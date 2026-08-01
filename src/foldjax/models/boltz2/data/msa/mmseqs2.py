"""ColabFold MMseqs2 server client used by Boltz-JAX preprocessing.

The wire protocol follows ColabFold/Boltz, while adding bounded retries,
response validation, atomic downloads, and safe archive extraction. Completed
archives are reusable, so restarting preprocessing does not resubmit a search.
"""

from __future__ import annotations

import logging
import os
import random
import tarfile
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import requests
from requests.auth import HTTPBasicAuth
from tqdm import tqdm

logger = logging.getLogger(__name__)

TQDM_BAR_FORMAT = (
    "{l_bar}{bar}| {n_fmt}/{total_fmt} [elapsed: {elapsed} remaining: {remaining}]"
)
_ACTIVE_STATUSES = frozenset({"UNKNOWN", "RUNNING", "PENDING", "RATELIMIT"})
_RESUBMIT_STATUSES = frozenset({"UNKNOWN", "RATELIMIT"})


def _retry(
    operation: Callable[[], Any],
    *,
    description: str,
    max_retries: int,
) -> Any:
    """Run one HTTP operation with bounded exponential backoff."""
    for attempt in range(max_retries + 1):
        try:
            response = operation()
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            if attempt >= max_retries:
                raise RuntimeError(
                    f"MSA server {description} failed after {attempt + 1} attempts"
                ) from exc
            delay = min(2**attempt, 30) + random.random()
            logger.warning(
                "MSA server %s failed (%d/%d): %s; retrying in %.1fs",
                description,
                attempt + 1,
                max_retries + 1,
                exc,
                delay,
            )
            time.sleep(delay)

    raise AssertionError("unreachable")


def _json_response(response: requests.Response, *, operation: str) -> dict[str, Any]:
    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError(
            f"MSA server returned non-JSON data for {operation}: "
            f"{response.text[:200]!r}"
        ) from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("status"), str):
        raise RuntimeError(f"invalid MSA server response for {operation}: {payload!r}")
    return payload


def _validate_auth(
    username: str | None,
    password: str | None,
    auth_headers: dict[str, str] | None,
) -> HTTPBasicAuth | None:
    if (username is None) != (password is None):
        raise ValueError("MSA server basic auth requires both username and password")
    if username is not None and auth_headers:
        raise ValueError("MSA server basic auth and API-key headers are mutually exclusive")
    return HTTPBasicAuth(username, password) if username is not None else None


def _safe_extract(archive_path: Path, result_dir: Path) -> None:
    """Validate a downloaded archive and extract regular files safely."""
    try:
        with tarfile.open(archive_path, mode="r:*") as archive:
            base = result_dir.resolve()
            for member in archive.getmembers():
                destination = (base / member.name).resolve()
                if (
                    member.issym()
                    or member.islnk()
                    or Path(member.name).is_absolute()
                    or os.path.commonpath((base, destination)) != str(base)
                ):
                    raise RuntimeError(
                        f"unsafe tar entry from MSA server: {member.name}"
                    )
            archive.extractall(result_dir, filter="data")
    except (tarfile.TarError, EOFError) as exc:
        archive_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"invalid MSA result archive; removed cache file {archive_path}"
        ) from exc
    except RuntimeError:
        archive_path.unlink(missing_ok=True)
        raise


def _read_a3m_results(files: Sequence[Path], query_ids: Sequence[int]) -> list[str]:
    blocks: dict[int, list[str]] = {}
    for path in files:
        if not path.is_file():
            raise RuntimeError(f"MSA result archive is missing {path.name}")
        update_query = True
        query_id: int | None = None
        with path.open(encoding="utf-8", errors="replace") as handle:
            for raw_line in handle:
                line = raw_line.replace("\x00", "")
                if line != raw_line:
                    update_query = True
                if line.startswith(">") and update_query:
                    try:
                        query_id = int(line[1:].strip())
                    except ValueError as exc:
                        raise RuntimeError(
                            f"invalid query header in MSA result {path}: {line.strip()!r}"
                        ) from exc
                    update_query = False
                    blocks.setdefault(query_id, [])
                if query_id is None:
                    if line.strip():
                        raise RuntimeError(f"MSA result {path} starts without a query header")
                    continue
                blocks[query_id].append(line)

    missing = sorted(set(query_ids) - blocks.keys())
    if missing:
        raise RuntimeError(f"MSA server result is missing query IDs: {missing}")
    return ["".join(blocks[query_id]) for query_id in query_ids]


def run_mmseqs2(  # noqa: C901, PLR0912, PLR0915
    x: str | list[str],
    prefix: str = "tmp",
    use_env: bool = True,
    use_filter: bool = True,
    use_pairing: bool = False,
    pairing_strategy: str = "greedy",
    host_url: str = "https://api.colabfold.com",
    msa_server_username: str | None = None,
    msa_server_password: str | None = None,
    auth_headers: dict[str, str] | None = None,
    *,
    request_timeout: float = 6.02,
    max_retries: int = 5,
    max_wait_seconds: float = 3600.0,
) -> list[str]:
    """Generate unpaired or paired A3M blocks through an MMseqs2 server.

    The returned list follows the original input order, including duplicates.
    A successful downloaded archive is cached below ``prefix``.
    """
    if not x or (isinstance(x, list) and not x):
        raise ValueError("at least one protein sequence is required")
    if pairing_strategy not in {"greedy", "complete"}:
        raise ValueError("pairing_strategy must be 'greedy' or 'complete'")
    if request_timeout <= 0 or max_retries < 0 or max_wait_seconds <= 0:
        raise ValueError("timeouts must be positive and max_retries non-negative")

    auth = _validate_auth(
        msa_server_username, msa_server_password, auth_headers
    )
    headers = {"User-Agent": "boltz-jax/0.1.0"}
    if auth_headers:
        headers.update(auth_headers)
    host_url = host_url.rstrip("/")
    if not host_url:
        raise ValueError("host_url must not be empty")

    sequences = [x] if isinstance(x, str) else list(x)
    if any(not sequence for sequence in sequences):
        raise ValueError("protein sequences must not be empty")
    unique_sequences = list(dict.fromkeys(sequences))
    start_id = 101
    query_ids = [start_id + unique_sequences.index(seq) for seq in sequences]
    query = "".join(
        f">{start_id + index}\n{sequence}\n"
        for index, sequence in enumerate(unique_sequences)
    )

    if use_pairing:
        mode = "pairgreedy" if pairing_strategy == "greedy" else "paircomplete"
        if use_env:
            mode += "-env"
        endpoint = "ticket/pair"
    else:
        mode = ("env" if use_env else "all") if use_filter else (
            "env-nofilter" if use_env else "nofilter"
        )
        endpoint = "ticket/msa"

    result_dir = Path(f"{prefix}_{mode}")
    result_dir.mkdir(parents=True, exist_ok=True)
    archive_path = result_dir / "out.tar.gz"

    def submit() -> dict[str, Any]:
        response = _retry(
            lambda: requests.post(
                f"{host_url}/{endpoint}",
                data={"q": query, "mode": mode},
                timeout=request_timeout,
                headers=headers,
                auth=auth,
            ),
            description="submission",
            max_retries=max_retries,
        )
        return _json_response(response, operation="submission")

    def status(job_id: str) -> dict[str, Any]:
        response = _retry(
            lambda: requests.get(
                f"{host_url}/ticket/{job_id}",
                timeout=request_timeout,
                headers=headers,
                auth=auth,
            ),
            description="status request",
            max_retries=max_retries,
        )
        return _json_response(response, operation="status request")

    if not archive_path.is_file():
        estimate = 150 * len(unique_sequences)
        with tqdm(total=estimate, bar_format=TQDM_BAR_FORMAT) as progress:
            progress.set_description("SUBMIT")
            submitted_at = time.monotonic()
            response = submit()
            while response["status"] in _RESUBMIT_STATUSES:
                if time.monotonic() - submitted_at > max_wait_seconds:
                    raise TimeoutError("timed out while submitting the MSA search")
                time.sleep(5 + random.randint(0, 5))
                response = submit()

            state = response["status"]
            if state == "MAINTENANCE":
                raise RuntimeError("MMseqs2 API is undergoing maintenance")
            if state == "ERROR":
                raise RuntimeError(
                    "MMseqs2 API rejected the search; check the protein sequences"
                )
            job_id = response.get("id")
            if not isinstance(job_id, str) or not job_id:
                raise RuntimeError(f"MSA submission response has no job id: {response!r}")

            progress.set_description(state)
            while state in _ACTIVE_STATUSES:
                if time.monotonic() - submitted_at > max_wait_seconds:
                    raise TimeoutError(f"MSA search {job_id} exceeded max_wait_seconds")
                delay = 5 + random.randint(0, 5)
                time.sleep(delay)
                response = status(job_id)
                state = response["status"]
                progress.set_description(state)
                progress.update(min(delay, max(estimate - progress.n, 0)))

            if state != "COMPLETE":
                raise RuntimeError(f"MSA search {job_id} ended with status {state!r}")

            partial_path = archive_path.with_name(f"{archive_path.name}.part")
            partial_path.unlink(missing_ok=True)
            download = _retry(
                lambda: requests.get(
                    f"{host_url}/result/download/{job_id}",
                    timeout=request_timeout,
                    headers=headers,
                    auth=auth,
                ),
                description="result download",
                max_retries=max_retries,
            )
            partial_path.write_bytes(download.content)
            partial_path.replace(archive_path)
            progress.update(max(estimate - progress.n, 0))

    expected_files = [result_dir / "pair.a3m"] if use_pairing else [
        result_dir / "uniref.a3m"
    ]
    if not use_pairing and use_env:
        expected_files.append(
            result_dir / "bfd.mgnify30.metaeuk30.smag30.a3m"
        )
    if any(not path.is_file() for path in expected_files):
        _safe_extract(archive_path, result_dir)
    missing_files = [path.name for path in expected_files if not path.is_file()]
    if missing_files:
        archive_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"MSA result archive is missing required files: {missing_files}"
        )
    return _read_a3m_results(expected_files, query_ids)
