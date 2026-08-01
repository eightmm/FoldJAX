"""Deterministic, Torch-free protein MSA search and cache boundary."""

from __future__ import annotations

import hashlib
import io
import json
import shutil
import subprocess
import tarfile
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

_CACHE_SCHEMA_VERSION = 1
_MAX_ARCHIVE_BYTES = 512 * 1024 * 1024
_MAX_EXTRACTED_BYTES = 512 * 1024 * 1024
_MAX_MEMBERS = 10_000


class SearchError(RuntimeError):
    """An MSA provider or cached result violated the search contract."""


@dataclass(frozen=True)
class MsaPayload:
    paired: str
    unpaired: str
    source: Mapping[str, Any] = field(default_factory=dict)
    templates: str | None = None


class MsaBackend(Protocol):
    name: str
    version: str
    server_url: str | None
    cache_identity: Mapping[str, Any]

    def search(self, sequence: str) -> MsaPayload: ...


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _normalize_sequence(sequence: str) -> str:
    if not isinstance(sequence, str):
        raise TypeError("protein sequence must be a string")
    normalized = "".join(sequence.split()).upper()
    if not normalized:
        raise ValueError("protein sequence must not be empty")
    if not normalized.isascii() or not normalized.isalpha():
        raise ValueError("protein sequence must contain ASCII letters only")
    return normalized


def _first_a3m_sequence(a3m: str, label: str) -> str:
    header_seen = False
    sequence: list[str] = []
    for line in a3m.splitlines():
        if line.startswith(">"):
            if header_seen:
                break
            header_seen = True
        elif header_seen:
            sequence.append(line.strip())
        elif line.strip():
            raise SearchError(f"{label} does not start with a FASTA header")
    if not header_seen or not sequence:
        raise SearchError(f"{label} is empty or has no query sequence")
    return "".join(
        character
        for character in "".join(sequence)
        if character.isupper() and character != "-"
    )


def _validate_payload(sequence: str, payload: MsaPayload) -> None:
    for label, content in (
        ("paired MSA", payload.paired),
        ("unpaired MSA", payload.unpaired),
    ):
        if not isinstance(content, str) or not content.strip():
            raise SearchError(f"{label} response is missing")
        query = _first_a3m_sequence(content, label)
        if query != sequence:
            raise SearchError(
                f"{label} query does not match requested protein sequence: "
                f"expected {sequence!r}, got {query!r}"
            )


class MsaSearchPipeline:
    """Resolve paired/unpaired A3M files through a content-addressed cache."""

    def __init__(
        self,
        cache_dir: str | Path,
        backend: MsaBackend,
        *,
        options: Mapping[str, Any] | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.backend = backend
        self.options = dict(options or {})
        try:
            json.dumps(self.options, sort_keys=True, separators=(",", ":"))
        except (TypeError, ValueError) as exc:
            raise ValueError("MSA search options must be JSON-serializable") from exc

    def _identity(self, sequence: str) -> tuple[str, dict[str, Any]]:
        identity: dict[str, Any] = {
            "schema_version": _CACHE_SCHEMA_VERSION,
            "sequence": sequence,
            "backend": {
                "name": self.backend.name,
                "version": self.backend.version,
            },
            "server": self.backend.server_url,
            "backend_options": dict(
                getattr(self.backend, "cache_identity", {})
            ),
            "options": self.options,
        }
        canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"))
        return _sha256(canonical.encode()), identity

    @staticmethod
    def _paths(directory: Path) -> dict[str, str]:
        paths = {
            "pairedMsaPath": str((directory / "pairing.a3m").resolve()),
            "unpairedMsaPath": str((directory / "non_pairing.a3m").resolve()),
            "provenancePath": str((directory / "provenance.json").resolve()),
        }
        template_hits = directory / "pdb70.m8"
        if template_hits.is_file():
            paths["templateHitsPath"] = str(template_hits.resolve())
        return paths

    def _cached(
        self,
        directory: Path,
        sequence: str,
        cache_key: str,
        identity: Mapping[str, Any],
    ) -> dict[str, str] | None:
        if not directory.exists():
            return None
        if not directory.is_dir():
            raise SearchError(f"MSA cache path is not a directory: {directory}")
        provenance_path = directory / "provenance.json"
        if not provenance_path.is_file():
            raise SearchError(f"MSA cache is incomplete: {provenance_path} is missing")
        try:
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SearchError(
                f"invalid MSA cache provenance: {provenance_path}"
            ) from exc
        if not isinstance(provenance, dict):
            raise SearchError(f"invalid MSA cache provenance: {provenance_path}")
        if provenance.get("cache_key") != cache_key:
            raise SearchError(f"MSA cache provenance key mismatch: {provenance_path}")
        for key, expected in identity.items():
            if provenance.get(key) != expected:
                raise SearchError(
                    f"MSA cache provenance field mismatch for {key}: {provenance_path}"
                )
        if provenance.get("sequence_sha256") != _sha256(sequence.encode()):
            raise SearchError(f"MSA cache sequence hash mismatch: {provenance_path}")

        contents: dict[str, str] = {}
        file_specs = [
            ("pairing.a3m", "paired MSA"),
            ("non_pairing.a3m", "unpaired MSA"),
        ]
        if "pdb70.m8" in provenance.get("files", {}):
            file_specs.append(("pdb70.m8", "template hits"))
        for filename, label in file_specs:
            path = directory / filename
            if not path.is_file():
                raise SearchError(f"MSA cache is incomplete: {path} is missing")
            try:
                raw = path.read_bytes()
                content = raw.decode("utf-8")
            except (OSError, UnicodeDecodeError) as exc:
                raise SearchError(f"MSA cache file is unreadable: {path}") from exc
            metadata = provenance.get("files", {}).get(filename, {})
            if metadata.get("sha256") != _sha256(raw):
                raise SearchError(f"MSA cache content hash mismatch: {path}")
            if metadata.get("bytes") != len(raw):
                raise SearchError(f"MSA cache content size mismatch: {path}")
            contents[label] = content
        _validate_payload(
            sequence,
            MsaPayload(
                contents["paired MSA"],
                contents["unpaired MSA"],
                templates=contents.get("template hits"),
            ),
        )
        return self._paths(directory)

    def _materialize(
        self,
        directory: Path,
        sequence: str,
        cache_key: str,
        identity: Mapping[str, Any],
        payload: MsaPayload,
    ) -> dict[str, str]:
        _validate_payload(sequence, payload)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        temporary = Path(
            tempfile.mkdtemp(prefix=f".{cache_key}.", dir=self.cache_dir)
        )
        try:
            files: dict[str, dict[str, Any]] = {}
            outputs = [
                ("pairing.a3m", payload.paired),
                ("non_pairing.a3m", payload.unpaired),
            ]
            if payload.templates is not None:
                outputs.append(("pdb70.m8", payload.templates))
            for filename, content in outputs:
                raw = content.encode()
                (temporary / filename).write_bytes(raw)
                files[filename] = {"sha256": _sha256(raw), "bytes": len(raw)}
            provenance = {
                **identity,
                "cache_key": cache_key,
                "sequence_sha256": _sha256(sequence.encode()),
                "source": dict(payload.source),
                "files": files,
            }
            (temporary / "provenance.json").write_text(
                json.dumps(provenance, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            try:
                temporary.rename(directory)
            except FileExistsError:
                shutil.rmtree(temporary)
                cached = self._cached(directory, sequence, cache_key, identity)
                if cached is None:
                    raise AssertionError("MSA cache disappeared during materialization")
                return cached
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return self._paths(directory)

    def search(self, sequences: Sequence[str]) -> list[dict[str, str]]:
        normalized = [_normalize_sequence(sequence) for sequence in sequences]
        if not normalized:
            raise ValueError("at least one protein sequence is required")
        resolved: dict[str, dict[str, str]] = {}
        for sequence in dict.fromkeys(normalized):
            cache_key, identity = self._identity(sequence)
            directory = self.cache_dir / cache_key
            cached = self._cached(directory, sequence, cache_key, identity)
            if cached is None:
                cached = self._materialize(
                    directory,
                    sequence,
                    cache_key,
                    identity,
                    self.backend.search(sequence),
                )
            resolved[sequence] = cached
        return [resolved[sequence] for sequence in normalized]


class LocalMsaBackend:
    """Run a local wrapper that writes pairing and non-pairing A3M files."""

    name = "local"
    server_url = None

    def __init__(
        self,
        command: Sequence[str],
        *,
        version: str,
        runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
        timeout: float | None = None,
    ) -> None:
        if not command or any(
            not isinstance(item, str) or not item for item in command
        ):
            raise ValueError("local MSA command is required")
        if not version:
            raise ValueError("local MSA version is required")
        if timeout is not None and timeout <= 0:
            raise ValueError("local MSA timeout must be positive")
        self.command = tuple(command)
        self.version = version
        self.runner = runner
        self.timeout = timeout

    @property
    def cache_identity(self) -> Mapping[str, Any]:
        return {"command": list(self.command)}

    def search(self, sequence: str) -> MsaPayload:
        sequence = _normalize_sequence(sequence)
        with tempfile.TemporaryDirectory(prefix="chai-jax-msa-") as raw_dir:
            directory = Path(raw_dir)
            fasta = directory / "query.fasta"
            output = directory / "result"
            output.mkdir()
            fasta.write_text(f">query\n{sequence}\n", encoding="utf-8")
            command = [*self.command, "--input", str(fasta), "--output", str(output)]
            try:
                completed = self.runner(
                    command,
                    check=False,
                    capture_output=True,
                    text=True,
                    shell=False,
                    timeout=self.timeout,
                )
            except (OSError, subprocess.SubprocessError) as exc:
                raise SearchError(
                    f"failed to run local MSA command: {self.command[0]}"
                ) from exc
            if completed.returncode:
                detail = (completed.stderr or completed.stdout or "").strip()[-500:]
                raise SearchError(
                    f"local MSA command exited with {completed.returncode}: {detail}"
                )
            paired = output / "pairing.a3m"
            unpaired = output / "non_pairing.a3m"
            missing = [str(path) for path in (paired, unpaired) if not path.is_file()]
            if missing:
                raise SearchError(
                    f"local MSA command did not produce: {', '.join(missing)}"
                )
            try:
                payload = MsaPayload(
                    paired.read_text(encoding="utf-8"),
                    unpaired.read_text(encoding="utf-8"),
                    {"command": list(self.command), "version": self.version},
                )
            except (OSError, UnicodeDecodeError) as exc:
                raise SearchError("local MSA output is unreadable") from exc
            _validate_payload(sequence, payload)
            return payload


@dataclass(frozen=True)
class HttpResponse:
    status: int
    body: bytes


HttpTransport = Callable[
    [str, str, bytes | None, Mapping[str, str], float], HttpResponse
]


def _urllib_transport(
    method: str,
    url: str,
    data: bytes | None,
    headers: Mapping[str, str],
    timeout: float,
) -> HttpResponse:
    request = urllib.request.Request(
        url, data=data, headers=dict(headers), method=method
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return HttpResponse(response.status, response.read())
    except urllib.error.HTTPError as exc:
        return HttpResponse(exc.code, exc.read())


class RemoteMMseqs2Backend:
    """ColabFold-compatible ticket/poll/download client using stdlib HTTP."""

    name = "remote-mmseqs2"

    def __init__(
        self,
        server_url: str,
        *,
        version: str,
        headers: Mapping[str, str] | None = None,
        transport: HttpTransport = _urllib_transport,
        timeout: float = 30.0,
        poll_interval: float = 5.0,
        max_wait_seconds: float = 3600.0,
        use_env: bool = True,
        use_filter: bool = True,
        pairing_strategy: str = "greedy",
        use_templates: bool = False,
    ) -> None:
        parsed = urllib.parse.urlsplit(server_url.strip())
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("remote MSA server must be an HTTP(S) URL")
        if parsed.username or parsed.password:
            raise ValueError("remote MSA credentials must not be embedded in the URL")
        if not version:
            raise ValueError("remote MSA version is required")
        if timeout <= 0 or poll_interval < 0 or max_wait_seconds <= 0:
            raise ValueError("remote MSA timeout values are invalid")
        if pairing_strategy not in {"greedy", "complete"}:
            raise ValueError("pairing_strategy must be 'greedy' or 'complete'")
        self.server_url = server_url.rstrip("/")
        self.version = version
        self.headers = {"User-Agent": "chai-jax/0.1.0", **dict(headers or {})}
        self.transport = transport
        self.timeout = timeout
        self.poll_interval = poll_interval
        self.max_wait_seconds = max_wait_seconds
        self.use_env = use_env
        self.use_filter = use_filter
        self.pairing_strategy = pairing_strategy
        self.use_templates = use_templates

    @property
    def cache_identity(self) -> Mapping[str, Any]:
        return {
            "use_env": self.use_env,
            "use_filter": self.use_filter,
            "pairing_strategy": self.pairing_strategy,
            "use_templates": self.use_templates,
        }

    def _request(self, method: str, path: str, data: bytes | None = None) -> bytes:
        try:
            response = self.transport(
                method,
                f"{self.server_url}/{path.lstrip('/')}",
                data,
                self.headers,
                self.timeout,
            )
        except OSError as exc:
            raise SearchError(f"remote MSA request {path!r} failed") from exc
        if not 200 <= response.status < 300:
            raise SearchError(
                f"remote MSA request {path!r} failed with HTTP {response.status}"
            )
        return response.body

    def _json(
        self, method: str, path: str, data: bytes | None = None
    ) -> dict[str, Any]:
        try:
            payload = json.loads(self._request(method, path, data))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SearchError(f"remote MSA returned invalid JSON for {path!r}") from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("status"), str):
            raise SearchError(f"remote MSA returned invalid response for {path!r}")
        return payload

    @staticmethod
    def _read_archive_files(
        archive: bytes, required: Sequence[str]
    ) -> dict[str, str]:
        if len(archive) > _MAX_ARCHIVE_BYTES:
            raise SearchError("remote MSA archive exceeds the compressed size limit")
        try:
            with tarfile.open(fileobj=io.BytesIO(archive), mode="r:*") as tar:
                members = tar.getmembers()
                if len(members) > _MAX_MEMBERS:
                    raise SearchError("remote MSA archive contains too many entries")
                total_size = 0
                names: dict[str, list[tarfile.TarInfo]] = {}
                for member in members:
                    path = PurePosixPath(member.name)
                    if path.is_absolute() or ".." in path.parts:
                        raise SearchError(
                            f"remote MSA archive has unsafe path: {member.name}"
                        )
                    if member.issym() or member.islnk():
                        raise SearchError(
                            f"remote MSA archive has unsafe link: {member.name}"
                        )
                    if member.isfile():
                        if member.size < 0:
                            raise SearchError(
                                "remote MSA archive has invalid file size"
                            )
                        total_size += member.size
                        if total_size > _MAX_EXTRACTED_BYTES:
                            raise SearchError(
                                "remote MSA archive exceeds the extracted size limit"
                            )
                        names.setdefault(member.name, []).append(member)
                contents: dict[str, str] = {}
                for name in required:
                    matches = names.get(name, [])
                    if len(matches) != 1:
                        raise SearchError(
                            "remote MSA archive must contain exactly one "
                            f"regular file named {name}"
                        )
                    extracted = tar.extractfile(matches[0])
                    if extracted is None:
                        raise SearchError(f"remote MSA archive cannot read: {name}")
                    contents[name] = (
                        extracted.read().decode("utf-8").replace("\x00", "")
                    )
        except SearchError:
            raise
        except (tarfile.TarError, OSError, UnicodeDecodeError) as exc:
            raise SearchError("remote MSA returned an invalid archive") from exc
        return contents

    @staticmethod
    def _read_archive(archive: bytes, required: Sequence[str]) -> str:
        contents = RemoteMMseqs2Backend._read_archive_files(archive, required)
        return "".join(contents[name] for name in required)

    def _run(self, sequence: str, *, paired: bool) -> tuple[str, str, str | None]:
        endpoint = "ticket/pair" if paired else "ticket/msa"
        if paired:
            mode = f"pair{self.pairing_strategy}"
            if self.use_env:
                mode += "-env"
        elif self.use_filter:
            mode = "env" if self.use_env else "all"
        else:
            mode = "env-nofilter" if self.use_env else "nofilter"
        query = f">101\n{sequence}\n"
        body = urllib.parse.urlencode({"q": query, "mode": mode}).encode()
        response = self._json("POST", endpoint, body)
        status = response["status"]
        job_id = response.get("id")
        if status in {"ERROR", "MAINTENANCE"}:
            raise SearchError(f"remote MSA submission ended with status {status!r}")
        if not isinstance(job_id, str) or not job_id:
            raise SearchError("remote MSA submission response is missing a job id")
        deadline = time.monotonic() + self.max_wait_seconds
        pending = {"UNKNOWN", "RUNNING", "PENDING", "RATELIMIT"}
        while status in pending:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"remote MSA search {job_id} timed out")
            if self.poll_interval:
                time.sleep(self.poll_interval)
            status = self._json("GET", f"ticket/{job_id}")["status"]
        if status != "COMPLETE":
            raise SearchError(
                f"remote MSA search {job_id} ended with status {status!r}"
            )
        archive = self._request("GET", f"result/download/{job_id}")
        required = (
            ("pair.a3m",)
            if paired
            else (
                "uniref.a3m",
                "bfd.mgnify30.metaeuk30.smag30.a3m",
                *(("pdb70.m8",) if self.use_templates else ()),
            )
        )
        contents = self._read_archive_files(archive, required)
        msa_names = required[:1] if paired else required[:2]
        msa = "".join(contents[name] for name in msa_names)
        return msa, job_id, contents.get("pdb70.m8")

    def search(self, sequence: str) -> MsaPayload:
        sequence = _normalize_sequence(sequence)
        unpaired, unpaired_job, templates = self._run(sequence, paired=False)
        paired, paired_job, _ = self._run(sequence, paired=True)
        payload = MsaPayload(
            paired,
            unpaired,
            {"paired_job_id": paired_job, "unpaired_job_id": unpaired_job},
            templates,
        )
        _validate_payload(sequence, payload)
        return payload
