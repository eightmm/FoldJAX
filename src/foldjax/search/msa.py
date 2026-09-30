"""Deterministic MSA search with local and remote backends.

Extracted from the former standalone Protenix JAX port -- it had no model imports
at all -- so the search step can be shared instead of duplicated. The private
helper names and ColabFold MMseqs2 behavior remain compatible with that port.

Nothing was changed during the move: the cache layout, filenames and returned key
names are protenix's, so protenix's behaviour is byte-identical and a consumer with
different naming requirements adapts on its side.

That boundary was not obvious. Making the cache filenames caller-chosen looks
harmless -- the alignments are the same bytes -- but the cache key does not include
them, so a rename turns every subsequent cache hit into "cache is incomplete" while
looking up files under names the cache never wrote. A test caught it. Naming for a
downstream parser belongs to the consumer that has the requirement: OpenFold3, for
instance, selects alignment files by *stem* and only accepts database names such as
``colabfold_main``, and links the cached files to those names itself.

This package depends on nothing else in the workspace, which is what lets both
``foldjax`` (which depends on the ports) and the ports themselves use it without a
dependency cycle.
"""

from __future__ import annotations

import base64
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
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


class SearchError(RuntimeError):
    """An MSA provider returned an unusable or incomplete result."""


@dataclass(frozen=True)
class MsaPayload:
    paired: str
    unpaired: str
    source: Mapping[str, Any] = field(default_factory=dict)


class MsaBackend(Protocol):
    name: str
    version: str

    def search(self, sequence: str) -> MsaPayload: ...


@dataclass(frozen=True)
class ComplexPairPayload:
    """One paired alignment per submitted sequence, row-aligned across them."""

    paired: tuple[str, ...]
    source: Mapping[str, Any] = field(default_factory=dict)


#: How OpenFold3 v0.5.0 asks ColabFold to pair a complex: the greedy strategy
#: with the environmental databases (``pairing_strategy="greedy"``,
#: ``use_env=True`` in colabfold_msa_server.py:305-313).
COMPLEX_PAIRING_MODE = "pairgreedy-env"


def _split_colabfold_a3m(text: str, label: str) -> dict[int, str]:
    """Split a multi-query ColabFold A3M into its per-query blocks, keyed by M.

    The same gather loop OpenFold3 runs on ``pair.a3m``
    (colabfold_msa_server.py:470-490): a NUL byte starts a new block, and only
    the header right after it -- or the file's first header -- is the query
    number. Hit headers inside a block are not numbers and stay in the block.
    """
    blocks: dict[int, list[str]] = {}
    expect_query, query = True, None
    for line in io.StringIO(text):
        if "\x00" in line:
            line = line.replace("\x00", "")
            expect_query = True
        if line.startswith(">") and expect_query:
            try:
                query = int(line[1:].rstrip())
            except ValueError as exc:
                raise SearchError(
                    f"{label} block header {line.rstrip()!r} is not a query number"
                ) from exc
            expect_query = False
            blocks.setdefault(query, [])
        if query is None:
            if line.strip():
                raise SearchError(f"{label} does not start with a query header")
            continue
        blocks[query].append(line)
    return {key: "".join(lines) for key, lines in blocks.items()}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _normalize_sequence(sequence: str) -> str:
    normalized = "".join(sequence.split()).upper()
    if not normalized:
        raise ValueError("protein sequence must not be empty")
    if not normalized.isascii() or not normalized.isalpha():
        raise ValueError("protein sequence must contain ASCII letters only")
    return normalized


def _first_a3m_lines(lines: Iterable[str], label: str) -> str:
    header_seen = False
    complete = False
    sequence: list[str] = []
    for line in lines:
        if complete:
            continue
        if line.startswith(">"):
            if header_seen:
                complete = True
                continue
            header_seen = True
        elif header_seen:
            sequence.append(line.strip())
        elif line.strip():
            raise SearchError(f"{label} does not start with a FASTA header")
    if not header_seen or not sequence:
        raise SearchError(f"{label} is empty or has no query sequence")
    return "".join(c for c in "".join(sequence) if c.isupper() and c != "-")


def _first_a3m_sequence(a3m: str, label: str) -> str:
    return _first_a3m_lines(a3m.splitlines(), label)


def _first_verified_a3m_file_sequence(
    path: Path, label: str, expected_sha256: str | None
) -> str | None:
    """Hash one file, then validate its UTF-8/query through the same open fd."""

    digest = hashlib.sha256()
    with path.open("rb") as binary:
        for chunk in iter(lambda: binary.read(1 << 20), b""):
            digest.update(chunk)
        if digest.hexdigest() != expected_sha256:
            return None
        binary.seek(0)
        text = io.TextIOWrapper(binary, encoding="utf-8")
        try:

            def logical_lines() -> Iterable[str]:
                for physical_line in text:
                    yield from physical_line.splitlines()

            return _first_a3m_lines(logical_lines(), label)
        finally:
            text.detach()


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
    """Cache backend results by sequence plus immutable search provenance."""

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
        identity = {
            "schema_version": 1,
            "sequence": sequence,
            "backend": {"name": self.backend.name, "version": self.backend.version},
            "options": self.options,
        }
        canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"))
        return _sha256(canonical.encode()), identity

    def _paths(self, directory: Path) -> dict[str, str]:
        return {
            "pairedMsaPath": str((directory / "pairing.a3m").resolve()),
            "unpairedMsaPath": str((directory / "non_pairing.a3m").resolve()),
            "provenancePath": str((directory / "provenance.json").resolve()),
        }

    def _cached(
        self, directory: Path, sequence: str, cache_key: str
    ) -> dict[str, str] | None:
        provenance_path = directory / "provenance.json"
        if not directory.exists():
            return None
        if not provenance_path.is_file():
            raise SearchError(f"MSA cache is incomplete: {provenance_path} is missing")
        try:
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SearchError(
                f"invalid MSA cache provenance: {provenance_path}"
            ) from exc
        if provenance.get("cache_key") != cache_key:
            raise SearchError(f"MSA cache provenance key mismatch: {provenance_path}")
        for filename, label in (
            ("pairing.a3m", "paired MSA"),
            ("non_pairing.a3m", "unpaired MSA"),
        ):
            path = directory / filename
            if not path.is_file():
                raise SearchError(f"MSA cache is incomplete: {path} is missing")
            expected = provenance.get("files", {}).get(filename, {}).get("sha256")
            query = _first_verified_a3m_file_sequence(path, label, expected)
            if query is None:
                raise SearchError(f"MSA cache content hash mismatch: {path}")
            if query != sequence:
                raise SearchError(
                    f"{label} query does not match requested protein sequence: "
                    f"expected {sequence!r}, got {query!r}"
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
        temp_dir = Path(tempfile.mkdtemp(prefix=f".{cache_key}.", dir=self.cache_dir))
        try:
            files: dict[str, dict[str, Any]] = {}
            for filename, content in (
                ("pairing.a3m", payload.paired),
                ("non_pairing.a3m", payload.unpaired),
            ):
                raw = content.encode()
                (temp_dir / filename).write_bytes(raw)
                files[filename] = {"sha256": _sha256(raw), "bytes": len(raw)}
            provenance = {
                **identity,
                "cache_key": cache_key,
                "sequence_sha256": _sha256(sequence.encode()),
                "source": dict(payload.source),
                "files": files,
            }
            (temp_dir / "provenance.json").write_text(
                json.dumps(provenance, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            try:
                temp_dir.rename(directory)
            except FileExistsError:
                shutil.rmtree(temp_dir)
                cached = self._cached(directory, sequence, cache_key)
                if cached is None:
                    raise AssertionError("cache disappeared during materialization")
                return cached
        except Exception:
            shutil.rmtree(temp_dir, ignore_errors=True)
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
            cached = self._cached(directory, sequence, cache_key)
            resolved[sequence] = cached or self._materialize(
                directory,
                sequence,
                cache_key,
                identity,
                self.backend.search(sequence),
            )
        return [resolved[sequence] for sequence in normalized]

    @property
    def pairs_complexes(self) -> bool:
        """Whether the backend can pair several sequences in one search."""
        return callable(getattr(self.backend, "search_complex", None))

    def search_complex(self, sequences: Sequence[str]) -> list[dict[str, str]]:
        """Pair the distinct sequences of one complex in a single search.

        One entry per input sequence, in input order; repeated sequences share a
        file. The cache key is the ordered tuple of distinct sequences, so this
        never collides with -- or reuses -- a per-sequence ``search`` entry.
        """
        normalized = [_normalize_sequence(sequence) for sequence in sequences]
        unique = list(dict.fromkeys(normalized))
        if len(unique) < 2:
            raise ValueError("complex pairing needs at least two distinct sequences")
        if not self.pairs_complexes:
            raise SearchError(
                f"MSA backend {self.backend.name!r} cannot pair a complex"
            )
        identity = {
            "schema_version": 1,
            "kind": "complex_pairing",
            "sequences": unique,
            "backend": {"name": self.backend.name, "version": self.backend.version},
            "mode": getattr(self.backend, "complex_pairing_mode", None),
            "options": self.options,
        }
        canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"))
        cache_key = _sha256(canonical.encode())
        directory = self.cache_dir / cache_key
        paths = self._complex_cached(
            directory, unique, cache_key
        ) or self._complex_materialize(
            directory, unique, cache_key, identity, self.backend.search_complex(unique)
        )
        by_sequence = dict(zip(unique, paths, strict=True))
        return [by_sequence[sequence] for sequence in normalized]

    @staticmethod
    def _complex_name(index: int) -> str:
        return f"pair_{index:03d}.a3m"

    def _complex_paths(self, directory: Path, count: int) -> list[dict[str, str]]:
        provenance = str((directory / "provenance.json").resolve())
        return [
            {
                "pairedMsaPath": str((directory / self._complex_name(i)).resolve()),
                "provenancePath": provenance,
            }
            for i in range(count)
        ]

    def _complex_cached(
        self, directory: Path, sequences: list[str], cache_key: str
    ) -> list[dict[str, str]] | None:
        provenance_path = directory / "provenance.json"
        if not directory.exists():
            return None
        if not provenance_path.is_file():
            raise SearchError(f"MSA cache is incomplete: {provenance_path} is missing")
        try:
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SearchError(
                f"invalid MSA cache provenance: {provenance_path}"
            ) from exc
        if provenance.get("cache_key") != cache_key:
            raise SearchError(f"MSA cache provenance key mismatch: {provenance_path}")
        for index, sequence in enumerate(sequences):
            filename = self._complex_name(index)
            path = directory / filename
            if not path.is_file():
                raise SearchError(f"MSA cache is incomplete: {path} is missing")
            expected = provenance.get("files", {}).get(filename, {}).get("sha256")
            query = _first_verified_a3m_file_sequence(path, "paired MSA", expected)
            if query is None:
                raise SearchError(f"MSA cache content hash mismatch: {path}")
            if query != sequence:
                raise SearchError(
                    "paired MSA query does not match requested protein sequence: "
                    f"expected {sequence!r}, got {query!r}"
                )
        return self._complex_paths(directory, len(sequences))

    def _complex_materialize(
        self,
        directory: Path,
        sequences: list[str],
        cache_key: str,
        identity: Mapping[str, Any],
        payload: ComplexPairPayload,
    ) -> list[dict[str, str]]:
        if len(payload.paired) != len(sequences):
            raise SearchError(
                f"complex pairing returned {len(payload.paired)} alignments for "
                f"{len(sequences)} sequences"
            )
        for sequence, content in zip(sequences, payload.paired, strict=True):
            if not isinstance(content, str) or not content.strip():
                raise SearchError("paired MSA response is missing")
            query = _first_a3m_sequence(content, "paired MSA")
            if query != sequence:
                raise SearchError(
                    "paired MSA query does not match requested protein sequence: "
                    f"expected {sequence!r}, got {query!r}"
                )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        temp_dir = Path(tempfile.mkdtemp(prefix=f".{cache_key}.", dir=self.cache_dir))
        try:
            files: dict[str, dict[str, Any]] = {}
            for index, content in enumerate(payload.paired):
                raw = content.encode()
                filename = self._complex_name(index)
                (temp_dir / filename).write_bytes(raw)
                files[filename] = {"sha256": _sha256(raw), "bytes": len(raw)}
            provenance = {
                **identity,
                "cache_key": cache_key,
                "source": dict(payload.source),
                "files": files,
            }
            (temp_dir / "provenance.json").write_text(
                json.dumps(provenance, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            try:
                temp_dir.rename(directory)
            except FileExistsError:
                shutil.rmtree(temp_dir)
                cached = self._complex_cached(directory, sequences, cache_key)
                if cached is None:
                    raise AssertionError("cache disappeared during materialization")
                return cached
        except Exception:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise
        return self._complex_paths(directory, len(sequences))


@dataclass(frozen=True)
class RnaMsaPayload:
    """Unpaired RNA alignment returned by an nhmmer-compatible backend."""

    unpaired: str
    source: Mapping[str, Any] = field(default_factory=dict)


class RnaMsaBackend(Protocol):
    name: str
    version: str

    def search(self, sequence: str) -> RnaMsaPayload: ...


def _normalize_rna_sequence(sequence: str) -> str:
    normalized = "".join(sequence.split()).upper()
    if not normalized:
        raise ValueError("RNA sequence must not be empty")
    invalid = set(normalized) - set("AGCUN")
    if invalid:
        raise ValueError(
            f"RNA sequence contains unsupported residues: {sorted(invalid)}"
        )
    return normalized


def _validate_rna_payload(sequence: str, payload: RnaMsaPayload) -> None:
    if not isinstance(payload.unpaired, str) or not payload.unpaired.strip():
        raise SearchError("RNA unpaired MSA response is missing")
    query = _first_a3m_sequence(payload.unpaired, "RNA unpaired MSA")
    if query != sequence:
        raise SearchError(
            "RNA unpaired MSA query does not match requested sequence: "
            f"expected {sequence!r}, got {query!r}"
        )


class RnaMsaSearchPipeline:
    """Content-addressed cache around a real local RNA-MSA backend boundary."""

    def __init__(
        self,
        cache_dir: str | Path,
        backend: RnaMsaBackend,
        *,
        options: Mapping[str, Any] | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.backend = backend
        self.options = dict(options or {})
        try:
            json.dumps(self.options, sort_keys=True, separators=(",", ":"))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "RNA MSA search options must be JSON-serializable"
            ) from exc

    def _identity(self, sequence: str) -> tuple[str, dict[str, Any]]:
        identity = {
            "schema_version": 1,
            "kind": "rna",
            "sequence": sequence,
            "backend": {"name": self.backend.name, "version": self.backend.version},
            "options": self.options,
        }
        canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"))
        return _sha256(canonical.encode()), identity

    @staticmethod
    def _paths(directory: Path) -> dict[str, str]:
        return {
            "unpairedMsaPath": str((directory / "rna_msa.a3m").resolve()),
            "provenancePath": str((directory / "provenance.json").resolve()),
        }

    def _cached(
        self, directory: Path, sequence: str, cache_key: str
    ) -> dict[str, str] | None:
        if not directory.exists():
            return None
        provenance_path = directory / "provenance.json"
        msa_path = directory / "rna_msa.a3m"
        if not provenance_path.is_file() or not msa_path.is_file():
            raise SearchError(f"RNA MSA cache is incomplete: {directory}")
        try:
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SearchError(
                f"invalid RNA MSA cache provenance: {provenance_path}"
            ) from exc
        if provenance.get("cache_key") != cache_key:
            raise SearchError(
                f"RNA MSA cache provenance key mismatch: {provenance_path}"
            )
        expected = provenance.get("files", {}).get("rna_msa.a3m", {}).get("sha256")
        try:
            query = _first_verified_a3m_file_sequence(
                msa_path, "RNA unpaired MSA", expected
            )
        except UnicodeDecodeError as exc:
            raise SearchError(f"RNA MSA cache is not UTF-8: {msa_path}") from exc
        if query is None:
            raise SearchError(f"RNA MSA cache content hash mismatch: {msa_path}")
        if query != sequence:
            raise SearchError(
                "RNA unpaired MSA query does not match requested sequence: "
                f"expected {sequence!r}, got {query!r}"
            )
        return self._paths(directory)

    def _materialize(
        self,
        directory: Path,
        sequence: str,
        cache_key: str,
        identity: Mapping[str, Any],
        payload: RnaMsaPayload,
    ) -> dict[str, str]:
        _validate_rna_payload(sequence, payload)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        temporary = Path(
            tempfile.mkdtemp(prefix=f".{cache_key}.", dir=self.cache_dir)
        )
        try:
            raw = payload.unpaired.encode()
            (temporary / "rna_msa.a3m").write_bytes(raw)
            provenance = {
                **identity,
                "cache_key": cache_key,
                "sequence_sha256": _sha256(sequence.encode()),
                "source": dict(payload.source),
                "files": {
                    "rna_msa.a3m": {"sha256": _sha256(raw), "bytes": len(raw)}
                },
            }
            (temporary / "provenance.json").write_text(
                json.dumps(provenance, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            try:
                temporary.rename(directory)
            except FileExistsError:
                shutil.rmtree(temporary)
                cached = self._cached(directory, sequence, cache_key)
                if cached is None:
                    raise AssertionError("RNA MSA cache disappeared")
                return cached
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return self._paths(directory)

    def search(self, sequences: Sequence[str]) -> list[dict[str, str]]:
        normalized = [_normalize_rna_sequence(sequence) for sequence in sequences]
        if not normalized:
            raise ValueError("at least one RNA sequence is required")
        resolved: dict[str, dict[str, str]] = {}
        for sequence in dict.fromkeys(normalized):
            cache_key, identity = self._identity(sequence)
            directory = self.cache_dir / cache_key
            cached = self._cached(directory, sequence, cache_key)
            resolved[sequence] = cached or self._materialize(
                directory,
                sequence,
                cache_key,
                identity,
                self.backend.search(sequence),
            )
        return [resolved[sequence] for sequence in normalized]



class LocalRnaMsaClient:
    """Run an nhmmer workflow wrapper producing ``rna_msa.a3m``."""

    name = "local-rna"

    def __init__(
        self,
        command: Sequence[str],
        *,
        version: str,
        runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    ) -> None:
        if not command or not version:
            raise ValueError("local RNA MSA command and version are required")
        self.command = tuple(command)
        self.version = version
        self._runner = runner

    def search(self, sequence: str) -> RnaMsaPayload:
        with tempfile.TemporaryDirectory(prefix="protenix-jax-rna-msa-") as raw_dir:
            directory = Path(raw_dir)
            fasta = directory / "query.fasta"
            output = directory / "result"
            output.mkdir()
            fasta.write_text(f">query\n{sequence}\n", encoding="utf-8")
            command = [*self.command, "--input", str(fasta), "--output", str(output)]
            try:
                completed = self._runner(
                    command, check=False, capture_output=True, text=True
                )
            except OSError as exc:
                raise SearchError(
                    f"failed to start local RNA MSA command: {command[0]}"
                ) from exc
            if completed.returncode:
                detail = (completed.stderr or completed.stdout or "").strip()[-500:]
                raise SearchError(
                    "local RNA MSA command exited with "
                    f"{completed.returncode}: {detail}"
                )
            result = output / "rna_msa.a3m"
            if not result.is_file():
                raise SearchError(
                    f"local RNA MSA command did not produce: {result}"
                )
            return RnaMsaPayload(
                result.read_text(encoding="utf-8"),
                {"command": list(self.command), "version": self.version},
            )



class LocalMsaClient:
    """Run a local wrapper which writes pairing/non_pairing A3M files.

    The wrapper receives ``--input <fasta> --output <directory>``. This keeps
    database/tool selection outside the library while preserving its version in
    the content-addressed cache identity.
    """

    name = "local"

    def __init__(
        self,
        command: Sequence[str],
        *,
        version: str,
        runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    ) -> None:
        if not command or not version:
            raise ValueError("local MSA command and version are required")
        self.command = tuple(command)
        self.version = version
        self._runner = runner

    def search(self, sequence: str) -> MsaPayload:
        with tempfile.TemporaryDirectory(prefix="protenix-jax-msa-") as raw_dir:
            directory = Path(raw_dir)
            fasta = directory / "query.fasta"
            output = directory / "result"
            output.mkdir()
            fasta.write_text(f">query\n{sequence}\n", encoding="utf-8")
            command = [*self.command, "--input", str(fasta), "--output", str(output)]
            try:
                completed = self._runner(
                    command, check=False, capture_output=True, text=True
                )
            except OSError as exc:
                raise SearchError(
                    f"failed to start local MSA command: {command[0]}"
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
            return MsaPayload(
                paired.read_text(encoding="utf-8"),
                unpaired.read_text(encoding="utf-8"),
                {"command": list(self.command), "version": self.version},
            )


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


class RemoteMMseqs2Client:
    """Minimal ColabFold-compatible MMseqs2 ticket client using stdlib HTTP."""

    name = "remote-mmseqs2"
    complex_pairing_mode = COMPLEX_PAIRING_MODE

    def __init__(
        self,
        host_url: str,
        *,
        version: str,
        username: str | None = None,
        password: str | None = None,
        auth_headers: Mapping[str, str] | None = None,
        transport: HttpTransport = _urllib_transport,
        timeout: float = 30.0,
        poll_interval: float = 5.0,
        max_wait_seconds: float = 3600.0,
    ) -> None:
        if (username is None) != (password is None):
            raise ValueError("remote MSA basic auth requires username and password")
        if username is not None and auth_headers:
            raise ValueError("basic auth and auth_headers are mutually exclusive")
        if not host_url.strip() or not version:
            raise ValueError("remote MSA host URL and version are required")
        if timeout <= 0 or poll_interval < 0 or max_wait_seconds <= 0:
            raise ValueError("remote MSA timeout values are invalid")
        self.host_url = host_url.rstrip("/")
        self.version = version
        self.transport = transport
        self.timeout = timeout
        self.poll_interval = poll_interval
        self.max_wait_seconds = max_wait_seconds
        # The public ColabFold endpoint is a shared, free service whose operators
        # ask clients to identify themselves. This one used to say
        # "protenix-jax", which was true when the search lived in that port and
        # is now the name of one of six callers.
        try:
            from foldjax import __version__ as foldjax_version
        except ImportError:  # pragma: no cover - the package is always present
            foldjax_version = "0"
        self.headers = {
            "User-Agent": f"foldjax/{foldjax_version}",
            **dict(auth_headers or {}),
        }
        if username is not None:
            token = base64.b64encode(f"{username}:{password}".encode()).decode()
            self.headers["Authorization"] = f"Basic {token}"

    def _request(self, method: str, path: str, data: bytes | None = None) -> bytes:
        """Send one request, waiting out a busy server rather than failing on it.

        HTTP 429 is how this API says "at capacity, come back", and a public
        MMseqs2 server is at capacity often. Treating it as an error made a
        search fail on a condition whose entire meaning is that it is temporary,
        and lost the queue position of every sequence after it. The wait is
        bounded by ``max_wait_seconds``, the same budget the poll loop uses, so
        a server that never recovers still ends the search.
        """
        url = f"{self.host_url}/{path.lstrip('/')}"
        deadline = time.monotonic() + self.max_wait_seconds
        delay = max(self.poll_interval, 1.0)
        while True:
            response = self.transport(method, url, data, self.headers, self.timeout)
            if response.status != 429:
                break
            if time.monotonic() + delay >= deadline:
                raise SearchError(
                    f"remote MSA request {path!r} was rate-limited for the whole "
                    f"{self.max_wait_seconds:.0f}s budget"
                )
            time.sleep(delay)
            delay = min(delay * 2, 60.0)
        if response.status < 200 or response.status >= 300:
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

    def _run(self, sequence: str, *, paired: bool) -> tuple[str, str]:
        names = (
            ("pair.a3m",)
            if paired
            else (
                "uniref.a3m",
                "bfd.mgnify30.metaeuk30.smag30.a3m",
            )
        )
        texts, job_id = self._submit(
            f">101\n{sequence}\n",
            mode="paircomplete" if paired else "env",
            endpoint="ticket/pair" if paired else "ticket/msa",
            names=names,
        )
        return "".join(text.replace("\x00", "") for text in texts), job_id

    def search_complex(self, sequences: Sequence[str]) -> ComplexPairPayload:
        """Pair a complex in one job, the way OpenFold3 v0.5.0 does.

        The sequences are submitted together as queries 101, 102, ... and the
        returned ``pair.a3m`` is split back into one block per query.
        """
        query = "".join(
            f">{101 + index}\n{sequence}\n" for index, sequence in enumerate(sequences)
        )
        (text,), job_id = self._submit(
            query,
            mode=COMPLEX_PAIRING_MODE,
            endpoint="ticket/pair",
            names=("pair.a3m",),
        )
        blocks = _split_colabfold_a3m(text, "paired MSA")
        numbers = [101 + index for index in range(len(sequences))]
        missing = [number for number in numbers if number not in blocks]
        if missing:
            raise SearchError(f"remote paired MSA has no block for queries {missing}")
        return ComplexPairPayload(
            tuple(blocks[number] for number in numbers),
            {"paired_job_id": job_id, "mode": COMPLEX_PAIRING_MODE},
        )

    def _submit(
        self, query: str, *, mode: str, endpoint: str, names: Sequence[str]
    ) -> tuple[list[str], str]:
        """Run one ticket to completion and return the named archive members."""
        data = urllib.parse.urlencode({"q": query, "mode": mode}).encode()
        response = self._json("POST", endpoint, data)
        state = response["status"]
        job_id = response.get("id")
        if state in {"ERROR", "MAINTENANCE"}:
            raise SearchError(f"remote MSA submission ended with status {state!r}")
        if not isinstance(job_id, str) or not job_id:
            raise SearchError("remote MSA submission response is missing a job id")
        deadline = time.monotonic() + self.max_wait_seconds
        while state in {"UNKNOWN", "RUNNING", "PENDING", "RATELIMIT"}:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"remote MSA search {job_id} timed out")
            if self.poll_interval:
                time.sleep(self.poll_interval)
            response = self._json("GET", f"ticket/{job_id}")
            state = response["status"]
        if state != "COMPLETE":
            raise SearchError(f"remote MSA search {job_id} ended with status {state!r}")
        archive = self._request("GET", f"result/download/{job_id}")
        chunks: list[str] = []
        try:
            with tarfile.open(fileobj=io.BytesIO(archive), mode="r:*") as tar:
                for name in names:
                    member = tar.getmember(name)
                    if not member.isfile():
                        raise SearchError(
                            f"remote MSA archive entry is not a file: {name}"
                        )
                    extracted = tar.extractfile(member)
                    if extracted is None:
                        raise SearchError(f"remote MSA archive cannot read: {name}")
                    chunks.append(extracted.read().decode("utf-8"))
        except (tarfile.TarError, KeyError, UnicodeDecodeError) as exc:
            raise SearchError(
                "remote MSA returned an invalid or incomplete archive"
            ) from exc
        return chunks, job_id

    def search(self, sequence: str) -> MsaPayload:
        unpaired, unpaired_job = self._run(sequence, paired=False)
        paired, paired_job = self._run(sequence, paired=True)
        return MsaPayload(
            paired,
            unpaired,
            {"paired_job_id": paired_job, "unpaired_job_id": unpaired_job},
        )
