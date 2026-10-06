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
import errno
import hashlib
import http.client
import io
import json
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from foldjax.redaction import redact


class SearchError(RuntimeError):
    """An MSA provider returned an unusable or incomplete result."""


class _UnsplittableTicketError(SearchError):
    """A shared ticket's result did not separate back into its queries."""


#: Set to ``1`` to let a search or download use plain ``http://`` beyond this
#: machine. Off by default: the query sequence and any credential would cross
#: the network in clear text. Loopback addresses are always allowed.
ALLOW_INSECURE_HTTP_ENV = "FOLDJAX_ALLOW_INSECURE_HTTP"
_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
#: The most one remote response, or one member of a result archive, may hold.
#: Generous for the largest alignment or mmCIF a server returns; a server that
#: sends more is refused before it exhausts memory.
MAX_REMOTE_BYTES = 1 << 30
#: Queries per shared unpaired ticket (`RemoteMMseqs2Client.search_many`).
MAX_QUERIES_PER_TICKET = 16
#: A server job id is spliced into the next request's URL path.
_JOB_ID = re.compile(r"[A-Za-z0-9_-]+")


def require_https(url: str, *, what: str) -> str:
    """Return ``url`` when it is https (or loopback / opted-in plain http)."""
    try:
        parts = urllib.parse.urlsplit(url.strip())
        host = parts.hostname
    except ValueError as exc:
        raise ValueError(f"{what} is not a valid URL") from exc
    scheme = parts.scheme.lower()
    if scheme == "https" and host:
        return url
    if scheme == "http" and host:
        opt_in = os.environ.get(ALLOW_INSECURE_HTTP_ENV, "").strip().lower()
        if host in _LOOPBACK_HOSTS or opt_in in {"1", "true", "yes", "on"}:
            return url
    raise ValueError(
        f"{what} must be an https:// URL, got {scheme or 'no'} scheme for host "
        f"{host or '(none)'}; set {ALLOW_INSECURE_HTTP_ENV}=1 to allow plain http"
    )


def validate_job_id(job_id: object) -> str:
    """A server-issued job id, checked before it becomes part of a URL."""
    if not isinstance(job_id, str) or not _JOB_ID.fullmatch(job_id):
        raise SearchError("remote MSA submission returned an invalid job id")
    return job_id


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


# Cache entries: one directory per key, published by renaming a staged sibling
# into place. A shared store sees concurrent runs and the occasional damaged
# entry, and the helpers below are what every search cache does about both.
# The content hashes in provenance.json catch damage, not a hostile writer:
# a cache other accounts can write is trusted by everyone who reads it.


@contextmanager
def cache_key_lock(cache_dir: Path, cache_key: str) -> Iterator[None]:
    """Serialize work on one cache entry across processes, best effort.

    Two runs on one sequence used to search twice and race to publish. Held
    around check, search and publish, the lock makes the second run wait and
    then read the first one's entry. It is advisory: where ``flock`` is not
    honoured (some network filesystems) the publish below still resolves the
    race, so a lock that cannot be taken is never a failure.
    """
    try:
        import fcntl
    except ImportError:  # pragma: no cover - POSIX only
        yield
        return
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        handle = os.open(
            cache_dir / f".{cache_key}.lock",
            os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0),
            0o666,
        )
    except OSError:
        yield
        return
    try:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX)
        except OSError:
            pass
        yield
    finally:
        os.close(handle)


def staging_directory(cache_dir: Path, cache_key: str) -> Path:
    """A fresh hidden sibling to build an entry in, with the umask's mode.

    ``tempfile.mkdtemp`` makes it 0700, which the published entry kept, so a
    store shared by a group could not be read by the rest of it.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    while True:
        path = cache_dir / f".{cache_key}.{os.urandom(6).hex()}"
        try:
            os.mkdir(path)
        except FileExistsError:
            continue
        return path


def publish_directory(staged: Path, directory: Path) -> bool:
    """Rename ``staged`` to ``directory``; ``False`` when another run got there.

    Renaming onto an existing *non-empty* directory fails with ``ENOTEMPTY``,
    not ``FileExistsError``, so the loser of a race used to fail its run.
    """
    try:
        staged.rename(directory)
    except OSError as exc:
        if exc.errno not in (errno.EEXIST, errno.ENOTEMPTY):
            raise
        shutil.rmtree(staged, ignore_errors=True)
        return False
    return True


def quarantine_entry(directory: Path, cache_key: str, reason: object) -> None:
    """Move a damaged entry aside so the next lookup searches again.

    A hash mismatch or a missing file used to fail every later run of that
    sequence. The entry is kept, once, as ``.<key>.damaged`` for inspection.
    """
    import warnings

    aside = directory.parent / f".{cache_key}.damaged"
    shutil.rmtree(aside, ignore_errors=True)
    try:
        directory.rename(aside)
    except OSError:
        shutil.rmtree(directory, ignore_errors=True)
    warnings.warn(
        f"search cache entry {directory} was damaged ({reason}); moved aside to "
        f"{aside.name} and searching again",
        UserWarning,
        stacklevel=3,
    )


#: What one sequence's search may fail with and leave the others standing.
_SEARCH_FAILURES = (SearchError, TimeoutError, OSError, ValueError)


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
        temp_dir = staging_directory(self.cache_dir, cache_key)
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
                json.dumps(redact(provenance), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            if not publish_directory(temp_dir, directory):
                cached = self._cached(directory, sequence, cache_key)
                if cached is None:
                    raise AssertionError("cache disappeared during materialization")
                return cached
        except Exception:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise
        return self._paths(directory)

    def _usable(
        self, directory: Path, sequence: str, cache_key: str
    ) -> dict[str, str] | None:
        """The cached entry, or ``None`` after moving a damaged one aside."""
        try:
            return self._cached(directory, sequence, cache_key)
        except (SearchError, UnicodeDecodeError) as error:
            quarantine_entry(directory, cache_key, error)
            return None

    def search(self, sequences: Sequence[str]) -> list[dict[str, str]]:
        """One entry per sequence; the first sequence that failed raises."""
        results = self.search_each(sequences)
        for result in results:
            if isinstance(result, Exception):
                raise result
        return results  # type: ignore[return-value]

    def search_each(
        self, sequences: Sequence[str]
    ) -> list[dict[str, str] | Exception]:
        """One entry per sequence, or the error that sequence's search raised.

        A failure is the failing sequence's own: the others keep the alignment
        they found. Sequences missing from the cache go to the backend's
        ``search_many`` together when it has one (a shared ticket), else one
        ``search`` at a time; every miss is searched under its cache key's lock.
        """
        normalized = [_normalize_sequence(sequence) for sequence in sequences]
        if not normalized:
            raise ValueError("at least one protein sequence is required")
        identities = {
            sequence: self._identity(sequence) for sequence in dict.fromkeys(normalized)
        }
        outcomes: dict[str, dict[str, str] | Exception] = {}
        missing: list[str] = []
        for sequence, (cache_key, _) in identities.items():
            try:
                cached = self._cached(self.cache_dir / cache_key, sequence, cache_key)
            except (SearchError, UnicodeDecodeError):
                cached = None  # set aside under the lock below
            if cached is not None:
                outcomes[sequence] = cached
            else:
                missing.append(sequence)
        search_many = getattr(self.backend, "search_many", None)
        if callable(search_many) and len(missing) > 1:
            self._search_together(missing, identities, outcomes, search_many)
        else:
            for sequence in missing:
                cache_key, identity = identities[sequence]
                directory = self.cache_dir / cache_key
                try:
                    with cache_key_lock(self.cache_dir, cache_key):
                        cached = self._usable(directory, sequence, cache_key)
                        outcomes[sequence] = cached or self._materialize(
                            directory,
                            sequence,
                            cache_key,
                            identity,
                            self.backend.search(sequence),
                        )
                except _SEARCH_FAILURES as error:
                    outcomes[sequence] = error
        return [outcomes[sequence] for sequence in normalized]

    def _search_together(
        self,
        missing: list[str],
        identities: Mapping[str, tuple[str, dict[str, Any]]],
        outcomes: dict[str, dict[str, str] | Exception],
        search_many: Callable[[list[str]], list[MsaPayload | Exception]],
    ) -> None:
        with ExitStack() as locks:
            # Sorted, so two runs taking overlapping sets cannot deadlock.
            for cache_key in sorted(identities[sequence][0] for sequence in missing):
                locks.enter_context(cache_key_lock(self.cache_dir, cache_key))
            still: list[str] = []
            for sequence in missing:
                cache_key = identities[sequence][0]
                cached = self._usable(self.cache_dir / cache_key, sequence, cache_key)
                if cached is not None:
                    outcomes[sequence] = cached
                else:
                    still.append(sequence)
            if not still:
                return
            try:
                payloads: list[MsaPayload | Exception] = list(search_many(still))
            except _SEARCH_FAILURES as error:
                payloads = [error] * len(still)
            if len(payloads) != len(still):
                raise SearchError(
                    f"MSA backend returned {len(payloads)} results for "
                    f"{len(still)} sequences"
                )
            for sequence, payload in zip(still, payloads, strict=True):
                if isinstance(payload, Exception):
                    outcomes[sequence] = payload
                    continue
                cache_key, identity = identities[sequence]
                try:
                    outcomes[sequence] = self._materialize(
                        self.cache_dir / cache_key,
                        sequence,
                        cache_key,
                        identity,
                        payload,
                    )
                except _SEARCH_FAILURES as error:
                    outcomes[sequence] = error

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
        with cache_key_lock(self.cache_dir, cache_key):
            try:
                paths = self._complex_cached(directory, unique, cache_key)
            except (SearchError, UnicodeDecodeError) as error:
                quarantine_entry(directory, cache_key, error)
                paths = None
            paths = paths or self._complex_materialize(
                directory,
                unique,
                cache_key,
                identity,
                self.backend.search_complex(unique),
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
        temp_dir = staging_directory(self.cache_dir, cache_key)
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
                json.dumps(redact(provenance), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            if not publish_directory(temp_dir, directory):
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
        temporary = staging_directory(self.cache_dir, cache_key)
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
                json.dumps(redact(provenance), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            if not publish_directory(temporary, directory):
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
            with cache_key_lock(self.cache_dir, cache_key):
                try:
                    cached = self._cached(directory, sequence, cache_key)
                except SearchError as error:
                    quarantine_entry(directory, cache_key, error)
                    cached = None
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
        with _NO_REDIRECT_OPENER.open(request, timeout=timeout) as response:
            return HttpResponse(response.status, _read_capped(response, url))
    except urllib.error.HTTPError as exc:
        return HttpResponse(exc.code, _read_capped(exc, url))


class _RefuseRedirect(urllib.request.HTTPRedirectHandler):
    """Return a redirect as its 3xx status instead of following it.

    urllib's own handler resends every header, ``Authorization`` and API keys
    included, to whatever host the ``Location`` names, ``http://`` included.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


_NO_REDIRECT_OPENER = urllib.request.build_opener(_RefuseRedirect)


def _read_capped(response: Any, url: str) -> bytes:
    body = response.read(MAX_REMOTE_BYTES + 1)
    if len(body) > MAX_REMOTE_BYTES:
        host = urllib.parse.urlsplit(url).hostname
        raise SearchError(
            f"response from {host} exceeds {MAX_REMOTE_BYTES} bytes; refused"
        )
    return body


#: Seconds one remote search, or one request's retries, may take before it is
#: abandoned. Overridden by ``FOLDJAX_MSA_MAX_WAIT_SECONDS``.
MAX_WAIT_ENV = "FOLDJAX_MSA_MAX_WAIT_SECONDS"
DEFAULT_MAX_WAIT_SECONDS = 3600.0
#: Answers that mean "try again shortly": capacity (429) and a gateway or
#: server that is restarting. Anything else is the server's real answer.
_TRANSIENT_STATUSES = frozenset({429, 500, 502, 503, 504})
#: How often a connection failure or 5xx is retried; 429 is waited out for the
#: whole budget instead, since it is the server asking for exactly that.
_TRANSIENT_RETRIES = 5


def resolve_max_wait_seconds(value: float | None = None) -> float:
    """``value``, else ``FOLDJAX_MSA_MAX_WAIT_SECONDS``, else one hour."""
    if value is not None:
        return float(value)
    raw = os.environ.get(MAX_WAIT_ENV, "").strip()
    if not raw:
        return DEFAULT_MAX_WAIT_SECONDS
    try:
        parsed = float(raw)
    except ValueError as exc:
        raise ValueError(f"{MAX_WAIT_ENV} must be a number of seconds") from exc
    if parsed <= 0:
        raise ValueError(f"{MAX_WAIT_ENV} must be positive")
    return parsed


def _send(
    transport: HttpTransport,
    method: str,
    url: str,
    data: bytes | None,
    headers: Mapping[str, str],
    timeout: float,
    *,
    budget: float,
    first_delay: float,
    label: str,
) -> HttpResponse:
    """One request, retried through the failures that are worth waiting out.

    A dropped connection, a truncated body (``http.client.IncompleteRead``,
    which is not an ``OSError``) or a 5xx is retried a few times with
    exponential backoff; 429 is waited out for the whole ``budget``. What is
    still failing at the end is a `SearchError` that names the server, so a
    caller's ``except SearchError`` sees it rather than a stray exception type.
    """
    from foldjax import progress

    host = urllib.parse.urlsplit(url).hostname or url
    deadline = time.monotonic() + budget
    delay = max(first_delay, 1.0)
    retries = 0
    while True:
        try:
            response: HttpResponse | None = transport(
                method, url, data, headers, timeout
            )
        except (OSError, http.client.HTTPException) as exc:
            response, failure = None, f"{type(exc).__name__}: {exc}"
        else:
            assert response is not None
            if response.status not in _TRANSIENT_STATUSES:
                return response
            failure = f"HTTP {response.status}"
        limited = response is not None and response.status == 429
        if not limited:
            retries += 1
        if time.monotonic() + delay >= deadline:
            if limited:
                raise SearchError(
                    f"{label} was rate-limited for the whole {budget:.0f}s "
                    f"budget by {host}"
                )
            raise SearchError(
                f"{label} to {host} failed ({failure}) within {budget:.0f}s"
            )
        if retries > _TRANSIENT_RETRIES:
            raise SearchError(
                f"{label} to {host} failed after {retries} attempts ({failure})"
            )
        progress.message(
            f"  {host}: {label} "
            + ("rate-limited" if limited else f"failed ({failure})")
            + f"; retrying in {delay:.0f}s"
        )
        time.sleep(delay)
        delay = min(delay * 2, 60.0)


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
        max_wait_seconds: float | None = None,
    ) -> None:
        max_wait_seconds = resolve_max_wait_seconds(max_wait_seconds)
        if (username is None) != (password is None):
            raise ValueError("remote MSA basic auth requires username and password")
        if username is not None and auth_headers:
            raise ValueError("basic auth and auth_headers are mutually exclusive")
        if not host_url.strip() or not version:
            raise ValueError("remote MSA host URL and version are required")
        require_https(host_url, what="remote MSA host URL")
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
        a server that never recovers still ends the search. A dropped
        connection or a 5xx is retried a few times the same way (`_send`).
        """
        url = f"{self.host_url}/{path.lstrip('/')}"
        response = _send(
            self.transport,
            method,
            url,
            data,
            self.headers,
            self.timeout,
            budget=self.max_wait_seconds,
            first_delay=self.poll_interval,
            label=f"remote MSA request {path!r}",
        )
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
        from foldjax import progress

        host = urllib.parse.urlsplit(self.host_url).hostname or self.host_url
        data = urllib.parse.urlencode({"q": query, "mode": mode}).encode()
        deadline = time.monotonic() + self.max_wait_seconds
        response = self._json("POST", endpoint, data)
        state = response["status"]
        # RATELIMIT or UNKNOWN in answer to a submission means the ticket was
        # not taken; ColabFold's own client, and Boltz's, submit again. Polling
        # the id such an answer carries waited on a job that did not exist.
        delay = max(self.poll_interval, 1.0)
        while state in {"UNKNOWN", "RATELIMIT"}:
            if time.monotonic() + delay >= deadline:
                raise SearchError(
                    f"remote MSA server {host} answered {state} to every "
                    f"submission for {self.max_wait_seconds:.0f}s"
                )
            progress.message(
                f"  {host}: MSA submission {state.lower()}; resubmitting in "
                f"{delay:.0f}s"
            )
            time.sleep(delay)
            delay = min(delay * 2, 60.0)
            response = self._json("POST", endpoint, data)
            state = response["status"]
        job_id = response.get("id")
        if state in {"ERROR", "MAINTENANCE"}:
            raise SearchError(
                f"remote MSA submission to {host} ended with status {state!r}"
            )
        if not isinstance(job_id, str) or not job_id:
            raise SearchError("remote MSA submission response is missing a job id")
        validate_job_id(job_id)
        while state in {"UNKNOWN", "RUNNING", "PENDING", "RATELIMIT"}:
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"remote MSA search {job_id} on {host} did not finish within "
                    f"{self.max_wait_seconds:.0f}s (set {MAX_WAIT_ENV} to wait longer)"
                )
            if self.poll_interval:
                time.sleep(self.poll_interval)
            response = self._json("GET", f"ticket/{job_id}")
            state = response["status"]
        if state != "COMPLETE":
            raise SearchError(
                f"remote MSA search {job_id} on {host} ended with status {state!r}"
            )
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
                    if member.size > MAX_REMOTE_BYTES:
                        raise SearchError(
                            f"remote MSA archive entry {name} exceeds "
                            f"{MAX_REMOTE_BYTES} bytes; refused"
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

    def search_many(self, sequences: Sequence[str]) -> list[MsaPayload | Exception]:
        """`search` for several sequences, the unpaired search in shared tickets.

        Twenty sequences used to cost forty serial tickets. ColabFold's own
        client and Boltz's submit every query of a run in one ``env`` ticket;
        that search treats each query on its own, so a query's block in a
        shared ticket is what its own ticket returns (an inference from those
        clients, not something this code can check). Each block's query header
        is renumbered to the ``>101`` a single-query ticket writes, so the
        cached bytes do not depend on the batching. The per-chain
        ``paircomplete`` search stays one ticket per sequence: what it pairs
        depends on which queries share the ticket.

        One entry per sequence: its payload, or the error its search raised.
        """
        outcomes: list[MsaPayload | Exception] = []
        for start in range(0, len(sequences), MAX_QUERIES_PER_TICKET):
            chunk = list(sequences[start : start + MAX_QUERIES_PER_TICKET])
            try:
                unpaired, unpaired_job = self._run_many(chunk)
            except _UnsplittableTicketError:
                # A server whose shared result cannot be split per query is
                # asked one query at a time instead, as before batching.
                for sequence in chunk:
                    try:
                        outcomes.append(self.search(sequence))
                    except (SearchError, TimeoutError, OSError) as error:
                        outcomes.append(error)
                continue
            except (SearchError, TimeoutError, OSError) as error:
                outcomes.extend([error] * len(chunk))
                continue
            for sequence, text in zip(chunk, unpaired, strict=True):
                try:
                    paired, paired_job = self._run(sequence, paired=True)
                except (SearchError, TimeoutError, OSError) as error:
                    outcomes.append(error)
                    continue
                outcomes.append(
                    MsaPayload(
                        paired,
                        text,
                        {"paired_job_id": paired_job, "unpaired_job_id": unpaired_job},
                    )
                )
        return outcomes

    def _run_many(self, sequences: Sequence[str]) -> tuple[list[str], str]:
        """One ``env`` ticket for several queries, split back per query."""
        names = ("uniref.a3m", "bfd.mgnify30.metaeuk30.smag30.a3m")
        texts, job_id = self._submit(
            "".join(
                f">{101 + index}\n{sequence}\n"
                for index, sequence in enumerate(sequences)
            ),
            mode="env",
            endpoint="ticket/msa",
            names=names,
        )
        try:
            members = [
                _split_colabfold_a3m(text, name) for text, name in zip(texts, names)
            ]
        except SearchError as error:
            raise _UnsplittableTicketError(str(error)) from error
        unpaired = []
        for index in range(len(sequences)):
            number = 101 + index
            parts = []
            for blocks, name in zip(members, names, strict=True):
                if number not in blocks:
                    raise _UnsplittableTicketError(
                        f"remote MSA {name} has no block for query {number}"
                    )
                block = blocks[number]
                header, _, rest = block.partition("\n")
                parts.append(f">101\n{rest}" if header == f">{number}" else block)
            unpaired.append("".join(parts))
        return unpaired, job_id
