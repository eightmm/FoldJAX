"""Template hits from a ColabFold MMseqs2 server, and the structures they name.

The network half of ``templates='auto'``; `foldjax.template_search` turns the
hits into per-model templates. Like `foldjax.search.msa` this module imports
nothing outside the standard library, so the client and its caches can be
exercised -- and mocked -- without a structure parser or an aligner.

Where the hits come from follows OpenFold3 v0.5.0, the one carried upstream
that searches templates through this server: it submits the ordinary
``ticket/msa`` job and reads the ``pdb70.m8`` member of the result archive
(``colabfold_msa_server.py:866, 896-916``), which the server writes whether or
not templates were asked for. The structures are then downloaded from RCSB by
PDB id (upstream: biotite ``rcsb.fetch`` in
``pipelines/preprocessing/template.py:2208-2213``). Hit ids carry the
*author* chain (``1abc_A``).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from foldjax.redaction import redact
from foldjax.search.msa import (
    MAX_REMOTE_BYTES,
    HttpTransport,
    RemoteMMseqs2Client,
    SearchError,
    _normalize_sequence,
    _sha256,
    _urllib_transport,
    require_https,
)

#: The archive member the ColabFold server writes its PDB70 hits to.
HITS_MEMBER = "pdb70.m8"

#: The result columns of a BLAST-tabular ``.m8`` row, as OpenFold3 names them
#: (``io/sequence/template.py`` ``M8Parser``); a 13th column is a CIGAR string.
_M8_COLUMNS = 12


@dataclass(frozen=True)
class TemplateHit:
    """One ``.m8`` row. Positions are the file's own, 1-based and inclusive."""

    rank: int
    query: str
    target: str
    pdb_id: str
    chain_id: str
    identity: float
    alignment_length: int
    query_start: int
    query_end: int
    target_start: int
    target_end: int
    e_value: float
    bit_score: float

    def summary(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "target": self.target,
            "identity": self.identity,
            "e_value": self.e_value,
            "bit_score": self.bit_score,
        }


def parse_m8(text: str) -> list[TemplateHit]:
    """Parse ``.m8`` hits, ordered by e-value as OpenFold3's ``M8Parser`` does.

    The sort is stable, so hits with equal e-values keep the server's order.
    A row whose target is not ``<pdb id>_<chain>`` cannot name a structure
    and is refused rather than skipped: it means the file is not PDB70 hits.
    """
    hits: list[tuple[float, int, TemplateHit]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        columns = line.rstrip("\n").split("\t")
        if len(columns) < _M8_COLUMNS:
            columns = line.split()
        if len(columns) < _M8_COLUMNS:
            raise ValueError(
                f"template hits line {line_number} has {len(columns)} columns; "
                f"an .m8 row has {_M8_COLUMNS} or 13"
            )
        target = columns[1].strip()
        pdb_id, separator, chain_id = target.partition("_")
        if not separator or len(pdb_id) != 4 or not pdb_id.isalnum() or not chain_id:
            raise ValueError(
                f"template hit {target!r} on line {line_number} is not "
                "'<pdb id>_<chain>'"
            )
        try:
            hit = TemplateHit(
                rank=0,
                query=columns[0].strip(),
                target=target,
                pdb_id=pdb_id.lower(),
                chain_id=chain_id,
                identity=float(columns[2]),
                alignment_length=int(columns[3]),
                query_start=int(columns[6]),
                query_end=int(columns[7]),
                target_start=int(columns[8]),
                target_end=int(columns[9]),
                e_value=float(columns[10]),
                bit_score=float(columns[11]),
            )
        except ValueError as error:
            raise ValueError(
                f"template hits line {line_number} has a non-numeric field: {error}"
            ) from error
        hits.append((hit.e_value, len(hits), hit))
    hits.sort(key=lambda item: (item[0], item[1]))
    return [
        TemplateHit(**{**hit.__dict__, "rank": rank})
        for rank, (_, _, hit) in enumerate(hits)
    ]


@dataclass(frozen=True)
class TemplateHitsPayload:
    hits: str
    source: Mapping[str, Any] = field(default_factory=dict)


class TemplateHitsBackend(Protocol):
    name: str
    version: str

    def search(self, sequence: str) -> TemplateHitsPayload: ...


class RemoteTemplateHitsClient:
    """PDB70 hits from a ColabFold-compatible MMseqs2 server."""

    name = "remote-mmseqs2-pdb70"

    def __init__(
        self,
        host_url: str,
        *,
        version: str,
        transport: HttpTransport = _urllib_transport,
        timeout: float = 30.0,
        poll_interval: float = 5.0,
        max_wait_seconds: float = 3600.0,
    ) -> None:
        self._client = RemoteMMseqs2Client(
            host_url,
            version=version,
            transport=transport,
            timeout=timeout,
            poll_interval=poll_interval,
            max_wait_seconds=max_wait_seconds,
        )
        self.host_url = self._client.host_url
        self.version = version

    def search(self, sequence: str) -> TemplateHitsPayload:
        (hits,), job_id = self._client._submit(
            f">101\n{sequence}\n",
            mode="env",
            endpoint="ticket/msa",
            names=(HITS_MEMBER,),
        )
        return TemplateHitsPayload(
            hits, {"job_id": job_id, "mode": "env", "member": HITS_MEMBER}
        )


class LocalTemplateHitsClient:
    """Run a local wrapper that writes ``pdb70.m8`` for one query.

    The wrapper receives ``--input <fasta> --output <directory>``, the contract
    `foldjax.search.msa.LocalMsaClient` uses, and nothing leaves the machine
    on this path.
    """

    name = "local-template"

    def __init__(
        self,
        command: Sequence[str],
        *,
        version: str,
        runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    ) -> None:
        if not command or not version:
            raise ValueError("local template command and version are required")
        self.command = tuple(command)
        self.version = version
        self._runner = runner

    def search(self, sequence: str) -> TemplateHitsPayload:
        with tempfile.TemporaryDirectory(prefix="foldjax-templates-") as raw_dir:
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
                    f"failed to start local template command: {command[0]}"
                ) from exc
            if completed.returncode:
                detail = (completed.stderr or completed.stdout or "").strip()[-500:]
                raise SearchError(
                    "local template command exited with "
                    f"{completed.returncode}: {detail}"
                )
            result = output / HITS_MEMBER
            if not result.is_file():
                raise SearchError(f"local template command did not produce: {result}")
            return TemplateHitsPayload(
                result.read_text(encoding="utf-8"),
                {"command": list(self.command), "version": self.version},
            )


class TemplateHitsPipeline:
    """Cache hits by sequence plus search provenance, as the MSA cache does."""

    def __init__(
        self,
        cache_dir: str | Path,
        backend: TemplateHitsBackend,
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
                "template search options must be JSON-serializable"
            ) from exc

    def _identity(self, sequence: str) -> tuple[str, dict[str, Any]]:
        identity = {
            "schema_version": 1,
            "kind": "template_hits",
            "sequence": sequence,
            "backend": {"name": self.backend.name, "version": self.backend.version},
            "options": self.options,
        }
        canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"))
        return _sha256(canonical.encode()), identity

    @staticmethod
    def _paths(directory: Path) -> dict[str, str]:
        return {
            "hitsPath": str((directory / HITS_MEMBER).resolve()),
            "provenancePath": str((directory / "provenance.json").resolve()),
        }

    def _cached(self, directory: Path, cache_key: str) -> dict[str, str] | None:
        if not directory.exists():
            return None
        provenance_path = directory / "provenance.json"
        hits_path = directory / HITS_MEMBER
        if not provenance_path.is_file() or not hits_path.is_file():
            raise SearchError(f"template hits cache is incomplete: {directory}")
        try:
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SearchError(
                f"invalid template hits cache provenance: {provenance_path}"
            ) from exc
        if provenance.get("cache_key") != cache_key:
            raise SearchError(
                f"template hits cache provenance key mismatch: {provenance_path}"
            )
        expected = provenance.get("files", {}).get(HITS_MEMBER, {}).get("sha256")
        if _sha256(hits_path.read_bytes()) != expected:
            raise SearchError(f"template hits cache content hash mismatch: {hits_path}")
        return self._paths(directory)

    def search(self, sequence: str) -> dict[str, str]:
        """Return ``hitsPath`` and ``provenancePath`` for one protein sequence."""
        normalized = _normalize_sequence(sequence)
        cache_key, identity = self._identity(normalized)
        directory = self.cache_dir / cache_key
        cached = self._cached(directory, cache_key)
        if cached is not None:
            return cached
        payload = self.backend.search(normalized)
        if not isinstance(payload.hits, str):
            raise SearchError("template hits response is missing")
        try:
            parse_m8(payload.hits)
        except ValueError as exc:
            # Parsed at the boundary so that a malformed answer never enters
            # the cache, where every later run would trip over it.
            raise SearchError(f"template hits are malformed: {exc}") from exc
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{cache_key}.", dir=self.cache_dir))
        try:
            raw = payload.hits.encode()
            (temporary / HITS_MEMBER).write_bytes(raw)
            provenance = {
                **identity,
                "cache_key": cache_key,
                "sequence_sha256": _sha256(normalized.encode()),
                "source": dict(payload.source),
                "files": {HITS_MEMBER: {"sha256": _sha256(raw), "bytes": len(raw)}},
            }
            (temporary / "provenance.json").write_text(
                json.dumps(redact(provenance), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            try:
                temporary.rename(directory)
            except FileExistsError:
                shutil.rmtree(temporary)
                cached = self._cached(directory, cache_key)
                if cached is None:
                    raise AssertionError("template hits cache disappeared") from None
                return cached
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return self._paths(directory)


#: RCSB's download endpoint, the one biotite ``rcsb.fetch`` reads.
DEFAULT_STRUCTURE_URL = "https://files.rcsb.org/download"


class StructureStore:
    """Template mmCIFs by PDB id: a local mirror first, then a cached download.

    A PDB id is the only thing sent on this path, never a sequence. Without a
    ``base_url`` nothing is downloaded and a structure missing from both the
    mirror and the cache is a `SearchError` for that hit alone.
    """

    def __init__(
        self,
        cache_dir: str | Path,
        *,
        local_dir: str | Path | None = None,
        base_url: str | None = DEFAULT_STRUCTURE_URL,
        transport: HttpTransport = _urllib_transport,
        timeout: float = 60.0,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.local_dir = Path(local_dir) if local_dir else None
        self.base_url = base_url.rstrip("/") if base_url else None
        if self.base_url is not None:
            require_https(self.base_url, what="template structure URL")
        self.transport = transport
        self.timeout = timeout
        try:
            from foldjax import __version__ as foldjax_version
        except ImportError:  # pragma: no cover - the package is always present
            foldjax_version = "0"
        self.headers = {"User-Agent": f"foldjax/{foldjax_version}"}

    def describe(self) -> dict[str, Any]:
        return {
            "local_dir": str(self.local_dir) if self.local_dir else None,
            "url": self.base_url,
            "cache_dir": str(self.cache_dir),
        }

    @staticmethod
    def _source_key(source: str) -> str:
        """A directory name per source, so two servers never share a file."""
        readable = "".join(c if c.isalnum() else "-" for c in source)[-40:]
        return f"{readable.strip('-')}-{_sha256(source.encode())[:12]}"

    def _local(self, pdb_id: str) -> Path | None:
        """``1abc.cif`` or ``1abc.cif.gz``, flat or wwPDB-divided (``ab/``)."""
        if self.local_dir is None:
            return None
        for directory in (self.local_dir, self.local_dir / pdb_id[1:3]):
            for name in (pdb_id, pdb_id.upper()):
                for suffix in (".cif", ".cif.gz"):
                    candidate = directory / f"{name}{suffix}"
                    if candidate.is_file():
                        return candidate
        return None

    def _publish(self, directory: Path, pdb_id: str, body: bytes) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / f"{pdb_id}.cif"
        with tempfile.NamedTemporaryFile(
            dir=directory, prefix=f".{pdb_id}.", suffix=".cif", delete=False
        ) as staged:
            staged.write(body)
        os.replace(staged.name, target)
        return target.resolve()

    def path(self, pdb_id: str) -> Path:
        """A readable ``.cif`` for ``pdb_id``, fetching it once if needed."""
        pdb_id = pdb_id.strip().lower()
        if len(pdb_id) != 4 or not pdb_id.isalnum():
            raise SearchError(f"not a PDB id: {pdb_id!r}")
        local = self._local(pdb_id)
        if local is not None and local.suffix != ".gz":
            return local.resolve()
        if local is not None:
            # Readers downstream take a plain file, so a compressed mirror
            # entry is unpacked once into the cache, beside its source's key.
            directory = self.cache_dir / self._source_key(str(local.parent.resolve()))
            unpacked = directory / f"{pdb_id}.cif"
            if (
                unpacked.is_file()
                and unpacked.stat().st_mtime >= local.stat().st_mtime
                and _cached_mmcif_is(unpacked, pdb_id)
            ):
                return unpacked.resolve()
            import gzip

            try:
                body = gzip.decompress(local.read_bytes())
            except (OSError, EOFError) as exc:
                raise SearchError(f"template structure {local} is not gzip") from exc
            return self._publish(directory, pdb_id, body)
        if self.base_url is None:
            raise SearchError(
                f"template structure {pdb_id} is not in the local mirror and "
                "downloading is disabled"
            )
        directory = self.cache_dir / self._source_key(self.base_url)
        cached = directory / f"{pdb_id}.cif"
        # A cached file is only ever this store's own download, but anything
        # that can write the cache can also leave a different entry -- or a
        # truncated one -- under the name; such a file is fetched again.
        if cached.is_file() and _cached_mmcif_is(cached, pdb_id):
            return cached.resolve()
        url = f"{self.base_url}/{pdb_id.upper()}.cif"
        response = self.transport("GET", url, None, self.headers, self.timeout)
        if response.status != 200:
            raise SearchError(
                f"template structure {pdb_id} download failed with HTTP "
                f"{response.status}"
            )
        body = response.body
        if len(body) > MAX_REMOTE_BYTES:
            raise SearchError(f"template structure {pdb_id} is implausibly large")
        if _mmcif_block_name(body[:4096].decode("utf-8", "replace")) != pdb_id:
            raise SearchError(
                f"template structure {pdb_id} is not that entry's mmCIF file"
            )
        return self._publish(directory, pdb_id, body)


def _mmcif_block_name(text: str) -> str | None:
    """The lower-cased name of an mmCIF's first data block, if it has one."""
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped[:5].lower() == "data_":
            return stripped[5:].split()[0].lower() if stripped[5:] else None
        return None
    return None


def _cached_mmcif_is(path: Path, pdb_id: str) -> bool:
    """Whether a cached file is an mmCIF of ``pdb_id`` that can be parsed."""
    try:
        with path.open(encoding="utf-8", errors="replace") as handle:
            head = handle.read(4096)
    except OSError:
        return False
    if _mmcif_block_name(head) != pdb_id:
        return False
    try:
        import gemmi
    except ImportError:
        return True
    try:
        gemmi.cif.read(str(path)).sole_block()
    except (RuntimeError, ValueError):
        return False
    return True
