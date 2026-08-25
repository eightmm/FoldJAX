"""Parsers for Protenix's documented hmmsearch A3M and HHsearch HHR files."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from .msa import SearchError

_HIT_ID = re.compile(
    r"^(?:pdb\|)?(?P<pdb>[0-9][A-Za-z0-9]{3})(?:[|_](?P<chain>[A-Za-z0-9.]+))?"
)
_MAX_TEMPLATE_DATE = date(2021, 9, 30)
_MAX_TEMPLATE_HITS = 4
_MAX_TEMPLATE_CANDIDATES = 20
_RELEASE_DATES_ENV = "PROTENIX_TEMPLATE_RELEASE_DATES_FILE"
_OBSOLETE_PDBS_ENV = "PROTENIX_TEMPLATE_OBSOLETE_FILE"
_KALIGN_BINARY_ENV = "PROTENIX_KALIGN_BINARY"
#: Oldest Kalign this path accepts. It was an exact pin, and that was wrong
#: in a way worth recording: neither Protenix nor AlphaFold 3 asks for a
#: version -- upstream Protenix only asserts that a binary path was given --
#: and no commit here ever said why 3.3.5. It reads as the version that
#: happened to be installed. The cost was real: PyPI's `kalign-python` wheel
#: publishes 3.4.8 through 3.6.0 and nothing older, so an exact 3.3.5 made
#: templates reachable only through a system package.
#:
#: What is *not* established is that these versions align identically. Kalign
#: realigns the query onto the template chain here, so a different aligner can
#: give different templates and therefore a different structure. That
#: comparison has not been run. Pin an exact binary with the environment
#: variable when a run has to be reproduced against one.
_KALIGN_MINIMUM = (3, 3, 5)
_KALIGN_VERSION = ".".join(str(part) for part in _KALIGN_MINIMUM)

_PROTEIN_3TO1 = {
    "ALA": "A",
    "CYS": "C",
    "ASP": "D",
    "GLU": "E",
    "PHE": "F",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LYS": "K",
    "LEU": "L",
    "MET": "M",
    "ASN": "N",
    "PRO": "P",
    "GLN": "Q",
    "ARG": "R",
    "SER": "S",
    "THR": "T",
    "VAL": "V",
    "TRP": "W",
    "TYR": "Y",
}


@dataclass(frozen=True)
class TemplateSearchPayload:
    content: str
    suffix: str
    source: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _MmcifChainSequence:
    sequence: str
    seqres_start: int


class LocalTemplateSearchClient:
    """Run a local HMM/HH wrapper producing one ``.a3m`` or ``.hhr`` file."""

    name = "local-template"

    def __init__(
        self,
        command: Sequence[str],
        *,
        version: str,
        runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    ) -> None:
        if not command or not version:
            raise ValueError("template search command and version are required")
        self.command = tuple(command)
        self.version = version
        self._runner = runner

    def search(self, sequence: str, msa_path: str | Path) -> TemplateSearchPayload:
        msa = Path(msa_path)
        if not msa.is_file():
            raise FileNotFoundError(f"template search MSA input does not exist: {msa}")
        with tempfile.TemporaryDirectory(prefix="protenix-jax-template-") as raw_dir:
            output = Path(raw_dir) / "result"
            output.mkdir()
            command = [*self.command, "--input", str(msa), "--output", str(output)]
            try:
                completed = self._runner(
                    command, check=False, capture_output=True, text=True
                )
            except OSError as exc:
                raise SearchError(
                    f"failed to start template search command: {command[0]}"
                ) from exc
            if completed.returncode:
                detail = (completed.stderr or completed.stdout or "").strip()[-500:]
                raise SearchError(
                    f"template search exited with {completed.returncode}: {detail}"
                )
            artifacts = sorted(
                path
                for path in output.iterdir()
                if path.is_file() and path.suffix.lower() in (".a3m", ".hhr")
            )
            if len(artifacts) != 1:
                raise SearchError(
                    "template search command must produce exactly one .a3m or .hhr "
                    f"artifact, found {len(artifacts)}"
                )
            artifact = artifacts[0]
            content = artifact.read_text(encoding="utf-8")
            # Parse at the boundary so invalid wrapper output never enters cache.
            temporary = Path(raw_dir) / artifact.name
            temporary.write_text(content, encoding="utf-8")
            load_template_search_artifact(temporary, query_sequence=sequence)
            return TemplateSearchPayload(
                content,
                artifact.suffix.lower(),
                {"command": list(self.command), "version": self.version},
            )


class TemplateSearchPipeline:
    """Cache local template-search artifacts by sequence, MSA, and backend."""

    def __init__(
        self,
        cache_dir: str | Path,
        backend: LocalTemplateSearchClient,
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

    def search(self, sequence: str, msa_path: str | Path) -> dict[str, str]:
        msa = Path(msa_path)
        if not msa.is_file():
            raise FileNotFoundError(f"template search MSA input does not exist: {msa}")
        msa_raw = msa.read_bytes()
        identity = {
            "schema_version": 1,
            "sequence": sequence,
            "msa_sha256": hashlib.sha256(msa_raw).hexdigest(),
            "backend": {"name": self.backend.name, "version": self.backend.version},
            "options": self.options,
        }
        canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"))
        cache_key = hashlib.sha256(canonical.encode()).hexdigest()
        directory = self.cache_dir / cache_key
        provenance_path = directory / "provenance.json"
        if directory.exists():
            if not provenance_path.is_file():
                raise SearchError(f"template cache is incomplete: {directory}")
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
            if provenance.get("cache_key") != cache_key:
                raise SearchError(f"template cache key mismatch: {provenance_path}")
            artifact = directory / provenance["artifact"]["name"]
            if not artifact.is_file():
                raise SearchError(f"template cache artifact is missing: {artifact}")
            if (
                hashlib.sha256(artifact.read_bytes()).hexdigest()
                != provenance["artifact"]["sha256"]
            ):
                raise SearchError(f"template cache artifact hash mismatch: {artifact}")
            return {"templatesPath": str(artifact.resolve())}

        payload = self.backend.search(sequence, msa)
        if payload.suffix not in (".a3m", ".hhr"):
            raise SearchError(f"unsupported template artifact suffix: {payload.suffix}")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{cache_key}.", dir=self.cache_dir))
        try:
            artifact = temporary / f"templates{payload.suffix}"
            raw = payload.content.encode()
            artifact.write_bytes(raw)
            provenance = {
                **identity,
                "cache_key": cache_key,
                "source": dict(payload.source),
                "artifact": {
                    "name": artifact.name,
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "bytes": len(raw),
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
                return self.search(sequence, msa)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return {"templatesPath": str((directory / artifact.name).resolve())}


def apply_template_paths(
    inference: Mapping[str, Any], pipeline: TemplateSearchPipeline
) -> dict[str, Any]:
    """Apply template search to protein chains after protein MSA preprocessing."""

    import copy

    updated = copy.deepcopy(inference)
    for item in updated.get("sequences", []):
        if not isinstance(item, dict) or not isinstance(item.get("proteinChain"), dict):
            continue
        chain = item["proteinChain"]
        if chain.get("templatesPath"):
            continue
        msa_path = chain.get("unpairedMsaPath") or chain.get("pairedMsaPath")
        if not msa_path:
            raise SearchError(
                "automatic template search requires pairedMsaPath or "
                "unpairedMsaPath; enable protein MSA search first"
            )
        chain.update(pipeline.search(chain.get("sequence", ""), msa_path))
    return updated


def _indices(sequence: str, start: int) -> list[int]:
    result: list[int] = []
    position = start
    for residue in sequence:
        if residue == "-":
            result.append(-1)
        elif residue.islower():
            position += 1
        else:
            result.append(position)
            position += 1
    return result


def _mapping(query_indices: list[int], hit_indices: list[int]) -> dict[str, int]:
    if len(query_indices) != len(hit_indices):
        raise ValueError("template query and hit alignments have different lengths")
    return {
        str(query): hit
        for query, hit in zip(query_indices, hit_indices, strict=True)
        if query >= 0 and hit >= 0
    }


def _fasta_records(content: str) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    description: str | None = None
    sequence: list[str] = []
    for line in content.splitlines():
        if line.startswith(">"):
            if description is not None:
                records.append((description, "".join(sequence)))
            description, sequence = line[1:].strip(), []
        elif description is not None:
            sequence.append(line.strip())
        elif line.strip():
            raise ValueError("A3M artifact starts before its first FASTA header")
    if description is not None:
        records.append((description, "".join(sequence)))
    if not records or any(not sequence for _, sequence in records):
        raise ValueError("A3M artifact contains no complete FASTA records")
    return records


def _parse_a3m(content: str, query_sequence: str) -> list[dict[str, Any]]:
    records = _fasta_records(content)
    aligned_query = records[0][1]
    observed_query = "".join(c for c in aligned_query if c.isupper() and c != "-")
    if observed_query == query_sequence:
        hit_records = records[1:]
    elif "mol:protein" in records[0][0]:
        # Protenix release examples contain raw hmmsearch ``--domE`` A3M:
        # every record is a hit and the query row is omitted.  Lower-case hit
        # insertions do not consume a query column, so the number of non-lower
        # characters must equal the full query length and the aligned query is
        # reconstructed without guessing any residue identity.
        aligned_query = query_sequence
        hit_records = records
    else:
        raise ValueError(
            "template A3M query mismatch: "
            f"expected {query_sequence!r}, got {observed_query!r}"
        )
    query_indices = _indices(aligned_query, 0)
    hits: list[dict[str, Any]] = []
    for index, (description, hit_sequence) in enumerate(hit_records, start=1):
        if "mol:protein" not in description:
            continue
        coordinates = re.search(r"(?:^|[/\s])(\d+)-(\d+)(?:\s|$)", description)
        start = int(coordinates.group(1)) - 1 if coordinates else 0
        hit_indices = _indices(hit_sequence, start)
        hits.append(
            {
                "index": index,
                "name": description.split()[0],
                "aligned_cols": sum(i >= 0 for i in hit_indices),
                "sum_probs": None,
                "query": aligned_query,
                "hit_sequence": hit_sequence.upper(),
                "indices_query": query_indices,
                "indices_hit": hit_indices,
                "query_to_hit_mapping": _mapping(query_indices, hit_indices),
                "description": description,
            }
        )
    return hits


def _alignment_line(line: str) -> tuple[int, str] | None:
    match = re.match(r"^[QT]\s+\S+\s+(\d+)\s+([A-Za-z-]+)(?:\s+\d+)?", line)
    return (int(match.group(1)) - 1, match.group(2)) if match else None


def _parse_hhr(content: str) -> list[dict[str, Any]]:
    lines = content.splitlines()
    starts = [
        index for index, line in enumerate(lines) if re.match(r"^No\s+\d+\s*$", line)
    ]
    if not starts:
        raise ValueError("HHR artifact contains no hit blocks")
    starts.append(len(lines))
    hits: list[dict[str, Any]] = []
    for begin, end in zip(starts[:-1], starts[1:], strict=True):
        block = lines[begin:end]
        if len(block) < 3 or not block[1].startswith(">"):
            raise ValueError(f"invalid HHR hit block at line {begin + 1}")
        index = int(block[0].split()[-1])
        summary = next((line for line in block[2:] if "Aligned_cols=" in line), "")
        summary_match = re.search(
            r"Aligned_cols=(\d+).*?Sum_probs=([0-9]+(?:\.[0-9]+)?)", summary
        )
        if not summary_match:
            raise ValueError(f"invalid HHR summary for hit {index}")
        query_parts: list[str] = []
        hit_parts: list[str] = []
        query_indices: list[int] = []
        hit_indices: list[int] = []
        for line in block[3:]:
            if line.startswith(("Q ss_", "Q Consensus", "T ss_", "T Consensus")):
                continue
            parsed = _alignment_line(line)
            if parsed is None:
                continue
            start, sequence = parsed
            if line.startswith("Q "):
                query_parts.append(sequence)
                query_indices.extend(_indices(sequence, start))
            elif line.startswith("T "):
                hit_parts.append(sequence)
                hit_indices.extend(_indices(sequence, start))
        if not query_parts or not hit_parts:
            raise ValueError(f"HHR hit {index} contains no query/template alignment")
        hits.append(
            {
                "index": index,
                "name": block[1][1:].strip().split()[0],
                "aligned_cols": int(summary_match.group(1)),
                "sum_probs": float(summary_match.group(2)),
                "query": "".join(query_parts),
                "hit_sequence": "".join(hit_parts),
                "indices_query": query_indices,
                "indices_hit": hit_indices,
                "query_to_hit_mapping": _mapping(query_indices, hit_indices),
                "description": block[1][1:].strip(),
            }
        )
    return hits


def load_template_search_artifact(
    path: str | Path,
    *,
    query_sequence: str,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load `.a3m`/`.hhr` search hits and optionally persist JSON metadata.

    This intermediate intentionally contains alignments and PDB hit identifiers,
    not atom coordinates. Resolving coordinates requires a compatible local
    mmCIF database and remains an explicit downstream step.
    """
    artifact = Path(path)
    if not artifact.is_file():
        raise FileNotFoundError(f"template search artifact does not exist: {artifact}")
    query = "".join(query_sequence.split()).upper()
    if not query or not query.isascii() or not query.isalpha():
        raise ValueError("template query sequence must contain ASCII letters only")
    raw = artifact.read_bytes()
    try:
        content = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"template search artifact is not UTF-8 text: {artifact}"
        ) from exc
    suffix = artifact.suffix.lower()
    if suffix == ".a3m":
        artifact_format = "hmmsearch_a3m"
        hits = _parse_a3m(content, query)
    elif suffix == ".hhr":
        artifact_format = "hhr"
        hits = _parse_hhr(content)
    else:
        raise ValueError("supported template search artifacts are .a3m and .hhr")
    result = {
        "schema_version": 1,
        "format": artifact_format,
        "query_sequence": query,
        "source": {
            "path": str(artifact.resolve()),
            "sha256": hashlib.sha256(raw).hexdigest(),
        },
        "hits": hits,
    }
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.tmp")
        temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        temporary.replace(destination)
    return result


def _hit_pdb_and_chain(name: str) -> tuple[str, str | None]:
    """Extract a PDB accession and optional chain from HMM/HH hit names."""

    match = _HIT_ID.match(name)
    if match is None:
        raise ValueError(f"template hit name has no PDB accession: {name!r}")
    return match.group("pdb").lower(), match.group("chain")


def _find_mmcif(database: Path, pdb_id: str) -> Path:
    candidates = (
        database / f"{pdb_id}.cif",
        database / f"{pdb_id}.cif.gz",
        database / f"{pdb_id.upper()}.cif",
        database / f"{pdb_id.upper()}.cif.gz",
        database / pdb_id[1:3] / f"{pdb_id}.cif",
        database / pdb_id[1:3] / f"{pdb_id}.cif.gz",
    )
    match = next((candidate for candidate in candidates if candidate.is_file()), None)
    if match is None:
        raise FileNotFoundError(
            f"template mmCIF for PDB {pdb_id!r} was not found under {database}"
        )
    return match


def _mmcif_chain_sequences(mmcif_string: str) -> dict[str, _MmcifChainSequence]:
    """Read author-chain SEQRES using OpenDDE's mmCIF indexing convention."""

    import gemmi

    try:
        block = gemmi.cif.read_string(mmcif_string).sole_block()
    except (RuntimeError, ValueError) as exc:
        raise ValueError("template mmCIF could not be parsed") from exc

    entity_monomers: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for row in block.find(
        [
            "_entity_poly_seq.entity_id",
            "_entity_poly_seq.num",
            "_entity_poly_seq.mon_id",
        ]
    ):
        try:
            number = int(str(row[1]))
        except ValueError as exc:
            raise ValueError("template mmCIF has an invalid SEQRES index") from exc
        entity_monomers[str(row[0])].append((number, str(row[2]).upper()))
    if not entity_monomers:
        return {}

    label_to_entity = {
        str(row[0]): str(row[1])
        for row in block.find(["_struct_asym.id", "_struct_asym.entity_id"])
    }
    label_to_author: dict[str, str] = {}
    for row in block.find(
        [
            "_atom_site.label_asym_id",
            "_atom_site.auth_asym_id",
            "_atom_site.pdbx_PDB_model_num",
        ]
    ):
        if str(row[2]) != "1":
            continue
        label_chain, author_chain = str(row[0]), str(row[1])
        if author_chain not in (".", "?"):
            label_to_author.setdefault(label_chain, author_chain)

    chains: dict[str, _MmcifChainSequence] = {}
    for label_chain, author_chain in label_to_author.items():
        monomers = entity_monomers.get(label_to_entity.get(label_chain, ""))
        if not monomers:
            continue
        sequence = "".join(_PROTEIN_3TO1.get(name, "X") for _, name in monomers)
        chains.setdefault(
            author_chain,
            _MmcifChainSequence(
                sequence=sequence,
                seqres_start=min(number for number, _ in monomers),
            ),
        )
    if not chains:
        return {}
    return chains


def _template_chain_sequence(
    mmcif_string: str, chain_id: str | None
) -> tuple[str, _MmcifChainSequence]:
    chains = _mmcif_chain_sequences(mmcif_string)
    if not chains:
        raise ValueError("template mmCIF contains no observed protein chain SEQRES")
    if chain_id is not None and chain_id in chains:
        return chain_id, chains[chain_id]
    if len(chains) == 1:
        actual_chain_id = next(iter(chains))
        return actual_chain_id, chains[actual_chain_id]
    raise ValueError(
        f"template chain {chain_id!r} not found; available chains: {list(chains)}"
    )


def _resolve_kalign_binary(kalign_binary: str | Path | None) -> str:
    configured = kalign_binary or os.environ.get(_KALIGN_BINARY_ENV)
    if configured is None:
        # A `kalign` on PATH is accepted, and then held to the same version
        # check as a configured one -- which is what the pin actually protects.
        # Requiring the variable as well meant that installing the right
        # version was not enough, and refusing to look was the difference
        # between templates working and not on a machine that already had it.
        # `kalign-py` is the console script PyPI's `kalign-python` wheel
        # installs, and it is the only way to get this without a system
        # package. Looking only for `kalign` meant an environment that already
        # had a working aligner reported it as missing.
        found = next(
            (name for name in ("kalign", "kalign-py") if shutil.which(name)),
            None,
        )
        if found is None:
            raise ValueError(
                f"template search realignment requires Kalign {_KALIGN_VERSION} "
                "or newer; install it (`pip install kalign-python`, or "
                f"`apt install kalign`) or set {_KALIGN_BINARY_ENV}"
            )
        configured = shutil.which(found)
    executable = shutil.which(os.fspath(configured))
    if executable is None:
        raise FileNotFoundError(
            f"configured Kalign binary is not executable: {configured}"
        )
    try:
        completed = subprocess.run(
            [executable, "--version"],
            check=False,
            capture_output=True,
            text=True,
            errors="replace",
        )
    except OSError as exc:
        raise RuntimeError(f"failed to start Kalign binary: {executable}") from exc
    if completed.returncode:
        detail = (completed.stderr or completed.stdout or "").strip()[-500:]
        raise RuntimeError(
            f"Kalign version check exited with {completed.returncode}: {detail}"
        )
    version_output = "\n".join((completed.stdout, completed.stderr)).strip()
    # The wheel spells its own name `kalign-py`, so a pattern anchored on
    # `kalign` alone read its whole banner as the version string.
    match = re.search(r"(?mi)^kalign(?:-py)?\s+([^\s]+)\s*$", version_output)
    actual_version = match.group(1) if match is not None else version_output
    if _version_tuple(actual_version) < _KALIGN_MINIMUM:
        raise RuntimeError(
            f"template mapping requires Kalign {_KALIGN_VERSION} or newer; "
            f"configured binary reports {actual_version or 'no version'}"
        )
    return executable


def _version_tuple(reported: str) -> tuple[int, ...]:
    """Leading numeric components of a reported version, or nothing.

    An unparsable banner returns an empty tuple, which sorts below every real
    version and so fails the check rather than passing it by accident.
    """
    match = re.match(r"\s*v?(\d+(?:\.\d+)*)", reported or "")
    if match is None:
        return ()
    return tuple(int(part) for part in match.group(1).split("."))


def _parse_fasta_rows(content: str) -> list[str]:
    rows: list[str] = []
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            rows.append("")
        elif rows:
            rows[-1] += line
    return rows


def _align_query_to_template(
    query_sequence: str, target_sequence: str, *, kalign_binary: str
) -> dict[int, int]:
    for sequence in (query_sequence, target_sequence):
        if len(sequence) < 6:
            raise ValueError(
                "Kalign requires query and template sequences to be at least "
                f"6 residues long; got {len(sequence)}"
            )
    with tempfile.TemporaryDirectory(prefix="protenix-jax-kalign-") as raw_dir:
        directory = Path(raw_dir)
        input_path = directory / "input.fasta"
        output_path = directory / "output.a3m"
        input_path.write_text(
            f">sequence 1\n{query_sequence}\n>sequence 2\n{target_sequence}\n",
            encoding="utf-8",
        )
        try:
            completed = subprocess.run(
                [
                    kalign_binary,
                    "-i",
                    str(input_path),
                    "-o",
                    str(output_path),
                    "-format",
                    "fasta",
                ],
                check=False,
                capture_output=True,
                text=True,
                errors="replace",
            )
        except OSError as exc:
            raise RuntimeError(
                f"failed to start Kalign binary: {kalign_binary}"
            ) from exc
        if completed.returncode:
            detail = (completed.stderr or completed.stdout or "").strip()[-500:]
            raise RuntimeError(f"Kalign exited with {completed.returncode}: {detail}")
        if not output_path.is_file():
            raise RuntimeError("Kalign did not produce its requested output file")
        aligned = _parse_fasta_rows(output_path.read_text(encoding="utf-8"))
    if len(aligned) != 2 or len(aligned[0]) != len(aligned[1]):
        raise RuntimeError(
            "Kalign output must contain exactly two equally sized FASTA rows"
        )

    mapping: dict[int, int] = {}
    query_index = template_index = -1
    for query_residue, template_residue in zip(*aligned, strict=True):
        if query_residue != "-":
            query_index += 1
        if template_residue != "-":
            template_index += 1
        if query_residue != "-" and template_residue != "-":
            mapping[query_index] = template_index
    return mapping


def _metadata_path(
    configured: str | Path | None, *, environment: str, label: str, managed: str
) -> Path:
    """Caller, then environment, then the file `foldjax weights fetch` put down.

    The managed fallback is what makes a fetched asset reachable without also
    exporting a variable: both JSONs are published by the same project as the
    CCD, land in the same shared `assets/` directory, and are read here for the
    same reason -- so requiring the path twice would be a trap, not a policy.
    """
    value = configured or os.environ.get(environment)
    if value is None:
        from foldjax.paths import assets_dir

        fetched = assets_dir() / managed
        if fetched.is_file():
            return fetched
        raise ValueError(
            f"template {label} metadata is required for the "
            f"{_MAX_TEMPLATE_DATE.isoformat()} cutoff; run `foldjax weights "
            f"fetch --model protenix`, pass its path, or set {environment}"
        )
    path = Path(value)
    if not path.is_file():
        raise FileNotFoundError(f"template {label} metadata does not exist: {path}")
    return path


def _load_release_dates(path: Path) -> dict[str, date]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid template release-date JSON: {path}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"template release-date metadata must be an object: {path}")
    release_dates: dict[str, date] = {}
    for pdb_id, metadata in raw.items():
        if not isinstance(metadata, dict) or "release_date" not in metadata:
            continue
        try:
            release_dates[str(pdb_id).lower()] = date.fromisoformat(
                str(metadata["release_date"])
            )
        except ValueError as exc:
            raise ValueError(
                f"invalid release date for PDB {pdb_id!r} in {path}"
            ) from exc
    return release_dates


def _load_obsolete_pdbs(path: Path) -> dict[str, str]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid obsolete-PDB JSON: {path}") from exc
    if not isinstance(raw, dict) or any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in raw.items()
    ):
        raise ValueError(f"obsolete-PDB metadata must map strings to strings: {path}")
    return {key.lower(): value.lower() for key, value in raw.items()}


def _prefilter_template_hits(
    hits: Sequence[Mapping[str, Any]],
    *,
    query_sequence: str,
    release_dates: Mapping[str, date],
    obsolete_pdbs: Mapping[str, str],
) -> list[tuple[Mapping[str, Any], str, str | None]]:
    """Apply OpenDDE's quick date/coverage/duplicate/length hit filter."""

    valid: list[tuple[Mapping[str, Any], str, str | None]] = []
    for hit in hits:
        try:
            pdb_id, chain_id = _hit_pdb_and_chain(str(hit["name"]))
        except (KeyError, ValueError):
            continue
        coordinate_pdb_id = obsolete_pdbs.get(pdb_id, pdb_id)
        date_pdb_id = pdb_id if pdb_id in release_dates else coordinate_pdb_id
        released = release_dates.get(date_pdb_id)
        if released is None or released > _MAX_TEMPLATE_DATE:
            continue
        aligned_cols = int(hit["aligned_cols"])
        hit_sequence = str(hit["hit_sequence"]).replace("-", "")
        if aligned_cols / len(query_sequence) <= 0.1:
            continue
        if (
            hit_sequence in query_sequence
            and len(hit_sequence) / len(query_sequence) > 0.95
        ):
            continue
        if len(hit_sequence) < 10:
            continue
        valid.append((hit, coordinate_pdb_id, chain_id))

    valid.sort(
        key=lambda item: (
            float(item[0]["sum_probs"]) if item[0].get("sum_probs") is not None else 0.0
        ),
        reverse=True,
    )
    deduplicated: list[tuple[Mapping[str, Any], str, str | None]] = []
    seen_sequences: set[str] = set()
    for item in valid:
        sequence = str(item[0]["hit_sequence"]).replace("-", "")
        if sequence not in seen_sequences:
            seen_sequences.add(sequence)
            deduplicated.append(item)
    return deduplicated


def resolve_template_search_hits(
    artifact: str | Path,
    *,
    query_sequence: str,
    mmcif_dir: str | Path | None = None,
    max_templates: int = 4,
    release_dates_path: str | Path | None = None,
    obsolete_pdbs_path: str | Path | None = None,
    kalign_binary: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Resolve A3M/HHR hits to self-contained JSON-template dictionaries.

    The coordinate database is caller supplied, or read from
    ``PROTENIX_TEMPLATE_MMCIF_DIR``. Both flat and PDB divided directory
    layouts, with plain/gzipped mmCIF files, are accepted. Release-date and
    obsolete-PDB metadata are caller supplied, or read from
    ``PROTENIX_TEMPLATE_RELEASE_DATES_FILE`` and
    ``PROTENIX_TEMPLATE_OBSOLETE_FILE``. Query-to-coordinate mappings are rebuilt
    with an explicitly configured Kalign 3.3.5 binary, supplied by argument or
    ``PROTENIX_KALIGN_BINARY``. No database or executable is downloaded implicitly.
    """

    if max_templates <= 0:
        raise ValueError("max_templates must be positive")
    configured = mmcif_dir or os.environ.get("PROTENIX_TEMPLATE_MMCIF_DIR")
    if configured is None:
        raise ValueError(
            "templatesPath .a3m/.hhr requires a local mmCIF directory; pass "
            "mmcif_dir or set PROTENIX_TEMPLATE_MMCIF_DIR"
        )
    database = Path(configured)
    if not database.is_dir():
        raise FileNotFoundError(f"template mmCIF directory does not exist: {database}")
    release_dates_file = _metadata_path(
        release_dates_path,
        environment=_RELEASE_DATES_ENV,
        label="release-date",
        managed="release_date_cache.json",
    )
    obsolete_pdbs_file = _metadata_path(
        obsolete_pdbs_path,
        environment=_OBSOLETE_PDBS_ENV,
        label="obsolete-PDB",
        managed="obsolete_to_successor.json",
    )
    search = load_template_search_artifact(artifact, query_sequence=query_sequence)
    candidates = _prefilter_template_hits(
        search["hits"],
        query_sequence=search["query_sequence"],
        release_dates=_load_release_dates(release_dates_file),
        obsolete_pdbs=_load_obsolete_pdbs(obsolete_pdbs_file),
    )[:_MAX_TEMPLATE_CANDIDATES]
    limit = min(max_templates, _MAX_TEMPLATE_HITS)
    resolved: list[dict[str, Any]] = []
    missing_coordinates: list[str] = []
    resolved_kalign_binary: str | None = None
    for hit, pdb_id, chain_id in candidates:
        if len(resolved) >= limit:
            break
        try:
            path = _find_mmcif(database, pdb_id)
        except FileNotFoundError:
            missing_coordinates.append(pdb_id)
            continue
        if path.suffix == ".gz":
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                mmcif = handle.read()
        else:
            mmcif = path.read_text(encoding="utf-8")
        if resolved_kalign_binary is None:
            resolved_kalign_binary = _resolve_kalign_binary(kalign_binary)
        actual_chain_id, chain = _template_chain_sequence(mmcif, chain_id)
        mapping = _align_query_to_template(
            search["query_sequence"],
            chain.sequence,
            kalign_binary=resolved_kalign_binary,
        )
        query_indices = sorted(mapping)
        resolved.append(
            {
                "mmcif": mmcif,
                "queryIndices": query_indices,
                "templateIndices": [mapping[index] for index in query_indices],
                "chainId": actual_chain_id,
                "pdbId": pdb_id,
            }
        )
    if not resolved and missing_coordinates:
        missing = ", ".join(dict.fromkeys(missing_coordinates))
        raise FileNotFoundError(
            "no coordinate-backed template hits were found under "
            f"{database}; missing PDB mmCIF files: {missing}"
        )
    return resolved
