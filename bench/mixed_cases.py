"""Build a mixed-entity benchmark set from real depositions, one command per step.

Every case in `bench/spec.py`'s pinned set is protein and nothing else. That is
the cheap half of the cost curve: a ligand is tokenized per heavy atom rather
than per residue, nucleic acids take a different embedding path, and repeated
ion chains exercise the chain-id and asym bookkeeping that a single protein
never touches. A scale table measured on protein-only inputs therefore says
nothing about the shapes the ports actually meet.

The 2026-09-10 mixed set was built by hand -- an afternoon of RCSB queries,
a scratch ColabFold fetcher and five job files written out one at a time. This
module is that afternoon turned into four subcommands:

    python -m bench.mixed_cases select      --out candidates.json
    python -m bench.mixed_cases materialise --pdb 5NPK --case mixed_3k_5npk ...
    python -m bench.mixed_cases fetch-msa   --set-dir ...
    python -m bench.mixed_cases validate    --set-dir ...

`select` is the only step that guesses; the other three are deterministic given
their inputs, so a set can be rebuilt from the chosen PDB ids alone.

**Token counting.** A token here is what the models tokenize: one per polymer
residue and one per *heavy atom* of a ligand. Monatomic ions are one token by
arithmetic rather than by special case. The hand-built set's `sequences.json`
recorded one token per ligand *copy* instead, which is right for ions and wrong
for anything organic -- it undercounts 7Y7Q by 93 and 5NPK by 200. `validate`
reports both numbers side by side rather than silently preferring one.

**Formatting.** Job files are `json.dumps(..., indent=2, sort_keys=True)` and
`sequences.json` is `indent=1` in `length, sequence, pdb` order, both without a
trailing newline, because that is byte-for-byte what the reference set contains
and a byte comparison is the strongest schema test available offline.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tests._foldbench.rcsb import (
    _MOLECULE_TYPES,
    UnsupportedEntityError,
    cache_dir,
    fetch_cif,
)
from tests._foldbench.targets import CRYSTALLIZATION_ADDITIVES, UNSUPPORTED_COMPONENTS

#: Metal ions kept even though some of them double as buffer salts. A folding
#: model is asked to place these because they are part of the assembly -- the
#: zinc in a zinc finger, the magnesium in a polymerase active site -- so
#: dropping them would change the biology rather than the crystallography.
#:
#: `K` is the divergence from `tests._foldbench.targets.CRYSTALLIZATION_ADDITIVES`,
#: which denies it: potassium is usually a salt component, but it is also the
#: coordinating ion of K+ channels and G-quadruplexes, and this set exists to
#: exercise ion handling. `NA` and `CL` stay denied there and here, because they
#: are almost never the point of the structure.
STRUCTURAL_IONS = frozenset({"ZN", "MG", "CA", "MN", "FE", "K"})

#: Waters, cryoprotectants, buffers, precipitants and components no port can
#: express. Built on the curated list under `tests/` rather than restated, so a
#: component added there is excluded here too.
EXCLUDED_COMPONENTS = (
    CRYSTALLIZATION_ADDITIVES
    | frozenset(UNSUPPORTED_COMPONENTS)
    | frozenset({"HOH", "DOD"})
) - STRUCTURAL_IONS

#: The scale ladder the set covers, and how far a real entry may sit from a rung.
#: Depositions do not come in round sizes, so an exact target would select
#: nothing; 25% is wide enough that every rung has candidates and narrow enough
#: that the rungs stay ordered.
BUCKETS: tuple[int, ...] = (1000, 2000, 3000, 4000, 5000)
TOLERANCE = 0.25

_POLYMER_KINDS = frozenset({"protein", "dna", "rna"})
_SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
_CHEMCOMP_URL = "https://data.rcsb.org/rest/v1/core/chemcomp"
_COLABFOLD_HOST = "https://api.colabfold.com"

#: The two archive members ColabFold's `env` mode returns. Concatenated in this
#: order, which is the order `foldjax.search.msa.RemoteMMseqs2Client` uses.
_A3M_MEMBERS = ("uniref.a3m", "bfd.mgnify30.metaeuk30.smag30.a3m")

#: Rank of a candidate that could not be read or costed at all, so it sorts
#: behind every entry that was. Same arity as :func:`rank_key`, because a
#: shorter tuple would compare against its elements positionally and order by
#: whatever happened to line up.
_UNRANKABLE = (True, True, True, 10**9, 99.0)


# --------------------------------------------------------------------------- #
# Composition
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Entity:
    """One deposited entity, with the number of chains it appears as."""

    entity_id: int
    kind: str
    copies: int
    sequence: str | None = None
    ccd: str | None = None

    @property
    def is_polymer(self) -> bool:
        return self.kind in _POLYMER_KINDS

    @property
    def length(self) -> int:
        return len(self.sequence) if self.sequence is not None else 0


@dataclass(frozen=True)
class Composition:
    """A deposition reduced to what a job file needs, ligand copies included.

    `tests._foldbench.rcsb.parse_target` deliberately stops short of this: it
    reports *distinct* ligand codes, because its callers ask which components an
    entry has. A job file has to say how many chains each one occupies, so this
    reads `struct_asym` as well.
    """

    pdb_id: str
    entities: tuple[Entity, ...]
    resolution: float | None = None

    @property
    def polymer_tokens(self) -> int:
        return sum(e.length * e.copies for e in self.entities if e.is_polymer)

    @property
    def ligands(self) -> tuple[Entity, ...]:
        return tuple(e for e in self.entities if e.kind == "ligand")

    @property
    def molecule_types(self) -> frozenset[str]:
        return frozenset(e.kind for e in self.entities)

    @property
    def has_nucleic_acid(self) -> bool:
        return bool(self.molecule_types & {"dna", "rna"})

    @property
    def has_organic_ligand(self) -> bool:
        return any(e.ccd not in STRUCTURAL_IONS for e in self.ligands)

    def describe(self) -> str:
        parts = [
            f"{e.kind}x{e.copies}:{e.length}" if e.is_polymer else f"{e.ccd}x{e.copies}"
            for e in self.entities
        ]
        return f"{self.pdb_id} {' '.join(parts)}"


def parse_composition(
    path: str | Path, pdb_id: str | None = None, *, include_solvent: bool = False
) -> Composition:
    """Read an mmCIF into a :class:`Composition`.

    Entities keep their deposited `entity_id` order, which is polymers first and
    then non-polymers. That order is what decides chain-id assignment later, so
    it is preserved rather than sorted by anything else.

    Raises:
        UnsupportedEntityError: a polymer type outside protein/DNA/RNA is present.
    """
    import biotite.structure.io.pdbx as pdbx

    text = Path(path).read_text()
    block = pdbx.CIFFile.read(io.StringIO(text)).block
    identifier = (pdb_id or Path(path).stem).upper()

    # `struct_asym` has one row per chain, so counting its rows per entity is
    # how many copies of that entity the deposition contains. `entity_poly`
    # gives the same answer for polymers via `pdbx_strand_id`, but non-polymers
    # have no strand list at all and this is the only place their count lives.
    struct_asym = block.get("struct_asym")
    if struct_asym is None:
        raise UnsupportedEntityError(f"{identifier}: mmCIF has no struct_asym table")
    copies = Counter(str(value) for value in struct_asym["entity_id"].as_array())

    entities: list[Entity] = []
    entity_poly = block.get("entity_poly")
    if entity_poly is not None:
        columns = zip(
            entity_poly["entity_id"].as_array(),
            entity_poly["type"].as_array(),
            entity_poly["pdbx_seq_one_letter_code_can"].as_array(),
            strict=True,
        )
        for entity_id, kind, sequence in columns:
            molecule_type = _MOLECULE_TYPES.get(str(kind))
            if molecule_type is None:
                raise UnsupportedEntityError(
                    f"{identifier}: polymer type {kind!r} is not one of "
                    f"{sorted(_MOLECULE_TYPES)}"
                )
            entities.append(
                Entity(
                    entity_id=int(entity_id),
                    kind=molecule_type,
                    copies=copies[str(entity_id)],
                    # The one-letter code is wrapped across lines in the file.
                    sequence="".join(str(sequence).split()),
                )
            )

    nonpoly = block.get("pdbx_entity_nonpoly")
    if nonpoly is not None:
        for entity_id, comp_id in zip(
            nonpoly["entity_id"].as_array(),
            nonpoly["comp_id"].as_array(),
            strict=True,
        ):
            code = str(comp_id)
            if not include_solvent and code in EXCLUDED_COMPONENTS:
                continue
            entities.append(
                Entity(
                    entity_id=int(entity_id),
                    kind="ligand",
                    copies=copies[str(entity_id)],
                    ccd=code,
                )
            )

    resolution = _resolution(block)
    return Composition(
        pdb_id=identifier,
        entities=tuple(sorted(entities, key=lambda e: e.entity_id)),
        resolution=resolution,
    )


def _resolution(block: Any) -> float | None:
    """Best reported resolution, or None for a method that has none (NMR)."""
    for table, column in (
        ("refine", "ls_d_res_high"),
        ("em_3d_reconstruction", "resolution"),
    ):
        category = block.get(table)
        if category is None or column not in category:
            continue
        for value in category[column].as_array():
            try:
                return float(str(value))
            except ValueError:
                continue
    return None


# --------------------------------------------------------------------------- #
# Token counting
# --------------------------------------------------------------------------- #


def chemcomp_cache_path() -> Path:
    """Where heavy-atom counts are remembered, beside the cached mmCIFs."""
    return cache_dir() / "chemcomp.json"


def _fetch_heavy_atoms(ccd: str) -> int:
    with urllib.request.urlopen(f"{_CHEMCOMP_URL}/{ccd}", timeout=60) as response:
        payload = json.load(response)
    count = payload.get("rcsb_chem_comp_info", {}).get("atom_count_heavy")
    if not isinstance(count, int) or count <= 0:
        raise ValueError(f"RCSB reported no heavy-atom count for CCD {ccd!r}")
    return count


def heavy_atom_counts(
    ccds: Iterable[str],
    *,
    cache_path: str | Path | None = None,
    fetch: Callable[[str], int] | None = None,
) -> dict[str, int]:
    """Heavy atoms per CCD code, cached on disk between runs.

    The ions in :data:`STRUCTURAL_IONS` are monatomic, so their count is 1 by
    definition and never costs a request. A set built from ions alone therefore
    validates with no network at all, which is what makes the offline tests
    possible without a fixture for every code.
    """
    wanted = sorted({code.upper() for code in ccds})
    counts = {code: 1 for code in wanted if code in STRUCTURAL_IONS}
    missing = [code for code in wanted if code not in counts]
    if not missing:
        return counts

    path = Path(cache_path) if cache_path is not None else chemcomp_cache_path()
    cached: dict[str, int] = {}
    if path.is_file():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            loaded = {}
        if isinstance(loaded, dict):
            cached = {str(k): int(v) for k, v in loaded.items() if isinstance(v, int)}

    resolve = fetch if fetch is not None else _fetch_heavy_atoms
    fresh = False
    for code in missing:
        if code in cached:
            counts[code] = cached[code]
            continue
        counts[code] = cached[code] = resolve(code)
        fresh = True
    if fresh:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cached, indent=1, sort_keys=True) + "\n")
    return counts


def token_count(entities: Sequence[Entity], heavy_atoms: Mapping[str, int]) -> int:
    """One token per polymer residue, one per ligand heavy atom, per copy.

    Raises:
        KeyError: a ligand's CCD code has no heavy-atom count.
    """
    total = 0
    for entity in entities:
        if entity.is_polymer:
            total += entity.length * entity.copies
        else:
            assert entity.ccd is not None
            total += heavy_atoms[entity.ccd] * entity.copies
    return total


def bucket_band(target: int, tolerance: float = TOLERANCE) -> tuple[int, int]:
    """Inclusive token range accepted for a bucket."""
    return (
        int(round(target * (1.0 - tolerance))),
        int(round(target * (1.0 + tolerance))),
    )


def bucket_of(case: str) -> int | None:
    """The rung a case name claims, e.g. `mixed_3k_5npk` -> 3000."""
    parts = case.split("_")
    for part in parts:
        if part.endswith("k") and part[:-1].isdigit():
            return int(part[:-1]) * 1000
    return None


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #


def search_query(
    target: int,
    *,
    tolerance: float = TOLERANCE,
    max_resolution: float | None = 3.0,
    rows: int = 25,
    polymer_types: str = "Protein/NA",
) -> dict[str, Any]:
    """The RCSB search body for one bucket.

    Deliberately coarser than the final filter. RCSB can only count *polymer
    monomers*, which is a lower bound on the token count -- ligand atoms are on
    top of it -- so the remote query brackets the band generously and the exact
    token rule is applied locally against the downloaded mmCIF. Filtering
    remotely on a number that means something else would silently drop entries.
    """
    low, high = bucket_band(target, tolerance)
    nodes: list[dict[str, Any]] = [
        {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "rcsb_entry_info.deposited_polymer_monomer_count",
                "operator": "range",
                "value": {"from": int(low * 0.8), "to": high},
            },
        },
        {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "rcsb_entry_info.selected_polymer_entity_types",
                "operator": "exact_match",
                "value": polymer_types,
            },
        },
        {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "rcsb_entry_info.nonpolymer_entity_count",
                "operator": "greater",
                "value": 0,
            },
        },
    ]
    if max_resolution is not None:
        nodes.append(
            {
                "type": "terminal",
                "service": "text",
                "parameters": {
                    "attribute": "rcsb_entry_info.resolution_combined",
                    "operator": "less_or_equal",
                    "value": max_resolution,
                },
            }
        )
    return {
        "query": {"type": "group", "logical_operator": "and", "nodes": nodes},
        "return_type": "entry",
        "request_options": {
            "paginate": {"start": 0, "rows": rows},
            "results_content_type": ["experimental"],
            # `sort_by`, not `attribute`: the sort clause spells the field
            # differently from a terminal's parameters, and the v2 endpoint
            # answers the other spelling with a bare HTTP 400.
            "sort": [
                {
                    "sort_by": "rcsb_entry_info.resolution_combined",
                    "direction": "asc",
                }
            ],
        },
    }


def search_entries(query: Mapping[str, Any], *, timeout: float = 60.0) -> list[str]:
    """PDB ids matching a search body, best resolution first."""
    url = f"{_SEARCH_URL}?json={urllib.parse.quote(json.dumps(query))}"
    with urllib.request.urlopen(url, timeout=timeout) as response:
        payload = json.load(response)
    return [str(hit["identifier"]) for hit in payload.get("result_set", [])]


def rank_key(
    composition: Composition, target: int, *, refused: str | None = None
) -> tuple[Any, ...]:
    """Sort key for candidates in one bucket; smaller is better.

    A refused entry sorts last whatever else it has going for it. 7MJW is the
    case that put this term first: it is 1029 tokens, protein with RNA and four
    distinct ligands at 1.4 A -- the best candidate in the 1k band on every
    other axis -- and its RNA carries a modified nucleotide whose canonical
    mapping is `X`, which every backend refuses. Ranking without asking the
    harness recommends it, and the defect only surfaces at `validate`.

    Composition comes before closeness on purpose. A protein-plus-zinc entry at
    exactly the target token count would defeat the reason this set exists,
    while one 8% off the rung with DNA and a nucleotide cofactor is the case
    worth measuring.
    """
    return (
        refused is not None,
        not composition.has_nucleic_acid,
        not composition.has_organic_ligand,
        abs(composition.polymer_tokens - target),
        composition.resolution if composition.resolution is not None else 99.0,
    )


def select_candidates(
    target: int,
    *,
    tolerance: float = TOLERANCE,
    max_resolution: float | None = 3.0,
    limit: int = 12,
    polymer_types: Sequence[str] = ("Protein/NA", "Protein (only)"),
    cif_directory: str | Path | None = None,
    heavy_atoms: Mapping[str, int] | None = None,
    model: str = "boltz2",
) -> list[dict[str, Any]]:
    """Score real entries for one bucket, best first.

    Every candidate is turned into the job file it would become and put to
    `model`'s own validator, so a `refused` entry is one the benchmark could not
    have run. `heavy_atoms` may be supplied to keep the scoring offline; when it
    is None the counts are fetched (and cached) per ligand code encountered.
    """
    from foldjax.input import compatibility

    identifiers: list[str] = []
    for types in polymer_types:
        query = search_query(
            target,
            tolerance=tolerance,
            max_resolution=max_resolution,
            rows=limit,
            polymer_types=types,
        )
        for identifier in search_entries(query):
            if identifier not in identifiers:
                identifiers.append(identifier)

    low, high = bucket_band(target, tolerance)
    scored: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    for identifier in identifiers:
        try:
            composition = parse_composition(
                fetch_cif(identifier, directory=cif_directory), identifier
            )
        except (UnsupportedEntityError, OSError, KeyError, ValueError) as error:
            scored.append((_UNRANKABLE, {"pdb": identifier, "skipped": str(error)}))
            continue
        codes = {e.ccd for e in composition.ligands if e.ccd}
        try:
            table = (
                dict(heavy_atoms)
                if heavy_atoms is not None
                else heavy_atom_counts(codes)
            )
            tokens = token_count(composition.entities, table)
        except (KeyError, OSError, ValueError) as error:
            scored.append((_UNRANKABLE, {"pdb": identifier, "skipped": str(error)}))
            continue
        refused = compatibility(
            build_job(composition, case_name(composition.pdb_id, target)), model
        )
        scored.append(
            (
                rank_key(composition, target, refused=refused),
                {
                    "pdb": composition.pdb_id,
                    "tokens": tokens,
                    "polymer_tokens": composition.polymer_tokens,
                    "resolution": composition.resolution,
                    "in_band": low <= tokens <= high,
                    "refused": refused,
                    "molecule_types": sorted(composition.molecule_types),
                    "chains": sum(e.copies for e in composition.entities),
                    "entities": [
                        {
                            "type": e.kind,
                            "copies": e.copies,
                            **(
                                {"length": e.length}
                                if e.is_polymer
                                else {"ccd": e.ccd, "heavy_atoms": table.get(e.ccd)}
                            ),
                        }
                        for e in composition.entities
                    ],
                },
            )
        )
    scored.sort(key=lambda item: (not item[1].get("in_band", False), item[0]))
    return [record for _, record in scored]


# --------------------------------------------------------------------------- #
# Materialisation
# --------------------------------------------------------------------------- #


def case_name(pdb_id: str, target: int) -> str:
    """`mixed_3k_5npk`: the rung, so a set sorts by size, then the deposition."""
    return f"mixed_{target // 1000}k_{pdb_id.lower()}"


def build_job(
    composition: Composition,
    case: str,
    *,
    msa_directory: str = "../msa",
) -> dict[str, Any]:
    """The FoldJAX job document for one deposition.

    Chain ids are reassigned in entity order with `foldjax.input.chain_labels`
    rather than carried over from the deposition. Two reasons: the deposition's
    ids are not contiguous once excluded components are dropped, and ligands
    have no strand id at all, so something has to invent them anyway. Doing it
    for every entity keeps the naming one rule instead of two.
    """
    from foldjax.input import chain_labels

    labels = chain_labels()
    entities: list[dict[str, Any]] = []
    protein_index = 0
    for entity in composition.entities:
        ids = [next(labels) for _ in range(entity.copies)]
        if entity.kind == "ligand":
            entities.append({"ccd": entity.ccd, "id": ids, "type": "ligand"})
            continue
        body: dict[str, Any] = {
            "id": ids,
            "sequence": entity.sequence,
            "type": entity.kind,
        }
        if entity.kind == "protein":
            protein_index += 1
            body["unpaired_msa"] = (
                f"{msa_directory}/{case}_p{protein_index}_unpaired.a3m"
            )
        entities.append(body)
    return {"entities": entities, "name": case}


def job_sequence(job: Mapping[str, Any]) -> str:
    """Polymer chains joined by `:`, one part per copy -- `sequences.json`'s field."""
    parts: list[str] = []
    for entity in job["entities"]:
        if entity["type"] == "ligand":
            continue
        parts.extend([entity["sequence"]] * len(entity["id"]))
    return ":".join(parts)


def dump_job(job: Mapping[str, Any]) -> str:
    return json.dumps(job, indent=2, sort_keys=True)


def dump_sequences(document: Mapping[str, Mapping[str, Any]]) -> str:
    ordered = {
        name: {
            "length": document[name]["length"],
            "sequence": document[name]["sequence"],
            "pdb": document[name]["pdb"],
        }
        for name in sorted(document)
    }
    return json.dumps(ordered, indent=1)


def materialise(
    pdb_id: str,
    case: str,
    set_dir: str | Path,
    *,
    cif_directory: str | Path | None = None,
    heavy_atoms: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Write `jobs/<case>.json` and update `sequences.json`. Returns a summary."""
    root = Path(set_dir)
    composition = parse_composition(fetch_cif(pdb_id, directory=cif_directory), pdb_id)
    job = build_job(composition, case)
    codes = {e.ccd for e in composition.ligands if e.ccd}
    table = dict(heavy_atoms) if heavy_atoms is not None else heavy_atom_counts(codes)
    tokens = token_count(composition.entities, table)

    (root / "jobs").mkdir(parents=True, exist_ok=True)
    (root / "msa").mkdir(parents=True, exist_ok=True)
    (root / "jobs" / f"{case}.json").write_text(dump_job(job), encoding="utf-8")

    sequences_path = root / "sequences.json"
    document: dict[str, Any] = {}
    if sequences_path.is_file():
        document = json.loads(sequences_path.read_text(encoding="utf-8"))
    document[case] = {
        "length": tokens,
        "sequence": job_sequence(job),
        "pdb": composition.pdb_id,
    }
    sequences_path.write_text(dump_sequences(document), encoding="utf-8")
    return {
        "case": case,
        "pdb": composition.pdb_id,
        "tokens": tokens,
        "polymer_tokens": composition.polymer_tokens,
        "proteins": sum(1 for e in job["entities"] if e["type"] == "protein"),
        "describe": composition.describe(),
    }


# --------------------------------------------------------------------------- #
# MSA
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SanitisedA3m:
    """A cleaned alignment and what had to be removed to get there."""

    text: str
    records: int
    dropped_columns: int
    stripped_nuls: int


def _match_columns(sequence: str) -> int:
    """Positions the a3m aligns to the query: uppercase residues and gaps.

    Lowercase characters are insertions relative to the query and belong to no
    column, which is the whole point of the a3m format.
    """
    return sum(1 for character in sequence if character.isupper() or character == "-")


def _query_residues(sequence: str) -> str:
    return "".join(c for c in sequence if c.isupper() and c != "-")


def sanitise_a3m(text: str, query: str, *, label: str = "unpaired MSA") -> SanitisedA3m:
    """Make one alignment safe for every parser in this repo to read.

    Three defects are removed, each of which has broken a real run:

    * **NUL bytes.** The ColabFold archive carries trailing NULs. OpenFold3,
      Boltz-2, ESMFold2 and AlphaFold3 all read the a3m as text and their
      parsers fail on them at different, unhelpful places.
    * **A first record that is not the query.** Every consumer takes record zero
      as the query row and indexes the rest against it, so a reordered file is
      wrong rather than merely odd. This raises instead of repairing: a file
      whose query is missing is a failed search, not a dirty one.
    * **Records with the wrong number of match columns.** A truncated download
      produces these, and they turn into a ragged array much later.

    Duplicate copies of the query are *kept*. They occur in every file of the
    reference set, from both of the fetchers that built it, and a homolog that
    happens to equal the query is a legitimate row.

    Raises:
        ValueError: the file is empty, headerless, or its first record is not
            the query.
    """
    stripped_nuls = text.count("\x00")
    cleaned = text.replace("\x00", "")

    headers: list[str] = []
    sequences: list[str] = []
    for line in cleaned.splitlines():
        if line.startswith(">"):
            headers.append(line)
            sequences.append("")
        elif line.strip():
            if not headers:
                raise ValueError(f"{label} does not start with a FASTA header")
            sequences[-1] += line.strip()

    if not headers or not sequences[0]:
        raise ValueError(f"{label} is empty or has no query sequence")
    first = _query_residues(sequences[0])
    if first != query:
        raise ValueError(
            f"{label} query does not match the requested sequence: "
            f"expected {len(query)} residues, got {len(first)}"
        )

    expected = len(query)
    kept_headers: list[str] = []
    kept_sequences: list[str] = []
    dropped = 0
    for header, sequence in zip(headers, sequences, strict=True):
        if not sequence or _match_columns(sequence) != expected:
            dropped += 1
            continue
        kept_headers.append(header)
        kept_sequences.append(sequence)

    body = "\n".join(
        line
        for header, sequence in zip(kept_headers, kept_sequences, strict=True)
        for line in (header, sequence)
    )
    return SanitisedA3m(
        text=body + "\n",
        records=len(kept_headers),
        dropped_columns=dropped,
        stripped_nuls=stripped_nuls,
    )


def sequence_key(sequence: str) -> str:
    """Cache key for one query: the sequence, not the case it appears in.

    Keyed this way an identical chain in two cases is searched once. The scratch
    script keyed by case and protein index, and re-ran the search for every
    repeat.
    """
    return hashlib.sha256(sequence.strip().upper().encode()).hexdigest()


class ColabFoldUnpairedClient:
    """Ticket, poll and download one unpaired MMseqs2 alignment.

    `foldjax.search.msa.RemoteMMseqs2Client` already speaks this protocol and is
    the right thing for a prediction. It is the wrong thing here for two
    reasons, both about scale rather than correctness:

    * It runs the *paired* search as well, because `MsaSearchPipeline` requires
      both halves. A benchmark set needs only the unpaired half, and the public
      ColabFold endpoint is a free shared service -- doubling the load on it to
      discard the answer is not acceptable at 22 sequences.
    * It downloads the archive in one stdlib request. The largest member of the
      reference set is a 44 MB a3m, and that download failed often enough that
      the afternoon's scratch script was rewritten around `curl -C -`.

    The ticket state machine, the archive members and the NUL strip are the same
    as that client's, on purpose.
    """

    name = "colabfold-unpaired"

    def __init__(
        self,
        host: str = _COLABFOLD_HOST,
        *,
        poll_interval: float = 10.0,
        max_wait_seconds: float = 3600.0,
        timeout: float = 120.0,
        curl_timeout: float = 1800.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.host = host.rstrip("/")
        self.poll_interval = poll_interval
        self.max_wait_seconds = max_wait_seconds
        self.timeout = timeout
        self.curl_timeout = curl_timeout
        self.sleep = sleep
        try:
            from foldjax import __version__ as version
        except ImportError:  # pragma: no cover - the package is always present
            version = "0"
        # The endpoint's operators ask clients to identify themselves, and this
        # repo already decided what that name is.
        self.headers = {"User-Agent": f"foldjax/{version}"}

    def _json(self, path: str, data: bytes | None = None) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{self.host}/{path.lstrip('/')}", data=data, headers=self.headers
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            payload = json.load(response)
        if not isinstance(payload, dict) or not isinstance(payload.get("status"), str):
            raise RuntimeError(f"ColabFold returned an invalid response for {path!r}")
        return payload

    def submit(self, sequence: str, label: str) -> str:
        body = urllib.parse.urlencode(
            {"q": f">{label}\n{sequence}\n", "mode": "env"}
        ).encode()
        response = self._json("ticket/msa", body)
        state = response["status"]
        if state in {"ERROR", "MAINTENANCE"}:
            raise RuntimeError(f"ColabFold submission ended with status {state!r}")
        job_id = response.get("id")
        if not isinstance(job_id, str) or not job_id:
            raise RuntimeError("ColabFold submission response is missing a job id")
        return job_id

    def poll(self, job_id: str) -> None:
        deadline = time.monotonic() + self.max_wait_seconds
        state = "UNKNOWN"
        while state in {"UNKNOWN", "RUNNING", "PENDING", "RATELIMIT"}:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"ColabFold search {job_id} timed out")
            # A rate limit is the server saying "come back", so it gets a longer
            # wait than an ordinary poll rather than being retried immediately.
            self.sleep(60.0 if state == "RATELIMIT" else self.poll_interval)
            state = self._json(f"ticket/{job_id}")["status"]
        if state != "COMPLETE":
            raise RuntimeError(f"ColabFold search {job_id} ended with {state!r}")

    def download(self, job_id: str, destination: str | Path) -> Path:
        """Fetch the result archive, resuming a partial file rather than restarting.

        The archive path is named for the ticket, so `-C -` can only ever resume
        onto the partial file of the same job. Resuming onto another ticket's
        leftover bytes would produce a corrupt tarball that still extracts.
        """
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "curl",
                "-sS",
                "-L",
                "--retry",
                "8",
                "--retry-all-errors",
                "--retry-delay",
                "15",
                "-C",
                "-",
                "-A",
                self.headers["User-Agent"],
                "-o",
                str(path),
                f"{self.host}/result/download/{job_id}",
            ],
            check=True,
            timeout=self.curl_timeout,
        )
        return path

    @staticmethod
    def extract(archive: str | Path) -> str:
        """Concatenate the uniref and environmental alignments from the tarball."""
        import tarfile

        chunks: list[str] = []
        with tarfile.open(archive) as tar:
            names = {member.name for member in tar.getmembers()}
            for name in _A3M_MEMBERS:
                if name not in names:
                    continue
                handle = tar.extractfile(name)
                if handle is None:
                    raise RuntimeError(f"ColabFold archive cannot read {name}")
                chunks.append(handle.read().decode("utf-8", errors="replace"))
        if not chunks:
            raise RuntimeError(
                f"ColabFold archive has none of {_A3M_MEMBERS}: {sorted(names)}"
            )
        return "".join(chunks)

    def search(self, sequence: str, label: str, work_dir: str | Path) -> str:
        job_id = self.submit(sequence, label)
        self.poll(job_id)
        archive = self.download(job_id, Path(work_dir) / f"{job_id}.tar.gz")
        return self.extract(archive)


def fetch_unpaired_msa(
    sequence: str,
    label: str,
    cache_dir_path: str | Path,
    *,
    client: ColabFoldUnpairedClient | None = None,
) -> tuple[str, bool]:
    """Return a sanitised alignment for one sequence, from cache when possible.

    Returns the a3m text and whether it came from the cache.
    """
    root = Path(cache_dir_path)
    key = sequence_key(sequence)
    entry = root / key
    alignment = entry / "unpaired.a3m"
    if alignment.is_file() and alignment.stat().st_size > 0:
        return alignment.read_text(encoding="utf-8"), True

    query = "".join(sequence.split()).upper()
    searcher = client if client is not None else ColabFoldUnpairedClient()
    entry.mkdir(parents=True, exist_ok=True)
    raw = searcher.search(query, label, entry)
    result = sanitise_a3m(raw, query, label=label)
    # Written through a temporary name so an interrupted write cannot be cached,
    # which is how `tests._foldbench.rcsb.fetch_cif` guards the same hazard.
    scratch = entry / ".unpaired.partial"
    scratch.write_text(result.text, encoding="utf-8")
    scratch.replace(alignment)
    (entry / "provenance.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "backend": {"name": searcher.name, "host": searcher.host},
                "cache_key": key,
                "label": label,
                "records": result.records,
                "dropped_columns": result.dropped_columns,
                "stripped_nuls": result.stripped_nuls,
                "sequence_sha256": key,
                "length": len(query),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return result.text, False


def case_msa_requests(job: Mapping[str, Any]) -> list[tuple[str, str, str]]:
    """`(label, sequence, relative a3m path)` for each protein entity, in order."""
    requests: list[tuple[str, str, str]] = []
    index = 0
    for entity in job["entities"]:
        if entity["type"] != "protein":
            continue
        index += 1
        label = f"{job['name']}_p{index}"
        target = entity.get("unpaired_msa") or f"../msa/{label}_unpaired.a3m"
        requests.append((label, entity["sequence"], target))
    return requests


def fetch_case_msas(
    set_dir: str | Path,
    *,
    cases: Sequence[str] | None = None,
    client: ColabFoldUnpairedClient | None = None,
    log: Callable[[str], None] = print,
) -> list[dict[str, Any]]:
    """Fill in every missing `msa/*.a3m` a set's job files refer to."""
    root = Path(set_dir)
    cache = root / "msa-cache"
    written: list[dict[str, Any]] = []
    for job_path in sorted((root / "jobs").glob("*.json")):
        job = json.loads(job_path.read_text(encoding="utf-8"))
        if cases is not None and job["name"] not in cases:
            continue
        for label, sequence, relative in case_msa_requests(job):
            destination = (job_path.parent / relative).resolve()
            if destination.is_file() and destination.stat().st_size > 0:
                log(f"have {destination.name}")
                continue
            started = time.time()
            text, cached = fetch_unpaired_msa(sequence, label, cache, client=client)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(text, encoding="utf-8")
            depth = text.count("\n>") + (1 if text.startswith(">") else 0)
            log(
                f"{label} L={len(sequence)} depth={depth} "
                f"{'cached' if cached else f'{time.time() - started:.0f}s'}"
            )
            written.append({"label": label, "depth": depth, "cached": cached})
    return written


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CaseReport:
    """What `validate` found for one case."""

    case: str
    pdb: str | None
    bucket: int | None
    recorded_length: int | None
    tokens: int | None
    in_band: bool | None
    alignments: int
    problems: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.problems


def _a3m_problems(path: Path, query: str, label: str) -> list[str]:
    problems: list[str] = []
    raw = path.read_bytes()
    if b"\x00" in raw:
        problems.append(f"{label}: contains {raw.count(0)} NUL bytes")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        return [*problems, f"{label}: is not valid UTF-8 ({error.reason})"]

    header_seen = False
    first: list[str] = []
    expected = len(query)
    ragged = 0
    records = 0
    for line in text.splitlines():
        if line.startswith(">"):
            header_seen = True
            records += 1
        elif line.strip():
            if records == 1:
                first.append(line.strip())
            elif _match_columns(line.strip()) != expected:
                ragged += 1
    if not header_seen:
        return [*problems, f"{label}: has no FASTA header"]
    query_row = _query_residues("".join(first))
    if query_row != query:
        problems.append(
            f"{label}: first record is not the query "
            f"({len(query_row)} residues, expected {expected})"
        )
    if ragged:
        problems.append(f"{label}: {ragged} records have the wrong column count")
    return problems


def validate_set(
    set_dir: str | Path,
    *,
    tolerance: float = TOLERANCE,
    heavy_atoms: Mapping[str, int] | None = None,
    check_alignments: bool = True,
    model: str = "boltz2",
) -> list[CaseReport]:
    """Check every job file, alignment and token count in a set directory.

    `model` names the backend whose acceptance is asserted. Boltz-2 is the
    default because it is the strictest of the five on CCD codes, so a job it
    accepts the others will too.
    """
    root = Path(set_dir)
    sequences: dict[str, Any] = {}
    sequences_path = root / "sequences.json"
    if sequences_path.is_file():
        sequences = json.loads(sequences_path.read_text(encoding="utf-8"))

    reports: list[CaseReport] = []
    for job_path in sorted((root / "jobs").glob("*.json")):
        problems: list[str] = []
        case = job_path.stem
        try:
            job = json.loads(job_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            reports.append(
                CaseReport(
                    case, None, None, None, None, None, 0, (f"unparsable: {error}",)
                )
            )
            continue

        # The harness's own answer, not a restatement of it. Checking the key
        # set against `input._JOB_KEYS` would say what the schema *allows*;
        # `compatibility` says what a backend *accepts*, which is the thing a
        # benchmark row depends on, and it is where the `pdb` key is refused.
        from foldjax.input import compatibility

        refusal = compatibility(job, model)
        if refusal is not None:
            problems.append(f"{model} refuses this job: {refusal}")
        if job.get("name") != case:
            problems.append(f"name {job.get('name')!r} does not match the file name")

        entities = job.get("entities")
        if not isinstance(entities, list) or not entities:
            problems.append("no entities")
            entities = []

        parsed: list[Entity] = []
        for entity in entities:
            kind = entity.get("type")
            copies = len(entity.get("id", []))
            if kind == "ligand":
                parsed.append(Entity(0, "ligand", copies, ccd=entity.get("ccd")))
            else:
                parsed.append(
                    Entity(0, str(kind), copies, sequence=entity.get("sequence"))
                )

        codes = {e.ccd for e in parsed if e.kind == "ligand" and e.ccd}
        tokens: int | None = None
        try:
            table = (
                dict(heavy_atoms)
                if heavy_atoms is not None
                else heavy_atom_counts(codes)
            )
            tokens = token_count(parsed, table)
        except (KeyError, OSError, ValueError) as error:
            problems.append(f"cannot count tokens: {error}")

        bucket = bucket_of(case)
        in_band: bool | None = None
        if bucket is not None and tokens is not None:
            low, high = bucket_band(bucket, tolerance)
            in_band = low <= tokens <= high
            if not in_band:
                problems.append(f"{tokens} tokens outside {low}-{high} for {bucket}")

        record = sequences.get(case)
        recorded = None
        if record is None:
            problems.append("missing from sequences.json")
        else:
            recorded = record.get("length")
            if job_sequence(job) != record.get("sequence"):
                problems.append("sequences.json sequence does not match the job")

        alignments = 0
        for label, sequence, relative in case_msa_requests(job):
            path = (job_path.parent / relative).resolve()
            if not path.is_file():
                problems.append(f"{label}: {relative} is missing")
                continue
            alignments += 1
            if check_alignments:
                problems.extend(_a3m_problems(path, sequence, label))

        reports.append(
            CaseReport(
                case=case,
                pdb=(record or {}).get("pdb"),
                bucket=bucket,
                recorded_length=recorded,
                tokens=tokens,
                in_band=in_band,
                alignments=alignments,
                problems=tuple(problems),
            )
        )
    return reports


def format_reports(reports: Sequence[CaseReport]) -> str:
    """A table, then one line per problem.

    `length` and `tokens` are both shown because they disagree wherever a set
    predates this module's token rule, and a reader has to be able to see which
    number a downstream row was labelled with.
    """
    header = (
        f"{'case':22s} {'pdb':6s} {'bucket':>7s} {'length':>7s} "
        f"{'tokens':>7s} {'band':>5s} {'msa':>4s}  status"
    )
    lines = [header, "-" * len(header)]
    for report in reports:
        band = "-" if report.in_band is None else ("ok" if report.in_band else "OUT")
        drift = ""
        if (
            report.recorded_length is not None
            and report.tokens is not None
            and report.recorded_length != report.tokens
        ):
            drift = f" (length off by {report.tokens - report.recorded_length:+d})"
        bucket = report.bucket if report.bucket is not None else "-"
        length = report.recorded_length if report.recorded_length is not None else "-"
        tokens = report.tokens if report.tokens is not None else "-"
        status = "ok" if report.ok else f"{len(report.problems)} problem(s)"
        lines.append(
            f"{report.case:22s} {report.pdb or '-':6s} {bucket:>7} {length:>7} "
            f"{tokens:>7} {band:>5s} {report.alignments:>4d}  {status}{drift}"
        )
    for report in reports:
        for problem in report.problems:
            lines.append(f"  {report.case}: {problem}")
    passed = sum(1 for report in reports if report.ok)
    lines.append(f"{passed}/{len(reports)} cases clean")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _cmd_select(args: argparse.Namespace) -> int:
    document: dict[str, Any] = {"tolerance": args.tolerance, "buckets": []}
    for target in args.bucket:
        candidates = select_candidates(
            target,
            tolerance=args.tolerance,
            max_resolution=args.max_resolution,
            limit=args.limit,
            cif_directory=args.cif_dir,
        )
        low, high = bucket_band(target, args.tolerance)
        chosen = next(
            (c for c in candidates if c.get("in_band") and c.get("refused") is None),
            None,
        )
        document["buckets"].append(
            {
                "target": target,
                "band": [low, high],
                "chosen": chosen["pdb"] if chosen else None,
                "case": case_name(chosen["pdb"], target) if chosen else None,
                "candidates": candidates,
            }
        )
        label = chosen["pdb"] if chosen else "none in band"
        print(f"{target:>5}  {low}-{high}  -> {label}", file=sys.stderr)
    text = json.dumps(document, indent=2)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


def _cmd_materialise(args: argparse.Namespace) -> int:
    case = args.case or case_name(args.pdb, args.bucket or 0)
    summary = materialise(args.pdb, case, args.set_dir, cif_directory=args.cif_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _cmd_fetch_msa(args: argparse.Namespace) -> int:
    fetch_case_msas(args.set_dir, cases=args.case or None)
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    reports = validate_set(args.set_dir, tolerance=args.tolerance, model=args.model)
    print(format_reports(reports))
    return 0 if all(report.ok for report in reports) else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bench.mixed_cases", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    select_parser = sub.add_parser("select", help="find candidate depositions")
    select_parser.add_argument(
        "--bucket", type=int, action="append", default=None, metavar="TOKENS"
    )
    select_parser.add_argument("--tolerance", type=float, default=TOLERANCE)
    select_parser.add_argument("--max-resolution", type=float, default=3.0)
    select_parser.add_argument("--limit", type=int, default=12)
    select_parser.add_argument("--cif-dir", type=Path, default=None)
    select_parser.add_argument("--out", type=Path, default=None)
    select_parser.set_defaults(func=_cmd_select)

    materialise_parser = sub.add_parser("materialise", help="write a job file")
    materialise_parser.add_argument("--pdb", required=True)
    materialise_parser.add_argument("--case", default=None)
    materialise_parser.add_argument("--bucket", type=int, default=None)
    materialise_parser.add_argument("--set-dir", type=Path, required=True)
    materialise_parser.add_argument("--cif-dir", type=Path, default=None)
    materialise_parser.set_defaults(func=_cmd_materialise)

    msa_parser = sub.add_parser("fetch-msa", help="fill in missing alignments")
    msa_parser.add_argument("--set-dir", type=Path, required=True)
    msa_parser.add_argument("--case", action="append", default=None)
    msa_parser.set_defaults(func=_cmd_fetch_msa)

    validate_parser = sub.add_parser("validate", help="check a set directory")
    validate_parser.add_argument("--set-dir", type=Path, required=True)
    validate_parser.add_argument("--tolerance", type=float, default=TOLERANCE)
    validate_parser.add_argument(
        "--model",
        default="boltz2",
        help="backend whose acceptance is asserted (default: the strictest)",
    )
    validate_parser.set_defaults(func=_cmd_validate)

    args = parser.parse_args(argv)
    if getattr(args, "bucket", None) is None and args.command == "select":
        args.bucket = list(BUCKETS)
    if args.command == "materialise" and args.case is None and not args.bucket:
        parser.error("materialise needs --case or --bucket")
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
