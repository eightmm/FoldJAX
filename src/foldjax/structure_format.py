"""Legacy PDB copies of the canonical mmCIF structures.

``--structure-format pdb|both`` writes ``<job>_seed-<s>_sample-<nn>.pdb`` beside
each sample's mmCIF, converted with gemmi. The mmCIF stays in every mode: the
manifest's SHA-256, ``confidence_full.npz``'s atom axis and every reader in
FoldJAX refer to it, so the PDB is an additional file, never the record.

The PDB format cannot hold every structure, and gemmi's writer does not refuse
one that does not fit -- it silently switches to hybrid-36 atom serials past
99,999 and residue numbers past 9,999, and shifts columns for a two-character
chain id or a five-character CCD code. So FoldJAX refuses instead, and names
the limit:

- more than 99,999 atoms,
- a chain id longer than one character,
- a residue name longer than three characters (newer CCD codes have five),
- an atom name longer than four characters,
- a residue number outside -999..9999.

``pdb`` checks what it can from the job before the run (chain ids, CCD codes,
chain lengths) and fails the command after it if a written structure still
does not fit; ``both`` warns and keeps going, writing no PDB for that sample.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

FORMATS = ("cif", "pdb", "both")
MAX_ATOMS = 99_999
MAX_RESIDUE_NUMBER = 9_999
MIN_RESIDUE_NUMBER = -999


class StructureFormatError(ValueError):
    """A structure exceeds what the PDB format can hold."""


def pdb_violations(structure: Any) -> list[str]:
    """Every PDB-format limit a gemmi structure exceeds (first model)."""
    problems: list[str] = []
    if len(structure) == 0:
        return ["the structure has no model"]
    model = structure[0]
    atoms = sum(len(residue) for chain in model for residue in chain)
    if atoms > MAX_ATOMS:
        problems.append(f"{atoms} atoms (PDB holds at most {MAX_ATOMS})")
    long_chains = sorted({chain.name for chain in model if len(chain.name) > 1})
    if long_chains:
        problems.append(
            "chain ids longer than one character: " + ", ".join(long_chains[:10])
        )
    names: set[str] = set()
    atom_names: set[str] = set()
    numbers: set[int] = set()
    for chain in model:
        for residue in chain:
            if len(residue.name) > 3:
                names.add(residue.name)
            number = int(residue.seqid.num)
            if not MIN_RESIDUE_NUMBER <= number <= MAX_RESIDUE_NUMBER:
                numbers.add(number)
            for atom in residue:
                if len(atom.name) > 4:
                    atom_names.add(atom.name)
    if names:
        problems.append(
            "residue names longer than three characters: "
            + ", ".join(sorted(names)[:10])
        )
    if atom_names:
        problems.append(
            "atom names longer than four characters: "
            + ", ".join(sorted(atom_names)[:10])
        )
    if numbers:
        problems.append(
            f"residue numbers outside {MIN_RESIDUE_NUMBER}..{MAX_RESIDUE_NUMBER}: "
            + ", ".join(str(n) for n in sorted(numbers)[:5])
        )
    return problems


def convert(
    cif: str | os.PathLike[str], pdb: str | os.PathLike[str] | None = None
) -> Path:
    """Write the PDB copy of one mmCIF; refuse when it does not fit."""
    import gemmi

    cif = Path(cif)
    target = Path(pdb) if pdb is not None else cif.with_suffix(".pdb")
    structure = gemmi.read_structure(str(cif))
    structure.setup_entities()
    problems = pdb_violations(structure)
    if problems:
        raise StructureFormatError(
            f"{cif.name} cannot be written as PDB: "
            + "; ".join(problems)
            + ". The mmCIF is kept; use --structure-format cif or both"
        )
    options = gemmi.PdbWriteOptions()
    text = structure.make_pdb_string(options)
    staged = target.with_name(f".{target.name}.tmp")
    staged.write_text(text, encoding="utf-8")
    os.replace(staged, target)
    return target


def preflight(documents: Iterable[Mapping[str, Any]]) -> list[str]:
    """PDB limits a common job already exceeds before anything runs."""
    problems: list[str] = []
    for document in documents:
        name = document.get("name", "?")
        for entity in document.get("entities") or []:
            if not isinstance(entity, Mapping):
                continue
            ids = entity.get("id")
            for chain in ids if isinstance(ids, list) else [ids]:
                if isinstance(chain, str) and len(chain) > 1:
                    problems.append(
                        f"{name}: chain id {chain!r} is longer than one character"
                    )
            code = entity.get("ccd")
            if isinstance(code, str) and len(code) > 3:
                problems.append(
                    f"{name}: ligand CCD code {code!r} is longer than three characters"
                )
            sequence = entity.get("sequence")
            if isinstance(sequence, str) and len(sequence) > MAX_RESIDUE_NUMBER:
                problems.append(f"{name}: a chain of {len(sequence)} residues")
            for modification in entity.get("modifications") or []:
                if isinstance(modification, Mapping):
                    code = modification.get("ccd")
                    if isinstance(code, str) and len(code) > 3:
                        problems.append(
                            f"{name}: modification {code!r} is longer than three "
                            "characters"
                        )
    return problems


def write_formats(
    structures: Iterable[str | os.PathLike[str]], structure_format: str
) -> tuple[list[Path], list[str]]:
    """Write PDB copies for ``structure_format`` ``pdb``/``both``.

    Returns the files written and the per-structure refusals.
    """
    if structure_format not in FORMATS:
        raise ValueError(f"structure format must be one of {', '.join(FORMATS)}")
    written: list[Path] = []
    refused: list[str] = []
    if structure_format == "cif":
        return written, refused
    for path in structures:
        path = Path(path)
        if path.suffix.lower() not in {".cif", ".mmcif"} or not path.is_file():
            continue
        try:
            written.append(convert(path))
        except StructureFormatError as error:
            refused.append(str(error))
    return written, refused
