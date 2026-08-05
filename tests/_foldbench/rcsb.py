"""Fetch real structures from RCSB and describe them in a model-neutral way."""

from __future__ import annotations

import gzip
import io
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path

# Waters are stripped: every entry has them, no folding model takes them as input,
# and leaving them in would make ligand counts meaningless.
_SOLVENT = frozenset({"HOH", "DOD"})

# mmCIF entity_poly types mapped onto the three molecule types the models share.
_MOLECULE_TYPES = {
    "polypeptide(L)": "protein",
    "polypeptide(D)": "protein",
    "polyribonucleotide": "rna",
    "polydeoxyribonucleotide": "dna",
}


class UnsupportedEntityError(Exception):
    """An entry contains a polymer none of the models can express."""


@dataclass(frozen=True)
class Polymer:
    """One polymer entity, possibly present as several identical chains."""

    molecule_type: str
    sequence: str
    chain_ids: tuple[str, ...]

    @property
    def tokens(self) -> int:
        """Residues across all copies; one token per residue for polymers."""
        return len(self.sequence) * len(self.chain_ids)


@dataclass(frozen=True)
class Target:
    """A structure described so that any of the ports can be asked to predict it.

    Deliberately not a feature dict: the models disagree on almost every feature
    name, so the shared representation stops at the level they *do* agree on --
    sequences, molecule types, chain identity and ligand components.
    """

    pdb_id: str
    polymers: tuple[Polymer, ...]
    ligands: tuple[str, ...]

    @property
    def polymer_tokens(self) -> int:
        return sum(polymer.tokens for polymer in self.polymers)

    @property
    def biological_ligands(self) -> tuple[str, ...]:
        """Ligands with crystallization additives removed.

        ``ligands`` reports the deposition as it is, which is the right default
        for a fidelity check. This is the right one for a *cost* measurement:
        buffer and cryoprotectant components are tokenized per atom just like a
        cofactor, so leaving them in makes the token count of a target depend on
        how its crystal was grown rather than on the biology being predicted.
        """
        from tests._foldbench.targets import (
            CRYSTALLIZATION_ADDITIVES,
            UNSUPPORTED_COMPONENTS,
        )

        excluded = CRYSTALLIZATION_ADDITIVES | set(UNSUPPORTED_COMPONENTS)
        return tuple(code for code in self.ligands if code not in excluded)

    @property
    def molecule_types(self) -> tuple[str, ...]:
        return tuple(sorted({polymer.molecule_type for polymer in self.polymers}))

    def describe(self) -> str:
        parts = [
            f"{polymer.molecule_type}x{len(polymer.chain_ids)}:{len(polymer.sequence)}"
            for polymer in self.polymers
        ]
        ligands = f" +{','.join(self.ligands)}" if self.ligands else ""
        return f"{self.pdb_id} {' '.join(parts)}{ligands}"


def cache_dir() -> Path:
    """``$FOLDBENCH_CACHE``, else ``$XDG_CACHE_HOME``, else ``~/.cache``."""
    explicit = os.environ.get("FOLDBENCH_CACHE")
    if explicit:
        return Path(explicit)
    base = os.environ.get("XDG_CACHE_HOME")
    root = Path(base) if base else Path.home() / ".cache"
    return root / "tests._foldbench" / "rcsb"


def fetch_cif(pdb_id: str, *, directory: str | os.PathLike[str] | None = None) -> Path:
    """Download ``pdb_id``'s mmCIF once and return the local path.

    Cached because a benchmark run touches the same entries repeatedly, and
    re-downloading would make timings depend on the network.
    """
    target = Path(directory) if directory is not None else cache_dir()
    target.mkdir(parents=True, exist_ok=True)
    path = target / f"{pdb_id.upper()}.cif"
    if path.is_file() and path.stat().st_size > 0:
        return path

    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.cif.gz"
    with urllib.request.urlopen(url, timeout=120) as response:
        payload = gzip.decompress(response.read())
    # Written via a temporary name so an interrupted download cannot be cached.
    scratch = path.with_suffix(".partial")
    scratch.write_bytes(payload)
    scratch.replace(path)
    return path


def parse_target(path: str | os.PathLike[str], pdb_id: str | None = None) -> Target:
    """Read an mmCIF into a :class:`Target`.

    Raises:
        UnsupportedEntityError: a polymer type outside protein/DNA/RNA is present.
    """
    import biotite.structure.io.pdbx as pdbx

    text = Path(path).read_text()
    block = pdbx.CIFFile.read(io.StringIO(text)).block
    identifier = pdb_id or Path(path).stem

    polymers: list[Polymer] = []
    entity_poly = block.get("entity_poly")
    if entity_poly is not None:
        types = entity_poly["type"].as_array()
        sequences = entity_poly["pdbx_seq_one_letter_code_can"].as_array()
        strands = entity_poly["pdbx_strand_id"].as_array()
        for kind, sequence, strand in zip(types, sequences, strands, strict=True):
            molecule_type = _MOLECULE_TYPES.get(str(kind))
            if molecule_type is None:
                raise UnsupportedEntityError(
                    f"{identifier}: polymer type {kind!r} is not one of "
                    f"{sorted(_MOLECULE_TYPES)}"
                )
            # The one-letter code is wrapped across lines in the file.
            polymers.append(
                Polymer(
                    molecule_type=molecule_type,
                    sequence="".join(str(sequence).split()),
                    chain_ids=tuple(part.strip() for part in str(strand).split(",")),
                )
            )

    ligands: list[str] = []
    nonpoly = block.get("pdbx_entity_nonpoly")
    if nonpoly is not None:
        for comp_id in nonpoly["comp_id"].as_array():
            code = str(comp_id)
            if code not in _SOLVENT:
                ligands.append(code)

    return Target(
        pdb_id=identifier.upper(),
        polymers=tuple(polymers),
        ligands=tuple(sorted(set(ligands))),
    )


def load_target(pdb_id: str, *, directory: str | os.PathLike[str] | None = None):
    """Fetch and parse in one step."""
    return parse_target(fetch_cif(pdb_id, directory=directory), pdb_id=pdb_id)
