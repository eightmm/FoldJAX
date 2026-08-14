"""Build a FoldJAX job in Python instead of by hand.

The common schema already exists -- `foldjax.input` reads it, validates it
against what each backend can express, and translates it into five native
dialects. What it did not have was a way to *construct* it: a caller driving
FoldJAX from Python assembled `{"entities": [{"type": "protein", ...}]}` with
string keys and wrote it to a file, so a misspelled ``unpaired_msa`` became a
silently MSA-less prediction rather than an error.

These classes are that constructor and nothing more. `to_document()` returns the
same mapping a job file holds, and every rule about what the document may say
stays in `foldjax.input`: one set of rules, checked once, at the point where the
model is known. A ligand carrying both a CCD code and a SMILES string is refused
there, not here, because "which of these can this backend express" is the same
question and it has one answer.

    from foldjax import Job, Ligand, Protein

    job = Job(
        "1abc",
        [
            Protein("A", sequence, unpaired_msa="alignments/1abc.a3m"),
            Ligand("L", ccd="ATP"),
        ],
    )
    request = PredictionRequest(model="protenix", input=job.write(tmp / "job.json"))

Relative paths are written through unchanged: `foldjax.input` resolves them
against the document's own directory, so a job stays movable together with its
alignments.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "Bond",
    "Dna",
    "Job",
    "Ligand",
    "Modification",
    "Protein",
    "Rna",
]

#: An atom, addressed the way the common schema addresses one: chain id, 1-based
#: residue index, atom name.
Atom = tuple[str, int, str]


def _document_text(value: Any, *, name: str) -> str | None:
    """Normalize an optional string field without stringifying invalid data."""
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


@dataclass(frozen=True, slots=True)
class Modification:
    """A residue replaced by a CCD component, at a 1-based position."""

    ccd: str
    position: int

    def to_document(self) -> dict[str, Any]:
        return {"ccd": self.ccd, "position": self.position}


@dataclass(frozen=True, slots=True)
class Bond:
    """A covalent bond between two atoms named across the whole job."""

    first: Atom
    second: Atom

    def to_document(self) -> list[list[Any]]:
        return [list(self.first), list(self.second)]


@dataclass(frozen=True, slots=True)
class _Polymer:
    """One protein, DNA or RNA chain, with whatever is known about it."""

    id: str | tuple[str, ...]
    sequence: str
    unpaired_msa: str | Path | None = None
    paired_msa: str | Path | None = None
    modifications: tuple[Modification, ...] = ()

    #: Set by each subclass; this is what lands in the document's ``type``.
    kind: str = field(init=False, default="")

    def to_document(self) -> dict[str, Any]:
        document: dict[str, Any] = {
            "type": self.kind,
            "id": list(self.id) if isinstance(self.id, tuple) else self.id,
            "sequence": self.sequence,
        }
        if self.unpaired_msa is not None:
            document["unpaired_msa"] = str(self.unpaired_msa)
        if self.paired_msa is not None:
            document["paired_msa"] = str(self.paired_msa)
        if self.modifications:
            document["modifications"] = [
                item.to_document() for item in self.modifications
            ]
        return document


@dataclass(frozen=True, slots=True)
class Protein(_Polymer):
    kind: str = field(init=False, default="protein")


@dataclass(frozen=True, slots=True)
class Dna(_Polymer):
    kind: str = field(init=False, default="dna")


@dataclass(frozen=True, slots=True)
class Rna(_Polymer):
    kind: str = field(init=False, default="rna")


@dataclass(frozen=True, slots=True)
class Ligand:
    """A ligand, named either by CCD code or by SMILES.

    Both are accepted here and the pair is refused at materialization, together
    with the check that the chosen one is a representation the backend takes --
    OpenFold3 and Boltz do not read the same one.
    """

    id: str | tuple[str, ...]
    ccd: str | None = None
    smiles: str | None = None

    def to_document(self) -> dict[str, Any]:
        document: dict[str, Any] = {
            "type": "ligand",
            "id": list(self.id) if isinstance(self.id, tuple) else self.id,
        }
        if self.ccd is not None:
            document["ccd"] = self.ccd
        if self.smiles is not None:
            document["smiles"] = self.smiles
        return document


Entity = Protein | Dna | Rna | Ligand


@dataclass(frozen=True, slots=True)
class Job:
    """One prediction target: what is in the box, and what is bonded to what."""

    name: str
    entities: tuple[Entity, ...]
    bonds: tuple[Bond, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "entities", tuple(self.entities))
        object.__setattr__(self, "bonds", tuple(self.bonds))

    def to_document(self) -> dict[str, Any]:
        """The mapping a job file holds. Not validated -- see the module docstring."""
        document: dict[str, Any] = {
            "name": self.name,
            "entities": [entity.to_document() for entity in self.entities],
        }
        if self.bonds:
            document["bonds"] = [bond.to_document() for bond in self.bonds]
        return document

    def write(self, path: str | Path) -> Path:
        """Write the job as JSON and return the path, so it can be passed on.

        Returning the path is the whole ergonomic point: `PredictionRequest`
        takes a file, and this keeps that one interface rather than adding a
        second way for input to arrive.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_document(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def from_document(cls, document: dict[str, Any]) -> Job:
        """Structure a job mapping. Unknown fields raise rather than vanish."""
        from foldjax.input import (
            _JOB_KEYS,
            _LIGAND_KEYS,
            _POLYMER_KEYS,
            _bonds,
            _ids,
            _modifications,
        )

        if not isinstance(document, dict):
            raise ValueError("a FoldJAX job must be a mapping")
        unknown = set(document) - _JOB_KEYS
        if unknown:
            raise ValueError(f"unsupported top-level fields: {sorted(unknown)}")
        raw_entities = document.get("entities")
        if not isinstance(raw_entities, list) or not raw_entities:
            raise ValueError("FoldJAX input requires a non-empty entities list")

        entities: list[Entity] = []
        chains: set[str] = set()
        for raw in raw_entities:
            if not isinstance(raw, dict):
                raise ValueError("each entity must be an object")
            kind = raw.get("type")
            allowed = _LIGAND_KEYS if kind == "ligand" else _POLYMER_KEYS
            unknown = set(raw) - allowed
            if unknown:
                raise ValueError(f"unsupported {kind} entity fields: {sorted(unknown)}")
            identifiers = _ids(raw)
            if any(not identifier.strip() for identifier in identifiers):
                raise ValueError("every entity requires a non-empty id")
            for chain in identifiers:
                if chain in chains:
                    raise ValueError(f"duplicate chain id: {chain!r}")
                chains.add(chain)
            identifier: str | tuple[str, ...] = (
                tuple(identifiers)
                if isinstance(raw.get("id"), list)
                else identifiers[0]
            )
            if kind == "ligand":
                ligand_values: dict[str, str | None] = {}
                for field_name in ("ccd", "smiles"):
                    value = raw.get(field_name)
                    ligand_values[field_name] = _document_text(
                        value, name=f"ligand {field_name}"
                    )
                entities.append(
                    Ligand(
                        identifier,
                        ccd=ligand_values["ccd"],
                        smiles=ligand_values["smiles"],
                    )
                )
                continue
            polymer = {"protein": Protein, "dna": Dna, "rna": Rna}.get(str(kind))
            if polymer is None:
                raise ValueError(f"unsupported entity type: {kind!r}")
            sequence = raw.get("sequence")
            if not isinstance(sequence, str) or not sequence.strip():
                raise ValueError(
                    f"{kind} entity requires a non-empty string sequence"
                )
            entities.append(
                polymer(
                    identifier,
                    sequence=sequence.strip(),
                    unpaired_msa=_document_text(
                        raw.get("unpaired_msa"), name="unpaired_msa"
                    ),
                    paired_msa=_document_text(
                        raw.get("paired_msa"), name="paired_msa"
                    ),
                    modifications=tuple(
                        Modification(ccd, position)
                        for ccd, position in _modifications(raw)
                    ),
                )
            )

        bonds = tuple(
            Bond(first, second) for first, second in _bonds(document, chains)
        )
        return cls(str(document.get("name", "")), tuple(entities), bonds)

    @classmethod
    def read(cls, path: str | Path) -> Job:
        """Read a job file -- JSON or YAML, the same as prediction accepts."""
        from foldjax.input import read_job_document

        return cls.from_document(read_job_document(Path(path)))
