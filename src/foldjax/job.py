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
from collections.abc import Mapping, Sequence
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
    "Template",
    "parse_fasta",
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
class Template:
    """A structural template for one chain.

    ``query_to_template`` maps query residue index to template residue index.
    Boltz-2 aligns a bare mmCIF itself and takes no map; AlphaFold 3, Protenix
    and OpenDDE require one. Which of those a job can run is decided at
    materialization, where the model is known -- the same rule every other
    field follows.
    """

    mmcif: str | Path
    query_to_template: Mapping[int, int] | None = None
    chain_id: str | None = None

    def to_document(self) -> dict[str, Any]:
        document: dict[str, Any] = {"mmcif": str(self.mmcif)}
        if self.query_to_template is not None:
            document["query_indices"] = list(self.query_to_template.keys())
            document["template_indices"] = list(self.query_to_template.values())
        if self.chain_id is not None:
            document["chain_id"] = self.chain_id
        return document


@dataclass(frozen=True, slots=True)
class _Polymer:
    """One protein, DNA or RNA chain, with whatever is known about it."""

    id: str | tuple[str, ...]
    sequence: str
    unpaired_msa: str | Path | None = None
    paired_msa: str | Path | None = None
    modifications: tuple[Modification, ...] = ()
    templates: tuple[Template, ...] = ()

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
        if self.templates:
            document["templates"] = [item.to_document() for item in self.templates]
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
    #: The chain whose binding affinity to predict. Boltz-2 is the only carried
    #: model with that head, so every other backend refuses a job that asks.
    affinity_binder: str | None = None

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
        if self.affinity_binder is not None:
            document["properties"] = [
                {"affinity": {"binder": self.affinity_binder}}
            ]
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
        import copy

        from foldjax.input import (
            _JOB_KEYS,
            _LIGAND_KEYS,
            _POLYMER_KEYS,
            _affinity_binder,
            _bonds,
            _ids,
            _modifications,
            _normalize_sequence,
            _reject_unknown,
            _templates,
            assign_chain_ids,
        )

        if not isinstance(document, dict):
            raise ValueError("a FoldJAX job must be a mapping")
        # Chain ids are filled in below; a caller's own mapping is not the place
        # to leave that side effect.
        document = copy.deepcopy(document)
        _reject_unknown(set(document) - _JOB_KEYS, _JOB_KEYS, "top-level fields")
        raw_entities = document.get("entities")
        if not isinstance(raw_entities, list) or not raw_entities:
            raise ValueError("FoldJAX input requires a non-empty entities list")
        assign_chain_ids(raw_entities)

        entities: list[Entity] = []
        chains: set[str] = set()
        for raw in raw_entities:
            if not isinstance(raw, dict):
                raise ValueError("each entity must be an object")
            kind = raw.get("type")
            allowed = _LIGAND_KEYS if kind == "ligand" else _POLYMER_KEYS
            _reject_unknown(set(raw) - allowed, allowed, f"{kind} entity fields")
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
            raw["sequence"] = _normalize_sequence(
                raw.get("sequence"), kind=str(kind), chain=identifiers[0]
            )
            entities.append(
                polymer(
                    identifier,
                    sequence=raw["sequence"],
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
                    templates=tuple(
                        Template(
                            template["mmcif"],
                            query_to_template=(
                                dict(template["mapping"])
                                if template["mapping"]
                                else None
                            ),
                            chain_id=template["chain_id"],
                        )
                        for template in _templates(raw)
                    ),
                )
            )

        bonds = tuple(
            Bond(first, second) for first, second in _bonds(document, chains)
        )
        return cls(
            str(document.get("name", "")),
            tuple(entities),
            bonds,
            affinity_binder=_affinity_binder(document, chains),
        )

    @classmethod
    def read(cls, path: str | Path) -> Job:
        """Read a job file -- JSON or YAML, the same as prediction accepts."""
        from foldjax.input import read_job_document

        return cls.from_document(read_job_document(Path(path)))

    @classmethod
    def from_sequences(
        cls,
        protein: Sequence[str] = (),
        *,
        dna: Sequence[str] = (),
        rna: Sequence[str] = (),
        ligand_ccd: Sequence[str] = (),
        ligand_smiles: Sequence[str] = (),
        name: str = "job",
        affinity_binder: str | None = None,
    ) -> Job:
        """Build a job out of bare sequences, letting chain ids fall where they may.

        This is the smallest thing a person can hand a folding tool, and until
        now FoldJAX could not take it: the request needed a file, so the first
        prediction anyone ran began by learning a document schema. Chains are
        named in declaration order by `foldjax.input.assign_chain_ids`.
        """
        entities: list[Entity] = []
        for kind, sequences in (
            (Protein, protein),
            (Dna, dna),
            (Rna, rna),
        ):
            entities.extend(
                kind(_next_id(entities), sequence) for sequence in sequences
            )
        entities.extend(
            Ligand(_next_id(entities), ccd=code) for code in ligand_ccd
        )
        entities.extend(
            Ligand(_next_id(entities), smiles=smiles) for smiles in ligand_smiles
        )
        if not entities:
            raise ValueError("a job needs at least one sequence or ligand")
        return cls(name, tuple(entities), affinity_binder=affinity_binder)

    @classmethod
    def from_structure(cls, path: str | Path) -> Job:
        """Read an mmCIF or PDB file as a job. See `read_structure`."""
        return read_structure(path)

    @classmethod
    def from_fasta(cls, path: str | Path, *, kind: str = "protein") -> Job:
        """Read a FASTA file as one chain per record.

        Molecule type is *not* inferred from the letters: ``ACGT`` is a valid
        protein as well as a valid DNA sequence, and guessing wrong would fold
        the wrong polymer without saying so. Protein is the default because it
        is what a bare FASTA almost always is; anything else says so explicitly.
        """
        polymer = {"protein": Protein, "dna": Dna, "rna": Rna}.get(kind)
        if polymer is None:
            raise ValueError(f"FASTA chains must be protein, dna or rna; got {kind!r}")
        path = Path(path)
        records = parse_fasta(path.read_text(encoding="utf-8"))
        if not records:
            raise ValueError(f"no FASTA records in {path}")
        entities: list[Entity] = []
        for header, sequence in records:
            entities.append(polymer(_chain_from_header(header, entities), sequence))
        return cls(path.stem, tuple(entities))


#: Residue names that are solvent rather than chemistry worth folding.
_SOLVENT = frozenset({"HOH", "DOD", "WAT", "H2O"})


def _polymer_kind(polymer_type: Any) -> str | None:
    """Map gemmi's polymer classification onto the common schema's types."""
    name = str(polymer_type).rsplit(".", 1)[-1]
    if name.startswith("Peptide"):
        return "protein"
    if name == "Dna":
        return "dna"
    if name == "Rna":
        return "rna"
    return None


def read_structure(path: str | Path) -> Job:
    """Read an mmCIF or PDB file as a job: its chains, and what is bound to them.

    "Fold this deposition's sequence again, with a ligand" is one of the most
    common things anyone does with a structure predictor, and it used to mean
    opening the file and retyping the sequences. Only the chemistry is taken --
    sequences, ligand codes, chain names -- never the coordinates, because the
    point is to predict them.

    Solvent is dropped. A chain whose one-letter sequence gemmi cannot spell
    (a non-standard residue) is refused by name rather than silently truncated:
    the common schema expresses those as `Modification`, and guessing which one
    was meant would change the sequence being folded.
    """
    import gemmi

    path = Path(path)
    structure = gemmi.read_structure(str(path))
    structure.setup_entities()
    if not len(structure):
        raise ValueError(f"no models in structure: {path}")

    entities: list[Entity] = []
    for chain in structure[0]:
        polymer = chain.get_polymer()
        if len(polymer):
            kind = _polymer_kind(polymer.check_polymer_type())
            if kind is None:
                raise ValueError(
                    f"chain {chain.name!r} in {path} is not protein, DNA or RNA; "
                    "write the job document by hand for this one"
                )
            sequence = polymer.make_one_letter_sequence()
            if not sequence.isalpha():
                raise ValueError(
                    f"chain {chain.name!r} in {path} contains a residue with no "
                    f"one-letter code ({sequence!r}); express it as a "
                    "modification in a job document"
                )
            entities.append(
                {"protein": Protein, "dna": Dna, "rna": Rna}[kind](
                    chain.name, sequence
                )
            )
            continue
        for residue in chain:
            if residue.name.upper() in _SOLVENT:
                continue
            # Keep the deposition's own chain name where it is still free, so a
            # ligand in the job is the ligand a person sees in a viewer. Several
            # ligands sharing one chain get the next free labels.
            taken = {
                item
                for entity in entities
                for item in (
                    entity.id if isinstance(entity.id, tuple) else (entity.id,)
                )
            }
            identifier = (
                chain.name if chain.name not in taken else _next_id(entities)
            )
            entities.append(
                Ligand(identifier, ccd=residue.name.strip().upper())
            )
    if not entities:
        raise ValueError(f"no foldable chains or ligands in {path}")
    return Job(path.stem, tuple(entities))


def _next_id(entities: list[Entity]) -> str:
    """The next unused chain label, so a builder can hand ids out as it goes."""
    from foldjax.input import chain_labels

    taken = set()
    for entity in entities:
        identifier = entity.id
        taken.update(identifier if isinstance(identifier, tuple) else (identifier,))
    labels = chain_labels()
    label = next(labels)
    while label in taken:
        label = next(labels)
    return label


def _chain_from_header(header: str, entities: list[Entity]) -> str:
    """Use a FASTA header as the chain id when it is one, else the next label.

    A header like ``>A`` or ``>chain B`` is the writer naming the chain; a
    header like ``>sp|P69905|HBA_HUMAN`` is a description and must not become a
    chain id -- it is not a chain name, and it would end up in file names.
    """
    candidate = header.strip().split()[0] if header.strip() else ""
    if candidate.isalnum() and 1 <= len(candidate) <= 4:
        taken = {
            item
            for entity in entities
            for item in (
                entity.id if isinstance(entity.id, tuple) else (entity.id,)
            )
        }
        if candidate not in taken:
            return candidate
    return _next_id(entities)


def parse_fasta(text: str) -> list[tuple[str, str]]:
    """Return ``(header, sequence)`` pairs. Sequences keep their own validation.

    Deliberately small: whitespace and case are normalized where the document is
    validated (`foldjax.input`), so this only has to split records and refuse a
    file that is not FASTA at all.
    """
    records: list[tuple[str, str]] = []
    header: str | None = None
    chunks: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(">"):
            if header is not None:
                records.append((header, "".join(chunks)))
            header, chunks = stripped[1:], []
            continue
        if stripped.startswith(";") or not stripped:
            continue
        if header is None:
            raise ValueError("FASTA must start with a '>' header line")
        chunks.append(stripped)
    if header is not None:
        records.append((header, "".join(chunks)))
    empty = [head for head, sequence in records if not sequence]
    if empty:
        raise ValueError(f"FASTA record {empty[0]!r} has no sequence")
    return records
