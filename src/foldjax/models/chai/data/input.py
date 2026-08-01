"""Torch-free parsing of Chai's public FASTA input contract."""

from __future__ import annotations

import string
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class EntityType(Enum):
    """Integer values used by the released Chai feature pipeline."""

    PROTEIN = 0
    RNA = 1
    DNA = 2
    LIGAND = 3
    POLYMER_HYBRID = 4
    WATER = 5
    UNKNOWN = 6
    MANUAL_GLYCAN = 7


@dataclass(frozen=True)
class Input:
    sequence: str
    entity_type: int
    entity_name: str


def constituents_of_modified_fasta(sequence: str) -> list[str] | None:
    """Split one-letter and bracketed residues using Chai's exact grammar."""
    sequence = sequence.strip().upper()
    allowed = string.ascii_letters + "()[]" + string.digits
    if not all(character in allowed for character in sequence):
        return None

    current_modified: str | None = None
    constituents: list[str] = []
    for character in sequence:
        if character in "([":
            if current_modified is not None:
                return None
            current_modified = ""
        elif character in ")]":
            if current_modified is None or len(current_modified) <= 1:
                return None
            constituents.append(current_modified)
            current_modified = None
        elif current_modified is not None:
            current_modified += character
        else:
            if character not in string.ascii_letters:
                return None
            constituents.append(character)
    return constituents if current_modified is None else None


def identify_potential_entity_types(sequence: str) -> list[EntityType]:
    """Return the same lightweight type hints as the upstream parser."""
    sequence = sequence.strip()
    if not sequence:
        return []
    possible: list[EntityType] = []
    constituents = constituents_of_modified_fasta(sequence)
    if constituents is not None:
        one_letter = {value for value in constituents if len(value) == 1}
        if one_letter.issubset(set("AGTC")):
            possible.append(EntityType.DNA)
        if one_letter.issubset(set("AGUC")):
            possible.append(EntityType.RNA)
        if "U" not in one_letter:
            possible.append(EntityType.PROTEIN)

    smiles_symbols = string.ascii_letters + string.digits + ".-+=#$%:/\\[]()<>@"
    if set(sequence.upper()).issubset(set(smiles_symbols)):
        possible.extend((EntityType.LIGAND, EntityType.MANUAL_GLYCAN))
    return possible


def _read_fasta_records(path: str | Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    header: str | None = None
    sequence_lines: list[str] = []
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                records.append((header, "".join(sequence_lines)))
            header = line[1:].strip()
            sequence_lines = []
        else:
            if header is None:
                raise ValueError("FASTA sequence appears before its header")
            sequence_lines.append(line)
    if header is not None:
        records.append((header, "".join(sequence_lines)))
    if not records:
        raise ValueError("FASTA input contains no records")
    if any(not header or not sequence for header, sequence in records):
        raise ValueError("FASTA records require non-empty headers and sequences")
    return records


def read_inputs(fasta_file: str | Path, length_limit: int | None = None) -> list[Input]:
    """Parse Chai entity headers without importing BioPython or PyTorch."""
    inputs: list[Input] = []
    entity_names: set[str] = set()
    total_length = 0
    names = {
        "protein": EntityType.PROTEIN,
        "ligand": EntityType.LIGAND,
        "rna": EntityType.RNA,
        "dna": EntityType.DNA,
        "glycan": EntityType.MANUAL_GLYCAN,
    }
    for description, sequence in _read_fasta_records(fasta_file):
        entity_text, *parts = description.split("|")
        try:
            entity_type = names[entity_text.lower().strip()]
        except KeyError as error:
            raise ValueError(f"{entity_text} is not a valid entity type") from error
        if len(parts) != 1:
            raise ValueError(f"Unsupported inputs: desc={description!r}")
        label = parts[0].strip()
        if "=" in label:
            field_name, entity_name = label.split("=", maxsplit=1)
            if field_name != "name":
                raise ValueError(f"Unsupported input field: {field_name}")
        else:
            entity_name = label
        if not entity_name:
            raise ValueError(f"label is not provided in desc={description!r}")
        if entity_name in entity_names:
            raise ValueError(
                f"name={entity_name!r} used more than once in inputs; "
                "each entity must have a unique name"
            )
        entity_names.add(entity_name)
        inputs.append(Input(sequence, entity_type.value, entity_name))
        total_length += len(sequence)
    if length_limit is not None and total_length > length_limit:
        raise ValueError(
            f"[fasta] [{fasta_file}] too many chars "
            f"({total_length} > {length_limit}); skipping"
        )
    return inputs


def synthetic_chain_id(index: int) -> str:
    """Return Chai's A..Z, AA..ZZ synthetic input-chain identifiers."""
    if index < 0:
        raise ValueError("chain index must be non-negative")
    alphabet_size = len(string.ascii_uppercase)
    value = ""
    while index >= 0:
        value = string.ascii_uppercase[index % alphabet_size] + value
        index = index // alphabet_size - 1
    return value
