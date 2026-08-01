from __future__ import annotations

from pathlib import Path

import pytest

from foldjax.models.chai.data.input import (
    EntityType,
    constituents_of_modified_fasta,
    identify_potential_entity_types,
    read_inputs,
    synthetic_chain_id,
)


@pytest.mark.parametrize(
    ("sequence", "expected"),
    [
        ("RKDES", ["R", "K", "D", "E", "S"]),
        (
            "(KCJ)(SEP)(PPN)(B3S)(BAL)(PPN)KX(NH2)",
            ["KCJ", "SEP", "PPN", "B3S", "BAL", "PPN", "K", "X", "NH2"],
        ),
        ("A()", None),
        ("A(K)", None),
        ("A(SEP", None),
        ("A((SEP))", None),
    ],
)
def test_modified_fasta_constituents_match_chai(
    sequence: str, expected: list[str] | None
) -> None:
    assert constituents_of_modified_fasta(sequence) == expected


def test_entity_type_heuristics_match_chai() -> None:
    assert EntityType.PROTEIN in identify_potential_entity_types("RKDES")
    assert EntityType.DNA in identify_potential_entity_types("ACGT")
    assert EntityType.RNA in identify_potential_entity_types("ACGU")
    assert EntityType.LIGAND in identify_potential_entity_types("[Mg+2]")
    assert identify_potential_entity_types("") == []


def test_read_inputs_supports_all_public_headers_and_multiline_sequences(
    tmp_path: Path,
) -> None:
    path = tmp_path / "input.fasta"
    path.write_text(
        ">protein|name=prot\nRKD\nES\n"
        ">dna|dna-a\nACGT\n"
        ">rna|name=rna-a\nACGU\n"
        ">ligand|lig\n[Mg+2]\n"
        ">glycan|name=sugar\nNAG(4-1 NAG)\n",
        encoding="utf-8",
    )

    inputs = read_inputs(path)

    assert [value.entity_type for value in inputs] == [0, 2, 1, 3, 7]
    assert [value.entity_name for value in inputs] == [
        "prot",
        "dna-a",
        "rna-a",
        "lig",
        "sugar",
    ]
    assert inputs[0].sequence == "RKDES"


@pytest.mark.parametrize(
    "header",
    ["protein", "unknown|x", "protein|name=x|extra=y", "protein|bad=x"],
)
def test_read_inputs_rejects_invalid_public_headers(
    tmp_path: Path, header: str
) -> None:
    path = tmp_path / "bad.fasta"
    path.write_text(f">{header}\nRKDES\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_inputs(path)


def test_read_inputs_enforces_optional_character_limit(tmp_path: Path) -> None:
    path = tmp_path / "input.fasta"
    path.write_text(">protein|p\nRKDES\n", encoding="utf-8")
    with pytest.raises(ValueError, match="too many chars"):
        read_inputs(path, length_limit=4)


def test_read_inputs_rejects_duplicate_entity_names(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.fasta"
    path.write_text(
        ">protein|name=target\nAA\n>ligand|name=target\nCCO\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unique name"):
        read_inputs(path)


def test_synthetic_chain_ids_match_chai() -> None:
    assert [synthetic_chain_id(index) for index in (0, 25, 26, 27, 701)] == [
        "A",
        "Z",
        "AA",
        "AB",
        "ZZ",
    ]
