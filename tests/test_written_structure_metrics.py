"""Contract tests for strict all-atom written-CIF comparison."""

from __future__ import annotations

import json

import numpy as np
import pytest

from bench.written_structure_metrics import compare_written_structures, main

_HEADER = (
    "group_PDB",
    "id",
    "type_symbol",
    "label_atom_id",
    "label_alt_id",
    "label_comp_id",
    "label_asym_id",
    "label_entity_id",
    "label_seq_id",
    "pdbx_PDB_ins_code",
    "Cartn_x",
    "Cartn_y",
    "Cartn_z",
    "occupancy",
    "pdbx_PDB_model_num",
    "auth_seq_id",
    "auth_asym_id",
)


def write_cif(path, rows, *, polymer_entities=("1",), polymer_types=None):
    """Write a compact literal atom_site fixture; coordinates come from each row."""
    text = ["data_test", "#"]
    if polymer_entities:
        if polymer_types is None:
            text += ["loop_", "_entity_poly.entity_id"] + list(polymer_entities) + ["#"]
        else:
            text += ["loop_", "_entity_poly.entity_id", "_entity_poly.type"]
            text += [
                f"{entity} {polymer_types[entity]}" for entity in polymer_entities
            ] + ["#"]
    text += ["loop_"] + [f"_atom_site.{field}" for field in _HEADER]
    for serial, row in enumerate(rows, start=1):
        text.append(
            " ".join(
                (
                    row.get("group", "ATOM"),
                    str(serial),
                    row["element"],
                    row["atom"],
                    row.get("alt", "."),
                    row["comp"],
                    row["chain"],
                    row.get("entity", "1"),
                    str(row["seq"]),
                    row.get("ins", "?"),
                    *(str(value) for value in row["xyz"]),
                    row.get("occupancy", "1.00"),
                    row.get("model", "1"),
                    str(row.get("auth_seq", row["seq"])),
                    row.get("auth_chain", row["chain"]),
                )
            )
        )
    path.write_text("\n".join(text + ["#", ""]))


def protein_rows(points, chain="A", entity="1"):
    return [
        {
            "element": "C",
            "atom": "CA",
            "comp": "ALA",
            "chain": chain,
            "entity": entity,
            "seq": index + 1,
            "xyz": point,
        }
        for index, point in enumerate(points)
    ]


def test_proper_rigid_rotation_and_translation_is_zero(tmp_path):
    left_points = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]
    )
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    right_points = left_points @ rotation.T + np.array([4.0, -3.0, 2.0])
    left, right = tmp_path / "left.cif", tmp_path / "right.cif"
    write_cif(left, protein_rows(left_points))
    write_cif(right, protein_rows(right_points))

    result = compare_written_structures(left, right)

    assert result["metrics"]["all_atom_rmsd"] < 1e-12
    assert result["metrics"]["ca_rmsd_same_fit"] < 1e-12
    assert (
        result["policies"]["raw_network_atom_mask"] == "not_available_from_written_cif"
    )
    assert result["units"] == "angstrom"
    assert result["policies"]["left_to_right_transform"][
        "translation"
    ] == pytest.approx([4.0, -3.0, 2.0])


def test_reflection_of_noncoplanar_geometry_remains_nonzero(tmp_path):
    points = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]
    )
    left, right = tmp_path / "left.cif", tmp_path / "right.cif"
    write_cif(left, protein_rows(points))
    write_cif(right, protein_rows(points * np.array([-1.0, 1.0, 1.0])))

    assert compare_written_structures(left, right)["metrics"]["all_atom_rmsd"] > 0.1


def test_chain_metrics_reuse_global_fit(tmp_path):
    shape = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 1.0]])
    left_rows = protein_rows(shape, "A") + protein_rows(
        shape + [10.0, 0.0, 0.0], "B", "2"
    )
    right_rows = protein_rows(shape, "A") + protein_rows(
        shape + [15.0, 0.0, 0.0], "B", "2"
    )
    left, right = tmp_path / "left.cif", tmp_path / "right.cif"
    write_cif(left, left_rows, polymer_entities=("1", "2"))
    write_cif(right, right_rows, polymer_entities=("1", "2"))

    result = compare_written_structures(left, right)

    assert result["metrics"]["per_chain_same_fit"]["A"] > 1.0
    assert result["metrics"]["per_chain_same_fit"]["B"] > 1.0


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        (
            [
                {
                    "element": "C",
                    "atom": "CA",
                    "comp": "ALA",
                    "chain": "A",
                    "seq": 1,
                    "xyz": [0, 0, 0],
                },
                {
                    "element": "C",
                    "atom": "CA",
                    "comp": "ALA",
                    "chain": "A",
                    "seq": 1,
                    "xyz": [1, 0, 0],
                },
            ],
            "duplicate",
        ),
        (
            [
                {
                    "element": "C",
                    "atom": "CA",
                    "comp": "ALA",
                    "chain": "A",
                    "seq": 1,
                    "xyz": ["nan", 0, 0],
                }
            ],
            "nonfinite",
        ),
    ],
)
def test_duplicate_and_nonfinite_inputs_are_rejected(tmp_path, rows, message):
    bad, good = tmp_path / "bad.cif", tmp_path / "good.cif"
    write_cif(bad, rows)
    write_cif(good, protein_rows([[0.0, 0.0, 0.0]]))
    with pytest.raises(ValueError, match=message):
        compare_written_structures(bad, good)


def test_mismatched_identity_is_rejected(tmp_path):
    left, right = tmp_path / "left.cif", tmp_path / "right.cif"
    write_cif(left, protein_rows([[0.0, 0.0, 0.0]]))
    write_cif(right, protein_rows([[0.0, 0.0, 0.0]], chain="B"))
    with pytest.raises(ValueError, match="identity mismatch"):
        compare_written_structures(left, right)


def test_nonprotein_and_calcium_ca_are_not_protein_ca(tmp_path):
    rows = [
        {
            "group": "HETATM",
            "element": "P",
            "atom": "P",
            "comp": "A",
            "chain": "R",
            "entity": "2",
            "seq": 1,
            "xyz": [0, 0, 0],
        },
        {
            "group": "HETATM",
            "element": "C",
            "atom": "C1",
            "comp": "LIG",
            "chain": "L",
            "entity": "3",
            "seq": 1,
            "xyz": [1, 0, 0],
        },
        {
            "group": "HETATM",
            "element": "CA",
            "atom": "CA",
            "comp": "CA",
            "chain": "I",
            "entity": "4",
            "seq": 1,
            "xyz": [2, 0, 0],
        },
    ]
    left, right = tmp_path / "left.cif", tmp_path / "right.cif"
    write_cif(left, rows, polymer_entities=("2",))
    write_cif(right, rows, polymer_entities=("2",))

    result = compare_written_structures(left, right)

    assert result["metrics"]["all_atom_rmsd"] == 0.0
    assert result["metrics"]["ca_rmsd_same_fit"] is None
    assert result["coverage"]["ca"]["matched"] == 0


def test_insertion_codes_and_more_than_four_chains_are_preserved(tmp_path):
    rows = []
    for index, chain in enumerate("ABCDE"):
        rows.extend(protein_rows([[float(index), 0.0, 0.0]], chain, str(index + 1)))
    rows.append(
        {
            "element": "C",
            "atom": "CB",
            "comp": "ALA",
            "chain": "A",
            "entity": "1",
            "seq": 1,
            "ins": "A",
            "xyz": [0.0, 1.0, 0.0],
        }
    )
    rows.append(
        {
            "element": "C",
            "atom": "CB",
            "comp": "ALA",
            "chain": "A",
            "entity": "1",
            "seq": 1,
            "ins": "B",
            "xyz": [0.0, 2.0, 0.0],
        }
    )
    left, right = tmp_path / "left.cif", tmp_path / "right.cif"
    write_cif(left, rows, polymer_entities=("1", "2", "3", "4", "5"))
    write_cif(right, rows, polymer_entities=("1", "2", "3", "4", "5"))

    result = compare_written_structures(left, right)

    assert result["atom_identity"]["matched_atoms"] == 7
    assert set(result["metrics"]["per_chain_same_fit"]) == set("ABCDE")


def test_cli_json_has_hashes_and_no_nan(tmp_path, monkeypatch):
    left, right, output = (
        tmp_path / "left.cif",
        tmp_path / "right.cif",
        tmp_path / "metric.json",
    )
    rows = protein_rows([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    write_cif(left, rows)
    write_cif(right, rows)
    monkeypatch.setattr(
        "sys.argv",
        [
            "written_structure_metrics",
            "--left",
            str(left),
            "--right",
            str(right),
            "--output",
            str(output),
        ],
    )
    main()

    raw = output.read_text()
    assert "NaN" not in raw
    result = json.loads(raw)
    assert result["schema_version"] == "1.0"
    assert len(result["inputs"]["left"]["sha256"]) == 64


def test_matching_identity_rejects_element_and_ca_metadata_disagreement(tmp_path):
    rows = protein_rows([[0.0, 0.0, 0.0]])
    left, right = tmp_path / "left.cif", tmp_path / "right.cif"
    write_cif(left, rows)
    changed_element = [dict(rows[0], element="CA")]
    write_cif(right, changed_element)
    with pytest.raises(ValueError, match="element differs"):
        compare_written_structures(left, right)

    write_cif(left, rows, polymer_types={"1": "polypeptide(L)"})
    write_cif(right, rows, polymer_types={"1": "polyribonucleotide"})
    with pytest.raises(ValueError, match="ambiguous CA selection"):
        compare_written_structures(left, right)


def test_missing_or_unknown_element_is_rejected(tmp_path):
    rows = protein_rows([[0.0, 0.0, 0.0]])
    rows[0]["element"] = "?"
    left, right = tmp_path / "left.cif", tmp_path / "right.cif"
    write_cif(left, rows)
    write_cif(right, rows)
    with pytest.raises(ValueError, match="missing required type_symbol"):
        compare_written_structures(left, right)


def test_declared_polypeptide_selects_modified_protein_ca(tmp_path):
    rows = [
        {
            "element": "C",
            "atom": "CA",
            "comp": "PTR",
            "chain": "A",
            "entity": "1",
            "seq": 1,
            "xyz": [0.0, 0.0, 0.0],
        }
    ]
    left, right = tmp_path / "left.cif", tmp_path / "right.cif"
    types = {"1": "polypeptide(L)"}
    write_cif(left, rows, polymer_types=types)
    write_cif(right, rows, polymer_types=types)
    result = compare_written_structures(left, right)
    assert result["coverage"]["ca"]["matched"] == 1


def test_nonpolymer_residues_in_one_chain_entity_stay_separate(tmp_path):
    rows = [
        {
            "group": "HETATM",
            "element": "C",
            "atom": "C1",
            "comp": "LIG",
            "chain": "L",
            "entity": "2",
            "seq": sequence,
            "xyz": [float(sequence), 0.0, 0.0],
        }
        for sequence in (1, 2)
    ]
    left, right = tmp_path / "left.cif", tmp_path / "right.cif"
    write_cif(left, rows, polymer_entities=())
    write_cif(right, rows, polymer_entities=())
    result = compare_written_structures(left, right)
    counts = result["metrics"]["per_entity_instance_atom_counts"]
    assert len(counts) == 2
    assert set(counts.values()) == {1}
    assert result["policies"]["left"]["entity_metadata_uncertainty"]
