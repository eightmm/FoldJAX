"""Explicit pair-manifest contracts for written-structure diagnostics."""

from __future__ import annotations

import hashlib
import json

import pytest

from bench import written_structure_panel as panel

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


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_cif(path, rows):
    lines = ["data_test", "#", "loop_"] + [f"_atom_site.{field}" for field in _HEADER]
    for index, row in enumerate(rows, start=1):
        lines.append(
            " ".join(
                (
                    "ATOM",
                    str(index),
                    row["element"],
                    row["atom"],
                    ".",
                    row["component"],
                    row.get("chain", "A"),
                    "1",
                    str(row.get("sequence", index)),
                    "?",
                    *(str(value) for value in row["xyz"]),
                    "1.00",
                    "1",
                    str(row.get("sequence", index)),
                    row.get("chain", "A"),
                )
            )
        )
    path.write_text("\n".join(lines + ["#", ""]))


def manifest_pair(pair_id, left, right, **overrides):
    pair = {
        "pair_id": pair_id,
        "model": "model",
        "case": "case",
        "comparison": "cross",
        "noise_pairing": "unpaired",
        "left": {
            "path": left.name,
            "sha256": digest(left),
            "sample_id": "left-original",
        },
        "right": {
            "path": right.name,
            "sha256": digest(right),
            "sample_id": "right-original",
        },
    }
    pair.update(overrides)
    return pair


def write_manifest(path, pairs, schema_version=1):
    document = {"schema_version": schema_version, "pairs": pairs}
    path.write_text(json.dumps(document))
    return document


def test_success_preserves_manifest_and_full_comparator_output(tmp_path):
    left, right, manifest = (
        tmp_path / "left.cif",
        tmp_path / "right.cif",
        tmp_path / "pairs.json",
    )
    rows = [
        {"element": "C", "atom": "CA", "component": "ALA", "xyz": [0, 0, 0]},
        {"element": "C", "atom": "CA", "component": "ALA", "xyz": [1, 0, 0]},
        {"element": "C", "atom": "CA", "component": "ALA", "xyz": [0, 1, 0]},
        {"element": "C", "atom": "CA", "component": "ALA", "xyz": [0, 0, 1]},
    ]
    write_cif(left, rows)
    write_cif(right, rows)
    document = write_manifest(manifest, [manifest_pair("one", left, right)])

    result = panel.build_panel(manifest, tmp_path)

    record = result["pairs"][0]
    assert result["input_manifest"] == document
    assert record["manifest_pair"] == document["pairs"][0]
    assert record["status"] == "ok"
    assert record["result"]["metrics"]["all_atom_rmsd"] == pytest.approx(0.0, abs=1e-12)
    assert result["counts"] == {"total": 1, "ok": 1, "unscorable": 0}


def test_pure_rna_is_all_atom_with_ca_not_applicable(tmp_path):
    left, right, manifest = (
        tmp_path / "left.cif",
        tmp_path / "right.cif",
        tmp_path / "pairs.json",
    )
    rows = [{"element": "P", "atom": "P", "component": "A", "xyz": [0, 0, 0]}]
    write_cif(left, rows)
    write_cif(right, rows)
    write_manifest(manifest, [manifest_pair("rna", left, right)])

    result = panel.build_panel(manifest, tmp_path)

    metrics = result["pairs"][0]["result"]["metrics"]
    assert metrics["all_atom_rmsd"] == pytest.approx(0.0, abs=1e-12)
    assert metrics["ca_rmsd_same_fit"] is None


def test_identity_mismatch_is_retained_as_unscorable(tmp_path):
    left, right, manifest = (
        tmp_path / "left.cif",
        tmp_path / "right.cif",
        tmp_path / "pairs.json",
    )
    write_cif(
        left, [{"element": "C", "atom": "CA", "component": "ALA", "xyz": [0, 0, 0]}]
    )
    write_cif(
        right,
        [
            {
                "element": "C",
                "atom": "CA",
                "component": "ALA",
                "chain": "B",
                "xyz": [0, 0, 0],
            }
        ],
    )
    source_pair = manifest_pair("mismatch", left, right)
    write_manifest(manifest, [source_pair])

    result = panel.build_panel(manifest, tmp_path)

    record = result["pairs"][0]
    assert record["status"] == "unscorable"
    assert record["metrics"] is None
    assert record["manifest_pair"]["left"]["sample_id"] == "left-original"
    assert record["exception_type"] == "ValueError"
    assert "identity mismatch" in record["reason"]


@pytest.mark.parametrize(
    "change",
    [
        lambda pair: pair["left"].update(sha256="0" * 64),
        lambda pair: pair.update(comparison="not-a-comparison"),
        lambda pair: pair.update(noise_pairing="paired"),
    ],
)
def test_invalid_manifest_is_fatal_before_comparison(tmp_path, monkeypatch, change):
    left, right, manifest = (
        tmp_path / "left.cif",
        tmp_path / "right.cif",
        tmp_path / "pairs.json",
    )
    write_cif(left, [{"element": "P", "atom": "P", "component": "A", "xyz": [0, 0, 0]}])
    write_cif(
        right, [{"element": "P", "atom": "P", "component": "A", "xyz": [0, 0, 0]}]
    )
    pair = manifest_pair("one", left, right)
    change(pair)
    write_manifest(manifest, [pair])
    monkeypatch.setattr(
        panel, "compare_written_structures", lambda *_args: pytest.fail("called")
    )

    with pytest.raises(ValueError):
        panel.build_panel(manifest, tmp_path)


def test_duplicate_ids_missing_files_and_schema_are_fatal(tmp_path):
    left, right, manifest = (
        tmp_path / "left.cif",
        tmp_path / "right.cif",
        tmp_path / "pairs.json",
    )
    write_cif(left, [{"element": "P", "atom": "P", "component": "A", "xyz": [0, 0, 0]}])
    write_cif(
        right, [{"element": "P", "atom": "P", "component": "A", "xyz": [0, 0, 0]}]
    )
    pair = manifest_pair("same", left, right)
    write_manifest(manifest, [pair, pair])
    with pytest.raises(ValueError, match="duplicate"):
        panel.build_panel(manifest, tmp_path)

    missing = manifest_pair("missing", left, right)
    missing["right"]["path"] = "gone.cif"
    write_manifest(manifest, [missing])
    with pytest.raises(ValueError, match="missing"):
        panel.build_panel(manifest, tmp_path)

    write_manifest(manifest, [pair], schema_version=2)
    with pytest.raises(ValueError, match="schema"):
        panel.build_panel(manifest, tmp_path)


def test_identical_bytes_keep_distinct_manifest_sample_identities(tmp_path):
    left, right, manifest = (
        tmp_path / "left.cif",
        tmp_path / "right.cif",
        tmp_path / "pairs.json",
    )
    rows = [{"element": "P", "atom": "P", "component": "A", "xyz": [0, 0, 0]}]
    write_cif(left, rows)
    right.write_bytes(left.read_bytes())
    first = manifest_pair("first", left, right)
    second = manifest_pair("second", left, right)
    second["left"]["sample_id"] = "left-second-original"
    second["right"]["sample_id"] = "right-second-original"
    write_manifest(manifest, [first, second])

    result = panel.build_panel(manifest, tmp_path)

    assert [record["manifest_pair"]["pair_id"] for record in result["pairs"]] == [
        "first",
        "second",
    ]
    assert (
        result["pairs"][1]["manifest_pair"]["left"]["sample_id"]
        == "left-second-original"
    )


def test_cli_rejects_manifest_or_cif_output_alias(tmp_path, monkeypatch):
    left, right, manifest = (
        tmp_path / "left.cif",
        tmp_path / "right.cif",
        tmp_path / "pairs.json",
    )
    rows = [{"element": "P", "atom": "P", "component": "A", "xyz": [0, 0, 0]}]
    write_cif(left, rows)
    write_cif(right, rows)
    write_manifest(manifest, [manifest_pair("one", left, right)])
    monkeypatch.setattr(
        "sys.argv",
        [
            "written_structure_panel",
            "--manifest",
            str(manifest),
            "--root",
            str(tmp_path),
            "--output",
            str(manifest),
        ],
    )
    with pytest.raises(ValueError, match="manifest"):
        panel.main()

    monkeypatch.setattr(
        "sys.argv",
        [
            "written_structure_panel",
            "--manifest",
            str(manifest),
            "--root",
            str(tmp_path),
            "--output",
            str(left),
        ],
    )
    with pytest.raises(ValueError, match="referenced CIF"):
        panel.main()


@pytest.mark.parametrize("schema_version", [True, 1.0])
def test_schema_version_requires_exact_integer_one(tmp_path, schema_version):
    left, right, manifest = (
        tmp_path / "left.cif",
        tmp_path / "right.cif",
        tmp_path / "pairs.json",
    )
    rows = [{"element": "P", "atom": "P", "component": "A", "xyz": [0, 0, 0]}]
    write_cif(left, rows)
    write_cif(right, rows)
    write_manifest(manifest, [manifest_pair("one", left, right)], schema_version)

    with pytest.raises(ValueError, match="schema"):
        panel.build_panel(manifest, tmp_path)


def test_nonstring_comparison_is_explicit_validation_error(tmp_path, monkeypatch):
    left, right, manifest = (
        tmp_path / "left.cif",
        tmp_path / "right.cif",
        tmp_path / "pairs.json",
    )
    rows = [{"element": "P", "atom": "P", "component": "A", "xyz": [0, 0, 0]}]
    write_cif(left, rows)
    write_cif(right, rows)
    pair = manifest_pair("one", left, right, comparison=[])
    write_manifest(manifest, [pair])
    monkeypatch.setattr(
        panel, "compare_written_structures", lambda *_args: pytest.fail("called")
    )

    with pytest.raises(ValueError, match="invalid comparison"):
        panel.build_panel(manifest, tmp_path)


@pytest.mark.parametrize("target_name", ["manifest", "left"])
def test_cli_rejects_hardlink_alias_before_comparison(
    tmp_path, monkeypatch, target_name
):
    left, right, manifest = (
        tmp_path / "left.cif",
        tmp_path / "right.cif",
        tmp_path / "pairs.json",
    )
    rows = [{"element": "P", "atom": "P", "component": "A", "xyz": [0, 0, 0]}]
    write_cif(left, rows)
    write_cif(right, rows)
    write_manifest(manifest, [manifest_pair("one", left, right)])
    alias = tmp_path / "output-hardlink"
    alias.hardlink_to({"manifest": manifest, "left": left}[target_name])
    monkeypatch.setattr(
        panel, "compare_written_structures", lambda *_args: pytest.fail("called")
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "written_structure_panel",
            "--manifest",
            str(manifest),
            "--root",
            str(tmp_path),
            "--output",
            str(alias),
        ],
    )

    with pytest.raises(ValueError, match="manifest|referenced CIF"):
        panel.main()


def test_panel_passes_optional_explicit_chain_map_with_provenance(tmp_path):
    left, right, manifest = (
        tmp_path / "left.cif",
        tmp_path / "right.cif",
        tmp_path / "pairs.json",
    )
    left_rows = [
        {
            "element": "C",
            "atom": "CA",
            "component": "ALA",
            "chain": "A",
            "sequence": 1,
            "xyz": [0, 0, 0],
        },
        {
            "element": "C",
            "atom": "CA",
            "component": "ALA",
            "chain": "B",
            "sequence": 1,
            "xyz": [5, 1, 0],
        },
    ]
    right_rows = [
        dict(left_rows[1], chain="A"),
        dict(left_rows[0], chain="B"),
    ]
    write_cif(left, left_rows)
    write_cif(right, right_rows)
    pair = manifest_pair("mapped", left, right)
    pair["left_chain_to_right_chain"] = {"A": "B", "B": "A"}
    pair["chain_mapping_provenance"] = {
        "description": "externally recorded CA pair correspondence",
        "source_sha256": "a" * 64,
    }
    write_manifest(manifest, [pair])

    result = panel.build_panel(manifest, tmp_path)

    record = result["pairs"][0]
    assert record["status"] == "ok"
    assert record["manifest_pair"] == pair
    assert record["result"]["policies"]["left_chain_to_original_right_chain"] == {
        "A": "B",
        "B": "A",
    }


@pytest.mark.parametrize(
    "provenance",
    [
        None,
        {},
        {"description": "", "source_sha256": "a" * 64},
        {"description": "x", "source_sha256": "A" * 64},
    ],
)
def test_panel_rejects_missing_or_malformed_mapping_provenance_before_comparison(
    tmp_path, monkeypatch, provenance
):
    left, right, manifest = (
        tmp_path / "left.cif",
        tmp_path / "right.cif",
        tmp_path / "pairs.json",
    )
    row = {"element": "P", "atom": "P", "component": "A", "xyz": [0, 0, 0]}
    write_cif(left, [row])
    write_cif(right, [row])
    pair = manifest_pair("mapped", left, right)
    pair["left_chain_to_right_chain"] = {"A": "A"}
    if provenance is not None:
        pair["chain_mapping_provenance"] = provenance
    write_manifest(manifest, [pair])
    monkeypatch.setattr(
        panel,
        "compare_written_structures",
        lambda *_args, **_kwargs: pytest.fail("called"),
    )

    with pytest.raises(ValueError, match="chain_mapping_provenance"):
        panel.build_panel(manifest, tmp_path)
