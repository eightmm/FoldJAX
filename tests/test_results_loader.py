"""`load_results`, `results_table`, `aggregate_table`, `show --format` and `compare`.

Every directory here is written by the real pipeline from the fixtures in
``tests/fixtures/outputs`` (see `tests/test_output_contract.py`), laid out the
way a batch lays itself out: ``<root>/<model>/<input stem>``.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import pytest

import foldjax
from foldjax import cli
from foldjax.results import (
    BEST_SELECTION,
    aggregate_table,
    load_results,
    results_table,
    to_csv,
)
from tests.test_output_contract import _run, write_job

MONOMER = ("alphafold3", "boltz2", "esmfold2", "opendde", "openfold3", "protenix")


def _batch(tmp_path: Path, models=("boltz2", "protenix"), target="e9_8reh") -> Path:
    root = tmp_path / "batch"
    job = write_job(tmp_path / "jobs", target)
    for model in models:
        _run(
            tmp_path / "scratch" / model,
            f"{model}_{target}",
            out=root / model / target,
            job=job,
        )
    return root


def test_one_row_per_sample_from_canonical_files(tmp_path: Path) -> None:
    root = _batch(tmp_path, MONOMER)

    report = foldjax.load_results(root)
    rows = foldjax.results_table(report)

    assert len(report.runs) == 6
    assert len(rows) == 6
    assert {row["model"] for row in rows} == set(MONOMER)
    for row in rows:
        assert row["status"] == "ok"
        assert row["structure_verified"] is True
        assert len(row["structure_sha256"]) == 64
        assert Path(row["structure_path"]).is_file()
        assert row["summary_origin"] == "confidence.json"
        assert 50.0 < row["plddt"] <= 100.0
        assert row["iptm"] is None and "single chain" in row["summary_missing"]["iptm"]
        # Common-schema input: inspected, and nothing was dropped.
        assert row["ignored_msas"] == [] and row["ignored_templates"] == []
        assert row["sample"] == 0 and row["seed"] == 101
    by_model = {row["model"]: row for row in rows}
    assert by_model["boltz2"]["plddt_source"] == "scores.complex_plddt"
    assert by_model["protenix"]["native_rank"] == 3
    assert by_model["protenix"]["execution.num_recycles"] == 10
    assert by_model["protenix"]["score.has_clash"] == 0.0
    assert "score.num_recycles" not in by_model["protenix"]
    assert by_model["openfold3"]["ranking_value"] is None
    assert by_model["openfold3"]["best_within_model"] is False
    assert by_model["boltz2"]["best_within_model"] is True


def test_the_canonical_file_wins_and_native_side_files_are_never_read(
    tmp_path: Path,
) -> None:
    root = _batch(tmp_path, ("protenix",))
    run = root / "protenix" / "e9_8reh"
    confidence_path = run / "seed-101_sample-00" / "confidence.json"
    confidence = json.loads(confidence_path.read_text())
    confidence["scores"]["ptm"] = 0.123
    confidence["summary"]["ptm"]["value"] = 0.123
    confidence_path.write_text(json.dumps(confidence))
    for native in run.rglob("*_summary_confidence_sample_*.json"):
        native.write_text(json.dumps({"ptm": 0.999, "plddt": 1.0}))

    (row,) = results_table(load_results(root))

    assert row["ptm"] == 0.123
    assert row["score.ptm"] == 0.123


def test_a_run_from_before_the_summary_gets_one_computed_and_labelled(
    tmp_path: Path,
) -> None:
    root = _batch(tmp_path, ("protenix",))
    run = root / "protenix" / "e9_8reh"
    confidence_path = run / "seed-101_sample-00" / "confidence.json"
    legacy = json.loads(confidence_path.read_text())
    legacy = {key: legacy[key] for key in ("model", "seed", "sample", "scores")}
    legacy["scores"]["num_recycles"] = 10.0
    confidence_path.write_text(json.dumps(legacy))

    (row,) = results_table(load_results(root))

    assert row["summary_origin"] == "computed"
    assert row["plddt"] == pytest.approx(legacy["scores"]["plddt"])
    assert row["execution.num_recycles"] == 10.0
    assert "score.num_recycles" not in row


def test_failures_are_rows_and_a_finished_rerun_supersedes_one(tmp_path: Path) -> None:
    root = _batch(tmp_path, ("boltz2",))
    finished_input = json.loads(
        (root / "boltz2" / "e9_8reh" / "foldjax_run.json").read_text()
    )["input"]["path"]
    (root / "foldjax_failures.json").write_text(
        json.dumps(
            [
                {
                    "model": "openfold3",
                    "input": finished_input,
                    "seed": 101,
                    "output_dir": str(root / "openfold3" / "e9_8reh"),
                    "error_type": "MemoryError",
                    "error": "openfold3 ran out of memory",
                },
                {
                    "model": "boltz2",
                    "input": finished_input,
                    "seed": 101,
                    "output_dir": str(root / "boltz2" / "e9_8reh"),
                    "error_type": "PredictionError",
                    "error": "an earlier attempt that a later run finished",
                },
            ]
        )
    )

    rows = results_table(load_results(root))

    failed = [row for row in rows if row["status"] == "failed"]
    assert [(row["model"], row["error_type"]) for row in failed] == [
        ("openfold3", "MemoryError")
    ]
    assert len(rows) == 2
    summary = aggregate_table(rows)
    assert {(item["model"], item["samples"], item["failures"]) for item in summary} == {
        ("boltz2", 1, 0),
        ("openfold3", 0, 1),
    }


def test_a_changed_structure_is_flagged_not_trusted(tmp_path: Path) -> None:
    root = _batch(tmp_path, ("boltz2",))
    structure = next((root / "boltz2").rglob("*_sample-00.cif"))
    structure.write_text(structure.read_text() + "#\n")

    (row,) = results_table(load_results(root))

    assert row["status"] == "structure missing or changed"
    assert row["structure_verified"] is False


def test_a_manifest_cannot_point_the_loader_outside_its_root(tmp_path: Path) -> None:
    root = _batch(tmp_path, ("boltz2",))
    manifest_path = root / "boltz2" / "e9_8reh" / "foldjax_run.json"
    manifest = json.loads(manifest_path.read_text())
    outside = tmp_path / "outside.cif"
    outside.write_text("data_x\n")
    manifest["samples"][0]["structure_path"] = "../../../outside.cif"
    manifest_path.write_text(json.dumps(manifest))

    (row,) = results_table(load_results(root))

    assert row["structure_path"] is None
    assert row["status"] != "ok"


def test_aggregation_never_crosses_models_and_labels_best(tmp_path: Path) -> None:
    root = _batch(tmp_path, ("boltz2", "protenix"))
    # A second, identical configuration of boltz2 is a repeat and pools; the
    # same model on another checkpoint is another configuration and does not.
    job = next((tmp_path / "jobs").glob("*.json"))
    _run(tmp_path / "scratch" / "boltz2", "boltz2_e9_8reh", out=root / "again", job=job)
    _run(tmp_path / "scratch" / "other", "boltz2_e9_8reh", out=root / "other", job=job)

    summary = aggregate_table(load_results(root))

    assert sorted((item["model"], item["samples"]) for item in summary) == [
        ("boltz2", 1),
        ("boltz2", 2),
        ("protenix", 1),
    ]
    for item in summary:
        assert item["best_within_model"]["selection"] == BEST_SELECTION
        assert item["plddt_n"] == item["samples"]
        assert item["plddt_spread"] == pytest.approx(0.0)


def test_csv_has_one_header_and_json_nests(tmp_path: Path) -> None:
    rows = results_table(load_results(_batch(tmp_path)))
    parsed = list(csv.DictReader(io.StringIO(to_csv(rows))))

    assert len(parsed) == 2
    assert parsed[0]["ignored_msas"] == "[]"
    assert "score.ptm" in parsed[0]


@pytest.mark.parametrize("fmt", ["csv", "json"])
def test_show_format_prints_rows(tmp_path: Path, capsys, fmt: str) -> None:
    root = _batch(tmp_path)

    assert cli.main(["show", str(root), "--format", fmt]) == 0
    printed = capsys.readouterr().out

    if fmt == "json":
        rows = json.loads(printed)
        assert {row["model"] for row in rows} == {"boltz2", "protenix"}
    else:
        assert printed.splitlines()[0].startswith("status,model,input")
        assert len(printed.splitlines()) == 3


def test_show_aggregate_and_legacy_json_are_unchanged(tmp_path: Path, capsys) -> None:
    root = _batch(tmp_path)

    assert cli.main(["show", str(root), "--format", "json", "--aggregate"]) == 0
    aggregated = json.loads(capsys.readouterr().out)
    assert {item["model"] for item in aggregated} == {"boltz2", "protenix"}

    assert cli.main(["show", str(root), "--json"]) == 0
    manifests = json.loads(capsys.readouterr().out)
    assert all("schema_version" in document for document in manifests)


def test_compare_writes_every_ordered_pair_with_its_correspondence(
    tmp_path: Path, capsys
) -> None:
    root = _batch(tmp_path, ("alphafold3", "boltz2", "protenix"))
    out = tmp_path / "cmp"

    assert cli.main(["compare", str(root), "--out", str(out)]) == 0
    written = json.loads(capsys.readouterr().out)

    document = json.loads(Path(written["json"]).read_text())
    (entry,) = document["inputs"]
    assert len(entry["structures"]) == 3
    assert len(entry["pairs"]) == 6
    keys = {item["key"] for item in entry["structures"]}
    assert keys == {
        "alphafold3/seed-101/sample-00",
        "boltz2/seed-101/sample-00",
        "protenix/seed-101/sample-00",
    }
    for pair in entry["pairs"]:
        assert pair["reference"] != pair["mobile"]
        assert pair["rmsd_angstrom"] >= 0.0
        assert pair["coverage"] == pytest.approx(1.0)
        assert pair["matched_atoms"] == 129
        (correspondence,) = entry["correspondences"][pair["correspondence"]]
        assert correspondence["runs"] == [[1, 1, 129]]
    for item in entry["structures"]:
        assert item["ignored_msas"] == []
    pairs_csv = list(
        csv.DictReader(io.StringIO(Path(written["pairs_csv"]).read_text()))
    )
    assert len(pairs_csv) == 6
    assert {row["reference_model"] for row in pairs_csv} == {
        "alphafold3",
        "boltz2",
        "protenix",
    }
    assert Path(written["structures_csv"]).read_text().count("\n") == 4


def test_compare_best_only_falls_back_for_a_model_that_ranks_nothing(
    tmp_path: Path,
) -> None:
    from foldjax.compare import compare_rows

    root = _batch(tmp_path, ("boltz2", "openfold3"))

    document = compare_rows(root, samples="best")

    (entry,) = document["inputs"]
    flags = {item["model"]: item["best_within_model"] for item in entry["structures"]}
    assert flags == {"boltz2": True, "openfold3": False}


def test_confidence_array_records_validate_and_reach_the_row(tmp_path: Path) -> None:
    """The I4 manifest fields, in the shape that worker writes them."""
    from foldjax.summary import load_schema
    from tests._schema_lite import errors

    root = _batch(tmp_path, ("boltz2",))
    manifest_path = root / "boltz2" / "e9_8reh" / "foldjax_run.json"
    manifest = json.loads(manifest_path.read_text())
    assert results_table(load_results(root))[0]["confidence_arrays"] is None
    manifest["confidence_arrays"] = {
        "file": "confidence_full.npz",
        "schema_version": 1,
        "samples": 1,
        "arrays": ["pae", "plddt"],
        "unavailable": {"pde": "not written by this backend"},
    }
    manifest["samples"][0]["metadata"]["confidence_arrays"] = {
        "file": "confidence_full.npz",
        "schema_version": 1,
        "arrays": ["pae", "plddt"],
        "index_arrays": ["token_chain"],
        "unavailable": {"pde": "not written by this backend"},
    }
    manifest_path.write_text(json.dumps(manifest))

    assert errors(manifest, load_schema("run")) == []
    (row,) = results_table(load_results(root))
    assert row["confidence_arrays"] == ["pae", "plddt"]


def test_an_input_the_model_never_read_reaches_the_row(tmp_path: Path) -> None:
    """Boltz-2 reads no RNA alignment: dropped, recorded, and shown per row."""
    from foldjax.summary import load_schema
    from tests._schema_lite import errors

    jobs = tmp_path / "jobs"
    jobs.mkdir()
    (jobs / "job.a3m").write_text(">query\nACD\n")
    (jobs / "rna.a3m").write_text(">query\nACGU\n")
    job = jobs / "e9_8reh.json"
    job.write_text(
        json.dumps(
            {
                "name": "e9_8reh",
                "entities": [
                    {
                        "type": "protein",
                        "id": "A",
                        "sequence": "ACD",
                        "unpaired_msa": "job.a3m",
                    },
                    {
                        "type": "rna",
                        "id": "B",
                        "sequence": "ACGU",
                        "unpaired_msa": "rna.a3m",
                    },
                ],
            }
        )
    )
    out = tmp_path / "batch" / "boltz2" / "e9_8reh"
    with pytest.warns(UserWarning, match="RNA"):
        _confidence, manifest, _fixture = _run(
            tmp_path / "scratch", "boltz2_e9_8reh", out=out, job=job
        )

    assert errors(manifest, load_schema("run")) == []
    (row,) = results_table(load_results(tmp_path / "batch"))
    (ignored,) = row["ignored_msas"]
    assert ignored["field"] == "unpaired_msa" and ignored["type"] == "rna"
    assert "ignored_msas" in to_csv([row]).splitlines()[0]
