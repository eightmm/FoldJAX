from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from bench import evidence_index


def _workspace(tmp_path: Path, *, cards: str = "1", raw_count: int = 2) -> Path:
    root = tmp_path / "foldjax-bench"
    paper = root / "paper-2026-09"
    (paper / "structures").mkdir(parents=True)
    (root / "paper-single-gpu-audit-20260918").mkdir()
    (root / "scale-timing-20260910" / "results").mkdir(parents=True)
    (root / "raw").mkdir()
    for index in range(2):
        (root / "raw" / f"{index}.cif").write_text(f"raw {index}")
    copied = paper / "structures" / "copy.cif"
    copied.write_text("copied")
    (root / "result.json").write_text(json.dumps({"samples": [{}, {}], "options": {}}))
    with (paper / "benchmarks.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["row_id", "cards", "status", "n_structures", "result_json"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "row_id": "row-a",
                "cards": cards,
                "status": "ok",
                "n_structures": "2",
                "result_json": "result.json",
            }
        )
    with (paper / "structures" / "MANIFEST.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["row_id", "source_path", "published_path", "sha256"]
        )
        writer.writeheader()
        writer.writerow(
            {
                "row_id": "row-a",
                "source_path": "raw/0.cif",
                "published_path": "structures/copy.cif",
                "sha256": "",
            }
        )
    paths = [f"foldjax-bench/raw/{index}.cif" for index in range(2)]
    (
        root / "paper-single-gpu-audit-20260918" / "raw-structure-availability.json"
    ).write_text(
        json.dumps([{"row_id": "row-a", "raw_cif_count": raw_count, "paths": paths}])
    )
    return root


def test_build_index_preserves_raw_copy_identity_hashes_and_missing_provenance(
    tmp_path,
):
    root = _workspace(tmp_path)
    report = evidence_index.build_index(root.parent)
    row = report["paper_rows"][0]
    assert row["row_id"] == "row-a"
    assert len(row["raw_structures"]) == 2
    assert len(row["structures"]) == 1
    assert row["result_json_sha256"]
    assert row["missing_identity_reasons"]
    assert any(
        issue["kind"] == "packaging_count_mismatch"
        for issue in report["provenance_issues"]
    )


def test_build_index_rejects_cp_contradiction(tmp_path):
    root = _workspace(tmp_path)
    (root / "result.json").write_text(
        json.dumps({"samples": [{}], "options": {"cp_devices": 2}})
    )
    with pytest.raises(ValueError, match="contradictory CP"):
        evidence_index.build_index(root)


@pytest.mark.parametrize("kind", ["duplicate", "malformed", "count"])
def test_build_index_rejects_invalid_evidence(tmp_path, kind):
    root = _workspace(tmp_path)
    if kind == "duplicate":
        with (root / "paper-2026-09" / "benchmarks.csv").open("a") as stream:
            stream.write("row-a,1,ok,2,result.json\n")
        message = "duplicate"
    elif kind == "malformed":
        (root / "result.json").write_text("{")
        message = "Expecting"
    else:
        audit = (
            root / "paper-single-gpu-audit-20260918" / "raw-structure-availability.json"
        )
        body = json.loads(audit.read_text())
        body[0]["raw_cif_count"] = 1
        audit.write_text(json.dumps(body))
        message = "count mismatch"
    with pytest.raises((ValueError, json.JSONDecodeError), match=message):
        evidence_index.build_index(root)


def test_helper_cp_and_failure_contracts():
    assert evidence_index._cp_from_paper({"cards": "1"}) == "single_gpu"
    assert evidence_index._cp_from_paper({"cards": "2"}) == "cp_excluded"
    assert evidence_index._cp_from_paper({}) == "unknown"
    for value in (0, -1):
        with pytest.raises(ValueError):
            evidence_index._cp_from_paper({"cards": str(value)})
    for value in (0, -1, True, 1.0, "1"):
        with pytest.raises(ValueError):
            evidence_index._cp_from_result({"options": {"cp_devices": value}})
    assert evidence_index._status({"status": "ok"}, {"samples": []}) == "failure"
    assert (
        evidence_index._status({"status": "ok"}, {"returncode": 1, "samples": [{}]})
        == "failure"
    )


def test_build_index_rejects_numeric_cp_mismatch_and_preserves_distinct_rows(tmp_path):
    root = _workspace(tmp_path, cards="2")
    (root / "result.json").write_text(
        json.dumps({"samples": [{}, {}], "options": {"cp_devices": 4}})
    )
    with pytest.raises(ValueError, match="contradictory CP"):
        evidence_index.build_index(root)


def test_empty_result_is_failure_and_historical_missing_cp_is_unknown():
    assert evidence_index._status({"status": "ok"}, {}) == "failure"
    assert evidence_index._cp_from_result({"options": {}}) == "unknown"
