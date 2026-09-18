"""Contracts for source-preserving individual-run paper evidence."""

from __future__ import annotations

import hashlib
import json

import pytest

from bench import paper_run_table as table


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def result(path, *, model="model", case="case", samples=None, **extra):
    body = {
        "model": model,
        "case": case,
        "wall_s": 12.5,
        "peak_mib": 456.0,
        "samples": samples
        if samples is not None
        else [{"seed": 1, "scores": {"ptm": 0.7}}],
    }
    body.update(extra)
    path.write_text(json.dumps(body))
    return body


def run_entry(run_id, result_path=None, **extra):
    body = {
        "run_id": run_id,
        "model": "model",
        "case": "case",
        "arm": "foldjax",
        "protocol_id": "protocol-a",
        "timing_state": "warm",
        "status": "success",
        "result": None if result_path is None else result_path.name,
        "result_sha256": None if result_path is None else digest(result_path),
        "note": "literal run record",
    }
    body.update(extra)
    return body


def manifest(path, runs):
    body = {"schema_version": 1, "runs": runs}
    path.write_text(json.dumps(body))
    return body


def test_repeated_model_case_arm_runs_stay_individual_and_keep_all_scores(tmp_path):
    first, second, source = (
        tmp_path / "one.json",
        tmp_path / "two.json",
        tmp_path / "runs.json",
    )
    scores = [
        {"seed": 7, "sample_id": "x", "scores": {"ptm": 0.2, "plddt": 90.0}},
        {"seed": 8, "sample_id": "y", "scores": {"ptm": 0.9, "plddt": 20.0}},
    ]
    first_body = result(first, samples=scores)
    result(second, samples=[{"scores": {"confidence_score": 1.2}}])
    document = manifest(
        source,
        [run_entry("one", first), run_entry("two", second, protocol_id="protocol-b")],
    )

    built = table.build_table(source, tmp_path)

    assert built["input_manifest"] == document
    assert [row["run_id"] for row in built["runs"]] == ["one", "two"]
    assert [row["protocol_id"] for row in built["runs"]] == ["protocol-a", "protocol-b"]
    assert built["runs"][0]["samples"] == first_body["samples"]
    assert built["runs"][0]["source_result_sha256"] == digest(first)
    assert built["runs"][0]["sample_count"] == 2
    assert built["runs"][0]["structure_metrics_status"] == "not_joined"
    assert "speed-accuracy comparison" in table.render_markdown(built)


@pytest.mark.parametrize("status", ["failed", "oom", "timeout", "unmeasured"])
def test_failure_and_unmeasured_rows_are_preserved_without_timing_numbers(
    tmp_path, status
):
    source = tmp_path / "runs.json"
    entry = run_entry("not-measured", status=status, result=None, result_sha256=None)
    manifest(source, [entry])

    built = table.build_table(source, tmp_path)

    row = built["runs"][0]
    assert row["status"] == status
    assert row["wall_s"] is None
    assert row["peak_mib"] is None
    assert row["samples"] == []


def test_failed_source_hash_and_samples_are_preserved_but_metrics_are_null(tmp_path):
    source_result, source = tmp_path / "failed.json", tmp_path / "runs.json"
    result(source_result, samples=[], failed=True, returncode=1)
    manifest(source, [run_entry("failed", source_result, status="failed")])

    built = table.build_table(source, tmp_path)

    row = built["runs"][0]
    assert row["source_result_sha256"] == digest(source_result)
    assert row["samples"] == []
    assert row["wall_s"] is None
    assert row["peak_mib"] is None
    assert "| null | null |" in table.render_markdown(built)


@pytest.mark.parametrize(
    "change",
    [
        lambda run: run.update(run_id=""),
        lambda run: run.update(model="other"),
        lambda run: run.update(result_sha256="0" * 64),
        lambda run: run.update(result="../escape.json"),
        lambda run: run.update(status="unknown"),
    ],
)
def test_invalid_identity_hash_and_path_are_rejected(tmp_path, change):
    source_result, source = tmp_path / "result.json", tmp_path / "runs.json"
    result(source_result)
    entry = run_entry("one", source_result)
    change(entry)
    manifest(source, [entry])

    with pytest.raises(ValueError):
        table.build_table(source, tmp_path)


def test_duplicate_id_nonfinite_and_negative_measurements_are_rejected(tmp_path):
    source_result, source = tmp_path / "result.json", tmp_path / "runs.json"
    result(source_result, wall_s=-1)
    manifest(source, [run_entry("one", source_result)])
    with pytest.raises(ValueError, match="nonnegative"):
        table.build_table(source, tmp_path)

    source_result.write_text(
        '{"model":"model","case":"case","wall_s":NaN,"peak_mib":1,"samples":[]}'
    )
    manifest(source, [run_entry("one", source_result)])
    with pytest.raises(ValueError, match="invalid JSON"):
        table.build_table(source, tmp_path)

    result(source_result)
    one = run_entry("same", source_result)
    manifest(source, [one, dict(one)])
    with pytest.raises(ValueError, match="duplicate"):
        table.build_table(source, tmp_path)


def test_success_and_failure_result_state_conflicts_are_rejected(tmp_path):
    source_result, source = tmp_path / "result.json", tmp_path / "runs.json"
    result(source_result, failed=True)
    manifest(source, [run_entry("one", source_result)])
    with pytest.raises(ValueError, match="success status conflicts"):
        table.build_table(source, tmp_path)

    result(source_result, returncode=0)
    manifest(source, [run_entry("one", source_result, status="failed")])
    with pytest.raises(ValueError, match="failure status conflicts"):
        table.build_table(source, tmp_path)


def test_partial_failed_samples_are_preserved_with_null_measurements(tmp_path):
    source_result, source = tmp_path / "partial.json", tmp_path / "runs.json"
    partial = [{"seed": 3, "scores": {"ptm": 0.4}}]
    result(source_result, samples=partial, failed=True, returncode=1)
    manifest(source, [run_entry("partial", source_result, status="timeout")])

    built = table.build_table(source, tmp_path)

    row = built["runs"][0]
    assert row["samples"] == partial
    assert row["sample_count"] == 1
    assert row["wall_s"] is None
    assert row["peak_mib"] is None


def test_failure_result_may_omit_samples(tmp_path):
    source_result, source = tmp_path / "failed.json", tmp_path / "runs.json"
    body = result(source_result, failed=True, returncode=1)
    body.pop("samples")
    source_result.write_text(json.dumps(body))
    manifest(source, [run_entry("failed", source_result, status="failed")])

    assert table.build_table(source, tmp_path)["runs"][0]["samples"] == []


def test_nonfinite_nested_source_score_and_manifest_value_are_rejected(tmp_path):
    source_result, source = tmp_path / "result.json", tmp_path / "runs.json"
    source_result.write_text(
        '{"model":"model","case":"case","wall_s":1,"peak_mib":1,'
        '"samples":[{"scores":{"ptm":1e999}}]}'
    )
    manifest(source, [run_entry("one", source_result)])
    with pytest.raises(ValueError, match="nonfinite"):
        table.build_table(source, tmp_path)

    result(source_result)
    entry = run_entry("one", source_result)
    source.write_text(
        '{"schema_version":1,"runs":['
        + json.dumps(entry)
        + '],"extra":{"value":1e999}}'
    )
    with pytest.raises(ValueError, match="nonfinite"):
        table.build_table(source, tmp_path)


def test_cli_rejects_manifest_result_and_output_aliases(tmp_path, monkeypatch):
    source_result, source = tmp_path / "result.json", tmp_path / "runs.json"
    result(source_result)
    manifest(source, [run_entry("one", source_result)])
    markdown = tmp_path / "table.md"
    monkeypatch.setattr(
        "sys.argv",
        [
            "paper_run_table",
            "--manifest",
            str(source),
            "--artifact-root",
            str(tmp_path),
            "--json-out",
            str(source),
            "--markdown-out",
            str(markdown),
        ],
    )
    with pytest.raises(ValueError, match="manifest"):
        table.main()

    existing_json = tmp_path / "existing.json"
    existing_json.write_text("old")
    monkeypatch.setattr(
        "sys.argv",
        [
            "paper_run_table",
            "--manifest",
            str(source),
            "--artifact-root",
            str(tmp_path),
            "--json-out",
            str(existing_json),
            "--markdown-out",
            str(markdown),
        ],
    )
    table.main()
    assert json.loads(existing_json.read_text())["runs"][0]["run_id"] == "one"
    assert markdown.is_file()

    alias = tmp_path / "result-alias.json"
    alias.hardlink_to(source_result)
    monkeypatch.setattr(
        "sys.argv",
        [
            "paper_run_table",
            "--manifest",
            str(source),
            "--artifact-root",
            str(tmp_path),
            "--json-out",
            str(alias),
            "--markdown-out",
            str(markdown),
        ],
    )
    with pytest.raises(ValueError, match="referenced result"):
        table.main()

    monkeypatch.setattr(
        "sys.argv",
        [
            "paper_run_table",
            "--manifest",
            str(source),
            "--artifact-root",
            str(tmp_path),
            "--json-out",
            str(markdown),
            "--markdown-out",
            str(markdown),
        ],
    )
    with pytest.raises(ValueError, match="outputs"):
        table.main()
