"""Focused contracts for explicit descriptive performance aggregation."""

import hashlib
import json

import pytest

from bench import paper_performance_table as performance


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def result(
    path,
    *,
    wall,
    peak,
    model="boltz2",
    case="L1000_3og2",
    schedule=None,
    device=None,
    seed=101,
    options=None,
    runtime=None,
    source=None,
    failed=False,
):
    body = {
        "model": model,
        "case": case,
        "wall_s": wall,
        "peak_mib": peak,
        "samples": [] if failed else [{"scores": {"ptm": 0.7}}],
        "returncode": 1 if failed else 0,
        "failed": failed,
        "schedule": schedule or {"num_samples": 1, "num_steps": 200, "num_recycles": 3},
        "device": device
        or {
            "model": "GPU",
            "platform": "cuda",
            "compute_capability": "12.0",
            "driver_version": "x",
            "total_memory_mib": 100,
        },
        "seed": seed,
        "options": options,
        "runtime": runtime or {"python": {"version": "3.13"}},
        "impl": "implementation-a",
        "artifacts": {
            "checkpoints": {"model": {"sha256": "d" * 64}},
            "inputs": {"job": {"sha256": "e" * 64}},
        },
        "execution": {"timing_state": "warm-after-successful-prefill"},
        "source": source
        or {
            "foldjax": {"sha256": "a" * 64},
            "harness": {"runner": {"sha256": "b" * 64}},
        },
    }
    path.write_text(json.dumps(body))


def run(
    run_id, path, *, arm, timing, status="success", protocol="n1-r3", note="record"
):
    body = {
        "run_id": run_id,
        "model": "boltz2",
        "case": "L1000_3og2",
        "arm": arm,
        "protocol_id": protocol,
        "timing_state": timing,
        "status": status,
        "result": path.name,
        "result_sha256": digest(path),
        "note": note,
    }
    value = json.loads(path.read_text())
    if isinstance(value.get("execution"), dict):
        value["execution"]["timing_state"] = timing
        path.write_text(json.dumps(value))
    body["result_sha256"] = digest(path)
    return body


def cell(cell_id, run_ids, arm):
    return {
        "cell_id": cell_id,
        "run_ids": run_ids,
        "arm": arm,
        "precision_policy_id": "fp32-policy",
        "hardware_id": "gpu-a",
        "limitations": "operator parity is not established by this table",
    }


def write(tmp_path, runs, spec):
    manifest = tmp_path / "runs.json"
    manifest.write_text(json.dumps({"schema_version": 1, "runs": runs}))
    aggregate = tmp_path / "aggregate.json"
    aggregate.write_text(json.dumps({"schema_version": 1, **spec}))
    return manifest, aggregate


def test_warm_medians_need_three_successes_and_ratio_is_descriptive(tmp_path):
    paths = []
    for name, wall in [
        ("n-cold", 10),
        ("n1", 4),
        ("n2", 6),
        ("n3", 5),
        ("f-cold", 8),
        ("f1", 2),
        ("f2", 3),
        ("f3", 2.5),
    ]:
        path = tmp_path / f"{name}.json"
        result(path, wall=wall, peak=100)
        paths.append((name, path))
    runs = [
        run(
            name,
            path,
            arm="native" if name.startswith("n") else "foldjax",
            timing="cold-or-unspecified"
            if "cold" in name
            else "warm-after-successful-prefill",
        )
        for name, path in paths
    ]
    manifest, aggregate = write(
        tmp_path,
        runs,
        {
            "cells": [
                cell(
                    "native",
                    [name for name, _ in paths if name.startswith("n")],
                    "native",
                ),
                cell(
                    "foldjax",
                    [name for name, _ in paths if name.startswith("f")],
                    "foldjax",
                ),
            ],
            "comparisons": [
                {
                    "comparison_id": "matched",
                    "baseline_cell": "native",
                    "candidate_cell": "foldjax",
                    "rationale": "explicit requested comparison",
                    "limitations": "no operator parity claim",
                }
            ],
        },
    )
    table = performance.build_table(manifest, aggregate, tmp_path)
    assert table["cells"][0]["warm"]["wall_s_median"] == 5
    assert (
        table["comparisons"][0]["speed_ratio_warm_median_baseline_over_candidate"] == 2
    )
    assert "no statistical inference" in table["comparisons"][0]["interpretation"]


def test_missing_warm_repeat_and_oom_stay_separate_and_block_ratio(tmp_path):
    good = tmp_path / "good.json"
    oom = tmp_path / "oom.json"
    result(good, wall=4, peak=10)
    result(oom, wall=9, peak=99, failed=True)
    failed = json.loads(oom.read_text())
    for name in ("seed", "options", "runtime", "source", "schedule", "device"):
        failed.pop(name)
    oom.write_text(json.dumps(failed))
    runs = [
        run("n1", good, arm="native", timing="warm-after-successful-prefill"),
        run("n-oom", oom, arm="native", timing="cold-or-unspecified", status="oom"),
    ]
    manifest, aggregate = write(
        tmp_path,
        runs,
        {"cells": [cell("native", ["n1", "n-oom"], "native")], "comparisons": []},
    )
    row = performance.build_table(manifest, aggregate, tmp_path)["cells"][0]
    assert row["warm"]["missing_repeats"] == 2
    assert row["failure_rows"][0]["status"] == "oom"
    assert row["failure_rows"][0]["run_id"] == "n-oom"


def test_comparison_rejects_schedule_or_precision_mismatch(tmp_path):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    result(a, wall=1, peak=1)
    result(
        b,
        wall=1,
        peak=1,
        schedule={"num_samples": 1, "num_steps": 200, "num_recycles": 5},
    )
    manifest, aggregate = write(
        tmp_path,
        [
            run("a", a, arm="native", timing="warm-after-successful-prefill"),
            run("b", b, arm="foldjax", timing="warm-after-successful-prefill"),
        ],
        {
            "cells": [cell("a", ["a"], "native"), cell("b", ["b"], "foldjax")],
            "comparisons": [
                {
                    "comparison_id": "bad",
                    "baseline_cell": "a",
                    "candidate_cell": "b",
                    "rationale": "x",
                    "limitations": "x",
                }
            ],
        },
    )
    with pytest.raises(ValueError, match="schedule"):
        performance.build_table(manifest, aggregate, tmp_path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("options", {"dtype": "bf16"}),
        ("seed", 102),
        ("runtime", {"python": {"version": "3.14"}}),
        (
            "source",
            {
                "foldjax": {"sha256": "c" * 64},
                "harness": {"runner": {"sha256": "new"}},
            },
        ),
    ],
)
def test_cell_rejects_mixed_success_execution_identity(tmp_path, field, value):
    first, second = tmp_path / "first.json", tmp_path / "second.json"
    result(first, wall=1, peak=1)
    kwargs = {field: value}
    result(second, wall=2, peak=2, **kwargs)
    manifest, aggregate = write(
        tmp_path,
        [
            run("one", first, arm="native", timing="warm-after-successful-prefill"),
            run("two", second, arm="native", timing="warm-after-successful-prefill"),
        ],
        {"cells": [cell("native", ["one", "two"], "native")], "comparisons": []},
    )
    with pytest.raises(ValueError, match="execution identity"):
        performance.build_table(manifest, aggregate, tmp_path)


def test_missing_device_total_memory_is_a_clear_validation_error(tmp_path):
    source = tmp_path / "result.json"
    result(
        source,
        wall=1,
        peak=1,
        device={
            "model": "GPU",
            "platform": "cuda",
            "compute_capability": "12.0",
            "driver_version": "x",
        },
    )
    manifest, aggregate = write(
        tmp_path,
        [run("one", source, arm="native", timing="warm-after-successful-prefill")],
        {"cells": [cell("native", ["one"], "native")], "comparisons": []},
    )
    with pytest.raises(ValueError, match="total_memory_mib"):
        performance.build_table(manifest, aggregate, tmp_path)


@pytest.mark.parametrize("field", ["impl", "seed", "artifacts"])
def test_cell_rejects_mixed_implementation_seed_or_artifact_identity(tmp_path, field):
    first, second = tmp_path / "first.json", tmp_path / "second.json"
    result(first, wall=1, peak=1)
    result(second, wall=2, peak=2)
    changed = json.loads(second.read_text())
    if field == "impl":
        changed[field] = "implementation-b"
    elif field == "seed":
        changed[field] = 102
    else:
        changed[field]["inputs"]["job"]["sha256"] = "f" * 64
    second.write_text(json.dumps(changed))
    manifest, aggregate = write(
        tmp_path,
        [
            run("one", first, arm="native", timing="warm-after-successful-prefill"),
            run("two", second, arm="native", timing="warm-after-successful-prefill"),
        ],
        {"cells": [cell("native", ["one", "two"], "native")], "comparisons": []},
    )
    with pytest.raises(ValueError, match="execution identity"):
        performance.build_table(manifest, aggregate, tmp_path)


def test_success_timing_state_mismatch_and_cross_cell_seed_are_rejected(tmp_path):
    first, second = tmp_path / "first.json", tmp_path / "second.json"
    result(first, wall=1, peak=1)
    result(second, wall=1, peak=1, seed=102)
    one = run("one", first, arm="native", timing="warm-after-successful-prefill")
    two = run("two", second, arm="foldjax", timing="warm-after-successful-prefill")
    manifest, aggregate = write(
        tmp_path,
        [one, two],
        {
            "cells": [cell("one", ["one"], "native"), cell("two", ["two"], "foldjax")],
            "comparisons": [
                {
                    "comparison_id": "bad",
                    "baseline_cell": "one",
                    "candidate_cell": "two",
                    "rationale": "x",
                    "limitations": "x",
                }
            ],
        },
    )
    with pytest.raises(ValueError, match="seed"):
        performance.build_table(manifest, aggregate, tmp_path)
    stale = json.loads(first.read_text())
    stale["execution"]["timing_state"] = "cold-or-unspecified"
    first.write_text(json.dumps(stale))
    one["result_sha256"] = digest(first)
    manifest, aggregate = write(
        tmp_path, [one], {"cells": [cell("one", ["one"], "native")], "comparisons": []}
    )
    with pytest.raises(ValueError, match="timing_state"):
        performance.build_table(manifest, aggregate, tmp_path)
