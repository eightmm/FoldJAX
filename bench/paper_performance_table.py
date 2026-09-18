"""Build explicit, schedule-matched descriptive performance evidence tables.

The input run manifest is validated by :mod:`bench.paper_run_table`.  This
module never discovers a native/FoldJAX pair itself: the aggregate spec names
cells and comparisons, so incompatible schedules, precision policies, devices,
and failures cannot silently become a speedup claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any

from bench import paper_run_table as individual

SCHEMA_VERSION = 1
_COLD = "cold-or-unspecified"
_WARM = "warm-after-successful-prefill"
_STATES = frozenset({_COLD, _WARM})


def _reject_constant(value: str) -> None:
    raise ValueError(f"nonfinite JSON constant {value!r} is not allowed")


def _load(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    try:
        value = json.loads(raw, parse_constant=_reject_constant)
    except (ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"{path}: invalid {label} JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path}: {label} must be an object")
    individual._validate_json_finite(value, f"{path}: {label}")
    return value, raw


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")
    return value


def _schedule(result: dict[str, Any], run_id: str) -> dict[str, int]:
    value = result.get("schedule")
    if not isinstance(value, dict):
        raise ValueError(f"run {run_id!r} has no schedule object")
    out: dict[str, int] = {}
    for name in ("num_samples", "num_steps", "num_recycles"):
        item = value.get(name)
        if isinstance(item, bool) or not isinstance(item, int) or item < 1:
            raise ValueError(f"run {run_id!r} has invalid schedule.{name}")
        out[name] = item
    return out


def _device(result: dict[str, Any], run_id: str) -> dict[str, Any]:
    value = result.get("device")
    if not isinstance(value, dict):
        raise ValueError(f"run {run_id!r} has no device identity")
    required = ("model", "platform", "compute_capability", "driver_version")
    if any(not isinstance(value.get(name), str) for name in required):
        raise ValueError(f"run {run_id!r} has incomplete device identity")
    total = value.get("total_memory_mib")
    if (
        isinstance(total, bool)
        or not isinstance(total, (int, float))
        or not math.isfinite(total)
    ):
        raise ValueError(f"run {run_id!r} has invalid device total_memory_mib")
    return {name: value[name] for name in required} | {"total_memory_mib": total}


def _execution_identity(result: dict[str, Any], run_id: str) -> dict[str, Any]:
    implementation = result.get("impl")
    if not isinstance(implementation, str) or not implementation:
        raise ValueError(f"run {run_id!r} has invalid implementation")
    seed = result.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f"run {run_id!r} has invalid seed")
    options = result.get("options")
    if options is not None and not isinstance(options, dict):
        raise ValueError(f"run {run_id!r} has invalid options")
    runtime = result.get("runtime")
    if not isinstance(runtime, dict):
        raise ValueError(f"run {run_id!r} has no runtime identity")
    artifacts = result.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError(f"run {run_id!r} has no artifact identity")
    checkpoint_inputs = {
        name: artifacts.get(name) for name in ("checkpoints", "inputs")
    }
    if not all(
        isinstance(value, dict) and value for value in checkpoint_inputs.values()
    ):
        raise ValueError(f"run {run_id!r} has incomplete checkpoint/input identity")
    source = result.get("source")
    if not isinstance(source, dict):
        raise ValueError(f"run {run_id!r} has no source identity")
    upstream = result.get("upstream_git")
    foldjax = source.get("foldjax")
    if isinstance(upstream, dict):
        model_source = {"kind": "upstream_git", "identity": upstream}
    elif isinstance(foldjax, dict) and isinstance(foldjax.get("sha256"), str):
        model_source = {"kind": "foldjax_source", "identity": foldjax}
    else:
        raise ValueError(f"run {run_id!r} has no model executable source identity")
    harness = source.get("harness")
    if not isinstance(harness, dict):
        raise ValueError(f"run {run_id!r} has no harness source record")
    return {
        "implementation": implementation,
        "seed": seed,
        "options": options,
        "runtime": runtime,
        "model_execution_source": model_source,
        "checkpoint_input_artifacts": checkpoint_inputs,
        "harness_source": harness,
    }


def _success_summary(rows: list[dict[str, Any]], timing: str) -> dict[str, Any]:
    expected = 1 if timing == _COLD else 3
    successful = [row for row in rows if row["status"] == "success"]
    values = [row["wall_s"] for row in successful]
    peaks = [row["peak_mib"] for row in successful]
    return {
        "timing_state": timing,
        "expected_successful_repeats": expected,
        "successful_run_ids": [row["run"]["run_id"] for row in successful],
        "successful_n": len(successful),
        "missing_repeats": max(0, expected - len(successful)),
        "complete": len(successful) == expected,
        "wall_s_values": values,
        "wall_s_median": None if not values else median(values),
        "peak_live_allocated_mib_values": peaks,
        "peak_live_allocated_mib_median": None if not peaks else median(peaks),
        "peak_reserved_mib": None,
        "memory_definition": "live allocated high-water mark",
        "reserved_memory_status": "not reported",
    }


def _cell(spec: dict[str, Any], by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    cell_id = _text(spec.get("cell_id"), "cell_id")
    arm = _text(spec.get("arm"), "arm")
    precision = _text(spec.get("precision_policy_id"), "precision_policy_id")
    hardware = _text(spec.get("hardware_id"), "hardware_id")
    limitation = _text(spec.get("limitations"), "limitations")
    requested = spec.get("run_ids")
    if not isinstance(requested, list) or not requested:
        raise ValueError(f"cell {cell_id!r} run_ids must be a nonempty list")
    if len(requested) != len(set(requested)) or not all(
        isinstance(x, str) for x in requested
    ):
        raise ValueError(f"cell {cell_id!r} run_ids must be unique strings")
    try:
        rows = [by_id[run_id] for run_id in requested]
    except KeyError as error:
        raise ValueError(f"cell {cell_id!r} names an unknown run") from error
    if any(row["run"]["arm"] != arm for row in rows):
        raise ValueError(f"cell {cell_id!r} arm does not match all manifest runs")
    successful = [row for row in rows if row["status"] == "success"]
    for row in successful:
        execution = row["source_result"].get("execution")
        if (
            not isinstance(execution, dict)
            or execution.get("timing_state") != row["run"]["timing_state"]
        ):
            raise ValueError(
                f"run {row['run']['run_id']!r} execution timing_state "
                "differs from manifest"
            )
    successful_results = [row["source_result"] for row in successful]
    if any(result is None for result in successful_results):
        raise ValueError(f"cell {cell_id!r} success rows require result artifacts")
    first = rows[0]["run"]
    if any(
        row["run"][name] != first[name]
        for row in rows
        for name in ("model", "case", "protocol_id")
    ):
        raise ValueError(f"cell {cell_id!r} mixes model, case, or protocol")
    schedule = device = execution = None
    harness_sources: dict[str, dict[str, Any]] = {}
    if successful:
        schedules = [
            _schedule(result, row["run"]["run_id"])
            for row, result in zip(successful, successful_results, strict=True)
        ]
        devices = [
            _device(result, row["run"]["run_id"])
            for row, result in zip(successful, successful_results, strict=True)
        ]
        executions = [
            _execution_identity(result, row["run"]["run_id"])
            for row, result in zip(successful, successful_results, strict=True)
        ]
        schedule, device, execution = schedules[0], devices[0], executions[0]
        if any(value != schedule for value in schedules[1:]):
            raise ValueError(f"cell {cell_id!r} mixes schedules")
        if any(value != device for value in devices[1:]):
            raise ValueError(f"cell {cell_id!r} mixes hardware")
        identity = {
            name: execution[name]
            for name in (
                "implementation",
                "seed",
                "options",
                "runtime",
                "model_execution_source",
                "checkpoint_input_artifacts",
            )
        }
        if any(
            {name: value[name] for name in identity} != identity
            for value in executions[1:]
        ):
            raise ValueError(f"cell {cell_id!r} mixes execution identity")
        harness_sources = {
            row["run"]["run_id"]: value["harness_source"]
            for row, value in zip(successful, executions, strict=True)
        }

    grouped = {
        state: [row for row in rows if row["run"]["timing_state"] == state]
        for state in _STATES
    }
    unknown = [
        row["run"]["timing_state"]
        for row in rows
        if row["run"]["timing_state"] not in _STATES
    ]
    if unknown:
        raise ValueError(f"cell {cell_id!r} has unsupported timing state")
    failures = [
        {
            "run_id": row["run"]["run_id"],
            "timing_state": row["run"]["timing_state"],
            "status": row["status"],
            "note": row["manifest_run"]["note"],
            "result": row["manifest_run"].get("result"),
            "source_result_sha256": row["source_result_sha256"],
        }
        for row in rows
        if row["status"] != "success"
    ]
    return {
        "cell_id": cell_id,
        "model": first["model"],
        "case": first["case"],
        "protocol_id": first["protocol_id"],
        "arm": arm,
        "precision_policy_id": precision,
        "hardware_id": hardware,
        "device": device,
        "schedule": schedule,
        "success_execution_identity": (
            None
            if execution is None
            else {
                name: execution[name]
                for name in (
                    "implementation",
                    "seed",
                    "options",
                    "runtime",
                    "model_execution_source",
                    "checkpoint_input_artifacts",
                )
            }
        ),
        "harness_sources_by_successful_run": harness_sources,
        "successful_identity_verified": execution is not None,
        "operator_parity_verified": False,
        "limitations": limitation,
        "cold": _success_summary(grouped[_COLD], _COLD),
        "warm": _success_summary(grouped[_WARM], _WARM),
        "failure_rows": failures,
        "status_counts": dict(sorted(Counter(row["status"] for row in rows).items())),
    }


def _comparison(
    spec: dict[str, Any], cells: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    comparison_id = _text(spec.get("comparison_id"), "comparison_id")
    baseline = _text(spec.get("baseline_cell"), "baseline_cell")
    candidate = _text(spec.get("candidate_cell"), "candidate_cell")
    rationale = _text(spec.get("rationale"), "rationale")
    limitations = _text(spec.get("limitations"), "limitations")
    if baseline == candidate or baseline not in cells or candidate not in cells:
        raise ValueError(f"comparison {comparison_id!r} names invalid cells")
    left, right = cells[baseline], cells[candidate]
    if (
        not left["successful_identity_verified"]
        or not right["successful_identity_verified"]
    ):
        raise ValueError(
            f"comparison {comparison_id!r} requires successful execution identity"
        )
    if (
        left["success_execution_identity"]["seed"]
        != right["success_execution_identity"]["seed"]
    ):
        raise ValueError(f"comparison {comparison_id!r} has unmatched seed")
    for name in (
        "model",
        "case",
        "protocol_id",
        "schedule",
        "device",
        "hardware_id",
        "precision_policy_id",
    ):
        if left[name] != right[name]:
            raise ValueError(f"comparison {comparison_id!r} has unmatched {name}")
    ready = (
        left["warm"]["complete"]
        and right["warm"]["complete"]
        and not left["failure_rows"]
        and not right["failure_rows"]
    )
    ratio = (
        None
        if not ready
        else left["warm"]["wall_s_median"] / right["warm"]["wall_s_median"]
    )
    return {
        "comparison_id": comparison_id,
        "baseline_cell": baseline,
        "candidate_cell": candidate,
        "rationale": rationale,
        "limitations": limitations,
        "speed_ratio_warm_median_baseline_over_candidate": ratio,
        "comparison_complete": ready,
        "interpretation": (
            "descriptive warm-median ratio only; n=3 per arm, no statistical inference"
            if ready
            else (
                "not computed: each arm requires three successful warm repeats "
                "and no failure rows"
            )
        ),
        "operator_parity_verified": False,
    }


def build_table(
    runs_manifest: Path, aggregate_spec: Path, artifact_root: Path
) -> dict[str, Any]:
    manifest, manifest_raw = individual._load_json(Path(runs_manifest), "runs manifest")
    rows = individual._validate_manifest(manifest, Path(artifact_root).resolve())
    spec, spec_raw = _load(Path(aggregate_spec), "aggregate spec")
    if spec.get("schema_version") != SCHEMA_VERSION or not isinstance(
        spec.get("cells"), list
    ):
        raise ValueError("aggregate spec schema_version must be 1 with cells")
    by_id = {row["run"]["run_id"]: row for row in rows}
    if not all(isinstance(value, dict) for value in spec["cells"]):
        raise ValueError("aggregate spec cells must be objects")
    cells_list = [_cell(value, by_id) for value in spec["cells"]]
    if len({cell["cell_id"] for cell in cells_list}) != len(cells_list):
        raise ValueError("aggregate spec cells must be unique objects")
    used = [run_id for value in spec["cells"] for run_id in value.get("run_ids", [])]
    if len(used) != len(set(used)):
        raise ValueError("a run_id may appear in only one aggregate cell")
    cells = {cell["cell_id"]: cell for cell in cells_list}
    comparisons_spec = spec.get("comparisons", [])
    if not isinstance(comparisons_spec, list) or not all(
        isinstance(x, dict) for x in comparisons_spec
    ):
        raise ValueError("aggregate spec comparisons must be objects")
    comparisons = [_comparison(value, cells) for value in comparisons_spec]
    if len({item["comparison_id"] for item in comparisons}) != len(comparisons):
        raise ValueError("comparison_id values must be unique")
    return {
        "schema_version": SCHEMA_VERSION,
        "scope": "explicit_schedule_matched_descriptive_performance",
        "runs_manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "aggregate_spec_sha256": hashlib.sha256(spec_raw).hexdigest(),
        "memory_statement": (
            "peak_mib is live allocated high-water mark; reserved memory is "
            "not reported and is never inferred"
        ),
        "statistical_statement": (
            "warm median ratios are descriptive only; n=3 is not used for "
            "significance claims"
        ),
        "cells": cells_list,
        "comparisons": comparisons,
    }


def render_markdown(table: dict[str, Any]) -> str:
    lines = [
        "# Schedule-matched performance evidence",
        "",
        (
            "| cell | model | case | arm | protocol | precision | cold n/1 | "
            "warm n/3 | warm median s | live allocated MiB | failures |"
        ),
        "|---|---|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for cell in table["cells"]:
        lines.append(
            (
                "| {cell_id} | {model} | {case} | {arm} | {protocol_id} | "
                "{precision_policy_id} | {cold_n} | {warm_n} | {warm_s} | "
                "{warm_mib} | {failures} |"
            ).format(
                **cell,
                cold_n=cell["cold"]["successful_n"],
                warm_n=cell["warm"]["successful_n"],
                warm_s=cell["warm"]["wall_s_median"],
                warm_mib=cell["warm"]["peak_live_allocated_mib_median"],
                failures=len(cell["failure_rows"]),
            )
        )
    lines += [
        "",
        (
            "Ratios below are descriptive warm-median ratios only; n=3 supports "
            "no significance claim."
        ),
        "",
    ]
    for item in table["comparisons"]:
        lines.append(
            f"- `{item['comparison_id']}`: "
            f"{item['speed_ratio_warm_median_baseline_over_candidate']} — "
            f"{item['interpretation']}"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-manifest", type=Path, required=True)
    parser.add_argument("--aggregate-spec", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path, required=True)
    args = parser.parse_args()
    outputs = [args.json_out.resolve(), args.markdown_out.resolve()]
    if outputs[0] == outputs[1]:
        raise ValueError("JSON and Markdown outputs must differ")
    inputs = [args.runs_manifest.resolve(), args.aggregate_spec.resolve()]
    if any(output == source for output in outputs for source in inputs):
        raise ValueError("output must not overwrite an input")
    table = build_table(args.runs_manifest, args.aggregate_spec, args.artifact_root)
    args.json_out.write_text(json.dumps(table, indent=2, allow_nan=False) + "\n")
    args.markdown_out.write_text(render_markdown(table))


if __name__ == "__main__":
    main()
