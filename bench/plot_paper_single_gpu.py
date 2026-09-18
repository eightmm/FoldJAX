"""Export schedule-matched performance and structural evidence for the paper.

This standalone script deliberately depends on NumPy and Matplotlib only at plot
creation time.  It consumes the validated JSON emitted by
:mod:`bench.paper_performance_table` and an explicit, hash-pinned structure
index.  Structure rows are unpaired Cartesian prediction pairs; they are shown
as descriptive ranges and points, never as independent biological replicates.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import median
from typing import Any

SCHEMA_VERSION = 1
_STRUCTURE_FIELDS = (
    ("all_atom_rmsd", ("strict_allatom", "metrics", "all_atom_rmsd")),
    ("ca_rmsd_same_fit", ("strict_allatom", "metrics", "ca_rmsd_same_fit")),
    ("tm_matched_residue_normalized", ("ca_only", "tm_matched_residue_normalized")),
    ("ca_fit_rmsd", ("ca_only", "ca_fit_rmsd")),
)


def _reject_constant(value: str) -> None:
    raise ValueError(f"nonfinite JSON constant {value!r} is not allowed")


def _load_object(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    try:
        value = json.loads(raw, parse_constant=_reject_constant)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"{path}: invalid {label} JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path}: {label} must be an object")
    _finite(value, f"{path}: {label}")
    return value, raw


def _finite(value: Any, label: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{label} contains a nonfinite number")
    if isinstance(value, dict):
        for key, child in value.items():
            _finite(child, f"{label}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _finite(child, f"{label}[{index}]")


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _number_or_none(value: Any, label: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number or null")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return value


def _child(value: dict[str, Any], keys: tuple[str, ...], label: str) -> Any:
    current: Any = value
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            raise ValueError(f"{label} is missing {'.'.join(keys)}")
        current = current[key]
    return current


def _safe_relative_path(root: Path, relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError("structure metric path must be relative to artifact root")
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError("structure metric path escapes artifact root") from error
    if not resolved.is_file():
        raise ValueError(f"structure metric file does not exist: {relative}")
    return resolved


def _metric_rows(
    document: dict[str, Any], comparison_id: str, path: str
) -> list[dict[str, Any]]:
    raw_rows = document.get("pairs", [document])
    if not isinstance(raw_rows, list) or not raw_rows:
        raise ValueError(f"{path}: structure document has no metric pairs")
    rows: list[dict[str, Any]] = []
    for pair_index, raw in enumerate(raw_rows, start=1):
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: metric pair {pair_index} must be an object")
        row = {
            "comparison_id": comparison_id,
            "metric_path": path,
            "pair_index": pair_index,
        }
        for name, keys in _STRUCTURE_FIELDS:
            row[name] = _number_or_none(
                _child(raw, keys, f"{path} pair {pair_index}"),
                f"{path} {name}",
            )
        rows.append(row)
    return rows


def load_structure_index(
    index_path: Path, artifact_root: Path, comparisons: set[str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    index, raw = _load_object(index_path, "structure index")
    if index.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("structure index schema_version must be 1")
    entries = index.get("metrics")
    if not isinstance(entries, list):
        raise ValueError("structure index metrics must be a list")
    root = artifact_root.resolve()
    indexed: set[str] = set()
    rows: list[dict[str, Any]] = []
    status: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("structure index entries must be objects")
        comparison_id = _text(entry.get("comparison_id"), "comparison_id")
        if comparison_id not in comparisons or comparison_id in indexed:
            raise ValueError(
                "structure index comparison_id must name one unique aggregate "
                "comparison"
            )
        relative = _text(entry.get("path"), "structure metric path")
        expected = _text(entry.get("sha256"), "structure metric sha256")
        if len(expected) != 64 or any(
            char not in "0123456789abcdef" for char in expected
        ):
            raise ValueError("structure metric sha256 must be lowercase SHA256")
        resolved = _safe_relative_path(root, relative)
        actual = hashlib.sha256(resolved.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"structure metric SHA256 mismatch for {relative}")
        document, _ = _load_object(resolved, "structure metric")
        pair_rows = _metric_rows(document, comparison_id, relative)
        expected_pairs = entry.get("expected_pairs")
        left_count = entry.get("left_sample_count")
        right_count = entry.get("right_sample_count")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in (expected_pairs, left_count, right_count)
        ):
            raise ValueError(
                "expected_pairs and left_sample_count/right_sample_count must be "
                "positive integers"
            )
        if (
            expected_pairs != left_count * right_count
            or expected_pairs != len(pair_rows)
        ):
            raise ValueError(
                f"{relative}: expected_pairs must equal left_sample_count * "
                "right_sample_count and metric row count"
            )
        semantics = entry.get("pair_semantics")
        if semantics != "unpaired_cartesian":
            raise ValueError(
                "structure index pair_semantics must explicitly be unpaired_cartesian"
            )
        indexed.add(comparison_id)
        rows.extend(pair_rows)
        metrics_complete = all(
            row[name] is not None for row in pair_rows for name, _ in _STRUCTURE_FIELDS
        )
        status.append(
            {
                "comparison_id": comparison_id,
                "structural_evidence": "present" if metrics_complete else "missing",
                "structural_status_reason": (
                    "all required metrics present"
                    if metrics_complete
                    else "one or more required structural metrics are null"
                ),
                "metric_path": relative,
                "metric_sha256": actual,
                "pair_count": len(pair_rows),
                "pair_semantics": semantics,
                "left_sample_count": left_count,
                "right_sample_count": right_count,
            }
        )
    for comparison_id in sorted(comparisons - indexed):
        status.append(
            {
                "comparison_id": comparison_id,
                "structural_evidence": "missing",
                "structural_status_reason": "no indexed cross-metric evidence",
                "metric_path": None,
                "metric_sha256": None,
                "pair_count": 0,
                "pair_semantics": None,
                "left_sample_count": None,
                "right_sample_count": None,
            }
        )
    return rows, status


def _cell_map(table: dict[str, Any]) -> dict[str, dict[str, Any]]:
    cells = table.get("cells")
    if table.get("schema_version") != 1 or not isinstance(cells, list):
        raise ValueError("aggregate JSON is not schema version 1")
    out: dict[str, dict[str, Any]] = {}
    for cell in cells:
        if not isinstance(cell, dict):
            raise ValueError("aggregate cells must be objects")
        cell_id = _text(cell.get("cell_id"), "cell_id")
        if cell_id in out:
            raise ValueError("aggregate cell_id values must be unique")
        out[cell_id] = cell
    return out


def _state(cell: dict[str, Any], name: str) -> dict[str, Any]:
    state = cell.get(name)
    if not isinstance(state, dict):
        raise ValueError(f"cell {cell.get('cell_id')!r} lacks {name} summary")
    return state


def _performance_rows(table: dict[str, Any]) -> list[dict[str, Any]]:
    cells = _cell_map(table)
    comparisons = table.get("comparisons")
    if not isinstance(comparisons, list):
        raise ValueError("aggregate comparisons must be a list")
    rows: list[dict[str, Any]] = []
    for comparison in comparisons:
        if not isinstance(comparison, dict):
            raise ValueError("aggregate comparison must be an object")
        comparison_id = _text(comparison.get("comparison_id"), "comparison_id")
        left = cells.get(_text(comparison.get("baseline_cell"), "baseline_cell"))
        right = cells.get(_text(comparison.get("candidate_cell"), "candidate_cell"))
        if left is None or right is None:
            raise ValueError(f"comparison {comparison_id!r} references a missing cell")
        for arm, cell in (("baseline", left), ("candidate", right)):
            for timing in ("cold", "warm"):
                state = _state(cell, timing)
                rows.append(
                    {
                        "comparison_id": comparison_id,
                        "model": cell.get("model"),
                        "case": cell.get("case"),
                        "protocol_id": cell.get("protocol_id"),
                        "precision_policy_id": cell.get("precision_policy_id"),
                        "arm_role": arm,
                        "arm": cell.get("arm"),
                        "timing_state": timing,
                        "wall_s_values": state.get("wall_s_values", []),
                        "wall_s_median": state.get("wall_s_median"),
                        "peak_live_allocated_mib_values": state.get(
                            "peak_live_allocated_mib_values", []
                        ),
                        "peak_live_allocated_mib_median": state.get(
                            "peak_live_allocated_mib_median"
                        ),
                        "successful_n": state.get("successful_n"),
                        "expected_successful_repeats": state.get(
                            "expected_successful_repeats"
                        ),
                        "missing_repeats": state.get("missing_repeats"),
                        "comparison_complete": comparison.get("comparison_complete"),
                        "speed_ratio_warm_median_baseline_over_candidate": (
                            comparison.get(
                                "speed_ratio_warm_median_baseline_over_candidate"
                            )
                        ),
                        "operator_parity_verified": comparison.get(
                            "operator_parity_verified"
                        ),
                        "comparison_limitations": comparison.get("limitations"),
                    }
                )
    return rows


def build_export(
    aggregate_path: Path, structure_index_path: Path, artifact_root: Path
) -> dict[str, Any]:
    table, aggregate_raw = _load_object(aggregate_path, "aggregate")
    performance = _performance_rows(table)
    comparison_ids = {row["comparison_id"] for row in performance}
    structure_rows, structure_status = load_structure_index(
        structure_index_path, artifact_root, comparison_ids
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "scope": (
            "schedule-matched descriptive performance with unpaired-cartesian "
            "structural cross-metrics"
        ),
        "aggregate_sha256": hashlib.sha256(aggregate_raw).hexdigest(),
        "structure_index_sha256": hashlib.sha256(
            structure_index_path.read_bytes()
        ).hexdigest(),
        "memory_statement": (
            "peak values are live allocated high-water marks; reserved memory is "
            "not reported"
        ),
        "statistical_statement": (
            "warm ratios are descriptive medians with n=3 per arm when complete; "
            "no significance, confidence interval, histogram, or biological-replicate "
            "inference is made"
        ),
        "structure_statement": (
            "cross-metric Cartesian pair counts are descriptive and are not "
            "independent observations; plots show points and ranges only"
        ),
        "performance_rows": performance,
        "structure_rows": structure_rows,
        "structure_status": structure_status,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({field for row in rows for field in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _range_text(rows: list[dict[str, Any]], name: str) -> str:
    values = [row[name] for row in rows if row[name] is not None]
    if not values:
        return "missing"
    return f"{min(values):.4g}–{max(values):.4g}"


def _markdown(export: dict[str, Any]) -> str:
    lines = [
        "# Performance and structure evidence",
        "",
        export["memory_statement"] + ".",
        "",
        export["statistical_statement"] + ".",
        "",
        export["structure_statement"] + ".",
        "",
        (
            "| comparison | warm ratio | structural evidence | pairs | all-atom RMSD "
            "Å range | CA RMSD (all-atom fit) Å range | TM-score range | "
            "CA RMSD (CA fit) Å range | operator parity |"
        ),
        "|---|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    statuses = {row["comparison_id"]: row for row in export["structure_status"]}
    rows_by_comparison: dict[str, list[dict[str, Any]]] = {}
    for row in export["structure_rows"]:
        rows_by_comparison.setdefault(row["comparison_id"], []).append(row)
    seen: set[str] = set()
    for row in export["performance_rows"]:
        key = row["comparison_id"]
        if key in seen or row["timing_state"] != "warm":
            continue
        seen.add(key)
        status = statuses[key]
        ratio = row["speed_ratio_warm_median_baseline_over_candidate"]
        ratio_text = ratio if ratio is not None else "not complete"
        metric_rows = rows_by_comparison.get(key, [])
        values = [_range_text(metric_rows, name) for name, _ in _STRUCTURE_FIELDS]
        lines.append(
            f"| {key} | {ratio_text} | {status['structural_evidence']} | "
            f"{status['pair_count']} | " + " | ".join(values) + " | not verified |"
        )
    lines.append("")
    return "\n".join(lines)


def _plot(export: dict[str, Any], pdf: Path, png: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.lines import Line2D

    comparisons = sorted({row["comparison_id"] for row in export["performance_rows"]})
    figure_width = max(15, 1.9 * len(comparisons))
    figure, axes = plt.subplots(2, 3, figsize=(figure_width, 10))
    figure.subplots_adjust(top=0.80, bottom=0.18, left=0.07, right=0.98, hspace=0.65)
    figure.suptitle(
        "Schedule-matched descriptive performance and structural cross-metrics",
        y=0.97,
        fontsize=14,
    )
    figure.legend(
        handles=[
            Line2D(
                [], [], color="#4c78a8", marker="o", linestyle="None", label="baseline"
            ),
            Line2D(
                [], [], color="#f58518", marker="o", linestyle="None", label="candidate"
            ),
            Line2D([], [], color="black", marker="s", linestyle="None", label="cold"),
            Line2D([], [], color="black", marker="o", linestyle="None", label="warm"),
            Line2D([], [], color="#54a24b", linewidth=2, label="structural range"),
            Line2D(
                [],
                [],
                color="black",
                marker="_",
                markersize=12,
                linestyle="None",
                label="median",
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=6,
        frameon=False,
        fontsize=8,
    )
    time_axis, memory_axis = axes[0, 0], axes[0, 1]
    metric_axes = dict(zip((item[0] for item in _STRUCTURE_FIELDS), axes.flat[2:]))
    colors = {"baseline": "#4c78a8", "candidate": "#f58518"}
    labels: list[str] = []
    for comparison_id in comparisons:
        row = next(
            item
            for item in export["performance_rows"]
            if item["comparison_id"] == comparison_id
            and item["timing_state"] == "warm"
        )
        ratio = row["speed_ratio_warm_median_baseline_over_candidate"]
        labels.append(
            comparison_id
            + (f"\n{ratio:.2f}×" if ratio is not None else "\nincomplete")
        )
    for axis, value_key, median_key, title, ylabel in (
        (time_axis, "wall_s_values", "wall_s_median", "CLI time", "seconds"),
        (
            memory_axis,
            "peak_live_allocated_mib_values",
            "peak_live_allocated_mib_median",
            "Live GPU allocations",
            "MiB",
        ),
    ):
        for index, comparison_id in enumerate(comparisons):
            subset = [
                row
                for row in export["performance_rows"]
                if row["comparison_id"] == comparison_id
            ]
            for timing, marker in (("cold", "s"), ("warm", "o")):
                for role, offset in (("baseline", -0.17), ("candidate", 0.17)):
                    row = next(
                        item
                        for item in subset
                        if item["timing_state"] == timing
                        and item["arm_role"] == role
                    )
                    values = [
                        float(value)
                        for value in row[value_key]
                        if isinstance(value, (int, float))
                    ]
                    x = index + offset + (-0.045 if timing == "cold" else 0.045)
                    if values:
                        axis.scatter(
                            [x] * len(values),
                            values,
                            color=colors[role],
                            marker=marker,
                            alpha=0.8,
                            s=28,
                        )
                    if row[median_key] is not None:
                        axis.scatter(
                            [x],
                            [row[median_key]],
                            color="black",
                            marker="_",
                            s=260,
                            linewidths=2.5,
                        )
        axis.set_title(title, fontsize=10)
        axis.set_ylabel(ylabel)
        axis.set_xticks(
            range(len(comparisons)), labels, rotation=35, ha="right", fontsize=7
        )
        axis.grid(axis="y", alpha=0.25)
    metric_specs = {
        "all_atom_rmsd": ("All-atom RMSD", "Å"),
        "ca_rmsd_same_fit": ("CA RMSD: all-atom fit", "Å"),
        "tm_matched_residue_normalized": ("TM-score", "dimensionless"),
        "ca_fit_rmsd": ("CA RMSD: CA fit", "Å"),
    }
    for name, axis in metric_axes.items():
        for index, comparison_id in enumerate(comparisons):
            rows = [
                row
                for row in export["structure_rows"]
                if row["comparison_id"] == comparison_id
            ]
            values = [row[name] for row in rows if row[name] is not None]
            if not values:
                axis.text(index, 0.5, "missing", ha="center", va="center", fontsize=8)
                continue
            if len(values) != len(rows):
                axis.text(
                    index,
                    0.96,
                    "incomplete",
                    transform=axis.get_xaxis_transform(),
                    ha="center",
                    va="top",
                    fontsize=7,
                )
            axis.vlines(index, min(values), max(values), color="#54a24b", alpha=0.65)
            axis.scatter(
                np.full(len(values), index), values, color="#54a24b", s=14, alpha=0.7
            )
            axis.scatter(
                [index],
                [median(values)],
                color="black",
                marker="_",
                s=150,
                linewidths=2,
            )
        title, ylabel = metric_specs[name]
        axis.set_title(title, fontsize=10)
        axis.set_ylabel(ylabel)
        axis.set_xticks(
            range(len(comparisons)), comparisons, rotation=35, ha="right", fontsize=7
        )
        axis.grid(axis="y", alpha=0.25)
    figure.text(
        0.5,
        0.035,
        "Warm ratios are descriptive only (n=3 when complete); operator parity is not "
        "verified. Memory is live allocation high-water (reserved memory unreported). "
        "Structural Cartesian pair counts are not independent observations.",
        ha="center",
        va="center",
        fontsize=8,
        wrap=True,
    )
    figure.savefig(pdf)
    figure.savefig(png, dpi=220)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate-json", type=Path, required=True)
    parser.add_argument("--structure-index", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output in (args.aggregate_json.resolve(), args.structure_index.resolve()):
        raise ValueError("output directory must differ from input files")
    output.mkdir(parents=True, exist_ok=True)
    export = build_export(args.aggregate_json, args.structure_index, args.artifact_root)
    (output / "performance-structure.json").write_text(
        json.dumps(export, indent=2, allow_nan=False) + "\n"
    )
    _write_csv(output / "performance-comparisons.csv", export["performance_rows"])
    _write_csv(output / "structure-pairs.csv", export["structure_rows"])
    (output / "performance-structure.md").write_text(_markdown(export))
    _plot(
        export,
        output / "performance-structure.pdf",
        output / "performance-structure.png",
    )


if __name__ == "__main__":
    main()
