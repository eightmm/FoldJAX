"""Render an explicit per-run evidence table without cross-run aggregation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from copy import deepcopy
from numbers import Real
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
_STATUSES = frozenset({"success", "failed", "oom", "timeout", "unmeasured"})
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _reject_constant(value: str) -> None:
    raise ValueError(f"nonfinite JSON constant {value!r} is not allowed")


def _load_json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    try:
        value = json.loads(raw, parse_constant=_reject_constant)
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{path}: invalid JSON {label}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path}: {label} must be an object")
    _validate_json_finite(value, f"{path}: {label}")
    return value, raw


def _validate_json_finite(value: Any, location: str) -> None:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{location}: nonfinite number is not allowed")
    elif isinstance(value, dict):
        for key, child in value.items():
            _validate_json_finite(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _validate_json_finite(child, f"{location}[{index}]")


def _nonempty(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"run requires nonempty {field}")
    return value


def _relative_artifact(root: Path, value: Any) -> Path:
    relative = _nonempty(value, "result")
    path = Path(relative)
    if path.is_absolute():
        raise ValueError(f"result path must be relative to artifact root: {relative}")
    resolved = (root / path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"result path escapes artifact root: {relative}") from error
    if not resolved.is_file():
        raise ValueError(f"result is missing or not a file: {relative}")
    return resolved


def _finite_nonnegative(value: Any, field: str, *, required: bool) -> float | None:
    if value is None:
        if required:
            raise ValueError(f"success result requires {field}")
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"result {field} must be a finite nonnegative number")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"result {field} must be a finite nonnegative number")
    return result


def _validate_result(
    result: dict[str, Any], run: dict[str, Any], status: str
) -> tuple[float | None, float | None, list[Any]]:
    if result.get("model") != run["model"] or result.get("case") != run["case"]:
        raise ValueError(f"run {run['run_id']!r} result model/case does not match")
    samples_value = result.get("samples")
    if samples_value is None and status != "success":
        samples: list[Any] = []
    elif isinstance(samples_value, list) and all(
        isinstance(sample, dict) and isinstance(sample.get("scores"), dict)
        for sample in samples_value
    ):
        samples = samples_value
    else:
        raise ValueError(
            f"run {run['run_id']!r} result samples must be score-bearing objects"
        )
    failed_value = result.get("failed", False)
    if not isinstance(failed_value, bool):
        raise ValueError(f"run {run['run_id']!r} result failed must be boolean")
    failed = failed_value
    returncode = result.get("returncode", 0)
    if isinstance(returncode, bool) or not isinstance(returncode, int):
        raise ValueError(f"run {run['run_id']!r} result returncode must be an integer")
    broken = failed or returncode != 0
    wall_s = _finite_nonnegative(
        result.get("wall_s"), "wall_s", required=status == "success"
    )
    peak_mib = _finite_nonnegative(
        result.get("peak_mib"), "peak_mib", required=status == "success"
    )
    if status == "success":
        if broken or not samples:
            raise ValueError(
                f"run {run['run_id']!r} success status conflicts with failed, "
                "returncode, or empty samples"
            )
    elif status in {"failed", "oom", "timeout"}:
        if not broken and samples:
            raise ValueError(
                f"run {run['run_id']!r} failure status conflicts with "
                "successful samples"
            )
    return wall_s, peak_mib, deepcopy(samples)


def _validate_manifest(manifest: dict[str, Any], root: Path) -> list[dict[str, Any]]:
    if (
        type(manifest.get("schema_version")) is not int
        or manifest["schema_version"] != SCHEMA_VERSION
    ):
        raise ValueError("manifest schema_version must be 1")
    runs = manifest.get("runs")
    if not isinstance(runs, list):
        raise ValueError("manifest runs must be a list")
    validated: list[dict[str, Any]] = []
    run_ids: set[str] = set()
    for row in runs:
        if not isinstance(row, dict):
            raise ValueError("manifest run rows must be objects")
        run_id = _nonempty(row.get("run_id"), "run_id")
        if run_id in run_ids:
            raise ValueError(f"duplicate run_id {run_id!r}")
        run_ids.add(run_id)
        run = {
            field: _nonempty(row.get(field), field)
            for field in ("model", "case", "arm", "protocol_id", "timing_state")
        }
        run["run_id"] = run_id
        status = row.get("status")
        if not isinstance(status, str) or status not in _STATUSES:
            raise ValueError(f"run {run_id!r} has invalid status")
        note = row.get("note")
        if not isinstance(note, str):
            raise ValueError(f"run {run_id!r} note must be a string")
        result_value = row.get("result")
        result_path: Path | None = None
        source_result: dict[str, Any] | None = None
        source_sha256: str | None = None
        raw_samples: list[Any] = []
        wall_s = peak_mib = None
        if result_value is not None:
            result_path = _relative_artifact(root, result_value)
            expected = row.get("result_sha256")
            if not isinstance(expected, str) or _SHA256.fullmatch(expected) is None:
                raise ValueError(f"run {run_id!r} requires lowercase result_sha256")
            source_result, raw = _load_json(result_path, "result")
            source_sha256 = hashlib.sha256(raw).hexdigest()
            if source_sha256 != expected:
                raise ValueError(
                    f"run {run_id!r} result SHA256 does not match manifest"
                )
            wall_s, peak_mib, raw_samples = _validate_result(source_result, run, status)
        elif row.get("result_sha256") not in (None,):
            raise ValueError(f"run {run_id!r} result_sha256 requires a result")
        if status == "success" and result_path is None:
            raise ValueError(f"run {run_id!r} success status requires a result")
        validated.append(
            {
                "manifest_run": deepcopy(row),
                "run": run,
                "status": status,
                "result_path": result_path,
                "source_result_sha256": source_sha256,
                "source_result": source_result,
                "wall_s": wall_s if status == "success" else None,
                "peak_mib": peak_mib if status == "success" else None,
                "samples": raw_samples,
            }
        )
    return validated


def _build_table(
    manifest: dict[str, Any], raw_manifest: bytes, runs: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "scope": "individual_run_evidence",
        "input_manifest_sha256": hashlib.sha256(raw_manifest).hexdigest(),
        "input_manifest": manifest,
        "no_aggregation_statement": (
            "runs are individual evidence records; no confidence selection, "
            "unit conversion, sample pairing, speed ratio, or aggregate is computed"
        ),
        "runs": [
            {
                **entry["run"],
                "status": entry["status"],
                "result": (
                    None
                    if entry["result_path"] is None
                    else entry["manifest_run"]["result"]
                ),
                "source_result_sha256": entry["source_result_sha256"],
                "note": entry["manifest_run"]["note"],
                "wall_s": entry["wall_s"],
                "peak_mib": entry["peak_mib"],
                "sample_count": len(entry["samples"]),
                "samples": entry["samples"],
                "structure_metrics_status": "not_joined",
            }
            for entry in runs
        ],
    }


def build_table(manifest_path: Path, artifact_root: Path) -> dict[str, Any]:
    """Build a source-preserving record for every manifest run in input order."""
    manifest_path = Path(manifest_path).resolve()
    artifact_root = Path(artifact_root).resolve()
    if not artifact_root.is_dir():
        raise ValueError(f"artifact root is not a directory: {artifact_root}")
    manifest, raw = _load_json(manifest_path, "manifest")
    runs = _validate_manifest(manifest, artifact_root)
    return _build_table(manifest, raw, runs)


def _same_input_file(output: Path, input_path: Path) -> bool:
    return output == input_path or (
        output.exists() and input_path.exists() and output.samefile(input_path)
    )


def _validate_outputs(
    json_output: Path, markdown_output: Path, manifest: Path, runs: list[dict[str, Any]]
) -> None:
    if _same_input_file(json_output, markdown_output):
        raise ValueError("JSON and Markdown outputs must not alias")
    for output in (json_output, markdown_output):
        if _same_input_file(output, manifest):
            raise ValueError("output must not overwrite the manifest")
        for entry in runs:
            result_path = entry["result_path"]
            if result_path is not None and _same_input_file(output, result_path):
                raise ValueError("output must not overwrite a referenced result")


def _markdown_cell(value: Any) -> str:
    if value is None:
        return "null"
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_markdown(table: dict[str, Any]) -> str:
    """Render a per-run table without deriving a comparative claim."""
    lines = [
        "# Individual run evidence",
        "",
        "| run_id | model | case | arm | protocol | timing | status | wall_s | "
        "peak_mib | samples | structure metrics |",
        "|---|---|---|---|---|---|---|---:|---:|---:|---|",
    ]
    for run in table["runs"]:
        lines.append(
            "| "
            + " | ".join(
                _markdown_cell(run[field])
                for field in (
                    "run_id",
                    "model",
                    "case",
                    "arm",
                    "protocol_id",
                    "timing_state",
                    "status",
                    "wall_s",
                    "peak_mib",
                    "sample_count",
                    "structure_metrics_status",
                )
            )
            + " |"
        )
    lines += [
        "",
        "This individual-run table does not by itself establish a "
        "speed-accuracy comparison.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--json-out", required=True, type=Path)
    parser.add_argument("--markdown-out", required=True, type=Path)
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    json_output = args.json_out.resolve()
    markdown_output = args.markdown_out.resolve()
    artifact_root = args.artifact_root.resolve()
    manifest, _raw = _load_json(manifest_path, "manifest")
    runs = _validate_manifest(manifest, artifact_root)
    _validate_outputs(json_output, markdown_output, manifest_path, runs)
    table = _build_table(manifest, _raw, runs)
    json_output.write_text(json.dumps(table, indent=2, allow_nan=False) + "\n")
    markdown_output.write_text(render_markdown(table))


if __name__ == "__main__":
    main()
