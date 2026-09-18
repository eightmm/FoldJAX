"""Build an offline, evidence-preserving single-GPU index."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(path: Path, workspace: Path) -> str:
    try:
        return str(path.relative_to(workspace))
    except ValueError:
        return str(path)


def _issue(kind: str, row_id: str, detail: str) -> dict[str, str]:
    return {"kind": kind, "row_id": row_id, "detail": detail}


def _workspace_path(value: str, workspace: Path) -> Path:
    path = Path(value)
    if path.parts and path.parts[0] == workspace.name:
        return workspace.joinpath(*path.parts[1:])
    return workspace / path


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _cp_from_paper(row: dict[str, str]) -> str:
    cards = row.get("cards", "").strip()
    if not cards:
        return "unknown"
    try:
        value = int(cards)
    except ValueError as error:
        raise ValueError(f"invalid cards for {row.get('row_id')}: {cards!r}") from error
    if value < 1:
        raise ValueError(f"invalid cards for {row.get('row_id')}: {cards!r}")
    return "single_gpu" if value == 1 else "cp_excluded"


def _cp_from_result(result: dict[str, Any]) -> str:
    options = result.get("options", {})
    if not isinstance(options, dict):
        raise ValueError("result options must be an object")
    value = options.get("cp_devices")
    if value is None:
        return "unknown"
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"invalid result cp_devices: {value!r}")
    return "single_gpu" if value == 1 else "cp_excluded"


def _status(row: dict[str, str], result: dict[str, Any] | None) -> str:
    if row.get("status", "").strip().lower() not in {"", "ok", "success"}:
        return "failure"
    if result is not None and (
        result.get("failed")
        or int(result.get("returncode", 0) or 0) != 0
        or not result.get("samples")
    ):
        return "failure"
    return "success"


def _structure_records(
    row_id: str,
    manifest: list[dict[str, str]],
    workspace: Path,
    issues: list[dict[str, str]],
) -> list[dict[str, Any]]:
    records = []
    for entry in manifest:
        if entry.get("row_id") != row_id:
            continue
        original = workspace / entry["source_path"]
        copied = workspace / "paper-2026-09" / entry["published_path"]
        record: dict[str, Any] = {
            "original_path": _relative(original, workspace),
            "copied_path": _relative(copied, workspace),
            "original_exists": original.is_file(),
            "copied_exists": copied.is_file(),
            "manifest_sha256": entry.get("sha256") or None,
        }
        if original.is_file():
            record["original_sha256"] = _sha256(original)
        if copied.is_file():
            record["copied_sha256"] = _sha256(copied)
        if record.get("manifest_sha256") and (
            record.get("original_sha256") != record["manifest_sha256"]
            or record.get("copied_sha256") != record["manifest_sha256"]
        ):
            issues.append(_issue("manifest_hash_mismatch", row_id, str(record)))
        if not original.is_file() or not copied.is_file():
            issues.append(_issue("missing_packaged_structure", row_id, str(record)))
        records.append(record)
    return records


def build_index(workspace: Path) -> dict[str, Any]:
    workspace = workspace.resolve()
    bench_root = workspace / "foldjax-bench"
    if bench_root.is_dir():
        workspace = bench_root
    paper = workspace / "paper-2026-09"
    audit = workspace / "paper-single-gpu-audit-20260918"
    scale = workspace / "scale-timing-20260910" / "results"
    benchmarks = paper / "benchmarks.csv"
    manifest_path = paper / "structures" / "MANIFEST.csv"
    availability_path = audit / "raw-structure-availability.json"
    required = (benchmarks, manifest_path, availability_path, scale)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("missing required evidence: " + ", ".join(missing))

    with benchmarks.open(newline="", encoding="utf-8") as stream:
        paper_rows = list(csv.DictReader(stream))
    with manifest_path.open(newline="", encoding="utf-8") as stream:
        manifest = list(csv.DictReader(stream))
    availability = _read_json(availability_path)
    if not isinstance(availability, list):
        raise ValueError("raw-structure-availability.json must be a list")
    available: dict[str, list[str]] = {}
    for entry in availability:
        if not isinstance(entry, dict) or not isinstance(entry.get("row_id"), str):
            raise ValueError("invalid raw-structure-availability entry")
        row_id = entry["row_id"]
        if row_id in available:
            raise ValueError(f"duplicate raw availability row_id: {row_id}")
        paths = entry.get("paths", [])
        if not isinstance(paths, list) or not all(
            isinstance(path, str) for path in paths
        ):
            raise ValueError(f"invalid raw availability paths for {row_id}")
        if entry.get("raw_cif_count") != len(paths):
            raise ValueError(f"raw availability count mismatch for {row_id}")
        available[row_id] = paths

    seen: set[str] = set()
    issues: list[dict[str, str]] = []
    rows: list[dict[str, Any]] = []
    for source in paper_rows:
        row_id = source.get("row_id", "").strip()
        if not row_id or row_id in seen:
            raise ValueError(f"duplicate or empty paper row_id: {row_id!r}")
        seen.add(row_id)
        result_path = (
            workspace / source["result_json"] if source.get("result_json") else None
        )
        result = (
            _read_json(result_path) if result_path and result_path.is_file() else None
        )
        if result is not None and not isinstance(result, dict):
            raise ValueError(f"result is not an object: {result_path}")
        paper_cp = _cp_from_paper(source)
        result_cp = _cp_from_result(result) if result else "unknown"
        if paper_cp != "unknown" and result_cp != "unknown":
            paper_cards = int(source["cards"])
            result_devices = result.get("options", {}).get("cp_devices")
            if paper_cards != result_devices:
                raise ValueError(f"contradictory CP metadata for {row_id}")
        cp = result_cp if result_cp != "unknown" else paper_cp
        status = _status(source, result)
        structures = _structure_records(row_id, manifest, workspace, issues)
        raw_paths = available.get(row_id, [])
        if status == "success" and row_id not in available:
            issues.append(
                _issue(
                    "unavailable_inventory_scope",
                    row_id,
                    "not listed in raw availability",
                )
            )
        raw_structures = []
        for value in raw_paths:
            path = _workspace_path(value, workspace)
            record = {"path": _relative(path, workspace), "exists": path.is_file()}
            if path.is_file():
                record["sha256"] = _sha256(path)
            else:
                issues.append(_issue("missing_raw_structure", row_id, str(path)))
            raw_structures.append(record)
        if (
            status == "success"
            and row_id in available
            and len(raw_structures) != len(structures)
        ):
            issues.append(
                _issue(
                    "packaging_count_mismatch",
                    row_id,
                    f"raw {len(raw_structures)}, copied {len(structures)}",
                )
            )
        if result_path is None or not result_path.is_file():
            issues.append(_issue("missing_result_json", row_id, str(result_path)))
        expected = (
            int(source["n_structures"])
            if source.get("n_structures", "").isdigit()
            else None
        )
        if (
            status == "success"
            and expected is not None
            and row_id in available
            and len(raw_structures) != expected
        ):
            issues.append(
                _issue(
                    "raw_structure_count_mismatch",
                    row_id,
                    f"expected {expected}, found {len(raw_structures)}",
                )
            )
        artifacts = result.get("artifacts", {}) if isinstance(result, dict) else {}
        recorded_refs = artifacts if isinstance(artifacts, dict) else {}
        reasons = [
            "precision_not_assessed",
            "checkpoint_identity_not_assessed",
            "input_identity_not_assessed",
            "rng_tape_not_assessed",
            "independent_inputs_not_assessed",
        ]
        rows.append(
            {
                "row_id": row_id,
                "source": source,
                "result_json": _relative(result_path, workspace)
                if result_path
                else None,
                "result": result,
                "result_json_sha256": _sha256(result_path)
                if result_path and result_path.is_file()
                else None,
                "cp_classification": cp,
                "status": status,
                "structures": structures,
                "raw_structures": raw_structures,
                "provenance": "historical",
                "final_protocol_eligible": False,
                "recorded_refs": recorded_refs,
                "missing_identity_reasons": reasons,
            }
        )

    historical = []
    for path in sorted(scale.glob("*.json")):
        value = _read_json(path)
        if not isinstance(value, dict):
            raise ValueError(f"result is not an object: {path}")
        historical.append(
            {
                "source_file": _relative(path, workspace),
                "record": value,
                "source_sha256": _sha256(path),
                "cp_classification": _cp_from_result(value),
                "provenance": "historical",
                "final_protocol_eligible": False,
            }
        )
    if len({_item["source_file"] for _item in historical}) != len(historical):
        raise ValueError("duplicate historical source files")
    return {
        "schema_version": SCHEMA_VERSION,
        "inputs": {
            name: {
                "path": _relative(path, workspace),
                "sha256": _sha256(path) if path.is_file() else None,
            }
            for name, path in {
                "benchmarks": benchmarks,
                "manifest": manifest_path,
                "availability": availability_path,
            }.items()
        },
        "paper_rows": rows,
        "historical_results": historical,
        "provenance_issues": sorted(
            issues, key=lambda item: (item["row_id"], item["kind"], item["detail"])
        ),
        "counts": {
            "paper_total": len(rows),
            "single_gpu": sum(row["cp_classification"] == "single_gpu" for row in rows),
            "cp_excluded": sum(
                row["cp_classification"] == "cp_excluded" for row in rows
            ),
            "unknown_cp": sum(row["cp_classification"] == "unknown" for row in rows),
            "historical_results": len(historical),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(
        json.dumps(build_index(args.workspace), indent=2, sort_keys=True) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
