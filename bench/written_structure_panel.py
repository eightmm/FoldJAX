"""Run strict written-coordinate metrics over an explicit hashed pair manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

from bench.written_structure_metrics import compare_written_structures

SCHEMA_VERSION = 1
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_COMPARISONS = frozenset({"within_foldjax", "within_upstream", "cross"})


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _nonempty(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"pair requires nonempty {field}")
    return value


def _referenced_path(root: Path, value: Any) -> Path:
    relative = _nonempty(value, "path")
    path = Path(relative)
    if path.is_absolute():
        raise ValueError(f"pair path must be relative to root: {relative}")
    resolved = (root / path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"pair path escapes root: {relative}") from error
    if not resolved.is_file():
        raise ValueError(f"referenced CIF is missing or not a file: {relative}")
    return resolved


def _load_manifest(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    try:
        manifest = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}: invalid JSON manifest") from error
    if (
        not isinstance(manifest, dict)
        or type(manifest.get("schema_version")) is not int
        or manifest["schema_version"] != SCHEMA_VERSION
    ):
        raise ValueError("manifest schema_version must be 1")
    if not isinstance(manifest.get("pairs"), list):
        raise ValueError("manifest pairs must be a list")
    return manifest, hashlib.sha256(raw).hexdigest()


def _validate_pairs(manifest: dict[str, Any], root: Path) -> list[dict[str, Any]]:
    validated: list[dict[str, Any]] = []
    pair_ids: set[str] = set()
    digest_cache: dict[Path, str] = {}
    for row in manifest["pairs"]:
        if not isinstance(row, dict):
            raise ValueError("manifest pair rows must be objects")
        pair_id = _nonempty(row.get("pair_id"), "pair_id")
        if pair_id in pair_ids:
            raise ValueError(f"duplicate pair_id {pair_id!r}")
        pair_ids.add(pair_id)
        for field in ("model", "case"):
            _nonempty(row.get(field), field)
        comparison = row.get("comparison")
        if not isinstance(comparison, str) or comparison not in _COMPARISONS:
            raise ValueError(f"pair {pair_id!r} has invalid comparison")
        if row.get("noise_pairing") != "unpaired":
            raise ValueError(f"pair {pair_id!r} must declare noise_pairing='unpaired'")
        arms: dict[str, Path] = {}
        for arm in ("left", "right"):
            body = row.get(arm)
            if not isinstance(body, dict):
                raise ValueError(f"pair {pair_id!r} requires {arm} object")
            _nonempty(body.get("sample_id"), f"{arm}.sample_id")
            expected = body.get("sha256")
            if not isinstance(expected, str) or _SHA256.fullmatch(expected) is None:
                raise ValueError(f"pair {pair_id!r} requires lowercase {arm}.sha256")
            path = _referenced_path(root, body.get("path"))
            actual = digest_cache.get(path)
            if actual is None:
                actual = _digest(path)
                digest_cache[path] = actual
            if actual != expected:
                raise ValueError(
                    f"pair {pair_id!r} {arm} SHA256 does not match manifest"
                )
            arms[arm] = path
        validated.append(
            {"row": deepcopy(row), "left": arms["left"], "right": arms["right"]}
        )
    return validated


def _build_panel_from_validated(
    manifest: dict[str, Any], manifest_hash: str, pairs: list[dict[str, Any]]
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for pair in pairs:
        row = pair["row"]
        try:
            result = compare_written_structures(pair["left"], pair["right"])
        except ValueError as error:
            results.append(
                {
                    "manifest_pair": row,
                    "status": "unscorable",
                    "metrics": None,
                    "exception_type": type(error).__name__,
                    "reason": str(error),
                }
            )
        else:
            results.append({"manifest_pair": row, "status": "ok", "result": result})
    unscorable = sum(record["status"] == "unscorable" for record in results)
    return {
        "schema_version": SCHEMA_VERSION,
        "scope": "written_coordinates",
        "input_manifest_sha256": manifest_hash,
        "input_manifest": manifest,
        "noise_pairing": "unpaired",
        "no_equivalence_statement": (
            "written-coordinate diagnostics are not an equivalence, "
            "closure, or pass claim"
        ),
        "raw_masks_and_confidence": "not_assessed",
        "pairs": results,
        "counts": {
            "total": len(results),
            "ok": len(results) - unscorable,
            "unscorable": unscorable,
        },
    }


def build_panel(manifest_path: Path, root: Path) -> dict[str, Any]:
    """Compare only explicit unpaired, content-addressed manifest pairs."""
    manifest_path = Path(manifest_path).resolve()
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError(f"root is not a directory: {root}")
    manifest, manifest_hash = _load_manifest(manifest_path)
    pairs = _validate_pairs(manifest, root)
    return _build_panel_from_validated(manifest, manifest_hash, pairs)


def _same_input_file(output: Path, input_path: Path) -> bool:
    return output == input_path or (output.exists() and output.samefile(input_path))


def _validate_output_path(
    output: Path, manifest: Path, pairs: list[dict[str, Any]]
) -> None:
    if _same_input_file(output, manifest):
        raise ValueError("output must not overwrite the manifest")
    for pair in pairs:
        for arm in ("left", "right"):
            if _same_input_file(output, pair[arm]):
                raise ValueError("output must not overwrite a referenced CIF")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    root = args.root.resolve()
    output = args.output.resolve()
    if not root.is_dir():
        raise ValueError(f"root is not a directory: {root}")
    manifest, manifest_hash = _load_manifest(manifest_path)
    pairs = _validate_pairs(manifest, root)
    _validate_output_path(output, manifest_path, pairs)
    panel = _build_panel_from_validated(manifest, manifest_hash, pairs)
    args.output.write_text(json.dumps(panel, indent=2, allow_nan=False) + "\n")
    print(json.dumps(panel["counts"], sort_keys=True))


if __name__ == "__main__":
    main()
