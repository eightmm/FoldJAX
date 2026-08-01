"""Compare Torch/JAX prepared-only scientific parity artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scientific_parity import NUMERIC_PREPARED_TOLERANCES, validate_contract


def _load(path: Path) -> tuple[dict, dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"]))
        arrays = {
            name.removeprefix("prepared__"): np.asarray(archive[name])
            for name in archive.files
            if name.startswith("prepared__")
        }
    return metadata, arrays


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("torch_artifact", type=Path)
    parser.add_argument("jax_artifact", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch_metadata, torch_arrays = _load(args.torch_artifact)
    jax_metadata, jax_arrays = _load(args.jax_artifact)
    validate_contract(
        torch_metadata,
        jax_metadata,
        torch_arrays,
        jax_arrays,
    )
    torch_manifest = torch_metadata["prepared_tensors"]
    jax_manifest = jax_metadata["prepared_tensors"]
    common = sorted(
        name
        for name in set(torch_manifest) & set(jax_manifest)
        if name.startswith("features.")
    )
    exact = [
        name
        for name in common
        if torch_manifest[name]["semantic_sha256"]
        == jax_manifest[name]["semantic_sha256"]
    ]
    numeric_drift = {}
    tolerance_matched = []
    for name in sorted(set(common) - set(exact)):
        tolerance = NUMERIC_PREPARED_TOLERANCES[name]
        delta = jax_arrays[name].astype(np.float64) - torch_arrays[name].astype(
            np.float64
        )
        tolerance_matched.append(name)
        numeric_drift[name] = {
            **tolerance,
            "mae": float(np.mean(np.abs(delta))),
            "max_abs": float(np.max(np.abs(delta))),
        }
    report = {
        "status": "pass",
        "fixture_id": torch_metadata["fixture_id"],
        "config": torch_metadata["config"],
        "branches": torch_metadata["branches"],
        "common_feature_count": len(common),
        "exact_feature_count": len(exact),
        "tolerance_matched_features": tolerance_matched,
        "numeric_drift": numeric_drift,
        "note": "recycle count is recorded but does not alter prepared features",
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
