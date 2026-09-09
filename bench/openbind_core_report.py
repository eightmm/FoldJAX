"""Hash-bound shared-input diagnostics, never independent-input admission."""

import argparse
import json
from pathlib import Path

import numpy as np

from bench.boltz_historical_replay import digest, save_new
from bench.entity_parity import compare_entity_parity


def compare_raw_confidence(capture, output, *, trace_bound=False):
    path = capture / "raw-output.npz"
    names = (
        "plddt_logits", "pae_logits", "pde_logits", "distogram_logits",
        "experimentally_resolved_logits",
    )
    trace_path = capture / "trace.json"
    trace = json.loads(trace_path.read_text()) if trace_path.exists() else {}
    binding = "replay-bound trace verified" if trace_bound else "not replay-bound"
    if not path.exists():
        if "raw_output_sha256" in trace:
            raise ValueError("declared native raw output is missing")
        return {
            "status": "native raw output unavailable", "leaves": {},
            "trace_binding": binding,
            "missing_fields": list(names), "missing_from_native": list(names),
            "missing_from_candidate": [n for n in names if n not in output],
        }
    if "raw_output_sha256" not in trace:
        raise ValueError("undeclared native raw output")
    identity = trace["raw_output_sha256"]
    if digest(path) != identity["sha256"] or path.stat().st_size != identity["bytes"]:
        raise ValueError("native raw output identity mismatch")
    leaves, missing, missing_native, missing_candidate = {}, [], [], []
    with np.load(path, allow_pickle=False) as native:
        for name in names:
            if name not in native:
                missing_native.append(name)
            if name not in output:
                missing_candidate.append(name)
            if name not in native or name not in output:
                missing.append(name)
                continue
            left, right = native[name], output[name]
            # Native carries one outer batch axis; preserve sample/token axes.
            if left.shape != (1, *right.shape):
                raise ValueError(f"raw confidence shape mismatch: {name}")
            delta = left[0].astype(np.float64) - right.astype(np.float64)
            if not np.isfinite(delta).all():
                raise ValueError(f"nonfinite raw confidence: {name}")
            leaves[name] = {
                "scope": "all stored entries, including padding; no mask applied",
                "max_absolute_error": float(np.abs(delta).max()),
                "rmse": float(np.sqrt(np.mean(delta**2))),
            }
    return {
        "status": "measured", "leaves": leaves, "missing_fields": missing,
        "trace_binding": binding,
        "missing_from_native": missing_native,
        "missing_from_candidate": missing_candidate,
    }


def compare(capture, candidate):
    capture, candidate = Path(capture), Path(candidate)
    preflight = json.loads((candidate / "preflight.json").read_text())
    finished = json.loads((candidate / "finished.json").read_text())
    capture_identity = preflight.get("capture_provenance", {})
    public_bindings = None
    if "trace" in capture_identity:
        if digest(capture / "trace.json") != capture_identity["sha256"]:
            raise ValueError("native trace identity mismatch")
        trace = json.loads((capture / "trace.json").read_text())
        requested = trace.get("requested_triangle_backend")
        if requested is not None and requested != preflight["native_backend"]:
            raise ValueError("native triangle backend request/effective mismatch")
        public_bindings = trace.get("public_artifacts")

    def verify_public(path):
        if public_bindings is None:
            return
        identity = public_bindings.get(str(path.relative_to(capture)))
        if (identity is None or digest(path) != identity["sha256"]
                or path.stat().st_size != identity["bytes"]):
            raise ValueError(f"native public artifact identity mismatch: {path.name}")
    bindings = (
        (capture / "input.npz", preflight["input_sha256"]),
        (capture / "tape.npz", preflight["tape_sha256"]),
        (candidate / "prediction.npz", finished["prediction_sha256"]),
    )
    if any(digest(path) != expected for path, expected in bindings):
        raise ValueError("capture or prediction identity mismatch")
    with np.load(capture / "input.npz", allow_pickle=False) as features:
        columns = [
            features["atom_array.0.annotation." + name].tolist()
            for name in ("chain_id", "res_id", "res_name", "atom_name", "element")
        ]
        keys = list(zip(*columns, strict=True))
        labels = columns[0]
        atom_mask = features["atom_mask"]
        if atom_mask.shape != (1, len(keys)) or not np.isin(atom_mask, (0, 1)).all():
            raise ValueError("invalid captured atom mask")
        mask = np.broadcast_to(atom_mask.astype(bool), (5, len(keys)))
    verify_public(capture / "coordinate.npz")
    with np.load(capture / "coordinate.npz", allow_pickle=False) as native:
        coordinates = native["coordinate"]
    if coordinates.shape != (5, len(keys), 3):
        raise ValueError("expected five native coordinate samples")
    with np.load(candidate / "prediction.npz", allow_pickle=False) as output:
        geometry = compare_entity_parity(
            coordinates, output["coordinates"], keys, keys, labels, labels, mask, mask
        )
        confidence = {}
        for name in ("plddt", "ptm", "iptm"):
            values = []
            for sample in range(1, 6):
                suffix = (
                    "confidences.json"
                    if name == "plddt"
                    else "confidences_aggregated.json"
                )
                paths = list(
                    (capture / "predictions").rglob(f"*sample_{sample}_{suffix}")
                )
                if len(paths) != 1:
                    raise ValueError("expected one confidence file per sample")
                verify_public(paths[0])
                values.append(json.loads(paths[0].read_text())[name])
            reference = np.asarray(values, dtype=np.float64)
            measured = output[name].astype(np.float64) * (100 if name == "plddt" else 1)
            if reference.shape != measured.shape:
                raise ValueError(f"confidence shape mismatch: {name}")
            delta = reference - measured
            if not np.isfinite(delta).all():
                raise ValueError(f"nonfinite confidence: {name}")
            confidence[name] = {
                "max_absolute_error": float(np.abs(delta).max()),
                "rmse": float(np.sqrt(np.mean(delta**2))),
                "scale": "0-100" if name == "plddt" else "0-1",
            }
        raw_confidence = compare_raw_confidence(
            capture, output, trace_bound="trace" in capture_identity
        )
    return {
        "scope": (
            "shared captured atom order trusted; sample indices paired positionally; "
            "entities grouped by chain_id; no independent preprocessing proof"
        ),
        "native_public_artifact_binding": (
            "capture-time digests verified against replay-bound trace"
            if public_bindings is not None else
            "coordinate.npz and public confidence JSON lack capture-time digests"
        ),
        "coordinates": geometry,
        "public_confidence": confidence,
        "raw_confidence": raw_confidence,
        "native_backend": preflight["native_backend"],
        "candidate_backend": preflight["candidate_backend"],
        "private_pair_operators": preflight.get("private_pair_operators", False),
        "native_trunk_injection": preflight.get("native_trunk_injection"),
        "candidate_trunk_injection": preflight.get("candidate_trunk_injection"),
        "native_trunk_components": preflight.get("native_trunk_components"),
        "raw_confidence_admitted": False,
        "parity_admitted": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    save_new(args.out, compare(args.capture, args.candidate))


if __name__ == "__main__":
    main()
