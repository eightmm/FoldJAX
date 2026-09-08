"""Compare first-loop observed boundaries with an explicit reference bridge."""

import argparse
import json
from pathlib import Path

import numpy as np

from bench.boltz_closure_capture import save_new
from bench.boltz_pwa_averaging_probe import bitwise_comparison
from bench.esmfold2_tape import _npz, _sha256, _verify_reference_outputs
from bench.esmfold2_tape_report import build_report, compare_arrays, compare_coordinates

KEYS = {"msa_output", "lm_output", "injection_input", "injection_output", "trunk_input"}


def compare_boundaries(left, right, left_dtypes, right_dtypes):
    expected = (
        KEYS | {"msa_input_pair", "msa_input_embedding"}
        if "msa_input_pair" in left
        else KEYS
    )
    if "trunk_output" in left:
        expected = expected | {"trunk_output"}
    if "trunk_output_last" in left:
        expected = expected | {"trunk_output_last"}
    if "coda_input" in left:
        expected = expected | {"coda_input", "coda_output"}
    if any(set(v) != expected for v in (left, right, left_dtypes, right_dtypes)):
        raise ValueError("injection boundary coverage differs")
    result = {}
    for key in sorted(expected):
        a, b = left[key], right[key]
        rank = 3 if key == "msa_input_embedding" else 4
        if a.shape != b.shape or a.ndim != rank or min(a.shape) < 1:
            raise ValueError("injection boundary shape differs")
        if a.dtype != np.float32 or b.dtype != np.float32:
            raise ValueError("injection storage must be FP32")
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError("nonfinite injection boundary")
        result[key] = {
            "native_dtype": left_dtypes[key],
            "candidate_dtype": right_dtypes[key],
            "comparison": bitwise_comparison(a, b),
        }
    return result


def compare_paired_outputs(native, candidate, features):
    """Use the capture supplying boundaries, not the older tape reference."""
    return {
        "scope": "candidate versus the native capture providing observed boundaries",
        "coordinates": compare_coordinates(
            _npz(native / "upstream_coords.npz")["coords"],
            _npz(candidate / "jax_coords.npz")["coords"],
            _npz(features),
        ),
        "confidence": compare_arrays(
            _npz(native / "upstream_confidence.npz"),
            _npz(candidate / "jax_confidence.npz"),
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("native", "reference", "candidate", "features", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    roots = (args.native, args.reference, args.candidate)
    manifests = [root / "metadata.json" for root in roots]
    hashes = [_sha256(p) for p in manifests]
    native, reference, candidate = [json.loads(p.read_text()) for p in manifests]
    # The full output report enforces candidate -> reference manifest identity.
    core = build_report(args.reference, args.candidate, args.features)
    _verify_reference_outputs(args.native, native)
    for key in ("input_sha256", "tape_sha256", "config"):
        if native[key] != reference[key]:
            raise ValueError("native reference bridge differs: " + key)
    for key in ("checkpoint",):
        if native["binding"][key] != reference["binding"][key]:
            raise ValueError("native checkpoint bridge differs")
    if (
        native["binding"]["source"]["native"]
        != reference["binding"]["source"]["native"]
    ):
        raise ValueError("native source bridge differs")
    for filename in ("tape.npz", "features.npz", "upstream_lm.npz"):
        if _sha256(args.native / filename) != _sha256(args.reference / filename):
            raise ValueError("native archive bridge differs: " + filename)
    arrays, dtypes, bindings = (
        [],
        [],
        dict(zip(map(str, manifests), hashes, strict=True)),
    )
    for root, document in ((args.native, native), (args.candidate, candidate)):
        schema = document["injection_schema"]
        if schema["filename"] != "injection.npz":
            raise ValueError("invalid injection filename")
        path = root / schema["filename"]
        if _sha256(path) != schema["sha256"]:
            raise ValueError("injection archive changed")
        bindings[str(path)] = schema["sha256"]
        with np.load(path, allow_pickle=False) as archive:
            arrays.append({k: archive[k] for k in archive.files})
        dtypes.append(schema["dtypes"])
    comparison = compare_boundaries(*arrays, *dtypes)
    paired_outputs = compare_paired_outputs(args.native, args.candidate, args.features)
    if any(_sha256(Path(p)) != h for p, h in bindings.items()):
        raise ValueError("injection binding changed during comparison")
    save_new(
        args.output,
        {
            "boundaries": comparison,
            "bindings": bindings,
            "core": core,
            "paired_native_outputs": paired_outputs,
            "full_model_admission": False,
            "native_policy_recorded": native.get("native_policy"),
            "reference_policy_recorded": reference.get("native_policy"),
        },
    )


if __name__ == "__main__":
    main()
