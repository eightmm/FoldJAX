"""Bound native/candidate prefix comparison; no full-model admission."""

import argparse
import json
from pathlib import Path

import numpy as np

from bench.boltz_closure_capture import save_new
from bench.boltz_pwa_averaging_probe import bitwise_comparison
from bench.esmfold2_tape import _sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stack", action="store_true")
    for name in ("native", "candidate", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    reports, bindings = [], {}
    for root, engine in ((args.native, "native"), (args.candidate, "jax")):
        root = root.resolve(strict=True)
        report = json.loads((root / "report.json").read_text())
        scope_key = "stack" if args.stack else "full_msa_block"
        if report["engine"] != engine or not report.get(scope_key):
            raise ValueError("requires full first MSA block captures")
        for path, digest in report["bindings"].items():
            if path in bindings and bindings[path] != digest:
                raise ValueError("shared binding differs")
            bindings[path] = digest
        bindings[str(root / "report.json")] = _sha256(root / "report.json")
        bindings[str(root / "prefix.npz")] = report["archive_sha256"]
        reports.append(report)
    if any(_sha256(Path(p)) != h for p, h in bindings.items()):
        raise ValueError("capture binding changed")
    comparisons = {}
    with (
        np.load(args.native / "prefix.npz") as native,
        np.load(args.candidate / "prefix.npz") as candidate,
    ):
        if set(candidate.files) != set(reports[1]["dtypes"]):
            raise ValueError("candidate array schema differs")
        if args.stack:
            expected = {k for k in native.files if k.startswith("blocks.")}
            if not expected or set(candidate.files) != expected:
                raise ValueError("stack block coverage differs")
        for key in candidate.files:
            target = (
                "blocks.0.msa_transition.norm.input"
                if key == "blocks.0.msa_transition.input"
                else key
            )
            if target not in native.files:
                raise ValueError("native boundary missing: " + target)
            a, b = candidate[key], native[target]
            if a.shape != b.shape:
                raise ValueError("boundary shape differs: " + key)
            comparisons[key] = {
                "native_key": target,
                "candidate_dtype": reports[1]["dtypes"][key],
                "native_dtype": reports[0]["dtypes"][target],
                "comparison": bitwise_comparison(a, b),
            }
    if any(_sha256(Path(p)) != h for p, h in bindings.items()):
        raise ValueError("binding changed during comparison")
    save_new(
        args.output,
        {
            "comparisons": comparisons,
            "bindings": bindings,
            "full_model_admission": False,
            "scope": "instrumented MSA stack"
            if args.stack
            else "instrumented first block",
            "shared_native_start": True,
        },
    )


if __name__ == "__main__":
    main()
