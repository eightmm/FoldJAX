"""Compare FoldJAX's trunk against upstream stage by stage, and fail on drift.

Run `capture_torch_trunk_stages.py` first on the same feature npz; this reads
its `stages.json` and reports where the two trunks stop agreeing. Attribution
is the point: a whole-trunk comparison says only that something is wrong, while
the stage that first diverges names the module.

Exits non-zero if any stage drifts beyond `--rtol`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def stats(name: str, array) -> dict:
    array = np.asarray(array, dtype=np.float32)
    return {
        "stage": name,
        "mean": float(array.mean()),
        "std": float(array.std()),
        "absmax": float(np.abs(array).max()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--torch-stages", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument(
        "--rtol",
        type=float,
        default=2e-3,
        help="relative tolerance on each stage's std; float32 reductions over a "
        "deep trunk do not reproduce bit for bit across backends",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    import jax.numpy as jnp

    from foldjax.models.protenix.bridge.weights_io import load_native_weights
    from foldjax.models.protenix.data.static_io import load_static_feature_npz
    from foldjax.models.protenix.models.trunk_blocks.embedders import (
        input_feature_embedder,
    )
    from foldjax.models.protenix.models.trunk_blocks.msa import msa_module
    from foldjax.models.protenix.models.trunk_blocks.pairformer import pairformer_stack
    from foldjax.models.protenix.models.trunk_blocks.template import template_embedder
    from foldjax.models.protenix.models.trunk_blocks.trunk import (
        recycle_embeddings,
        relative_position_features,
        trunk_initial_embeddings,
    )

    features = load_static_feature_npz(args.features)
    params = load_native_weights(args.weights)
    trunk = params.pairformer_output
    n_token = int(features["restype"].shape[-2])

    rows = []
    s_inputs = input_feature_embedder(
        features, params.input_embedder, n_token=n_token, n_heads=4
    )
    rows.append(stats("s_inputs", s_inputs))

    relp = features.get("relp")
    if relp is None:
        relp = relative_position_features(features)
    s_init, z_init = trunk_initial_embeddings(
        s_inputs, relp, features["token_bonds"], trunk.trunk.initial
    )
    rows.append(stats("z_init", z_init))

    s = jnp.zeros_like(s_init)
    z = jnp.zeros_like(z_init)
    for cycle in range(args.cycles):
        last = cycle == args.cycles - 1
        s, z = recycle_embeddings(s_init, z_init, s, z, trunk.trunk.recycling)
        if last:
            rows.append(stats("after_recycle_z", z))
        z = z + template_embedder(features, z, None, trunk.template)
        if last:
            rows.append(stats("after_template", z))
        z = msa_module(features, z, s_inputs, None, trunk.msa)
        if last:
            rows.append(stats("after_msa", z))
        s, z = pairformer_stack(s, z, None, trunk.pairformer_stack, use_scan=True)
        if last:
            rows.append(stats("after_pairformer_z", z))
            rows.append(stats("after_pairformer_s", s))

    reference = json.loads(args.torch_stages.read_text())
    expected = {row["stage"]: row for row in reference["stages"]}

    print(f"{'stage':22s} {'foldjax std':>14s} {'torch std':>14s} {'rel':>10s}")
    worst = 0.0
    # A stage the reference does not mention used to print a dash and carry on,
    # which meant a reference file naming no stages at all -- an empty capture, a
    # renamed stage, the wrong file -- reported that everything agreed. An
    # unmatched stage is now a failure, because "not compared" and "compared and
    # equal" are the two things this script exists to tell apart.
    unmatched = [row["stage"] for row in rows if row["stage"] not in expected]
    missing = [stage for stage in expected if stage not in {r["stage"] for r in rows}]
    failures = [(stage, float("inf")) for stage in unmatched + missing]

    for row in rows:
        other = expected.get(row["stage"])
        if other is None:
            print(
                f"{row['stage']:22s} {row['std']:14.5f} {'-':>14s} "
                f"{'-':>10s}  <-- NOT IN REFERENCE"
            )
            continue
        scale = max(abs(other["std"]), 1e-6)
        relative = abs(row["std"] - other["std"]) / scale
        worst = max(worst, relative)
        flag = "" if relative <= args.rtol else "  <-- DRIFT"
        if flag:
            failures.append((row["stage"], relative))
        print(
            f"{row['stage']:22s} {row['std']:14.5f} {other['std']:14.5f} "
            f"{relative:10.2e}{flag}"
        )

    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "stages.json").write_text(
            json.dumps(
                {"impl": "foldjax", "cycles": args.cycles, "stages": rows}, indent=2
            )
        )

    if unmatched or missing:
        for stage in unmatched:
            print(f"\nFAIL: {stage} has no counterpart in the reference capture.")
        for stage in missing:
            print(f"\nFAIL: reference has {stage}; this run produced no such stage.")
        return 1
    if failures:
        first = failures[0][0]
        print(
            f"\nFAIL: {len(failures)} stage(s) beyond rtol={args.rtol:g}. The first "
            f"one, {first}, is where to look -- every stage before it agrees, so "
            "the modules feeding it are not the cause."
        )
        return 1
    print(f"\nOK: every stage within rtol={args.rtol:g} (worst {worst:.2e}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
