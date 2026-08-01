"""Profile the primitive groups in one warm Protenix Pairformer block."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--trunk", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--single-attention-backend",
        choices=("xla", "xla_jit", "xla_sdpa"),
        default="xla_jit",
    )
    parser.add_argument(
        "--triangle-attention-backend",
        choices=("xla", "xla_jit", "tokamax", "cueq", "cueq_jit"),
        default="xla_jit",
    )
    return parser.parse_args()


def warm_measure(fn):
    jax.block_until_ready(fn())
    started = time.perf_counter()
    value = jax.block_until_ready(fn())
    return value, time.perf_counter() - started


def main() -> None:
    args = parse_args()
    jax.config.update("jax_default_matmul_precision", "highest")

    from foldjax.models.protenix.bridge.weights_io import load_native_weights
    from foldjax.models.protenix.models.primitives.attention import attention_pair_bias
    from foldjax.models.protenix.models.primitives.primitives import compiled_transition
    from foldjax.models.protenix.models.triangle.triangle import (
        triangle_attention,
        triangle_multiplication,
    )
    from foldjax.models.protenix.models.trunk_blocks.pairformer import pairformer_stack

    params = load_native_weights(args.weights).pairformer_output.pairformer_stack
    block = params.blocks[0]
    with np.load(args.trunk) as data:
        s = jnp.asarray(data["s_trunk"])
        z = jnp.asarray(data["z_trunk"])

    out, tri_mul_out_seconds = warm_measure(
        lambda: triangle_multiplication(
            z,
            None,
            block.tri_mul_out,
            "outgoing",
            use_jit=args.triangle_attention_backend.endswith("_jit"),
        )
    )
    z = z + out
    out, tri_mul_in_seconds = warm_measure(
        lambda: triangle_multiplication(
            z,
            None,
            block.tri_mul_in,
            "incoming",
            use_jit=args.triangle_attention_backend.endswith("_jit"),
        )
    )
    z = z + out
    heads = int(block.tri_att_start.linear.weight.shape[0])
    out, tri_att_start_seconds = warm_measure(
        lambda: triangle_attention(
            z,
            None,
            block.tri_att_start,
            num_heads=heads,
            attention_backend=args.triangle_attention_backend,
        )
    )
    z = z + out
    z_t = jnp.swapaxes(z, -2, -3)
    out, tri_att_end_seconds = warm_measure(
        lambda: triangle_attention(
            z_t,
            None,
            block.tri_att_end,
            num_heads=heads,
            attention_backend=args.triangle_attention_backend,
        )
    )
    z = jnp.swapaxes(z_t + out, -2, -3)
    out, pair_transition_seconds = warm_measure(
        lambda: compiled_transition(z, block.pair_transition)
    )
    z = z + out
    apb = block.attention_pair_bias._replace(has_s=False, cross_attention_mode=False)
    pair_heads = int(apb.linear_z.weight.shape[0])
    out, single_attention_seconds = warm_measure(
        lambda: attention_pair_bias(
            s,
            None,
            z,
            apb,
            num_heads=pair_heads,
            attention_backend=args.single_attention_backend,
        )
    )
    s = s + out
    _, single_transition_seconds = warm_measure(
        lambda: compiled_transition(s, block.single_transition)
    )

    groups = {
        "triangle_multiplication_out_seconds": tri_mul_out_seconds,
        "triangle_multiplication_in_seconds": tri_mul_in_seconds,
        "triangle_attention_start_seconds": tri_att_start_seconds,
        "triangle_attention_end_seconds": tri_att_end_seconds,
        "pair_transition_seconds": pair_transition_seconds,
        "single_attention_seconds": single_attention_seconds,
        "single_transition_seconds": single_transition_seconds,
    }
    _, stack_scan_seconds = warm_measure(
        lambda: pairformer_stack(
            s,
            z,
            None,
            params,
            use_scan=True,
            single_attention_backend=args.single_attention_backend,
            triangle_attention_backend=args.triangle_attention_backend,
        )
    )
    _, stack_loop_seconds = warm_measure(
        lambda: pairformer_stack(
            s,
            z,
            None,
            params,
            use_scan=False,
            single_attention_backend=args.single_attention_backend,
            triangle_attention_backend=args.triangle_attention_backend,
        )
    )
    metrics = {
        "tokens": int(z.shape[-2]),
        "block_index": 0,
        "single_attention_backend": args.single_attention_backend,
        "triangle_attention_backend": args.triangle_attention_backend,
        "groups": groups,
        "group_sum_seconds": sum(groups.values()),
        "stack_scan_seconds": stack_scan_seconds,
        "stack_loop_seconds": stack_loop_seconds,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
