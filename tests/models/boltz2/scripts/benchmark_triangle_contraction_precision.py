#!/usr/bin/env python3
"""Benchmark triangle-contraction precision choices at production token count."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


def _block_until_ready(tree: object) -> None:
    jax.tree_util.tree_map(
        lambda value: (
            value.block_until_ready()
            if hasattr(value, "block_until_ready")
            else value
        ),
        tree,
    )


def _time(fn, iterations: int) -> tuple[float, list[float]]:
    _block_until_ready(fn())
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        _block_until_ready(fn())
        times.append((time.perf_counter() - start) * 1e3)
    return float(np.median(times)), times


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=952)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    key_a, key_b = jax.random.split(jax.random.key(0))
    shape = (1, args.tokens, args.tokens, args.channels)
    a = jax.random.normal(key_a, shape, dtype=jnp.bfloat16)
    b = jax.random.normal(key_b, shape, dtype=jnp.bfloat16)

    def contraction(mode: str) -> jax.Array:
        outputs = []
        for start in range(0, args.tokens, args.chunk_size):
            size = min(args.chunk_size, args.tokens - start)
            a_block = jax.lax.dynamic_slice_in_dim(a, start, size, axis=1)
            if mode == "fp32_operands":
                out = jnp.einsum(
                    "bikd,bjkd->bijd",
                    a_block.astype(jnp.float32),
                    b.astype(jnp.float32),
                )
            elif mode == "fp32_preferred":
                out = jnp.einsum(
                    "bikd,bjkd->bijd",
                    a_block,
                    b,
                    preferred_element_type=jnp.float32,
                )
            else:
                out = jnp.einsum("bikd,bjkd->bijd", a_block, b)
            outputs.append(out.astype(jnp.bfloat16))
        return jnp.concatenate(outputs, axis=1)

    fns = {
        mode: jax.jit(lambda mode=mode: contraction(mode))
        for mode in ("fp32_operands", "fp32_preferred", "bf16")
    }
    reference = np.asarray(fns["fp32_operands"]()).astype(np.float32)
    results = []
    for mode, fn in fns.items():
        median_ms, times_ms = _time(fn, args.iterations)
        value = np.asarray(fn()).astype(np.float32)
        delta = value - reference
        results.append(
            {
                "mode": mode,
                "warm_median_ms": median_ms,
                "warm_times_ms": times_ms,
                "rmse_vs_fp32_operands": float(np.sqrt(np.mean(delta * delta))),
                "max_abs_vs_fp32_operands": float(np.max(np.abs(delta))),
            }
        )

    payload = {
        "tokens": args.tokens,
        "channels": args.channels,
        "chunk_size": args.chunk_size,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
