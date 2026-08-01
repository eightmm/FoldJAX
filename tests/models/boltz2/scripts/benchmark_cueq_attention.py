#!/usr/bin/env python3
"""Benchmark JAX triangle-attention backends and compare with Torch cuEq."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from foldjax.models.boltz2.models.triangle.triangle_attention import _attention_core
from foldjax.models.boltz2.models.triangle.triangle_attention_pallas import (
    pallas_attention_core,
)
from foldjax.models.boltz2.models.triangle.triangle_cueq import cueq_attention_core


def _time(fn, iterations: int) -> tuple[float, list[float]]:
    jax.block_until_ready(fn())
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        jax.block_until_ready(fn())
        times.append((time.perf_counter() - start) * 1e3)
    return statistics.median(times), times


def _metrics(value: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    error = value - reference
    rmse = float(np.sqrt(np.mean(error * error)))
    return {
        "rmse": rmse,
        "relative_rmse": rmse / float(np.sqrt(np.mean(reference * reference))),
        "max_abs": float(np.max(np.abs(error))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=952)
    parser.add_argument("--outer", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--fixture-output", type=Path)
    parser.add_argument("--torch-output", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    keys = jax.random.split(jax.random.key(0), 4)
    shape = (1, args.outer, 4, args.tokens, 32)
    q = jax.random.normal(keys[0], shape, dtype=jnp.bfloat16)
    k = jax.random.normal(keys[1], shape, dtype=jnp.bfloat16)
    v = jax.random.normal(keys[2], shape, dtype=jnp.bfloat16)
    bias = jax.random.normal(
        keys[3], (1, 1, 4, args.tokens, args.tokens), dtype=jnp.bfloat16
    )
    mask = jnp.ones((1, args.outer, 1, 1, args.tokens), dtype=jnp.bool_)
    mask_bias = jnp.where(mask, 0.0, -1e9).astype(jnp.float32)
    scale = 32**-0.5

    functions = {
        "xla": jax.jit(
            lambda: _attention_core(
                q * jnp.asarray(scale, q.dtype),
                k,
                v,
                bias,
                mask_bias,
                args.outer,
                None,
            )
        ),
        "pallas": jax.jit(
            lambda: pallas_attention_core(
                q * jnp.asarray(scale, q.dtype), k, v, bias, mask_bias
            )
        ),
        "cueq": jax.jit(
            lambda: cueq_attention_core(
                q, k, v, bias, mask_bias, scale=scale, precision=None
            )
        ),
    }
    values = {
        name: np.asarray(jax.block_until_ready(fn())).astype(np.float32)
        for name, fn in functions.items()
    }
    results = {}
    for name, fn in functions.items():
        median, times = _time(fn, args.iterations)
        results[name] = {
            "warm_median_ms": median,
            "warm_times_ms": times,
            "vs_xla": _metrics(values[name], values["xla"]),
        }

    if args.fixture_output is not None:
        fixture = {
            "q_bits": np.asarray(q).view(np.uint16),
            "k_bits": np.asarray(k).view(np.uint16),
            "v_bits": np.asarray(v).view(np.uint16),
            "bias_bits": np.asarray(bias).view(np.uint16),
            "mask": np.asarray(mask),
            "scale": np.asarray(scale),
        }
        args.fixture_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.fixture_output, **fixture)
    if args.torch_output is not None:
        bf16_dtype = np.asarray(jnp.zeros((), dtype=jnp.bfloat16)).dtype
        torch = np.load(args.torch_output).view(bf16_dtype).astype(np.float32)
        for name in results:
            results[name]["vs_torch"] = _metrics(values[name], torch)

    payload = {
        "tokens": args.tokens,
        "outer": args.outer,
        "dtype": "bfloat16",
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
