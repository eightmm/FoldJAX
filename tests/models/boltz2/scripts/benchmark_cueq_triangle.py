#!/usr/bin/env python3
"""Compare cuEquivariance and XLA triangle multiplication at production shape."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from foldjax.models.boltz2.bridge.native import load_params
from foldjax.models.boltz2.models.triangle.triangle import (
    triangle_multiplication_forward,
)
from foldjax.models.boltz2.models.triangle.triangle_cueq import (
    cueq_triangle_multiplication_forward,
)


def _time(fn, iterations: int) -> tuple[float, list[float]]:
    jax.block_until_ready(fn())
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        jax.block_until_ready(fn())
        times.append((time.perf_counter() - start) * 1e3)
    return statistics.median(times), times


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=952)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument(
        "--weights", type=Path, default=Path("outputs/native_weights/boltz2_conf")
    )
    parser.add_argument("--fixture-output", type=Path)
    parser.add_argument("--torch-output", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    params = load_params(args.weights)["trunk"]["pairformer_module"]["layers"][0]
    params = jax.tree.map(
        lambda value: value.astype(jnp.bfloat16), params["tri_mul_out"]
    )
    x = jax.random.normal(
        jax.random.key(0), (1, args.tokens, args.tokens, 128), dtype=jnp.bfloat16
    )
    mask = jnp.ones((1, args.tokens, args.tokens), dtype=jnp.bfloat16)

    xla_fn = jax.jit(
        lambda: triangle_multiplication_forward(
            params,
            x,
            mask,
            "outgoing",
            chunk_size=128,
            contraction_precision="float32",
        )
    )
    cueq_fn = jax.jit(
        lambda: cueq_triangle_multiplication_forward(
            params, x, mask, "outgoing"
        )
    )

    xla = np.asarray(jax.block_until_ready(xla_fn())).astype(np.float32)
    cueq = np.asarray(jax.block_until_ready(cueq_fn())).astype(np.float32)
    delta = cueq - xla
    xla_median, xla_times = _time(xla_fn, args.iterations)
    cueq_median, cueq_times = _time(cueq_fn, args.iterations)
    payload = {
        "tokens": args.tokens,
        "dtype": "bfloat16",
        "xla_fp32_contraction": {
            "warm_median_ms": xla_median,
            "warm_times_ms": xla_times,
        },
        "cueq": {
            "warm_median_ms": cueq_median,
            "warm_times_ms": cueq_times,
            "speedup_vs_xla": xla_median / cueq_median,
            "rmse_vs_xla": float(np.sqrt(np.mean(delta * delta))),
            "relative_rmse_vs_xla": float(
                np.sqrt(np.mean(delta * delta))
                / np.sqrt(np.mean(xla * xla))
            ),
            "max_abs_vs_xla": float(np.max(np.abs(delta))),
            "finite": bool(np.isfinite(cueq).all()),
        },
    }
    if args.fixture_output is not None:
        bf16_dtype = np.asarray(x).dtype
        fixture = {
            "x_bits": np.asarray(x).view(np.uint16),
            "mask_bits": np.asarray(mask).view(np.uint16),
        }
        for module_name, values in params.items():
            for value_name, value in values.items():
                fixture[f"{module_name}_{value_name}_bits"] = np.asarray(value).view(
                    np.uint16
                )
        fixture["bf16_itemsize"] = np.asarray(bf16_dtype.itemsize)
        args.fixture_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.fixture_output, **fixture)
    if args.torch_output is not None:
        bf16_dtype = np.asarray(jnp.zeros((), dtype=jnp.bfloat16)).dtype
        torch = np.load(args.torch_output).view(bf16_dtype).astype(np.float32)
        for name, value in (("xla_vs_torch", xla), ("cueq_vs_torch", cueq)):
            error = value - torch
            payload[name] = {
                "rmse": float(np.sqrt(np.mean(error * error))),
                "relative_rmse": float(
                    np.sqrt(np.mean(error * error))
                    / np.sqrt(np.mean(torch * torch))
                ),
                "max_abs": float(np.max(np.abs(error))),
            }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
