#!/usr/bin/env python3
"""Compare Torch cuEq with saved Protenix JAX cuEq production-shape outputs."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch


def _bf16(bits: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(bits.copy()).view(torch.bfloat16).cuda()


def _metrics(value: torch.Tensor, reference_bits: np.ndarray) -> dict[str, float]:
    reference = _bf16(reference_bits).float()
    error = value.float() - reference
    rmse = torch.sqrt(torch.mean(error.square()))
    scale = torch.sqrt(torch.mean(reference.square()))
    return {
        "rmse": float(rmse.item()),
        "relative_rmse": float((rmse / scale).item()),
        "max_abs": float(error.abs().max().item()),
    }


def _time(fn, iterations: int) -> tuple[torch.Tensor, float, list[float]]:
    value = fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        value = fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1e3)
    return value, statistics.median(times), times


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from cuequivariance_torch.primitives.triangle import (
        triangle_attention,
        triangle_multiplicative_update,
    )

    f = np.load(args.fixture)
    z, mask = _bf16(f["z_bits"])[None], torch.from_numpy(f["mask"])[None].cuda()
    p_in = torch.cat((_bf16(f["linear_a_p_bits"]), _bf16(f["linear_b_p_bits"])))
    g_in = torch.cat((_bf16(f["linear_a_g_bits"]), _bf16(f["linear_b_g_bits"])))
    common = dict(
        x=z,
        mask=mask,
        norm_in_weight=_bf16(f["norm_in_weight_bits"]),
        norm_in_bias=_bf16(f["norm_in_bias_bits"]),
        p_in_weight=p_in,
        g_in_weight=g_in,
        norm_out_weight=_bf16(f["norm_out_weight_bits"]),
        norm_out_bias=_bf16(f["norm_out_bias_bits"]),
        p_out_weight=_bf16(f["linear_z_bits"]),
        g_out_weight=_bf16(f["linear_g_bits"]),
        eps=1e-5,
    )
    results = {}
    with torch.inference_mode():
        for direction in ("outgoing", "incoming"):
            def fn(direction=direction):
                return triangle_multiplicative_update(
                    direction=direction, **common
                )[0]

            value, median, times = _time(fn, args.iterations)
            results[direction] = {
                "warm_median_ms": median,
                "warm_times_ms": times,
                "vs_jax": _metrics(value, f[f"jax_{direction}_bits"]),
            }

        q, k, v = (_bf16(f[name]) for name in ("q_bits", "k_bits", "v_bits"))
        bias = _bf16(f["bias_bits"]).float()
        attn_mask = mask[:, :, None, None, :]
        def attn_fn():
            return triangle_attention(
                q[None], k[None], v[None], bias[None], attn_mask, scale=32**-0.5
            )[0]

        value, median, times = _time(attn_fn, args.iterations)
        results["attention"] = {
            "warm_median_ms": median,
            "warm_times_ms": times,
            "vs_jax": _metrics(value, f["jax_attention_bits"]),
            "vs_jax_xla": _metrics(value, f["jax_attention_xla_bits"]),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
