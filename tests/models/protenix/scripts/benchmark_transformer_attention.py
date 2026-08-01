#!/usr/bin/env python3
"""Benchmark Transformer pair-bias attention cores at Protenix shapes."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

CASES = {
    "pairformer": (1, 1, 16, 245, 245, 24, jnp.bfloat16),
    "diffusion": (1, 5, 16, 245, 245, 48, jnp.float32),
    "atom_local": (80, 1, 4, 32, 128, 32, jnp.bfloat16),
    "atom_local_samples": (400, 1, 4, 32, 128, 32, jnp.float32),
}


def _xla_attention(q, k, v, bias, mask):
    logits = jnp.einsum("bnhqd,bnhkd->bnhqk", q, k)
    logits = logits + bias + jnp.where(mask, 0.0, -1.0e10)
    probs = jax.nn.softmax(logits.astype(jnp.float32), axis=-1).astype(v.dtype)
    return jnp.einsum("bnhqk,bnhkd->bnhqd", probs, v)


def _cueq_attention(q, k, v, bias, mask):
    from cuequivariance_jax import triangle_attention

    return triangle_attention(q, k, v, bias, mask, scale=1.0)[0]


def _builtin_attention(q, k, v, bias, mask, implementation):
    batch, outer, heads, q_len, head_dim = q.shape
    k_len = k.shape[-2]
    q = jnp.moveaxis(q, -3, -2).reshape(batch * outer, q_len, heads, head_dim)
    k = jnp.moveaxis(k, -3, -2).reshape(batch * outer, k_len, heads, head_dim)
    v = jnp.moveaxis(v, -3, -2).reshape(batch * outer, k_len, heads, head_dim)
    bias = jnp.broadcast_to(bias, (batch, outer, heads, q_len, k_len)).reshape(
        batch * outer, heads, q_len, k_len
    )
    mask = jnp.broadcast_to(mask, (batch, outer, 1, q_len, k_len)).reshape(
        batch * outer, 1, q_len, k_len
    )
    out = jax.nn.dot_product_attention(
        q,
        k,
        v,
        bias=bias,
        mask=mask,
        scale=1.0,
        implementation=implementation,
    )
    return jnp.moveaxis(
        out.reshape(batch, outer, q_len, heads, head_dim), -2, -3
    )


def _time(fn, warmup, iterations):
    for _ in range(warmup):
        jax.block_until_ready(fn())
    values = []
    for _ in range(iterations):
        started = time.perf_counter()
        jax.block_until_ready(fn())
        values.append((time.perf_counter() - started) * 1e3)
    return statistics.median(values), values


def _error(value, reference):
    value = np.asarray(value, dtype=np.float32)
    reference = np.asarray(reference, dtype=np.float32)
    delta = value - reference
    rmse = float(np.sqrt(np.mean(delta * delta)))
    denom = float(np.sqrt(np.mean(reference * reference)))
    return {
        "rmse": rmse,
        "relative_rmse": rmse / denom,
        "max_abs": float(np.max(np.abs(delta))),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=CASES, action="append")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    results = {}
    for index, name in enumerate(args.case or CASES):
        batch, outer, heads, q_len, k_len, head_dim, dtype = CASES[name]
        keys = jax.random.split(jax.random.key(index), 4)
        q = jax.random.normal(
            keys[0], (batch, outer, heads, q_len, head_dim), dtype=dtype
        ) / jnp.sqrt(jnp.asarray(head_dim, dtype=dtype))
        k = jax.random.normal(
            keys[1], (batch, outer, heads, k_len, head_dim), dtype=dtype
        )
        v = jax.random.normal(
            keys[2], (batch, outer, heads, k_len, head_dim), dtype=dtype
        )
        bias = jax.random.normal(
            keys[3], (batch, 1, heads, q_len, k_len), dtype=dtype
        )
        mask = jnp.ones((batch, outer, 1, 1, k_len), dtype=jnp.bool_)
        xla_fn = jax.jit(lambda: _xla_attention(q, k, v, bias, mask))
        cueq_fn = jax.jit(lambda: _cueq_attention(q, k, v, bias, mask))
        xla_sdpa_fn = jax.jit(
            lambda: _builtin_attention(q, k, v, bias, mask, "xla")
        )
        cudnn_fn = jax.jit(
            lambda: _builtin_attention(q, k, v, bias, mask, "cudnn")
        )
        reference = jax.block_until_ready(xla_fn())
        xla_median, xla_times = _time(xla_fn, args.warmup, args.iterations)
        results[name] = {
            "shape": [batch, outer, heads, q_len, k_len, head_dim],
            "dtype": jnp.dtype(dtype).name,
            "xla_warm_median_ms": xla_median,
            "xla_times_ms": xla_times,
        }
        for backend, fn in {
            "cueq": cueq_fn,
            "xla_sdpa": xla_sdpa_fn,
            "cudnn": cudnn_fn,
        }.items():
            try:
                candidate = jax.block_until_ready(fn())
                median, times = _time(fn, args.warmup, args.iterations)
                results[name][backend] = {
                    "warm_median_ms": median,
                    "speedup": xla_median / median,
                    "times_ms": times,
                    "vs_xla": _error(candidate, reference),
                }
            except Exception as error:  # noqa: BLE001 - benchmark records support
                results[name][backend] = {"error": repr(error)}

    payload = {"backend": jax.default_backend(), "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
