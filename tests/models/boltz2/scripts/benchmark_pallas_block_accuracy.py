"""Compare Pallas triangle-attention block sizes against the XLA reference."""

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=952)
    parser.add_argument("--outer", type=int, default=32)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=32)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, 4)
    shape = (1, args.outer, args.heads, args.tokens, args.head_dim)
    q = jax.random.normal(keys[0], shape, dtype=jnp.bfloat16) / np.sqrt(args.head_dim)
    k = jax.random.normal(keys[1], shape, dtype=jnp.bfloat16)
    v = jax.random.normal(keys[2], shape, dtype=jnp.bfloat16)
    tri = jax.random.normal(
        keys[3], (1, 1, args.heads, args.tokens, args.tokens), dtype=jnp.bfloat16
    )
    mask = jnp.zeros((1, args.outer, 1, 1, args.tokens), dtype=jnp.float32)

    reference_fn = jax.jit(
        lambda: _attention_core(q, k, v, tri, mask, args.outer, None)
    )
    reference = np.asarray(jax.block_until_ready(reference_fn())).astype(np.float32)

    results = []
    for block_q, block_k in ((32, 64), (64, 64), (128, 64), (128, 128)):
        candidate_fn = jax.jit(
            lambda bq=block_q, bk=block_k: pallas_attention_core(
                q, k, v, tri, mask, block_q=bq, block_k=bk
            )
        )
        candidate = candidate_fn()
        candidate.block_until_ready()
        times = []
        for _ in range(args.iters):
            start = time.perf_counter()
            candidate = candidate_fn()
            candidate.block_until_ready()
            times.append((time.perf_counter() - start) * 1000)
        candidate_np = np.asarray(candidate).astype(np.float32)
        delta = candidate_np - reference
        results.append(
            {
                "block_q": block_q,
                "block_k": block_k,
                "warm_median_ms": statistics.median(times),
                "warm_times_ms": times,
                "rmse_vs_xla": float(np.sqrt(np.mean(delta * delta))),
                "max_abs_vs_xla": float(np.max(np.abs(delta))),
            }
        )

    payload = {
        "tokens": args.tokens,
        "outer": args.outer,
        "heads": args.heads,
        "head_dim": args.head_dim,
        "dtype": "bfloat16",
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
