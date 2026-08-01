#!/usr/bin/env python3
"""Create production-shape Protenix cuEq fixtures and JAX outputs."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from foldjax.models.protenix.models.primitives.primitives import (
    LayerNormParams,
    LinearParams,
)
from foldjax.models.protenix.models.triangle.triangle import (
    TriangleMultiplicationParams,
    _triangle_attention_block,
)
from foldjax.models.protenix.models.triangle.triangle_cueq import (
    cueq_attention_core,
    cueq_triangle_multiplication,
)


def _bits(value: jax.Array) -> np.ndarray:
    return np.asarray(value).view(np.uint16)


def _time(fn, iterations: int) -> tuple[float, list[float]]:
    jax.block_until_ready(fn())
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        jax.block_until_ready(fn())
        times.append((time.perf_counter() - start) * 1e3)
    return float(np.median(times)), times


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=245)
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    keys = iter(jax.random.split(jax.random.key(7), 16))
    n, c = args.tokens, args.channels
    z = jax.random.normal(next(keys), (n, n, c), dtype=jnp.bfloat16)
    mask = jax.random.bernoulli(next(keys), 0.9, (n, n))

    def weight(shape):
        return jax.random.normal(next(keys), shape, dtype=jnp.bfloat16) * 0.05

    params = TriangleMultiplicationParams(
        LayerNormParams(weight((c,)), weight((c,))),
        LayerNormParams(weight((c,)), weight((c,))),
        LinearParams(weight((c, c))),
        LinearParams(weight((c, c))),
        LinearParams(weight((c, c))),
        LinearParams(weight((c, c))),
        LinearParams(weight((c, c))),
        LinearParams(weight((c, c))),
    )
    mul_functions = {
        direction: jax.jit(
            lambda direction=direction: cueq_triangle_multiplication(
                z, mask, params, direction
            )
        )
        for direction in ("outgoing", "incoming")
    }

    q_shape = (n, 4, n, c // 4)
    q = jax.random.normal(next(keys), q_shape, dtype=jnp.bfloat16)
    k = jax.random.normal(next(keys), q_shape, dtype=jnp.bfloat16)
    v = jax.random.normal(next(keys), q_shape, dtype=jnp.bfloat16)
    bias = jax.random.normal(next(keys), (1, 4, n, n), dtype=jnp.bfloat16)
    mask_bias = jnp.where(mask[:, None, None, :], 0.0, -1e9).astype(jnp.float32)
    attention_fn = jax.jit(
        lambda: cueq_attention_core(
            q, k, v, bias, mask_bias, scale=float((c // 4) ** -0.5)
        )
    )
    attention_xla_fn = jax.jit(
        lambda: _triangle_attention_block(
            q * jnp.asarray((c // 4) ** -0.5, q.dtype),
            k,
            v,
            mask_bias,
            bias,
        )
    )

    values = {name: jax.block_until_ready(fn()) for name, fn in mul_functions.items()}
    values["attention"] = jax.block_until_ready(attention_fn())
    values["attention_xla"] = jax.block_until_ready(attention_xla_fn())
    timings = {}
    for name, fn in (*mul_functions.items(), ("attention", attention_fn)):
        median, times = _time(fn, args.iterations)
        timings[name] = {"warm_median_ms": median, "warm_times_ms": times}

    fixture = {
        "z_bits": _bits(z),
        "mask": np.asarray(mask),
        "norm_in_weight_bits": _bits(params.layer_norm_in.weight),
        "norm_in_bias_bits": _bits(params.layer_norm_in.bias),
        "norm_out_weight_bits": _bits(params.layer_norm_out.weight),
        "norm_out_bias_bits": _bits(params.layer_norm_out.bias),
        "linear_a_p_bits": _bits(params.linear_a_p.weight),
        "linear_a_g_bits": _bits(params.linear_a_g.weight),
        "linear_b_p_bits": _bits(params.linear_b_p.weight),
        "linear_b_g_bits": _bits(params.linear_b_g.weight),
        "linear_z_bits": _bits(params.linear_z.weight),
        "linear_g_bits": _bits(params.linear_g.weight),
        "q_bits": _bits(q),
        "k_bits": _bits(k),
        "v_bits": _bits(v),
        "bias_bits": _bits(bias),
        "jax_outgoing_bits": _bits(values["outgoing"]),
        "jax_incoming_bits": _bits(values["incoming"]),
        "jax_attention_bits": _bits(values["attention"]),
        "jax_attention_xla_bits": _bits(values["attention_xla"]),
    }
    args.fixture.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.fixture, **fixture)
    payload = {"tokens": n, "channels": c, "dtype": "bfloat16", **timings}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
