"""Exact-parity tests for query-row chunking in triangle attention."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.boltz2.models.triangle.triangle_attention import (
    triangle_attention_forward,
)


def _random_params(rng: np.random.Generator, dim: int, no_heads: int, hidden: int):
    return _random_params_dtype(rng, dim, no_heads, hidden, jnp.float32)


def _random_params_dtype(
    rng: np.random.Generator,
    dim: int,
    no_heads: int,
    hidden: int,
    dtype: jnp.dtype,
):
    def w(*shape):
        return jnp.asarray(rng.standard_normal(shape) * 0.1, dtype=dtype)

    inner = no_heads * hidden
    return {
        "layer_norm": {"scale": w(dim), "bias": w(dim)},
        "linear": {"kernel": w(dim, no_heads)},
        "mha": {
            "linear_q": {"kernel": w(dim, inner)},
            "linear_k": {"kernel": w(dim, inner)},
            "linear_v": {"kernel": w(dim, inner)},
            "linear_g": {"kernel": w(dim, inner)},
            "linear_o": {"kernel": w(inner, dim)},
        },
    }


@pytest.mark.parametrize("starting", [True, False])
def test_chunked_matches_single_shot(starting: bool) -> None:
    rng = np.random.default_rng(0)
    n, dim, no_heads, hidden = 64, 24, 4, 6
    params = _random_params(rng, dim, no_heads, hidden)
    x = jnp.asarray(rng.standard_normal((1, n, n, dim)), dtype=jnp.float32)
    mask = jnp.asarray((rng.random((1, n, n)) > 0.2).astype(np.float32))

    fwd = jax.jit(
        triangle_attention_forward,
        static_argnames=("starting", "chunk_size", "q_chunk_size"),
    )
    single = fwd(params, x, mask, starting=starting, chunk_size=0)
    chunked = fwd(params, x, mask, starting=starting, chunk_size=16)
    inner_chunked = fwd(
        params,
        x,
        mask,
        starting=starting,
        chunk_size=16,
        q_chunk_size=17,
    )

    diff = float(jnp.max(jnp.abs(single - chunked)))
    assert diff < 1e-6, f"starting={starting} max abs diff={diff}"
    inner_diff = float(jnp.max(jnp.abs(single - inner_chunked)))
    assert inner_diff < 1e-6, f"starting={starting} max abs diff={inner_diff}"


def test_triangle_attention_preserves_bfloat16_activation_dtype() -> None:
    rng = np.random.default_rng(1)
    n, dim, no_heads, hidden = 32, 24, 4, 6
    params = _random_params_dtype(rng, dim, no_heads, hidden, jnp.bfloat16)
    x = jnp.asarray(rng.standard_normal((1, n, n, dim)), dtype=jnp.bfloat16)
    mask = jnp.ones((1, n, n), dtype=jnp.bfloat16)

    out = triangle_attention_forward(
        params,
        x,
        mask,
        chunk_size=8,
        q_chunk_size=7,
    )

    assert out.dtype == jnp.bfloat16
