"""Exact-parity tests for query-axis chunking in attention_pair_bias_forward."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from foldjax.models.boltz2.models.primitives.attention import (
    attention_pair_bias_forward,
)


def _random_params(rng: np.random.Generator, c_s: int, c_z: int, num_heads: int):
    def w(*shape):
        return jnp.asarray(rng.standard_normal(shape) * 0.1, dtype=jnp.float32)

    return {
        "proj_q": {"kernel": w(c_s, c_s), "bias": w(c_s)},
        "proj_k": {"kernel": w(c_s, c_s)},
        "proj_v": {"kernel": w(c_s, c_s)},
        "proj_g": {"kernel": w(c_s, c_s)},
        "proj_o": {"kernel": w(c_s, c_s)},
        "proj_z_norm": {"scale": w(c_z), "bias": w(c_z)},
        "proj_z": {"kernel": w(c_z, num_heads)},
    }


def test_query_chunk_matches_single_shot() -> None:
    rng = np.random.default_rng(0)
    n, c_s, c_z, num_heads = 48, 16, 12, 4
    params = _random_params(rng, c_s, c_z, num_heads)
    s = jnp.asarray(rng.standard_normal((1, n, c_s)), dtype=jnp.float32)
    z = jnp.asarray(rng.standard_normal((1, n, n, c_z)), dtype=jnp.float32)
    mask = jnp.asarray((rng.random((1, n)) > 0.2).astype(np.float32))

    fwd = jax.jit(attention_pair_bias_forward, static_argnames=("chunk_size",))
    single = fwd(params, s, z, mask, chunk_size=None)
    chunked = fwd(params, s, z, mask, chunk_size=13)

    diff = float(jnp.max(jnp.abs(single - chunked)))
    assert diff < 1e-6, f"max abs diff={diff}"
