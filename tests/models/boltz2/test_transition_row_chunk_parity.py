"""Exact-parity tests for outer-row chunking in transition_forward (rank-4)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from foldjax.models.boltz2.models.primitives.transition import transition_forward


def _random_params(rng: np.random.Generator, dim: int, hidden: int):
    def w(*shape):
        return jnp.asarray(rng.standard_normal(shape) * 0.1, dtype=jnp.float32)

    return {
        "norm": {"scale": w(dim), "bias": w(dim)},
        "fc1": {"kernel": w(dim, hidden)},
        "fc2": {"kernel": w(dim, hidden)},
        "fc3": {"kernel": w(hidden, dim)},
    }


def test_row_chunk_matches_single_shot() -> None:
    rng = np.random.default_rng(0)
    n, dim, hidden = 40, 8, 20
    params = _random_params(rng, dim, hidden)
    z = jnp.asarray(rng.standard_normal((1, n, n, dim)), dtype=jnp.float32)

    fwd = jax.jit(transition_forward, static_argnames=("row_chunk_size", "chunk_size"))
    single = fwd(params, z)
    chunked = fwd(params, z, row_chunk_size=11)

    diff = float(jnp.max(jnp.abs(single - chunked)))
    assert diff < 1e-6, f"max abs diff={diff}"


def test_row_and_hidden_chunk_matches_single_shot() -> None:
    rng = np.random.default_rng(2)
    n, dim, hidden = 40, 8, 32
    params = _random_params(rng, dim, hidden)
    z = jnp.asarray(rng.standard_normal((1, n, n, dim)), dtype=jnp.float32)

    fwd = jax.jit(transition_forward, static_argnames=("row_chunk_size", "chunk_size"))
    single = fwd(params, z)
    chunked = fwd(params, z, row_chunk_size=11, chunk_size=7)

    np.testing.assert_allclose(
        np.asarray(chunked),
        np.asarray(single),
        rtol=1e-5,
        atol=1e-5,
    )


def test_row_chunk_noop_when_rank3() -> None:
    rng = np.random.default_rng(1)
    dim, hidden = 8, 20
    params = _random_params(rng, dim, hidden)
    s = jnp.asarray(rng.standard_normal((1, 40, dim)), dtype=jnp.float32)

    single = transition_forward(params, s)
    with_row = transition_forward(params, s, row_chunk_size=11)
    assert float(jnp.max(jnp.abs(single - with_row))) == 0.0
