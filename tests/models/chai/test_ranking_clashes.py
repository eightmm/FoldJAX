"""Regression tests for memory-bounded clash accumulation."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from foldjax.models.chai.ranking.clashes import (
    _chain_chain_clashes_tiled,
    _compute_clashes,
)


def test_tiled_chain_clashes_exactly_match_dense_reference() -> None:
    rng = np.random.default_rng(20260715)
    batch_size, atom_count, chain_count = 2, 11, 4
    coords = jnp.asarray(
        rng.normal(size=(batch_size, atom_count, 3)).astype(np.float32)
    )
    atom_mask = jnp.asarray(
        [
            [True, True, False, True, True, True, True, False, True, True, True],
            [True, False, True, True, True, False, True, True, True, True, False],
        ]
    )
    atom_asym_id = jnp.asarray([0, 0, 1, 1, 2, 2, 3, 3, 0, 1, 2])
    chain_one_hot = jax.nn.one_hot(
        atom_asym_id, chain_count, dtype=jnp.int32
    )
    threshold = 1.75

    dense = _compute_clashes(coords, atom_mask, threshold).astype(jnp.int32)
    dense_counts = jnp.einsum(
        "...ij,ic,jd->...cd", dense, chain_one_hot, chain_one_hot
    )
    tiled_counts = _chain_chain_clashes_tiled(
        coords,
        atom_mask,
        chain_one_hot,
        threshold,
        tile_size=4,
    )

    np.testing.assert_array_equal(np.asarray(tiled_counts), np.asarray(dense_counts))
