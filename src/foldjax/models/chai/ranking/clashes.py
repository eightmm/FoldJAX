"""Chai-1 atom-clash metrics for inference ranking."""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from foldjax.models.chai.ranking._common import POLYMER_ENTITY_TYPES


class ClashScores(NamedTuple):
    total_clashes: jnp.ndarray
    total_inter_chain_clashes: jnp.ndarray
    chain_chain_clashes: jnp.ndarray
    has_inter_chain_clashes: jnp.ndarray


def _compute_clashes(
    atom_coords: jnp.ndarray,
    atom_mask: jnp.ndarray,
    clash_threshold: float,
) -> jnp.ndarray:
    distances = jnp.linalg.norm(
        atom_coords[..., :, None, :] - atom_coords[..., None, :, :], axis=-1
    )
    valid = atom_mask[..., :, None] & atom_mask[..., None, :]
    valid &= ~jnp.eye(atom_coords.shape[-2], dtype=bool)
    return valid & (distances < clash_threshold)


@partial(jax.jit, static_argnames=("clash_threshold",))
def _chain_clash_tile(
    query_coords: jnp.ndarray,
    key_coords: jnp.ndarray,
    query_mask: jnp.ndarray,
    key_mask: jnp.ndarray,
    query_indices: jnp.ndarray,
    key_indices: jnp.ndarray,
    query_chain_one_hot: jnp.ndarray,
    key_chain_one_hot: jnp.ndarray,
    *,
    clash_threshold: float,
) -> jnp.ndarray:
    """Accumulate one exact directed query/key tile by chain pair."""

    distances = jnp.linalg.norm(
        query_coords[..., :, None, :] - key_coords[..., None, :, :], axis=-1
    )
    valid = query_mask[..., :, None] & key_mask[..., None, :]
    valid &= query_indices[:, None] != key_indices[None, :]
    clashes = (valid & (distances < clash_threshold)).astype(jnp.int32)
    return jnp.einsum(
        "...ij,...ic,...jd->...cd",
        clashes,
        query_chain_one_hot,
        key_chain_one_hot,
    )


def _chain_chain_clashes_tiled(
    atom_coords: jnp.ndarray,
    atom_mask: jnp.ndarray,
    chain_one_hot: jnp.ndarray,
    clash_threshold: float,
    *,
    tile_size: int = 4096,
) -> jnp.ndarray:
    """Stream exact clash counts without materializing the atom-square matrix."""

    atom_count = atom_coords.shape[-2]
    chain_count = chain_one_hot.shape[-1]
    counts = jnp.zeros((*atom_coords.shape[:-2], chain_count, chain_count), jnp.int32)
    for query_start in range(0, atom_count, tile_size):
        query_stop = min(query_start + tile_size, atom_count)
        query_indices = jnp.arange(query_start, query_stop, dtype=jnp.int32)
        row_counts = jnp.zeros_like(counts)
        for key_start in range(0, atom_count, tile_size):
            key_stop = min(key_start + tile_size, atom_count)
            key_indices = jnp.arange(key_start, key_stop, dtype=jnp.int32)
            row_counts = row_counts + _chain_clash_tile(
                atom_coords[..., query_start:query_stop, :],
                atom_coords[..., key_start:key_stop, :],
                atom_mask[..., query_start:query_stop],
                atom_mask[..., key_start:key_stop],
                query_indices,
                key_indices,
                chain_one_hot[..., query_start:query_stop, :],
                chain_one_hot[..., key_start:key_stop, :],
                clash_threshold=clash_threshold,
            )
        counts = counts + row_counts
        jax.block_until_ready(counts)
    return counts


def _has_inter_chain_clashes(
    atom_mask: jnp.ndarray,
    atom_asym_id: jnp.ndarray,
    atom_entity_type: jnp.ndarray,
    inter_chain_clashes: jnp.ndarray,
    *,
    n_chains: int,
    max_clashes: int,
    max_clash_ratio: float,
) -> jnp.ndarray:
    chain_one_hot = jax.nn.one_hot(atom_asym_id, n_chains, dtype=jnp.int32)
    atoms_per_chain = jnp.sum(
        chain_one_hot * atom_mask[..., :, None].astype(jnp.int32), axis=-2
    )
    has_clashes = inter_chain_clashes >= max_clashes
    has_clashes |= (
        inter_chain_clashes / jnp.maximum(atoms_per_chain[..., :, None], 1)
        >= max_clash_ratio
    )
    has_clashes |= (
        inter_chain_clashes / jnp.maximum(atoms_per_chain[..., None, :], 1)
        >= max_clash_ratio
    )

    is_polymer_atom = jnp.any(
        atom_entity_type[..., None] == POLYMER_ENTITY_TYPES, axis=-1
    )
    polymer_counts = jnp.sum(
        chain_one_hot * (atom_mask & is_polymer_atom)[..., :, None].astype(jnp.int32),
        axis=-2,
    )
    polymer_chains = polymer_counts > 0
    polymer_pairs = polymer_chains[..., :, None] & polymer_chains[..., None, :]
    return jnp.any(has_clashes & polymer_pairs, axis=(-2, -1))


def get_scores(
    atom_coords: jnp.ndarray,
    atom_mask: jnp.ndarray,
    atom_asym_id: jnp.ndarray,
    atom_entity_type: jnp.ndarray,
    *,
    clash_threshold: float = 1.1,
    max_clashes: int = 100,
    max_clash_ratio: float = 0.5,
) -> ClashScores:
    """Compute exact Chai clash counts and polymer clash flag.

    Chain count is data-dependent, matching upstream output postprocessing; this
    function is not intended to sit inside the compiled model graph.
    """

    atom_asym_id = atom_asym_id.astype(jnp.int32) - 1
    if bool(jnp.any(atom_asym_id < 0)):
        raise ValueError("atom asym IDs must be one-based positive integers")
    n_chains = int(jnp.max(atom_asym_id).item()) + 1
    chain_one_hot = jax.nn.one_hot(atom_asym_id, n_chains, dtype=jnp.int32)
    chain_chain = _chain_chain_clashes_tiled(
        atom_coords,
        atom_mask,
        chain_one_hot,
        clash_threshold,
    )
    total_clashes = jnp.sum(chain_chain, axis=(-2, -1)) // 2
    eye = jnp.eye(n_chains, dtype=jnp.int32)
    chain_chain = chain_chain // (1 + eye)
    inter_chain = chain_chain * (1 - eye)
    total_inter_chain = jnp.sum(inter_chain, axis=(-2, -1)) // 2
    has_inter_chain = _has_inter_chain_clashes(
        atom_mask,
        atom_asym_id,
        atom_entity_type,
        inter_chain,
        n_chains=n_chains,
        max_clashes=max_clashes,
        max_clash_ratio=max_clash_ratio,
    )
    return ClashScores(
        total_clashes=total_clashes,
        total_inter_chain_clashes=total_inter_chain,
        chain_chain_clashes=chain_chain,
        has_inter_chain_clashes=has_inter_chain,
    )
