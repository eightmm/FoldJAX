"""Sequence-local atom blocking (AF3 Algorithm 7 neighbourhoods).

Atom attention is local: each query block of ``n_query`` atoms attends to a
``n_key``-wide window centred on that block. These helpers build the gather
indices and masks for those windows.

The edge handling is the subtle part and the reason this is ported as its own
unit. A window is shifted rather than clipped: a block that would start before
atom 0 slides right to start at 0, and a block that would run past the last real
atom slides left to end there. ``n_atom`` comes from the mask, not the padded
tensor length, so padding never enters a window.
"""

from __future__ import annotations

import math

import jax.numpy as jnp


def query_block_padding(n_atom: int, n_query: int) -> int:
    """Return the right padding that makes ``n_atom`` divisible by ``n_query``."""
    return (-n_atom) % n_query


def block_indices(
    atom_mask: jnp.ndarray, *, n_query: int, n_key: int
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return per-block key gather indices and their invalid-position mask.

    Args:
        atom_mask: ``[..., N_atom]`` atom mask. Its row sums give each sample's
            true atom count, which is what the shifting logic uses.
        n_query: query block height.
        n_key: key window width.

    Returns:
        ``(safe_indices, invalid_mask)``, both ``[..., N_blocks, N_key]``.
        ``safe_indices`` is clamped into ``[0, n_atom - 1]`` so it is always a
        legal gather; ``invalid_mask`` marks the positions that were out of range
        before clamping and whose gathered values must be zeroed.
    """
    batch_dims = atom_mask.shape[:-1]
    n_atom_padded = atom_mask.shape[-1]
    offset = n_query // 2
    num_blocks = math.ceil(n_atom_padded / n_query)

    centers = offset + jnp.arange(num_blocks) * n_query
    centers = jnp.broadcast_to(centers, (*batch_dims, num_blocks))

    n_atom = jnp.sum(atom_mask, axis=-1, keepdims=True)
    n_atom = jnp.broadcast_to(n_atom, (*batch_dims, num_blocks))

    window = jnp.arange(-(n_key // 2), n_key - n_key // 2)
    initial = (centers[..., None] + window[None, :]).astype(jnp.int32)

    underflow = jnp.maximum(-initial[..., 0], 0)
    overflow = jnp.maximum(initial[..., -1] - (n_atom - 1), 0)
    shift = jnp.where(underflow > 0, underflow, -overflow)

    final = initial + shift[..., None]

    n_atom = n_atom[..., None]
    invalid = (final < 0) | (final >= n_atom)
    safe = jnp.clip(final, 0, jnp.maximum(n_atom - 1, 0))
    return safe.astype(jnp.int32), invalid


def pair_atom_block_mask(
    atom_mask: jnp.ndarray,
    *,
    n_query: int,
    indices: jnp.ndarray,
    invalid: jnp.ndarray,
) -> jnp.ndarray:
    """Return the ``[..., N_blocks, N_query, N_key]`` query/key block pair mask."""
    n_atom = atom_mask.shape[-1]
    num_blocks = math.ceil(n_atom / n_query)
    pad = query_block_padding(n_atom, n_query)

    padded = jnp.pad(atom_mask, [(0, 0)] * (atom_mask.ndim - 1) + [(0, pad)])
    mask_q = padded.reshape((*atom_mask.shape[:-1], num_blocks, n_query))

    # Upstream's indices come out float, because the clamp bound is derived from
    # a summed (float) mask; it casts with .long() at every use site.
    mask_k = jnp.take_along_axis(
        atom_mask[..., None, :], indices.astype(jnp.int32), axis=-1
    )
    mask_k = jnp.where(invalid, 0.0, mask_k)

    return mask_q[..., None] * mask_k[..., None, :]


def single_rep_to_blocks(
    ql: jnp.ndarray,
    atom_mask: jnp.ndarray,
    *,
    n_query: int,
    n_key: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Split an atom single representation into local query and key blocks.

    Args:
        ql: ``[..., N_atom, C]`` atom single representation.
        atom_mask: ``[..., N_atom]`` atom mask.
        n_query: query block height.
        n_key: key window width.

    Returns:
        ``(query_blocks, key_blocks, pair_mask)`` shaped
        ``[..., N_blocks, N_query, C]``, ``[..., N_blocks, N_key, C]`` and
        ``[..., N_blocks, N_query, N_key]``.
    """
    batch_dims = ql.shape[:-2]
    n_atom, n_dim = ql.shape[-2:]
    num_blocks = math.ceil(n_atom / n_query)
    pad = query_block_padding(n_atom, n_query)

    padded = jnp.pad(ql, [(0, 0)] * (ql.ndim - 2) + [(0, pad), (0, 0)])
    query_blocks = padded.reshape((*batch_dims, num_blocks, n_query, n_dim))

    atom_mask = jnp.broadcast_to(atom_mask, (*batch_dims, n_atom))
    indices, invalid = block_indices(atom_mask, n_query=n_query, n_key=n_key)

    # [..., N_blocks, N_key, C]
    key_blocks = jnp.take_along_axis(
        ql[..., None, :, :], indices[..., None].astype(jnp.int32), axis=-2
    )
    key_blocks = jnp.where(invalid[..., None], 0.0, key_blocks)

    pair_mask = pair_atom_block_mask(
        atom_mask, n_query=n_query, indices=indices, invalid=invalid
    )
    return query_blocks, key_blocks, pair_mask


def pair_rep_to_blocks(
    zij_trunk: jnp.ndarray,
    atom_to_token_index: jnp.ndarray,
    atom_mask: jnp.ndarray,
    *,
    n_query: int,
    n_key: int,
) -> jnp.ndarray:
    """Gather a token pair representation into sequence-local atom pair blocks.

    Each atom maps to its owning token, so the ``[N_token, N_token]`` pair
    representation is indexed twice: once by the query block's token ids and once
    by the key window's. Out-of-range key slots are zeroed, then the whole block
    is masked by the query/key atom pair mask.

    Args:
        zij_trunk: ``[..., N_token, N_token, C]`` trunk pair representation.
        atom_to_token_index: ``[..., N_atom]`` owning token of each atom.
        atom_mask: ``[..., N_atom]`` atom mask.
        n_query: query block height.
        n_key: key window width.

    Returns:
        ``[..., N_blocks, N_query, N_key, C]`` atom pair conditioning.
    """
    batch_dims = zij_trunk.shape[:-3]
    n_atom = atom_to_token_index.shape[-1]
    num_blocks = math.ceil(n_atom / n_query)
    pad = query_block_padding(n_atom, n_query)
    flat_batch = math.prod(batch_dims) if batch_dims else 1

    padded_q = jnp.pad(
        atom_to_token_index,
        [(0, 0)] * (atom_to_token_index.ndim - 1) + [(0, pad)],
    )
    q_indices = padded_q.reshape((flat_batch, num_blocks, n_query)).astype(jnp.int32)

    atom_mask = jnp.broadcast_to(atom_mask, (*batch_dims, n_atom))
    key_indices, invalid = block_indices(atom_mask, n_query=n_query, n_key=n_key)

    k_indices = jnp.take_along_axis(
        atom_to_token_index[..., None, :], key_indices.astype(jnp.int32), axis=-1
    )
    k_indices = k_indices.reshape((flat_batch, num_blocks, n_key)).astype(jnp.int32)
    invalid_flat = invalid.reshape((flat_batch, num_blocks, n_key))

    flat_z = zij_trunk.reshape((flat_batch, *zij_trunk.shape[-3:]))
    batch_index = jnp.arange(flat_batch).reshape(-1, 1, 1, 1)
    # [flat, N_blocks, N_query, N_key, C]
    plm = flat_z[batch_index, q_indices[..., None], k_indices[..., None, :]]
    plm = jnp.where(invalid_flat[..., None, :, None], 0.0, plm)

    pair_mask = pair_atom_block_mask(
        atom_mask, n_query=n_query, indices=key_indices, invalid=invalid
    )
    plm = plm.reshape((*batch_dims, num_blocks, n_query, n_key, plm.shape[-1]))
    return plm * pair_mask[..., None]
