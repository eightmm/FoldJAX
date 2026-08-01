"""Static Chai padding buckets and blocked atom-attention indices."""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

AVAILABLE_MODEL_SIZES = (256, 384, 512, 768, 1024, 1536, 2048)


class PadSizes(NamedTuple):
    n_tokens: int
    n_atoms: int


def get_pad_sizes(num_tokens: int, num_atoms: int) -> PadSizes:
    """Choose the released Chai bucket and its fixed 23-atoms/token size."""
    if num_tokens < 1:
        raise ValueError("num_tokens must be positive")
    if num_tokens > AVAILABLE_MODEL_SIZES[-1]:
        raise ValueError(
            f"num_tokens={num_tokens} exceeds {AVAILABLE_MODEL_SIZES[-1]}"
        )
    padded_tokens = next(
        size for size in AVAILABLE_MODEL_SIZES if size >= num_tokens
    )
    padded_atoms = 23 * padded_tokens
    if num_atoms < 0 or num_atoms > padded_atoms:
        raise ValueError(
            f"num_atoms={num_atoms} exceeds atoms available in bucket: {padded_atoms}"
        )
    return PadSizes(padded_tokens, padded_atoms)


def qkv_indices_for_blocks(
    sequence_length: int,
    query_block_size: int = 32,
    key_block_size: int = 128,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return Chai's circular local-attention indices and non-wrapped mask."""
    if sequence_length <= 0 or query_block_size <= 0 or key_block_size <= 0:
        raise ValueError("sequence and block sizes must be positive")
    if sequence_length % query_block_size:
        raise ValueError(
            f"sequence_length={sequence_length} is not divisible by "
            f"query_block_size={query_block_size}"
        )
    query = jnp.arange(sequence_length).reshape(-1, query_block_size)
    keys_unwrapped = query[:, :1] + (query_block_size - key_block_size) // 2
    keys_unwrapped = keys_unwrapped + jnp.arange(key_block_size)
    key_valid = (keys_unwrapped >= 0) & (keys_unwrapped < sequence_length)
    keys = keys_unwrapped % sequence_length
    return query, keys, key_valid


def block_atom_pair_mask(
    atom_mask: jnp.ndarray,
    query_indices: jnp.ndarray,
    key_indices: jnp.ndarray,
    key_is_not_wrapped: jnp.ndarray,
) -> jnp.ndarray:
    """Build the exact `(batch, block, query, key)` local pair mask."""
    if atom_mask.ndim != 2:
        raise ValueError("atom_mask must have shape (batch, atoms)")
    query_mask = atom_mask[:, query_indices]
    key_mask = atom_mask[:, key_indices]
    return (
        query_mask[..., :, None]
        & key_mask[..., None, :]
        & key_is_not_wrapped[None, :, None, :]
    )
