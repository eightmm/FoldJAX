"""Deterministic atom-to-token reductions without a dense owner matrix.

ESMFold2 packs each token's atoms contiguously and masks only a suffix. The
released chemistry has at most 23 atoms per token. Those three facts let us
gather a fixed-width group for each token and reduce it in rank order, without
materialising ``[atoms, tokens]`` or using colliding scatter-add atomics.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

#: The released pLDDT/resolved heads carry one row for each of these ranks.
MAX_ATOMS_PER_TOKEN = 23


def sum_by_token(
    values: jnp.ndarray,
    atom_to_token: jnp.ndarray,
    n_tokens: int,
    valid_mask: jnp.ndarray,
) -> jnp.ndarray:
    """Sum ``[batch, atoms, ...]`` values by contiguous token owner.

    ``valid_mask`` must be a prefix and valid owners must be nondecreasing,
    which is the ESMFold2 featurizer contract. Low-precision values accumulate
    in float32 and narrow only after the fixed-rank reduction, matching the
    historical dense dot's dtype boundary. The loop is deliberate: a tree
    reduction changes enough float32 additions to move some bfloat16 outputs.
    """
    if values.ndim < 2:
        raise ValueError("values must have batch and atom axes")
    if atom_to_token.shape != values.shape[:2]:
        raise ValueError(
            "atom_to_token must match the batch and atom axes of values"
        )
    if valid_mask.shape != atom_to_token.shape:
        raise ValueError("valid_mask must match atom_to_token")
    if n_tokens < 1:
        raise ValueError("n_tokens must be positive")

    result_dtype = values.dtype
    accumulation_dtype = (
        jnp.float32
        if jnp.issubdtype(result_dtype, jnp.inexact)
        and jnp.dtype(result_dtype).itemsize < jnp.dtype(jnp.float32).itemsize
        else result_dtype
    )
    n_atoms = values.shape[1]
    if n_atoms == 0:
        return jnp.zeros(
            (values.shape[0], n_tokens, *values.shape[2:]), dtype=result_dtype
        )

    def reduce_row(
        row: jnp.ndarray, owner: jnp.ndarray, row_mask: jnp.ndarray
    ) -> jnp.ndarray:
        active = row_mask.astype(bool) & (owner >= 0) & (owner < n_tokens)
        # The inactive suffix sorts after every real owner. Searchsorted then
        # finds each token's group without an atom-by-token comparison matrix.
        sortable_owner = jnp.where(active, owner, n_tokens)
        token = jnp.arange(n_tokens, dtype=owner.dtype)
        starts = jnp.searchsorted(
            sortable_owner, token, side="left", method="scan"
        )
        ends = jnp.searchsorted(
            sortable_owner, token, side="right", method="scan"
        )
        initial = jnp.zeros(
            (n_tokens, *row.shape[1:]), dtype=accumulation_dtype
        )

        def add_rank(rank: int, total: jnp.ndarray) -> jnp.ndarray:
            # Gather only one [tokens, ...] rank at a time. Materialising all
            # 23 ranks together would itself become a multi-GiB buffer at the
            # released 32 diffusion samples and token width.
            indices = jnp.minimum(starts + rank, n_atoms - 1)
            gathered = row[indices].astype(accumulation_dtype)
            member = rank < (ends - starts)
            if row.ndim > 1:
                member = member.reshape(
                    member.shape + (1,) * (row.ndim - 1)
                )
            return total + jnp.where(member, gathered, 0)

        reduced = jax.lax.fori_loop(
            0,
            MAX_ATOMS_PER_TOKEN,
            add_rank,
            initial,
        )
        return reduced.astype(result_dtype)

    return jax.vmap(reduce_row)(values, atom_to_token, valid_mask)


__all__ = ["MAX_ATOMS_PER_TOKEN", "sum_by_token"]
