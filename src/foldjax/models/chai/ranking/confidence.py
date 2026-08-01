"""Confidence-logit postprocessing independent of the confidence network."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from foldjax.models.chai.ranking._common import expectation, midpoint_bin_centers


class ConfidenceScores(NamedTuple):
    pae: jnp.ndarray
    pde: jnp.ndarray
    plddt: jnp.ndarray
    atom_plddt: jnp.ndarray


def confidence_logits_to_scores(
    pae_logits: jnp.ndarray,
    pde_logits: jnp.ndarray,
    plddt_logits: jnp.ndarray,
    *,
    token_mask: jnp.ndarray,
    atom_mask: jnp.ndarray,
    atom_token_index: jnp.ndarray,
) -> ConfidenceScores:
    """Convert Chai confidence logits to public candidate scores.

    Masks and atom-to-token indices are one-dimensional because Chai inference
    fixes the feature batch size to one and concatenates confidence samples on
    the leading axis before this postprocessing step.
    """

    if token_mask.ndim != 1 or atom_mask.ndim != 1 or atom_token_index.ndim != 1:
        raise ValueError("confidence postprocessing expects unbatched masks/indices")
    pair_centers = midpoint_bin_centers(0.0, 32.0, 64)
    plddt_centers = midpoint_bin_centers(0.0, 1.0, plddt_logits.shape[-1])
    pae_all = expectation(pae_logits, pair_centers)
    pde_all = expectation(pde_logits, pair_centers)
    atom_plddt = expectation(plddt_logits, plddt_centers)

    token_indices = jnp.flatnonzero(token_mask.astype(bool))
    pae = pae_all[:, token_indices][:, :, token_indices]
    pde = pde_all[:, token_indices][:, :, token_indices]

    n_tokens = token_mask.shape[0]
    masked_atom_scores = atom_plddt * atom_mask[None, :]
    token_sum = jax.vmap(
        lambda values: jax.ops.segment_sum(values, atom_token_index, n_tokens)
    )(masked_atom_scores)
    token_count = jax.ops.segment_sum(
        atom_mask.astype(atom_plddt.dtype), atom_token_index, n_tokens
    )
    token_plddt = token_sum / jnp.maximum(token_count[None, :], 1)
    return ConfidenceScores(
        pae=pae,
        pde=pde,
        plddt=token_plddt[:, token_indices],
        atom_plddt=atom_plddt,
    )
