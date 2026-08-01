"""pLDDT expectation and aggregation matching Chai-1."""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from foldjax.models.chai.ranking._common import (
    expectation,
    get_chain_masks_and_asyms,
    masked_mean,
)


class PLDDTScores(NamedTuple):
    complex_plddt: jnp.ndarray
    per_chain_plddt: jnp.ndarray
    per_atom_plddt: jnp.ndarray


def plddt(
    logits: jnp.ndarray,
    mask: jnp.ndarray,
    bin_centers: jnp.ndarray,
    *,
    per_atom: bool = False,
) -> jnp.ndarray:
    expectations = expectation(logits, bin_centers)
    if per_atom:
        return expectations
    return masked_mean(mask, expectations, axis=-1)


def per_chain_plddt(
    logits: jnp.ndarray,
    atom_mask: jnp.ndarray,
    asym_id: jnp.ndarray,
    bin_centers: jnp.ndarray,
) -> jnp.ndarray:
    chain_masks, _ = get_chain_masks_and_asyms(asym_id, atom_mask)
    values = expectation(logits, bin_centers)[..., None, :]
    return masked_mean(chain_masks, values, axis=-1)


def get_scores(
    lddt_logits: jnp.ndarray,
    atom_mask: jnp.ndarray,
    atom_asym_id: jnp.ndarray,
    bin_centers: jnp.ndarray,
) -> PLDDTScores:
    return PLDDTScores(
        complex_plddt=plddt(lddt_logits, atom_mask, bin_centers),
        per_chain_plddt=per_chain_plddt(
            lddt_logits, atom_mask, atom_asym_id, bin_centers
        ),
        per_atom_plddt=plddt(lddt_logits, atom_mask, bin_centers, per_atom=True),
    )
