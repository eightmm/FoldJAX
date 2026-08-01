"""Chai-1 sample ranking assembled from confidence and clash metrics."""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from foldjax.models.chai.ranking import clashes, plddt, ptm
from foldjax.models.chai.ranking._common import get_chain_masks_and_asyms


class SampleRanking(NamedTuple):
    asym_ids: jnp.ndarray
    aggregate_score: jnp.ndarray
    ptm_scores: ptm.PTMScores
    clash_scores: clashes.ClashScores
    plddt_scores: plddt.PLDDTScores


def rank(
    atom_coords: jnp.ndarray,
    atom_mask: jnp.ndarray,
    atom_token_index: jnp.ndarray,
    token_exists_mask: jnp.ndarray,
    token_asym_id: jnp.ndarray,
    token_entity_type: jnp.ndarray,
    token_valid_frames_mask: jnp.ndarray,
    lddt_logits: jnp.ndarray,
    lddt_bin_centers: jnp.ndarray,
    pae_logits: jnp.ndarray,
    pae_bin_centers: jnp.ndarray,
    *,
    clash_threshold: float = 1.1,
    max_clashes: int = 100,
    max_clash_ratio: float = 0.5,
) -> SampleRanking:
    ptm_scores = ptm.get_scores(
        pae_logits,
        token_exists_mask,
        token_valid_frames_mask,
        pae_bin_centers,
        token_asym_id,
    )
    atom_asym_id = jnp.take_along_axis(
        token_asym_id, atom_token_index.astype(jnp.int32), axis=-1
    )
    atom_entity_type = jnp.take_along_axis(
        token_entity_type, atom_token_index.astype(jnp.int32), axis=-1
    )
    clash_scores = clashes.get_scores(
        atom_coords,
        atom_mask,
        atom_asym_id,
        atom_entity_type,
        clash_threshold=clash_threshold,
        max_clashes=max_clashes,
        max_clash_ratio=max_clash_ratio,
    )
    plddt_scores = plddt.get_scores(
        lddt_logits, atom_mask, atom_asym_id, lddt_bin_centers
    )
    aggregate_score = (
        0.2 * ptm_scores.complex_ptm
        + 0.8 * ptm_scores.interface_ptm
        - 100.0 * clash_scores.has_inter_chain_clashes.astype(jnp.float32)
    )
    _, asyms = get_chain_masks_and_asyms(token_asym_id, token_exists_mask)
    return SampleRanking(
        asym_ids=asyms,
        aggregate_score=aggregate_score,
        ptm_scores=ptm_scores,
        clash_scores=clash_scores,
        plddt_scores=plddt_scores,
    )


def get_scores(ranking_data: SampleRanking) -> dict[str, np.ndarray]:
    """Return the exact arrays written to Chai's score NPZ."""

    return {
        "aggregate_score": np.asarray(ranking_data.aggregate_score),
        "ptm": np.asarray(ranking_data.ptm_scores.complex_ptm),
        "iptm": np.asarray(ranking_data.ptm_scores.interface_ptm),
        "per_chain_ptm": np.asarray(ranking_data.ptm_scores.per_chain_ptm),
        "per_chain_pair_iptm": np.asarray(ranking_data.ptm_scores.per_chain_pair_iptm),
        "has_inter_chain_clashes": np.asarray(
            ranking_data.clash_scores.has_inter_chain_clashes
        ),
        "chain_chain_clashes": np.asarray(
            ranking_data.clash_scores.chain_chain_clashes
        ),
    }
