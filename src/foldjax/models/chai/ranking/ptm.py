"""Predicted TM metrics ported from Chai-1."""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from foldjax.models.chai.ranking._common import expectation, get_chain_masks_and_asyms


class PTMScores(NamedTuple):
    complex_ptm: jnp.ndarray
    interface_ptm: jnp.ndarray
    per_chain_ptm: jnp.ndarray
    per_chain_pair_iptm: jnp.ndarray


def tm_d0(n_tokens: jnp.ndarray) -> jnp.ndarray:
    """Compute the Chai TM-score normalization distance."""

    n_tokens = jnp.maximum(n_tokens, 19)
    return 1.24 * (n_tokens - 15) ** (1.0 / 3.0) - 1.8


def _compute_ptm(
    logits: jnp.ndarray,
    query_res_mask: jnp.ndarray,
    query_has_frame_mask: jnp.ndarray,
    key_res_mask: jnp.ndarray,
    bin_centers: jnp.ndarray,
) -> jnp.ndarray:
    num_key_tokens = jnp.sum(key_res_mask, axis=-1).astype(logits.dtype)
    d0 = tm_d0(num_key_tokens)[..., None]
    bin_weights = 1.0 / (1.0 + (bin_centers / d0) ** 2)
    bin_weights = bin_weights[..., None, None, :]
    valid_pairs = (query_has_frame_mask & query_res_mask)[..., :, None]
    valid_pairs = valid_pairs & key_res_mask[..., None, :]
    expected_pair_tm = expectation(logits, bin_weights)
    qk_weights = valid_pairs.astype(logits.dtype) / jnp.maximum(
        num_key_tokens[..., None, None], 1
    )
    query_key_tm = jnp.sum(qk_weights * expected_pair_tm, axis=-1)
    return jnp.max(query_key_tm, axis=-1)


def complex_ptm(
    pae_logits: jnp.ndarray,
    token_exists_mask: jnp.ndarray,
    valid_frames_mask: jnp.ndarray,
    bin_centers: jnp.ndarray,
) -> jnp.ndarray:
    return _compute_ptm(
        pae_logits,
        token_exists_mask,
        valid_frames_mask,
        token_exists_mask,
        bin_centers,
    )


def interface_ptm(
    pae_logits: jnp.ndarray,
    token_exists_mask: jnp.ndarray,
    valid_frames_mask: jnp.ndarray,
    bin_centers: jnp.ndarray,
    token_asym_id: jnp.ndarray,
) -> jnp.ndarray:
    chain_mask, _ = get_chain_masks_and_asyms(token_asym_id, token_exists_mask)
    scores = _compute_ptm(
        jnp.expand_dims(pae_logits, axis=-4),
        chain_mask,
        valid_frames_mask[..., None, :],
        ~chain_mask & token_exists_mask[..., None, :],
        bin_centers,
    )
    return jnp.max(scores, axis=-1)


def per_chain_ptm(
    pae_logits: jnp.ndarray,
    token_exists_mask: jnp.ndarray,
    valid_frames_mask: jnp.ndarray,
    bin_centers: jnp.ndarray,
    token_asym_id: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    chain_mask, asyms = get_chain_masks_and_asyms(token_asym_id, token_exists_mask)
    result = _compute_ptm(
        jnp.expand_dims(pae_logits, axis=-4),
        chain_mask,
        valid_frames_mask[..., None, :],
        chain_mask,
        bin_centers,
    )
    return result, asyms


def per_chain_pair_iptm(
    pae_logits: jnp.ndarray,
    token_exists_mask: jnp.ndarray,
    valid_frames_mask: jnp.ndarray,
    bin_centers: jnp.ndarray,
    token_asym_id: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    chain_mask, asyms = get_chain_masks_and_asyms(token_asym_id, token_exists_mask)
    query_mask = chain_mask[..., :, None, :]
    key_mask = chain_mask[..., None, :, :]
    result = _compute_ptm(
        pae_logits[..., None, None, :, :, :],
        query_mask,
        valid_frames_mask[..., None, None, :],
        key_mask,
        bin_centers,
    )
    return result, asyms


def get_scores(
    pae_logits: jnp.ndarray,
    token_exists_mask: jnp.ndarray,
    valid_frames_mask: jnp.ndarray,
    bin_centers: jnp.ndarray,
    token_asym_id: jnp.ndarray,
) -> PTMScores:
    pair, _ = per_chain_pair_iptm(
        pae_logits,
        token_exists_mask,
        valid_frames_mask,
        bin_centers,
        token_asym_id,
    )
    per_chain, _ = per_chain_ptm(
        pae_logits,
        token_exists_mask,
        valid_frames_mask,
        bin_centers,
        token_asym_id,
    )
    return PTMScores(
        complex_ptm=complex_ptm(
            pae_logits, token_exists_mask, valid_frames_mask, bin_centers
        ),
        interface_ptm=interface_ptm(
            pae_logits,
            token_exists_mask,
            valid_frames_mask,
            bin_centers,
            token_asym_id,
        ),
        per_chain_ptm=per_chain,
        per_chain_pair_iptm=pair,
    )
