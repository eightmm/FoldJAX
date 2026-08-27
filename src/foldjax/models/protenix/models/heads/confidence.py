"""Confidence head pieces for the Protenix JAX inference port."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.nn as jnn
import jax.numpy as jnp
import numpy as np

from foldjax.models._cp import shard_pair_rows
from foldjax.models.protenix.models.primitives.primitives import (
    LayerNormParams,
    LinearParams,
    layer_norm,
    linear,
)
from foldjax.models.protenix.models.trunk_blocks.pairformer import (
    PairformerStackParams,
    pairformer_stack,
)


class ConfidenceDistanceEmbeddingParams(NamedTuple):
    """Parameters for confidence-head pair distance embedding."""

    lower_bins: jnp.ndarray
    upper_bins: jnp.ndarray
    linear_d: LinearParams
    linear_d_wo_onehot: LinearParams


class ConfidenceOutputParams(NamedTuple):
    """Parameters for confidence-head output projections."""

    pae_ln: LayerNormParams
    pde_ln: LayerNormParams
    plddt_ln: LayerNormParams
    resolved_ln: LayerNormParams
    linear_pae: LinearParams
    linear_pde: LinearParams
    plddt_weight: jnp.ndarray
    resolved_weight: jnp.ndarray


class ConfidenceHeadParams(NamedTuple):
    """Parameters for the single-sample confidence head path."""

    input_strunk_ln: LayerNormParams
    linear_s1: LinearParams
    linear_s2: LinearParams
    distance_embedding: ConfidenceDistanceEmbeddingParams
    pairformer_stack: PairformerStackParams
    output: ConfidenceOutputParams


RDKIT_VDWS = jnp.asarray(
    [
        1.2,
        1.4,
        2.2,
        1.9,
        1.8,
        1.7,
        1.6,
        1.55,
        1.5,
        1.54,
        2.4,
        2.2,
        2.1,
        2.1,
        1.95,
        1.8,
        1.8,
        1.88,
        2.8,
        2.4,
        2.3,
        2.15,
        2.05,
        2.05,
        2.05,
        2.05,
        2.0,
        2.0,
        2.0,
        2.1,
        2.1,
        2.1,
        2.05,
        1.9,
        1.9,
        2.02,
        2.9,
        2.55,
        2.4,
        2.3,
        2.15,
        2.1,
        2.05,
        2.05,
        2.0,
        2.05,
        2.1,
        2.2,
        2.2,
        2.25,
        2.2,
        2.2,
        2.1,
        2.1,
        2.16,
        3.0,
        2.7,
        2.5,
        2.48,
        2.47,
        2.45,
        2.43,
        2.42,
        2.4,
        2.38,
        2.37,
        2.35,
        2.33,
        2.32,
        2.3,
        2.28,
        2.27,
        2.25,
        2.2,
        2.1,
        2.05,
        2.0,
        2.0,
        2.05,
        2.1,
        2.05,
        2.2,
        2.3,
        2.3,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.4,
        2.0,
        2.3,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
    ],
    dtype=jnp.float32,
)


def get_bin_centers(
    min_bin: float,
    max_bin: float,
    no_bins: int,
    *,
    dtype: jnp.dtype = jnp.float32,
) -> jnp.ndarray:
    """Return Protenix score bin centers."""

    bin_width = (max_bin - min_bin) / no_bins
    boundaries = jnp.linspace(
        min_bin,
        max_bin - bin_width,
        no_bins,
        dtype=dtype,
    )
    return boundaries + 0.5 * bin_width


def _contract_bins(pae_prob: jnp.ndarray, per_bin_weight: jnp.ndarray) -> jnp.ndarray:
    """Contract the trailing bin axis: ``sum(pae_prob * per_bin_weight, axis=-1)``.

    Written as a dot product rather than broadcast-multiply-then-sum because this
    runs **eagerly** -- this port jits individual primitives, not the graph, so
    nothing fuses the product away. The broadcast form materializes a temporary the
    full size of ``pae_prob`` only to reduce it to 1/n_bins of that. At 2030 tokens
    and 5 samples the temporary is 4.91 GiB, and ``calculate_chain_based_ptm`` calls
    into here once per chain and once per chain pair -- roughly 12 times for a
    4-chain target -- which is what OOM'd a finished 2030-token prediction.

    ``precision=HIGHEST`` is not optional: callers set
    ``jax_default_matmul_precision="default"``, under which XLA would run this f32
    dot in TF32 and quietly drop ~10 bits off every pTM/ipTM score.
    """
    return jnp.tensordot(
        pae_prob.astype(jnp.float32),
        per_bin_weight,
        axes=([-1], [0]),
        precision=jax.lax.Precision.HIGHEST,
    )


def logits_to_score(
    logits: jnp.ndarray,
    *,
    min_bin: float,
    max_bin: float,
    no_bins: int | None = None,
    return_prob: bool = False,
) -> jnp.ndarray | tuple[jnp.ndarray, jnp.ndarray]:
    """Convert binned logits to Protenix-style expected scores."""

    if no_bins is None:
        no_bins = int(logits.shape[-1])
    prob = jnn.softmax(logits.astype(jnp.float32), axis=-1)
    bin_centers = get_bin_centers(
        min_bin,
        max_bin,
        no_bins,
        dtype=prob.dtype,
    )
    score = prob @ bin_centers
    if return_prob:
        return score, prob
    return score


def calculate_normalization(n_token: int | jnp.ndarray) -> jnp.ndarray:
    """TM-score normalization constant used by Protenix."""

    n = jnp.asarray(n_token, dtype=jnp.float32)
    return 1.24 * (jnp.maximum(n, 19.0) - 15.0) ** (1.0 / 3.0) - 1.8


def calculate_ptm(
    pae_prob: jnp.ndarray,
    has_frame: jnp.ndarray,
    *,
    min_bin: float,
    max_bin: float,
    no_bins: int | None = None,
    token_mask: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Compute Protenix pTM with static-shape masking."""

    n_token = int(pae_prob.shape[-3])
    if no_bins is None:
        no_bins = int(pae_prob.shape[-1])
    if token_mask is None:
        token_mask = jnp.ones((n_token,), dtype=bool)
    else:
        token_mask = token_mask.astype(bool)
    has_frame = has_frame.astype(bool)
    valid_rows = token_mask & has_frame

    n_d = jnp.sum(token_mask.astype(jnp.float32))
    norm = calculate_normalization(n_d)
    centers = get_bin_centers(min_bin, max_bin, no_bins, dtype=jnp.float32)
    per_bin_weight = 1.0 / (1.0 + (centers / norm) ** 2)
    token_token_ptm = _contract_bins(pae_prob, per_bin_weight)

    col_mask = token_mask.astype(token_token_ptm.dtype)
    denom = jnp.maximum(jnp.sum(col_mask), 1.0)
    row_mean = jnp.sum(token_token_ptm * col_mask, axis=-1) / denom
    row_score = jnp.where(valid_rows, row_mean, -jnp.inf)
    score = jnp.max(row_score, axis=-1)
    return jnp.where(jnp.any(valid_rows), score, jnp.zeros_like(score))


def calculate_iptm(
    pae_prob: jnp.ndarray,
    has_frame: jnp.ndarray,
    asym_id: jnp.ndarray,
    *,
    min_bin: float,
    max_bin: float,
    no_bins: int | None = None,
    token_mask: jnp.ndarray | None = None,
    eps: float = 1e-8,
) -> jnp.ndarray:
    """Compute Protenix ipTM with static-shape masking."""

    n_token = int(pae_prob.shape[-3])
    if no_bins is None:
        no_bins = int(pae_prob.shape[-1])
    if token_mask is None:
        token_mask = jnp.ones((n_token,), dtype=bool)
    else:
        token_mask = token_mask.astype(bool)
    has_frame = has_frame.astype(bool)
    asym_id = asym_id.astype(jnp.int32)
    valid_rows = token_mask & has_frame

    n_d = jnp.sum(token_mask.astype(jnp.float32))
    norm = calculate_normalization(n_d)
    centers = get_bin_centers(min_bin, max_bin, no_bins, dtype=jnp.float32)
    per_bin_weight = 1.0 / (1.0 + (centers / norm) ** 2)
    token_token_ptm = _contract_bins(pae_prob, per_bin_weight)

    col_mask = token_mask.astype(token_token_ptm.dtype)
    is_diff_chain = (asym_id[None, :] != asym_id[:, None]).astype(token_token_ptm.dtype)
    denom = jnp.sum(is_diff_chain * col_mask[None, :], axis=-1)
    row_score = jnp.sum(token_token_ptm * is_diff_chain * col_mask, axis=-1) / (
        eps + denom
    )
    row_score = jnp.where(valid_rows, row_score, -jnp.inf)
    score = jnp.max(row_score, axis=-1)
    return jnp.where(jnp.any(valid_rows), score, jnp.zeros_like(score))


def calculate_chain_based_ptm(
    pae_prob: jnp.ndarray,
    has_frame: jnp.ndarray,
    asym_id: jnp.ndarray,
    token_is_ligand: jnp.ndarray,
    *,
    min_bin: float,
    max_bin: float,
    no_bins: int | None = None,
    n_chain: int | None = None,
    token_mask: jnp.ndarray | None = None,
) -> dict[str, jnp.ndarray]:
    """Compute the complete chain pTM/ipTM summaries used by Protenix."""

    if no_bins is None:
        no_bins = int(pae_prob.shape[-1])
    if token_mask is not None:
        token_mask = jnp.asarray(token_mask).astype(bool)
        if token_mask.shape != asym_id.shape:
            raise ValueError("token_mask must match the chain pTM token axis")
    if n_chain is None:
        chain_ids = (
            asym_id if token_mask is None else jnp.where(token_mask, asym_id, -1)
        )
        n_chain = int(jnp.max(chain_ids)) + 1
    asym_id = asym_id.astype(jnp.int32)
    has_frame = has_frame.astype(bool)
    token_is_ligand = token_is_ligand.astype(bool)
    batch_shape = pae_prob.shape[:-3]

    pair_rows: list[list[jnp.ndarray]] = []
    for aid_1 in range(n_chain):
        row = []
        for aid_2 in range(n_chain):
            if aid_1 == aid_2:
                value = jnp.zeros(batch_shape, dtype=jnp.float32)
            elif aid_2 < aid_1:
                value = pair_rows[aid_2][aid_1]
            else:
                pair_mask = (asym_id == aid_1) | (asym_id == aid_2)
                if token_mask is not None:
                    pair_mask = pair_mask & token_mask
                value = calculate_iptm(
                    pae_prob,
                    has_frame,
                    asym_id,
                    min_bin=min_bin,
                    max_bin=max_bin,
                    no_bins=no_bins,
                    token_mask=pair_mask,
                )
            row.append(value)
        pair_rows.append(row)
    chain_pair_iptm = jnp.stack(
        [jnp.stack(row, axis=-1) for row in pair_rows], axis=-2
    )

    chain_ptm = jnp.stack(
        [
            calculate_ptm(
                pae_prob,
                has_frame,
                min_bin=min_bin,
                max_bin=max_bin,
                no_bins=no_bins,
                token_mask=(
                    asym_id == aid
                    if token_mask is None
                    else (asym_id == aid) & token_mask
                ),
            )
            for aid in range(n_chain)
        ],
        axis=-1,
    )

    chain_has_frame = [
        jnp.any(
            (asym_id == aid) & has_frame
            if token_mask is None
            else (asym_id == aid) & has_frame & token_mask
        )
        for aid in range(n_chain)
    ]
    chain_iptm_values = []
    for aid in range(n_chain):
        values = []
        for i in range(n_chain):
            for j in range(n_chain):
                if (i == aid or j == aid) and i != j:
                    values.append(
                        jnp.where(
                            chain_has_frame[i],
                            chain_pair_iptm[..., i, j],
                            jnp.zeros(batch_shape, dtype=jnp.float32),
                        )
                    )
        if values:
            valid = jnp.stack(
                [
                    jnp.broadcast_to(chain_has_frame[i], batch_shape)
                    for i in range(n_chain)
                    for j in range(n_chain)
                    if (i == aid or j == aid) and i != j
                ],
                axis=-1,
            )
            stacked = jnp.stack(values, axis=-1)
            chain_iptm_values.append(
                jnp.sum(stacked, axis=-1)
                / jnp.maximum(jnp.sum(valid, axis=-1), 1)
            )
        else:
            chain_iptm_values.append(jnp.zeros(batch_shape, dtype=jnp.float32))
    chain_iptm = jnp.stack(chain_iptm_values, axis=-1)

    chain_is_ligand = []
    for aid in range(n_chain):
        chain_mask = asym_id == aid
        if token_mask is not None:
            chain_mask = chain_mask & token_mask
        chain_is_ligand.append(
            jnp.sum(token_is_ligand & chain_mask)
            >= jnp.sum(chain_mask.astype(jnp.int32)) // 2
        )
    global_rows = []
    for aid_1 in range(n_chain):
        row = []
        for aid_2 in range(n_chain):
            if aid_1 == aid_2:
                value = jnp.zeros(batch_shape, dtype=jnp.float32)
            else:
                # Whether a chain is a ligand is a *value*, so the choice is
                # selected rather than branched on: a Python `if` here reads a
                # tracer once the whole graph is traced. Same result, same
                # precedence -- chain 1 first, then chain 2, then the mean.
                value = jnp.where(
                    chain_is_ligand[aid_1],
                    chain_iptm[..., aid_1],
                    jnp.where(
                        chain_is_ligand[aid_2],
                        chain_iptm[..., aid_2],
                        0.5 * (chain_iptm[..., aid_1] + chain_iptm[..., aid_2]),
                    ),
                )
            row.append(value)
        global_rows.append(jnp.stack(row, axis=-1))
    chain_pair_iptm_global = jnp.stack(global_rows, axis=-2)
    return {
        "chain_ptm": chain_ptm.astype(jnp.float32),
        "chain_iptm": chain_iptm.astype(jnp.float32),
        "chain_pair_iptm": chain_pair_iptm.astype(jnp.float32),
        "chain_pair_iptm_global": chain_pair_iptm_global.astype(jnp.float32),
    }


def calculate_chain_based_plddt(
    atom_plddt: jnp.ndarray,
    asym_id: jnp.ndarray,
    atom_to_token_idx: jnp.ndarray,
    *,
    n_chain: int | None = None,
    atom_mask: jnp.ndarray | None = None,
) -> dict[str, jnp.ndarray]:
    """Compute Protenix chain pLDDT summaries."""

    if n_chain is None:
        n_chain = int(jnp.max(asym_id)) + 1
    atom_chain_id = asym_id.astype(jnp.int32)[atom_to_token_idx]
    if atom_mask is not None:
        atom_mask = jnp.asarray(atom_mask).astype(bool)
        if atom_mask.shape != atom_chain_id.shape:
            raise ValueError("atom_mask must match the pLDDT atom axis")
        atom_chain_id = jnp.where(atom_mask, atom_chain_id, -1)

    chain_vals = []
    for aid in range(n_chain):
        atom_mask = (atom_chain_id == aid).astype(atom_plddt.dtype)
        denom = jnp.maximum(jnp.sum(atom_mask), 1.0)
        chain_vals.append(jnp.sum(atom_plddt * atom_mask, axis=-1) / denom)
    chain_plddt = jnp.stack(chain_vals, axis=-1)

    pair_rows = []
    for aid_1 in range(n_chain):
        pair_cols = []
        for aid_2 in range(n_chain):
            if aid_1 == aid_2:
                pair_cols.append(jnp.zeros(atom_plddt.shape[:-1], dtype=jnp.float32))
            else:
                atom_mask = (
                    (atom_chain_id == aid_1) | (atom_chain_id == aid_2)
                ).astype(atom_plddt.dtype)
                denom = jnp.maximum(jnp.sum(atom_mask), 1.0)
                pair_cols.append(jnp.sum(atom_plddt * atom_mask, axis=-1) / denom)
        pair_rows.append(jnp.stack(pair_cols, axis=-1))
    chain_pair_plddt = jnp.stack(pair_rows, axis=-2)
    return {
        "chain_plddt": chain_plddt.astype(jnp.float32),
        "chain_pair_plddt": chain_pair_plddt.astype(jnp.float32),
    }


def calculate_chain_based_gpde(
    token_pair_pde: jnp.ndarray,
    contact_probs: jnp.ndarray,
    asym_id: jnp.ndarray,
    *,
    n_chain: int | None = None,
    eps: float = 1e-8,
    token_mask: jnp.ndarray | None = None,
) -> dict[str, jnp.ndarray]:
    """Compute Protenix chain and chain-pair gPDE summaries."""

    if token_mask is not None:
        token_mask = jnp.asarray(token_mask).astype(bool)
        if token_mask.shape != asym_id.shape:
            raise ValueError("token_mask must match the chain gPDE token axis")
    if n_chain is None:
        chain_ids = (
            asym_id if token_mask is None else jnp.where(token_mask, asym_id, -1)
        )
        n_chain = int(jnp.max(chain_ids)) + 1
    asym_id = asym_id.astype(jnp.int32)
    contact_probs = contact_probs.astype(token_pair_pde.dtype)

    def _weighted_mean(mask_1, mask_2):
        if token_mask is not None:
            mask_1 = mask_1 & token_mask
            mask_2 = mask_2 & token_mask
        pair_mask = (mask_1[:, None] & mask_2[None, :]).astype(token_pair_pde.dtype)
        weights = contact_probs * pair_mask
        return jnp.sum(token_pair_pde * weights, axis=(-1, -2)) / (
            jnp.sum(weights, axis=(-1, -2)) + eps
        )

    chain_vals = []
    for aid in range(n_chain):
        mask = asym_id == aid
        chain_vals.append(_weighted_mean(mask, mask))
    chain_gpde = jnp.stack(chain_vals, axis=-1)

    pair_rows = []
    for aid_1 in range(n_chain):
        pair_cols = []
        for aid_2 in range(n_chain):
            if aid_1 == aid_2:
                pair_cols.append(
                    jnp.zeros(token_pair_pde.shape[:-2], dtype=jnp.float32)
                )
            elif aid_2 < aid_1:
                pair_cols.append(pair_rows[aid_2][..., aid_1])
            else:
                pair_cols.append(_weighted_mean(asym_id == aid_1, asym_id == aid_2))
        pair_rows.append(jnp.stack(pair_cols, axis=-1))
    chain_pair_gpde = jnp.stack(pair_rows, axis=-2)
    return {
        "chain_gpde": chain_gpde.astype(jnp.float32),
        "chain_pair_gpde": chain_pair_gpde.astype(jnp.float32),
    }


def calculate_chain_pair_pae(
    token_pair_pae: jnp.ndarray,
    asym_id: jnp.ndarray,
    token_has_frame: jnp.ndarray,
    contact_probs: jnp.ndarray | None = None,
    *,
    n_chain: int | None = None,
    eps: float = 1e-8,
    token_mask: jnp.ndarray | None = None,
) -> dict[str, jnp.ndarray]:
    """Compute Protenix chain-pair PAE mean and minimum summaries."""

    if token_mask is not None:
        token_mask = jnp.asarray(token_mask).astype(bool)
        if token_mask.shape != asym_id.shape:
            raise ValueError("token_mask must match the chain PAE token axis")
    if n_chain is None:
        chain_ids = (
            asym_id if token_mask is None else jnp.where(token_mask, asym_id, -1)
        )
        n_chain = int(jnp.max(chain_ids)) + 1
    asym_id = asym_id.astype(jnp.int32)
    token_has_frame = token_has_frame.astype(bool)
    if contact_probs is None:
        contact_probs = jnp.ones(token_pair_pae.shape[-2:], dtype=token_pair_pae.dtype)
    else:
        contact_probs = contact_probs.astype(token_pair_pae.dtype)

    frame_mask = token_has_frame[:, None] & token_has_frame[None, :]
    if token_mask is not None:
        frame_mask = frame_mask & token_mask[:, None] & token_mask[None, :]
    mean_rows = []
    min_rows = []
    for aid_1 in range(n_chain):
        mean_cols = []
        min_cols = []
        for aid_2 in range(n_chain):
            pair_mask = (
                (asym_id[:, None] == aid_1)
                & (asym_id[None, :] == aid_2)
                & frame_mask
            )
            pair_mask_f = pair_mask.astype(token_pair_pae.dtype)
            weights = contact_probs * pair_mask_f
            weight_sum = jnp.sum(weights, axis=(-1, -2))
            mean_cols.append(
                jnp.where(
                    weight_sum > 0,
                    jnp.sum(token_pair_pae * weights, axis=(-1, -2))
                    / (weight_sum + eps),
                    jnp.nan,
                )
            )
            min_cols.append(
                jnp.min(
                    jnp.where(pair_mask, token_pair_pae, jnp.inf),
                    axis=(-1, -2),
                )
            )
        mean_rows.append(jnp.stack(mean_cols, axis=-1))
        min_rows.append(jnp.stack(min_cols, axis=-1))
    chain_pair_pae_mean = jnp.stack(mean_rows, axis=-2)
    chain_pair_pae_min = jnp.stack(min_rows, axis=-2)
    chain_pair_pae_min = jnp.where(
        jnp.isfinite(chain_pair_pae_min),
        chain_pair_pae_min,
        jnp.nan,
    )
    return {
        "chain_pair_pae_mean": chain_pair_pae_mean.astype(jnp.float32),
        "chain_pair_pae_min": chain_pair_pae_min.astype(jnp.float32),
    }


#: Atom rows per block in the clash scans. The pairwise distance matrix is
#: ``[samples, n_atom, n_atom]``: at 5 samples and the 16134 atoms of a 2030-token
#: target that is 5.2 TB, so it cannot be built, and an unchunked version dies during
#: autotuning on a ``f32[5, 16134, 16134]`` fusion. Both clash reductions partition
#: exactly over rows, so a block at a time is the same answer -- 2048 rows is ~165 MB.
CLASH_ROW_CHUNK = 2048


def _chain_pairs(n_chain: int) -> tuple[tuple[int, int], ...]:
    """Ordered ``(first, second)`` chain pairs, first < second."""
    return tuple(
        (first, second)
        for first in range(n_chain)
        for second in range(first + 1, n_chain)
    )


def _row_blocks(
    atom_coordinate: jnp.ndarray, atom_chain_id: jnp.ndarray, chunk: int
) -> tuple[jnp.ndarray, jnp.ndarray, int]:
    """Pad the atom axis to whole blocks, and return the block count.

    Padded rows are given a chain id of ``-1``, which no real chain has, so they
    match no chain pair and contribute nothing to either reduction. Only the *row*
    axis is padded; columns keep the original coordinates.
    """
    n_atom = atom_coordinate.shape[-2]
    blocks = -(-n_atom // chunk)
    pad = blocks * chunk - n_atom
    if pad == 0:
        return atom_coordinate, atom_chain_id, blocks
    width = [(0, 0)] * atom_coordinate.ndim
    width[-2] = (0, pad)
    return (
        jnp.pad(atom_coordinate, width),
        jnp.pad(atom_chain_id, (0, pad), constant_values=-1),
        blocks,
    )


def calculate_clash(
    atom_coordinate: jnp.ndarray,
    asym_id: jnp.ndarray,
    atom_to_token_idx: jnp.ndarray,
    *,
    atom_is_polymer: jnp.ndarray | None = None,
    threshold: float = 1.1,
    n_chain: int | None = None,
    row_chunk_size: int = CLASH_ROW_CHUNK,
    atom_mask: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Compute a Protenix AF3-style inter-chain clash flag per sample.

    This covers the AF3 polymer-style clash penalty used in ranking. VDW
    ligand/polymer clash requires element radii and remains separate.

    Counted a block of atom rows at a time; see :data:`CLASH_ROW_CHUNK`. Counting is
    a sum, so partitioning the rows gives the same total.
    """

    if n_chain is None:
        n_chain = int(jnp.max(asym_id)) + 1
    atom_chain_id = asym_id.astype(jnp.int32)[atom_to_token_idx]
    if atom_mask is not None:
        atom_mask = jnp.asarray(atom_mask).astype(bool)
        if atom_mask.shape != atom_chain_id.shape:
            raise ValueError("atom_mask must match the clash atom axis")
        atom_chain_id = jnp.where(atom_mask, atom_chain_id, -1)
    if atom_is_polymer is None:
        chain_is_polymer = jnp.ones((n_chain,), dtype=bool)
    else:
        atom_is_polymer = jnp.asarray(atom_is_polymer, dtype=bool)
        if atom_is_polymer.shape != atom_chain_id.shape:
            raise ValueError(
                "atom_is_polymer must match the atom axis, got "
                f"{atom_is_polymer.shape} for {atom_chain_id.shape}"
            )
        chain_is_polymer = jnp.stack(
            [
                jnp.any(atom_is_polymer & (atom_chain_id == aid))
                for aid in range(n_chain)
            ]
        )
    pairs = _chain_pairs(n_chain)
    if not pairs:
        return jnp.zeros(atom_coordinate.shape[:-2], dtype=bool)

    sizes = jnp.stack(
        [
            jnp.sum((atom_chain_id == aid).astype(jnp.float32))
            for aid in range(n_chain)
        ]
    )
    rows_all, chain_all, blocks = _row_blocks(
        atom_coordinate, atom_chain_id, row_chunk_size
    )
    # Squared comparison: sqrt is monotone on non-negative values, so this is the
    # same test without a second pass over a [..., chunk, n_atom] array.
    squared_threshold = threshold * threshold

    def body(index, totals):
        start = index * row_chunk_size
        rows = jax.lax.dynamic_slice_in_dim(rows_all, start, row_chunk_size, axis=-2)
        row_chain = jax.lax.dynamic_slice_in_dim(
            chain_all, start, row_chunk_size, axis=0
        )
        diff = rows[..., :, None, :] - atom_coordinate[..., None, :, :]
        close = jnp.sum(jnp.square(diff), axis=-1) < squared_threshold
        return tuple(
            total
            + jnp.sum(
                (
                    close
                    & (row_chain[:, None] == first)
                    & (atom_chain_id[None, :] == second)
                ).astype(jnp.float32),
                axis=(-1, -2),
            )
            for total, (first, second) in zip(totals, pairs, strict=True)
        )

    sample_shape = atom_coordinate.shape[:-2]
    totals = jax.lax.fori_loop(
        0,
        blocks,
        body,
        tuple(jnp.zeros(sample_shape, dtype=jnp.float32) for _ in pairs),
    )

    flags = [
        chain_is_polymer[first]
        & chain_is_polymer[second]
        & (
            (total > 100.0)
            | (
                total
                / jnp.maximum(jnp.minimum(sizes[first], sizes[second]), 1.0)
                > 0.5
            )
        )
        for total, (first, second) in zip(totals, pairs, strict=True)
    ]
    return jnp.any(jnp.stack(flags, axis=-1), axis=-1)


def calculate_vdw_clash(
    atom_coordinate: jnp.ndarray,
    asym_id: jnp.ndarray,
    atom_to_token_idx: jnp.ndarray,
    elements_one_hot: jnp.ndarray,
    *,
    mol_id: jnp.ndarray | None = None,
    threshold: float = 0.75,
    n_chain: int | None = None,
    row_chunk_size: int = CLASH_ROW_CHUNK,
    atom_mask: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Compute inter-chain VDW clash flags using Protenix/RDKit radii.

    Scanned a block of atom rows at a time, like :func:`calculate_clash`; ``any`` is
    associative over the row partition, so the flags are unchanged.
    """

    if n_chain is None:
        n_chain = int(jnp.max(asym_id)) + 1
    atom_chain_id = asym_id.astype(jnp.int32)[atom_to_token_idx]
    if atom_mask is not None:
        atom_mask = jnp.asarray(atom_mask).astype(bool)
        if atom_mask.shape != atom_chain_id.shape:
            raise ValueError("atom_mask must match the VDW clash atom axis")
        atom_chain_id = jnp.where(atom_mask, atom_chain_id, -1)
    element_order = jnp.argmax(elements_one_hot, axis=-1)
    radii = RDKIT_VDWS[element_order]
    pairs = _chain_pairs(n_chain)
    if not pairs:
        return jnp.zeros(atom_coordinate.shape[:-2], dtype=bool)

    # Whether a pair is skipped depends only on chain membership, so it is decided
    # once rather than inside the row scan.
    skip = []
    for first, second in pairs:
        if mol_id is None:
            skip.append(jnp.asarray(False))
            continue
        mol_first = jnp.max(jnp.where(atom_chain_id == first, mol_id, -1))
        mol_second = jnp.max(jnp.where(atom_chain_id == second, mol_id, -2))
        skip.append(mol_first == mol_second)

    rows_all, chain_all, blocks = _row_blocks(
        atom_coordinate, atom_chain_id, row_chunk_size
    )
    radii_all = (
        radii
        if rows_all.shape[-2] == radii.shape[0]
        else jnp.pad(radii, (0, rows_all.shape[-2] - radii.shape[0]))
    )

    def body(index, found):
        start = index * row_chunk_size
        rows = jax.lax.dynamic_slice_in_dim(rows_all, start, row_chunk_size, axis=-2)
        row_chain = jax.lax.dynamic_slice_in_dim(
            chain_all, start, row_chunk_size, axis=0
        )
        row_radii = jax.lax.dynamic_slice_in_dim(
            radii_all, start, row_chunk_size, axis=0
        )
        # ``dist / vdw_sum < threshold`` is ``dist**2 < (threshold * vdw_sum)**2``
        # for non-negative operands, which avoids the sqrt over the whole block.
        limit = threshold * jnp.maximum(row_radii[:, None] + radii[None, :], 1e-8)
        diff = rows[..., :, None, :] - atom_coordinate[..., None, :, :]
        close = jnp.sum(jnp.square(diff), axis=-1) < jnp.square(limit)
        return tuple(
            seen
            | jnp.any(
                close
                & (row_chain[:, None] == first)
                & (atom_chain_id[None, :] == second),
                axis=(-1, -2),
            )
            for seen, (first, second) in zip(found, pairs, strict=True)
        )

    sample_shape = atom_coordinate.shape[:-2]
    found = jax.lax.fori_loop(
        0,
        blocks,
        body,
        tuple(jnp.zeros(sample_shape, dtype=bool) for _ in pairs),
    )
    flags = [
        jnp.where(skipped, jnp.zeros_like(seen), seen)
        for seen, skipped in zip(found, skip, strict=True)
    ]
    return jnp.any(jnp.stack(flags, axis=-1), axis=-1)


def compute_contact_prob(
    distogram_logits: jnp.ndarray,
    *,
    min_bin: float = 2.3125,
    max_bin: float = 21.6875,
    no_bins: int | None = None,
    thres: float = 8.0,
) -> jnp.ndarray:
    """Compute Protenix contact probabilities from distogram logits."""

    if no_bins is None:
        no_bins = int(distogram_logits.shape[-1])
    prob = jnn.softmax(distogram_logits.astype(jnp.float32), axis=-1)
    centers = get_bin_centers(min_bin, max_bin, no_bins, dtype=prob.dtype)
    contact_mask = centers < thres
    return jnp.sum(prob * contact_mask.astype(prob.dtype), axis=-1)


def confidence_scores_from_logits(
    *,
    plddt_logits: jnp.ndarray,
    pae_logits: jnp.ndarray,
    pde_logits: jnp.ndarray,
    distogram_logits: jnp.ndarray,
    plddt_min_bin: float = 0.0,
    plddt_max_bin: float = 1.0,
    plddt_no_bins: int | None = None,
    pae_min_bin: float = 0.0,
    pae_max_bin: float = 32.0,
    pae_no_bins: int | None = None,
    pde_min_bin: float = 0.0,
    pde_max_bin: float = 32.0,
    pde_no_bins: int | None = None,
    distogram_min_bin: float = 2.3125,
    distogram_max_bin: float = 21.6875,
    distogram_no_bins: int | None = None,
    contact_threshold: float = 8.0,
    token_has_frame: jnp.ndarray | None = None,
    token_asym_id: jnp.ndarray | None = None,
    atom_to_token_idx: jnp.ndarray | None = None,
    atom_coordinate: jnp.ndarray | None = None,
    atom_is_polymer: jnp.ndarray | None = None,
    elements_one_hot: jnp.ndarray | None = None,
    mol_id: jnp.ndarray | None = None,
    clash_threshold: float = 1.1,
    vdw_clash_threshold: float = 0.75,
    token_mask: jnp.ndarray | None = None,
    atom_mask: jnp.ndarray | None = None,
    token_is_ligand: jnp.ndarray | None = None,
    num_recycles: int | None = None,
    n_chain: int | None = None,
    include_chain_pair_pae: bool = True,
) -> dict[str, jnp.ndarray]:
    """Compute the basic full-data confidence scores used in inference."""

    atom_plddt = logits_to_score(
        plddt_logits,
        min_bin=plddt_min_bin,
        max_bin=plddt_max_bin,
        no_bins=plddt_no_bins,
    )
    token_pair_pde = logits_to_score(
        pde_logits,
        min_bin=pde_min_bin,
        max_bin=pde_max_bin,
        no_bins=pde_no_bins,
    )
    token_pair_pae, pae_prob = logits_to_score(
        pae_logits,
        min_bin=pae_min_bin,
        max_bin=pae_max_bin,
        no_bins=pae_no_bins,
        return_prob=True,
    )
    contact_probs = compute_contact_prob(
        distogram_logits,
        min_bin=distogram_min_bin,
        max_bin=distogram_max_bin,
        no_bins=distogram_no_bins,
        thres=contact_threshold,
    )
    if atom_mask is None:
        summary_plddt = jnp.mean(atom_plddt, axis=-1) * 100.0
    else:
        atom_mask_f = jnp.asarray(atom_mask, dtype=atom_plddt.dtype)
        if atom_mask_f.shape != atom_plddt.shape[-1:]:
            raise ValueError("atom_mask must match the pLDDT atom axis")
        summary_plddt = (
            jnp.sum(atom_plddt * atom_mask_f, axis=-1)
            / jnp.maximum(jnp.sum(atom_mask_f), 1.0)
            * 100.0
        )
    if token_mask is None:
        gpde_weights = contact_probs
    else:
        token_mask_bool = jnp.asarray(token_mask).astype(bool)
        if token_mask_bool.shape != contact_probs.shape[-2:-1]:
            raise ValueError("token_mask must match the confidence token axis")
        score_pair_mask = (token_mask_bool[:, None] & token_mask_bool[None, :]).astype(
            contact_probs.dtype
        )
        gpde_weights = contact_probs * score_pair_mask
    gpde_numer = jnp.sum(token_pair_pde * gpde_weights, axis=(-1, -2))
    gpde_denom = jnp.sum(gpde_weights, axis=(-1, -2))
    summary_gpde = jnp.where(gpde_denom > 0, gpde_numer / gpde_denom, 0.0)
    scores = {
        "atom_plddt": atom_plddt.astype(jnp.float32),
        "token_pair_pde": token_pair_pde.astype(jnp.float32),
        "token_pair_pae": token_pair_pae.astype(jnp.float32),
        "contact_probs": contact_probs.astype(jnp.float32),
        "summary_plddt": summary_plddt.astype(jnp.float32),
        "summary_gpde": summary_gpde.astype(jnp.float32),
    }
    if token_has_frame is not None and token_asym_id is not None:
        summary_ptm = calculate_ptm(
            pae_prob,
            token_has_frame,
            min_bin=pae_min_bin,
            max_bin=pae_max_bin,
            no_bins=pae_no_bins,
            token_mask=token_mask,
        )
        summary_iptm = calculate_iptm(
            pae_prob,
            token_has_frame,
            token_asym_id,
            min_bin=pae_min_bin,
            max_bin=pae_max_bin,
            no_bins=pae_no_bins,
            token_mask=token_mask,
        )
        ranking_score = 0.8 * summary_iptm + 0.2 * summary_ptm
        if atom_to_token_idx is not None and atom_coordinate is not None:
            has_clash = calculate_clash(
                atom_coordinate,
                token_asym_id,
                atom_to_token_idx,
                atom_is_polymer=atom_is_polymer,
                threshold=clash_threshold,
                n_chain=n_chain,
                atom_mask=atom_mask,
            )
            ranking_score = ranking_score - 100.0 * has_clash.astype(
                ranking_score.dtype
            )
            scores["has_clash"] = has_clash
            if elements_one_hot is not None:
                has_vdw_clash = calculate_vdw_clash(
                    atom_coordinate,
                    token_asym_id,
                    atom_to_token_idx,
                    elements_one_hot,
                    mol_id=mol_id,
                    threshold=vdw_clash_threshold,
                    n_chain=n_chain,
                    atom_mask=atom_mask,
                )
                scores["has_vdw_clash"] = has_vdw_clash
                scores["summary_ranking_score_vdw_penalized"] = (
                    ranking_score
                    - 100.0 * has_vdw_clash.astype(ranking_score.dtype)
                ).astype(jnp.float32)
        scores.update(
            {
                "summary_ptm": summary_ptm.astype(jnp.float32),
                "summary_iptm": summary_iptm.astype(jnp.float32),
                "summary_ranking_score": ranking_score.astype(jnp.float32),
            }
        )
        scores.update(
            calculate_chain_based_ptm(
                pae_prob,
                token_has_frame,
                token_asym_id,
                (
                    jnp.zeros_like(token_asym_id, dtype=bool)
                    if token_is_ligand is None
                    else token_is_ligand
                ),
                min_bin=pae_min_bin,
                max_bin=pae_max_bin,
                no_bins=pae_no_bins,
                n_chain=n_chain,
                token_mask=token_mask,
            )
        )
        scores.update(
            calculate_chain_based_gpde(
                token_pair_pde,
                contact_probs,
                token_asym_id,
                n_chain=n_chain,
                token_mask=token_mask,
            )
        )
        if include_chain_pair_pae:
            scores.update(
                calculate_chain_pair_pae(
                    token_pair_pae,
                    token_asym_id,
                    token_has_frame,
                    contact_probs,
                    n_chain=n_chain,
                    token_mask=token_mask,
                )
            )
        if atom_to_token_idx is not None:
            scores.update(
                calculate_chain_based_plddt(
                    atom_plddt,
                    token_asym_id,
                    atom_to_token_idx,
                    n_chain=n_chain,
                    atom_mask=atom_mask,
                )
            )
        scores["disorder"] = jnp.zeros_like(summary_ptm, dtype=jnp.float32)
        scores["ranking_score"] = scores["summary_ranking_score"]
        if num_recycles is not None:
            scores["num_recycles"] = jnp.asarray(num_recycles, dtype=jnp.int32)
    return scores


def confidence_one_hot(
    x: jnp.ndarray,
    lower_bins: jnp.ndarray,
    upper_bins: jnp.ndarray,
) -> jnp.ndarray:
    """Open-interval distance binning matching Protenix ``one_hot``."""

    return ((x[..., None] > lower_bins) & (x[..., None] < upper_bins)).astype(x.dtype)


def can_compact_confidence_distance_embedding(
    params: ConfidenceDistanceEmbeddingParams | None,
) -> bool:
    """Return whether indexed bin projection is exact for concrete parameters.

    This is deliberately a host-only predicate.  Released Protenix and OpenDDE
    checkpoints use finite, adjacent open intervals, for which a distance can
    select at most one column of ``linear_d``.  Custom parameters may overlap,
    leave gaps, contain non-finite values, or rely on dense-dot IEEE behaviour;
    those stay on the historical dense path.
    """

    if params is None:
        return False
    try:
        arrays = (
            params.lower_bins,
            params.upper_bins,
            params.linear_d.weight,
        )
        if any(isinstance(value, jax.core.Tracer) for value in arrays):
            return False
        lower, upper, weight = (
            np.asarray(jax.device_get(value)) for value in arrays
        )
        bias = params.linear_d.bias
        if isinstance(bias, jax.core.Tracer):
            return False
        bias_host = None if bias is None else np.asarray(jax.device_get(bias))
    except (AttributeError, TypeError, ValueError):
        return False

    def _floating(value: np.ndarray) -> bool:
        try:
            return bool(jnp.issubdtype(jnp.dtype(value.dtype), jnp.floating))
        except TypeError:
            return False

    if (
        lower.ndim != 1
        or upper.ndim != 1
        or lower.shape != upper.shape
        or lower.size <= 1
        or lower.dtype != upper.dtype
        or not _floating(lower)
        or not np.isfinite(lower).all()
        or not np.isfinite(upper).all()
        or not np.all(lower[1:] > lower[:-1])
        or not np.all(upper > lower)
        or not np.array_equal(upper[:-1], lower[1:])
    ):
        return False
    if (
        weight.ndim != 2
        or weight.shape[1] != lower.size
        or not _floating(weight)
        or not np.isfinite(weight).all()
    ):
        return False
    if bias_host is not None and (
        bias_host.ndim != 1
        or bias_host.shape[0] != weight.shape[0]
        or not _floating(bias_host)
        or not np.isfinite(bias_host).all()
    ):
        return False
    return True


def _compact_confidence_bin_projection(
    distance: jnp.ndarray,
    params: ConfidenceDistanceEmbeddingParams,
) -> jnp.ndarray:
    """Project one selected open-interval bin without an ``N x N x B`` tensor."""

    lower = params.lower_bins
    upper = params.upper_bins
    index = jnp.searchsorted(lower, distance, side="left") - 1
    safe_index = jnp.clip(index, 0, lower.shape[0] - 1)
    valid = (
        (index >= 0)
        & (distance > lower[safe_index])
        & (distance < upper[safe_index])
    )

    # ``linear`` stores weights as [out, in].  A dense dot over a one-hot row
    # selects this same [out] row, while its multi-term reduction canonicalises
    # a selected -0 to +0.  Preserve that last detail explicitly.
    dtype = jnp.result_type(distance.dtype, params.linear_d.weight.dtype)
    table = jnp.swapaxes(params.linear_d.weight, -1, -2).astype(dtype)
    selected = table[safe_index]
    positive_zero = jnp.asarray(0.0, dtype=dtype)
    selected = jnp.where(selected == 0, positive_zero, selected)
    projected = jnp.where(valid[..., None], selected, positive_zero)
    if params.linear_d.bias is not None:
        projected = projected + params.linear_d.bias
    return projected


def confidence_distance_embedding(
    x_pred_rep_coords: jnp.ndarray,
    params: ConfidenceDistanceEmbeddingParams,
    *,
    compact_bins: bool = False,
) -> jnp.ndarray:
    """Embed representative-atom pair distances for ConfidenceHead.

    ``compact_bins=False`` preserves the public/direct dense implementation.
    Production compiled wrappers enable the compact path only after validating
    the concrete checkpoint with :func:`can_compact_confidence_distance_embedding`.
    """

    coords = x_pred_rep_coords.astype(jnp.float32)
    diff = coords[..., :, None, :] - coords[..., None, :, :]
    distance = jnp.sqrt(jnp.sum(jnp.square(diff), axis=-1))
    if compact_bins:
        binned = _compact_confidence_bin_projection(distance, params)
    else:
        binned = linear(
            confidence_one_hot(distance, params.lower_bins, params.upper_bins),
            params.linear_d,
        )
    return binned + linear(distance[..., None], params.linear_d_wo_onehot)


def confidence_output_logits(
    s_single: jnp.ndarray,
    z_pair: jnp.ndarray,
    atom_to_token_idx: jnp.ndarray,
    atom_to_tokatom_idx: jnp.ndarray,
    params: ConfidenceOutputParams,
) -> dict[str, jnp.ndarray]:
    """Project pair and atom confidence logits."""

    pae = linear(layer_norm(z_pair, params.pae_ln), params.linear_pae)
    pde = linear(
        layer_norm(z_pair + jnp.swapaxes(z_pair, -2, -3), params.pde_ln),
        params.linear_pde,
    )
    atom_single = s_single[..., atom_to_token_idx, :]
    plddt_weight = params.plddt_weight[atom_to_tokatom_idx]
    resolved_weight = params.resolved_weight[atom_to_tokatom_idx]
    plddt = jnp.einsum(
        "...nc,ncb->...nb",
        layer_norm(atom_single, params.plddt_ln),
        plddt_weight,
    )
    resolved = jnp.einsum(
        "...nc,ncb->...nb",
        layer_norm(atom_single, params.resolved_ln),
        resolved_weight,
    )
    return {
        "plddt": plddt.astype(jnp.float32),
        "pae": pae.astype(jnp.float32),
        "pde": pde.astype(jnp.float32),
        "resolved": resolved.astype(jnp.float32),
    }


def confidence_head_single_sample(
    s_inputs: jnp.ndarray,
    s_trunk: jnp.ndarray,
    z_trunk: jnp.ndarray,
    pair_mask: jnp.ndarray | None,
    x_pred_rep_coords: jnp.ndarray,
    atom_to_token_idx: jnp.ndarray,
    atom_to_tokatom_idx: jnp.ndarray,
    params: ConfidenceHeadParams,
    *,
    use_embedding: bool = True,
    compact_distance_bins: bool = False,
    use_scan: bool = True,
    triangle_mul_chunk_size: int | None = None,
    triangle_att_q_chunk_size: int | None = None,
    single_att_q_chunk_size: int | None = None,
    triangle_attention_backend: str | None = None,
) -> dict[str, jnp.ndarray]:
    """Run the ConfidenceHead inference path for one predicted sample."""

    s_trunk = layer_norm(jnp.clip(s_trunk, -512.0, 512.0), params.input_strunk_ln)
    z_base = z_trunk if use_embedding else jnp.zeros_like(z_trunk)
    z_init = linear(s_inputs, params.linear_s1)[..., :, None, :] + linear(
        s_inputs,
        params.linear_s2,
    )[..., None, :, :]
    # Born sharded under context parallelism: the distance embedding and the
    # outer-sum init are otherwise materialized whole on every device before
    # the stack's own constraints take over.
    z_pair = shard_pair_rows(
        z_base
        + z_init
        + confidence_distance_embedding(
            x_pred_rep_coords,
            params.distance_embedding,
            compact_bins=compact_distance_bins,
        )
    )
    if params.pairformer_stack.blocks:
        s_single, z_pair = pairformer_stack(
            s_trunk,
            z_pair,
            pair_mask,
            params.pairformer_stack,
            use_scan=use_scan,
            triangle_mul_chunk_size=triangle_mul_chunk_size,
            triangle_att_q_chunk_size=triangle_att_q_chunk_size,
            single_att_q_chunk_size=single_att_q_chunk_size,
            triangle_attention_backend=triangle_attention_backend,
        )
        if s_single is None:
            raise ValueError("ConfidenceHead requires PairformerStack single output")
    else:
        s_single = s_trunk
    return confidence_output_logits(
        s_single.astype(jnp.float32),
        z_pair.astype(jnp.float32),
        atom_to_token_idx,
        atom_to_tokatom_idx,
        params.output,
    )


def confidence_head(
    input_feature_dict: dict[str, jnp.ndarray | dict[str, jnp.ndarray]],
    s_inputs: jnp.ndarray,
    s_trunk: jnp.ndarray,
    z_trunk: jnp.ndarray,
    pair_mask: jnp.ndarray | None,
    x_pred_coords: jnp.ndarray,
    params: ConfidenceHeadParams,
    *,
    use_embedding: bool = True,
    compact_distance_bins: bool = False,
    use_scan: bool = True,
    triangle_mul_chunk_size: int | None = None,
    triangle_att_q_chunk_size: int | None = None,
    single_att_q_chunk_size: int | None = None,
    triangle_attention_backend: str | None = None,
) -> dict[str, jnp.ndarray]:
    """Run the Protenix confidence head over the sample axis.

    The original PyTorch inference path loops over samples to reduce peak pair
    memory. This JAX path keeps that contract and stacks outputs on the same
    axes as Protenix.
    """

    n_token = int(s_inputs.shape[-2])
    rep_atom_idx = jnp.nonzero(
        input_feature_dict["distogram_rep_atom_mask"].astype(bool),
        size=n_token,
    )[0]
    x_pred_rep_coords = jnp.take(x_pred_coords, rep_atom_idx, axis=-2)
    num_samples = int(x_pred_rep_coords.shape[-3])
    atom_to_token_idx = input_feature_dict["atom_to_token_idx"]
    atom_to_tokatom_idx = input_feature_dict["atom_to_tokatom_idx"]

    outputs = []
    for sample_index in range(num_samples):
        outputs.append(
            confidence_head_single_sample(
                s_inputs,
                s_trunk,
                z_trunk,
                pair_mask,
                jnp.take(x_pred_rep_coords, sample_index, axis=-3),
                atom_to_token_idx,
                atom_to_tokatom_idx,
                params,
                use_embedding=use_embedding,
                compact_distance_bins=compact_distance_bins,
                use_scan=use_scan,
                triangle_mul_chunk_size=triangle_mul_chunk_size,
                triangle_att_q_chunk_size=triangle_att_q_chunk_size,
                single_att_q_chunk_size=single_att_q_chunk_size,
                triangle_attention_backend=triangle_attention_backend,
            )
        )

    return {
        "plddt": jnp.stack([out["plddt"] for out in outputs], axis=-3),
        "pae": jnp.stack([out["pae"] for out in outputs], axis=-4),
        "pde": jnp.stack([out["pde"] for out in outputs], axis=-4),
        "resolved": jnp.stack([out["resolved"] for out in outputs], axis=-3),
    }
