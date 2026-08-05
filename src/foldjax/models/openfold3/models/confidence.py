"""Confidence metrics computed from head logits (AF3 SI 5.7 and 5.9.1).

Weight-free arithmetic over binned distributions. Two details carry most of the
risk:

* Bin centers are ``boundaries[:-1] + width/2``, i.e. the *left* edges shifted by
  half a width. Using ``linspace`` midpoints or the right edges shifts every
  expectation.
* pTM's ``d0`` is derived from the number of *considered* tokens clipped at 19,
  not the full token count, and the final score is a max over ``i`` restricted to
  tokens with a valid frame.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def bin_centers(bin_min: float, bin_max: float, no_bins: int) -> jnp.ndarray:
    """Return ``[no_bins]`` bin centers for a uniform binning."""
    width = (bin_max - bin_min) / float(no_bins)
    boundaries = jnp.linspace(bin_min, bin_max, no_bins + 1)
    return boundaries[:-1] + 0.5 * width


def probs_to_expected_error(
    probs: jnp.ndarray, *, bin_min: float, bin_max: float, no_bins: int
) -> jnp.ndarray:
    """Return the expectation of the binned distribution over its last axis."""
    return jnp.sum(probs * bin_centers(bin_min, bin_max, no_bins), axis=-1)


def compute_plddt(logits: jnp.ndarray) -> jnp.ndarray:
    """Return pLDDT from per-atom logits.

    Upstream hardcodes the ``[0, 1]`` / 50-bin configuration here, so this does
    too rather than inventing parameters.
    """
    return probs_to_expected_error(
        jax.nn.softmax(logits, axis=-1), bin_min=0.0, bin_max=1.0, no_bins=50
    )


def compute_ptm(
    logits: jnp.ndarray,
    has_frame: jnp.ndarray,
    mask_i: jnp.ndarray,
    *,
    bin_min: float,
    bin_max: float,
    no_bins: int,
    asym_id: jnp.ndarray | None = None,
    interface: bool = False,
    eps: float = 1e-8,
) -> jnp.ndarray:
    """Return pTM, or ipTM when ``interface`` is set (AF3 SI 5.9.1, Eqs. 17-18).

    Args:
        logits: ``[n_sample, N_token, N_token, no_bins]`` pair-distance logits.
        has_frame: ``[n_sample, N_token]`` tokens with a valid frame.
        mask_i: ``[N_token]`` boolean mask of the token set to consider.
        bin_min: lower bin edge.
        bin_max: upper bin edge.
        no_bins: bin count.
        asym_id: ``[N_token]`` chain ids; required when ``interface``.
        interface: exclude same-chain pairs, giving ipTM.
        eps: denominator floor.

    Returns:
        ``[n_sample]`` score.
    """
    if interface and asym_id is None:
        raise ValueError("asym_id is required when interface=True")

    mask_i = mask_i.astype(bool)
    considered = jnp.maximum(jnp.sum(mask_i), 1).astype(logits.dtype)
    # d0 uses the considered-token count clipped at 19, per the SI.
    clipped = jnp.maximum(considered, 19.0)
    d0 = 1.24 * jnp.maximum(clipped - 15.0, 0.0) ** (1.0 / 3.0) - 1.8

    weight = 1.0 / (1.0 + (bin_centers(bin_min, bin_max, no_bins) / d0) ** 2)

    ptm_ij = jnp.sum(jax.nn.softmax(logits, axis=-1) * weight, axis=-1)

    # Upstream slices both pair axes down to the considered set. Slicing is a
    # boolean index, which jit cannot trace, so the same restriction is applied
    # as a mask on the j-sum instead. Rows outside the set are zeroed before the
    # final max, which is equivalent to their not existing.
    keep_pair = mask_i[..., :, None] & mask_i[..., None, :]
    if interface:
        chains = asym_id
        keep_pair = keep_pair & (chains[..., :, None] != chains[..., None, :])
        denominator = jnp.maximum(jnp.sum(keep_pair, axis=-1), eps)
    else:
        denominator = considered

    tm_i = jnp.sum(ptm_ij * keep_pair, axis=-1) / denominator
    tm_i = jnp.where(has_frame.astype(bool) & mask_i, tm_i, 0.0)
    return jnp.max(tm_i, axis=-1)


def compute_chain_ptm(
    logits: jnp.ndarray,
    has_frame: jnp.ndarray,
    token_mask: jnp.ndarray,
    asym_id: jnp.ndarray,
    *,
    n_chain: int,
    bin_min: float,
    bin_max: float,
    no_bins: int,
) -> jnp.ndarray:
    """Return per-chain pTM (AF3 SI 5.9.3, item 2).

    Each chain's score is pTM restricted to that chain's tokens, so this is
    ``compute_ptm(interface=False)`` once per chain id. ``n_chain`` is static
    because the loop is unrolled at trace time.

    Args:
        logits: ``[n_sample, N_token, N_token, no_bins]`` pair-distance logits.
        has_frame: ``[n_sample, N_token]`` tokens with a valid frame.
        token_mask: ``[N_token]`` valid-token mask.
        asym_id: ``[N_token]`` chain id per token.
        n_chain: static chain-id upper bound.
        bin_min: lower bin edge.
        bin_max: upper bin edge.
        no_bins: bin count.

    Returns:
        ``[n_sample, n_chain]``. Chains with no tokens score 0.
    """
    scores = []
    for chain in range(n_chain):
        mask_i = token_mask.astype(bool) & (asym_id == chain)
        present = jnp.any(mask_i)
        chain_ptm = compute_ptm(
            logits,
            has_frame,
            mask_i,
            bin_min=bin_min,
            bin_max=bin_max,
            no_bins=no_bins,
        )
        scores.append(jnp.where(present, chain_ptm, 0.0))
    return jnp.stack(scores, axis=-1)


def compute_chain_pair_iptm(
    logits: jnp.ndarray,
    has_frame: jnp.ndarray,
    token_mask: jnp.ndarray,
    asym_id: jnp.ndarray,
    *,
    n_chain: int,
    bin_min: float,
    bin_max: float,
    no_bins: int,
) -> jnp.ndarray:
    """Return the symmetric chain-pair ipTM matrix (AF3 SI 5.9.3, item 3).

    Each entry is ipTM computed over the *union* of two chains' tokens, with the
    interface term still excluding same-chain pairs. The diagonal stays zero: a
    chain has no interface with itself.

    Args:
        logits: ``[n_sample, N_token, N_token, no_bins]`` pair-distance logits.
        has_frame: ``[n_sample, N_token]`` tokens with a valid frame.
        token_mask: ``[N_token]`` valid-token mask.
        asym_id: ``[N_token]`` chain id per token.
        n_chain: static chain-id upper bound.
        bin_min: lower bin edge.
        bin_max: upper bin edge.
        no_bins: bin count.

    Returns:
        ``[n_sample, n_chain, n_chain]``, symmetric with a zero diagonal.
    """
    valid = token_mask.astype(bool)
    chain_masks = [valid & (asym_id == chain) for chain in range(n_chain)]

    rows = []
    for i in range(n_chain):
        row = []
        for j in range(n_chain):
            if i == j:
                row.append(jnp.zeros(logits.shape[0], dtype=logits.dtype))
                continue
            pair_mask = chain_masks[i] | chain_masks[j]
            both_present = jnp.any(chain_masks[i]) & jnp.any(chain_masks[j])
            value = compute_ptm(
                logits,
                has_frame,
                pair_mask,
                bin_min=bin_min,
                bin_max=bin_max,
                no_bins=no_bins,
                asym_id=asym_id,
                interface=True,
            )
            row.append(jnp.where(both_present, value, 0.0))
        rows.append(jnp.stack(row, axis=-1))
    return jnp.stack(rows, axis=-2)


def compute_chain_mean_iptm(
    chain_pair_iptm: jnp.ndarray,
    has_frame: jnp.ndarray,
    token_mask: jnp.ndarray,
    asym_id: jnp.ndarray,
    *,
    n_chain: int,
) -> jnp.ndarray:
    """Average each chain's interface scores (AF3 SI 5.9.3).

    The upstream averaging is asymmetric in a way worth spelling out: row entries
    ``[i, j]`` are included only when chain ``i`` has a frame, while column
    entries ``[j, i]`` are included per-``j`` when chain ``j`` has a frame. Both
    directions are averaged together, so a chain with no frame still gets a score
    from the columns of framed partners.

    ``has_frame`` is reduced over samples as well as tokens, matching upstream's
    unqualified ``.any()``.

    Args:
        chain_pair_iptm: ``[n_sample, n_chain, n_chain]`` pair matrix.
        has_frame: ``[n_sample, N_token]`` tokens with a valid frame.
        token_mask: ``[N_token]`` valid-token mask.
        asym_id: ``[N_token]`` chain id per token.
        n_chain: static chain-id upper bound.

    Returns:
        ``[n_sample, n_chain]``; chains with no contributing entries score 0.
    """
    valid = token_mask.astype(bool)
    any_frame = jnp.any(has_frame.astype(bool), axis=0)
    framed = jnp.stack(
        [
            jnp.any(valid & (asym_id == chain) & any_frame)
            for chain in range(n_chain)
        ]
    )

    off_diagonal = ~jnp.eye(n_chain, dtype=bool)
    row_take = off_diagonal & framed[:, None]
    col_take = off_diagonal & framed[None, :]

    transposed = jnp.swapaxes(chain_pair_iptm, -1, -2)
    total = jnp.sum(chain_pair_iptm * row_take, axis=-1) + jnp.sum(
        transposed * col_take, axis=-1
    )
    count = jnp.sum(row_take, axis=-1) + jnp.sum(col_take, axis=-1)
    return jnp.where(count > 0, total / jnp.maximum(count, 1), 0.0)


def compute_bespoke_iptm(
    chain_mean_iptm: jnp.ndarray,
    token_mask: jnp.ndarray,
    asym_id: jnp.ndarray,
    is_ligand: jnp.ndarray,
    *,
    n_chain: int,
) -> jnp.ndarray:
    """Return the ligand-aware "bespoke" ipTM matrix (AF3 SI 5.9.3).

    For a protein-ligand pair the ligand's own mean carries the score; for a
    protein-protein pair the two means are averaged. When *both* chains are
    ligands the row chain wins, because upstream tests ``i`` first.

    A chain counts as a ligand when at least half its tokens are ligand tokens
    (upstream's ``(mask & is_ligand).sum() * 2 >= mask.sum()``). Note this makes
    an empty chain a ligand chain, since ``0 >= 0``; that is upstream behaviour
    and is left intact.

    Args:
        chain_mean_iptm: ``[n_sample, n_chain]`` per-chain mean ipTM.
        token_mask: ``[N_token]`` valid-token mask.
        asym_id: ``[N_token]`` chain id per token.
        is_ligand: ``[N_token]`` ligand flag per token.
        n_chain: static chain-id upper bound.

    Returns:
        ``[n_sample, n_chain, n_chain]`` with a zero diagonal.
    """
    valid = token_mask.astype(bool)
    ligand_flags = []
    for chain in range(n_chain):
        chain_tokens = valid & (asym_id == chain)
        ligand_count = jnp.sum(chain_tokens & is_ligand.astype(bool))
        ligand_flags.append(ligand_count * 2 >= jnp.sum(chain_tokens))
    is_ligand_chain = jnp.stack(ligand_flags)

    row_mean = chain_mean_iptm[..., :, None]
    col_mean = chain_mean_iptm[..., None, :]

    pairwise = jnp.where(
        is_ligand_chain[:, None],
        row_mean,
        jnp.where(is_ligand_chain[None, :], col_mean, 0.5 * (row_mean + col_mean)),
    )
    off_diagonal = ~jnp.eye(n_chain, dtype=bool)
    return jnp.where(off_diagonal, pairwise, 0.0)
