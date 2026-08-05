"""Inter-chain clash detection (AF3 sample ranking).

Upstream loops over chain pairs in Python and slices each chain's atoms out with
boolean indexing, so both the loop count and the slice sizes depend on the data.
This port keeps the shapes static: a chain-membership one-hot turns the per-pair
clash count into one einsum over the full distance matrix.

That costs an ``[n_sample, N_atom, N_atom]`` intermediate, which is the honest
trade for static shapes. For large complexes this should be chunked over atoms
before it goes near a real structure; correctness comes first here.
"""

from __future__ import annotations

import jax.numpy as jnp


def compute_has_clash(
    asym_id: jnp.ndarray,
    atom_positions: jnp.ndarray,
    atom_mask: jnp.ndarray,
    is_polymer: jnp.ndarray,
    *,
    n_chain: int,
    threshold: float = 1.1,
    violation_abs: int = 100,
    violation_frac: float = 0.5,
) -> jnp.ndarray:
    """Return ``[n_sample]``, 1.0 where two distinct polymer chains clash.

    A chain counts as a polymer chain only if *every* one of its atoms is marked
    polymer, matching upstream's ``((asym_id != aid) | is_polymer).all()``.

    Args:
        asym_id: ``[N_atom]`` chain id per atom.
        atom_positions: ``[n_sample, N_atom, 3]`` predicted coordinates.
        atom_mask: ``[N_atom]`` valid-atom mask.
        is_polymer: ``[N_atom]`` polymer flag per atom.
        n_chain: static upper bound on chain ids (ids must be in ``[0, n_chain)``).
        threshold: clash distance in the same units as ``atom_positions``.
        violation_abs: absolute clash count that trips the flag.
        violation_frac: clash count over the smaller chain's atom count that
            trips the flag.

    Returns:
        ``[n_sample]`` float array of 0.0/1.0.
    """
    chains = jnp.arange(n_chain)
    # [N_atom, n_chain]
    in_chain = asym_id[:, None] == chains[None, :]

    # A chain is a polymer chain only if all of its atoms are polymer atoms.
    chain_is_polymer = jnp.all(~in_chain | is_polymer[:, None].astype(bool), axis=0)
    chain_exists = jnp.any(in_chain, axis=0)

    member = (
        in_chain
        & atom_mask[:, None].astype(bool)
        & (chain_is_polymer & chain_exists)[None, :]
    ).astype(atom_positions.dtype)

    # [n_chain] valid atoms per polymer chain.
    counts = jnp.sum(member, axis=0)

    # [n_sample, N_atom, N_atom]
    deltas = atom_positions[:, :, None, :] - atom_positions[:, None, :, :]
    close = (jnp.sqrt(jnp.sum(deltas**2, axis=-1)) < threshold).astype(
        atom_positions.dtype
    )

    # [n_sample, n_chain, n_chain] clashes between each ordered chain pair.
    pair_clashes = jnp.einsum("sab,ac,bd->scd", close, member, member)

    smaller = jnp.minimum(counts[:, None], counts[None, :])
    fraction = pair_clashes / jnp.maximum(smaller, 1.0)
    tripped = (pair_clashes > violation_abs) | (fraction > violation_frac)

    # Distinct chains only, each unordered pair once, both non-empty.
    both_present = (counts[:, None] > 0) & (counts[None, :] > 0)
    upper = jnp.triu(jnp.ones((n_chain, n_chain), dtype=bool), k=1)
    considered = upper & both_present

    return jnp.any(tripped & considered[None, :, :], axis=(-2, -1)).astype(
        atom_positions.dtype
    )


def sample_ranking_score(
    ptm: jnp.ndarray,
    iptm: jnp.ndarray,
    has_clash: jnp.ndarray,
    disorder: jnp.ndarray | None = None,
    *,
    ptm_weight: float = 0.2,
    iptm_weight: float = 0.8,
    disorder_weight: float = 0.5,
    has_clash_weight: float = 100.0,
) -> jnp.ndarray:
    """Return the AF3 sample ranking score.

    ``0.8*ipTM + 0.2*pTM + 0.5*disorder - 100*has_clash``. The clash weight is
    large enough to push any clashing sample below every clean one, so it acts as
    a veto rather than a penalty. ``disorder`` defaults to zero, matching upstream
    when RASA is unavailable.

    Args:
        ptm: ``[n_sample]`` pTM.
        iptm: ``[n_sample]`` ipTM.
        has_clash: ``[n_sample]`` 0.0/1.0 clash flag.
        disorder: ``[n_sample]`` disorder fraction, or ``None`` for zeros.
        ptm_weight: pTM coefficient.
        iptm_weight: ipTM coefficient.
        disorder_weight: disorder coefficient.
        has_clash_weight: clash penalty.

    Returns:
        ``[n_sample]`` ranking score; higher is better.
    """
    if disorder is None:
        disorder = jnp.zeros_like(ptm)
    return (
        iptm_weight * iptm
        + ptm_weight * ptm
        + disorder_weight * disorder
        - has_clash_weight * has_clash
    )
