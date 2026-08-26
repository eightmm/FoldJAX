"""Token-to-atom broadcasting.

Upstream builds atom-level features with ``torch.repeat_interleave``, whose output
length depends on the data. JAX needs static shapes, so this port takes the atom
count as an argument and computes the same mapping with a cumulative sum and a
gather: atom ``a`` belongs to the token whose cumulative atom boundary first
exceeds ``a``.

That is equivalent, not merely similar — the parity gate checks it against the
upstream ``repeat_interleave`` path directly. For inference the atom count is a
fixed bucket, so requiring it up front costs nothing.
"""

from __future__ import annotations

import jax.numpy as jnp


def broadcast_token_feat_to_atoms(
    token_mask: jnp.ndarray,
    num_atoms_per_token: jnp.ndarray,
    token_feat: jnp.ndarray,
    *,
    n_atom: int,
    atom_to_token_index: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Broadcast a per-token feature to per-atom.

    Args:
        token_mask: ``[..., N_token]`` token mask.
        num_atoms_per_token: ``[..., N_token]`` atom count per token.
        token_feat: ``[..., N_token, C]`` token feature.
        n_atom: static output atom count.
        atom_to_token_index: optional validated ``[..., n_atom]`` owner table.
            Supplying it avoids rebuilding the owner table from all token
            boundaries. Its real-atom prefix must describe the same runs as
            ``num_atoms_per_token``; padded owner values are ignored.

    Returns:
        ``[..., n_atom, C]``. Atoms past the last token's boundary are zero, as
        are atoms of masked-out tokens.
    """
    if token_feat.ndim < 2:
        raise ValueError("token_feat must have a trailing feature axis")

    token_feat = token_feat * token_mask[..., None]
    n_token = token_mask.shape[-1]
    counts = num_atoms_per_token * token_mask
    positions = jnp.arange(n_atom)

    if atom_to_token_index is not None:
        if atom_to_token_index.shape[-1] != n_atom:
            raise ValueError(
                "atom_to_token_index length must equal the requested atom count"
            )
        safe = jnp.clip(atom_to_token_index.astype(jnp.int32), 0, n_token - 1)
        atom_feat = jnp.take_along_axis(token_feat, safe[..., None], axis=-2)
        # Preserve the count-based primitive's exact validity semantics. The
        # high-level owner table has the same prefix contract, but deriving
        # validity here also keeps direct calls compatible for padded lanes.
        valid = positions < jnp.sum(counts, axis=-1, keepdims=True)
        return jnp.where(valid[..., None], atom_feat, 0.0)

    # [..., N_token] exclusive-end boundaries of each token's atom run.
    boundaries = jnp.cumsum(counts, axis=-1)

    # Number of boundaries at or below each position = owning token index.
    index = jnp.sum(
        positions[..., None] >= boundaries[..., None, :], axis=-1
    ).astype(jnp.int32)
    safe = jnp.clip(index, 0, n_token - 1)

    atom_feat = jnp.take_along_axis(token_feat, safe[..., None], axis=-2)

    # Positions beyond the final boundary have no owning token.
    valid = positions < boundaries[..., -1:]
    return jnp.where(valid[..., None], atom_feat, 0.0)


def max_atom_per_token_masked_select(
    atom_feat: jnp.ndarray, mask: jnp.ndarray, *, n_atom: int
) -> jnp.ndarray:
    """Compact per-token-padded atom features down to the real atoms.

    Upstream uses ``torch.masked_select``, whose output length depends on the
    data. This port takes the output length and compacts with a stable argsort:
    sorting on ``~mask`` moves every valid slot to the front while preserving
    their original relative order, which is exactly ``masked_select``'s ordering.

    Args:
        atom_feat: ``[..., N_token * max_atoms_per_token, C]`` padded features.
        mask: ``[..., N_token * max_atoms_per_token]`` valid-atom mask.
        n_atom: static output atom count.

    Returns:
        ``[..., n_atom, C]``, zero-padded past the number of valid atoms.
    """
    valid = mask.astype(bool)
    # Stable sort on the negated mask: valid slots first, order preserved.
    order = jnp.argsort(~valid, axis=-1, stable=True)
    gathered = jnp.take_along_axis(atom_feat, order[..., None], axis=-2)
    kept = jnp.take_along_axis(valid, order, axis=-1)
    return jnp.where(kept[..., :n_atom, None], gathered[..., :n_atom, :], 0.0)


def aggregate_atom_feat_to_tokens(
    atom_feat: jnp.ndarray,
    atom_to_token_index: jnp.ndarray,
    atom_mask: jnp.ndarray,
    *,
    n_token: int,
    aggregate: str = "mean",
    eps: float = 1e-9,
) -> jnp.ndarray:
    """Scatter per-atom features up to per-token features.

    Masked atoms are routed to an extra ``n_token``-th bin which is then dropped,
    which is how upstream keeps them out of the sums without a gather. That
    overflow bin is the reason the scatter target has ``n_token + 1`` rows.

    Args:
        atom_feat: ``[..., N_atom, C]`` atom features.
        atom_to_token_index: ``[..., N_atom]`` owning token of each atom.
        atom_mask: ``[..., N_atom]`` atom mask.
        n_token: static token count.
        aggregate: ``"mean"`` or ``"sum"``.
        eps: denominator floor for the mean.

    Returns:
        ``[..., n_token, C]`` token features.
    """
    if aggregate not in ("mean", "sum"):
        raise ValueError(f"invalid aggregation function: {aggregate}")

    mask = atom_mask.astype(atom_feat.dtype)
    atom_feat = atom_feat * mask[..., None]

    # Masked atoms land in the discarded overflow bin.
    index = (atom_to_token_index * mask + n_token * (1.0 - mask)).astype(jnp.int32)

    # A one-hot contraction rather than a scatter-add: GPU scatter uses atomics,
    # whose summation order is not reproducible, and the diffusion rollout
    # amplifies that into visibly different coordinates for the same PRNG key.
    # This form is deterministic and computes the same sums.
    membership = (
        index[..., :, None] == jnp.arange(n_token + 1)
    ).astype(atom_feat.dtype)

    totals = jnp.einsum("...ac,...at->...tc", atom_feat, membership)
    token_feat = totals[..., :n_token, :]

    if aggregate == "sum":
        return token_feat

    counts = jnp.einsum("...a,...at->...t", mask, membership)
    return token_feat / (counts[..., :n_token, None] + eps)
