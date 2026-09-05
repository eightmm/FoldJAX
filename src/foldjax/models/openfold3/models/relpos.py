"""Relative position encoding for the pair representation (AF3 Algorithm 3).

Three clipped relative-offset one-hots plus a same-entity flag, concatenated:

* residue offset, valid only within a chain,
* token offset, valid only within the same residue of the same chain,
* chain (sym_id) offset, valid only within an entity.

Each ``relpos`` block spends its last bin on "condition failed" — pairs that are
not in the same chain/residue/entity land there rather than being zeroed, so the
model can tell "far apart" from "unrelated".
"""

from __future__ import annotations

from collections.abc import Mapping

import jax.numpy as jnp


def binned_one_hot(x: jnp.ndarray, bins: jnp.ndarray) -> jnp.ndarray:
    """One-hot the nearest bin, matching upstream's argmin-of-absolute-difference."""
    nearest = jnp.argmin(jnp.abs(x[..., None] - bins), axis=-1)
    return jnp.eye(bins.shape[0], dtype=jnp.float32)[nearest]


def _relpos(
    pos: jnp.ndarray,
    condition: jnp.ndarray,
    rel_clip_idx: int,
    cyclic_offsets: tuple[jnp.ndarray, jnp.ndarray] | None = None,
) -> jnp.ndarray:
    """Clipped relative-offset one-hot with an out-of-condition bin at the end."""
    offset = pos[..., :, None] - pos[..., None, :]
    if cyclic_offsets is not None:
        mask, wrapped = cyclic_offsets
        offset = jnp.where(mask, wrapped, offset)
    clipped = jnp.clip(offset + rel_clip_idx, 0, 2 * rel_clip_idx)
    # The extra bin index 2*k+1 means "condition not satisfied".
    final = jnp.where(condition, clipped, 2 * rel_clip_idx + 1)
    bins = jnp.arange(0, 2 * rel_clip_idx + 2)
    return binned_one_hot(final, bins)


def relpos_complex(
    batch: Mapping[str, jnp.ndarray],
    *,
    max_relative_idx: int,
    max_relative_chain: int,
) -> jnp.ndarray:
    """Return the ``[..., N_token, N_token, C]`` relative position features.

    Args:
        batch: needs ``residue_index``, ``asym_id``, ``entity_id``,
            ``token_index`` and ``sym_id``, each ``[..., N_token]``.
        max_relative_idx: clip for residue and token offsets.
        max_relative_chain: clip for chain offsets.

    Returns:
        ``[..., N_token, N_token, 4*max_relative_idx + 2*max_relative_chain + 5]``
        concatenation of residue, token, same-entity and chain features.
    """
    res_idx = batch["residue_index"]
    asym_id = batch["asym_id"]
    entity_id = batch["entity_id"]

    same_chain = asym_id[..., :, None] == asym_id[..., None, :]
    same_res = res_idx[..., :, None] == res_idx[..., None, :]
    same_entity = entity_id[..., :, None] == entity_id[..., None, :]

    cyclic_offsets = None
    if "cyclic_mask" in batch:
        mask = jnp.asarray(batch["cyclic_mask"], dtype=bool)
        within = same_chain & mask[..., :, None] & mask[..., None, :]
        # Native wraps the ordered cyclic token subset, not residue numbers.
        # Its round(length / 2) uses ties-to-even, including odd-length rings.
        order = jnp.arange(mask.shape[-1])
        rank = jnp.sum(within & (order[:, None] > order[None, :]), axis=-1)
        length = jnp.maximum(jnp.sum(within, axis=-1), 1)
        distance = (rank[..., None, :] - rank[..., :, None]) % length[..., :, None]
        centre = jnp.rint(length.astype(jnp.float32) / 2).astype(length.dtype)
        wrapped = jnp.where(
            distance <= centre[..., :, None],
            -distance,
            length[..., :, None] - distance,
        )
        cyclic_offsets = within, wrapped

    rel_pos = _relpos(res_idx, same_chain, max_relative_idx, cyclic_offsets)
    rel_token = _relpos(
        batch["token_index"], same_chain & same_res, max_relative_idx, cyclic_offsets
    )
    rel_chain = _relpos(
        batch["sym_id"], same_entity, max_relative_chain, cyclic_offsets
    )

    # Order matters: it fixes the layout linear_z expects.
    return jnp.concatenate(
        [rel_pos, rel_token, same_entity[..., None].astype(rel_pos.dtype), rel_chain],
        axis=-1,
    )
