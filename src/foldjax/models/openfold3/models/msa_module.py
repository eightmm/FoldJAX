"""MSA module block and stack (AF3 Algorithm 8, lines 5-15).

Two upstream details drive the shape of this port:

* ``opm_first`` moves the outer product mean from after the MSA update to before
  it, which changes which ``m`` the pair update sees.
* The last block sets ``skip_msa_update = last_block and opm_first``, and when it
  does, ``msa_att_row`` and ``msa_transition`` are ``None`` and absent from the
  checkpoint. Its MSA output is therefore passed through untouched.

The AF2-only MSA attention variants (row attention with pair bias, column
attention, column global attention) are not part of this path: AF3 replaces them
with ``MSAPairWeightedAveraging``.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from foldjax.models.openfold3.models.msa import (
    MSAPairWeightedAveragingParams,
    OuterProductMeanParams,
    msa_pair_weighted_averaging,
    outer_product_mean,
)
from foldjax.models.openfold3.models.pair_block import PairBlockParams, pair_block
from foldjax.models.openfold3.models.primitives import (
    SwiGLUTransitionParams,
    swiglu_transition,
)
from foldjax.models.openfold3.models.stacking import can_scan, scan_stack


class MSAModuleBlockParams(NamedTuple):
    """Parameters for one ``MSAModuleBlock``.

    ``msa_att_row`` and ``msa_transition`` are ``None`` for a block that skips
    the MSA update, mirroring the absent checkpoint entries.
    """

    outer_product_mean: OuterProductMeanParams
    pair_stack: PairBlockParams
    msa_att_row: MSAPairWeightedAveragingParams | None = None
    msa_transition: SwiGLUTransitionParams | None = None


class MSAModuleStackParams(NamedTuple):
    """Parameters for an ``MSAModuleStack``: one entry per block, in order."""

    blocks: tuple[MSAModuleBlockParams, ...]


def msa_module_block(
    m: jnp.ndarray,
    z: jnp.ndarray,
    params: MSAModuleBlockParams,
    *,
    msa_mask: jnp.ndarray,
    pair_mask: jnp.ndarray,
    no_heads_msa: int,
    no_heads_pair: int,
    opm_first: bool = True,
    inf: float = 1e9,
    mask_transition: bool = True,
    eps: float = 1e-5,
    chunk_size: int | None = None,
    opm_eps: float = 1e-3,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Apply one MSA module block.

    Args:
        m: ``[..., N_seq, N_token, C_m]`` MSA embedding.
        z: ``[..., N_token, N_token, C_z]`` pair embedding.
        params: mapped block parameters.
        msa_mask: ``[..., N_seq, N_token]`` MSA mask.
        pair_mask: ``[..., N_token, N_token]`` pair mask.
        no_heads_msa: head count for pair-weighted averaging.
        no_heads_pair: head count for triangle attention in the pair stack.
        opm_first: run the outer product mean before the MSA update.
        inf: masking constant.
        mask_transition: upstream's ``_mask_trans``.
        eps: layer norm epsilon.
        opm_eps: outer product mean normalization epsilon.

    Returns:
        ``(m, z)`` updated representations.
    """
    if opm_first:
        z = z + outer_product_mean(
            m,
            params.outer_product_mean,
            mask=msa_mask,
            eps=opm_eps,
            layer_norm_eps=eps,
        )

    if params.msa_att_row is not None:
        if params.msa_transition is None:
            raise ValueError(
                "a block with msa_att_row must also carry msa_transition"
            )
        m = m + msa_pair_weighted_averaging(
            m,
            z,
            params.msa_att_row,
            no_heads=no_heads_msa,
            mask=pair_mask,
            inf=inf,
            eps=eps,
        )
        m = m + swiglu_transition(
            m,
            params.msa_transition,
            mask=msa_mask if mask_transition else None,
            eps=eps,
        )
        if not opm_first:
            z = z + outer_product_mean(
                m,
                params.outer_product_mean,
                mask=msa_mask,
                eps=opm_eps,
                layer_norm_eps=eps,
            )

    z = pair_block(
        z,
        params.pair_stack,
        pair_mask=pair_mask,
        no_heads_pair=no_heads_pair,
        inf=inf,
        mask_transition=mask_transition,
        eps=eps,
        chunk_size=chunk_size,
    )
    return m, z


def msa_module_stack(
    m: jnp.ndarray,
    z: jnp.ndarray,
    params: MSAModuleStackParams,
    *,
    msa_mask: jnp.ndarray,
    pair_mask: jnp.ndarray,
    no_heads_msa: int,
    no_heads_pair: int,
    opm_first: bool = True,
    inf: float = 1e9,
    mask_transition: bool = True,
    eps: float = 1e-5,
    chunk_size: int | None = None,
    opm_eps: float = 1e-3,
) -> jnp.ndarray:
    """Run every MSA module block and return the pair representation.

    Upstream's ``_wrap_up`` discards the MSA embedding, so only ``z`` escapes the
    stack.
    """
    settings = dict(
        msa_mask=msa_mask,
        pair_mask=pair_mask,
        no_heads_msa=no_heads_msa,
        no_heads_pair=no_heads_pair,
        opm_first=opm_first,
        inf=inf,
        mask_transition=mask_transition,
        eps=eps,
        chunk_size=chunk_size,
        opm_eps=opm_eps,
    )
    # A block configured with ``skip_msa_update`` has no msa_att_row at all, so a
    # stack that mixes the two cannot be scanned; can_scan detects exactly that.
    if can_scan(params.blocks):
        def body(carry, block):
            return msa_module_block(carry[0], carry[1], block, **settings)

        _, z = scan_stack(body, (m, z), params.blocks)
        return z
    for block in params.blocks:
        m, z = msa_module_block(m, z, block, **settings)
    return z
