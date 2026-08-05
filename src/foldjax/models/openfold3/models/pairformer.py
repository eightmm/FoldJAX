"""Pairformer block and stack (AF3 Algorithm 17).

The block updates the pair representation through the shared pair stack, then
updates the single representation with pair-biased attention and a transition.
The stack is a plain sequential loop over blocks; upstream's activation
checkpointing and chunk-size tuning are training/memory concerns with no effect
on the computed function.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from foldjax.models.openfold3.models.attention_pair_bias import (
    AttentionPairBiasParams,
    attention_pair_bias,
)
from foldjax.models.openfold3.models.pair_block import PairBlockParams, pair_block
from foldjax.models.openfold3.models.primitives import (
    SwiGLUTransitionParams,
    swiglu_transition,
)
from foldjax.models.openfold3.models.stacking import can_scan, scan_stack


class PairformerBlockParams(NamedTuple):
    """Parameters for one ``PairFormerBlock``."""

    pair_stack: PairBlockParams
    attn_pair_bias: AttentionPairBiasParams
    single_transition: SwiGLUTransitionParams


class PairformerStackParams(NamedTuple):
    """Parameters for a ``PairFormerStack``: one entry per block, in order."""

    blocks: tuple[PairformerBlockParams, ...]


def pairformer_block(
    s: jnp.ndarray,
    z: jnp.ndarray,
    params: PairformerBlockParams,
    *,
    single_mask: jnp.ndarray,
    pair_mask: jnp.ndarray,
    no_heads_pair: int,
    no_heads_pair_bias: int,
    inf: float = 1e9,
    mask_transition: bool = True,
    eps: float = 1e-5,
    chunk_size: int | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Apply one Pairformer block.

    Args:
        s: ``[..., N, C_s]`` single representation.
        z: ``[..., N, N, C_z]`` pair representation.
        params: mapped block parameters.
        single_mask: ``[..., N]`` single mask.
        pair_mask: ``[..., N, N]`` pair mask.
        no_heads_pair: head count for triangle attention.
        no_heads_pair_bias: head count for the pair-biased single attention.
        inf: masking constant.
        mask_transition: upstream's ``_mask_trans``.
        eps: layer norm epsilon.

    Returns:
        ``(s, z)`` updated representations. The pair stack runs first, so the
        single update sees the already-updated ``z``.
    """
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

    s = s + attention_pair_bias(
        s,
        z,
        params.attn_pair_bias,
        no_heads=no_heads_pair_bias,
        mask=single_mask,
        inf=inf,
        eps=eps,
    )

    s = s + swiglu_transition(
        s,
        params.single_transition,
        mask=single_mask if mask_transition else None,
        eps=eps,
    )

    return s, z


def pairformer_stack(
    s: jnp.ndarray,
    z: jnp.ndarray,
    params: PairformerStackParams,
    *,
    single_mask: jnp.ndarray,
    pair_mask: jnp.ndarray,
    no_heads_pair: int,
    no_heads_pair_bias: int,
    inf: float = 1e9,
    mask_transition: bool = True,
    eps: float = 1e-5,
    chunk_size: int | None = None,
    scan_blocks: bool = True,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Run every Pairformer block in order.

    ``scan_blocks`` runs them as a ``lax.scan`` over stacked parameters instead of
    unrolling. The released stack is 48 blocks, and unrolling it puts 48 copies of
    the block body in one HLO module: XLA then plans buffers across the whole thing
    at once, which is where the memory goes, and compiles it block by block, which
    is where the compile time goes. The arithmetic is identical either way.
    """
    if scan_blocks and can_scan(params.blocks):
        return _scan_blocks(
            s,
            z,
            params,
            single_mask=single_mask,
            pair_mask=pair_mask,
            no_heads_pair=no_heads_pair,
            no_heads_pair_bias=no_heads_pair_bias,
            inf=inf,
            mask_transition=mask_transition,
            eps=eps,
            chunk_size=chunk_size,
        )
    for block in params.blocks:
        s, z = pairformer_block(
            s,
            z,
            block,
            single_mask=single_mask,
            pair_mask=pair_mask,
            no_heads_pair=no_heads_pair,
            no_heads_pair_bias=no_heads_pair_bias,
            inf=inf,
            mask_transition=mask_transition,
            eps=eps,
            chunk_size=chunk_size,
        )
    return s, z


def _scan_blocks(
    s: jnp.ndarray,
    z: jnp.ndarray,
    params: PairformerStackParams,
    **kwargs,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Run the blocks as one scanned body over stacked parameters."""

    def body(carry, block):
        single, pair = carry
        return pairformer_block(single, pair, block, **kwargs)

    return scan_stack(body, (s, z), params.blocks)
