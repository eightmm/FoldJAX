"""Diffusion transformer block and stack.

Each block is AdaLN-conditioned pair-biased attention followed by a conditioned
transition, both added residually. Upstream notes two deliberate deviations from
the AF3 supplement that this port keeps:

* residual connections are added around both sub-layers, and
* the transition consumes the *updated* ``a`` from the attention step rather than
  the block's input.

Only the self-attention configuration is ported. The cross-attention variant
(``CrossAttentionPairBias``, used by the atom encoder/decoder with a neighbourhood
mask) applies a single ``layer_norm_z`` at the stack level and is not covered.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from foldjax.models.openfold3.models.attention_pair_bias import (
    AdaAttentionPairBiasParams,
    CrossAttentionPairBiasParams,
    ada_attention_pair_bias,
    cross_attention_pair_bias,
)
from foldjax.models.openfold3.models.primitives import (
    ConditionedTransitionBlockParams,
    LayerNormParams,
    conditioned_transition_block,
    layer_norm,
)
from foldjax.models.openfold3.models.stacking import can_scan, scan_stack


class DiffusionTransformerBlockParams(NamedTuple):
    """Parameters for one ``DiffusionTransformerBlock``."""

    attention_pair_bias: AdaAttentionPairBiasParams
    conditioned_transition: ConditionedTransitionBlockParams


class DiffusionTransformerParams(NamedTuple):
    """Parameters for a ``DiffusionTransformer``: one entry per block, in order."""

    blocks: tuple[DiffusionTransformerBlockParams, ...]


def diffusion_transformer_block(
    a: jnp.ndarray,
    s: jnp.ndarray,
    z: jnp.ndarray,
    params: DiffusionTransformerBlockParams,
    *,
    no_heads: int,
    mask: jnp.ndarray | None = None,
    inf: float = 1e9,
    mask_transition: bool = True,
    eps: float = 1e-5,
) -> jnp.ndarray:
    """Apply one diffusion transformer block.

    Args:
        a: ``[..., N, C_token]`` activation.
        s: ``[..., N, C_s]`` conditioning single representation.
        z: ``[..., N, N, C_z]`` pair embedding.
        params: mapped block parameters.
        no_heads: attention head count.
        mask: ``[..., N]`` mask.
        inf: masking constant.
        mask_transition: upstream's ``_mask_trans``.
        eps: layer norm epsilon.

    Returns:
        ``[..., N, C_token]`` updated activation.
    """
    a = a + ada_attention_pair_bias(
        a,
        s,
        z,
        params.attention_pair_bias,
        no_heads=no_heads,
        mask=mask,
        inf=inf,
        eps=eps,
    )
    return a + conditioned_transition_block(
        a,
        s,
        params.conditioned_transition,
        mask=mask if mask_transition else None,
        eps=eps,
    )


def diffusion_transformer(
    a: jnp.ndarray,
    s: jnp.ndarray,
    z: jnp.ndarray,
    params: DiffusionTransformerParams,
    *,
    no_heads: int,
    mask: jnp.ndarray | None = None,
    inf: float = 1e9,
    mask_transition: bool = True,
    eps: float = 1e-5,
    scan_blocks: bool = True,
) -> jnp.ndarray:
    """Run every diffusion transformer block in order.

        scan_blocks: run the blocks as one scanned body over stacked parameters
            instead of unrolling them. Identical arithmetic; see
            :mod:`foldjax.models.openfold3.models.stacking` for the measured effect.
    """
    settings = dict(
        no_heads=no_heads,
        mask=mask,
        inf=inf,
        mask_transition=mask_transition,
        eps=eps,
    )
    if scan_blocks and can_scan(params.blocks):
        return scan_stack(
            lambda carry, block: diffusion_transformer_block(
                carry, s, z, block, **settings
            ),
            a,
            params.blocks,
        )
    for block in params.blocks:
        a = diffusion_transformer_block(
            a,
            s,
            z,
            block,
            **settings,
        )
    return a


class AtomTransformerBlockParams(NamedTuple):
    """Parameters for a cross-attention (sequence-local) transformer block."""

    attention_pair_bias: CrossAttentionPairBiasParams
    conditioned_transition: ConditionedTransitionBlockParams


class AtomTransformerParams(NamedTuple):
    """Parameters for the atom transformer.

    ``layer_norm_z`` is applied **once at the stack level**, not per block: the
    cross-attention blocks project the pair bias from an already-normed ``z``.
    """

    blocks: tuple[AtomTransformerBlockParams, ...]
    layer_norm_z: LayerNormParams


def atom_transformer(
    a: jnp.ndarray,
    s: jnp.ndarray,
    z: jnp.ndarray,
    params: AtomTransformerParams,
    *,
    no_heads: int,
    n_query: int,
    n_key: int,
    mask: jnp.ndarray | None = None,
    inf: float = 1e9,
    mask_transition: bool = True,
    eps: float = 1e-5,
    scan_blocks: bool = True,
) -> jnp.ndarray:
    """Run the sequence-local atom transformer.

    Args:
        a: ``[..., N_atom, C_a]`` atom single representation.
        s: ``[..., N_atom, C_s]`` atom single conditioning.
        z: ``[..., N_blocks, N_query, N_key, C_z]`` blocked pair conditioning.
        params: mapped parameters.
        no_heads: attention head count.
        n_query: query block height.
        n_key: key window width.
        mask: ``[..., N_atom]`` atom mask.
        inf: masking constant.
        mask_transition: upstream's ``_mask_trans``.
        eps: layer norm epsilon.

    Returns:
        ``[..., N_atom, C_a]`` updated atom representation.
    """
    # One norm for the whole stack, matching upstream's use_cross_attention path.
    z = layer_norm(z, params.layer_norm_z, eps=eps)

    def one_block(current: jnp.ndarray, block: AtomTransformerBlockParams):
        current = current + cross_attention_pair_bias(
            current,
            z,
            block.attention_pair_bias,
            no_heads=no_heads,
            n_query=n_query,
            n_key=n_key,
            mask=mask,
            s=s,
            inf=inf,
            eps=eps,
        )
        return current + conditioned_transition_block(
            current,
            s,
            block.conditioned_transition,
            mask=mask if mask_transition else None,
            eps=eps,
        )

    # This stack runs inside the diffusion rollout, so its body is emitted once per
    # block *per rollout step* when unrolled. Scanning it keeps the rollout body
    # one block wide.
    if scan_blocks and can_scan(params.blocks):
        return scan_stack(one_block, a, params.blocks)
    for block in params.blocks:
        a = one_block(a, block)
    return a
