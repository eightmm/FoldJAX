"""Diffusion transformer block and stack.

Each block is AdaLN-conditioned pair-biased attention followed by a conditioned
transition, both added residually. Upstream notes two deliberate deviations from
the AF3 supplement that this port keeps:

* residual connections are added around both sub-layers, and
* the transition consumes the *updated* ``a`` from the attention step rather than
  the block's input.

OpenFold3 v0.5 applies ``layer_norm_z`` once at the stack level before sharing
the normalized pair representation across all attention blocks.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from foldjax.models._stacking import stacked_or_stack
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
    linear,
)
from foldjax.models.openfold3.models.stacking import can_scan, scan_stack
from foldjax.models.openfold3.models.triangle import permute_final_dims


class DiffusionTransformerBlockParams(NamedTuple):
    """Parameters for one ``DiffusionTransformerBlock``."""

    attention_pair_bias: AdaAttentionPairBiasParams
    conditioned_transition: ConditionedTransitionBlockParams


class DiffusionTransformerParams(NamedTuple):
    """Parameters for an OpenFold3 v0.5 ``DiffusionTransformer`` stack."""

    blocks: tuple[DiffusionTransformerBlockParams, ...]
    layer_norm_z: LayerNormParams


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
    glu_backend: str = "xla",
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
        glu_backend=glu_backend,
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
    glu_backend: str = "xla",
    scan_blocks: bool = True,
) -> jnp.ndarray:
    """Run every diffusion transformer block in order.

    scan_blocks: run the blocks as one scanned body over stacked parameters
        instead of unrolling them. Identical arithmetic; see
        :mod:`foldjax.models.openfold3.models.stacking` for the measured effect.
    """
    z = layer_norm(z, params.layer_norm_z, eps=eps)

    settings = dict(
        no_heads=no_heads,
        mask=mask,
        inf=inf,
        mask_transition=mask_transition,
        eps=eps,
        glu_backend=glu_backend,
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


def atom_transformer_pair_biases(
    z: jnp.ndarray,
    params: AtomTransformerParams,
    *,
    eps: float = 1e-5,
) -> jnp.ndarray:
    """Precompute per-block attention pair biases for :func:`atom_transformer`.

    Applies the stack-level ``layer_norm_z`` once, then projects each block's
    ``linear_z`` and permutes ``(..., N_blocks, N_query, N_key, H)`` to
    ``(no_blocks, ..., N_blocks, H, N_query, N_key)``.
    """
    if len(params.blocks) == 0:
        return jnp.empty(
            (0, *z.shape[:-3], 0, z.shape[-3], z.shape[-2]), dtype=z.dtype
        )
    z = layer_norm(z, params.layer_norm_z, eps=eps)
    if can_scan(params.blocks):
        stacked = stacked_or_stack(params.blocks)
        return jax.lax.map(
            lambda lin_z: permute_final_dims(linear(z, lin_z), (2, 0, 1)),
            stacked.attention_pair_bias.linear_z,
        )
    return jnp.stack(
        [
            permute_final_dims(
                linear(z, block.attention_pair_bias.linear_z), (2, 0, 1)
            )
            for block in params.blocks
        ],
        axis=0,
    )


def atom_transformer(
    a: jnp.ndarray,
    s: jnp.ndarray,
    z: jnp.ndarray | None,
    params: AtomTransformerParams,
    *,
    no_heads: int,
    n_query: int,
    n_key: int,
    mask: jnp.ndarray | None = None,
    inf: float = 1e9,
    mask_transition: bool = True,
    eps: float = 1e-5,
    glu_backend: str = "xla",
    scan_blocks: bool = True,
    pair_biases: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Run the sequence-local atom transformer.

    Args:
        a: ``[..., N_atom, C_a]`` atom single representation.
        s: ``[..., N_atom, C_s]`` atom single conditioning.
        z: ``[..., N_blocks, N_query, N_key, C_z]`` blocked pair conditioning,
            or ``None`` when ``pair_biases`` is supplied.
        params: mapped parameters.
        no_heads: attention head count.
        n_query: query block height.
        n_key: key window width.
        mask: ``[..., N_atom]`` atom mask.
        inf: masking constant.
        mask_transition: upstream's ``_mask_trans``.
        eps: layer norm epsilon.
        pair_biases: optional precomputed per-block pair biases from
            :func:`atom_transformer_pair_biases`, shaped
            ``[no_blocks, ..., N_blocks, H, N_query, N_key]``.

    Returns:
        ``[..., N_atom, C_a]`` updated atom representation.
    """
    if pair_biases is None:
        if z is None:
            raise ValueError("either z or pair_biases is required")
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
                glu_backend=glu_backend,
            )

        # This stack runs inside the diffusion rollout, so its body is emitted
        # once per block *per rollout step* when unrolled. Scanning it keeps
        # the rollout body one block wide.
        if scan_blocks and can_scan(params.blocks):
            return scan_stack(one_block, a, params.blocks)
        for block in params.blocks:
            a = one_block(a, block)
        return a

    if z is not None:
        raise ValueError("pass either z or pair_biases, not both")
    n_blocks = len(params.blocks)
    if pair_biases.shape[0] != n_blocks:
        raise ValueError(
            f"pair_biases leading block axis {pair_biases.shape[0]} does not "
            f"match {n_blocks} atom transformer blocks"
        )
    if n_blocks == 0:
        return a

    def one_cached_block(
        current: jnp.ndarray,
        block: AtomTransformerBlockParams,
        block_pair_bias: jnp.ndarray,
    ):
        current = current + cross_attention_pair_bias(
            current,
            None,
            block.attention_pair_bias,
            no_heads=no_heads,
            n_query=n_query,
            n_key=n_key,
            mask=mask,
            s=s,
            inf=inf,
            eps=eps,
            pair_bias=block_pair_bias,
        )
        return current + conditioned_transition_block(
            current,
            s,
            block.conditioned_transition,
            mask=mask if mask_transition else None,
            eps=eps,
            glu_backend=glu_backend,
        )

    if scan_blocks and can_scan(params.blocks):
        stacked = stacked_or_stack(params.blocks)

        def scanned(state: jnp.ndarray, item):
            block, block_pair_bias = item
            return one_cached_block(state, block, block_pair_bias), None

        result, _ = jax.lax.scan(scanned, a, (stacked, pair_biases))
        return result
    for index, block in enumerate(params.blocks):
        a = one_cached_block(a, block, pair_biases[index])
    return a
