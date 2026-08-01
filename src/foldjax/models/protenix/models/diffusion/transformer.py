"""Transformer leaf blocks for the Protenix JAX port."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from foldjax.models._stacking import stacked_or_stack
from foldjax.models.protenix.models.primitives.attention import (
    AttentionPairBiasParams,
    attention_pair_bias,
    local_attention_pair_bias,
)
from foldjax.models.protenix.models.primitives.primitives import (
    AdaptiveLayerNormParams,
    LinearParams,
    adaptive_layer_norm,
    linear,
    sigmoid,
    silu,
)


class ConditionedTransitionParams(NamedTuple):
    """Parameters for ``ConditionedTransitionBlock``."""

    adaln: AdaptiveLayerNormParams
    linear_a1: LinearParams
    linear_a2: LinearParams
    linear_b: LinearParams
    linear_s: LinearParams


class DiffusionTransformerBlockParams(NamedTuple):
    """Parameters for one Protenix ``DiffusionTransformerBlock``."""

    attention_pair_bias: AttentionPairBiasParams
    conditioned_transition: ConditionedTransitionParams


class DiffusionTransformerStackParams(NamedTuple):
    """Parameters for a Protenix ``DiffusionTransformer`` stack."""

    blocks: tuple[DiffusionTransformerBlockParams, ...]


def conditioned_transition_block(
    a: jnp.ndarray,
    s: jnp.ndarray,
    params: ConditionedTransitionParams,
) -> jnp.ndarray:
    """Apply Protenix ``ConditionedTransitionBlock``."""

    a = adaptive_layer_norm(a, s, params.adaln)
    hidden = silu(linear(a, params.linear_a1)) * linear(a, params.linear_a2)
    return sigmoid(linear(s, params.linear_s)) * linear(hidden, params.linear_b)


_compiled_conditioned_transition_block = jax.jit(conditioned_transition_block)


def diffusion_transformer_block(
    a: jnp.ndarray,
    s: jnp.ndarray,
    z: jnp.ndarray,
    params: DiffusionTransformerBlockParams,
    *,
    num_heads: int,
    n_queries: int | None = None,
    n_keys: int | None = None,
    global_q_chunk_size: int | None = None,
    attention_backend: str = "xla",
    z_is_normalized: bool = False,
    extra_attn_bias: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Apply one inference-mode DiffusionTransformer block."""

    if n_queries is not None and n_keys is not None:
        if extra_attn_bias is not None:
            raise ValueError(
                "extra attention bias is only supported by global token attention"
            )
        attn_out = local_attention_pair_bias(
            a,
            s,
            z,
            params.attention_pair_bias,
            num_heads=num_heads,
            n_queries=n_queries,
            n_keys=n_keys,
            attention_backend=attention_backend,
        )
    else:
        attn_out = attention_pair_bias(
            a,
            s,
            z,
            params.attention_pair_bias,
            num_heads=num_heads,
            q_chunk_size=global_q_chunk_size,
            attention_backend=attention_backend,
            z_is_normalized=z_is_normalized,
            extra_attn_bias=extra_attn_bias,
        )
    a = a + attn_out
    transition_fn = (
        _compiled_conditioned_transition_block
        if attention_backend == "xla_jit"
        else conditioned_transition_block
    )
    return a + transition_fn(a, s, params.conditioned_transition)


def diffusion_transformer_stack(
    a: jnp.ndarray,
    s: jnp.ndarray,
    z: jnp.ndarray,
    params: DiffusionTransformerStackParams,
    *,
    num_heads: int,
    n_queries: int | None = None,
    n_keys: int | None = None,
    use_scan: bool = False,
    global_q_chunk_size: int | None = None,
    attention_backend: str = "xla",
    z_is_normalized: bool = False,
    extra_attn_bias: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Apply a DiffusionTransformer stack in inference mode."""

    if not params.blocks:
        raise ValueError("DiffusionTransformerStack requires at least one block")
    if not use_scan:
        for block in params.blocks:
            a = diffusion_transformer_block(
                a,
                s,
                z,
                block,
                num_heads=num_heads,
                n_queries=n_queries,
                n_keys=n_keys,
                global_q_chunk_size=global_q_chunk_size,
                attention_backend=attention_backend,
                z_is_normalized=z_is_normalized,
                extra_attn_bias=extra_attn_bias,
            )
        return a

    has_s = params.blocks[0].attention_pair_bias.has_s
    cross_attention_mode = params.blocks[0].attention_pair_bias.cross_attention_mode
    for block in params.blocks[1:]:
        block_attention = block.attention_pair_bias
        if (
            block_attention.has_s != has_s
            or block_attention.cross_attention_mode != cross_attention_mode
        ):
            raise ValueError(
                "scan requires homogeneous attention pair-bias configuration"
            )
    stacked = stack_diffusion_transformer_block_params(params.blocks)

    def body(a_carry, block_params):
        attention_params = block_params.attention_pair_bias._replace(
            has_s=has_s,
            cross_attention_mode=cross_attention_mode,
        )
        block_params = block_params._replace(attention_pair_bias=attention_params)
        return (
            diffusion_transformer_block(
                a_carry,
                s,
                z,
                block_params,
                num_heads=num_heads,
                n_queries=n_queries,
                n_keys=n_keys,
                global_q_chunk_size=global_q_chunk_size,
                attention_backend=attention_backend,
                z_is_normalized=z_is_normalized,
                extra_attn_bias=extra_attn_bias,
            ),
            None,
        )

    a, _ = jax.lax.scan(body, a, stacked)
    return a


def stack_diffusion_transformer_block_params(
    blocks: tuple[DiffusionTransformerBlockParams, ...],
) -> DiffusionTransformerBlockParams:
    """Stack DiffusionTransformer block params for ``lax.scan``."""

    if not blocks:
        raise ValueError("stack_diffusion_transformer_block_params requires blocks")
    return stacked_or_stack(blocks)

