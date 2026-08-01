"""Chai's 16-block conditioned diffusion transformer.

The exported TorchScript inlines every block.  Each block computes pair-biased
attention and its conditioned SwiGLU transition in parallel from the same
pre-residual single representation, then adds both updates together.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from foldjax.models._stacking import stacked_or_stack
from foldjax.models.chai.models.primitives import layer_norm, linear

NUM_HEADS = 16
HEAD_DIM = 48
MASK_FILL = -10000.0
CONDITIONED_NORM_EPS = 0.1


class DiffusionTransformerBlockParams(NamedTuple):
    q_bias: jnp.ndarray
    transition_norm_weight: jnp.ndarray
    transition_a_weight: jnp.ndarray
    transition_b_weight: jnp.ndarray
    transition_gate_weight: jnp.ndarray
    transition_gate_bias: jnp.ndarray
    attention_norm_weight: jnp.ndarray
    qkv_weight: jnp.ndarray
    attention_gate_weight: jnp.ndarray
    attention_gate_bias: jnp.ndarray
    pair_norm_weight: jnp.ndarray
    pair_norm_bias: jnp.ndarray
    pair_linear_weight: jnp.ndarray
    output_weight: jnp.ndarray


class DiffusionTransformerParams(NamedTuple):
    blocks: tuple[DiffusionTransformerBlockParams, ...]


def map_diffusion_transformer_block(
    state: Mapping[str, Any], prefix: str
) -> DiffusionTransformerBlockParams:
    """Map the 14 official tensors belonging to one block."""
    transition = f"{prefix}.transition"
    return DiffusionTransformerBlockParams(
        q_bias=jnp.asarray(state[f"{prefix}.q_bias"]),
        transition_norm_weight=jnp.asarray(
            state[f"{transition}.ada_ln.lin_s_merged.weight"]
        ),
        transition_a_weight=jnp.asarray(
            state[f"{transition}.linear_a_nobias_double.weight"]
        ),
        transition_b_weight=jnp.asarray(state[f"{transition}.linear_b_nobias.weight"]),
        transition_gate_weight=jnp.asarray(
            state[f"{transition}.linear_s_biasinit_m2.weight"]
        ),
        transition_gate_bias=jnp.asarray(
            state[f"{transition}.linear_s_biasinit_m2.bias"]
        ),
        attention_norm_weight=jnp.asarray(
            state[f"{prefix}.norm_in.lin_s_merged.weight"]
        ),
        qkv_weight=jnp.asarray(state[f"{prefix}.to_qkv.weight"]),
        attention_gate_weight=jnp.asarray(state[f"{prefix}.gate_proj.0.weight"]),
        attention_gate_bias=jnp.asarray(state[f"{prefix}.gate_proj.0.bias"]),
        pair_norm_weight=jnp.asarray(state[f"{prefix}.pair_layer_norm.weight"]),
        pair_norm_bias=jnp.asarray(state[f"{prefix}.pair_layer_norm.bias"]),
        pair_linear_weight=jnp.asarray(state[f"{prefix}.pair_linear.weight"]),
        output_weight=jnp.asarray(state[f"{prefix}.to_out.weight"]),
    )


def map_diffusion_transformer(
    state: Mapping[str, Any],
    prefix: str = "diffusion_transformer",
    *,
    num_blocks: int = 16,
) -> DiffusionTransformerParams:
    """Map the complete official 16-block, 224-tensor stack."""
    return DiffusionTransformerParams(
        blocks=tuple(
            map_diffusion_transformer_block(state, f"{prefix}.blocks.{index}")
            for index in range(num_blocks)
        )
    )


def _conditioned_norm(
    single: jnp.ndarray, cond: jnp.ndarray, weight: jnp.ndarray
) -> jnp.ndarray:
    normalized = layer_norm(single.astype(jnp.float32), eps=CONDITIONED_NORM_EPS)
    scale, shift = jnp.split(linear(cond, weight), 2, axis=-1)
    return normalized * (scale + 1.0) + shift


def _scaled_dot_product_attention(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    pair_bias: jnp.ndarray,
) -> jnp.ndarray:
    scale = 1.0 / jnp.sqrt(jnp.asarray(q.shape[-1], dtype=jnp.float32))
    logits = jnp.einsum(
        "...qd,...kd->...qk", q, k, precision=jax.lax.Precision.HIGHEST
    ) * scale
    probabilities = jax.nn.softmax(logits + pair_bias, axis=-1)
    return jnp.einsum(
        "...qk,...kd->...qd",
        probabilities,
        v,
        precision=jax.lax.Precision.HIGHEST,
    )


def _attention_update(
    single: jnp.ndarray,
    cond: jnp.ndarray,
    pair: jnp.ndarray,
    token_mask: jnp.ndarray,
    params: DiffusionTransformerBlockParams,
    query_chunk_size: int | None,
) -> jnp.ndarray:
    pair_mask = token_mask[:, :, None] & token_mask[:, None, :]
    pair_bias = diffusion_transformer_pair_bias(pair, pair_mask, params)
    return diffusion_transformer_attention_with_bias(
        single, cond, pair_bias, params, query_chunk_size=query_chunk_size
    )


def diffusion_transformer_pair_bias(
    pair: jnp.ndarray,
    pair_mask: jnp.ndarray,
    params: DiffusionTransformerBlockParams,
) -> jnp.ndarray:
    """Project full or row-chunked pair features to attention bias."""

    pair_normalized = layer_norm(
        pair.astype(jnp.float32),
        params.pair_norm_weight,
        params.pair_norm_bias,
    )
    pair_bias = linear(pair_normalized, params.pair_linear_weight)
    pair_bias = jnp.moveaxis(pair_bias, -1, 1)[:, :, None]
    pair_bias = jnp.where(pair_mask[:, None, None], pair_bias, MASK_FILL)
    return pair_bias


def diffusion_transformer_attention_with_bias(
    single: jnp.ndarray,
    cond: jnp.ndarray,
    pair_bias: jnp.ndarray,
    params: DiffusionTransformerBlockParams,
    *,
    query_chunk_size: int | None,
) -> jnp.ndarray:
    """Compute the attention branch from a preprojected official pair bias."""

    batch, samples, tokens, _ = single.shape

    normalized = _conditioned_norm(single, cond, params.attention_norm_weight)
    qkv = linear(normalized, params.qkv_weight)
    qkv = qkv.reshape(batch, samples, tokens, NUM_HEADS, 3 * HEAD_DIM)
    qkv = jnp.transpose(qkv, (0, 3, 1, 2, 4))
    q, k, v = jnp.split(qkv, 3, axis=-1)
    q = q + params.q_bias.reshape(1, NUM_HEADS, 1, 1, HEAD_DIM)

    if query_chunk_size is None or query_chunk_size >= tokens:
        attended = _scaled_dot_product_attention(q, k, v, pair_bias)
    else:
        chunks = []
        for start in range(0, tokens, query_chunk_size):
            stop = min(start + query_chunk_size, tokens)
            chunks.append(
                _scaled_dot_product_attention(
                    q[..., start:stop, :],
                    k,
                    v,
                    pair_bias[..., start:stop, :],
                )
            )
        attended = jnp.concatenate(chunks, axis=-2)

    attended = jnp.transpose(attended, (0, 2, 3, 1, 4)).reshape(
        batch, samples, tokens, NUM_HEADS * HEAD_DIM
    )
    attended = linear(attended, params.output_weight)
    gate = jax.nn.sigmoid(
        linear(
            cond,
            params.attention_gate_weight,
            params.attention_gate_bias,
        )
    )
    return gate * attended


def _transition_update(
    single: jnp.ndarray,
    cond: jnp.ndarray,
    params: DiffusionTransformerBlockParams,
) -> jnp.ndarray:
    normalized = _conditioned_norm(single, cond, params.transition_norm_weight)
    value, gate = jnp.split(linear(normalized, params.transition_a_weight), 2, axis=-1)
    hidden = jax.nn.silu(value) * gate
    update = linear(hidden, params.transition_b_weight)
    output_gate = jax.nn.sigmoid(
        linear(
            cond,
            params.transition_gate_weight,
            params.transition_gate_bias,
        )
    )
    return output_gate * update


def diffusion_transformer_block(
    single: jnp.ndarray,
    cond: jnp.ndarray,
    pair: jnp.ndarray,
    token_mask: jnp.ndarray,
    params: DiffusionTransformerBlockParams,
    *,
    query_chunk_size: int | None = None,
) -> jnp.ndarray:
    """Apply one official block with a static optional query chunk size."""
    if query_chunk_size is not None and query_chunk_size <= 0:
        raise ValueError("query_chunk_size must be positive")
    attention = _attention_update(
        single, cond, pair, token_mask, params, query_chunk_size
    )
    transition = _transition_update(single, cond, params)
    return single + attention + transition


def diffusion_transformer_stack(
    single: jnp.ndarray,
    cond: jnp.ndarray,
    pair: jnp.ndarray,
    token_mask: jnp.ndarray,
    params: DiffusionTransformerParams,
    *,
    query_chunk_size: int | None = None,
    use_scan: bool = True,
) -> jnp.ndarray:
    """Apply all mapped blocks in checkpoint order."""
    if not params.blocks:
        raise ValueError("diffusion transformer requires at least one block")

    def one_block(
        current: jnp.ndarray, block: DiffusionTransformerBlockParams
    ) -> jnp.ndarray:
        return diffusion_transformer_block(
            current,
            cond,
            pair,
            token_mask,
            block,
            query_chunk_size=query_chunk_size,
        )

    # Scanned rather than unrolled, matching ``_pairformer_stack`` in trunk.py: the
    # released stack is 16 blocks, and unrolling puts 16 copies of the body in one
    # HLO module for XLA to schedule and compile as a whole. Same arithmetic.
    if use_scan and len(params.blocks) > 1:
        stacked = stacked_or_stack(params.blocks)
        return jax.lax.scan(
            lambda carry, block: (one_block(carry, block), None),
            single,
            stacked,
        )[0]
    for block in params.blocks:
        single = one_block(single, block)
    return single
