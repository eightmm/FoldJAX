"""Chai trunk MSA stack recovered from ``trunk.pt`` ``forward_256``.

Chai uses four MSA blocks.  Every block first adds an outer-product update to
the pair representation and then applies a pair-only block.  The first three
blocks also update the MSA representation with parallel transition and
pair-weighted-averaging branches; the fourth block discards the MSA state.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from foldjax.models.chai.models.pairformer import (
    PairformerPairBlockParams,
    PairformerTransitionParams,
    fused_triangle_attention,
    fused_triangle_multiplication,
    map_fused_triangle_attention,
    map_fused_triangle_multiplication,
    map_pairformer_transition,
    pairformer_transition,
)
from foldjax.models.chai.models.primitives import layer_norm, linear_bf16


class OuterProductMeanParams(NamedTuple):
    weight_ab: jnp.ndarray
    output_norm_weight: jnp.ndarray
    output_norm_bias: jnp.ndarray
    output_weight: jnp.ndarray
    output_bias: jnp.ndarray


class MSAPairWeightedAveragingParams(NamedTuple):
    msa_norm_weight: jnp.ndarray
    msa_norm_bias: jnp.ndarray
    msa_value_gate_weight: jnp.ndarray
    pair_norm_weight: jnp.ndarray
    pair_norm_bias: jnp.ndarray
    pair_weight: jnp.ndarray
    output_weight: jnp.ndarray


class MSABlockParams(NamedTuple):
    outer_product_mean: OuterProductMeanParams
    pair: PairformerPairBlockParams
    weighted_averaging: MSAPairWeightedAveragingParams | None
    msa_transition: PairformerTransitionParams | None


class MSAModuleParams(NamedTuple):
    linear_s2m_weight: jnp.ndarray
    blocks: tuple[MSABlockParams, ...]


def _base(prefix: str) -> str:
    return f"{prefix}." if prefix else ""


def map_outer_product_mean(
    state: Mapping[str, Any], prefix: str
) -> OuterProductMeanParams:
    base = _base(prefix)
    return OuterProductMeanParams(
        weight_ab=jnp.asarray(state[f"{base}weight_ab"]),
        output_norm_weight=jnp.asarray(state[f"{base}ln_out.weight"]),
        output_norm_bias=jnp.asarray(state[f"{base}ln_out.bias"]),
        output_weight=jnp.asarray(state[f"{base}linear_out.weight"]),
        output_bias=jnp.asarray(state[f"{base}linear_out.bias"]),
    )


def map_msa_pair_weighted_averaging(
    state: Mapping[str, Any], prefix: str
) -> MSAPairWeightedAveragingParams:
    base = _base(prefix)
    return MSAPairWeightedAveragingParams(
        msa_norm_weight=jnp.asarray(state[f"{base}layernorm_msa.weight"]),
        msa_norm_bias=jnp.asarray(state[f"{base}layernorm_msa.bias"]),
        msa_value_gate_weight=jnp.asarray(
            state[f"{base}linear_msa2vg.weight"]
        ),
        pair_norm_weight=jnp.asarray(state[f"{base}layernorm_pair.weight"]),
        pair_norm_bias=jnp.asarray(state[f"{base}layernorm_pair.bias"]),
        pair_weight=jnp.asarray(state[f"{base}linear_pair.weight"]),
        output_weight=jnp.asarray(state[f"{base}linear_out_no_bias.weight"]),
    )


def _map_msa_pair_block(
    state: Mapping[str, Any], prefix: str, index: int
) -> PairformerPairBlockParams:
    return PairformerPairBlockParams(
        transition_pair=map_pairformer_transition(
            state, f"{prefix}.pair_transition.{index}"
        ),
        triangle_multiplication=map_fused_triangle_multiplication(
            state, f"{prefix}.triangular_multiplication.{index}"
        ),
        triangle_attention=map_fused_triangle_attention(
            state, f"{prefix}.triangular_attention.{index}"
        ),
    )


def map_msa_module(
    state: Mapping[str, Any], prefix: str = "msa_module"
) -> MSAModuleParams:
    """Map all 110 tensors in the official Chai MSA module."""
    blocks = []
    for index in range(4):
        weighted = None
        transition = None
        if index < 3:
            weighted = map_msa_pair_weighted_averaging(
                state, f"{prefix}.msa_pair_weighted_averaging.{index}"
            )
            transition = map_pairformer_transition(
                state, f"{prefix}.msa_transition.{index}"
            )
        blocks.append(
            MSABlockParams(
                outer_product_mean=map_outer_product_mean(
                    state, f"{prefix}.outer_product_mean.{index}"
                ),
                pair=_map_msa_pair_block(state, prefix, index),
                weighted_averaging=weighted,
                msa_transition=transition,
            )
        )
    return MSAModuleParams(
        linear_s2m_weight=jnp.asarray(state[f"{prefix}.linear_s2m.weight"]),
        blocks=tuple(blocks),
    )


def _outer_product_chunk(
    msa: jnp.ndarray,
    msa_mask: jnp.ndarray,
    weight_ab: jnp.ndarray,
) -> jnp.ndarray:
    normalized = layer_norm(msa.astype(jnp.float32))
    normalized = normalized * msa_mask[..., None].astype(normalized.dtype)
    normalized = normalized.astype(jnp.bfloat16)
    projection_shape = weight_ab.shape[1:3]
    projection_width = projection_shape[0] * projection_shape[1]
    first = linear_bf16(
        normalized, weight_ab[0].reshape(projection_width, -1)
    ).reshape(*normalized.shape[:-1], *projection_shape)
    second = linear_bf16(
        normalized, weight_ab[1].reshape(projection_width, -1)
    ).reshape(*normalized.shape[:-1], *projection_shape)
    output = jnp.einsum(
        "briax,brjay->bijaxy",
        first,
        second,
        preferred_element_type=jnp.float32,
    ).astype(jnp.bfloat16)
    return output.reshape(output.shape[:3] + (-1,))


def outer_product_mean(
    msa: jnp.ndarray,
    msa_mask: jnp.ndarray,
    params: OuterProductMeanParams,
    *,
    chunk_size: int = 4096,
) -> jnp.ndarray:
    """Return Chai's masked MSA outer-product update.

    The exported graph sums four 4096-row chunks.  Smaller MSA buckets use the
    same math with only their non-empty chunks.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    output_shape = (msa.shape[0], msa.shape[2], msa.shape[2], 512)
    if msa.shape[1] >= chunk_size and msa.shape[1] % chunk_size == 0:
        num_chunks = msa.shape[1] // chunk_size
        msa_chunks = jnp.moveaxis(
            msa.reshape(
                msa.shape[0], num_chunks, chunk_size, msa.shape[2], msa.shape[3]
            ),
            1,
            0,
        )
        mask_chunks = jnp.moveaxis(
            msa_mask.reshape(
                msa_mask.shape[0], num_chunks, chunk_size, msa_mask.shape[2]
            ),
            1,
            0,
        )

        def accumulate(current, inputs):
            value, mask = inputs
            return current + _outer_product_chunk(
                value, mask, params.weight_ab
            ), None

        accumulated, _ = jax.lax.scan(
            accumulate,
            jnp.zeros(output_shape, dtype=jnp.bfloat16),
            (msa_chunks, mask_chunks),
        )
    else:
        accumulated = jnp.zeros(output_shape, dtype=jnp.bfloat16)
        for start in range(0, msa.shape[1], chunk_size):
            stop = min(start + chunk_size, msa.shape[1])
            accumulated += _outer_product_chunk(
                msa[:, start:stop],
                msa_mask[:, start:stop],
                params.weight_ab,
            )
    normalized = layer_norm(
        accumulated.astype(jnp.float32),
        params.output_norm_weight,
        params.output_norm_bias,
        eps=0.1,
    )
    return linear_bf16(
        normalized, params.output_weight, params.output_bias
    )


def msa_pair_weighted_averaging(
    msa: jnp.ndarray,
    pair: jnp.ndarray,
    msa_mask: jnp.ndarray,
    pair_mask: jnp.ndarray,
    params: MSAPairWeightedAveragingParams,
    *,
    chunk_size: int = 8192,
) -> jnp.ndarray:
    """Return Chai's pair-conditioned MSA update without its residual."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    pair_normalized = layer_norm(
        pair.astype(jnp.float32),
        params.pair_norm_weight,
        params.pair_norm_bias,
    )
    logits = linear_bf16(pair_normalized, params.pair_weight)
    logits = jnp.moveaxis(logits, -1, 1)
    logits = jnp.where(pair_mask[:, None], logits, -10000.0)
    weights = jax.nn.softmax(logits.astype(jnp.float32), axis=-1).astype(
        jnp.bfloat16
    )
    num_heads = params.pair_weight.shape[0]
    def process_chunk(value, mask):
        value = layer_norm(
            value.astype(jnp.float32),
            params.msa_norm_weight,
            params.msa_norm_bias,
        )
        value_gate = linear_bf16(value, params.msa_value_gate_weight)
        head_dim = value_gate.shape[-1] // (2 * num_heads)
        value_gate = value_gate.reshape(
            *value_gate.shape[:-1], 2, num_heads, head_dim
        )
        value, gate = jnp.moveaxis(value_gate, -3, 0)
        value = value * mask[..., None, None].astype(value.dtype)
        attended = jnp.einsum(
            "abcd,aedbf->aecbf",
            weights,
            value,
            preferred_element_type=jnp.float32,
        ).astype(jnp.bfloat16)
        attended = jax.nn.sigmoid(gate) * attended
        attended = attended.reshape(attended.shape[:-2] + (-1,))
        return linear_bf16(attended, params.output_weight)

    if msa.shape[1] >= chunk_size and msa.shape[1] % chunk_size == 0:
        num_chunks = msa.shape[1] // chunk_size
        msa_chunks = jnp.moveaxis(
            msa.reshape(
                msa.shape[0], num_chunks, chunk_size, msa.shape[2], msa.shape[3]
            ),
            1,
            0,
        )
        mask_chunks = jnp.moveaxis(
            msa_mask.reshape(
                msa_mask.shape[0], num_chunks, chunk_size, msa_mask.shape[2]
            ),
            1,
            0,
        )

        def scan_chunk(_, inputs):
            value, mask = inputs
            return None, process_chunk(value, mask)

        _, updates = jax.lax.scan(scan_chunk, None, (msa_chunks, mask_chunks))
        return jnp.moveaxis(updates, 0, 1).reshape(msa.shape)

    updates = [
        process_chunk(
            msa[:, start : min(start + chunk_size, msa.shape[1])],
            msa_mask[:, start : min(start + chunk_size, msa.shape[1])],
        )
        for start in range(0, msa.shape[1], chunk_size)
    ]
    return jnp.concatenate(updates, axis=1) if updates else jnp.zeros_like(msa)


def msa_transition(
    msa: jnp.ndarray,
    params: PairformerTransitionParams,
    *,
    chunk_size: int = 8192,
) -> jnp.ndarray:
    """Apply the row-independent MSA transition with bounded intermediates."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if msa.shape[1] >= chunk_size and msa.shape[1] % chunk_size == 0:
        num_chunks = msa.shape[1] // chunk_size
        chunks = jnp.moveaxis(
            msa.reshape(
                msa.shape[0], num_chunks, chunk_size, msa.shape[2], msa.shape[3]
            ),
            1,
            0,
        )

        def scan_chunk(_, value):
            return None, pairformer_transition(value, params)

        _, updates = jax.lax.scan(scan_chunk, None, chunks)
        return jnp.moveaxis(updates, 0, 1).reshape(msa.shape)
    updates = [
        pairformer_transition(
            msa[:, start : min(start + chunk_size, msa.shape[1])], params
        )
        for start in range(0, msa.shape[1], chunk_size)
    ]
    return jnp.concatenate(updates, axis=1) if updates else jnp.zeros_like(msa)


def _msa_pair_block(
    pair: jnp.ndarray,
    pair_mask: jnp.ndarray,
    params: PairformerPairBlockParams,
) -> jnp.ndarray:
    """Pair-only block order used specifically inside Chai's MSA stack."""
    intermediate = pair + fused_triangle_multiplication(
        pair, pair_mask, params.triangle_multiplication
    )
    intermediate += pairformer_transition(pair, params.transition_pair)
    return intermediate + fused_triangle_attention(
        intermediate, pair_mask, params.triangle_attention
    )


def msa_module(
    msa_features: jnp.ndarray,
    msa_mask: jnp.ndarray,
    single: jnp.ndarray,
    pair: jnp.ndarray,
    pair_mask: jnp.ndarray,
    params: MSAModuleParams,
    *,
    outer_product_chunk_size: int = 4096,
    weighted_averaging_chunk_size: int = 8192,
    transition_chunk_size: int = 8192,
) -> jnp.ndarray:
    """Apply the official four-block MSA stack and return the pair state."""
    msa = msa_features + linear_bf16(single, params.linear_s2m_weight)[:, None]
    return msa_module_from_embedded(
        msa,
        msa_mask,
        pair,
        pair_mask,
        params,
        outer_product_chunk_size=outer_product_chunk_size,
        weighted_averaging_chunk_size=weighted_averaging_chunk_size,
        transition_chunk_size=transition_chunk_size,
    )


def msa_module_from_embedded(
    msa: jnp.ndarray,
    msa_mask: jnp.ndarray,
    pair: jnp.ndarray,
    pair_mask: jnp.ndarray,
    params: MSAModuleParams,
    *,
    outer_product_chunk_size: int = 4096,
    weighted_averaging_chunk_size: int = 8192,
    transition_chunk_size: int = 8192,
) -> jnp.ndarray:
    """Apply four MSA blocks to an already single-conditioned MSA tensor."""
    if len(params.blocks) != 4:
        raise ValueError("Chai MSA module requires exactly four blocks")
    for index, block in enumerate(params.blocks):
        pair = pair + outer_product_mean(
            msa,
            msa_mask,
            block.outer_product_mean,
            chunk_size=outer_product_chunk_size,
        )
        if index < 3:
            if block.msa_transition is None or block.weighted_averaging is None:
                raise ValueError("first three MSA blocks require MSA updates")
            msa_input = msa
            msa = msa_input + msa_transition(
                msa_input,
                block.msa_transition,
                chunk_size=transition_chunk_size,
            )
            msa += msa_pair_weighted_averaging(
                msa_input,
                pair,
                msa_mask,
                pair_mask,
                block.weighted_averaging,
                chunk_size=weighted_averaging_chunk_size,
            )
        pair = _msa_pair_block(pair, pair_mask, block.pair)
    return pair
