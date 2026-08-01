"""Pure JAX port of the official Chai-1 ``confidence_head.pt`` component.

The component is a four-block Pairformer confidence stack. Its parameter
packing and parallel residual order are specific to Chai and were recovered
from the official ``forward_256`` TorchScript graph; no Protenix/Boltz model
code is assumed to be graph-equivalent here.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from foldjax.models._stacking import stacked_or_stack
from foldjax.models.chai.models.primitives import layer_norm, linear_bf16

NUM_BLOCKS = 4
NUM_TRIANGLE_HEADS = 4
TRIANGLE_HEAD_DIM = 64
NUM_SINGLE_HEADS = 16
SINGLE_HEAD_DIM = 24
NUM_ATOM_SLOTS = 37
NUM_PLDDT_BINS = 50
MASK_FILL = -10000.0
TRIANGLE_QUERY_CHUNK_SIZE = 16
TRIANGLE_KEY_CHUNK_SIZE = 128


class TransitionParams(NamedTuple):
    norm_weight: jnp.ndarray
    norm_bias: jnp.ndarray
    linear_ab_weight: jnp.ndarray
    linear_out_weight: jnp.ndarray


class TriangleMultiplicationParams(NamedTuple):
    norm_weight: jnp.ndarray
    norm_bias: jnp.ndarray
    linear_p_weight: jnp.ndarray
    linear_g_weight: jnp.ndarray
    linear_out_weight: jnp.ndarray


class TriangleAttentionParams(NamedTuple):
    norm_weight: jnp.ndarray
    norm_bias: jnp.ndarray
    qkvgb_weight: jnp.ndarray
    linear_out_weight: jnp.ndarray


class AttentionPairBiasParams(NamedTuple):
    single_norm_weight: jnp.ndarray
    single_norm_bias: jnp.ndarray
    pair_norm_weight: jnp.ndarray
    pair_norm_bias: jnp.ndarray
    pair_linear_weight: jnp.ndarray
    query_bias: jnp.ndarray
    qkvg_weight: jnp.ndarray
    output_weight: jnp.ndarray


class ConfidenceBlockParams(NamedTuple):
    transition_pair: TransitionParams
    triangle_multiplication: TriangleMultiplicationParams
    triangle_attention: TriangleAttentionParams
    transition_single: TransitionParams
    attention_pair_bias: AttentionPairBiasParams


class ConfidenceHeadParams(NamedTuple):
    atom_distance_bins: jnp.ndarray
    single_to_pair_weight: jnp.ndarray
    atom_distance_projection_weight: jnp.ndarray
    blocks: tuple[ConfidenceBlockParams, ...]
    plddt_projection_weight: jnp.ndarray
    pae_projection_weight: jnp.ndarray
    pde_projection_weight: jnp.ndarray


def _asarray(value: Any) -> jnp.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return jnp.asarray(np.asarray(value))


def map_confidence_head(state: Mapping[str, Any]) -> ConfidenceHeadParams:
    """Map all 106 official tensors exactly once and reject extras/missing keys."""

    consumed: set[str] = set()

    def take(key: str) -> jnp.ndarray:
        if key in consumed:
            raise ValueError(f"confidence tensor consumed more than once: {key}")
        if key not in state:
            raise KeyError(f"missing confidence tensor: {key}")
        consumed.add(key)
        return _asarray(state[key])

    blocks = []
    for index in range(NUM_BLOCKS):
        prefix = f"blocks.{index}"
        pair = f"{prefix}.transition_pair"
        multiplication = f"{prefix}.triangle_multiplication"
        triangle_attention = f"{prefix}.triangle_attention"
        single = f"{prefix}.transition_single"
        attention = f"{prefix}.attention_pair_bias"
        blocks.append(
            ConfidenceBlockParams(
                transition_pair=TransitionParams(
                    take(f"{pair}.layer_norm.weight"),
                    take(f"{pair}.layer_norm.bias"),
                    take(f"{pair}.linear_no_bias_ab.weight"),
                    take(f"{pair}.linear_out.weight"),
                ),
                triangle_multiplication=TriangleMultiplicationParams(
                    take(f"{multiplication}.layernorm_z_in.weight"),
                    take(f"{multiplication}.layernorm_z_in.bias"),
                    take(f"{multiplication}.merged_linear_p.weight"),
                    take(f"{multiplication}.merged_linear_g.weight"),
                    take(f"{multiplication}.linear_z_out.weight"),
                ),
                triangle_attention=TriangleAttentionParams(
                    take(f"{triangle_attention}.pair_layer_norm.weight"),
                    take(f"{triangle_attention}.pair_layer_norm.bias"),
                    take(f"{triangle_attention}.pair2qkvgb.weight"),
                    take(f"{triangle_attention}.linear_out.weight"),
                ),
                transition_single=TransitionParams(
                    take(f"{single}.layer_norm.weight"),
                    take(f"{single}.layer_norm.bias"),
                    take(f"{single}.linear_no_bias_ab.weight"),
                    take(f"{single}.linear_out.weight"),
                ),
                attention_pair_bias=AttentionPairBiasParams(
                    take(f"{attention}.single_layer_norm.weight"),
                    take(f"{attention}.single_layer_norm.bias"),
                    take(f"{attention}.pair_layer_norm.weight"),
                    take(f"{attention}.pair_layer_norm.bias"),
                    take(f"{attention}.pair_linear.weight"),
                    take(f"{attention}.attention.query_bias"),
                    take(f"{attention}.attention.input2qkvg.weight"),
                    take(f"{attention}.attention.output_proj.weight"),
                ),
            )
        )

    params = ConfidenceHeadParams(
        atom_distance_bins=take("atom_distance_v_bins"),
        single_to_pair_weight=take("single_to_pair_proj.weight"),
        atom_distance_projection_weight=take("atom_distance_bins_projection.weight"),
        blocks=tuple(blocks),
        plddt_projection_weight=take("plddt_projection.weight"),
        pae_projection_weight=take("pae_projection.weight"),
        pde_projection_weight=take("pde_projection.weight"),
    )
    extras = set(state) - consumed
    if extras:
        raise ValueError(f"unexpected confidence tensors: {sorted(extras)}")
    if len(consumed) != 106:
        raise ValueError(f"expected 106 confidence tensors, consumed {len(consumed)}")
    return params


def _transition(x: jnp.ndarray, params: TransitionParams) -> jnp.ndarray:
    normalized = layer_norm(x.astype(jnp.float32), params.norm_weight, params.norm_bias)
    ab = linear_bf16(normalized, params.linear_ab_weight)
    a, b = jnp.split(ab, 2, axis=-1)
    product = jax.nn.silu(a.astype(jnp.float32)).astype(jnp.bfloat16) * b
    return linear_bf16(product, params.linear_out_weight)


def confidence_transition_projection(
    x: jnp.ndarray, params: TransitionParams, *, second: bool
) -> jnp.ndarray:
    """Compute one confidence-transition projection for row staging."""

    normalized = layer_norm(x.astype(jnp.float32), params.norm_weight, params.norm_bias)
    split = params.linear_ab_weight.shape[0] // 2
    weight = (
        params.linear_ab_weight[split:] if second else params.linear_ab_weight[:split]
    )
    projected = linear_bf16(normalized, weight)
    return (
        projected
        if second
        else jax.nn.silu(projected.astype(jnp.float32)).astype(jnp.bfloat16)
    )


def confidence_transition_output(
    first: jnp.ndarray, second: jnp.ndarray, params: TransitionParams
) -> jnp.ndarray:
    """Finish a staged confidence transition without changing its arithmetic."""

    return linear_bf16(first * second, params.linear_out_weight)


def _triangle_multiplication(
    z: jnp.ndarray,
    mask: jnp.ndarray,
    params: TriangleMultiplicationParams,
) -> jnp.ndarray:
    channels = z.shape[-1]
    normalized = layer_norm(z.astype(jnp.float32), params.norm_weight, params.norm_bias)
    projected = linear_bf16(normalized, params.linear_p_weight)
    gates = jax.nn.sigmoid(
        linear_bf16(normalized, params.linear_g_weight).astype(jnp.float32)
    ).astype(jnp.bfloat16)
    ab = projected * gates[..., :-channels]
    output_gate = gates[..., -channels:]
    outgoing, incoming = jnp.split(ab, 2, axis=-1)
    outgoing = jnp.where(mask[..., None], outgoing, 0)
    incoming_mask = jnp.swapaxes(mask, -1, -2)[..., None]
    incoming = jnp.where(incoming_mask, incoming, 0)
    out_a, out_b = jnp.split(outgoing, 2, axis=-1)
    in_a, in_b = jnp.split(incoming, 2, axis=-1)
    out_update = jnp.einsum(
        "...ikd,...jkd->...ijd",
        out_a,
        out_b,
    )
    in_update = jnp.einsum(
        "...kid,...kjd->...ijd",
        in_a,
        in_b,
    )
    combined = layer_norm(out_update.astype(jnp.float32)) + layer_norm(
        in_update.astype(jnp.float32)
    )
    return linear_bf16(combined, params.linear_out_weight) * output_gate


def confidence_triangle_multiplication_direction_tile(
    z: jnp.ndarray,
    mask: jnp.ndarray,
    params: TriangleMultiplicationParams,
    *,
    incoming: bool,
    start: int,
    size: int,
    column_start: int,
    column_size: int,
) -> jnp.ndarray:
    """Compute one full-k confidence triangle contraction output tile."""

    channels = z.shape[-1]
    offset = 2 if incoming else 0

    def project(value: jnp.ndarray, index: int) -> jnp.ndarray:
        normalized = layer_norm(
            value.astype(jnp.float32), params.norm_weight, params.norm_bias
        )
        begin = index * channels
        end = begin + channels
        projected = linear_bf16(normalized, params.linear_p_weight[begin:end])
        gate = jax.nn.sigmoid(
            linear_bf16(normalized, params.linear_g_weight[begin:end]).astype(
                jnp.float32
            )
        ).astype(jnp.bfloat16)
        return projected * gate

    if incoming:
        a_input = jax.lax.dynamic_slice_in_dim(z, start, size, axis=-2)
        b_input = jax.lax.dynamic_slice_in_dim(z, column_start, column_size, axis=-2)
        transposed_mask = jnp.swapaxes(mask, -1, -2)
        a_mask = jax.lax.dynamic_slice_in_dim(transposed_mask, start, size, axis=-1)
        b_mask = jax.lax.dynamic_slice_in_dim(
            transposed_mask, column_start, column_size, axis=-1
        )
        a = jnp.where(a_mask[..., None], project(a_input, offset), 0)
        b = jnp.where(b_mask[..., None], project(b_input, offset + 1), 0)
        update = jnp.einsum("...kid,...kjd->...ijd", a, b)
    else:
        a_input = jax.lax.dynamic_slice_in_dim(z, start, size, axis=-3)
        b_input = jax.lax.dynamic_slice_in_dim(z, column_start, column_size, axis=-3)
        a_mask = jax.lax.dynamic_slice_in_dim(mask, start, size, axis=-2)
        b_mask = jax.lax.dynamic_slice_in_dim(mask, column_start, column_size, axis=-2)
        a = jnp.where(a_mask[..., None], project(a_input, offset), 0)
        b = jnp.where(b_mask[..., None], project(b_input, offset + 1), 0)
        update = jnp.einsum("...ikd,...jkd->...ijd", a, b)
    return layer_norm(update.astype(jnp.float32))


def confidence_triangle_multiplication_output_tile(
    z: jnp.ndarray,
    outgoing: jnp.ndarray,
    incoming: jnp.ndarray,
    params: TriangleMultiplicationParams,
) -> jnp.ndarray:
    """Apply the official output gate/projection to one contraction tile."""

    channels = z.shape[-1]
    normalized = layer_norm(z.astype(jnp.float32), params.norm_weight, params.norm_bias)
    output_gate = jax.nn.sigmoid(
        linear_bf16(
            normalized, params.linear_g_weight[4 * channels : 5 * channels]
        ).astype(jnp.float32)
    ).astype(jnp.bfloat16)
    combined = outgoing + incoming
    return linear_bf16(combined, params.linear_out_weight) * output_gate


def _sdpa(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    bias: jnp.ndarray,
) -> jnp.ndarray:
    logits = _attention_logits(q, k, bias)
    probabilities = jax.nn.softmax(logits, axis=-1)
    return jnp.einsum(
        "...qk,...kd->...qd",
        probabilities,
        v.astype(jnp.float32),
    ).astype(jnp.bfloat16)


def _attention_logits(
    q: jnp.ndarray,
    k: jnp.ndarray,
    bias: jnp.ndarray,
) -> jnp.ndarray:
    return jnp.einsum(
        "...qd,...kd->...qk",
        q.astype(jnp.float32),
        k.astype(jnp.float32),
    ) / jnp.sqrt(jnp.asarray(q.shape[-1], dtype=jnp.float32)) + bias.astype(jnp.float32)


def _chunked_sdpa(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    bias: jnp.ndarray,
    *,
    query_chunk_size: int,
    key_chunk_size: int = TRIANGLE_KEY_CHUNK_SIZE,
) -> jnp.ndarray:
    """Evaluate SDPA with bounded logits and online fp32 softmax reduction."""
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4 or bias.ndim != 4:
        raise ValueError("chunked triangle attention expects four-dimensional inputs")
    query_count = q.shape[-2]
    key_count = k.shape[-2]
    if query_count <= query_chunk_size and key_count <= key_chunk_size:
        return _sdpa(q, k, v, bias)
    if query_count % query_chunk_size:
        raise ValueError(
            f"query count {query_count} must be divisible by chunk size "
            f"{query_chunk_size}"
        )
    if bias.shape[-2] != query_count:
        raise ValueError("attention bias query axis must match q")
    if key_count % key_chunk_size:
        raise ValueError(
            f"key count {key_count} must be divisible by chunk size {key_chunk_size}"
        )
    if bias.shape[-1] != key_count:
        raise ValueError("attention bias key axis must match k")

    groups, outer, _, query_dim = q.shape
    value_dim = v.shape[-1]
    query_chunks = query_count // query_chunk_size
    key_chunks = key_count // key_chunk_size
    output = jnp.zeros((groups, outer, query_count, value_dim), jnp.bfloat16)

    def query_body(query_index: int, result: jnp.ndarray) -> jnp.ndarray:
        query_start = query_index * query_chunk_size
        q_chunk = jax.lax.dynamic_slice(
            q,
            (0, 0, query_start, 0),
            (groups, outer, query_chunk_size, query_dim),
        )
        bias_query = jax.lax.dynamic_slice(
            bias,
            (0, 0, query_start, 0),
            (groups, bias.shape[1], query_chunk_size, key_count),
        )
        running_max = jnp.full((groups, outer, query_chunk_size), -jnp.inf, jnp.float32)
        denominator = jnp.zeros((groups, outer, query_chunk_size), jnp.float32)
        numerator = jnp.zeros((groups, outer, query_chunk_size, value_dim), jnp.float32)

        def key_body(
            key_index: int,
            state: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
        ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
            old_max, old_denominator, old_numerator = state
            key_start = key_index * key_chunk_size
            k_chunk = jax.lax.dynamic_slice(
                k,
                (0, 0, key_start, 0),
                (groups, outer, key_chunk_size, query_dim),
            )
            v_chunk = jax.lax.dynamic_slice(
                v,
                (0, 0, key_start, 0),
                (groups, outer, key_chunk_size, value_dim),
            )
            bias_chunk = jax.lax.dynamic_slice(
                bias_query,
                (0, 0, 0, key_start),
                (groups, bias.shape[1], query_chunk_size, key_chunk_size),
            )
            logits = _attention_logits(q_chunk, k_chunk, bias_chunk)
            chunk_max = jnp.max(logits, axis=-1)
            new_max = jnp.maximum(old_max, chunk_max)
            old_scale = jnp.exp(old_max - new_max)
            probabilities = jnp.exp(logits - new_max[..., None])
            new_denominator = old_scale * old_denominator + jnp.sum(
                probabilities, axis=-1
            )
            weighted_values = jnp.einsum(
                "...qk,...kd->...qd",
                probabilities,
                v_chunk.astype(jnp.float32),
            )
            new_numerator = old_scale[..., None] * old_numerator + weighted_values
            return new_max, new_denominator, new_numerator

        _, denominator, numerator = jax.lax.fori_loop(
            0,
            key_chunks,
            key_body,
            (running_max, denominator, numerator),
        )
        attended = (numerator / denominator[..., None]).astype(jnp.bfloat16)
        return jax.lax.dynamic_update_slice(result, attended, (0, 0, query_start, 0))

    return jax.lax.fori_loop(0, query_chunks, query_body, output)


def triangle_attention_prepare(
    z: jnp.ndarray,
    mask: jnp.ndarray,
    params: TriangleAttentionParams,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Project triangle-attention inputs without materializing attention logits."""
    batch, tokens = z.shape[:2]
    normalized = layer_norm(z.astype(jnp.float32), params.norm_weight, params.norm_bias)
    component_width = NUM_TRIANGLE_HEADS * TRIANGLE_HEAD_DIM
    directional_components = []
    for component_index in range(4):
        directions = []
        for direction_index in range(2):
            weight_start = (direction_index * 4 + component_index) * component_width
            projected = linear_bf16(
                normalized,
                params.qkvgb_weight[weight_start : weight_start + component_width],
            )
            if direction_index == 1:
                projected = jnp.swapaxes(projected, 1, 2)
            directions.append(
                projected.reshape(
                    batch,
                    tokens,
                    tokens,
                    NUM_TRIANGLE_HEADS,
                    TRIANGLE_HEAD_DIM,
                )
            )
        directional = jnp.stack(directions, axis=1)
        directional = jnp.transpose(directional, (0, 1, 4, 2, 3, 5))
        directional_components.append(
            directional.reshape(
                batch * 2 * NUM_TRIANGLE_HEADS,
                tokens,
                tokens,
                TRIANGLE_HEAD_DIM,
            )
        )
    q, k, v, gate = directional_components

    pair_bias = linear_bf16(normalized, params.qkvgb_weight[2048:])
    pair_bias = pair_bias.reshape(batch, tokens, tokens, 2, NUM_TRIANGLE_HEADS)
    pair_bias = jnp.transpose(pair_bias, (0, 3, 4, 1, 2))
    pair_mask = mask[:, None, None, :, :]
    pair_bias = jnp.where(pair_mask, pair_bias, MASK_FILL)
    pair_bias = pair_bias.reshape(batch * 2 * NUM_TRIANGLE_HEADS, 1, tokens, tokens)
    return q, k, v, gate, pair_bias


def triangle_attention_finalize(
    attended: jnp.ndarray,
    gate: jnp.ndarray,
    params: TriangleAttentionParams,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Apply the triangle-attention gate and output projection."""
    groups, tokens, _, _ = attended.shape
    batch = groups // (2 * NUM_TRIANGLE_HEADS)
    attended = attended * jax.nn.sigmoid(gate.astype(jnp.float32)).astype(jnp.bfloat16)
    attended = attended.reshape(
        batch, 2, NUM_TRIANGLE_HEADS, tokens, tokens, TRIANGLE_HEAD_DIM
    )
    attended = jnp.transpose(attended, (0, 3, 4, 1, 2, 5)).reshape(
        batch, tokens, tokens, 512
    )
    attended = linear_bf16(attended, params.linear_out_weight)
    attended = attended.reshape(
        batch, tokens, tokens, 2, NUM_TRIANGLE_HEADS, TRIANGLE_HEAD_DIM
    )
    start_update = attended[..., 0, :, :].reshape(batch, tokens, tokens, 256)
    end_update = attended[..., 1, :, :].reshape(batch, tokens, tokens, 256)
    # The graph transposes the ending update before and after feature dropout.
    # Dropout probability is zero, so those two transposes cancel exactly.
    return start_update, end_update


def triangle_attention_query_slice(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    bias: jnp.ndarray,
) -> jnp.ndarray:
    """Attend one bounded query slice against the complete key/value axes."""
    return _sdpa(q, k, v, bias)


def _triangle_attention(
    z: jnp.ndarray,
    mask: jnp.ndarray,
    params: TriangleAttentionParams,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    q, k, v, gate, pair_bias = triangle_attention_prepare(z, mask, params)
    attended = _chunked_sdpa(
        q,
        k,
        v,
        pair_bias,
        query_chunk_size=TRIANGLE_QUERY_CHUNK_SIZE,
    )
    return triangle_attention_finalize(attended, gate, params)


def _attention_pair_bias(
    single: jnp.ndarray,
    pair: jnp.ndarray,
    pair_mask: jnp.ndarray,
    single_mask: jnp.ndarray,
    params: AttentionPairBiasParams,
) -> jnp.ndarray:
    single_normalized = layer_norm(
        single.astype(jnp.float32),
        params.single_norm_weight,
        params.single_norm_bias,
    )
    pair_normalized = layer_norm(
        pair.astype(jnp.float32), params.pair_norm_weight, params.pair_norm_bias
    )
    bias = linear_bf16(pair_normalized, params.pair_linear_weight)
    bias = jnp.transpose(bias, (0, 3, 1, 2))
    bias = jnp.where(pair_mask[:, None], bias, MASK_FILL)

    qkvg = jnp.einsum(
        "bae,efhd->fbhad",
        single_normalized.astype(jnp.bfloat16),
        params.qkvg_weight.astype(jnp.bfloat16),
    )
    q, k, v, gate = qkvg
    q = q + params.query_bias.astype(jnp.bfloat16)[None, :, None, :]
    attended = _sdpa(q, k, v, bias)
    attended *= jax.nn.sigmoid(gate.astype(jnp.float32) + 1.0).astype(jnp.bfloat16)
    output = jnp.einsum(
        "bhnd,hdo->bno",
        attended,
        params.output_weight.astype(jnp.bfloat16),
    )
    return output * single_mask[..., None]


def confidence_attention_pair_bias_projection(
    pair: jnp.ndarray,
    pair_mask: jnp.ndarray,
    params: AttentionPairBiasParams,
) -> jnp.ndarray:
    """Project full or row-chunked pair features to single-attention bias."""

    pair_normalized = layer_norm(
        pair.astype(jnp.float32), params.pair_norm_weight, params.pair_norm_bias
    )
    bias = linear_bf16(pair_normalized, params.pair_linear_weight)
    bias = jnp.transpose(bias, (0, 3, 1, 2))
    return jnp.where(pair_mask[:, None], bias, MASK_FILL)


def confidence_attention_pair_bias_with_bias(
    single: jnp.ndarray,
    bias: jnp.ndarray,
    single_mask: jnp.ndarray,
    params: AttentionPairBiasParams,
) -> jnp.ndarray:
    """Compute single attention from a preprojected official pair bias."""

    single_normalized = layer_norm(
        single.astype(jnp.float32),
        params.single_norm_weight,
        params.single_norm_bias,
    )
    qkvg = jnp.einsum(
        "bae,efhd->fbhad",
        single_normalized.astype(jnp.bfloat16),
        params.qkvg_weight.astype(jnp.bfloat16),
    )
    q, k, v, gate = qkvg
    q = q + params.query_bias.astype(jnp.bfloat16)[None, :, None, :]
    attended = _sdpa(q, k, v, bias)
    attended *= jax.nn.sigmoid(gate.astype(jnp.float32) + 1.0).astype(jnp.bfloat16)
    output = jnp.einsum(
        "bhnd,hdo->bno",
        attended,
        params.output_weight.astype(jnp.bfloat16),
    )
    return output * single_mask[..., None]


def confidence_block_without_triangle_attention(
    single: jnp.ndarray,
    pair: jnp.ndarray,
    single_mask: jnp.ndarray,
    params: ConfidenceBlockParams,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute confidence-block updates independent of triangle attention."""

    pair_mask = single_mask[..., :, None] & single_mask[..., None, :]
    multiplication = _triangle_multiplication(
        pair, pair_mask, params.triangle_multiplication
    )
    pair_transition = _transition(pair, params.transition_pair)
    single_out = (
        single
        + _attention_pair_bias(
            single,
            pair,
            pair_mask,
            single_mask,
            params.attention_pair_bias,
        )
        + _transition(single, params.transition_single)
    )
    return single_out, multiplication, pair_transition


def confidence_block_add_pair_updates(
    pair: jnp.ndarray,
    multiplication: jnp.ndarray,
    attention_start: jnp.ndarray,
    attention_end: jnp.ndarray,
    pair_transition: jnp.ndarray,
) -> jnp.ndarray:
    """Combine pair updates in the exact official residual order."""
    return pair + multiplication + attention_start + attention_end + pair_transition


def confidence_block_forward(
    single: jnp.ndarray,
    pair: jnp.ndarray,
    single_mask: jnp.ndarray,
    params: ConfidenceBlockParams,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Run one Chai confidence Pairformer block in its parallel residual order."""

    single_out, multiplication, pair_transition = (
        confidence_block_without_triangle_attention(single, pair, single_mask, params)
    )
    pair_mask = single_mask[..., :, None] & single_mask[..., None, :]
    attention_start, attention_end = _triangle_attention(
        pair, pair_mask, params.triangle_attention
    )
    pair_out = confidence_block_add_pair_updates(
        pair,
        multiplication,
        attention_start,
        attention_end,
        pair_transition,
    )
    return single_out, pair_out


def confidence_head_initialize(
    token_single_input_repr: jnp.ndarray,
    token_single_trunk_repr: jnp.ndarray,
    token_pair_trunk_repr: jnp.ndarray,
    atom_single_mask: jnp.ndarray,
    atom_coords: jnp.ndarray,
    token_reference_atom_index: jnp.ndarray,
    params: ConfidenceHeadParams,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Build the official confidence single and pair input representations.

    ``atom_single_mask`` is accepted for signature compatibility. The official
    TorchScript graph does not consume it.
    """

    del atom_single_mask
    single = token_single_trunk_repr.astype(jnp.bfloat16)
    pair = token_pair_trunk_repr.astype(jnp.bfloat16)
    single_pair = linear_bf16(token_single_input_repr, params.single_to_pair_weight)
    left, right = jnp.split(single_pair, 2, axis=-1)
    pair = pair + left[:, :, None, :] + right[:, None, :, :]

    batch_indices = jnp.arange(atom_coords.shape[0])[:, None]
    reference_coords = atom_coords[batch_indices, token_reference_atom_index]
    distances = jnp.linalg.norm(
        reference_coords[:, :, None, :] - reference_coords[:, None, :, :],
        axis=-1,
    )
    distance_bins = jnp.searchsorted(params.atom_distance_bins, distances)
    distance_one_hot = jax.nn.one_hot(distance_bins, 16, dtype=jnp.float32)
    pair += linear_bf16(distance_one_hot, params.atom_distance_projection_weight)
    return single, pair


def confidence_head_project(
    single: jnp.ndarray,
    pair: jnp.ndarray,
    atom_token_index: jnp.ndarray,
    atom_within_token_index: jnp.ndarray,
    params: ConfidenceHeadParams,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Project final confidence representations to public logits."""
    single = layer_norm(single.astype(jnp.float32))
    pair = layer_norm(pair.astype(jnp.float32))
    pae_logits = linear_bf16(pair, params.pae_projection_weight)
    pde_logits = linear_bf16(
        pair + jnp.swapaxes(pair, 1, 2), params.pde_projection_weight
    )
    plddt = linear_bf16(single, params.plddt_projection_weight)
    plddt = plddt.reshape(
        plddt.shape[0], plddt.shape[1], NUM_ATOM_SLOTS, NUM_PLDDT_BINS
    )
    batch_indices = jnp.arange(atom_token_index.shape[0])[:, None]
    plddt_logits = plddt[batch_indices, atom_token_index, atom_within_token_index]
    return pae_logits, pde_logits, plddt_logits


def confidence_head_forward(
    token_single_input_repr: jnp.ndarray,
    token_single_trunk_repr: jnp.ndarray,
    token_pair_trunk_repr: jnp.ndarray,
    token_single_mask: jnp.ndarray,
    atom_single_mask: jnp.ndarray,
    atom_coords: jnp.ndarray,
    token_reference_atom_index: jnp.ndarray,
    atom_token_index: jnp.ndarray,
    atom_within_token_index: jnp.ndarray,
    params: ConfidenceHeadParams,
    *,
    use_scan: bool = True,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Match official ``confidence_head.pt::forward_<crop>``."""
    single, pair = confidence_head_initialize(
        token_single_input_repr,
        token_single_trunk_repr,
        token_pair_trunk_repr,
        atom_single_mask,
        atom_coords,
        token_reference_atom_index,
        params,
    )

    if use_scan and len(params.blocks) > 1:
        stacked = stacked_or_stack(params.blocks)
        single, pair = jax.lax.scan(
            lambda carry, block: (
                confidence_block_forward(carry[0], carry[1], token_single_mask, block),
                None,
            ),
            (single, pair),
            stacked,
        )[0]
    else:
        for block in params.blocks:
            single, pair = confidence_block_forward(
                single, pair, token_single_mask, block
            )
    return confidence_head_project(
        single, pair, atom_token_index, atom_within_token_index, params
    )
