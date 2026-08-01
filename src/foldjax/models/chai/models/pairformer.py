"""Chai Pairformer blocks recovered from the exported trunk graph."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from foldjax.models.chai.models.primitives import layer_norm, linear_bf16

TRIANGLE_ATTENTION_OUTER_CHUNK_SIZE = 128


def _triangle_attention_backend() -> str:
    configured = os.environ.get("CHAI_JAX_TRIANGLE_ATTENTION_BACKEND", "auto")
    if configured not in {"auto", "xla", "cueq"}:
        raise ValueError(
            "CHAI_JAX_TRIANGLE_ATTENTION_BACKEND must be 'auto', 'xla', or 'cueq'"
        )
    if configured != "auto":
        return configured
    from foldjax.models.chai.models.triangle_cueq import cueq_available

    has_gpu = any(device.platform == "gpu" for device in jax.devices())
    return "cueq" if has_gpu and cueq_available() else "xla"


class PairformerTransitionParams(NamedTuple):
    layer_norm_weight: jnp.ndarray
    layer_norm_bias: jnp.ndarray
    linear_ab_weight: jnp.ndarray
    linear_out_weight: jnp.ndarray


class FusedTriangleMultiplicationParams(NamedTuple):
    layer_norm_weight: jnp.ndarray
    layer_norm_bias: jnp.ndarray
    merged_linear_p_weight: jnp.ndarray
    merged_linear_g_weight: jnp.ndarray
    linear_z_out_weight: jnp.ndarray


class FusedTriangleAttentionParams(NamedTuple):
    out_scalers: jnp.ndarray
    pair2b_weight: jnp.ndarray
    pair2qkvg1_weight: jnp.ndarray
    pair2qkvg2_weight: jnp.ndarray
    linear_out_weight: jnp.ndarray


class AttentionPairBiasParams(NamedTuple):
    single_layer_norm_weight: jnp.ndarray
    single_layer_norm_bias: jnp.ndarray
    pair_layer_norm_weight: jnp.ndarray
    pair_layer_norm_bias: jnp.ndarray
    pair_linear_weight: jnp.ndarray
    query_bias: jnp.ndarray
    input2qkvg_weight: jnp.ndarray
    output_proj_weight: jnp.ndarray


class PairformerBlockParams(NamedTuple):
    transition_pair: PairformerTransitionParams
    triangle_multiplication: FusedTriangleMultiplicationParams
    triangle_attention: FusedTriangleAttentionParams
    transition_single: PairformerTransitionParams
    attention_pair_bias: AttentionPairBiasParams


class PairformerPairBlockParams(NamedTuple):
    transition_pair: PairformerTransitionParams
    triangle_multiplication: FusedTriangleMultiplicationParams
    triangle_attention: FusedTriangleAttentionParams


def map_pairformer_transition(
    state: Mapping[str, Any], prefix: str
) -> PairformerTransitionParams:
    """Map one unconditioned Pairformer transition from ``trunk.pt``."""
    base = f"{prefix}." if prefix else ""
    return PairformerTransitionParams(
        layer_norm_weight=jnp.asarray(state[f"{base}layer_norm.weight"]),
        layer_norm_bias=jnp.asarray(state[f"{base}layer_norm.bias"]),
        linear_ab_weight=jnp.asarray(state[f"{base}linear_no_bias_ab.weight"]),
        linear_out_weight=jnp.asarray(state[f"{base}linear_out.weight"]),
    )


def map_fused_triangle_multiplication(
    state: Mapping[str, Any], prefix: str = ""
) -> FusedTriangleMultiplicationParams:
    """Map Chai's single fused outgoing+incoming triangle module."""
    base = f"{prefix}." if prefix else ""
    return FusedTriangleMultiplicationParams(
        layer_norm_weight=jnp.asarray(state[f"{base}layernorm_z_in.weight"]),
        layer_norm_bias=jnp.asarray(state[f"{base}layernorm_z_in.bias"]),
        merged_linear_p_weight=jnp.asarray(state[f"{base}merged_linear_p.weight"]),
        merged_linear_g_weight=jnp.asarray(state[f"{base}merged_linear_g.weight"]),
        linear_z_out_weight=jnp.asarray(state[f"{base}linear_z_out.weight"]),
    )


def map_fused_triangle_attention(
    state: Mapping[str, Any], prefix: str = ""
) -> FusedTriangleAttentionParams:
    """Map Chai's fused two-direction triangle attention."""
    base = f"{prefix}." if prefix else ""
    return FusedTriangleAttentionParams(
        out_scalers=jnp.asarray(state[f"{base}out_scalers"]),
        pair2b_weight=jnp.asarray(state[f"{base}pair2b.weight"]),
        pair2qkvg1_weight=jnp.asarray(state[f"{base}pair2qkvg1.weight"]),
        pair2qkvg2_weight=jnp.asarray(state[f"{base}pair2qkvg2.weight"]),
        linear_out_weight=jnp.asarray(state[f"{base}linear_out.weight"]),
    )


def map_attention_pair_bias(
    state: Mapping[str, Any], prefix: str = ""
) -> AttentionPairBiasParams:
    """Map Chai's token self-attention with pair bias."""
    base = f"{prefix}." if prefix else ""
    return AttentionPairBiasParams(
        single_layer_norm_weight=jnp.asarray(state[f"{base}single_layer_norm.weight"]),
        single_layer_norm_bias=jnp.asarray(state[f"{base}single_layer_norm.bias"]),
        pair_layer_norm_weight=jnp.asarray(state[f"{base}pair_layer_norm.weight"]),
        pair_layer_norm_bias=jnp.asarray(state[f"{base}pair_layer_norm.bias"]),
        pair_linear_weight=jnp.asarray(state[f"{base}pair_linear.weight"]),
        query_bias=jnp.asarray(state[f"{base}attention.query_bias"]),
        input2qkvg_weight=jnp.asarray(state[f"{base}attention.input2qkvg.weight"]),
        output_proj_weight=jnp.asarray(state[f"{base}attention.output_proj.weight"]),
    )


def map_pairformer_block(
    state: Mapping[str, Any], prefix: str = ""
) -> PairformerBlockParams:
    """Map all 26 tensors in one Chai Pairformer block."""
    base = f"{prefix}." if prefix else ""
    return PairformerBlockParams(
        transition_pair=map_pairformer_transition(
            state, f"{base}transition_pair".removesuffix(".")
        ),
        triangle_multiplication=map_fused_triangle_multiplication(
            state, f"{base}triangle_multiplication".removesuffix(".")
        ),
        triangle_attention=map_fused_triangle_attention(
            state, f"{base}triangle_attention".removesuffix(".")
        ),
        transition_single=map_pairformer_transition(
            state, f"{base}transition_single".removesuffix(".")
        ),
        attention_pair_bias=map_attention_pair_bias(
            state, f"{base}attention_pair_bias".removesuffix(".")
        ),
    )


def map_pairformer_pair_block(
    state: Mapping[str, Any], prefix: str = ""
) -> PairformerPairBlockParams:
    """Map the 14 tensors in a Chai pair-only Pairformer block."""
    base = f"{prefix}." if prefix else ""
    return PairformerPairBlockParams(
        transition_pair=map_pairformer_transition(state, f"{base}transition_pair"),
        triangle_multiplication=map_fused_triangle_multiplication(
            state, f"{base}triangle_multiplication"
        ),
        triangle_attention=map_fused_triangle_attention(
            state, f"{base}triangle_attention"
        ),
    )


LinearFn = Callable[..., jnp.ndarray]


def _sdpa_implementation() -> str:
    configured = os.environ.get("CHAI_JAX_SDPA_IMPLEMENTATION")
    if configured is not None:
        if configured not in {"cudnn", "xla"}:
            raise ValueError("CHAI_JAX_SDPA_IMPLEMENTATION must be 'cudnn' or 'xla'")
        return configured
    # cuDNN fused attention can fail plan construction for heavily padded Chai
    # buckets (including valid one-residue inference on Blackwell). XLA is the
    # portable default; users may opt into cuDNN after validating their shapes.
    return "xla"


def pairformer_transition(
    x: jnp.ndarray,
    params: PairformerTransitionParams,
    *,
    lin: LinearFn = linear_bf16,
) -> jnp.ndarray:
    """Return the Pairformer transition update without adding the residual."""
    normalized = layer_norm(
        x.astype(jnp.float32),
        params.layer_norm_weight,
        params.layer_norm_bias,
    )
    a, b = jnp.split(lin(normalized, params.linear_ab_weight), 2, axis=-1)
    return lin(jax.nn.silu(a) * b, params.linear_out_weight)


def pairformer_transition_projection(
    x: jnp.ndarray,
    params: PairformerTransitionParams,
    *,
    second: bool,
    lin: LinearFn = linear_bf16,
) -> jnp.ndarray:
    """Compute one transition projection for low-memory staging."""

    normalized = layer_norm(
        x.astype(jnp.float32),
        params.layer_norm_weight,
        params.layer_norm_bias,
    )
    split = params.linear_ab_weight.shape[0] // 2
    weight = (
        params.linear_ab_weight[split:] if second else params.linear_ab_weight[:split]
    )
    projected = lin(normalized, weight)
    return projected if second else jax.nn.silu(projected)


def pairformer_transition_output(
    first: jnp.ndarray,
    second: jnp.ndarray,
    params: PairformerTransitionParams,
    *,
    lin: LinearFn = linear_bf16,
) -> jnp.ndarray:
    """Finish the official transition product and output projection."""

    return lin(first * second, params.linear_out_weight)


def fused_triangle_multiplication(
    z: jnp.ndarray,
    pair_mask: jnp.ndarray,
    params: FusedTriangleMultiplicationParams,
    *,
    lin: LinearFn = linear_bf16,
) -> jnp.ndarray:
    """Return Chai's parallel outgoing+incoming triangle residual update."""
    c_z = z.shape[-1]
    normalized = layer_norm(
        z.astype(jnp.float32),
        params.layer_norm_weight,
        params.layer_norm_bias,
    )
    p_weight = params.merged_linear_p_weight
    g_weight = params.merged_linear_g_weight

    def gated_projection(index: int) -> jnp.ndarray:
        start = index * c_z
        stop = start + c_z
        projected = lin(normalized, p_weight[start:stop])
        gate = jax.nn.sigmoid(lin(normalized, g_weight[start:stop]))
        return projected * gate

    out_a, out_b, in_a, in_b = (gated_projection(index) for index in range(4))
    out_a = jnp.where(pair_mask[..., None], out_a, 0)
    out_b = jnp.where(pair_mask[..., None], out_b, 0)
    transposed_mask = jnp.swapaxes(pair_mask, -1, -2)
    in_a = jnp.where(transposed_mask[..., None], in_a, 0)
    in_b = jnp.where(transposed_mask[..., None], in_b, 0)
    out = jnp.einsum(
        "...ikd,...jkd->...ijd",
        out_a,
        out_b,
        preferred_element_type=jnp.float32,
    ).astype(out_a.dtype)
    inc = jnp.einsum(
        "...kid,...kjd->...ijd",
        in_a,
        in_b,
        preferred_element_type=jnp.float32,
    ).astype(in_a.dtype)
    out = layer_norm(out.astype(jnp.float32))
    inc = layer_norm(inc.astype(jnp.float32))
    projected = lin(out + inc, params.linear_z_out_weight)
    out_gate = jax.nn.sigmoid(lin(normalized, g_weight[4 * c_z : 5 * c_z]))
    return projected * out_gate


def fused_triangle_multiplication_direction(
    z: jnp.ndarray,
    pair_mask: jnp.ndarray,
    params: FusedTriangleMultiplicationParams,
    *,
    incoming: bool,
    lin: LinearFn = linear_bf16,
) -> jnp.ndarray:
    """Compute one normalized triangle product for low-memory staging."""

    c_z = z.shape[-1]
    normalized = layer_norm(
        z.astype(jnp.float32),
        params.layer_norm_weight,
        params.layer_norm_bias,
    )
    offset = 2 if incoming else 0

    def gated_projection(index: int) -> jnp.ndarray:
        start = index * c_z
        stop = start + c_z
        projected = lin(normalized, params.merged_linear_p_weight[start:stop])
        gate = jax.nn.sigmoid(
            lin(normalized, params.merged_linear_g_weight[start:stop])
        )
        return projected * gate

    a = gated_projection(offset)
    b = gated_projection(offset + 1)
    if incoming:
        mask = jnp.swapaxes(pair_mask, -1, -2)
        a = jnp.where(mask[..., None], a, 0)
        b = jnp.where(mask[..., None], b, 0)
        product = jnp.einsum(
            "...kid,...kjd->...ijd",
            a,
            b,
            preferred_element_type=jnp.float32,
        ).astype(a.dtype)
    else:
        a = jnp.where(pair_mask[..., None], a, 0)
        b = jnp.where(pair_mask[..., None], b, 0)
        product = jnp.einsum(
            "...ikd,...jkd->...ijd",
            a,
            b,
            preferred_element_type=jnp.float32,
        ).astype(a.dtype)
    return layer_norm(product.astype(jnp.float32))


def fused_triangle_multiplication_direction_chunk(
    z: jnp.ndarray,
    pair_mask: jnp.ndarray,
    params: FusedTriangleMultiplicationParams,
    *,
    incoming: bool,
    start: int,
    size: int,
    column_start: int,
    column_size: int,
    lin: LinearFn = linear_bf16,
) -> jnp.ndarray:
    """Compute one output-row chunk without materializing both full projections."""

    c_z = z.shape[-1]
    offset = 2 if incoming else 0

    def gated_projection(value: jnp.ndarray, index: int) -> jnp.ndarray:
        normalized = layer_norm(
            value.astype(jnp.float32),
            params.layer_norm_weight,
            params.layer_norm_bias,
        )
        begin = index * c_z
        end = begin + c_z
        projected = lin(normalized, params.merged_linear_p_weight[begin:end])
        gate = jax.nn.sigmoid(lin(normalized, params.merged_linear_g_weight[begin:end]))
        return projected * gate

    if incoming:
        a_input = jax.lax.dynamic_slice_in_dim(z, start, size, axis=-2)
        a = gated_projection(a_input, offset)
        b_input = jax.lax.dynamic_slice_in_dim(z, column_start, column_size, axis=-2)
        b = gated_projection(b_input, offset + 1)
        transposed_mask = jnp.swapaxes(pair_mask, -1, -2)
        a_mask = jax.lax.dynamic_slice_in_dim(transposed_mask, start, size, axis=-1)
        a = jnp.where(a_mask[..., None], a, 0)
        b_mask = jax.lax.dynamic_slice_in_dim(
            transposed_mask, column_start, column_size, axis=-1
        )
        b = jnp.where(b_mask[..., None], b, 0)
        product = jnp.einsum(
            "...kid,...kjd->...ijd",
            a,
            b,
            preferred_element_type=jnp.float32,
        ).astype(a.dtype)
    else:
        a_input = jax.lax.dynamic_slice_in_dim(z, start, size, axis=-3)
        a = gated_projection(a_input, offset)
        b_input = jax.lax.dynamic_slice_in_dim(z, column_start, column_size, axis=-3)
        b = gated_projection(b_input, offset + 1)
        a_mask = jax.lax.dynamic_slice_in_dim(pair_mask, start, size, axis=-2)
        a = jnp.where(a_mask[..., None], a, 0)
        b_mask = jax.lax.dynamic_slice_in_dim(
            pair_mask, column_start, column_size, axis=-2
        )
        b = jnp.where(b_mask[..., None], b, 0)
        product = jnp.einsum(
            "...ikd,...jkd->...ijd",
            a,
            b,
            preferred_element_type=jnp.float32,
        ).astype(a.dtype)
    return layer_norm(product.astype(jnp.float32))


def fused_triangle_multiplication_output(
    z: jnp.ndarray,
    outgoing: jnp.ndarray,
    incoming: jnp.ndarray,
    params: FusedTriangleMultiplicationParams,
    *,
    lin: LinearFn = linear_bf16,
) -> jnp.ndarray:
    """Apply the exact official final projection and output gate."""

    c_z = z.shape[-1]
    normalized = layer_norm(
        z.astype(jnp.float32),
        params.layer_norm_weight,
        params.layer_norm_bias,
    )
    projected = lin(outgoing + incoming, params.linear_z_out_weight)
    out_gate = jax.nn.sigmoid(
        lin(
            normalized,
            params.merged_linear_g_weight[4 * c_z : 5 * c_z],
        )
    )
    return projected * out_gate


def _triangle_attention_direction(
    x: jnp.ndarray,
    weight: jnp.ndarray,
    bias: jnp.ndarray,
    *,
    num_heads: int,
    lin: LinearFn,
    outer_chunk_size: int | None,
    attention_backend: str,
) -> jnp.ndarray:
    batch, rows, columns, _ = x.shape
    projected = lin(x, weight)
    head_dim = projected.shape[-1] // (4 * num_heads)
    qkvg = projected.reshape(batch, rows, columns, num_heads, 4, head_dim)
    q, k, v, gate = jnp.transpose(qkvg, (4, 0, 3, 1, 2, 5))

    if attention_backend == "cueq":
        from foldjax.models.chai.models.triangle_cueq import cueq_attention_core

        cueq_q = jnp.transpose(q, (0, 2, 1, 3, 4))
        cueq_k = jnp.transpose(k, (0, 2, 1, 3, 4))
        cueq_v = jnp.transpose(v, (0, 2, 1, 3, 4))
        cueq_gate = jnp.transpose(gate, (0, 2, 1, 3, 4))
        # ``bias`` already contains Chai's -10000 pair-mask bias. Passing an
        # all-valid cuEq mask retains the official Torch masking semantics.
        cueq_mask = jnp.ones((batch, rows, 1, 1, columns), dtype=jnp.bool_)
        attended = cueq_attention_core(
            cueq_q,
            cueq_k,
            cueq_v,
            bias[:, None],
            cueq_mask,
            scale=head_dim**-0.5,
        )
        attended = attended.astype(v.dtype) * jax.nn.sigmoid(cueq_gate)
        return jnp.transpose(attended, (0, 1, 3, 2, 4)).reshape(
            batch, rows, columns, num_heads * head_dim
        )

    if attention_backend != "xla":
        raise ValueError(
            f"unsupported triangle attention backend: {attention_backend!r}"
        )

    q = q.reshape(batch * num_heads, rows, columns, head_dim)
    k = k.reshape(batch * num_heads, rows, columns, head_dim)
    v = v.reshape(batch * num_heads, rows, columns, head_dim)
    gate = gate.reshape(batch * num_heads, rows, columns, head_dim)

    bias_rows, bias_columns = bias.shape[-2:]
    attention_bias = bias.reshape(batch * num_heads, 1, bias_rows, bias_columns)

    def attend_block(start: int, size: int) -> jnp.ndarray:
        q_block = jax.lax.dynamic_slice_in_dim(q, start, size, axis=1)
        k_block = jax.lax.dynamic_slice_in_dim(k, start, size, axis=1)
        v_block = jax.lax.dynamic_slice_in_dim(v, start, size, axis=1)
        block_bias = jnp.broadcast_to(
            attention_bias[:, None],
            (batch * num_heads, size, 1, bias_rows, bias_columns),
        ).reshape(batch * num_heads * size, 1, bias_rows, bias_columns)
        return jax.nn.dot_product_attention(
            q_block.reshape(batch * num_heads * size, columns, 1, head_dim),
            k_block.reshape(batch * num_heads * size, columns, 1, head_dim),
            v_block.reshape(batch * num_heads * size, columns, 1, head_dim),
            bias=block_bias,
            implementation=_sdpa_implementation(),
        ).reshape(batch * num_heads, size, columns, head_dim)

    if outer_chunk_size is None or outer_chunk_size <= 0 or outer_chunk_size >= rows:
        attended = attend_block(0, rows)
    else:
        attended = jnp.concatenate(
            [
                attend_block(start, min(outer_chunk_size, rows - start))
                for start in range(0, rows, outer_chunk_size)
            ],
            axis=1,
        )
    attended = attended.astype(v.dtype)
    attended = attended * jax.nn.sigmoid(gate)
    attended = attended.reshape(batch, num_heads, rows, columns, head_dim)
    return jnp.transpose(attended, (0, 2, 3, 1, 4)).reshape(
        batch, rows, columns, num_heads * head_dim
    )


def fused_triangle_attention(
    z: jnp.ndarray,
    pair_mask: jnp.ndarray,
    params: FusedTriangleAttentionParams,
    *,
    lin: LinearFn = linear_bf16,
    outer_chunk_size: int | None = TRIANGLE_ATTENTION_OUTER_CHUNK_SIZE,
    attention_backend: str | None = None,
) -> jnp.ndarray:
    """Return Chai's two-direction triangle-attention residual update."""
    backend = (
        _triangle_attention_backend()
        if attention_backend is None
        else attention_backend
    )
    if backend not in {"xla", "cueq"}:
        raise ValueError(f"unsupported triangle attention backend: {backend!r}")
    pair = layer_norm(z.astype(jnp.float32))
    num_heads = params.pair2b_weight.shape[0] // 2
    bias = lin(pair, params.pair2b_weight)
    bias = bias.reshape(*bias.shape[:-1], 2, num_heads)
    bias = jnp.transpose(bias, (0, 3, 4, 1, 2))
    bias = jnp.where(pair_mask[:, None, None], bias, -10000.0)
    first = _triangle_attention_direction(
        pair,
        params.pair2qkvg1_weight,
        bias[:, 0],
        num_heads=num_heads,
        lin=lin,
        outer_chunk_size=outer_chunk_size,
        attention_backend=backend,
    )
    second = _triangle_attention_direction(
        jnp.swapaxes(pair, 1, 2),
        params.pair2qkvg2_weight,
        bias[:, 1],
        num_heads=num_heads,
        lin=lin,
        outer_chunk_size=outer_chunk_size,
        attention_backend=backend,
    )
    merged = jnp.concatenate([first, second], axis=-1)
    output_weight = params.linear_out_weight * params.out_scalers[:, None]
    return lin(merged, output_weight)


def fused_triangle_attention_direction(
    z: jnp.ndarray,
    pair_mask: jnp.ndarray,
    params: FusedTriangleAttentionParams,
    *,
    direction: int,
    lin: LinearFn = linear_bf16,
    outer_chunk_size: int | None = TRIANGLE_ATTENTION_OUTER_CHUNK_SIZE,
    attention_backend: str | None = None,
) -> jnp.ndarray:
    """Compute one exact directional output for low-memory staging."""

    if direction not in {0, 1}:
        raise ValueError("triangle attention direction must be 0 or 1")
    backend = (
        _triangle_attention_backend()
        if attention_backend is None
        else attention_backend
    )
    pair = layer_norm(z.astype(jnp.float32))
    num_heads = params.pair2b_weight.shape[0] // 2
    bias = lin(pair, params.pair2b_weight)
    bias = bias.reshape(*bias.shape[:-1], 2, num_heads)
    bias = jnp.transpose(bias, (0, 3, 4, 1, 2))
    bias = jnp.where(pair_mask[:, None, None], bias, -10000.0)
    value = pair if direction == 0 else jnp.swapaxes(pair, 1, 2)
    weight = params.pair2qkvg1_weight if direction == 0 else params.pair2qkvg2_weight
    return _triangle_attention_direction(
        value,
        weight,
        bias[:, direction],
        num_heads=num_heads,
        lin=lin,
        outer_chunk_size=outer_chunk_size,
        attention_backend=backend,
    )


def fused_triangle_attention_bias(
    z: jnp.ndarray,
    pair_mask: jnp.ndarray,
    params: FusedTriangleAttentionParams,
    *,
    lin: LinearFn = linear_bf16,
) -> jnp.ndarray:
    """Prepare the shared full attention bias for outer-row staging."""

    pair = layer_norm(z.astype(jnp.float32))
    num_heads = params.pair2b_weight.shape[0] // 2
    bias = lin(pair, params.pair2b_weight)
    bias = bias.reshape(*bias.shape[:-1], 2, num_heads)
    bias = jnp.transpose(bias, (0, 3, 4, 1, 2))
    return jnp.where(pair_mask[:, None, None], bias, -10000.0)


def fused_triangle_attention_direction_chunk(
    value: jnp.ndarray,
    bias: jnp.ndarray,
    params: FusedTriangleAttentionParams,
    *,
    direction: int,
    lin: LinearFn = linear_bf16,
) -> jnp.ndarray:
    """Run one backend-portable outer-row chunk with the shared attention bias."""

    if direction not in {0, 1}:
        raise ValueError("triangle attention direction must be 0 or 1")
    backend = _triangle_attention_backend()
    pair = layer_norm(value.astype(jnp.float32))
    num_heads = params.pair2b_weight.shape[0] // 2
    weight = params.pair2qkvg1_weight if direction == 0 else params.pair2qkvg2_weight
    return _triangle_attention_direction(
        pair,
        weight,
        bias[:, direction],
        num_heads=num_heads,
        lin=lin,
        outer_chunk_size=None,
        attention_backend=backend,
    )


def fused_triangle_attention_output(
    first: jnp.ndarray,
    second: jnp.ndarray,
    params: FusedTriangleAttentionParams,
    *,
    lin: LinearFn = linear_bf16,
) -> jnp.ndarray:
    """Apply the unchanged official projection to directional outputs."""

    merged = jnp.concatenate([first, second], axis=-1)
    output_weight = params.linear_out_weight * params.out_scalers[:, None]
    return lin(merged, output_weight)


def attention_pair_bias(
    s: jnp.ndarray,
    z: jnp.ndarray,
    pair_mask: jnp.ndarray,
    token_mask: jnp.ndarray,
    params: AttentionPairBiasParams,
    *,
    lin: LinearFn = linear_bf16,
) -> jnp.ndarray:
    """Return Chai's masked single-attention residual update."""
    single = layer_norm(
        s.astype(jnp.float32),
        params.single_layer_norm_weight,
        params.single_layer_norm_bias,
    )
    pair = layer_norm(
        z.astype(jnp.float32),
        params.pair_layer_norm_weight,
        params.pair_layer_norm_bias,
    )
    bias = lin(pair, params.pair_linear_weight)
    bias = jnp.moveaxis(bias, -1, 1)
    bias = jnp.where(pair_mask[:, None], bias, -10000.0)

    input_dim, _, num_heads, head_dim = params.input2qkvg_weight.shape
    qkvg_weight = jnp.transpose(params.input2qkvg_weight, (1, 2, 3, 0)).reshape(
        4 * num_heads * head_dim, input_dim
    )
    qkvg = lin(single, qkvg_weight)
    qkvg = qkvg.reshape(s.shape[0], s.shape[1], 4, num_heads, head_dim)
    q, k, v, gate = jnp.transpose(qkvg, (2, 0, 3, 1, 4))
    q = (q + params.query_bias[None, :, None, :]).astype(k.dtype)
    attended = jax.nn.dot_product_attention(
        jnp.transpose(q, (0, 2, 1, 3)),
        jnp.transpose(k, (0, 2, 1, 3)),
        jnp.transpose(v, (0, 2, 1, 3)),
        bias=bias.astype(jnp.float32),
        implementation=_sdpa_implementation(),
    )
    attended = jnp.transpose(attended, (0, 2, 1, 3)).astype(v.dtype)
    attended = attended * jax.nn.sigmoid(gate + 1)
    attended = jnp.transpose(attended, (0, 2, 1, 3)).reshape(
        s.shape[0], s.shape[1], num_heads * head_dim
    )
    output_dim = params.output_proj_weight.shape[-1]
    output_weight = jnp.transpose(params.output_proj_weight, (2, 0, 1)).reshape(
        output_dim, num_heads * head_dim
    )
    update = lin(attended, output_weight)
    return update * token_mask[..., None]


def pairformer_block(
    s: jnp.ndarray,
    z: jnp.ndarray,
    token_mask: jnp.ndarray,
    pair_mask: jnp.ndarray,
    params: PairformerBlockParams,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Apply one Chai block with all branches reading pre-block ``s``/``z``."""
    z_out = z + fused_triangle_multiplication(
        z, pair_mask, params.triangle_multiplication
    )
    z_out += fused_triangle_attention(z, pair_mask, params.triangle_attention)
    z_out += pairformer_transition(z, params.transition_pair)
    s_out = s + attention_pair_bias(
        s, z, pair_mask, token_mask, params.attention_pair_bias
    )
    s_out += pairformer_transition(s, params.transition_single)
    return s_out, z_out


def pairformer_pair_block(
    z: jnp.ndarray,
    pair_mask: jnp.ndarray,
    params: PairformerPairBlockParams,
) -> jnp.ndarray:
    """Apply a Chai pair-only block with parallel pre-block branches."""
    out = z + fused_triangle_multiplication(
        z, pair_mask, params.triangle_multiplication
    )
    out += fused_triangle_attention(z, pair_mask, params.triangle_attention)
    out += pairformer_transition(z, params.transition_pair)
    return out
