"""JAX attention blocks matching Protenix inference modules."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from foldjax.models.protenix.models.primitives.primitives import (
    AdaptiveLayerNormParams,
    LayerNormParams,
    LinearParams,
    adaptive_layer_norm,
    layer_norm,
    linear,
    sigmoid,
)
from foldjax.models.protenix.models.primitives.windows import (
    gather_overlapping_windows,
)


class AttentionParams(NamedTuple):
    """Parameters for ``protenix.model.modules.primitives.Attention``."""

    linear_q: LinearParams
    linear_k: LinearParams
    linear_v: LinearParams
    linear_o: LinearParams
    linear_g: LinearParams | None


class AttentionPairBiasParams(NamedTuple):
    """Parameters for standard ``AttentionPairBias``."""

    layernorm_a: LayerNormParams | AdaptiveLayerNormParams
    layernorm_kv: LayerNormParams | AdaptiveLayerNormParams | None
    attention: AttentionParams
    layernorm_z: LayerNormParams
    linear_z: LinearParams
    linear_a_last: LinearParams | None = None
    has_s: bool = False
    cross_attention_mode: bool = False


def prepare_qkv(
    q_x: jnp.ndarray,
    kv_x: jnp.ndarray,
    params: AttentionParams,
    num_heads: int,
    *,
    apply_scale: bool = True,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Prepare q/k/v in Protenix layout: ``[..., H, Q/K/V, C_hidden]``."""

    q = _project_heads(q_x, params.linear_q, num_heads)
    k = _project_heads(kv_x, params.linear_k, num_heads)
    v = _project_heads(kv_x, params.linear_v, num_heads)
    if apply_scale:
        q = q / jnp.sqrt(jnp.asarray(q.shape[-1], dtype=q.dtype))
    return q, k, v


def attention(
    q_x: jnp.ndarray,
    kv_x: jnp.ndarray,
    params: AttentionParams,
    num_heads: int,
    attn_bias: jnp.ndarray | None = None,
    *,
    q_chunk_size: int | None = None,
    attention_backend: str = "xla",
) -> jnp.ndarray:
    """Run standard full attention with optional pair bias and gating."""

    if attention_backend == "xla_jit":
        return _compiled_attention(
            q_x,
            kv_x,
            params,
            num_heads,
            attn_bias,
            q_chunk_size=q_chunk_size,
            attention_backend="xla",
        )
    q, k, v = prepare_qkv(q_x, kv_x, params, num_heads, apply_scale=True)
    out = _attention_qkv(
        q,
        k,
        v,
        attn_bias,
        q_chunk_size=q_chunk_size,
        attention_backend=attention_backend,
    )
    out = jnp.swapaxes(out, -2, -3)
    if params.linear_g is not None:
        gate = sigmoid(linear(q_x, params.linear_g))
        gate = gate.reshape(gate.shape[:-1] + (num_heads, -1))
        out = out * gate
    out = out.reshape(out.shape[:-2] + (-1,))
    return linear(out, params.linear_o)


_compiled_attention = jax.jit(
    attention,
    static_argnames=("num_heads", "q_chunk_size", "attention_backend"),
)


def attention_pair_bias(
    a: jnp.ndarray,
    s: jnp.ndarray | None,
    z: jnp.ndarray,
    params: AttentionPairBiasParams,
    *,
    num_heads: int,
    q_chunk_size: int | None = None,
    attention_backend: str = "xla",
    z_is_normalized: bool = False,
    extra_attn_bias: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Run standard global ``AttentionPairBias``.

    This implements the non-local path used by token Pairformer/Confidence
    blocks. Atom local attention is intentionally separate.
    """

    q = _apply_apb_norm(a, s, params.layernorm_a, params.has_s)
    if params.cross_attention_mode:
        kv = _apply_apb_norm(q, s, params.layernorm_kv, params.has_s)
    else:
        kv = q
    if z_is_normalized:
        z_norm = z
        if params.layernorm_z.weight is not None:
            z_norm = z_norm * params.layernorm_z.weight
        if params.layernorm_z.bias is not None:
            z_norm = z_norm + params.layernorm_z.bias
    else:
        z_norm = layer_norm(z, params.layernorm_z)
    bias = linear(z_norm, params.linear_z)
    bias = jnp.moveaxis(bias, -1, -3)
    if extra_attn_bias is not None:
        if extra_attn_bias.shape[-2:] != bias.shape[-2:]:
            raise ValueError(
                "extra attention bias pair axes must match projected pair bias: "
                f"{extra_attn_bias.shape[-2:]} != {bias.shape[-2:]}"
            )
        extra_attn_bias = jnp.asarray(extra_attn_bias, dtype=bias.dtype)
        while extra_attn_bias.ndim < bias.ndim - 1:
            extra_attn_bias = jnp.expand_dims(extra_attn_bias, axis=0)
        if extra_attn_bias.ndim == bias.ndim - 1:
            extra_attn_bias = jnp.expand_dims(extra_attn_bias, axis=-3)
        bias = bias + extra_attn_bias
    target_ndim = q.ndim + 1
    while bias.ndim < target_ndim:
        bias = jnp.expand_dims(bias, axis=bias.ndim - 3)
    out = attention(
        q,
        kv,
        params.attention,
        num_heads,
        bias,
        q_chunk_size=q_chunk_size,
        attention_backend=attention_backend,
    )
    if params.linear_a_last is not None:
        if s is None:
            raise ValueError("linear_a_last requires conditioning tensor s")
        out = sigmoid(linear(s, params.linear_a_last)) * out
    return out


def local_attention(
    q_x: jnp.ndarray,
    kv_x: jnp.ndarray,
    params: AttentionParams,
    num_heads: int,
    *,
    trunked_attn_bias: jnp.ndarray | None,
    n_queries: int,
    n_keys: int,
    sequence_mask: jnp.ndarray | None = None,
    inf: float = 1.0e10,
    attention_backend: str = "xla",
) -> jnp.ndarray:
    """Run local blocked attention used by AtomTransformer."""

    if attention_backend == "xla_jit":
        return _compiled_local_attention(
            q_x,
            kv_x,
            params,
            num_heads,
            trunked_attn_bias=trunked_attn_bias,
            n_queries=n_queries,
            n_keys=n_keys,
            sequence_mask=sequence_mask,
            inf=inf,
            attention_backend="xla",
        )
    q, k, v = prepare_qkv(q_x, kv_x, params, num_heads, apply_scale=True)
    q_trunked, k_trunked, mask, q_pad = _local_qk_trunks(
        q,
        k,
        n_queries=n_queries,
        n_keys=n_keys,
    )
    _, v_trunked, _, _ = _local_qk_trunks(
        q,
        v,
        n_queries=n_queries,
        n_keys=n_keys,
    )
    if sequence_mask is not None:
        sequence_mask = jnp.asarray(sequence_mask).astype(bool)
        if sequence_mask.ndim != 1 or sequence_mask.shape[0] != q_x.shape[-2]:
            raise ValueError("local attention sequence_mask must have shape [N_atom]")
        q_valid, k_valid, _, _ = _local_qk_trunks(
            sequence_mask[:, None],
            sequence_mask[:, None],
            n_queries=n_queries,
            n_keys=n_keys,
        )
        mask = mask & q_valid[..., 0, None] & k_valid[..., None, :, 0]
    mask = mask.reshape((1,) * (q_trunked.ndim - 4) + mask.shape)
    if attention_backend == "xla":
        logits = jnp.einsum("...htqd,...htkd->...htqk", q_trunked, k_trunked)
        logits = logits + jnp.where(mask, 0.0, -inf)
        if trunked_attn_bias is not None:
            if trunked_attn_bias.ndim == logits.ndim - 1:
                trunked_attn_bias = jnp.expand_dims(trunked_attn_bias, axis=-4)
            logits = logits + trunked_attn_bias
        probs = jax.nn.softmax(logits.astype(jnp.float32), axis=-1).astype(v.dtype)
        out = jnp.einsum("...htqk,...htkd->...htqd", probs, v_trunked)
    else:
        out = _local_builtin_sdpa(
            q_trunked,
            k_trunked,
            v_trunked,
            trunked_attn_bias,
            mask,
            implementation=_sdpa_implementation(attention_backend),
        )
    out = out.reshape(out.shape[:-4] + (num_heads, -1, out.shape[-1]))
    if q_pad > 0:
        out = out[..., :-q_pad, :]
    out = jnp.swapaxes(out, -2, -3)
    if params.linear_g is not None:
        gate = sigmoid(linear(q_x, params.linear_g))
        gate = gate.reshape(gate.shape[:-1] + (num_heads, -1))
        out = out * gate
    out = out.reshape(out.shape[:-2] + (-1,))
    out = linear(out, params.linear_o)
    if sequence_mask is not None:
        out = out * sequence_mask.astype(out.dtype)[..., None]
    return out


_compiled_local_attention = jax.jit(
    local_attention,
    static_argnames=(
        "num_heads",
        "n_queries",
        "n_keys",
        "inf",
        "attention_backend",
    ),
)


def local_attention_pair_bias(
    a: jnp.ndarray,
    s: jnp.ndarray,
    z: jnp.ndarray,
    params: AttentionPairBiasParams,
    *,
    num_heads: int,
    n_queries: int,
    n_keys: int,
    sequence_mask: jnp.ndarray | None = None,
    attention_backend: str = "xla",
) -> jnp.ndarray:
    """Run adaptive local ``AttentionPairBias`` for AtomTransformer."""

    q = _apply_apb_norm(a, s, params.layernorm_a, params.has_s)
    kv = (
        _apply_apb_norm(q, s, params.layernorm_kv, params.has_s)
        if params.cross_attention_mode
        else q
    )
    bias = linear(layer_norm(z, params.layernorm_z), params.linear_z)
    bias = jnp.moveaxis(bias, -1, -4)
    out = local_attention(
        q,
        kv,
        params.attention,
        num_heads,
        trunked_attn_bias=bias,
        n_queries=n_queries,
        n_keys=n_keys,
        sequence_mask=sequence_mask,
        attention_backend=attention_backend,
    )
    if params.linear_a_last is not None:
        out = sigmoid(linear(s, params.linear_a_last)) * out
    return out


def _project_heads(
    x: jnp.ndarray,
    params: LinearParams,
    num_heads: int,
) -> jnp.ndarray:
    y = linear(x, params)
    y = y.reshape(y.shape[:-1] + (num_heads, -1))
    return jnp.swapaxes(y, -2, -3)


def _attention_qkv(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    attn_bias: jnp.ndarray | None,
    *,
    q_chunk_size: int | None,
    attention_backend: str,
) -> jnp.ndarray:
    n_q = q.shape[-2]
    if q_chunk_size is None or q_chunk_size <= 0 or q_chunk_size >= n_q:
        return _attention_qkv_chunk(
            q, k, v, attn_bias, attention_backend=attention_backend
        )
    chunks = []
    for start in range(0, n_q, q_chunk_size):
        end = min(start + q_chunk_size, n_q)
        q_chunk = q[..., start:end, :]
        bias_chunk = None
        if attn_bias is not None:
            bias_chunk = _normalize_attention_bias(attn_bias, q.ndim)[
                ..., start:end, :
            ]
        chunks.append(
            _attention_qkv_chunk(
                q_chunk,
                k,
                v,
                bias_chunk,
                attention_backend=attention_backend,
            )
        )
    return jnp.concatenate(chunks, axis=-2)


def _attention_qkv_chunk(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    attn_bias: jnp.ndarray | None,
    *,
    attention_backend: str,
) -> jnp.ndarray:
    if attention_backend != "xla":
        bias = None
        if attn_bias is not None:
            bias = _normalize_attention_bias(attn_bias, q.ndim)
        out = jax.nn.dot_product_attention(
            jnp.swapaxes(q, -3, -2),
            jnp.swapaxes(k, -3, -2),
            jnp.swapaxes(v, -3, -2),
            bias=bias,
            scale=1.0,
            implementation=_sdpa_implementation(attention_backend),
        )
        return jnp.swapaxes(out, -3, -2)
    logits = jnp.einsum("...hid,...hjd->...hij", q, k)
    if attn_bias is not None:
        logits = logits + _normalize_attention_bias(attn_bias, logits.ndim)
    probs = jax.nn.softmax(logits.astype(jnp.float32), axis=-1).astype(v.dtype)
    return jnp.einsum("...hij,...hjd->...hid", probs, v)


def _local_builtin_sdpa(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    bias: jnp.ndarray | None,
    mask: jnp.ndarray,
    *,
    implementation: str,
) -> jnp.ndarray:
    q_dpa = jnp.moveaxis(q, -4, -2)
    k_dpa = jnp.moveaxis(k, -4, -2)
    v_dpa = jnp.moveaxis(v, -4, -2)
    prefix = q_dpa.shape[:-3]
    q_flat = q_dpa.reshape((-1,) + q_dpa.shape[-3:])
    k_flat = k_dpa.reshape((-1,) + k_dpa.shape[-3:])
    v_flat = v_dpa.reshape((-1,) + v_dpa.shape[-3:])
    mask_flat = jnp.broadcast_to(mask, prefix + mask.shape[-2:]).reshape(
        (-1, 1) + mask.shape[-2:]
    )
    bias_flat = None
    if bias is not None:
        if bias.ndim == q.ndim - 1:
            bias = jnp.expand_dims(bias, axis=-4)
        bias = jnp.broadcast_to(bias, q.shape[:-1] + (k.shape[-2],))
        bias_dpa = jnp.moveaxis(bias, -4, -3)
        bias_flat = bias_dpa.reshape((-1,) + bias_dpa.shape[-3:])
    out = jax.nn.dot_product_attention(
        q_flat,
        k_flat,
        v_flat,
        bias=bias_flat,
        mask=mask_flat,
        scale=1.0,
        implementation=implementation,
    )
    return jnp.moveaxis(out.reshape(q_dpa.shape), -2, -4)


def _sdpa_implementation(attention_backend: str) -> str:
    if attention_backend == "xla_sdpa":
        return "xla"
    if attention_backend == "cudnn":
        return "cudnn"
    raise ValueError(f"unsupported attention backend: {attention_backend!r}")


def _normalize_attention_bias(
    attn_bias: jnp.ndarray,
    logits_ndim: int,
) -> jnp.ndarray:
    if attn_bias.ndim == logits_ndim:
        return attn_bias
    if attn_bias.ndim == logits_ndim - 1:
        return jnp.expand_dims(attn_bias, axis=-3)
    raise ValueError("attention bias rank must match logits rank or omit head axis")


def _apply_apb_norm(
    a: jnp.ndarray,
    s: jnp.ndarray | None,
    params: LayerNormParams | AdaptiveLayerNormParams | None,
    has_s: bool,
) -> jnp.ndarray:
    if params is None:
        raise ValueError("missing attention pair-bias normalization params")
    if has_s:
        if s is None:
            raise ValueError("adaptive attention pair-bias normalization requires s")
        if not isinstance(params, AdaptiveLayerNormParams):
            raise TypeError("expected AdaptiveLayerNormParams when has_s=True")
        return adaptive_layer_norm(a, s, params)
    if not isinstance(params, LayerNormParams):
        raise TypeError("expected LayerNormParams when has_s=False")
    return layer_norm(a, params)


def _local_qk_trunks(
    q: jnp.ndarray,
    k: jnp.ndarray,
    *,
    n_queries: int,
    n_keys: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, int]:
    if q.shape != k.shape:
        raise ValueError("local attention requires q and kv to share shape")
    if (
        n_queries <= 0
        or n_keys <= 0
        or n_keys < n_queries
        or n_queries % 2
        or n_keys % 2
    ):
        raise ValueError("invalid local attention window sizes")
    n = q.shape[-2]
    if n == 0:
        raise ValueError("local attention requires at least one atom")
    n_trunks = (n + n_queries - 1) // n_queries
    q_pad = n_trunks * n_queries - n
    pad_left = (n_keys - n_queries) // 2
    pad_right = int((n_trunks - 0.5) * n_queries + n_keys / 2 - n + 0.5)
    pad_q = ((0, 0),) * (q.ndim - 2) + ((0, q_pad), (0, 0))
    pad_k = ((0, 0),) * (k.ndim - 2) + ((pad_left, pad_right), (0, 0))
    q_padded = jnp.pad(q, pad_q)
    k_padded = jnp.pad(k, pad_k)
    q_trunked = q_padded.reshape(
        q.shape[:-2] + (n_trunks, n_queries, q.shape[-1])
    )
    k_trunked = gather_overlapping_windows(
        k_padded,
        axis=-2,
        n_windows=n_trunks,
        window_size=n_keys,
        stride=n_queries,
    )
    q_abs = jnp.arange(n_trunks * n_queries).reshape(n_trunks, n_queries)
    k_abs = (
        jnp.arange(n_keys)[None, :]
        + jnp.arange(n_trunks)[:, None] * n_queries
        - pad_left
    )
    mask = (q_abs[..., None] < n) & (k_abs[:, None, :] >= 0) & (
        k_abs[:, None, :] < n
    )
    return q_trunked, k_trunked, mask, q_pad
