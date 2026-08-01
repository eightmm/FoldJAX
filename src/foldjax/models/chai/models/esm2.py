"""Exact JAX forward for Chai's traced fp16 ESM2-t36-3B encoder."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

Esm2State = Mapping[str, Any]

_LAYER_KEYS = (
    "q_weight",
    "q_bias",
    "k_weight",
    "k_bias",
    "v_weight",
    "v_bias",
    "out_weight",
    "out_bias",
    "rotary_inv_freq",
    "attention_norm_weight",
    "attention_norm_bias",
    "fc1_weight",
    "fc1_bias",
    "fc2_weight",
    "fc2_bias",
    "final_norm_weight",
    "final_norm_bias",
)


def _linear(x: jax.Array, weight: jax.Array, bias: jax.Array) -> jax.Array:
    return x @ jnp.swapaxes(weight, -1, -2) + bias


def _layer_norm(
    x: jax.Array, weight: jax.Array, bias: jax.Array
) -> jax.Array:
    source_dtype = x.dtype
    x32 = x.astype(jnp.float32)
    mean = jnp.mean(x32, axis=-1, keepdims=True)
    variance = jnp.mean(jnp.square(x32 - mean), axis=-1, keepdims=True)
    normalized = (x32 - mean) * jax.lax.rsqrt(variance + 1e-5)
    return (
        normalized * weight.astype(jnp.float32) + bias.astype(jnp.float32)
    ).astype(source_dtype)


def _rotate_half(x: jax.Array) -> jax.Array:
    left, right = jnp.split(x, 2, axis=-1)
    return jnp.concatenate([-right, left], axis=-1)


def apply_rotary(
    query: jax.Array,
    key: jax.Array,
    inv_freq: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Apply the artifact's float32-frequency, fp16-output rotary embedding."""

    positions = jnp.arange(query.shape[1], dtype=jnp.float32)
    frequencies = positions[:, None] * inv_freq.astype(jnp.float32)[None, :]
    embedding = jnp.concatenate([frequencies, frequencies], axis=-1)
    cosine = jnp.cos(embedding).astype(query.dtype)[None, :, None, :]
    sine = jnp.sin(embedding).astype(query.dtype)[None, :, None, :]
    return (
        query * cosine + _rotate_half(query) * sine,
        key * cosine + _rotate_half(key) * sine,
    )


def esm2_layer(
    x: jax.Array,
    padding_mask: jax.Array,
    layer: Esm2State,
    *,
    attention_implementation: str = "xla",
) -> jax.Array:
    """One traced ESM2 pre-LN attention/FFN residual block."""

    residual = x
    normalized = _layer_norm(
        x,
        layer["attention_norm_weight"],
        layer["attention_norm_bias"],
    )
    query = _linear(normalized, layer["q_weight"], layer["q_bias"])
    key = _linear(normalized, layer["k_weight"], layer["k_bias"])
    value = _linear(normalized, layer["v_weight"], layer["v_bias"])
    head_dim = 2 * layer["rotary_inv_freq"].shape[-1]
    if query.shape[-1] % head_dim:
        raise ValueError("ESM hidden size must be divisible by rotary head size")
    num_heads = query.shape[-1] // head_dim
    head_shape = query.shape[:-1] + (num_heads, head_dim)
    query, key, value = (
        query.reshape(head_shape),
        key.reshape(head_shape),
        value.reshape(head_shape),
    )
    query, key = apply_rotary(query, key, layer["rotary_inv_freq"])
    allowed_keys = (~padding_mask)[:, None, None, :]
    attended = jax.nn.dot_product_attention(
        query,
        key,
        value,
        mask=allowed_keys,
        implementation=attention_implementation,
    ).reshape(x.shape)
    x = residual + _linear(attended, layer["out_weight"], layer["out_bias"])

    residual = x
    normalized = _layer_norm(
        x,
        layer["final_norm_weight"],
        layer["final_norm_bias"],
    )
    hidden = _linear(normalized, layer["fc1_weight"], layer["fc1_bias"])
    hidden = hidden * 0.5 * (1.0 + jax.lax.erf(hidden / jnp.sqrt(2.0)))
    return residual + _linear(hidden, layer["fc2_weight"], layer["fc2_bias"])


def stack_esm2_layers(layers: Sequence[Esm2State]) -> dict[str, np.ndarray]:
    """Stack identically-shaped layer states for a compact ``lax.scan``."""

    if not layers:
        raise ValueError("at least one ESM2 layer is required")
    if any(set(layer) != set(_LAYER_KEYS) for layer in layers):
        raise ValueError("ESM2 layer tensor names mismatch")
    return {
        key: np.stack([np.asarray(layer[key]) for layer in layers])
        for key in _LAYER_KEYS
    }


def esm2_forward(
    tokens: jax.Array,
    state: Esm2State,
    *,
    attention_implementation: str = "xla",
) -> jax.Array:
    """Return Chai's final fp16 ESM hidden state including special tokens."""

    tokens = jnp.asarray(tokens, dtype=jnp.int32)
    if tokens.ndim != 2:
        raise ValueError("ESM tokens must have shape (batch, sequence)")
    padding_mask = tokens == 1
    x = jnp.asarray(state["embed_tokens_weight"])[tokens]
    masked_tokens = tokens == 32
    x = jnp.where(masked_tokens[..., None], 0.0, x)
    source_lengths = jnp.sum(~padding_mask, axis=-1)
    observed_mask_ratio = jnp.sum(masked_tokens, axis=-1) / source_lengths
    x = x * 0.88 / (1.0 - observed_mask_ratio[:, None, None])
    x = jnp.where(padding_mask[..., None], 0.0, x)

    layers = jax.tree.map(jnp.asarray, state["layers"])

    def apply_layer(carry: jax.Array, layer: Esm2State):
        output = esm2_layer(
            carry,
            padding_mask,
            layer,
            attention_implementation=attention_implementation,
        )
        return output, None

    x, _ = jax.lax.scan(apply_layer, x, layers)
    return _layer_norm(
        x,
        jnp.asarray(state["final_norm_weight"]),
        jnp.asarray(state["final_norm_bias"]),
    ).astype(jnp.float16)


__all__ = [
    "apply_rotary",
    "esm2_forward",
    "esm2_layer",
    "stack_esm2_layers",
]
