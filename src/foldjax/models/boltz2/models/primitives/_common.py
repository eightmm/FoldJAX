"""Shared leaf primitives used across Boltz-2 JAX modules.

``layer_norm`` and ``linear`` were copy-pasted byte-identically across most
model files; they are centralized here. Modules import them aliased to the
private names they already use (``_layer_norm`` / ``_linear``) so call sites are
unchanged and numerics stay bit-identical.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def layer_norm(
    x: jnp.ndarray,
    scale: jnp.ndarray,
    bias: jnp.ndarray,
    eps: float,
) -> jnp.ndarray:
    in_dtype = x.dtype
    xf = x.astype(jnp.float32)
    mean = jnp.mean(xf, axis=-1, keepdims=True)
    variance = jnp.mean(jnp.square(xf - mean), axis=-1, keepdims=True)
    normed = ((xf - mean) * jax.lax.rsqrt(variance + eps)).astype(in_dtype)
    return normed * scale + bias


def linear(
    x: jnp.ndarray,
    kernel: jnp.ndarray,
    bias: jnp.ndarray | None = None,
) -> jnp.ndarray:
    out = x @ kernel
    if bias is not None:
        out = out + bias
    return out
