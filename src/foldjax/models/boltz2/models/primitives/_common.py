"""Shared leaf primitives used across Boltz-2 JAX modules.

``layer_norm`` and ``linear`` were copy-pasted byte-identically across most
model files; they are centralized here. Modules import them aliased to the
private names they already use (``_layer_norm`` / ``_linear``) so call sites are
unchanged and numerics stay bit-identical.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def sigmoid(x: jnp.ndarray) -> jnp.ndarray:
    # Torch's low-precision sigmoid is one FP32 operator followed by one cast,
    # not separately rounded exp/add/div operations in the storage dtype.
    if x.dtype in (jnp.bfloat16, jnp.float16):
        return jax.nn.sigmoid(x.astype(jnp.float32)).astype(x.dtype)
    return jax.nn.sigmoid(x)


def layer_norm(
    x: jnp.ndarray,
    scale: jnp.ndarray,
    bias: jnp.ndarray,
    eps: float,
) -> jnp.ndarray:
    out_dtype = jnp.result_type(x, scale, bias)
    xf = x.astype(jnp.float32)
    mean = jnp.mean(xf, axis=-1, keepdims=True)
    variance = jnp.mean(jnp.square(xf - mean), axis=-1, keepdims=True)
    # CUDA AMP keeps ordinary LayerNorm and its original affine in FP32.
    # Narrowing the normalized value before the affine loses this island.
    normed = (xf - mean) * jax.lax.rsqrt(variance + eps)
    return (normed * scale.astype(jnp.float32) + bias.astype(jnp.float32)).astype(
        out_dtype
    )


def linear(
    x: jnp.ndarray,
    kernel: jnp.ndarray,
    bias: jnp.ndarray | None = None,
    *,
    compute_dtype: jnp.dtype | None = None,
) -> jnp.ndarray:
    if compute_dtype is None and kernel.dtype in (jnp.bfloat16, jnp.float16):
        compute_dtype = kernel.dtype
    if compute_dtype is not None:
        x = x.astype(compute_dtype)
        kernel = kernel.astype(compute_dtype)
        if bias is not None:
            bias = bias.astype(compute_dtype)
    if x.dtype in (jnp.bfloat16, jnp.float16) and (
        bias is not None or kernel.shape[-1] == 1
    ):
        # Native autocast Linear narrows its bias too, but adds it to the GEMM
        # accumulator before the single low-precision output rounding.
        # One-column GPU dots can lower to a low-precision reduction instead
        # of a tensor-core GEMM; explicitly retain the native FP32 accumulator.
        out = jnp.matmul(x, kernel, preferred_element_type=jnp.float32)
        if bias is not None:
            out = out + bias.astype(jnp.float32)
        return out.astype(x.dtype)
    out = x @ kernel
    if bias is not None:
        out = out + bias
    return out
