"""Pure JAX triangle multiplication blocks for the Boltz-2 port."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Literal

import jax
import jax.numpy as jnp

from foldjax.models.boltz2.models.primitives._common import layer_norm as _layer_norm
from foldjax.models.boltz2.models.primitives.glu_backend import gated_linear_unit

TriangleDirection = Literal["outgoing", "incoming"]
TriangleMultiplicationParams = Mapping[str, Mapping[str, jnp.ndarray]]


def triangle_multiplication_forward(
    params: TriangleMultiplicationParams,
    x: jnp.ndarray,
    mask: jnp.ndarray,
    direction: TriangleDirection,
    eps: float = 1e-5,
    chunk_size: int = 128,
    glu_backend: str = "xla",
    contraction_precision: str = "float32",
) -> jnp.ndarray:
    """Run Boltz triangle multiplication with mapped PyTorch parameters.

    ``glu_backend="tokamax"`` runs the projection+gate sigmoid GLU through the
    fused Triton kernel (GPU, low precision); the triangle contraction stays in
    XLA either way (cuBLAS-bound, no custom kernel — matches AF3). ``"xla"``
    (default) keeps the bit-exact elementwise gate.
    """

    backend = os.getenv("BOLTZ_JAX_TRIANGLE_MULTIPLICATION_BACKEND", "cueq")
    if backend == "cueq":
        from foldjax.models.boltz2.models.triangle.triangle_cueq import (
            cueq_triangle_multiplication_forward,
        )

        return cueq_triangle_multiplication_forward(
            params, x, mask, direction, eps=eps
        )
    if backend != "xla":
        msg = f"Unsupported triangle multiplication backend: {backend!r}"
        raise ValueError(msg)

    x = _layer_norm(x, params["norm_in"]["scale"], params["norm_in"]["bias"], eps)
    out_dtype = x.dtype
    x_in = x
    mask = mask.astype(x.dtype)
    # sigmoid GLU: sigmoid(g_in(x)) * p_in(x)
    projected = gated_linear_unit(
        x,
        params["g_in"]["kernel"],
        params["p_in"]["kernel"],
        jax.nn.sigmoid,
        backend=glu_backend,
    )
    projected = projected * mask[..., None]
    # Coordinate sampling amplifies the small reduction-order differences of a
    # BF16-output contraction. Keep the historical FP32 contraction as the
    # accuracy default; the explicit BF16 mode is available for throughput
    # experiments where that drift has been validated for the target workload.
    if contraction_precision == "float32":
        contraction_input = projected.astype(jnp.float32)
    elif contraction_precision == "bf16":
        contraction_input = projected
    else:
        msg = f"Unsupported contraction_precision: {contraction_precision!r}"
        raise ValueError(msg)
    a, b = jnp.split(contraction_input, 2, axis=-1)

    out = _chunked_triangle_einsum(a, b, direction, chunk_size)
    out = out.astype(out_dtype)

    out = _layer_norm(out, params["norm_out"]["scale"], params["norm_out"]["bias"], eps)
    out = _linear(out, params["p_out"]["kernel"])
    gate = jax.nn.sigmoid(_linear(x_in, params["g_out"]["kernel"]))
    return out * gate


def _chunked_triangle_einsum(
    a: jnp.ndarray,
    b: jnp.ndarray,
    direction: TriangleDirection,
    chunk_size: int,
) -> jnp.ndarray:
    """Compute the triangle contraction in chunks over the output i axis.

    The contraction is over k (not split), so chunking the output i axis is
    exact: out[:, i_block] depends only on the i-block slice of ``a``.
    """

    if direction == "outgoing":
        n = a.shape[1]  # i is axis 1 of a in "bikd,bjkd->bijd"

        def block(start: int, size: int) -> jnp.ndarray:
            return jnp.einsum(
                "bikd,bjkd->bijd", jax.lax.dynamic_slice_in_dim(a, start, size, 1), b
            )

    elif direction == "incoming":
        n = a.shape[2]  # i is axis 2 of a in "bkid,bkjd->bijd"

        def block(start: int, size: int) -> jnp.ndarray:
            return jnp.einsum(
                "bkid,bkjd->bijd", jax.lax.dynamic_slice_in_dim(a, start, size, 2), b
            )

    else:
        msg = f"Unsupported triangle multiplication direction: {direction!r}"
        raise ValueError(msg)

    if chunk_size <= 0 or chunk_size >= n:
        return block(0, n)

    out = jnp.zeros((a.shape[0], n, b.shape[1], a.shape[-1]), dtype=a.dtype)
    for start in range(0, n, chunk_size):
        size = min(chunk_size, n - start)
        out = out.at[:, start : start + size].set(block(start, size))
    return out


def _linear(x: jnp.ndarray, kernel: jnp.ndarray) -> jnp.ndarray:
    return x @ kernel
