"""Context-parallel dispatch for Boltz-2 triangle attention.

The serial implementation remains the numerical ground truth. A 1-D CP mesh
continues to use its row-sharded path; a 2-D mesh uses the gather-free Fold-CP
ring from :mod:`foldjax.models._cp_attention`.
"""

from __future__ import annotations

from collections.abc import Mapping

import jax
import jax.numpy as jnp

from foldjax.models._cp import cp_layout, shard_pair_rows
from foldjax.models._cp_attention import ring_triangle_attention_2d
from foldjax.models.boltz2.models.primitives._common import layer_norm as _layer_norm
from foldjax.models.boltz2.models.triangle.triangle_attention import (
    resolve_matmul_precision,
    resolve_triangle_attention_chunk,
    resolve_triangle_attention_q_chunk,
)
from foldjax.models.boltz2.models.triangle.triangle_attention import (
    triangle_attention_forward as _triangle_attention_forward,
)

TriangleAttentionParams = Mapping[
    str, Mapping[str, jnp.ndarray | Mapping[str, jnp.ndarray]]
]


def _linear(
    x: jax.Array,
    kernel: jax.Array,
    precision: jax.lax.Precision,
) -> jax.Array:
    return jnp.matmul(x, kernel, precision=precision)


def triangle_attention_forward(
    params: TriangleAttentionParams,
    x: jnp.ndarray,
    mask: jnp.ndarray | None = None,
    starting: bool = True,
    inf: float = 1e9,
    eps: float = 1e-5,
    chunk_size: int = 128,
    q_chunk_size: int | None = None,
    matmul_precision: str = "highest",
    triangle_backend: str = "xla",
) -> jnp.ndarray:
    """Dispatch Boltz triangle attention to serial/1-D or the 2-D ring.

    ``chunk_size`` and ``q_chunk_size`` are intentionally unused on the ring
    path: the local ``N/sqrt(P)`` key tile is the score-memory bound, and
    splitting that tile would add launch overhead without reducing global
    communication.
    """

    if cp_layout() != "2d":
        return _triangle_attention_forward(
            params,
            x,
            mask,
            starting=starting,
            inf=inf,
            eps=eps,
            chunk_size=chunk_size,
            q_chunk_size=q_chunk_size,
            matmul_precision=matmul_precision,
            triangle_backend=triangle_backend,
        )
    if triangle_backend != "xla":
        raise ValueError(
            "2-D context-parallel triangle attention requires "
            "triangle_backend='xla': a fused kernel normalises one local tile "
            "before the ring can merge its softmax statistics."
        )
    if x.ndim != 4:
        raise ValueError(
            "Boltz 2-D triangle attention expects [B, N, N, C], "
            f"got shape {x.shape}"
        )

    precision = resolve_matmul_precision(matmul_precision)
    if mask is None:
        mask = jnp.ones(x.shape[:-1], dtype=x.dtype)
    if not starting:
        x = jnp.swapaxes(x, -2, -3)
        mask = jnp.swapaxes(mask, -1, -2)

    x = shard_pair_rows(x, row_axis=-3)
    x = _layer_norm(
        x,
        params["layer_norm"]["scale"],
        params["layer_norm"]["bias"],
        eps,
    )
    mask_bias = inf * (mask.astype(jnp.float32) - 1.0)
    mask_bias = mask_bias[..., :, None, None, :]
    triangle_bias = _linear(x, params["linear"]["kernel"], precision)
    triangle_bias = jnp.transpose(triangle_bias, (0, 3, 1, 2))[:, None]

    no_heads = triangle_bias.shape[-3]
    hidden = params["mha"]["linear_g"]["kernel"].shape[-1] // no_heads
    qg = _linear(
        x,
        jnp.concatenate(
            (
                params["mha"]["linear_q"]["kernel"],
                params["mha"]["linear_g"]["kernel"],
            ),
            axis=-1,
        ),
        precision,
    )
    query, gate = jnp.split(qg, 2, axis=-1)
    kv = _linear(
        x,
        jnp.concatenate(
            (
                params["mha"]["linear_k"]["kernel"],
                params["mha"]["linear_v"]["kernel"],
            ),
            axis=-1,
        ),
        precision,
    )
    key, value = jnp.split(kv, 2, axis=-1)

    def split_heads(array: jax.Array) -> jax.Array:
        array = array.reshape(array.shape[:-1] + (no_heads, hidden))
        return jnp.swapaxes(array, -2, -3)

    query = split_heads(query)
    key = split_heads(key)
    value = split_heads(value)
    query = query / jnp.sqrt(jnp.asarray(hidden, dtype=query.dtype))
    out = ring_triangle_attention_2d(
        query,
        key,
        value,
        triangle_bias,
        mask_bias,
        precision=precision,
    )
    out = jnp.swapaxes(out, -2, -3)

    gate = jax.nn.sigmoid(gate)
    gate = gate.reshape(gate.shape[:-1] + (no_heads, hidden))
    out = out * gate
    out = out.reshape(out.shape[:-2] + (no_heads * hidden,))
    out = _linear(out, params["mha"]["linear_o"]["kernel"], precision)
    if not starting:
        out = jnp.swapaxes(out, -2, -3)
    return shard_pair_rows(out, row_axis=-3)


__all__ = [
    "resolve_matmul_precision",
    "resolve_triangle_attention_chunk",
    "resolve_triangle_attention_q_chunk",
    "triangle_attention_forward",
]
