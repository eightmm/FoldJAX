"""Context-parallel dispatch for OpenFold3 triangle attention.

The serial module remains the numerical reference. A one-dimensional CP mesh
uses the existing row-sharded implementation, while a square two-dimensional
mesh uses the gather-free Fold-CP ring implemented in
:mod:`foldjax.models._cp_attention`.
"""

from __future__ import annotations

import jax.numpy as jnp

from foldjax.models._cp import cp_layout, shard_pair_rows
from foldjax.models._cp_attention import ring_triangle_attention_2d
from foldjax.models.openfold3.models.attention import flatten_heads, split_heads
from foldjax.models.openfold3.models.primitives import (
    jax_sigmoid,
    layer_norm,
    linear,
)
from foldjax.models.openfold3.models.triangle_attention import (
    TriangleAttentionParams,
    _project_triangle_bias,
)
from foldjax.models.openfold3.models.triangle_attention import (
    triangle_attention as _triangle_attention,
)


def triangle_attention(
    x: jnp.ndarray,
    params: TriangleAttentionParams,
    *,
    no_heads: int,
    mask: jnp.ndarray | None = None,
    starting: bool = True,
    transpose_bias: bool = False,
    inf: float = 1e9,
    eps: float = 1e-5,
    chunk_size: int | None = None,
    backend: str | None = None,
) -> jnp.ndarray:
    """Apply serial/1-D attention or exact 2-D Fold-CP ring attention.

    On the 2-D path the query tile remains resident and K/V/mask/bias tiles
    rotate around the mesh. A fused local attention backend is rejected: it
    normalises one key tile before the online softmax can combine all ring
    steps, so using it would change the function.
    """

    if cp_layout() != "2d":
        return _triangle_attention(
            x,
            params,
            no_heads=no_heads,
            mask=mask,
            starting=starting,
            transpose_bias=transpose_bias,
            inf=inf,
            eps=eps,
            chunk_size=chunk_size,
            backend=backend,
        )
    resolved_backend = "xla" if backend is None else backend
    if resolved_backend != "xla":
        raise ValueError(
            "2-D context-parallel OpenFold3 triangle attention requires "
            "backend='xla'; fused local softmax outputs cannot be merged "
            "across rotating key tiles."
        )
    if x.ndim < 3:
        raise ValueError(
            "OpenFold3 2-D triangle attention expects [..., N, N, C], "
            f"got shape {x.shape}"
        )

    if mask is None:
        mask = jnp.ones(x.shape[:-1], dtype=x.dtype)
    if not starting:
        x = jnp.swapaxes(x, -2, -3)
        mask = jnp.swapaxes(mask, -1, -2)

    x = shard_pair_rows(x, row_axis=-3)
    x = layer_norm(x, params.layer_norm, eps=eps)
    mask_bias = (inf * (mask.astype(jnp.float32) - 1.0))[..., :, None, None, :]
    triangle_bias = _project_triangle_bias(x, params, transpose_bias)
    triangle_bias = jnp.expand_dims(triangle_bias, -4)

    query = jnp.swapaxes(split_heads(linear(x, params.mha.linear_q), no_heads), -2, -3)
    key = jnp.swapaxes(split_heads(linear(x, params.mha.linear_k), no_heads), -2, -3)
    value = jnp.swapaxes(split_heads(linear(x, params.mha.linear_v), no_heads), -2, -3)
    query = query / jnp.sqrt(jnp.asarray(query.shape[-1], dtype=query.dtype))

    out = ring_triangle_attention_2d(
        query,
        key,
        value,
        triangle_bias,
        mask_bias,
    )
    out = jnp.swapaxes(out, -2, -3)
    if params.mha.linear_g is not None:
        gate = split_heads(jax_sigmoid(linear(x, params.mha.linear_g)), no_heads)
        out = out * gate
    out = linear(flatten_heads(out), params.mha.linear_o)

    if not starting:
        out = jnp.swapaxes(out, -2, -3)
    return shard_pair_rows(out, row_axis=-3)


__all__ = ["TriangleAttentionParams", "triangle_attention"]
