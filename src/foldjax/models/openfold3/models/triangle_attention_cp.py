"""Context-parallel dispatch for OpenFold3 triangle attention.

The serial module remains the numerical reference. A one-dimensional CP mesh
uses the existing row-sharded implementation, while a square two-dimensional
mesh uses the gather-free Fold-CP ring implemented in
:mod:`foldjax.models._cp_attention`.
"""

from __future__ import annotations

import jax.numpy as jnp

from foldjax.models._cp import cp_layout, shard_pair_rows
from foldjax.models._cp_attention import ring_triangle_attention_2d_from_pair
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

    ``chunk_size`` reaches the ring as its local row block, the axis the
    chunked serial path blocks as well: the rotation happens inside that loop,
    so one block bounds its own projections, its score tile and its
    accumulators. ``None`` leaves the ring's own rule
    (:func:`~foldjax.models._cp_attention.resolve_ring_row_block`) in force.
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

    def project(mha, rows):
        def heads(array: jnp.ndarray) -> jnp.ndarray:
            return jnp.swapaxes(split_heads(array, no_heads), -2, -3)

        query = heads(linear(rows, mha.linear_q))
        query = query / jnp.sqrt(jnp.asarray(query.shape[-1], dtype=query.dtype))
        gate = None
        if mha.linear_g is not None:
            gate = heads(jax_sigmoid(linear(rows, mha.linear_g)))
        return (
            query,
            heads(linear(rows, mha.linear_k)),
            heads(linear(rows, mha.linear_v)),
            gate,
        )

    out = ring_triangle_attention_2d_from_pair(
        x,
        triangle_bias,
        mask_bias,
        params.mha,
        project=project,
        q_block=chunk_size,
    )
    out = jnp.swapaxes(out, -2, -3)
    out = linear(flatten_heads(out), params.mha.linear_o)

    if not starting:
        out = jnp.swapaxes(out, -2, -3)
    return shard_pair_rows(out, row_axis=-3)


__all__ = ["TriangleAttentionParams", "triangle_attention"]
