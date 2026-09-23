"""Context-parallel dispatch for OpenFold3 triangle attention.

The serial module remains the numerical reference. A one-dimensional CP mesh
uses the existing row-sharded implementation, while a square two-dimensional
mesh uses the gather-free Fold-CP ring implemented in
:mod:`foldjax.models._cp_attention`.
"""

from __future__ import annotations

import functools

import jax.numpy as jnp

from foldjax.models._cp import cp_layout, shard_pair_rows
from foldjax.models._cp_attention import (
    gather_triangle_attention_2d_from_pair,
    resolve_gather_attention_body,
    ring_triangle_attention_2d_from_pair,
    triangle_attention_grid,
)
from foldjax.models.openfold3.models.attention import flatten_heads, split_heads
from foldjax.models.openfold3.models.primitives import (
    jax_sigmoid,
    layer_norm,
    linear,
)
from foldjax.models.openfold3.models.triangle_attention import (
    TriangleAttentionParams,
    _project_triangle_bias,
    _refuse_ring_tile_scope,
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

    The fused *tile* kernel Boltz-2 and Protenix can opt into is refused here
    rather than ignored. Nothing about it is port-specific -- the ring below is
    the same object -- so what is missing is only that no OpenFold3 backend
    option reaches this scope and no OpenFold3 run has measured it. A scope
    this adapter silently dropped would make a spelled request and an omitted
    one compile the same program under two names.

    ``triangle_attention_grid`` -- a scope this port's backend *does* offer
    (:func:`~foldjax.models._cp_attention.triangle_attention_grid`) --
    replaces the ring with the streamed gather
    (:func:`~foldjax.models._cp_attention.gather_triangle_attention_2d_from_pair`),
    whose local body is cuEquivariance's kernel on a GPU and the XLA reference
    elsewhere. The query then leaves ``project`` unscaled and the body applies
    ``D ** -0.5``, where this port's serial cuEquivariance call applies it
    (``triangle_attention._cueq_attention``). The serial module's own 2-D
    branch (``triangle_attention._ring_attention``) dispatches the same way.
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
            "backend='xla'; a fused attention over the global token axis "
            "cannot be partitioned."
        )
    _refuse_ring_tile_scope()
    gather_body = (
        resolve_gather_attention_body()
        if triangle_attention_grid() == "gather"
        else None
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

    def project(mha, rows, *, scaled=True):
        def heads(array: jnp.ndarray) -> jnp.ndarray:
            return jnp.swapaxes(split_heads(array, no_heads), -2, -3)

        query = heads(linear(rows, mha.linear_q))
        if scaled:
            query = query / jnp.sqrt(
                jnp.asarray(query.shape[-1], dtype=query.dtype)
            )
        gate = None
        if mha.linear_g is not None:
            gate = heads(jax_sigmoid(linear(rows, mha.linear_g)))
        return (
            query,
            heads(linear(rows, mha.linear_k)),
            heads(linear(rows, mha.linear_v)),
            gate,
        )

    if gather_body is not None:
        head_dim = params.mha.linear_q.weight.shape[0] // no_heads
        out = gather_triangle_attention_2d_from_pair(
            x,
            triangle_bias,
            mask_bias,
            params.mha,
            project=functools.partial(project, scaled=False),
            scale=float(head_dim) ** -0.5,
            q_block=chunk_size,
            body=gather_body,
        )
    else:
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
