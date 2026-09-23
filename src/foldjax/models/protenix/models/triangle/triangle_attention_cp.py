"""Context-parallel dispatch for Protenix triangle attention."""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp

from foldjax.models._cp import cp_layout, shard_pair_rows
from foldjax.models._cp_attention import (
    gather_triangle_attention_2d_from_pair,
    resolve_gather_attention_body,
    resolve_ring_tile_kernel,
    ring_tile_kernel,
    ring_triangle_attention_2d_from_pair,
    triangle_attention_grid,
)
from foldjax.models.protenix.models.primitives.primitives import (
    layer_norm,
    linear,
    sigmoid,
)
from foldjax.models.protenix.models.triangle.triangle import (
    TriangleAttentionParams,
)
from foldjax.models.protenix.models.triangle.triangle import (
    triangle_attention as _triangle_attention,
)


def _project_heads(
    x: jax.Array,
    params,
    num_heads: int,
) -> jax.Array:
    projected = linear(x, params)
    projected = projected.reshape(projected.shape[:-1] + (num_heads, -1))
    return jnp.swapaxes(projected, -2, -3)


def triangle_attention(
    x: jnp.ndarray,
    mask: jnp.ndarray | None,
    params: TriangleAttentionParams,
    *,
    num_heads: int,
    starting: bool = True,
    inf: float = 1e9,
    q_chunk_size: int | None = None,
    attention_backend: str | None = None,
) -> jnp.ndarray:
    """Use the exact 2-D Fold-CP ring when a square mesh is active.

    ``q_chunk_size`` reaches the ring as its local row block, the same axis it
    blocks on the serial path: the ring rotates inside the row loop, so one
    block bounds the projections, the score tile and the accumulators at once
    without touching the rotation schedule. ``None`` leaves the ring's own
    rule (:func:`~foldjax.models._cp_attention.resolve_ring_row_block`) in
    force.

    What one ring step evaluates its tile with comes from the scope
    (:func:`~foldjax.models._cp_attention.ring_tile_kernel`) rather than from
    this signature, for the reason the Boltz-2 adapter records: nothing
    between the backend and here would have a use for the argument.
    ``attention_backend`` remains the serial kernel choice -- a fused
    attention over the global token axis, which no mesh can partition -- and
    is still refused for anything but ``xla`` here.

    ``triangle_attention_grid`` -- another scope
    (:func:`~foldjax.models._cp_attention.triangle_attention_grid`) --
    replaces the ring with the streamed gather
    (:func:`~foldjax.models._cp_attention.gather_triangle_attention_2d_from_pair`),
    whose local body is cuEquivariance's kernel on a GPU and the XLA reference
    elsewhere. The query then leaves ``project`` unscaled and the body applies
    ``hidden ** -0.5``, where this port's serial cuEquivariance call applies
    it (``triangle._triangle_attention_dense``); the ring's tile kernel is not
    read, because there is no ring to be a body of. The serial module's own
    2-D branch (``triangle._triangle_attention_ring_2d``) dispatches the same
    way.
    """

    if cp_layout() != "2d":
        return _triangle_attention(
            x,
            mask,
            params,
            num_heads=num_heads,
            starting=starting,
            inf=inf,
            q_chunk_size=q_chunk_size,
            attention_backend=attention_backend,
        )
    backend = (
        "xla"
        if attention_backend is None
        else attention_backend.removesuffix("_jit")
    )
    if backend != "xla":
        raise ValueError(
            "2-D context-parallel Protenix triangle attention requires the "
            "XLA ring backend; a fused attention over the global token axis "
            "cannot be partitioned. To run a fused kernel on each ring tile "
            "instead, pass triangle_attention_ring_kernel=tokamax."
        )
    gather_body = (
        resolve_gather_attention_body()
        if triangle_attention_grid() == "gather"
        else None
    )
    tile_kernel = (
        resolve_ring_tile_kernel(ring_tile_kernel()) if gather_body is None else None
    )
    if x.ndim != 3:
        raise ValueError(
            "Protenix 2-D triangle attention expects [N, N, C], "
            f"got shape {x.shape}"
        )

    if mask is None:
        mask = jnp.ones(x.shape[:-1], dtype=x.dtype)
    if not starting:
        x = jnp.swapaxes(x, -2, -3)
        mask = jnp.swapaxes(mask, -1, -2)
    x = shard_pair_rows(x, row_axis=-3)
    x = layer_norm(x, params.layer_norm)

    mask_bias = inf * (mask.astype(jnp.float32) - 1.0)
    mask_bias = mask_bias[..., :, None, None, :]
    triangle_bias = linear(x, params.linear)
    triangle_bias = jnp.moveaxis(triangle_bias, -1, -3)
    triangle_bias = jnp.expand_dims(triangle_bias, axis=-4)

    hidden = params.attention.linear_k.weight.shape[0] // num_heads
    scale = jnp.asarray(hidden**-0.5, dtype=x.dtype)

    def project(attention, rows, *, scaled=True):
        gate = None
        if attention.linear_g is not None:
            gate = sigmoid(linear(rows, attention.linear_g))
            gate = gate.reshape(gate.shape[:-1] + (num_heads, -1))
            gate = jnp.swapaxes(gate, -2, -3)
        query = _project_heads(rows, attention.linear_q, num_heads)
        return (
            query * scale if scaled else query,
            _project_heads(rows, attention.linear_k, num_heads),
            _project_heads(rows, attention.linear_v, num_heads),
            gate,
        )

    if gather_body is not None:
        out = gather_triangle_attention_2d_from_pair(
            x,
            triangle_bias,
            mask_bias,
            params.attention,
            project=functools.partial(project, scaled=False),
            scale=float(hidden**-0.5),
            q_block=q_chunk_size,
            body=gather_body,
        )
    else:
        out = ring_triangle_attention_2d_from_pair(
            x,
            triangle_bias,
            mask_bias,
            params.attention,
            project=project,
            q_block=q_chunk_size,
            tile_kernel=tile_kernel,
        )
    out = jnp.swapaxes(out, -2, -3)
    out = out.reshape(out.shape[:-2] + (-1,))
    out = linear(out, params.attention.linear_o)
    if not starting:
        out = jnp.swapaxes(out, -2, -3)
    return shard_pair_rows(out, row_axis=-3)


__all__ = ["TriangleAttentionParams", "triangle_attention"]
