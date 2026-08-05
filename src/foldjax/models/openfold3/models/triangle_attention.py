"""Triangle attention (AF3 Algorithms 14 and 15).

Upstream builds this as one class with a ``starting`` flag that transposes the
two spatial axes before and after the attention. ``PairBlock`` notably does *not*
use ``starting=False``: it constructs two ``starting=True`` layers and transposes
the pair representation itself between them, so the flag here exists for
faithfulness to the standalone module rather than for the Pairformer path.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from foldjax.models.openfold3.models.attention import AttentionParams, attention
from foldjax.models.openfold3.models.primitives import (
    LayerNormParams,
    LinearParams,
    layer_norm,
    linear,
)
from foldjax.models.openfold3.models.triangle import permute_final_dims


class TriangleAttentionParams(NamedTuple):
    """Parameters for ``TriangleAttention``.

    ``linear_z`` projects the pair representation to one bias per head and is
    bias-free upstream (``mha_bias_init``).
    """

    layer_norm: LayerNormParams
    linear_z: LinearParams
    mha: AttentionParams


def triangle_attention(
    x: jnp.ndarray,
    params: TriangleAttentionParams,
    *,
    no_heads: int,
    mask: jnp.ndarray | None = None,
    starting: bool = True,
    inf: float = 1e9,
    eps: float = 1e-5,
    chunk_size: int | None = None,
) -> jnp.ndarray:
    """Apply one triangle attention layer.

    Args:
        x: ``[..., I, J, C_in]`` pair representation.
        params: mapped layer parameters.
        no_heads: attention head count.
        mask: ``[..., I, J]`` pair mask; ``None`` means all ones.
        starting: ``True`` for Algorithm 14, ``False`` for Algorithm 15.
        inf: masking constant; upstream uses ``inf * (mask - 1)``.
        eps: layer norm epsilon.
        chunk_size: rows of ``I`` to attend at a time. ``None`` does it in one
            go. See :func:`_chunked_attention` for why this is exact.

    Returns:
        ``[..., I, J, C_in]`` update. Upstream returns the update only.
    """
    if mask is None:
        mask = jnp.ones(x.shape[:-1], dtype=x.dtype)

    if not starting:
        x = jnp.swapaxes(x, -2, -3)
        mask = jnp.swapaxes(mask, -1, -2)

    x = layer_norm(x, params.layer_norm, eps=eps)

    # [..., I, 1, 1, J]
    mask_bias = (inf * (mask - 1.0))[..., :, None, None, :]

    # [..., H, I, J] -> [..., 1, H, I, J]
    triangle_bias = permute_final_dims(linear(x, params.linear_z), (2, 0, 1))
    triangle_bias = jnp.expand_dims(triangle_bias, -4)

    if chunk_size is None:
        out = attention(
            x, x, params.mha, no_heads=no_heads, biases=(mask_bias, triangle_bias)
        )
    else:
        out = _chunked_attention(
            x,
            mask_bias,
            triangle_bias,
            params.mha,
            no_heads=no_heads,
            chunk_size=chunk_size,
        )

    if not starting:
        out = jnp.swapaxes(out, -2, -3)
    return out


def _chunked_attention(
    x: jnp.ndarray,
    mask_bias: jnp.ndarray,
    triangle_bias: jnp.ndarray,
    params: AttentionParams,
    *,
    no_heads: int,
    chunk_size: int,
) -> jnp.ndarray:
    """Attend over ``J`` for a block of ``I`` rows at a time.

    This is exact, not an approximation. ``I`` is a batch axis here -- each row of
    the pair representation attends over ``J`` independently -- so splitting it
    changes nothing about what any output element depends on. What it changes is the
    peak: the attention logits are ``[I, H, J, J]``, which at 384 tokens is 906 MiB
    and at 849 tokens is 9.8 GiB, and a chunk of 64 rows cuts that by ``I /
    chunk_size``.

    ``triangle_bias`` is deliberately *not* chunked. Its ``I`` axis is the query
    axis of the attention, not the batch axis -- the pair representation is square,
    so the two have the same length and slicing the wrong one is easy and silent.
    It is also shared across every row, so it must be built from the whole input.

    ``lax.map`` rather than a Python loop: the loop would unroll one attention body
    per chunk, and with 48 blocks and two of these per block the compile time grows
    with sequence length instead of staying flat.
    """
    rows = x.shape[-3]
    n_chunks = -(-rows // chunk_size)
    padding = n_chunks * chunk_size - rows

    lead = x.shape[:-3]
    flat_x = x.reshape((-1, *x.shape[-3:]))
    flat_mask = mask_bias.reshape((-1, *mask_bias.shape[-4:]))
    flat_bias = triangle_bias.reshape((-1, *triangle_bias.shape[-4:]))

    if padding:
        # Padding rows are attended over and then discarded. The mask bias is padded
        # with zeros, not with -inf: a fully masked row makes softmax divide by zero
        # and returns NaN, which would propagate out of the slice that drops it.
        flat_x = jnp.pad(flat_x, ((0, 0), (0, padding), (0, 0), (0, 0)))
        flat_mask = jnp.pad(
            flat_mask, ((0, 0), (0, padding), (0, 0), (0, 0), (0, 0))
        )

    def one_chunk(index: jnp.ndarray) -> jnp.ndarray:
        start = index * chunk_size
        return attention(
            jax.lax.dynamic_slice_in_dim(flat_x, start, chunk_size, axis=1),
            jax.lax.dynamic_slice_in_dim(flat_x, start, chunk_size, axis=1),
            params,
            no_heads=no_heads,
            biases=(
                jax.lax.dynamic_slice_in_dim(flat_mask, start, chunk_size, axis=1),
                flat_bias,
            ),
        )

    # [n_chunks, B, chunk, J, C] -> [B, n_chunks * chunk, J, C]
    chunks = jax.lax.map(one_chunk, jnp.arange(n_chunks))
    merged = jnp.swapaxes(chunks, 0, 1).reshape(
        (flat_x.shape[0], n_chunks * chunk_size, *chunks.shape[-2:])
    )
    return merged[:, :rows].reshape((*lead, rows, *chunks.shape[-2:]))
