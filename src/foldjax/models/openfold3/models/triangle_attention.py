"""Triangle attention (AF3 Algorithms 14 and 15).

Upstream builds this as one class with a ``starting`` flag that transposes the
two spatial axes before and after the attention. ``PairBlock`` notably does *not*
use ``starting=False``: it constructs two ``starting=True`` layers and transposes
the pair representation itself between them, so the flag here exists for
faithfulness to the standalone module rather than for the Pairformer path.
"""

from __future__ import annotations

import os
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec

from foldjax.models._cp import (
    cp_layout,
    cp_mesh,
    cp_row_shards,
    pair_row_spec,
    shard_pair_rows,
)
from foldjax.models._cp_attention import ring_triangle_attention_2d
from foldjax.models.openfold3.models.attention import (
    AttentionParams,
    attention,
    flatten_heads,
    split_heads,
)
from foldjax.models.openfold3.models.primitives import (
    LayerNormParams,
    LinearParams,
    jax_sigmoid,
    layer_norm,
    linear,
)
from foldjax.models.openfold3.models.triangle import permute_final_dims


def _default_backend() -> str:
    """Which kernel to use, from ``OPENFOLD3_TRIANGLE_BACKEND``.

    Defaults to the fused cuEquivariance kernel, which is what the other three
    AF3-family ports here already run. This is a deliberate step away from
    upstream's *default* -- ``Attention.__init__`` takes
    ``use_cueq_triangle_kernels`` and ships it ``False`` -- but not away from
    upstream's code: it is one of the three fused paths upstream offers. Measured
    on this port, one sample, warm cache:

    ============  ==================  =================
    tokens        peak                time
    ============  ==================  =================
    1003          22.13 -> 10.17 GiB  80.8 -> 58.0 s
    3012          62.47 -> 62.49 GiB  1134 -> 934 s
    ============  ==================  =================

    The memory win is size-dependent and disappears at 3012, where the XLA path
    already blocks the score tensor to 58 rows and the peak is set by a dozen
    pair-sized buffers instead. The time win holds at both. Nothing measured
    worse.

    Confidence moves by about +0.0014 mean pLDDT and +0.0012 pTM, consistent with
    the kernel's accumulation order and its TF32 matmuls -- and TF32 is what
    upstream runs too (``set_float32_matmul_precision("high")``), so the fused
    path is arguably the closer match on that axis rather than the further one.

    ``OPENFOLD3_TRIANGLE_BACKEND=xla`` restores the blocked path -- on a machine
    without the cuEquivariance wheel, or to reproduce a number taken before this
    changed. There is no automatic fallback, for the reason the Protenix port
    gives: a backend that silently changes kernel when a wheel is missing means
    two machines running two different programs under one setting.

    Read here rather than threaded through the config so the switch reaches every
    triangle attention in the model, including the ones inside the template stack
    and the confidence head, without six signatures growing an argument.
    """
    return os.environ.get("OPENFOLD3_TRIANGLE_BACKEND", "cueq").lower()


class TriangleAttentionParams(NamedTuple):
    """Parameters for ``TriangleAttention``.

    ``linear_z`` projects the pair representation to one bias per head and is
    bias-free upstream (``mha_bias_init``).
    """

    layer_norm: LayerNormParams
    linear_z: LinearParams
    mha: AttentionParams


def _project_triangle_bias(
    x: jnp.ndarray,
    params: TriangleAttentionParams,
    transpose_bias: bool,
) -> jnp.ndarray:
    """Project the pair bias with the checkpoint's ending-node orientation."""
    projected = linear(x, params.linear_z)
    permutation = (2, 1, 0) if transpose_bias else (2, 0, 1)
    return permute_final_dims(projected, permutation)


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
    """Apply one triangle attention layer.

    Args:
        x: ``[..., I, J, C_in]`` pair representation.
        params: mapped layer parameters.
        no_heads: attention head count.
        mask: ``[..., I, J]`` pair mask; ``None`` means all ones.
        starting: ``True`` for Algorithm 14, ``False`` for Algorithm 15.
        transpose_bias: use OpenBind v0.5's corrected ending-node pair-bias
            orientation. Pair blocks set this only for their second attention.
        inf: masking constant; upstream uses ``inf * (mask - 1)``.
        eps: layer norm epsilon.
        chunk_size: rows of ``I`` to attend at a time. ``None`` does it in one
            go. See :func:`_chunked_attention` for why this is exact.
        backend: ``"xla"`` builds the score tensor and blocks it by ``chunk_size``;
            ``"cueq"`` calls the fused cuEquivariance kernel, which never forms it
            and therefore ignores ``chunk_size``. Upstream has the same choice --
            ``Attention`` takes ``use_cueq_triangle_kernels`` -- and defaults it off,
            which is why this does too.

    Returns:
        ``[..., I, J, C_in]`` update. Upstream returns the update only.
    """
    if cp_mesh() is not None:
        # The fused kernel is an FFI call the SPMD partitioner cannot split --
        # the same constraint upstream Fold-CP states as "torch kernels only".
        # An *unset* backend resolves to the blocked XLA path instead of the
        # cueq environment default; an explicit request is honoured, because
        # inside `shard_map` the kernel runs on whole local rows.
        if backend is None and "OPENFOLD3_TRIANGLE_BACKEND" not in os.environ:
            backend = "xla"
        elif backend is None:
            backend = _default_backend()
        if backend not in {"xla", "cueq"}:
            raise ValueError(
                "context-parallel triangle attention supports the XLA and "
                f"cueq backends; got {backend!r}."
            )
        return _triangle_attention_cp(
            x,
            params,
            no_heads=no_heads,
            mask=mask,
            starting=starting,
            transpose_bias=transpose_bias,
            inf=inf,
            eps=eps,
            chunk_size=chunk_size,
            local_backend=backend,
        )
    if backend is None:
        backend = _default_backend()
    if backend not in {"xla", "cueq"}:
        raise ValueError(f"unsupported triangle attention backend: {backend!r}")

    if mask is None:
        mask = jnp.ones(x.shape[:-1], dtype=x.dtype)

    if not starting:
        x = jnp.swapaxes(x, -2, -3)
        mask = jnp.swapaxes(mask, -1, -2)

    x = layer_norm(x, params.layer_norm, eps=eps)

    # [..., I, 1, 1, J]
    mask_bias = (inf * (mask - 1.0))[..., :, None, None, :]

    # [..., H, I, J] -> [..., 1, H, I, J]
    triangle_bias = _project_triangle_bias(x, params, transpose_bias)
    triangle_bias = jnp.expand_dims(triangle_bias, -4)

    if backend == "cueq":
        out = _cueq_attention(
            x, mask_bias, triangle_bias, params.mha, no_heads=no_heads
        )
    elif chunk_size is None:
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


def _cueq_attention(
    x: jnp.ndarray,
    mask_bias: jnp.ndarray,
    triangle_bias: jnp.ndarray,
    params: AttentionParams,
    *,
    no_heads: int,
) -> jnp.ndarray:
    """Attend with the fused kernel, which never materialises the scores.

    The two biases are already in the layouts the kernel wants -- ``mask_bias`` is
    ``[..., I, 1, 1, J]`` and ``triangle_bias`` is ``[..., 1, H, I, J]`` -- because
    the XLA path needs the same broadcast shapes. The row axis of the triangle bias
    is 1, so the kernel broadcasts one copy across all rows instead of holding one
    per row; that is the whole saving, and passing a per-row bias here would undo it.

    Scaling moves into the kernel: :func:`attention` divides the queries by
    ``sqrt(D)`` itself, and doing both would scale twice.
    """
    from foldjax.models._cueq import cueq_attention_core

    # [..., I, H, J, D] -- head axis before the attended axis, as the kernel wants.
    query = jnp.swapaxes(split_heads(linear(x, params.linear_q), no_heads), -2, -3)
    key = jnp.swapaxes(split_heads(linear(x, params.linear_k), no_heads), -2, -3)
    value = jnp.swapaxes(split_heads(linear(x, params.linear_v), no_heads), -2, -3)

    out = cueq_attention_core(
        query,
        key,
        value,
        triangle_bias,
        mask_bias,
        scale=float(query.shape[-1]) ** -0.5,
    )
    # [..., I, J, H, D], which is what the gate and the output projection expect.
    out = jnp.swapaxes(out, -2, -3)
    if params.linear_g is not None:
        out = out * split_heads(jax_sigmoid(linear(x, params.linear_g)), no_heads)
    return linear(flatten_heads(out), params.linear_o)


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
        flat_mask = jnp.pad(flat_mask, ((0, 0), (0, padding), (0, 0), (0, 0), (0, 0)))

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


def _ring_attention(
    x: jnp.ndarray,
    mask_bias: jnp.ndarray,
    triangle_bias: jnp.ndarray,
    params: AttentionParams,
    *,
    no_heads: int,
) -> jnp.ndarray:
    """Exact two-dimensional ring counterpart of dense AF3 attention."""

    query = jnp.swapaxes(split_heads(linear(x, params.linear_q), no_heads), -2, -3)
    key = jnp.swapaxes(split_heads(linear(x, params.linear_k), no_heads), -2, -3)
    value = jnp.swapaxes(split_heads(linear(x, params.linear_v), no_heads), -2, -3)
    query = query / jnp.sqrt(jnp.asarray(query.shape[-1], dtype=query.dtype))
    out = ring_triangle_attention_2d(
        query,
        key,
        value,
        triangle_bias,
        mask_bias,
    )
    out = jnp.swapaxes(out, -2, -3)
    if params.linear_g is not None:
        out = out * split_heads(jax_sigmoid(linear(x, params.linear_g)), no_heads)
    return linear(flatten_heads(out), params.linear_o)


def _triangle_attention_cp(
    x: jnp.ndarray,
    params: TriangleAttentionParams,
    *,
    no_heads: int,
    mask: jnp.ndarray | None,
    starting: bool,
    transpose_bias: bool,
    inf: float,
    eps: float,
    chunk_size: int | None,
    local_backend: str = "xla",
) -> jnp.ndarray:
    """Triangle attention with the row axis sharded across the ``cp`` mesh.

    Each row of the pair representation is an independent attention, so a row
    shard needs no other shard's queries, keys or values -- the one cross-row
    input is the triangle bias, whose ``[i, j]`` indices read every row. The
    bias is therefore projected from the sharded tensor in its cheapest form
    (``heads`` channels, not ``C``) and handed to every shard whole, and the
    existing blocked row loop runs unchanged on each device's local rows. This
    is the one routing the SPMD partitioner cannot derive on its own: the
    chunked path slices the sharded axis at traced offsets, which a
    constraint-only program could satisfy only by gathering the whole pair
    tensor back onto every device.

    Leading batch axes (the template stack's ``[N_templ, N, N, C_t]``, the
    confidence re-embedding's sample axis) stay unsharded; only the row axis
    at ``-3`` is split.
    """
    mesh = cp_mesh()
    if mask is None:
        mask = jnp.ones(x.shape[:-1], dtype=x.dtype)
    if not starting:
        x = jnp.swapaxes(x, -2, -3)
        mask = jnp.swapaxes(mask, -1, -2)
    # The reshard after a transpose is the Fold-CP "transpose exchange":
    # ending-node attention runs on columns, so the columns become the
    # sharded rows.
    x = shard_pair_rows(x, row_axis=-3)

    x = layer_norm(x, params.layer_norm, eps=eps)
    mask_bias = (inf * (mask - 1.0))[..., :, None, None, :]
    triangle_bias = _project_triangle_bias(x, params, transpose_bias)
    triangle_bias = jnp.expand_dims(triangle_bias, -4)

    if cp_layout() == "2d":
        out = _ring_attention(
            x,
            mask_bias,
            triangle_bias,
            params.mha,
            no_heads=no_heads,
        )
        if not starting:
            out = jnp.swapaxes(out, -2, -3)
        return shard_pair_rows(out, row_axis=-3)

    # `shard_map` requires the sharded axis to divide evenly; pad the rows and
    # slice them back. The mask bias is padded with zeros, not -inf: a fully
    # masked row makes softmax divide by zero, and the padded rows are
    # discarded either way.
    n_rows = x.shape[-3]
    pad = (-n_rows) % cp_row_shards()
    if pad:
        width = [(0, 0)] * x.ndim
        width[-3] = (0, pad)
        x = jnp.pad(x, width)
        bias_width = [(0, 0)] * mask_bias.ndim
        bias_width[-4] = (0, pad)
        mask_bias = jnp.pad(mask_bias, bias_width)

    def local_rows(x_l, mask_bias_l, bias_l, params_l):
        if local_backend == "cueq":
            # Each device holds whole rows here, which is exactly the layout
            # the fused kernel takes; it never sees the sharded axis.
            return _cueq_attention(
                x_l, mask_bias_l, bias_l, params_l.mha, no_heads=no_heads
            )
        if chunk_size is None:
            return attention(
                x_l,
                x_l,
                params_l.mha,
                no_heads=no_heads,
                biases=(mask_bias_l, bias_l),
            )
        return _chunked_attention(
            x_l,
            mask_bias_l,
            bias_l,
            params_l.mha,
            no_heads=no_heads,
            chunk_size=chunk_size,
        )

    out = jax.shard_map(
        local_rows,
        mesh=mesh,
        in_specs=(
            pair_row_spec(x.ndim, row_axis=-3),
            pair_row_spec(mask_bias.ndim, row_axis=-4),
            PartitionSpec(),
            PartitionSpec(),
        ),
        out_specs=pair_row_spec(x.ndim, row_axis=-3),
    )(x, mask_bias, triangle_bias, params)
    if pad:
        # Slicing the padded rows away would otherwise let the partitioner
        # fall back to a replicated result.
        out = shard_pair_rows(
            jax.lax.slice_in_dim(out, 0, n_rows, axis=-3), row_axis=-3
        )
    if not starting:
        out = jnp.swapaxes(out, -2, -3)
    return out
