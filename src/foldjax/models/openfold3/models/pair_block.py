"""The pair stack block shared by the Pairformer and Evoformer.

Mirrors ``openfold3.core.model.latent.base_blocks.PairBlock``. Dropout is
identity at inference, so the residual structure reduces to five sequential
updates.

Two details are easy to get wrong and are pinned by the parity gate:

* ``tri_att_end`` is built with ``starting=True`` upstream, not
  ``starting=False``. The caller transposes ``z`` around it instead.
* The block returns the *updated* ``z``, not an update to be added. Each
  sub-layer's output is added internally.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec

from foldjax._openfold3_compile import resolve_triangle_kernel
from foldjax.models._cp import cp_grid, cp_mesh, pair_spec, shard_pair_rows
from foldjax.models.openfold3.models.primitives import (
    SwiGLUTransitionParams,
    swiglu_transition,
)
from foldjax.models.openfold3.models.row_chunking import map_row_chunks
from foldjax.models.openfold3.models.triangle import (
    TriangleMultiplicationParams,
    triangle_multiplication,
)
from foldjax.models.openfold3.models.triangle_attention import TriangleAttentionParams
from foldjax.models.openfold3.models.triangle_attention_cp import triangle_attention


class PairBlockParams(NamedTuple):
    """Parameters for ``PairBlock`` with a SwiGLU transition.

    The ReLU transition variant (AF2) and the fused triangular multiplication
    variant are not mapped; only the AF3 SwiGLU / non-fused layout is mapped.
    """

    tri_mul_out: TriangleMultiplicationParams
    tri_mul_in: TriangleMultiplicationParams
    tri_att_start: TriangleAttentionParams
    tri_att_end: TriangleAttentionParams
    pair_transition: SwiGLUTransitionParams


def _native_pair_backend() -> bool:
    # compile_predict includes this resolved choice in its executable identity.
    return resolve_triangle_kernel(None, cp_shards=1) == "native-private"


def _pair_attention(z, params, *, no_heads, mask, chunk_size=None, **kwargs):
    if not _native_pair_backend():
        return triangle_attention(
            z, params, no_heads=no_heads, mask=mask, chunk_size=chunk_size, **kwargs
        )
    from foldjax.models.openfold3.models import native_triangle_ops as ops

    if no_heads != 4:
        raise ValueError("native-private triangle attention requires four heads")
    return ops.map_native_samples(
        lambda one, one_mask: ops.native_triangle_attention_update(
            one,
            params,
            mask=one_mask,
            chunk_size=1024 if chunk_size is None else chunk_size,
            **kwargs,
        ),
        z,
        mask,
    )


def tri_mul_out_in(
    z: jnp.ndarray,
    params: PairBlockParams,
    *,
    pair_mask: jnp.ndarray,
    eps: float = 1e-5,
) -> jnp.ndarray:
    """Apply the outgoing then incoming triangular multiplicative updates."""
    if _native_pair_backend():
        from foldjax.models.openfold3.models import native_triangle_ops as ops

        def one(pair, mask):
            pair = ops.native_triangle_multiplication_residual(
                pair, params.tri_mul_out, outgoing=True, mask=mask, eps=eps
            )
            return ops.native_triangle_multiplication_residual(
                pair, params.tri_mul_in, outgoing=False, mask=mask, eps=eps
            )

        return ops.map_native_samples(one, z, pair_mask)
    z = z + triangle_multiplication(
        z, params.tri_mul_out, outgoing=True, mask=pair_mask, eps=eps
    )
    return z + triangle_multiplication(
        z, params.tri_mul_in, outgoing=False, mask=pair_mask, eps=eps
    )


def tri_att_start_end(
    z: jnp.ndarray,
    params: PairBlockParams,
    *,
    pair_mask: jnp.ndarray,
    no_heads_pair: int,
    inf: float = 1e9,
    eps: float = 1e-5,
    chunk_size: int | None = None,
) -> jnp.ndarray:
    """Apply the starting then ending triangle attention layers.

    Both layers are starting-node modules upstream; the transposes around the
    second one are what make it an ending-node update.
    """
    z = z + _pair_attention(
        z,
        params.tri_att_start,
        no_heads=no_heads_pair,
        mask=pair_mask,
        inf=inf,
        eps=eps,
        chunk_size=chunk_size,
    )
    z = jnp.swapaxes(z, -2, -3)
    # The Fold-CP transpose exchange: the ending-node update treats columns as
    # rows, so under context parallelism the sharded axis moves with the
    # transpose (an all-to-all under the partitioner, an identity otherwise).
    z = shard_pair_rows(z)
    z = z + _pair_attention(
        z,
        params.tri_att_end,
        no_heads=no_heads_pair,
        mask=jnp.swapaxes(pair_mask, -1, -2),
        transpose_bias=True,
        inf=inf,
        eps=eps,
        chunk_size=chunk_size,
    )
    return shard_pair_rows(jnp.swapaxes(z, -2, -3))


def _swiglu_row_blocks(
    params: SwiGLUTransitionParams,
    z: jnp.ndarray,
    pair_mask: jnp.ndarray | None,
    *,
    eps: float,
    glu_backend: str,
    chunk_size: int | None,
) -> jnp.ndarray:
    """The SwiGLU transition over whatever rows it is handed.

    Nothing here reads the sharding: the caller decides whether the row block
    is taken on the global row axis or inside a shard.
    """

    def one(rows: jnp.ndarray, mask: jnp.ndarray | None = None) -> jnp.ndarray:
        return swiglu_transition(
            rows, params, mask=mask, eps=eps, glu_backend=glu_backend
        )

    if pair_mask is None:
        return map_row_chunks(one, z, chunk_size=chunk_size)
    return map_row_chunks(one, z, pair_mask, chunk_size=chunk_size, row_axes=(-3, -2))


def _pair_transition(
    z: jnp.ndarray,
    params: SwiGLUTransitionParams,
    *,
    pair_mask: jnp.ndarray | None,
    eps: float,
    glu_backend: str,
    chunk_size: int | None,
) -> jnp.ndarray:
    """Row-block the pair transition, on local rows when a mesh is active.

    SwiGLU widens the pair representation to ``4 * C_z`` twice before the
    output projection reads the product, and leaving that widened form
    unblocked made it the largest buffer in the context-parallel program: at
    2,112 tokens on four devices it was ``f32[1115136, 512]``, 2,178 MiB, at 18
    allocated sites -- *identically* under the 1-D layout (528 x 2112) and the
    2-D one (1056 x 1056), because ``N/4 x N`` and ``N/2 x N/2`` are the same
    area, so the square grid does not touch it.

    Restoring the block on the *global* row axis does not recover that: a
    128-row slice of a 4-way-sharded axis is not shard-aligned, so the
    partitioner has to gather. The block therefore has to be taken inside the
    shard, the way ``triangle_attention`` already runs its chunked row loop and
    the way Boltz-2's ``_cp_pair_transition`` does it. Every op in the
    transition is elementwise in the two token axes and contracts only over
    channels, so a local tile is the whole computation for its own rows and
    columns: no collective, and the same arithmetic per element that the
    unblocked sharded program did -- the sharded blocked output is bitwise
    equal to the *serial* blocked output in both layouts and in both dtypes,
    where the unblocked sharded one was 3.8e-6 away from it at float32.

    Measured by compiling the shipped program on four fake CPU devices at
    2,112 tokens, all 18 full-tile widened buffers go to zero and are replaced
    by the serial block width -- ``f32[270336, 512]``, 528 MiB, under 1-D and
    ``f32[135168, 512]`` (128 x 1056), 264 MiB, under 2-D, with the bfloat16
    twins halving alongside. The collective census is unchanged at both sizes
    (1-D 175 -> 175, 2-D 640 -> 640): the block adds no communication.

    Two things the block does *not* buy. It makes ``map_row_chunks``
    materialise the local tile restacked into blocks, a new per-device tenant
    of ``f32[5, 1, 128, 2112, 128]`` (660 MiB) under 1-D and
    ``f32[9, 1, 128, 1056, 128]`` (594 MiB) under 2-D -- still a third of the
    buffer it removes, and the serial program has always paid the same thing at
    full width (``f32[17, 1, 128, 2112, 128]``, 2,240 MiB). And the CPU temp
    arena *rises* 8-9%, which is the XLA-CPU packing artefact Boltz-2 recorded
    from the same experiment: that backend merges blocks where it likes and
    materialises the ones it keeps rather than fusing them. The per-device
    shape is the part of a CPU partitioning probe that transfers; the arena is
    not.
    """

    mesh = cp_mesh()
    settings = {"eps": eps, "glu_backend": glu_backend}
    if mesh is None:
        return _swiglu_row_blocks(
            params, z, pair_mask, chunk_size=chunk_size, **settings
        )

    rows, columns = cp_grid()
    n_rows, n_columns = z.shape[-3], z.shape[-2]
    # `shard_map` needs both sharded axes to divide the grid. The padded region
    # is its own set of rows and columns, and the transition never mixes them
    # with a kept one, so the padded output is sliced away unread.
    row_pad, column_pad = (-n_rows) % rows, (-n_columns) % columns
    local_rows = (n_rows + row_pad) // rows
    if chunk_size is None or chunk_size <= 0 or chunk_size >= local_rows:
        # A block at least as wide as the local tile would not block anything;
        # asking for it on the global axis is what forces the gather, so the
        # request is dropped rather than moved, leaving the sharded program
        # exactly as it was.
        return _swiglu_row_blocks(params, z, pair_mask, chunk_size=None, **settings)

    operands = [z] if pair_mask is None else [z, pair_mask]
    # The first token axis of each operand: `[..., N, N, C]` is at -3 and its
    # mask `[..., N, N]` at -2, with the second token axis the next one along.
    first_axes = [-3] if pair_mask is None else [-3, -2]
    padded, specs = [], []
    for array, first_axis in zip(operands, first_axes, strict=True):
        row = first_axis % array.ndim
        if row_pad or column_pad:
            width = [(0, 0)] * array.ndim
            width[row] = (0, row_pad)
            width[row + 1] = (0, column_pad)
            array = jnp.pad(array, width)
        padded.append(array)
        specs.append(pair_spec(array.ndim, row_axis=row, col_axis=row + 1))

    def local(*sharded: jnp.ndarray) -> jnp.ndarray:
        *arrays, params_local = sharded
        return _swiglu_row_blocks(
            params_local,
            arrays[0],
            arrays[1] if len(arrays) > 1 else None,
            chunk_size=chunk_size,
            **settings,
        )

    # Parameters go in as a replicated operand rather than a closure, as in
    # `triangle_attention._triangle_attention_cp`: under a layer scan they are
    # traced values, and an operand keeps the whole tree on the mesh.
    out = jax.shard_map(
        local,
        mesh=mesh,
        in_specs=(*specs, PartitionSpec()),
        out_specs=specs[0],
    )(*padded, params)
    if row_pad:
        out = jax.lax.slice_in_dim(out, 0, n_rows, axis=-3)
    if column_pad:
        out = jax.lax.slice_in_dim(out, 0, n_columns, axis=-2)
    if row_pad or column_pad:
        # Re-pinning the slice keeps the partitioner from answering the
        # narrower shape with a replicated result.
        out = shard_pair_rows(out)
    return out


def pair_block(
    z: jnp.ndarray,
    params: PairBlockParams,
    *,
    pair_mask: jnp.ndarray,
    no_heads_pair: int,
    inf: float = 1e9,
    mask_transition: bool = True,
    tri_mul_first: bool = True,
    eps: float = 1e-5,
    chunk_size: int | None = None,
    glu_backend: str = "xla",
) -> jnp.ndarray:
    """Apply one pair block.

    Args:
        z: ``[..., N, N, C_z]`` pair representation.
        params: mapped block parameters.
        pair_mask: ``[..., N, N]`` pair mask.
        no_heads_pair: head count for both triangle attention layers.
        inf: triangle attention masking constant.
        mask_transition: upstream's ``_mask_trans``; ``True`` masks the
            transition output, which is the default.
        tri_mul_first: run the multiplicative updates before triangle attention.
            ``PairBlock`` is always ``True``; ``TemplatePairBlock`` exposes it as
            a configuration option.
        eps: layer norm epsilon.

    Returns:
        ``[..., N, N, C_z]`` updated pair representation.
    """
    # Under context parallelism the pair representation is sharded along its
    # rows; pinning it here keeps every block of every stack -- trunk, MSA,
    # template, confidence re-embedding -- on the same layout. Both chunked
    # stages keep their block in that mode, and both take it *inside* a
    # `shard_map` on local rows: `map_row_chunks` on the global row axis is a
    # `lax.map` over slices of the sharded axis, which the partitioner could
    # only satisfy by gathering the whole tensor. Triangle attention's blocked
    # loop already ran there; see `_pair_transition` for the transition's.
    z = shard_pair_rows(z)
    if tri_mul_first:
        z = tri_mul_out_in(z, params, pair_mask=pair_mask, eps=eps)
        z = tri_att_start_end(
            z,
            params,
            pair_mask=pair_mask,
            no_heads_pair=no_heads_pair,
            inf=inf,
            eps=eps,
            chunk_size=chunk_size,
        )
    else:
        z = tri_att_start_end(
            z,
            params,
            pair_mask=pair_mask,
            no_heads_pair=no_heads_pair,
            inf=inf,
            eps=eps,
            chunk_size=chunk_size,
        )
        z = tri_mul_out_in(z, params, pair_mask=pair_mask, eps=eps)

    # SwiGLU widens the pair representation to ``4 * C_z`` twice before the output
    # projection reads the product. Rows are independent, so ``chunk_size`` caps that
    # widened tensor without changing a value -- the same knob the triangle attention
    # above uses, and for the same reason.
    return z + _pair_transition(
        z,
        params.pair_transition,
        pair_mask=pair_mask if mask_transition else None,
        eps=eps,
        glu_backend=glu_backend,
        chunk_size=chunk_size,
    )
