"""Pure JAX MSA module for the Boltz-2 port."""

from __future__ import annotations

from collections.abc import Mapping

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec

from foldjax.models._cp import (
    CP_COL_AXIS,
    CP_ROW_AXIS,
    cp_grid,
    cp_layout,
    cp_mesh,
    msa_spec,
    pair_spec,
    shard_msa,
    shard_pair_rows,
)
from foldjax.models.boltz2.models.primitives._common import layer_norm as _layer_norm
from foldjax.models.boltz2.models.primitives._common import linear as _linear
from foldjax.models.boltz2.models.primitives._common import (
    residual_cast as _residual_cast,
)
from foldjax.models.boltz2.models.primitives._scan_utils import stack_layer_params
from foldjax.models.boltz2.models.primitives.native_amp_norm import amp_layer_norm
from foldjax.models.boltz2.models.primitives.native_pwa_mma import (
    pair_weighted_contraction,
)
from foldjax.models.boltz2.models.primitives.native_pwa_weights import (
    observed_profile,
    pwa_logits,
    pwa_softmax,
)
from foldjax.models.boltz2.models.primitives.transition import transition_forward
from foldjax.models.boltz2.models.triangle.triangle import (
    triangle_multiplication_forward,
)
from foldjax.models.boltz2.models.triangle.triangle_attention import (
    resolve_triangle_attention_chunk,
    resolve_triangle_attention_q_chunk,
    triangle_attention_forward,
)

Params = Mapping[str, object]


def _amp_dtype(kernel: jnp.ndarray) -> jnp.dtype | None:
    return kernel.dtype if kernel.dtype in (jnp.bfloat16, jnp.float16) else None


# Publisher trunkv2.py uses physical token count. Padding across this boundary
# changes rounding and needs separate parity admission, not mask-based inference.
_NATIVE_CHUNK_THRESHOLD = 384


# --- the MSA stack on the two-dimensional grid ------------------------------
#
# Under the 2-D layout the MSA tensor ``m`` [B, M, N, C] is split on BOTH grid
# axes -- alignment depth ``M`` on the column axis, tokens ``N`` on the row
# axis -- so rank ``(r, s)`` owns alignment rows ``M_s`` against token
# positions ``N_r``.  The pair tensor at that same rank owns token rows ``N_r``
# against token columns ``N_s``.  Without the depth split a square grid only
# halves this stack while it quarters the pair stack, and the stack is what
# owns the peak: of the 16,956 MiB per-device peak-live set measured at 2,096
# tokens over an 8,192-row alignment on a 2x2 grid, the residual stream ``m``
# is two f32[1,8192,1048,64] (2,096 MiB each), the MSA transition's
# accumulators eight bf16 tiles of 1,048 MiB, and PairWeightedAveraging's
# output one more bf16[1,8192,1048,64].
#
# The column axis therefore names alignment rows here and token columns next
# door, and the two operators that read one layout and write the other bridge
# them with explicit collectives rather than a reshard:
#
# * ``PairWeightedAveraging`` takes its weights from the pair tensor per
#   ``(i, j)`` and applies them to every alignment row independently.  Each
#   rank projects its own pair tile to head logits, masks them there -- on the
#   tile that owns the key block -- and ``all-gather``s those small logits
#   along the column axis, so it holds the weights for its own token rows
#   against every key token and normalises over the complete key axis.  The
#   widened MSA values then ride a ``collective-permute`` ring along the row
#   axis, which keeps the alignment shard fixed, and each hop contributes the
#   key block it carries.  Nothing is summed along the column axis: those
#   ranks hold different alignment rows.
#
# * ``OuterProductMean`` needs, for output block ``(I, J)``, the ``a`` operand
#   of token block ``I`` and the ``b`` operand of token block ``J`` over ALL
#   alignment rows.  ``b`` and its mask ride a ring along the row axis, so at
#   step ``t`` every rank of grid row ``r`` holds key block ``(r - t) % side``
#   -- the same block for the whole column, which is what makes the column
#   reduction coherent.  Each rank contributes its own alignment shard's
#   numerator and its own mask count, both are ``psum``ed along the column
#   axis, the mean is taken after that reduction, and the output bias is added
#   once.  Exactly one rank of each reduction group owns the block it has just
#   computed, and that rank keeps it.
#
# Everything else in the stack -- the transition, the channel norms, the gates
# -- is elementwise in both split axes and contracts only over channels, so it
# runs on the local tile with no collective at all.
#
# The result is not bitwise equal to the serial program and cannot be: summing
# a shard at a time reassociates both reductions.  It is equal to it within the
# port's context-parallel tolerance, and the three things reassociation is not
# allowed to become -- averaging locally normalised means, adding the output
# bias once per shard, or a mask that counts different rows -- are each pinned
# by a test rather than by inspection.


def _on_msa_grid() -> bool:
    """Whether the MSA stack is running on the two-dimensional grid."""

    return cp_mesh() is not None and cp_layout() == "2d"


def _grid_side() -> int:
    """Devices along one side of the square grid."""

    return cp_grid()[0]


def _grid_pads(depth: int, tokens: int) -> tuple[int, int]:
    """Alignment rows and token positions ``shard_map`` needs added.

    Both split axes have to divide the grid.  The token axis is already
    aligned by the port's context-parallel padding plan, so this is a no-op
    there; the alignment depth is not -- it is whatever the alignment holds --
    and :func:`msa_module_forward` pads it once for the whole stack rather
    than letting every operator pad the residual stream again.
    """

    side = _grid_side()
    return (-depth) % side, (-tokens) % side


def _ring_perm(side: int) -> list[tuple[int, int]]:
    """One hop of the grid-row ring: rank ``r`` sends to rank ``r + 1``.

    After ``t`` hops a rank therefore holds what rank ``r - t`` started with,
    which is the token block ``(r - t) % side``.  Both bridges below index
    their other operand by exactly that.
    """

    return [(source, (source + 1) % side) for source in range(side)]


def msa_module_forward(
    params: Params,
    z: jnp.ndarray,
    emb: jnp.ndarray,
    feats: Mapping[str, jnp.ndarray],
    num_tokens: int = 33,
    eps: float = 1e-5,
    use_scan: bool = True,
    chunk_size: int = 128,
    triangle_attention_chunk: int | None = None,
    triangle_attention_q_chunk: int | None = None,
    transition_hidden_chunk: int | None = None,
    pair_averaging_chunk: int | None = None,
    matmul_precision: str = "highest",
    triangle_backend: str = "xla",
    glu_backend: str = "xla",
    subsample_msa: bool = False,
    num_subsampled_msa: int = 1024,
    msa_key: jnp.ndarray | None = None,
    msa_rows: jnp.ndarray | None = None,
    pair_residual_dtype: jnp.dtype | None = None,
) -> jnp.ndarray:
    """Run Boltz MSAModule.

    ``use_scan=False`` (default) unrolls the layer stack in Python (lower steady
    latency). ``use_scan=True`` runs the stack via ``lax.scan`` (faster compile).

    ``subsample_msa`` caps the MSA depth at ``num_subsampled_msa`` (default
    1024). ``MSAModuleArgs`` defaults it to False and `boltz predict` declares
    ``--subsample_msa`` as a click flag, which defaults to False whatever the
    Python signature says -- so upstream subsamples only when that flag is
    given.

    ``msa_key`` selects *which* rows. Upstream draws a fresh
    ``torch.randperm(n_msa)[:1024]`` inside every forward pass
    (``trunkv2.py``), so across a recycled trunk it sees a different slice of
    the alignment each pass and effectively ensembles over the whole thing.
    Passing a key reproduces that; passing ``None`` keeps the first
    ``num_subsampled_msa`` rows, which is deterministic and reproducible but is
    *not* what upstream computes -- with 2,372 rows and eleven passes it shows
    the model the same top 1,024 eleven times instead of most of the alignment.

    ``pair_residual_dtype`` pins the pair carry ``z``. ``None`` (default)
    leaves it at whatever width the caller handed in -- float32 in the
    released trunk, and float32 again from the first OuterProductMean above
    384 tokens, which adds the original FP32 bias outside its AMP matmuls.
    The MSA carry ``m`` is not pinned; see ``msa_layer_forward``.

    ``msa_rows`` overrides the draw with explicit row indices and takes
    precedence over ``msa_key``. It exists so a matched-tape parity harness can
    hand this module the very permutation upstream drew: without it the two
    trunks read different alignment rows and their coordinates cannot agree no
    matter how well the sampler is matched. ``None`` (default) is inert.
    """

    # Subsample MSA depth (axis 1) to num_subsampled_msa, matching Boltz CLI's
    # inference-time cap; take the first rows (deterministic, query-first order).
    msa_d = feats["msa"]
    has_del = feats["has_deletion"]
    del_val = feats["deletion_value"]
    msa_paired = feats["msa_paired"]
    msa_mask = feats["msa_mask"]
    if subsample_msa and msa_d.shape[1] > num_subsampled_msa:
        if msa_rows is not None:
            rows = jnp.asarray(msa_rows, dtype=jnp.int32)
            if rows.ndim != 1 or rows.shape[0] != num_subsampled_msa:
                raise ValueError(
                    f"msa_rows must be {num_subsampled_msa} row indices, got "
                    f"shape {rows.shape}"
                )
        elif msa_key is None:
            rows = jnp.arange(num_subsampled_msa)
        else:
            rows = jax.random.permutation(msa_key, msa_d.shape[1])[:num_subsampled_msa]
        msa_d = jnp.take(msa_d, rows, axis=1)
        has_del = jnp.take(has_del, rows, axis=1)
        del_val = jnp.take(del_val, rows, axis=1)
        msa_paired = jnp.take(msa_paired, rows, axis=1)
        msa_mask = jnp.take(msa_mask, rows, axis=1)
    if _on_msa_grid():
        # One pad of the entry features, for the whole stack. The alignment
        # depth is split on the grid's column axis and no padding plan aligns
        # it, so `shard_map` would otherwise refuse -- and `m` is never
        # returned, so these rows are never removed again. Padding here costs
        # one copy of the integer MSA per pass instead of three copies of the
        # widened residual stream per layer.
        #
        # The padded rows are masked, which is what keeps them out of
        # OuterProductMean's mean and so out of every pair value this module
        # returns. They still run through the gates and the transition, where
        # they are finite and unread.
        depth_pad, _ = _grid_pads(msa_d.shape[1], msa_d.shape[2])
        if depth_pad:
            widths = ((0, 0), (0, depth_pad), (0, 0))
            msa_d = jnp.pad(msa_d, widths)
            has_del = jnp.pad(has_del, widths)
            del_val = jnp.pad(del_val, widths)
            msa_paired = jnp.pad(msa_paired, widths)
            msa_mask = jnp.pad(msa_mask, widths)
    m = _msa_input_embedding(
        params,
        emb,
        msa_d,
        has_del,
        del_val,
        msa_paired,
        num_tokens=num_tokens,
    )

    mask_dtype = (
        jnp.float32 if _amp_dtype(params["msa_proj"]["kernel"]) is not None else m.dtype
    )
    token_mask = feats["token_pad_mask"].astype(mask_dtype)
    token_mask = token_mask[:, :, None] * token_mask[:, None, :]
    msa_mask = msa_mask.astype(m.dtype)

    z = _residual_cast(z, pair_residual_dtype)
    layers = params["layers"]
    if not use_scan:
        for layer in layers:
            z, m = msa_layer_forward(
                layer,
                z,
                m,
                token_mask,
                msa_mask,
                eps=eps,
                chunk_size=chunk_size,
                triangle_attention_chunk=triangle_attention_chunk,
                triangle_attention_q_chunk=triangle_attention_q_chunk,
                transition_hidden_chunk=transition_hidden_chunk,
                pair_averaging_chunk=pair_averaging_chunk,
                matmul_precision=matmul_precision,
                triangle_backend=triangle_backend,
                glu_backend=glu_backend,
                pair_residual_dtype=pair_residual_dtype,
            )
        return z

    # Native eval dropout multiplies the first PWA update by FP32 ones, so
    # every residual after that addition is FP32. Promote the scan carry early
    # without changing its BF16-representable values; PWA already normalizes
    # in FP32, and scan requires a stable carry dtype across all layers.
    if m.dtype in (jnp.bfloat16, jnp.float16):
        m = m.astype(jnp.float32)
    m = shard_msa(m)
    stacked = stack_layer_params(layers)

    def body(carry, layer):
        z_c, m_c = carry
        z_c, m_c = msa_layer_forward(
            layer,
            z_c,
            m_c,
            token_mask,
            msa_mask,
            eps=eps,
            chunk_size=chunk_size,
            triangle_attention_chunk=triangle_attention_chunk,
            triangle_attention_q_chunk=triangle_attention_q_chunk,
            transition_hidden_chunk=transition_hidden_chunk,
            pair_averaging_chunk=pair_averaging_chunk,
            matmul_precision=matmul_precision,
            triangle_backend=triangle_backend,
            glu_backend=glu_backend,
            pair_residual_dtype=pair_residual_dtype,
        )
        return (z_c, m_c), None

    (z, m), _ = jax.lax.scan(body, (z, m), stacked)
    return z


def _msa_input_embedding(
    params: Params,
    emb: jnp.ndarray,
    msa: jnp.ndarray,
    has_deletion: jnp.ndarray,
    deletion_value: jnp.ndarray,
    msa_paired: jnp.ndarray,
    *,
    num_tokens: int,
) -> jnp.ndarray:
    """Project one compact or publisher-native Boltz MSA feature block."""

    # Equivalent to one_hot(msa, num_tokens) @ kernel[:num_tokens] but without
    # materializing the [batch, num_seq, N, num_tokens] one-hot tensor.
    # Concatenation order in the original code is: one-hot block (rows
    # [0:num_tokens]), then has_deletion, deletion_value, msa_paired
    # (rows [num_tokens:num_tokens+3]). Split the kernel at call time; the
    # stored param pytree is left unchanged.
    kernel = params["msa_proj"]["kernel"]
    kernel_onehot = kernel[:num_tokens]  # [num_tokens, d]
    kernel_extra = kernel[num_tokens:]  # [3, d]
    # The MSA features arrive replicated: nothing in the input registry places
    # an alignment axis. Constraining the two operands of the projection --
    # rather than only its result -- is what keeps the partitioner from
    # materialising a replicated ``[B, M, N, d]`` gather and slicing it, which
    # is a full-width value in a program whose whole point is that none exists.
    msa_idx = shard_msa(msa.astype(jnp.int32))
    # The high-level path stores exact binary MSA flags as bool.  Cast every
    # lane to the historical projection dtype before stacking; this is also
    # robust under JAX's strict dtype-promotion mode and leaves custom callers'
    # numerical values unchanged.
    extra = shard_msa(
        jnp.stack(
            tuple(
                jnp.asarray(value, dtype=kernel_extra.dtype)
                for value in (has_deletion, deletion_value, msa_paired)
            ),
            axis=-1,
        )
    )
    if _amp_dtype(kernel) is not None:
        # Native projects the concatenated one-hot and scalar features in one
        # Linear. Keep the sparse form, but round only the completed projection.
        scalar_projection = jnp.matmul(
            extra, kernel_extra, preferred_element_type=jnp.float32
        )
        m = (kernel_onehot[msa_idx].astype(jnp.float32) + scalar_projection).astype(
            kernel.dtype
        )
    else:
        m = kernel_onehot[msa_idx] + _linear(extra, kernel_extra)
    return shard_msa(m + _linear(emb, params["s_proj"]["kernel"])[:, None])


def msa_layer_forward(
    params: Params,
    z: jnp.ndarray,
    m: jnp.ndarray,
    token_mask: jnp.ndarray,
    msa_mask: jnp.ndarray,
    eps: float = 1e-5,
    chunk_size: int = 128,
    triangle_attention_chunk: int | None = None,
    triangle_attention_q_chunk: int | None = None,
    transition_hidden_chunk: int | None = None,
    pair_averaging_chunk: int | None = None,
    matmul_precision: str = "highest",
    triangle_backend: str = "xla",
    glu_backend: str = "xla",
    pair_residual_dtype: jnp.dtype | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Run one Boltz MSALayer in eval mode.

    ``pair_residual_dtype`` pins the pair carry only. ``m`` stays FP32: its
    promotion is not a storage choice but upstream arithmetic -- eval-mode
    ``get_dropout_mask`` returns an FP32 tensor
    (``boltz/model/layers/dropout.py:42-43``) that multiplies every MSA
    update, so narrowing ``m`` would compute different numbers rather than
    store the same ones more tightly.
    """

    msa_update = pair_weighted_averaging_forward(
        params["pair_weighted_averaging"],
        m,
        z,
        token_mask,
        eps=eps,
        row_chunk_size=pair_averaging_chunk,
    )
    if msa_update.dtype in (jnp.bfloat16, jnp.float16):
        # get_dropout_mask(..., training=False) is still FP32 upstream.
        msa_update = msa_update.astype(jnp.float32)
    m = m + msa_update
    transition_dtype = _amp_dtype(params["msa_transition"]["fc1"]["kernel"])
    # The transition is elementwise in both of the MSA tensor's split axes and
    # contracts only over channels, so it needs no collective -- but on the
    # grid its row block has to be taken *inside* the shard, because axis 1 is
    # now the alignment depth the column axis splits. `_msa_transition_grid`
    # is that one `shard_map`; off the grid this is the same callable, and the
    # two call sites below are unchanged.
    transition = _msa_transition_grid if _on_msa_grid() else transition_forward
    if transition_dtype is not None:
        m = m + transition(
            params["msa_transition"],
            m,
            eps=eps,
            glu_backend=glu_backend,
            compute_dtype=transition_dtype,
            native_amp_norm=transition_dtype == jnp.bfloat16,
            chunk_size=32 if z.shape[1] > _NATIVE_CHUNK_THRESHOLD else None,
            # Axis 1 of `m` is alignment depth. The 1-D layout does not shard
            # it, so the row block the serial program takes is shard-aligned
            # as written; the 2-D grid does shard it, and there the same flag
            # marks the tile inside `_msa_transition_grid`'s `shard_map` as
            # the one to block. Without the block the eight unrolled
            # hidden-chunk accumulators stay at full local width:
            # `bf16[8585216, 64]` x8, 8.4 GiB, half the per-device peak of the
            # 2,096-token 2-D program before the depth axis was split.
            cp_msa=True,
        )
    else:
        m = m + transition(
            params["msa_transition"],
            m,
            eps=eps,
            glu_backend=glu_backend,
            cp_msa=True,
        )
    # Above 384 tokens OuterProductMean returns FP32 (it adds the original
    # FP32 bias outside its AMP matmuls), so this add is the one promoter on
    # the pair path and the pin has to sit on it or the scan carry changes
    # width between layers.
    z = _residual_cast(
        z
        + outer_product_mean_forward(
            params["outer_product_mean"],
            m,
            msa_mask,
            eps,
            chunk_size=chunk_size,
            preserve_native_amp_shape=True,
        ),
        pair_residual_dtype,
    )
    z = pairformer_no_seq_layer_forward(
        params["pairformer_layer"],
        z,
        token_mask,
        eps,
        chunk_size=chunk_size,
        triangle_attention_chunk=triangle_attention_chunk,
        triangle_attention_q_chunk=triangle_attention_q_chunk,
        transition_hidden_chunk=transition_hidden_chunk,
        matmul_precision=matmul_precision,
        triangle_backend=triangle_backend,
        glu_backend=glu_backend,
        pair_residual_dtype=pair_residual_dtype,
    )
    return _residual_cast(z, pair_residual_dtype), shard_msa(m)


def _msa_transition_grid(
    params: Params,
    x: jnp.ndarray,
    **kwargs: object,
) -> jnp.ndarray:
    """Run the MSA transition on each device's own tile of the grid.

    One ``shard_map`` and no collective. Inside it ``transition_forward`` sees
    the local tile, so ``cp_msa=True`` still means what it meant before the
    depth axis was split -- take the row block, it is shard-aligned -- only
    now the rows it blocks are the device's own share of the alignment.
    Blocking the *global* depth axis instead would fight the partitioner
    exactly as blocking the global pair rows does (``_cp_pair_transition``).
    """

    mesh = cp_mesh()
    depth, tokens = x.shape[1], x.shape[2]
    depth_pad, token_pad = _grid_pads(depth, tokens)
    if depth_pad or token_pad:
        x = jnp.pad(x, ((0, 0), (0, depth_pad), (0, token_pad), (0, 0)))
    spec = msa_spec(x.ndim)

    def local(x_local: jnp.ndarray, params_local: Params) -> jnp.ndarray:
        return transition_forward(params_local, x_local, **kwargs)  # type: ignore[arg-type]

    # Parameters as a replicated operand rather than a closure: under the
    # layer scan they are traced values (`_cp_pair_transition` says the same).
    out = jax.shard_map(
        local,
        mesh=mesh,
        in_specs=(spec, PartitionSpec()),
        out_specs=spec,
    )(x, params)
    if depth_pad or token_pad:
        # Re-pinning the slice keeps the partitioner from answering the
        # narrower shape with a replicated result.
        out = shard_msa(
            jax.lax.slice(
                out,
                (0, 0, 0, 0),
                (out.shape[0], depth, tokens, out.shape[3]),
            )
        )
    return out


#: Byte budget for one block of PairWeightedAveraging's widened MSA tensors.
#:
#: This block widens the MSA from ``c_m`` (64) to ``num_heads * c_h`` (256)
#: channels three times over -- the projected values ``v``, the gate ``g``, and
#: the averaged output ``o`` -- and ``v`` and ``o`` are live at the same
#: instant. Read off XLA's buffer assignment for the 490-token benchmark job
#: (7,917 alignment rows), those two were ``f32[1,7917,490,256]`` and
#: ``f32[8,1,7917,32,490]``: 3.70 GiB each, 7.4 GiB of a 9,963 MiB temp arena,
#: while every other block in the trunk was already chunked down to a few
#: hundred MiB. That is the whole of this port's memory gap against upstream at
#: full alignment depth, and it appears only above 384 tokens because that is
#: where upstream turns its own version of this on (``chunk_heads_pwa`` in
#: ``trunkv2.py``, gated on ``const.chunk_size_threshold``).
#:
#: Chunking over MSA rows rather than over heads is the safer of the two: rows
#: are a pure batch axis of the einsum and of all four projections, so every
#: reduction stays whole and the accumulation order does not change. Upstream's
#: head chunk instead splits ``proj_o``'s 256-long reduction into eight partial
#: sums.
#:
#: 512 MiB rather than the transition's 256 MiB, so that the 132-token
#: benchmark job -- widened form 306 MiB over its 2,371 rows -- stays on the
#: single-op path. That job is the one the matched-tape parity harness replays,
#: and leaving it unsplit is what makes this change provably inert there: the
#: module it lowers to is byte-identical to the one that shipped (sha256 of
#: ``jax.jit(...).lower(...).as_text()``, asserted in the MSA parity tests). It
#: is also what upstream runs at that size, below its own 384-token threshold.
#:
#: The identity matters because the harness cannot settle the question by
#: measurement: six runs of unmodified code spanned 0.14522 to 0.14768 A, and
#: two with XLA autotuning disabled still differed in the fourth decimal, so a
#: repeated RMSD would not have shown a shift this small either way. A chunk is
#: exact but not bit-identical -- XLA tiles the smaller GEMMs differently -- so
#: "unchanged" has to mean the same program, not a similar number.
#:
#: Above the budget it splits: the 490-token job into eight blocks, and its
#: trunk temp arena from 9,963 MiB to 3,649 MiB.
_PWA_BUDGET_BYTES = 512 * 1024**2


def _auto_pair_averaging_chunk(
    m: jnp.ndarray,
    params: Params,
    *,
    budget: int = _PWA_BUDGET_BYTES,
) -> int | None:
    """MSA rows whose widened form fits the budget, or None to do it whole.

    ``budget`` is only ever lowered: the grid bridge halves it, because there
    the ring holds the block it is contracting and the block it has just
    received at the same instant, and the pair of them is what has to stay
    inside the byte budget one block stayed inside before.
    """
    if m.ndim != 4 or m.shape[1] < 2:
        return None
    wide = params["proj_m"]["kernel"].shape[-1]
    per_row = m.shape[2] * wide * m.dtype.itemsize
    if per_row <= 0 or per_row * m.shape[1] <= budget:
        return None
    return max(1, budget // per_row)


def pair_weighted_averaging_forward(
    params: Params,
    m: jnp.ndarray,
    z: jnp.ndarray,
    mask: jnp.ndarray,
    eps: float = 1e-5,
    inf: float = 1e6,
    row_chunk_size: int | None = None,
) -> jnp.ndarray:
    """Run Boltz PairWeightedAveraging.

    ``row_chunk_size`` splits axis 1 (MSA depth) into independent blocks, so the
    widened ``[b, n_msa, n_token, num_heads * c_h]`` intermediates are never
    resident all at once. Left as ``None`` the block size is derived from
    ``_PWA_BUDGET_BYTES``; pass ``0`` to force the single-op path.

    Below the budget this runs the original single-op body, statement for
    statement. Sharing one body with the chunked path would mean hoisting the
    pair weights above the MSA layer norm -- algebraically the same program, but
    a different HLO, which is exactly what the budget is set to avoid. Two
    bodies is the price of leaving the jobs that never needed a chunk on the
    program they already ran.

    The AMP path additionally preserves native head-wise accumulation above
    384 tokens; MSA-row blocking never replaces that reduction order.
    """

    if _on_msa_grid():
        return _pair_weighted_averaging_grid(
            params, m, z, mask, eps, inf, row_chunk_size
        )

    if _amp_dtype(params["proj_m"]["kernel"]) is not None:
        return _pair_weighted_averaging_amp(
            params, m, z, mask, eps, inf, row_chunk_size
        )

    if row_chunk_size is None:
        row_chunk_size = _auto_pair_averaging_chunk(m, params)
    if row_chunk_size is not None and 0 < row_chunk_size < m.shape[1]:
        return _pair_weighted_averaging_chunked(
            params, m, z, mask, eps, inf, row_chunk_size
        )

    m = _layer_norm(m, params["norm_m"]["scale"], params["norm_m"]["bias"], eps)
    z = _layer_norm(z, params["norm_z"]["scale"], params["norm_z"]["bias"], eps)
    num_heads = params["proj_z"]["kernel"].shape[-1]
    c_h = params["proj_m"]["kernel"].shape[-1] // num_heads

    v = _linear(m, params["proj_m"]["kernel"])
    v = jnp.reshape(v, (*v.shape[:3], num_heads, c_h))
    v = jnp.transpose(v, (0, 3, 1, 2, 4))
    b = _linear(z, params["proj_z"]["kernel"])
    b = jnp.transpose(b, (0, 3, 1, 2))
    # Mask + softmax in fp32 so the `inf` constant stays finite (in fp16 it would
    # saturate to inf -> fully-masked rows give NaN). fp32 runtime is unchanged.
    mask = mask.astype(jnp.float32)
    b = b.astype(jnp.float32) + (1.0 - mask[:, None]) * (-inf)
    w = jax.nn.softmax(b, axis=-1).astype(v.dtype)
    g = jax.nn.sigmoid(_linear(m, params["proj_g"]["kernel"]))
    o = jnp.einsum("bhij,bhsjd->bhsid", w, v)
    o = jnp.transpose(o, (0, 2, 3, 1, 4))
    o = jnp.reshape(o, (*o.shape[:3], num_heads * c_h))
    return _linear(g * o, params["proj_o"]["kernel"])


def _pair_weighted_averaging_amp(
    params: Params,
    m: jnp.ndarray,
    z: jnp.ndarray,
    mask: jnp.ndarray,
    eps: float,
    inf: float,
    row_chunk_size: int | None,
) -> jnp.ndarray:
    """Preserve native head-wise BF16 sums while optionally splitting MSA rows."""
    # Row-slice evidence binds the native full-MSA operands, not arbitrary native
    # GEMMs with the smaller S. Preserve explicit custom chunk profiles unchanged.
    native_row_profile = (
        row_chunk_size is None or row_chunk_size <= 0 or row_chunk_size >= m.shape[1]
    )
    dtype = params["proj_m"]["kernel"].dtype
    norm = amp_layer_norm if dtype == jnp.bfloat16 else _layer_norm
    z = norm(
        z.astype(jnp.float32), params["norm_z"]["scale"], params["norm_z"]["bias"], eps
    )
    num_heads = params["proj_z"]["kernel"].shape[-1]
    c_h = params["proj_m"]["kernel"].shape[-1] // num_heads
    heads_per_group = 1 if z.shape[1] > _NATIVE_CHUNK_THRESHOLD else num_heads
    weights = []
    for first in range(0, num_heads, heads_per_group):
        last = first + heads_per_group
        kernel = params["proj_z"]["kernel"][:, first:last]
        native_weights = observed_profile(z, kernel)
        logits = pwa_logits(z, kernel) if native_weights else _linear(z, kernel)
        logits = logits.transpose(0, 3, 1, 2).astype(jnp.float32)
        logits = logits + (1 - mask[:, None].astype(jnp.float32)) * -inf
        weights.append(
            (pwa_softmax(logits) if native_weights else jax.nn.softmax(logits, axis=-1))
            .astype(dtype)
        )

    def block(value):
        value = norm(
            value.astype(jnp.float32),
            params["norm_m"]["scale"],
            params["norm_m"]["bias"],
            eps,
        )
        result = None
        for group, weight in enumerate(weights):
            start = group * heads_per_group * c_h
            stop = start + heads_per_group * c_h
            v = _linear(value, params["proj_m"]["kernel"][:, start:stop])
            v = v.reshape(*v.shape[:3], heads_per_group, c_h).transpose(0, 3, 1, 2, 4)
            gate = _linear(value, params["proj_g"]["kernel"][:, start:stop])
            gate = jax.nn.sigmoid(gate.astype(jnp.float32)).astype(dtype)
            output = (
                pair_weighted_contraction(weight, v, original_msa_rows=m.shape[1])
                if native_row_profile
                else jnp.einsum("bhij,bhsjd->bhsid", weight, v)
            )
            output = output.transpose(0, 2, 3, 1, 4)
            output = output.reshape(*output.shape[:3], stop - start)
            output = _linear(gate * output, params["proj_o"]["kernel"][start:stop])
            result = output if result is None else result + output
        return result

    if row_chunk_size is None:
        row_chunk_size = _auto_pair_averaging_chunk(m, params)
    if row_chunk_size is None or row_chunk_size <= 0 or row_chunk_size >= m.shape[1]:
        return block(m)
    return jnp.concatenate(
        [
            block(m[:, start : start + row_chunk_size])
            for start in range(0, m.shape[1], row_chunk_size)
        ],
        axis=1,
    )


def _pair_weighted_averaging_chunked(
    params: Params,
    m: jnp.ndarray,
    z: jnp.ndarray,
    mask: jnp.ndarray,
    eps: float,
    inf: float,
    row_chunk_size: int,
) -> jnp.ndarray:
    """PairWeightedAveraging over blocks of MSA rows.

    The pair weights do not depend on the MSA axis, so they are computed once
    and every block reads them. Everything that does depend on it -- the MSA
    layer norm, both widening projections, the weighted average, and the
    narrowing projection -- is done a block at a time; each reduction still runs
    over its whole axis, so no sum is split.
    """

    z = _layer_norm(z, params["norm_z"]["scale"], params["norm_z"]["bias"], eps)
    num_heads = params["proj_z"]["kernel"].shape[-1]
    c_h = params["proj_m"]["kernel"].shape[-1] // num_heads

    b = _linear(z, params["proj_z"]["kernel"])
    b = jnp.transpose(b, (0, 3, 1, 2))
    mask = mask.astype(jnp.float32)
    b = b.astype(jnp.float32) + (1.0 - mask[:, None]) * (-inf)
    w = jax.nn.softmax(b, axis=-1).astype(m.dtype)

    def block(m_block: jnp.ndarray) -> jnp.ndarray:
        # Normalized per block rather than once up front: the normalized copy is
        # as big as the MSA itself, and nothing outside this block reads it.
        m_block = _layer_norm(
            m_block, params["norm_m"]["scale"], params["norm_m"]["bias"], eps
        )
        v = _linear(m_block, params["proj_m"]["kernel"])
        v = jnp.reshape(v, (*v.shape[:3], num_heads, c_h))
        v = jnp.transpose(v, (0, 3, 1, 2, 4))
        g = jax.nn.sigmoid(_linear(m_block, params["proj_g"]["kernel"]))
        o = jnp.einsum("bhij,bhsjd->bhsid", w, v)
        o = jnp.transpose(o, (0, 2, 3, 1, 4))
        o = jnp.reshape(o, (*o.shape[:3], num_heads * c_h))
        return _linear(g * o, params["proj_o"]["kernel"])

    return jnp.concatenate(
        [
            block(m[:, start : start + row_chunk_size])
            for start in range(0, m.shape[1], row_chunk_size)
        ],
        axis=1,
    )


def _pair_weighted_averaging_grid(
    params: Params,
    m: jnp.ndarray,
    z: jnp.ndarray,
    mask: jnp.ndarray,
    eps: float,
    inf: float,
    row_chunk_size: int | None,
) -> jnp.ndarray:
    """PairWeightedAveraging across the grid's two different layouts.

    The weights live in the pair layout and the values in the MSA layout, and
    the bridge is two collectives per head group:

    * the head logits are projected from each rank's own pair tile and masked
      there -- on the rank that owns the key block -- and then ``all-gather``ed
      along the *column* axis, so a rank holds the logits of its own token
      rows against every key token.  The softmax that follows normalises over
      the complete key axis, exactly as the serial one does, rather than over
      a shard of it.  What travels is ``[B, heads, N/side, N]``: 70 MiB at
      2,096 tokens and eight heads, against the 4.4 GiB the widened values
      would have cost.
    * the widened values ride a ``collective-permute`` ring along the *row*
      axis.  Every hop keeps a rank in its own grid column, which is what
      keeps the alignment shard fixed -- a rank must never see another rank's
      alignment rows here, and nothing is ever summed along that axis.  After
      ``t`` hops a rank holds token block ``(r - t) % side``, and it is that
      block of the gathered weights it contracts against.

    The gates, both projections and the output narrowing are local to a rank's
    own token rows, so the ring is the only thing that moves.  It sits inside
    the MSA-row block for the reason the serial chunk exists at all: the
    widened form is the largest thing here, and the ring holds two of them.

    The head grouping is the serial AMP path's, read off the global token
    count.  The three native CUDA routes that path can take -- ``pwa_logits``,
    ``pwa_softmax`` and ``pair_weighted_contraction`` -- each refuse an active
    mesh and fall back to ``linear``, ``jax.nn.softmax`` and the einsum, so
    this body spells those three out rather than calling helpers that could
    only decline.
    """

    mesh = cp_mesh()
    side = _grid_side()
    depth, tokens = m.shape[1], m.shape[2]
    if z.shape[1] != tokens or z.shape[2] != tokens or mask.shape[1:] != (
        tokens,
        tokens,
    ):
        msg = (
            "the grid PairWeightedAveraging needs one token width for the MSA "
            f"and pair tensors; got {m.shape} against {z.shape} and {mask.shape}"
        )
        raise ValueError(msg)
    depth_pad, token_pad = _grid_pads(depth, tokens)
    if depth_pad or token_pad:
        # Both split axes have to divide the grid. Padded alignment rows and
        # padded token positions are their own rows and columns; the padded
        # output region is sliced away unread below.
        m = jnp.pad(m, ((0, 0), (0, depth_pad), (0, token_pad), (0, 0)))
        z = jnp.pad(z, ((0, 0), (0, token_pad), (0, token_pad), (0, 0)))
        # Minus one, not zero, and that is the whole of the token padding's
        # numerics. The mask enters as `(1 - mask) * -inf`, so a masked real
        # column gets `-inf` and a padded one `-2 * inf`. A query row with any
        # unmasked key cannot tell the two apart -- both weights underflow to
        # zero -- but a row with *no* unmasked key can: the serial softmax
        # there is the softmax of the unmasked logits, the shared `-inf`
        # having cancelled, and it must stay a softmax over the real tokens
        # rather than pick up two columns that do not exist. One penalty level
        # is what makes it, and a padded column still vanishes from every row.
        mask = jnp.pad(
            mask.astype(jnp.float32),
            ((0, 0), (0, token_pad), (0, token_pad)),
            constant_values=-1.0,
        )

    amp = _amp_dtype(params["proj_m"]["kernel"])
    num_heads = params["proj_z"]["kernel"].shape[-1]
    c_h = params["proj_m"]["kernel"].shape[-1] // num_heads
    # Read off the GLOBAL token count, as the serial program reads it: the
    # per-device width would flip this switch on the shard and run a different
    # head grouping than the program this is meant to agree with.
    heads_per_group = (
        1 if amp is not None and tokens > _NATIVE_CHUNK_THRESHOLD else num_heads
    )

    def local(
        m_local: jnp.ndarray,
        z_local: jnp.ndarray,
        mask_local: jnp.ndarray,
        p: Params,
    ) -> jnp.ndarray:
        row = jax.lax.axis_index(CP_ROW_AXIS)
        tile = int(m_local.shape[2])
        norm = amp_layer_norm if amp == jnp.bfloat16 else _layer_norm
        if amp is None:
            z_normed = _layer_norm(
                z_local, p["norm_z"]["scale"], p["norm_z"]["bias"], eps
            )
        else:
            z_normed = norm(
                z_local.astype(jnp.float32),
                p["norm_z"]["scale"],
                p["norm_z"]["bias"],
                eps,
            )
        mask_f32 = mask_local.astype(jnp.float32)

        weights = []
        for first in range(0, num_heads, heads_per_group):
            kernel = p["proj_z"]["kernel"][:, first : first + heads_per_group]
            logits = _linear(z_normed, kernel)
            logits = jnp.transpose(logits, (0, 3, 1, 2)).astype(jnp.float32)
            logits = logits + (1.0 - mask_f32[:, None]) * (-inf)
            # Masked before the gather, so the gathered value is the very
            # logit the serial softmax is fed rather than one this path has to
            # mask again against a mask it would also have to gather.
            logits = jax.lax.all_gather(logits, CP_COL_AXIS, axis=3, tiled=True)
            weight = jax.nn.softmax(logits, axis=-1)
            # Narrowed here rather than at each ring step, which is where the
            # serial path narrows it too, and halves what is held across the
            # MSA-row loop.
            weights.append(weight if amp is None else weight.astype(amp))

        def block(value: jnp.ndarray) -> jnp.ndarray:
            if amp is None:
                normed = _layer_norm(
                    value, p["norm_m"]["scale"], p["norm_m"]["bias"], eps
                )
            else:
                normed = norm(
                    value.astype(jnp.float32),
                    p["norm_m"]["scale"],
                    p["norm_m"]["bias"],
                    eps,
                )
            result = None
            for group, weight in enumerate(weights):
                start = group * heads_per_group * c_h
                stop = start + heads_per_group * c_h
                v = _linear(normed, p["proj_m"]["kernel"][:, start:stop])
                v = jnp.reshape(v, (*v.shape[:3], heads_per_group, c_h))
                v = jnp.transpose(v, (0, 3, 1, 2, 4))
                gate = _linear(normed, p["proj_g"]["kernel"][:, start:stop])
                if amp is None:
                    gate = jax.nn.sigmoid(gate)
                else:
                    gate = jax.nn.sigmoid(gate.astype(jnp.float32)).astype(amp)
                total = None
                work = v
                for step in range(side):
                    # `work` started on rank `row - step`, so it carries that
                    # rank's token block, and the weights it pairs with are
                    # that block's columns of the gathered key axis.
                    offset = ((row - step) % side) * tile
                    keys = jax.lax.dynamic_slice_in_dim(weight, offset, tile, 3)
                    partial = jnp.einsum(
                        "bhij,bhsjd->bhsid",
                        keys.astype(v.dtype),
                        work,
                        preferred_element_type=jnp.float32,
                    )
                    total = partial if total is None else total + partial
                    if step + 1 < side:
                        work = jax.lax.ppermute(
                            work, axis_name=CP_ROW_AXIS, perm=_ring_perm(side)
                        )
                # One rounding of the completed sum, where the serial einsum
                # also rounds once -- the ring's own accumulation stays in the
                # FP32 the einsum accumulates in either way.
                output = jnp.transpose(total.astype(v.dtype), (0, 2, 3, 1, 4))
                output = jnp.reshape(output, (*output.shape[:3], stop - start))
                output = _linear(gate * output, p["proj_o"]["kernel"][start:stop])
                result = output if result is None else result + output
            return result

        rows = row_chunk_size
        if rows is None:
            # Half the serial budget: the ring holds the block it contracts
            # and the block it received at the same instant.
            rows = _auto_pair_averaging_chunk(
                m_local, p, budget=_PWA_BUDGET_BYTES // 2
            )
        if rows is None or rows <= 0 or rows >= m_local.shape[1]:
            return block(m_local)
        return jnp.concatenate(
            [
                block(m_local[:, start : start + rows])
                for start in range(0, m_local.shape[1], rows)
            ],
            axis=1,
        )

    out = jax.shard_map(
        local,
        mesh=mesh,
        in_specs=(
            msa_spec(4),
            pair_spec(4),
            pair_spec(3, row_axis=1, col_axis=2),
            PartitionSpec(),
        ),
        out_specs=msa_spec(4),
    )(m, z, mask, params)
    if depth_pad or token_pad:
        out = shard_msa(
            jax.lax.slice(
                out,
                (0, 0, 0, 0),
                (out.shape[0], depth, tokens, out.shape[3]),
            )
        )
    return out


#: Byte budget for the historical FP32 OuterProductMean product. The memory
#: measurements below predate faithful native AMP contraction; the separate
#: AMP path reuses this conservative ceiling with native hidden-axis chunks.
#:
#: This block widens 32 channels to 32*32 before the output projection narrows
#: them to 128, and it computes that product in float32 -- ``a.float(),
#: b.float()`` is what upstream writes, and it is the largest float32 left in an
#: otherwise bfloat16 trunk. At 1,003 tokens one 128-row block is
#: ``f32[1,128,1003,32,32]``, 502 MiB; the row loop is unrolled, so several of
#: them are live at once. Narrowing the block took the trunk's temp arena from
#: 6,116 MiB to 3,860 MiB at that size -- measured with ``memory_analysis`` on
#: the trunk compiled by itself, since the full prediction program shares its
#: arena with the diffusion and confidence modules, which are float32 by design.
#:
#: 256 MiB rather than a derivation, because the response is not monotonic and
#: only measurement finds the shelf. Swept at 1,003 tokens over an 8,192-row
#: alignment, temp arena against budget: 384 MiB -> 5.93, 320 -> 3.74,
#: 256 -> 3.86, 192 -> 3.86, 128 -> 3.86, 96 -> 5.79, 64 -> 5.80 GiB. Four
#: consecutive settings share the floor and both edges fall off it, so this sits
#: in the middle of the shelf rather than on a lucky point -- but it is a
#: scheduling shelf, so re-sweep it after an XLA upgrade rather than trusting
#: the constant.
#:
#: The budget only ever *tightens* the chunk the caller asked for. Raising it
#: would move the 132-token matched-tape job off the two-block program it
#: already runs, which is the one thing the transition and pair-averaging
#: budgets above are also written to avoid. Below 513 tokens the budget does
#: not engage at all and the float32 default lowers to the program it shipped
#: with; above it the chunk narrows in both dtypes, which is exact but not
#: bit-identical -- XLA tiles the smaller GEMMs differently -- exactly as for
#: the two budgets above.
_OPM_BUDGET_BYTES = 256 * 1024**2


def _auto_outer_product_chunk(n_token: int, widened: int, chunk_size: int) -> int:
    """Token rows whose widened float32 product fits the budget."""
    per_row = n_token * widened * 4
    if per_row <= 0:
        return chunk_size
    return max(1, min(chunk_size, _OPM_BUDGET_BYTES // per_row))


def outer_product_mean_forward(
    params: Params,
    m: jnp.ndarray,
    mask: jnp.ndarray,
    eps: float = 1e-5,
    chunk_size: int = 128,
    preserve_native_amp_shape: bool = False,
) -> jnp.ndarray:
    """Run Boltz OuterProductMean.

    Computes the result in chunks over the i (token) axis so the full
    [b, i, j, c, d] fp32 intermediate is never materialized at once. Peak
    intermediate goes from [N, N, c*d] to [chunk, N, c*d], where ``chunk`` is
    ``chunk_size`` narrowed by ``_OPM_BUDGET_BYTES``.

    ``preserve_native_amp_shape`` retains the native full-token GEMM when its
    product fits that same budget. BF16 reduction rounding depends on GEMM
    tiling, so a gratuitous token split is not numerically inert.
    """

    if _on_msa_grid():
        return _outer_product_mean_grid(
            params, m, mask, eps, chunk_size, preserve_native_amp_shape
        )

    if _amp_dtype(params["proj_a"]["kernel"]) is not None:
        return _outer_product_mean_amp(
            params, m, mask, eps, chunk_size, preserve_native_amp_shape
        )

    mask = mask.astype(m.dtype)
    m = _layer_norm(m, params["norm"]["scale"], params["norm"]["bias"], eps)
    a = _linear(m, params["proj_a"]["kernel"]) * mask[..., None]
    b = _linear(m, params["proj_b"]["kernel"]) * mask[..., None]
    # Upstream writes `torch.einsum(..., a.float(), b.float())`. Both operands
    # are already exactly representable in the compute dtype, so a float32
    # accumulator fed from the narrow operands computes the same products from
    # the same values -- without the widened copies, which are f32[8192,32096]
    # (1,003 MiB) at this job's alignment depth. Measured on an isolated dot,
    # the two forms agree to 1e-6 relative, which is reduction order and
    # nothing else. Under the float32 default the casts were no-ops and
    # `preferred_element_type` is the output dtype the dot already had, so
    # neither line changes that path at all.
    #
    # `num_mask` is the same sum written as a contraction: it counts, for each
    # token pair, the alignment rows where both are present. As a broadcast
    # product it is a [b, s, i, j] tensor -- 16.5 GiB of elementwise work at
    # 8,192 rows, fused away by XLA but still computed -- and the mask is 0/1,
    # so summing it in float32 is exact whatever the order.
    num_mask = jnp.maximum(
        jnp.einsum(
            "bsi,bsj->bij", mask, mask, preferred_element_type=jnp.float32
        ).astype(m.dtype)[..., None],
        1.0,
    )

    proj_o = params["proj_o"]
    n = a.shape[2]
    out_dtype = m.dtype
    chunk = _auto_outer_product_chunk(n, a.shape[-1] * b.shape[-1], chunk_size)
    out = jnp.zeros((a.shape[0], n, n, proj_o["kernel"].shape[-1]), dtype=out_dtype)
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        a_blk = a[:, :, start:end]
        z = jnp.einsum("bsic,bsjd->bijcd", a_blk, b, preferred_element_type=jnp.float32)
        z = jnp.reshape(z, (*z.shape[:3], -1)) / num_mask[:, start:end]
        out = out.at[:, start:end].set(
            _linear(z.astype(out_dtype), proj_o["kernel"], proj_o["bias"])
        )
    return out


def _outer_product_mean_amp(
    params: Params,
    m: jnp.ndarray,
    mask: jnp.ndarray,
    eps: float,
    token_chunk_size: int,
    preserve_native_shape: bool = False,
) -> jnp.ndarray:
    """CUDA AMP contracts to BF16 before FP32 division, even after ``.float()``."""
    dtype = params["proj_a"]["kernel"].dtype
    mask = mask.astype(m.dtype)
    norm = amp_layer_norm if dtype == jnp.bfloat16 else _layer_norm
    m = norm(
        m.astype(jnp.float32), params["norm"]["scale"], params["norm"]["bias"], eps
    )
    a = _linear(m, params["proj_a"]["kernel"]) * mask[..., None]
    b = _linear(m, params["proj_b"]["kernel"]) * mask[..., None]
    # Validity masks are binary: this counts the same entries as the native
    # FP32 sum without materializing its [batch, MSA, token, token] product.
    count = jnp.einsum(
        "bsi,bsj->bij",
        mask.astype(jnp.float32),
        mask.astype(jnp.float32),
        preferred_element_type=jnp.float32,
    )
    count = jnp.maximum(count, 1)[..., None]
    n_tokens, hidden = a.shape[2:]
    native_chunked = n_tokens > _NATIVE_CHUNK_THRESHOLD
    hidden_chunk = 4 if native_chunked else hidden
    if preserve_native_shape and (
        a.shape[0] * n_tokens * n_tokens * hidden_chunk * b.shape[-1] * 4
        <= _OPM_BUDGET_BYTES
    ):
        token_chunk_size = n_tokens
    token_chunk_size = _auto_outer_product_chunk(
        n_tokens, hidden_chunk * b.shape[-1], token_chunk_size
    )
    proj_o = params["proj_o"]
    blocks = []
    for first in range(0, n_tokens, token_chunk_size):
        last = min(first + token_chunk_size, n_tokens)
        result = None
        for start in range(0, hidden, hidden_chunk):
            stop = min(start + hidden_chunk, hidden)
            product = jnp.einsum(
                "bsic,bsjd->bijcd",
                a[:, :, first:last, start:stop].astype(dtype),
                b.astype(dtype),
            )
            product = product.reshape(*product.shape[:3], -1).astype(jnp.float32)
            product = product / count[:, first:last]
            output = _linear(
                product,
                proj_o["kernel"][start * b.shape[-1] : stop * b.shape[-1]],
                None if native_chunked else proj_o["bias"],
            )
            result = output if result is None else result + output
        # Above the threshold native adds the original FP32 bias outside its
        # AMP matmuls, promoting the accumulated low-precision output to FP32.
        if native_chunked:
            result = result + proj_o["bias"]
        blocks.append(result)
    return jnp.concatenate(blocks, axis=1)


def _outer_product_mean_grid(
    params: Params,
    m: jnp.ndarray,
    mask: jnp.ndarray,
    eps: float,
    chunk_size: int,
    preserve_native_shape: bool,
) -> jnp.ndarray:
    """OuterProductMean from the MSA layout into the pair layout.

    Output block ``(I, J)`` is a mean over the whole alignment of an outer
    product between the ``a`` operand of token block ``I`` and the ``b``
    operand of token block ``J``.  On the grid a rank holds neither whole
    thing: it has both operands for its own token block only, and only for its
    own alignment shard.  So:

    * ``b`` and its mask ride a ``collective-permute`` ring along the *row*
      axis.  After ``t`` hops every rank of grid row ``r`` holds key block
      ``(r - t) % side`` -- the same block for the entire grid column, which
      is what makes the reduction below coherent rather than a sum of
      different blocks.
    * the numerator and the mask count are each ``psum``ed along the *column*
      axis, which is the axis that splits the alignment.  Both are blocked --
      over output rows and, on the AMP path, over the hidden axis as well --
      so the full outer-product tensor is never built and no single collective
      carries more than the byte budget one serial block carried.
    * the mean is taken *after* that reduction, against the reduced count and
      with the same clamp, and the output bias is added once.  Averaging
      locally normalised means or adding the bias per shard would both be
      wrong here rather than merely reassociated, and both are pinned by test.
    * exactly one rank of each reduction group owns the block the group just
      computed -- the rank whose grid column is ``(r - t) % side`` -- and it
      keeps it.  The other ranks discard their copy; what is redundant is the
      reduction's fan-out, not the arithmetic, which totals the serial
      operation count exactly.

    The deliberate deviation, and the only one, is that the cross-shard sum of
    the AMP numerator runs in FP32 rather than in the BF16 the local einsum
    emits.  The serial program sums the whole alignment inside one einsum,
    which accumulates in FP32 and rounds once; summing FP32 partials and
    rounding once is nearer to that than rounding the sum a second time.
    """

    mesh = cp_mesh()
    side = _grid_side()
    depth, tokens = m.shape[1], m.shape[2]
    if mask.shape[1:] != (depth, tokens):
        msg = (
            "the grid OuterProductMean needs a mask shaped like the MSA "
            f"tensor's two leading axes; got {mask.shape} against {m.shape}"
        )
        raise ValueError(msg)
    depth_pad, token_pad = _grid_pads(depth, tokens)
    if depth_pad or token_pad:
        m = jnp.pad(m, ((0, 0), (0, depth_pad), (0, token_pad), (0, 0)))
        # Zero: a padded alignment row contributes to neither the numerator
        # nor the count, and a padded token position is sliced away below.
        mask = jnp.pad(mask, ((0, 0), (0, depth_pad), (0, token_pad)))

    amp = _amp_dtype(params["proj_a"]["kernel"])
    # The GLOBAL token count, as the serial program reads it: this switch
    # chooses the native hidden chunking and where the FP32 bias is added.
    native_chunked = tokens > _NATIVE_CHUNK_THRESHOLD

    def local(
        m_local: jnp.ndarray,
        mask_local: jnp.ndarray,
        p: Params,
    ) -> jnp.ndarray:
        row = jax.lax.axis_index(CP_ROW_AXIS)
        column = jax.lax.axis_index(CP_COL_AXIS)
        tile = int(m_local.shape[2])
        proj_o = p["proj_o"]
        mask_cast = mask_local.astype(m_local.dtype)
        if amp is None:
            normed = _layer_norm(
                m_local, p["norm"]["scale"], p["norm"]["bias"], eps
            )
        else:
            norm = amp_layer_norm if amp == jnp.bfloat16 else _layer_norm
            normed = norm(
                m_local.astype(jnp.float32),
                p["norm"]["scale"],
                p["norm"]["bias"],
                eps,
            )
        a = _linear(normed, p["proj_a"]["kernel"]) * mask_cast[..., None]
        b = _linear(normed, p["proj_b"]["kernel"]) * mask_cast[..., None]
        hidden, width = a.shape[-1], b.shape[-1]
        out_dtype = normed.dtype

        if amp is None:
            hidden_chunk = hidden
            token_chunk = _auto_outer_product_chunk(tile, hidden * width, chunk_size)
        else:
            # Exactly the cast the AMP einsum applies to this operand, hoisted
            # above the ring so the hops carry half the bytes. Bitwise inert.
            b = b.astype(amp)
            hidden_chunk = 4 if native_chunked else hidden
            token_chunk = chunk_size
            if preserve_native_shape and (
                a.shape[0] * tile * tile * hidden_chunk * width * 4
                <= _OPM_BUDGET_BYTES
            ):
                token_chunk = tile
            token_chunk = _auto_outer_product_chunk(
                tile, hidden_chunk * width, token_chunk
            )

        # Binary masks, so an FP32 count is exact whatever the order.
        counts = mask_cast.astype(jnp.float32)
        keys, key_counts = b, counts
        result = None
        # The value that keeps every reduction below on one chain; see the
        # barrier inside the block loop for what it is buying.
        chained = None
        for step in range(side):
            owner = column == ((row - step) % side)
            count = jax.lax.psum(
                jnp.einsum(
                    "bsi,bsj->bij",
                    counts,
                    key_counts,
                    preferred_element_type=jnp.float32,
                ),
                CP_COL_AXIS,
            )
            if amp is None:
                count = jnp.maximum(count.astype(out_dtype)[..., None], 1.0)
            else:
                count = jnp.maximum(count, 1)[..., None]
            rows = []
            for first in range(0, tile, token_chunk):
                last = min(first + token_chunk, tile)
                partial = None
                for start in range(0, hidden, hidden_chunk):
                    stop = min(start + hidden_chunk, hidden)
                    operand = (
                        a[:, :, first:last]
                        if amp is None
                        else a[:, :, first:last, start:stop]
                    )
                    if chained is not None:
                        # These reductions are independent, and that is the
                        # problem: an unrolled loop of independent collectives
                        # is one XLA merges into a single collective whose
                        # every operand and every result is then co-live.
                        # Measured at 2,112 tokens on a 2x2 CPU grid, the 272
                        # reductions of this loop and the ring became two
                        # all-reduces of 256 and 16 members, 8,712 MiB of
                        # results -- the whole of that program's arena growth,
                        # against 35 MiB for one block. Threading each
                        # reduction's operand through the previous one's
                        # result makes the sequence a chain the scheduler has
                        # to walk instead, at the cost of the overlap between
                        # one block's communication and the next block's
                        # arithmetic. It is the same trade the ring's row
                        # block took when it became a `lax.scan`.
                        operand, chained = jax.lax.optimization_barrier(
                            (operand, chained)
                        )
                    if amp is None:
                        product = jnp.einsum(
                            "bsic,bsjd->bijcd",
                            operand,
                            keys,
                            preferred_element_type=jnp.float32,
                        )
                        product = jnp.reshape(product, (*product.shape[:3], -1))
                    else:
                        product = jnp.einsum(
                            "bsic,bsjd->bijcd",
                            operand.astype(amp),
                            keys,
                        )
                        product = jnp.reshape(
                            product, (*product.shape[:3], -1)
                        ).astype(jnp.float32)
                    product = jax.lax.psum(product, CP_COL_AXIS)
                    product = product / count[:, first:last]
                    if amp is None:
                        piece = _linear(
                            product.astype(out_dtype),
                            proj_o["kernel"],
                            proj_o["bias"],
                        )
                    else:
                        piece = _linear(
                            product,
                            proj_o["kernel"][start * width : stop * width],
                            None if native_chunked else proj_o["bias"],
                        )
                    partial = piece if partial is None else partial + piece
                    chained = partial
                if amp is not None and native_chunked:
                    # Native adds the original FP32 bias outside its AMP
                    # matmuls, once per output block and not once per chunk.
                    partial = partial + proj_o["bias"]
                rows.append(partial)
            block = jnp.concatenate(rows, axis=1)
            if result is None:
                result = jnp.zeros_like(block)
            result = jnp.where(owner, block, result)
            if step + 1 < side:
                # The next step's reductions join the same chain: its key tile
                # is what the hops produce, and nothing else in it depends on
                # this step's result.
                keys, result = jax.lax.optimization_barrier((keys, result))
                chained = result
                keys = jax.lax.ppermute(
                    keys, axis_name=CP_ROW_AXIS, perm=_ring_perm(side)
                )
                key_counts = jax.lax.ppermute(
                    key_counts, axis_name=CP_ROW_AXIS, perm=_ring_perm(side)
                )
        return result

    out = jax.shard_map(
        local,
        mesh=mesh,
        in_specs=(
            msa_spec(4),
            msa_spec(3),
            PartitionSpec(),
        ),
        out_specs=pair_spec(4),
    )(m, mask, params)
    if token_pad:
        # Re-pinning the slice keeps the partitioner from answering the
        # narrower shape with a replicated result.
        out = shard_pair_rows(
            jax.lax.slice(
                out,
                (0, 0, 0, 0),
                (out.shape[0], tokens, tokens, out.shape[3]),
            )
        )
    return out


def pairformer_no_seq_layer_forward(
    params: Params,
    z: jnp.ndarray,
    pair_mask: jnp.ndarray,
    eps: float = 1e-5,
    chunk_size: int = 128,
    triangle_attention_chunk: int | None = None,
    triangle_attention_q_chunk: int | None = None,
    transition_hidden_chunk: int | None = None,
    matmul_precision: str = "highest",
    triangle_backend: str = "xla",
    glu_backend: str = "xla",
    pair_residual_dtype: jnp.dtype | None = None,
) -> jnp.ndarray:
    """Run Boltz PairformerNoSeqLayer in eval mode."""

    # Row-sharded under context parallelism; identity otherwise. Same seam as
    # `pairformer_layer_forward`.
    z = _residual_cast(shard_pair_rows(z), pair_residual_dtype)
    # A narrowed pair residual is still the CUDA-autocast configuration, so
    # the triangle ops are told so rather than reading it off the activation
    # width and dropping to the FP32-model program.
    native_amp = None if pair_residual_dtype is None else True
    tri_att_chunk = resolve_triangle_attention_chunk(
        z.shape[1], chunk_size, triangle_attention_chunk
    )
    tri_att_q_chunk = resolve_triangle_attention_q_chunk(
        z.shape[1], triangle_attention_q_chunk
    )
    z = _residual_cast(
        z
        + triangle_multiplication_forward(
            params["tri_mul_out"],
            z,
            pair_mask,
            "outgoing",
            eps=eps,
            chunk_size=chunk_size,
            glu_backend=glu_backend,
            native_amp=native_amp,
        ),
        pair_residual_dtype,
    )
    z = _residual_cast(
        z
        + triangle_multiplication_forward(
            params["tri_mul_in"],
            z,
            pair_mask,
            "incoming",
            eps=eps,
            chunk_size=chunk_size,
            glu_backend=glu_backend,
            native_amp=native_amp,
        ),
        pair_residual_dtype,
    )
    z = _residual_cast(
        z
        + triangle_attention_forward(
            params["tri_att_start"],
            z,
            pair_mask,
            starting=True,
            eps=eps,
            chunk_size=tri_att_chunk,
            q_chunk_size=tri_att_q_chunk,
            matmul_precision=matmul_precision,
            triangle_backend=triangle_backend,
            native_amp=native_amp,
        ),
        pair_residual_dtype,
    )
    z = _residual_cast(
        z
        + triangle_attention_forward(
            params["tri_att_end"],
            z,
            pair_mask,
            starting=False,
            eps=eps,
            chunk_size=tri_att_chunk,
            q_chunk_size=tri_att_q_chunk,
            matmul_precision=matmul_precision,
            triangle_backend=triangle_backend,
            native_amp=native_amp,
        ),
        pair_residual_dtype,
    )
    z = _residual_cast(
        z
        + transition_forward(
            params["transition_z"],
            z,
            chunk_size=transition_hidden_chunk,
            eps=eps,
            row_chunk_size=chunk_size,
            glu_backend=glu_backend,
            native_amp_norm=(
                params["transition_z"]["fc1"]["kernel"].dtype == jnp.bfloat16
            ),
            # `z` is the pair tensor the active layout shards, so the row
            # block above is taken inside the shard instead of being dropped.
            cp_pair=True,
        ),
        pair_residual_dtype,
    )
    return z
