"""ESMFold2's pair trunk: triangle updates, transitions, and the block stack.

Written against `docs/ports/esmfold2/trunk-spec.md`, which records what the
torch source does and where it is easy to read it wrongly. The three places
that matter here, all checked by the parity tests beside this file:

* native explicit `.float()` contraction operands are narrowed again by CUDA
  BF16 autocast; the opt-in native path preserves that BF16 output boundary;
* `proj_bundle` packs `[left | right | gate_left | gate_right]`, the input
  gate multiplies before the mask, and the output gate is computed from the
  same normalised input rather than from the contraction;
* `Transition` carries its residual internally and `PairTransition` does not,
  so the caller adds one and not the other.
"""

from __future__ import annotations

from collections.abc import Mapping
from math import prod

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec

from foldjax.models._cp import (
    CP_COL_AXIS,
    CP_ROW_AXIS,
    col_skew_perm,
    cp_grid,
    cp_mesh,
    pair_spec,
    permute,
    ring_perm,
    row_skew_perm,
    shard_pair_rows,
    transpose_perm,
)
from foldjax.models._cp import cp_layout as _cp_layout
from foldjax.models.esmfold2.models.primitives import layer_norm, linear, swiglu

Params = Mapping[str, jnp.ndarray]


def _cannon_contract(
    a: jnp.ndarray,
    b: jnp.ndarray,
    *,
    outgoing: bool,
) -> jnp.ndarray:
    """Contract two doubly-sharded pair projections by Cannon's algorithm.

    Under the ``2d`` layout device ``(p, q)`` holds the ``(p, q)`` tile of every
    pair tensor, so the contracted axis lives on a different device from the
    operand that needs it and no sharding constraint can express the schedule.
    Cannon's algorithm is what the other ports' pair cores use, through the same
    shared tile routing: align the operands once so every device starts on a
    matching block of ``k``, then alternate a local product with a one-hop
    shift, ``side`` times. The per-device transient is two tiles --
    ``O((N/side)^2 C)`` -- and nothing full-width is ever built.

    The local einsum is deliberately the *same* one the dense path uses, once
    per direction. Cannon permutes whole tiles; it does not change what the axes
    inside a tile mean, so a tile of ``a`` is still indexed ``[i, k]`` for
    outgoing and ``[k, i]`` for incoming. Rewriting the step as a canonical
    ``[i,k] @ [k,j]`` matmul is the tempting error and it is wrong in both
    directions -- measured elsewhere in this tree at 15.9 (outgoing) and 12.3
    (incoming) on values of order one.

    The result is float32, which is what ``preferred_element_type`` gives the
    serial and one-dimensional einsums this replaces; the native-autocast
    caller narrows it again itself, exactly where its own chunk loop did.
    """

    mesh = cp_mesh()
    side = cp_grid()[0]
    if a.ndim < 3 or a.shape[-3] != a.shape[-2]:
        raise ValueError(
            "the 2-D context-parallel triangle contraction needs square pair "
            f"axes, got shape {tuple(a.shape)}"
        )
    n = a.shape[-3]
    pad = (-n) % side
    if pad:
        # `shard_map` needs both pair axes to divide the grid. The projections
        # are padded rather than the pair state, so the padding is exactly zero
        # by construction rather than by way of the mask: a padded block of `k`
        # contributes nothing to the sum, and the padded output region is
        # sliced off below.
        widths = [(0, 0)] * a.ndim
        widths[-3] = widths[-2] = (0, pad)
        a = jnp.pad(a, widths)
        b = jnp.pad(b, widths)
    spec = pair_spec(a.ndim)
    subscript = "...ikd,...jkd->...ijd" if outgoing else "...kid,...kjd->...ijd"

    def body(lhs: jnp.ndarray, rhs: jnp.ndarray) -> jnp.ndarray:
        if outgoing:
            rhs = permute(rhs, transpose_perm(side))
        else:
            lhs = permute(lhs, transpose_perm(side))
        lhs = permute(lhs, row_skew_perm(side))
        rhs = permute(rhs, col_skew_perm(side))
        total = None
        correction = None
        for step in range(side):
            # Float32 accumulation without widening the operands, which is what
            # the dense path's `preferred_element_type` does; with a float32
            # trunk the two forms are identical.
            term = jnp.einsum(
                subscript, lhs, rhs, preferred_element_type=jnp.float32
            )
            if total is None:
                total = term
                correction = jnp.zeros_like(term)
            else:
                updated = total + term
                residual = jnp.where(
                    jnp.abs(total) >= jnp.abs(term),
                    (total - updated) + term,
                    (term - updated) + total,
                )
                total = updated
                correction = correction + residual
            if step + 1 < side:
                lhs = permute(lhs, ring_perm(side, axis=CP_COL_AXIS, delta=-1))
                rhs = permute(rhs, ring_perm(side, axis=CP_ROW_AXIS, delta=-1))
        return total + correction

    out = jax.shard_map(body, mesh=mesh, in_specs=(spec, spec), out_specs=spec)(a, b)
    if pad:
        out = jax.lax.slice_in_dim(out, 0, n, axis=-3)
        out = jax.lax.slice_in_dim(out, 0, n, axis=-2)
    return out


def _native_bf16_linear(x, weight):
    # Native autocast materializes this BF16 output before residual addition.
    # The barrier prevents GEMM beta=1 fusion across that rounding boundary;
    # native cuBLAS algorithm selection remains a separate compiler control.
    # A vector-shaped dot may become elementwise BF16 multiply + FP32 reduce,
    # rounding each product before summation. Native linear retains the product
    # in FP32. Keep the measured BF16 GEMM destination for non-vector shapes.
    vector = prod(x.shape[:-1]) == 1 or weight.shape[0] == 1
    lhs, rhs = x.astype(jnp.bfloat16), weight.astype(jnp.bfloat16)
    if vector:
        # Otherwise excess-precision simplification can erase the input casts
        # after it rewrites the vector dot to FP32 elementwise operations.
        lhs, rhs = jax.lax.optimization_barrier((lhs, rhs))
    result = jnp.matmul(
        lhs,
        rhs.T,
        precision="default",
        preferred_element_type=jnp.float32 if vector else None,
    )
    return jax.lax.optimization_barrier(result.astype(jnp.bfloat16))


def _autocast_linear_fallback(x, params, prefix):
    result = jnp.matmul(
        x.astype(jnp.bfloat16),
        params[f"{prefix}.weight"].astype(jnp.bfloat16).T,
        preferred_element_type=jnp.float32,
    )
    if f"{prefix}.bias" in params:
        result = result + params[f"{prefix}.bias"].astype(jnp.bfloat16).astype(
            jnp.float32
        )
    return result.astype(jnp.bfloat16)


def _autocast_linear(x, params, prefix):
    if f"{prefix}.bias" not in params and cp_mesh() is None:
        return jax.lax.platform_dependent(
            x,
            params[f"{prefix}.weight"],
            cuda=_native_bf16_linear,
            default=lambda a, w: _autocast_linear_fallback(
                a, {f"{prefix}.weight": w}, prefix
            ),
        )
    # Bias-bearing linears, CPU/TPU and CP retain the existing implementation
    # until their native output/accumulation policies are independently checked.
    return _autocast_linear_fallback(x, params, prefix)


def _autocast_norm(x, params, prefix, eps=1e-5):
    from foldjax.models._cp import cp_mesh
    from foldjax.models.boltz2.models.primitives.native_amp_norm import _cuda_layer_norm

    x = x.astype(jnp.float32)
    weight, bias = params[f"{prefix}.weight"], params[f"{prefix}.bias"]
    # The shared vector-4 CUDA Welford/FMA implementation matched the captured
    # first width256 LM norm bitwise, including effective BF16 rounding ties.
    # Later norm sites and full-block parity require separate evidence.
    # Preserve the ESM formula on other devices, widths and CP layouts.
    msa_norm = (
        x.shape[-1] == 128
        and prefix.startswith("msa_encoder.blocks.")
        and prefix.endswith((
            ".outer_product_mean.norm", ".msa_pair_weighted_averaging.norm_single",
            ".msa_transition.norm",
        ))
    )
    # Captured first OPM, PWA and transition MSA norms match this reduction.
    if (x.shape[-1] == 256 or msa_norm) and cp_mesh() is None:
        return jax.lax.platform_dependent(
            x,
            weight,
            bias,
            cuda=lambda a, w, b: _cuda_layer_norm(a, w, b, eps)[0],
            default=lambda a, w, b: layer_norm(a, w, b, eps=eps),
        )
    return layer_norm(
        x,
        weight,
        bias,
        eps=eps,
    )


#: Rows of a pair tensor the native autocast path shapes at once.
#:
#: The same 64-row rule `_autocast_transition` below spells inline, named so
#: the triangle prologue and a test can share it.
_AUTOCAST_ROWS = 64


#: `proj_bundle`'s two contraction operands, in the order it emits them.
_LEFT, _RIGHT = 0, 1


def _route(bundled, mask):
    """`proj_bundle`'s output through its own gate and the visibility mask.

    Split in half rather than into the two operands: the halves are the signal
    and the gate logits, and `[left | right]` sits *inside* each of them. A
    bundle restricted to one operand's rows (`_operand_params`) is therefore
    routed by this same function, unchanged.
    """
    signal, logits = jnp.split(bundled, 2, axis=-1)
    routed = signal * jax.nn.sigmoid(logits.astype(jnp.float32)).astype(jnp.bfloat16)
    if mask is not None:
        # Cast down, the way `triangle_multiplicative`'s own body does and the
        # way upstream does at `modeling_esmfold2.py:1098` -- "so masking does
        # not promote the O(N^3) contraction to fp32". `pair_mask` is float32
        # (`model.py:1310`) and `routed` is bfloat16, so without this the
        # multiply promotes the widest tensor this block owns.
        routed = routed * mask[..., None].astype(routed.dtype)
    return routed


def _output_gate(normalized, params, engine):
    """`proj_gate`, which reads the normalised input and not the contraction."""
    return jax.nn.sigmoid(
        _autocast_linear(normalized, params, f"{engine}.proj_gate").astype(jnp.float32)
    ).astype(jnp.bfloat16)


def _triangle_prologue(pair, params, engine, mask, eps):
    """`norm_start` through the split, over whatever rows it is handed.

    Everything here is elementwise in the two token axes and contracts only
    over channels, so a block of rows is the whole computation for its own
    rows: nothing reads a row it was not given, and no value depends on how
    the rows were divided.

    `proj_gate` is computed here rather than in the epilogue for that reason.
    It is a pure function of `normalized`, so moving it changes no value --
    and it is the only other consumer of `normalized`, whose live range
    otherwise straddles the whole O(N^3) contraction and is what forces the
    float32 normalisation to exist at full width. Handing the epilogue the
    bfloat16 gate instead halves what the contraction has to step over.

    Both operands and the gate at once, which is what the contraction needs
    when it is fed whole -- the sharded schedules below, and any pair small
    enough that a block would divide nothing. The serial program streams the
    contraction instead and asks for one operand at a time
    (`_autocast_triangle_streamed`).
    """
    normalized = _autocast_norm(pair, params, f"{engine}.norm_start", eps)
    routed = _route(_autocast_linear(normalized, params, f"{engine}.proj_bundle"), mask)
    left, right = jnp.split(routed, 2, axis=-1)
    return left, right, _output_gate(normalized, params, engine)


def _operand_params(params, engine, half):
    """`proj_bundle` restricted to the rows one contraction operand reads.

    The bundle emits `[signal | logits]`, each of which is `[left | right]`,
    so operand `half` keeps output columns `[h*L, (h+1)*L)` of the signal and
    the matching columns of the logits, `2L` further along -- rows of the
    weight, which is stored `[out, in]`.

    A matmul's output columns do not depend on one another, so taking those
    rows alone computes exactly the columns that operand keeps, for half the
    arithmetic. That is what lets the streamed contraction below pass over
    `norm_start` twice and still spend one `proj_bundle` in total rather than
    two. Measured bitwise identical to slicing the full product, per operand,
    on CPU.
    """
    prefix = f"{engine}.proj_bundle"
    latent = params[f"{prefix}.weight"].shape[0] // 4
    selected = {}
    for name in ("weight", "bias"):
        if f"{prefix}.{name}" not in params:
            continue
        rows = params[f"{prefix}.{name}"]
        selected[f"{prefix}.{name}"] = jnp.concatenate(
            [
                rows[half * latent : (half + 1) * latent],
                rows[(2 + half) * latent : (3 + half) * latent],
            ],
            axis=0,
        )
    return selected


def _triangle_operand(pair, params, engine, mask, eps, *, half, gate):
    """One contraction operand, and optionally the output gate beside it.

    `half` is `_LEFT` or `_RIGHT`. Everything here is elementwise in the two
    token axes, so the caller may hand this a block of rows *or* a block of
    columns and get that block of the operand back.
    """
    normalized = _autocast_norm(pair, params, f"{engine}.norm_start", eps)
    operand = _route(
        _autocast_linear(
            normalized, _operand_params(params, engine, half), f"{engine}.proj_bundle"
        ),
        mask,
    )
    if not gate:
        return operand, None
    return operand, _output_gate(normalized, params, engine)


def _triangle_prologue_rows(pair, params, engine, mask, eps, *, rows):
    """`_triangle_prologue` in row blocks, assembling the whole operands.

    The sharded arrangement, and the reason it is not the serial one: a
    sharded contraction is a fixed schedule over whole operands -- Cannon's
    algorithm on the grid, one dense einsum plus an all-reduce on the row mesh
    -- so its operands have to be assembled before it starts, and this keeps
    only what the block was for. What the block removes is everything between
    the normalisation and the split: the float32 `normalized`, the
    `proj_bundle` output that `_native_bf16_linear`'s barrier pins at four
    times the pair width, and `routed`, none of which the contraction reads.

    The three assembled operands are themselves full width, which the serial
    program does not pay: it streams the contraction and never assembles more
    than one (`_autocast_triangle_streamed`).
    """
    n_rows = pair.shape[-3]
    if rows >= n_rows:
        return _triangle_prologue(pair, params, engine, mask, eps)
    pieces = []
    for start in range(0, n_rows, rows):
        # `min` because a trailing block is shorter whenever the axis does not
        # divide, and `slice_in_dim` rejects an overrun rather than clamping
        # it the way Python slicing would.
        stop = min(start + rows, n_rows)
        pieces.append(
            _triangle_prologue(
                jax.lax.slice_in_dim(pair, start, stop, axis=-3),
                params,
                engine,
                # `[B, N, N]` against the pair's `[B, N, N, C]`: the mask's
                # first token axis is one further along.
                None
                if mask is None
                else jax.lax.slice_in_dim(mask, start, stop, axis=-2),
                eps,
            )
        )
    return tuple(
        jnp.concatenate(parts, axis=-3) for parts in zip(*pieces, strict=True)
    )


def _triangle_prologue_blocked(pair, params, engine, mask, eps):
    """Row-block the prologue, on local rows whenever a mesh is active.

    A block of a sharded row axis is a slice the partitioner can only serve by
    moving data -- the cost `_cp_pair_transition` records 104 `all-to-all`s
    for -- so under either layout the block is taken inside a `shard_map`, on
    the tile the device already holds. Every operation in the prologue is
    elementwise in the two token axes and contracts only over channels, so a
    tile is the whole computation for its own rows and columns: no collective,
    and the same arithmetic per element the unblocked sharded program did.

    A block at least as wide as the local tile would divide nothing, and
    asking for it on the global axis is what forces the move, so the request
    is dropped rather than relocated and the sharded program is left exactly
    as it was. At the released 64 rows that is every context-parallel program
    below 64 rows per device, which is every fixture the suite runs.

    Serially there is nothing to relocate and the prologue is whole: the
    blocked serial route is `_autocast_triangle_streamed`, which cuts the
    contraction rather than the prologue, so the only serial caller left here
    is a pair the block would not divide.
    """
    mesh = cp_mesh()
    if mesh is None:
        return _triangle_prologue(pair, params, engine, mask, eps)
    grid_rows, grid_columns = cp_grid()
    n_rows, n_columns = pair.shape[-3], pair.shape[-2]
    row_pad, column_pad = (-n_rows) % grid_rows, (-n_columns) % grid_columns
    if _AUTOCAST_ROWS >= (n_rows + row_pad) // grid_rows:
        return _triangle_prologue(pair, params, engine, mask, eps)
    operands, specs = [], []
    # The first token axis of each operand: `[B, N, N, C]` has it at -3 and its
    # mask `[B, N, N]` at -2, with the second token axis the next one along.
    for array, first_axis in ((pair, -3), *(() if mask is None else ((mask, -2),))):
        row = first_axis % array.ndim
        if row_pad or column_pad:
            # `shard_map` needs both sharded axes to divide the grid. The
            # padded rows and columns are their own, and the prologue never
            # mixes them with a kept one, so the padded output is sliced away
            # unread before the contraction sees it.
            width = [(0, 0)] * array.ndim
            width[row], width[row + 1] = (0, row_pad), (0, column_pad)
            array = jnp.pad(array, width)
        # Pinning each operand to the layout its blocks are cut on keeps
        # `shard_map` from having to reshard the mask, which is built from a
        # per-token mask rather than placed as a pair tensor.
        operands.append(shard_pair_rows(array, row_axis=row, col_axis=row + 1))
        specs.append(pair_spec(array.ndim, row_axis=row, col_axis=row + 1))

    def local(*sharded):
        *arrays, params_local = sharded
        return _triangle_prologue_rows(
            arrays[0],
            params_local,
            engine,
            arrays[1] if len(arrays) > 1 else None,
            eps,
            rows=_AUTOCAST_ROWS,
        )

    # Parameters go in as a replicated operand rather than a closure: the
    # trunk runs inside the recycling `lax.scan`, where they are traced
    # values, and an operand keeps the whole tree on the mesh. Only this
    # engine's own leaves, the way `_cp_pair_transition` takes them -- the
    # whole trunk tree as an operand is a `shard_map` input per direction per
    # layer, which the compiler then has to prune.
    block_params = {
        name: value
        for name, value in params.items()
        if name.startswith(f"{engine}.")
    }
    out_spec = pair_spec(pair.ndim)
    out = jax.shard_map(
        local,
        mesh=mesh,
        in_specs=(*specs, PartitionSpec()),
        out_specs=(out_spec, out_spec, out_spec),
    )(*operands, block_params)
    if not (row_pad or column_pad):
        return out
    # Re-pinning each slice keeps the partitioner from answering the narrower
    # shape with a replicated result.
    return tuple(
        shard_pair_rows(
            jax.lax.slice_in_dim(
                jax.lax.slice_in_dim(part, 0, n_rows, axis=-3),
                0,
                n_columns,
                axis=-2,
            )
        )
        for part in out
    )


def _triangle_operand_rows(pair, params, engine, mask, eps, *, rows, half):
    """The one operand the streamed contraction needs whole, in row blocks.

    Row blocks and not column blocks only because the pair's first token axis
    is the major one; the prologue is elementwise in both, so either would
    give the same value. What the block keeps out is the `proj_bundle` output
    at twice this operand's width, pinned by `_native_bf16_linear`'s barrier.
    """
    n_rows = pair.shape[1]
    if rows >= n_rows:
        return _triangle_operand(
            pair, params, engine, mask, eps, half=half, gate=False
        )[0]
    pieces = []
    for start in range(0, n_rows, rows):
        # `min` because a trailing block is shorter whenever the axis does not
        # divide, and `slice_in_dim` rejects an overrun rather than clamping it
        # the way Python slicing would.
        stop = min(start + rows, n_rows)
        pieces.append(
            _triangle_operand(
                jax.lax.slice_in_dim(pair, start, stop, axis=1),
                params,
                engine,
                # The pair's `[B, N, N, C]` and its mask's `[B, N, N]` carry
                # the token axes at 1 and 2 alike, which is why one index
                # serves both here and in the loop below.
                None
                if mask is None
                else jax.lax.slice_in_dim(mask, start, stop, axis=1),
                eps,
                half=half,
                gate=False,
            )[0]
        )
    return jnp.concatenate(pieces, axis=1)


def _autocast_triangle_streamed(pair, params, engine, outgoing, mask, eps):
    """Contract in blocks of the output, with one operand whole and the rest cut.

    The contraction sums over `k`, so one of its two operands is read entirely
    by every block of the output and the other is read a block at a time::

        outgoing  out[i, j] = sum_k left[i, k] * right[j, k]
        incoming  out[i, j] = sum_k left[k, i] * right[k, j]

    Cutting the output on `i` for the outgoing direction and on `j` for the
    incoming one is what makes the *same* slice of the pair serve both the
    operand that is cut and the output gate the epilogue multiplies by -- so
    each block costs one `norm_start` rather than two, and the whole operand
    is `right` outgoing and `left` incoming. That is the one full-width
    bfloat16 pair tensor this block builds; `left`/`right`/`output_gate`
    assembled whole, and the contraction's own concatenated destination, were
    four of them, and at 2,096 tokens each is 2,145 MiB.

    The epilogue is elementwise in the two token axes like the prologue, so it
    runs on the block too and the emitted pair update is written straight into
    the result. Nothing here reduces over a cut axis; the block is a choice of
    shapes, and the GEMM tiling it reaches is the reason the tests assert a
    bfloat16 ULP rather than bit equality.
    """
    rows = _AUTOCAST_ROWS
    axis = 1 if outgoing else 2
    whole = _triangle_operand_rows(
        pair,
        params,
        engine,
        mask,
        eps,
        rows=rows,
        half=_RIGHT if outgoing else _LEFT,
    )
    equation = "bikd,bjkd->bijd" if outgoing else "bkid,bkjd->bijd"
    blocks = []
    for start in range(0, pair.shape[axis], rows):
        stop = min(start + rows, pair.shape[axis])
        operand, output_gate = _triangle_operand(
            jax.lax.slice_in_dim(pair, start, stop, axis=axis),
            params,
            engine,
            None
            if mask is None
            else jax.lax.slice_in_dim(mask, start, stop, axis=axis),
            eps,
            half=_LEFT if outgoing else _RIGHT,
            gate=True,
        )
        lhs, rhs = (operand, whole) if outgoing else (whole, operand)
        # The float32 accumulation is `preferred_element_type`'s, as it was
        # when this loop only chunked the einsum: upstream spells
        # `routed.float().chunk(2, ...)` here and lets autocast narrow the
        # operands back at the GEMM, which is bfloat16 in and float32 out.
        contracted = jnp.einsum(
            equation,
            lhs.astype(jnp.bfloat16),
            rhs.astype(jnp.bfloat16),
            preferred_element_type=jnp.float32,
        ).astype(jnp.bfloat16)
        blocks.append(
            _finish_autocast_triangle(contracted, output_gate, params, engine, eps)
        )
    return jnp.concatenate(blocks, axis=axis)


def _autocast_triangle(pair, params, prefix, outgoing, mask, eps):
    engine = f"{prefix}._engine" if prefix else "_engine"
    if cp_mesh() is None and _AUTOCAST_ROWS < pair.shape[1]:
        return _autocast_triangle_streamed(pair, params, engine, outgoing, mask, eps)
    left, right, output_gate = _triangle_prologue_blocked(
        pair, params, engine, mask, eps
    )
    if _cp_layout() == "2d":
        # Both pair axes are sharded, so the contraction is Cannon's algorithm
        # -- skew, then one local matmul per ring hop -- and nothing full-width
        # is built. The narrowing back to bfloat16 is the one the chunk loop
        # below does to each of its chunks, done once to the whole result:
        # `preferred_element_type` keeps the float32 accumulation inside the
        # ring exactly as it keeps it inside the dense einsum.
        #
        # This branch is reachable and the reference path's is not the only one
        # that needs it: `lm_encoder_params` is gated on the trunk dtype alone
        # (`models/model.py:1444`), so the language-model encoder's trunk runs
        # native autocast under a mesh while every other pair stack falls back.
        return _finish_autocast_triangle(
            _cannon_contract(
                left.astype(jnp.bfloat16),
                right.astype(jnp.bfloat16),
                outgoing=outgoing,
            ).astype(jnp.bfloat16),
            output_gate,
            params,
            engine,
            eps,
        )
    equation = "bikd,bjkd->bijd" if outgoing else "bkid,bkjd->bijd"
    # Pinned native default chunks the output i dimension at 64. Upstream
    # spells `routed.float().chunk(2, ...)` here and lets autocast narrow the
    # operands back at the GEMM; this reproduces the arithmetic without
    # materialising the float32 in between. Bit-identical, not merely exact:
    # bfloat16 -> float32 -> bfloat16 is the identity, and `pair_mask` is
    # 0.0/1.0, so neither step the promotion used to add could change a value.
    # The float32 accumulation is kept by `preferred_element_type` below, as
    # it already was.
    chunks = []
    for start in range(0, pair.shape[1], 64):
        operand = (
            left[:, start : start + 64] if outgoing else left[:, :, start : start + 64]
        )
        chunks.append(
            jnp.einsum(
                equation,
                operand.astype(jnp.bfloat16),
                right.astype(jnp.bfloat16),
                preferred_element_type=jnp.float32,
            ).astype(jnp.bfloat16)
        )
    contracted = jnp.concatenate(chunks, axis=1)
    return _finish_autocast_triangle(contracted, output_gate, params, engine, eps)


def _finish_autocast_triangle(contracted, output_gate, params, engine, eps):
    """The native block's epilogue, shared by its dense and Cannon branches.

    The output gate arrives already computed: it is a function of the
    normalised *input*, not of the contraction, and `_triangle_prologue`
    produces it beside `left` and `right` so the float32 normalisation does
    not have to outlive the contraction to be read here.
    """
    mixed = _autocast_linear(
        _autocast_norm(contracted, params, f"{engine}.norm_mix", eps),
        params,
        f"{engine}.proj_emit",
    )
    return mixed * output_gate


def _autocast_transition(x, params, prefix, residual, eps):
    dot = f"{prefix}." if prefix else ""
    outputs = []
    for start in range(0, x.shape[1], 64):
        part = x[:, start : start + 64]
        packed = _autocast_linear(
            _autocast_norm(part, params, f"{dot}norm", eps),
            params,
            f"{dot}ffn.w12",
        )
        gate, value = jnp.split(packed, 2, axis=-1)
        hidden = jax.nn.silu(gate.astype(jnp.float32)).astype(jnp.bfloat16) * value
        update = _autocast_linear(hidden, params, f"{dot}ffn.w3")
        outputs.append(part + update if residual else update)
    return jnp.concatenate(outputs, axis=1)


def triangle_multiplicative(
    pair: jnp.ndarray,
    params: Params,
    prefix: str,
    *,
    outgoing: bool,
    mask: jnp.ndarray | None = None,
    eps: float = 1e-5,
    native_autocast: bool = False,
) -> jnp.ndarray:
    """`TriangleMultiplicativeBlock`, reference path.

    `prefix` names the `TriangleMultiplicativeUpdate` that owns it; the
    `_engine` level upstream inserts is added here so callers spell the module
    the way the checkpoint does.
    """
    if native_autocast:
        return _autocast_triangle(pair, params, prefix, outgoing, mask, eps)
    engine = f"{prefix}._engine" if prefix else "_engine"
    normalised = layer_norm(
        pair,
        params[f"{engine}.norm_start.weight"],
        params[f"{engine}.norm_start.bias"],
        eps=eps,
    )
    bundled = linear(normalised, params, f"{engine}.proj_bundle")
    width = bundled.shape[-1] // 2
    signal, gate_logits = bundled[..., :width], bundled[..., width:]
    routed = signal * jax.nn.sigmoid(gate_logits)
    if mask is not None:
        routed = routed * mask[..., None].astype(routed.dtype)

    # No cast here. Upstream does the opposite, and says so at
    # `modeling_esmfold2.py:1098`: it casts the fp32 visibility mask *down* to
    # `routed.dtype` "so masking does not promote the O(N^3) contraction to
    # fp32". The line above already reproduces that down-cast; promoting the
    # whole tensor immediately afterwards undid it, on the two widest buffers
    # this block owns -- `routed` is [N, N, 2*dim] and `left`/`right` are half
    # that each. The float32 accumulation upstream gets from its bf16 GEMM is
    # kept explicitly by `preferred_element_type` on the einsum below, so the
    # arithmetic that mattered is unchanged; only the operand storage narrows.
    #
    # The comment this replaces said "Upstream forces float32 here". It does
    # not, and never did in any released version.
    half = routed.shape[-1] // 2
    left, right = routed[..., :half], routed[..., half:]
    if _cp_layout() == "2d":
        # Fold-CP's own schedule: both pair axes are sharded, so the
        # contraction runs as Cannon's algorithm -- skew, then one local matmul
        # per ring hop. Nothing full-width is ever built, so the row-transpose
        # rewrite below, which only moves the all-reduce off the sharded axis,
        # has nothing left to fix.
        contracted = _cannon_contract(left, right, outgoing=outgoing)
    else:
        contract_outgoing = outgoing
        if cp_mesh() is not None and not outgoing:
            # The incoming contraction sums over the sharded row axis, which
            # the partitioner realises as a full-size float32 partial plus an
            # all-reduce per device. Swapping the pair axes and contracting in
            # the outgoing form is the same arithmetic with sharded partials.
            left = shard_pair_rows(jnp.swapaxes(left, 1, 2))
            right = shard_pair_rows(jnp.swapaxes(right, 1, 2))
            contract_outgoing = True
        equation = "bikd,bjkd->bijd" if contract_outgoing else "bkid,bkjd->bijd"
        contracted = jnp.einsum(
            equation, left, right, preferred_element_type=jnp.float32
        )
    contracted = shard_pair_rows(contracted)

    mixed = layer_norm(
        contracted,
        params[f"{engine}.norm_mix.weight"],
        params[f"{engine}.norm_mix.bias"],
        eps=eps,
    )
    mixed = linear(mixed.astype(pair.dtype), params, f"{engine}.proj_emit")
    out_gate = jax.nn.sigmoid(linear(normalised, params, f"{engine}.proj_gate"))
    return mixed * out_gate


def transition(
    x: jnp.ndarray,
    params: Params,
    prefix: str,
    *,
    residual: bool,
    eps: float = 1e-5,
    native_autocast: bool = False,
    cp_pair: bool = False,
) -> jnp.ndarray:
    """Norm, SwiGLU, and upstream's two different opinions about the residual.

    `common.Transition` adds `x` back; `modeling.PairTransition` returns the
    update alone and lets its caller add it. Same parameters, same shapes --
    only the caller can tell them apart, so it is a flag rather than a guess.

    `cp_pair` declares `x` a pair tensor `[B, N, N, C]` laid out by the active
    context-parallel layout, which is what lets the row block survive the
    square grid -- see `_cp_pair_transition`. Any other caller, and every
    caller on the row mesh, keeps the path it already had.
    """
    if cp_pair and _cp_layout() == "2d" and x.ndim == 4:
        return _cp_pair_transition(
            x,
            params,
            prefix,
            residual=residual,
            eps=eps,
            native_autocast=native_autocast,
        )
    if native_autocast:
        return _autocast_transition(x, params, prefix, residual, eps)
    dot = f"{prefix}." if prefix else ""
    normalised = layer_norm(
        x, params[f"{dot}norm.weight"], params[f"{dot}norm.bias"], eps=eps
    )
    update = swiglu(normalised, params, f"{dot}ffn")
    return x + update if residual else update


def _cp_pair_transition(
    x: jnp.ndarray,
    params: Params,
    prefix: str,
    *,
    residual: bool,
    eps: float,
    native_autocast: bool,
) -> jnp.ndarray:
    """Row-block the transition on each device's own tile of a pair tensor.

    `w12` widens `[..., C]` to `[..., 2 * hidden]` and the split halves plus
    their product are live at once, so `primitives.swiglu` blocks a leading
    axis once the widened form passes its budget -- and the axis it picks on a
    pair tensor is the token rows. Under the square grid those rows are
    sharded, and a block of them is a slice of a sharded axis: the partitioner
    cannot serve it without moving data. Measured on the two-layer CPU fixture
    at twelve tokens on a 2x2 mesh, with the budget lowered so the block
    fires, the partitioned trunk carried 104 `all-to-all`s that the unblocked
    program has none of.

    The block therefore has to be taken inside the shard, the way
    `_cannon_contract` above takes the contraction. Every operation in the
    transition is elementwise in the two token axes and contracts only over
    channels, so a device's tile is the whole computation for its own rows and
    columns: no collective, and the arithmetic per element is the arithmetic
    the unblocked sharded program did.

    The row mesh is deliberately left alone. Its block is a slice of the
    sharded rows too, but the one-dimensional program is what every recorded
    ESMFold2 context-parallel number describes, and it stays byte-identical.

    The body is `transition` itself with the flag off rather than a third
    function, so the serial and one-dimensional programs keep the call stack
    they had: a frame this path added would appear in their debug metadata,
    which is the one part of an "identical program" claim that is checkable
    and would then be false.
    """

    mesh = cp_mesh()
    rows, columns = cp_grid()
    n_rows, n_columns = x.shape[1], x.shape[2]
    # `shard_map` needs both sharded axes to divide the grid. The padded region
    # is its own set of rows and columns and the transition never mixes them
    # with a kept one, so the padded output is sliced away unread.
    row_pad, column_pad = (-n_rows) % rows, (-n_columns) % columns
    if row_pad or column_pad:
        x = jnp.pad(x, ((0, 0), (0, row_pad), (0, column_pad), (0, 0)))
    spec = pair_spec(x.ndim)
    dot = f"{prefix}." if prefix else ""
    # This block's own parameters, spelled the way the checkpoint spells them
    # so the body reads them under the same prefix it always did.
    block_params = (
        {name: value for name, value in params.items() if name.startswith(dot)}
        if dot
        else dict(params)
    )

    def local(x_local: jnp.ndarray, params_local: Params) -> jnp.ndarray:
        return transition(
            x_local,
            params_local,
            prefix,
            residual=residual,
            eps=eps,
            native_autocast=native_autocast,
        )

    # Parameters go in as a replicated operand rather than a closure: the
    # trunk runs inside the recycling `lax.scan`, where they are traced
    # values, and an operand keeps the whole tree on the mesh.
    out = jax.shard_map(
        local,
        mesh=mesh,
        in_specs=(spec, PartitionSpec()),
        out_specs=spec,
    )(x, block_params)
    if row_pad or column_pad:
        # Re-pinning the slice keeps the partitioner from answering the
        # narrower shape with a replicated result.
        out = shard_pair_rows(
            jax.lax.slice(
                out,
                (0, 0, 0, 0),
                (out.shape[0], n_rows, n_columns, out.shape[3]),
            )
        )
    return out


def pair_update_block(
    pair: jnp.ndarray,
    params: Params,
    prefix: str,
    *,
    mask: jnp.ndarray | None = None,
    native_autocast: bool = False,
) -> jnp.ndarray:
    """One `PairUpdateBlock`: two triangle updates then a transition.

    The residuals are sequential rather than parallel -- the incoming update
    reads the pair the outgoing one just wrote -- and the transition's own
    residual is internal. Dropout is `r=0.0` throughout the trunk and disabled
    at inference besides, so it is simply absent here.
    """
    dot = f"{prefix}." if prefix else ""
    # Under context parallelism the pair state is sharded along its rows;
    # pinning it at block entry and exit keeps the whole stack -- the main
    # trunk, the parcae coda, the lm encoder, and the confidence head's trunk
    # -- on one layout without the partitioner re-deriving it per consumer.
    pair = shard_pair_rows(pair)
    pair = pair + triangle_multiplicative(
        pair,
        params,
        f"{dot}tri_mul_out",
        outgoing=True,
        mask=mask,
        native_autocast=native_autocast,
    )
    pair = pair + triangle_multiplicative(
        pair,
        params,
        f"{dot}tri_mul_in",
        outgoing=False,
        mask=mask,
        native_autocast=native_autocast,
    )
    return shard_pair_rows(
        transition(
            pair,
            params,
            f"{dot}pair_transition",
            residual=True,
            native_autocast=native_autocast,
            # `pair` is the tensor the active layout shards, so under the grid
            # the row block is taken inside the shard instead of slicing a
            # sharded axis.
            cp_pair=True,
        )
    )


def folding_trunk(
    pair: jnp.ndarray,
    params: Params,
    prefix: str = "",
    *,
    n_layers: int,
    mask: jnp.ndarray | None = None,
    native_autocast: bool = False,
) -> jnp.ndarray:
    """`FoldingTrunk`: `n_layers` blocks in sequence, no output norm.

    Whether the result replaces the pair or is added to it is the caller's
    business and upstream disagrees with itself about it: the main loop
    overwrites, the confidence head adds. Neither is done here.
    """
    dot = f"{prefix}." if prefix else ""
    for index in range(n_layers):
        pair = pair_update_block(
            pair,
            params,
            f"{dot}blocks.{index}",
            mask=mask,
            native_autocast=native_autocast,
        )
    return pair


#: Bytes of the widened `c * d` outer product allowed live at once.
_OPM_OUTER_BUDGET_BYTES = 512 * 1024**2


def outer_product_mean(
    msa: jnp.ndarray,
    params: Params,
    prefix: str,
    *,
    msa_mask: jnp.ndarray,
    eps: float = 1e-5,
    native_autocast: bool = False,
) -> jnp.ndarray:
    """`OuterProductMean`, in the released arrangement.

    The division happens *after* the projection, so `Wout`'s bias is scaled by
    it too. The experimental module divides before, under the same key names;
    reproducing the release means this order.
    """
    dot = f"{prefix}." if prefix else ""
    if native_autocast:
        normalised = _autocast_norm(msa, params, f"{dot}norm", eps)
        projected = _autocast_linear(normalised, params, f"{dot}W")
    else:
        normalised = layer_norm(
            msa, params[f"{dot}norm.weight"], params[f"{dot}norm.bias"], eps=eps
        )
        projected = linear(normalised, params, f"{dot}W")
    # Native uses m_norm.dtype for masking: autocast Linear emits BF16 but
    # LayerNorm emits FP32, so this intermediate is promoted back to FP32.
    mask_dtype = normalised.dtype if native_autocast else projected.dtype
    projected = projected * msa_mask[..., None].astype(mask_dtype)
    half = projected.shape[-1] // 2
    a, b = projected[..., :half], projected[..., half:]

    if native_autocast:
        # Both the count matmul and outer einsum are autocast operations.
        a, b = a.astype(jnp.bfloat16), b.astype(jnp.bfloat16)
    mask = msa_mask.astype(a.dtype)
    valid = jnp.maximum(jnp.einsum("bim,bjm->bij", mask, mask)[..., None], 1.0)

    # The outer product is `[B, N, N, c, d]` and the projection immediately
    # narrows it to `[B, N, N, C_z]`, so building it whole allocates `c * d /
    # C_z` times the result -- at this model's widths, `32 * 32 / 256` = four
    # times over, and 1,965 MiB at 1,003 tokens in bfloat16, which XLA's arena
    # accounting named four of. Projecting inside the block is what keeps the
    # `c * d` tensor from ever existing at full width; it is the arrangement
    # OpenFold3's own `outer_product_mean` already uses, for the same reason.
    #
    # Rows of `i` are independent -- the contraction is over `m` -- so this is
    # exact. The division stays outside, after the projection, because that
    # order is what scales `Wout`'s bias and is the released arrangement.
    def project(rows: jnp.ndarray) -> jnp.ndarray:
        block = jnp.einsum("bimc,bjmd->bijcd", rows, b)
        block = block.reshape(block.shape[:-2] + (half * half,))
        if native_autocast:
            return _autocast_linear(block, params, f"{dot}Wout")
        return linear(block, params, f"{dot}Wout")

    tokens = a.shape[1]
    per_row = b.shape[1] * half * half * a.dtype.itemsize * a.shape[0]
    chunk = (
        max(1, _OPM_OUTER_BUDGET_BYTES // per_row)
        if per_row > 0 and per_row * tokens > _OPM_OUTER_BUDGET_BYTES
        else tokens
    )
    if chunk >= tokens:
        return project(a) / valid
    return (
        jnp.concatenate(
            [project(a[:, start : start + chunk]) for start in range(0, tokens, chunk)],
            axis=1,
        )
        / valid
    )


def msa_pair_weighted_averaging(
    msa: jnp.ndarray,
    pair: jnp.ndarray,
    params: Params,
    prefix: str,
    *,
    pair_mask: jnp.ndarray,
    eps: float = 1e-5,
    native_autocast: bool = False,
) -> jnp.ndarray:
    """`MSAPairWeightedAveraging`.

    The softmax is over the *second* token axis and the gate multiplies after
    the sum over it, not before. The masked-out bias is -1e5 rather than
    -inf: upstream fills a finite value, and a fully masked row therefore
    still produces a finite softmax.
    """
    dot = f"{prefix}." if prefix else ""
    normalised = (
        _autocast_norm(msa, params, f"{dot}norm_single", eps)
        if native_autocast
        else layer_norm(
            msa,
            params[f"{dot}norm_single.weight"],
            params[f"{dot}norm_single.bias"],
            eps=eps,
        )
    )
    bias = (
        _autocast_norm(pair, params, f"{dot}compute_bias.0", eps)
        if native_autocast
        else layer_norm(
            pair,
            params[f"{dot}compute_bias.0.weight"],
            params[f"{dot}compute_bias.0.bias"],
            eps=eps,
        )
    )
    project = _autocast_linear if native_autocast else linear
    bias = project(bias, params, f"{dot}compute_bias.1")
    bias = jnp.where(pair_mask[..., None].astype(bool), bias, -1e5)
    if native_autocast:
        from foldjax.models.esmfold2.models.native_softmax import pwa_softmax

        attention = pwa_softmax(bias.astype(jnp.float32))
    else:
        attention = jax.nn.softmax(bias, axis=-2)

    heads = bias.shape[-1]
    value = project(normalised, params, f"{dot}Wv")
    value = value.reshape(value.shape[:-1] + (heads, value.shape[-1] // heads))
    gate_logits = project(normalised, params, f"{dot}Wgate")
    gate = jax.nn.sigmoid(
        gate_logits.astype(jnp.float32) if native_autocast else gate_logits
    ).astype(gate_logits.dtype)
    gate = gate.reshape(value.shape)

    if native_autocast:
        # Native autocast narrows attention for the contraction, then rounds
        # its result before the elementwise gate (captured first PWA).
        out = jnp.einsum("bijh,bjmhd->bimhd", attention.astype(jnp.bfloat16), value)
        out = out.astype(jnp.bfloat16) * gate
    else:
        out = jnp.einsum("bijh,bjmhd,bimhd->bimhd", attention, value, gate)
    out = out.reshape(out.shape[:-2] + (heads * value.shape[-1],))
    return project(out, params, f"{dot}Wout")
