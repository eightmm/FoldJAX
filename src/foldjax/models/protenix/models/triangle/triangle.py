"""Triangle pair-update blocks for the Protenix JAX port."""

from __future__ import annotations

import os
import warnings
from collections.abc import Callable
from typing import Literal, NamedTuple

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec

from foldjax.models._cp import (
    CP_COL_AXIS,
    CP_ROW_AXIS,
    col_skew_perm,
    cp_grid,
    cp_layout,
    cp_mesh,
    cp_row_shards,
    pair_row_spec,
    pair_spec,
    permute,
    ring_perm,
    row_skew_perm,
    shard_pair_rows,
    transpose_perm,
)
from foldjax.models._cp_attention import ring_triangle_attention_2d
from foldjax.models.protenix.models.primitives.attention import AttentionParams
from foldjax.models.protenix.models.primitives.primitives import (
    LayerNormParams,
    LinearParams,
    layer_norm,
    linear,
    sigmoid,
)


def _triangle_attention_backend() -> str:
    """Return the configured triangle-attention backend.

    cuEquivariance, which is what upstream Protenix runs
    (`triangle_attention: "cuequivariance"`, configs_base.py:130).

    This defaulted to the blocked XLA path for a while, on a 490-token
    measurement: there cueq and unblocked XLA both peaked at 6,048 MiB against
    4,348 for blocked XLA, so blocking looked like the memory-safe choice. That
    does not survive a real length. At 1,531 tokens cueq is both faster and
    smaller -- 172.6 s / 21,475 MiB against 254.5 s / 24,764 MiB -- because the
    cost that grows is the `[rows, heads, N, N]` score tensor the blocked path
    writes to HBM and the fused kernel never materialises. Blocking bounds that
    tensor; it does not stop paying for it.

    The 32% is bandwidth, not arithmetic: narrowing the same blocked path to
    bfloat16 moved it 254.5 -> 255.6 s, and bfloat16 on top of cueq moves
    nothing either (172.8 s).

    There is no automatic fallback. An `auto` mode that quietly chose XLA when
    the wheel was missing would mean two machines running two different kernels
    under one default, and this repository has spent enough time on knobs that
    look set and are not: if the fused kernel cannot load, the import raises and
    says so. `PROTENIX_TRIANGLE_BACKEND=xla_jit` is the way to ask for the
    blocked path -- on a card where the fused arena does not fit, and as the
    only backend that honours a requested chunk size.
    """
    backend = os.environ.get("PROTENIX_TRIANGLE_BACKEND", "cueq_jit").lower()
    if backend not in {"xla", "xla_jit", "tokamax", "cueq", "cueq_jit"}:
        raise ValueError(f"unsupported triangle attention backend: {backend!r}")
    return backend


TriangleDirection = Literal["outgoing", "incoming"]


class TriangleMultiplicationParams(NamedTuple):
    """Parameters for ``TriangleMultiplicativeUpdate``."""

    layer_norm_in: LayerNormParams
    layer_norm_out: LayerNormParams
    linear_a_p: LinearParams
    linear_a_g: LinearParams
    linear_b_p: LinearParams
    linear_b_g: LinearParams
    linear_z: LinearParams
    linear_g: LinearParams


class TriangleAttentionParams(NamedTuple):
    """Parameters for Protenix ``TriangleAttention``."""

    layer_norm: LayerNormParams
    linear: LinearParams
    attention: AttentionParams


def triangle_multiplication(
    z: jnp.ndarray,
    mask: jnp.ndarray | None,
    params: TriangleMultiplicationParams,
    direction: TriangleDirection,
    *,
    chunk_size: int | None = None,
    use_jit: bool = False,
) -> jnp.ndarray:
    """Apply Protenix triangle multiplication without eval in-place mutation."""

    # Context parallelism changes three routing decisions at once. The cueq
    # kernel is an FFI call the SPMD partitioner cannot split, so the XLA
    # einsum runs instead (upstream Fold-CP makes the same trade: torch
    # kernels only). The blocked contraction is a `lax.scan` of dynamic
    # slices over the row axis -- the sharded axis -- which the partitioner
    # can only satisfy by gathering, so the contraction runs whole and the
    # partitioner splits the one einsum. And the inner `jit` is bypassed:
    # its trace cache does not key on the mesh, so a graph traced without
    # constraints could be replayed inside a context-parallel trace.
    cp_active = cp_mesh() is not None
    if use_jit and not cp_active:
        return _compiled_triangle_multiplication(
            z,
            mask,
            params,
            direction,
            chunk_size=chunk_size,
            use_jit=False,
        )
    if mask is None:
        mask = jnp.ones(z.shape[:-1], dtype=z.dtype)
    backend = os.environ.get("PROTENIX_TRIANGLE_MULTIPLICATION_BACKEND", "cueq").lower()
    # Upstream carries the same guard and falls back the same way: the kernel
    # requires the hidden width to equal the pair width (Protenix says so at
    # triangular.py:491). The template stack has c_z=64, c_hidden=128, so both
    # sides run the unfused path there -- that is a shape fact, not a policy.
    cueq_supported = (
        params.linear_a_p.weight.shape[0] == z.shape[-1] and z.shape[-1] % 32 == 0
    )
    if backend == "cueq" and cueq_supported and not cp_active:
        if chunk_size is not None and 0 < chunk_size < z.shape[-3]:
            _warn_unchunkable_multiplication(
                chunk_size,
                rows=z.shape[-3],
                cols=z.shape[-2],
                c_hidden=params.linear_a_p.weight.shape[0],
                itemsize=z.dtype.itemsize,
            )
        from foldjax.models.protenix.models.triangle.triangle_cueq import (
            cueq_triangle_multiplication,
        )

        return cueq_triangle_multiplication(z, mask, params, direction)
    if backend not in {"cueq", "xla"}:
        raise ValueError(f"unsupported triangle multiplication backend: {backend!r}")
    mask = mask.astype(z.dtype)[..., None]
    # Same reasoning as the attention: the block belongs to the tensor, not to
    # the token count. One row of `a` costs `N * c_hidden * itemsize`, so the
    # chunk policy's 256 means 373 MiB per block on OpenDDE's structural branch
    # and a fifth of that on a narrower trunk. Swept there against the policy's
    # value, the peak went 9,809 -> 9,652 -> 9,628 MiB at 256 -> 128 -> 64 rows,
    # against a 9 MiB run-to-run spread.
    chunk_size = (
        None
        if cp_active
        else _row_block(
            rows=z.shape[-3],
            per_row=(
                z.shape[-2] * params.linear_a_p.weight.shape[0] * z.dtype.itemsize
            ),
            requested=chunk_size,
            budget=_PROJECTION_BUDGET_BYTES,
        )
    )

    z_norm = layer_norm(z, params.layer_norm_in)
    if cp_layout() == "2d":
        # Fold-CP's own schedule: both pair axes are sharded, so the
        # contraction runs as Cannon's algorithm -- skew, then one local
        # matmul per ring hop. Nothing full-width is ever built.
        mask_c = mask if mask.shape == z_norm.shape[:-1] + (1,) else mask
        a = _project_pair(z_norm, mask_c, params.linear_a_g, params.linear_a_p)
        b = _project_pair(z_norm, mask_c, params.linear_b_g, params.linear_b_p)
        out = _cannon_contract(a, b, direction).astype(z.dtype)
        out = layer_norm(out, params.layer_norm_out)
        out = linear(out, params.linear_z)
        return out * sigmoid(linear(z_norm, params.linear_g))

    contract_z = z_norm
    contract_mask = mask
    contract_direction = direction
    if cp_active and direction == "incoming":
        # The incoming contraction sums over the *sharded* axis, and the
        # partitioner realises that as a full-size float32 partial sum plus an
        # all-reduce on every device -- 49 GiB per device at 3,012 residues,
        # measured as the allocation that killed the first cluster run. The
        # projections are elementwise, so projecting the transpose and
        # contracting in the outgoing form is the same arithmetic
        # (out[i,j] = sum_k a[k,i] b[k,j] either way) with the partials
        # sharded like every other pair tensor. The gate below still reads
        # the untransposed `z_norm`.
        contract_z = shard_pair_rows(jnp.swapaxes(z_norm, -2, -3))
        contract_mask = jnp.swapaxes(mask, -3, -2)
        contract_direction = "outgoing"
    b = contract_mask * sigmoid(linear(contract_z, params.linear_b_g))
    b = b * linear(contract_z, params.linear_b_p)

    def project_a(z_slice: jnp.ndarray, mask_slice: jnp.ndarray) -> jnp.ndarray:
        gated = mask_slice * sigmoid(linear(z_slice, params.linear_a_g))
        return gated * linear(z_slice, params.linear_a_p)

    # Keep the projections in the trunk's own dtype and accumulate in float32
    # instead of widening them first. `a` and `b` are each [N, N, c_hidden] --
    # 1,311 MiB apiece on OpenDDE's structural branch -- so casting them up
    # meant a bfloat16 trunk still paid float32 for its two largest buffers.
    # `preferred_element_type` gives the same float32 accumulation without the
    # widened operands, which is what AlphaFold 3's BF16_BF16_F32 algorithm
    # does. With a float32 trunk both forms are identical.
    #
    # `a` is also built one block at a time when the contraction is blocked.
    # Each output row block reduces over the whole of `b` but only its own rows
    # of `a`, so the rest of `a` never has to exist -- the same trade the
    # attention makes with its queries.
    out = _triangle_contract(
        project_a,
        contract_z,
        contract_mask,
        b,
        contract_direction,
        chunk_size,
        z.dtype,
    )
    # Pin the contraction result to the pair layout before the epilogue; the
    # partitioner otherwise sometimes materialises it replicated.
    out = shard_pair_rows(out)
    out = out.astype(z.dtype)
    out = layer_norm(out, params.layer_norm_out)
    out = linear(out, params.linear_z)
    return out * sigmoid(linear(z_norm, params.linear_g))


_compiled_triangle_multiplication = jax.jit(
    triangle_multiplication,
    static_argnames=("direction", "chunk_size", "use_jit"),
)


def _project_pair(
    z_norm: jnp.ndarray,
    mask: jnp.ndarray,
    gate_params: LinearParams,
    value_params: LinearParams,
) -> jnp.ndarray:
    """One gated pair projection, elementwise and therefore shard-local."""
    return mask * sigmoid(linear(z_norm, gate_params)) * linear(z_norm, value_params)


def _cannon_contract(
    a: jnp.ndarray,
    b: jnp.ndarray,
    direction: TriangleDirection,
) -> jnp.ndarray:
    """Contract two 2-D-sharded pair projections by Cannon's algorithm.

    The triangle contraction is a per-channel matrix product:
    ``outgoing`` is ``out[i,j] = sum_k a[i,k] b[j,k]`` and ``incoming`` is
    ``out[i,j] = sum_k a[k,i] b[k,j]``. Both become ``A @ B`` once the
    operand whose contracted axis sits on the wrong grid axis is transposed
    across the grid -- ``b`` for outgoing, ``a`` for incoming -- which is why
    the two directions differ by one flag rather than by a rewrite.

    Cannon then aligns the operands so every device starts on matching
    indices: the left operand is skewed along grid columns by its row
    coordinate, the right along grid rows by its column coordinate. Each of
    the ``P`` steps multiplies the two resident tiles and shifts both one hop,
    so the per-device transient is two tiles -- ``O((N/P)^2 C)`` -- and no
    all-gather appears anywhere.
    """
    mesh = cp_mesh()
    side = cp_grid()[0]
    if a.ndim != 3:
        raise ValueError(
            "the 2-D context-parallel triangle multiplication takes unbatched "
            f"[N, N, C] operands, got rank {a.ndim}"
        )
    n = a.shape[0]
    pad = (-n) % side
    if pad:
        # Padded rows and columns project to zero (their mask is zero), so a
        # padded k block contributes nothing to the sum and the padded output
        # region is sliced away.
        widths = ((0, pad), (0, pad), (0, 0))
        a = jnp.pad(a, widths)
        b = jnp.pad(b, widths)
    spec = pair_spec(3)

    # A grid permutation moves whole tiles; it never permutes the axes *inside*
    # a tile. So after the transpose and the skews the resident operands still
    # carry their dense index meaning -- outgoing holds `a[i, k]` against
    # `b[j, k]`, incoming `a[k, i]` against `b[k, j]` -- and the local
    # contraction stays the dense einsum of that direction. Reading the moved
    # `b` as if it were `[k, j]` is the error this comment exists to prevent:
    # it type-checks, it runs, and it silently computes a different function.
    equations = {"outgoing": "ikd,jkd->ijd", "incoming": "kid,kjd->ijd"}
    if direction not in equations:
        raise ValueError(f"unsupported triangle direction: {direction!r}")
    equation = equations[direction]

    def body(lhs, rhs):
        if direction == "outgoing":
            rhs = permute(rhs, transpose_perm(side))
        else:
            lhs = permute(lhs, transpose_perm(side))
        lhs = permute(lhs, row_skew_perm(side))
        rhs = permute(rhs, col_skew_perm(side))
        rows = lhs.shape[1] if direction == "incoming" else lhs.shape[0]
        cols = rhs.shape[1] if direction == "incoming" else rhs.shape[0]
        total = jnp.zeros((rows, cols) + lhs.shape[2:], dtype=jnp.float32)
        correction = jnp.zeros_like(total)
        for step in range(side):
            partial = jnp.einsum(equation, lhs, rhs, preferred_element_type=jnp.float32)
            updated = total + partial
            residual = jnp.where(
                jnp.abs(total) >= jnp.abs(partial),
                (total - updated) + partial,
                (partial - updated) + total,
            )
            total = updated
            correction = correction + residual
            if step + 1 < side:
                lhs = permute(lhs, ring_perm(side, axis=CP_COL_AXIS, delta=-1))
                rhs = permute(rhs, ring_perm(side, axis=CP_ROW_AXIS, delta=-1))
        return total + correction

    out = jax.shard_map(body, mesh=mesh, in_specs=(spec, spec), out_specs=spec)(a, b)
    if pad:
        out = out[:n, :n]
    return shard_pair_rows(out)


def _triangle_contract(
    project_a: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    z_norm: jnp.ndarray,
    mask: jnp.ndarray,
    b: jnp.ndarray,
    direction: TriangleDirection,
    chunk_size: int | None,
    out_dtype: jnp.dtype,
) -> jnp.ndarray:
    """Contract ``a`` against ``b``, building ``a`` whole or one block at a time.

    ``project_a`` takes a slice of the normalised pair representation and its
    matching mask slice and returns that slice of ``a``. Blocking the output
    rows only ever needs the matching rows of ``a``, so when the contraction is
    blocked the full projection is never materialised.

    ``out_dtype`` is the caller's ``z.dtype``. The blocked destination is held
    at that width, not at float32.
    """
    n = z_norm.shape[-3]
    if chunk_size is None or chunk_size <= 0 or chunk_size >= n:
        return _triangle_contract_block(project_a(z_norm, mask), b, direction)

    # The destination is held at the trunk's own width, not at float32. Each
    # einsum still accumulates in float32 (`preferred_element_type`), and each
    # output element is written exactly once -- a row block reduces over the
    # whole of `b`, so no element is ever revisited by a later block. The
    # narrowing rounding is therefore the same single rounding the caller
    # performs on the next line (`out.astype(z.dtype)`), applied at the same
    # point in the arithmetic. This comment used to claim the opposite, that a
    # narrow destination would round "every block on the way in" -- true, but
    # it is the rounding that was going to happen anyway.
    #
    # What it is worth, measured on Protenix at 3,012 tokens, where the
    # template stack runs this path because c_z=64 != c_hidden=128 keeps it off
    # the fused kernel. The destination went f32[128, 3012, 3040] 4,471 MiB ->
    # bf16 2,235 MiB, exactly half, with every other entry in the arena cover
    # byte-identical and no new buffer taking its place. The temp arena went
    # 39.102 -> 37.367 GiB.
    #
    # Note those two do not agree: the buffer shrank 2,236 MiB and the arena
    # only 1,777, so **21% of the freed space was reabsorbed by repacking**.
    # Neither all-or-nothing prediction was right, and a width change is worth
    # measuring at the arena rather than at the buffer.
    #
    # With a float32 trunk `out_dtype` is float32 and nothing here changes,
    # which is the whole story on OpenDDE: its CLI default is fp32, faithful to
    # upstream's own fp32 storage, so this is live there only on the opt-in
    # bfloat16 path. Protenix defaults to bf16 and gets it by default.
    axis = -3 if direction == "outgoing" else -2
    # A real `lax.scan` over the row blocks, not an unrolled Python loop. The
    # unrolled form -- whether it concatenated the block outputs or wrote them
    # into a preallocated destination -- left XLA free to merge the per-block
    # `project_a` gemms horizontally: same weights, slices of the same tensor,
    # so it reassembled the operand and ran whole-width gemms. At 3,012 tokens
    # the template stack's a-side projections came back as four simultaneous
    # bf16[128, 4*N^2] operands, 34.6 GiB of temp arena, and the chunking
    # bought nothing (verified twice on the compiled program's buffer
    # assignment, once per form). Iterations of one loop body cannot be
    # merged, so inside a scan the projection exists one block at a time by
    # construction. The row axis is padded to a whole number of blocks (mask
    # is zero-padded, so the padded rows project to zeros) and the result is
    # sliced back.
    pad = (-n) % chunk_size
    z_blocked = z_norm
    mask_blocked = mask
    if pad:
        widths = [(0, 0)] * z_norm.ndim
        widths[axis % z_norm.ndim] = (0, pad)
        z_blocked = jnp.pad(z_norm, widths)
        mask_blocked = jnp.pad(mask, widths)

    def body(out: jnp.ndarray, start: jnp.ndarray):
        a_block = project_a(
            jax.lax.dynamic_slice_in_dim(z_blocked, start, chunk_size, axis=axis),
            jax.lax.dynamic_slice_in_dim(mask_blocked, start, chunk_size, axis=axis),
        )
        block = _triangle_contract_block(a_block, b, direction).astype(out_dtype)
        return (
            jax.lax.dynamic_update_slice_in_dim(out, block, start, axis=-3),
            None,
        )

    shape = list(z_norm.shape)
    shape[-3] = n + pad
    shape[-1] = b.shape[-1]
    out, _ = jax.lax.scan(
        body,
        jnp.zeros(shape, dtype=out_dtype),
        jnp.arange(0, n + pad, chunk_size),
    )
    if pad:
        out = jax.lax.slice_in_dim(out, 0, n, axis=-3)
    return out


def _triangle_contract_block(
    a: jnp.ndarray,
    b: jnp.ndarray,
    direction: TriangleDirection,
) -> jnp.ndarray:
    # NOTE: writing this as permute_final_dims+matmul (to mirror torch's GEMM)
    # is a no-op in JAX — XLA lowers both einsum and matmul to the same
    # dot_general, so the reduction order is XLA's choice regardless. The
    # einsum form is kept for clarity.
    if direction == "outgoing":
        return jnp.einsum(
            "...ikd,...jkd->...ijd", a, b, preferred_element_type=jnp.float32
        )
    if direction == "incoming":
        return jnp.einsum(
            "...kid,...kjd->...ijd", a, b, preferred_element_type=jnp.float32
        )
    raise ValueError(f"unsupported triangle direction: {direction!r}")


_WARNED_UNCHUNKABLE = False


def _warn_unchunkable_backend(q_chunk_size: int) -> None:
    """Say that a requested chunk size is not used, once.

    The cuEquivariance kernel takes the whole row axis, so a chunk size does not
    reach it. That is not a loss: the kernel never forms the
    ``[rows, heads, N, N]`` score tensor the chunk size exists to bound, so there
    is nothing left to block.

    This used to say the opposite -- that cuEquivariance "does not make up for it
    by fusing the score tensor away" -- on a 490-token measurement where cueq and
    the unblocked XLA path peaked identically (6,048 and 6,049 MiB) against 4,348
    blocked. That size is too small to separate the two: the score tensor is not
    yet what dominates. At 1531 tokens cueq is both faster and smaller (172.6 s /
    21,475 MiB against 254.5 s / 24,764 MiB), which is the measurement the module
    docstring records and the reason cueq is the default. The warning text was
    left behind when the default changed.

    Not an exception, because the automatic chunk policy legitimately emits a
    size without knowing which backend will run.
    """
    global _WARNED_UNCHUNKABLE
    if _WARNED_UNCHUNKABLE:
        return
    _WARNED_UNCHUNKABLE = True
    warnings.warn(
        f"triangle attention q_chunk_size={q_chunk_size} is not used by the "
        "'cueq' backend, which never forms the score tensor the chunk size "
        "exists to bound. Nothing is lost; set PROTENIX_TRIANGLE_BACKEND=xla_jit "
        "to get the blocked path that honours it.",
        RuntimeWarning,
        stacklevel=2,
    )


_WARNED_UNCHUNKABLE_MUL = False


def _warn_unchunkable_multiplication(
    chunk_size: int,
    *,
    rows: int,
    cols: int,
    c_hidden: int,
    itemsize: int,
) -> None:
    """Say that a requested chunk size is not used, once -- and what it costs.

    The cuEquivariance kernel has no parameter of this family to accept:
    ``cuequivariance_jax`` takes ``(x, direction, key, mask, weights, biases,
    eps, precision, fallback)`` and nothing else, so this is not a call site
    dropping an argument. There is nowhere to put it.

    Deliberately *not* the sentence the attention path uses. There the knob is
    moot and nothing is lost, because the fused kernel never forms the score
    tensor the chunk size exists to bound. Here the fused kernel does
    materialise both full ``[N, N, c_hidden]`` projections -- 5,299 MiB apiece
    on OpenDDE's structural branch at 1,003 tokens, 53.5% of that arena -- which
    is exactly what a chunk size looks like it would bound. A knob that is inert
    next to a buffer it appears to promise to bound is worse than a knob that is
    inert next to nothing.

    What the escape is worth, measured, and it changed sign once the blocked
    destination stopped being float32. Per pair element, with ``s`` the width of
    ``z`` and ``Npad/N`` the row padding::

        fused projections   2*(2c)*s        -- 768 concatenated channels
        blocked accumulator 2*c*s*(Npad/N)  -- only the 384 output channels
        blocked input copy  1*c*s*(Npad/N)
        ratio = 0.75 * (Npad/N) ~= 0.76

    So the blocked path is about 24% cheaper **at any dtype**, and the two
    looked equal only while the destination was float32 against a bfloat16
    fused path -- 4 bytes against 2, cancelling the 0.75 exactly. Measured on
    OpenDDE, two passes per arm: at 1,003 tokens bf16, peak 25.185 -> 19.944
    GiB across that change, putting the blocked arena at 17,147 MiB against
    fused's 19,797. The identity predicts the gap as one ``c``-wide pair tensor,
    2,650 MiB; measured 2,650.

    Which is why this says to measure rather than naming a winner: the fused
    path is still the default, the memory evidence now favours the escape, and
    the time evidence does not resolve -- 166.2 s fused against 190.6 and 145.2
    s blocked, a spread that straddles the comparison on a card that swings that
    far under throttle.

    **The 24% is the ratio for these two projections, not for a run.** The
    warning below used to say "about 24% cheaper on memory at every size tried",
    which reads as a whole-peak claim and is not one. Measured end to end on
    Protenix at 4,100 tokens, one warm pass per arm: peak 77,158.2 MiB fused
    against 71,360.6 blocked, which is **7.5%** -- the projections are only part
    of what sets that peak. The same pair resolves the time question in the
    direction the escape loses: 2,329 s fused against 2,715 blocked, **+17%**.
    It bought 6.6 points of pool occupancy (88.1% to 81.5%) and did not rescue
    any of the three models that fail at that size.
    """
    global _WARNED_UNCHUNKABLE_MUL
    if _WARNED_UNCHUNKABLE_MUL:
        return
    _WARNED_UNCHUNKABLE_MUL = True
    projections_mib = 2 * rows * cols * c_hidden * itemsize // 1024**2
    warnings.warn(
        f"triangle multiplication chunk_size={chunk_size} is not used by the "
        "'cueq' backend, which takes the whole row axis and has no parameter "
        "for one. Unlike the attention path this is not free: the fused kernel "
        f"materialises both full projections ({projections_mib:,} MiB here), "
        "which is what the chunk size looks like it would bound. Set "
        "PROTENIX_TRIANGLE_MULTIPLICATION_BACKEND=xla for the blocked path "
        "that honours it: about 24% cheaper on these projections, which was "
        "7.5% of the whole-process peak on Protenix at 4,100 tokens, for 17% "
        "more wall time. It is a trade, not a saving.",
        RuntimeWarning,
        stacklevel=2,
    )


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
    """Apply Protenix triangle attention in the dense XLA path."""

    backend = (
        _triangle_attention_backend()
        if attention_backend is None
        else attention_backend
    )
    if cp_mesh() is not None:
        # The SPMD partitioner cannot split an FFI call -- but triangle
        # attention is a batch of independent per-row attentions, and inside
        # `shard_map` each device holds whole rows, so the cueq kernel can run
        # unchanged on the local shard (it takes q/k/v for a block of rows
        # plus the full bias, which is exactly what the body has). The
        # inner-jit route stays bypassed because its trace cache does not key
        # on the mesh. Tokamax remains unvalidated under a mesh.
        local_backend = backend.removesuffix("_jit")
        if local_backend not in {"xla", "cueq"}:
            raise ValueError(
                "context-parallel triangle attention supports the XLA and "
                f"cueq backends; got {backend!r}."
            )
        return _triangle_attention_cp(
            x,
            mask,
            params,
            num_heads=num_heads,
            starting=starting,
            inf=inf,
            q_chunk_size=q_chunk_size,
            local_backend=local_backend,
        )
    if backend in {"xla_jit", "cueq_jit"}:
        return _compiled_triangle_attention(
            x,
            mask,
            params,
            num_heads=num_heads,
            starting=starting,
            inf=inf,
            q_chunk_size=q_chunk_size,
            attention_backend=backend.removesuffix("_jit"),
        )
    if backend not in {"xla", "tokamax", "cueq"}:
        raise ValueError(f"unsupported triangle attention backend: {backend!r}")
    if mask is None:
        mask = jnp.ones(x.shape[:-1], dtype=x.dtype)
    if not starting:
        x = jnp.swapaxes(x, -2, -3)
        mask = jnp.swapaxes(mask, -1, -2)

    x = layer_norm(x, params.layer_norm)
    mask_bias = inf * (mask.astype(jnp.float32) - 1.0)
    mask_bias = mask_bias[..., :, None, None, :]
    triangle_bias = linear(x, params.linear)
    triangle_bias = jnp.moveaxis(triangle_bias, -1, -3)
    triangle_bias = jnp.expand_dims(triangle_bias, axis=-4)

    out = _triangle_attention_dense(
        x,
        params,
        num_heads,
        mask_bias,
        triangle_bias,
        q_chunk_size,
        backend,
    )
    if not starting:
        out = jnp.swapaxes(out, -2, -3)
    return out


_compiled_triangle_attention = jax.jit(
    triangle_attention,
    static_argnames=(
        "num_heads",
        "starting",
        "inf",
        "q_chunk_size",
        "attention_backend",
    ),
)


def _triangle_attention_ring_2d(
    x: jnp.ndarray,
    params: TriangleAttentionParams,
    num_heads: int,
    mask_bias: jnp.ndarray,
    triangle_bias: jnp.ndarray,
) -> jnp.ndarray:
    """Two-dimensional Protenix attention without a full-column gather."""

    scale = float(params.attention.linear_k.weight.shape[0] // num_heads) ** -0.5
    q = _project_heads(x, params.attention.linear_q, num_heads)
    k = _project_heads(x, params.attention.linear_k, num_heads)
    v = _project_heads(x, params.attention.linear_v, num_heads)
    q = q * jnp.asarray(scale, dtype=q.dtype)
    out = ring_triangle_attention_2d(
        q,
        k,
        v,
        triangle_bias,
        mask_bias,
    )
    out = jnp.swapaxes(out, -2, -3)
    if params.attention.linear_g is not None:
        gate = sigmoid(linear(x, params.attention.linear_g))
        gate = gate.reshape(gate.shape[:-1] + (num_heads, -1))
        out = out * gate
    out = out.reshape(out.shape[:-2] + (-1,))
    return linear(out, params.attention.linear_o)


def _triangle_attention_cp(
    x: jnp.ndarray,
    mask: jnp.ndarray | None,
    params: TriangleAttentionParams,
    *,
    num_heads: int,
    starting: bool,
    inf: float,
    q_chunk_size: int | None,
    local_backend: str = "xla",
) -> jnp.ndarray:
    """Triangle attention with the row axis sharded across the ``cp`` mesh.

    Each row of the pair representation is an independent attention, so a row
    shard needs no other shard's ``q``/``k``/``v`` -- the one cross-row input
    is the triangle bias, whose ``[j, k]`` indices read every row. The bias is
    therefore projected from the sharded tensor (its cheapest form: ``heads``
    channels, not ``C``) and handed to every shard whole, and the blocked row
    loop then runs unchanged on each device's local rows. This is the one
    routing the SPMD partitioner cannot derive on its own: the row loop
    slices the sharded axis, which a constraint-only program could satisfy
    only by gathering the whole pair tensor back.
    """
    mesh = cp_mesh()
    if x.ndim != 3:
        raise ValueError(
            "context-parallel triangle attention supports unbatched "
            f"[N, N, C] inputs, got rank {x.ndim}"
        )
    if mask is None:
        mask = jnp.ones(x.shape[:-1], dtype=x.dtype)
    if not starting:
        x = jnp.swapaxes(x, -2, -3)
        mask = jnp.swapaxes(mask, -1, -2)
    # The reshard after the transpose is the Fold-CP "transpose exchange":
    # ending-node attention runs on columns, so the columns become the
    # sharded rows.
    x = shard_pair_rows(x, row_axis=-3)

    x = layer_norm(x, params.layer_norm)
    mask_bias = inf * (mask.astype(jnp.float32) - 1.0)
    mask_bias = mask_bias[..., :, None, None, :]
    triangle_bias = linear(x, params.linear)
    triangle_bias = jnp.moveaxis(triangle_bias, -1, -3)
    triangle_bias = jnp.expand_dims(triangle_bias, axis=-4)

    if cp_layout() == "2d":
        out = _triangle_attention_ring_2d(
            x,
            params,
            num_heads,
            mask_bias,
            triangle_bias,
        )
        if not starting:
            out = jnp.swapaxes(out, -2, -3)
        return shard_pair_rows(out, row_axis=-3)

    # `shard_map` requires the sharded axis to divide evenly; pad the rows and
    # slice them back. Padded rows attend over real columns and are discarded,
    # so their values never reach a kept output.
    n_rows = x.shape[-3]
    pad = (-n_rows) % cp_row_shards()
    if pad:
        x = jnp.pad(x, ((0, pad), (0, 0), (0, 0)))
        mask_bias = jnp.pad(mask_bias, ((0, pad), (0, 0), (0, 0), (0, 0)))

    def local_rows(x_l, mask_bias_l, bias_l, params_l):
        return _triangle_attention_dense(
            x_l,
            params_l,
            num_heads,
            mask_bias_l,
            bias_l,
            q_chunk_size,
            local_backend,
        )

    # The specs name the mesh's *row* axis rather than a literal, because the
    # 2-D layout calls it something else. Triangle attention stays row-sharded
    # under either layout -- its softmax spans whole columns -- so the column
    # axis stays unmanaged here and the partitioner supplies whole rows.
    row_spec = pair_row_spec(1, row_axis=0)
    out = jax.shard_map(
        local_rows,
        mesh=mesh,
        in_specs=(row_spec, row_spec, PartitionSpec(), PartitionSpec()),
        out_specs=row_spec,
    )(x, mask_bias, triangle_bias, params)
    if pad:
        # Slicing the padded rows away would otherwise let the partitioner
        # fall back to a replicated result -- one whole pair tensor per
        # device, transiently, after every block.
        out = shard_pair_rows(
            jax.lax.slice_in_dim(out, 0, n_rows, axis=-3), row_axis=-3
        )
    if not starting:
        out = jnp.swapaxes(out, -2, -3)
    return out


# One row of the score tensor costs `heads * N * N * 4` bytes -- the logits are
# float32 whatever the trunk dtype is, because the mask bias is. Blocking rows
# is what bounds it, and both branches were swept independently, at two sizes
# each:
#
#   protenix, 4 heads:   64 rows optimal at 490 tokens and again at 976
#   opendde, 12 heads:   ~25 rows optimal at 948 tokens and again at 1,892
#
# The optimum is the same row count at both sizes for a given branch, so it is
# not a byte budget -- a budget halves the count every time the tokens double,
# and at 1,892 tokens that meant 6 rows, which measured 43,569 MiB against
# 32,641 at 25. What is invariant is `rows * heads`: 64*4 = 256, 25*12 = 300.
# Too few rows pays per-block overhead across hundreds of blocks; too many lets
# the score tensor take over.
#
# Bytes stay on as a ceiling, because a fixed row count grows the score as N^2
# and would run away on a very large complex.
_SCORE_ROWS_TIMES_HEADS = 288
_SCORE_CEILING_BYTES = 8 * 1024**3
_PROJECTION_BUDGET_BYTES = 128 * 1024**2
_MAX_ROWS_PER_BLOCK = 64
_MIN_ROWS_PER_BLOCK = 8


def _row_block(
    *,
    rows: int,
    per_row: int,
    requested: int | None,
    cap: int = _MAX_ROWS_PER_BLOCK,
    budget: int = _SCORE_CEILING_BYTES,
) -> int | None:
    """Rows per block: the caller's request, narrowed to what the budget allows."""
    if per_row <= 0 or rows < 2:
        return requested
    allowed = max(_MIN_ROWS_PER_BLOCK, min(cap, budget // per_row))
    if allowed >= rows:
        return requested
    return allowed if requested is None else min(requested, allowed)


def _score_rows(*, rows: int, cols: int, num_heads: int, requested: int | None):
    """Row block for a triangle attention, from its own head count and width."""
    return _row_block(
        rows=rows,
        per_row=num_heads * cols * cols * 4,
        requested=requested,
        cap=min(_MAX_ROWS_PER_BLOCK, max(1, _SCORE_ROWS_TIMES_HEADS // num_heads)),
    )


def _triangle_attention_dense(
    x: jnp.ndarray,
    params: TriangleAttentionParams,
    num_heads: int,
    mask_bias: jnp.ndarray,
    triangle_bias: jnp.ndarray,
    q_chunk_size: int | None,
    attention_backend: str,
) -> jnp.ndarray:
    # Triangle attention is a batch of independent attentions, one per row of
    # the pair representation: `q`, `k` and `v` are [N_row, heads, N_col, d] and
    # nothing in the softmax crosses a row. Blocking that row axis therefore
    # bounds all three projections *and* the [N_row, heads, N_col, N_col] score
    # tensor at once, for the same block count, the same score budget and the
    # same arithmetic -- each row is projected exactly once either way.
    #
    # Blocking the query axis instead, which is what this did, bounds only the
    # scores and `q`; `k` and `v` stay whole, at 1,311 MiB apiece on OpenDDE's
    # structural branch.
    if attention_backend not in {"cueq", "tokamax"}:
        q_chunk_size = _score_rows(
            rows=x.shape[-3],
            cols=x.shape[-2],
            num_heads=num_heads,
            requested=q_chunk_size,
        )
    chunked = (
        q_chunk_size is not None
        and attention_backend not in {"cueq", "tokamax"}
        and 0 < q_chunk_size < x.shape[-3]
    )
    scale = float(params.attention.linear_k.weight.shape[0] // num_heads) ** -0.5
    if chunked:
        q = k = v = None
    else:
        q = _project_heads(x, params.attention.linear_q, num_heads)
        k = _project_heads(x, params.attention.linear_k, num_heads)
        v = _project_heads(x, params.attention.linear_v, num_heads)

    if attention_backend == "cueq" and x.shape[-2] > 16:
        # The cuEquivariance kernel is fused and takes the whole thing, so a
        # chunk size does not reach it. Nothing is materialised that a chunk
        # size would have bounded; say so once rather than let a knob look
        # like it worked.
        if q_chunk_size is not None and 0 < q_chunk_size < x.shape[-2]:
            _warn_unchunkable_backend(q_chunk_size)
        from foldjax.models.protenix.models.triangle.triangle_cueq import (
            cueq_attention_core,
        )

        out = cueq_attention_core(
            q,
            k,
            v,
            triangle_bias,
            mask_bias,
            scale=scale,
        )
    elif q is not None:
        # The chunked path scales each block as it is projected instead.
        q = q * jnp.asarray(scale, dtype=q.dtype)

    if attention_backend == "tokamax":
        # q/k/v are [..., N_row, h, j, d], triangle_bias [..., 1, h, N, N],
        # mask_bias [..., N, 1, 1, N] — already the layout tokamax expects; q is
        # pre-scaled (scale=1.0 inside). Returns [..., h, i, d] like the block.
        from foldjax.models.protenix.models.triangle.triangle_attention_tokamax import (
            tokamax_attention_core,
        )

        out = tokamax_attention_core(q, k, v, triangle_bias, mask_bias)
    elif attention_backend == "cueq" and x.shape[-2] > 16:
        pass
    elif not chunked:
        out = _triangle_attention_block(q, k, v, mask_bias, triangle_bias)
    else:
        rows = x.shape[-3]
        # Written into one preallocated buffer rather than concatenated at the
        # end: at 1,892 structural tokens the row budget picks blocks of six,
        # and holding all 316 results live to concatenate them costs a second
        # copy of the [N_row, heads, N_col, d] output -- 5,244 MiB.
        head_dim = params.attention.linear_q.weight.shape[0] // num_heads
        out = jnp.zeros(
            x.shape[:-3] + (rows, num_heads, x.shape[-2], head_dim),
            dtype=x.dtype,
        )
        for start in range(0, rows, q_chunk_size):
            size = min(q_chunk_size, rows - start)
            # One block of rows, projected straight from `x`. Slicing the row
            # axis and then projecting is the same arithmetic as projecting and
            # then slicing -- the linear contracts over the channel axis -- and
            # each row is still projected exactly once across the whole loop.
            x_rows = jax.lax.dynamic_slice_in_dim(x, start, size, axis=-3)
            q_block = _project_heads(x_rows, params.attention.linear_q, num_heads)
            q_block = q_block * jnp.asarray(scale, dtype=q_block.dtype)
            block = _triangle_attention_block(
                q_block,
                _project_heads(x_rows, params.attention.linear_k, num_heads),
                _project_heads(x_rows, params.attention.linear_v, num_heads),
                # mask_bias is [..., N_row, 1, 1, N_col]; the row axis is -4.
                # triangle_bias is [..., 1, heads, N, N] -- already broadcast
                # over rows, so it is passed through whole.
                jax.lax.dynamic_slice_in_dim(mask_bias, start, size, axis=-4),
                triangle_bias,
            )
            out = out.at[..., start : start + size, :, :, :].set(
                block.astype(out.dtype)
            )

    out = jnp.swapaxes(out, -2, -3)
    if params.attention.linear_g is not None:
        gate = sigmoid(linear(x, params.attention.linear_g))
        gate = gate.reshape(gate.shape[:-1] + (num_heads, -1))
        out = out * gate
    out = out.reshape(out.shape[:-2] + (-1,))
    return linear(out, params.attention.linear_o)


def _triangle_attention_block(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    mask_bias: jnp.ndarray,
    triangle_bias: jnp.ndarray,
) -> jnp.ndarray:
    logits = jnp.einsum("...hid,...hjd->...hij", q, k)
    logits = logits + mask_bias + triangle_bias
    probs = jax.nn.softmax(logits.astype(jnp.float32), axis=-1).astype(v.dtype)
    return jnp.einsum("...hij,...hjd->...hid", probs, v)


def _project_heads(
    x: jnp.ndarray,
    params: LinearParams,
    num_heads: int,
) -> jnp.ndarray:
    y = linear(x, params)
    y = y.reshape(y.shape[:-1] + (num_heads, -1))
    return jnp.swapaxes(y, -2, -3)
