"""Gather-free two-dimensional Fold-CP attention primitives.

The pair representation is tiled over a square ``cp_row x cp_col`` mesh.
Queries stay resident while key, value, mask and pair-bias tiles rotate through
a ring. An fp32 online-softmax accumulator makes the result mathematically
equivalent to dense attention without materialising a full token axis.

The ring runs one block of local pair rows at a time, in a ``lax.scan`` around
the whole rotation. Every pair row is an independent attention, so a block
needs nothing but its own rows of ``q``/``k``/``v`` -- which
:func:`ring_triangle_attention_2d_from_pair` projects inside the block -- and
its score tile and accumulators are the block's width, not the tile's. See
:func:`resolve_ring_row_block`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec

from foldjax.models._cp import (
    CP_COL_AXIS,
    CP_ROW_AXIS,
    cp_grid,
    cp_layout,
    cp_mesh,
    permute,
)


def _flat(
    pairs: Sequence[tuple[tuple[int, int], tuple[int, int]]],
    side: int,
) -> list[tuple[int, int]]:
    return [(a * side + b, c * side + d) for (a, b), (c, d) in pairs]


def triangle_bias_stage0_perm(side: int) -> list[tuple[int, int]]:
    """Flatten lower diagonals onto rows: ``(r, c) -> (r-c, c)``."""

    return _flat(
        [
            ((row, col), ((row - col) % side, col))
            for row in range(side)
            for col in range(side)
        ],
        side,
    )


def triangle_bias_stage1_perm(side: int) -> list[tuple[int, int]]:
    """Rotate flattened diagonals: ``(r, c) -> (r, c+r)``."""

    return _flat(
        [
            ((row, col), (row, (col + row) % side))
            for row in range(side)
            for col in range(side)
        ],
        side,
    )


def triangle_kv_initial_perm(side: int) -> list[tuple[int, int]]:
    """Offset K/V tiles onto the redistributed bias diagonal."""

    return triangle_bias_stage1_perm(side)


def triangle_kv_ring_perm(side: int) -> list[tuple[int, int]]:
    """Advance K/V/mask one key tile around each grid row."""

    return _flat(
        [
            ((row, col), (row, (col + 1) % side))
            for row in range(side)
            for col in range(side)
        ],
        side,
    )


def triangle_bias_ring_perm(side: int) -> list[tuple[int, int]]:
    """Advance bias one matching key tile up each grid column."""

    return _flat(
        [
            ((row, col), ((row - 1) % side, col))
            for row in range(side)
            for col in range(side)
        ],
        side,
    )


def _resolve_axis(axis: int, ndim: int, *, name: str) -> int:
    resolved = axis + ndim if axis < 0 else axis
    if not 0 <= resolved < ndim:
        raise ValueError(f"{name} {axis} is out of range for rank {ndim}")
    return resolved


def _two_axis_spec(
    ndim: int,
    row_axis: int,
    col_axis: int,
) -> PartitionSpec:
    row = _resolve_axis(row_axis, ndim, name="row axis")
    col = _resolve_axis(col_axis, ndim, name="column axis")
    if row == col:
        raise ValueError("row and column axes must differ")
    entries: list[str | None] = [None] * ndim
    entries[row] = CP_ROW_AXIS
    entries[col] = CP_COL_AXIS
    return PartitionSpec(*entries)


def fold_cp_pad_width(size: int) -> int:
    """Rows to append so ``size`` divides the square mesh side."""

    side = cp_grid()[0]
    return 0 if side <= 1 else (-size) % side


def _widen(
    array: jax.Array,
    pads: Sequence[tuple[int, int]],
) -> jax.Array:
    if not any(width for _, width in pads):
        return array
    widths = [(0, 0)] * array.ndim
    for axis, width in pads:
        widths[_resolve_axis(axis, array.ndim, name="pad axis")] = (0, width)
    return jnp.pad(array, widths)


def _softmax_rescale(
    source_maximum: jax.Array,
    next_maximum: jax.Array,
) -> jax.Array:
    """Stable rescale, including empty ``-inf`` and ``+inf`` blocks."""

    finite = jnp.isfinite(source_maximum) & jnp.isfinite(next_maximum)
    positive_infinity = jnp.isposinf(source_maximum) & jnp.isposinf(next_maximum)
    difference = jnp.where(finite, source_maximum - next_maximum, 0.0)
    return jnp.where(
        positive_infinity,
        jnp.ones_like(difference),
        jnp.where(finite, jnp.exp(difference), jnp.zeros_like(difference)),
    )


def _compensated_add(
    total: jax.Array,
    correction: jax.Array,
    term: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Neumaier-add one ring tile without changing its communication schedule.

    A three-by-three mesh combines three independently reduced key tiles. Plain
    fp32 addition can lose enough low bits there to become visible after several
    Pairformer residual blocks. The correction tensor has the same local output
    shape, so the per-device asymptotic memory remains ``O(N^2/P)`` and no
    collective is introduced.
    """

    updated = total + term
    residual = jnp.where(
        jnp.abs(total) >= jnp.abs(term),
        (total - updated) + term,
        (term - updated) + total,
    )
    return updated, correction + residual


def online_softmax_update(
    output: jax.Array,
    normalizer: jax.Array,
    maximum: jax.Array,
    block_output: jax.Array,
    block_normalizer: jax.Array,
    block_maximum: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Merge one locally normalised key tile into an online accumulator.

    This public helper keeps its historical three-state API. The production
    ring below additionally carries compensation terms for the numerator and
    denominator so repeated residual blocks stay close to the dense fp32 path.
    """

    next_maximum = jnp.maximum(maximum, block_maximum)
    previous_scale = _softmax_rescale(maximum, next_maximum)
    block_scale = _softmax_rescale(block_maximum, next_maximum)
    return (
        previous_scale * output + block_scale * block_output,
        previous_scale * normalizer + block_scale * block_normalizer,
        next_maximum,
    )


#: Rows per local block times heads. The serial and 1-D paths pick their row
#: block from this same product (protenix ``_SCORE_ROWS_TIMES_HEADS``), because
#: one block's score tile is ``rows * heads * keys^2``: what has to stay
#: bounded is the product, and a swept optimum of 64 rows at 4 heads and ~25 at
#: 12 is the same 288 either way.
RING_SCORE_ROWS_TIMES_HEADS = 288
#: Ceiling on one block's score tile. A fixed row count still grows it as
#: ``keys^2``, so this is what narrows the block again on a very wide local
#: tile -- the case the 2-D layout exists for.
RING_SCORE_CEILING_BYTES = 8 * 1024**3
RING_MAX_ROWS_PER_BLOCK = 64
RING_MIN_ROWS_PER_BLOCK = 8


def resolve_ring_row_block(
    local_rows: int,
    *,
    heads: int,
    local_keys: int,
    requested: int | None = None,
) -> int:
    """Return the local pair rows one ring block runs at a time.

    The blocked axis is the *pair row* -- ``i`` in ``z[i, j]`` -- exactly as on
    the serial and 1-D paths. Triangle attention is a batch of independent
    attentions, one per row, and nothing in the softmax crosses a row, so one
    block of rows bounds ``q``/``k``/``v``, the score tile and the online
    accumulators together. Blocking the query axis instead, which is what this
    ring did, bounds only the score tile: ``q``, ``k``, ``v`` and both fp32
    accumulators stayed whole and the ring's live set stayed quadratic in the
    local width (1,969 MiB of loop carry at 1,024 OpenDDE structural tokens on
    a 2x2 mesh, five ``[N/2, heads, N/2, 32]`` tensors at once).

    The rule is the serial path's: ``rows * heads`` near
    :data:`RING_SCORE_ROWS_TIMES_HEADS`, capped at
    :data:`RING_MAX_ROWS_PER_BLOCK`, and narrowed further when one block's
    score tile would pass :data:`RING_SCORE_CEILING_BYTES`. The key extent
    here is already local (``N / sqrt(P)``), so the tile a given row count
    buys is ``P`` times smaller than the serial path's.

    ``requested`` is the caller's knob (``triangle_att_q_chunk_size`` /
    ``q_chunk_size`` / OpenFold3's ``chunk_size``): ``None`` takes the rule,
    a positive value is narrowed by it, and a non-positive one means one block
    -- the unblocked ring, which is the program this had before blocking
    existed and the same "dense" the serial knob spells that way.
    """

    if local_rows < 2:
        return local_rows
    if requested is not None and requested <= 0:
        return local_rows
    cap = min(RING_MAX_ROWS_PER_BLOCK, max(1, RING_SCORE_ROWS_TIMES_HEADS // heads))
    per_row = heads * local_keys * local_keys * 4
    if per_row <= 0:
        allowed = cap
    else:
        allowed = max(
            RING_MIN_ROWS_PER_BLOCK,
            min(cap, RING_SCORE_CEILING_BYTES // per_row),
        )
    block = allowed if requested is None else min(requested, allowed)
    return min(max(block, 1), local_rows)


def _rows_of(
    array: jax.Array,
    start: int | jax.Array | None,
    size: int,
    axis: int,
) -> jax.Array:
    """One block of rows, or the whole axis when ``start`` is ``None``.

    ``None`` is the single-block case and takes no slice at all, so the
    unblocked ring lowers to the program it lowered to before row blocking.
    """

    if start is None:
        return array
    return jax.lax.dynamic_slice_in_dim(array, start, size, axis=axis)


def _tile_scores(
    q_b: jax.Array,
    k_t: jax.Array,
    bias_b: jax.Array,
    mask_t: jax.Array,
    precision: jax.lax.Precision | None,
) -> jax.Array:
    scores = jnp.matmul(
        q_b.astype(jnp.float32),
        jnp.swapaxes(k_t.astype(jnp.float32), -1, -2),
        precision=precision,
    )
    return scores + bias_b.astype(jnp.float32) + mask_t.astype(jnp.float32)


def _tile_terms(
    q_b: jax.Array,
    k_t: jax.Array,
    v_t: jax.Array,
    bias_b: jax.Array,
    mask_t: jax.Array,
    maximum_b: jax.Array,
    precision: jax.lax.Precision | None,
) -> tuple[jax.Array, jax.Array]:
    scores = _tile_scores(q_b, k_t, bias_b, mask_t, precision)
    finite_maximum = jnp.isfinite(maximum_b)
    positive_infinity = jnp.isposinf(maximum_b)
    shifted = jnp.where(
        finite_maximum,
        scores - maximum_b,
        -jnp.inf,
    )
    probabilities = jnp.where(
        positive_infinity,
        jnp.isposinf(scores).astype(jnp.float32),
        jnp.exp(shifted),
    )
    block_normalizer = jnp.sum(
        probabilities,
        axis=-1,
        keepdims=True,
    )
    block_output = jnp.matmul(
        probabilities,
        v_t.astype(jnp.float32),
        precision=precision,
    )
    return block_output, block_normalizer


def _ring_local_rows(
    *,
    tiles: Callable[
        [int | jax.Array | None, int],
        tuple[jax.Array, jax.Array, jax.Array, jax.Array | None],
    ],
    mask_l: jax.Array,
    bias_l: jax.Array,
    rows: int,
    block: int,
    side: int,
    heads: int,
    precision: jax.lax.Precision | None,
) -> jax.Array:
    """Run the two-pass ring on one ``shard_map`` shard, a row block at a time.

    ``tiles(start, size)`` returns that block's ``q``, ``k``, ``v`` and gate --
    projected inside the block by the pair-representation entry point, sliced
    from already-projected tensors by the q/k/v one.
    """

    bias_init0 = triangle_bias_stage0_perm(side)
    diagonal_init = triangle_bias_stage1_perm(side)
    kv_hop = triangle_kv_ring_perm(side)
    bias_hop = triangle_bias_ring_perm(side)

    # The two initial bias skews do not depend on the row block, so they run
    # once above the loop; the ring hops inside it then start every block from
    # the same tile ownership.
    bias_l = permute(bias_l, bias_init0)
    bias_l = permute(bias_l, diagonal_init)

    def one_block(start: int | jax.Array | None, size: int) -> jax.Array:
        q_b, k_b, v_b, gate_b = tiles(start, size)
        if q_b.shape[-3] != heads or q_b.shape[-4] != size:
            raise ValueError(
                "ring row block expects [..., rows, heads, keys, channels]; "
                f"got {q_b.shape} for {size} rows and {heads} heads"
            )
        mask_b = _rows_of(mask_l, start, size, -4)
        # `diagonal_init` and `kv_hop` both keep the grid row, so a device owns
        # the same pair rows for the whole ring: slicing them before the skew
        # sends the same bytes in smaller messages and lands on the same tiles.
        k_t = permute(k_b, diagonal_init)
        v_t = permute(v_b, diagonal_init)
        mask_t = permute(mask_b, diagonal_init)
        bias_t = bias_l

        # Pass 1 fixes one global row maximum before any exponentials are
        # accumulated. Rotating K/bias/mask through a complete cycle returns
        # every tile to its initial owner, so pass 2 needs no saved full-width
        # operand and the per-device memory remains O(N^2 / P).
        maximum = jnp.full(
            q_b.shape[:-1] + (1,),
            -jnp.inf,
            dtype=jnp.float32,
        )
        for _ in range(side):
            scores = _tile_scores(q_b, k_t, bias_t, mask_t, precision)
            maximum = jnp.maximum(
                maximum,
                jnp.max(scores, axis=-1, keepdims=True),
            )
            # A full cycle restores the initial tile ownership. V is not used
            # in the max pass and therefore stays at its initial owner.
            k_t = permute(k_t, kv_hop)
            mask_t = permute(mask_t, kv_hop)
            bias_t = permute(bias_t, bias_hop)

        normalizer = jnp.zeros_like(maximum)
        normalizer_correction = jnp.zeros_like(maximum)
        output = jnp.zeros(q_b.shape, dtype=jnp.float32)
        output_correction = jnp.zeros_like(output)

        # Pass 2 evaluates every tile against the same maximum and combines
        # tile-local numerator/denominator terms with Neumaier compensation.
        # This removes the repeated online-rescaling error that becomes visible
        # after a 3x3 Pairformer stack while preserving the gather-free ring.
        for step in range(side):
            block_output, block_normalizer = _tile_terms(
                q_b,
                k_t,
                v_t,
                bias_t,
                mask_t,
                maximum,
                precision,
            )
            output, output_correction = _compensated_add(
                output,
                output_correction,
                block_output,
            )
            normalizer, normalizer_correction = _compensated_add(
                normalizer,
                normalizer_correction,
                block_normalizer,
            )
            if step + 1 < side:
                k_t = permute(k_t, kv_hop)
                v_t = permute(v_t, kv_hop)
                mask_t = permute(mask_t, kv_hop)
                bias_t = permute(bias_t, bias_hop)

        output = output + output_correction
        normalizer = normalizer + normalizer_correction
        tiny = jnp.asarray(jnp.finfo(jnp.float32).tiny, dtype=jnp.float32)
        result = jnp.where(
            normalizer > 0,
            output / jnp.maximum(normalizer, tiny),
            jnp.zeros_like(output),
        )
        result = result.astype(v_b.dtype)
        if gate_b is not None:
            # Same order the callers applied it in outside the ring: the fp32
            # accumulator is rounded to the value dtype first and only then
            # gated, and an elementwise product commutes with the caller's
            # transpose back to [..., rows, keys, heads, channels].
            result = result * gate_b
        return result

    if block >= rows:
        return one_block(None, rows)

    full_blocks, remainder = divmod(rows, block)
    # A ragged tail is its own static block, not a padded-and-masked axis: each
    # pair row is independent, so there is nothing a mask would protect, and
    # padding the row axis would copy the pair tile to compute rows that are
    # then discarded.
    peel_start, peel_size = (
        (full_blocks * block, remainder) if remainder else (0, block)
    )
    scan_starts = jnp.arange(full_blocks, dtype=jnp.int32) * block
    if not remainder:
        scan_starts = scan_starts[1:]

    # One block is peeled off ahead of the scan -- the ragged one where there
    # is one -- for two reasons, either of which would be enough: a `shard_map`
    # scan refuses an initial carry that does not already carry this mesh's
    # varying-ness, which a freshly zeroed destination does not; and the peeled
    # block makes the loop's carry depend on a real tile, so XLA retires that
    # tile before the loop instead of scheduling the two side by side.
    peeled = one_block(peel_start, peel_size)
    destination = jnp.zeros(
        peeled.shape[:-4] + (rows,) + peeled.shape[-3:],
        dtype=peeled.dtype,
    )
    destination = jax.lax.dynamic_update_slice_in_dim(
        destination,
        peeled,
        peel_start,
        axis=-4,
    )

    def write_block(current: jax.Array, start: jax.Array):
        return (
            jax.lax.dynamic_update_slice_in_dim(
                current,
                one_block(start, block),
                start,
                axis=-4,
            ),
            None,
        )

    # A `lax.scan` rather than a Python loop because an unrolled loop only
    # hints: the blocks are independent, so XLA schedules them together and
    # keeps every block's tile live at once. Measured on one GPU, Boltz-2 at
    # 2,096 tokens on a 2x2 mesh then asked for 68.18 GiB against the
    # unblocked path's 56.78 GiB -- blocking made it worse. A scan holds one
    # block at a time by construction; only the output destination persists.
    destination, _ = jax.lax.scan(write_block, destination, scan_starts)
    return destination


def _check_ring_mesh() -> int:
    if cp_layout() != "2d":
        raise RuntimeError("ring_triangle_attention_2d requires an active 2-D CP mesh")
    mesh = cp_mesh()
    if mesh is None:
        raise RuntimeError("context-parallel mesh is not active")
    side_row, side_col = cp_grid()
    if side_row != side_col:
        raise ValueError(f"triangle ring requires a square mesh, got {cp_grid()}")
    return side_row


def _check_ring_biases(
    triangle_bias: jax.Array,
    mask_bias: jax.Array,
    *,
    rows: int,
    tokens: int,
    ndim: int,
) -> int:
    if triangle_bias.ndim != ndim or mask_bias.ndim != ndim:
        raise ValueError(
            "triangle_bias and mask_bias must have the same rank as Q/K/V; "
            f"got {triangle_bias.ndim}, {mask_bias.ndim}, {ndim}"
        )
    if triangle_bias.shape[-2:] != (tokens, tokens):
        raise ValueError(
            "triangle-bias query/key axes do not match Q/K tokens: "
            f"{triangle_bias.shape[-2:]} vs {(tokens, tokens)}"
        )
    if mask_bias.shape[-4] != rows or mask_bias.shape[-1] != tokens:
        raise ValueError(
            "mask-bias outer/key axes do not match Q/K: "
            f"shape={mask_bias.shape}, outer={rows}, tokens={tokens}"
        )
    return triangle_bias.shape[-3]


def _pad_ring_biases(
    triangle_bias: jax.Array,
    mask_bias: jax.Array,
    *,
    pad_rows: int,
    pad_tokens: int,
) -> tuple[jax.Array, jax.Array]:
    triangle_bias = _widen(
        triangle_bias,
        ((-2, pad_tokens), (-1, pad_tokens)),
    )
    mask_bias = _widen(mask_bias, ((-4, pad_rows),))
    if pad_tokens:
        # Padded keys are absent, not merely very unlikely. Using -inf is
        # safe because the online recurrence explicitly handles a completely
        # empty tile and a globally empty query row.
        mask_bias = jnp.concatenate(
            [
                mask_bias,
                jnp.full(
                    mask_bias.shape[:-1] + (pad_tokens,),
                    -jnp.inf,
                    dtype=mask_bias.dtype,
                ),
            ],
            axis=-1,
        )
    return triangle_bias, mask_bias


def _unpad_ring_output(
    out: jax.Array,
    *,
    mesh: Any,
    spec: PartitionSpec,
    rows: int,
    tokens: int,
    pad_rows: int,
    pad_tokens: int,
) -> jax.Array:
    if pad_rows:
        out = jax.lax.slice_in_dim(out, 0, rows, axis=-4)
    if pad_tokens:
        out = jax.lax.slice_in_dim(out, 0, tokens, axis=-2)
    if pad_rows or pad_tokens:
        out = jax.lax.with_sharding_constraint(out, NamedSharding(mesh, spec))
    return out


def ring_triangle_attention_2d(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    triangle_bias: jax.Array,
    mask_bias: jax.Array,
    *,
    precision: jax.lax.Precision | None = None,
    q_block: int | None = None,
) -> jax.Array:
    """Run exact gather-free triangle attention on a square two-dimensional mesh.

    Semantic layouts::

        query/key/value  [..., outer, heads, token, channels]
        triangle_bias   [..., 1, heads, query_token, key_token]
        mask_bias       [..., outer, 1, 1, key_token]

    ``q_block`` sets the local pair rows -- the ``outer`` axis -- one ring
    block runs at a time; see :func:`resolve_ring_row_block` for the rule and
    the knob's contract. Blocking leaves the communication schedule untouched
    for Q/K/V/mask (the same bytes in more, smaller rotations, because every
    skew and hop keeps a device's grid row) and repeats the bias rotations once
    per block. Each query row still reduces over the same whole local key axis
    in the same order, so the arithmetic is unchanged. Floating-point results
    can still move by the last bits, because XLA picks its contraction
    schedule from the operand extents.

    Q/K/V arrive projected here, so a block bounds the score tile and the
    accumulators but not the projections.
    :func:`ring_triangle_attention_2d_from_pair` projects inside the block and
    bounds those too.
    """

    side = _check_ring_mesh()
    mesh = cp_mesh()

    if query.ndim < 4:
        raise ValueError(
            "triangle ring expects [..., outer, heads, token, channels], "
            f"got shape {query.shape}"
        )
    if key.shape != query.shape or value.shape != query.shape:
        raise ValueError(
            "query, key and value must have identical global shapes; got "
            f"{query.shape}, {key.shape}, {value.shape}"
        )

    outer = query.shape[-4]
    tokens = query.shape[-2]
    heads = query.shape[-3]
    _check_ring_biases(
        triangle_bias,
        mask_bias,
        rows=outer,
        tokens=tokens,
        ndim=query.ndim,
    )

    pad_outer = fold_cp_pad_width(outer)
    pad_tokens = fold_cp_pad_width(tokens)
    if pad_outer or pad_tokens:
        query = _widen(query, ((-4, pad_outer), (-2, pad_tokens)))
        key = _widen(key, ((-4, pad_outer), (-2, pad_tokens)))
        value = _widen(value, ((-4, pad_outer), (-2, pad_tokens)))
        triangle_bias, mask_bias = _pad_ring_biases(
            triangle_bias,
            mask_bias,
            pad_rows=pad_outer,
            pad_tokens=pad_tokens,
        )

    qkv_spec = _two_axis_spec(query.ndim, -4, -2)
    bias_spec = _two_axis_spec(triangle_bias.ndim, -2, -1)
    mask_spec = _two_axis_spec(mask_bias.ndim, -4, -1)

    local_rows = (outer + pad_outer) // side
    local_keys = (tokens + pad_tokens) // side
    block = resolve_ring_row_block(
        local_rows,
        heads=heads,
        local_keys=local_keys,
        requested=q_block,
    )

    def local_ring(q_l, k_l, v_l, bias_l, mask_l):
        def tiles(start, size):
            return (
                _rows_of(q_l, start, size, -4),
                _rows_of(k_l, start, size, -4),
                _rows_of(v_l, start, size, -4),
                None,
            )

        return _ring_local_rows(
            tiles=tiles,
            mask_l=mask_l,
            bias_l=bias_l,
            rows=local_rows,
            block=block,
            side=side,
            heads=heads,
            precision=precision,
        )

    out = jax.shard_map(
        local_ring,
        mesh=mesh,
        in_specs=(qkv_spec, qkv_spec, qkv_spec, bias_spec, mask_spec),
        out_specs=qkv_spec,
    )(query, key, value, triangle_bias, mask_bias)
    return _unpad_ring_output(
        out,
        mesh=mesh,
        spec=qkv_spec,
        rows=outer,
        tokens=tokens,
        pad_rows=pad_outer,
        pad_tokens=pad_tokens,
    )


def ring_triangle_attention_2d_from_pair(
    pair: Any,
    triangle_bias: jax.Array,
    mask_bias: jax.Array,
    params: Any,
    *,
    project: Callable[[Any, Any], tuple[jax.Array, jax.Array, jax.Array, Any]],
    precision: jax.lax.Precision | None = None,
    q_block: int | None = None,
) -> jax.Array:
    """Project Q/K/V one row block at a time and run the ring on each block.

    ``pair`` is the pre-projection pair representation -- one array, or a
    pytree of them sharing the ``[..., rows, keys, channels]`` geometry (the
    two operands a Q/KV-split caller holds) -- and ``project(params, rows)``
    returns that block's ``query``, ``key``, ``value`` and gate (or ``None``)
    in the ring's ``[..., rows, heads, keys, channels]`` layout, with any
    query scale already applied.

    Projecting inside the block is what keeps the ring's live set from scaling
    with the square of the local width: ``q``, ``k``, ``v`` and the gate are a
    block wide, and only the output destination spans the local rows. A linear
    contracts over the channel axis, so slicing the rows and then projecting is
    the same arithmetic as projecting and then slicing, and each row is still
    projected exactly once.
    """

    side = _check_ring_mesh()
    mesh = cp_mesh()

    leaves = jax.tree.leaves(pair)
    if not leaves:
        raise ValueError("pair representation has no arrays")
    ndim = leaves[0].ndim
    if ndim < 3:
        raise ValueError(
            "pair representation expects [..., rows, keys, channels], "
            f"got shape {leaves[0].shape}"
        )
    rows = leaves[0].shape[-3]
    tokens = leaves[0].shape[-2]
    for leaf in leaves[1:]:
        if leaf.ndim != ndim or leaf.shape[-3:-1] != (rows, tokens):
            raise ValueError(
                "pair operands must share the row and key axes; got "
                f"{leaf.shape} against {leaves[0].shape}"
            )
    heads = _check_ring_biases(
        triangle_bias,
        mask_bias,
        rows=rows,
        tokens=tokens,
        ndim=ndim + 1,
    )

    pad_rows = fold_cp_pad_width(rows)
    pad_tokens = fold_cp_pad_width(tokens)
    if pad_rows or pad_tokens:
        pair = jax.tree.map(
            lambda leaf: _widen(leaf, ((-3, pad_rows), (-2, pad_tokens))),
            pair,
        )
        triangle_bias, mask_bias = _pad_ring_biases(
            triangle_bias,
            mask_bias,
            pad_rows=pad_rows,
            pad_tokens=pad_tokens,
        )

    pair_specs = jax.tree.map(
        lambda leaf: _two_axis_spec(leaf.ndim, -3, -2),
        pair,
    )
    bias_spec = _two_axis_spec(triangle_bias.ndim, -2, -1)
    mask_spec = _two_axis_spec(mask_bias.ndim, -4, -1)
    out_spec = _two_axis_spec(ndim + 1, -4, -2)

    local_rows = (rows + pad_rows) // side
    local_keys = (tokens + pad_tokens) // side
    block = resolve_ring_row_block(
        local_rows,
        heads=heads,
        local_keys=local_keys,
        requested=q_block,
    )

    def local_ring(pair_l, bias_l, mask_l, params_l):
        def tiles(start, size):
            rows_l = jax.tree.map(
                lambda leaf: _rows_of(leaf, start, size, -3),
                pair_l,
            )
            projected = project(params_l, rows_l)
            if len(projected) != 4:
                raise ValueError(
                    "project must return (query, key, value, gate); got "
                    f"{len(projected)} values"
                )
            return projected

        return _ring_local_rows(
            tiles=tiles,
            mask_l=mask_l,
            bias_l=bias_l,
            rows=local_rows,
            block=block,
            side=side,
            heads=heads,
            precision=precision,
        )

    out = jax.shard_map(
        local_ring,
        mesh=mesh,
        # Weights are replicated, like the 1-D path's `params` operand: the
        # projection contracts over the channel axis, which no shard splits.
        in_specs=(pair_specs, bias_spec, mask_spec, PartitionSpec()),
        out_specs=out_spec,
    )(pair, triangle_bias, mask_bias, params)
    return _unpad_ring_output(
        out,
        mesh=mesh,
        spec=out_spec,
        rows=rows,
        tokens=tokens,
        pad_rows=pad_rows,
        pad_tokens=pad_tokens,
    )
