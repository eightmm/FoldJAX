"""Gather-free two-dimensional Fold-CP attention primitives.

The pair representation is tiled over a square ``cp_row x cp_col`` mesh.
Queries stay resident while key, value, mask and pair-bias tiles rotate through
a ring. An fp32 online-softmax accumulator makes the result mathematically
equivalent to dense attention without materialising a full token axis.

Each ring step evaluates its local tile in blocks of query rows, so the live
score tensor is ``q_block x (N/s)^2 x heads`` rather than the whole
``(N/s)^3 x heads`` tile. See :func:`resolve_ring_query_block`.
"""

from __future__ import annotations

from collections.abc import Sequence

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


#: Query rows per local score block inside one ring step. The serial and 1-D
#: paths use the same 512 for their inner query chunk.
RING_QUERY_BLOCK = 512


def resolve_ring_query_block(local_queries: int, q_block: int | None = None) -> int:
    """Return the query-row block one local ring tile is evaluated in.

    The serial and 1-D paths bound their score tensor with
    ``resolve_triangle_attention_q_chunk``, whose gate -- 512 rows once the
    sequence passes 2,048 tokens -- reads the *global* token count. That gate
    cannot be reused verbatim here, because the ring's query axis is already
    ``N / sqrt(P)``: at the sizes for which a square mesh is chosen for its
    memory (2,096 tokens over a 2x2 mesh leaves 1,048 local rows) it would
    never fire and the tile would stay whole. The default is therefore the
    same 512 rows applied unconditionally to the local axis and clamped to it.

    A non-positive block, or one covering the whole local axis, means a single
    block. That case takes the unblocked path below and lowers to the exact
    program this ring had before blocking existed.
    """

    if q_block is None:
        return min(RING_QUERY_BLOCK, local_queries)
    if q_block <= 0 or q_block >= local_queries:
        return local_queries
    return q_block


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

    ``q_block`` sets the query rows per local score block; see
    :func:`resolve_ring_query_block` for the default. Blocking leaves the
    communication schedule untouched -- it happens inside one ring step, after
    the rotations have already placed the tile -- and each query row still
    reduces over the same whole local key axis, so the arithmetic is unchanged.
    Floating-point results can still move by the last bits, because XLA picks
    its contraction schedule from the operand extents.
    """

    if cp_layout() != "2d":
        raise RuntimeError("ring_triangle_attention_2d requires an active 2-D CP mesh")
    mesh = cp_mesh()
    if mesh is None:
        raise RuntimeError("context-parallel mesh is not active")
    side_row, side_col = cp_grid()
    if side_row != side_col:
        raise ValueError(f"triangle ring requires a square mesh, got {cp_grid()}")
    side = side_row

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
    if triangle_bias.ndim != query.ndim or mask_bias.ndim != query.ndim:
        raise ValueError(
            "triangle_bias and mask_bias must have the same rank as Q/K/V; "
            f"got {triangle_bias.ndim}, {mask_bias.ndim}, {query.ndim}"
        )

    outer = query.shape[-4]
    tokens = query.shape[-2]
    if triangle_bias.shape[-2:] != (tokens, tokens):
        raise ValueError(
            "triangle-bias query/key axes do not match Q/K tokens: "
            f"{triangle_bias.shape[-2:]} vs {(tokens, tokens)}"
        )
    if mask_bias.shape[-4] != outer or mask_bias.shape[-1] != tokens:
        raise ValueError(
            "mask-bias outer/key axes do not match Q/K: "
            f"shape={mask_bias.shape}, outer={outer}, tokens={tokens}"
        )

    pad_outer = fold_cp_pad_width(outer)
    pad_tokens = fold_cp_pad_width(tokens)
    if pad_outer or pad_tokens:
        query = _widen(query, ((-4, pad_outer), (-2, pad_tokens)))
        key = _widen(key, ((-4, pad_outer), (-2, pad_tokens)))
        value = _widen(value, ((-4, pad_outer), (-2, pad_tokens)))
        triangle_bias = _widen(
            triangle_bias,
            ((-2, pad_tokens), (-1, pad_tokens)),
        )
        mask_bias = _widen(mask_bias, ((-4, pad_outer),))
        if pad_tokens:
            # Padded keys are absent, not merely very unlikely. Using -inf is
            # now safe because the online recurrence explicitly handles a
            # completely empty tile and a globally empty query row.
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

    qkv_spec = _two_axis_spec(query.ndim, -4, -2)
    bias_spec = _two_axis_spec(triangle_bias.ndim, -2, -1)
    mask_spec = _two_axis_spec(mask_bias.ndim, -4, -1)

    bias_init0 = triangle_bias_stage0_perm(side)
    diagonal_init = triangle_bias_stage1_perm(side)
    kv_hop = triangle_kv_ring_perm(side)
    bias_hop = triangle_bias_ring_perm(side)

    def local_ring(q_l, k_l, v_l, bias_l, mask_l):
        bias_l = permute(bias_l, bias_init0)
        bias_l = permute(bias_l, diagonal_init)
        k_l = permute(k_l, diagonal_init)
        v_l = permute(v_l, diagonal_init)
        mask_l = permute(mask_l, diagonal_init)

        # Blocking the local query rows is what keeps the score tile off the
        # device: one ring step would otherwise materialise
        # f32[..., rows, heads, N/s, N/s] whole, which is cubic in the local
        # token extent and made the 2-D layout cost more memory than the 1-D
        # one it exists to beat. The blocks are cut after the rotations, so
        # the schedule above and below is untouched.
        local_queries = q_l.shape[-2]
        block = resolve_ring_query_block(local_queries, q_block)
        blocked = block < local_queries
        starts = range(0, local_queries, block)

        def query_blocks(array):
            # Every other axis is shared, including the key axis each row
            # still reduces over whole.
            if not blocked:
                return (array,)
            return tuple(
                jax.lax.slice_in_dim(
                    array,
                    start,
                    min(start + block, local_queries),
                    axis=-2,
                )
                for start in starts
            )

        def joined(parts):
            return parts[0] if len(parts) == 1 else jnp.concatenate(parts, axis=-2)

        def tile_scores(q_b, k_t, bias_b, mask_t):
            scores = jnp.matmul(
                q_b.astype(jnp.float32),
                jnp.swapaxes(k_t.astype(jnp.float32), -1, -2),
                precision=precision,
            )
            return scores + bias_b.astype(jnp.float32) + mask_t.astype(jnp.float32)

        # Pass 1 fixes one global row maximum before any exponentials are
        # accumulated. Rotating K/bias/mask through a complete cycle returns
        # every tile to its initial owner, so pass 2 needs no saved full-width
        # operand and the per-device memory remains O(N^2 / P).
        maximum = jnp.full(
            q_l.shape[:-1] + (1,),
            -jnp.inf,
            dtype=jnp.float32,
        )
        for _ in range(side):
            tile_maximum = joined(
                [
                    jnp.max(
                        tile_scores(q_b, k_l, bias_b, mask_l),
                        axis=-1,
                        keepdims=True,
                    )
                    for q_b, bias_b in zip(
                        query_blocks(q_l),
                        query_blocks(bias_l),
                        strict=True,
                    )
                ]
            )
            maximum = jnp.maximum(maximum, tile_maximum)
            # A full cycle restores the initial tile ownership. V is not used
            # in the max pass and therefore stays at its initial owner.
            k_l = permute(k_l, kv_hop)
            mask_l = permute(mask_l, kv_hop)
            bias_l = permute(bias_l, bias_hop)

        normalizer = jnp.zeros_like(maximum)
        normalizer_correction = jnp.zeros_like(maximum)
        output = jnp.zeros(q_l.shape, dtype=jnp.float32)
        output_correction = jnp.zeros_like(output)

        # Pass 2 evaluates every tile against the same maximum and combines
        # tile-local numerator/denominator terms with Neumaier compensation.
        # This removes the repeated online-rescaling error that becomes visible
        # after a 3x3 Pairformer stack while preserving the gather-free ring.
        for step in range(side):
            block_outputs = []
            block_normalizers = []
            for q_b, bias_b, maximum_b in zip(
                query_blocks(q_l),
                query_blocks(bias_l),
                query_blocks(maximum),
                strict=True,
            ):
                scores = tile_scores(q_b, k_l, bias_b, mask_l)
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
                block_normalizers.append(
                    jnp.sum(
                        probabilities,
                        axis=-1,
                        keepdims=True,
                    )
                )
                block_outputs.append(
                    jnp.matmul(
                        probabilities,
                        v_l.astype(jnp.float32),
                        precision=precision,
                    )
                )
            block_output = joined(block_outputs)
            block_normalizer = joined(block_normalizers)
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
                k_l = permute(k_l, kv_hop)
                v_l = permute(v_l, kv_hop)
                mask_l = permute(mask_l, kv_hop)
                bias_l = permute(bias_l, bias_hop)

        output = output + output_correction
        normalizer = normalizer + normalizer_correction
        tiny = jnp.asarray(jnp.finfo(jnp.float32).tiny, dtype=jnp.float32)
        result = jnp.where(
            normalizer > 0,
            output / jnp.maximum(normalizer, tiny),
            jnp.zeros_like(output),
        )
        return result.astype(v_l.dtype)

    out = jax.shard_map(
        local_ring,
        mesh=mesh,
        in_specs=(qkv_spec, qkv_spec, qkv_spec, bias_spec, mask_spec),
        out_specs=qkv_spec,
    )(query, key, value, triangle_bias, mask_bias)
    if pad_outer:
        out = jax.lax.slice_in_dim(out, 0, outer, axis=-4)
    if pad_tokens:
        out = jax.lax.slice_in_dim(out, 0, tokens, axis=-2)
    if pad_outer or pad_tokens:
        out = jax.lax.with_sharding_constraint(
            out,
            NamedSharding(mesh, qkv_spec),
        )
    return out
