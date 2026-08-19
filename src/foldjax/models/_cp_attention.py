"""Gather-free 2-D Fold-CP attention primitives.

The pair representation is laid out on a square ``cp_row x cp_col`` mesh.
Triangle attention contains three token roles (outer row, query and key), so a
plain SPMD annotation cannot keep all three local. The official Fold-CP
schedule keeps the query tile resident, redistributes the pair bias onto the
query/key diagonal, and rotates key/value/mask tiles in a ring while updating a
numerically stable online softmax accumulator. This module expresses that
schedule with ``jax.shard_map`` and ``lax.ppermute``.

No operation in :func:`ring_triangle_attention_2d` materialises a full token
axis on one device. Inputs must therefore already be padded so both pair axes
split evenly over the square mesh; data-layer padding is an explicit part of
the Fold-CP contract rather than a hidden gather-and-slice fallback.
"""

from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec

from foldjax.models._cp import (
    CP_COL_AXIS,
    CP_ROW_AXIS,
    cp_grid,
    cp_layout,
    cp_mesh,
    permute,
)


def _flat(
    pairs: Sequence[tuple[tuple[int, int], tuple[int, int]]], side: int
) -> list[tuple[int, int]]:
    """Flatten 2-D grid-coordinate permutations in row-major rank order."""

    return [(a * side + b, c * side + d) for (a, b), (c, d) in pairs]


def triangle_bias_stage0_perm(side: int) -> list[tuple[int, int]]:
    """Flatten lower diagonals onto rows: ``(r, c) -> (r-c, c)``.

    This is stage one of ``Ring2DCommTriAttn`` for starting-node attention.
    Keeping it separate from stage two preserves the official topology-aware
    schedule instead of replacing it with one arbitrary cross-grid exchange.
    """

    return _flat(
        [
            ((row, col), ((row - col) % side, col))
            for row in range(side)
            for col in range(side)
        ],
        side,
    )


def triangle_bias_stage1_perm(side: int) -> list[tuple[int, int]]:
    """Rotate the flattened diagonals: ``(r, c) -> (r, c+r)``."""

    return _flat(
        [
            ((row, col), (row, (col + row) % side))
            for row in range(side)
            for col in range(side)
        ],
        side,
    )


def triangle_kv_initial_perm(side: int) -> list[tuple[int, int]]:
    """Offset K/V tiles onto the same diagonal as the redistributed bias."""

    return triangle_bias_stage1_perm(side)


def triangle_kv_ring_perm(side: int) -> list[tuple[int, int]]:
    """Advance K/V/mask one key tile around each device-grid row."""

    return _flat(
        [
            ((row, col), (row, (col + 1) % side))
            for row in range(side)
            for col in range(side)
        ],
        side,
    )


def triangle_bias_ring_perm(side: int) -> list[tuple[int, int]]:
    """Advance bias one matching key tile up each device-grid column."""

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


def require_fold_cp_divisible(size: int, *, what: str = "token axis") -> None:
    """Fail before tracing when a 2-D Fold-CP axis is not evenly shardable."""

    side = cp_grid()[0]
    if side > 1 and size % side:
        raise ValueError(
            f"2-D Fold-CP requires {what}={size} to be divisible by the "
            f"mesh side {side}. Pad semantic token/atom axes at the data layer; "
            "an internal gather-and-slice fallback would forfeit the memory bound."
        )


def online_softmax_update(
    output: jax.Array,
    normalizer: jax.Array,
    maximum: jax.Array,
    block_output: jax.Array,
    block_normalizer: jax.Array,
    block_maximum: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Merge one key tile into an unnormalised online-softmax accumulator.

    ``output`` and ``block_output`` contain ``sum(exp(score-max) * value)``;
    ``normalizer`` contains the matching exponential sum. All accumulator
    arithmetic is fp32 even when Q/K/V are bf16.
    """

    next_maximum = jnp.maximum(maximum, block_maximum)
    previous_scale = jnp.exp(maximum - next_maximum)
    block_scale = jnp.exp(block_maximum - next_maximum)
    return (
        previous_scale * output + block_scale * block_output,
        previous_scale * normalizer + block_scale * block_normalizer,
        next_maximum,
    )


def ring_triangle_attention_2d(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    triangle_bias: jax.Array,
    mask_bias: jax.Array,
    *,
    precision: jax.lax.Precision | None = None,
) -> jax.Array:
    """Run exact gather-free triangle attention on a square 2-D CP mesh.

    Expected semantic layouts are::

        query/key/value  [..., outer, heads, token, channels]
        triangle_bias   [..., 1, heads, query_token, key_token]
        mask_bias       [..., outer, 1, 1, key_token]

    The global arrays remain ordinary JAX arrays; ``shard_map`` assigns
    ``outer``/``query_token`` and ``query_token``/``key_token`` to the two mesh
    axes. Inside each device the resident query tile never moves. K/V/mask and
    the pair-bias tile rotate for ``sqrt(P)`` steps, and
    :func:`online_softmax_update` combines the partial softmaxes exactly.
    """

    if cp_layout() != "2d":
        raise RuntimeError("ring_triangle_attention_2d requires an active 2-D CP mesh")
    mesh = cp_mesh()
    if mesh is None:  # defensive; cp_layout() already excludes this
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
    require_fold_cp_divisible(outer, what="outer token axis")
    require_fold_cp_divisible(tokens, what="attended token axis")

    qkv_spec = _two_axis_spec(query.ndim, -4, -2)
    bias_spec = _two_axis_spec(triangle_bias.ndim, -2, -1)
    mask_spec = _two_axis_spec(mask_bias.ndim, -4, -1)

    bias_init0 = triangle_bias_stage0_perm(side)
    diagonal_init = triangle_bias_stage1_perm(side)
    kv_hop = triangle_kv_ring_perm(side)
    bias_hop = triangle_bias_ring_perm(side)

    def local_ring(q_l, k_l, v_l, bias_l, mask_l):
        # Official two-stage pair-bias redistribution. After these two
        # permutations, device (r, c) owns bias(query=c, key=c-r).
        bias_l = permute(bias_l, bias_init0)
        bias_l = permute(bias_l, diagonal_init)

        # Device (r, c) originally owns K/V key tile c. Shifting by row r
        # aligns it to key c-r, matching the redistributed bias.
        k_l = permute(k_l, diagonal_init)
        v_l = permute(v_l, diagonal_init)
        mask_l = permute(mask_l, diagonal_init)

        maximum = jnp.full(q_l.shape[:-1] + (1,), -jnp.inf, dtype=jnp.float32)
        normalizer = jnp.zeros_like(maximum)
        output = jnp.zeros(q_l.shape, dtype=jnp.float32)

        for step in range(side):
            scores = jnp.einsum(
                "...hqd,...hkd->...hqk",
                q_l.astype(jnp.float32),
                k_l.astype(jnp.float32),
                precision=precision,
            )
            scores = (
                scores
                + bias_l.astype(jnp.float32)
                + mask_l.astype(jnp.float32)
            )
            block_maximum = jnp.max(scores, axis=-1, keepdims=True)
            probabilities = jnp.exp(scores - block_maximum)
            block_normalizer = jnp.sum(probabilities, axis=-1, keepdims=True)
            block_output = jnp.einsum(
                "...hqk,...hkd->...hqd",
                probabilities,
                v_l.astype(jnp.float32),
                precision=precision,
            )
            output, normalizer, maximum = online_softmax_update(
                output,
                normalizer,
                maximum,
                block_output,
                block_normalizer,
                block_maximum,
            )

            if step + 1 < side:
                k_l = permute(k_l, kv_hop)
                v_l = permute(v_l, kv_hop)
                mask_l = permute(mask_l, kv_hop)
                bias_l = permute(bias_l, bias_hop)

        tiny = jnp.asarray(jnp.finfo(jnp.float32).tiny, dtype=jnp.float32)
        return (output / jnp.maximum(normalizer, tiny)).astype(v_l.dtype)

    return jax.shard_map(
        local_ring,
        mesh=mesh,
        in_specs=(qkv_spec, qkv_spec, qkv_spec, bias_spec, mask_spec),
        out_specs=qkv_spec,
    )(query, key, value, triangle_bias, mask_bias)
