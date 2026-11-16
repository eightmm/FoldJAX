"""Deterministic segment sums for atom-to-token aggregation."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


class SegmentPlan(NamedTuple):
    """The owner grouping a segment sum needs, reusable while owners are fixed.

    ``key`` is each atom's owner under ``.at[]`` index rules (``num_segments``
    when dropped), ``order`` the atoms grouped by owner in their original
    order, and ``starts``/``counts`` each owner's range in ``order``. ``counts``
    is also the exact number of atoms each owner sums.
    """

    key: jax.Array
    order: jax.Array
    starts: jax.Array
    counts: jax.Array


def _owner_key(segment_ids: jax.Array, num_segments: int) -> jax.Array:
    ids = segment_ids.astype(jnp.int32)
    ids = jnp.where(ids < 0, ids + num_segments, ids)
    return jnp.where((ids >= 0) & (ids < num_segments), ids, num_segments)


def segment_plan(segment_ids: jax.Array, num_segments: int) -> SegmentPlan:
    """Group ``segment_ids`` by owner once, for every sum over the same owners.

    ``segment_ids`` is 1-D; vmap for per-row owners. Passing the plan to
    :func:`segment_sum` gives bit-identical sums without the per-call sort.
    """

    segment_ids = jnp.asarray(segment_ids)
    if segment_ids.ndim != 1:
        raise ValueError("segment_ids must be 1-D")
    num_segments = int(num_segments)
    key = _owner_key(segment_ids, num_segments)
    order = jnp.argsort(key, stable=True).astype(jnp.int32)
    sorted_key = key[order]
    owners = jnp.arange(num_segments, dtype=jnp.int32)
    starts = jnp.searchsorted(sorted_key, owners, side="left", method="scan")
    ends = jnp.searchsorted(sorted_key, owners, side="right", method="scan")
    return SegmentPlan(
        key, order, starts.astype(jnp.int32), (ends - starts).astype(jnp.int32)
    )


def _member_shape(ndim: int, axis: int) -> tuple[int, ...]:
    return (-1,) + (1,) * (ndim - axis - 1)


def _segment_sum_impl(
    values: jax.Array, plan: SegmentPlan, num_segments: int, axis: int
) -> jax.Array:
    shape = values.shape[:axis] + (num_segments,) + values.shape[axis + 1 :]
    total = jnp.zeros(shape, dtype=values.dtype)
    n = int(plan.order.shape[0])
    if n == 0 or num_segments == 0:
        return total
    zero = jnp.zeros((), dtype=values.dtype)

    def add_rank(rank, total):
        source = plan.order[jnp.minimum(plan.starts + rank, n - 1)]
        member = (rank < plan.counts).reshape(_member_shape(values.ndim, axis))
        return total + jnp.where(member, jnp.take(values, source, axis=axis), zero)

    # Rank 0 is peeled so the carry takes the inputs' shard_map varying axes.
    return jax.lax.fori_loop(1, jnp.max(plan.counts), add_rank, add_rank(0, total))


def _segment_sum_fwd(values, plan, num_segments, axis):
    return _segment_sum_impl(values, plan, num_segments, axis), plan.key


def _segment_sum_bwd(num_segments, axis, key, cotangent):
    valid = key < num_segments
    gathered = jnp.take(cotangent, jnp.where(valid, key, 0), axis=axis)
    member = valid.reshape(_member_shape(cotangent.ndim, axis))
    return jnp.where(member, gathered, jnp.zeros((), cotangent.dtype)), None


_segment_sum = jax.custom_vjp(_segment_sum_impl, nondiff_argnums=(2, 3))
_segment_sum.defvjp(_segment_sum_fwd, _segment_sum_bwd)
# Eager callers (the guided sampler) would otherwise retrace the loop and
# compile a fresh program on every call.
_segment_sum = jax.jit(_segment_sum, static_argnums=(2, 3))


def segment_sum(
    values: jax.Array,
    segment_ids: jax.Array | None,
    num_segments: int,
    *,
    axis: int = 0,
    plan: SegmentPlan | None = None,
) -> jax.Array:
    """Sum ``values`` along ``axis`` into ``num_segments`` owners.

    The same result as ``zeros.at[..., segment_ids, ...].add(values)`` up to
    floating-point rounding, with the same index rules: a negative owner wraps
    once and an owner still out of range is dropped. ``segment_ids`` is 1-D over
    ``axis``; vmap for per-row owners. A ``plan`` from :func:`segment_plan`
    replaces ``segment_ids``. Sums accumulate in the input dtype, as the
    scatter-add does; a caller whose upstream accumulates wider (a float32
    contraction) casts first.

    A gather-and-add over each owner's atoms in their original order rather
    than a scatter-add: GPU scatter-add uses atomics, whose summation order is
    not reproducible, and the diffusion rollout amplifies that into visibly
    different coordinates for the same PRNG key. The loop runs once per atom
    rank of the largest owner, so callers route masked atoms to a dropped owner
    rather than piling them on one token; it holds only a ``[num_segments,
    ...]`` buffer, and its trip count is data-dependent, so the gradient is
    supplied directly.
    """

    values = jnp.asarray(values)
    axis = axis % values.ndim
    num_segments = int(num_segments)
    if plan is None:
        if segment_ids is None:
            raise ValueError("segment_ids or plan is required")
        segment_ids = jnp.asarray(segment_ids)
        if segment_ids.ndim != 1 or segment_ids.shape[0] != values.shape[axis]:
            raise ValueError(
                "segment_ids must be 1-D and match values along the summed axis"
            )
        plan = segment_plan(segment_ids, num_segments)
    elif plan.order.shape != (values.shape[axis],) or plan.counts.shape != (
        num_segments,
    ):
        raise ValueError("plan does not match values and num_segments")
    return _segment_sum(values, plan, num_segments, axis)


__all__ = ["SegmentPlan", "segment_plan", "segment_sum"]
