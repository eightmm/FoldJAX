"""Shared distance-binning helpers for confidence, template, and affinity heads."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def distance_bins(distances: jnp.ndarray, boundaries: jnp.ndarray) -> jnp.ndarray:
    """Count the sorted ``boundaries`` below each distance.

    A binary search rather than a compare-and-sum over every boundary: XLA
    tiled that reduction 128 x 128 outputs at a time and spilled 133 KB of
    registers per thread, and at 2,304 padded tokens the kernel's local
    memory no longer fit beside the preallocated pool. The indices are
    equal, edges included, and in the default integer dtype a boolean sum
    returns; a NaN distance, which no boundary is below, keeps bin 0.

    ``boundaries`` broadcast as the compare-and-sum broadcast them: leading
    singleton axes widen the result's rank as they did there, and a boundary
    array with any wider leading axis keeps that original expression.
    """
    distances = jnp.asarray(distances)
    boundaries = jnp.asarray(boundaries)
    if boundaries.ndim != 1 and any(size != 1 for size in boundaries.shape[:-1]):
        return jnp.sum(distances[..., None] > boundaries, axis=-1)
    shape = jnp.broadcast_shapes(distances.shape, boundaries.shape[:-1])
    return jnp.broadcast_to(
        jnp.where(
            jnp.isnan(distances),
            0,
            # Sorted so the count stays independent of the stored order, as
            # the compare-and-sum it replaces was.
            jnp.searchsorted(
                jnp.sort(boundaries.reshape(-1)),
                distances,
                side="left",
                method="scan_unrolled",
            ),
        ).astype(jax.dtypes.canonicalize_dtype(int)),
        shape,
    )


def open_interval_bin(
    x: jnp.ndarray,
    lower: jnp.ndarray,
    upper: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Locate the strict open interval ``(lower[k], upper[k])`` containing ``x``.

    ``lower`` must be sorted ascending and non-empty with the same 1-D shape as
    ``upper``. Returns ``(safe_index, valid)`` where ``safe_index`` is clipped
    to ``[0, lower.shape[0] - 1]`` for safe gathering and ``valid`` rejects
    exact bin edges, out-of-range values, infinities outside the open bounds,
    and NaNs.
    """
    index = jnp.searchsorted(lower, x, side="left") - 1
    safe_index = jnp.clip(index, 0, lower.shape[0] - 1)
    valid = (
        (index >= 0)
        & (x > lower[safe_index])
        & (x < upper[safe_index])
    )
    return safe_index, valid


__all__ = ["distance_bins", "open_interval_bin"]
