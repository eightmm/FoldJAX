"""Static-shape overlapping windows without unrolled slice stacks."""

from __future__ import annotations

import jax.numpy as jnp


def gather_overlapping_windows(
    array: jnp.ndarray,
    *,
    axis: int,
    n_windows: int,
    window_size: int,
    stride: int,
) -> jnp.ndarray:
    """Replace one tiled axis with ``[window, element]`` axes.

    The source length must describe an exact series of full windows. Indices
    are static, so JAX lowers this to one gather instead of cloning one slice
    into the graph for every atom-query block.
    """

    if array.ndim == 0:
        raise ValueError("overlapping windows require an array axis")
    if not -array.ndim <= axis < array.ndim:
        raise ValueError(f"axis {axis} is out of bounds for rank {array.ndim}")
    if window_size < 1:
        raise ValueError("window_size must be positive")
    if stride < 1:
        raise ValueError("stride must be positive")
    if n_windows < 1:
        raise ValueError("n_windows must be positive")

    axis %= array.ndim
    length = array.shape[axis]
    required_length = (n_windows - 1) * stride + window_size
    if length != required_length:
        raise ValueError(
            f"axis length {length} does not match {n_windows} windows of "
            f"size {window_size} at stride {stride}; expected {required_length}"
        )

    indices = (
        jnp.arange(n_windows, dtype=jnp.int32)[:, None] * stride
        + jnp.arange(window_size, dtype=jnp.int32)[None]
    )
    return jnp.take(array, indices, axis=axis, mode="clip")


__all__ = ["gather_overlapping_windows"]
