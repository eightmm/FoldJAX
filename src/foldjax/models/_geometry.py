"""Shared float32 rigid-coordinate primitives for diffusion ports."""

from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.numpy as jnp


def uniform_random_rotations(
    key: jax.Array,
    shape: Sequence[int],
) -> jnp.ndarray:
    """Generate uniformly distributed proper 3D rotation matrices."""

    quaternion = jax.random.normal(key, (*shape, 4), dtype=jnp.float32)
    quaternion = quaternion / jnp.linalg.norm(quaternion, axis=-1, keepdims=True)
    w, x, y, z = jnp.moveaxis(quaternion, -1, 0)
    return jnp.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ),
        axis=-1,
    ).reshape(*shape, 3, 3)


def apply_scalar_rigid_transform(
    coordinates: jnp.ndarray,
    rotations: jnp.ndarray,
    translations: jnp.ndarray,
) -> jnp.ndarray:
    """Apply float32 scalar rigid rotation and translation without TF32 dots."""

    rotation = jnp.asarray(rotations, dtype=jnp.float32)
    x_axis, y_axis, z_axis = (
        jnp.asarray(coordinates, dtype=jnp.float32)[..., axis] for axis in range(3)
    )
    rotated = jnp.stack(
        tuple(
            rotation[..., axis, 0, None] * x_axis
            + rotation[..., axis, 1, None] * y_axis
            + rotation[..., axis, 2, None] * z_axis
            for axis in range(3)
        ),
        axis=-1,
    )
    return rotated + jnp.asarray(translations, dtype=jnp.float32)[..., None, :]


__all__ = ["apply_scalar_rigid_transform", "uniform_random_rotations"]
