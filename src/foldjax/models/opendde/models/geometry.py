"""Rigid-coordinate operations used by OpenDDE diffusion sampling."""

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


def centre_random_augmentation(
    coordinates: jnp.ndarray,
    *,
    num_samples: int = 1,
    s_trans: float = 1.0,
    centre_only: bool = False,
    mask: jnp.ndarray | None = None,
    eps: float = 1e-12,
    key: jax.Array | None = None,
    rotations: jnp.ndarray | None = None,
    translations: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Center coordinates and apply OpenDDE's random rigid augmentation.

    ``rotations`` and ``translations`` accept a shared random tape for exact
    PyTorch/JAX parity. Otherwise ``key`` generates the same distributions,
    without claiming bitwise agreement with SciPy or PyTorch RNG streams.
    """

    coordinates = jnp.asarray(coordinates)
    if mask is None:
        center = jnp.mean(coordinates, axis=-2, keepdims=True)
    else:
        mask = jnp.asarray(mask, dtype=coordinates.dtype)
        numerator = jnp.sum(coordinates * mask[..., :, None], axis=-2)
        denominator = jnp.sum(mask, axis=-1, keepdims=True) + eps
        center = (numerator / denominator)[..., None, :]
    centered = (coordinates - center).astype(jnp.float32)
    sample_shape = (*centered.shape[:-2], num_samples)
    expanded = jnp.broadcast_to(
        centered[..., None, :, :],
        (*sample_shape, *centered.shape[-2:]),
    )
    if centre_only:
        return expanded

    if rotations is None and translations is None:
        if key is None:
            raise ValueError(
                "random augmentation requires key or rotations/translations"
            )
        rotation_key, translation_key = jax.random.split(key)
        rotations = uniform_random_rotations(rotation_key, sample_shape)
        translations = s_trans * jax.random.normal(
            translation_key,
            (*sample_shape, 3),
            dtype=jnp.float32,
        )
    elif rotations is None or translations is None:
        raise ValueError("rotations and translations must be provided together")

    augmented = jnp.einsum(
        "...sij,...saj->...sai",
        jnp.asarray(rotations, dtype=jnp.float32),
        expanded,
    )
    augmented = augmented + jnp.asarray(translations, dtype=jnp.float32)[..., None, :]
    if mask is not None:
        augmented = augmented * mask[..., None, :, None]
    return augmented
