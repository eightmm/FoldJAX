"""Rigid-coordinate operations used by OpenDDE diffusion sampling."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from foldjax.models._geometry import (
    apply_scalar_rigid_transform,
    uniform_random_rotations,
)


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

    # Native rot_vec_mul deliberately uses scalar FP32 arithmetic: a matrix
    # contraction here inherits TF32 and changes the sampler's actual input.
    augmented = apply_scalar_rigid_transform(expanded, rotations, translations)
    if mask is not None:
        augmented = augmented * mask[..., None, :, None]
    return augmented


__all__ = ["centre_random_augmentation", "uniform_random_rotations"]
