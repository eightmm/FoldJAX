"""Chai-specific EDM schedule and sampler primitives."""

from __future__ import annotations

import math
from collections.abc import Callable

import jax
import jax.numpy as jnp


def chai_noise_schedule(
    num_timesteps: int,
    *,
    s_max: float = 80.0,
    s_min: float = 4e-4,
    rho: float = 7.0,
    sigma_data: float = 16.0,
) -> jnp.ndarray:
    """Return Chai's midpoint schedule of exactly ``num_timesteps`` sigmas."""
    if num_timesteps < 2:
        raise ValueError("num_timesteps must be at least 2")
    times = jnp.linspace(0.0, 1.0, 2 * num_timesteps + 1)[1::2]
    interpolated = (
        times * s_min ** (1.0 / rho)
        + (1.0 - times) * s_max ** (1.0 / rho)
    ) ** rho
    return sigma_data * interpolated


def chai_diffusion_gammas(
    sigmas: jnp.ndarray,
    *,
    num_timesteps: int,
    s_churn: float = 80.0,
    s_tmin: float = 4e-4,
    s_tmax: float = 80.0,
) -> jnp.ndarray:
    """Compute the upstream Chai per-sigma churn factors."""
    gamma = min(s_churn / num_timesteps, math.sqrt(2.0) - 1.0)
    active = (sigmas >= s_tmin) & (sigmas <= s_tmax)
    return jnp.where(active, gamma, 0.0)


def center_augmentation(
    coords: jnp.ndarray,
    atom_mask: jnp.ndarray,
    rotations: jnp.ndarray,
    translations: jnp.ndarray,
) -> jnp.ndarray:
    """Apply Chai's masked centering and supplied rigid augmentation."""
    weights = atom_mask.astype(coords.dtype)
    weights = weights / jnp.maximum(jnp.sum(weights, axis=-1, keepdims=True), 1e-4)
    centroid = jnp.sum(coords * weights[..., None], axis=-2, keepdims=True)
    centered = coords - centroid
    rotated = jnp.einsum(
        "bij,baj->bai",
        rotations,
        centered,
        precision=jax.lax.Precision.HIGHEST,
    )
    return rotated + translations


DenoiseFn = Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]


def edm_heun_step(
    atom_pos: jnp.ndarray,
    atom_mask: jnp.ndarray,
    *,
    sigma_curr: float | jnp.ndarray,
    sigma_next: float | jnp.ndarray,
    gamma: float | jnp.ndarray,
    noise: jnp.ndarray,
    rotations: jnp.ndarray,
    translations: jnp.ndarray,
    denoise: DenoiseFn,
    s_noise: float = 1.003,
    second_order: bool = True,
) -> jnp.ndarray:
    """Run one Chai EDM interval with all randomness explicitly supplied."""
    augmented = center_augmentation(
        atom_pos, atom_mask, rotations, translations
    )
    sigma_hat = sigma_curr * (1.0 + gamma)
    noise_scale = jnp.sqrt(
        jnp.maximum(sigma_hat**2 - sigma_curr**2, 1e-6)
    )
    atom_pos_hat = augmented + s_noise * noise * noise_scale

    denoised = denoise(atom_pos_hat, jnp.asarray(sigma_hat))
    derivative = (atom_pos_hat - denoised) / sigma_hat
    euler = atom_pos_hat + (sigma_next - sigma_hat) * derivative
    if not second_order:
        return euler

    denoised_next = denoise(euler, jnp.asarray(sigma_next))
    derivative_next = (euler - denoised_next) / sigma_next
    return euler + (sigma_next - sigma_hat) * (
        (derivative_next + derivative) / 2.0
    )
