"""EDM noise schedule and pre/post-conditioning (AF3 page 24).

These are weight-free, so they are exact arithmetic rather than a learned
mapping. They are ported as their own unit because they are where an EDM
implementation most often goes subtly wrong: the schedule is a ``p``-power
interpolation in ``sigma^(1/p)`` space, not a linear or cosine ramp, and the
denoiser output is a ``sigma``-dependent blend of the noisy input and the network
prediction rather than the prediction alone.
"""

from __future__ import annotations

import jax.numpy as jnp


def noise_schedule(
    num_steps: int,
    *,
    sigma_data: float,
    s_max: float,
    s_min: float,
    p: int,
) -> jnp.ndarray:
    """Return the ``num_steps + 1`` noise levels, descending.

    Args:
        num_steps: number of sampler steps.
        sigma_data: data standard deviation.
        s_max: maximum noise level, in units of ``sigma_data``.
        s_min: minimum noise level, in units of ``sigma_data``.
        p: interpolation exponent; larger ``p`` spends more steps near ``s_min``.

    Returns:
        ``[num_steps + 1]`` noise levels.
    """
    t = jnp.arange(0, 1 + num_steps, dtype=jnp.float32) / num_steps
    return sigma_data * (
        s_max ** (1 / p) + t * (s_min ** (1 / p) - s_max ** (1 / p))
    ) ** p


def scale_noisy_positions(
    xl_noisy: jnp.ndarray, t: jnp.ndarray, *, sigma_data: float
) -> jnp.ndarray:
    """Rescale noisy coordinates to unit variance before the network.

    Args:
        xl_noisy: ``[..., N_atom, 3]`` noisy coordinates.
        t: ``[...]`` noise level, broadcast over atoms and axes.
        sigma_data: data standard deviation.

    Returns:
        ``[..., N_atom, 3]`` network input.
    """
    return xl_noisy / jnp.sqrt(t[..., None, None] ** 2 + sigma_data**2)


def combine_denoiser_output(
    xl_noisy: jnp.ndarray,
    rl_update: jnp.ndarray,
    t: jnp.ndarray,
    *,
    sigma_data: float,
) -> jnp.ndarray:
    """Blend the noisy input with the network prediction (EDM postconditioning).

    At high noise the prediction dominates; at low noise the input does. Returning
    ``rl_update`` alone is the classic mistake this function exists to prevent.

    Args:
        xl_noisy: ``[..., N_atom, 3]`` noisy coordinates.
        rl_update: ``[..., N_atom, 3]`` network output.
        t: ``[...]`` noise level.
        sigma_data: data standard deviation.

    Returns:
        ``[..., N_atom, 3]`` denoised coordinates.
    """
    t = t[..., None, None]
    variance = sigma_data**2 + t**2
    return (sigma_data**2 / variance) * xl_noisy + (
        sigma_data * t / jnp.sqrt(variance)
    ) * rl_update
