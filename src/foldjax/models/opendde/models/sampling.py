"""OpenDDE diffusion sampler with per-step rigid augmentation."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp

from foldjax.models.opendde.models.geometry import (
    centre_random_augmentation,
    uniform_random_rotations,
)


def sample_diffusion(
    denoise_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    noise_schedule: jnp.ndarray,
    *,
    n_sample: int,
    n_atom: int,
    key: jax.Array | None,
    init_noise: jnp.ndarray | None = None,
    step_noises: Sequence[jnp.ndarray] | None = None,
    rotations: jnp.ndarray | None = None,
    translations: jnp.ndarray | None = None,
    batch_shape: Sequence[int] = (),
    gamma0: float = 0.8,
    gamma_min: float = 1.0,
    noise_scale_lambda: float = 1.003,
    step_scale_eta: float = 1.5,
    dtype: jnp.dtype = jnp.float32,
    use_scan: bool = False,
) -> jnp.ndarray:
    """Run OpenDDE's loop sampler with optional shared random tapes.

    ``use_scan`` rolls the step loop into ``lax.scan`` instead of writing one
    copy of the denoiser per step into the graph. Every other repeated stack in
    this port is already scanned; the sampler was the one that was not, and it
    is the one whose count is a user-facing knob. At the released 200-step
    schedule the unrolled form puts 200 copies of the structural refiner in a
    single executable, which is why OpenDDE was the slowest model here by far.
    """

    noise_schedule = jnp.asarray(noise_schedule, dtype=dtype)
    n_steps = int(noise_schedule.shape[0]) - 1
    if n_steps < 1:
        raise ValueError("noise schedule must contain at least two levels")
    tapes_complete = all(
        tape is not None for tape in (init_noise, step_noises, rotations, translations)
    )
    if key is None and not tapes_complete:
        raise ValueError("sample_diffusion requires key or complete random tapes")

    if key is not None:
        init_key, step_key, rotation_key, translation_key = jax.random.split(key, 4)
    if init_noise is None:
        init_noise = jax.random.normal(
            init_key,
            (*batch_shape, n_sample, n_atom, 3),
            dtype=dtype,
        )
    else:
        init_noise = jnp.asarray(init_noise, dtype=dtype)
    expected_shape = (*batch_shape, n_sample, n_atom, 3)
    if tuple(init_noise.shape) != expected_shape:
        raise ValueError(
            f"init_noise expected shape {expected_shape}, got {init_noise.shape}"
        )

    leading_shape = init_noise.shape[:-2]
    if step_noises is None:
        step_keys = jax.random.split(step_key, n_steps)
        step_noises = tuple(
            jax.random.normal(step_key_i, init_noise.shape, dtype=dtype)
            for step_key_i in step_keys
        )
    else:
        step_noises = tuple(jnp.asarray(noise, dtype=dtype) for noise in step_noises)
    if len(step_noises) != n_steps or any(
        noise.shape != init_noise.shape for noise in step_noises
    ):
        raise ValueError("step_noises must match the number of steps and init shape")

    if rotations is None:
        rotations = uniform_random_rotations(
            rotation_key,
            (n_steps, *leading_shape),
        )
    else:
        rotations = jnp.asarray(rotations, dtype=jnp.float32)
    expected_rotations = (n_steps, *leading_shape, 3, 3)
    if tuple(rotations.shape) != expected_rotations:
        raise ValueError(
            f"rotations expected shape {expected_rotations}, got {rotations.shape}"
        )

    if translations is None:
        translations = jax.random.normal(
            translation_key,
            (n_steps, *leading_shape, 3),
            dtype=jnp.float32,
        )
    else:
        translations = jnp.asarray(translations, dtype=jnp.float32)
    expected_translations = (n_steps, *leading_shape, 3)
    if tuple(translations.shape) != expected_translations:
        raise ValueError(
            "translations expected shape "
            f"{expected_translations}, got {translations.shape}"
        )

    x_l = noise_schedule[0] * init_noise

    def one_step(x_current, c_tau_last, c_tau, step_noise, rotation, translation):
        """One Algorithm 18 step. Shared by the rolled and unrolled paths."""
        augmented = centre_random_augmentation(
            x_current,
            n_sample=1,
            rotations=rotation[..., None, :, :],
            translations=translation[..., None, :],
        )
        x_current = jnp.squeeze(augmented, axis=-3).astype(dtype)
        gamma = jnp.where(c_tau > gamma_min, gamma0, 0.0).astype(dtype)
        t_hat_scalar = c_tau_last * (gamma + 1.0)
        # `t_hat**2 - c**2` is a difference of nearly equal squares and exactly
        # zero when gamma is zero, so XLA's contraction can leave it slightly
        # negative and `sqrt` returns NaN. The identity
        # `t_hat**2 - c**2 == c**2 * gamma * (gamma + 2)` avoids the
        # cancellation. Same defect and same fix as the Protenix sampler.
        delta_noise_level = c_tau_last * jnp.sqrt(gamma * (gamma + 2.0))
        x_noisy = x_current + noise_scale_lambda * delta_noise_level * step_noise
        t_hat = jnp.full(x_noisy.shape[:-2], t_hat_scalar, dtype=dtype)
        x_denoised = denoise_fn(x_noisy, t_hat)
        delta = (x_noisy - x_denoised) / t_hat[..., None, None]
        dt = c_tau - t_hat
        return x_noisy + step_scale_eta * dt[..., None, None] * delta

    if use_scan:
        stacked_noises = jnp.stack(tuple(step_noises), axis=0)

        def body(x_carry, xs):
            c_tau_last, c_tau, step_noise, rotation, translation = xs
            return one_step(
                x_carry, c_tau_last, c_tau, step_noise, rotation, translation
            ), None

        x_l, _ = jax.lax.scan(
            body,
            x_l,
            xs=(
                noise_schedule[:-1],
                noise_schedule[1:],
                stacked_noises,
                rotations,
                translations,
            ),
        )
        return x_l

    for step_index in range(n_steps):
        x_l = one_step(
            x_l,
            noise_schedule[step_index],
            noise_schedule[step_index + 1],
            step_noises[step_index],
            rotations[step_index],
            translations[step_index],
        )
    return x_l
