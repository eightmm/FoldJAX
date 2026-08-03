from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.opendde.models.sampling import sample_diffusion


def test_sample_diffusion_applies_shared_rigid_tape_before_euler_step() -> None:
    schedule = jnp.asarray([2.0, 0.0], dtype=jnp.float32)
    init_noise = jnp.asarray(
        [[[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]]],
        dtype=jnp.float32,
    )
    step_noises = (jnp.zeros_like(init_noise),)
    rotations = jnp.eye(3, dtype=jnp.float32)[None, None, :, :]
    translations = jnp.asarray([[[1.0, 2.0, 3.0]]], dtype=jnp.float32)

    actual = sample_diffusion(
        lambda x, t: jnp.zeros_like(x),
        schedule,
        n_sample=1,
        n_atom=2,
        key=None,
        init_noise=init_noise,
        step_noises=step_noises,
        rotations=rotations,
        translations=translations,
    )

    initial = 2.0 * np.asarray(init_noise)
    augmented = initial - initial.mean(axis=-2, keepdims=True)
    augmented = augmented + np.asarray(translations[0])[..., None, :]
    expected = -0.5 * augmented
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_sample_diffusion_churns_noise_when_the_level_is_above_gamma_min() -> None:
    """The noise-churn branch, which every other schedule here switches off.

    `gamma = where(c_tau > gamma_min, gamma0, 0)`, and each existing test uses a
    schedule whose *next* level sits at or below `gamma_min`, so gamma is zero
    throughout: `t_hat` collapses to `c_tau_last` and `delta_noise_level` to 0.
    In that regime `gamma0`, the `sqrt(gamma * (gamma + 2))` polynomial and
    `noise_scale_lambda` are all unobservable -- and `t_hat` cancels out of the
    Euler tail as well -- so the whole churn can be wrong and every assertion
    still holds. This picks a schedule that keeps the level above `gamma_min`
    for one step and pins the arithmetic by hand.
    """

    gamma0, gamma_min, noise_scale_lambda, step_scale_eta = 0.8, 1.0, 1.003, 1.5
    c_tau_last, c_tau = 8.0, 4.0
    schedule = jnp.asarray([c_tau_last, c_tau], dtype=jnp.float32)
    init_noise = jnp.asarray([[[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]], dtype=jnp.float32)
    step_noise = jnp.asarray([[[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]]], dtype=jnp.float32)
    rotations = jnp.eye(3, dtype=jnp.float32)[None, None, :, :]
    translations = jnp.zeros((1, 1, 3), dtype=jnp.float32)

    actual = sample_diffusion(
        lambda x, t: jnp.zeros_like(x),
        schedule,
        n_sample=1,
        n_atom=2,
        key=None,
        init_noise=init_noise,
        step_noises=(step_noise,),
        rotations=rotations,
        translations=translations,
        gamma0=gamma0,
        gamma_min=gamma_min,
        noise_scale_lambda=noise_scale_lambda,
        step_scale_eta=step_scale_eta,
    )

    # c_tau (4.0) is above gamma_min, so gamma really is gamma0 here.
    gamma = gamma0
    t_hat = c_tau_last * (1.0 + gamma)
    delta_noise_level = c_tau_last * np.sqrt(gamma * (gamma + 2.0))
    x = c_tau_last * np.asarray(init_noise)
    x = x - x.mean(axis=-2, keepdims=True)
    x_noisy = x + noise_scale_lambda * delta_noise_level * np.asarray(step_noise)
    # denoise_fn returns zeros, so delta = x_noisy / t_hat.
    expected = x_noisy + step_scale_eta * (c_tau - t_hat) * (x_noisy / t_hat)

    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


def test_sample_diffusion_key_path_is_reproducible() -> None:
    schedule = jnp.asarray([2.0, 1.0, 0.0], dtype=jnp.float32)

    first = sample_diffusion(
        lambda x, t: 0.25 * x,
        schedule,
        n_sample=2,
        n_atom=3,
        key=jax.random.key(7),
    )
    second = sample_diffusion(
        lambda x, t: 0.25 * x,
        schedule,
        n_sample=2,
        n_atom=3,
        key=jax.random.key(7),
    )

    assert first.shape == (2, 3, 3)
    np.testing.assert_allclose(first, second, rtol=0.0, atol=0.0)


def test_sample_diffusion_requires_key_or_complete_tapes() -> None:
    schedule = jnp.asarray([2.0, 0.0], dtype=jnp.float32)
    init_noise = jnp.zeros((1, 2, 3), dtype=jnp.float32)

    with pytest.raises(ValueError, match="key or complete random tapes"):
        sample_diffusion(
            lambda x, t: x,
            schedule,
            n_sample=1,
            n_atom=2,
            key=None,
            init_noise=init_noise,
        )
