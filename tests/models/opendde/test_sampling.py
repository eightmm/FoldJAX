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
