"""The Algorithm 18 noise-level step must survive XLA's fusion.

Raising the noise level from `c` to `t_hat = c * (gamma + 1)` adds
`sqrt(t_hat**2 - c**2)`. Written literally that is a difference of two nearly
equal squares which is *exactly* zero whenever gamma is zero — every step once
the noise level falls below `gamma_min`. Under `lax.scan`, XLA contracts the
expression and the cancellation leaves rounding noise: on CPU at num_steps=20 one
step produced -4.3e-10, and `sqrt` of that is NaN, which then poisoned every
coordinate and confidence score. GPU fused differently and happened not to cross
zero, so this presented as a CPU-only fault.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.protenix.models.diffusion.diffusion import (
    _sample_diffusion_chunk,
    inference_noise_schedule,
    sample_diffusion,
)

GAMMA0 = 0.8
GAMMA_MIN = 1.0


def _delta_noise_level(c_last, c_tau):
    """The expression as the sampler now computes it."""
    gamma = jnp.where(c_tau > GAMMA_MIN, GAMMA0, 0.0).astype(jnp.float32)
    return c_last * jnp.sqrt(gamma * (gamma + 2.0))


@pytest.mark.parametrize("num_steps", [5, 8, 20, 50, 200])
def test_noise_level_step_is_finite_under_scan(num_steps: int) -> None:
    schedule = inference_noise_schedule(num_steps=num_steps)

    def body(carry, xs):
        return carry, _delta_noise_level(xs[0], xs[1])

    _, values = jax.lax.scan(body, 0.0, (schedule[:-1], schedule[1:]))
    values = np.asarray(values)
    bad = np.flatnonzero(~np.isfinite(values))
    assert np.isfinite(values).all(), f"non-finite at steps {bad}"
    assert (values >= 0).all()


def test_it_is_exactly_zero_when_no_noise_is_added() -> None:
    """gamma == 0 means Algorithm 18 adds nothing; 1e-4 of noise is not nothing."""
    schedule = inference_noise_schedule(num_steps=20)

    def body(carry, xs):
        return carry, _delta_noise_level(xs[0], xs[1])

    _, values = jax.lax.scan(body, 0.0, (schedule[:-1], schedule[1:]))
    quiet = np.asarray(schedule[1:]) <= GAMMA_MIN
    assert quiet.any(), "schedule no longer reaches the no-churn regime"
    assert (np.asarray(values)[quiet] == 0.0).all()


def test_it_still_matches_the_algebraic_form_where_noise_is_added() -> None:
    """Equal to sqrt(t_hat**2 - c**2) wherever that form is well conditioned."""
    schedule = inference_noise_schedule(num_steps=20)
    loud = np.asarray(schedule[1:]) > GAMMA_MIN
    c_last = np.asarray(schedule[:-1])[loud]

    reference = np.sqrt((c_last * (GAMMA0 + 1.0)) ** 2 - c_last**2)
    actual = c_last * np.sqrt(GAMMA0 * (GAMMA0 + 2.0))
    np.testing.assert_allclose(actual, reference, rtol=1e-6)


@pytest.mark.parametrize("use_scan", [False, True])
def test_the_sampler_returns_finite_coordinates(use_scan: bool) -> None:
    """End to end through the sampler, with a denoiser that cannot itself NaN."""
    schedule = inference_noise_schedule(num_steps=20)
    coordinates = _sample_diffusion_chunk(
        lambda x_noisy, t_hat: x_noisy * 0.5,
        schedule,
        num_samples=1,
        n_atom=8,
        key=jax.random.PRNGKey(0),
        init_noise=None,
        step_noises=None,
        gamma0=GAMMA0,
        gamma_min=GAMMA_MIN,
        noise_scale_lambda=1.003,
        step_scale_eta=1.5,
        dtype=jnp.float32,
        centre_each_step=True,
        use_scan=use_scan,
        guidance_config=None,
        guidance_features=None,
    )
    assert np.isfinite(np.asarray(coordinates)).all()


def test_scan_and_loop_sampler_paths_agree() -> None:
    """They differed only because one of them was producing NaN."""
    schedule = inference_noise_schedule(num_steps=20)
    key = jax.random.PRNGKey(3)
    shared = dict(
        num_samples=1, n_atom=6, key=key, gamma0=GAMMA0, gamma_min=GAMMA_MIN,
        centre_each_step=False,
    )
    denoise = lambda x_noisy, t_hat: x_noisy * 0.25  # noqa: E731
    looped = np.asarray(sample_diffusion(denoise, schedule, use_scan=False, **shared))
    scanned = np.asarray(sample_diffusion(denoise, schedule, use_scan=True, **shared))
    assert np.isfinite(looped).all() and np.isfinite(scanned).all()
    np.testing.assert_allclose(scanned, looped, rtol=1e-5, atol=1e-5)
