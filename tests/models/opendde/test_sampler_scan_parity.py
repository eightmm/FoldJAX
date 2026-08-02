"""Rolling the sampler into `lax.scan` must not change what it computes.

Every other repeated stack in this port is already scanned; the diffusion
sampler was the one that was not, and its step count is a user-facing knob. At
the released 200-step schedule the unrolled form writes 200 copies of the
denoiser into one executable.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.opendde.models.sampling import sample_diffusion


def _denoise(x_noisy: jnp.ndarray, t_hat: jnp.ndarray) -> jnp.ndarray:
    """A cheap stand-in that still depends on both arguments and on position."""
    scale = 1.0 / (1.0 + t_hat[..., None, None])
    return jnp.tanh(x_noisy) * scale + 0.1 * jnp.roll(x_noisy, 1, axis=-2)


@pytest.mark.parametrize("n_steps", [4, 25])
def test_scanned_and_unrolled_samplers_agree(n_steps: int) -> None:
    n_sample, n_atom = 3, 7
    schedule = jnp.linspace(8.0, 0.05, n_steps + 1)
    key = jax.random.PRNGKey(0)

    shared = dict(
        noise_schedule=schedule,
        n_sample=n_sample,
        n_atom=n_atom,
        key=key,
        gamma0=0.8,
        gamma_min=1.0,
    )
    rolled = sample_diffusion(_denoise, use_scan=True, **shared)
    unrolled = sample_diffusion(_denoise, use_scan=False, **shared)

    assert rolled.shape == (n_sample, n_atom, 3)
    # Same arithmetic in a different graph shape: XLA may reassociate, so this
    # is a numerical agreement rather than a bit-for-bit one.
    np.testing.assert_allclose(rolled, unrolled, rtol=2e-5, atol=2e-5)


def test_the_scan_consumes_the_same_random_tape(tmp_path=None) -> None:
    """With explicit tapes both paths must be driven by identical noise."""
    n_sample, n_atom, n_steps = 2, 5, 6
    schedule = jnp.linspace(6.0, 0.1, n_steps + 1)
    rng = np.random.default_rng(7)
    init = jnp.asarray(rng.normal(size=(n_sample, n_atom, 3)), dtype=jnp.float32)
    noises = tuple(
        jnp.asarray(rng.normal(size=(n_sample, n_atom, 3)), dtype=jnp.float32)
        for _ in range(n_steps)
    )
    rotations = jnp.broadcast_to(jnp.eye(3), (n_steps, n_sample, 3, 3))
    translations = jnp.zeros((n_steps, n_sample, 3))

    shared = dict(
        noise_schedule=schedule,
        n_sample=n_sample,
        n_atom=n_atom,
        key=None,
        init_noise=init,
        step_noises=noises,
        rotations=rotations,
        translations=translations,
    )
    np.testing.assert_allclose(
        sample_diffusion(_denoise, use_scan=True, **shared),
        sample_diffusion(_denoise, use_scan=False, **shared),
        rtol=2e-5,
        atol=2e-5,
    )
