"""Exact Chai EDM schedule and one-step parity against the Torch formula."""

from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import jax.numpy as jnp  # noqa: E402

from foldjax.models.chai.models.diffusion import (  # noqa: E402
    chai_diffusion_gammas,
    chai_noise_schedule,
    edm_heun_step,
)


def test_chai_midpoint_schedule_matches_upstream_torch() -> None:
    steps = 7
    times = torch.linspace(0, 1, 2 * steps + 1)[1::2]
    expected = 16.0 * (
        times * 4e-4 ** (1 / 7) + (1 - times) * 80.0 ** (1 / 7)
    ) ** 7

    actual = chai_noise_schedule(steps)

    assert actual.shape == (steps,)
    np.testing.assert_allclose(np.asarray(actual), expected.numpy(), rtol=2e-6)
    assert float(actual[-1]) > 0.0


def test_chai_gamma_contract_matches_upstream() -> None:
    sigmas = jnp.asarray([100.0, 80.0, 1.0, 4e-4, 3e-4])
    actual = chai_diffusion_gammas(sigmas, num_timesteps=200)
    expected_gamma = min(80 / 200, math.sqrt(2) - 1)
    np.testing.assert_allclose(
        np.asarray(actual),
        np.asarray([0.0, expected_gamma, expected_gamma, expected_gamma, 0.0]),
    )


def test_edm_heun_step_matches_torch_with_injected_randomness() -> None:
    rng = np.random.default_rng(42)
    coords = rng.normal(size=(2, 5, 3)).astype(np.float32)
    mask = np.asarray([[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]], dtype=bool)
    noise = rng.normal(size=coords.shape).astype(np.float32)
    rotations = np.broadcast_to(np.eye(3, dtype=np.float32), (2, 3, 3)).copy()
    translations = rng.normal(size=(2, 1, 3)).astype(np.float32)
    sigma_curr = 5.0
    sigma_next = 2.0
    gamma = 0.2

    x = torch.from_numpy(coords)
    m = torch.from_numpy(mask)
    centroid = (x * m[..., None]).sum(1, keepdim=True) / m.sum(1)[:, None, None]
    augmented = torch.einsum(
        "bij,baj->bai", torch.from_numpy(rotations), x - centroid
    ) + torch.from_numpy(translations)
    sigma_hat = sigma_curr * (1 + gamma)
    noise_scale = max(sigma_hat**2 - sigma_curr**2, 1e-6) ** 0.5
    atom_hat = augmented + 1.003 * torch.from_numpy(noise) * noise_scale

    def torch_denoise(value, sigma):
        return 0.75 * value + 0.01 * sigma

    denoised = torch_denoise(atom_hat, sigma_hat)
    derivative = (atom_hat - denoised) / sigma_hat
    euler = atom_hat + (sigma_next - sigma_hat) * derivative
    derivative_next = (euler - torch_denoise(euler, sigma_next)) / sigma_next
    expected = euler + (sigma_next - sigma_hat) * (
        (derivative_next + derivative) / 2
    )

    actual = edm_heun_step(
        jnp.asarray(coords),
        jnp.asarray(mask),
        sigma_curr=sigma_curr,
        sigma_next=sigma_next,
        gamma=gamma,
        noise=jnp.asarray(noise),
        rotations=jnp.asarray(rotations),
        translations=jnp.asarray(translations),
        denoise=lambda value, sigma: 0.75 * value + 0.01 * sigma,
    )

    np.testing.assert_allclose(
        np.asarray(actual), expected.numpy(), rtol=1e-5, atol=1e-5
    )


def test_gamma_zero_preserves_upstream_noise_floor() -> None:
    coords = jnp.zeros((1, 2, 3), dtype=jnp.float32)
    noise = jnp.ones_like(coords)
    actual = edm_heun_step(
        coords,
        jnp.ones((1, 2), dtype=bool),
        sigma_curr=100.0,
        sigma_next=90.0,
        gamma=0.0,
        noise=noise,
        rotations=jnp.eye(3)[None],
        translations=jnp.zeros((1, 1, 3)),
        denoise=lambda value, sigma: value,
        second_order=False,
    )
    np.testing.assert_allclose(np.asarray(actual), 1.003e-3, rtol=1e-6)
