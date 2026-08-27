"""Torch-vs-JAX parity for the EDM noise schedule and conditioning.

These are weight-free, so there is nothing to randomize; the tests compare exact
arithmetic against upstream and pin the properties an EDM schedule must have.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.openfold3.models.diffusion_schedule import (
    combine_denoiser_output,
    noise_schedule,
    scale_noisy_positions,
)

pytestmark = pytest.mark.torch_parity

SIGMA_DATA = 16.0


def _torch():
    import torch

    torch.manual_seed(0)
    return torch


def _upstream():
    from openfold3.core.model.structure.diffusion_module import create_noise_schedule

    return create_noise_schedule


@pytest.mark.parametrize(
    ("steps", "s_max", "s_min", "p"),
    [
        (200, 160.0, 4e-4, 7),
        (20, 160.0, 4e-4, 7),
        (5, 80.0, 1e-3, 2),
        (50, 160.0, 4e-4, 1),
    ],
)
def test_noise_schedule_matches_torch(
    openfold3_source: Path, steps: int, s_max: float, s_min: float, p: int
) -> None:
    torch = _torch()
    expected = _upstream()(
        num_steps=steps,
        sigma_data=SIGMA_DATA,
        s_max=s_max,
        s_min=s_min,
        p=p,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    actual = noise_schedule(
        steps, sigma_data=SIGMA_DATA, s_max=s_max, s_min=s_min, p=p
    )
    assert actual.shape == (steps + 1,)
    # Schedule values reach O(1e3), where one fp32 ulp is already ~1e-4, and the
    # p-power is evaluated in a slightly different order by torch and JAX. This
    # is a relative comparison for that reason; the observed worst case is 5e-6.
    np.testing.assert_allclose(
        np.asarray(actual, dtype=np.float64),
        expected.numpy().astype(np.float64),
        rtol=1e-5,
        atol=1e-5,
    )


def test_noise_schedule_is_monotonically_decreasing(openfold3_source: Path) -> None:
    """A schedule that is not descending would break the sampler's rollout."""
    schedule = np.asarray(
        noise_schedule(50, sigma_data=SIGMA_DATA, s_max=160.0, s_min=4e-4, p=7)
    )
    assert np.all(np.diff(schedule) < 0)
    # Endpoints are sigma_data * s_max and sigma_data * s_min.
    assert schedule[0] == pytest.approx(SIGMA_DATA * 160.0, rel=1e-5)
    assert schedule[-1] == pytest.approx(SIGMA_DATA * 4e-4, rel=1e-4)


def test_p_controls_where_steps_are_spent(openfold3_source: Path) -> None:
    """Larger p front-loads coarse noise, spending more steps near s_min."""
    low_p = np.asarray(
        noise_schedule(20, sigma_data=SIGMA_DATA, s_max=160.0, s_min=4e-4, p=1)
    )
    high_p = np.asarray(
        noise_schedule(20, sigma_data=SIGMA_DATA, s_max=160.0, s_min=4e-4, p=7)
    )
    midpoint = len(low_p) // 2
    assert high_p[midpoint] < low_p[midpoint]


def test_input_scaling_matches_upstream_expression(openfold3_source: Path) -> None:
    torch = _torch()
    xl = torch.randn(2, 7, 3)
    t = torch.tensor([1.0, 40.0])
    expected = xl / torch.sqrt(t[..., None, None] ** 2 + SIGMA_DATA**2)
    actual = scale_noisy_positions(
        jnp.asarray(xl.numpy()), jnp.asarray(t.numpy()), sigma_data=SIGMA_DATA
    )
    np.testing.assert_allclose(
        np.asarray(actual, dtype=np.float64),
        expected.numpy().astype(np.float64),
        rtol=1e-6,
        atol=1e-6,
    )


def test_output_blend_matches_upstream_expression(openfold3_source: Path) -> None:
    torch = _torch()
    xl = torch.randn(2, 7, 3)
    update = torch.randn(2, 7, 3)
    t = torch.tensor([0.5, 100.0])
    expected = (
        SIGMA_DATA**2 / (SIGMA_DATA**2 + t[..., None, None] ** 2) * xl
        + SIGMA_DATA
        * t[..., None, None]
        / torch.sqrt(SIGMA_DATA**2 + t[..., None, None] ** 2)
        * update
    )
    actual = combine_denoiser_output(
        jnp.asarray(xl.numpy()),
        jnp.asarray(update.numpy()),
        jnp.asarray(t.numpy()),
        sigma_data=SIGMA_DATA,
    )
    np.testing.assert_allclose(
        np.asarray(actual, dtype=np.float64),
        expected.numpy().astype(np.float64),
        rtol=1e-6,
        atol=1e-6,
    )


def test_blend_limits_are_input_at_low_noise_and_prediction_at_high(
    openfold3_source: Path,
) -> None:
    """Returning the raw prediction would fail this; that is the classic bug."""
    xl = jnp.asarray(np.random.default_rng(0).normal(size=(1, 5, 3)))
    update = jnp.asarray(np.random.default_rng(1).normal(size=(1, 5, 3)))

    near_zero = combine_denoiser_output(
        xl, update, jnp.asarray([1e-6]), sigma_data=SIGMA_DATA
    )
    np.testing.assert_allclose(
        np.asarray(near_zero), np.asarray(xl), rtol=1e-5, atol=1e-5
    )

    # At sigma >> sigma_data the prediction term dominates and grows like sigma.
    huge = combine_denoiser_output(
        xl, update, jnp.asarray([1e6]), sigma_data=SIGMA_DATA
    )
    np.testing.assert_allclose(
        np.asarray(huge), np.asarray(update) * SIGMA_DATA, rtol=1e-3, atol=1e-3
    )
