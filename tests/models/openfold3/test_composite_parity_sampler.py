"""Sampler-rollout parity against upstream's ``SampleDiffusion.forward``.

The rollout was the last composite path with no upstream comparison: its own tests
check schedule arithmetic and determinism, but nothing checked it against
Algorithm 18 as upstream implements it. The obstacle is that both draw random
tensors -- the initial coordinates, the per-step injection, and the random
augmentation -- and the two PRNG streams cannot be matched.

The fix is to remove the randomness rather than to skip the comparison: the same
pre-generated noise is injected into both (``noise_fn`` here, a patched
``torch.randn``/``randn_like`` there) and augmentation is made the identity on both
sides. What remains is the loop arithmetic -- gamma gating, the inflated ``t``, the
step from ``xl_noisy`` rather than ``xl``, and ``step_scale`` -- which is exactly
what was ungated.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.openfold3.models.diffusion_schedule import noise_schedule
from foldjax.models.openfold3.models.sampler import sample_diffusion

pytestmark = pytest.mark.torch_parity

RTOL = 1e-4
ATOL = 1e-4

N_SAMPLE, N_ATOM, STEPS = 2, 7, 5
KW = {
    "gamma_0": 0.8,
    "gamma_min": 1.0,
    "noise_scale": 1.003,
    "step_scale": 1.5,
}
SCHEDULE_KW = {"s_max": 160.0, "s_min": 4e-4, "p": 7, "sigma_data": 16.0}


def test_rollout_matches_upstream(openfold3_source: Path, monkeypatch) -> None:
    import torch
    from openfold3.core.model.structure import diffusion_module as upstream_module

    generator = torch.Generator().manual_seed(7)
    shape = (1, N_SAMPLE, N_ATOM, 3)
    # One draw for the initial coordinates, then one per rollout step.
    draws = [torch.randn(shape, generator=generator) for _ in range(STEPS + 1)]

    schedule = noise_schedule(STEPS, **SCHEDULE_KW)
    torch_schedule = torch.tensor(np.asarray(schedule), dtype=torch.float32)

    # A denoiser that is deterministic and depends on both arguments, so a wrong
    # t or a wrong xl_noisy cannot cancel out.
    def torch_denoise(xl_noisy: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return 0.3 * xl_noisy + 0.05 * t.reshape(-1)[0] - 0.1

    def jax_denoise(xl_noisy: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        return 0.3 * xl_noisy + 0.05 * t.reshape(-1)[0] - 0.1

    class _Module(torch.nn.Module):
        def forward(self, *, xl_noisy, t, **_kwargs):
            return torch_denoise(xl_noisy, t)

    sampler = upstream_module.SampleDiffusion(diffusion_module=_Module(), **KW)

    remaining = list(draws)

    def fake_randn(*_args, **_kwargs):
        return remaining.pop(0)

    # Both sides get the same stream and no augmentation, so only the rollout
    # arithmetic is under test.
    monkeypatch.setattr(upstream_module.torch, "randn", fake_randn)
    monkeypatch.setattr(upstream_module.torch, "randn_like", fake_randn)
    monkeypatch.setattr(
        upstream_module, "centre_random_augmentation", lambda xl, atom_mask: xl
    )

    batch = {
        "atom_mask": torch.ones((1, N_ATOM)),
        "token_mask": torch.ones((1, N_ATOM)),
    }
    with torch.no_grad():
        expected = sampler(
            batch=batch,
            noise_schedule=torch_schedule,
            si_input=None,
            si_trunk=None,
            zij_trunk=None,
            no_rollout_samples=N_SAMPLE,
            use_conditioning=True,
        )
    assert not remaining, "upstream drew a different number of noise tensors"

    actual = sample_diffusion(
        jax.random.key(0),
        schedule,
        shape,
        jax_denoise,
        augment_fn=None,
        noise_fn=lambda step, _shape: jnp.asarray(draws[step].numpy()),
        **KW,
    )
    assert actual.shape == tuple(expected.shape)
    np.testing.assert_allclose(
        np.asarray(actual, dtype=np.float64),
        expected.detach().numpy().astype(np.float64),
        rtol=RTOL,
        atol=ATOL,
        err_msg="sampler rollout diverged from upstream SampleDiffusion",
    )


def test_the_comparison_is_sensitive_to_step_scale(
    openfold3_source: Path, monkeypatch
) -> None:
    """Guards the gate: a wrong step_scale must change the result."""
    import torch

    generator = torch.Generator().manual_seed(7)
    shape = (1, N_SAMPLE, N_ATOM, 3)
    draws = [torch.randn(shape, generator=generator) for _ in range(STEPS + 1)]
    schedule = noise_schedule(STEPS, **SCHEDULE_KW)

    def run(step_scale: float):
        return sample_diffusion(
            jax.random.key(0),
            schedule,
            shape,
            lambda xl, t: 0.3 * xl + 0.05 * t.reshape(-1)[0] - 0.1,
            augment_fn=None,
            noise_fn=lambda step, _shape: jnp.asarray(draws[step].numpy()),
            **{**KW, "step_scale": step_scale},
        )

    assert not np.allclose(np.asarray(run(1.5)), np.asarray(run(1.0)))
