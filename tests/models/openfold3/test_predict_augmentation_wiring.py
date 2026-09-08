"""Real predict-to-sampler wiring with inexpensive network boundaries."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.openfold3 import inference
from foldjax.models.openfold3.models.augmentation import (
    AugmentationTape,
    centre_augmentation_from_draws,
)
from foldjax.models.openfold3.models.sampler import sample_diffusion


@pytest.mark.parametrize("taped", [False, True])
def test_real_predict_forwards_augmentation_and_expanded_mask(monkeypatch, taped):
    config = inference.InferenceConfig(
        n_atom=4,
        n_token=2,
        num_samples=3,
        num_steps=2,
        n_query=2,
        n_key=4,
        atom_heads=1,
        token_heads=1,
        no_heads_msa=1,
        no_heads_pair=1,
        no_heads_pair_bias=1,
        max_relative_idx=2,
        max_relative_chain=2,
        num_recycles=1,
        max_atoms_per_token=2,
        plddt_bins=4,
        pae_bins=4,
        pae_bin_max=4.0,
        msa_depth=None,
    )
    batch = {
        "atom_mask": jnp.array([[1.0, 1.0, 0.0, 1.0]]),
        "token_mask": jnp.ones((1, 2)),
    }
    single, pair = jnp.zeros((1, 2, 4)), jnp.zeros((1, 2, 2, 4))
    monkeypatch.setattr(inference, "trunk", lambda *a, **k: (single, single, pair))
    monkeypatch.setattr(inference, "pair_conditioning", lambda *a, **k: pair)
    monkeypatch.setattr(inference, "single_conditioning", lambda *a, **k: single)
    monkeypatch.setattr(inference, "denoise", lambda batch, x, *a, **k: x * 0.25)
    tape = (
        AugmentationTape(
            jnp.tile(jnp.array([1.0, 0.0, 0.0, 0.0]), (2, 3, 1)), jnp.ones((2, 3, 3))
        )
        if taped
        else None
    )
    captured = {}

    class SamplerCompletedError(Exception):
        pass

    def observe(key, schedule, shape, denoiser, **kwargs):
        captured.update(kwargs)
        captured["coordinates"] = sample_diffusion(
            key, schedule, shape, denoiser, **kwargs
        )
        raise SamplerCompletedError

    monkeypatch.setattr(inference, "sample_diffusion", observe)
    params = SimpleNamespace(trunk=None, diffusion_conditioning=None, denoiser=None)
    with pytest.raises(SamplerCompletedError):
        inference.predict(
            jax.random.key(3), batch, params, config, None, augmentation_tape=tape
        )
    assert captured["coordinates"].shape == (3, 4, 3)
    assert np.isfinite(captured["coordinates"]).all()
    if taped:
        assert captured["augment_fn"] is None
        np.testing.assert_array_equal(
            captured["augmentation_tape"].quaternions, tape.quaternions
        )
        np.testing.assert_array_equal(
            captured["atom_mask"], jnp.broadcast_to(batch["atom_mask"], (3, 4))
        )
    else:
        assert callable(captured["augment_fn"])
        assert captured["atom_mask"] is None
        assert captured["augmentation_tape"] is None


@pytest.mark.parametrize("compiled", [False, True])
def test_tape_leaves_ordinary_initial_and_churn_noise_unchanged(compiled, monkeypatch):
    draws = []
    native_normal = jax.random.normal

    def observed_normal(*args, **kwargs):
        value = native_normal(*args, **kwargs)
        jax.debug.callback(
            lambda x: draws.append(np.asarray(x).copy()), value, ordered=True
        )
        return value

    monkeypatch.setattr(jax.random, "normal", observed_normal)
    q = jnp.tile(jnp.array([1.0, 2.0, 3.0, 4.0]), (3, 1))
    t = jnp.arange(9, dtype=jnp.float32).reshape(3, 3) / 10
    mask = jnp.array([[1.0, 1.0, 0.0, 1.0]])
    tape = AugmentationTape(jnp.stack([q, q]), jnp.stack([t, t]))

    def run(key, taped):
        kwargs = (
            {"augmentation_tape": tape, "atom_mask": mask}
            if taped
            else {
                "augment_fn": lambda key, x: centre_augmentation_from_draws(
                    x, mask, q, t
                )
            }
        )
        return sample_diffusion(
            key,
            jnp.array([2.0, 1.0, 0.0]),
            (3, 4, 3),
            lambda x, time: x * 0.25,
            gamma_0=0.8,
            gamma_min=0.1,
            noise_scale=1.003,
            step_scale=1.5,
            **kwargs,
        )

    call = jax.jit(run, static_argnums=1) if compiled else run
    expected = call(jax.random.key(4), False).block_until_ready()
    original_draws = draws.copy()
    draws.clear()
    call(jax.random.key(4), True).block_until_ready()
    assert len(draws) == len(original_draws) == 3
    for taped_draw, original_draw in zip(draws, original_draws, strict=True):
        np.testing.assert_array_equal(taped_draw, original_draw)
    assert not np.array_equal(call(jax.random.key(5), True), expected)
