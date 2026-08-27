from __future__ import annotations

import inspect

import jax
import jax.numpy as jnp
import numpy as np

from foldjax.models.protenix.models.diffusion.diffusion import (
    inference_noise_schedule,
    sample_diffusion,
)
from foldjax.models.protenix.models.model import protenix_infer_static
from foldjax.models.protenix.models.predict import protenix_predict_static


def test_production_inference_defaults_to_sampler_scan() -> None:
    assert (
        inspect.signature(protenix_infer_static).parameters["use_sampler_scan"].default
        is True
    )
    assert (
        inspect.signature(protenix_predict_static)
        .parameters["use_sampler_scan"]
        .default
        is True
    )


def test_slow_scan_and_unhelpful_fusion_are_disabled_by_default() -> None:
    infer_signature = inspect.signature(protenix_infer_static).parameters
    predict_signature = inspect.signature(protenix_predict_static).parameters

    assert infer_signature["use_confidence_scan"].default is False
    assert predict_signature["use_confidence_scan"].default is False
    assert infer_signature["use_diffusion_efficient_fusion"].default is False
    assert predict_signature["use_diffusion_efficient_fusion"].default is False


def test_inference_noise_schedule_matches_protenix_formula() -> None:
    schedule = inference_noise_schedule(
        num_steps=4,
        s_max=10.0,
        s_min=0.1,
        rho=2.0,
        sigma_data=3.0,
        dtype=jnp.float32,
    )
    indices = np.arange(5, dtype=np.float32)
    expected = 3.0 * (10.0**0.5 + indices / 4.0 * (0.1**0.5 - 10.0**0.5)) ** 2
    expected[-1] = 0.0

    np.testing.assert_allclose(np.asarray(schedule), expected, rtol=1e-6, atol=1e-6)


def test_sample_diffusion_zero_denoiser_matches_euler_formula() -> None:
    noise_schedule = jnp.asarray([2.0, 1.0], dtype=jnp.float32)
    init_noise = jnp.asarray([[[[1.0, -1.0, 0.5], [0.0, 2.0, -2.0]]]])
    step_noise = jnp.zeros_like(init_noise)

    def zero_denoiser(x_noisy, t_hat):
        assert x_noisy.shape == init_noise.shape
        assert t_hat.shape == (1, 1)
        return jnp.zeros_like(x_noisy)

    out = sample_diffusion(
        zero_denoiser,
        noise_schedule,
        num_samples=1,
        n_atom=2,
        key=None,
        init_noise=init_noise,
        step_noises=(step_noise,),
        gamma0=0.5,
        gamma_min=0.5,
        noise_scale_lambda=1.0,
        step_scale_eta=1.5,
        centre_each_step=False,
    )

    x_l = 2.0 * np.asarray(init_noise)
    t_hat = 2.0 * 1.5
    x_noisy = x_l
    delta = x_noisy / t_hat
    expected = x_noisy + 1.5 * (1.0 - t_hat) * delta
    np.testing.assert_allclose(np.asarray(out), expected, rtol=1e-6, atol=1e-6)


def test_sample_diffusion_chunks_samples() -> None:
    noise_schedule = jnp.asarray([1.0, 0.0], dtype=jnp.float32)
    init_noise = jnp.ones((3, 2, 3), dtype=jnp.float32)
    step_noise = jnp.zeros_like(init_noise)

    def identity_denoiser(x_noisy, t_hat):
        del t_hat
        return x_noisy

    out = sample_diffusion(
        identity_denoiser,
        noise_schedule,
        num_samples=3,
        n_atom=2,
        key=None,
        init_noise=init_noise,
        step_noises=(step_noise,),
        diffusion_chunk_size=2,
        centre_each_step=False,
    )

    assert out.shape == (3, 2, 3)
    np.testing.assert_allclose(np.asarray(out), np.asarray(init_noise), atol=1e-6)


def test_sample_diffusion_scan_matches_loop_with_injected_noise() -> None:
    noise_schedule = jnp.asarray([2.0, 1.0, 0.25, 0.0], dtype=jnp.float32)
    init_noise = jnp.arange(18, dtype=jnp.float32).reshape(2, 3, 3) / 10.0
    step_noises = tuple(
        jnp.full_like(init_noise, value) for value in (0.25, -0.5, 0.75)
    )

    def shrink_denoiser(x_noisy, t_hat):
        return x_noisy / (1.0 + t_hat[..., None, None])

    kwargs = {
        "num_samples": 2,
        "n_atom": 3,
        "key": None,
        "init_noise": init_noise,
        "step_noises": step_noises,
        "gamma0": 0.4,
        "gamma_min": 0.5,
        "noise_scale_lambda": 1.003,
        "step_scale_eta": 1.5,
        "centre_each_step": True,
    }
    loop = sample_diffusion(
        shrink_denoiser,
        noise_schedule,
        use_scan=False,
        **kwargs,
    )
    scanned = sample_diffusion(
        shrink_denoiser,
        noise_schedule,
        use_scan=True,
        **kwargs,
    )

    np.testing.assert_allclose(scanned, loop, rtol=1e-6, atol=1e-6)


def test_disabled_guidance_is_bitwise_identical() -> None:
    schedule = jnp.asarray([2.0, 0.5, 0.0], dtype=jnp.float32)
    init_noise = jnp.arange(18, dtype=jnp.float32).reshape(2, 3, 3) / 10
    noises = (jnp.full_like(init_noise, 0.2), jnp.full_like(init_noise, -0.3))

    def denoiser(x_noisy, t_hat):
        return x_noisy / (1.0 + t_hat[..., None, None])

    kwargs = dict(
        num_samples=2,
        n_atom=3,
        key=None,
        init_noise=init_noise,
        step_noises=noises,
        centre_each_step=False,
    )
    baseline = sample_diffusion(denoiser, schedule, **kwargs)
    disabled = sample_diffusion(
        denoiser,
        schedule,
        guidance_config={"enable": False},
        guidance_features={},
        **kwargs,
    )
    np.testing.assert_array_equal(disabled, baseline)

    random_baseline = sample_diffusion(
        denoiser,
        schedule,
        num_samples=2,
        n_atom=3,
        key=jax.random.key(11),
        centre_each_step=False,
    )
    random_disabled = sample_diffusion(
        denoiser,
        schedule,
        num_samples=2,
        n_atom=3,
        key=jax.random.key(11),
        centre_each_step=False,
        guidance_config={"enable": False},
        guidance_features={},
    )
    np.testing.assert_array_equal(random_disabled, random_baseline)


def test_enabled_guidance_changes_sampler_result() -> None:
    schedule = jnp.asarray([2.0, 0.0], dtype=jnp.float32)
    init_noise = jnp.asarray([[[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]]], dtype=jnp.float32)
    kwargs = dict(
        num_samples=1,
        n_atom=2,
        key=None,
        init_noise=init_noise,
        step_noises=(jnp.zeros_like(init_noise),),
        gamma0=0.0,
        centre_each_step=False,
    )
    baseline = sample_diffusion(lambda x, _: x, schedule, **kwargs)
    guided = sample_diffusion(
        lambda x, _: x,
        schedule,
        guidance_config={
            "enable": True,
            "mu": 0.1,
            "steps": {"tfg_inner": 1, "projection_outer": 0},
            "terms": {"InterchainBondPotential": {"weight": 1.0, "buffer": 1.0}},
        },
        guidance_features={
            "interchain_bond_index": jnp.asarray([[0], [1]], dtype=jnp.int32)
        },
        **kwargs,
    )
    assert not np.array_equal(guided, baseline)


def test_guidance_mc_sampler_is_deterministic_for_key() -> None:
    schedule = jnp.asarray([2.0, 0.0], dtype=jnp.float32)
    config = {
        "enable": True,
        "mu": 0.1,
        "mc": {"std": 0.25, "batch": 4},
        "steps": {"tfg_inner": 1, "projection_outer": 0},
        "terms": {"InterchainBondPotential": {"weight": 1.0, "buffer": 1.0}},
    }
    kwargs = dict(
        denoise_fn=lambda x, _: x,
        noise_schedule=schedule,
        num_samples=1,
        n_atom=2,
        init_noise=jnp.asarray([[[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]]]),
        step_noises=(jnp.zeros((1, 2, 3), dtype=jnp.float32),),
        gamma0=0.0,
        centre_each_step=False,
        guidance_config=config,
        guidance_features={
            "interchain_bond_index": jnp.asarray([[0], [1]], dtype=jnp.int32)
        },
    )
    first = sample_diffusion(key=jax.random.key(3), **kwargs)
    repeated = sample_diffusion(key=jax.random.key(3), **kwargs)
    other = sample_diffusion(key=jax.random.key(4), **kwargs)
    np.testing.assert_array_equal(first, repeated)
    assert not np.array_equal(first, other)
