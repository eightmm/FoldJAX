from __future__ import annotations

import inspect
from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models._stacking import StackedLayers
from foldjax.models.protenix.models import model as protenix_model
from foldjax.models.protenix.models.diffusion import diffusion as protenix_diffusion
from foldjax.models.protenix.models.diffusion.atom import (
    AtomAttentionDecoderParams,
    AtomAttentionEncoderCacheParams,
    AtomAttentionEncoderParams,
    AtomPairMlpParams,
    rearrange_qk_to_dense_trunk,
)
from foldjax.models.protenix.models.diffusion.diffusion import (
    DiffusionConditioningParams,
    DiffusionModuleParams,
    diffusion_conditioning_prepare_cache,
    inference_noise_schedule,
    sample_diffusion,
    sample_diffusion_with_module,
)
from foldjax.models.protenix.models.diffusion.transformer import (
    ConditionedTransitionParams,
    DiffusionTransformerBlockParams,
    DiffusionTransformerStackParams,
)
from foldjax.models.protenix.models.input_precision import (
    native_diffusion_autocast_params,
)
from foldjax.models.protenix.models.model import (
    protenix_infer_compiled,
    protenix_infer_static,
)
from foldjax.models.protenix.models.predict import protenix_predict_static
from foldjax.models.protenix.models.primitives.attention import (
    AttentionPairBiasParams,
    AttentionParams,
)
from foldjax.models.protenix.models.primitives.primitives import (
    AdaptiveLayerNormParams,
    LayerNormParams,
    LinearParams,
    TransitionParams,
)
from foldjax.models.protenix.models.trunk_blocks.embedders import (
    FourierParams,
    RelativePositionParams,
)


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


def test_the_slow_confidence_scan_is_disabled_by_default() -> None:
    infer_signature = inspect.signature(protenix_infer_static).parameters
    predict_signature = inspect.signature(protenix_predict_static).parameters

    assert infer_signature["use_confidence_scan"].default is False
    assert predict_signature["use_confidence_scan"].default is False


def test_the_two_entry_points_agree_on_the_fusion_default() -> None:
    """Both say what upstream's released inference config says.

    Asserted as an equivalence against the upstream value rather than as a
    flipped literal: a test that pins a spelling passes a rename and fails a
    behaviour change it should have allowed. What matters is that the two
    entry points cannot disagree, and that they follow
    `protenix/configs/configs_inference.py:33` rather than the training base
    at `configs_base.py:132`, which is where the port's old default came from.
    """
    infer_signature = inspect.signature(protenix_infer_static).parameters
    predict_signature = inspect.signature(protenix_predict_static).parameters
    upstream_inference_value = True

    assert (
        infer_signature["use_diffusion_efficient_fusion"].default
        == predict_signature["use_diffusion_efficient_fusion"].default
        == upstream_inference_value
    )


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
        "rotations": jnp.broadcast_to(jnp.eye(3), (3, 2, 3, 3)),
        "translations": jnp.zeros((3, 2, 3), dtype=jnp.float32),
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


def _rand_linear(
    rng: np.random.Generator, out_dim: int, in_dim: int, *, with_bias: bool = False
) -> LinearParams:
    scale = 0.25 / np.sqrt(max(in_dim, 1))
    weight = jnp.asarray(rng.normal(scale=scale, size=(out_dim, in_dim)), jnp.float32)
    bias = (
        jnp.asarray(rng.normal(scale=0.05, size=(out_dim,)), jnp.float32)
        if with_bias
        else None
    )
    return LinearParams(weight=weight, bias=bias)


def _rand_norm(
    rng: np.random.Generator, dim: int, *, with_bias: bool = True
) -> LayerNormParams:
    weight = jnp.asarray(1.0 + rng.normal(scale=0.05, size=(dim,)), jnp.float32)
    bias = (
        jnp.asarray(rng.normal(scale=0.02, size=(dim,)), jnp.float32)
        if with_bias
        else None
    )
    return LayerNormParams(weight=weight, bias=bias)


def _rand_transition(rng: np.random.Generator, dim: int) -> TransitionParams:
    hidden = 2 * dim
    return TransitionParams(
        layer_norm=_rand_norm(rng, dim, with_bias=False),
        linear_a=_rand_linear(rng, hidden, dim),
        linear_b=_rand_linear(rng, hidden, dim),
        linear_out=_rand_linear(rng, dim, hidden),
    )


def _rand_transformer_block(
    rng: np.random.Generator,
    *,
    c_a: int,
    c_s: int,
    c_z: int,
    heads: int,
    cross_attention: bool,
    layernorm_z_bias: bool = False,
) -> DiffusionTransformerBlockParams:
    def make_adaln() -> AdaptiveLayerNormParams:
        return AdaptiveLayerNormParams(
            layernorm_a=LayerNormParams(weight=None, bias=None),
            layernorm_s=_rand_norm(rng, c_s, with_bias=False),
            linear_s=_rand_linear(rng, c_a, c_s, with_bias=True),
            linear_no_bias_s=_rand_linear(rng, c_a, c_s),
        )

    adaln_a = make_adaln()
    adaln_kv = make_adaln() if cross_attention else None
    return DiffusionTransformerBlockParams(
        attention_pair_bias=AttentionPairBiasParams(
            layernorm_a=adaln_a,
            layernorm_kv=adaln_kv,
            attention=AttentionParams(
                linear_q=_rand_linear(rng, c_a, c_a, with_bias=True),
                linear_k=_rand_linear(rng, c_a, c_a),
                linear_v=_rand_linear(rng, c_a, c_a),
                linear_o=_rand_linear(rng, c_a, c_a),
                linear_g=_rand_linear(rng, c_a, c_a),
            ),
            layernorm_z=_rand_norm(rng, c_z, with_bias=layernorm_z_bias),
            linear_z=_rand_linear(rng, heads, c_z),
            linear_a_last=_rand_linear(rng, c_a, c_s, with_bias=True),
            has_s=True,
            cross_attention_mode=cross_attention,
        ),
        conditioned_transition=ConditionedTransitionParams(
            adaln=make_adaln(),
            linear_a1=_rand_linear(rng, 2 * c_a, c_a),
            linear_a2=_rand_linear(rng, 2 * c_a, c_a),
            linear_b=_rand_linear(rng, c_a, 2 * c_a),
            linear_s=_rand_linear(rng, c_a, c_s, with_bias=True),
        ),
    )


def _protenix_pair_bias_cache_fixture(
    seed: int = 19,
    *,
    n_blocks: int = 3,
    heads: int = 2,
    stacked: bool = False,
) -> dict[str, Any]:
    """Tiny random multi-block Protenix diffusion fixture with padding masks."""
    rng = np.random.default_rng(seed)
    n_samples = 2
    n_token = 5
    n_atom = 7
    n_queries = 2
    n_keys = 4
    c_s_inputs, c_s_trunk, c_z_trunk = 5, 6, 4
    c_s, c_z, c_atom, c_atompair, c_tok = 4, 4, 8, 4, 8

    conditioning = DiffusionConditioningParams(
        relpe=RelativePositionParams(linear_no_bias=_rand_linear(rng, c_z, 4)),
        layernorm_z=_rand_norm(rng, c_z_trunk + c_z),
        linear_z=_rand_linear(rng, c_z, c_z_trunk + c_z),
        transition_z1=_rand_transition(rng, c_z),
        transition_z2=_rand_transition(rng, c_z),
        layernorm_s=_rand_norm(rng, c_s_trunk + c_s_inputs),
        linear_s=_rand_linear(rng, c_s, c_s_trunk + c_s_inputs),
        fourier=FourierParams(
            w=jnp.asarray(rng.normal(size=(4,)), jnp.float32),
            b=jnp.asarray(rng.normal(size=(4,)), jnp.float32),
        ),
        layernorm_n=_rand_norm(rng, 4),
        linear_n=_rand_linear(rng, c_s, 4),
        transition_s1=_rand_transition(rng, c_s),
        transition_s2=_rand_transition(rng, c_s),
    )
    enc_block = _rand_transformer_block(
        rng, c_a=c_atom, c_s=c_atom, c_z=c_atompair, heads=heads, cross_attention=True
    )
    tok_blocks: Any = tuple(
        _rand_transformer_block(
            rng,
            c_a=c_tok,
            c_s=c_s,
            c_z=c_z,
            heads=heads,
            cross_attention=False,
            layernorm_z_bias=True,
        )
        for _ in range(n_blocks)
    )
    if stacked:
        tok_blocks = StackedLayers.from_layers(tok_blocks)
    dec_block = _rand_transformer_block(
        rng, c_a=c_atom, c_s=c_atom, c_z=c_atompair, heads=heads, cross_attention=True
    )
    atom_encoder = AtomAttentionEncoderParams(
        cache=AtomAttentionEncoderCacheParams(
            linear_ref_pos=_rand_linear(rng, c_atom, 3),
            linear_ref_charge=_rand_linear(rng, c_atom, 1),
            linear_f=_rand_linear(rng, c_atom, 385),
            linear_d=_rand_linear(rng, c_atompair, 3),
            linear_invd=_rand_linear(rng, c_atompair, 1),
            linear_v=_rand_linear(rng, c_atompair, 1),
        ),
        linear_cl=_rand_linear(rng, c_atompair, c_atom),
        linear_cm=_rand_linear(rng, c_atompair, c_atom),
        small_mlp=AtomPairMlpParams(
            linear_1=_rand_linear(rng, c_atompair, c_atompair),
            linear_2=_rand_linear(rng, c_atompair, c_atompair),
            linear_3=_rand_linear(rng, c_atompair, c_atompair),
        ),
        atom_transformer=DiffusionTransformerStackParams(blocks=(enc_block,)),
        linear_q=_rand_linear(rng, c_tok, c_atom),
        layernorm_s=_rand_norm(rng, c_s_trunk),
        linear_s=_rand_linear(rng, c_atom, c_s_trunk),
        layernorm_z=_rand_norm(rng, c_z),
        linear_z=_rand_linear(rng, c_atompair, c_z),
        linear_r=_rand_linear(rng, c_atom, 3),
    )
    diffusion_params = DiffusionModuleParams(
        conditioning=conditioning,
        atom_encoder=atom_encoder,
        layernorm_s=_rand_norm(rng, c_s),
        linear_s=_rand_linear(rng, c_tok, c_s),
        diffusion_transformer=DiffusionTransformerStackParams(blocks=tok_blocks),
        layernorm_a=_rand_norm(rng, c_tok),
        atom_decoder=AtomAttentionDecoderParams(
            linear_a=_rand_linear(rng, c_atom, c_tok),
            layernorm_q=_rand_norm(rng, c_atom),
            linear_out=_rand_linear(rng, 3, c_atom),
            atom_transformer=DiffusionTransformerStackParams(blocks=(dec_block,)),
        ),
    )

    atom_to_token_idx = jnp.asarray([0, 0, 1, 2, 2, 3, 4], dtype=jnp.int32)
    token_mask = jnp.asarray([True, True, True, False, False])
    atom_mask = jnp.asarray([True, True, False, True, True, False, False])
    ref_pos = jnp.asarray(rng.normal(size=(n_atom, 3)), jnp.float32)
    ref_charge = jnp.asarray(rng.normal(size=(n_atom,)), jnp.float32)
    ref_mask = atom_mask.astype(jnp.float32)
    ref_atom_name_chars = jnp.zeros((n_atom, 4, 64), dtype=jnp.float32)
    ref_element = jnp.zeros((n_atom, 128), dtype=jnp.float32)
    pos_q, pos_k, pad_info = rearrange_qk_to_dense_trunk(
        ref_pos, ref_pos, n_queries=n_queries, n_keys=n_keys
    )
    d_lm = pos_q[:, :, None, :] - pos_k[:, None, :, :]
    id_q, id_k, _ = rearrange_qk_to_dense_trunk(
        atom_to_token_idx[:, None],
        atom_to_token_idx[:, None],
        n_queries=n_queries,
        n_keys=n_keys,
        compute_mask=False,
    )
    v_lm = (id_q[:, :, None, 0] == id_k[:, None, :, 0])[..., None].astype(jnp.float32)
    relp = jnp.asarray(rng.normal(size=(n_token, n_token, 4)), jnp.float32)
    s_inputs = jnp.asarray(rng.normal(size=(n_token, c_s_inputs)), jnp.float32)
    s_trunk = jnp.asarray(rng.normal(size=(n_token, c_s_trunk)), jnp.float32)
    z_trunk = jnp.asarray(rng.normal(size=(n_token, n_token, c_z_trunk)), jnp.float32)
    pair_z = diffusion_conditioning_prepare_cache(relp, z_trunk, conditioning)
    features = {
        "atom_to_token_idx": atom_to_token_idx,
        "ref_pos": ref_pos,
        "ref_charge": ref_charge,
        "ref_mask": ref_mask,
        "ref_atom_name_chars": ref_atom_name_chars,
        "ref_element": ref_element,
        "d_lm": d_lm,
        "v_lm": v_lm,
        "pad_info": pad_info,
        "restype": jnp.zeros((n_token, 32), dtype=jnp.float32),
        "profile": jnp.zeros((n_token, 32), dtype=jnp.float32),
        "deletion_mean": jnp.zeros((n_token,), dtype=jnp.float32),
        "relp": relp,
        "token_padding_mask": token_mask,
        "atom_padding_mask": atom_mask,
    }
    init_noise = (
        jnp.asarray(rng.normal(size=(n_samples, n_atom, 3)), jnp.float32)
        * atom_mask[None, :, None].astype(jnp.float32)
    )
    schedule = jnp.asarray([2.0, 1.0, 0.5, 0.0], dtype=jnp.float32)
    step_noises = tuple(
        jnp.asarray(rng.normal(scale=0.1, size=(n_samples, n_atom, 3)), jnp.float32)
        for _ in range(3)
    )
    rotations = jnp.broadcast_to(jnp.eye(3, dtype=jnp.float32), (3, n_samples, 3, 3))
    translations = jnp.zeros((3, n_samples, 3), dtype=jnp.float32)
    return {
        "features": features,
        "s_inputs": s_inputs,
        "s_trunk": s_trunk,
        "z_trunk": z_trunk,
        "pair_z": pair_z,
        "diffusion_params": diffusion_params,
        "schedule": schedule,
        "init_noise": init_noise,
        "step_noises": step_noises,
        "rotations": rotations,
        "translations": translations,
        "n_samples": n_samples,
        "n_token": n_token,
        "n_blocks": n_blocks,
        "heads": heads,
        "n_queries": n_queries,
        "n_keys": n_keys,
        "exact_cache_bytes": n_blocks * heads * n_token * n_token * 4,
    }


def _spy_pair_bias_builds(monkeypatch: pytest.MonkeyPatch) -> list[None]:
    """Record each pair-bias cache the sampler builds, not its shape probe."""
    builds: list[None] = []
    probing = False
    real_build = protenix_diffusion.build_diffusion_pair_bias_cache
    real_eval_shape = jax.eval_shape

    def spy_build(*args: Any, **kwargs: Any) -> jnp.ndarray:
        if not probing:
            builds.append(None)
        return real_build(*args, **kwargs)

    def eval_shape(*args: Any, **kwargs: Any) -> Any:
        nonlocal probing
        probing = True
        try:
            return real_eval_shape(*args, **kwargs)
        finally:
            probing = False

    monkeypatch.setattr(
        protenix_diffusion, "build_diffusion_pair_bias_cache", spy_build
    )
    monkeypatch.setattr(jax, "eval_shape", eval_shape)
    return builds


@pytest.mark.parametrize("use_scan", [False, True])
@pytest.mark.parametrize("stacked", [False, True])
def test_protenix_infer_static_pair_bias_cache_builds_once_and_matches_uncached(
    monkeypatch: pytest.MonkeyPatch,
    use_scan: bool,
    stacked: bool,
) -> None:
    """Default gated cache builds once outside the sampler and matches uncached."""
    fixture = _protenix_pair_bias_cache_fixture(
        seed=41, n_blocks=3, heads=2, stacked=stacked
    )
    monkeypatch.setattr(
        protenix_model,
        "input_feature_embedder",
        lambda *_a, **_k: fixture["s_inputs"],
    )
    monkeypatch.setattr(
        protenix_model,
        "pairformer_output_from_s_inputs",
        lambda *_a, **_k: (
            fixture["s_inputs"],
            fixture["s_trunk"],
            fixture["z_trunk"],
        ),
    )
    monkeypatch.setattr(
        protenix_model,
        "distogram_head",
        lambda z, _p, **_k: jnp.zeros((fixture["n_token"], fixture["n_token"], 2)),
    )

    builds = _spy_pair_bias_builds(monkeypatch)

    forward_calls: list[dict[str, Any]] = []
    real_forward = protenix_diffusion.diffusion_module_forward

    def spy_forward(*args: Any, **kwargs: Any) -> jnp.ndarray:
        forward_calls.append(dict(kwargs))
        return real_forward(*args, **kwargs)

    monkeypatch.setattr(protenix_diffusion, "diffusion_module_forward", spy_forward)

    params = SimpleNamespace(
        input_embedder=object(),
        pairformer_output=object(),
        diffusion=fixture["diffusion_params"],
        distogram=object(),
        confidence=None,
    )

    def run_infer(*, autocast: bool = False) -> jnp.ndarray:
        model_params = (
            SimpleNamespace(
                **{
                    **vars(params),
                    "diffusion": native_diffusion_autocast_params(
                        fixture["diffusion_params"]
                    ),
                }
            )
            if autocast
            else params
        )
        return protenix_infer_static(
            fixture["features"],
            model_params,
            fixture["schedule"],
            key=None,
            num_samples=fixture["n_samples"],
            init_noise=fixture["init_noise"],
            step_noises=fixture["step_noises"],
            rotations=fixture["rotations"],
            translations=fixture["translations"],
            num_recycles=1,
            atom_encoder_heads=fixture["heads"],
            token_heads=fixture["heads"],
            atom_decoder_heads=fixture["heads"],
            n_queries=fixture["n_queries"],
            n_keys=fixture["n_keys"],
            sigma_data=4.0,
            use_diffusion_scan=use_scan,
            use_sampler_scan=use_scan,
            use_diffusion_efficient_fusion=True,
            diffusion_attention_backend="xla",
            diffusion_autocast=autocast,
            run_confidence=False,
        )["coordinate"]

    cached = run_infer()
    assert len(builds) == 1
    assert forward_calls
    assert all(call.get("precomputed_pair_bias") is not None for call in forward_calls)
    assert all(call.get("transformer_z") is None for call in forward_calls)
    assert all(call.get("extra_attn_bias") is None for call in forward_calls)

    builds.clear()
    forward_calls.clear()
    with monkeypatch.context() as mp:
        mp.setattr(protenix_diffusion, "_DIFFUSION_PAIR_BIAS_CACHE_MAX_BYTES", 0)
        uncached = run_infer()
    assert len(builds) == 0
    assert forward_calls
    assert all("precomputed_pair_bias" not in call for call in forward_calls)
    assert all(call.get("transformer_z") is not None for call in forward_calls)

    np.testing.assert_array_equal(np.asarray(cached), np.asarray(uncached))

    # Also verify under BF16 diffusion autocast (linear_z still delivers FP32 bias).
    if not stacked:
        builds.clear()
        forward_calls.clear()
        cached_bf16 = run_infer(autocast=True)
        assert len(builds) == 1
        assert cached_bf16.dtype == jnp.float32
        with monkeypatch.context() as mp:
            mp.setattr(protenix_diffusion, "_DIFFUSION_PAIR_BIAS_CACHE_MAX_BYTES", 0)
            uncached_bf16 = run_infer(autocast=True)
        np.testing.assert_array_equal(
            np.asarray(cached_bf16), np.asarray(uncached_bf16)
        )


def test_protenix_infer_compiled_pair_bias_cache_matches_uncached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The consolidated JIT graph builds the cache once and matches uncached."""
    fixture = _protenix_pair_bias_cache_fixture(seed=53, n_blocks=2, heads=2)
    monkeypatch.setattr(
        protenix_model,
        "input_feature_embedder",
        lambda *_a, **_k: fixture["s_inputs"],
    )
    monkeypatch.setattr(
        protenix_model,
        "pairformer_output_from_s_inputs",
        lambda *_a, **_k: (
            fixture["s_inputs"],
            fixture["s_trunk"],
            fixture["z_trunk"],
        ),
    )
    monkeypatch.setattr(
        protenix_model,
        "distogram_head",
        lambda z, _p, **_k: jnp.zeros((fixture["n_token"], fixture["n_token"], 2)),
    )
    params = protenix_model.ProtenixInferenceParams(
        input_embedder=None,
        pairformer_output=None,
        diffusion=fixture["diffusion_params"],
        distogram=None,
        confidence=None,
    )
    call_kw: dict[str, Any] = {
        "key": None,
        "num_samples": fixture["n_samples"],
        "init_noise": fixture["init_noise"],
        "step_noises": jnp.stack(fixture["step_noises"], axis=0),
        "rotations": fixture["rotations"],
        "translations": fixture["translations"],
        "num_recycles": 1,
        "atom_encoder_heads": fixture["heads"],
        "token_heads": fixture["heads"],
        "atom_decoder_heads": fixture["heads"],
        "n_queries": fixture["n_queries"],
        "n_keys": fixture["n_keys"],
        "sigma_data": 4.0,
        "use_diffusion_scan": True,
        "use_sampler_scan": True,
        "use_diffusion_efficient_fusion": True,
        "diffusion_attention_backend": "xla",
        "run_confidence": False,
    }
    builds = _spy_pair_bias_builds(monkeypatch)

    protenix_model._compiled_protenix_infer.clear_cache()
    try:
        cached = protenix_infer_compiled(
            fixture["features"], params, fixture["schedule"], **call_kw
        )["coordinate"]
        assert len(builds) == 1

        protenix_model._compiled_protenix_infer.clear_cache()
        builds.clear()
        with monkeypatch.context() as mp:
            mp.setattr(protenix_diffusion, "_DIFFUSION_PAIR_BIAS_CACHE_MAX_BYTES", 0)
            uncached = protenix_infer_compiled(
                fixture["features"], params, fixture["schedule"], **call_kw
            )["coordinate"]
        assert len(builds) == 0
    finally:
        protenix_model._compiled_protenix_infer.clear_cache()

    np.testing.assert_array_equal(np.asarray(cached), np.asarray(uncached))


def test_pair_bias_cache_gate_refuses_cap_cp_unfused_and_guidance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Gate admits exact cap, refuses above cap, CP, unfused, guidance, no pair_z."""
    fixture = _protenix_pair_bias_cache_fixture(seed=61, n_blocks=2, heads=2)
    exact_bytes = fixture["exact_cache_bytes"]

    builds = _spy_pair_bias_builds(monkeypatch)

    forward_calls: list[dict[str, Any]] = []
    real_forward = protenix_diffusion.diffusion_module_forward

    def spy_forward(*args: Any, **kwargs: Any) -> jnp.ndarray:
        forward_calls.append(dict(kwargs))
        return real_forward(*args, **kwargs)

    monkeypatch.setattr(protenix_diffusion, "diffusion_module_forward", spy_forward)

    def run_sampler(**overrides: Any) -> jnp.ndarray:
        forward_calls.clear()
        kw: dict[str, Any] = {
            "num_samples": fixture["n_samples"],
            "key": None,
            "init_noise": fixture["init_noise"],
            "step_noises": fixture["step_noises"],
            "rotations": fixture["rotations"],
            "translations": fixture["translations"],
            "pair_z": fixture["pair_z"],
            "atom_encoder_heads": fixture["heads"],
            "token_heads": fixture["heads"],
            "atom_decoder_heads": fixture["heads"],
            "n_queries": fixture["n_queries"],
            "n_keys": fixture["n_keys"],
            "sigma_data": 4.0,
            "use_scan": False,
            "use_sampler_scan": False,
            "use_efficient_fusion": True,
            "attention_backend": "xla",
        }
        kw.update(overrides)
        return sample_diffusion_with_module(
            fixture["features"],
            fixture["s_inputs"],
            fixture["s_trunk"],
            fixture["z_trunk"],
            fixture["diffusion_params"],
            fixture["schedule"],
            **kw,
        )

    # Exact byte cap: admitted.
    with monkeypatch.context() as mp:
        mp.setattr(
            protenix_diffusion, "_DIFFUSION_PAIR_BIAS_CACHE_MAX_BYTES", exact_bytes
        )
        builds.clear()
        at_cap = run_sampler()
        assert len(builds) == 1
        assert all(
            call.get("precomputed_pair_bias") is not None for call in forward_calls
        )
        assert all(call.get("transformer_z") is None for call in forward_calls)

    # One byte over the cap, zero, and negative caps: refused, uncached path intact.
    for cap in (exact_bytes - 1, 0, -1):
        with monkeypatch.context() as mp:
            mp.setattr(protenix_diffusion, "_DIFFUSION_PAIR_BIAS_CACHE_MAX_BYTES", cap)
            builds.clear()
            over_cap = run_sampler()
            assert len(builds) == 0
            assert all("precomputed_pair_bias" not in call for call in forward_calls)
            assert all(call.get("transformer_z") is not None for call in forward_calls)
            np.testing.assert_array_equal(np.asarray(at_cap), np.asarray(over_cap))

    # Context parallelism: refused (the denoiser is stubbed, so no real mesh).
    with monkeypatch.context() as mp:
        mp.setattr(protenix_diffusion, "cp_mesh", lambda: object())
        seen_cp_calls: list[dict[str, Any]] = []
        mp.setattr(
            protenix_diffusion,
            "diffusion_module_forward",
            lambda *a, **k: seen_cp_calls.append(dict(k)) or jnp.zeros_like(a[9]),
        )
        builds.clear()
        run_sampler()
        assert len(builds) == 0
        assert seen_cp_calls
        assert all("precomputed_pair_bias" not in c for c in seen_cp_calls)
        assert all(c.get("transformer_z") is not None for c in seen_cp_calls)

    # A separately jitted denoiser: refused, and the result is unchanged.
    builds.clear()
    jitted = run_sampler(use_denoiser_jit=True)
    assert len(builds) == 0
    np.testing.assert_allclose(np.asarray(jitted), np.asarray(at_cap), atol=1e-5)

    # Efficient fusion off: refused, transformer_z stays None.
    builds.clear()
    run_sampler(use_efficient_fusion=False)
    assert len(builds) == 0
    assert all("precomputed_pair_bias" not in call for call in forward_calls)
    assert all(call.get("transformer_z") is None for call in forward_calls)

    # No precomputed pair_z: refused.
    builds.clear()
    run_sampler(pair_z=None)
    assert len(builds) == 0
    assert all("precomputed_pair_bias" not in call for call in forward_calls)
    assert all(call.get("transformer_z") is None for call in forward_calls)

    # Active TFG guidance: refused; disabled TFG guidance: admitted.
    active_guidance = {
        "enable": True,
        "mu": 0.1,
        "steps": {"tfg_inner": 1, "projection_outer": 0},
        "terms": {"InterchainBondPotential": {"weight": 1.0, "buffer": 1.0}},
    }
    guidance_features = {
        "interchain_bond_index": jnp.asarray([[0], [1]], dtype=jnp.int32)
    }
    builds.clear()
    guided_default = run_sampler(
        guidance_config=active_guidance,
        guidance_features=guidance_features,
    )
    assert len(builds) == 0
    assert all("precomputed_pair_bias" not in call for call in forward_calls)
    with monkeypatch.context() as mp:
        mp.setattr(protenix_diffusion, "_DIFFUSION_PAIR_BIAS_CACHE_MAX_BYTES", 0)
        guided_uncached = run_sampler(
            guidance_config=active_guidance,
            guidance_features=guidance_features,
        )
    np.testing.assert_array_equal(
        np.asarray(guided_default), np.asarray(guided_uncached)
    )

    builds.clear()
    disabled_guidance = run_sampler(
        guidance_config={"enable": False}, guidance_features={}
    )
    assert len(builds) == 1
    np.testing.assert_array_equal(np.asarray(disabled_guidance), np.asarray(at_cap))

    # A traced guidance flag counts as on.
    traced: list[bool] = []
    jax.eval_shape(
        lambda flag: traced.append(
            protenix_diffusion._guidance_enabled({"enable": flag})
        )
        or flag,
        jax.ShapeDtypeStruct((), jnp.bool_),
    )
    assert traced == [True]

    # Explicit extra_attn_bias is folded into the cache and stripped from denoiser args.
    extra_bias = jnp.asarray(
        np.random.default_rng(77).normal(
            scale=0.5, size=(fixture["n_token"], fixture["n_token"])
        ),
        jnp.float32,
    )
    builds.clear()
    cached_extra = run_sampler(extra_attn_bias=extra_bias)
    assert len(builds) == 1
    assert all(call.get("extra_attn_bias") is None for call in forward_calls)
    assert all(call.get("precomputed_pair_bias") is not None for call in forward_calls)
    with monkeypatch.context() as mp:
        mp.setattr(protenix_diffusion, "_DIFFUSION_PAIR_BIAS_CACHE_MAX_BYTES", 0)
        builds.clear()
        uncached_extra = run_sampler(extra_attn_bias=extra_bias)
    assert len(builds) == 0
    assert all(call.get("extra_attn_bias") is extra_bias for call in forward_calls)
    np.testing.assert_array_equal(np.asarray(cached_extra), np.asarray(uncached_extra))
    assert not np.allclose(np.asarray(cached_extra), np.asarray(at_cap))
