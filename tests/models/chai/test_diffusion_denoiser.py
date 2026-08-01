# ruff: noqa: E402, I001

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.official_parity

torch = pytest.importorskip("torch")

from foldjax.models.chai.models.diffusion_denoiser import (
    AtomEncoderOutput,
    DiffusionDenoiserParams,
    decompose_diffusion_state,
    diffusion_denoiser,
    map_full_diffusion_denoiser,
    map_diffusion_denoiser,
)


@pytest.fixture(scope="module")
def official_module(official_asset_path):
    component = official_asset_path("diffusion_module.pt")
    return torch.jit.load(str(component), map_location="cpu").eval()


@pytest.fixture(scope="module")
def official_state(official_module):
    return {
        key: value.detach().cpu().numpy()
        for key, value in official_module.state_dict().items()
    }


def test_state_decomposition_accounts_for_all_343_tensors(official_state) -> None:
    decomposition = decompose_diffusion_state(official_state)

    assert len(official_state) == 343
    assert len(decomposition.conditioning) == 31
    assert len(decomposition.atom_encoder) == 46
    assert len(decomposition.transformer) == 224
    assert len(decomposition.atom_decoder) == 37
    assert len(decomposition.top_level) == 5
    assert not decomposition.unknown
    groups = (
        decomposition.conditioning,
        decomposition.atom_encoder,
        decomposition.transformer,
        decomposition.atom_decoder,
        decomposition.top_level,
    )
    assert len(set().union(*groups)) == sum(map(len, groups)) == 343


def test_top_level_mapping_reports_exact_unconsumed_set(official_state) -> None:
    params, unconsumed = map_diffusion_denoiser(official_state)
    expected = {
        key
        for key in official_state
        if key.startswith(
            (
                "diffusion_conditioning.",
                "atom_attention_encoder.",
                "diffusion_transformer.",
                "atom_attention_decoder.",
            )
        )
    }

    assert len(unconsumed) == 338
    assert unconsumed == expected
    assert params.structure_projection_weight.shape == (768, 384)
    assert params.post_attention_norm_weight.shape == (768,)
    assert params.post_atom_condition_norm_weight.shape == (128,)


def test_full_mapping_consumes_all_343_tensors(official_state) -> None:
    params = map_full_diffusion_denoiser(official_state)

    assert len(jax.tree.leaves(params)) == 343


def test_top_level_projections_match_official_callable_children(
    official_module, official_state
) -> None:
    params, _ = map_diffusion_denoiser(official_state)
    rng = np.random.default_rng(11)
    structure = rng.normal(size=(1, 2, 3, 384)).astype(np.float32)
    token = rng.normal(size=(1, 2, 3, 768)).astype(np.float32)
    atom = rng.normal(size=(1, 4, 128)).astype(np.float32)

    with torch.no_grad():
        expected_structure = official_module.structure_cond_to_token_structure_proj(
            torch.from_numpy(structure)
        ).numpy()
        expected_token = official_module.post_attn_layernorm(
            torch.from_numpy(token)
        ).numpy()
        expected_atom = official_module.post_atom_cond_layernorm(
            torch.from_numpy(atom)
        ).numpy()

    np.testing.assert_allclose(
        structure @ np.asarray(params.structure_projection_weight).T,
        expected_structure,
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        _layer_norm_np(
            token,
            np.asarray(params.post_attention_norm_weight),
            np.asarray(params.post_attention_norm_bias),
        ),
        expected_token,
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        _layer_norm_np(
            atom,
            np.asarray(params.post_atom_condition_norm_weight),
            np.asarray(params.post_atom_condition_norm_bias),
        ),
        expected_atom,
        rtol=1e-5,
        atol=1e-5,
    )


def _layer_norm_np(x, weight, bias):
    mean = x.mean(axis=-1, keepdims=True)
    variance = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(variance + 1e-5) * weight + bias


def test_exact_denoiser_orchestration_and_sigma_scaling() -> None:
    batch, samples, tokens, atoms = 1, 2, 3, 4
    calls = {}
    call_order = []
    initial_single = jnp.zeros((batch, tokens, 384), jnp.float32)
    initial_pair = jnp.zeros((batch, tokens, tokens, 256), jnp.float32)
    trunk_single = jnp.ones((batch, tokens, 384), jnp.float32)
    trunk_pair = jnp.ones((batch, tokens, tokens, 256), jnp.float32)
    noise_sigma = jnp.array([[2.0, 4.0]], jnp.float32)
    noised_coords = jnp.arange(batch * samples * atoms * 3, dtype=jnp.float32)
    noised_coords = noised_coords.reshape(batch, samples, atoms, 3)
    atom_features = jnp.zeros((batch, atoms, 128), jnp.float32)
    atom_pair_features = jnp.zeros((batch, 1, atoms, atoms, 16), jnp.float32)
    atom_mask = jnp.ones((batch, atoms), bool)
    pair_mask = jnp.ones((batch, 1, atoms, atoms), bool)
    token_mask = jnp.array([[1, 1, 0]], bool)
    block_h = jnp.arange(atoms).reshape(1, atoms)
    block_w = jnp.arange(atoms).reshape(1, atoms)
    atom_token_indices = jnp.array([[0, 0, 1, 2]], jnp.int32)

    conditioned_single = jnp.full((batch, samples, tokens, 384), 2.0)
    conditioned_pair = jnp.full((batch, tokens, tokens, 256), 3.0)
    encoder_output = AtomEncoderOutput(
        token_single=jnp.full((batch, samples, tokens, 768), 5.0),
        atom_single=jnp.full((batch, samples, atoms, 128), 7.0),
        atom_condition=jnp.arange(batch * atoms * 128, dtype=jnp.float32).reshape(
            batch, atoms, 128
        ),
        atom_pair=jnp.full((batch, samples, 1, atoms, atoms, 16), 11.0),
    )
    transformer_output = jnp.arange(
        batch * samples * tokens * 768, dtype=jnp.float32
    ).reshape(batch, samples, tokens, 768)
    unit_update = jnp.full((batch, samples, atoms, 3), 13.0)
    params = DiffusionDenoiserParams(
        structure_projection_weight=jnp.ones((768, 384), jnp.float32),
        post_attention_norm_weight=jnp.ones((768,), jnp.float32),
        post_attention_norm_bias=jnp.zeros((768,), jnp.float32),
        post_atom_condition_norm_weight=jnp.ones((128,), jnp.float32),
        post_atom_condition_norm_bias=jnp.zeros((128,), jnp.float32),
    )

    def conditioner(*args):
        call_order.append("conditioning")
        calls["conditioner"] = args
        return conditioned_single, conditioned_pair

    def encoder(*args):
        call_order.append("atom_encoder")
        calls["encoder"] = args
        return encoder_output

    def transformer(token, condition, pair, mask):
        call_order.append("transformer")
        calls["transformer"] = (token, condition, pair, mask)
        return transformer_output

    def decoder(*args):
        call_order.append("atom_decoder")
        calls["decoder"] = args
        return unit_update

    output = diffusion_denoiser(
        initial_single,
        initial_pair,
        trunk_single,
        trunk_pair,
        atom_features,
        atom_pair_features,
        atom_mask,
        pair_mask,
        token_mask,
        block_h,
        block_w,
        noised_coords,
        noise_sigma,
        atom_token_indices,
        params,
        conditioning_fn=conditioner,
        atom_encoder_fn=encoder,
        transformer_fn=transformer,
        atom_decoder_fn=decoder,
    )

    assert call_order == ["conditioning", "atom_encoder", "transformer", "atom_decoder"]
    assert calls["conditioner"][-1] is noise_sigma
    expected_scaled = noised_coords / jnp.sqrt(
        noise_sigma[..., None, None] ** 2 + 256.0
    )
    np.testing.assert_allclose(calls["encoder"][3], expected_scaled)
    expected_transformer_input = encoder_output.token_single + 384.0 * 2.0
    np.testing.assert_allclose(calls["transformer"][0], expected_transformer_input)
    np.testing.assert_allclose(calls["transformer"][1], conditioned_single)
    np.testing.assert_allclose(calls["transformer"][2], conditioned_pair)
    np.testing.assert_array_equal(calls["transformer"][3], token_mask)
    expected_post_attention = _layer_norm_np(
        np.asarray(transformer_output),
        np.ones((768,), np.float32),
        np.zeros((768,), np.float32),
    )
    expected_post_atom = _layer_norm_np(
        np.asarray(encoder_output.atom_condition),
        np.ones((128,), np.float32),
        np.zeros((128,), np.float32),
    )
    np.testing.assert_allclose(
        calls["decoder"][0], expected_post_attention, rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(calls["decoder"][1], encoder_output.atom_single)
    np.testing.assert_allclose(
        calls["decoder"][2], expected_post_atom, rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(calls["decoder"][3], encoder_output.atom_pair)

    sigma = noise_sigma[..., None, None]
    expected = noised_coords * (256.0 / (sigma**2 + 256.0))
    expected += unit_update * (sigma * 16.0 / jnp.sqrt(sigma**2 + 256.0))
    np.testing.assert_allclose(output, expected.reshape(batch * samples, atoms, 3))
