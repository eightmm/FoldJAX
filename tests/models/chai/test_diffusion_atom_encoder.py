from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.chai.models.diffusion_atom_encoder import (
    diffusion_atom_encoder,
    diffusion_atom_encoder_with_aux,
    map_diffusion_atom_encoder,
)

pytestmark = pytest.mark.official_parity

@pytest.fixture(scope="module")
def diffusion_module(official_asset_path):
    torch = pytest.importorskip("torch")
    path = official_asset_path("diffusion_module.pt")
    return torch.jit.load(str(path), map_location="cpu").eval()


@pytest.fixture(scope="module")
def encoder_state(diffusion_module):
    return {
        key: value.detach().cpu().numpy()
        for key, value in diffusion_module.state_dict().items()
        if key.startswith("atom_attention_encoder.")
    }


def _inputs() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(43)
    return {
        "atom_single_input_feats": rng.normal(size=(1, 4, 128)).astype(np.float32),
        "token_single_trunk_repr": rng.normal(size=(1, 3, 384)).astype(np.float32),
        "token_pair_repr": rng.normal(size=(1, 3, 3, 256)).astype(np.float32),
        "atom_scaled_coords": rng.normal(size=(1, 2, 4, 3)).astype(np.float32),
        "atom_block_pair_input_feats": rng.normal(size=(1, 1, 4, 4, 16)).astype(
            np.float32
        ),
        "atom_single_mask": np.array([[True, True, True, False]]),
        "atom_block_pair_mask": np.array(
            [[[[True, True, True, False]] * 4]], dtype=np.bool_
        ),
        "block_indices_h": np.array([[0, 1, 2, 3]], dtype=np.int64),
        "block_indices_w": np.array([[0, 1, 2, 3]], dtype=np.int64),
        "atom_token_indices": np.array([[0, 0, 1, 2]], dtype=np.int64),
    }


def test_mapping_covers_all_46_encoder_tensors(encoder_state) -> None:
    params = map_diffusion_atom_encoder(encoder_state)
    assert len(encoder_state) == 46
    assert len(jax.tree.leaves(params)) == 46


@pytest.mark.parametrize(
    ("child_name", "input_key"),
    [
        ("to_atom_cond", "atom_single_input_feats"),
        ("token_to_atom_single", "token_single_trunk_repr"),
        ("prev_pos_embed", "atom_scaled_coords"),
        ("token_pair_to_atom_pair", "token_pair_repr"),
    ],
)
def test_callable_input_projections_match_official(
    diffusion_module,
    encoder_state,
    child_name: str,
    input_key: str,
) -> None:
    torch = pytest.importorskip("torch")
    inputs = _inputs()
    child = getattr(diffusion_module.atom_attention_encoder, child_name)
    with torch.no_grad():
        expected = child(torch.from_numpy(inputs[input_key])).numpy()

    params = map_diffusion_atom_encoder(encoder_state)
    if child_name == "to_atom_cond":
        actual = jnp.einsum(
            "...c,oc->...o", inputs[input_key], params.to_atom_cond_weight
        )
    elif child_name == "token_to_atom_single":
        normalized = jax.nn.standardize(
            jnp.asarray(inputs[input_key]), axis=-1, epsilon=1e-5
        )
        normalized = (
            normalized * params.token_to_atom_norm_weight
            + params.token_to_atom_norm_bias
        )
        actual = jnp.einsum("...c,oc->...o", normalized, params.token_to_atom_weight)
    elif child_name == "prev_pos_embed":
        actual = jnp.einsum("...c,oc->...o", inputs[input_key], params.prev_pos_weight)
    else:
        normalized = jax.nn.standardize(
            jnp.asarray(inputs[input_key]), axis=-1, epsilon=1e-5
        )
        normalized = (
            normalized * params.token_pair_norm_weight + params.token_pair_norm_bias
        )
        actual = jnp.einsum("...c,oc->...o", normalized, params.token_pair_weight)
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=3e-5, atol=3e-5)


def _torch_encoder_reference(module, inputs):
    torch = pytest.importorskip("torch")
    functional = torch.nn.functional
    encoder = module.atom_attention_encoder
    transformer = encoder.atom_transformer.local_diffn_transformer

    raw = encoder.to_atom_cond(inputs["atom_single_input_feats"])
    token_single = encoder.token_to_atom_single(inputs["token_single_trunk_repr"])
    batch = torch.arange(raw.shape[0])[:, None]
    cond = raw + token_single[batch, inputs["atom_token_indices"]]
    cond = functional.layer_norm(cond, (cond.shape[-1],))

    single = raw[:, None] + encoder.prev_pos_embed(inputs["atom_scaled_coords"]).float()
    samples = single.shape[1]
    cond_samples = cond[:, None].expand(-1, samples, -1, -1)
    h_cond = cond_samples[:, :, inputs["block_indices_h"]]
    w_cond = cond_samples[:, :, inputs["block_indices_w"]]

    token_pair = encoder.token_pair_to_atom_pair(inputs["token_pair_repr"])
    h_token = inputs["atom_token_indices"][:, inputs["block_indices_h"]][..., None]
    w_token = inputs["atom_token_indices"][:, inputs["block_indices_w"]][:, :, None]
    pair_batch = torch.arange(raw.shape[0])[:, None, None, None]
    atom_pair = token_pair[pair_batch, h_token, w_token]
    pair = encoder.pair_update_block(
        inputs["atom_block_pair_input_feats"], atom_pair, h_cond, w_cond
    )

    batch_samples, atoms = raw.shape[0] * samples, raw.shape[1]
    single = single.reshape(batch_samples, atoms, 128)
    cond = cond_samples.reshape(batch_samples, atoms, 128)
    single_mask = (
        inputs["atom_single_mask"][:, None]
        .expand(-1, samples, -1)
        .reshape(batch_samples, atoms)
    )
    pair_mask = (
        inputs["atom_block_pair_mask"][:, None]
        .expand(-1, samples, -1, -1, -1)
        .reshape(batch_samples, 1, 4, 4)
    )
    pair = pair.reshape(batch_samples, 1, 4, 4, 16)

    pair_features = getattr(transformer.blocked_pairs2blocked_bias, "0")(pair)
    bias_weights = transformer.blocked_pairs2blocked_bias.state_dict()["1.weight"]
    kv_idx = torch.arange(atoms).reshape(1, 4)
    single = single.masked_fill(~single_mask[..., None], 0.0)

    for index in range(3):
        attention = getattr(transformer.local_attentions, str(index))
        transition = getattr(transformer.transitions, str(index))
        bias = torch.einsum(
            "blqkc,hc->bhlqk", pair_features, bias_weights[index]
        ).masked_fill(~pair_mask[:, None], -10000.0)

        feat = functional.layer_norm(single, (128,), eps=0.1)
        scale, shift = attention.single_layer_norm.lin_s_merged(cond).chunk(2, -1)
        feat = feat * (scale + 1.0) + shift
        q, k, v = attention.to_qkv(feat).unbind()
        heads, dim = attention.q_bias.shape
        q = q + attention.q_bias[None].expand(batch_samples, -1, -1).reshape(
            batch_samples * heads, 1, dim
        )
        q = q.reshape(batch_samples * heads, 1, 4, dim)
        k = k[:, kv_idx]
        v = v[:, kv_idx]
        local = functional.scaled_dot_product_attention(
            q, k, v, bias.reshape(batch_samples * heads, 1, 4, 4)
        )
        local = (
            local.reshape(batch_samples, heads, 1, 4, dim)
            .permute(0, 2, 3, 1, 4)
            .reshape(batch_samples, atoms, 128)
        )
        local = local * torch.sigmoid(attention.out_proj(cond))

        feat = functional.layer_norm(single, (128,), eps=0.1)
        scale, shift = transition.ada_ln.lin_s_merged(cond).chunk(2, -1)
        feat = feat * (scale + 1.0) + shift
        value, gate = transition.linear_a_nobias_double(feat).chunk(2, -1)
        value = functional.silu(value) * gate
        value = transition.linear_b_nobias(value)
        value = value * torch.sigmoid(transition.linear_s_biasinit_m2(cond))
        single = single + local + value
        if index < 2:
            single = single.masked_fill(~single_mask[..., None], 0.0)

    atom_single = encoder.to_token_single(
        single.reshape(raw.shape[0], samples, atoms, 128)
    ).reshape(batch_samples, atoms, 768)
    mask = single_mask[..., None].to(atom_single.dtype)
    atom_single = atom_single * mask
    token_indices = (
        inputs["atom_token_indices"][:, None]
        .expand(-1, samples, -1)
        .reshape(batch_samples, atoms)
    )
    pooled = atom_single.new_zeros(batch_samples, 3, 768)
    pooled.scatter_add_(1, token_indices[..., None].expand(-1, -1, 768), atom_single)
    counts = atom_single.new_zeros(batch_samples, 3)
    counts.scatter_add_(1, token_indices, single_mask.to(atom_single.dtype))
    return (pooled / counts[..., None].clamp_min(1.0)).reshape(
        raw.shape[0], samples, 3, 768
    )


def test_complete_encoder_matches_official_leaf_formula(
    diffusion_module, encoder_state
) -> None:
    torch = pytest.importorskip("torch")
    numpy_inputs = _inputs()
    torch_inputs = {key: torch.from_numpy(value) for key, value in numpy_inputs.items()}
    with torch.no_grad():
        expected = _torch_encoder_reference(diffusion_module, torch_inputs)

    actual = diffusion_atom_encoder(
        **{key: jnp.asarray(value) for key, value in numpy_inputs.items()},
        params=map_diffusion_atom_encoder(encoder_state),
    )
    assert actual.shape == (1, 2, 3, 768)
    np.testing.assert_allclose(
        np.asarray(actual), expected.numpy(), rtol=4e-4, atol=4e-4
    )


def test_encoder_exposes_decoder_intermediates_without_changing_pooled_output(
    encoder_state,
) -> None:
    inputs = {key: jnp.asarray(value) for key, value in _inputs().items()}
    params = map_diffusion_atom_encoder(encoder_state)

    output = diffusion_atom_encoder_with_aux(**inputs, params=params)

    assert output.token_single.shape == (1, 2, 3, 768)
    assert output.atom_single.shape == (1, 2, 4, 128)
    assert output.atom_condition.shape == (1, 4, 128)
    assert output.atom_pair.shape == (1, 2, 1, 4, 4, 16)
    np.testing.assert_allclose(
        np.asarray(output.token_single),
        np.asarray(diffusion_atom_encoder(**inputs, params=params)),
    )
