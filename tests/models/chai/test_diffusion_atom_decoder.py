# ruff: noqa: E402, I001

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.official_parity

torch = pytest.importorskip("torch")

from foldjax.models.chai.models.diffusion_atom_decoder import (
    atom_decoder_forward,
    map_atom_decoder,
)


_PREFIX = "atom_attention_decoder."


@pytest.fixture(scope="module")
def decoder_state(official_asset_path) -> dict[str, np.ndarray]:
    component = official_asset_path("diffusion_module.pt")
    module = torch.jit.load(str(component), map_location="cpu").eval()
    return {
        key: value.detach().cpu().numpy()
        for key, value in module.state_dict().items()
        if key.startswith(_PREFIX)
    }


def test_decoder_maps_every_official_tensor_exactly_once(decoder_state) -> None:
    params = map_atom_decoder(decoder_state)

    leaves = torch.utils._pytree.tree_leaves(params)
    assert len(decoder_state) == len(leaves) == 37
    assert sum(value.size for value in decoder_state.values()) == 837_600
    assert sum(value.size for value in leaves) == 837_600


def test_decoder_mapping_rejects_missing_and_unknown_keys(decoder_state) -> None:
    missing = dict(decoder_state)
    missing.pop(next(iter(missing)))
    with pytest.raises(ValueError, match="missing"):
        map_atom_decoder(missing)

    unknown = dict(decoder_state)
    unknown[f"{_PREFIX}unexpected.weight"] = np.zeros((1,), np.float32)
    with pytest.raises(ValueError, match="unexpected"):
        map_atom_decoder(unknown)


def _inputs(seed: int = 17) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    batch, samples, tokens, atoms = 1, 2, 3, 8
    n_blocks, query, keys = 2, 4, 6
    return {
        "token_single": rng.normal(size=(batch, samples, tokens, 768)).astype(
            np.float32
        ),
        "atom_single": rng.normal(size=(batch, samples, atoms, 128)).astype(
            np.float32
        ),
        "atom_cond": rng.normal(size=(batch, atoms, 128)).astype(np.float32),
        "atom_pair": rng.normal(
            size=(batch, samples, n_blocks, query, keys, 16)
        ).astype(np.float32),
        "atom_mask": np.array([[1, 1, 1, 1, 1, 1, 1, 0]], dtype=bool),
        "pair_mask": np.array(
            [
                [
                    [[1, 1, 1, 1, 1, 0]] * query,
                    [[1, 1, 1, 1, 0, 0]] * query,
                ]
            ],
            dtype=bool,
        ),
        "atom_token_indices": np.array([[0, 0, 1, 1, 1, 2, 2, 2]], np.int64),
    }


def _linear(x, weight, bias=None):
    return torch.nn.functional.linear(x, weight, bias)


def _torch_reference(state, inputs):
    tf = torch.nn.functional
    prefix = _PREFIX

    def weight(name):
        return torch.from_numpy(state[f"{prefix}{name}"])

    token = torch.from_numpy(inputs["token_single"])
    atom = torch.from_numpy(inputs["atom_single"])
    cond = torch.from_numpy(inputs["atom_cond"])
    pair = torch.from_numpy(inputs["atom_pair"])
    atom_mask = torch.from_numpy(inputs["atom_mask"])
    pair_mask = torch.from_numpy(inputs["pair_mask"])
    token_index = torch.from_numpy(inputs["atom_token_indices"])

    batch, samples, atoms, channels = atom.shape
    _, _, n_blocks, query, keys, _ = pair.shape
    token = token.reshape(batch * samples, token.shape[2], token.shape[3])
    atom = atom.reshape(batch * samples, atoms, channels)
    cond = cond[:, None].expand(-1, samples, -1, -1).reshape(
        batch * samples, atoms, channels
    )
    pair = pair.reshape(batch * samples, n_blocks, query, keys, 16)
    atom_mask = atom_mask[:, None].expand(-1, samples, -1).reshape(
        batch * samples, atoms
    )
    pair_mask = pair_mask[:, None].expand(-1, samples, -1, -1, -1).reshape(
        batch * samples, n_blocks, query, keys
    )
    token_index = token_index[:, None].expand(-1, samples, -1).reshape(
        batch * samples, atoms
    )

    token_atoms = _linear(token, weight("token_to_atom.weight"))
    batch_index = torch.arange(batch * samples)[:, None]
    single = atom + token_atoms[batch_index, token_index]
    single = single.masked_fill(~atom_mask[..., None], 0.0)

    b2b = tf.layer_norm(
        pair,
        (16,),
        weight("atom_transformer.local_diffn_transformer."
               "blocked_pairs2blocked_bias.0.weight"),
        weight("atom_transformer.local_diffn_transformer."
               "blocked_pairs2blocked_bias.0.bias"),
    )
    q_indices = torch.arange(atoms).reshape(n_blocks, query)
    kv_index = (
        q_indices[:, :1] + (query - keys) // 2 + torch.arange(keys)
    ) % atoms
    base = "atom_transformer.local_diffn_transformer"

    for block in range(3):
        attn = f"{base}.local_attentions.{block}"
        trans = f"{base}.transitions.{block}"
        bias_weight = weight(f"{base}.blocked_pairs2blocked_bias.1.weight")[block]
        bias = torch.einsum("blqkc,hc->bhlqk", b2b, bias_weight)
        bias = bias.masked_fill(~pair_mask[:, None], -10000.0)

        feat = tf.layer_norm(single, (128,), eps=0.1)
        scale, shift = _linear(
            cond, weight(f"{attn}.single_layer_norm.lin_s_merged.weight")
        ).chunk(2, dim=-1)
        feat = feat * (scale + 1.0) + shift
        qkv = torch.einsum("nhdc,bac->nbhad", weight(f"{attn}.to_qkv.weight"), feat)
        q, k, value = qkv.unbind(0)
        q = q + weight(f"{attn}.q_bias")[None, :, None, :]
        heads, head_dim = q.shape[1], q.shape[-1]
        q = q.reshape(batch * samples * heads, n_blocks, query, head_dim)
        k = k.reshape(batch * samples * heads, atoms, head_dim)[:, kv_index]
        value = value.reshape(batch * samples * heads, atoms, head_dim)[:, kv_index]
        local = tf.scaled_dot_product_attention(
            q, k, value, bias.reshape(batch * samples * heads, n_blocks, query, keys)
        )
        local = local.reshape(batch * samples, heads, n_blocks, query, head_dim)
        local = local.permute(0, 2, 3, 1, 4).reshape(
            batch * samples, atoms, heads * head_dim
        )
        local = local * torch.sigmoid(
            _linear(
                cond,
                weight(f"{attn}.out_proj.weight"),
                weight(f"{attn}.out_proj.bias"),
            )
        )

        feat = tf.layer_norm(single, (128,), eps=0.1)
        scale, shift = _linear(
            cond, weight(f"{trans}.ada_ln.lin_s_merged.weight")
        ).chunk(2, dim=-1)
        feat = feat * (scale + 1.0) + shift
        value0, gate = _linear(
            feat, weight(f"{trans}.linear_a_nobias_double.weight")
        ).chunk(2, dim=-1)
        hidden = tf.silu(value0) * gate
        transition = torch.sigmoid(
            _linear(
                cond,
                weight(f"{trans}.linear_s_biasinit_m2.weight"),
                weight(f"{trans}.linear_s_biasinit_m2.bias"),
            )
        ) * _linear(hidden, weight(f"{trans}.linear_b_nobias.weight"))
        single = single + transition + local
        if block < 2:
            single = single.masked_fill(~atom_mask[..., None], 0.0)

    output = tf.layer_norm(
        single,
        (128,),
        weight("to_pos_updates.0.weight"),
        weight("to_pos_updates.0.bias"),
    )
    output = _linear(output, weight("to_pos_updates.1.weight"))
    return output.reshape(batch, samples, atoms, 3)


def test_decoder_fp32_parity_with_official_weights(decoder_state) -> None:
    inputs = _inputs()
    reference = _torch_reference(decoder_state, inputs).numpy()
    params = map_atom_decoder(decoder_state)

    actual = np.asarray(
        atom_decoder_forward(
            inputs["token_single"],
            inputs["atom_single"],
            inputs["atom_cond"],
            inputs["atom_pair"],
            inputs["atom_mask"],
            inputs["pair_mask"],
            inputs["atom_token_indices"],
            params,
        )
    )

    assert actual.shape == reference.shape == (1, 2, 8, 3)
    np.testing.assert_allclose(actual, reference, rtol=1e-4, atol=1e-4)
