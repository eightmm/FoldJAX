import sys
from functools import partial
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
import torch

from foldjax.models.boltz2.bridge.torch_checkpoint import load_checkpoint_state_dict
from foldjax.models.boltz2.bridge.torch_mapping import (
    map_atom_attention_decoder_state_dict,
    map_atom_attention_encoder_state_dict,
)
from foldjax.models.boltz2.models.diffusion.atom import (
    atom_attention_decoder_forward,
    atom_attention_encoder_forward,
    get_indexing_matrix,
    single_to_keys,
)

CHECKPOINT = (
    Path(__file__).resolve().parents[4] / "boltz/.cache/boltz/boltz2_conf.ckpt"
)
BOLTZ_SRC = Path(__file__).resolve().parents[4] / "boltz/src"
ENC_PREFIX = "structure_module.score_model.atom_attention_encoder"
DEC_PREFIX = "structure_module.score_model.atom_attention_decoder"


@pytest.fixture(scope="module")
def checkpoint_state() -> dict[str, torch.Tensor]:
    if not CHECKPOINT.exists():
        pytest.skip(f"Boltz-2 checkpoint not found: {CHECKPOINT}")
    return load_checkpoint_state_dict(CHECKPOINT)


def test_checkpoint_atom_attention_encoder_matches_boltz_torch(
    checkpoint_state: dict[str, torch.Tensor],
) -> None:
    torch_module = _load_torch_encoder(checkpoint_state)
    params = map_atom_attention_encoder_state_dict(checkpoint_state, ENC_PREFIX)
    feats, q, c, bias, r = _encoder_inputs()

    torch_to_keys_fn, jax_to_keys_fn = _to_keys_fns()
    with torch.no_grad():
        expected_a, expected_q, expected_c, _ = torch_module(
            feats=feats,
            q=q,
            c=c,
            atom_enc_bias=bias,
            to_keys=torch_to_keys_fn,
            r=r,
            multiplicity=1,
        )
    actual_a, actual_q, actual_c = atom_attention_encoder_forward(
        params,
        feats=_jax_feats(feats),
        q=jnp.asarray(q.numpy()),
        c=jnp.asarray(c.numpy()),
        atom_enc_bias=jnp.asarray(bias.numpy()),
        to_keys=jax_to_keys_fn,
        r=jnp.asarray(r.numpy()),
        multiplicity=1,
    )

    np.testing.assert_allclose(
        np.asarray(actual_a),
        expected_a.detach().numpy(),
        rtol=2e-3,
        atol=2e-3,
    )
    np.testing.assert_allclose(
        np.asarray(actual_q),
        expected_q.detach().numpy(),
        rtol=2e-3,
        atol=2e-3,
    )
    np.testing.assert_allclose(
        np.asarray(actual_c),
        expected_c.detach().numpy(),
        rtol=2e-3,
        atol=2e-3,
    )


def test_checkpoint_atom_attention_decoder_matches_boltz_torch(
    checkpoint_state: dict[str, torch.Tensor],
) -> None:
    torch_module = _load_torch_decoder(checkpoint_state)
    params = map_atom_attention_decoder_state_dict(checkpoint_state, DEC_PREFIX)
    feats, a, q, c, bias = _decoder_inputs()

    torch_to_keys_fn, jax_to_keys_fn = _to_keys_fns()
    with torch.no_grad():
        expected = torch_module(
            a=a,
            q=q,
            c=c,
            atom_dec_bias=bias,
            feats=feats,
            to_keys=torch_to_keys_fn,
            multiplicity=1,
        )
    actual = atom_attention_decoder_forward(
        params,
        a=jnp.asarray(a.numpy()),
        q=jnp.asarray(q.numpy()),
        c=jnp.asarray(c.numpy()),
        atom_dec_bias=jnp.asarray(bias.numpy()),
        feats=_jax_feats(feats),
        to_keys=jax_to_keys_fn,
        multiplicity=1,
    )

    np.testing.assert_allclose(
        np.asarray(actual),
        expected.detach().numpy(),
        rtol=2e-3,
        atol=2e-3,
    )


def _load_torch_encoder(state: dict[str, torch.Tensor]) -> torch.nn.Module:
    sys.path.insert(0, str(BOLTZ_SRC))
    from boltz.model.modules.encodersv2 import AtomAttentionEncoder

    module = AtomAttentionEncoder(
        atom_s=128,
        token_s=384,
        atoms_per_window_queries=32,
        atoms_per_window_keys=128,
        atom_encoder_depth=3,
        atom_encoder_heads=4,
        structure_prediction=True,
    ).eval()
    module_state = {
        key.removeprefix(f"{ENC_PREFIX}."): value
        for key, value in state.items()
        if key.startswith(f"{ENC_PREFIX}.")
    }
    module.load_state_dict(module_state)
    return module


def _load_torch_decoder(state: dict[str, torch.Tensor]) -> torch.nn.Module:
    sys.path.insert(0, str(BOLTZ_SRC))
    from boltz.model.modules.encodersv2 import AtomAttentionDecoder

    module = AtomAttentionDecoder(
        atom_s=128,
        token_s=384,
        attn_window_queries=32,
        attn_window_keys=128,
        atom_decoder_depth=3,
        atom_decoder_heads=4,
    ).eval()
    module_state = {
        key.removeprefix(f"{DEC_PREFIX}."): value
        for key, value in state.items()
        if key.startswith(f"{DEC_PREFIX}.")
    }
    module.load_state_dict(module_state)
    return module


def _to_keys_fns():
    sys.path.insert(0, str(BOLTZ_SRC))
    from boltz.model.modules.encodersv2 import (
        get_indexing_matrix as torch_index,
    )
    from boltz.model.modules.encodersv2 import (
        single_to_keys as torch_to_keys,
    )

    torch_keys = torch_index(K=2, W=32, H=128, device="cpu")
    torch_to_keys_fn = partial(torch_to_keys, indexing_matrix=torch_keys, W=32, H=128)
    jax_keys = get_indexing_matrix(k=2, w=32, h_keys=128)
    return torch_to_keys_fn, lambda x: single_to_keys(x, jax_keys, w=32, h_keys=128)


def _base_feats() -> dict[str, torch.Tensor]:
    atoms = 64
    tokens = 8
    atom_to_token = torch.zeros(1, atoms, tokens)
    atom_to_token[0, torch.arange(atoms), torch.arange(atoms) % tokens] = 1.0
    atom_pad_mask = torch.ones(1, atoms, dtype=torch.float32)
    atom_pad_mask[:, -3:] = 0.0
    return {
        "ref_pos": torch.linspace(-0.3, 0.3, steps=atoms * 3).reshape(1, atoms, 3),
        "atom_pad_mask": atom_pad_mask,
        "atom_to_token": atom_to_token,
    }


def _encoder_inputs():
    atoms = 64
    feats = _base_feats()
    q = torch.linspace(-0.2, 0.2, steps=atoms * 128).reshape(1, atoms, 128)
    c = torch.linspace(0.25, -0.25, steps=atoms * 128).reshape(1, atoms, 128)
    bias = torch.linspace(-0.1, 0.1, steps=2 * 32 * 128 * 12)
    bias = bias.reshape(1, 2, 32, 128, 12)
    r = torch.linspace(0.15, -0.15, steps=atoms * 3).reshape(1, atoms, 3)
    return feats, q, c, bias, r


def _decoder_inputs():
    atoms = 64
    tokens = 8
    feats = _base_feats()
    a = torch.linspace(-0.3, 0.3, steps=tokens * 768).reshape(1, tokens, 768)
    q = torch.linspace(-0.2, 0.2, steps=atoms * 128).reshape(1, atoms, 128)
    c = torch.linspace(0.25, -0.25, steps=atoms * 128).reshape(1, atoms, 128)
    bias = torch.linspace(-0.1, 0.1, steps=2 * 32 * 128 * 12)
    bias = bias.reshape(1, 2, 32, 128, 12)
    return feats, a, q, c, bias


def _jax_feats(feats: dict[str, torch.Tensor]) -> dict[str, jnp.ndarray]:
    return {key: jnp.asarray(value.numpy()) for key, value in feats.items()}
