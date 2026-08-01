import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
import torch

from foldjax.models.boltz2.bridge.torch_checkpoint import load_checkpoint_state_dict
from foldjax.models.boltz2.bridge.torch_mapping import map_input_embedder_state_dict
from foldjax.models.boltz2.models.trunk_blocks.input_embedder import (
    input_embedder_forward,
)

CHECKPOINT = (
    Path(__file__).resolve().parents[4] / "boltz/.cache/boltz/boltz2_conf.ckpt"
)
BOLTZ_SRC = Path(__file__).resolve().parents[4] / "boltz/src"
PREFIX = "input_embedder"


@pytest.fixture(scope="module")
def checkpoint_state() -> dict[str, torch.Tensor]:
    if not CHECKPOINT.exists():
        pytest.skip(f"Boltz-2 checkpoint not found: {CHECKPOINT}")
    return load_checkpoint_state_dict(CHECKPOINT)


def test_checkpoint_input_embedder_matches_boltz_torch(
    checkpoint_state: dict[str, torch.Tensor],
) -> None:
    torch_module = _load_torch_input_embedder(checkpoint_state)
    params = map_input_embedder_state_dict(checkpoint_state, PREFIX)
    feats = _input_feats()

    with torch.no_grad():
        expected = torch_module(feats)
    actual = input_embedder_forward(params, _jax_feats(feats))

    np.testing.assert_allclose(
        np.asarray(actual),
        expected.detach().numpy(),
        rtol=2e-3,
        atol=2e-3,
    )


def _load_torch_input_embedder(state: dict[str, torch.Tensor]) -> torch.nn.Module:
    sys.path.insert(0, str(BOLTZ_SRC))
    from boltz.model.modules.trunkv2 import InputEmbedder

    module = InputEmbedder(
        atom_s=128,
        atom_z=16,
        token_s=384,
        token_z=128,
        atoms_per_window_queries=32,
        atoms_per_window_keys=128,
        atom_feature_dim=388,
        atom_encoder_depth=3,
        atom_encoder_heads=4,
        add_method_conditioning=True,
        add_modified_flag=True,
        add_cyclic_flag=True,
        add_mol_type_feat=True,
    ).eval()
    module_state = {
        key.removeprefix(f"{PREFIX}."): value
        for key, value in state.items()
        if key.startswith(f"{PREFIX}.")
    }
    module.load_state_dict(module_state)
    return module


def _input_feats() -> dict[str, torch.Tensor]:
    atoms = 64
    tokens = 8
    atom_to_token = torch.zeros(1, atoms, tokens)
    atom_to_token[0, torch.arange(atoms), torch.arange(atoms) % tokens] = 1.0
    ref_element = torch.zeros(1, atoms, 128)
    ref_element[0, torch.arange(atoms), torch.arange(atoms) % 128] = 1.0
    chars = torch.zeros(1, atoms, 4, 64)
    for index in range(4):
        chars[0, torch.arange(atoms), index, (torch.arange(atoms) + index) % 64] = 1.0
    res_type = torch.zeros(1, tokens, 33)
    res_type[0, torch.arange(tokens), torch.arange(tokens) % 33] = 1.0
    profile = torch.zeros(1, tokens, 33)
    profile[0, torch.arange(tokens), (torch.arange(tokens) + 3) % 33] = 1.0

    feats = {
        "ref_pos": torch.linspace(-0.3, 0.3, steps=atoms * 3).reshape(
            1, atoms, 3
        ),
        "atom_pad_mask": torch.ones(1, atoms, dtype=torch.float32),
        "ref_space_uid": (torch.arange(atoms) // 8).reshape(1, atoms),
        "ref_charge": torch.linspace(-0.5, 0.5, steps=atoms).reshape(1, atoms),
        "ref_element": ref_element,
        "ref_atom_name_chars": chars,
        "atom_to_token": atom_to_token,
        "res_type": res_type,
        "profile": profile,
        "deletion_mean": torch.linspace(0.0, 1.0, steps=tokens).reshape(1, tokens),
        "method_feature": (torch.arange(tokens) % 12).reshape(1, tokens),
        "modified": (torch.arange(tokens) % 2).reshape(1, tokens),
        "cyclic_period": torch.linspace(0.0, 2.0, steps=tokens).reshape(1, tokens),
        "mol_type": (torch.arange(tokens) % 4).reshape(1, tokens),
    }
    feats["atom_pad_mask"][:, -3:] = 0.0
    return feats


def _jax_feats(feats: dict[str, torch.Tensor]) -> dict[str, jnp.ndarray]:
    return {key: jnp.asarray(value.numpy()) for key, value in feats.items()}
