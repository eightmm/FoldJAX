import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
import torch

from foldjax.models.boltz2.bridge.torch_checkpoint import load_checkpoint_state_dict
from foldjax.models.boltz2.bridge.torch_mapping import (
    map_pairwise_conditioning_state_dict,
    map_single_conditioning_state_dict,
)
from foldjax.models.boltz2.models.trunk_blocks.conditioning import (
    pairwise_conditioning_forward,
    single_conditioning_forward,
)

CHECKPOINT = (
    Path(__file__).resolve().parents[4] / "boltz/.cache/boltz/boltz2_conf.ckpt"
)
BOLTZ_SRC = Path(__file__).resolve().parents[4] / "boltz/src"
SINGLE_PREFIX = "structure_module.score_model.single_conditioner"
PAIRWISE_PREFIX = "structure_module.score_model.pairwise_conditioner"


@pytest.fixture(scope="module")
def checkpoint_state() -> dict[str, torch.Tensor]:
    if not CHECKPOINT.exists():
        pytest.skip(f"Boltz-2 checkpoint not found: {CHECKPOINT}")
    return load_checkpoint_state_dict(CHECKPOINT)


def test_checkpoint_single_conditioning_matches_boltz_torch(
    checkpoint_state: dict[str, torch.Tensor],
) -> None:
    torch_module = _load_torch_single_conditioning(checkpoint_state)
    params = map_single_conditioning_state_dict(checkpoint_state, SINGLE_PREFIX)
    times, s_trunk, s_inputs = _single_inputs()

    with torch.no_grad():
        expected_s, expected_fourier = torch_module(times, s_trunk, s_inputs)
    actual_s, actual_fourier = single_conditioning_forward(
        params,
        jnp.asarray(times.numpy()),
        jnp.asarray(s_trunk.numpy()),
        jnp.asarray(s_inputs.numpy()),
    )

    np.testing.assert_allclose(
        np.asarray(actual_s),
        expected_s.detach().numpy(),
        rtol=1.5e-3,
        atol=1.5e-3,
    )
    np.testing.assert_allclose(
        np.asarray(actual_fourier),
        expected_fourier.detach().numpy(),
        rtol=1e-5,
        atol=1e-5,
    )


def test_checkpoint_pairwise_conditioning_matches_boltz_torch_if_present(
    checkpoint_state: dict[str, torch.Tensor],
) -> None:
    if not any(key.startswith(f"{PAIRWISE_PREFIX}.") for key in checkpoint_state):
        torch.manual_seed(0)
        torch_module = _new_torch_pairwise_conditioning()
        state = {
            f"{PAIRWISE_PREFIX}.{key}": value
            for key, value in torch_module.state_dict().items()
        }
    else:
        state = checkpoint_state
        torch_module = _load_torch_pairwise_conditioning(state)
    params = map_pairwise_conditioning_state_dict(state, PAIRWISE_PREFIX)
    z_trunk, token_rel_pos_feats = _pairwise_inputs()

    with torch.no_grad():
        expected = torch_module(z_trunk, token_rel_pos_feats)
    actual = pairwise_conditioning_forward(
        params,
        jnp.asarray(z_trunk.numpy()),
        jnp.asarray(token_rel_pos_feats.numpy()),
    )

    np.testing.assert_allclose(
        np.asarray(actual),
        expected.detach().numpy(),
        rtol=1.5e-3,
        atol=1.5e-3,
    )


def _load_torch_single_conditioning(
    state: dict[str, torch.Tensor],
) -> torch.nn.Module:
    sys.path.insert(0, str(BOLTZ_SRC))
    from boltz.model.modules.encodersv2 import SingleConditioning

    module = SingleConditioning(
        sigma_data=16.0,
        token_s=384,
        dim_fourier=256,
        num_transitions=2,
    ).eval()
    module_state = {
        key.removeprefix(f"{SINGLE_PREFIX}."): value
        for key, value in state.items()
        if key.startswith(f"{SINGLE_PREFIX}.")
    }
    module.load_state_dict(module_state)
    return module


def _load_torch_pairwise_conditioning(
    state: dict[str, torch.Tensor],
) -> torch.nn.Module:
    module = _new_torch_pairwise_conditioning()
    module_state = {
        key.removeprefix(f"{PAIRWISE_PREFIX}."): value
        for key, value in state.items()
        if key.startswith(f"{PAIRWISE_PREFIX}.")
    }
    module.load_state_dict(module_state)
    return module


def _new_torch_pairwise_conditioning() -> torch.nn.Module:
    sys.path.insert(0, str(BOLTZ_SRC))
    from boltz.model.modules.encodersv2 import PairwiseConditioning

    return PairwiseConditioning(
        token_z=128,
        dim_token_rel_pos_feats=3,
        num_transitions=2,
    ).eval()


def _single_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    residues = 4
    times = torch.tensor([0.17], dtype=torch.float32)
    trunk_values = torch.linspace(-0.3, 0.3, steps=residues * 384)
    input_values = torch.linspace(0.2, -0.2, steps=residues * 384)
    return (
        times,
        trunk_values.reshape(1, residues, 384),
        input_values.reshape(1, residues, 384),
    )


def _pairwise_inputs() -> tuple[torch.Tensor, torch.Tensor]:
    residues = 4
    z_values = torch.linspace(-0.2, 0.2, steps=residues * residues * 128)
    rel_pos_values = torch.linspace(-1.0, 1.0, steps=residues * residues * 3)
    return (
        z_values.reshape(1, residues, residues, 128),
        rel_pos_values.reshape(1, residues, residues, 3),
    )
