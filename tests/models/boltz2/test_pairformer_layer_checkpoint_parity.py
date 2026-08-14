import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
import torch

from foldjax.models.boltz2.bridge.torch_mapping import map_pairformer_layer_state_dict
from foldjax.models.boltz2.models.trunk_blocks.pairformer import (
    pairformer_layer_forward,
)
from tests.models.boltz2.torch_checkpoint import load_torch_checkpoint_state_dict

CHECKPOINT = (
    Path(__file__).resolve().parents[4] / "boltz/.cache/boltz/boltz2_conf.ckpt"
)
BOLTZ_SRC = Path(__file__).resolve().parents[4] / "boltz/src"
PREFIX = "pairformer_module.layers.0"


@pytest.fixture(scope="module")
def checkpoint_state() -> dict[str, torch.Tensor]:
    if not CHECKPOINT.exists():
        pytest.skip(f"Boltz-2 checkpoint not found: {CHECKPOINT}")
    return load_torch_checkpoint_state_dict(CHECKPOINT)


def test_checkpoint_pairformer_layer_matches_boltz_torch(
    checkpoint_state: dict[str, torch.Tensor],
) -> None:
    torch_layer = _load_torch_pairformer_layer(checkpoint_state)
    params = map_pairformer_layer_state_dict(checkpoint_state, PREFIX)
    s, z, mask, pair_mask = _pairformer_inputs()

    with torch.no_grad():
        expected_s, expected_z = torch_layer(
            s,
            z,
            mask,
            pair_mask,
            chunk_size_tri_attn=None,
            use_kernels=False,
        )
    actual_s, actual_z = pairformer_layer_forward(
        params,
        jnp.asarray(s.numpy()),
        jnp.asarray(z.numpy()),
        jnp.asarray(mask.numpy()),
        jnp.asarray(pair_mask.numpy()),
    )

    np.testing.assert_allclose(
        np.asarray(actual_s),
        expected_s.detach().numpy(),
        rtol=8e-4,
        atol=8e-4,
    )
    np.testing.assert_allclose(
        np.asarray(actual_z),
        expected_z.detach().numpy(),
        rtol=8e-4,
        atol=8e-4,
    )


def _load_torch_pairformer_layer(
    state: dict[str, torch.Tensor],
) -> torch.nn.Module:
    sys.path.insert(0, str(BOLTZ_SRC))
    from boltz.model.layers.pairformer import PairformerLayer

    layer = PairformerLayer(
        token_s=384,
        token_z=128,
        num_heads=16,
        pairwise_head_width=32,
        pairwise_num_heads=4,
        v2=True,
    ).eval()
    layer_state = {
        key.removeprefix(f"{PREFIX}."): value
        for key, value in state.items()
        if key.startswith(f"{PREFIX}.")
    }
    layer.load_state_dict(layer_state)
    return layer


def _pairformer_inputs() -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    residues = 4
    s_values = torch.linspace(-0.5, 0.5, steps=residues * 384, dtype=torch.float32)
    z_values = torch.linspace(-0.25, 0.25, steps=residues * residues * 128)
    s = s_values.reshape(1, residues, 384)
    z = z_values.reshape(1, residues, residues, 128)
    mask = torch.tensor([[1.0, 1.0, 0.0, 1.0]], dtype=torch.float32)
    pair_mask = mask[:, :, None] * mask[:, None, :]
    return s, z, mask, pair_mask
