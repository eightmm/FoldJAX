import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
import torch

from foldjax.models.boltz2.bridge.torch_checkpoint import load_checkpoint_state_dict
from foldjax.models.boltz2.bridge.torch_mapping import map_msa_module_state_dict
from foldjax.models.boltz2.models.trunk_blocks.msa import (
    msa_module_forward,
    pair_weighted_averaging_forward,
)

CHECKPOINT = (
    Path(__file__).resolve().parents[4] / "boltz/.cache/boltz/boltz2_conf.ckpt"
)
BOLTZ_SRC = Path(__file__).resolve().parents[4] / "boltz/src"
PREFIX = "msa_module"


@pytest.fixture(scope="module")
def checkpoint_state() -> dict[str, torch.Tensor]:
    if not CHECKPOINT.exists():
        pytest.skip(f"Boltz-2 checkpoint not found: {CHECKPOINT}")
    return load_checkpoint_state_dict(CHECKPOINT)


def test_checkpoint_msa_module_matches_boltz_torch(
    checkpoint_state: dict[str, torch.Tensor],
) -> None:
    num_layers = 2
    torch_module = _load_torch_msa_module(checkpoint_state, num_layers)
    params = map_msa_module_state_dict(checkpoint_state, PREFIX, num_layers=num_layers)
    z, emb, feats = _msa_inputs()

    with torch.no_grad():
        expected = torch_module(z, emb, feats, use_kernels=False)
    actual = msa_module_forward(
        params,
        jnp.asarray(z.numpy()),
        jnp.asarray(emb.numpy()),
        _jax_feats(feats),
    )

    np.testing.assert_allclose(
        np.asarray(actual),
        expected.detach().numpy(),
        rtol=2e-3,
        atol=2e-3,
    )


def test_masked_msa_depth_padding_preserves_jax_output(
    checkpoint_state: dict[str, torch.Tensor],
) -> None:
    params = map_msa_module_state_dict(checkpoint_state, PREFIX, num_layers=2)
    z, emb, feats_torch = _msa_inputs()
    feats = _jax_feats(feats_torch)
    expected = msa_module_forward(
        params, jnp.asarray(z.numpy()), jnp.asarray(emb.numpy()), feats
    )
    padded = dict(feats)
    for key in ("msa", "has_deletion", "deletion_value", "msa_paired", "msa_mask"):
        padded[key] = jnp.pad(feats[key], ((0, 0), (0, 5), (0, 0)))

    actual = msa_module_forward(
        params, jnp.asarray(z.numpy()), jnp.asarray(emb.numpy()), padded
    )

    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), atol=2e-5)


def test_pair_weighted_averaging_preserves_bfloat16_activation_dtype() -> None:
    rng = np.random.default_rng(0)
    c_m, c_z, heads, c_h = 8, 12, 4, 3

    def w(*shape):
        return jnp.asarray(rng.standard_normal(shape) * 0.1, dtype=jnp.bfloat16)

    params = {
        "norm_m": {"scale": w(c_m), "bias": w(c_m)},
        "norm_z": {"scale": w(c_z), "bias": w(c_z)},
        "proj_m": {"kernel": w(c_m, heads * c_h)},
        "proj_z": {"kernel": w(c_z, heads)},
        "proj_g": {"kernel": w(c_m, heads * c_h)},
        "proj_o": {"kernel": w(heads * c_h, c_m)},
    }
    m = jnp.asarray(rng.standard_normal((1, 2, 5, c_m)), dtype=jnp.bfloat16)
    z = jnp.asarray(rng.standard_normal((1, 5, 5, c_z)), dtype=jnp.bfloat16)
    mask = jnp.ones((1, 5, 5), dtype=jnp.bfloat16)

    out = pair_weighted_averaging_forward(params, m, z, mask)

    assert out.dtype == jnp.bfloat16


def _load_torch_msa_module(
    state: dict[str, torch.Tensor],
    num_layers: int,
) -> torch.nn.Module:
    sys.path.insert(0, str(BOLTZ_SRC))
    from boltz.model.modules.trunkv2 import MSAModule

    module = MSAModule(
        msa_s=64,
        token_z=128,
        token_s=384,
        msa_blocks=num_layers,
        msa_dropout=0.15,
        z_dropout=0.25,
        pairwise_head_width=32,
        pairwise_num_heads=4,
        use_paired_feature=True,
    ).eval()
    module_state = {}
    for key, value in state.items():
        if not key.startswith(f"{PREFIX}."):
            continue
        local_key = key.removeprefix(f"{PREFIX}.")
        if local_key.startswith("layers."):
            index = int(local_key.split(".")[1])
            if index >= num_layers:
                continue
        module_state[local_key] = value
    module.load_state_dict(module_state)
    return module


def _msa_inputs() -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    tokens = 4
    msa_rows = 3
    z = torch.linspace(-0.1, 0.1, steps=tokens * tokens * 128).reshape(
        1, tokens, tokens, 128
    )
    emb = torch.linspace(0.2, -0.2, steps=tokens * 384).reshape(1, tokens, 384)
    feats = {
        "msa": (torch.arange(msa_rows * tokens).reshape(1, msa_rows, tokens) % 33),
        "has_deletion": torch.zeros(1, msa_rows, tokens),
        "deletion_value": torch.linspace(0.0, 1.0, steps=msa_rows * tokens).reshape(
            1, msa_rows, tokens
        ),
        "msa_paired": torch.ones(1, msa_rows, tokens),
        "msa_mask": torch.ones(1, msa_rows, tokens),
        "token_pad_mask": torch.tensor([[1.0, 1.0, 1.0, 0.0]]),
    }
    feats["msa_mask"][:, -1, -1] = 0.0
    return z, emb, feats


def _jax_feats(feats: dict[str, torch.Tensor]) -> dict[str, jnp.ndarray]:
    return {key: jnp.asarray(value.numpy()) for key, value in feats.items()}
