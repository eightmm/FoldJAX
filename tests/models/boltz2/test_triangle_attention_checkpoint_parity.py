from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
import torch
import torch.nn.functional as functional

from foldjax.models.boltz2.bridge.torch_mapping import map_triangle_attention_state_dict
from foldjax.models.boltz2.models.triangle.triangle_attention import (
    triangle_attention_forward,
)
from tests.models.boltz2.torch_checkpoint import load_torch_checkpoint_state_dict

CHECKPOINT = (
    Path(__file__).resolve().parents[4] / "boltz/.cache/boltz/boltz2_conf.ckpt"
)


@pytest.fixture(scope="module")
def checkpoint_state() -> dict[str, torch.Tensor]:
    if not CHECKPOINT.exists():
        pytest.skip(f"Boltz-2 checkpoint not found: {CHECKPOINT}")
    return load_torch_checkpoint_state_dict(CHECKPOINT)


@pytest.mark.parametrize(
    ("prefix", "starting"),
    [
        ("pairformer_module.layers.0.tri_att_start", True),
        ("pairformer_module.layers.0.tri_att_end", False),
    ],
)
def test_checkpoint_triangle_attention_matches_torch(
    checkpoint_state: dict[str, torch.Tensor],
    prefix: str,
    starting: bool,
) -> None:
    x, mask = _triangle_attention_inputs(checkpoint_state, prefix)
    params = map_triangle_attention_state_dict(checkpoint_state, prefix)

    expected = _torch_triangle_attention_forward(
        checkpoint_state,
        prefix,
        x,
        mask,
        starting=starting,
    )
    actual = triangle_attention_forward(
        params,
        jnp.asarray(x.numpy()),
        jnp.asarray(mask.numpy()),
        starting=starting,
    )

    np.testing.assert_allclose(
        np.asarray(actual),
        expected.detach().numpy(),
        rtol=3e-4,
        atol=3e-4,
    )


def test_triangle_attention_mapping_reports_missing_key(
    checkpoint_state: dict[str, torch.Tensor],
) -> None:
    prefix = "pairformer_module.layers.0.tri_att_start"
    state = dict(checkpoint_state)
    del state[f"{prefix}.mha.linear_v.weight"]

    with pytest.raises(KeyError, match="Missing required TriangleAttention"):
        map_triangle_attention_state_dict(state, prefix)


def _triangle_attention_inputs(
    state: dict[str, torch.Tensor],
    prefix: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    dim = state[f"{prefix}.layer_norm.weight"].shape[0]
    values = torch.linspace(-0.5, 0.5, steps=1 * 4 * 4 * dim, dtype=torch.float32)
    x = values.reshape(1, 4, 4, dim)
    mask = torch.tensor(
        [[
            [1.0, 1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0, 1.0],
            [0.0, 1.0, 1.0, 1.0],
        ]],
        dtype=torch.float32,
    )
    return x, mask


def _torch_triangle_attention_forward(
    state: dict[str, torch.Tensor],
    prefix: str,
    x: torch.Tensor,
    mask: torch.Tensor,
    starting: bool,
) -> torch.Tensor:
    if not starting:
        x = x.transpose(-2, -3)
        mask = mask.transpose(-1, -2)

    x = functional.layer_norm(
        x,
        normalized_shape=(x.shape[-1],),
        weight=state[f"{prefix}.layer_norm.weight"],
        bias=state[f"{prefix}.layer_norm.bias"],
        eps=1e-5,
    )
    mask_bias = 1e9 * (mask[..., :, None, None, :] - 1)
    tri_bias = functional.linear(x, state[f"{prefix}.linear.weight"])
    tri_bias = tri_bias.permute(0, 3, 1, 2).unsqueeze(1)

    out = _torch_mha(state, prefix, x, x, tri_bias, mask_bias)
    if not starting:
        out = out.transpose(-2, -3)
    return out


def _torch_mha(
    state: dict[str, torch.Tensor],
    prefix: str,
    q_x: torch.Tensor,
    kv_x: torch.Tensor,
    tri_bias: torch.Tensor,
    mask_bias: torch.Tensor,
) -> torch.Tensor:
    heads = state[f"{prefix}.linear.weight"].shape[0]
    hidden = state[f"{prefix}.mha.linear_q.weight"].shape[0] // heads
    q = functional.linear(q_x, state[f"{prefix}.mha.linear_q.weight"])
    k = functional.linear(kv_x, state[f"{prefix}.mha.linear_k.weight"])
    v = functional.linear(kv_x, state[f"{prefix}.mha.linear_v.weight"])
    q = q.reshape(q.shape[:-1] + (heads, hidden)).transpose(-2, -3)
    k = k.reshape(k.shape[:-1] + (heads, hidden)).transpose(-2, -3)
    v = v.reshape(v.shape[:-1] + (heads, hidden)).transpose(-2, -3)
    q = q / (hidden**0.5)
    scores = torch.matmul(q, k.transpose(-1, -2))
    scores = scores + mask_bias + tri_bias
    attn = scores.softmax(dim=-1)
    out = torch.matmul(attn, v).transpose(-2, -3)
    gate = torch.sigmoid(functional.linear(q_x, state[f"{prefix}.mha.linear_g.weight"]))
    gate = gate.reshape(gate.shape[:-1] + (heads, hidden))
    out = out * gate
    out = out.reshape(out.shape[:-2] + (heads * hidden,))
    return functional.linear(out, state[f"{prefix}.mha.linear_o.weight"])
