from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
import torch.nn.functional as functional

from foldjax.models.boltz2.bridge.torch_mapping import (
    map_attention_pair_bias_state_dict,
)
from foldjax.models.boltz2.models.primitives.attention import (
    attention_pair_bias_forward,
)
from tests.models.boltz2.torch_checkpoint import load_torch_checkpoint_state_dict

PREFIX = "pairformer_module.layers.0.attention"
CHECKPOINT = (
    Path(__file__).resolve().parents[4] / "boltz/.cache/boltz/boltz2_conf.ckpt"
)


@pytest.fixture(scope="module")
def attention_state() -> dict[str, torch.Tensor]:
    if not CHECKPOINT.exists():
        pytest.skip(f"Boltz-2 checkpoint not found: {CHECKPOINT}")
    return load_torch_checkpoint_state_dict(CHECKPOINT)


def test_checkpoint_attention_pair_bias_matches_torch(
    attention_state: dict[str, torch.Tensor],
) -> None:
    inputs = _attention_inputs(attention_state)
    params = map_attention_pair_bias_state_dict(attention_state, PREFIX)

    expected = _torch_attention_pair_bias_forward(attention_state, **inputs)
    actual = attention_pair_bias_forward(
        params,
        jnp.asarray(inputs["s"].numpy()),
        jnp.asarray(inputs["z"].numpy()),
        jnp.asarray(inputs["mask"].numpy()),
        k_in=jnp.asarray(inputs["k_in"].numpy()),
    )

    np.testing.assert_allclose(
        np.asarray(actual),
        expected.detach().numpy(),
        rtol=2e-4,
        atol=2e-4,
    )


def test_checkpoint_attention_pair_bias_accepts_flash_backend(
    attention_state: dict[str, torch.Tensor],
) -> None:
    inputs = _attention_inputs(attention_state)
    params = map_attention_pair_bias_state_dict(attention_state, PREFIX)
    s = jnp.asarray(inputs["s"].numpy())
    z = jnp.asarray(inputs["z"].numpy())
    mask = jnp.asarray(inputs["mask"].numpy())
    k_in = jnp.asarray(inputs["k_in"].numpy())

    expected = attention_pair_bias_forward(params, s, z, mask, k_in=k_in)
    compiled = jax.jit(
        lambda p, s_, z_, mask_, k_: attention_pair_bias_forward(
            p,
            s_,
            z_,
            mask_,
            k_in=k_,
            attention_backend="tokamax",
        )
    )
    actual = compiled(params, s, z, mask, k_in)

    np.testing.assert_allclose(
        np.asarray(actual),
        np.asarray(expected),
        rtol=2e-3,
        atol=2e-3,
    )


def test_attention_pair_bias_mapping_reports_missing_key(
    attention_state: dict[str, torch.Tensor],
) -> None:
    state = dict(attention_state)
    missing_key = f"{PREFIX}.proj_z.1.weight"
    del state[missing_key]

    with pytest.raises(KeyError, match="Missing required AttentionPairBias"):
        map_attention_pair_bias_state_dict(state, PREFIX)


def _attention_inputs(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    token_s = state[f"{PREFIX}.proj_q.weight"].shape[1]
    token_z = state[f"{PREFIX}.proj_z.1.weight"].shape[1]
    values_s = torch.linspace(-0.75, 0.75, steps=1 * 5 * token_s, dtype=torch.float32)
    values_z = torch.linspace(-0.5, 0.5, steps=1 * 5 * 5 * token_z, dtype=torch.float32)
    return {
        "s": values_s.reshape(1, 5, token_s),
        "k_in": values_s.reshape(1, 5, token_s).flip(dims=(-2,)),
        "z": values_z.reshape(1, 5, 5, token_z),
        "mask": torch.tensor([[1.0, 1.0, 1.0, 0.0, 1.0]], dtype=torch.float32),
    }


def _torch_attention_pair_bias_forward(
    state: dict[str, torch.Tensor],
    s: torch.Tensor,
    z: torch.Tensor,
    mask: torch.Tensor,
    k_in: torch.Tensor,
) -> torch.Tensor:
    batch = s.shape[0]
    num_heads = state[f"{PREFIX}.proj_z.1.weight"].shape[0]
    head_dim = s.shape[-1] // num_heads

    q = functional.linear(
        s,
        state[f"{PREFIX}.proj_q.weight"],
        state[f"{PREFIX}.proj_q.bias"],
    ).reshape(batch, -1, num_heads, head_dim)
    k = functional.linear(k_in, state[f"{PREFIX}.proj_k.weight"]).reshape(
        batch, -1, num_heads, head_dim
    )
    v = functional.linear(k_in, state[f"{PREFIX}.proj_v.weight"]).reshape(
        batch, -1, num_heads, head_dim
    )
    bias = functional.layer_norm(
        z,
        normalized_shape=(z.shape[-1],),
        weight=state[f"{PREFIX}.proj_z.0.weight"],
        bias=state[f"{PREFIX}.proj_z.0.bias"],
        eps=1e-5,
    )
    bias = functional.linear(bias, state[f"{PREFIX}.proj_z.1.weight"])
    bias = bias.permute(0, 3, 1, 2)
    gate = torch.sigmoid(functional.linear(s, state[f"{PREFIX}.proj_g.weight"]))

    attn = torch.einsum("bihd,bjhd->bhij", q.float(), k.float())
    attn = attn / (head_dim**0.5) + bias.float()
    attn = attn + (1 - mask[:, None, None].float()) * -1e6
    attn = attn.softmax(dim=-1)
    out = torch.einsum("bhij,bjhd->bihd", attn, v.float()).to(v.dtype)
    out = out.reshape(batch, -1, s.shape[-1])
    return functional.linear(gate * out, state[f"{PREFIX}.proj_o.weight"])
