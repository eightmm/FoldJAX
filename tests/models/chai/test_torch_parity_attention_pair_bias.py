"""Parity for Chai trunk single attention with pair bias."""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.official_parity

torch = pytest.importorskip("torch")

import jax.numpy as jnp  # noqa: E402

from foldjax.models.chai.models.pairformer import (  # noqa: E402
    attention_pair_bias,
    map_attention_pair_bias,
)
from foldjax.models.chai.models.primitives import linear  # noqa: E402


@pytest.fixture(scope="module")
def attention_child(chai_trunk_module):
    block0 = getattr(chai_trunk_module.pairformer_stack.blocks, "0")
    return block0.attention_pair_bias


def _state(child):
    return {
        name: value.detach().cpu().numpy()
        for name, value in child.state_dict().items()
    }


def _torch_attention(s, z, pair_mask, token_mask, child, *, bf16: bool):
    state = _state(child)
    single = torch.nn.functional.layer_norm(
        s.float(),
        (s.shape[-1],),
        torch.from_numpy(state["single_layer_norm.weight"]),
        torch.from_numpy(state["single_layer_norm.bias"]),
    )
    pair = torch.nn.functional.layer_norm(
        z.float(),
        (z.shape[-1],),
        torch.from_numpy(state["pair_layer_norm.weight"]),
        torch.from_numpy(state["pair_layer_norm.bias"]),
    )
    pair_weight = torch.from_numpy(state["pair_linear.weight"])
    if bf16:
        pair = pair.bfloat16()
        pair_weight = pair_weight.bfloat16()
    bias = torch.nn.functional.linear(pair, pair_weight).permute(0, 3, 1, 2)
    bias = bias.masked_fill(~pair_mask[:, None], -10000)

    if bf16:
        method = child.attention.input2qkvg._c._get_method("forward")
        q, k, v, gate = method(single).unbind(0)
    else:
        qkvg_weight = torch.from_numpy(state["attention.input2qkvg.weight"])
        q, k, v, gate = torch.einsum(
            "bnc,cxhd->xbhnd", single, qkvg_weight
        ).unbind(0)
    query_bias = torch.from_numpy(state["attention.query_bias"])
    q = q + query_bias[None, :, None, :]
    if bf16:
        q = q.bfloat16()
    attended = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=bias
    )
    gated = attended * torch.sigmoid(gate + 1)
    output_weight = torch.from_numpy(state["attention.output_proj.weight"])
    expected = torch.einsum("bhnd,hdc->bnc", gated, output_weight.to(gated.dtype))
    return expected * token_mask[..., None]


def _inputs(seed: int):
    rng = np.random.default_rng(seed)
    s = rng.normal(size=(1, 6, 384)).astype(np.float32)
    z = rng.normal(size=(1, 6, 6, 256)).astype(np.float32)
    pair_mask = np.asarray(
        [
            [
                [1, 1, 1, 1, 0, 0],
                [1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1],
                [0, 1, 1, 1, 1, 1],
                [0, 0, 1, 1, 1, 1],
            ]
        ],
        dtype=bool,
    )
    token_mask = np.asarray([[1, 1, 1, 1, 0, 0]], dtype=bool)
    return s, z, pair_mask, token_mask


def test_attention_pair_bias_fp32_matches_torch(attention_child) -> None:
    s, z, pair_mask, token_mask = _inputs(71)
    state = _state(attention_child)
    expected = _torch_attention(
        torch.from_numpy(s),
        torch.from_numpy(z),
        torch.from_numpy(pair_mask),
        torch.from_numpy(token_mask),
        attention_child,
        bf16=False,
    )
    actual = attention_pair_bias(
        jnp.asarray(s),
        jnp.asarray(z),
        jnp.asarray(pair_mask),
        jnp.asarray(token_mask),
        map_attention_pair_bias(state),
        lin=linear,
    )
    np.testing.assert_allclose(
        np.asarray(actual), expected.numpy(), rtol=3e-4, atol=3e-4
    )


def test_attention_pair_bias_bf16_matches_torch(attention_child) -> None:
    s, z, pair_mask, token_mask = _inputs(72)
    state = _state(attention_child)
    expected = _torch_attention(
        torch.from_numpy(s),
        torch.from_numpy(z),
        torch.from_numpy(pair_mask),
        torch.from_numpy(token_mask),
        attention_child,
        bf16=True,
    ).float().detach().numpy()
    actual = np.asarray(
        attention_pair_bias(
            jnp.asarray(s),
            jnp.asarray(z),
            jnp.asarray(pair_mask),
            jnp.asarray(token_mask),
            map_attention_pair_bias(state),
        ),
        dtype=np.float32,
    )
    error = actual - expected
    nrmse = float(np.sqrt(np.mean(error**2)) / np.sqrt(np.mean(expected**2)))
    correlation = float(np.corrcoef(actual.ravel(), expected.ravel())[0, 1])
    assert float(np.max(np.abs(error))) <= 0.25
    assert nrmse <= 1e-2
    assert correlation >= 0.9999
