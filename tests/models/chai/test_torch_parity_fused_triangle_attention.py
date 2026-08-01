"""Parity for Chai's fused two-direction triangle attention."""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.official_parity

torch = pytest.importorskip("torch")

import jax.numpy as jnp  # noqa: E402

from foldjax.models.chai.models.pairformer import (  # noqa: E402
    fused_triangle_attention,
    map_fused_triangle_attention,
)
from foldjax.models.chai.models.primitives import linear  # noqa: E402


@pytest.fixture(scope="module")
def triangle_attention_state(chai_trunk_module):
    block0 = getattr(chai_trunk_module.pairformer_stack.blocks, "0")
    return {
        name: value.detach().cpu().numpy()
        for name, value in block0.triangle_attention.state_dict().items()
    }


def _torch_direction(x, weight, bias, *, heads: int):
    batch, rows, columns, _ = x.shape
    projected = torch.nn.functional.linear(x, weight)
    head_dim = projected.shape[-1] // (4 * heads)
    qkvg = projected.reshape(batch, rows, columns, heads, 4, head_dim)
    q, k, v, gate = qkvg.permute(4, 0, 3, 1, 2, 5).unbind(0)
    q = q.reshape(batch * heads, rows, columns, head_dim)
    k = k.reshape(batch * heads, rows, columns, head_dim)
    v = v.reshape(batch * heads, rows, columns, head_dim)
    gate = gate.reshape(batch * heads, rows, columns, head_dim)
    attended = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=bias.reshape(batch * heads, 1, rows, columns)
    )
    attended = attended * torch.sigmoid(gate)
    attended = attended.reshape(batch, heads, rows, columns, head_dim)
    return attended.permute(0, 2, 3, 1, 4).reshape(
        batch, rows, columns, heads * head_dim
    )


def _torch_attention(z, mask, state, *, bf16: bool):
    def weight(name):
        value = torch.from_numpy(state[name])
        return value.bfloat16() if bf16 else value

    heads = state["pair2b.weight"].shape[0] // 2
    pair = torch.nn.functional.layer_norm(z.float(), (z.shape[-1],))
    if bf16:
        pair = pair.bfloat16()
    bias = torch.nn.functional.linear(pair, weight("pair2b.weight"))
    bias = bias.reshape(*bias.shape[:-1], 2, heads).permute(0, 3, 4, 1, 2)
    bias = bias.masked_fill(~mask[:, None, None], -10000)
    first = _torch_direction(pair, weight("pair2qkvg1.weight"), bias[:, 0], heads=heads)
    second = _torch_direction(
        pair.transpose(1, 2),
        weight("pair2qkvg2.weight"),
        bias[:, 1],
        heads=heads,
    )
    merged = torch.cat([first, second], dim=-1)
    output_weight = torch.from_numpy(state["linear_out.weight"])
    output_weight = output_weight * torch.from_numpy(state["out_scalers"])[:, None]
    if bf16:
        output_weight = output_weight.bfloat16()
    return torch.nn.functional.linear(merged, output_weight)


def _inputs(seed: int):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(1, 5, 5, 256)).astype(np.float32)
    mask = np.asarray(
        [
            [
                [1, 1, 1, 0, 0],
                [1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1],
                [0, 1, 1, 1, 1],
                [0, 0, 1, 1, 1],
            ]
        ],
        dtype=bool,
    )
    return z, mask


def test_fused_triangle_attention_fp32_matches_torch(
    triangle_attention_state,
) -> None:
    z, mask = _inputs(61)
    params = map_fused_triangle_attention(triangle_attention_state)
    expected = _torch_attention(
        torch.from_numpy(z),
        torch.from_numpy(mask),
        triangle_attention_state,
        bf16=False,
    )

    actual = fused_triangle_attention(
        jnp.asarray(z), jnp.asarray(mask), params, lin=linear
    )

    np.testing.assert_allclose(
        np.asarray(actual), expected.numpy(), rtol=3e-4, atol=3e-4
    )


def test_fused_triangle_attention_bf16_matches_torch(
    triangle_attention_state,
) -> None:
    z, mask = _inputs(62)
    params = map_fused_triangle_attention(triangle_attention_state)
    expected = (
        _torch_attention(
            torch.from_numpy(z),
            torch.from_numpy(mask),
            triangle_attention_state,
            bf16=True,
        )
        .float()
        .numpy()
    )

    actual = np.asarray(
        fused_triangle_attention(jnp.asarray(z), jnp.asarray(mask), params),
        dtype=np.float32,
    )
    error = actual - expected
    nrmse = float(np.sqrt(np.mean(error**2)) / np.sqrt(np.mean(expected**2)))
    correlation = float(np.corrcoef(actual.ravel(), expected.ravel())[0, 1])
    assert float(np.max(np.abs(error))) <= 0.25
    assert nrmse <= 1e-2
    assert correlation >= 0.9999


def test_outer_chunked_triangle_attention_matches_dense_exactly(
    triangle_attention_state,
) -> None:
    z, mask = _inputs(63)
    params = map_fused_triangle_attention(triangle_attention_state)

    dense = fused_triangle_attention(
        jnp.asarray(z),
        jnp.asarray(mask),
        params,
        outer_chunk_size=None,
    )
    chunked = fused_triangle_attention(
        jnp.asarray(z),
        jnp.asarray(mask),
        params,
        outer_chunk_size=2,
    )

    np.testing.assert_array_equal(np.asarray(chunked), np.asarray(dense))
