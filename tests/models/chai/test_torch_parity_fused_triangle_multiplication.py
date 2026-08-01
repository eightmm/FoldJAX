"""Parity for Chai's fused outgoing+incoming triangle multiplication."""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.official_parity

torch = pytest.importorskip("torch")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from foldjax.models.chai.models.pairformer import (  # noqa: E402
    fused_triangle_multiplication,
    map_fused_triangle_multiplication,
)
from foldjax.models.chai.models.primitives import (  # noqa: E402
    layer_norm,
    linear,
    linear_bf16,
)


@pytest.fixture(scope="module")
def triangle_state(chai_trunk_module):
    block0 = getattr(chai_trunk_module.pairformer_stack.blocks, "0")
    return {
        name: value.detach().cpu().numpy()
        for name, value in block0.triangle_multiplication.state_dict().items()
    }


def _torch_triangle(z, mask, state, *, bf16: bool):
    def weight(name):
        value = torch.from_numpy(state[name])
        return value.bfloat16() if bf16 else value

    c_z = z.shape[-1]
    normalized = torch.nn.functional.layer_norm(
        z.float(),
        (c_z,),
        torch.from_numpy(state["layernorm_z_in.weight"]),
        torch.from_numpy(state["layernorm_z_in.bias"]),
    )
    if bf16:
        normalized = normalized.bfloat16()
    p = torch.nn.functional.linear(normalized, weight("merged_linear_p.weight"))
    gates = torch.sigmoid(
        torch.nn.functional.linear(normalized, weight("merged_linear_g.weight"))
    )
    ab = p * gates[..., :-c_z]
    out_gate = gates[..., -c_z:]
    outgoing, incoming = torch.chunk(ab, 2, dim=-1)
    outgoing = outgoing.masked_fill(~mask[..., None], 0)
    incoming = incoming.masked_fill(~mask.transpose(1, 2)[..., None], 0)
    out_a, out_b = torch.chunk(outgoing, 2, dim=-1)
    in_a, in_b = torch.chunk(incoming, 2, dim=-1)
    out = torch.einsum("bikd,bjkd->bijd", out_a, out_b)
    inc = torch.einsum("bkid,bkjd->bijd", in_a, in_b)
    out = torch.nn.functional.layer_norm(out.float(), (c_z,))
    inc = torch.nn.functional.layer_norm(inc.float(), (c_z,))
    merged = out + inc
    if bf16:
        merged = merged.bfloat16()
    projected = torch.nn.functional.linear(
        merged, weight("linear_z_out.weight")
    )
    return projected * out_gate


def _inputs(seed: int = 51):
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


def _merged_jax_triangle(z, mask, params):
    c_z = z.shape[-1]
    normalized = layer_norm(
        z.astype(jnp.float32),
        params.layer_norm_weight,
        params.layer_norm_bias,
    )
    p = linear_bf16(normalized, params.merged_linear_p_weight)
    gates = jax.nn.sigmoid(
        linear_bf16(normalized, params.merged_linear_g_weight)
    )
    outgoing, incoming = jnp.split(p * gates[..., :-c_z], 2, axis=-1)
    outgoing = jnp.where(mask[..., None], outgoing, 0)
    incoming = jnp.where(jnp.swapaxes(mask, -1, -2)[..., None], incoming, 0)
    out_a, out_b = jnp.split(outgoing, 2, axis=-1)
    in_a, in_b = jnp.split(incoming, 2, axis=-1)
    out = jnp.einsum(
        "...ikd,...jkd->...ijd",
        out_a,
        out_b,
        preferred_element_type=jnp.float32,
    ).astype(out_a.dtype)
    inc = jnp.einsum(
        "...kid,...kjd->...ijd",
        in_a,
        in_b,
        preferred_element_type=jnp.float32,
    ).astype(in_a.dtype)
    merged = layer_norm(out.astype(jnp.float32)) + layer_norm(
        inc.astype(jnp.float32)
    )
    projected = linear_bf16(merged, params.linear_z_out_weight)
    return projected * gates[..., -c_z:]


def test_fused_triangle_multiplication_fp32_matches_torch(triangle_state) -> None:
    z, mask = _inputs()
    params = map_fused_triangle_multiplication(triangle_state)
    expected = _torch_triangle(
        torch.from_numpy(z), torch.from_numpy(mask), triangle_state, bf16=False
    )

    actual = fused_triangle_multiplication(
        jnp.asarray(z), jnp.asarray(mask), params, lin=linear
    )

    np.testing.assert_allclose(
        np.asarray(actual), expected.numpy(), rtol=2e-4, atol=2e-4
    )


def test_fused_triangle_multiplication_bf16_matches_torch(triangle_state) -> None:
    z, mask = _inputs(52)
    params = map_fused_triangle_multiplication(triangle_state)
    expected = _torch_triangle(
        torch.from_numpy(z), torch.from_numpy(mask), triangle_state, bf16=True
    ).float().numpy()

    actual = np.asarray(
        fused_triangle_multiplication(jnp.asarray(z), jnp.asarray(mask), params),
        dtype=np.float32,
    )
    error = actual - expected
    nrmse = float(np.sqrt(np.mean(error**2)) / np.sqrt(np.mean(expected**2)))
    correlation = float(np.corrcoef(actual.ravel(), expected.ravel())[0, 1])
    assert float(np.max(np.abs(error))) <= 0.5
    assert nrmse <= 1e-2
    assert correlation >= 0.9999


def test_split_projection_matches_merged_bf16_exactly(triangle_state) -> None:
    z, mask = _inputs(53)
    params = map_fused_triangle_multiplication(triangle_state)

    expected = _merged_jax_triangle(jnp.asarray(z), jnp.asarray(mask), params)
    actual = fused_triangle_multiplication(
        jnp.asarray(z), jnp.asarray(mask), params
    )

    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
