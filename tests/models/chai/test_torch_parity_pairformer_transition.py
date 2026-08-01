"""Parity gates for Chai trunk Pairformer transitions using official weights."""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.official_parity

torch = pytest.importorskip("torch")

import jax.numpy as jnp  # noqa: E402

from foldjax.models.chai.models.pairformer import (  # noqa: E402
    map_pairformer_transition,
    pairformer_transition,
)
from foldjax.models.chai.models.primitives import linear  # noqa: E402

_PREFIX = ""


@pytest.fixture(scope="module")
def transition_state(chai_trunk_module):
    block0 = getattr(chai_trunk_module.pairformer_stack.blocks, "0")
    return {
        name: value.detach().cpu().numpy()
        for name, value in block0.transition_pair.state_dict().items()
    }


def _torch_transition(x, state, *, bf16: bool):
    def weight(name):
        return torch.from_numpy(state[name])

    normalized = torch.nn.functional.layer_norm(
        x.float(),
        (x.shape[-1],),
        weight("layer_norm.weight"),
        weight("layer_norm.bias"),
    )
    if bf16:
        normalized = normalized.bfloat16()
    ab_weight = weight("linear_no_bias_ab.weight")
    out_weight = weight("linear_out.weight")
    if bf16:
        ab_weight = ab_weight.bfloat16()
        out_weight = out_weight.bfloat16()
    a, b = torch.chunk(
        torch.nn.functional.linear(normalized, ab_weight), 2, dim=-1
    )
    return torch.nn.functional.linear(torch.nn.functional.silu(a) * b, out_weight)


def test_pairformer_transition_fp32_matches_torch(transition_state) -> None:
    rng = np.random.default_rng(21)
    x = rng.normal(size=(1, 3, 4, 256)).astype(np.float32)
    params = map_pairformer_transition(transition_state, _PREFIX)

    expected = _torch_transition(torch.from_numpy(x), transition_state, bf16=False)
    actual = pairformer_transition(jnp.asarray(x), params, lin=linear)

    np.testing.assert_allclose(
        np.asarray(actual), expected.numpy(), rtol=1e-4, atol=1e-4
    )


def test_pairformer_transition_bf16_matches_torch(transition_state) -> None:
    rng = np.random.default_rng(22)
    x = rng.normal(size=(1, 3, 4, 256)).astype(np.float32)
    params = map_pairformer_transition(transition_state, _PREFIX)

    expected = _torch_transition(torch.from_numpy(x), transition_state, bf16=True)
    actual = pairformer_transition(jnp.asarray(x), params)

    assert actual.dtype == jnp.bfloat16
    actual_fp32 = np.asarray(actual, dtype=np.float32)
    expected_fp32 = expected.float().numpy()
    error = actual_fp32 - expected_fp32
    max_abs = float(np.max(np.abs(error)))
    nrmse = float(
        np.sqrt(np.mean(error**2)) / np.sqrt(np.mean(expected_fp32**2))
    )
    correlation = float(np.corrcoef(actual_fp32.ravel(), expected_fp32.ravel())[0, 1])
    assert max_abs <= 0.25
    assert nrmse <= 4e-3
    assert correlation >= 0.99999
