"""Torch-vs-JAX parity for OpenFold3's multi-head attention."""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.openfold3.bridge.torch_mapping import map_attention
from foldjax.models.openfold3.models.attention import attention, split_heads

pytestmark = pytest.mark.torch_parity

RTOL = 1e-4
ATOL = 1e-4

C_Q, C_K, C_HIDDEN, HEADS = 12, 10, 4, 3


def _torch():
    import torch

    torch.manual_seed(0)
    return torch


def _assert_close(actual: jnp.ndarray, expected, name: str) -> None:
    np.testing.assert_allclose(
        np.asarray(actual, dtype=np.float64),
        np.asarray(expected.detach().numpy(), dtype=np.float64),
        rtol=RTOL,
        atol=ATOL,
        err_msg=f"{name} diverged from the OpenFold3 reference",
    )


def _module(torch, gating: bool):
    from openfold3.core.model.primitives import Attention

    return Attention(
        c_q=C_Q,
        c_k=C_K,
        c_v=C_K,
        c_hidden=C_HIDDEN,
        no_heads=HEADS,
        gating=gating,
    )


@pytest.mark.parametrize("gating", [True, False])
def test_attention_matches_torch(
    openfold3_source: Path, randomized, gating: bool
) -> None:
    torch = _torch()
    module = randomized(_module(torch, gating))
    q_x = torch.randn(2, 7, C_Q)
    kv_x = torch.randn(2, 5, C_K)
    with torch.no_grad():
        expected = module(q_x, kv_x)
    params = map_attention(dict(module.state_dict()))
    assert (params.linear_g is not None) == gating
    actual = attention(
        jnp.asarray(q_x.numpy()),
        jnp.asarray(kv_x.numpy()),
        params,
        no_heads=HEADS,
    )
    _assert_close(actual, expected, f"Attention(gating={gating})")


def test_attention_matches_torch_with_biases(
    openfold3_source: Path, randomized
) -> None:
    torch = _torch()
    module = randomized(_module(torch, gating=True))
    q_x = torch.randn(2, 7, C_Q)
    kv_x = torch.randn(2, 5, C_K)
    # One broadcast mask bias and one full pair bias, as the trunk supplies.
    mask_bias = torch.zeros(2, 1, 1, 5)
    mask_bias[:, :, :, 3:] = -1e9
    pair_bias = torch.randn(2, HEADS, 7, 5)
    with torch.no_grad():
        expected = module(q_x, kv_x, biases=[mask_bias, pair_bias])
    actual = attention(
        jnp.asarray(q_x.numpy()),
        jnp.asarray(kv_x.numpy()),
        map_attention(dict(module.state_dict())),
        no_heads=HEADS,
        biases=(jnp.asarray(mask_bias.numpy()), jnp.asarray(pair_bias.numpy())),
    )
    _assert_close(actual, expected, "Attention(biases)")


def test_attention_matches_torch_with_extra_batch_dims(
    openfold3_source: Path, randomized
) -> None:
    torch = _torch()
    module = randomized(_module(torch, gating=True))
    q_x = torch.randn(2, 3, 7, C_Q)
    kv_x = torch.randn(2, 3, 5, C_K)
    with torch.no_grad():
        expected = module(q_x, kv_x)
    actual = attention(
        jnp.asarray(q_x.numpy()),
        jnp.asarray(kv_x.numpy()),
        map_attention(dict(module.state_dict())),
        no_heads=HEADS,
    )
    _assert_close(actual, expected, "Attention(extra batch dims)")


def test_head_split_matches_torch_view_order(openfold3_source: Path) -> None:
    """Head-major vs channel-major reshape both give valid shapes; pin the order."""
    torch = _torch()
    flat = torch.arange(2 * 1 * (HEADS * C_HIDDEN), dtype=torch.float32).reshape(
        2, 1, HEADS * C_HIDDEN
    )
    expected = flat.view(flat.shape[:-1] + (HEADS, -1))
    actual = split_heads(jnp.asarray(flat.numpy()), HEADS)
    _assert_close(actual, expected, "split_heads")


def test_mapper_reports_a_missing_projection(openfold3_source: Path) -> None:
    torch = _torch()
    state = dict(_module(torch, gating=True).state_dict())
    del state["linear_v.weight"]
    with pytest.raises(KeyError, match="linear_v.weight"):
        map_attention(state)


def test_attention_state_dict_layout(openfold3_source: Path) -> None:
    torch = _torch()
    assert set(_module(torch, gating=True).state_dict()) == {
        "linear_q.weight",
        "linear_k.weight",
        "linear_v.weight",
        "linear_o.weight",
        "linear_g.weight",
    }
    assert "linear_g.weight" not in _module(torch, gating=False).state_dict()
