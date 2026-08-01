"""Torch-free unit tests for state_dict mappers."""

from __future__ import annotations

import numpy as np

from foldjax.models.chai.bridge.torch_mapping import (
    apply_layer_norm,
    apply_linear,
    map_bond_loss_input_proj,
    map_layer_norm,
    map_linear,
)


def test_map_linear_keeps_torch_layout() -> None:
    state = {
        "proj.weight": np.arange(6, dtype=np.float32).reshape(2, 3),
        "proj.bias": np.array([0.5, -0.5], dtype=np.float32),
    }
    p = map_linear(state, "proj")
    np.testing.assert_array_equal(np.asarray(p.weight), state["proj.weight"])
    x = np.ones((4, 3), dtype=np.float32)
    y = np.asarray(apply_linear(p, x))
    assert y.shape == (4, 2)
    np.testing.assert_allclose(y[0], state["proj.weight"].sum(1) + state["proj.bias"])


def test_map_bond_loss_input_proj_no_bias() -> None:
    state = {"weight": np.random.randn(512, 1).astype(np.float32)}
    p = map_bond_loss_input_proj(state)
    assert p.bias is None
    assert tuple(p.weight.shape) == (512, 1)


def test_layer_norm_matches_reference() -> None:
    state = {"ln.weight": np.ones(3, np.float32), "ln.bias": np.zeros(3, np.float32)}
    p = map_layer_norm(state, "ln")
    x = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    y = np.asarray(apply_layer_norm(p, x))
    ref = (x - x.mean()) / np.sqrt(x.var() + 1e-5)
    np.testing.assert_allclose(y, ref, rtol=1e-5, atol=1e-5)
