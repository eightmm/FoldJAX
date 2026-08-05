"""Torch-vs-JAX parity for ConditionedTransitionBlock (AF3 Algorithm 25)."""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.openfold3.bridge.torch_mapping import (
    map_conditioned_transition_block,
)
from foldjax.models.openfold3.models.primitives import conditioned_transition_block

pytestmark = pytest.mark.torch_parity

RTOL = 1e-4
ATOL = 1e-4

C_A, C_S, N = 8, 6, 5


def _torch():
    import torch

    torch.manual_seed(0)
    return torch


def _module():
    from openfold3.core.model.layers.transition import ConditionedTransitionBlock

    return ConditionedTransitionBlock(c_a=C_A, c_s=C_S, n=2)


def _assert_close(actual: jnp.ndarray, expected, name: str) -> None:
    np.testing.assert_allclose(
        np.asarray(actual, dtype=np.float64),
        np.asarray(expected.detach().numpy(), dtype=np.float64),
        rtol=RTOL,
        atol=ATOL,
        err_msg=f"{name} diverged from the OpenFold3 reference",
    )


def test_conditioned_transition_matches_torch(
    openfold3_source: Path, randomized
) -> None:
    torch = _torch()
    module = randomized(_module())
    a = torch.randn(1, N, C_A)
    s = torch.randn(1, N, C_S)
    with torch.no_grad():
        expected = module(a=a, s=s)
    actual = conditioned_transition_block(
        jnp.asarray(a.numpy()),
        jnp.asarray(s.numpy()),
        map_conditioned_transition_block(dict(module.state_dict())),
    )
    _assert_close(actual, expected, "ConditionedTransitionBlock")


def test_conditioned_transition_matches_torch_with_mask(
    openfold3_source: Path, randomized
) -> None:
    torch = _torch()
    module = randomized(_module())
    a = torch.randn(1, N, C_A)
    s = torch.randn(1, N, C_S)
    mask = torch.zeros(1, N)
    mask[:, :3] = 1.0
    with torch.no_grad():
        # forward() unsqueezes internally, so the mask is [*, N] here.
        expected = module(a=a, s=s, mask=mask)
    actual = conditioned_transition_block(
        jnp.asarray(a.numpy()),
        jnp.asarray(s.numpy()),
        map_conditioned_transition_block(dict(module.state_dict())),
        mask=jnp.asarray(mask.numpy()),
    )
    _assert_close(actual, expected, "ConditionedTransitionBlock(masked)")
    assert np.allclose(np.asarray(actual)[:, 3:], 0.0)


def test_gate_starts_near_closed_at_default_init(openfold3_source: Path) -> None:
    """gating_ada_zero sets linear_g.bias to -2, so sigmoid(-2) ~ 0.12."""
    _torch()
    module = _module()
    state = dict(module.state_dict())
    np.testing.assert_allclose(
        state["linear_g.bias"].numpy(), np.full(C_A, -2.0), rtol=1e-6
    )
    # linear_out is zero-init only for "final"; here it is "default", so the
    # block is not identically zero at init — but the gate keeps it small.
    assert float(state["linear_out.weight"].abs().sum()) > 0.0


def test_output_is_not_a_residual(openfold3_source: Path, randomized) -> None:
    """Unlike SwiGLUTransition, this block replaces rather than adds to `a`."""
    torch = _torch()
    module = randomized(_module())
    a = torch.randn(1, N, C_A)
    s = torch.randn(1, N, C_S)
    with torch.no_grad():
        expected = module(a=a, s=s)
    # If the port had added a residual, the difference would be exactly `a`.
    diff = expected.detach().numpy() - a.numpy()
    assert not np.allclose(diff, 0.0, atol=1e-6)


def test_state_dict_layout(openfold3_source: Path) -> None:
    _torch()
    assert set(_module().state_dict()) == {
        "layer_norm.layer_norm_s.weight",
        "layer_norm.linear_g.weight",
        "layer_norm.linear_g.bias",
        "layer_norm.linear_s.weight",
        "swiglu.linear_a.weight",
        "swiglu.linear_b.weight",
        "linear_g.weight",
        "linear_g.bias",
        "linear_out.weight",
    }


def test_mapper_reports_a_missing_gate_bias(openfold3_source: Path) -> None:
    _torch()
    state = dict(_module().state_dict())
    del state["linear_g.bias"]
    with pytest.raises(KeyError, match="linear_g.bias"):
        map_conditioned_transition_block(state)
