"""Torch-vs-JAX numerical parity for every ported OpenFold3 primitive.

This gate exists from the first module because `protenix_jax` shipped a real
model bug that a parity test would have caught. Random initialization is enough:
the gate tests the forward math and the checkpoint key mapping, not the weights.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.openfold3.bridge.torch_mapping import (
    map_adaln,
    map_layer_norm,
    map_linear,
    map_swiglu,
    map_swiglu_transition,
)
from foldjax.models.openfold3.models.primitives import (
    adaln,
    layer_norm,
    linear,
    swiglu,
    swiglu_transition,
)

pytestmark = pytest.mark.torch_parity

# fp32 leaf tolerance, matching the sibling ports' convention.
RTOL = 1e-4
ATOL = 1e-4


def _torch():
    import torch

    torch.manual_seed(0)
    return torch


def _state(module) -> dict:
    return dict(module.state_dict())


def _assert_close(actual: jnp.ndarray, expected, name: str) -> None:
    np.testing.assert_allclose(
        np.asarray(actual, dtype=np.float64),
        np.asarray(expected.detach().numpy(), dtype=np.float64),
        rtol=RTOL,
        atol=ATOL,
        err_msg=f"{name} diverged from the OpenFold3 reference",
    )


def test_linear_matches_torch(openfold3_source: Path, randomized) -> None:
    torch = _torch()
    from openfold3.core.model.primitives import Linear

    for bias in (True, False):
        module = randomized(Linear(6, 4, bias=bias))
        x = torch.randn(3, 5, 6)
        with torch.no_grad():
            expected = module(x)
        params = map_linear(_state(module), bias=bias)
        actual = linear(jnp.asarray(x.numpy()), params)
        _assert_close(actual, expected, f"Linear(bias={bias})")


def test_layer_norm_matches_torch_including_optional_scale_and_offset(
    openfold3_source: Path, randomized
) -> None:
    torch = _torch()
    from openfold3.core.model.primitives import LayerNorm

    # AdaLN relies on both flags, so both are covered here.
    for create_scale, create_offset in ((True, True), (False, False), (True, False)):
        module = LayerNorm(
            8, create_scale=create_scale, create_offset=create_offset
        )
        module = randomized(module)
        x = torch.randn(2, 7, 8)
        with torch.no_grad():
            expected = module(x)
        params = map_layer_norm(_state(module))
        assert (params.weight is not None) == create_scale
        assert (params.bias is not None) == create_offset
        actual = layer_norm(jnp.asarray(x.numpy()), params)
        _assert_close(actual, expected, f"LayerNorm({create_scale},{create_offset})")


def test_swiglu_matches_torch(openfold3_source: Path, randomized) -> None:
    torch = _torch()
    from openfold3.core.model.primitives import SwiGLU

    module = randomized(SwiGLU(8, 16))
    x = torch.randn(2, 5, 8)
    with torch.no_grad():
        expected = module(x)
    actual = swiglu(jnp.asarray(x.numpy()), map_swiglu(_state(module)))
    _assert_close(actual, expected, "SwiGLU")


def test_adaln_matches_torch(openfold3_source: Path, randomized) -> None:
    torch = _torch()
    from openfold3.core.model.primitives import AdaLN

    module = randomized(AdaLN(c_a=8, c_s=6))
    a = torch.randn(2, 5, 8)
    s = torch.randn(2, 5, 6)
    with torch.no_grad():
        expected = module(a, s)
    params = map_adaln(_state(module))
    actual = adaln(jnp.asarray(a.numpy()), jnp.asarray(s.numpy()), params)
    _assert_close(actual, expected, "AdaLN")


def test_swiglu_transition_matches_torch(
    openfold3_source: Path, randomized
) -> None:
    torch = _torch()
    from openfold3.core.model.layers.transition import SwiGLUTransition

    module = randomized(SwiGLUTransition(c_in=8, n=4))
    x = torch.randn(2, 5, 8)
    with torch.no_grad():
        expected = module(x)
    params = map_swiglu_transition(_state(module))
    actual = swiglu_transition(jnp.asarray(x.numpy()), params)
    _assert_close(actual, expected, "SwiGLUTransition")


def test_swiglu_transition_honours_the_mask(
    openfold3_source: Path, randomized
) -> None:
    torch = _torch()
    from openfold3.core.model.layers.transition import SwiGLUTransition

    module = randomized(SwiGLUTransition(c_in=8, n=4))
    x = torch.randn(2, 5, 8)
    mask = torch.zeros(2, 5)
    mask[:, :3] = 1.0
    with torch.no_grad():
        expected = module(x, mask=mask)
    params = map_swiglu_transition(_state(module))
    actual = swiglu_transition(
        jnp.asarray(x.numpy()), params, mask=jnp.asarray(mask.numpy())
    )
    _assert_close(actual, expected, "SwiGLUTransition(masked)")
    # A zero mask row must zero the update, not merely shrink it.
    assert np.allclose(np.asarray(actual)[:, 3:], 0.0)


def test_mapper_rejects_a_missing_key(openfold3_source: Path) -> None:
    _torch()
    from openfold3.core.model.layers.transition import SwiGLUTransition

    state = _state(SwiGLUTransition(c_in=8, n=4))
    del state["swiglu.linear_a.weight"]
    with pytest.raises(KeyError, match="swiglu.linear_a.weight"):
        map_swiglu_transition(state)


def test_state_dict_layout_is_what_the_mappers_assume(openfold3_source: Path) -> None:
    """Pin the upstream key names the mappers depend on."""
    _torch()
    from openfold3.core.model.layers.transition import SwiGLUTransition
    from openfold3.core.model.primitives import AdaLN

    assert set(_state(SwiGLUTransition(c_in=8, n=4))) == {
        "layer_norm.weight",
        "layer_norm.bias",
        "swiglu.linear_a.weight",
        "swiglu.linear_b.weight",
        "linear_out.weight",
    }
    assert set(_state(AdaLN(c_a=8, c_s=6))) == {
        "layer_norm_s.weight",
        "linear_g.weight",
        "linear_g.bias",
        "linear_s.weight",
    }
