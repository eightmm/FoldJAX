"""Fused triangular multiplication maps onto the unfused forward pass.

Whether the released OpenFold3 weights store this layer fused or unfused was the
port's largest open risk. It turns out not to need a second forward path: the
fused variant is one 2*c_hidden projection split in half, which is the same
function as two c_hidden projections. This gates that claim against upstream's
own fused module.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.openfold3.bridge.torch_mapping import (
    map_fused_triangle_multiplication,
)
from foldjax.models.openfold3.models.triangle import triangle_multiplication

pytestmark = pytest.mark.torch_parity

RTOL = 1e-4
ATOL = 1e-4

C_Z, C_HIDDEN, N = 8, 6, 5


def _torch():
    import torch

    torch.manual_seed(0)
    return torch


def _module(outgoing: bool):
    from openfold3.core.model.layers.triangular_multiplicative_update import (
        FusedTriangleMultiplicationIncoming,
        FusedTriangleMultiplicationOutgoing,
    )

    cls = (
        FusedTriangleMultiplicationOutgoing
        if outgoing
        else FusedTriangleMultiplicationIncoming
    )
    return cls(c_z=C_Z, c_hidden=C_HIDDEN)


@pytest.mark.parametrize("outgoing", [True, False])
def test_fused_checkpoint_matches_the_unfused_forward(
    openfold3_source: Path, randomized, outgoing: bool
) -> None:
    torch = _torch()
    module = randomized(_module(outgoing))
    z = torch.randn(1, N, N, C_Z)
    mask = torch.ones(1, N, N)
    mask[:, 4:, :] = 0.0

    with torch.no_grad():
        expected = module(z, mask=mask)

    params = map_fused_triangle_multiplication(dict(module.state_dict()))
    actual = triangle_multiplication(
        jnp.asarray(z.numpy()),
        params,
        outgoing=outgoing,
        mask=jnp.asarray(mask.numpy()),
    )
    np.testing.assert_allclose(
        np.asarray(actual, dtype=np.float64),
        expected.detach().numpy().astype(np.float64),
        rtol=RTOL,
        atol=ATOL,
        err_msg=f"fused->unfused mapping diverged (outgoing={outgoing})",
    )


def test_fused_state_dict_layout(openfold3_source: Path) -> None:
    """The fused layout stores one ab projection, not separate a and b."""
    _torch()
    keys = set(_module(outgoing=True).state_dict())
    assert "linear_ab_p.weight" in keys
    assert "linear_ab_g.weight" in keys
    assert "linear_a_p.weight" not in keys
    assert "linear_b_p.weight" not in keys


def test_split_halves_are_the_a_then_b_order(
    openfold3_source: Path, randomized
) -> None:
    """First half is a, second is b; swapping them would change the result."""
    _torch()
    module = randomized(_module(outgoing=True))
    state = dict(module.state_dict())
    params = map_fused_triangle_multiplication(state)
    ab_p = state["linear_ab_p.weight"].numpy()
    np.testing.assert_allclose(np.asarray(params.linear_a_p.weight), ab_p[:C_HIDDEN])
    np.testing.assert_allclose(np.asarray(params.linear_b_p.weight), ab_p[C_HIDDEN:])


def test_rejects_an_odd_fused_width(openfold3_source: Path) -> None:
    state = {
        "linear_ab_p.weight": np.zeros((5, C_Z), dtype=np.float32),
        "linear_ab_g.weight": np.zeros((5, C_Z), dtype=np.float32),
        "linear_g.weight": np.zeros((C_Z, C_Z), dtype=np.float32),
        "linear_z.weight": np.zeros((C_Z, C_HIDDEN), dtype=np.float32),
    }
    with pytest.raises(ValueError, match="must be even"):
        map_fused_triangle_multiplication(state)
