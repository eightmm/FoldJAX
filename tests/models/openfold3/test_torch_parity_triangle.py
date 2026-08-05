"""Torch-vs-JAX parity for the triangular multiplicative update.

Outgoing and incoming share a parameter layout and differ only in which spatial
axis is contracted, so a swapped contraction produces correct shapes and wrong
numbers. Both directions are gated separately, and a cross-check asserts they
actually disagree.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.openfold3.bridge.torch_mapping import map_triangle_multiplication
from foldjax.models.openfold3.models.triangle import (
    permute_final_dims,
    triangle_multiplication,
)

pytestmark = pytest.mark.torch_parity

RTOL = 1e-4
ATOL = 1e-4

C_Z, C_HIDDEN, N = 8, 6, 5


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


def _module(outgoing: bool):
    from openfold3.core.model.layers.triangular_multiplicative_update import (
        TriangleMultiplicationIncoming,
        TriangleMultiplicationOutgoing,
    )

    cls = TriangleMultiplicationOutgoing if outgoing else TriangleMultiplicationIncoming
    return cls(c_z=C_Z, c_hidden=C_HIDDEN)


@pytest.mark.parametrize("outgoing", [True, False])
def test_triangle_multiplication_matches_torch(
    openfold3_source: Path, randomized, outgoing: bool
) -> None:
    torch = _torch()
    module = randomized(_module(outgoing))
    z = torch.randn(2, N, N, C_Z)
    with torch.no_grad():
        expected = module(z)
    actual = triangle_multiplication(
        jnp.asarray(z.numpy()),
        map_triangle_multiplication(dict(module.state_dict())),
        outgoing=outgoing,
    )
    _assert_close(actual, expected, f"TriangleMultiplication(outgoing={outgoing})")


@pytest.mark.parametrize("outgoing", [True, False])
def test_triangle_multiplication_matches_torch_with_mask(
    openfold3_source: Path, randomized, outgoing: bool
) -> None:
    torch = _torch()
    module = randomized(_module(outgoing))
    z = torch.randn(2, N, N, C_Z)
    mask = torch.ones(2, N, N)
    mask[:, 3:, :] = 0.0
    mask[:, :, 4:] = 0.0
    with torch.no_grad():
        expected = module(z, mask=mask)
    actual = triangle_multiplication(
        jnp.asarray(z.numpy()),
        map_triangle_multiplication(dict(module.state_dict())),
        outgoing=outgoing,
        mask=jnp.asarray(mask.numpy()),
    )
    _assert_close(actual, expected, f"TriangleMultiplication(masked,{outgoing})")


def test_outgoing_and_incoming_actually_differ(
    openfold3_source: Path, randomized
) -> None:
    """Guards against a contraction bug that would pass one direction's test."""
    torch = _torch()
    module = randomized(_module(outgoing=True))
    params = map_triangle_multiplication(dict(module.state_dict()))
    z = jnp.asarray(torch.randn(1, N, N, C_Z).numpy())
    out = triangle_multiplication(z, params, outgoing=True)
    inc = triangle_multiplication(z, params, outgoing=False)
    assert not np.allclose(np.asarray(out), np.asarray(inc), rtol=1e-3, atol=1e-3)


def test_permute_final_dims_matches_torch(openfold3_source: Path) -> None:
    torch = _torch()
    from openfold3.core.utils.tensor_utils import permute_final_dims as torch_permute

    x = torch.randn(2, 3, 4, 5)
    for inds in ((2, 0, 1), (2, 1, 0), (1, 2, 0)):
        expected = torch_permute(x, list(inds))
        actual = permute_final_dims(jnp.asarray(x.numpy()), inds)
        assert actual.shape == tuple(expected.shape), inds
        _assert_close(actual, expected, f"permute_final_dims{inds}")


def test_mapper_reports_a_missing_projection(openfold3_source: Path) -> None:
    _torch()
    state = dict(_module(outgoing=True).state_dict())
    del state["linear_b_g.weight"]
    with pytest.raises(KeyError, match="linear_b_g.weight"):
        map_triangle_multiplication(state)


def test_triangle_state_dict_layout(openfold3_source: Path) -> None:
    _torch()
    assert set(_module(outgoing=True).state_dict()) == {
        "layer_norm_in.weight",
        "layer_norm_in.bias",
        "layer_norm_out.weight",
        "layer_norm_out.bias",
        "linear_a_p.weight",
        "linear_a_g.weight",
        "linear_b_p.weight",
        "linear_b_g.weight",
        "linear_g.weight",
        "linear_z.weight",
    }
