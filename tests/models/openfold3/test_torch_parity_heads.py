"""Torch-vs-JAX parity for the pair-level prediction heads."""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.openfold3.bridge.torch_mapping import map_pair_head
from foldjax.models.openfold3.models.heads import (
    distogram_head,
    predicted_aligned_error_head,
    predicted_distance_error_head,
)

pytestmark = pytest.mark.torch_parity

RTOL = 1e-4
ATOL = 1e-4

C_Z, C_OUT, N = 8, 6, 5


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


def _heads():
    from openfold3.core.model.heads.prediction_heads import (
        DistogramHead,
        PredictedAlignedErrorHead,
        PredictedDistanceErrorHead,
    )

    return {
        "pae": PredictedAlignedErrorHead(c_z=C_Z, c_out=C_OUT),
        "pde": PredictedDistanceErrorHead(c_z=C_Z, c_out=C_OUT),
        "distogram": DistogramHead(c_z=C_Z, c_out=C_OUT),
    }


def test_pae_head_matches_torch(openfold3_source: Path, randomized) -> None:
    torch = _torch()
    module = randomized(_heads()["pae"])
    z = torch.randn(1, N, N, C_Z)
    with torch.no_grad():
        expected = module(z)
    actual = predicted_aligned_error_head(
        jnp.asarray(z.numpy()), map_pair_head(dict(module.state_dict()))
    )
    _assert_close(actual, expected, "PredictedAlignedErrorHead")


def test_pde_head_matches_torch(openfold3_source: Path, randomized) -> None:
    torch = _torch()
    module = randomized(_heads()["pde"])
    z = torch.randn(1, N, N, C_Z)
    with torch.no_grad():
        expected = module(z)
    actual = predicted_distance_error_head(
        jnp.asarray(z.numpy()), map_pair_head(dict(module.state_dict()))
    )
    _assert_close(actual, expected, "PredictedDistanceErrorHead")


def test_distogram_head_matches_torch(openfold3_source: Path, randomized) -> None:
    torch = _torch()
    module = randomized(_heads()["distogram"])
    z = torch.randn(1, N, N, C_Z)
    with torch.no_grad():
        expected = module(z)
    actual = distogram_head(
        jnp.asarray(z.numpy()),
        map_pair_head(dict(module.state_dict()), layer_norm=False),
    )
    _assert_close(actual, expected, "DistogramHead")


def test_pae_is_asymmetric_but_pde_and_distogram_are_not(
    openfold3_source: Path, randomized
) -> None:
    """The three heads differ exactly here; shapes alone cannot tell them apart."""
    torch = _torch()
    heads = _heads()
    z = jnp.asarray(torch.randn(1, N, N, C_Z).numpy())

    pae = predicted_aligned_error_head(
        z, map_pair_head(dict(randomized(heads["pae"]).state_dict()))
    )
    pde = predicted_distance_error_head(
        z, map_pair_head(dict(randomized(heads["pde"]).state_dict()))
    )
    disto = distogram_head(
        z, map_pair_head(dict(randomized(heads["distogram"]).state_dict()),
                         layer_norm=False)
    )

    def symmetric(x: jnp.ndarray) -> bool:
        return bool(
            np.allclose(np.asarray(x), np.asarray(jnp.swapaxes(x, -2, -3)), atol=1e-5)
        )

    assert not symmetric(pae), "PAE must stay directional"
    assert symmetric(pde), "PDE must be symmetrized"
    assert symmetric(disto), "the distogram must be symmetrized"


def test_distogram_head_has_no_layer_norm(openfold3_source: Path) -> None:
    _torch()
    state = set(_heads()["distogram"].state_dict())
    assert state == {"linear.weight"}
    # Asking for a norm that does not exist must fail loudly.
    with pytest.raises(ValueError, match="no layer norm"):
        distogram_head(
            jnp.zeros((1, N, N, C_Z)),
            map_pair_head(dict(_heads()["pae"].state_dict())),
        )


def test_error_head_state_dict_layouts(openfold3_source: Path) -> None:
    _torch()
    heads = _heads()
    for name in ("pae", "pde"):
        assert set(heads[name].state_dict()) == {
            "layer_norm.weight",
            "layer_norm.bias",
            "linear.weight",
        }, name


def test_mapper_reports_a_missing_projection(openfold3_source: Path) -> None:
    _torch()
    state = dict(_heads()["pae"].state_dict())
    del state["linear.weight"]
    with pytest.raises(KeyError, match="linear.weight"):
        map_pair_head(state)
