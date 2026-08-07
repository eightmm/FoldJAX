"""Torch-vs-JAX parity for the MSA leaves."""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.openfold3.bridge.torch_mapping import (
    map_msa_pair_weighted_averaging,
    map_outer_product_mean,
)
from foldjax.models.openfold3.models.msa import (
    msa_pair_weighted_averaging,
    outer_product_mean,
)

pytestmark = pytest.mark.torch_parity

RTOL = 1e-4
ATOL = 1e-4

C_M, C_Z, C_HIDDEN, HEADS = 8, 6, 4, 2
N_SEQ, N_TOKEN = 3, 5


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


def _opm():
    from openfold3.core.model.layers.outer_product_mean import OuterProductMean

    return OuterProductMean(c_m=C_M, c_z=C_Z, c_hidden=C_HIDDEN)


def _pwa():
    from openfold3.core.model.layers.msa import MSAPairWeightedAveraging

    return MSAPairWeightedAveraging(
        c_in=C_M, c_hidden=C_HIDDEN, c_z=C_Z, no_heads=HEADS
    )


def test_outer_product_mean_matches_torch(openfold3_source: Path, randomized) -> None:
    torch = _torch()
    module = randomized(_opm())
    m = torch.randn(1, N_SEQ, N_TOKEN, C_M)
    with torch.no_grad():
        expected = module(m)
    actual = outer_product_mean(
        jnp.asarray(m.numpy()), map_outer_product_mean(dict(module.state_dict()))
    )
    assert actual.shape == (1, N_TOKEN, N_TOKEN, C_Z)
    _assert_close(actual, expected, "OuterProductMean")


def test_outer_product_mean_matches_torch_with_mask(
    openfold3_source: Path, randomized
) -> None:
    torch = _torch()
    module = randomized(_opm())
    m = torch.randn(1, N_SEQ, N_TOKEN, C_M)
    mask = torch.ones(1, N_SEQ, N_TOKEN)
    # Drop a whole MSA row and a trailing token column.
    mask[:, 2, :] = 0.0
    mask[:, :, 4:] = 0.0
    with torch.no_grad():
        expected = module(m, mask=mask)
    actual = outer_product_mean(
        jnp.asarray(m.numpy()),
        map_outer_product_mean(dict(module.state_dict())),
        mask=jnp.asarray(mask.numpy()),
    )
    _assert_close(actual, expected, "OuterProductMean(masked)")


def test_outer_product_mean_normalization_epsilon_matters(
    openfold3_source: Path, randomized
) -> None:
    """A fully masked token pair divides by eps alone; it must stay finite."""
    torch = _torch()
    module = randomized(_opm())
    m = torch.randn(1, N_SEQ, N_TOKEN, C_M)
    mask = torch.zeros(1, N_SEQ, N_TOKEN)
    mask[:, :, 0] = 1.0
    with torch.no_grad():
        expected = module(m, mask=mask)
    actual = outer_product_mean(
        jnp.asarray(m.numpy()),
        map_outer_product_mean(dict(module.state_dict())),
        mask=jnp.asarray(mask.numpy()),
    )
    assert np.isfinite(np.asarray(actual)).all()
    _assert_close(actual, expected, "OuterProductMean(mostly masked)")


def test_outer_product_mean_chunking_matches_torch_chunking(
    openfold3_source: Path, randomized
) -> None:
    """Both sides take ``chunk_size`` to mean the same axis.

    ``test_outer_product_mean_chunking`` already proves our chunked result equals
    our unchunked one, which is the correctness argument. This proves the argument
    means what upstream means by it -- a rank-1 chunk over the first token axis of
    the transposed ``a``, not over the MSA rows the sum runs across.
    """
    torch = _torch()
    module = randomized(_opm())
    m = torch.randn(1, N_SEQ, N_TOKEN, C_M)
    with torch.no_grad():
        expected = module(m, chunk_size=2)
    actual = outer_product_mean(
        jnp.asarray(m.numpy()),
        map_outer_product_mean(dict(module.state_dict())),
        chunk_size=2,
    )
    _assert_close(actual, expected, "OuterProductMean(chunked)")


def test_msa_pair_weighted_averaging_matches_torch(
    openfold3_source: Path, randomized
) -> None:
    torch = _torch()
    module = randomized(_pwa())
    m = torch.randn(1, N_SEQ, N_TOKEN, C_M)
    z = torch.randn(1, N_TOKEN, N_TOKEN, C_Z)
    with torch.no_grad():
        expected = module(m=m, z=z)
    actual = msa_pair_weighted_averaging(
        jnp.asarray(m.numpy()),
        jnp.asarray(z.numpy()),
        map_msa_pair_weighted_averaging(dict(module.state_dict())),
        no_heads=HEADS,
    )
    assert actual.shape == (1, N_SEQ, N_TOKEN, C_M)
    _assert_close(actual, expected, "MSAPairWeightedAveraging")


def test_msa_pair_weighted_averaging_matches_torch_with_mask(
    openfold3_source: Path, randomized
) -> None:
    torch = _torch()
    module = randomized(_pwa())
    m = torch.randn(1, N_SEQ, N_TOKEN, C_M)
    z = torch.randn(1, N_TOKEN, N_TOKEN, C_Z)
    # This layer takes the PAIR mask, not the MSA mask.
    token = torch.ones(1, N_TOKEN)
    token[:, 4:] = 0.0
    mask = token[..., None] * token[..., None, :]
    with torch.no_grad():
        expected = module(m=m, z=z, mask=mask)
    actual = msa_pair_weighted_averaging(
        jnp.asarray(m.numpy()),
        jnp.asarray(z.numpy()),
        map_msa_pair_weighted_averaging(dict(module.state_dict())),
        no_heads=HEADS,
        mask=jnp.asarray(mask.numpy()),
    )
    _assert_close(actual, expected, "MSAPairWeightedAveraging(masked)")


def test_msa_state_dict_layouts(openfold3_source: Path) -> None:
    _torch()
    assert set(_opm().state_dict()) == {
        "layer_norm.weight",
        "layer_norm.bias",
        "linear_1.weight",
        "linear_2.weight",
        "linear_out.weight",
        "linear_out.bias",
    }
    assert set(_pwa().state_dict()) == {
        "layer_norm_m.weight",
        "layer_norm_m.bias",
        "layer_norm_z.weight",
        "layer_norm_z.bias",
        "linear_z.weight",
        "linear_v.weight",
        "linear_g.weight",
        "linear_o.weight",
    }


def test_msa_mappers_report_missing_keys(openfold3_source: Path) -> None:
    _torch()
    state = dict(_opm().state_dict())
    del state["linear_out.bias"]
    with pytest.raises(KeyError, match="linear_out.bias"):
        map_outer_product_mean(state)

    state = dict(_pwa().state_dict())
    del state["linear_v.weight"]
    with pytest.raises(KeyError, match="linear_v.weight"):
        map_msa_pair_weighted_averaging(state)
