"""Torch-vs-JAX parity for reference atom feature embedding (AF3 Alg. 5)."""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.openfold3.bridge.torch_mapping import map_ref_atom_feature_embedder
from foldjax.models.openfold3.models.atom_features import ref_atom_feature_embedder

pytestmark = pytest.mark.torch_parity

RTOL = 1e-4
ATOL = 1e-4

C_ATOM, C_ATOM_PAIR = 8, 6
C_ELEMENT, C_CHARS = 12, 5
N_ATOM, N_QUERY, N_KEY = 12, 4, 8


def _torch():
    import torch

    torch.manual_seed(0)
    return torch


def _module(torch):
    from ml_collections import ConfigDict
    from openfold3.core.model.layers.sequence_local_atom_attention import (
        RefAtomFeatureEmbedder,
    )

    return RefAtomFeatureEmbedder(
        c_atom_ref=ConfigDict({"element": C_ELEMENT, "name_chars": 4 * C_CHARS}),
        c_atom=C_ATOM,
        c_atom_pair=C_ATOM_PAIR,
    )


def _batch(torch, *, n_valid: int = N_ATOM, spaces: int = 2) -> dict:
    atom_mask = torch.zeros(1, N_ATOM)
    atom_mask[:, :n_valid] = 1.0
    # Split the atoms across `spaces` reference conformers so the vlm term
    # actually zeroes some cross-conformer pairs.
    uid = torch.arange(N_ATOM).float() // max(N_ATOM // spaces, 1)
    return {
        "ref_pos": torch.randn(1, N_ATOM, 3),
        "ref_charge": torch.randn(1, N_ATOM) * 2.0,
        "ref_mask": atom_mask.clone(),
        "ref_element": torch.rand(1, N_ATOM, C_ELEMENT),
        "ref_atom_name_chars": torch.rand(1, N_ATOM, 4, C_CHARS),
        "ref_space_uid": uid.reshape(1, N_ATOM),
        "atom_mask": atom_mask,
    }


def _as_jax(batch: dict) -> dict:
    return {key: jnp.asarray(value.numpy()) for key, value in batch.items()}


@pytest.mark.parametrize(("n_valid", "spaces"), [(N_ATOM, 2), (7, 3), (12, 1)])
def test_ref_atom_feature_embedder_matches_torch(
    openfold3_source: Path, randomized, n_valid: int, spaces: int
) -> None:
    torch = _torch()
    module = randomized(_module(torch))
    batch = _batch(torch, n_valid=n_valid, spaces=spaces)
    with torch.no_grad():
        expected_cl, expected_plm = module(batch, n_query=N_QUERY, n_key=N_KEY)
    actual_cl, actual_plm = ref_atom_feature_embedder(
        _as_jax(batch),
        map_ref_atom_feature_embedder(dict(module.state_dict())),
        n_query=N_QUERY,
        n_key=N_KEY,
    )
    for actual, expected, name in (
        (actual_cl, expected_cl, "cl"),
        (actual_plm, expected_plm, "plm"),
    ):
        assert actual.shape == tuple(expected.shape), name
        np.testing.assert_allclose(
            np.asarray(actual, dtype=np.float64),
            expected.detach().numpy().astype(np.float64),
            rtol=RTOL,
            atol=ATOL,
            err_msg=f"{name} diverged from the OpenFold3 reference",
        )


def test_charge_goes_through_arcsinh(openfold3_source: Path, randomized) -> None:
    """A raw-charge port would agree at small values and drift at large ones."""
    torch = _torch()
    module = randomized(_module(torch))
    batch = _batch(torch)
    batch["ref_charge"] = torch.full((1, N_ATOM), 50.0)
    with torch.no_grad():
        expected_cl, _ = module(batch, n_query=N_QUERY, n_key=N_KEY)
    actual_cl, _ = ref_atom_feature_embedder(
        _as_jax(batch),
        map_ref_atom_feature_embedder(dict(module.state_dict())),
        n_query=N_QUERY,
        n_key=N_KEY,
    )
    np.testing.assert_allclose(
        np.asarray(actual_cl, dtype=np.float64),
        expected_cl.detach().numpy().astype(np.float64),
        rtol=RTOL,
        atol=ATOL,
    )


def test_cross_conformer_pairs_are_zeroed(openfold3_source: Path, randomized) -> None:
    """Pair features only survive between atoms sharing a ref_space_uid."""
    torch = _torch()
    module = randomized(_module(torch))
    batch = _batch(torch, spaces=N_ATOM)  # every atom its own conformer
    _cl, plm = ref_atom_feature_embedder(
        _as_jax(batch),
        map_ref_atom_feature_embedder(dict(module.state_dict())),
        n_query=N_QUERY,
        n_key=N_KEY,
    )
    # With no two atoms sharing a uid, only self-pairs can be non-zero.
    array = np.asarray(plm)
    assert np.isfinite(array).all()
    nonzero_fraction = float((np.abs(array).sum(-1) > 1e-6).mean())
    assert nonzero_fraction < 0.2, nonzero_fraction


def test_state_dict_layout(openfold3_source: Path) -> None:
    torch = _torch()
    assert set(_module(torch).state_dict()) == {
        "linear_ref_pos.weight",
        "linear_ref_charge.weight",
        "linear_ref_mask.weight",
        "linear_ref_element.weight",
        "linear_ref_atom_chars.weight",
        "linear_ref_offset.weight",
        "linear_inv_sq_dists.weight",
        "linear_valid_mask.weight",
    }


def test_mapper_reports_a_missing_projection(openfold3_source: Path) -> None:
    torch = _torch()
    state = dict(_module(torch).state_dict())
    del state["linear_inv_sq_dists.weight"]
    with pytest.raises(KeyError, match="linear_inv_sq_dists.weight"):
        map_ref_atom_feature_embedder(state)
