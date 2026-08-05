"""Torch-vs-JAX parity for the per-atom logit heads and mask compaction."""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.openfold3.bridge.torch_mapping import map_atom_logit_head
from foldjax.models.openfold3.models.atomize import max_atom_per_token_masked_select
from foldjax.models.openfold3.models.heads import atom_logit_head

pytestmark = pytest.mark.torch_parity

RTOL = 1e-4
ATOL = 1e-4

C_S, C_OUT, N_TOKEN, MAX_ATOMS = 8, 5, 4, 3


def _torch():
    import torch

    torch.manual_seed(0)
    return torch


def _upstream_select():
    from openfold3.core.utils.atomize_utils import max_atom_per_token_masked_select

    return max_atom_per_token_masked_select


def _heads():
    from openfold3.core.model.heads.prediction_heads import (
        ExperimentallyResolvedHeadAllAtom,
        PerResidueLDDTAllAtom,
    )

    kwargs = {
        "c_s": C_S,
        "c_out": C_OUT,
        "max_atoms_per_token": MAX_ATOMS,
    }
    return {
        "plddt": PerResidueLDDTAllAtom(**kwargs),
        "resolved": ExperimentallyResolvedHeadAllAtom(**kwargs),
    }


def _mask(torch, counts: list[int]):
    """Mark the first `counts[i]` of each token's MAX_ATOMS slots as valid."""
    mask = torch.zeros(1, N_TOKEN, MAX_ATOMS)
    for index, count in enumerate(counts):
        mask[:, index, :count] = 1.0
    return mask.reshape(1, N_TOKEN * MAX_ATOMS)


@pytest.mark.parametrize(
    "counts", [[3, 3, 3, 3], [3, 1, 2, 3], [1, 1, 1, 1], [3, 0, 3, 1]]
)
def test_masked_select_matches_torch(
    openfold3_source: Path, counts: list[int]
) -> None:
    torch = _torch()
    feat = torch.randn(1, N_TOKEN * MAX_ATOMS, C_OUT)
    mask = _mask(torch, counts)
    expected = _upstream_select()(atom_feat=feat, max_atom_per_token_mask=mask)
    actual = max_atom_per_token_masked_select(
        jnp.asarray(feat.numpy()),
        jnp.asarray(mask.numpy()),
        n_atom=expected.shape[-2],
    )
    assert actual.shape == tuple(expected.shape)
    np.testing.assert_allclose(
        np.asarray(actual, dtype=np.float64),
        expected.detach().numpy().astype(np.float64),
        rtol=1e-6,
        atol=1e-6,
    )


def test_masked_select_preserves_order(openfold3_source: Path) -> None:
    """Compaction must keep the original slot order, not just the right values."""
    feat = jnp.arange(6, dtype=jnp.float32).reshape(1, 6, 1)
    mask = jnp.asarray([[1.0, 0.0, 1.0, 1.0, 0.0, 0.0]])
    actual = max_atom_per_token_masked_select(feat, mask, n_atom=3)
    np.testing.assert_allclose(np.asarray(actual).reshape(-1), [0.0, 2.0, 3.0])


def test_masked_select_pads_past_the_valid_count(openfold3_source: Path) -> None:
    feat = jnp.arange(4, dtype=jnp.float32).reshape(1, 4, 1) + 1.0
    mask = jnp.asarray([[1.0, 1.0, 0.0, 0.0]])
    actual = max_atom_per_token_masked_select(feat, mask, n_atom=4)
    np.testing.assert_allclose(
        np.asarray(actual).reshape(-1), [1.0, 2.0, 0.0, 0.0]
    )


@pytest.mark.parametrize("name", ["plddt", "resolved"])
def test_atom_head_matches_torch(
    openfold3_source: Path, randomized, name: str
) -> None:
    torch = _torch()
    module = randomized(_heads()[name])
    s = torch.randn(1, N_TOKEN, C_S)
    mask = _mask(torch, [3, 1, 2, 3])
    with torch.no_grad():
        expected = module(s=s, max_atom_per_token_mask=mask)
    actual = atom_logit_head(
        jnp.asarray(s.numpy()),
        map_atom_logit_head(dict(module.state_dict())),
        jnp.asarray(mask.numpy()),
        max_atoms_per_token=MAX_ATOMS,
        c_out=C_OUT,
        n_atom=expected.shape[-2],
    )
    assert actual.shape == tuple(expected.shape)
    np.testing.assert_allclose(
        np.asarray(actual, dtype=np.float64),
        expected.detach().numpy().astype(np.float64),
        rtol=RTOL,
        atol=ATOL,
        err_msg=f"{name} head diverged from the OpenFold3 reference",
    )


def test_atom_head_state_dict_layout(openfold3_source: Path) -> None:
    _torch()
    for name, module in _heads().items():
        assert set(module.state_dict()) == {
            "layer_norm.weight",
            "layer_norm.bias",
            "linear.weight",
        }, name
        # The projection fans out to max_atoms_per_token * c_out channels.
        assert module.state_dict()["linear.weight"].shape == (
            MAX_ATOMS * C_OUT,
            C_S,
        ), name


def test_mapper_reports_a_missing_projection(openfold3_source: Path) -> None:
    _torch()
    state = dict(_heads()["plddt"].state_dict())
    del state["linear.weight"]
    with pytest.raises(KeyError, match="linear.weight"):
        map_atom_logit_head(state)
