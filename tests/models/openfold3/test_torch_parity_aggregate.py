"""Torch-vs-JAX parity for atom-to-token aggregation."""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.openfold3.models.atomize import aggregate_atom_feat_to_tokens

pytestmark = pytest.mark.torch_parity

C_FEAT = 5


def _torch():
    import torch

    torch.manual_seed(0)
    return torch


def _upstream():
    from openfold3.core.utils.atomize_utils import aggregate_atom_feat_to_tokens

    return aggregate_atom_feat_to_tokens


def _scene(torch, counts: list[int], n_masked: int = 0):
    n_atom = sum(counts)
    atom_to_token = torch.cat(
        [torch.full((count,), index) for index, count in enumerate(counts)]
    ).float()
    atom_mask = torch.ones(1, n_atom)
    if n_masked:
        atom_mask[:, -n_masked:] = 0.0
    return (
        torch.ones(1, len(counts)),
        atom_to_token.reshape(1, n_atom),
        atom_mask,
        torch.randn(1, n_atom, C_FEAT),
    )


@pytest.mark.parametrize("aggregate", ["mean", "sum"])
@pytest.mark.parametrize(
    ("counts", "n_masked"), [([3, 2, 4], 0), ([3, 2, 4], 3), ([1, 1, 1], 0)]
)
def test_aggregate_matches_torch(
    openfold3_source: Path, aggregate: str, counts: list[int], n_masked: int
) -> None:
    torch = _torch()
    token_mask, atom_to_token, atom_mask, atom_feat = _scene(
        torch, counts, n_masked
    )
    expected = _upstream()(
        token_mask=token_mask,
        atom_to_token_index=atom_to_token,
        atom_mask=atom_mask,
        atom_feat=atom_feat,
        atom_dim=-2,
        aggregate_fn=aggregate,
    )
    actual = aggregate_atom_feat_to_tokens(
        jnp.asarray(atom_feat.numpy()),
        jnp.asarray(atom_to_token.numpy()),
        jnp.asarray(atom_mask.numpy()),
        n_token=len(counts),
        aggregate=aggregate,
    )
    assert actual.shape == tuple(expected.shape)
    np.testing.assert_allclose(
        np.asarray(actual, dtype=np.float64),
        expected.detach().numpy().astype(np.float64),
        rtol=1e-5,
        atol=1e-5,
    )


def test_masked_atoms_do_not_reach_any_token(openfold3_source: Path) -> None:
    """Masked atoms go to a discarded overflow bin, not to token 0."""
    feat = jnp.asarray([[[1.0], [1.0], [100.0]]])
    atom_to_token = jnp.asarray([[0.0, 1.0, 0.0]])
    atom_mask = jnp.asarray([[1.0, 1.0, 0.0]])
    actual = aggregate_atom_feat_to_tokens(
        feat, atom_to_token, atom_mask, n_token=2, aggregate="sum"
    )
    np.testing.assert_allclose(np.asarray(actual).reshape(-1), [1.0, 1.0])


def test_empty_token_is_zero_not_nan(openfold3_source: Path) -> None:
    """A token with no atoms divides by eps alone and must stay finite."""
    feat = jnp.ones((1, 2, 1))
    atom_to_token = jnp.asarray([[0.0, 0.0]])
    atom_mask = jnp.ones((1, 2))
    actual = aggregate_atom_feat_to_tokens(
        feat, atom_to_token, atom_mask, n_token=3, aggregate="mean"
    )
    array = np.asarray(actual)
    assert np.isfinite(array).all()
    np.testing.assert_allclose(array.reshape(-1), [1.0, 0.0, 0.0])


def test_rejects_an_unknown_aggregation(openfold3_source: Path) -> None:
    with pytest.raises(ValueError, match="invalid aggregation"):
        aggregate_atom_feat_to_tokens(
            jnp.ones((1, 2, 1)),
            jnp.zeros((1, 2)),
            jnp.ones((1, 2)),
            n_token=1,
            aggregate="max",
        )
