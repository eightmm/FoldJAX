"""Torch-vs-JAX parity for token-to-atom broadcasting.

Upstream uses a data-dependent ``repeat_interleave``; this port uses a static
cumsum + gather. These tests assert the two agree exactly, which is the whole
justification for the different formulation.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.openfold3.models.atomize import broadcast_token_feat_to_atoms

pytestmark = pytest.mark.torch_parity


def _torch():
    import torch

    torch.manual_seed(0)
    return torch


def _upstream():
    from openfold3.core.utils.atomize_utils import broadcast_token_feat_to_atoms

    return broadcast_token_feat_to_atoms


@pytest.mark.parametrize(
    "counts",
    [
        [1, 1, 1, 1],
        [4, 2, 1, 3],
        # A masked-out trailing token contributes no atoms.
        [3, 5, 2, 0],
        # Single token holding every atom.
        [7],
        # Wide variation, including a 1-atom token between large ones.
        [9, 1, 6, 2, 4],
    ],
)
def test_broadcast_matches_torch(
    openfold3_source: Path, counts: list[int]
) -> None:
    torch = _torch()
    upstream = _upstream()
    c_feat = 5
    n_token = len(counts)
    token_mask = torch.ones(1, n_token)
    num_atoms = torch.tensor([counts], dtype=torch.float32)
    token_feat = torch.randn(1, n_token, c_feat)

    expected = upstream(
        token_mask=token_mask,
        num_atoms_per_token=num_atoms,
        token_feat=token_feat,
        token_dim=-2,
    )
    actual = broadcast_token_feat_to_atoms(
        jnp.asarray(token_mask.numpy()),
        jnp.asarray(num_atoms.numpy()),
        jnp.asarray(token_feat.numpy()),
        n_atom=expected.shape[-2],
    )
    assert actual.shape == tuple(expected.shape)
    np.testing.assert_allclose(
        np.asarray(actual, dtype=np.float64),
        expected.detach().numpy().astype(np.float64),
        rtol=1e-6,
        atol=1e-6,
    )


def test_masked_tokens_contribute_no_atoms(openfold3_source: Path) -> None:
    """A masked token's atom count is zeroed before the broadcast."""
    torch = _torch()
    upstream = _upstream()
    token_mask = torch.tensor([[1.0, 0.0, 1.0]])
    num_atoms = torch.tensor([[2.0, 3.0, 2.0]])
    token_feat = torch.randn(1, 3, 4)

    expected = upstream(
        token_mask=token_mask,
        num_atoms_per_token=num_atoms,
        token_feat=token_feat,
        token_dim=-2,
    )
    actual = broadcast_token_feat_to_atoms(
        jnp.asarray(token_mask.numpy()),
        jnp.asarray(num_atoms.numpy()),
        jnp.asarray(token_feat.numpy()),
        n_atom=expected.shape[-2],
    )
    np.testing.assert_allclose(
        np.asarray(actual, dtype=np.float64),
        expected.detach().numpy().astype(np.float64),
        rtol=1e-6,
        atol=1e-6,
    )
    # The masked token's feature must appear nowhere in the output.
    assert np.allclose(np.asarray(actual)[0, 2:4], token_feat[0, 2].numpy())


def test_each_atom_gets_its_owning_token(openfold3_source: Path) -> None:
    """Distinct per-token values make the run boundaries directly checkable."""
    counts = [3, 1, 2]
    token_mask = jnp.ones((1, 3))
    num_atoms = jnp.asarray([counts], dtype=jnp.float32)
    # Token i carries the constant value i + 1.
    token_feat = jnp.asarray([[[1.0], [2.0], [3.0]]])
    actual = broadcast_token_feat_to_atoms(
        token_mask, num_atoms, token_feat, n_atom=sum(counts)
    )
    np.testing.assert_allclose(
        np.asarray(actual).reshape(-1), [1.0, 1.0, 1.0, 2.0, 3.0, 3.0]
    )


def test_atoms_past_the_last_boundary_are_zero(openfold3_source: Path) -> None:
    """Requesting more atoms than the tokens own must pad with zeros."""
    token_mask = jnp.ones((1, 2))
    num_atoms = jnp.asarray([[2.0, 1.0]], dtype=jnp.float32)
    token_feat = jnp.asarray([[[5.0], [7.0]]])
    actual = broadcast_token_feat_to_atoms(
        token_mask, num_atoms, token_feat, n_atom=6
    )
    np.testing.assert_allclose(
        np.asarray(actual).reshape(-1), [5.0, 5.0, 7.0, 0.0, 0.0, 0.0]
    )


def test_rejects_a_featureless_token_tensor(openfold3_source: Path) -> None:
    with pytest.raises(ValueError, match="trailing feature axis"):
        broadcast_token_feat_to_atoms(
            jnp.ones(3), jnp.ones(3), jnp.ones(3), n_atom=3
        )
