"""Torch-vs-JAX parity for relative position encoding (AF3 Algorithm 3)."""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.openfold3.models.relpos import binned_one_hot, relpos_complex

pytestmark = pytest.mark.torch_parity

MAX_IDX, MAX_CHAIN = 4, 2


def _torch():
    import torch

    torch.manual_seed(0)
    return torch


def _upstream():
    from openfold3.core.utils.relpos import relpos_complex

    return relpos_complex


def _batch(torch, *, chains=(0, 0, 0, 1, 1), entities=(0, 0, 0, 0, 0)):
    n = len(chains)
    return {
        "residue_index": torch.arange(n).reshape(1, n),
        "asym_id": torch.tensor([list(chains)]),
        "entity_id": torch.tensor([list(entities)]),
        "token_index": torch.arange(n).reshape(1, n),
        "sym_id": torch.tensor([list(chains)]),
    }


@pytest.mark.parametrize(
    ("chains", "entities"),
    [
        ((0, 0, 0, 1, 1), (0, 0, 0, 0, 0)),   # two chains, one entity
        ((0, 0, 1, 1, 2), (0, 0, 1, 1, 2)),   # three chains, three entities
        ((0, 0, 0, 0, 0), (0, 0, 0, 0, 0)),   # a single chain
    ],
)
def test_relpos_matches_torch(
    openfold3_source: Path, chains: tuple, entities: tuple
) -> None:
    torch = _torch()
    batch = _batch(torch, chains=chains, entities=entities)
    expected = _upstream()(
        batch=batch,
        max_relative_idx=MAX_IDX,
        max_relative_chain=MAX_CHAIN,
    )
    actual = relpos_complex(
        {key: jnp.asarray(value.numpy()) for key, value in batch.items()},
        max_relative_idx=MAX_IDX,
        max_relative_chain=MAX_CHAIN,
    )
    assert actual.shape == tuple(expected.shape)
    np.testing.assert_allclose(
        np.asarray(actual, dtype=np.float64),
        expected.detach().numpy().astype(np.float64),
        rtol=1e-6,
        atol=1e-6,
    )


def test_binned_one_hot_matches_torch(openfold3_source: Path) -> None:
    torch = _torch()
    from openfold3.core.utils.tensor_utils import binned_one_hot as torch_binned

    x = torch.tensor([[0.0, 1.4, 2.6, 9.0]])
    bins = torch.arange(0, 5)
    expected = torch_binned(x, bins)
    actual = binned_one_hot(jnp.asarray(x.numpy()), jnp.asarray(bins.numpy()))
    np.testing.assert_allclose(
        np.asarray(actual, dtype=np.float64),
        expected.detach().numpy().astype(np.float64),
    )


def test_cross_chain_pairs_use_the_final_bin(openfold3_source: Path) -> None:
    """Unrelated pairs get their own bin rather than being zeroed."""
    torch = _torch()
    batch = _batch(torch, chains=(0, 0, 1, 1, 1))
    actual = np.asarray(
        relpos_complex(
            {k: jnp.asarray(v.numpy()) for k, v in batch.items()},
            max_relative_idx=MAX_IDX,
            max_relative_chain=MAX_CHAIN,
        )
    )
    residue_block = actual[..., : 2 * MAX_IDX + 2]
    # Token 0 (chain 0) vs token 3 (chain 1): condition fails -> last bin hot.
    assert residue_block[0, 0, 3, -1] == 1.0
    # Same chain: the last bin must not fire.
    assert residue_block[0, 0, 1, -1] == 0.0
    # Every position is exactly one-hot.
    np.testing.assert_allclose(residue_block.sum(-1), np.ones((1, 5, 5)))


def test_feature_width_is_the_documented_concatenation(
    openfold3_source: Path,
) -> None:
    torch = _torch()
    actual = relpos_complex(
        {k: jnp.asarray(v.numpy()) for k, v in _batch(torch).items()},
        max_relative_idx=MAX_IDX,
        max_relative_chain=MAX_CHAIN,
    )
    expected_width = (2 * MAX_IDX + 2) * 2 + 1 + (2 * MAX_CHAIN + 2)
    assert actual.shape[-1] == expected_width
