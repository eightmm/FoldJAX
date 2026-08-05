"""Torch-vs-JAX parity for sequence-local atom blocking.

These are pure index/mask computations, so there are no weights to randomize.
The edge cases are the point: a window that underflows shifts right, one that
overflows shifts left, and the true atom count comes from the mask rather than
the padded length.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.openfold3.models.atom_blocks import (
    block_indices,
    pair_atom_block_mask,
    query_block_padding,
    single_rep_to_blocks,
)

pytestmark = pytest.mark.torch_parity


def _torch():
    import torch

    torch.manual_seed(0)
    return torch


def _upstream():
    from openfold3.core.utils import atom_attention_block_utils as utils

    return utils


@pytest.mark.parametrize(
    ("n_atom", "n_query"), [(10, 4), (12, 4), (1, 8), (7, 1), (33, 8)]
)
def test_query_block_padding_matches_torch(
    openfold3_source: Path, n_atom: int, n_query: int
) -> None:
    assert query_block_padding(n_atom, n_query) == _upstream().get_query_block_padding(
        n_atom=n_atom, n_query=n_query
    )


def _atom_mask(torch, n_atom: int, valid: list[int]):
    mask = torch.zeros(len(valid), n_atom)
    for row, count in enumerate(valid):
        mask[row, :count] = 1.0
    return mask


@pytest.mark.parametrize(
    ("n_atom", "valid", "n_query", "n_key"),
    [
        # Full-length, exact multiple.
        (16, [16], 4, 8),
        # Ragged batch: one short sample forces an overflow shift.
        (16, [16, 6, 1], 4, 8),
        # Padded length not a multiple of n_query.
        (14, [14, 9], 4, 8),
        # n_key wider than the whole structure.
        (8, [8, 3], 4, 16),
        # n_key == n_query, no surrounding context.
        (12, [12, 5], 4, 4),
    ],
)
def test_block_indices_match_torch(
    openfold3_source: Path, n_atom: int, valid: list[int], n_query: int, n_key: int
) -> None:
    torch = _torch()
    mask = _atom_mask(torch, n_atom, valid)
    expected_idx, expected_invalid = _upstream().get_block_indices(
        atom_mask=mask, n_query=n_query, n_key=n_key, device=torch.device("cpu")
    )
    actual_idx, actual_invalid = block_indices(
        jnp.asarray(mask.numpy()), n_query=n_query, n_key=n_key
    )
    np.testing.assert_array_equal(
        np.asarray(actual_idx), expected_idx.numpy().astype(np.int32)
    )
    np.testing.assert_array_equal(
        np.asarray(actual_invalid), expected_invalid.numpy()
    )


def test_block_indices_never_point_at_padding(openfold3_source: Path) -> None:
    """Every valid index must land on a real atom, not on mask padding."""
    torch = _torch()
    mask = _atom_mask(torch, 16, [16, 5, 2])
    indices, invalid = block_indices(
        jnp.asarray(mask.numpy()), n_query=4, n_key=8
    )
    counts = np.asarray(mask.sum(-1).numpy(), dtype=np.int64)
    idx = np.asarray(indices)
    bad = np.asarray(invalid)
    for row, count in enumerate(counts):
        live = idx[row][~bad[row]]
        assert live.size > 0
        assert live.min() >= 0
        assert live.max() < count, f"row {row} gathered padding"


@pytest.mark.parametrize(
    ("n_atom", "valid", "n_query", "n_key"),
    [(16, [16, 6], 4, 8), (14, [14, 9, 1], 4, 8), (8, [8, 3], 4, 16)],
)
def test_pair_atom_block_mask_matches_torch(
    openfold3_source: Path, n_atom: int, valid: list[int], n_query: int, n_key: int
) -> None:
    torch = _torch()
    import math

    mask = _atom_mask(torch, n_atom, valid)
    num_blocks = math.ceil(n_atom / n_query)
    pad = query_block_padding(n_atom, n_query)
    idx, invalid = _upstream().get_block_indices(
        atom_mask=mask, n_query=n_query, n_key=n_key, device=torch.device("cpu")
    )
    expected = _upstream().get_pair_atom_block_mask(
        atom_mask=mask,
        num_blocks=num_blocks,
        n_query=n_query,
        n_key=n_key,
        pad_len_right_q=pad,
        key_block_idxs=idx,
        invalid_mask=invalid,
    )
    actual = pair_atom_block_mask(
        jnp.asarray(mask.numpy()),
        n_query=n_query,
        indices=jnp.asarray(idx.numpy()),
        invalid=jnp.asarray(invalid.numpy()),
    )
    np.testing.assert_allclose(
        np.asarray(actual), expected.numpy(), rtol=1e-6, atol=1e-6
    )


@pytest.mark.parametrize(
    ("n_atom", "valid", "n_query", "n_key"),
    [(16, [16, 6], 4, 8), (14, [14, 9], 4, 8), (12, [12, 5], 4, 4)],
)
def test_single_rep_to_blocks_matches_torch(
    openfold3_source: Path, n_atom: int, valid: list[int], n_query: int, n_key: int
) -> None:
    torch = _torch()
    c_atom = 6
    ql = torch.randn(len(valid), n_atom, c_atom)
    mask = _atom_mask(torch, n_atom, valid)
    expected_q, expected_k, expected_pair = _upstream().convert_single_rep_to_blocks(
        ql=ql, n_query=n_query, n_key=n_key, atom_mask=mask
    )
    actual_q, actual_k, actual_pair = single_rep_to_blocks(
        jnp.asarray(ql.numpy()),
        jnp.asarray(mask.numpy()),
        n_query=n_query,
        n_key=n_key,
    )
    for actual, expected, name in (
        (actual_q, expected_q, "query blocks"),
        (actual_k, expected_k, "key blocks"),
        (actual_pair, expected_pair, "pair mask"),
    ):
        assert actual.shape == tuple(expected.shape), name
        np.testing.assert_allclose(
            np.asarray(actual, dtype=np.float64),
            expected.numpy().astype(np.float64),
            rtol=1e-6,
            atol=1e-6,
            err_msg=f"{name} diverged from the OpenFold3 reference",
        )


def test_invalid_key_positions_are_zeroed(openfold3_source: Path) -> None:
    """Out-of-range key slots must be zero, not a wrapped-around atom."""
    torch = _torch()
    ql = torch.randn(1, 8, 4) + 10.0  # far from zero so zeros are unambiguous
    mask = _atom_mask(torch, 8, [3])
    _q, keys, _pair = single_rep_to_blocks(
        jnp.asarray(ql.numpy()), jnp.asarray(mask.numpy()), n_query=4, n_key=8
    )
    indices, invalid = block_indices(
        jnp.asarray(mask.numpy()), n_query=4, n_key=8
    )
    zeroed = np.asarray(keys)[np.asarray(invalid)]
    assert zeroed.size > 0, "this fixture should produce invalid positions"
    assert np.allclose(zeroed, 0.0)
