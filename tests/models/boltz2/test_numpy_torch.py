"""The NumPy array layer Boltz-2's vendored featurizer runs on.

These pin the places where torch and NumPy disagree. Each one, left alone,
returns a plausible array of the right shape and silently changes a feature —
which is the failure mode a shim like this has to be defended against.
"""

from __future__ import annotations

import numpy as np
import pytest

from foldjax.models.boltz2.data import _numpy_torch as t


def test_transpose_swaps_two_axes_rather_than_reversing_all() -> None:
    """np.transpose would reverse every axis; torch swaps the two named ones."""
    x = t.Tensor(np.zeros((2, 3, 4)))
    assert x.transpose(0, 1).shape == (3, 2, 4)
    assert x.permute(2, 0, 1).shape == (4, 2, 3)


def test_pad_counts_pairs_from_the_last_dimension_backwards() -> None:
    """torch's pad spec starts at the last dim; NumPy's starts at the first."""
    x = t.Tensor(np.ones((2, 3)))
    padded = t.nn.functional.pad(x, (1, 0))
    assert padded.shape == (2, 4), "padding landed on the wrong axis"
    np.testing.assert_array_equal(padded.array[:, 0], np.zeros(2))

    both = t.nn.functional.pad(x, (0, 1, 2, 0))
    assert both.shape == (4, 4)


def test_pad_uses_the_requested_fill() -> None:
    x = t.Tensor(np.ones((2, 2)))
    assert t.nn.functional.pad(x, (1, 0), value=7).array[0, 0] == 7


def test_dividing_a_float_by_an_integer_keeps_the_float_dtype() -> None:
    """float32 / int64 is float32 in torch and float64 in NumPy.

    The featurizer divides float32 coordinates by an integer count, so following
    NumPy here hands the model float64 `coords`.
    """
    coords = t.Tensor(np.ones((2, 3), dtype=np.float32))
    count = t.Tensor(np.asarray(4, dtype=np.int64))
    assert (coords / count).array.dtype == np.float32
    assert (coords * count).array.dtype == np.float32
    assert (coords + count).array.dtype == np.float32


def test_maximum_of_a_float_and_an_integer_keeps_the_float_dtype() -> None:
    """`torch.maximum(float32, int64)` is what produced float64 cyclic_period."""
    result = t.maximum(
        t.Tensor(np.ones(3, dtype=np.float32)),
        t.Tensor(np.zeros(3, dtype=np.int64)),
    )
    assert result.array.dtype == np.float32


def test_python_scalars_do_not_widen_a_tensor() -> None:
    x = t.Tensor(np.ones(3, dtype=np.float32))
    assert (x * 2.5).array.dtype == np.float32
    assert (x + 1).array.dtype == np.float32


def test_one_hot_matches_torch_shape_and_dtype() -> None:
    encoded = t.nn.functional.one_hot(t.Tensor(np.asarray([0, 2])), num_classes=3)
    assert encoded.shape == (2, 3)
    assert encoded.array.dtype == np.int64
    np.testing.assert_array_equal(encoded.array, [[1, 0, 0], [0, 0, 1]])


def test_reductions_accept_both_dim_and_axis() -> None:
    """torch accepts `axis` as an alias, and the vendored featurizer uses both."""
    x = t.Tensor(np.arange(6).reshape(2, 3))
    assert x.sum(dim=0).shape == (3,)
    assert x.mean(axis=0).shape == (3,)
    assert x.sum().item() == 15


def test_cdist_is_pairwise_over_the_last_dimension() -> None:
    a = t.Tensor(np.asarray([[[0.0, 0.0], [3.0, 4.0]]], dtype=np.float32))
    distances = t.cdist(a, a)
    assert distances.shape == (1, 2, 2)
    np.testing.assert_allclose(distances.array[0], [[0.0, 5.0], [5.0, 0.0]], atol=1e-6)


def test_indexing_accepts_tensor_masks_and_indices() -> None:
    x = t.Tensor(np.arange(5))
    mask = t.Tensor(np.asarray([True, False, True, False, True]))
    np.testing.assert_array_equal(x[mask].array, [0, 2, 4])
    x[mask] = t.Tensor(np.asarray([9, 9, 9]))
    np.testing.assert_array_equal(x.array, [9, 1, 9, 3, 9])


def test_unimplemented_operations_raise_rather_than_approximate() -> None:
    """A silent fallback here would be a silently different feature array."""
    with pytest.raises(NotImplementedError, match="pad mode"):
        t.nn.functional.pad(t.Tensor(np.ones(3)), (1, 1), mode="reflect")
    with pytest.raises(AttributeError):
        t.some_operation_that_does_not_exist  # noqa: B018


def test_the_selector_reports_which_backend_is_in_use() -> None:
    from foldjax.models.boltz2.data._torch import BACKEND

    assert BACKEND == "numpy"
