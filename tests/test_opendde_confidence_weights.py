from collections import namedtuple

import numpy as np
import pytest

from bench.opendde_confidence_weights import TrackedState, compare_trees

Weights = namedtuple("Weights", "array optional setting")


@pytest.mark.parametrize("change", [None, "array", "nonfinite", "optional", "static"])
def test_weight_proof_requires_exact_finite_arrays_and_static_leaves(change):
    left = Weights(np.asarray([1.0, 2.0], np.float32), None, 3)
    right = Weights(left.array.copy(), None, 3)
    if change == "array":
        right.array[1] += 1e-6
    elif change == "nonfinite":
        left.array[0] = right.array[0] = np.nan
    elif change == "optional":
        right = right._replace(optional=1)
    elif change == "static":
        right = right._replace(setting=4)
    result = compare_trees(left, right)
    assert result["passed"] is (change is None)
    if change is None:
        assert (
            result["canonical_contents_sha256"][0]
            == result["canonical_contents_sha256"][1]
        )


def test_native_leaf_consumption_is_counted_not_assumed():
    values = TrackedState({"read": 1, "unread": 2})
    assert values["read"] == values["read"] == 1
    assert values.reads == {"read": 2}
    assert values.keys() - values.reads.keys() == {"unread"}
