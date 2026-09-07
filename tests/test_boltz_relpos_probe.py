from types import SimpleNamespace

import numpy as np
import pytest

from bench.boltz_relpos_probe import (
    bf16_round,
    comparison,
    dense_relative_features,
    exact_sparse_projection,
    torch_policy,
    validate_features,
)


def _features():
    return {
        "asym_id": np.asarray([[0, 0, 1]], dtype=np.int64),
        "residue_index": np.asarray([[0, 0, 7000]], dtype=np.int64),
        "entity_id": np.asarray([[0, 0, 1]], dtype=np.int64),
        "token_index": np.asarray([[0, 1, 2]], dtype=np.int64),
        "sym_id": np.asarray([[0, 0, 0]], dtype=np.int64),
        "cyclic_period": np.zeros((1, 3), dtype=np.float32),
    }


def test_dense_native_category_order_and_entity_slot():
    dense = dense_relative_features(_features())
    assert dense.shape == (1, 3, 3, 139)
    assert dense.dtype == np.float32
    np.testing.assert_array_equal(np.flatnonzero(dense[0, 0, 1]), [32, 97, 132, 135])
    np.testing.assert_array_equal(np.flatnonzero(dense[0, 0, 2]), [65, 131, 138])


def test_native_skips_period_wrap_when_all_periods_zero():
    features = _features()
    features["asym_id"][:] = 0
    dense = dense_relative_features(features)
    assert dense[0, 0, 2, 0] == 1
    assert dense[0, 2, 0, 64] == 1


def test_native_cyclic_distance_wrap():
    features = _features()
    features["asym_id"][:] = 0
    features["residue_index"][0, 2] = 5
    features["cyclic_period"][:] = 6
    dense = dense_relative_features(features)
    assert dense[0, 0, 2, 33] == 1
    assert dense[0, 2, 0, 31] == 1


def test_bf16_round_is_direct_round_to_nearest_even():
    values = np.asarray([1 + 1 / 256, 1 + 3 / 256, -1 - 1 / 256, -1 - 3 / 256])
    np.testing.assert_array_equal(bf16_round(values), [1, 1 + 1 / 64, -1, -1 - 1 / 64])
    np.testing.assert_array_equal(
        bf16_round([2.0**-134, 3 * 2.0**-134]), [0, 2.0**-132]
    )
    with pytest.raises(ValueError, match="finite"):
        bf16_round([np.inf])


def test_exact_sum_retains_selected_terms_lost_by_bf16_partial_sums():
    dense = np.zeros((1, 2, 139), dtype=np.float32)
    dense[..., [0, 66, 133]] = 1
    dense[:, 1, 132] = 1
    weight = np.zeros((1, 139), dtype=np.float32)
    weight[0, [0, 66, 132, 133]] = [1, 1 / 256, 1 / 512, -1]
    actual = exact_sparse_projection(dense, weight)
    assert actual.dtype == np.float64
    np.testing.assert_array_equal(actual[..., 0], [[1 / 256, 3 / 512]])
    dense[:, 0, 1] = 1
    with pytest.raises(ValueError, match="one-hot"):
        exact_sparse_projection(dense, weight)


@pytest.mark.parametrize("defect", ["extra", "missing", "shape", "float", "nan"])
def test_feature_schema_rejects_unknown_or_malformed_input(defect):
    features = _features()
    if defect == "extra":
        features["other"] = np.zeros((1, 3))
    elif defect == "missing":
        del features["sym_id"]
    elif defect == "shape":
        features["sym_id"] = np.zeros((1, 2), dtype=np.int64)
    elif defect == "float":
        features["sym_id"] = features["sym_id"].astype(np.float32)
    else:
        features["cyclic_period"][0, 0] = np.nan
    with pytest.raises(ValueError):
        validate_features(features)


def test_counterfactual_restores_native_policy_even_on_error():
    state = {"precision": "high"}
    torch = SimpleNamespace(
        backends=SimpleNamespace(
            cuda=SimpleNamespace(
                matmul=SimpleNamespace(allow_bf16_reduced_precision_reduction=True)
            )
        ),
        get_float32_matmul_precision=lambda: state["precision"],
        set_float32_matmul_precision=lambda value: state.update(precision=value),
    )
    with pytest.raises(RuntimeError):
        with torch_policy(torch, reduction=False, precision="highest"):
            assert (
                torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
                is False
            )
            assert state["precision"] == "highest"
            raise RuntimeError("probe failure")
    assert torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction is True
    assert state["precision"] == "high"


def test_comparison_keeps_small_nonzero_differences():
    result = comparison(np.asarray([1.0, 1.0625]), np.asarray([1.0, 1.0]))
    assert result["max_abs"] == 0.0625
    assert result["unequal"] == 1
    assert result["values_equal"] is False
    with pytest.raises(ValueError, match="nonfinite"):
        comparison(np.asarray([np.nan]), np.asarray([1.0]))
