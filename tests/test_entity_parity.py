import json

import numpy as np
import pytest

from bench.entity_parity import compare_entity_parity, compare_feature_dicts


@pytest.mark.parametrize("dtype", [np.float32, np.int64, np.uint64])
def test_empty_feature_arrays_compare_exactly(dtype):
    features = {"empty_template": np.empty((0, 4), dtype=dtype)}
    report = compare_feature_dicts(features, features)
    assert report["equal"]
    assert report["max_absolute_error"]["empty_template"] == 0


def test_nonfinite_feature_on_only_one_side_is_rejected():
    with pytest.raises(ValueError, match="nonfinite"):
        compare_feature_dicts({"missing": np.array([np.nan])}, {})


def test_scalar_integer_features_report_exact_errors():
    report = compare_feature_dicts(
        {"n": np.array(5, np.int64)}, {"n": np.array(7, np.int64)}
    )
    assert report["max_absolute_error"]["n"] == 2
    assert report["value_mismatches"]["n"] == 1


def _coordinates():
    left = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [5.0, 5.0, 5.0]],
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [5.0, 5.0, 5.0]],
        ]
    )
    right = left.copy()
    right[1, 2, 0] += 0.5
    return left, right


def test_entity_parity_reports_each_sample_max_and_excludes_padding():
    left, right = _coordinates()
    right[:, :3] += np.array([4.0, -3.0, 2.0])
    right[:, 3] = 10_000.0
    report = compare_entity_parity(
        left,
        right,
        ["a", "b", "c", "pad"],
        ["a", "b", "c", "pad"],
        ["protein", "protein", "ligand", "padding"],
        ["protein", "protein", "ligand", "padding"],
        np.array([[True, True, True, False], [True, True, True, False]]),
        np.array([[True, True, True, False], [True, True, True, False]]),
    )
    assert json.loads(json.dumps(report)) == report
    assert report["atom_counts"] == [3, 3]
    assert report["global_rmsd"][0] < 1e-12
    assert report["entity_max_rmsd"]["ligand"] > 0.0
    assert report["coordinates_equal"] == [False, False]


def test_entity_parity_reorders_keys_and_uses_one_global_fit_for_ligand():
    left = np.array(
        [[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [1.0, 1.0, 1.0]]]
    )
    right = left[:, [2, 0, 3, 1]].copy()
    right[0, 2, 0] += 0.01
    report = compare_entity_parity(
        left,
        right,
        ["a", "b", "c", "lig"],
        ["c", "a", "lig", "b"],
        ["protein", "protein", "protein", "ligand"],
        ["protein", "protein", "ligand", "protein"],
        np.ones((1, 4), dtype=bool),
        np.ones((1, 4), dtype=bool),
    )
    assert list(report["entity_rmsd"]) == ["protein", "ligand"]
    assert report["entity_rmsd"]["ligand"][0] > 0.0


def test_entity_parity_matches_rigidly_transformed_coordinates():
    left = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    right = left @ rotation.T + np.array([5.0, -3.0, 2.0])
    report = compare_entity_parity(
        left,
        right,
        ["a", "b", "c"],
        ["a", "b", "c"],
        ["protein", "protein", "protein"],
        ["protein", "protein", "protein"],
        np.ones((1, 3), dtype=bool),
        np.ones((1, 3), dtype=bool),
    )
    assert report["global_max_rmsd"] < 1e-12


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"right_atom_keys": ["a", "a", "c", "d"]}, "duplicate"),
        ({"right_atom_keys": ["a", "b", "c", "missing"]}, "key sets"),
        ({"right_entity_labels": ["x", "p", "l", "pad"]}, "entity labels"),
        (
            {
                "right_mask": np.array(
                    [[True, False, True, False], [True, True, True, False]]
                )
            },
            "masks",
        ),
    ],
)
def test_entity_parity_rejects_correspondence_differences(kwargs, message):
    left, right = _coordinates()
    values = dict(
        left_atom_keys=["a", "b", "c", "d"],
        right_atom_keys=["a", "b", "c", "d"],
        left_entity_labels=["p", "p", "l", "pad"],
        right_entity_labels=["p", "p", "l", "pad"],
        left_mask=np.array([[True, True, True, False], [True, True, True, False]]),
        right_mask=np.array([[True, True, True, False], [True, True, True, False]]),
    )
    values.update(kwargs)
    with pytest.raises(ValueError, match=message):
        compare_entity_parity(left, right, **values)


@pytest.mark.parametrize(
    ("left", "right"),
    [
        (np.empty((0, 1, 3)), np.empty((0, 1, 3))),
        (np.empty((1, 0, 3)), np.empty((1, 0, 3))),
        (np.full((1, 1, 3), np.nan), np.zeros((1, 1, 3))),
    ],
)
def test_entity_parity_rejects_empty_and_nonfinite_coordinates(left, right):
    with pytest.raises(ValueError):
        compare_entity_parity(
            left,
            right,
            ["a"] if left.shape[1] else [],
            ["a"] if right.shape[1] else [],
            ["p"] if left.shape[1] else [],
            ["p"] if right.shape[1] else [],
            np.ones(left.shape[:2], dtype=bool),
            np.ones(right.shape[:2], dtype=bool),
        )


def test_feature_dicts_accept_strings_bools_and_report_safe_integer_errors():
    report = compare_feature_dicts(
        {
            "text": np.array(["A"]),
            "flag": np.array([True]),
            "uint": np.array([0], dtype=np.uint8),
            "wide": np.array([np.iinfo(np.int64).min], dtype=np.int64),
        },
        {
            "text": np.array(["A"]),
            "flag": np.array([True]),
            "uint": np.array([255], dtype=np.uint8),
            "wide": np.array([np.iinfo(np.int64).max], dtype=np.int64),
        },
    )
    assert report["equal"] is False
    assert "text" not in report["value_mismatches"]
    assert "flag" not in report["max_absolute_error"]
    assert report["max_absolute_error"]["uint"] == 255.0
    assert report["max_absolute_error"]["wide"] == float(2**64 - 1)


def test_feature_dicts_match_strings_and_booleans_exactly():
    report = compare_feature_dicts(
        {"text": np.array(["A"]), "flag": np.array([True])},
        {"text": np.array(["A"]), "flag": np.array([True])},
    )
    assert report["equal"] is True


@pytest.mark.parametrize(
    "left,right",
    [
        ({"a": np.array([1])}, {"b": np.array([1])}),
        ({"a": np.array([1])}, {"a": np.array([[1]])}),
        ({"a": np.array([1], dtype=np.int32)}, {"a": np.array([1], dtype=np.int64)}),
    ],
)
def test_feature_dicts_report_key_shape_and_dtype_differences(left, right):
    assert compare_feature_dicts(left, right)["equal"] is False


def test_feature_dicts_reject_nonfinite_numeric_values():
    with pytest.raises(ValueError, match="nonfinite"):
        compare_feature_dicts({"a": np.array([np.nan])}, {"a": np.array([0.0])})
