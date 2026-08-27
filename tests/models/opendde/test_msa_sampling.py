from __future__ import annotations

import numpy as np
import pytest

from foldjax.models.opendde.models.msa_sampling import (
    drop_sampled_msa_source_features,
    sample_opendde_msa_cycle_features,
)


def _features(msa: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "msa": msa,
        "has_deletion": msa.astype(np.float32) + 100.0,
        "deletion_value": msa.astype(np.float32) + 200.0,
    }


def _complete_cycle(depth: int = 2, tokens: int = 3) -> dict[str, np.ndarray]:
    shape = (depth, tokens)
    return {
        "msa": np.zeros(shape, dtype=np.int64),
        "has_deletion": np.zeros(shape, dtype=np.float32),
        "deletion_value": np.zeros(shape, dtype=np.float32),
        "msa_mask": np.ones(shape, dtype=np.float32),
    }


def test_sampled_msa_source_drop_is_surgical() -> None:
    source = {
        **_complete_cycle(depth=7),
        "profile": np.zeros((3, 32), dtype=np.float32),
        "deletion_mean": np.zeros((3,), dtype=np.float32),
        "constraint_feature": {"contact": np.asarray([1.0])},
        "template_aatype": np.asarray([[1, 2, 3]]),
        "writer_metadata": {"chain": "A"},
        "custom_feature": object(),
    }

    pruned = drop_sampled_msa_source_features(source, (_complete_cycle(),))

    assert pruned is not source
    for name in ("msa", "has_deletion", "deletion_value", "msa_mask"):
        assert name not in pruned
    for name in (
        "profile",
        "deletion_mean",
        "constraint_feature",
        "template_aatype",
        "writer_metadata",
        "custom_feature",
    ):
        assert pruned[name] is source[name]


@pytest.mark.parametrize("cycles", [None, ()])
def test_sampled_msa_source_drop_preserves_no_cycle_fallback(cycles) -> None:
    source = _complete_cycle()

    assert drop_sampled_msa_source_features(source, cycles) is source


@pytest.mark.parametrize(
    "malformation",
    [
        "missing",
        "mismatched",
        "different_cycle_shape",
        "ragged",
        "empty",
        "non_mapping",
        "later_incomplete",
    ],
)
def test_sampled_msa_source_drop_preserves_malformed_cycle_fallback(
    malformation: str,
) -> None:
    source = _complete_cycle(depth=7)
    cycle = _complete_cycle()
    cycles: tuple[object, ...]
    if malformation == "missing":
        cycle.pop("msa_mask")
        cycles = (cycle,)
    elif malformation == "mismatched":
        cycle["msa_mask"] = np.ones((2, 4), dtype=np.float32)
        cycles = (cycle,)
    elif malformation == "different_cycle_shape":
        cycles = (cycle, _complete_cycle(depth=3))
    elif malformation == "ragged":
        cycle["msa"] = [[1, 2, 3], [4]]  # type: ignore[assignment]
        cycles = (cycle,)
    elif malformation == "empty":
        cycles = (_complete_cycle(depth=0),)
    elif malformation == "non_mapping":
        cycles = (object(),)
    else:
        incomplete = _complete_cycle()
        incomplete.pop("deletion_value")
        cycles = (cycle, incomplete)

    assert drop_sampled_msa_source_features(source, cycles) is source  # type: ignore[arg-type]


def test_opendde_msa_sampling_uses_fixed_depth_and_preserves_duplicates() -> None:
    msa = np.asarray(
        [
            [1, 2, 3],
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12],
        ],
        dtype=np.int64,
    )

    cycles = sample_opendde_msa_cycle_features(
        _features(msa),
        num_recycles=3,
        seed=101,
        msa_depth=1280,
    )

    assert len(cycles) == 3
    expected_rows = sorted(map(tuple, msa.tolist()))
    for cycle in cycles:
        assert cycle["msa"].shape == (5, 3)
        assert sorted(map(tuple, cycle["msa"].tolist())) == expected_rows
        np.testing.assert_array_equal(cycle["msa_mask"], 1.0)
        np.testing.assert_array_equal(
            cycle["has_deletion"] - 100.0,
            cycle["msa"],
        )
        np.testing.assert_array_equal(
            cycle["deletion_value"] - 200.0,
            cycle["msa"],
        )


def test_opendde_msa_sampling_prioritizes_valid_rows_before_all_gap_rows() -> None:
    gap = 31
    msa = np.asarray(
        [
            [gap, gap, gap],
            [1, gap, gap],
            [gap, 2, gap],
            [gap, gap, 3],
            [gap, gap, gap],
        ],
        dtype=np.int64,
    )

    cycles = sample_opendde_msa_cycle_features(
        _features(msa),
        num_recycles=4,
        seed=37,
        msa_depth=3,
        gap_token=gap,
    )

    for cycle in cycles:
        assert cycle["msa"].shape == (3, 3)
        assert np.all(np.any(cycle["msa"] != gap, axis=-1))


def test_opendde_msa_sampling_uses_mask_only_for_row_priority() -> None:
    msa = np.asarray([[1, 2], [3, 4], [5, 6]], dtype=np.int64)
    features = _features(msa)
    features["msa_mask"] = np.asarray(
        [[1.0, 1.0], [0.0, 0.0], [1.0, 0.0]],
        dtype=np.float32,
    )

    (cycle,) = sample_opendde_msa_cycle_features(
        features,
        num_recycles=1,
        seed=17,
        msa_depth=2,
    )

    assert {tuple(row) for row in cycle["msa"]} == {(1, 2), (5, 6)}
    np.testing.assert_array_equal(cycle["msa_mask"], 1.0)
