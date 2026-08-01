from __future__ import annotations

import numpy as np

from foldjax.models.opendde.models.msa_sampling import sample_opendde_msa_cycle_features


def _features(msa: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "msa": msa,
        "has_deletion": msa.astype(np.float32) + 100.0,
        "deletion_value": msa.astype(np.float32) + 200.0,
    }


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
        n_cycle=3,
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
        n_cycle=4,
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
        n_cycle=1,
        seed=17,
        msa_depth=2,
    )

    assert {tuple(row) for row in cycle["msa"]} == {(1, 2), (5, 6)}
    np.testing.assert_array_equal(cycle["msa_mask"], 1.0)
