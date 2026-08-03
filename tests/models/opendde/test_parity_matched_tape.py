"""Cover the MSA row injection the OpenDDE matched-tape harness depends on.

The harness's whole claim rests on one thing: that the rows it hands the JAX
trunk really are the rows upstream's MSA module read. If `build_cycle_msa`
quietly returned the wrong rows -- or the untouched alignment -- the parity run
would still print a number, and the number would be wrong in the direction that
looks like success only if you never check it. So the selection is tested
directly, including the featurizer-agreement report that decides whether the
cheaper `indices` mode is equivalent at all.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.models.opendde.scripts.parity_matched_tape import build_cycle_msa

FIELDS = ("msa", "has_deletion", "deletion_value")


def _features(n_row: int = 6, n_token: int = 4) -> dict[str, np.ndarray]:
    msa = np.arange(n_row * n_token, dtype=np.int32).reshape(n_row, n_token) % 31
    return {
        "msa": msa,
        "has_deletion": msa.astype(np.float32) + 100.0,
        "deletion_value": msa.astype(np.float32) + 200.0,
    }


def _archive(features: dict[str, np.ndarray], rows: np.ndarray) -> dict:
    archive = {"rows": rows}
    for name in FIELDS:
        archive[f"input_{name}"] = features[name]
        archive[f"selected_{name}"] = np.stack(
            [features[name][row] for row in rows]
        )
    return archive


def test_upstream_source_hands_back_the_rows_upstream_selected() -> None:
    features = _features()
    rows = np.asarray([[4, 1, 0], [2, 5, 3]], dtype=np.int64)
    archive = _archive(features, rows)

    cycles, agreement = build_cycle_msa(
        features, archive, source="upstream", n_cycle=2
    )

    assert cycles is not None and len(cycles) == 2
    for cycle_index, cycle in enumerate(cycles):
        for name in FIELDS:
            expected = features[name][rows[cycle_index]]
            np.testing.assert_array_equal(np.asarray(cycle[name]), expected)
        # Upstream uses its mask only to prioritize rows; once selected every
        # row enters the module unmasked, so a mask that is anything but ones
        # would silently reweight the alignment.
        assert np.all(np.asarray(cycle["msa_mask"]) == 1.0)
    assert agreement["upstream_sampled_rows"] == 3
    assert agreement["featurizer_msa_rows_identical"] is True


def test_indices_source_matches_upstream_source_when_featurizers_agree() -> None:
    features = _features()
    rows = np.asarray([[5, 0, 2]], dtype=np.int64)
    archive = _archive(features, rows)

    by_rows, _ = build_cycle_msa(features, archive, source="indices", n_cycle=1)
    by_arrays, _ = build_cycle_msa(features, archive, source="upstream", n_cycle=1)

    assert by_rows is not None and by_arrays is not None
    for name in FIELDS:
        np.testing.assert_array_equal(
            np.asarray(by_rows[0][name]), np.asarray(by_arrays[0][name])
        )


def test_featurizer_disagreement_is_reported_rather_than_assumed() -> None:
    """`indices` is only equivalent if both featurizers emit the same rows.

    That equivalence is an assumption until it is measured, and a harness that
    assumed it would attribute a featurizer difference to the model.
    """
    features = _features()
    rows = np.asarray([[0, 1, 2]], dtype=np.int64)
    archive = _archive(features, rows)
    shifted = dict(features)
    shifted["msa"] = features["msa"][::-1].copy()

    _, agreement = build_cycle_msa(shifted, archive, source="upstream", n_cycle=1)

    assert agreement["featurizer_msa_rows_identical"] is False
    assert agreement["featurizer_msa_row_match_fraction"] < 1.0


def test_whole_source_leaves_the_alignment_untouched() -> None:
    features = _features()
    archive = _archive(features, np.asarray([[0, 1]], dtype=np.int64))

    cycles, agreement = build_cycle_msa(features, archive, source="whole", n_cycle=1)

    assert cycles is None
    assert agreement["jax_msa_rows"] == 6


def test_a_capture_with_the_wrong_cycle_count_is_rejected() -> None:
    features = _features()
    archive = _archive(features, np.asarray([[0, 1]], dtype=np.int64))

    with pytest.raises(ValueError, match="MSA row draws"):
        build_cycle_msa(features, archive, source="upstream", n_cycle=2)
