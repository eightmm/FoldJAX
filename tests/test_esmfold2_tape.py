from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

# ruff: noqa: E501
from bench import esmfold2_tape as tape


def _shapes() -> tape.TapeShapes:
    return tape.TapeShapes(1, 3, 4, 2, 5, 2, 2, None, 1024, 0.1)


def _events(shapes: tape.TapeShapes):
    pair = np.zeros((1, 3, 3, 2), np.float32)
    atom = np.zeros((5, 4, 3), np.float32)
    rotation = np.zeros((5, 4), np.float32)
    translation = np.zeros((5, 1, 3), np.float32)
    return (
        [pair],
        [np.ones_like(pair), np.ones_like(pair)],
        [],
        [],
        [atom, rotation, translation, atom, rotation, translation, atom],
    )


def test_classifies_complete_native_tape() -> None:
    shapes = _shapes()
    initial, dropout, rand, perm, normal = _events(shapes)
    got = tape.classify_random_events(
        shapes,
        initial_pair=initial,
        dropout=dropout,
        rand=rand,
        randperm=perm,
        normal=normal,
    )
    assert got["lm_dropout_masks"].shape == (2, 1, 3, 3, 2)
    assert got["diffusion_churn_normals"].shape == (2, 5, 4, 3)
    assert "msa_row_choices" not in got
    assert "msa_column_keep" not in got


def test_msa_row_tape_retains_query_and_integer_indices():
    shapes = replace(_shapes(), msa_depth=5, max_msa_depth=3)
    initial, dropout, _, _, normal = _events(shapes)
    got = tape.classify_random_events(
        shapes,
        initial_pair=initial,
        dropout=dropout,
        rand=[np.ones((1, 3))],
        randperm=[np.array([3, 1, 0, 2]), np.array([2, 0, 3, 1])],
        normal=normal,
    )
    np.testing.assert_array_equal(got["msa_row_choices"], [[0, 2, 4], [0, 1, 3]])
    assert got["msa_row_choices"].dtype == np.int64


def test_rejects_unclassified_or_missing_random_consumer() -> None:
    shapes = _shapes()
    initial, dropout, rand, perm, normal = _events(shapes)
    with pytest.raises(ValueError, match="diffusion normal"):
        tape.classify_random_events(
            shapes,
            initial_pair=initial,
            dropout=dropout,
            rand=rand,
            randperm=perm,
            normal=normal[:-1],
        )
def test_replay_rejects_missing_core_contract_before_loading():
    from argparse import Namespace

    from bench.esmfold2_tape import _replay

    # No paths or checkpoint attributes: a path access would fail this test.
    with pytest.raises(RuntimeError, match="full tape replay is not implemented"):
        _replay(Namespace())
