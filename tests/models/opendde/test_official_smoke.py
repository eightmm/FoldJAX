from __future__ import annotations

import numpy as np
import pytest

from .scripts.run_official_smoke import (
    collect_raw_arrays,
    coordinate_error_metrics,
    load_random_tape,
)


def test_load_random_tape_validates_and_returns_shared_arrays(tmp_path) -> None:
    path = tmp_path / "tape.npz"
    np.savez_compressed(
        path,
        noise_schedule=np.asarray([2560.0, 0.0], dtype=np.float32),
        init_noise=np.zeros((1, 3, 3), dtype=np.float32),
        step_noises=np.ones((1, 1, 3, 3), dtype=np.float32),
        rotations=np.eye(3, dtype=np.float32)[None, None],
        translations=np.zeros((1, 1, 3), dtype=np.float32),
    )

    schedule, tape = load_random_tape(
        path,
        num_steps=1,
        num_samples=1,
        n_atom=3,
    )

    np.testing.assert_array_equal(schedule, [2560.0, 0.0])
    assert tuple(tape["init_noise"].shape) == (1, 3, 3)
    assert len(tape["step_noises"]) == 1
    assert tuple(tape["rotations"].shape) == (1, 1, 3, 3)
    assert tuple(tape["translations"].shape) == (1, 1, 3)


def test_load_random_tape_rejects_wrong_atom_count(tmp_path) -> None:
    path = tmp_path / "tape.npz"
    np.savez_compressed(
        path,
        noise_schedule=np.asarray([2560.0, 0.0], dtype=np.float32),
        init_noise=np.zeros((1, 2, 3), dtype=np.float32),
        step_noises=np.zeros((1, 1, 2, 3), dtype=np.float32),
        rotations=np.eye(3, dtype=np.float32)[None, None],
        translations=np.zeros((1, 1, 3), dtype=np.float32),
    )

    with pytest.raises(ValueError, match="init_noise expected shape"):
        load_random_tape(path, num_steps=1, num_samples=1, n_atom=3)


def test_collect_raw_arrays_preserves_named_model_tensors() -> None:
    output = {
        "coordinate": np.zeros((1, 2, 3), dtype=np.float32),
        "s_trunk": np.ones((2, 4), dtype=np.float32),
        "metadata": "ignored",
    }

    arrays = collect_raw_arrays(output)

    assert sorted(arrays) == ["coordinate", "s_trunk"]
    np.testing.assert_array_equal(arrays["s_trunk"], 1.0)


def test_coordinate_error_metrics_reports_component_and_atom_errors() -> None:
    actual = np.asarray([[[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]])
    reference = np.zeros((2, 3), dtype=np.float64)

    metrics = coordinate_error_metrics(actual, reference)

    assert metrics["raw_coordinate_rmse_angstrom"] == pytest.approx(np.sqrt(5.0 / 6.0))
    assert metrics["all_atom_rmsd_angstrom"] == pytest.approx(np.sqrt(5.0 / 2.0))
    assert metrics["coordinate_max_abs_error_angstrom"] == 2.0


def test_coordinate_error_metrics_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="reference coordinates expected shape"):
        coordinate_error_metrics(
            np.zeros((1, 2, 3)),
            np.zeros((3, 3)),
        )
