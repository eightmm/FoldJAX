from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


def _load_module():
    path = (
        Path(__file__).resolve().parent
        / "scripts"
        / "reference_augmentation_tape.py"
    )
    spec = importlib.util.spec_from_file_location("reference_augmentation_tape", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tape_module = _load_module()


def test_reference_augmentation_tape_is_rigid_and_replay_exact(tmp_path: Path) -> None:
    path = tmp_path / "tape.npz"
    tape_module.save_tape(path, seed=7, count=2)
    position = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [-2.0, 1.0, 0.5]], np.float32
    )

    first = tape_module.ReferenceAugmentationTape(path)
    second = tape_module.ReferenceAugmentationTape(path)
    transformed = first.transform(position)
    np.testing.assert_array_equal(transformed, second.transform(position))
    np.testing.assert_allclose(
        np.linalg.norm(transformed[:, None] - transformed[None, :], axis=-1),
        np.linalg.norm(position[:, None] - position[None, :], axis=-1),
        atol=5e-7,
        rtol=5e-7,
    )
    first.transform(position)
    first.assert_exhausted()


def test_reference_augmentation_tape_fails_on_consumption_mismatch(
    tmp_path: Path,
) -> None:
    path = tmp_path / "tape.npz"
    tape_module.save_tape(path, seed=0, count=1)
    tape = tape_module.ReferenceAugmentationTape(path)
    with pytest.raises(ValueError, match="consumption mismatch"):
        tape.assert_exhausted()
    tape.transform(np.zeros((1, 3), np.float32))
    with pytest.raises(ValueError, match="exhausted"):
        tape.transform(np.zeros((1, 3), np.float32))
