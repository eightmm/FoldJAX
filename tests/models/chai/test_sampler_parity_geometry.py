from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


def _load_module():
    scripts = Path(__file__).resolve().parent / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        path = scripts / "benchmark_sampler_parity.py"
        spec = importlib.util.spec_from_file_location("benchmark_sampler_parity", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(scripts))


sampler = _load_module()


def test_geometry_metrics_remove_rigid_motion_and_preserve_distances() -> None:
    reference = np.asarray(
        [[[0, 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 3]]], dtype=np.float32
    )
    rotation = np.asarray([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32)
    actual = reference @ rotation + np.asarray([[[5, -3, 2]]], dtype=np.float32)

    metrics = sampler._coordinate_drift(reference, actual, np.ones((1, 4), dtype=bool))

    assert metrics["all_atom_raw_rmsd"] > 1.0
    assert metrics["all_atom_kabsch_rmsd"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["centroid_rmsd"] > 1.0
    assert metrics["local_pair_distance_mae"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["local_pair_lddt"] == pytest.approx(1.0)


def test_geometry_metrics_detect_non_rigid_local_drift() -> None:
    reference = np.asarray([[[0, 0, 0], [1, 0, 0], [2, 0, 0]]], dtype=np.float32)
    actual = reference.copy()
    actual[0, 2, 0] += 2.0

    metrics = sampler._coordinate_drift(reference, actual, np.ones((1, 3), dtype=bool))

    assert metrics["all_atom_kabsch_rmsd"] > 0.5
    assert metrics["local_pair_distance_mae"] > 0.5
    assert metrics["local_pair_lddt"] < 1.0
