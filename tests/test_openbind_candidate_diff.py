import json

import numpy as np
import pytest

from bench.boltz_historical_replay import digest
from bench.openbind_candidate_diff import compare


def _candidate(root, coordinates, *, backend="cueq", source=None):
    root.mkdir()
    with (root / "prediction.npz").open("xb") as stream:
        np.savez_compressed(
            stream,
            coordinates=coordinates,
            plddt=np.full((5, 3), 0.9, np.float32),
        )
    (root / "preflight.json").write_text(json.dumps({
        "input_sha256": "in", "tape_sha256": "tape", "checkpoint_sha256": "ckpt",
        "candidate_backend": backend, "source": source or {"a.py": "1"},
    }))
    (root / "finished.json").write_text(json.dumps({
        "prediction_sha256": digest(root / "prediction.npz"),
    }))
    return root


@pytest.fixture
def capture(tmp_path):
    root = tmp_path / "capture"
    root.mkdir()
    fields = {
        "chain_id": ["A", "A", "B"], "res_id": [1, 2, 1],
        "res_name": ["GLY", "ALA", "LIG"], "atom_name": ["CA", "CA", "C1"],
        "element": ["C", "C", "C"],
    }
    with (root / "input.npz").open("xb") as stream:
        np.savez_compressed(
            stream,
            atom_mask=np.ones((1, 3), np.float32),
            **{"atom_array.0.annotation." + k: np.array(v) for k, v in fields.items()},
        )
    return root


def test_identical_candidates_are_bitwise_equal(tmp_path, capture):
    coordinates = np.random.default_rng(0).normal(size=(5, 3, 3)).astype(np.float32)
    left = _candidate(tmp_path / "left", coordinates)
    right = _candidate(tmp_path / "right", coordinates.copy())
    report = compare(capture, left, right)
    assert report["fields"]["coordinates"]["bitwise_equal"]
    assert report["same_source"] and report["same_backend"]
    assert max(report["coordinates"]["entity_max_rmsd"].values()) < 1e-12


def test_moved_atoms_are_measured_not_hidden(tmp_path, capture):
    coordinates = np.random.default_rng(1).normal(size=(5, 3, 3)).astype(np.float32)
    moved = coordinates.copy()
    moved[0, 2] += 1.0
    left = _candidate(tmp_path / "left", coordinates)
    right = _candidate(tmp_path / "right", moved, source={"a.py": "2"})
    report = compare(capture, left, right)
    assert not report["fields"]["coordinates"]["bitwise_equal"]
    assert not report["same_source"]
    assert report["coordinates"]["entity_max_rmsd"]["B"] > 0.0


def test_different_captures_are_refused(tmp_path, capture):
    coordinates = np.zeros((5, 3, 3), np.float32)
    left = _candidate(tmp_path / "left", coordinates)
    right = _candidate(tmp_path / "right", coordinates)
    preflight = json.loads((right / "preflight.json").read_text())
    preflight["tape_sha256"] = "other"
    (right / "preflight.json").write_text(json.dumps(preflight))
    with pytest.raises(ValueError, match="different captures"):
        compare(capture, left, right)
