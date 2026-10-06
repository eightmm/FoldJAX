"""Boltz-2 affinity reaches the files, from the sample upstream would score.

Upstream ranks its samples by ``confidence_score``, saves the first-ranked one
as ``pre_affinity_<id>.npz`` for the affinity model, and writes the affinity
summary to ``affinity_<id>.json`` (``data/write/writer.py:73-79,178,303-324``).
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from foldjax.backends import boltz2 as boltz2_backend
from foldjax.backends.boltz2 import Boltz2Backend
from foldjax.models.boltz2 import api as native_api
from foldjax.schema import PredictionRequest


def test_the_affinity_sample_is_upstreams_first_ranked_not_the_best_iptm() -> None:
    out = {
        "confidence_score": np.asarray([0.2, 0.9, 0.5]),
        "iptm": np.asarray([0.95, 0.1, 0.1]),
    }
    assert native_api._affinity_input_sample(out) == 1
    # No confidence summary: upstream's identity ranking, sample 0.
    assert native_api._affinity_input_sample({"iptm": np.asarray([0.1, 0.9])}) == 0


def test_the_affinity_json_carries_upstreams_fields(tmp_path: Path) -> None:
    out = {
        "affinity_pred_value": np.asarray([1.5], np.float32),
        "affinity_probability_binary": np.asarray([0.25], np.float32),
        "affinity_pred_value1": np.asarray([1.0], np.float32),
        "affinity_probability_binary1": np.asarray([0.5], np.float32),
        "affinity_pred_value2": np.asarray([2.0], np.float32),
        "affinity_probability_binary2": np.asarray([0.0], np.float32),
        "plddt": np.ones(3),
    }
    path = native_api._write_affinity_summary(tmp_path, "job", out)
    assert path == tmp_path / "affinity_job.json"
    assert json.loads(path.read_text()) == {
        "affinity_pred_value": 1.5,
        "affinity_probability_binary": 0.25,
        "affinity_pred_value1": 1.0,
        "affinity_probability_binary1": 0.5,
        "affinity_pred_value2": 2.0,
        "affinity_probability_binary2": 0.0,
    }
    # Nothing to say without an affinity stage, so nothing is written.
    assert native_api._write_affinity_summary(tmp_path, "none", {}) is None
    assert not (tmp_path / "affinity_none.json").exists()


def test_the_adapter_field_list_is_the_native_one() -> None:
    assert boltz2_backend._AFFINITY_FIELDS == native_api.AFFINITY_SUMMARY_FIELDS


def test_affinity_lands_in_the_scored_samples_scores_only(
    tmp_path: Path, monkeypatch
) -> None:
    mols = tmp_path / "mols"
    mols.mkdir()
    job = tmp_path / "job.yaml"
    job.write_text("{}")
    weights = tmp_path / "weights"
    weights.mkdir()

    def native_predict(**kwargs):
        return {
            "coords": np.zeros((3, 2, 3)),
            "plddt": np.ones((3, 2)),
            "raw": {
                "confidence_score": np.asarray([0.2, 0.9, 0.5]),
                "affinity_pred_value": np.asarray([1.5]),
                "affinity_probability_binary": np.asarray([0.25]),
            },
            "out_paths": [None, None, None],
            "affinity_pred_value": np.asarray([1.5]),
            "affinity_probability_binary": np.asarray([0.25]),
            "affinity_input_sample": 1,
        }

    monkeypatch.setattr(
        "foldjax.backends.boltz2.import_module",
        lambda name: SimpleNamespace(predict=native_predict),
    )
    request = PredictionRequest(
        model="boltz2",
        input=job,
        weights=weights,
        output_dir=tmp_path / "out",
        seed=5,
        cache_dir=tmp_path / "cache",
        options={"mols": mols, "glu_backend": "xla"},
    )
    result = Boltz2Backend().predict(request)
    scores = [sample.scores for sample in result.samples]
    assert scores[1]["affinity_pred_value"] == 1.5
    assert scores[1]["affinity_probability_binary"] == 0.25
    assert not any(key.startswith("affinity_") for key in scores[0])
    assert not any(key.startswith("affinity_") for key in scores[2])
