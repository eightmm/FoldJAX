import json
import sys

import pytest

from bench.esmfold2_msa_transition_replay import main


def test_pair_updates_reject_incomplete_capture(tmp_path, monkeypatch):
    (tmp_path / "report.json").write_text(
        json.dumps(
            {
                "engine": "native",
                "full_msa_transition": True,
            }
        )
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "probe",
            "--pair-updates",
            "--reference",
            str(tmp_path),
            "--weights",
            str(tmp_path / "weights"),
            "--out",
            str(tmp_path / "output.json"),
        ],
    )
    with pytest.raises(ValueError, match="full MSA block"):
        main()
    assert not (tmp_path / "output.json").exists()


@pytest.mark.parametrize("engine,transition", [("jax", True), ("native", False)])
def test_rejects_wrong_capture_before_gpu(tmp_path, monkeypatch, engine, transition):
    (tmp_path / "report.json").write_text(
        json.dumps(
            {
                "engine": engine,
                "full_msa_transition": transition,
            }
        )
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "probe",
            "--reference",
            str(tmp_path),
            "--weights",
            str(tmp_path / "weights"),
            "--out",
            str(tmp_path / "output.json"),
        ],
    )
    with pytest.raises(ValueError, match="requires native MSA transition"):
        main()
    assert not (tmp_path / "output.json").exists()


def test_rejects_unbound_weights_before_gpu(tmp_path, monkeypatch):
    (tmp_path / "weights").write_bytes(b"not-a-checkpoint")
    (tmp_path / "report.json").write_text(
        json.dumps(
            {
                "engine": "native",
                "full_msa_transition": True,
                "bindings": {},
                "archive_sha256": "unused",
            }
        )
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "probe",
            "--reference",
            str(tmp_path),
            "--weights",
            str(tmp_path / "weights"),
            "--out",
            str(tmp_path / "output.json"),
        ],
    )
    with pytest.raises(ValueError, match="weights differ"):
        main()
    assert not (tmp_path / "output.json").exists()
