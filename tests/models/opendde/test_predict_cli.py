from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

from foldjax.models.opendde.cli import predict as predict_impl


def test_predict_cli_runs_native_json_to_ranked_output(
    tmp_path, monkeypatch, capsys
) -> None:
    input_path = tmp_path / "tiny.json"
    weights_path = tmp_path / "opendde.jax"
    output_dir = tmp_path / "out"
    components_path = tmp_path / "components.cif"
    rdkit_cache_path = tmp_path / "components.cif.rdkit_mol.pkl"
    template_mmcif_dir = tmp_path / "template_mmcif"
    template_release_dates = tmp_path / "release_date_cache.json"
    template_obsolete_map = tmp_path / "obsolete_to_successor.json"
    kalign_binary = tmp_path / "kalign"
    job = {"name": "tiny", "modelSeeds": [101], "sequences": []}
    input_path.write_text(json.dumps([job]), encoding="utf-8")
    weights_path.write_bytes(b"native fixture")
    components_path.write_text("data_TST\n", encoding="utf-8")
    rdkit_cache_path.write_bytes(b"trusted fixture")
    template_mmcif_dir.mkdir()
    template_release_dates.write_text("{}", encoding="utf-8")
    template_obsolete_map.write_text("{}", encoding="utf-8")
    kalign_binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    kalign_binary.chmod(0o755)
    monkeypatch.delenv("PROTENIX_CCD_COMPONENTS_FILE", raising=False)
    monkeypatch.delenv("PROTENIX_CCD_RDKIT_MOL_FILE", raising=False)
    monkeypatch.delenv("PROTENIX_TEMPLATE_MMCIF_DIR", raising=False)
    monkeypatch.delenv("PROTENIX_TEMPLATE_RELEASE_DATES_FILE", raising=False)
    monkeypatch.delenv("PROTENIX_TEMPLATE_OBSOLETE_FILE", raising=False)
    monkeypatch.delenv("PROTENIX_KALIGN_BINARY", raising=False)
    features = {"restype": np.zeros((2, 32), dtype=np.float32)}
    raw_output = {"coordinate": np.zeros((1, 3, 3), dtype=np.float32)}
    scored_output = {**raw_output, "atom_plddt": np.ones((1, 3))}
    params = object()
    calls = []
    featurize_calls = []

    monkeypatch.setattr(predict_impl, "_load_jobs", lambda path: [job])

    def fake_featurize(value, **kwargs):
        featurize_calls.append((value, kwargs))
        return features

    monkeypatch.setattr(predict_impl, "_featurize", fake_featurize)
    monkeypatch.setattr(predict_impl, "_load_weights", lambda path: params)

    def fake_predict(value, model_params, **kwargs):
        calls.append((value, model_params, kwargs))
        return raw_output

    monkeypatch.setattr(predict_impl, "_predict", fake_predict)
    monkeypatch.setattr(
        predict_impl,
        "_score",
        lambda output, value, *, num_recycles: scored_output,
    )
    expected_path = output_dir / "tiny" / "seed_101" / "predictions" / "tiny.cif"
    monkeypatch.setattr(
        predict_impl,
        "_write",
        lambda root, **kwargs: [expected_path],
    )

    predict_impl.main(
        [
            "--input-json",
            str(input_path),
            "--weights",
            str(weights_path),
            "--out",
            str(output_dir),
            "--n-sample",
            "1",
            "--n-step",
            "2",
            "--n-cycle",
            "3",
            "--components-cif",
            str(components_path),
            "--ccd-rdkit-cache",
            str(rdkit_cache_path),
            "--template-mmcif-dir",
            str(template_mmcif_dir),
            "--template-release-dates",
            str(template_release_dates),
            "--template-obsolete-map",
            str(template_obsolete_map),
            "--kalign-binary",
            str(kalign_binary),
        ]
    )

    assert calls[0][0] is features
    assert calls[0][1] is params
    assert calls[0][2]["seed"] == 101
    assert calls[0][2]["n_sample"] == 1
    assert calls[0][2]["n_step"] == 2
    assert calls[0][2]["n_cycle"] == 3
    assert featurize_calls[0][0] is job
    assert featurize_calls[0][1]["seed"] == 101
    assert os.environ["PROTENIX_CCD_COMPONENTS_FILE"] == str(components_path.resolve())
    assert os.environ["PROTENIX_CCD_RDKIT_MOL_FILE"] == str(rdkit_cache_path.resolve())
    assert os.environ["PROTENIX_TEMPLATE_MMCIF_DIR"] == str(
        template_mmcif_dir.resolve()
    )
    assert os.environ["PROTENIX_TEMPLATE_RELEASE_DATES_FILE"] == str(
        template_release_dates.resolve()
    )
    assert os.environ["PROTENIX_TEMPLATE_OBSOLETE_FILE"] == str(
        template_obsolete_map.resolve()
    )
    assert os.environ["PROTENIX_KALIGN_BINARY"] == str(kalign_binary.resolve())
    assert f"wrote: {expected_path.parent}" in capsys.readouterr().out


def test_predict_cli_rejects_pt_weight_at_runtime(tmp_path) -> None:
    input_path = tmp_path / "tiny.json"
    checkpoint = tmp_path / "opendde.pt"
    input_path.write_text("[]", encoding="utf-8")
    checkpoint.write_bytes(b"torch fixture")

    with pytest.raises(SystemExit, match="export-weights"):
        predict_impl.main(
            [
                "--input-json",
                str(input_path),
                "--weights",
                str(checkpoint),
                "--out",
                str(tmp_path / "out"),
            ]
        )


def test_predict_cli_module_does_not_import_torch() -> None:
    assert "torch" not in predict_impl.__dict__
    assert not any(
        name == "opendde" or name.startswith("opendde.") for name in sys.modules
    )


def test_empty_model_seeds_use_release_default() -> None:
    assert predict_impl._job_seeds({"modelSeeds": []}, None) == [101]
