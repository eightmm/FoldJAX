from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

from foldjax.models.opendde.cli import predict as predict_impl
from foldjax.schema import PredictionError


def _params_double():
    """A weights stand-in shaped like the real parameter tree.

    `object()` was enough while OpenDDE shipped float32, because that path
    casts nothing and never touches the tree. The shipped trunk is bfloat16
    since 2026-08-28, so `cast_trunk_params` runs on every default prediction
    -- these flows had simply never exercised it. Empty subtrees keep the
    double cheap: `jax.tree.map` over `{}` is a no-op, and `_replace` is what
    the cast actually needs.
    """
    from foldjax.models.opendde.models.model import OpenDDEInferenceParams

    return OpenDDEInferenceParams(
        **{name: {} for name in OpenDDEInferenceParams._fields}
    )


def _predict_kwargs() -> dict[str, object]:
    return {
        "seed": 101,
        "num_samples": 1,
        "num_steps": 2,
        "num_recycles": 2,
        "n_queries": 8,
        "n_keys": 16,
        "diffusion_attention_backend": "xla",
        "trunk_single_attention_backend": "xla",
        "structural_single_attention_backend": "xla",
    }


def _raw_msa_features(*, include_mask: bool = True) -> dict[str, object]:
    depth, tokens = 5, 3
    metadata = {"chain": "A"}
    features: dict[str, object] = {
        "restype": np.zeros((tokens, 32), dtype=np.float32),
        "msa": np.arange(depth * tokens, dtype=np.int64).reshape(depth, tokens),
        "has_deletion": np.full((depth, tokens), np.nan, dtype=np.float32),
        "deletion_value": np.full((depth, tokens), np.inf, dtype=np.float32),
        "profile": np.zeros((tokens, 32), dtype=np.float32),
        "deletion_mean": np.zeros((tokens,), dtype=np.float32),
        "constraint_feature": {"contact": np.asarray([1.0])},
        "template_aatype": np.asarray([[1, 2, 3]]),
        "writer_metadata": metadata,
        "custom_feature": object(),
    }
    if include_mask:
        features["msa_mask"] = np.ones((depth, tokens), dtype=np.float32)
    return features


def _capture_predict_route(monkeypatch):
    from foldjax.models.opendde.models import model as model_impl

    captured = {}

    def fake_infer(features, params, noise_schedule, **kwargs):
        captured["features"] = features
        captured["params"] = params
        captured["noise_schedule"] = noise_schedule
        captured["kwargs"] = kwargs
        return {"coordinate": np.zeros((1, 1, 3), dtype=np.float32)}

    monkeypatch.setattr(model_impl, "opendde_infer_compiled", fake_infer)
    monkeypatch.setattr(model_impl, "opendde_infer_static", fake_infer)

    def fake_preflight(features, _trunk_dtype):
        captured["preflight"] = features
        return None

    monkeypatch.setattr(predict_impl, "_preflight_arena", fake_preflight)
    return captured


@pytest.mark.parametrize("include_mask", [False, True])
def test_predict_drops_raw_msa_after_default_cycle_sampling(
    monkeypatch, include_mask: bool
) -> None:
    captured = _capture_predict_route(monkeypatch)
    features = _raw_msa_features(include_mask=include_mask)
    params = _params_double()

    predict_impl._predict(features, params, **_predict_kwargs())

    routed = captured["features"]
    assert captured["preflight"] is routed
    assert captured["params"] is params
    assert routed is not features
    for name in ("msa", "has_deletion", "deletion_value", "msa_mask"):
        assert name not in routed
    for name in (
        "restype",
        "profile",
        "deletion_mean",
        "constraint_feature",
        "template_aatype",
        "writer_metadata",
        "custom_feature",
    ):
        assert routed[name] is features[name]
    sampled = captured["kwargs"]["cycle_msa_features"]
    assert len(sampled) == 2
    assert all(cycle["msa"].shape == (5, 3) for cycle in sampled)


def test_predict_compacts_complete_binary_msa_cycles(monkeypatch) -> None:
    captured = _capture_predict_route(monkeypatch)
    features = _raw_msa_features()
    shape = np.shape(features["msa"])
    features["has_deletion"] = (
        np.arange(np.prod(shape)).reshape(shape) % 2
    ).astype(np.float32)
    features["deletion_value"] = np.linspace(
        0.0, 1.0, np.prod(shape), dtype=np.float32
    ).reshape(shape)

    predict_impl._predict(features, _params_double(), **_predict_kwargs())

    sampled = captured["kwargs"]["cycle_msa_features"]
    assert len(sampled) == 2
    for cycle in sampled:
        assert cycle["msa"].dtype == np.uint8
        assert cycle["has_deletion"].dtype == np.bool_
        assert cycle["msa_mask"].dtype == np.bool_
        assert cycle["deletion_value"].dtype == np.float32


@pytest.mark.parametrize("fallback", ["none", "incomplete"])
def test_predict_preserves_direct_msa_fallback(monkeypatch, fallback: str) -> None:
    captured = _capture_predict_route(monkeypatch)
    features = _raw_msa_features()
    if fallback == "none":
        cycles = ()
        expected_cycles = None
    else:
        cycles = (
            {
                "msa": np.zeros((2, 3), dtype=np.int64),
                "has_deletion": np.zeros((2, 3), dtype=np.float32),
                "deletion_value": np.zeros((2, 3), dtype=np.float32),
            },
        )
        expected_cycles = cycles

    predict_impl._predict(
        features,
        _params_double(),
        graph_jit=False,
        cycle_msa_features=cycles,
        **_predict_kwargs(),
    )

    assert captured["preflight"] is features
    assert captured["features"] is features
    assert captured["kwargs"]["cycle_msa_features"] is expected_cycles


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
    features = {
        "restype": np.zeros((2, 32), dtype=np.float32),
        "ref_element": np.eye(128, dtype=np.int64)[[6, 7, 8]],
        "ref_atom_name_chars": np.eye(64, dtype=np.int64)[
            [[32, 32, 32, 32], [33, 33, 33, 33], [34, 34, 34, 34]]
        ],
    }
    raw_output = {"coordinate": np.zeros((1, 3, 3), dtype=np.float32)}
    scored_output = {**raw_output, "atom_plddt": np.ones((1, 3))}
    params = _params_double()
    calls = []
    featurize_calls = []

    monkeypatch.setattr(predict_impl, "_load_jobs", lambda path: [job])

    def fake_featurize(value, **kwargs):
        featurize_calls.append((value, kwargs))
        return features

    monkeypatch.setattr(predict_impl, "_featurize", fake_featurize)
    monkeypatch.setattr(
        predict_impl,
        "_load_prepared_params",
        lambda path, trunk_dtype: params,
    )

    def fake_predict(value, model_params, **kwargs):
        calls.append((value, model_params, kwargs))
        return raw_output

    monkeypatch.setattr(predict_impl, "_predict", fake_predict)
    monkeypatch.setattr(
        predict_impl,
        "_score",
        lambda output, value, *, num_recycles, return_confidence_details: scored_output,
    )
    expected_path = output_dir / "tiny" / "seed_101" / "predictions" / "tiny.cif"
    write_calls = []

    def fake_write(root, **kwargs):
        write_calls.append((root, kwargs))
        return [expected_path]

    monkeypatch.setattr(predict_impl, "_write", fake_write)

    argv = [
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
    predict_impl.main(argv)

    model_features = calls[0][0]
    assert model_features is not features
    assert "ref_element" not in model_features
    assert "ref_atom_name_chars" not in model_features
    assert "_foldjax_compact_ref_atom_categories" in model_features
    assert write_calls[0][1]["features"] is features
    assert features["ref_element"].dtype == np.int64
    assert features["ref_atom_name_chars"].dtype == np.int64
    # Equality, not identity: the shipped bfloat16 trunk casts the weights on
    # the way in, so `cast_trunk_params` hands the model a rebuilt tree. What
    # this pins is that the loaded parameters are what reach the model, which
    # an identity check only expressed by accident while float32 shipped.
    assert calls[0][1] == params
    assert calls[0][2]["seed"] == 101
    assert calls[0][2]["num_samples"] == 1
    assert calls[0][2]["num_steps"] == 2
    assert calls[0][2]["num_recycles"] == 3
    assert calls[0][2]["return_confidence_details"] is False
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

    calls.clear()
    monkeypatch.setattr(
        predict_impl,
        "_load_prepared_params",
        lambda _path, _dtype: pytest.fail(
            "injected params must bypass checkpoint loading"
        ),
    )
    predict_impl.main(
        argv,
        _prepared_params_loader=lambda _path, _dtype, _cacheable: params,
    )
    assert len(calls) == 1
    assert calls[0][1] is params

    calls.clear()
    predict_impl.main(
        [*argv, "--include-raw"],
        _prepared_params_loader=lambda _path, _dtype, _cacheable: params,
    )
    assert len(calls) == 1
    assert calls[0][2]["return_confidence_details"] is True

    def poisoned_loader(_path, _dtype, _cacheable):
        raise PredictionError("prepared weight session is poisoned")

    with pytest.raises(PredictionError, match="session is poisoned"):
        predict_impl.main(argv, _prepared_params_loader=poisoned_loader)


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
