import numpy as np
import pytest

from bench.openbind_core_replay import (
    candidate_trunk_arrays,
    compiler_environment,
    model_feature_batch,
    native_trunk_arrays,
    positive_count,
    replace_trunk_components,
)
from foldjax.models.openfold3.data.featurize import MODEL_FEATURES


def test_metadata_is_not_forwarded_and_required_features_are_not_dropped():
    features = {name: np.zeros(1) for name in MODEL_FEATURES}
    features["atom_array.0.annotation.atom_name"] = np.array(["CA"])
    batch = model_feature_batch(features)
    assert set(batch) == set(MODEL_FEATURES)
    assert all(batch[name] is features[name] for name in MODEL_FEATURES)
    assert "atom_array.0.annotation.atom_name" in features
    del features["token_mask"]
    with pytest.raises(ValueError, match="token_mask"):
        model_feature_batch(features)


def test_repeat_count_must_be_positive():
    import argparse

    assert positive_count("3") == 3
    for value in ("0", "-1"):
        with pytest.raises(argparse.ArgumentTypeError, match="positive"):
            positive_count(value)


@pytest.mark.parametrize("extra", [
    ["--backend", "cueq"],
    ["--backend", "native-private"],
    ["--backend", "xla", "--inject-native-trunk"],
    ["--backend", "xla", "--inject-candidate-trunk", "unused"],
])
def test_private_control_rejects_other_backend_or_trunk_injection(extra, capsys):
    from bench.openbind_core_replay import main

    args = ["--capture", "unused", "--checkpoint", "unused",
            "--source-root", "unused", "--out-dir", "unused",
            "--private-pair-operators", *extra]
    with pytest.raises(SystemExit) as error:
        main(args)
    assert error.value.code == 2
    assert "requires XLA and no trunk injection" in capsys.readouterr().err


def test_compiler_environment_is_allowlisted_and_preserves_absence(monkeypatch):
    monkeypatch.setenv("XLA_FLAGS", "--xla_gpu_autotune_level=0")
    monkeypatch.delenv("JAX_DEFAULT_MATMUL_PRECISION", raising=False)
    monkeypatch.setenv("PRIVATE_TEST_TOKEN", "must-not-be-recorded")
    recorded = compiler_environment()
    assert recorded["XLA_FLAGS"] == "--xla_gpu_autotune_level=0"
    assert recorded["JAX_DEFAULT_MATMUL_PRECISION"] is None
    assert "PRIVATE_TEST_TOKEN" not in recorded


def test_native_trunk_injection_requires_bound_valid_arrays(tmp_path):
    import json

    from bench.boltz_historical_replay import digest

    path = tmp_path / "trunk-00.npz"
    arrays = [np.zeros(s, np.float32) for s in
              ((1, 2, 449), (1, 2, 384), (1, 2, 2, 128))]
    np.savez(path, **{str(i): a for i, a in enumerate(arrays)})
    (tmp_path / "trace.json").write_text(json.dumps({"trunk_arrays": [{
        "file": path.name,
        "sha256": {"sha256": digest(path), "bytes": path.stat().st_size},
    }]}))
    loaded, _ = native_trunk_arrays(tmp_path, 2)
    assert all(np.array_equal(a, b) for a, b in zip(arrays, loaded, strict=True))
    with pytest.raises(ValueError, match="shape"):
        native_trunk_arrays(tmp_path, 3)
    np.savez(path, **{"0": np.zeros(1)})
    with pytest.raises(ValueError, match="identity"):
        native_trunk_arrays(tmp_path, 2)


def test_candidate_trunk_requires_matching_noninjected_provenance(tmp_path):
    import json

    from bench.boltz_historical_replay import digest

    expected = {"input_sha256": "input", "tape_sha256": "tape"}
    preflight = tmp_path / "preflight.json"
    preflight.write_text(json.dumps(expected))
    path = tmp_path / "prediction.npz"
    np.savez(path, single_inputs=np.zeros((1, 2, 449), np.float32),
             single=np.zeros((1, 2, 384), np.float32),
             pair=np.zeros((1, 2, 2, 128), np.float32))
    (tmp_path / "finished.json").write_text(json.dumps({
        "prediction_sha256": digest(path),
    }))
    arrays, identity = candidate_trunk_arrays(tmp_path, expected, 2)
    assert len(arrays) == 3 and identity["prediction_sha256"] == digest(path)
    with pytest.raises(ValueError, match="provenance"):
        candidate_trunk_arrays(tmp_path, {"input_sha256": "wrong"}, 2)
    preflight.write_text(json.dumps({**expected, "native_trunk_injection": {"x": 1}}))
    with pytest.raises(ValueError, match="non-injected"):
        candidate_trunk_arrays(tmp_path, expected, 2)
    preflight.write_text(json.dumps(expected))
    np.savez(path, single=np.zeros(1))
    with pytest.raises(ValueError, match="identity"):
        candidate_trunk_arrays(tmp_path, expected, 2)


def test_component_replacement_preserves_each_selected_array():
    candidate = tuple(object() for _ in range(3))
    native = tuple(object() for _ in range(3))
    assert replace_trunk_components(candidate, native, "single") == (
        native[0], native[1], candidate[2]
    )
    assert replace_trunk_components(candidate, native, "pair") == (
        candidate[0], candidate[1], native[2]
    )
    with pytest.raises(ValueError, match="unknown"):
        replace_trunk_components(candidate, native, "other")
