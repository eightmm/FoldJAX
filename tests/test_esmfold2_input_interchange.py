import json
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bench.esmfold2_tape import (
    _replay,
    _sha256,
    load_native_input_control,
    predict_with_native_inputs,
    predict_with_native_shim,
    validate_saved_input_control,
)


@pytest.mark.parametrize("count", [0, 1, 2, "error"])
def test_dynamic_input_control_restores_and_checks_consumption(count):
    def original():
        return 0

    module = SimpleNamespace(inputs_embedding=original)

    def predict():
        if count == "error":
            raise RuntimeError("failed")
        return sum(module.inputs_embedding() for _ in range(count))

    if count == 1:
        run = jax.jit(lambda x: predict_with_native_inputs(predict, module, x))
        assert float(run(jnp.array(3.0))) == 3
        assert float(run(jnp.array(9.0))) == 9
    else:
        with pytest.raises(RuntimeError if count == "error" else ValueError):
            predict_with_native_inputs(predict, module, jnp.array(3.0))
    assert module.inputs_embedding is original


def test_input_control_rejects_unmatched_outer_arms_before_loading():
    with pytest.raises(ValueError, match="requires --native-lm"):
        _replay(SimpleNamespace(native_input_embedding="not-read", native_lm=False))


def test_nested_shim_and_input_controls_keep_both_operands_dynamic():
    def original():
        raise AssertionError("must not compute the substituted representation")

    module = SimpleNamespace(inputs_embedding=original, language_model_pair=original)

    def actual_predict():
        return module.inputs_embedding() + 2 * module.language_model_pair()

    def with_shim(*, diagnostic_shim_pair):
        return predict_with_native_shim(actual_predict, module, diagnostic_shim_pair)

    run = jax.jit(
        lambda x, pair: predict_with_native_inputs(
            with_shim, module, x, diagnostic_shim_pair=pair
        )
    )
    assert float(run(jnp.array(3.0), jnp.array(5.0))) == 13
    assert float(run(jnp.array(7.0), jnp.array(11.0))) == 29
    assert module.inputs_embedding is original
    assert module.language_model_pair is original


@pytest.mark.parametrize("change", [None, "dtype", "shape", "nan", "hash", "binding"])
def test_native_input_control_checks_original_reference(tmp_path, change):
    reference, weights = tmp_path / "reference", tmp_path / "weights"
    reference.mkdir()
    weights.mkdir()
    paths = [
        reference / "metadata.json",
        weights / "config.json",
        weights / "model.safetensors",
    ]
    for path in paths:
        path.write_bytes(b"fixture")
    value = np.ones((1, 2, 3), np.float32)
    if change == "dtype":
        value = value.astype(np.float64)
    if change == "shape":
        value = value[:, :1]
    if change == "nan":
        value.flat[0] = np.nan
    archive = tmp_path / "native.npz"
    np.savez(archive, baseline=value)
    report = {
        "archive_sha256": _sha256(archive),
        "bindings": {str(p.resolve()): _sha256(p) for p in paths},
    }
    if change == "hash":
        report["archive_sha256"] = "0" * 64
    if change == "binding":
        report["bindings"].pop(str(paths[0].resolve()))
    (tmp_path / "report.json").write_text(json.dumps(report))
    if change:
        with pytest.raises(ValueError):
            load_native_input_control(tmp_path, reference, weights, (1, 2, 3))
    else:
        actual, _ = load_native_input_control(tmp_path, reference, weights, (1, 2, 3))
        np.testing.assert_array_equal(actual, value)


@pytest.mark.parametrize("change", [None, "path", "shape", "hash", "keys"])
def test_saved_control_is_not_trusted_from_metadata_alone(tmp_path, change):
    path = tmp_path / "native_input_control.npz"
    np.savez(
        path,
        **{
            ("wrong" if change == "keys" else "embedding"): np.zeros(
                (1, 2, 3), np.float32
            )
        },
    )
    control = {
        "filename": path.name,
        "sha256": _sha256(path),
        "dtype": "float32",
        "shape": [1, 2, 3],
        "full_model_admission": None,
    }
    if change == "path":
        control["filename"] = "../native_input_control.npz"
    if change == "shape":
        control["shape"] = [1, 1, 3]
    if change == "hash":
        control["sha256"] = "0" * 64
    if change:
        with pytest.raises(ValueError):
            validate_saved_input_control(tmp_path, control, (1, 2, 3))
    else:
        validate_saved_input_control(tmp_path, control, (1, 2, 3))


@pytest.mark.parametrize("change", [None, "tape", "dtype", "shape", "hash"])
def test_observed_native_input_checks_bridge(tmp_path, change):
    native, reference, weights = [tmp_path / k for k in ("native", "ref", "weights")]
    for p in (native, reference, weights):
        p.mkdir()
    (weights / "model.safetensors").write_bytes(b"fixture")
    value = np.ones((1, 2, 3), np.float32)
    np.savez(native / "injection.npz", msa_input_embedding=value)
    document = {
        "input_sha256": "input",
        "tape_sha256": "tape",
        "config": {},
        "binding": {
            "checkpoint": {"model.safetensors": _sha256(weights / "model.safetensors")},
            "source": {"native": {"source.py": "digest"}},
        },
        "output_schema": {"schema_version": 1, "arrays": {}},
        "injection_schema": {
            "filename": "injection.npz",
            "sha256": _sha256(native / "injection.npz"),
            "dtypes": {"msa_input_embedding": "torch.float32"},
        },
    }
    for name in ("features.npz", "tape.npz", "upstream_lm.npz"):
        for root in (native, reference):
            (root / name).write_bytes(b"same fixture")
    (reference / "metadata.json").write_text(json.dumps(document))
    if change == "tape":
        (native / "tape.npz").write_bytes(b"different")
    elif change == "dtype":
        document["injection_schema"]["dtypes"]["msa_input_embedding"] = "torch.bfloat16"
    elif change == "hash":
        document["injection_schema"]["sha256"] = "bad"
    (native / "metadata.json").write_text(json.dumps(document))
    if change:
        with pytest.raises(ValueError):
            load_native_input_control(
                native,
                reference,
                weights,
                (1, 1, 3) if change == "shape" else value.shape,
            )
    else:
        actual, bindings = load_native_input_control(
            native, reference, weights, value.shape
        )
        np.testing.assert_array_equal(actual, value)
        assert str((native / "metadata.json").resolve()) in bindings
