from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bench import esmfold2_lm_shim_candidate, esmfold2_tape, esmfold2_tape_report


@pytest.mark.parametrize("change", [None, "shape", "dtype", "nan", "precision"])
def test_shim_control_validates_pair_after_native_provenance(
    tmp_path, monkeypatch, change
):
    pair = np.ones((1, 2, 2, 4), np.float32)
    if change == "shape":
        pair = pair[:, :1]
    elif change == "dtype":
        pair = pair.astype(np.float64)
    elif change == "nan":
        pair.flat[0] = np.nan
    esmfold2_tape._save_npz(tmp_path / "native.npz", {"pair": pair})
    (tmp_path / "report.json").write_text("{}")
    calls = []

    def validate(*args):
        calls.append(args)
        return {"precision_control": change == "precision"}

    monkeypatch.setattr(esmfold2_lm_shim_candidate, "validate_native", validate)
    args = (tmp_path, {}, tmp_path, tmp_path, (1, 2, 2, 4))
    if change:
        with pytest.raises(ValueError):
            esmfold2_tape.load_native_shim_control(*args)
    else:
        result, bindings = esmfold2_tape.load_native_shim_control(*args)
        np.testing.assert_array_equal(result, pair)
        assert bindings[
            str((tmp_path / "native.npz").resolve())
        ] == esmfold2_tape._sha256(tmp_path / "native.npz")
    assert len(calls) == 1


def test_shim_control_requires_provenance_before_array_loading(tmp_path, monkeypatch):
    def reject(*args):
        raise ValueError("native artifacts differ")

    monkeypatch.setattr(esmfold2_lm_shim_candidate, "validate_native", reject)
    with pytest.raises(ValueError, match="artifacts differ"):
        esmfold2_tape.load_native_shim_control(
            tmp_path, {}, tmp_path, tmp_path, (1, 2, 2, 4)
        )


@pytest.mark.parametrize("calls", [0, 1, 2, "exception"])
def test_shim_control_scoped_restoration_and_exact_consumption(calls):
    def original():
        return jnp.array(-1.0)

    module = SimpleNamespace(language_model_pair=original)

    def predict():
        if calls == "exception":
            raise RuntimeError("probe failure")
        result = 0
        for _ in range(calls):
            result += module.language_model_pair()
        return result

    if calls == 1:
        run = jax.jit(
            lambda pair: esmfold2_tape.predict_with_native_shim(predict, module, pair)
        )
        np.testing.assert_array_equal(run(jnp.array(3.0)), 3)
        np.testing.assert_array_equal(run(jnp.array(9.0)), 9)
    else:
        with pytest.raises(RuntimeError if calls == "exception" else ValueError):
            esmfold2_tape.predict_with_native_shim(predict, module, jnp.array(3.0))
    assert module.language_model_pair is original


def test_shim_control_rejects_independent_lm_before_loading():
    args = SimpleNamespace(native_shim="not-loaded", native_lm=False)
    with pytest.raises(ValueError, match="requires --native-lm"):
        esmfold2_tape._replay(args)


def test_shim_metadata_rejects_saved_pair_tampering(tmp_path):
    path = tmp_path / "pair.npz"
    esmfold2_tape._save_npz(path, {"pair": np.ones((1, 2, 2, 4), np.float32)})
    args = SimpleNamespace(
        output_dir=tmp_path,
        native_shim_control={
            "filename": path.name,
            "sha256": esmfold2_tape._sha256(path),
        },
    )
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="saved native shim artifact changed"):
        esmfold2_tape._write_metadata(args, {}, {})


@pytest.mark.parametrize(
    "bad", [None, "hash", "filename", "shape", "dtype", "nan", "fields", "arm"]
)
def test_report_validates_and_scopes_native_shim_artifact(tmp_path, bad):
    pair = np.ones((1, 2, 2, 4), np.float32)
    if bad == "nan":
        pair.flat[0] = np.nan
    if bad == "dtype":
        pair = pair.astype(np.float64)
    fields = {"wrong" if bad == "fields" else "pair": pair}
    path = tmp_path / "pair.npz"
    esmfold2_tape._save_npz(path, fields)
    shim = {
        "filename": "../pair.npz" if bad == "filename" else "pair.npz",
        "sha256": "wrong" if bad == "hash" else esmfold2_tape._sha256(path),
        "shape": [1, 2, 2, 3] if bad == "shape" else list(pair.shape),
        "dtype": "float32",
    }
    kwargs = dict(downstream_only=bad != "arm", expected_shape=(1, 2, 2, 4))
    if bad:
        with pytest.raises(ValueError, match="native shim control"):
            esmfold2_tape_report.validate_shim_control(tmp_path, shim, **kwargs)
    else:
        esmfold2_tape_report.validate_shim_control(tmp_path, shim, **kwargs)
