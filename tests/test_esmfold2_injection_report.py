import numpy as np
import pytest

from bench.esmfold2_injection_report import KEYS, compare_boundaries


def test_paired_outputs_use_capture_native_not_tape_reference(tmp_path, monkeypatch):
    from bench import esmfold2_injection_report as report

    native, candidate = tmp_path / "observed-native", tmp_path / "candidate"
    features = tmp_path / "features.npz"
    fixtures = {
        native / "upstream_coords.npz": {"coords": "native-coords"},
        candidate / "jax_coords.npz": {"coords": "candidate-coords"},
        native / "upstream_confidence.npz": {"score": "native-score"},
        candidate / "jax_confidence.npz": {"score": "candidate-score"},
        features: {"feature": "identity"},
    }
    monkeypatch.setattr(report, "_npz", fixtures.__getitem__)
    monkeypatch.setattr(report, "compare_coordinates", lambda *args: args)
    monkeypatch.setattr(report, "compare_arrays", lambda *args: args)
    result = report.compare_paired_outputs(native, candidate, features)
    assert result["coordinates"] == (
        "native-coords",
        "candidate-coords",
        fixtures[features],
    )
    assert result["confidence"] == (
        {"score": "native-score"},
        {"score": "candidate-score"},
    )


def fixture():
    return (
        {k: np.zeros((1, 2, 2, 3), np.float32) for k in KEYS},
        {k: "bfloat16" for k in KEYS},
    )


def test_boundary_comparison_retains_dtype_and_exactness():
    arrays, dtypes = fixture()
    result = compare_boundaries(arrays, arrays, dtypes, dtypes)
    assert set(result) == KEYS
    assert all(v["comparison"]["bitwise_equal"] for v in result.values())


def test_coda_requires_complete_symmetric_pair():
    arrays, dtypes = fixture()
    arrays.update(
        coda_input=arrays["trunk_input"].copy(),
        coda_output=arrays["trunk_input"].copy(),
    )
    dtypes.update(coda_input="bfloat16", coda_output="bfloat16")
    assert compare_boundaries(arrays, arrays, dtypes, dtypes)["coda_output"][
        "comparison"
    ]["bitwise_equal"]
    for key in ("coda_input", "coda_output"):
        missing = {k: v for k, v in arrays.items() if k != key}
        with pytest.raises(ValueError):
            compare_boundaries(arrays, missing, dtypes, dtypes)
        with pytest.raises(ValueError):
            compare_boundaries(missing, arrays, dtypes, dtypes)


def test_optional_msa_inputs_require_symmetric_complete_capture():
    arrays, dtypes = fixture()
    arrays.update(
        msa_input_pair=np.zeros((1, 2, 2, 3), np.float32),
        msa_input_embedding=np.zeros((1, 2, 3), np.float32),
    )
    dtypes.update(msa_input_pair="bfloat16", msa_input_embedding="bfloat16")
    result = compare_boundaries(arrays, arrays, dtypes, dtypes)
    assert len(result) == 7
    for key in ("msa_input_pair", "msa_input_embedding"):
        incomplete = {k: v for k, v in arrays.items() if k != key}
        with pytest.raises(ValueError):
            compare_boundaries(arrays, incomplete, dtypes, dtypes)
        with pytest.raises(ValueError):
            compare_boundaries(incomplete, arrays, dtypes, dtypes)


@pytest.mark.parametrize("key", ["trunk_output", "trunk_output_last"])
def test_optional_trunk_output_requires_both_arms(key):
    arrays, dtypes = fixture()
    extended = {**arrays, key: arrays["trunk_input"].copy()}
    extended_dtypes = {**dtypes, key: "bfloat16"}
    result = compare_boundaries(extended, extended, extended_dtypes, extended_dtypes)
    assert result[key]["comparison"]["bitwise_equal"]
    with pytest.raises(ValueError):
        compare_boundaries(extended, arrays, extended_dtypes, dtypes)
    with pytest.raises(ValueError):
        compare_boundaries(arrays, extended, dtypes, extended_dtypes)


@pytest.mark.parametrize("change", ["missing", "shape", "storage", "nan", "schema"])
def test_invalid_boundary_fails(change):
    a, da = fixture()
    b, db = fixture()
    key = "msa_output"
    if change == "missing":
        b.pop(key)
    elif change == "shape":
        b[key] = b[key][:, :1]
    elif change == "storage":
        b[key] = b[key].astype(np.float64)
    elif change == "nan":
        b[key].flat[0] = np.nan
    else:
        db.pop(key)
    with pytest.raises(ValueError):
        compare_boundaries(a, b, da, db)
