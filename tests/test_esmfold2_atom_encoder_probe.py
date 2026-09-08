import json

import numpy as np
import pytest

from bench.esmfold2_atom_encoder_probe import difference, validate_boundary
from bench.esmfold2_input_boundary import INPUT_NAMES
from bench.esmfold2_tape import _sha256


def boundary(tmp_path):
    arrays = {name: np.zeros((1, 2), np.float32) for name in INPUT_NAMES}
    archive = tmp_path / "inputs.npz"
    np.savez(archive, **arrays)
    report = {
        "engine": "native",
        "archive_sha256": _sha256(archive),
        "fields": {
            name: {"shape": [1, 2], "dtype": "float32", "storage_dtype": "float32"}
            for name in arrays
        },
    }
    (tmp_path / "report.json").write_text(json.dumps(report))
    return arrays, report


def test_validate_boundary(tmp_path):
    arrays, _ = boundary(tmp_path)
    _, actual = validate_boundary(tmp_path)
    for key in arrays:
        np.testing.assert_array_equal(actual[key], arrays[key])


@pytest.mark.parametrize("change", ["engine", "hash", "dtype", "shape", "keys", "nan"])
def test_reject_invalid_boundary(tmp_path, change):
    arrays, report = boundary(tmp_path)
    if change == "engine":
        report["engine"] = "jax"
    elif change == "hash":
        report["archive_sha256"] = "0" * 64
    elif change == "dtype":
        report["fields"]["profile"]["dtype"] = "bfloat16"
    elif change == "shape":
        report["fields"]["profile"]["shape"] = [2, 1]
    elif change == "keys":
        arrays.pop("profile")
    else:
        arrays["profile"][0, 0] = np.nan
    if change in ("keys", "nan"):
        np.savez(tmp_path / "inputs.npz", **arrays)
        report["archive_sha256"] = _sha256(tmp_path / "inputs.npz")
    (tmp_path / "report.json").write_text(json.dumps(report))
    with pytest.raises(ValueError):
        validate_boundary(tmp_path)


def test_difference():
    a = np.array([0, 1], np.float32)
    b = np.array([0, 3], np.float32)
    result = difference(a, b)
    assert result["unequal"] == 1
    assert result["max_abs"] == 2
    assert result["rmse"] == pytest.approx(np.sqrt(2))
    assert difference(a, a)["unequal"] == 0
    with pytest.raises(ValueError, match="schema"):
        difference(a, b.astype(np.float64))
