import json

import numpy as np
import pytest

from bench.esmfold2_lm_shim_probe import boundary_slice, load_reference
from bench.esmfold2_tape import _save_lm


def test_native_lm_reference_preserves_full_float32_values(tmp_path):
    value = np.arange(160, dtype=np.float32).reshape(1, 4, 5, 8)
    schema = _save_lm(tmp_path, "upstream", value, native=False)
    (tmp_path / "metadata.json").write_text(json.dumps({"lm_schema": schema}))
    _, actual = load_reference(tmp_path)
    np.testing.assert_array_equal(actual, value)
    np.testing.assert_array_equal(boundary_slice(actual), value[:, :3, :3, :])


@pytest.mark.parametrize("bad", ["hash", "dtype", "shape", "path"])
def test_native_lm_reference_rejects_changed_contract(tmp_path, bad):
    value = np.ones((1, 4, 5, 8), np.float32)
    schema = _save_lm(tmp_path, "upstream", value, native=False)
    if bad == "hash":
        schema["sha256"] = "changed"
    elif bad == "dtype":
        schema["original_dtype"] = "bfloat16"
    elif bad == "shape":
        schema["shape"] = [1, 4, 5, 1]
    else:
        schema["filename"] = "../unexpected.npz"
    (tmp_path / "metadata.json").write_text(json.dumps({"lm_schema": schema}))
    with pytest.raises(ValueError):
        load_reference(tmp_path)
