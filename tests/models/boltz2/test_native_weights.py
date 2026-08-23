"""Focused tests for Boltz-2's torch-free native weight loader."""

from __future__ import annotations

import json
import weakref
from pathlib import Path
from typing import Any

import numpy as np

from foldjax.models import _stacking
from foldjax.models._stacking import StackedLayers
from foldjax.models.boltz2.bridge import native


def _layer(value: float) -> dict[str, Any]:
    return {
        "weight": np.asarray([value, value + 1], dtype=np.float32),
        "bias": np.asarray(value - 1, dtype=np.float32),
        "num_heads": 2,
    }


def test_load_params_prestacks_without_changing_values(tmp_path: Path) -> None:
    source = {
        "layers": [_layer(1.0), _layer(3.0)],
        "projection": np.arange(6, dtype=np.float32).reshape(2, 3),
        "optional": None,
    }
    saved = native.save_params(source, tmp_path / "weights")

    loaded = native.load_params(saved["weights_path"])

    assert isinstance(loaded["layers"], StackedLayers)
    assert loaded["optional"] is None
    np.testing.assert_array_equal(loaded["projection"], source["projection"])
    for actual, expected in zip(loaded["layers"], source["layers"], strict=True):
        assert actual["num_heads"] == expected["num_heads"]
        np.testing.assert_array_equal(actual["weight"], expected["weight"])
        np.testing.assert_array_equal(actual["bias"], expected["bias"])


def test_load_params_releases_flat_mapping_before_prestack(
    tmp_path: Path, monkeypatch: Any
) -> None:
    weights = tmp_path / "weights.safetensors"
    weights.touch()
    weights.with_suffix(".safetensors.json").write_text(
        json.dumps({"scalars": {}}), encoding="utf-8"
    )

    class TrackedArrays(dict[str, np.ndarray]):
        pass

    mapping_ref: weakref.ReferenceType[TrackedArrays] | None = None

    def fake_load_file(_path: str) -> TrackedArrays:
        nonlocal mapping_ref
        arrays = TrackedArrays(
            {
                "d:layers/i:0/d:weight": np.asarray([1.0, 2.0]),
                "d:layers/i:0/d:bias": np.asarray([0.0]),
                "d:layers/i:1/d:weight": np.asarray([3.0, 4.0]),
                "d:layers/i:1/d:bias": np.asarray([1.0]),
            }
        )
        mapping_ref = weakref.ref(arrays)
        return arrays

    original_prestack = _stacking.prestack_layer_lists
    observed: list[bool] = []

    def spy_prestack(params: Any) -> Any:
        assert mapping_ref is not None
        observed.append(mapping_ref() is None)
        return original_prestack(params)

    monkeypatch.setattr("safetensors.numpy.load_file", fake_load_file)
    monkeypatch.setattr(_stacking, "prestack_layer_lists", spy_prestack)

    loaded = native.load_params(weights)

    assert observed and all(observed)
    np.testing.assert_array_equal(loaded["layers"][0]["weight"], [1.0, 2.0])
    np.testing.assert_array_equal(loaded["layers"][1]["weight"], [3.0, 4.0])


def test_load_params_transfers_without_jnp_staging(
    tmp_path: Path, monkeypatch: Any
) -> None:
    source = {
        "float64": np.asarray([1.25, -2.5], dtype=np.float64),
        "float32": np.arange(6, dtype=np.float32).reshape(2, 3),
        "int64": np.asarray([1, -2], dtype=np.int64),
        "int32": np.asarray([3, -4], dtype=np.int32),
        "bool": np.asarray([True, False], dtype=bool),
    }
    saved = native.save_params(source, tmp_path / "weights")
    original_device_put = native.jax.device_put
    transferred: list[np.ndarray] = []

    def spy_device_put(value: np.ndarray) -> Any:
        transferred.append(value)
        return original_device_put(value)

    def forbid_asarray(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise AssertionError("parameter transfer used a staged jnp.asarray")

    monkeypatch.setattr(native.jax, "device_put", spy_device_put)
    monkeypatch.setattr(native.jnp, "asarray", forbid_asarray)

    loaded = native.load_params(saved["weights_path"])

    assert len(transferred) == len(source)
    assert all(isinstance(value, np.ndarray) for value in transferred)
    for name, expected in source.items():
        canonical_dtype = native.jax.dtypes.canonicalize_dtype(expected.dtype)
        np.testing.assert_array_equal(
            np.asarray(loaded[name]),
            expected.astype(canonical_dtype),
        )
        assert loaded[name].dtype == canonical_dtype
