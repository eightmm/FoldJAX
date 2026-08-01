from __future__ import annotations

import json

import numpy as np
import pytest

from foldjax.models.chai.bridge.component_io import (
    NATIVE_MANIFEST_KEY,
    convert_component_to_native,
    load_native_component_state_dict,
    save_native_component_state_dict,
)


def test_native_component_round_trip_without_torch(tmp_path) -> None:
    state = {
        "layer.weight": np.arange(6, dtype=np.float32).reshape(2, 3),
        "embedding.offsets": np.array([0, 3, 8], dtype=np.int64),
        "counter": np.array(7, dtype=np.int32),
    }
    path = tmp_path / "token_embedder.chai-jax.npz"

    save_native_component_state_dict(state, path, component="token_embedder.pt")
    restored = load_native_component_state_dict(
        path, expected_component="token_embedder.pt"
    )

    assert restored.keys() == state.keys()
    for name, expected in state.items():
        np.testing.assert_array_equal(restored[name], expected)
        assert restored[name].dtype == expected.dtype


def test_native_component_rejects_wrong_component(tmp_path) -> None:
    path = tmp_path / "component.npz"
    save_native_component_state_dict(
        {"weight": np.ones((2,), dtype=np.float32)},
        path,
        component="trunk.pt",
    )

    with pytest.raises(ValueError, match="component mismatch"):
        load_native_component_state_dict(
            path, expected_component="confidence_head.pt"
        )


def test_native_component_rejects_modified_tensor(tmp_path) -> None:
    path = tmp_path / "component.npz"
    save_native_component_state_dict(
        {"weight": np.ones((2,), dtype=np.float32)},
        path,
        component="trunk.pt",
    )
    with np.load(path, allow_pickle=False) as archive:
        manifest = archive[NATIVE_MANIFEST_KEY].copy()
    np.savez(
        path,
        **{
            NATIVE_MANIFEST_KEY: manifest,
            "weight": np.zeros(2, dtype=np.float32),
        },
    )

    with pytest.raises(ValueError, match="checksum mismatch"):
        load_native_component_state_dict(path)


def test_native_component_rejects_unknown_format(tmp_path) -> None:
    manifest = {
        "format": "chai-jax-component",
        "version": 999,
        "component": "trunk.pt",
        "tensors": {},
    }
    path = tmp_path / "future.npz"
    np.savez(
        path,
        **{
            NATIVE_MANIFEST_KEY: np.frombuffer(
                json.dumps(manifest).encode("utf-8"), dtype=np.uint8
            )
        },
    )

    with pytest.raises(ValueError, match="unsupported native component version"):
        load_native_component_state_dict(path)


def test_convert_component_uses_torch_only_for_export(tmp_path, monkeypatch) -> None:
    source = tmp_path / "token_embedder.pt"
    source.touch()
    destination = tmp_path / "token_embedder.npz"
    expected = {"weight": np.arange(3, dtype=np.float32)}
    monkeypatch.setattr(
        "foldjax.models.chai.bridge.component_io.load_component_state_dict",
        lambda path: expected,
    )

    count = convert_component_to_native(source, destination)

    assert count == 1
    restored = load_native_component_state_dict(
        destination, expected_component="token_embedder.pt"
    )
    np.testing.assert_array_equal(restored["weight"], expected["weight"])
