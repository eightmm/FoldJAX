from __future__ import annotations

import numpy as np
import pytest

from foldjax import torch_archive
from foldjax.models.opendde.bridge import export_weights as export_impl
from foldjax.models.opendde.bridge import weights_io as weights_impl


def test_native_weights_roundtrip_without_torch(tmp_path) -> None:
    path = tmp_path / "opendde.jax"
    params = {
        "weight": np.arange(6, dtype=np.float32).reshape(2, 3),
        "flag": True,
    }

    weights_impl.save_native_weights(path, params, compress=False)
    actual = weights_impl.load_native_weights(path)

    np.testing.assert_array_equal(actual["weight"], params["weight"])
    assert actual["flag"] is True
    assert "torch" not in weights_impl.__dict__


def test_torch_checkpoint_reader_is_lazy_safe_and_maps_opendde(
    tmp_path, monkeypatch
) -> None:
    checkpoint_path = tmp_path / "opendde.pt"
    checkpoint_path.write_bytes(b"trusted fixture")
    loaded = {"model": {"module.weight": np.asarray([1.0], dtype=np.float32)}}
    calls = []

    def fake_load(path):
        calls.append(path)
        return loaded

    monkeypatch.setattr(torch_archive, "load", fake_load)
    sentinel = object()
    monkeypatch.setattr(
        weights_impl,
        "map_opendde_inference_state_dict",
        lambda state: sentinel if set(state) == {"weight"} else None,
    )

    actual = weights_impl.load_torch_checkpoint(checkpoint_path)

    assert actual is sentinel
    assert calls == [checkpoint_path]


def test_torch_checkpoint_reader_rejects_missing_file(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="missing checkpoint"):
        weights_impl.load_torch_checkpoint(tmp_path / "missing.pt")


def test_export_cli_writes_native_weights(tmp_path, monkeypatch, capsys) -> None:
    checkpoint = tmp_path / "opendde.pt"
    output = tmp_path / "opendde.jax"
    checkpoint.write_bytes(b"trusted fixture")
    params = object()
    calls = []
    monkeypatch.setattr(export_impl, "load_torch_checkpoint", lambda path: params)
    monkeypatch.setattr(
        export_impl,
        "save_native_weights",
        lambda path, value, *, compress: calls.append((path, value, compress)),
    )

    export_impl.main(
        [
            "--checkpoint",
            str(checkpoint),
            "--out",
            str(output),
            "--no-compress",
        ]
    )

    assert calls == [(output, params, False)]
    assert f"wrote native weights: {output}" in capsys.readouterr().out
