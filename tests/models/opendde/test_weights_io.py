from __future__ import annotations

import pickle
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax import torch_archive
from foldjax.models.opendde.bridge import export_weights as export_impl
from foldjax.models.opendde.bridge import weights_io as weights_impl
from foldjax.models.opendde.models.model import (
    OpenDDEInferenceParams,
    cast_trunk_params,
)


def _mixed_precision_params() -> OpenDDEInferenceParams:
    fields = {}
    for index, name in enumerate(OpenDDEInferenceParams._fields):
        fields[name] = {
            "float": np.asarray(
                [index + 0.1234567, -index - 0.7654321], dtype=np.float32
            ),
            "integer": np.asarray([index], dtype=np.int32),
        }
    return OpenDDEInferenceParams(**fields)


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


def test_prepared_native_weights_match_historical_device_cast_exactly(
    tmp_path,
) -> None:
    path = tmp_path / "opendde.jax"
    weights_impl.save_native_weights(
        path,
        _mixed_precision_params(),
        compress=False,
    )

    historical = cast_trunk_params(
        weights_impl.load_native_weights(path),
        jnp.bfloat16,
    )
    prepared = weights_impl._load_prepared_native_weights(
        path,
        jnp.bfloat16,
        _cast_batch_bytes=8,
    )

    assert jax.tree.structure(prepared) == jax.tree.structure(historical)
    for actual, expected in zip(
        jax.tree.leaves(prepared),
        jax.tree.leaves(historical),
        strict=True,
    ):
        if hasattr(expected, "dtype"):
            assert actual.dtype == expected.dtype
            np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
        else:
            assert actual == expected

    for name in weights_impl._TRUNK_FIELDS:
        assert getattr(prepared, name)["float"].dtype == jnp.bfloat16
    for name in ("diffusion", "distogram", "confidence"):
        assert getattr(prepared, name)["float"].dtype == jnp.float32


def test_prepared_fallback_matches_nonfinite_and_float64_device_cast_bits(
    tmp_path,
) -> None:
    path = tmp_path / "opendde.jax"
    params = _mixed_precision_params()
    # Two distinct NaN payloads, both infinities, and signed zero exercise the
    # values for which a host cast must not replace the historical JAX cast.
    params.input_embedder["float32_edges"] = np.asarray(
        [
            0x7FC00001,
            0x7FA12345,
            0x7F800000,
            0xFF800000,
            0x00000000,
            0x80000000,
        ],
        dtype=np.uint32,
    ).view(np.float32)
    # Finite float64 deliberately takes the device fallback too: with x64
    # disabled JAX first canonicalizes it to float32, so a direct host
    # float64-to-bfloat16 conversion could double-round differently.
    params.pairformer_output["finite_float64"] = np.asarray(
        [1.0000000596046448, -3.141592653589793],
        dtype=np.float64,
    )
    weights_impl.save_native_weights(path, params, compress=False)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        historical = cast_trunk_params(
            weights_impl.load_native_weights(path),
            jnp.bfloat16,
        )
        prepared = weights_impl._load_prepared_native_weights(
            path,
            jnp.bfloat16,
            _cast_batch_bytes=8,
        )

    for field, name in (
        ("input_embedder", "float32_edges"),
        ("pairformer_output", "finite_float64"),
    ):
        expected = np.asarray(getattr(historical, field)[name])
        actual = np.asarray(getattr(prepared, field)[name])
        np.testing.assert_array_equal(
            actual.view(np.uint16),
            expected.view(np.uint16),
        )

    edge = np.asarray(prepared.input_embedder["float32_edges"]).astype(np.float32)
    np.testing.assert_array_equal(
        np.isnan(edge), [True, True, False, False, False, False]
    )
    np.testing.assert_array_equal(
        np.isposinf(edge), [False, False, True, False, False, False]
    )
    np.testing.assert_array_equal(
        np.isneginf(edge), [False, False, False, True, False, False]
    )
    np.testing.assert_array_equal(np.signbit(edge[-2:]), [False, True])


def test_prepared_loader_matches_shared_cross_precision_container(
    tmp_path,
) -> None:
    path = tmp_path / "opendde.jax"
    params = _mixed_precision_params()
    shared = {
        "float": np.asarray([0.1234567, -0.7654321], dtype=np.float32),
        "integer": np.asarray([7], dtype=np.int32),
    }
    params = params._replace(input_embedder=shared, diffusion=shared)
    # Dump directly so pickle preserves the shared container alias.  The
    # ordinary serializer tree-maps its input and may rebuild containers first.
    with path.open("wb") as handle:
        pickle.dump(params, handle, protocol=pickle.HIGHEST_PROTOCOL)

    historical = cast_trunk_params(
        weights_impl.load_native_weights(path),
        jnp.bfloat16,
    )
    prepared = weights_impl._load_prepared_native_weights(path, jnp.bfloat16)

    assert jax.tree.structure(prepared) == jax.tree.structure(historical)
    for actual, expected in zip(
        jax.tree.leaves(prepared),
        jax.tree.leaves(historical),
        strict=True,
    ):
        if hasattr(expected, "dtype"):
            assert actual.dtype == expected.dtype
            np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
        else:
            assert actual == expected
    assert prepared.input_embedder["float"].dtype == jnp.bfloat16
    assert prepared.diffusion["float"].dtype == jnp.float32


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
