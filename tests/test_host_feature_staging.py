"""Concrete feature transfers preserve graph inputs without identity JITs."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.backends.alphafold3 import _stage_model_features
from foldjax.models.boltz2.api import _graph_features
from foldjax.models.opendde import postprocess
from tests.models.cp_probe_env import inherited_environment


@pytest.fixture
def stage_calls(monkeypatch):
    from jax._src import dispatch

    calls = []
    original = dispatch.apply_primitive

    def apply(primitive, *args, **kwargs):
        if primitive.name == "stage":
            calls.append(primitive.name)
        return original(primitive, *args, **kwargs)

    monkeypatch.setattr(dispatch, "apply_primitive", apply)
    return calls


def _host_features():
    # Noncontiguous leaves and strong scalar arrays exercise transfer semantics
    # that differ from Python scalar conversion and from numerical casts.
    return {
        "mask": np.asarray([True, False, True, False])[::2],
        "int8": np.arange(12, dtype=np.int8).reshape(3, 4)[:, ::2],
        "int32": np.arange(3, dtype=np.int32),
        "int64": np.arange(3, dtype=np.int64),
        "float32": np.asarray([0.0, -0.0, np.nan], dtype=np.float32),
        "float64": np.asarray([0.0, -0.0, np.nan], dtype=np.float64),
        "scalar": np.asarray(1, dtype=np.int32),
    }


def _assert_array_contract(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert actual.weak_type == expected.weak_type
    assert actual.committed == expected.committed
    assert actual.devices() == expected.devices()
    assert np.asarray(actual).tobytes() == np.asarray(expected).tobytes()


@pytest.mark.parametrize("x64", [False, True])
@pytest.mark.parametrize("model", ["alphafold3", "boltz2"])
def test_numeric_feature_transfer_keeps_the_same_lowering_without_stage(
    stage_calls, model, x64
):
    features = _host_features()
    device = jax.devices("cpu")[0]
    graph = jax.jit(lambda features: features)
    with jax.enable_x64(x64), jax.default_device(device):
        expected = {key: jnp.asarray(value) for key, value in features.items()}
        if model == "alphafold3":
            expected = jax.device_put(expected, device)
        assert stage_calls, "the reference path must exercise concrete staging"
        old_hlo = graph.lower(expected).as_text()
        stage_calls.clear()
        if model == "alphafold3":
            actual = _stage_model_features(features, device=device)
        else:
            actual = _graph_features(features, place=True)
        for key, value in expected.items():
            _assert_array_contract(actual[key], value)
        assert graph.lower(actual).as_text() == old_hlo
        assert not stage_calls
    graph.clear_cache()


def test_af3_staging_preserves_the_native_metadata_filter_boundary(stage_calls):
    # The carried runner remains responsible for dropping non-model metadata;
    # the adapter must not expand its dtype allowlist or modify those objects.
    metadata = np.asarray(["A", "B"], dtype=object)
    unsupported = np.asarray([1], dtype=np.int16)
    features = {"chain_name": metadata, "unsupported": unsupported, "label": "A"}
    actual = _stage_model_features(features)
    assert all(actual[key] is value for key, value in features.items())
    assert not stage_calls


def test_boltz_host_first_graph_keeps_unused_features_unplaced(stage_calls):
    features = _host_features()
    actual = _graph_features(features, place=False)
    assert all(actual[key] is value for key, value in features.items())
    assert not stage_calls


@pytest.mark.parametrize("x64", [False, True])
def test_opendde_host_summary_scores_preserve_array_contract_without_stage(
    monkeypatch, stage_calls, x64
):
    output = {
        "summary_ranking_score": np.asarray([0.5], dtype=np.float32),
        "chain_pair_iptm": np.arange(8, dtype=np.float64).reshape(2, 4)[:, ::2],
        "num_recycles": np.asarray(3, dtype=np.int64),
        "has_clash": np.asarray([False, True]),
        "ignored": np.asarray(["host metadata"], dtype=object),
    }
    monkeypatch.setattr(
        postprocess, "_shape_complementarity_scores", lambda *a, **k: {}
    )
    device = jax.devices("cpu")[0]
    with jax.enable_x64(x64), jax.default_device(device):
        expected = {
            key: jnp.asarray(value)
            for key, value in output.items()
            if key in postprocess._OPENDDE_SCORE_KEYS
        }
        assert stage_calls
        stage_calls.clear()
        actual = postprocess.opendde_confidence_scores(
            output, {"asym_id": np.asarray([0, 0, 1, 1])}, num_recycles=3
        )
        assert actual.keys() == expected.keys()
        for key, value in expected.items():
            _assert_array_contract(actual[key], value)
        assert not stage_calls


def test_transfer_honors_selected_device_and_non_numpy_weak_types():
    script = textwrap.dedent(
        """
        import jax
        import jax.numpy as jnp
        import numpy as np
        from jax._src import dispatch
        from foldjax.backends.alphafold3 import _stage_model_features
        from foldjax.models.boltz2.api import _graph_features
        from foldjax.models.opendde import postprocess

        devices = jax.devices("cpu")
        assert len(devices) == 2, devices
        host = np.arange(3, dtype=np.int64)
        with jax.default_device(devices[1]):
            weak = jnp.asarray(0.5)
            reference = jnp.asarray(host)
            af3_reference = jax.device_put(reference, devices[1])
            original = dispatch.apply_primitive
            def no_stage(primitive, *args, **kwargs):
                assert primitive.name != "stage", primitive.name
                return original(primitive, *args, **kwargs)
            dispatch.apply_primitive = no_stage
            af3 = _stage_model_features({"x": host}, device=devices[1])["x"]
            boltz = _graph_features({"x": host, "weak": weak}, place=True)
            postprocess._shape_complementarity_scores = lambda *a, **k: {}
            scores = postprocess.opendde_confidence_scores(
                {"summary_ranking_score": weak, "num_recycles": host},
                {"asym_id": host}, num_recycles=1
            )
        for new, old in [(af3, af3_reference), (boltz["x"], reference),
                         (scores["num_recycles"], reference)]:
            assert new.devices() == old.devices() == {devices[1]}
            assert new.dtype == old.dtype
            assert new.weak_type == old.weak_type
            assert new.committed == old.committed
            np.testing.assert_array_equal(new, old)
        assert boltz["weak"] is weak
        assert scores["summary_ranking_score"] is weak
        assert weak.weak_type
        print("SELECTED_HOST_FEATURE_DEVICE_OK")
        """
    )
    environment = inherited_environment()
    environment.update(
        JAX_PLATFORMS="cpu", XLA_FLAGS="--xla_force_host_platform_device_count=2"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout
    assert "SELECTED_HOST_FEATURE_DEVICE_OK" in completed.stdout
