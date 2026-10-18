"""Production Boltz graph identity and realized chunks without model weights."""

from __future__ import annotations

import dataclasses
from contextlib import nullcontext
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax import execution, memory_policy
from foldjax.api import resolve_cache_dir
from foldjax.backends.boltz2 import Boltz2Backend
from foldjax.executable_cache_verify import (
    UnsupportedCacheVerificationError,
    lowered_cache_key,
)
from foldjax.models.boltz2 import api as native_api
from foldjax.models.boltz2.models.trunk_blocks.trunk import (
    resolve_long_sequence_chunks,
)
from tests.test_boltz2_session import _features, _request


@pytest.fixture(autouse=True)
def isolated_record():
    execution.clear_chunking_record()
    memory_policy.clear_record()
    yield
    execution.clear_chunking_record()
    memory_policy.clear_record()


def _install_runtime(monkeypatch, tmp_path, *, n_token=512, affinity=False):
    features = _features(affinity=affinity)
    features["token_pad_mask"] = np.ones((1, n_token), np.float32)
    features["mol_type"] = np.zeros((1, n_token), np.int32)
    features["affinity_token_mask"] = np.zeros((1, n_token), np.float32)
    if affinity:
        features["affinity_token_mask"][0, -1] = 1
    features["token_to_rep_atom"] = np.zeros((1, n_token, 3), np.int64)
    features["token_to_rep_atom"][0, :, 0] = 1
    features["atom_to_token"] = np.zeros((1, 3, n_token), np.int64)
    features["atom_to_token"][0, :, 0] = 1
    monkeypatch.setattr(
        native_api, "featurize", lambda **kwargs: (features, "job", tmp_path)
    )
    monkeypatch.setattr(
        "foldjax.models.boltz2.bridge.native.load_params",
        lambda _: {"trunk": {"weight": np.asarray([1.0], np.float32)}},
    )
    counts = {"traces": 0, "calls": 0}
    calls = []
    original_jit = native_api._boltz_jit

    def tiny_predict(params, feats, key, **kwargs):
        # The learned blocks are stubbed. The real API still chooses feature
        # storage, static kwargs, compiler policy, graph closure and owner.
        counts["traces"] += 1
        chunks = resolve_long_sequence_chunks(
            feats["token_pad_mask"].shape[-1],
            chunk_size=128,
            triangle_attention_chunk=None,
            triangle_attention_q_chunk=kwargs["triangle_attention_q_chunk"],
            token_attention_chunk=kwargs["token_attention_chunk"],
        )
        value = (
            params["trunk"]["weight"][0]
            + jax.random.uniform(key, ())
            + chunks["triangle_attention_chunk"] * 1e-6
        )
        samples = kwargs["multiplicity"]
        atoms = feats["atom_pad_mask"].shape[-1]
        tokens = feats["token_pad_mask"].shape[-1]
        return {
            "sample_atom_coords": jnp.full((samples, atoms, 3), value),
            "plddt": jnp.ones((samples, tokens)),
            "iptm": jnp.full((samples,), value),
        }

    class ObservedRunner:
        def __init__(self, function, **kwargs):
            self.runner = original_jit(function, **kwargs)

        def lower(self, *args):
            lowered = self.runner.lower(*args)
            calls.append(lowered)
            return lowered

        def __call__(self, *args):
            counts["calls"] += 1
            self.lower(*args)
            return self.runner(*args)

        def _cache_size(self):
            return self.runner._cache_size()

        def clear_cache(self):
            self.runner.clear_cache()

    monkeypatch.setattr(native_api, "_boltz_jit", ObservedRunner)
    monkeypatch.setattr(
        "foldjax.models.boltz2.models.predict.boltz2_predict", tiny_predict
    )
    return counts, calls


@pytest.mark.parametrize(
    ("n_token", "expected"),
    [
        (512, (128, 128, None, None)),
        (1536, (128, 128, None, 128)),
        (2048, (128, 32, None, 128)),
        (3072, (64, 16, 256, 64)),
    ],
)
def test_record_uses_native_token_rungs(n_token, expected):
    native_api._record_prediction_chunking(
        {"token_pad_mask": np.zeros((1, n_token), np.float32)},
        {
            "multiplicity": 5,
            "diffusion_chunk_size": None,
            "run_confidence": True,
            "confidence_sequentially": True,
        },
    )
    record = execution.recorded_chunking()["profiles"][0]
    assert record["model"] == "boltz2"
    assert record["n_token"] == n_token
    assert record["policy"] == "token_rungs"
    chunks = record["chunks"]
    assert tuple(
        chunks[key]
        for key in (
            "chunk_size",
            "triangle_attention_chunk",
            "triangle_attention_q_chunk",
            "token_attention_chunk",
        )
    ) == expected
    assert chunks["diffusion_chunk_size"] == 5
    assert chunks["confidence_sample_chunk_size"] == 1
    if n_token > 2048:
        assert chunks["confidence_chunk_size"] == 128
        assert chunks["confidence_triangle_attention_chunk"] == 32
        assert chunks["confidence_triangle_attention_q_chunk"] == 512


def test_record_preserves_explicit_dense_widths_and_diffusion_chunk():
    native_api._record_prediction_chunking(
        {"token_pad_mask": np.zeros((1, 3072), np.float32)},
        {
            "multiplicity": 8,
            "diffusion_chunk_size": 3,
            "token_attention_chunk": 0,
            "triangle_attention_q_chunk": 0,
        },
    )
    chunks = execution.recorded_chunking()["profiles"][0]["chunks"]
    assert chunks["diffusion_chunk_size"] == 3
    assert chunks["token_attention_chunk"] == 0
    assert chunks["triangle_attention_q_chunk"] == 0


def test_same_gpu_kind_ignores_ecc_budget_in_namespace(tmp_path, monkeypatch):
    from foldjax import _gpu
    from foldjax.cache import runtime_profile

    request = _request(tmp_path)
    backend = Boltz2Backend()
    profiles, namespaces, runtimes = [], [], []
    for gib in (22, 24):
        device = SimpleNamespace(
            platform="gpu",
            device_kind="NVIDIA GeForce RTX 4090",
            id=0,
            process_index=0,
            local_hardware_id=0,
            memory_stats=lambda: {"bytes_limit": gib * 2**30},
        )
        with monkeypatch.context() as patch:
            patch.setattr(_gpu, "default_device", lambda *args: device)
            patch.setattr(jax, "default_device", lambda _: nullcontext())
            patch.setattr(jax, "devices", lambda **kwargs: [device])
            patch.setattr(jax, "local_device_count", lambda **kwargs: 1)
            patch.setattr(jax, "process_count", lambda: 1)
            current = dataclasses.replace(
                request,
                options={**request.options, "memory_budget_gib": float(gib)},
            )
            profiles.append(backend.cache_profile(current))
            namespaces.append(resolve_cache_dir(current, backend))
            runtimes.append(runtime_profile())
    assert profiles[0] == profiles[1]
    assert namespaces[0] == namespaces[1]
    assert runtimes[0] == runtimes[1]


def test_actual_predict_owner_records_each_call_and_lowers_once_across_ecc_budgets(
    tmp_path, monkeypatch
):
    request = _request(tmp_path)
    counts, lowered = _install_runtime(monkeypatch, tmp_path)
    backend = Boltz2Backend()
    records = []
    with backend.session((request,)):
        for gib in (22, 24):
            execution.clear_chunking_record()
            native_api.predict(
                input=request.input,
                weights=request.weights,
                mols=request.options["mols"],
                out_dir=tmp_path,
                seed=0,
                write_fmt=None,
                memory_budget_gib=float(gib),
                _runtime=backend,
            )
            records.append(execution.recorded_chunking())
    assert counts == {"traces": 1, "calls": 2}
    assert records[0] == records[1]
    assert records[0]["profiles"][0]["n_token"] == 512
    assert lowered[0].as_text() == lowered[1].as_text()
    assert lowered_cache_key(lowered[0]) == lowered_cache_key(lowered[1])


@pytest.mark.parametrize("padded", [False, True])
@pytest.mark.parametrize("prefix_supported", [False, True])
def test_lower_only_matches_the_production_graph_without_execution_or_cache_writes(
    tmp_path, monkeypatch, padded, prefix_supported
):
    from jax._src import compiler

    request = dataclasses.replace(
        _request(tmp_path),
        num_seeds=1,
        padding={"tokens": 512, "atoms": 32} if padded else None,
    )
    counts, lowered = _install_runtime(monkeypatch, tmp_path, n_token=3)
    monkeypatch.setattr(
        native_api, "_prefix_rng_is_supported", lambda: prefix_supported
    )
    backend = Boltz2Backend()
    backend.predict(request)
    production = lowered[-1]
    previous_calls = counts["calls"]
    # A verifier must not enter the persistent cache scope even when callers
    # supplied the same compile_cache path as predict.
    monkeypatch.setattr(
        native_api,
        "compilation_cache_scope",
        lambda *a, **k: pytest.fail("lower-only initialized the compilation cache"),
    )
    monkeypatch.setattr(
        compiler,
        "backend_compile_and_load",
        lambda *a, **k: pytest.fail("lower-only compiled a program"),
    )
    verified = backend.lower_cache_graphs(request)
    assert counts["calls"] == previous_calls
    assert len(verified) == 1
    name, actual = verified[0]
    assert name == "primary"
    assert actual.as_text() == production.as_text()
    assert actual.in_avals == production.in_avals
    assert actual.in_tree == production.in_tree
    assert lowered_cache_key(actual) == lowered_cache_key(production)


def test_lower_only_refuses_cp_before_feature_preparation(tmp_path, monkeypatch):
    request = _request(tmp_path)
    monkeypatch.setattr(
        native_api,
        "featurize",
        lambda **kwargs: pytest.fail("unsupported CP started feature preparation"),
    )
    with pytest.raises(UnsupportedCacheVerificationError, match="cp_devices=1"):
        native_api.predict(
            input=request.input,
            weights=request.weights,
            mols=request.options["mols"],
            cp_devices=4,
            _lower_only=True,
        )


def test_lower_only_refuses_affinity_without_executing_primary(tmp_path, monkeypatch):
    request = _request(tmp_path)
    counts, _ = _install_runtime(monkeypatch, tmp_path, affinity=True)
    with pytest.raises(UnsupportedCacheVerificationError, match="affinity stage"):
        Boltz2Backend().lower_cache_graphs(request)
    assert counts == {"traces": 0, "calls": 0}


def test_lower_only_refuses_eager_steering(tmp_path, monkeypatch):
    request = _request(tmp_path)
    _install_runtime(monkeypatch, tmp_path)
    with pytest.raises(UnsupportedCacheVerificationError, match="eager steering"):
        native_api.predict(
            input=request.input,
            weights=request.weights,
            mols=request.options["mols"],
            use_potentials=True,
            _lower_only=True,
        )
