"""Cache activity belongs to the executing request, including private fallbacks."""

from __future__ import annotations

import dataclasses
import json
from contextlib import contextmanager

import jax
import numpy as np
import pytest

import foldjax
from foldjax import api, cache, memory_policy
from foldjax.backends import _tokamax_autotune as tuning
from foldjax.registry import backend_override
from tests.test_manifest import _job, _Recorder, _weights
from tests.test_resume_manifest import _backends, _request


def test_private_fallback_is_namespaced_like_shared_and_recorded(monkeypatch, tmp_path):
    shared = tmp_path / "shared"
    private = tmp_path / "private"
    monkeypatch.setenv("FOLDJAX_PRIVATE_COMPILE_CACHE", str(private))
    # An unavailable shared namespace must still reach a working private cache.
    monkeypatch.setattr(cache, "_layered_shared_directory", lambda directory: None)
    calls = []
    original_scope = api.compilation_cache_scope

    @contextmanager
    def observe_scope(directory, **kwargs):
        calls.append((directory, kwargs.get("private_directory")))
        with original_scope(directory, **kwargs):
            yield

    monkeypatch.setattr(api, "compilation_cache_scope", observe_scope)

    class _CompiledRecorder(_Recorder):
        def predict(self, request):
            # A fresh Python callable bypasses JAX's in-memory executable cache;
            # identical HLO can still be loaded from the persistent cache.
            operation = jax.jit(lambda value: value @ value.T + value)
            jax.block_until_ready(operation(np.eye(8, dtype=np.float32)))
            return super().predict(request)

    request = foldjax.PredictionRequest(
        model="opendde",
        input=_job(tmp_path),
        weights=_weights(tmp_path),
        profile="released",
        output_dir=tmp_path / "out",
        cache_dir=shared,
        seed=17,
    )
    with backend_override("opendde", _CompiledRecorder):
        foldjax.predict(request)
        first = json.loads((request.output_dir / "foldjax_run.json").read_text())
        foldjax.predict(dataclasses.replace(request, output_dir=tmp_path / "second"))
        second = json.loads((tmp_path / "second" / "foldjax_run.json").read_text())

    shared_namespace, private_namespace = calls[0]
    assert shared_namespace.relative_to(shared) == private_namespace.relative_to(
        private
    )
    assert calls[1] == calls[0]
    assert not shared.exists()
    assert private_namespace.stat().st_mode & 0o777 == 0o700
    assert first["compilation_cache"]["locations"] == {
        "shared": None,
        "private": str(private_namespace),
    }
    assert first["compilation_cache"]["writes"]["private"]["success"] >= 1
    assert first["compilation_cache"]["compiles"]["success"] >= 1
    assert second["compilation_cache"]["loads"]["private"]["success"] >= 1
    cache.clear_record()


def test_no_cache_does_not_create_a_private_namespace(monkeypatch, tmp_path):
    private = tmp_path / "private"
    monkeypatch.setenv("FOLDJAX_PRIVATE_COMPILE_CACHE", str(private))
    request = _request(tmp_path)
    with _backends([]):
        foldjax.predict(request)
    document = json.loads((request.output_dir / "foldjax_run.json").read_text())
    assert not private.exists()
    assert "compilation_cache" not in document


def test_all_resumed_seeds_do_not_inherit_previous_run_telemetry(tmp_path):
    request = _request(tmp_path, seed=None, seeds=(7, 8), representations=())
    calls = []
    try:
        with _backends(calls):
            foldjax.predict(request)
            with pytest.warns(RuntimeWarning, match="no alphafold3 peak law"):
                memory_policy.admit_unmeasured(
                    model="alphafold3",
                    detail="no fitted law",
                    budget=memory_policy.resolve_budget(
                        pool_bytes=None, card_bytes=None, override_gib=None
                    ),
                )
            tuning.start_record(strategy="autotune", persistent_installed=True)
            report = foldjax.predict_batch(dataclasses.replace(request, resume=True))
        assert len(calls) == 2
        assert len(report.skipped) == 2
        document = json.loads((request.output_dir / "foldjax_run.json").read_text())
        assert document["memory"] is None
        assert "kernel_tuning" not in document
        assert "compilation_cache" not in document
    finally:
        memory_policy.clear_record()
        memory_policy.reset_warnings()
        tuning.clear_record()
        cache.clear_record()
