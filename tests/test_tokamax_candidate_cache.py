"""Only model executables, not measured Tokamax candidates, persist."""

from __future__ import annotations

import builtins
import functools
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event

import jax
import numpy as np
import pytest

from foldjax import cache
from foldjax.backends import _tokamax_autotune as persistent
from tests.test_alphafold3_tokamax_autotune import (
    _FakeBoundArguments,
    _FakeData,
    _FakeResult,
    _FakeTokamax,
    _inputs,
    _store,
)


def _candidate_factory(config):
    def candidate(x):
        return x * np.float32(config) + np.float32(0.25)

    return candidate


def _apply_fn(x):
    return x * x + np.float32(1)


@pytest.mark.parametrize("parallel", [False, True])
def test_real_candidate_compilation_never_reads_or_writes_persistent_entries(
    tmp_path, monkeypatch, parallel
):
    from jax._src import compiler
    from tokamax._src.autotuning.autotuner import Autotuner

    monkeypatch.setattr(cache, "PERSISTENT_CACHE_MIN_COMPILE_SECS", 0)
    read, write = compiler._cache_read, compiler._cache_write
    model = jax.jit(_apply_fn)
    shared = tmp_path / "compile"

    def read_model(module_name, *args, **kwargs):
        assert not module_name.startswith("jit_forward")
        return read(module_name, *args, **kwargs)

    def write_model(key, elapsed, module_name, *args, **kwargs):
        assert not module_name.startswith("jit_forward")
        return write(key, elapsed, module_name, *args, **kwargs)

    monkeypatch.setattr(compiler, "_cache_read", read_model)
    monkeypatch.setattr(compiler, "_cache_write", write_model)
    with cache.compilation_cache_scope(shared, min_entry_size_bytes=0):
        # Initialize JAX's process-wide is_cache_used latch before disabling
        # candidate persistence; the configuration flag alone then fails.
        model(np.zeros((13,), np.float32)).block_until_ready()
        with persistent.without_candidate_compilation_cache():
            tuner = Autotuner() if parallel else Autotuner(compile_executor_fn=None)
            result = tuner.autotune(
                _candidate_factory, {1, 2}, np.ones((17,), np.float32)
            )
            assert result.fastest_config in {1, 2}
            assert jax.config.jax_enable_compilation_cache
            model(np.zeros((19,), np.float32)).block_until_ready()
        assert not list(shared.glob("jit_forward-*-cache"))
        assert len(list(shared.glob("jit__apply_fn-*-cache"))) == 2
    model.clear_cache()


def test_measurement_persists_the_model_and_next_store_hit_never_retunes(
    tmp_path, monkeypatch
):
    from jax._src import compiler
    from tokamax._src.autotuning.autotuner import Autotuner

    monkeypatch.setattr(cache, "PERSISTENT_CACHE_MIN_COMPILE_SECS", 0)
    tokamax = _FakeTokamax()
    monkeypatch.setattr(persistent, "_tokamax_api", lambda: tokamax)
    argument = _FakeBoundArguments("one-measured-kernel")
    tuning_root = tmp_path / "tuning"
    key, batch = _inputs()
    model = jax.jit(_apply_fn)
    compiled = []
    original_compile = compiler.backend_compile_and_load

    def compile_model(*args, **kwargs):
        compiled.append(None)
        return original_compile(*args, **kwargs)

    def tune(bound_args, *, progress_bar):
        assert not progress_bar and tuple(bound_args) == (argument,)
        tokamax.tune_calls += 1
        measured = Autotuner().autotune(
            _candidate_factory, {1, 2}, np.ones((23,), np.float32)
        )
        return _FakeResult(
            "mock-gpu", ((argument, _FakeData({str(measured.fastest_config): 0.1})),)
        )

    monkeypatch.setattr(tokamax, "autotune", tune)
    monkeypatch.setattr(compiler, "backend_compile_and_load", compile_model)

    def invoke():
        assert jax.config.jax_enable_compilation_cache
        return model(np.ones((29,), np.float32)).block_until_ready()

    try:
        with cache.compilation_cache_scope(
            tmp_path / "compile", min_entry_size_bytes=0
        ):
            persistent.start_record(strategy="autotune", persistent_installed=True)
            first = _store(tuning_root)
            np.testing.assert_array_equal(
                first.call(
                    rng_key=key, batch=batch, lower=lambda: (argument,), invoke=invoke
                ),
                np.full((29,), 2, np.float32),
            )
            assert persistent.recorded()["sources"] == {"measured": 1}
            assert (
                len(list((tuning_root / persistent._CACHE_DIRECTORY).glob("*.json")))
                == 1
            )
            assert not list((tmp_path / "compile").glob("jit_forward-*-cache"))
            assert list((tmp_path / "compile").glob("jit__apply_fn-*-cache"))
            model.clear_cache()
            before = len(compiled)
            persistent.start_record(strategy="autotune", persistent_installed=True)
            second = _store(tuning_root)
            second.call(
                rng_key=key,
                batch=batch,
                lower=lambda: pytest.fail("a store hit must not discover candidates"),
                invoke=invoke,
            )
            assert persistent.recorded()["sources"] == {"store": 1}
            assert tokamax.tune_calls == 1
            assert len(compiled) == before
    finally:
        persistent.clear_record()
        model.clear_cache()


def test_nested_and_failed_candidate_scopes_restore_the_original_hooks(monkeypatch):
    from jax._src import compiler, config
    from tokamax._src import benchmarking

    def original_cache_key(*_args, **_kwargs):
        return "model-key"

    def benchmark(*_args, **_kwargs):
        assert not jax.config.jax_enable_compilation_cache
        assert compiler._get_cache_key() is None
        raise TimeoutError("candidate timeout")

    monkeypatch.setattr(compiler, "_get_cache_key", original_cache_key)
    monkeypatch.setattr(benchmarking, "compile_benchmark", benchmark)
    for enabled in (False, True):
        with config.enable_compilation_cache(enabled):
            with pytest.raises(ValueError, match="stop discovery"):
                with persistent.without_candidate_compilation_cache():
                    wrapped = benchmarking.compile_benchmark
                    with persistent.without_candidate_compilation_cache():
                        with pytest.raises(TimeoutError, match="candidate timeout"):
                            benchmarking.compile_benchmark()
                    assert benchmarking.compile_benchmark is wrapped
                    assert compiler._get_cache_key() == "model-key"
                    assert jax.config.jax_enable_compilation_cache is enabled
                    raise ValueError("stop discovery")
            assert benchmarking.compile_benchmark is benchmark
            assert compiler._get_cache_key is original_cache_key
            assert persistent._CANDIDATE_CACHE_PATCH is None


def test_missing_optional_tokamax_keeps_the_model_cache_untouched(monkeypatch):
    from jax._src import compiler

    original_import = builtins.__import__
    original_key = compiler._get_cache_key

    def without_tokamax(name, *args, **kwargs):
        if name == "tokamax._src":
            raise ImportError("optional Tokamax is absent")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_tokamax)
    with persistent.without_candidate_compilation_cache():
        assert compiler._get_cache_key is original_key
    assert persistent._CANDIDATE_CACHE_PATCH is None


def test_no_store_af3_fallback_still_guards_only_the_candidate(monkeypatch, tmp_path):
    from jax._src import compiler
    from tokamax._src import benchmarking

    from foldjax.backends.alphafold3 import AlphaFold3Backend
    from tests.test_alphafold3_session import _MockRuntime, _request, _weights

    runtime = _MockRuntime(tmp_path, monkeypatch)
    original = runtime.runner.predict_structure
    calls = []

    def original_cache_key(*_args, **_kwargs):
        return "model-key"

    def benchmark(*_args, **_kwargs):
        assert not jax.config.jax_enable_compilation_cache
        assert compiler._get_cache_key() is None
        calls.append("candidate")

    @functools.wraps(original)
    def predict(*args, **kwargs):
        benchmarking.compile_benchmark()
        assert jax.config.jax_enable_compilation_cache
        assert compiler._get_cache_key() == "model-key"
        return original(*args, **kwargs)

    monkeypatch.setattr(compiler, "_get_cache_key", original_cache_key)
    monkeypatch.setattr(benchmarking, "compile_benchmark", benchmark)
    monkeypatch.setattr(runtime.runner, "predict_structure", predict)
    AlphaFold3Backend().predict(
        _request(tmp_path, weights=_weights(tmp_path), output="out", cache_dir=None)
    )
    assert calls == ["candidate"]
    assert benchmarking.compile_benchmark is benchmark
    assert compiler._get_cache_key is original_cache_key


def test_overlapping_scopes_keep_worker_guards_until_the_last_scope_exits(monkeypatch):
    from jax._src import compiler
    from tokamax._src import benchmarking

    ready, leave_first, first_left = Barrier(3), Event(), Event()

    def original_cache_key(*_args, **_kwargs):
        return "model-key"

    def benchmark(*_args, **_kwargs):
        assert not jax.config.jax_enable_compilation_cache
        assert compiler._get_cache_key() is None
        return "candidate"

    monkeypatch.setattr(compiler, "_get_cache_key", original_cache_key)
    monkeypatch.setattr(benchmarking, "compile_benchmark", benchmark)

    def first():
        with persistent.without_candidate_compilation_cache():
            ready.wait(timeout=10)
            assert leave_first.wait(10)
        first_left.set()

    def second():
        with persistent.without_candidate_compilation_cache():
            ready.wait(timeout=10)
            leave_first.set()
            assert first_left.wait(10)
            # This fresh worker inherits neither ContextVar nor JAX state.
            with ThreadPoolExecutor(max_workers=1) as worker:
                assert worker.submit(benchmarking.compile_benchmark).result(10) == (
                    "candidate"
                )
            assert compiler._get_cache_key() == "model-key"
            assert jax.config.jax_enable_compilation_cache

    with ThreadPoolExecutor(max_workers=2) as executor:
        pending = [executor.submit(first), executor.submit(second)]
        ready.wait(timeout=10)
        assert compiler._get_cache_key() == "model-key"
        for future in pending:
            future.result(20)
    assert benchmarking.compile_benchmark is benchmark
    assert compiler._get_cache_key is original_cache_key
    assert persistent._CANDIDATE_CACHE_PATCH is None


def test_timed_out_candidate_restores_its_worker_context(monkeypatch):
    from jax._src import compiler
    from tokamax._src import benchmarking

    def original_cache_key(*_args, **_kwargs):
        return "model-key"

    def benchmark(*_args, **_kwargs):
        assert compiler._get_cache_key() is None
        assert not jax.config.jax_enable_compilation_cache
        raise TimeoutError("candidate compile timeout")

    monkeypatch.setattr(compiler, "_get_cache_key", original_cache_key)
    monkeypatch.setattr(benchmarking, "compile_benchmark", benchmark)

    def worker():
        with pytest.raises(TimeoutError, match="candidate compile timeout"):
            benchmarking.compile_benchmark()
        assert compiler._get_cache_key() == "model-key"
        assert jax.config.jax_enable_compilation_cache

    with persistent.without_candidate_compilation_cache():
        with ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(worker).result(10)
    assert benchmarking.compile_benchmark is benchmark
    assert compiler._get_cache_key is original_cache_key


@pytest.mark.parametrize("route", ["selected", "batch", "glu", "retry"])
def test_all_measurement_routes_enter_the_candidate_guard(monkeypatch, route):
    from jax._src import compiler
    from tokamax._src import benchmarking

    calls = []

    def benchmark(*_args, **_kwargs):
        assert not jax.config.jax_enable_compilation_cache
        assert compiler._get_cache_key() is None
        calls.append(route)

    monkeypatch.setattr(benchmarking, "compile_benchmark", benchmark)
    tokamax = _FakeTokamax()

    class BoundArguments(_FakeBoundArguments):
        heuristics_config = "heuristic"

        def autotune(self, **_kwargs):
            benchmarking.compile_benchmark()
            return _FakeData({"heuristic": 0.1})

    argument = BoundArguments("measured")

    def tune(bound_args, *, progress_bar):
        benchmarking.compile_benchmark()
        return _FakeResult("mock-gpu", ((argument, _FakeData({"batch": 0.1})),))

    monkeypatch.setattr(tokamax, "autotune", tune)
    monkeypatch.setattr(persistent, "_is_glu_bound_arg", lambda _ba: route == "glu")
    if route == "selected":
        persistent._tune_selected_device(
            (argument,), device_kind="mock-gpu", tokamax=tokamax
        )
    elif route == "retry":
        persistent._retry_missing_heuristics(
            (argument,), device_kind="mock-gpu", tokamax=tokamax
        )
    else:
        persistent._tune_missing(
            (argument,), device_kind="mock-gpu", tokamax=tokamax, parallel=True
        )
    assert calls == [route]
    assert persistent._CANDIDATE_CACHE_PATCH is None
