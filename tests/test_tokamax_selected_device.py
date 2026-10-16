"""Tokamax must follow selected hardware without changing JAX for other callers."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from threading import Barrier, Event
from types import SimpleNamespace

import jax
import numpy as np
import pytest

from foldjax import _tokamax_device as selected
from foldjax.backends import _tokamax_autotune as persistent
from tests.test_alphafold3_tokamax_autotune import (
    _FakeBoundArguments,
    _FakeData,
    _FakeTokamax,
    _inputs,
    _store,
)


def _gpu(cores=80):
    return SimpleNamespace(
        platform="gpu",
        core_count=cores,
        compute_capability="8.9",
        device_kind="mock-gpu",
    )


def _helper_modules():
    from tokamax._src import benchmarking, precision
    from tokamax._src.ops.normalization import pallas_triton_config
    from tokamax._src.pallas import grid

    return grid, pallas_triton_config, precision, benchmarking


def _assert_helpers_use(device):
    grid, norm, precision, benchmark = _helper_modules()
    assert grid.jax.devices() == [device]
    cost = grid._get_group_size_cost(1, 8, 8, 32, 32)
    live = device.core_count
    assert cost == min((live + 7) // 8, 8) * 32 + min(live, 8) * 32
    config = norm.get_heuristics_config(
        jax.ShapeDtypeStruct((4096, 128), np.float32),
        jax.ShapeDtypeStruct((128,), np.float32),
        None,
        axis=-1,
        vmap_axis_sizes=(),
    )
    assert config.block_m >= 1
    assert precision.to_dot_algorithm_preset(np.float32, np.float32, "high") == (
        jax.lax.DotAlgorithmPreset.TF32_TF32_F32
    )
    assert benchmark.jax.default_backend() == "gpu"


def test_actual_grid_normalization_and_precision_use_selected_gpu():
    device = _gpu()
    modules = _helper_modules()
    originals = [module.jax for module in modules]
    devices, backend = jax.devices, jax.default_backend
    with selected.selected_tokamax_device(device):
        _assert_helpers_use(device)
        grid = modules[0]
        pids = grid.get_cheapest_grid_pids(
            0, grid_m=8, grid_n=8, block_m_cost=32, block_n_cost=32
        )
        assert tuple(map(int, pids)) == (0, 0)
        assert jax.devices is devices and jax.default_backend is backend
        assert jax.devices()[0].platform == "cpu"
        assert jax.default_backend() == "cpu"
    assert [module.jax for module in modules] == originals


def test_explicit_backend_queries_keep_the_original_backend_semantics():
    calls = []
    original = SimpleNamespace(
        devices=lambda backend=None: calls.append(backend) or [backend],
        default_backend=lambda: "cpu",
    )
    proxy = selected._SelectedJax(original)
    device = _gpu()
    with selected.selected_tokamax_device(device):
        assert proxy.devices() == [device]
        assert proxy.devices(None) == [device]
        assert proxy.devices("cpu") == ["cpu"]
        assert proxy.devices(backend="gpu") == ["gpu"]
        assert calls == ["cpu", "gpu"]
    assert proxy.devices() == [None]
    assert proxy.default_backend() == "cpu"


def test_nested_scopes_restore_selected_hardware_after_an_exception():
    grid = _helper_modules()[0]
    original = grid.jax
    first, second = _gpu(16), _gpu(132)
    with selected.selected_tokamax_device(first):
        _assert_helpers_use(first)
        with pytest.raises(ValueError, match="stop"):
            with selected.selected_tokamax_device(second):
                _assert_helpers_use(second)
                raise ValueError("stop")
        _assert_helpers_use(first)
    assert grid.jax is original
    assert not selected._PATCHES


def test_overlapping_threads_keep_context_and_restore_only_after_last_exit():
    grid = _helper_modules()[0]
    original = grid.jax
    ready, finish_first, first_left = Barrier(3), Event(), Event()
    first, second = _gpu(16), _gpu(132)

    def run_first():
        with selected.selected_tokamax_device(first):
            ready.wait(timeout=5)
            assert finish_first.wait(5)
            _assert_helpers_use(first)
        first_left.set()

    def run_second():
        with selected.selected_tokamax_device(second):
            ready.wait(timeout=5)
            finish_first.set()
            assert first_left.wait(5)
            assert grid.jax is not original
            _assert_helpers_use(second)

    with ThreadPoolExecutor(max_workers=2) as executor:
        pending = [executor.submit(run_first), executor.submit(run_second)]
        ready.wait(timeout=5)
        # This caller has no selected-device context even while proxies exist.
        assert grid.jax.devices()[0].platform == "cpu"
        for future in pending:
            future.result(timeout=10)
    assert grid.jax is original
    assert not selected._PATCHES


def test_managed_prediction_enters_the_selected_tokamax_scope(monkeypatch, tmp_path):
    from foldjax import api

    device = _gpu()
    monkeypatch.setattr("foldjax._gpu.default_device", lambda _request: device)
    monkeypatch.setattr(jax, "default_device", lambda _device: nullcontext())

    def run(*_args, **_kwargs):
        _assert_helpers_use(device)
        return "selected-device prediction"

    monkeypatch.setattr(api, "_predict_once_on_device", run)
    assert (
        api._predict_once(SimpleNamespace(), 0, tmp_path)
        == "selected-device prediction"
    )
    assert not selected._PATCHES


def test_af3_synchronous_discovery_uses_actual_tokamax_hardware_helpers(
    monkeypatch, tmp_path
):
    tokamax = _FakeTokamax()
    monkeypatch.setattr(persistent, "_tokamax_api", lambda: tokamax)
    device = _gpu()

    class BoundArguments(_FakeBoundArguments):
        def autotune(self, *, cache_results, autotuner, timeout):
            _assert_helpers_use(device)
            assert autotuner.compile_executor_fn is None
            return _FakeData({"selected": 1.0})

    arguments = (BoundArguments("selected-grid-normalization-precision"),)
    store = _store(tmp_path / "cache", can_tune=False)
    key, batch = _inputs()

    def lower():
        _assert_helpers_use(device)
        return arguments

    with selected.selected_tokamax_device(device):
        assert (
            store.call(rng_key=key, batch=batch, lower=lower, invoke=lambda: "ok")
            == "ok"
        )
    assert tokamax.tune_calls == 0
    assert (
        len(list((tmp_path / "cache" / persistent._CACHE_DIRECTORY).glob("*.json")))
        == 1
    )
