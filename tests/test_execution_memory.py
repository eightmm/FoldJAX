"""Bounded live ownership across model/input boundaries, without a GPU."""

import sys
import warnings
import weakref
from contextlib import contextmanager
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

import foldjax
from foldjax import _execution_memory as memory
from foldjax.backends.base import Backend
from foldjax.registry import backend_override
from foldjax.schema import (
    ModelCapabilities,
    PredictionRequest,
    PredictionResult,
    PredictionSample,
)


def test_input_boundaries_release_cycles_and_weights_but_keep_result_arrays(tmp_path):
    live = []
    live_before_load = []
    weights_seen = []

    class State:
        def __init__(self):
            self.weights = np.ones((128, 128))
            self.features = np.ones((64, 64))
            self.cycle = self

    class ToyBackend(Backend):
        name = "boltz2"
        session_reuse = True

        def __init__(self):
            self.state = None

        def capabilities(self):
            return ModelCapabilities(model=self.name, input_formats=("native",))

        @contextmanager
        def session(self, requests):
            try:
                yield self
            finally:
                self.invalidate_session()

        def invalidate_session(self):
            self.state = None

        def predict(self, request):
            if self.state is None:
                # Counts current live bytes, unlike ru_maxrss's high-water mark.
                live_before_load.append(
                    sum(
                        ref().weights.nbytes + ref().features.nbytes
                        for ref in live
                        if ref() is not None
                    )
                )
                self.state = State()
                live.append(weakref.ref(self.state))
            weights_seen.append(id(self.state))
            return PredictionResult(
                model=self.name,
                samples=(
                    PredictionSample(
                        seed=request.seed,
                        coordinates=np.full((1, 3), request.seed),
                    ),
                ),
                output_dir=request.output_dir,
                raw={"array": np.array([request.seed])},
            )

    inputs = tuple(tmp_path / f"job-{i}.json" for i in range(6))
    for source in inputs:
        source.write_text("{}")
    weights = tmp_path / "weights"
    weights.mkdir()
    backend = ToyBackend()
    with backend_override("boltz2", lambda: backend):
        results = foldjax.predict(
            PredictionRequest(
                model="boltz2",
                inputs=inputs,
                input_format="native",
                weights=weights,
                output_dir=tmp_path / "out",
                seeds=(7, 8),
                use_compile_cache=False,
            )
        )
    assert live_before_load == [0] * 6
    assert all(ref() is None for ref in live)
    assert all(weights_seen[i] == weights_seen[i + 1] for i in range(0, 12, 2))
    assert len(results) == 6
    for result in results:
        np.testing.assert_array_equal(result.samples[0].coordinates, [[7, 7, 7]])
        np.testing.assert_array_equal(result.raw[0]["array"], [7])


def test_loaded_model_pools_and_tokamax_are_cleared_without_imports(monkeypatch):
    from contextvars import ContextVar

    events = []
    for module_name, names in memory._CACHES.items():
        module = ModuleType(module_name)
        for name in names:
            setattr(
                module,
                name,
                SimpleNamespace(clear_cache=lambda name=name: events.append(name)),
            )
        monkeypatch.setitem(sys.modules, module_name, module)
    overlay = ContextVar("test_tuning_overlay", default=())
    tuning = {"key": object()}
    monkeypatch.setitem(
        sys.modules,
        "tokamax._src.ops.op",
        SimpleNamespace(
            AUTOTUNING_CACHE_OVERLAY_STACK=overlay,
            _AUTOTUNING_CACHE=tuning,
        ),
    )
    # A fake loaded JAX proves that cleanup needs no device/backend probe.
    monkeypatch.setitem(
        sys.modules,
        "jax",
        SimpleNamespace(
            clear_caches=lambda: events.append("jax"),
        ),
    )
    monkeypatch.setattr(
        memory._managed_memory.gc, "collect", lambda: events.append("gc")
    )
    monkeypatch.setattr(
        memory._managed_memory, "_malloc_trim", lambda: events.append("trim")
    )
    memory.release_execution_memory()
    assert tuning == {}
    assert len(events) == sum(map(len, memory._CACHES.values())) + 3
    assert events[-3:] == ["jax", "gc", "trim"]
    tuning["outer"] = object()
    with_overlay = (object(),)
    token = overlay.set(with_overlay)
    try:
        memory.release_execution_memory()
        assert overlay.get() is with_overlay
        assert "outer" in tuning
    finally:
        overlay.reset(token)


def test_cleanup_keeps_going_with_warnings_as_errors(monkeypatch):
    events = []

    def fail():
        raise RuntimeError("cleanup failed")

    monkeypatch.setattr(memory, "_CACHES", {})
    monkeypatch.delitem(sys.modules, "tokamax._src.ops.op", raising=False)
    monkeypatch.setitem(sys.modules, "jax", SimpleNamespace(clear_caches=fail))
    monkeypatch.setattr(memory._managed_memory.gc, "collect", fail)
    monkeypatch.setattr(
        memory._managed_memory, "_malloc_trim", lambda: events.append("trim")
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        memory.release_execution_memory()
    assert events == ["trim"]


@pytest.mark.parametrize("failure", [ValueError("bad input"), KeyboardInterrupt()])
def test_input_state_is_released_on_failure_or_cancellation(tmp_path, failure):
    from tests.test_api import DummyBackend, _request

    class FailingBackend(DummyBackend):
        session_reuse = True

        def __init__(self):
            super().__init__()
            self.state = None

        def predict(self, request):
            self.state = np.ones((128, 128))
            raise failure

        def invalidate_session(self):
            self.state = None

    backend = FailingBackend()
    with backend_override("boltz2", lambda: backend):
        with pytest.raises(type(failure)):
            foldjax.predict(_request(tmp_path, use_compile_cache=False))
    assert backend.state is None
