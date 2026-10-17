"""GLU resource limits prune candidates without turning heuristics into measurements."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import pytest

from foldjax._cache_tiers import CacheLocations
from foldjax.backends import _tokamax_autotune as persistent
from tests.test_alphafold3_tokamax_autotune import (
    _FakeBoundArguments,
    _FakeData,
    _FakeTokamax,
    _inputs,
)

tile = pytest.importorskip("tokamax._src.ops.gated_linear_unit.pallas_triton")


def _device(capability="8.9"):
    return SimpleNamespace(
        platform="gpu", compute_capability=capability,
        core_count=128, device_kind="mock-gpu",
    )


class _GLU(_FakeBoundArguments):
    def __init__(self, *, failure="data", weight_dtype=jnp.bfloat16):
        super().__init__("glu")
        self.op = tile.PallasTritonGatedLinearUnit()
        self.args = (
            jax.ShapeDtypeStruct((512, 384), jnp.bfloat16),
            jax.ShapeDtypeStruct((384, 2, 384), weight_dtype),
        )
        self.failure = failure
        self.candidates = []

    @property
    def heuristics_config(self):
        return self.op._get_heuristics_config(self)

    def autotune(self, *, configs=None, cache_results, autotuner, timeout):
        assert not cache_results and timeout in (60.0, 600.0)
        configs = configs or (
            self.op._get_autotuning_configs(self) | {self.heuristics_config}
        )
        self.candidates.append(configs)
        resource = RuntimeError(
            "Shared memory size limit exceeded: requested 110592, available: 101376"
            if self.failure == "shared-limit"
            else "RESOURCE_EXHAUSTED: shared memory required 110592, limit 101376"
        )
        oom = RuntimeError(
            "cudaErrorMemoryAllocation before calling cuModuleGetFunction"
        )
        if self.failure == "group":
            raise ExceptionGroup("All configs failed", [resource, oom])
        if self.failure == "programming-group":
            raise ExceptionGroup("All configs failed", [oom, TypeError("bad kernel")])
        if self.failure == "programming":
            return _FakeData({cfg: TypeError("bad kernel") for cfg in configs})
        if self.failure == "unsupported":
            raise NotImplementedError(
                "RESOURCE_EXHAUSTED: GLU is unsupported on the selected device"
            )
        return _FakeData({
            cfg: resource if index % 2 else oom
            for index, cfg in enumerate(configs)
        })


@pytest.fixture
def runtime(monkeypatch):
    tokamax = _FakeTokamax()
    monkeypatch.setattr(persistent, "_tokamax_api", lambda: tokamax)
    yield tokamax
    persistent.clear_record()


def _store(root: Path, *, parallel=True, layered=False):
    shared = root / "shared"
    shared.mkdir(mode=0o700)
    return persistent._PersistentTokamaxStore(
        shared, identity={"runtime": "test"}, strategy="autotune",
        device_kind="mock-gpu", can_tune=parallel,
        cache_locations=(
            CacheLocations(shared, shared, root / "private") if layered else None
        ),
    )


@pytest.mark.parametrize("weight_dtype", [jnp.bfloat16, jnp.float32])
def test_4090_glu_candidates_are_pruned_before_measurement(weight_dtype):
    glu = _GLU(weight_dtype=weight_dtype)
    raw = glu.op._get_autotuning_configs(glu)
    assert len(raw) == 120
    original = type(glu.op).__dict__["_get_autotuning_configs"]
    with persistent.glu_heuristics_within_shared_memory(_device()):
        filtered = glu.op._get_autotuning_configs(glu)
        itemsize, weight_itemsize = persistent._glu_itemsizes(glu)
        budget = persistent._shared_memory_budget(_device())
        assert budget == 101376
        assert filtered and len(filtered) < len(raw)
        assert all(
            cfg.num_stages * persistent._glu_stage_bytes(
                cfg, itemsize, weight_itemsize=weight_itemsize
            ) <= budget
            for cfg in filtered | {glu.heuristics_config}
        )
        with persistent.glu_heuristics_within_shared_memory(_device("9.0")):
            assert glu.op._get_autotuning_configs(glu) == raw
        assert glu.op._get_autotuning_configs(glu) == filtered
    assert type(glu.op).__dict__["_get_autotuning_configs"] is original


@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("layered", [False, True])
@pytest.mark.parametrize("failure", ["data", "group", "shared-limit"])
def test_resource_only_failure_runs_clamped_heuristic_with_measured_partial(
    tmp_path, runtime, parallel, layered, failure
):
    store = _store(tmp_path, parallel=parallel, layered=layered)
    glu, measured = _GLU(failure=failure), _FakeBoundArguments("measured")
    rng, batch = _inputs()
    persistent.start_record(strategy="autotune", persistent_installed=True)
    arguments = (glu, measured)
    executions = []

    def invoke():
        assert runtime.fallbacks[-1] == "heuristics"
        assert glu.cached_autotuning_data is None
        assert measured.cached_autotuning_data.fastest_config == "measured-fast"
        config = glu.heuristics_config
        assert config.num_stages * persistent._glu_stage_bytes(config, 2) <= 101376
        executions.append(config)
        return "heuristic-result"

    with persistent.glu_heuristics_within_shared_memory(_device()):
        assert store.call(
            rng_key=rng, batch=batch, lower=lambda: arguments, invoke=invoke
        ) == "heuristic-result"
        assert store.call(
            rng_key=rng, batch=batch,
            lower=lambda: pytest.fail("heuristic memory reuse must not retune"),
            invoke=invoke,
        ) == "heuristic-result"
    assert len(glu.candidates) == 2
    assert len(glu.candidates[1]) == 1  # The existing heuristic measurement retry.
    assert len(executions) == 2
    assert persistent.recorded()["sources"] == {"heuristic": 2}
    assert not store._results and not store._persisted
    assert not store._partial_results and len(store._heuristic_results) == 1
    assert not list(tmp_path.rglob("*.json"))


@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("failure", ["programming", "programming-group", "unsupported"])
def test_unexpected_glu_failure_is_not_hidden_by_a_heuristic(
    tmp_path, runtime, parallel, failure
):
    store = _store(tmp_path, parallel=parallel)
    glu = _GLU(failure=failure)
    rng, batch = _inputs()
    with persistent.glu_heuristics_within_shared_memory(_device()):
        with pytest.raises(persistent._UnexpectedTuningError, match="unexpectedly"):
            store.call(
                rng_key=rng, batch=batch, lower=lambda: (glu,),
                invoke=lambda: pytest.fail("a programming failure must not execute"),
            )
    assert not store._heuristic_results and not list(tmp_path.rglob("*.json"))


def test_wrong_tuning_device_is_not_hidden_by_a_heuristic(
    tmp_path, runtime, monkeypatch
):
    store = _store(tmp_path)
    original = runtime.autotune

    def wrong_device(*args, **kwargs):
        result = original(*args, **kwargs)
        result.device_kind = "another-gpu"
        return result

    monkeypatch.setattr(runtime, "autotune", wrong_device)
    rng, batch = _inputs()
    arguments = (_GLU(), _FakeBoundArguments("measured"))
    with persistent.glu_heuristics_within_shared_memory(_device()):
        with pytest.raises(persistent._UnexpectedTuningError, match="wrong device"):
            store.call(
                rng_key=rng, batch=batch, lower=lambda: arguments,
                invoke=lambda: pytest.fail("wrong-device configs must not execute"),
            )


def test_heuristic_lowering_rejects_newly_uncovered_operations(tmp_path, runtime):
    store = _store(tmp_path)
    glu, new = _GLU(), _FakeBoundArguments("newly-uncovered")
    rng, batch = _inputs()

    def lower():
        return (glu, new) if len(glu.candidates) == 2 else (glu,)

    with persistent.glu_heuristics_within_shared_memory(_device()):
        with pytest.raises(persistent._UnexpectedTuningError, match="uncovered"):
            store.call(
                rng_key=rng, batch=batch, lower=lower,
                invoke=lambda: pytest.fail("uncovered ops must not execute"),
            )
    assert not store._heuristic_results and not list(tmp_path.rglob("*.json"))


def test_cached_heuristic_requires_the_original_device_guard(tmp_path, runtime):
    store, glu = _store(tmp_path), _GLU()
    rng, batch = _inputs()

    def call():
        return store.call(
            rng_key=rng, batch=batch, lower=lambda: (glu,), invoke=lambda: None
        )

    with persistent.glu_heuristics_within_shared_memory(_device()):
        call()
    with pytest.raises(persistent._UnexpectedTuningError, match="memory guard"):
        call()
    wrong = _device()
    wrong.device_kind = "another-gpu"
    with persistent.glu_heuristics_within_shared_memory(wrong):
        with pytest.raises(persistent._UnexpectedTuningError, match="wrong device"):
            call()


def test_unknown_shared_memory_budget_keeps_existing_partial_semantics(
    tmp_path, runtime
):
    store = _store(tmp_path)
    rng, batch = _inputs()
    arguments = (_GLU(), _FakeBoundArguments("measured"))
    persistent.start_record(strategy="autotune", persistent_installed=True)
    with persistent.glu_heuristics_within_shared_memory(_device("9.0")):
        store.call(
            rng_key=rng, batch=batch, lower=lambda: arguments, invoke=lambda: None
        )
    assert persistent.recorded()["sources"] == {"partial": 1}
    assert not store._heuristic_results


def test_real_tokamax_overlay_consumes_the_clamped_config_without_measurements(
    monkeypatch,
):
    import tokamax
    from tokamax._src.ops import op as op_lib

    device = _device()
    monkeypatch.setattr(op_lib.backend, "get_default_device", lambda: device)
    monkeypatch.setattr(
        tile.PallasTritonGatedLinearUnit, "get_autotuning_cache", lambda *_: {}
    )
    monkeypatch.setattr(tokamax.autotuning, "get_bound_args", lambda value: value)
    ba = tile.PallasTritonGatedLinearUnit().bind(*_GLU().args)
    partial = tokamax.AutotuningResult("mock-gpu", ())
    error = persistent._IncompleteMeasurementsError(
        "resource failure", partial=partial, missing=(ba,), heuristic_ops=(ba,)
    )
    with persistent.glu_heuristics_within_shared_memory(device):
        retained = persistent._resource_heuristic_result(
            error, lower=lambda: (ba,), device_kind="mock-gpu"
        )
        with retained, tokamax.config.autotuning_cache_miss_fallback("heuristics"):
            assert ba.cached_autotuning_data is None
            assert ba.default_config == ba.heuristics_config
            config = ba.default_config
            assert config.num_stages * persistent._glu_stage_bytes(config, 2) <= 101376
    assert not partial.data


def test_real_triton_resource_errors_are_recognized_without_masking_other_errors():
    errors = pytest.importorskip("triton.runtime.errors")
    resource = errors.OutOfResources(139264, 101376, "shared memory")
    assert persistent._is_resource_failure(resource)
    assert persistent._is_resource_failure(ExceptionGroup(
        "all failed", [resource, MemoryError("CUDA out of memory")]
    ))
    assert not persistent._is_resource_failure(ExceptionGroup(
        "all failed", [resource, TypeError("wrong dtype")]
    ))


@pytest.mark.parametrize(
    ("error_type", "expected"),
    [
        (RuntimeError, True),
        (jax.errors.JaxRuntimeError, True),
        (ValueError, False),
        (TypeError, False),
        (NotImplementedError, False),
    ],
)
def test_shared_memory_size_limit_errors_keep_the_runtime_type_guard(
    error_type, expected
):
    error = error_type(
        "Shared memory size limit exceeded: requested 110592, available: 101376"
    )
    assert persistent._is_resource_failure(error) is expected
