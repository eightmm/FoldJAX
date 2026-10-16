"""Failed candidates must not discard measured kernels or poison a store."""

from __future__ import annotations

import json

import jax
import numpy as np
import pytest

from foldjax.backends import _tokamax_autotune as persistent
from tests.test_alphafold3_tokamax_autotune import (
    _FakeBoundArguments,
    _FakeData,
    _FakeResult,
    _FakeTokamax,
    _inputs,
    _store,
)


def test_real_timeout_compile_failure_and_success_persist_only_the_best():
    import tokamax
    from tokamax._src import precision
    from tokamax._src.autotuning.autotuner import AutotuningData
    from tokamax._src.benchmarking import BenchmarkData
    from tokamax._src.ops.gated_linear_unit import pallas_triton as glu

    argument = glu.PallasTritonGatedLinearUnit().bind(
        jax.ShapeDtypeStruct((4, 8), np.float32),
        jax.ShapeDtypeStruct((8, 2, 8), np.float32),
        activation=jax.nn.silu,
        precision=precision.canonicalize_precision("high"),
        return_residuals=False,
    )
    configs = [
        glu.Config(block_m=32, block_n=32, block_k=32, num_warps=4, num_stages=stages)
        for stages in (1, 2, 3, 4)
    ]
    data = AutotuningData(
        {
            configs[0]: TimeoutError("candidate compile timeout"),
            configs[1]: RuntimeError("RET_CHECK candidate compile failure"),
            configs[2]: BenchmarkData(1.0, 1.0, (0.5,), {}),
            configs[3]: BenchmarkData(1.0, 1.0, (0.3,), {}),
        }
    )
    kind = jax.devices()[0].device_kind
    complete = persistent._complete_result(
        (argument,), tokamax.AutotuningResult(kind, ((argument, data),)), kind
    )
    assert complete.data[0][1].fastest_config == configs[3]
    assert len(complete.data[0][1]) == 1
    payload = persistent._manifest_bytes(
        complete,
        signature="a" * 64,
        call_sha256="b" * 64,
        identity={"runtime": "test"},
        expected_count=1,
        device_kind=kind,
    )
    assert b"candidate compile timeout" not in payload
    assert b"RET_CHECK" not in payload
    loaded = persistent._decode_manifest(
        payload,
        signature="a" * 64,
        call_sha256="b" * 64,
        identity={"runtime": "test"},
        device_kind=kind,
    )
    assert loaded.data[0][1].fastest_config == configs[3]


@pytest.mark.parametrize("recover", [False, True])
def test_retry_only_the_uncovered_operation_and_never_publish_partial_results(
    monkeypatch, tmp_path, caplog, recover
):
    tokamax = _FakeTokamax()
    monkeypatch.setattr(persistent, "_tokamax_api", lambda: tokamax)
    retries = []

    class BoundArguments(_FakeBoundArguments):
        @property
        def heuristics_config(self):
            return "heuristic"

        def autotune(self, *, configs, cache_results, autotuner, timeout):
            retries.append(self.key)
            assert configs == {"heuristic"}
            assert not cache_results and autotuner.compile_executor_fn is None
            return _FakeData(
                {
                    "heuristic": 0.3
                    if recover
                    else RuntimeError("RET_CHECK retry failed")
                }
            )

    arguments = (BoundArguments("already-measured"), BoundArguments("missing-kernel"))

    def tune(bound_args, *, progress_bar):
        tokamax.tune_calls += 1
        assert tuple(ba.key for ba in bound_args) == (
            "already-measured",
            "missing-kernel",
        )
        return _FakeResult(
            "mock-gpu",
            (
                (arguments[0], _FakeData({"measured": 0.1})),
                (
                    arguments[1],
                    _FakeData(
                        {
                            "timeout": TimeoutError("30 candidate timeouts"),
                            "compile": RuntimeError(
                                "151 RET_CHECK compilation failures"
                            ),
                        }
                    ),
                ),
            ),
        )

    monkeypatch.setattr(tokamax, "autotune", tune)
    store = _store(tmp_path / "cache")
    key, batch = _inputs()

    overlay_ids = []

    def invoke():
        assert arguments[0].cached_autotuning_data.fastest_config == "measured"
        if recover:
            assert arguments[1].cached_autotuning_data.fastest_config == "heuristic"
        else:
            assert arguments[1].cached_autotuning_data is None
        overlay_ids.append(id(_FakeResult.active[-1]))
        return "executed"

    persistent.start_record(strategy="autotune", persistent_installed=True)

    assert (
        store.call(rng_key=key, batch=batch, lower=lambda: arguments, invoke=invoke)
        == "executed"
    )
    assert (
        store.call(
            rng_key=key,
            batch=batch,
            lower=lambda: pytest.fail(
                "retained results must not rediscover all kernels"
            ),
            invoke=invoke,
        )
        == "executed"
    )
    assert overlay_ids[0] == overlay_ids[1]
    assert tokamax.tune_calls == 1
    assert retries == ["missing-kernel"]
    assert persistent.recorded()["sources"] == (
        {"measured": 1, "store": 1} if recover else {"partial": 2}
    )
    persistent.clear_record()
    manifests = list((tmp_path / "cache" / persistent._CACHE_DIRECTORY).glob("*.json"))
    if recover:
        assert len(manifests) == 1
        document = json.loads(manifests[0].read_text())
        assert document["coverage"]["entries"] == 2
        assert document["coverage"]["configs_per_entry"] == 1
    else:
        assert not manifests
        assert "missing-kernel" in caplog.text
        assert "TimeoutError" in caplog.text and "RET_CHECK" in caplog.text
        assert "single-heuristic retry remains incomplete" in caplog.text


def test_synchronous_operation_failure_keeps_previously_measured_operations(
    monkeypatch,
):
    tokamax = _FakeTokamax()
    monkeypatch.setattr(persistent, "_tokamax_api", lambda: tokamax)

    class FailingArguments(_FakeBoundArguments):
        def autotune(self, **_kwargs):
            raise RuntimeError("RET_CHECK synchronous compile failed")

    good = _FakeBoundArguments("completed-kernel")
    failed = FailingArguments("raising-kernel")
    tuned = persistent._tune_selected_device(
        (good, failed), device_kind="mock-gpu", tokamax=tokamax
    )
    assert len(tuned.data) == 2
    with tuned:
        assert good.cached_autotuning_data.fastest_config == "completed-kernel-fast"
    with pytest.raises(persistent._IncompleteCoverageError) as caught:
        persistent._complete_result((good, failed), tuned, "mock-gpu")
    assert "raising-kernel" in str(caught.value)
    assert "RET_CHECK synchronous compile failed" in str(caught.value)
    assert len(caught.value.partial.data) == 1


def test_selected_device_total_failure_names_the_kernel_in_the_terminal_error(
    monkeypatch, tmp_path
):
    tokamax = _FakeTokamax()
    monkeypatch.setattr(persistent, "_tokamax_api", lambda: tokamax)

    class FailingArguments(_FakeBoundArguments):
        def autotune(self, **_kwargs):
            raise RuntimeError("RET_CHECK no candidate compiled")

    store = _store(tmp_path / "cache", can_tune=False)
    key, batch = _inputs()
    with pytest.raises(RuntimeError, match="selected-device tuning") as caught:
        store.call(
            rng_key=key,
            batch=batch,
            lower=lambda: (FailingArguments("uncovered-kernel"),),
            invoke=lambda: pytest.fail(
                "an uncovered selected GPU cannot use inline tuning"
            ),
        )
    assert "uncovered-kernel" in str(caught.value)
    assert "RET_CHECK no candidate compiled" in str(caught.value)
    assert not list((tmp_path / "cache" / persistent._CACHE_DIRECTORY).glob("*.json"))
