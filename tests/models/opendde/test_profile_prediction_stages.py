from __future__ import annotations

import json
from contextlib import contextmanager
from types import SimpleNamespace

import jax
import numpy as np
import pytest
from jax._src import compiler

from bench import opendde_profile
from foldjax.models.opendde import runner


@pytest.mark.parametrize(
    ("environment", "option", "expected"),
    [
        (None, [], 0.9),
        ("0.85", [], 0.85),
        (None, ["--mem-fraction", "0.8"], 0.8),
        ("0.85", ["--mem-fraction=0.8"], 0.8),
    ],
)
@pytest.mark.parametrize("failure", [False, True])
def test_cli_configures_allocator_before_the_profiler_can_import_the_model(
    tmp_path, monkeypatch, environment, option, expected, failure
):
    from foldjax import cli, oom
    from foldjax.schema import BatchReport

    monkeypatch.delenv(oom.CURRENT_FRACTION_ENV, raising=False)
    monkeypatch.delenv(oom.FRACTION_ENV, raising=False)
    if environment is not None:
        monkeypatch.setenv(oom.FRACTION_ENV, environment)
    monkeypatch.setattr(cli, "_apply_gc_threshold", lambda: None)
    monkeypatch.setattr(cli, "_enable_release_reclaim", lambda: None)
    monkeypatch.setattr(
        cli, "_request", lambda *a, **kw: SimpleNamespace(models=None, inputs=None)
    )
    observed = []

    @contextmanager
    def profile(report):
        assert oom.mem_fraction() == expected
        observed.append("enter")
        try:
            yield
        finally:
            observed.append("exit")

    def predict(*args, **kwargs):
        assert observed == ["enter"]
        if failure:
            raise RuntimeError("prediction failed")
        return BatchReport()

    monkeypatch.setattr(opendde_profile, "profile_prediction", profile)
    monkeypatch.setattr(cli, "_run_predictions", predict)
    output = tmp_path / "profile.json"
    command = [
        "--profile-out",
        str(output),
        "--",
        "predict",
        "--model",
        "opendde",
        "--quiet",
        *option,
    ]
    if failure:
        with pytest.raises(RuntimeError, match="prediction failed"):
            opendde_profile.main(command)
        assert json.loads(output.read_text())["error_type"] == "RuntimeError"
    else:
        assert opendde_profile.main(command) == 0
        assert json.loads(output.read_text())["exit_code"] == 0
    assert observed == ["enter", "exit"]
    assert cli._run_predictions is predict


def test_diagnostic_observes_real_executable_without_changing_outputs_or_owners(
    monkeypatch,
) -> None:
    @jax.jit
    def graph(value):
        return value @ value + 1.0

    def failing_score(*args, **kwargs):
        raise ValueError("score failed")

    monkeypatch.setattr(runner, "_predict", graph)
    monkeypatch.setattr(runner, "_score", failing_score)
    monkeypatch.setattr(opendde_profile, "_device_memory", lambda: {})
    original_compile = compiler.compile_or_get_cached
    value = np.arange(9, dtype=np.float32).reshape(3, 3)
    report = {}

    with pytest.raises(ValueError, match="score failed"):
        with opendde_profile.profile_prediction(report):
            actual = runner._predict(value)
            runner._score(actual)

    np.testing.assert_array_equal(np.asarray(actual), value @ value + 1.0)
    assert compiler.compile_or_get_cached is original_compile
    assert runner._predict is graph
    assert runner._score is failing_score
    assert [(phase["phase"], phase["status"]) for phase in report["phases"]] == [
        ("_predict", "ok"),
        ("_score", "error"),
    ]
    assert all(phase["seconds"] >= 0 for phase in report["phases"])
    executable = next(
        item for item in report["executables"] if item["module"] == "jit_graph"
    )
    assert executable["compile_or_load_seconds"] >= 0
    assert executable["memory"]["argument_size_in_bytes"] == value.nbytes
    assert executable["memory"]["output_size_in_bytes"] == value.nbytes


def test_device_observer_failure_cannot_replace_phase_error(monkeypatch):
    original = ValueError("original inference failure")

    def fail_predict(*args):
        raise original

    def fail_memory():
        raise RuntimeError("device memory is unavailable")

    monkeypatch.setattr(runner, "_predict", fail_predict)
    monkeypatch.setattr(opendde_profile, "_device_memory", fail_memory)
    report = {}
    with pytest.raises(ValueError) as caught:
        with opendde_profile.profile_prediction(report):
            runner._predict()
    assert caught.value is original
    phase, = report["phases"]
    assert phase["status"] == "error"
    assert phase["device_before"] == phase["device_after"] == {
        "unavailable": "RuntimeError"
    }
