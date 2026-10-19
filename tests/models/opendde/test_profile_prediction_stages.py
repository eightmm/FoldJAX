from __future__ import annotations

import json
import os
import subprocess
import sys
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
    from foldjax import cache, cli, oom
    from foldjax.backends import _tokamax_autotune
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
    selected_device = {"kind": "mock CUDA", "id": 1}
    cache_record = {
        "locations": {"shared": "/shared/model/weights/profile", "private": None},
        "loads": {"shared": {"hit": 1}},
    }
    tuning_record = {"sources": {"store": 1}}
    monkeypatch.setattr(opendde_profile, "_device", lambda request: selected_device)
    monkeypatch.setattr(
        opendde_profile, "_memory", lambda request: {"bytes_limit": 100}
    )
    monkeypatch.setattr(cache, "recorded", lambda: cache_record)
    monkeypatch.setattr(_tokamax_autotune, "recorded", lambda: tuning_record)
    monkeypatch.setenv("XLA_FLAGS", "--xla_dump_hlo_as_text")
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
    report = json.loads(output.read_text())
    assert report["device"] == selected_device
    assert report["effective_mem_fraction"] == expected
    assert report["environment"]["XLA_FLAGS"] == "--xla_dump_hlo_as_text"
    assert report["compilation_cache"] == cache_record
    assert report["kernel_tuning"] == tuning_record
    assert report["device_before"] == report["device_after"] == {"bytes_limit": 100}
    assert report["prediction_arguments"] == command[3:]


def test_importing_the_diagnostic_keeps_jax_uninitialized():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import bench.opendde_profile; assert 'jax' not in sys.modules",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_fresh_process_can_dump_cold_compiles_without_changing_cache_stores(tmp_path):
    program = """
import json
import sys
from pathlib import Path
import jax
import numpy as np
from bench.compilation_profile import observe_compilation
from foldjax.cache import compilation_cache_scope

root = Path(sys.argv[1])
shared, private = root / 'shared', root / 'private'
shared.mkdir(mode=0o700)
store = shared / '.tokamax-autotuning-v2'
store.mkdir(mode=0o700)
sentinel = store / ('a' * 64 + '.json')
sentinel.write_bytes(b'existing tuning store')
sentinel.chmod(0o600)
assert jax.config.jax_enable_compilation_cache is False

@jax.jit
def dump_probe(value):
    return value @ value + 1.0

report = {}
with (
    observe_compilation(report),
    compilation_cache_scope(shared, private_directory=private),
):
    assert jax.config.jax_enable_compilation_cache is False
    value = np.arange(9, dtype=np.float32).reshape(3, 3)
    np.testing.assert_array_equal(np.asarray(dump_probe(value)), value @ value + 1.0)
entry, = [e for e in report['executables'] if e['module'] == 'jit_dump_probe']
assert entry['backend_compile_and_load_calls'] == 1
assert not entry.get('cache_reads')
assert not list(shared.rglob('*-cache')) and not list(private.rglob('*-cache'))
assert sentinel.read_bytes() == b'existing tuning store'
print(json.dumps({'backend_compiled': True, 'persistent_cache_reads': 0}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", program, str(tmp_path)],
        env={
            **os.environ,
            "JAX_PLATFORMS": "cpu",
            "JAX_ENABLE_COMPILATION_CACHE": "false",
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


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
