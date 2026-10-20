from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import jax
import numpy as np
import pytest
from jax._src import compilation_cache, compiler

from bench import compilation_profile, opendde_profile, prediction_profile
from foldjax import cache
from foldjax._jax_cache_tiers import LayeredCache


@pytest.mark.parametrize("fingerprints", [False, True])
def test_observer_loads_real_layered_cache_and_preserves_keys_and_outputs(
    tmp_path, trust_ancestors_above_tmp_path, fingerprints
):
    @jax.jit
    def observed_graph(value):
        return value @ value + 3.0

    value = np.arange(16, dtype=np.float32).reshape(4, 4)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    reports = [{}, {}]
    for report in reports:
        jax.clear_caches()
        with (
            compilation_profile.observe_compilation(
                report, fingerprint_hlo=fingerprints
            ),
            cache.compilation_cache_scope(
                tmp_path / "shared", private_directory=tmp_path / "private"
            ),
        ):
            actual = observed_graph(value)
            np.testing.assert_array_equal(np.asarray(actual), value @ value + 3.0)

    entries = [
        next(e for e in r["executables"] if e["module"] == "jit_observed_graph")
        for r in reports
    ]
    first, second = [entry["cache_reads"] for entry in entries]
    assert first and not first[0]["hit"]
    assert second and second[0]["hit"]
    assert first[0]["key"] == second[0]["key"]
    assert second[0]["layered_get_calls"] == 1
    assert second[0]["layered_get_seconds"] >= 0
    assert second[0]["decompress_seconds"] >= 0
    assert second[0]["load_residual_seconds"] >= 0
    assert "backend_compile_and_load_seconds" in entries[0]
    assert "backend_compile_and_load_seconds" not in entries[1]
    assert entries[0]["memory"] == entries[1]["memory"]
    if fingerprints:
        assert entries[0]["hlo_fingerprints"] == entries[1]["hlo_fingerprints"]
        hashes = entries[0]["hlo_fingerprints"]
        assert len(hashes["pre_compile_mlir_sha256"]) == 64
        assert len(hashes["compile_options_sha256"]) == 64
        assert len(hashes["optimized_hlo_sha256"]) == 1
        assert len(hashes["optimized_hlo_sha256"][0]) == 64
        assert "device_before_compile_or_load" in entries[0]
        assert "device_after_compile_or_load" in entries[1]
    else:
        assert all("hlo_fingerprints" not in entry for entry in entries)


@pytest.mark.parametrize("fingerprints", [False, True])
def test_observer_restores_every_hook_and_retains_failed_compilation(
    monkeypatch, fingerprints
):
    def fail(*args, **kwargs):
        raise RuntimeError("compile failed")

    monkeypatch.setattr(compiler, "compile_or_get_cached", fail)
    owners = [
        (compiler, "compile_or_get_cached"),
        (compiler, "_cache_read"),
        (compiler, "backend_compile_and_load"),
        (compilation_cache, "decompress_executable"),
        (LayeredCache, "get"),
    ]
    originals = [getattr(owner, name) for owner, name in owners]
    report = {}
    module = SimpleNamespace(operation=SimpleNamespace(attributes={"sym_name": "x"}))
    with pytest.raises(RuntimeError, match="compile failed"):
        with compilation_profile.observe_compilation(
            report, fingerprint_hlo=fingerprints
        ):
            compiler.compile_or_get_cached(None, module)
    assert [getattr(owner, name) for owner, name in owners] == originals
    assert report["executables"][0]["status"] == "error"
    assert report["executables"][0]["error_type"] == "RuntimeError"


@pytest.mark.parametrize("failure", [False, True])
@pytest.mark.parametrize("fingerprints", [False, True])
def test_native_profile_enters_after_cli_allocator_policy_and_flushes_on_error(
    tmp_path, monkeypatch, failure, fingerprints
):
    from foldjax import cli, oom
    from foldjax.schema import BatchReport

    monkeypatch.delenv(oom.CURRENT_FRACTION_ENV, raising=False)
    monkeypatch.delenv(oom.FRACTION_ENV, raising=False)
    monkeypatch.setattr(cli, "_apply_gc_threshold", lambda: None)
    monkeypatch.setattr(cli, "_enable_release_reclaim", lambda: None)
    monkeypatch.setattr(
        cli, "_request", lambda *a, **kw: SimpleNamespace(models=None, inputs=None)
    )
    monkeypatch.setattr(prediction_profile, "_memory", lambda *a: {})
    monkeypatch.setattr(prediction_profile, "_device", lambda *a: {})
    events = []

    @contextmanager
    def observe(report, **options):
        assert options == ({"fingerprint_hlo": True} if fingerprints else {})
        assert oom.mem_fraction() == 0.8
        events.append("enter")
        try:
            yield
        finally:
            events.append("exit")

    def predict(*args, **kwargs):
        assert events == ["enter"]
        if failure:
            raise RuntimeError("prediction failed")
        return BatchReport()

    monkeypatch.setattr(prediction_profile, "observe_compilation", observe)
    monkeypatch.setattr(cli, "_run_predictions", predict)
    output = tmp_path / "diagnostics.json"
    command = [
        "--profile-out",
        str(output),
        "--",
        "predict",
        "--model",
        "boltz2",
        "--quiet",
        "--mem-fraction",
        "0.8",
    ]
    if fingerprints:
        command.insert(2, "--hlo-fingerprints")
    if failure:
        with pytest.raises(RuntimeError, match="prediction failed"):
            prediction_profile.main(command)
    else:
        assert prediction_profile.main(command) == 0
    assert events == ["enter", "exit"]
    assert cli._run_predictions is predict
    report = json.loads(output.read_text())
    assert report["prediction_seconds"] >= 0
    assert ("error_type" in report) == failure
    assert ("exit_code" in report) != failure


@pytest.mark.parametrize("profiler", [prediction_profile, opendde_profile])
@pytest.mark.parametrize("failure_kind", ["exception", "exit_code", "success"])
def test_report_io_failure_preserves_original_failure(
    tmp_path, monkeypatch, capsys, profiler, failure_kind
):
    from foldjax import cli

    original = RuntimeError("original prediction failure")

    def run(*args):
        if failure_kind == "exception":
            raise original
        return 1 if failure_kind == "exit_code" else 0

    def fail_write(*args, **kwargs):
        raise OSError("diagnostic filesystem full")

    monkeypatch.setattr(cli, "main", run)
    monkeypatch.setattr(Path, "write_text", fail_write)
    args = [
        "--profile-out",
        str(tmp_path / "profile.json"),
        "--",
        "predict",
        "--model",
        "opendde",
    ]
    if failure_kind == "exception":
        with pytest.raises(RuntimeError) as caught:
            profiler.main(args)
        assert caught.value is original
    elif failure_kind == "exit_code":
        assert profiler.main(args) == 1
    else:
        with pytest.raises(OSError, match="filesystem full"):
            profiler.main(args)
    if failure_kind != "success":
        assert "retaining the original prediction failure" in capsys.readouterr().err
