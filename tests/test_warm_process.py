"""Fresh interpreters release one bucket before warming the next one."""

from __future__ import annotations

import dataclasses
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from foldjax import _warm_process, warm_buckets, warmup
from foldjax.schema import JobSource, PaddingConfig, PredictionRequest


def _install_worker_stubs():
    """Installed only by the test's child command, never by production code."""
    import jax
    import jax.numpy as jnp

    from foldjax import _gpu
    from foldjax.cache import CacheSnapshot

    original_device = _gpu.default_device

    def device(request=None):
        if request is not None and request.options.get("_test_split_af3"):
            from types import SimpleNamespace

            return SimpleNamespace(platform="gpu")
        return original_device(request)

    _gpu.default_device = device

    tiny = jax.jit(lambda value: value + 1)

    def resolve(request, *, jobs_root=None):
        plural = request.models is not None or request.inputs is not None
        return tuple(
            dataclasses.replace(
                request,
                model=model,
                models=None,
                input=source,
                inputs=None,
                weights=request.weights or source,
                seed=(3 if request.seed is None else request.seed)
                if request.seeds is None
                else None,
                output_dir=(
                    request.output_dir / model / source.stem
                    if plural and request.output_dir is not None
                    else request.output_dir
                ),
            )
            for model in request.resolved_models
            for source in request.resolved_inputs
        )

    def run(request, *, keep_output, output_root):
        preparing = warmup._AF3_PREPARATION.get()
        bucket = (request.padding.tokens or 8) if request.padding else 8
        scratch = warmup._PROCESS_SCRATCH.get()
        scratch.mkdir(parents=True, exist_ok=True)
        (scratch / "partial.cif").write_text("unfinished")
        event = {
            "pid": os.getpid(),
            "model": request.model,
            "bucket": bucket,
            "scratch": str(scratch),
            "output": str(request.output_dir),
            "preparing": preparing,
        }
        if request.options.get("events"):
            with Path(request.options["events"]).open("a") as handle:
                handle.write(json.dumps(event) + "\n")
        if request.options.get("block") and (
            not request.options.get("_test_block_verify") or not preparing
        ):
            while True:
                signal.pause()
        if request.options.get("kill_bucket") == bucket:
            os.kill(os.getpid(), signal.SIGKILL)
        if (
            request.options.get("fail_bucket") == bucket
            and request.options.get("fail_model", request.model) == request.model
            and (not request.options.get("_test_fail_verify") or not preparing)
        ):
            kind = request.options.get("fail_kind", "oom")
            if kind == "oom":
                raise MemoryError("mock memory admission refusal")
            if kind == "exit":
                raise SystemExit(23)
            if kind == "interrupt":
                raise KeyboardInterrupt
            raise ValueError("mock invalid profile")
        namespace = request.cache_dir / request.model / str(bucket)
        before, after = CacheSnapshot(), CacheSnapshot()
        if request.options.get("_test_real_cache"):
            import numpy as np

            from foldjax import cache

            cache.clear_record()
            before = cache.cache_snapshot(namespace)
            with cache.compilation_cache_scope(
                namespace,
                private_directory=Path(request.options["_test_private_cache"]),
                min_entry_size_bytes=0,
            ):
                assert jax.config.jax_persistent_cache_min_compile_time_secs == 0
                assert jax.config.jax_persistent_cache_min_entry_size_bytes == 0
                value = np.arange(bucket, dtype=np.float32)
                actual = tiny(value).block_until_ready()
                np.testing.assert_array_equal(np.asarray(actual), value + 1)
            compilation_cache = cache.recorded()
            after = cache.cache_snapshot(namespace)
            kernel_tuning = None
        else:
            if not preparing:
                tiny(jnp.zeros((bucket,))).block_until_ready()
            compilation_cache = {
                "loads": {"shared": {"success": 1}},
                "writes": {"private": {"success": 1}},
            }
            kernel_tuning = {"calls": [{"cache_read_tiers": {"private": 1}}]}
        if request.options.get("_test_split_af3"):
            source = (
                "measured"
                if preparing or request.options.get("_test_remeasure")
                else "store"
            )
            kernel_tuning.update(sources={source: 1})
            kernel_tuning["kernel_autotuning"] = request.options.get(
                "kernel_autotuning", "autotune"
            )
            kernel_tuning["calls"][0].update(
                store_identity_sha256="mock_store", model_input_abi_sha256="mock_abi"
            )
        verification = warmup._AF3_VERIFICATION.get()
        if verification is not None:
            verification.append({"cache_key": "fake", "source": "loaded"})
        print("python worker progress", flush=True)
        os.write(1, b"native worker progress\n")
        return warmup.CacheWarmResult(
            model=request.model,
            input=request.input,
            weights=request.weights,
            cache_dir=namespace,
            seed=request.resolved_seeds[0],
            seconds=0.01,
            before=before,
            after=after,
            samples=0 if preparing else 1,
            output_dir=request.output_dir if keep_output else None,
            shape_profile={
                "target": {"tokens": bucket},
                "worker_pid": os.getpid(),
                "jit_cache_size": tiny._cache_size(),
            },
            compilation_cache=compilation_cache,
            kernel_tuning=kernel_tuning,
            strategy="compile_only" if preparing else "execute_once",
            preparation={
                "module": "jit_apply_fn",
                "cache_key": "fake",
                "source": "written",
            }
            if preparing
            else None,
            verification={
                "main_cache_reads": list(verification),
                "kernel_miss_policy": request.options.get("kernel_autotuning"),
            }
            if verification is not None
            else None,
        )

    warmup.resolve_requests = resolve
    warmup._warm_resolved = run
    warm_buckets.device_summary = lambda _request: {
        "platform": "cpu",
        "device_kind": "mock CPU",
        "card_bytes": None,
        "pool_bytes": None,
    }


def _stub_command(request, result):
    command = _warm_process._worker_command_original(request, result)
    command[2] = command[2].replace(
        "from foldjax._warm_process import _worker_main; ",
        "import runpy; runpy.run_path(sys.argv[4])['_install_worker_stubs'](); "
        "from foldjax._warm_process import _worker_main; ",
    )
    return [*command, str(Path(__file__).resolve())]


@pytest.fixture
def process_request(tmp_path, monkeypatch):
    source = tmp_path / "input.json"
    source.write_text("{}")
    request = PredictionRequest(
        model="boltz2",
        input=source,
        cache_dir=tmp_path / "cache",
        msa="single",
        padding=PaddingConfig(msa=7),
        max_msa_depth=7,
        options={"events": str(tmp_path / "events.jsonl")},
    )
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    monkeypatch.setattr(
        _warm_process,
        "_worker_command_original",
        _warm_process._worker_command,
        raising=False,
    )
    monkeypatch.setattr(_warm_process, "_worker_command", _stub_command)

    def parent_device_query(*_args, **_kwargs):
        raise AssertionError("parent must not query the device or resolve a profile")

    monkeypatch.setattr(warm_buckets, "device_summary", parent_device_query)
    monkeypatch.setattr(warmup, "resolve_requests", parent_device_query)
    from foldjax import _gpu, memory_policy

    monkeypatch.setattr(_gpu, "default_device", parent_device_query)
    monkeypatch.setattr(memory_policy, "device_memory_budget", parent_device_query)
    return request


def _events(request):
    return [
        json.loads(line)
        for line in Path(request.options["events"]).read_text().splitlines()
    ]


def test_two_buckets_use_fresh_processes_and_keep_both_lengths_together(
    process_request,
    capfd,
):
    report = warm_buckets.warm_token_buckets(
        process_request, "8,16", isolation="process"
    )
    rows = report["models"][0]["buckets"]
    assert report["models"][0]["successful_buckets"] == [8, 16]
    assert report["device"]["platform"] == "cpu"
    assert rows[0]["process"]["pid"] != rows[1]["process"]["pid"]
    for row in rows:
        assert row["process"]["exit_code"] == 0
        assert {run["shape_profile"]["worker_pid"] for run in row["runs"]} == {
            row["process"]["pid"]
        }
        assert [run["shape_profile"]["jit_cache_size"] for run in row["runs"]] == [1, 1]
        assert row["verification"]["padded_targets_match"]
        assert row["verification"]["cache_namespace_match"]
        assert row["runs"][0]["compilation_cache"]["writes"]["private"]["success"] == 1
        assert row["runs"][1]["kernel_tuning"]["calls"][0]["cache_read_tiers"] == {
            "private": 1
        }
    assert all(
        not Path(event["scratch"]).parent.exists() for event in _events(process_request)
    )
    captured = capfd.readouterr()
    assert captured.out == ""
    assert "native worker progress" in captured.err
    assert "python worker progress" in captured.err


@pytest.mark.parametrize("retuned", [False, True])
def test_af3_preparation_exits_before_fresh_two_length_execution(
    process_request, retuned
):
    request = dataclasses.replace(
        process_request,
        model="alphafold3",
        weights=process_request.input,
        options={
            **process_request.options,
            "_test_split_af3": True,
            "_test_remeasure": retuned,
        },
    )
    report = warm_buckets.warm_token_buckets(request, "8", isolation="process")
    (row,) = report["models"][0]["buckets"]
    assert row["status"] == ("failed" if retuned else "success")
    if retuned:
        assert "incomplete persistent Tokamax coverage" in row["error"]
        assert report["models"][0]["successful_buckets"] == []
        assert row["runs"][0]["samples_executed"] == 1
        return
    preparation = row["preparation"]
    assert preparation["process"]["pid"] != row["process"]["pid"]
    assert [event["preparing"] for event in _events(request)] == [
        True,
        True,
        False,
        False,
    ]
    assert {event["pid"] for event in _events(request)[:2]} == {
        preparation["process"]["pid"]
    }
    assert {event["pid"] for event in _events(request)[2:]} == {row["process"]["pid"]}
    for pre, run in zip(preparation["runs"], row["runs"], strict=True):
        assert pre["samples_executed"] == 0 and pre["status"] == "prepared"
        assert run["samples_executed"] == 1
        assert run["preparation"]["complete_store_reused"] is not retuned
        assert run["preparation"]["verification_retuned"] is retuned
        assert run["verification"]["kernel_miss_policy"] == "error"
    assert row["verification"]["fresh_process"] == row["process"]
    assert all(not Path(event["scratch"]).parent.exists() for event in _events(request))


def test_af3_failed_preparation_does_not_start_verification(process_request):
    request = dataclasses.replace(
        process_request,
        model="alphafold3",
        weights=process_request.input,
        options={**process_request.options, "_test_split_af3": True, "fail_bucket": 8},
    )
    report = warm_buckets.warm_token_buckets(request, "8,16", isolation="process")
    (model,) = report["models"]
    assert model["successful_buckets"] == [] and model["stopped_on_oom"]
    (row,) = model["buckets"]
    assert row["status"] == "failed" and row["oom"]
    assert row["preparation"]["process"] == row["process"]
    assert [event["preparing"] for event in _events(request)] == [True]


def test_af3_only_verification_receives_the_retained_output(process_request, tmp_path):
    output = tmp_path / "kept"
    request = dataclasses.replace(
        process_request,
        model="alphafold3",
        weights=process_request.input,
        output_dir=output,
        options={**process_request.options, "_test_split_af3": True},
    )
    result = warmup.warm_cache(request, isolation="process")
    assert result.strategy == "compile_then_execute_fresh"
    assert result.samples == 1 and result.output_dir == output
    assert result.preparation["run"]["samples_executed"] == 0
    assert result.verification["kernel_miss_policy"] == "error"
    assert "kernel_autotuning" not in request.options
    assert [event["output"] for event in _events(request)] == ["None", str(output)]
    assert result.process["pid"] != result.preparation["process"]["pid"]


def test_external_af3_runtime_keeps_ordinary_process_warming(process_request):
    request = dataclasses.replace(
        process_request,
        model="alphafold3",
        weights=process_request.input,
        options={
            **process_request.options,
            "_test_split_af3": True,
            "source": "/external/checkout",
        },
    )
    result = warmup.warm_cache(request, isolation="process")
    assert result.strategy == "execute_once" and result.samples == 1
    assert result.preparation is None and result.verification is None
    assert [event["preparing"] for event in _events(request)] == [False]
    assert result.kernel_tuning["kernel_autotuning"] == "autotune"
    assert "kernel_autotuning" not in request.options


def _write_native_af3_jobs(source):
    source.write_text(
        json.dumps(
            [
                {
                    "name": name,
                    "modelSeeds": [7],
                    "sequences": [{"proteinChain": {"sequence": sequence, "count": 1}}],
                }
                for name, sequence in (("short", "ACDE"), ("long", "ACDEFGHI"))
            ]
        )
    )
    return source


def test_af3_native_multijob_fixture_is_accepted_by_existing_loader(
    tmp_path, alphafold3_runtime
):
    pytest.importorskip("zstandard")
    from foldjax.models.alphafold3 import build

    build.register_runtime()
    from alphafold3.common import folding_input

    source = _write_native_af3_jobs(tmp_path / "jobs.json")
    jobs = tuple(folding_input.load_fold_inputs_from_path(source))
    assert [job.name for job in jobs] == ["short", "long"]


@pytest.mark.parametrize("input_format", ["auto", "native"])
def test_native_af3_job_list_keeps_ordinary_warming(process_request, input_format):
    source = _write_native_af3_jobs(process_request.input)
    request = dataclasses.replace(
        process_request,
        model="alphafold3",
        weights=source,
        input_format=input_format,
        options={**process_request.options, "_test_split_af3": True},
    )
    result = warmup.warm_cache(request, isolation="process")
    assert result.strategy == "execute_once" and result.samples == 1
    assert result.preparation is None and result.verification is None
    assert [event["preparing"] for event in _events(request)] == [False]
    assert result.kernel_tuning["kernel_autotuning"] == "autotune"
    assert "kernel_autotuning" not in request.options


def test_neutral_plural_inputs_keep_split_af3_warming(process_request, tmp_path):
    second = tmp_path / "second.json"
    second.write_text('{"entities": []}')
    request = dataclasses.replace(
        process_request,
        model="alphafold3",
        weights=process_request.input,
        input=None,
        inputs=(process_request.input, second),
        options={**process_request.options, "_test_split_af3": True},
    )
    results = warmup.warm_cache(request, isolation="process")
    assert len(results) == 2
    assert all(result.strategy == "compile_then_execute_fresh" for result in results)
    assert [event["preparing"] for event in _events(request)] == [
        True,
        True,
        False,
        False,
    ]


@pytest.mark.parametrize("changed", ["namespace", "identity"])
def test_af3_preparation_cannot_verify_another_profile(changed):
    prepared = {
        "model": "alphafold3",
        "cache_dir": "original",
        "weights": "weights",
        "strategy": "compile_only",
        "preparation": {"cache_key": "key"},
        "kernel_tuning": {
            "calls": [
                {"store_identity_sha256": "store", "model_input_abi_sha256": "abi"}
            ]
        },
    }
    verified = json.loads(json.dumps(prepared))
    if changed == "namespace":
        verified["cache_dir"] = "different"
    else:
        verified["kernel_tuning"]["calls"][0]["store_identity_sha256"] = "different"
    with pytest.raises(
        warmup.PredictionError, match="different profiles|changed model input ABI"
    ):
        _warm_process._prepared_summary(verified, prepared, {"pid": 1}, {"pid": 2})


def test_af3_failed_fresh_execution_is_not_a_successful_warm(process_request):
    request = dataclasses.replace(
        process_request,
        model="alphafold3",
        weights=process_request.input,
        options={
            **process_request.options,
            "_test_split_af3": True,
            "_test_fail_verify": True,
            "fail_bucket": 8,
        },
    )
    report = warm_buckets.warm_token_buckets(request, "8,16", isolation="process")
    (model,) = report["models"]
    assert model["successful_buckets"] == []
    assert model["stopped_on_oom"]
    (row,) = model["buckets"]
    assert row["status"] == "failed" and row["oom"]
    assert row["preparation"]["process"]["pid"] != row["process"]["pid"]
    assert [event["preparing"] for event in _events(request)] == [True, True, False]
    assert all(not Path(event["scratch"]).parent.exists() for event in _events(request))


@pytest.mark.parametrize(
    "kind,stop", [("oom", True), ("oom", False), ("invalid", True)]
)
def test_bucket_failure_preserves_oom_policy_and_releases_the_worker(
    process_request, kind, stop
):
    request = dataclasses.replace(
        process_request,
        options={
            **process_request.options,
            "fail_bucket": 8,
            "fail_kind": kind,
        },
    )
    report = warm_buckets.warm_token_buckets(
        request, "8,16", isolation="process", stop_on_oom=stop
    )
    model = report["models"][0]
    assert model["first_failed_bucket"] == 8
    assert model["buckets"][0]["oom"] is (kind == "oom")
    assert model["stopped_on_oom"] is (kind == "oom" and stop)
    assert len(model["buckets"]) == (1 if kind == "oom" and stop else 2)
    assert all(not Path(event["scratch"]).parent.exists() for event in _events(request))


def test_abnormal_worker_exit_records_signal_and_cleans_parent_owned_scratch(
    process_request,
):
    request = dataclasses.replace(
        process_request, options={**process_request.options, "kill_bucket": 8}
    )
    report = warm_buckets.warm_token_buckets(request, "8,16", isolation="process")
    row = report["models"][0]["buckets"][0]
    assert row["status"] == "failed" and not row["oom"]
    assert row["process"]["exit_code"] == -signal.SIGKILL
    assert "SIGKILL" in row["error"]
    assert report["models"][0]["successful_buckets"] == [16]
    assert all(not Path(event["scratch"]).parent.exists() for event in _events(request))


def test_oom_stops_only_the_current_model(process_request):
    request = dataclasses.replace(
        process_request,
        model=None,
        models=("boltz2", "esmfold2"),
        options={**process_request.options, "fail_bucket": 8, "fail_model": "boltz2"},
    )
    report = warm_buckets.warm_token_buckets(request, "8,16", isolation="process")
    assert report["models"][0]["stopped_on_oom"]
    assert report["models"][1]["successful_buckets"] == [8, 16]


def test_exact_plural_warming_returns_objects_and_separates_models(
    process_request, tmp_path
):
    request = dataclasses.replace(
        process_request,
        model=None,
        models=("boltz2", "esmfold2"),
        output_dir=tmp_path / "kept",
        seeds=(7, 11),
    )
    results = warmup.warm_cache(request, isolation="process")
    assert isinstance(results, tuple) and len(results) == 2
    assert all(isinstance(item, warmup.CacheWarmResult) for item in results)
    assert results[0].process["pid"] != results[1].process["pid"]
    for item in results:
        assert item.seed == 7 and item.status == "populated"
        assert item.output_dir == request.output_dir / item.model / request.input.stem
        assert item.summary()["process"] == item.process


def test_exact_workers_restore_a_real_shared_jax_executable_after_fresh_exec(
    process_request,
    monkeypatch,
    tmp_path,
):
    """Model execution is a tiny CPU stub; cache I/O and telemetry are real."""
    from foldjax import cache

    monkeypatch.setenv(cache.TRUST_SHARED_COMPILE_CACHE_ENV, "1")
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    request = dataclasses.replace(
        process_request,
        options={
            **process_request.options,
            "_test_real_cache": True,
            "_test_private_cache": str(private),
        },
    )
    first = warmup.warm_cache(request, isolation="process")
    assert first.compilation_cache["writes"]["shared"]["success"] >= 1
    assert first.compilation_cache["compiles"]["success"] >= 1
    assert first.after.files > first.before.files
    entries = {path.name: path.read_bytes() for path in first.cache_dir.glob("*-cache")}
    assert entries

    second = warmup.warm_cache(request, isolation="process")
    assert first.process["pid"] != second.process["pid"]
    assert first.cache_dir == second.cache_dir
    assert second.compilation_cache["loads"]["shared"]["success"] >= 1
    assert not second.compilation_cache["compiles"]
    assert not second.compilation_cache["writes"]
    assert second.compilation_cache["locations"] == {
        "shared": str(first.cache_dir),
        "private": str(private),
    }
    assert entries == {
        path.name: path.read_bytes() for path in second.cache_dir.glob("*-cache")
    }
    assert private.stat().st_mode & 0o777 == 0o700
    assert not list(private.glob("*-cache"))


@pytest.mark.parametrize(
    "kind,error", [("exit", SystemExit), ("interrupt", KeyboardInterrupt)]
)
def test_worker_cancellation_exceptions_are_not_bucket_failures(
    process_request, kind, error
):
    request = dataclasses.replace(
        process_request,
        options={
            **process_request.options,
            "fail_bucket": 8,
            "fail_kind": kind,
        },
    )
    with pytest.raises(error) as failure:
        warm_buckets.warm_token_buckets(request, "8,16", isolation="process")
    if kind == "exit":
        assert failure.value.code == 23
    assert len(_events(request)) == 1
    assert not Path(_events(request)[0]["scratch"]).parent.exists()


def test_request_transport_preserves_known_types_and_rejects_host_only_options(
    process_request,
):
    request = dataclasses.replace(
        process_request,
        source=JobSource(process_request.input, 0, "test"),
        seeds=(2, 3),
        representations=("single",),
        options={"nested": {"values": [True, None, "test", 7]}},
    )
    assert (
        _warm_process._request_from_packet(_warm_process.request_packet(request))
        == request
    )
    with pytest.raises(TypeError, match="isolation='in-process'"):
        warmup.warm_cache(
            dataclasses.replace(request, options={"callback": lambda: None}),
            isolation="process",
        )


def test_production_worker_reports_typed_validation_error_without_gpu_init(
    process_request,
    monkeypatch,
):
    monkeypatch.setattr(
        _warm_process,
        "_worker_command",
        _warm_process._worker_command_original,
    )
    packet = _warm_process.request_packet(process_request)
    process_request.input.unlink()
    with pytest.raises(FileNotFoundError, match="input does not exist") as failure:
        _warm_process.run_worker({"kind": "exact", "request": packet})
    assert failure.value.warm_process["pid"] != os.getpid()
    assert failure.value.warm_process["exit_code"] == 1


@pytest.mark.parametrize("mismatch", [False, True])
def test_invalid_worker_report_cleans_transport_directory(
    process_request,
    monkeypatch,
    tmp_path,
    mismatch,
):
    marker = tmp_path / "worker-root"

    def command(_request, result):
        packet = {
            "version": 1 if mismatch else 99,
            "package_directory": "wrong-source",
            "ok": True,
            "result": None,
        }
        return [
            sys.executable,
            "-c",
            "import pathlib,sys; pathlib.Path(sys.argv[1]).write_text(sys.argv[2]); "
            "pathlib.Path(sys.argv[3]).write_text(str(pathlib.Path(sys.argv[1]).parent))",
            str(result),
            json.dumps(packet),
            str(marker),
        ]

    monkeypatch.setattr(_warm_process, "_worker_command", command)
    message = "another installation" if mismatch else "without a valid report"
    with pytest.raises(Exception, match=message) as failure:
        warmup.warm_cache(process_request, isolation="process")
    assert failure.value.warm_process["exit_code"] == 0
    assert not Path(marker.read_text()).exists()


class _SpawnedWorker:
    pid = 12345

    def __init__(self):
        self.events = []
        self.terminated = False
        self.finished = False

    def poll(self):
        return 0 if self.terminated or self.finished else None

    def terminate(self):
        self.events.append("terminate")
        self.terminated = True

    def wait(self, timeout=None):
        self.events.append(("wait", timeout))
        if timeout is None:
            self.finished = True
        return 0


@pytest.mark.parametrize(
    "kind,fields",
    [
        ("exact", {"ok": 1, "result": []}),
        ("exact", {"ok": True}),
        ("exact", {"ok": True, "result": {}}),
        ("bucket", {"ok": True, "result": []}),
        ("exact", {"ok": False, "error": None}),
        ("bucket", {"ok": False, "error": []}),
        ("exact", {"ok": False, "error": {"type": 1, "message": "failed"}}),
        ("bucket", {"ok": False, "error": {"type": "ValueError", "message": None}}),
    ],
)
def test_malformed_same_version_envelopes_keep_provenance_and_cleanup(
    monkeypatch,
    kind,
    fields,
):
    roots = []

    def spawn(command, **_kwargs):
        result = Path(command[-1])
        roots.append(result.parent)
        _warm_process._write_packet(
            result,
            {
                "version": 1,
                "package_directory": str(_warm_process._PACKAGE_DIRECTORY),
                **fields,
            },
        )
        return _SpawnedWorker()

    monkeypatch.setattr(_warm_process.subprocess, "Popen", spawn)
    with pytest.raises(
        warmup.PredictionError, match="invalid cache-warm worker report"
    ) as failure:
        _warm_process.run_worker({"kind": kind})
    assert failure.value.warm_process == {"pid": 12345, "exit_code": 0}
    assert len(roots) == 1 and not roots[0].exists()


@pytest.mark.skipif(os.name != "posix", reason="POSIX Python signal delivery")
@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM])
def test_cancellation_during_spawn_keeps_the_returned_handle_for_cleanup(
    monkeypatch,
    signum,
):
    worker = _SpawnedWorker()
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}

    def spawn(*_args, **_kwargs):
        worker.events.append("created")
        signal.raise_signal(signum)
        worker.events.append("returned")
        return worker

    monkeypatch.setattr(_warm_process.subprocess, "Popen", spawn)
    expected = KeyboardInterrupt if signum == signal.SIGINT else SystemExit
    with pytest.raises(expected) as failure:
        _warm_process.run_worker({"kind": "probe"})
    if signum == signal.SIGTERM:
        assert failure.value.code == 143
    assert worker.events == ["created", "returned", "terminate", ("wait", 10)]
    assert {sig: signal.getsignal(sig) for sig in previous} == previous


@pytest.mark.skipif(os.name != "posix", reason="POSIX Python signal delivery")
@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM])
def test_spawn_replays_a_custom_handler_after_creation_and_restores_it(
    monkeypatch,
    signum,
):
    worker = _SpawnedWorker()
    previous = signal.getsignal(signum)

    def handler(received, _frame):
        assert received == signum
        worker.events.append("handler")

    def spawn(command, **_kwargs):
        worker.events.append("created")
        signal.raise_signal(signum)
        worker.events.append("returned")
        _warm_process._write_packet(
            Path(command[-1]),
            {
                "version": 1,
                "package_directory": str(_warm_process._PACKAGE_DIRECTORY),
                "ok": True,
                "result": "finished",
            },
        )
        return worker

    signal.signal(signum, handler)
    monkeypatch.setattr(_warm_process.subprocess, "Popen", spawn)
    try:
        result, provenance = _warm_process.run_worker({"kind": "probe"})
        assert result == "finished" and provenance["exit_code"] == 0
        assert worker.events == ["created", "returned", "handler", ("wait", None)]
        assert signal.getsignal(signum) is handler
    finally:
        signal.signal(signum, previous)


def test_spawn_error_is_preserved_and_handlers_are_restored(monkeypatch):
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    error = OSError("cannot create worker")

    def spawn(*_args, **_kwargs):
        raise error

    monkeypatch.setattr(_warm_process.subprocess, "Popen", spawn)
    with pytest.raises(OSError) as failure:
        _warm_process.run_worker({"kind": "probe"})
    assert failure.value is error
    assert {sig: signal.getsignal(sig) for sig in previous} == previous


@pytest.mark.parametrize("bucket", [False, True])
def test_cli_defaults_to_process_without_resolving_in_the_parent(
    process_request,
    monkeypatch,
    capfd,
    bucket,
):
    from foldjax import cli

    def no_parent_resolution(*_args, **_kwargs):
        raise AssertionError("CLI must leave device-sensitive resolution in the child")

    monkeypatch.setattr(cli, "resolve_requests", no_parent_resolution)
    args = [
        "cache",
        "warm",
        "--model",
        "boltz2",
        "--cache-dir",
        str(process_request.cache_dir),
        "--msa",
        "single",
        "--option",
        f"events={process_request.options['events']}",
    ]
    args += (
        ["--token-buckets", "8", "--json"]
        if bucket
        else [
            "--input",
            str(process_request.input),
            "--input-format",
            "native",
            "--padding",
        ]
    )
    assert cli.main(args) == 0
    captured = capfd.readouterr()
    result = json.loads(captured.out)
    if bucket:
        assert result["isolation"] == "process"
        assert result["models"][0]["successful_buckets"] == [8]
    else:
        assert result["process"]["pid"] != os.getpid()
    assert "native worker progress" in captured.err


@pytest.mark.skipif(os.name != "posix", reason="POSIX cancellation and child reaping")
@pytest.mark.parametrize("model", ["boltz2", "alphafold3"])
def test_parent_sigterm_reaps_its_worker_and_removes_scratch(tmp_path, model):
    marker = tmp_path / "events.jsonl"
    source = tmp_path / "input.json"
    source.write_text("{}")
    code = """
import runpy,sys
from pathlib import Path
from foldjax import _warm_process, warm_buckets
from foldjax.schema import PredictionRequest
namespace=runpy.run_path(sys.argv[1])
_warm_process._worker_command_original=_warm_process._worker_command
_warm_process._worker_command=namespace['_stub_command']
request=PredictionRequest(model=sys.argv[5], input=Path(sys.argv[2]),
 cache_dir=Path(sys.argv[3]), msa='single',
 options={'events':sys.argv[4], 'block':True, '_test_block_verify':True,
 '_test_split_af3':sys.argv[5]=='alphafold3'})
warm_buckets.warm_token_buckets(request,'8',isolation='process')
"""
    driver = subprocess.Popen(
        [
            sys.executable,
            "-c",
            code,
            __file__,
            str(source),
            str(tmp_path / "cache"),
            str(marker),
            model,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={**os.environ, "JAX_PLATFORMS": "cpu"},
    )
    event = None
    try:
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            if marker.exists() and marker.read_text().strip():
                events = [json.loads(line) for line in marker.read_text().splitlines()]
                active = [item for item in events if not item["preparing"]]
                if active:
                    event = active[0]
                    break
            if driver.poll() is not None:
                pytest.fail(driver.communicate()[1])
            time.sleep(0.05)
        assert event is not None, "worker did not reach the blocking prediction"
        driver.terminate()
        _, stderr = driver.communicate(timeout=15)
        assert driver.returncode == 143, stderr
        with pytest.raises(ProcessLookupError):
            os.kill(event["pid"], 0)
        assert not Path(event["scratch"]).parent.exists()
    finally:
        if driver.poll() is None:
            driver.kill()
        driver.wait(timeout=10)
