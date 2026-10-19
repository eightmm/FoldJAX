"""Private JSON transport for fresh-interpreter cache warming."""

from __future__ import annotations

import dataclasses
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any

from foldjax.schema import (
    JobSource,
    PaddingConfig,
    PredictionError,
    PredictionOutputError,
    PredictionRequest,
)

_PACKET_VERSION = 1
_MAX_PACKET_BYTES = 64 << 20
_PACKAGE_DIRECTORY = Path(__file__).resolve().parent
_PATH_FIELDS = {"input", "weights", "output_dir", "cache_dir", "template_dir"}


def _json_options(value: Any) -> Any:
    if value is None or type(value) in (bool, int, float, str):
        return value
    if isinstance(value, list):
        return [_json_options(item) for item in value]
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return {key: _json_options(item) for key, item in value.items()}
    raise TypeError(
        "process-isolated warming requires JSON options; use isolation='in-process' "
        "for callbacks, device objects or other host-only values"
    )


def request_packet(request: PredictionRequest) -> dict[str, Any]:
    packet = {
        field.name: getattr(request, field.name)
        for field in dataclasses.fields(request)
    }
    for name in _PATH_FIELDS:
        if packet[name] is not None:
            packet[name] = str(packet[name])
    for name in ("inputs", "models", "seeds", "representations"):
        if isinstance(packet[name], tuple):
            packet[name] = [
                str(value) if isinstance(value, Path) else value
                for value in packet[name]
            ]
    packet["padding"] = (
        request.padding.summary() if request.padding is not None else None
    )
    packet["source"] = request.source.summary() if request.source is not None else None
    packet["options"] = _json_options(dict(request.options))
    return packet


def _request_from_packet(packet: dict[str, Any]) -> PredictionRequest:
    values = dict(packet)
    if values.get("padding") is not None:
        values["padding"] = PaddingConfig(**values["padding"])
    if values.get("source") is not None:
        values["source"] = JobSource(**values["source"])
    return PredictionRequest(**values)


def _write_packet(path: Path, packet: dict[str, Any]) -> None:
    payload = json.dumps(packet, allow_nan=False).encode()
    if len(payload) > _MAX_PACKET_BYTES:
        raise ValueError("cache-warm worker report exceeds its transport limit")
    fd, temporary = tempfile.mkstemp(prefix=".packet-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _read_packet(path: Path) -> dict[str, Any]:
    if path.stat().st_size > _MAX_PACKET_BYTES:
        raise ValueError("cache-warm worker report exceeds its transport limit")
    packet = json.loads(path.read_text())
    if not isinstance(packet, dict) or packet.get("version") != _PACKET_VERSION:
        raise ValueError("invalid cache-warm worker report")
    return packet


def _validate_report(packet: dict[str, Any], kind: str) -> None:
    if type(packet.get("ok")) is not bool:
        raise ValueError("ok must be a boolean")
    if packet["ok"]:
        if "result" not in packet:
            raise ValueError("successful report has no result")
        expected = {"exact": list, "bucket": dict, "prepare": dict}.get(kind)
        if expected is not None and not isinstance(packet["result"], expected):
            raise ValueError(f"{kind} result must be a {expected.__name__}")
    else:
        error = packet.get("error")
        if not isinstance(error, dict) or not all(
            isinstance(error.get(name), str) for name in ("type", "message")
        ):
            raise ValueError("failed report needs an error type and message")


def _worker_command(request: Path, result: Path) -> list[str]:
    # Redirect the OS descriptor before importing FoldJAX: native libraries may
    # write beneath Python's stdout redirection. Pin imports to this installation.
    bootstrap = (
        "import os,sys; os.dup2(2,1); sys.path.insert(0,sys.argv[1]); "
        "from foldjax._warm_process import _worker_main; "
        "sys.exit(_worker_main(sys.argv[2],sys.argv[3]))"
    )
    return [
        sys.executable,
        "-c",
        bootstrap,
        str(_PACKAGE_DIRECTORY.parent),
        str(request),
        str(result),
    ]


def _stop_child(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


@contextmanager
def _defer_spawn_signals():
    """Keep the child handle reachable before replaying Python cancellation.

    Python signal handlers can only be installed in the main thread. Ignore
    and OS-default handlers retain their existing semantics; the warm scope
    already supplies a Python SIGTERM handler for the normal default case.
    """
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    pending = []
    handlers = {}

    def defer(signum, frame):
        pending.append((signum, frame))

    try:
        with ExitStack() as restore:
            for signum in (signal.SIGINT, signal.SIGTERM):
                previous = signal.getsignal(signum)
                if callable(previous):
                    handlers[signum] = previous
                    restore.callback(signal.signal, signum, previous)
                    signal.signal(signum, defer)
            yield
    finally:
        for signum, frame in pending:
            handlers[signum](signum, frame)


def _worker_error(
    message: str,
    *,
    process: dict[str, int],
    oom: bool = False,
    kind: str = "PredictionError",
):
    # Preserve common public API failures without importing or unpickling a
    # worker-provided class. Unknown native exceptions keep their type in text.
    classes = {
        cls.__name__: cls
        for cls in (
            PredictionError,
            PredictionOutputError,
            MemoryError,
            ValueError,
            TypeError,
            RuntimeError,
            NotImplementedError,
            OSError,
            FileNotFoundError,
            PermissionError,
            IsADirectoryError,
            NotADirectoryError,
        )
    }
    error = classes.get(kind, PredictionError)(
        message if kind in classes else f"{kind}: {message}"
    )
    error.warm_process = process
    error.warm_oom = oom
    return error


def _run_worker_once(payload: dict[str, Any]) -> tuple[Any, dict[str, int]]:
    """Run only our child; cancellation also removes its discarded outputs."""
    from foldjax.warmup import _warm_directory

    with _warm_directory(
        Path(tempfile.gettempdir()), prefix=".foldjax-warm-process-"
    ) as root:
        request_file, result_file = root / "request.json", root / "result.json"
        _write_packet(
            request_file,
            {
                "version": _PACKET_VERSION,
                **payload,
                "scratch_root": str(root / "outputs"),
                "jobs_root": str(root / "jobs"),
            },
        )
        process = None
        try:
            with _defer_spawn_signals():
                process = subprocess.Popen(
                    _worker_command(request_file, result_file), stdin=subprocess.DEVNULL
                )
            returncode = process.wait()
        finally:
            if process is not None:
                _stop_child(process)
        provenance = {"pid": process.pid, "exit_code": returncode}
        try:
            packet = _read_packet(result_file)
        except (OSError, ValueError) as error:
            if returncode < 0:
                detail = f"signal {signal.Signals(-returncode).name}"
            else:
                detail = f"exit status {returncode}"
            raise _worker_error(
                f"cache-warm worker ended with {detail} without a valid report",
                process=provenance,
            ) from error
        if packet.get("package_directory") != str(_PACKAGE_DIRECTORY):
            raise _worker_error(
                "cache-warm worker imported another installation", process=provenance
            )
        try:
            _validate_report(packet, payload["kind"])
        except ValueError as error:
            raise _worker_error(
                f"invalid cache-warm worker report: {error}", process=provenance
            ) from error
        if packet.get("ok") is True and returncode == 0:
            return packet["result"], provenance
        detail = packet.get("error", {})
        kind, message = (
            detail.get("type", "PredictionError"),
            detail.get("message", "worker failed"),
        )
        if kind == "SystemExit":
            raise SystemExit(detail.get("code", returncode))
        if kind == "KeyboardInterrupt":
            raise KeyboardInterrupt
        raise _worker_error(
            message, kind=kind, process=provenance, oom=bool(detail.get("oom", False))
        )


def _prepared_summary(summary, prepared, preparation_process, verification_process):
    if prepared.get("strategy") != "compile_only" or not prepared.get("preparation"):
        raise PredictionError("AF3 preparation did not confirm a model artifact")
    if any(
        summary.get(name) != prepared.get(name)
        for name in ("model", "cache_dir", "weights")
    ):
        raise PredictionError(
            "AF3 preparation and verification selected different profiles"
        )

    def identities(run):
        return {
            (call.get("store_identity_sha256"), call.get("model_input_abi_sha256"))
            for call in (run.get("kernel_tuning") or {}).get("calls", ())
        }

    prepared_identities = identities(prepared)
    if not prepared_identities or any(None in item for item in prepared_identities):
        raise PredictionError("AF3 preparation has no verifiable Tokamax call identity")
    if not prepared_identities <= identities(summary):
        raise PredictionError(
            "AF3 preparation and verification changed model input ABI"
        )
    tuning = summary.get("kernel_tuning") or {}
    sources = tuning.get("sources") or {}
    reads = (summary.get("verification") or {}).get("main_cache_reads", ())
    if prepared["preparation"].get("cache_key") not in {
        read.get("cache_key") for read in reads if read.get("source") == "loaded"
    }:
        raise PredictionError(
            "AF3 verification did not load its exact prepared main key"
        )
    policy = (prepared.get("kernel_tuning") or {}).get("kernel_autotuning", "autotune")
    if policy == "autotune" and set(sources) != {"store"}:
        raise PredictionError(
            "AF3 verification found incomplete persistent Tokamax coverage; "
            "fresh tuning cannot verify the prepared cache"
        )
    cache = prepared.get("compilation_cache") or {}
    published = any(
        events.get("success", 0) for events in cache.get("writes", {}).values()
    )
    published |= any(
        call.get("cache_write_tiers")
        for call in (prepared.get("kernel_tuning") or {}).get("calls", ())
    )
    summary.update(
        strategy="compile_then_execute_fresh",
        seconds=round(summary["seconds"] + prepared["seconds"], 3),
        preparation={
            "process": preparation_process,
            "run": prepared,
            "published": bool(published),
            "verification_process": verification_process,
            "verification_retuned": bool(sources.get("measured")),
            "verification_sources": sources,
            "complete_store_reused": bool(sources) and set(sources) == {"store"},
        },
    )
    summary["cache"]["before"] = prepared["cache"]["before"]
    for name in ("files", "bytes"):
        summary["cache"][f"new_{name}"] = max(
            0, summary["cache"]["after"][name] - prepared["cache"]["before"][name]
        )
    if published:
        summary["status"] = "populated"


def run_worker(payload: dict[str, Any]) -> tuple[Any, dict[str, int]]:
    """Release AF3 preparation's CUDA context before verified model execution."""
    request = payload.get("request", {})
    models = request.get("models") or [request.get("model")]
    model = payload.get("model") or (models[0] if len(models) == 1 else None)
    if payload.get("kind") not in {"exact", "bucket"} or model != "alphafold3":
        return _run_worker_once(payload)
    started = time.perf_counter()
    prepared, preparation_process = _run_worker_once(
        {
            **payload,
            "kind": "prepare",
            "operation": payload["kind"],
            "request": {**request, "output_dir": None},
        }
    )
    if prepared.get("enabled") and payload["kind"] == "bucket":
        result = prepared["result"]
        if result["row"]["status"] != "success":
            result["row"]["preparation"] = {
                "process": preparation_process,
                "runs": result["row"]["runs"],
            }
            return result, preparation_process
    verification_payload = payload
    if prepared.get("enabled"):
        preparation = prepared["result"]
        runs = preparation if payload["kind"] == "exact" else preparation["row"]["runs"]
        options = dict(request.get("options", {}))
        if options.get("kernel_autotuning", "autotune") == "autotune":
            options["kernel_autotuning"] = "error"
        verification_payload = {
            **payload,
            "request": {**request, "options": options},
            "verify_keys": [run["preparation"]["cache_key"] for run in runs],
        }
    try:
        verified, verification_process = _run_worker_once(verification_payload)
    except Exception as error:
        error.warm_preparation = {"process": preparation_process, **prepared}
        if payload["kind"] == "exact" or not prepared.get("enabled"):
            raise
        from foldjax.warm_buckets import _bucket_row, _failed_row

        row = _bucket_row(payload["bucket"], payload["previous"])
        _failed_row(row, error)
        row["preparation"] = error.warm_preparation
        row["seconds"] = round(time.perf_counter() - started, 3)
        return {
            "row": row,
            "device": prepared.get("result", {}).get("device", {}),
        }, error.warm_process
    if not prepared.get("enabled"):
        return verified, verification_process
    preparation = prepared["result"]
    try:
        if payload["kind"] == "exact":
            if len(verified) != len(preparation):
                raise PredictionError(
                    "AF3 preparation and verification resolved different jobs"
                )
            for summary, before in zip(verified, preparation, strict=True):
                _prepared_summary(
                    summary, before, preparation_process, verification_process
                )
        else:
            row, before = verified["row"], preparation["row"]
            row["preparation"] = {
                "process": preparation_process,
                "runs": before["runs"],
            }
            if row["status"] == "success":
                if len(row["runs"]) != len(before["runs"]):
                    raise PredictionError(
                        "AF3 preparation and verification resolved different lengths"
                    )
                for summary, pre in zip(row["runs"], before["runs"], strict=True):
                    _prepared_summary(
                        summary, pre, preparation_process, verification_process
                    )
                row["verification"]["fresh_process"] = verification_process
            row["seconds"] = round(time.perf_counter() - started, 3)
    except Exception as error:
        error.warm_preparation = {"process": preparation_process, **prepared}
        error.warm_process = verification_process
        if payload["kind"] == "exact":
            raise
        from foldjax.warm_buckets import _failed_row

        _failed_row(verified["row"], error)
    return verified, verification_process


def _has_native_job_list(request: PredictionRequest) -> bool:
    if request.input_format == "foldjax":
        return False
    from foldjax.input import read_job_document

    # The native AF3 loader yields one Server fold job per top-level list
    # element. Leave its validation authoritative; inspect only the container.
    for path in request.resolved_inputs:
        try:
            document = read_job_document(path)
        except (OSError, ValueError):
            continue
        if isinstance(document, list) and len(document) > 1:
            return True
    return False


def _worker_main(request_file: str, result_file: str) -> int:
    from foldjax import oom
    from foldjax.cli import _apply_gc_threshold
    from foldjax.warmup import (
        _prepare_af3_model,
        _scratch_root,
        _verify_af3_artifacts,
        warm_cache,
    )

    _apply_gc_threshold()
    code = 0
    try:
        packet = _read_packet(Path(request_file))
        request = _request_from_packet(packet["request"])

        def execute(operation):
            if operation == "exact":
                warmed = warm_cache(request)
                results = warmed if isinstance(warmed, tuple) else (warmed,)
                return [item.summary() for item in results]
            elif operation == "bucket":
                from foldjax.warm_buckets import _warm_bucket

                result = _warm_bucket(
                    request,
                    model=packet["model"],
                    bucket=packet["bucket"],
                    previous=packet["previous"],
                    jobs=Path(packet["jobs_root"]),
                )
                if result["row"]["status"] != "success":
                    from foldjax.backends._tokamax_autotune import (
                        recorded as tuning_record,
                    )
                    from foldjax.cache import recorded as cache_record

                    result["row"]["failed_execution"] = {
                        "compilation_cache": cache_record(),
                        "kernel_tuning": tuning_record(),
                    }
                return result
            else:
                raise ValueError("unknown cache-warm worker operation")

        with _scratch_root(Path(packet["scratch_root"])):
            if packet["kind"] == "prepare":
                from foldjax._gpu import default_device

                selected = dataclasses.replace(request, model="alphafold3", models=None)
                if request.options.get("source") is not None:
                    result = {"enabled": False, "reason": "external runtime"}
                elif packet["operation"] == "exact" and _has_native_job_list(request):
                    result = {"enabled": False, "reason": "native multi-job input"}
                elif default_device(selected).platform != "gpu":
                    result = {"enabled": False, "reason": "non-GPU execution"}
                else:
                    with _prepare_af3_model():
                        prepared = execute(packet["operation"])
                    result = {"enabled": True, "result": prepared}
            elif "verify_keys" in packet:
                with _verify_af3_artifacts(tuple(packet["verify_keys"])):
                    result = execute(packet["kind"])
            else:
                result = execute(packet["kind"])
        response = {"ok": True, "result": result}
    except BaseException as error:
        code = 130 if isinstance(error, KeyboardInterrupt) else 1
        if isinstance(error, SystemExit):
            code = (
                error.code
                if isinstance(error.code, int)
                else 0
                if error.code is None
                else 1
            )
        response = {
            "ok": False,
            "error": {
                "type": type(error).__name__,
                "message": str(error) or type(error).__name__,
                "oom": isinstance(error, MemoryError) or oom.is_out_of_memory(error),
            },
        }
        if isinstance(error, SystemExit):
            response["error"]["code"] = error.code
    _write_packet(
        Path(result_file),
        {
            "version": _PACKET_VERSION,
            "package_directory": str(_PACKAGE_DIRECTORY),
            **response,
        },
    )
    return code
