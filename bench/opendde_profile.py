"""Diagnose one OpenDDE prediction without changing its kernels or options.

Run each size in a fresh process. For example::

    python -m bench.opendde_profile --profile-out /tmp/opendde-768.json -- \
        predict --model opendde --input job.yaml --msa auto --fixed-msa 2048

This adds synchronization at phase boundaries. Its phase times include
compilation/cache loading and instrumentation; use ``bench.run_foldjax`` for
ordinary latency measurements. ``XLA_FLAGS=--xla_dump_to=...`` can additionally
retain XLA's buffer assignments for operation-level arena attribution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import time
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from functools import wraps
from pathlib import Path
from typing import Any
from unittest.mock import patch


def _device_memory() -> dict[str, int]:
    from foldjax._gpu import default_device

    stats = default_device().memory_stats() or {}
    return {
        name: int(stats[name])
        for name in ("bytes_in_use", "peak_bytes_in_use", "bytes_limit")
        if name in stats
    }


def _executable_memory(executable: Any) -> dict[str, int] | None:
    method = getattr(executable, "get_compiled_memory_stats", None)
    if method is None:
        return None
    try:
        stats = method()
    except (NotImplementedError, RuntimeError):
        return None
    return {
        name: int(getattr(stats, name))
        for name in (
            "argument_size_in_bytes",
            "output_size_in_bytes",
            "alias_size_in_bytes",
            "temp_size_in_bytes",
            "generated_code_size_in_bytes",
        )
        if hasattr(stats, name)
    }


@contextmanager
def profile_prediction(report: dict[str, Any]) -> Iterator[None]:
    """Observe actual dispatch and loaded executables in one diagnostic run."""
    import jax
    from jax._src import compiler

    from foldjax.models.opendde import runner

    phases: list[dict[str, Any]] = report.setdefault("phases", [])
    executables: list[dict[str, Any]] = report.setdefault("executables", [])
    original_compile = compiler.compile_or_get_cached

    @wraps(original_compile)
    def observed_compile(backend, computation, *args, **kwargs):
        entry = {
            "module": str(computation.operation.attributes["sym_name"]).strip('"'),
        }
        started = time.perf_counter()
        try:
            executable = original_compile(backend, computation, *args, **kwargs)
            entry["memory"] = _executable_memory(executable)
            entry["status"] = "ok"
            return executable
        except BaseException:
            entry["status"] = "error"
            raise
        finally:
            entry["compile_or_load_seconds"] = time.perf_counter() - started
            executables.append(entry)

    def phase(name, function):
        @wraps(function)
        def measured(*args, **kwargs):
            entry = {"phase": name, "device_before": _device_memory()}
            started = time.perf_counter()
            try:
                result = function(*args, **kwargs)
                jax.block_until_ready(result)
                entry["status"] = "ok"
                return result
            except BaseException:
                entry["status"] = "error"
                raise
            finally:
                entry["seconds"] = time.perf_counter() - started
                entry["device_after"] = _device_memory()
                # Linux ru_maxrss is a lifetime high-water mark, not a reset
                # per-phase peak or the current resident set.
                entry["process_peak_rss_kib"] = resource.getrusage(
                    resource.RUSAGE_SELF
                ).ru_maxrss
                phases.append(entry)

        return measured

    with ExitStack() as stack:
        stack.enter_context(
            patch.object(compiler, "compile_or_get_cached", observed_compile)
        )
        for name in (
            "_featurize",
            "_load_prepared_params",
            "_predict",
            "_score",
            "_write",
        ):
            stack.enter_context(
                patch.object(runner, name, phase(name, getattr(runner, name)))
            )
        yield


def _sources() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    relative_paths = (
        "bench/opendde_profile.py",
        "src/foldjax/models/opendde/runner.py",
        "src/foldjax/models/opendde/models/model.py",
        "src/foldjax/models/opendde/models/structural_refiner.py",
        "src/foldjax/models/opendde/models/shape_complementarity.py",
    )
    return {
        path: hashlib.sha256((root / path).read_bytes()).hexdigest()
        for path in relative_paths
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-out", type=Path, required=True)
    parser.add_argument("prediction", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    prediction = args.prediction
    if prediction[:1] == ["--"]:
        prediction = prediction[1:]
    if prediction[:1] != ["predict"] or "--model" not in prediction:
        parser.error("pass a FoldJAX predict --model opendde command after --")
    model_index = prediction.index("--model") + 1
    if model_index >= len(prediction) or prediction[model_index] != "opendde":
        parser.error("this diagnostic supports only --model opendde")

    from foldjax.cli import main as predict_main

    report: dict[str, Any] = {
        "schema": 1,
        "measurement_kind": "synchronized_diagnostic",
        "source_files_sha256": _sources(),
        "notes": [
            "phase times include compilation/cache loading and synchronization",
            "device peak and process RSS are process-lifetime high-water marks",
            "executable memory is XLA accounting, not total GPU/context memory",
            "use a fresh process per size and a normal benchmark for latency",
        ],
    }
    try:
        with profile_prediction(report):
            code = predict_main(prediction)
        report["exit_code"] = code
        return code
    except BaseException as error:
        report["error_type"] = type(error).__name__
        raise
    finally:
        args.profile_out.parent.mkdir(parents=True, exist_ok=True)
        args.profile_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
