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
import os
import resource
import sys
import time
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from functools import wraps
from pathlib import Path
from typing import Any
from unittest.mock import patch

from bench.compilation_profile import device_memory as _device_memory
from bench.compilation_profile import observe_compilation, write_report
from bench.prediction_profile import _device, _memory


def _memory_snapshot():
    try:
        return _device_memory()
    except Exception as error:
        return {"unavailable": type(error).__name__}


@contextmanager
def profile_prediction(report: dict[str, Any]) -> Iterator[None]:
    """Observe actual dispatch and loaded executables in one diagnostic run."""
    import jax

    from foldjax.models.opendde import runner

    phases: list[dict[str, Any]] = report.setdefault("phases", [])

    def phase(name, function):
        @wraps(function)
        def measured(*args, **kwargs):
            entry = {"phase": name, "device_before": _memory_snapshot()}
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
                entry["device_after"] = _memory_snapshot()
                # Linux ru_maxrss is a lifetime high-water mark, not a reset
                # per-phase peak or the current resident set.
                entry["process_peak_rss_kib"] = resource.getrusage(
                    resource.RUSAGE_SELF
                ).ru_maxrss
                phases.append(entry)

        return measured

    with ExitStack() as stack:
        stack.enter_context(observe_compilation(report))
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
        "bench/compilation_profile.py",
        "bench/prediction_profile.py",
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

    from foldjax import cli

    report: dict[str, Any] = {
        "schema": 1,
        "measurement_kind": "synchronized_diagnostic",
        "prediction_arguments": prediction,
        "source_files_sha256": _sources(),
        "notes": [
            "phase times include compilation/cache loading and synchronization",
            "device peak and process RSS are process-lifetime high-water marks",
            "executable memory is XLA accounting, not total GPU/context memory",
            "load residual includes deserialization/module loading and bookkeeping",
            "use a fresh process per size and a normal benchmark for latency",
        ],
    }
    original_predictions = cli._run_predictions

    @wraps(original_predictions)
    def observed_predictions(*args, **kwargs):
        # Runner imports can initialize CUDA. Let the native CLI configure its
        # allocator and runtime flags before the profiler imports the model.
        with profile_prediction(report):
            from foldjax import oom

            request = args[0] if args else kwargs.get("request")
            report["device"] = _device(request)
            report["effective_mem_fraction"] = oom.mem_fraction()
            report["environment"] = {
                name: os.environ.get(name)
                for name in (
                    "CUDA_LAUNCH_BLOCKING",
                    "CUDA_VISIBLE_DEVICES",
                    "JAX_PLATFORMS",
                    "JAX_ENABLE_COMPILATION_CACHE",
                    "XLA_CLIENT_MEM_FRACTION",
                    "XLA_PYTHON_CLIENT_MEM_FRACTION",
                    "XLA_FLAGS",
                    "FOLDJAX_PRIVATE_COMPILE_CACHE",
                )
            }
            report["device_before"] = _memory(request)
            try:
                return original_predictions(*args, **kwargs)
            finally:
                report["device_after"] = _memory(request)
                for name, module_name in (
                    ("compilation_cache", "foldjax.cache"),
                    ("kernel_tuning", "foldjax.backends._tokamax_autotune"),
                ):
                    module = sys.modules.get(module_name)
                    try:
                        report[name] = module.recorded() if module is not None else None
                    except Exception as error:
                        report[name] = {"unavailable": type(error).__name__}

    try:
        with patch.object(cli, "_run_predictions", observed_predictions):
            code = cli.main(prediction)
        report["exit_code"] = code
        return code
    except BaseException as error:
        report["error_type"] = type(error).__name__
        raise
    finally:
        write_report(args.profile_out, report)


if __name__ == "__main__":
    raise SystemExit(main())
