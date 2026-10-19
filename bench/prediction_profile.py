"""Observe one native prediction in an existing allocation, in a fresh process.

python -m bench.prediction_profile --profile-out /tmp/af3.json -- \
    predict --model alphafold3 --input job.yaml --padding

No model options, cache roots or CUDA synchronization policy are changed. This
is a diagnostic run, not a latency benchmark. XLA dump flags must be supplied
in the environment before starting Python. A killed process cannot flush its
report; retain stderr and the external GPU-memory trace as well.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import resource
import sys
import time
from functools import wraps
from pathlib import Path
from unittest.mock import patch

from bench.compilation_profile import device_memory, observe_compilation, write_report


def _memory(request=None):
    try:
        return device_memory(request)
    except Exception as error:
        return {"unavailable": type(error).__name__}


def _device(request=None):
    import jax
    import jaxlib

    from foldjax._gpu import compute_capability, default_device

    try:
        device = default_device(request)
        return {
            "platform": device.platform,
            "kind": device.device_kind,
            "id": device.id,
            "compute_capability": compute_capability(device),
            "core_count": getattr(device, "core_count", None),
            "platform_version": device.client.platform_version,
            "jax": jax.__version__,
            "jaxlib": jaxlib.__version__,
        }
    except Exception as error:
        return {"unavailable": type(error).__name__}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-out", required=True, type=Path)
    parser.add_argument("prediction", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    prediction = args.prediction
    if prediction[:1] == ["--"]:
        prediction = prediction[1:]
    if prediction[:1] != ["predict"] or "--model" not in prediction:
        parser.error("pass a native predict --model MODEL command after --")

    from foldjax import cli

    root = Path(__file__).resolve().parents[1]
    report = {
        "schema": 1,
        "measurement_kind": "native_prediction_diagnostic",
        "prediction_arguments": prediction,
        "source_files_sha256": {
            relative: hashlib.sha256((root / relative).read_bytes()).hexdigest()
            for relative in (
                "bench/prediction_profile.py",
                "bench/compilation_profile.py",
            )
        },
        "environment": {
            key: os.environ.get(key)
            for key in (
                "CUDA_LAUNCH_BLOCKING",
                "CUDA_VISIBLE_DEVICES",
                "JAX_PLATFORMS",
                "XLA_CLIENT_MEM_FRACTION",
                "XLA_PYTHON_CLIENT_MEM_FRACTION",
                "XLA_FLAGS",
            )
        },
        "notes": [
            "device peak and process RSS are process-lifetime high-water marks",
            "XLA executable accounting excludes CUDA context/module allocations",
            "layered_get includes trust validation and filesystem reading",
            "load_residual includes deserialization/module loading and bookkeeping",
            "unobserved in-memory hits do not appear as compiler invocations",
        ],
    }
    original_predictions = cli._run_predictions

    @wraps(original_predictions)
    def observed_predictions(*args, **kwargs):
        # Imports of JAX and a model must follow native allocator configuration.
        from foldjax import oom

        request = args[0] if args else kwargs.get("request")
        with observe_compilation(report):
            report["device"] = _device(request)
            report["effective_mem_fraction"] = oom.mem_fraction()
            report["device_before"] = _memory(request)
            started = time.perf_counter()
            try:
                return original_predictions(*args, **kwargs)
            finally:
                report["prediction_seconds"] = time.perf_counter() - started
                report["device_after"] = _memory(request)
                for name, module_name in (
                    ("compilation_cache", "foldjax.cache"),
                    ("kernel_tuning", "foldjax.backends._tokamax_autotune"),
                ):
                    module = sys.modules.get(module_name)
                    report[name] = module.recorded() if module is not None else None
                report["process_peak_rss_kib"] = resource.getrusage(
                    resource.RUSAGE_SELF
                ).ru_maxrss

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
