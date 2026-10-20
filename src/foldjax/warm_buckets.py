"""Warm serving buckets through the ordinary prediction path at two lengths."""

from __future__ import annotations

import dataclasses
import logging
import re
import tempfile
import time
from pathlib import Path
from typing import Any

from foldjax import oom
from foldjax.backends.alphafold3 import AF3_TOKEN_BUCKETS
from foldjax.job import Job
from foldjax.padding import TOKEN_BUCKETS
from foldjax.registry import normalize_model_name
from foldjax.schema import PaddingConfig, PredictionRequest
from foldjax.warmup import warm_cache


def parse_token_buckets(
    spec: str,
    *,
    model: str,
    options: dict[str, Any] | None = None,
) -> tuple[int, ...]:
    model = normalize_model_name(model)
    grid = AF3_TOKEN_BUCKETS if model == "alphafold3" else TOKEN_BUCKETS
    if model == "alphafold3" and options and "buckets" in options:
        from foldjax.backends.alphafold3 import _validated_buckets

        grid = _validated_buckets(options["buckets"])
        if not grid:
            raise ValueError(
                "--token-buckets needs a non-empty AlphaFold 3 bucket list"
            )
    text = spec.strip()
    interval = re.fullmatch(r"([0-9]+)\.\.([0-9]+)", text)
    if text == "auto":
        selected = tuple(grid)
    elif interval:
        low, high = map(int, interval.groups())
        if low < 1 or high < low:
            raise ValueError("token bucket range must be positive and increasing")
        selected = tuple(value for value in grid if low <= value <= high)
    elif re.fullmatch(r"[0-9]+(?:\s*,\s*[0-9]+)*", text):
        selected = tuple(sorted({int(value) for value in text.split(",")}))
        if any(value < 2 for value in selected):
            raise ValueError(
                "token buckets must be at least 2 for two-length verification"
            )
        if model == "alphafold3" and options and "buckets" in options:
            if any(value not in grid for value in selected):
                raise ValueError(
                    "token buckets must belong to the requested native AF3 buckets"
                )
    else:
        raise ValueError(
            "--token-buckets must be auto, a comma-separated list, or LOW..HIGH"
        )
    if not selected:
        raise ValueError("token bucket selection is empty")
    if any(value < 2 for value in selected):
        raise ValueError("two-length verification requires token buckets of at least 2")
    if model == "alphafold3" and options and "buckets" in options:
        for value in selected:
            previous = max((item for item in grid if item < value), default=0)
            if value - previous < 2:
                raise ValueError(
                    "native AF3 bucket intervals need two distinct real lengths"
                )
    return selected


def representative_lengths(bucket: int, *, previous: int = 0) -> tuple[int, int]:
    """Two positive real sizes in this bucket's interval, including tiny pins."""
    if bucket < 2:
        raise ValueError(
            "two-length verification requires a token bucket of at least 2"
        )
    margin = min(8, max(1, (bucket - previous) // 4))
    second = max(previous + 1, bucket - margin)
    second = min(bucket, max(2, second))
    first = max(previous + 1, second - margin)
    if first >= second:
        first, second = bucket - 1, bucket
    return first, second


class _CompileRequests(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        if message.startswith("Compiling "):
            self.messages.append(message.split(" with ", 1)[0])


def device_summary(request: PredictionRequest | None = None) -> dict[str, Any]:
    import jax

    from foldjax._gpu import default_device
    from foldjax.memory_policy import device_memory_budget

    device = default_device(request)
    with jax.default_device(device):
        budget = device_memory_budget()
    return {
        "platform": device.platform,
        "device_kind": device.device_kind,
        "card_bytes": budget.card_bytes,
        "pool_bytes": budget.pool_bytes,
    }


def _padded_targets(profile: Any) -> list[dict[str, Any]]:
    if isinstance(profile, dict):
        if isinstance(profile.get("target"), dict):
            return [profile["target"]]
        return [
            target for value in profile.values() for target in _padded_targets(value)
        ]
    if isinstance(profile, (list, tuple)):
        return [target for value in profile for target in _padded_targets(value)]
    return []


def _bucket_row(bucket: int, previous: int) -> dict[str, Any]:
    return {
        "bucket": bucket,
        "lengths": list(representative_lengths(bucket, previous=previous)),
        "runs": [],
        "status": "success",
        "error": None,
        "oom": False,
    }


def _failed_row(row: dict[str, Any], error: Exception) -> None:
    row.update(
        status="failed",
        error=str(error) or type(error).__name__,
        oom=isinstance(error, MemoryError)
        or getattr(error, "warm_oom", False)
        or oom.is_out_of_memory(error),
    )
    if getattr(error, "warm_process", None) is not None:
        row["process"] = error.warm_process


def bucket_request(
    request: PredictionRequest,
    *,
    model: str,
    bucket: int,
    length: int,
    index: int,
    jobs: Path,
) -> PredictionRequest:
    """Build the representative request shared by warming and presence checks."""
    jobs.mkdir(parents=True, exist_ok=True)
    source = Job.from_sequences(("A" * length,), name=f"warm-{bucket}-{index}").write(
        jobs / f"{model}-{bucket}-{index}.json"
    )
    native_buckets = model == "alphafold3" and "buckets" in request.options
    padding = (
        None
        if native_buckets
        else dataclasses.replace(request.padding or PaddingConfig(), tokens=bucket)
    )
    output = (
        None
        if request.output_dir is None
        else Path(request.output_dir) / model / str(bucket) / str(length)
    )
    return dataclasses.replace(
        request,
        model=model,
        models=None,
        input=source,
        inputs=None,
        input_format="foldjax",
        source=None,
        padding=padding,
        options=dict(request.options),
        output_dir=output,
        msa="single",
        resume=False,
        on_error="stop",
    )


def _warm_bucket(
    request: PredictionRequest,
    *,
    model: str,
    bucket: int,
    previous: int,
    jobs: Path,
) -> dict[str, Any]:
    """Keep both real lengths together, including their device and reuse probe."""
    import jax

    row = _bucket_row(bucket, previous)
    lengths = row["lengths"]
    device = device_summary(dataclasses.replace(request, model=model, models=None))
    jobs.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    try:
        for index, length in enumerate(lengths):
            scalar = bucket_request(
                request,
                model=model,
                bucket=bucket,
                length=length,
                index=index,
                jobs=jobs,
            )
            output = scalar.output_dir
            if output is not None:
                from foldjax.api import _prepare_output_directory

                _prepare_output_directory(output, boundary=Path(request.output_dir))
            observed = _CompileRequests()
            logger = logging.getLogger("jax")
            logger.addHandler(observed)
            try:
                with jax.log_compiles(True):
                    result = warm_cache(scalar)
            finally:
                logger.removeHandler(observed)
            row["runs"].append(
                {
                    **result.summary(),
                    "real_tokens": length,
                    "compile_requests": len(observed.messages),
                    "compile_request_names": observed.messages,
                }
            )
        profiles = [run.get("shape_profile") for run in row["runs"]]
        targets = [_padded_targets(profile) for profile in profiles]
        matching_targets = targets[0] == targets[1] if all(targets) else None
        matching_namespace = row["runs"][0]["cache_dir"] == row["runs"][1]["cache_dir"]
        row["verification"] = {
            "second_length_compile_requests": row["runs"][1]["compile_requests"],
            "shape_profiles": profiles,
            "padded_targets_match": matching_targets,
            "cache_namespace_match": matching_namespace,
            "note": "compile requests may restore persistent entries; "
            "not a cache-miss count",
        }
        if matching_targets is False or not matching_namespace:
            raise RuntimeError(
                "two lengths selected different padded targets or cache namespaces"
            )
    except Exception as error:  # noqa: BLE001 - each bucket has a failure row
        _failed_row(row, error)
    finally:
        row["seconds"] = round(time.perf_counter() - started, 3)
    return {"row": row, "device": device}


def warm_token_buckets(
    request: PredictionRequest,
    spec: str,
    *,
    stop_on_oom: bool = True,
    isolation: str = "in-process",
) -> dict[str, Any]:
    """Warm two lengths per bucket, optionally using a fresh process per bucket.

    Compilation requests can restore persistent executables and are not miss
    counts. Process isolation inherits environment and JSON request options;
    ambient JAX contexts/config and custom backends require in-process mode.
    """
    if isolation not in {"process", "in-process"}:
        raise ValueError("isolation must be 'process' or 'in-process'")
    if not request.use_compile_cache:
        raise ValueError("cache warm cannot be used with use_compile_cache=False")
    models = tuple(normalize_model_name(name) for name in request.resolved_models)
    if len(set(models)) != len(models):
        raise ValueError("models resolve to the same backend more than once")
    if len(models) > 1 and request.weights is not None:
        raise ValueError("one explicit weights path cannot be shared by several models")
    selections = {
        model: parse_token_buckets(spec, model=model, options=dict(request.options))
        for model in models
    }
    if request.msa not in {"none", "single"}:
        raise ValueError(
            "synthetic bucket warming uses --msa single; "
            "use an input job to warm searched MSAs"
        )
    if (
        "alphafold3" in models
        and "buckets" in request.options
        and request.padding is not None
    ):
        raise ValueError("native AF3 buckets and neutral padding cannot both be set")
    if isolation == "process":
        from foldjax._warm_process import request_packet, run_worker

        packet = request_packet(request)
    report: dict[str, Any] = {
        "strategy": "token_buckets",
        "isolation": isolation,
        "device": {
            "platform": "unknown",
            "device_kind": "unavailable",
            "card_bytes": None,
            "pool_bytes": None,
        },
        "stop_on_oom": stop_on_oom,
        "models": [],
    }
    with tempfile.TemporaryDirectory(prefix="foldjax-bucket-jobs-") as temporary:
        for model in models:
            grid = selections[model]
            native_buckets = model == "alphafold3" and "buckets" in request.options
            model_report: dict[str, Any] = {
                "model": model,
                "requested_buckets": list(grid),
                "successful_buckets": [],
                "first_failed_bucket": None,
                "failure_reason": None,
                "buckets": [],
                "stopped_on_oom": False,
            }
            if model == "opendde":
                padding = request.padding or PaddingConfig()
                automatic = [
                    name
                    for name in ("atoms", "structural_tokens")
                    if getattr(padding, name) is None
                ]
                if automatic:
                    model_report["shape_coverage"] = {
                        "kind": "representative_concrete_shapes",
                        "automatic_axes": automatic,
                        "note": "OpenDDE atom/structural capacities depend on actual "
                        "stored features. These synthetic lengths do not cover every "
                        "shape in the token bucket; warm real input jobs or use the "
                        "same explicit axis pins for warming and prediction.",
                    }
            report["models"].append(model_report)
            default_grid = (
                tuple(request.options["buckets"])
                if native_buckets
                else AF3_TOKEN_BUCKETS
                if model == "alphafold3"
                else TOKEN_BUCKETS
            )
            for bucket in grid:
                previous = max(
                    (size for size in default_grid if size < bucket), default=0
                )
                started = time.perf_counter()
                try:
                    if isolation == "process":
                        result, process = run_worker(
                            {
                                "kind": "bucket",
                                "request": packet,
                                "model": model,
                                "bucket": bucket,
                                "previous": previous,
                            }
                        )
                        result["row"]["process"] = process
                    else:
                        result = _warm_bucket(
                            request,
                            model=model,
                            bucket=bucket,
                            previous=previous,
                            jobs=Path(temporary),
                        )
                    row = result["row"]
                    row["device"] = result["device"]
                    if report["device"]["platform"] == "unknown":
                        report["device"] = result["device"]
                except Exception as error:  # noqa: BLE001 - worker failures are bucket failures
                    row = _bucket_row(bucket, previous)
                    _failed_row(row, error)
                    row["seconds"] = round(time.perf_counter() - started, 3)
                model_report["buckets"].append(row)
                if row["status"] == "success":
                    model_report["successful_buckets"].append(bucket)
                elif model_report["first_failed_bucket"] is None:
                    model_report["first_failed_bucket"] = bucket
                    model_report["failure_reason"] = row["error"]
                if row["oom"] and stop_on_oom:
                    model_report["stopped_on_oom"] = True
                    break
    return report


def format_bucket_report(report: dict[str, Any]) -> str:
    device = report["device"]
    card = device["card_bytes"]
    capacity = "unknown" if card is None else f"{card / 1024**3:.1f} GiB"
    lines = [
        f"device: {device['device_kind']} ({capacity})",
        "model        bucket  status    seconds  second-length compile requests",
    ]
    for model in report["models"]:
        for row in model["buckets"]:
            requests = row.get("verification", {}).get(
                "second_length_compile_requests", "-"
            )
            lines.append(
                f"{model['model']:<12s} {row['bucket']:>6}  "
                f"{row['status']:<8s} {row['seconds']:>7.2f}  {requests}"
            )
            if row["error"]:
                lines.append(f"  {row['error']}")
        lines.append(
            f"  successful: {model['successful_buckets']}; "
            f"first failed: {model['first_failed_bucket']}"
        )
        if coverage := model.get("shape_coverage"):
            lines.append(f"  coverage: {coverage['note']}")
    return "\n".join(lines)
