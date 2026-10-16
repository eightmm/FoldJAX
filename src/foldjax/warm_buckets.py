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


def warm_token_buckets(
    request: PredictionRequest,
    spec: str,
    *,
    stop_on_oom: bool = True,
) -> dict[str, Any]:
    """Run two synthetic single-chain jobs per bucket, stopping a model on OOM.

    Compilation log requests on the second length are diagnostic: a request
    can restore a persistent executable, so these are not claimed as misses.
    Concrete shape profiles are reported for both lengths to expose mismatches.
    """
    import jax

    models = tuple(
        normalize_model_name(name) for name in (request.models or (request.model,))
    )
    selections = {
        model: parse_token_buckets(spec, model=model, options=dict(request.options))
        for model in models
    }
    if request.msa not in {"none", "single"}:
        raise ValueError(
            "synthetic bucket warming uses --msa single; "
            "use an input job to warm searched MSAs"
        )
    device = device_summary(dataclasses.replace(request, model=models[0], models=None))
    report: dict[str, Any] = {
        "strategy": "token_buckets",
        "device": device,
        "stop_on_oom": stop_on_oom,
        "models": [],
    }
    with tempfile.TemporaryDirectory(prefix="foldjax-bucket-jobs-") as temporary:
        jobs = Path(temporary)
        for model in models:
            grid = selections[model]
            native_buckets = model == "alphafold3" and "buckets" in request.options
            if native_buckets and request.padding is not None:
                raise ValueError(
                    "native AF3 buckets and neutral padding cannot both be set"
                )
            model_report: dict[str, Any] = {
                "model": model,
                "requested_buckets": list(grid),
                "successful_buckets": [],
                "first_failed_bucket": None,
                "failure_reason": None,
                "buckets": [],
                "stopped_on_oom": False,
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
                lengths = representative_lengths(bucket, previous=previous)
                row: dict[str, Any] = {
                    "bucket": bucket,
                    "lengths": list(lengths),
                    "runs": [],
                    "status": "success",
                    "error": None,
                    "oom": False,
                }
                model_report["buckets"].append(row)
                started = time.perf_counter()
                try:
                    for index, length in enumerate(lengths):
                        source = Job.from_sequences(
                            ("A" * length,), name=f"warm-{bucket}-{index}"
                        ).write(jobs / f"{model}-{bucket}-{index}.json")
                        padding = (
                            None
                            if native_buckets
                            else dataclasses.replace(
                                request.padding or PaddingConfig(), tokens=bucket
                            )
                        )
                        options = dict(request.options)
                        output = (
                            None
                            if request.output_dir is None
                            else Path(request.output_dir)
                            / model
                            / str(bucket)
                            / str(length)
                        )
                        scalar = dataclasses.replace(
                            request,
                            model=model,
                            models=None,
                            input=source,
                            inputs=None,
                            input_format="foldjax",
                            source=None,
                            padding=padding,
                            options=options,
                            output_dir=output,
                            msa="single",
                            resume=False,
                            on_error="stop",
                        )
                        if output is not None:
                            from foldjax.api import _prepare_output_directory

                            _prepare_output_directory(
                                output, boundary=Path(request.output_dir)
                            )
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
                    matching_targets = (
                        targets[0] == targets[1] if all(targets) else None
                    )
                    matching_namespace = (
                        row["runs"][0]["cache_dir"] == row["runs"][1]["cache_dir"]
                    )
                    row["verification"] = {
                        "second_length_compile_requests": row["runs"][1][
                            "compile_requests"
                        ],
                        "shape_profiles": profiles,
                        "padded_targets_match": matching_targets,
                        "cache_namespace_match": matching_namespace,
                        "note": "compile requests may restore persistent entries; "
                        "not a cache-miss count",
                    }
                    if matching_targets is False or not matching_namespace:
                        raise RuntimeError(
                            "two lengths selected different padded targets "
                            "or cache namespaces"
                        )
                    model_report["successful_buckets"].append(bucket)
                except Exception as error:  # noqa: BLE001 - each bucket has a failure row
                    row.update(
                        status="failed",
                        error=str(error) or type(error).__name__,
                        oom=isinstance(error, MemoryError)
                        or oom.is_out_of_memory(error),
                    )
                    if model_report["first_failed_bucket"] is None:
                        model_report["first_failed_bucket"] = bucket
                        model_report["failure_reason"] = row["error"]
                finally:
                    row["seconds"] = round(time.perf_counter() - started, 3)
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
    return "\n".join(lines)
