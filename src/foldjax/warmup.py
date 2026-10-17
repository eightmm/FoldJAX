"""Populate FoldJAX's persistent compilation cache with a representative job.

JAX executables depend on the accelerator, runtime, input shapes and static
model options.  A useful warm-up therefore has to follow the exact prediction
path on the deployment machine.  FoldJAX executes one seed, waits for the full
result, and discards prediction files unless the caller explicitly supplied an
output directory.
"""

from __future__ import annotations

import dataclasses
import signal
import tempfile
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path

from foldjax.api import _predict_resolved, resolve_cache_dir, resolve_requests
from foldjax.cache import CacheSnapshot, cache_snapshot
from foldjax.manifest import device_peak_bytes
from foldjax.paths import runtime_dir
from foldjax.registry import get_backend
from foldjax.schema import PredictionError, PredictionRequest


class _WarmTermination(BaseException):
    """Unwind native exception adapters without turning a signal into a failure."""


_PROCESS_SCRATCH: ContextVar[Path | None] = ContextVar(
    "warm_process_scratch", default=None
)


@contextmanager
def _scratch_root(root: Path) -> Iterator[None]:
    token = _PROCESS_SCRATCH.set(root)
    try:
        yield
    finally:
        _PROCESS_SCRATCH.reset(token)


@contextmanager
def _warm_directory(root: Path, *, prefix: str) -> Iterator[Path]:
    previous = None
    installed = False

    def terminate(signum, _frame):
        # Unwind the temporary directory and native session before exiting.
        raise _WarmTermination(signum)

    try:
        if threading.current_thread() is threading.main_thread():
            previous = signal.getsignal(signal.SIGTERM)
            if previous == signal.SIG_DFL:
                signal.signal(signal.SIGTERM, terminate)
                installed = True
        with tempfile.TemporaryDirectory(prefix=prefix, dir=root) as temporary:
            yield Path(temporary)
    except _WarmTermination as error:
        raise SystemExit(128 + error.args[0]) from None
    finally:
        if installed:
            signal.signal(signal.SIGTERM, previous)


@dataclass(frozen=True, slots=True)
class CacheWarmResult:
    """What one execute-once cache warm changed."""

    model: str
    input: Path
    weights: Path
    cache_dir: Path
    seed: int
    seconds: float
    before: CacheSnapshot
    after: CacheSnapshot
    samples: int
    peak_device_bytes: int | None = None
    output_dir: Path | None = None
    strategy: str = "execute_once"
    shape_profile: dict[str, object] | None = None
    kernel_tuning: dict[str, object] | None = None
    compilation_cache: dict[str, object] | None = None
    process: dict[str, int] | None = None

    @property
    def new_files(self) -> int:
        return max(0, self.after.files - self.before.files)

    @property
    def new_bytes(self) -> int:
        return max(0, self.after.bytes - self.before.bytes)

    @property
    def status(self) -> str:
        cache = self.compilation_cache or {}
        tuning_calls = (self.kernel_tuning or {}).get("calls", ())
        published = any(
            events.get("success", 0) > 0 for events in cache.get("writes", {}).values()
        ) or any(call.get("cache_write_tiers") for call in tuning_calls)
        if self.new_files or self.new_bytes or published:
            return "populated"
        loaded = any(
            events.get("success", 0) > 0 for events in cache.get("loads", {}).values()
        ) or any(
            call.get("cache_read_tiers") or call.get("cache_reuse_tiers")
            for call in tuning_calls
        )
        if self.before.files or loaded:
            # Directory snapshots cannot distinguish a true XLA hit from an
            # executable that was ineligible for persistence. Do not claim a
            # hit that JAX did not expose as structured data.
            return "no_new_entry_observed"
        return "executed_no_persistent_entry_observed"

    def summary(self) -> dict[str, object]:
        """Return stable terminal/API output without exposing prediction arrays."""

        summary = {
            "model": self.model,
            "input": str(self.input),
            "weights": str(self.weights),
            "cache_dir": str(self.cache_dir),
            "seed": self.seed,
            "strategy": self.strategy,
            "status": self.status,
            "seconds": self.seconds,
            "peak_device_bytes": self.peak_device_bytes,
            "samples_executed": self.samples,
            "prediction_output": (
                None if self.output_dir is None else str(self.output_dir)
            ),
            "cache": {
                "before": self.before.summary(),
                "after": self.after.summary(),
                "new_files": self.new_files,
                "new_bytes": self.new_bytes,
            },
        }
        if self.shape_profile is not None:
            summary["shape_profile"] = dict(self.shape_profile)
        if self.kernel_tuning is not None:
            summary["kernel_tuning"] = dict(self.kernel_tuning)
        if self.compilation_cache is not None:
            summary["compilation_cache"] = dict(self.compilation_cache)
        if self.process is not None:
            summary["process"] = dict(self.process)
        return summary


def _warm_resolved(
    request: PredictionRequest,
    *,
    keep_output: bool,
    output_root: Path | None,
) -> CacheWarmResult:
    """Warm one already resolved model/input pair with its first requested seed."""

    backend = get_backend(request.model)
    if request.cache_dir is None:
        raise ValueError("cache warm requires an enabled persistent cache")
    namespace = resolve_cache_dir(request, backend)
    before = cache_snapshot(namespace)
    seed = request.resolved_seeds[0]
    scalar = dataclasses.replace(
        request,
        seed=seed,
        seeds=None,
        num_seeds=None,
    )

    started = time.perf_counter()
    # A warm has one seed and no batch to survive, so neither policy applies: a
    # failure here is the answer to "does this profile compile and run", and a
    # reused manifest would report a cache delta for work that did not happen.
    scalar = dataclasses.replace(scalar, resume=False, on_error="stop")
    if keep_output:
        result = _predict_resolved(scalar, output_root=output_root)
        retained = scalar.output_dir
    else:
        scratch_root = _PROCESS_SCRATCH.get() or runtime_dir("cache-warm")
        scratch_root.mkdir(parents=True, exist_ok=True)
        with _warm_directory(scratch_root, prefix=f".{backend.name}-") as scratch:
            ephemeral = dataclasses.replace(scalar, output_dir=scratch)
            result = _predict_resolved(ephemeral, output_root=scratch)
        retained = None
    if result is None:
        raise PredictionError(f"{backend.name} cache warm produced no prediction")
    seconds = time.perf_counter() - started
    peak = device_peak_bytes()
    after = cache_snapshot(namespace)
    from foldjax.backends._tokamax_autotune import recorded as recorded_kernel_tuning

    tuning = recorded_kernel_tuning() if backend.name == "alphafold3" else None
    from foldjax.cache import recorded as recorded_compilation_cache

    compilation_cache = recorded_compilation_cache()
    assert request.weights is not None
    return CacheWarmResult(
        model=backend.name,
        input=request.input,
        weights=request.weights,
        cache_dir=namespace,
        seed=seed,
        seconds=round(seconds, 3),
        before=before,
        after=after,
        samples=len(result.samples),
        peak_device_bytes=peak,
        output_dir=retained,
        shape_profile=result.shape_profile,
        kernel_tuning=tuning,
        compilation_cache=compilation_cache,
    )


def warm_cache(
    request: PredictionRequest,
    *,
    isolation: str = "in-process",
) -> CacheWarmResult | tuple[CacheWarmResult, ...]:
    """Execute each representative job once and populate its JAX cache.

    The request accepts the same model, profile, input and sampling options as
    :func:`foldjax.predict`.  Only the first requested seed is needed because a
    seed changes runtime values, not the compiled program.  Scalar calls return
    one result; plural model/input requests mirror :func:`foldjax.predict` and
    return a tuple in declaration order.

    This is deliberately named *warm*, not compile-only: model execution and
    GPU kernel autotuning occur once.  When ``output_dir`` is omitted all native
    prediction files live in a temporary runtime directory and are removed.

    ``isolation='process'`` runs each model in a fresh interpreter, inheriting
    the environment and JSON-compatible request options. Ambient JAX contexts,
    runtime config changes and custom backend registrations are not inherited;
    pass device/platform options explicitly or use the default in-process mode.
    """

    if not request.use_compile_cache:
        raise ValueError("cache warm cannot be used with use_compile_cache=False")
    if isolation not in {"process", "in-process"}:
        raise ValueError("isolation must be 'process' or 'in-process'")
    if isolation == "process":
        return _warm_processes(request)
    plural = request.models is not None or request.inputs is not None
    keep_output = request.output_dir is not None
    output_root = Path(request.output_dir) if plural and keep_output else None
    scratch = _PROCESS_SCRATCH.get()
    resolved = (
        resolve_requests(request, jobs_root=scratch.parent / "jobs")
        if scratch is not None and not keep_output
        else resolve_requests(request)
    )
    results = tuple(
        _warm_resolved(
            item,
            keep_output=keep_output,
            output_root=output_root,
        )
        for item in resolved
    )
    return results if plural else results[0]


def _warm_processes(request: PredictionRequest):
    from foldjax._warm_process import request_packet, run_worker
    from foldjax.registry import normalize_model_name

    # Resolve weights, input expansion, device and cache profiles inside the
    # child: even a preflight cache profile may initialize the CUDA runtime.
    models = tuple(normalize_model_name(model) for model in request.resolved_models)
    if len(set(models)) != len(models):
        raise ValueError(
            "models resolve to the same backend more than once; "
            "remove aliases or duplicate model names"
        )
    if len(models) > 1 and request.weights is not None:
        raise ValueError("one explicit weights path cannot be shared by several models")
    plural = request.models is not None or request.inputs is not None
    results = []
    for model in models:
        scalar = dataclasses.replace(
            request,
            model=None if request.models is not None else model,
            models=(model,) if request.models is not None else None,
        )
        summaries, process = run_worker(
            {"kind": "exact", "request": request_packet(scalar)}
        )
        for summary in summaries:
            results.append(
                CacheWarmResult(
                    model=summary["model"],
                    input=Path(summary["input"]),
                    weights=Path(summary["weights"]),
                    cache_dir=Path(summary["cache_dir"]),
                    seed=summary["seed"],
                    seconds=summary["seconds"],
                    before=CacheSnapshot(**summary["cache"]["before"]),
                    after=CacheSnapshot(**summary["cache"]["after"]),
                    samples=summary["samples_executed"],
                    peak_device_bytes=summary["peak_device_bytes"],
                    output_dir=None
                    if summary["prediction_output"] is None
                    else Path(summary["prediction_output"]),
                    strategy=summary["strategy"],
                    shape_profile=summary.get("shape_profile"),
                    kernel_tuning=summary.get("kernel_tuning"),
                    compilation_cache=summary.get("compilation_cache"),
                    process=process,
                )
            )
    return tuple(results) if plural else results[0]


__all__ = ["CacheWarmResult", "warm_cache"]
