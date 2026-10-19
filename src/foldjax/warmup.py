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
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

from foldjax.api import _predict_resolved, resolve_cache_dir, resolve_requests
from foldjax.backends.base import Backend
from foldjax.cache import CacheSnapshot, cache_snapshot
from foldjax.manifest import device_peak_bytes
from foldjax.paths import runtime_dir
from foldjax.registry import get_backend
from foldjax.schema import PredictionError, PredictionRequest, PredictionResult


class _WarmTermination(BaseException):
    """Unwind native exception adapters without turning a signal into a failure."""


_PROCESS_SCRATCH: ContextVar[Path | None] = ContextVar(
    "warm_process_scratch", default=None
)
_AF3_PREPARATION: ContextVar[bool] = ContextVar("af3_warm_preparation", default=False)
_AF3_VERIFICATION: ContextVar[list[dict[str, object]] | None] = ContextVar(
    "af3_warm_verification", default=None
)


class _AF3PreparedError(Exception):
    def __init__(self, artifact: dict[str, object]):
        super().__init__("AF3 model compilation prepared without model execution")
        self.artifact = artifact


@contextmanager
def _prepare_af3_model() -> Iterator[None]:
    """Stop after the managed model is persisted, before its first dispatch.

    Pinned JAX returns a loaded executable here only after its cache write.
    Tokamax benchmark compilations retain their normal execution; a model
    checkpoint requires both the candidate guard and a completed store policy.
    """
    from jax._src import compiler

    from foldjax._jax_cache_tiers import LayeredCache
    from foldjax.backends import _tokamax_autotune as tuning

    owner = threading.current_thread()
    active: ContextVar[dict | None] = ContextVar("af3_preparing_graph", default=None)
    original_compile, original_read = (
        compiler.compile_or_get_cached,
        compiler._cache_read,
    )
    original_put = LayeredCache.put

    def read(module_name, key, *args, **kwargs):
        result = original_read(module_name, key, *args, **kwargs)
        if (state := active.get()) is not None and module_name == "jit_apply_fn":
            state.update(key=key, loaded=result[0] is not None)
        return result

    def put(adapter, key, value):
        original_put(adapter, key, value)
        if (state := active.get()) is not None and state.get("key") == key:
            # put() deliberately swallows filesystem failures. Confirm the exact
            # serialized bytes through the same trusted shared/private adapter.
            state["written"] = adapter.get(key) == value

    def compile_or_checkpoint(backend, computation, *args, **kwargs):
        name = str(computation.operation.attributes["sym_name"]).strip('"')
        if (
            name != "jit_apply_fn"
            or threading.current_thread() is not owner
            or tuning._CANDIDATE_COMPILATION.get()
        ):
            return original_compile(backend, computation, *args, **kwargs)
        record = tuning.recorded() or {}
        if not record.get("persistent_store") or not record.get("sources"):
            raise PredictionError(
                "AF3 warm preparation reached a model without a completed "
                "persistent Tokamax policy; model execution was not attempted"
            )
        state: dict[str, object] = {}
        token = active.set(state)
        try:
            original_compile(backend, computation, *args, **kwargs)
        finally:
            active.reset(token)
        if not state.get("loaded") and not state.get("written"):
            raise PredictionError(
                "AF3 warm preparation did not confirm a persistent model "
                "executable; check cache write permissions and cache size limits. "
                "Model execution was not attempted"
            )
        raise _AF3PreparedError(
            {
                "module": name,
                "cache_key": state["key"],
                "source": "loaded" if state.get("loaded") else "written",
            }
        )

    token = _AF3_PREPARATION.set(True)
    try:
        with ExitStack() as stack:
            for target, name, replacement in (
                (compiler, "compile_or_get_cached", compile_or_checkpoint),
                (compiler, "_cache_read", read),
                (LayeredCache, "put", put),
            ):
                stack.enter_context(patch.object(target, name, replacement))
            yield
    finally:
        _AF3_PREPARATION.reset(token)


@contextmanager
def _verify_af3_artifacts(keys: tuple[str, ...]) -> Iterator[None]:
    """Permit the prepared model's persistent reads, never a new compilation."""
    from jax._src import compilation_cache, compiler

    from foldjax import oom
    from foldjax.backends import _tokamax_autotune as tuning

    if not keys or not all(isinstance(key, str) and key for key in keys):
        raise PredictionError("AF3 preparation did not report persistent model keys")
    original_read = compiler._cache_read
    original_compile = compiler.backend_compile_and_load
    original_load = compilation_cache.get_executable_and_time
    reads: list[dict[str, object]] = []
    load_oom: ContextVar[BaseException | None] = ContextVar(
        "af3_cache_load_oom", default=None
    )

    def load(*args, **kwargs):
        try:
            return original_load(*args, **kwargs)
        except Exception as error:
            if isinstance(error, MemoryError) or oom.is_out_of_memory(error):
                load_oom.set(error)
            raise

    def read(module_name, key, *args, **kwargs):
        if module_name != "jit_apply_fn" or tuning._CANDIDATE_COMPILATION.get():
            return original_read(module_name, key, *args, **kwargs)
        if key not in keys:
            raise PredictionError(
                "AF3 verification selected an unprepared model cache key"
            )
        token = load_oom.set(None)
        try:
            result = original_read(module_name, key, *args, **kwargs)
            error = load_oom.get()
        finally:
            load_oom.reset(token)
        if result[0] is None:
            if error is not None:
                raise error
            raise PredictionError(
                "AF3 verification could not load its prepared model executable; "
                "new model compilation was not attempted"
            )
        reads.append({"cache_key": key, "source": "loaded"})
        return result

    def compile_model(backend, computation, *args, **kwargs):
        name = str(computation.operation.attributes["sym_name"]).strip('"')
        if name == "jit_apply_fn" and not tuning._CANDIDATE_COMPILATION.get():
            raise PredictionError("AF3 verification refuses new model compilation")
        return original_compile(backend, computation, *args, **kwargs)

    token = _AF3_VERIFICATION.set(reads)
    try:
        with ExitStack() as stack:
            stack.enter_context(patch.object(compiler, "_cache_read", read))
            stack.enter_context(
                patch.object(compiler, "backend_compile_and_load", compile_model)
            )
            # Enter before the prediction installs its layered getter. A
            # shared OOM followed by an absent private entry returns no error;
            # retain it only when neither tier produced an executable.
            stack.enter_context(
                patch.object(
                    compilation_cache,
                    "get_executable_and_time",
                    load,
                )
            )
            yield
    finally:
        _AF3_VERIFICATION.reset(token)


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
    preparation: dict[str, object] | None = None
    verification: dict[str, object] | None = None

    @property
    def new_files(self) -> int:
        return max(0, self.after.files - self.before.files)

    @property
    def new_bytes(self) -> int:
        return max(0, self.after.bytes - self.before.bytes)

    @property
    def status(self) -> str:
        if self.strategy == "compile_only":
            return "prepared"
        cache = self.compilation_cache or {}
        tuning_calls = (self.kernel_tuning or {}).get("calls", ())
        published = any(
            events.get("success", 0) > 0 for events in cache.get("writes", {}).values()
        ) or any(call.get("cache_write_tiers") for call in tuning_calls)
        if (
            self.new_files
            or self.new_bytes
            or published
            or (self.preparation or {}).get("published")
        ):
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
        if self.preparation is not None:
            summary["preparation"] = dict(self.preparation)
        if self.verification is not None:
            summary["verification"] = dict(self.verification)
        return summary


def _warm_prediction(
    request: PredictionRequest,
    backend: Backend,
    *,
    output_root: Path | None,
) -> PredictionResult | None:
    # A session also selects staging and compact input routes. Warming must
    # follow those routes even when only one seed will be executed.
    if getattr(backend, "session_reuse", False):
        with backend.session((request,)) as active:
            return _predict_resolved(request, backend=active, output_root=output_root)
    return _predict_resolved(request, output_root=output_root)


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
    prepared = None
    # A warm has one seed and no batch to survive, so neither policy applies: a
    # failure here is the answer to "does this profile compile and run", and a
    # reused manifest would report a cache delta for work that did not happen.
    scalar = dataclasses.replace(scalar, resume=False, on_error="stop")
    try:
        if keep_output:
            result = _warm_prediction(scalar, backend, output_root=output_root)
            retained = scalar.output_dir
        else:
            scratch_root = _PROCESS_SCRATCH.get() or runtime_dir("cache-warm")
            scratch_root.mkdir(parents=True, exist_ok=True)
            with _warm_directory(scratch_root, prefix=f".{backend.name}-") as scratch:
                ephemeral = dataclasses.replace(scalar, output_dir=scratch)
                result = _warm_prediction(ephemeral, backend, output_root=scratch)
            retained = None
    except _AF3PreparedError as checkpoint:
        prepared = checkpoint.artifact
        result, retained = None, None
    if _AF3_PREPARATION.get() and prepared is None:
        raise PredictionError(
            "AF3 warm preparation did not observe its model checkpoint"
        )
    if result is None and prepared is None:
        raise PredictionError(f"{backend.name} cache warm produced no prediction")
    verification = _AF3_VERIFICATION.get()
    if verification is not None and not verification:
        raise PredictionError(
            "AF3 verification did not observe a persistent model read"
        )
    seconds = time.perf_counter() - started
    peak = device_peak_bytes()
    from foldjax.backends._tokamax_autotune import recorded as recorded_kernel_tuning

    tuning = recorded_kernel_tuning() if backend.name == "alphafold3" else None
    from foldjax.cache import recorded as recorded_compilation_cache

    compilation_cache = recorded_compilation_cache()
    shared = (compilation_cache or {}).get("locations", {}).get("shared")
    after = cache_snapshot(
        namespace,
        warn_unshareable=compilation_cache is None or shared is not None,
    )
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
        samples=len(result.samples) if result is not None else 0,
        peak_device_bytes=peak,
        output_dir=retained,
        strategy="compile_only" if prepared is not None else "execute_once",
        shape_profile=result.shape_profile if result is not None else None,
        kernel_tuning=tuning,
        compilation_cache=compilation_cache,
        preparation=prepared,
        verification={
            "main_cache_reads": list(verification),
            "kernel_miss_policy": request.options.get("kernel_autotuning", "autotune"),
        }
        if verification is not None
        else None,
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
                    preparation=summary.get("preparation"),
                    verification=summary.get("verification"),
                )
            )
    return tuple(results) if plural else results[0]


__all__ = ["CacheWarmResult", "warm_cache"]
