"""Populate FoldJAX's persistent compilation cache with a representative job.

JAX executables depend on the accelerator, runtime, input shapes and static
model options.  A useful warm-up therefore has to follow the exact prediction
path on the deployment machine.  FoldJAX executes one seed, waits for the full
result, and discards prediction files unless the caller explicitly supplied an
output directory.
"""

from __future__ import annotations

import dataclasses
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from foldjax.api import _predict_resolved, resolve_cache_dir, resolve_requests
from foldjax.cache import CacheSnapshot, cache_snapshot
from foldjax.manifest import device_peak_bytes
from foldjax.paths import runtime_dir
from foldjax.registry import get_backend
from foldjax.schema import PredictionError, PredictionRequest


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

    @property
    def new_files(self) -> int:
        return max(0, self.after.files - self.before.files)

    @property
    def new_bytes(self) -> int:
        return max(0, self.after.bytes - self.before.bytes)

    @property
    def status(self) -> str:
        if self.new_files or self.new_bytes:
            return "populated"
        if self.before.files:
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
        scratch_root = runtime_dir("cache-warm")
        scratch_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=f".{backend.name}-", dir=scratch_root
        ) as temporary:
            scratch = Path(temporary)
            ephemeral = dataclasses.replace(scalar, output_dir=scratch)
            result = _predict_resolved(ephemeral, output_root=scratch)
        retained = None
    if result is None:
        raise PredictionError(f"{backend.name} cache warm produced no prediction")
    seconds = time.perf_counter() - started
    peak = device_peak_bytes()
    after = cache_snapshot(namespace)
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
    )


def warm_cache(
    request: PredictionRequest,
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
    """

    if not request.use_compile_cache:
        raise ValueError("cache warm cannot be used with use_compile_cache=False")
    plural = request.models is not None or request.inputs is not None
    keep_output = request.output_dir is not None
    output_root = Path(request.output_dir) if plural and keep_output else None
    resolved = resolve_requests(request)
    results = tuple(
        _warm_resolved(
            item,
            keep_output=keep_output,
            output_root=output_root,
        )
        for item in resolved
    )
    return results if plural else results[0]


__all__ = ["CacheWarmResult", "warm_cache"]
