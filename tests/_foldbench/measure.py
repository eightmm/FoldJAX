"""Time and peak-memory measurement that means the same thing for every port.

Both frameworks run kernels asynchronously, so a naive wall-clock measurement times
the *dispatch* rather than the work. Each backend is therefore given an explicit
synchronisation point, and the first call is treated as warm-up because it carries
compilation (JAX) or autotuning and allocator growth (torch).
"""

from __future__ import annotations

import gc
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Measurement:
    """What one measured call cost."""

    label: str
    seconds: float
    peak_bytes: int | None = None
    warmup_seconds: float | None = None
    repeats: int = 1
    samples: list[float] = field(default_factory=list)
    note: str = ""

    @property
    def peak_gib(self) -> float | None:
        return None if self.peak_bytes is None else self.peak_bytes / 2**30

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "seconds": self.seconds,
            "peak_bytes": self.peak_bytes,
            "peak_gib": self.peak_gib,
            "warmup_seconds": self.warmup_seconds,
            "repeats": self.repeats,
            "samples": self.samples,
            "note": self.note,
        }


def _torch_sync() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _jax_sync(result: Any) -> None:
    import jax

    jax.block_until_ready(result)


def measure_torch(
    label: str, call: Callable[[], Any], *, repeats: int = 3, warmup: bool = True
) -> Measurement:
    """Measure a torch callable, resetting the allocator's peak counter first."""
    import torch

    cuda = torch.cuda.is_available()
    warmup_seconds = None
    if warmup:
        start = time.perf_counter()
        call()
        _torch_sync()
        warmup_seconds = time.perf_counter() - start

    gc.collect()
    if cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        call()
        _torch_sync()
        samples.append(time.perf_counter() - start)

    return Measurement(
        label=label,
        seconds=statistics.median(samples),
        peak_bytes=int(torch.cuda.max_memory_allocated()) if cuda else None,
        warmup_seconds=warmup_seconds,
        repeats=repeats,
        samples=samples,
    )


def measure_jax(
    label: str, call: Callable[[], Any], *, repeats: int = 3, warmup: bool = True
) -> Measurement:
    """Measure a JAX callable.

    The warm-up call is where compilation happens, so its time is reported
    separately rather than folded into the result -- mixing them would make a
    compiled function look slower than an eager one at small sizes.
    """
    import jax

    warmup_seconds = None
    if warmup:
        start = time.perf_counter()
        _jax_sync(call())
        warmup_seconds = time.perf_counter() - start

    device = jax.devices()[0]
    # Peak is cumulative for the process, so it is read before and after and the
    # difference reported; JAX exposes no reset.
    before = (device.memory_stats() or {}).get("peak_bytes_in_use")

    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        _jax_sync(call())
        samples.append(time.perf_counter() - start)

    stats = device.memory_stats() or {}
    peak = stats.get("peak_bytes_in_use")
    note = ""
    if peak is not None and before is not None and peak == before:
        note = "peak unchanged from before this call; it may reflect earlier work"

    return Measurement(
        label=label,
        seconds=statistics.median(samples),
        peak_bytes=None if peak is None else int(peak),
        warmup_seconds=warmup_seconds,
        repeats=repeats,
        samples=samples,
        note=note,
    )
