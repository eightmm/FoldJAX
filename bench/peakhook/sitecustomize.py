"""Report an observed process's live device memory without editing that project.

Every upstream here ships its own virtualenv and its own entry point, so the
measurement cannot live inside the call. Putting this directory on PYTHONPATH
makes Python import it as `sitecustomize` at startup, before the project's own
code runs; it registers an atexit hook that writes
`torch.cuda.max_memory_allocated()` by default, or an already-initialized JAX
GPU's `peak_bytes_in_use` when ``BENCH_PEAK_ENGINE=jax``, to the file named by
``BENCH_PEAK_FILE``.

`max_memory_allocated` is the live-bytes high-water mark, which is the same
quantity `peak_bytes_in_use` reports on the JAX side. `nvidia-smi` is not: it
shows the caching allocator's reserved pool, which grows by doubling and
therefore tracks the allocator's schedule rather than the model's need.

With BENCH_TRACE_DIR set, the established torch observer also samples that
quantity over time, so a trace and a table row measure one thing. See
`benchtrace.py`.
"""

from __future__ import annotations

import atexit
import os
import sys


def _peak_engine() -> str:
    """Keep torch's established observer unless JAX was explicitly requested."""
    return "jax" if os.environ.get("BENCH_PEAK_ENGINE") == "jax" else "torch"


def _torch_peak() -> int | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return int(torch.cuda.max_memory_allocated())
    except Exception:  # pragma: no cover - the run itself already failed
        return None


def _jax_peak() -> int | None:
    """Read an initialized CUDA backend without importing or initializing JAX.

    JAX 0.11 keeps initialized clients in ``xla_bridge._backends``.  Calling
    the public ``jax.devices()`` or ``xla_bridge.backends()`` here would create
    a backend for an otherwise unloaded process, changing the benchmark the
    observer is meant only to describe.
    """
    bridge = sys.modules.get("jax._src.xla_bridge")
    backends = getattr(bridge, "_backends", None)
    if not isinstance(backends, dict):
        return None
    for name, backend in backends.items():
        if name not in {"cuda", "gpu"} and getattr(backend, "platform", None) not in {
            "cuda",
            "gpu",
        }:
            continue
        try:
            devices = backend.local_devices()
        except Exception:
            continue
        for device in devices:
            try:
                stats = device.memory_stats()
                peak = stats.get("peak_bytes_in_use") if stats is not None else None
                if isinstance(peak, int) and not isinstance(peak, bool) and peak >= 0:
                    return peak
            except Exception:
                continue
    return None


def _peak_for_engine(engine: str) -> int | None:
    return _jax_peak() if engine == "jax" else _torch_peak()


def _report(destination: str, engine: str) -> None:
    peak = _peak_for_engine(engine)
    if peak is None:
        return
    try:
        with open(destination, "w", encoding="utf-8") as handle:
            handle.write(str(peak))
    except OSError:
        pass


def _install() -> None:
    destination = os.environ.get("BENCH_PEAK_FILE")
    if not destination:
        return
    engine = _peak_engine()

    def report() -> None:
        _report(destination, engine)

    atexit.register(report)


def _install_trace() -> None:
    """Start the memory-over-time sampler, if this run asked for one.

    Guarded because it is an observer: an upstream prediction must not fail
    because the instrument did.
    """
    try:
        import benchtrace  # same directory, already on sys.path via PYTHONPATH

        benchtrace.start_when_torch_ready()
    except Exception:
        pass


_install()
_install_trace()
