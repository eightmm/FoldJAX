"""Report a torch process's device memory without editing that project.

Every upstream here ships its own virtualenv and its own entry point, so the
measurement cannot live inside the call. Putting this directory on PYTHONPATH
makes Python import it as `sitecustomize` at startup, before the project's own
code runs; it registers an atexit hook that writes
`torch.cuda.max_memory_allocated()` to the file named by BENCH_PEAK_FILE.

`max_memory_allocated` is the live-bytes high-water mark, which is the same
quantity `peak_bytes_in_use` reports on the JAX side. `nvidia-smi` is not: it
shows the caching allocator's reserved pool, which grows by doubling and
therefore tracks the allocator's schedule rather than the model's need.

With BENCH_TRACE_DIR set, the same process also samples that quantity over
time, so a trace and a table row measure one thing. See `benchtrace.py`.
"""

from __future__ import annotations

import atexit
import os


def _install() -> None:
    destination = os.environ.get("BENCH_PEAK_FILE")
    if not destination:
        return

    def report() -> None:
        try:
            import torch

            if not torch.cuda.is_available():
                return
            peak = torch.cuda.max_memory_allocated()
        except Exception:  # pragma: no cover - the run itself already failed
            return
        try:
            with open(destination, "w", encoding="utf-8") as handle:
                handle.write(str(peak))
        except OSError:
            pass

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
