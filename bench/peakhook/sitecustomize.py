"""Report a torch process's peak device memory without editing that project.

Every upstream here ships its own virtualenv and its own entry point, so the
measurement cannot live inside the call. Putting this directory on PYTHONPATH
makes Python import it as `sitecustomize` at startup, before the project's own
code runs; it registers an atexit hook that writes
`torch.cuda.max_memory_allocated()` to the file named by BENCH_PEAK_FILE.

`max_memory_allocated` is the live-bytes high-water mark, which is the same
quantity `peak_bytes_in_use` reports on the JAX side. `nvidia-smi` is not: it
shows the caching allocator's reserved pool, which grows by doubling and
therefore tracks the allocator's schedule rather than the model's need.
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


_install()
