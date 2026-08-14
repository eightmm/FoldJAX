"""Sample a running process's live device bytes, for a memory-over-time trace.

Every peak in this directory is the live-bytes high-water mark --
`max_memory_allocated` under torch, `peak_bytes_in_use` under JAX -- and
`bench/README.md` says why `nvidia-smi` is not used: it reports the caching
allocator's reserved pool, which grows by doubling and so tracks the
allocator's schedule rather than the model's need.

A trace has to sample *that same quantity*, for two reasons beyond consistency:

* The driver's used-bytes is monotone over a run. Neither allocator returns
  memory to the driver mid-prediction -- XLA's pool only grows, and torch's
  caching allocator retains its segments -- so an `nvidia-smi` curve is a
  running maximum, and the shape a trace exists to show (a sampler spiking once
  per step, a trunk holding a plateau) is not in that signal at all.
* A figure whose maximum disagrees with the table printed beside it is worse
  than no figure. Sampling live bytes makes the curve's top the table's number.

So the sampler runs *inside* the measured process, which also means the trace
needs nothing turned off: it reads correctly with XLA preallocation on, which is
what `foldjax predict` and every row in `results/` actually run under.

Two callers, one loop. `sitecustomize` starts the torch side without editing any
upstream file; `bench/run_foldjax.py` starts the JAX side directly, where the
device handle is already in hand.
"""

from __future__ import annotations

import atexit
import json
import os
import sys
import threading
import time

#: 4 Hz. Fine enough to show a 200-step sampler's envelope, coarse enough that
#: the sampling thread never competes with the run for the GIL.
DEFAULT_INTERVAL = 0.25

#: How long to wait for a process to touch CUDA before concluding it never
#: will. The upstream drivers spawn the real worker and are themselves
#: CUDA-free, so this expiring is the normal case for them, not a failure.
DEFAULT_WAIT_S = 1800.0

_lock = threading.Lock()


def _write(handle, row: dict) -> None:
    """One JSON line, flushed.

    Written as the run goes rather than collected and dumped at exit: a run
    that dies on an allocation is exactly the run whose curve is worth having,
    and it never reaches an exit handler.
    """
    with _lock:
        handle.write(json.dumps(row) + "\n")
        handle.flush()


def start(read_current, read_peak, *, backend: str, directory: str | None = None,
          interval: float | None = None) -> bool:
    """Sample `read_current()` until the process exits. False if not tracing.

    The file is named by pid, because more than one process can inherit
    `BENCH_TRACE_DIR` through the environment and only one of them is the
    measurement. `bench.trace` reassembles them and says so when it finds more
    than one that produced samples.
    """
    directory = directory or os.environ.get("BENCH_TRACE_DIR")
    if not directory:
        return False
    if interval is None:
        interval = float(os.environ.get("BENCH_TRACE_INTERVAL", DEFAULT_INTERVAL))

    os.makedirs(directory, exist_ok=True)
    handle = open(
        os.path.join(directory, f"trace-{os.getpid()}.jsonl"), "w", encoding="utf-8"
    )
    started = time.perf_counter()
    _write(
        handle,
        {
            "pid": os.getpid(),
            "backend": backend,
            "interval_s": interval,
            "argv": sys.argv,
        },
    )

    def loop() -> None:
        while True:
            try:
                _write(
                    handle,
                    {
                        "t": round(time.perf_counter() - started, 3),
                        "bytes": int(read_current()),
                    },
                )
            except Exception:  # the run itself owns the failure, not the trace
                return
            time.sleep(interval)

    def footer() -> None:
        # The sampler is a 4 Hz observer of a process that allocates far faster
        # than that, so its largest sample is a lower bound. The high-water mark
        # the allocator kept is the real peak, and it is the number the tables
        # report -- record it so the figure can draw the difference instead of
        # quietly understating it.
        try:
            _write(
                handle,
                {
                    "peak_bytes": int(read_peak()),
                    "t_end": round(time.perf_counter() - started, 3),
                },
            )
        except Exception:
            pass

    threading.Thread(target=loop, daemon=True).start()
    atexit.register(footer)
    return True


def start_when_torch_ready(wait_s: float = DEFAULT_WAIT_S) -> None:
    """Begin sampling once this process has actually initialized CUDA.

    `sitecustomize` runs before the project's first import, so there is nothing
    to read yet. Waiting on `sys.modules` rather than importing torch keeps a
    process that never uses CUDA -- an upstream driver, say -- from paying for
    the import or writing an empty trace that would then have to be told apart
    from the real one.
    """
    if not os.environ.get("BENCH_TRACE_DIR"):
        return

    def wait() -> None:
        deadline = time.perf_counter() + wait_s
        while time.perf_counter() < deadline:
            torch = sys.modules.get("torch")
            try:
                if torch is not None and torch.cuda.is_initialized():
                    start(
                        torch.cuda.memory_allocated,
                        torch.cuda.max_memory_allocated,
                        backend="torch",
                    )
                    return
            except Exception:
                # `sys.modules` holds the module object from the moment its
                # import *begins*, so a poll landing mid-import finds a `torch`
                # with no `cuda` attribute yet. Treating that as terminal is
                # what silently produced empty traces: keep waiting instead.
                pass
            time.sleep(0.2)

    threading.Thread(target=wait, daemon=True).start()
