"""Process-wide ownership for large, lazily loaded model data.

The native featurizers keep immutable chemistry dictionaries in module globals
so repeated seeds do not reload gigabytes of Python and RDKit objects.  Common
backend predictions lease those caches while they can be read and release the
last process-wide owner at a request-session boundary.
"""

from __future__ import annotations

import ctypes
import gc
import platform
import sys
import warnings
from collections.abc import Callable, Hashable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from threading import RLock


@dataclass
class _LeaseState:
    owners: int
    release_cache: Callable[[], bool]


_REGISTRY_LOCK = RLock()
_LEASES: dict[Hashable, _LeaseState] = {}
_RECLAIM_AT_RELEASE = True


def set_reclaim_at_release(enabled: bool) -> None:
    """Choose whether releasing a loaded cache also collects and trims.

    The cache is dropped either way, so its memory goes back to the allocator
    for whatever the process does next. Only the full ``gc.collect()`` and
    ``malloc_trim`` -- which hand pages back to the operating system, ~0.5 s
    in a warm ESMFold2 prediction -- are skipped. That is for a process that
    owns its lifetime, such as the CLI; a library would be deciding for its
    host, so the default reclaims.
    """

    global _RECLAIM_AT_RELEASE
    _RECLAIM_AT_RELEASE = bool(enabled)


def _malloc_trim() -> None:
    """Best-effort return of freed glibc arenas to the operating system."""

    if sys.platform != "linux":
        return
    try:
        if platform.libc_ver()[0].lower() != "glibc":
            return
        trim = ctypes.CDLL(None).malloc_trim
        trim.argtypes = (ctypes.c_size_t,)
        trim.restype = ctypes.c_int
        trim(0)
    except Exception:  # noqa: BLE001 - a trim is an optimisation, never a failure
        return


def _warn_release(what: str, error: Exception) -> None:
    warnings.warn(
        f"managed memory: {what} failed ({type(error).__name__}: {error}); "
        "the cache may stay resident until the process exits",
        RuntimeWarning,
        stacklevel=3,
    )


def _cleanup(state: _LeaseState) -> None:
    """Clear one cache and collect it without propagating cleanup failures.

    An ordinary failure is reported and absorbed, so it cannot replace the
    prediction outcome already in flight. ``KeyboardInterrupt`` and
    ``SystemExit`` still propagate: a cancelled run must stop when cancelled.
    """

    loaded = False
    try:
        loaded = bool(state.release_cache())
    except Exception as error:  # noqa: BLE001 - reported, never raised
        _warn_release("releasing a cache", error)
    if not loaded or not _RECLAIM_AT_RELEASE:
        return
    try:
        gc.collect()
    except Exception as error:  # noqa: BLE001 - reported, never raised
        _warn_release("collecting a released cache", error)
    try:
        _malloc_trim()
    except Exception as error:  # noqa: BLE001 - reported, never raised
        _warn_release("returning freed memory to the system", error)


@contextmanager
def lease(
    key: Hashable,
    release_cache: Callable[[], bool],
) -> Iterator[None]:
    """Keep one keyed cache alive until its last process-wide owner exits.

    Cleanup runs while the registry lock is held.  A new owner therefore
    cannot begin loading the same cache between its clear and allocator trim.
    Different backend instances using the same key share the same count.
    """

    with _REGISTRY_LOCK:
        state = _LEASES.get(key)
        if state is None:
            state = _LeaseState(owners=0, release_cache=release_cache)
            _LEASES[key] = state
        elif state.release_cache is not release_cache:
            raise RuntimeError(f"managed memory key {key!r} has two release helpers")
        state.owners += 1
    try:
        yield
    finally:
        with _REGISTRY_LOCK:
            current = _LEASES.get(key)
            if current is state:
                state.owners -= 1
                if state.owners == 0:
                    del _LEASES[key]
                    _cleanup(state)


__all__ = ["lease", "set_reclaim_at_release"]
