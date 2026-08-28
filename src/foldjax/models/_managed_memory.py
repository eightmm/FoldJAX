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
    except BaseException:  # cleanup is never allowed to replace prediction state
        return


def _cleanup(state: _LeaseState) -> None:
    """Clear one cache and collect it without propagating cleanup failures."""

    loaded = False
    try:
        loaded = bool(state.release_cache())
    except BaseException:
        pass
    if not loaded:
        return
    try:
        gc.collect()
    except BaseException:
        pass
    try:
        _malloc_trim()
    except BaseException:
        pass


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


__all__ = ["lease"]
