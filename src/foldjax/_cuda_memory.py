"""Read CUDA capacity without inferring device ordinals or importing JAX."""

from __future__ import annotations

import ctypes
import functools
import os
from typing import Any

from foldjax._gpu import compute_capability


@functools.lru_cache(maxsize=1)
def _visible_device_driver():
    # The pre-JAX query also works on drivers without cuStreamGetDevice.
    library = ctypes.CDLL("nvcuda.dll" if os.name == "nt" else "libcuda.so.1")
    library.cuInit.argtypes = [ctypes.c_uint]
    library.cuInit.restype = ctypes.c_int
    library.cuDeviceGetCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
    library.cuDeviceGetCount.restype = ctypes.c_int
    library.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    library.cuDeviceGet.restype = ctypes.c_int
    library.cuDeviceTotalMem_v2.argtypes = [
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_int,
    ]
    library.cuDeviceTotalMem_v2.restype = ctypes.c_int
    return library


def visible_device_bytes() -> tuple[int, ...] | None:
    """Query CUDA-visible capacities before JAX creates an allocator or context.

    CUDA resolves visibility masks, UUIDs and device ordering. Incomplete or
    implausible inventory is unknown rather than a partially usable result.
    """
    try:
        driver = _visible_device_driver()
        if driver.cuInit(0):
            return None
        count = ctypes.c_int()
        if driver.cuDeviceGetCount(ctypes.byref(count)):
            return None
        if not 0 < count.value <= 1024:
            return None
        capacities = []
        for ordinal in range(count.value):
            device = ctypes.c_int()
            if driver.cuDeviceGet(ctypes.byref(device), ordinal):
                return None
            total = ctypes.c_size_t()
            if driver.cuDeviceTotalMem_v2(ctypes.byref(total), device):
                return None
            # Reject wrapped negative values and corrupted native results.
            if not 0 < total.value <= 2**50:
                return None
            capacities.append(total.value)
        return tuple(capacities)
    except (
        AttributeError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        OverflowError,
    ):
        return None


@functools.lru_cache(maxsize=1)
def _driver():
    library = ctypes.CDLL("nvcuda.dll" if os.name == "nt" else "libcuda.so.1")
    library.cuStreamGetDevice.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
    library.cuStreamGetDevice.restype = ctypes.c_int
    library.cuDeviceTotalMem_v2.argtypes = [
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_int,
    ]
    library.cuDeviceTotalMem_v2.restype = ctypes.c_int
    return library


def total_device_bytes(device: Any) -> int | None:
    if getattr(device, "platform", None) not in {"gpu", "cuda"}:
        return None
    if compute_capability(device) is None:
        return None
    try:
        stream = device.get_stream_for_external_ready_events()
        if stream is None:
            return None
        driver = _driver()
        cuda_device = ctypes.c_int()
        # The stream identifies the actual device even with visibility masks,
        # reordered ordinals and a different process-default JAX backend.
        if driver.cuStreamGetDevice(ctypes.c_void_p(stream), ctypes.byref(cuda_device)):
            return None
        total = ctypes.c_size_t()
        if driver.cuDeviceTotalMem_v2(ctypes.byref(total), cuda_device):
            return None
        return total.value or None
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
        # Older CUDA drivers may not export cuStreamGetDevice. An unknown
        # capacity is safer than dividing a pool by a changed environment knob.
        return None
