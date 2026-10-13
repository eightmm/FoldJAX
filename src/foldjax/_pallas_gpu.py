"""Register unnamed CUDA devices in JAX's Pallas-Triton hardware registry."""

from __future__ import annotations

import threading
from typing import Any

from foldjax._gpu import compute_capability

_LOCK = threading.Lock()


def install(device: Any | None = None) -> int:
    """Preserve JAX's known/AOT targets and custom entries; add real CUDA devices."""
    import jax
    from jax._src.pallas.triton import gpu_info

    devices = list(jax.local_devices())
    chosen = device if device is not None else jax.config.jax_default_device
    if isinstance(chosen, str):
        devices.extend(jax.local_devices(backend=chosen))
    elif chosen is not None:
        devices.append(chosen)
    added = 0
    with _LOCK:
        for device in devices:
            if device.platform not in {"gpu", "cuda"}:
                continue
            kind = device.device_kind
            capability = compute_capability(device)
            if (
                capability is None
                or gpu_info.gpu_version_from_device_kind(kind) is not None
            ):
                continue
            if kind in gpu_info.registry:
                continue
            major, minor = capability
            info = gpu_info.GpuInfo(
                gpu_version=None,
                arch_name=f"{major}.{minor}",
                compute_capability=major * 10 + minor,
            )
            gpu_info.registry[kind] = lambda info=info: info
            added += 1
    return added
