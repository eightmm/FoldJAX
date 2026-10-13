"""Read CUDA architecture without inferring it from a GPU marketing name."""

from __future__ import annotations

import re
from typing import Any


def compute_capability(device: Any) -> tuple[int, int] | None:
    value = getattr(device, "compute_capability", None)
    if callable(value):
        value = value()
    if isinstance(value, (tuple, list)) and len(value) == 2:
        major, minor = value
        if not all(isinstance(v, int) and not isinstance(v, bool) for v in value):
            return None
    else:
        if isinstance(value, bool):
            return None
        match = re.fullmatch(r"(?:sm_?)?(\d+)(?:\.(\d+))?", str(value))
        if match is None:
            return None
        major = int(match[1])
        if match[2] is not None:
            minor = int(match[2])
        elif major >= 50:
            major, minor = divmod(major, 10)
        else:
            minor = 0
    return (major, minor) if major > 0 and 0 <= minor < 10 else None


def default_device(request: Any | None = None):
    import jax

    if request is not None and request.model == "alphafold3":
        options = request.options
        platform = options.get("platform")
        if platform is not None or "device" in options:
            platform = platform or default_device().platform
            return jax.local_devices(backend=platform)[int(options.get("device", 0))]
    chosen = jax.config.jax_default_device
    if isinstance(chosen, str):
        return jax.local_devices(backend=chosen)[0]
    return chosen if chosen is not None else jax.local_devices()[0]


def unsupported_gpu_reason(device: Any) -> str | None:
    if device.platform not in {"gpu", "cuda"}:
        return None
    capability = compute_capability(device)
    if capability is not None and capability < (8, 0):
        major, minor = capability
        return (
            "FoldJAX requires NVIDIA compute capability sm_80 or newer; "
            f"{device.device_kind} reports sm_{major}{minor}. "
            "This GPU architecture is unsupported."
        )
    return None


def require_supported_gpu(device: Any) -> None:
    if reason := unsupported_gpu_reason(device):
        raise ValueError(reason)


def require_fused_cuda(what: str) -> None:
    if reason := unsupported_gpu_reason(default_device()):
        raise ValueError(f"{what}: {reason}")


def compatibility_notes(devices) -> list[str]:
    return [reason for device in devices if (reason := unsupported_gpu_reason(device))]
