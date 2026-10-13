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


def default_device():
    import jax

    chosen = jax.config.jax_default_device
    if isinstance(chosen, str):
        return jax.local_devices(backend=chosen)[0]
    return chosen if chosen is not None else jax.local_devices()[0]


def require_fused_cuda(what: str) -> None:
    device = default_device()
    capability = compute_capability(device)
    if capability is not None and capability < (8, 0):
        major, minor = capability
        raise ValueError(
            f"{what} requires NVIDIA compute capability sm_80 or newer; "
            f"{device.device_kind} reports sm_{major}{minor}. "
            "Use the model's XLA kernel options with float32 precision "
            "(for Boltz-2/OpenFold3: --option triangle_kernel=xla "
            "--option glu_backend=xla --option dtype=float32; Boltz-2 also needs "
            "BOLTZ_JAX_TRIANGLE_MULTIPLICATION_BACKEND=xla), or run on CPU. "
            "This is a fused-kernel limitation, not GPU global-memory exhaustion."
        )


def compatibility_notes(devices) -> list[str]:
    notes = []
    for device in devices:
        capability = compute_capability(device)
        if capability is not None and capability < (8, 0):
            major, minor = capability
            notes.append(
                f"{device.device_kind} (sm_{major}{minor}): fused Pallas/Tokamax "
                "paths require sm_80+; use model-specific XLA options with "
                "float32 precision. JAX GPU support does not imply "
                "fused-kernel support."
            )
    return notes
