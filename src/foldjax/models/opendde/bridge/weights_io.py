"""Native OpenDDE-JAX weights and isolated trusted-checkpoint conversion."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from foldjax.models.opendde.bridge.checkpoint import unwrap_state_dict
from foldjax.models.opendde.bridge.torch_mapping import map_opendde_inference_state_dict
from foldjax.models.protenix.bridge.weights_io import (
    load_native_weights,
    save_native_weights,
)


def load_torch_checkpoint(path: str | Path) -> Any:
    """Convert one trusted official ``.pt`` checkpoint to JAX parameters.

    PyTorch is imported only inside this optional bridge. Production prediction
    loads the native archive with :func:`load_native_weights` and never imports
    PyTorch. The safe ``weights_only`` loader is deliberately not followed by
    an unrestricted pickle fallback.
    """

    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"missing checkpoint: {checkpoint_path}")

    try:
        from foldjax import torch_archive

        checkpoint = torch_archive.load(checkpoint_path)
    except Exception:
        import torch

        checkpoint = torch.load(
            str(checkpoint_path),
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
    return map_opendde_inference_state_dict(unwrap_state_dict(checkpoint))


__all__ = [
    "load_native_weights",
    "load_torch_checkpoint",
    "save_native_weights",
]
