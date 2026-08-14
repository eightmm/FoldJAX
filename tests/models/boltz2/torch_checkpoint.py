"""Publisher-parity conversion for NumPy-backed Boltz checkpoint values."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from foldjax.models.boltz2.bridge.checkpoint import load_checkpoint_state_dict


def load_torch_checkpoint_state_dict(path: Path) -> dict[str, Any]:
    """Load with FoldJAX's restricted reader, then make parity tensors."""

    return torch_state_dict(load_checkpoint_state_dict(path))


def torch_state_dict(state: dict[str, Any]) -> dict[str, Any]:
    """Convert NumPy arrays while preserving non-array checkpoint metadata."""

    import torch

    converted = state.copy()
    for key, value in state.items():
        if not isinstance(value, np.ndarray):
            continue
        if value.dtype.name == "bfloat16":
            # torch.from_numpy does not accept ml_dtypes.bfloat16 arrays.
            converted[key] = torch.from_numpy(value.astype(np.float32)).to(
                dtype=torch.bfloat16
            )
            continue
        array = value if value.flags.writeable else value.copy()
        converted[key] = torch.from_numpy(array)
    return converted
