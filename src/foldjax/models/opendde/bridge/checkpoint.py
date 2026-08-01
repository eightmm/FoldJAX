"""OpenDDE checkpoint envelope handling."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def unwrap_state_dict(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    """Return a normalized OpenDDE model state dict.

    OpenDDE stores parameters under ``checkpoint["model"]`` and accepts DDP
    checkpoints whose keys start with ``module.``. A raw state dict remains
    useful for focused parity fixtures, so it is accepted as well.
    """

    payload: Any
    if "model" in checkpoint:
        payload = checkpoint["model"]
    elif "state_dict" in checkpoint:
        payload = checkpoint["state_dict"]
    else:
        payload = checkpoint
    if not isinstance(payload, Mapping):
        raise TypeError("OpenDDE checkpoint payload must be a model state dict")
    return {str(key).removeprefix("module."): value for key, value in payload.items()}
