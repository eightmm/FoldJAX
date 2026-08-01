"""Small JAX memory helpers with backend-dependent fallbacks."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def device_memory_stats(device: Any | None = None) -> Mapping[str, Any]:
    """Return memory stats for a JAX device when the backend exposes them."""

    try:
        import jax
    except ImportError as exc:  # pragma: no cover - import environment dependent
        msg = "Install JAX before collecting memory stats."
        raise SystemExit(msg) from exc

    selected = device or jax.devices()[0]
    stats_fn = getattr(selected, "memory_stats", None)
    if stats_fn is None:
        return {}
    stats = stats_fn()
    return stats or {}


def peak_bytes(device: Any | None = None) -> int | None:
    """Return peak allocated bytes if available."""

    stats = device_memory_stats(device)
    for key in ("peak_bytes_in_use", "bytes_limit"):
        value = stats.get(key)
        if isinstance(value, int):
            return value
    return None
