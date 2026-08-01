"""Stable namespacing for backend-specific caches.

Every backend hands FoldJAX's ``cache_dir`` to a native JAX compilation cache.
XLA keys entries by executable fingerprint, so one shared directory stays
correct but opaque: four models times several weight sets and shape buckets land
in a single flat pile that cannot be inspected, measured, or invalidated per
backend. Namespacing the root keeps each backend/weight/runtime combination in
its own subtree.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

_UNSAFE = re.compile(r"[^A-Za-z0-9_.-]+")


def cache_namespace(
    root: Path,
    *,
    model: str,
    weight_id: str,
    profile: Mapping[str, Any],
) -> Path:
    """Return ``root/model/weight_id/digest`` for one compile-relevant profile."""
    safe_weight = _UNSAFE.sub("_", weight_id).strip("_") or "weights"
    payload = json.dumps(dict(profile), sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode()).hexdigest()[:16]
    return Path(root) / model / safe_weight / digest


def weight_identity(weights: Path) -> tuple[str, str]:
    """Return a readable label and a fully qualifying identity for ``weights``.

    The label names the cache subdirectory; the identity goes into the profile
    digest so two checkpoints that happen to share a basename cannot collide.
    File weights include their size because sibling checkpoints are commonly
    replaced in place under one name.
    """
    resolved = Path(weights).resolve()
    identity = str(resolved)
    if resolved.is_file():
        identity = f"{identity}:{resolved.stat().st_size}"
    return resolved.name, identity


def runtime_profile() -> dict[str, str]:
    """Return the JAX runtime identity that compiled executables depend on."""
    import jax

    return {
        "jax": jax.__version__,
        "platform": jax.default_backend(),
    }
