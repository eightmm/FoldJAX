"""Stable namespacing for backend-specific caches.

Every backend hands FoldJAX's ``cache_dir`` to a native JAX compilation cache.
XLA keys entries by executable fingerprint, so one shared directory stays
correct but opaque: six models times several weight sets and shape buckets land
in a single flat pile that cannot be inspected, measured, or invalidated per
backend. Namespacing the root keeps each backend/weight/runtime combination in
its own subtree.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any

_UNSAFE = re.compile(r"[^A-Za-z0-9_.-]+")
_CONFIG_LOCK = RLock()


@dataclass(frozen=True, slots=True)
class CacheSnapshot:
    """Cheap, JSON-friendly size summary for one compilation namespace."""

    files: int = 0
    bytes: int = 0

    def summary(self) -> dict[str, int]:
        return {"files": self.files, "bytes": self.bytes}


def cache_snapshot(directory: Path) -> CacheSnapshot:
    """Count regular cache files without following user-controlled symlinks."""

    directory = Path(directory)
    if not directory.is_dir() or directory.is_symlink():
        return CacheSnapshot()
    files = 0
    bytes_ = 0
    for root, directories, names in os.walk(directory, followlinks=False):
        parent = Path(root)
        directories[:] = [
            name for name in directories if not (parent / name).is_symlink()
        ]
        for name in names:
            path = parent / name
            try:
                if path.is_symlink() or not path.is_file():
                    continue
                size = path.stat().st_size
            except OSError:
                # JAX may atomically replace a cache entry while it is counted.
                # A snapshot is telemetry, not a readiness gate, so skip the
                # transient path rather than turning a successful warm into an
                # unrelated filesystem race.
                continue
            files += 1
            bytes_ += size
    return CacheSnapshot(files=files, bytes=bytes_)


def _profile_value(value: Any) -> Any:
    """Turn option values into stable JSON without erasing their basic type."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): _profile_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_profile_value(item) for item in value]
    return str(value)


def cache_namespace(
    root: Path,
    *,
    model: str,
    weight_id: str,
    profile: Mapping[str, Any],
) -> Path:
    """Return ``root/model/weight_id/digest`` for one compile-relevant profile."""
    safe_weight = _UNSAFE.sub("_", weight_id).strip("_") or "weights"
    payload = json.dumps(
        _profile_value(dict(profile)), sort_keys=True, separators=(",", ":")
    )
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


@contextmanager
def compilation_cache_scope(directory: Path | None):
    """Apply one request's JAX cache setting and restore the host's afterwards.

    JAX exposes the persistent cache through process-wide config. FoldJAX
    predictions serialize while that setting is in force, and an embedding
    application gets its original config back even when prediction raises.
    Unrelated JAX compilation in another thread cannot be isolated from a
    process-global setting; mixed workloads must use a separate process.
    """
    import jax
    from jax.experimental.compilation_cache import compilation_cache

    names = (
        "jax_compilation_cache_dir",
        "jax_persistent_cache_min_compile_time_secs",
        "jax_persistent_cache_min_entry_size_bytes",
    )
    with _CONFIG_LOCK:
        previous = {name: getattr(jax.config, name) for name in names}
        try:
            # JAX constructs its file-cache object at most once. Updating the
            # config alone after the first compilation leaves that object
            # pointed at the old directory, so reset before selecting this
            # request's namespace (or disabling it).
            compilation_cache.reset_cache()
            if directory is None:
                jax.config.update("jax_compilation_cache_dir", None)
            else:
                Path(directory).mkdir(parents=True, exist_ok=True)
                jax.config.update("jax_compilation_cache_dir", str(directory))
                jax.config.update(
                    "jax_persistent_cache_min_compile_time_secs", 1.0
                )
            yield
        finally:
            # Drop the request-scoped file-cache object before restoring the
            # embedding application's config. Its next compilation lazily
            # recreates the object at the original directory.
            try:
                compilation_cache.reset_cache()
            finally:
                for name, value in previous.items():
                    jax.config.update(name, value)
