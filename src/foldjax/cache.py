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

#: ``jax_persistent_cache_min_compile_time_secs`` wherever a prediction opens
#: its compilation cache: every executable is written, not only the ones that
#: took a second. A 254-token warm process compiled 46-175 small programs
#: afresh every time (L250_3dha, fixed-cost job 2329: 1.4 s OpenFold3, 1.6 s
#: Protenix, 2.7 s OpenDDE, 3.0 s ESMFold2, 11-34 ms each), while reading a
#: small entry back cost ~3 ms (ESMFold2's six sub-second-retrieval hits:
#: 20 ms together). Loading an entry returns the executable the first
#: process compiled for the same key, so what runs is what ran.
PERSISTENT_CACHE_MIN_COMPILE_SECS = 0.0


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


def _device_identity(device: Any) -> dict[str, Any]:
    """Return stable, public device attributes without retaining a JAX object."""

    def attribute(name: str, default: Any = None) -> Any:
        try:
            return getattr(device, name)
        except (AttributeError, RuntimeError):
            return default

    identity: dict[str, Any] = {
        "platform": str(attribute("platform", "unknown")),
        "device_kind": str(attribute("device_kind", "unknown")),
    }
    for name in ("id", "process_index", "local_hardware_id", "slice_index"):
        value = attribute(name)
        if value is not None:
            try:
                identity[name] = int(value)
            except (TypeError, ValueError, OverflowError):
                identity[name] = str(value)
    coordinates = attribute("coords")
    if coordinates is not None:
        try:
            identity["coords"] = [int(value) for value in coordinates]
        except (TypeError, ValueError, OverflowError):
            identity["coords"] = str(coordinates)
    return identity


def _topology_identity(
    devices: Any,
    *,
    process_count: int,
    local_device_count: int,
) -> dict[str, Any]:
    """Normalize JAX's selected topology into deterministic JSON."""
    identities = [_device_identity(device) for device in devices]
    identities.sort(key=lambda item: json.dumps(item, sort_keys=True))
    return {
        "process_count": int(process_count),
        "global_device_count": len(identities),
        "local_device_count": int(local_device_count),
        "devices": identities,
    }


def runtime_profile() -> dict[str, str]:
    """Return the JAX runtime and topology identity compiled code depends on.

    The original ``jax`` and ``platform`` fields remain stable API. ``jaxlib``
    and the selected device topology prevent operational cache namespaces from
    mixing executables produced by different compiler builds or accelerator
    layouts, even when the high-level JAX version is unchanged.
    """
    import jax
    import jaxlib

    devices = tuple(jax.devices())
    topology = _topology_identity(
        devices,
        process_count=jax.process_count(),
        local_device_count=jax.local_device_count(),
    )
    device_kinds = sorted(
        {str(device["device_kind"]) for device in topology["devices"]}
    )

    return {
        "jax": jax.__version__,
        "jaxlib": jaxlib.__version__,
        "platform": jax.default_backend(),
        "device_kind": ", ".join(device_kinds) if device_kinds else "unknown",
        # Keep the existing ``dict[str, str]`` API while retaining the full
        # identity in canonical JSON for cache hashing and diagnostics.
        "topology": json.dumps(topology, sort_keys=True, separators=(",", ":")),
    }


#: Set to ``1`` to load executables from a compile cache that other accounts
#: can write -- a lab's shared store, typically group-writable and setgid. JAX
#: runs whatever executable it finds under the key, so by default FoldJAX only
#: uses a cache directory no other account can write into.
TRUST_SHARED_COMPILE_CACHE_ENV = "FOLDJAX_TRUST_SHARED_COMPILE_CACHE"
_TRUE = {"1", "true", "yes", "on"}
_WARNED_CACHE_DIRS: set[str] = set()


def _make_cache_directories(path: Path) -> None:
    """Create missing components without the umask's group-write bit.

    ``Path.mkdir(parents=True)`` under a collaborative umask of 0002 makes
    every new namespace 0775, which the check below would then refuse.
    """
    missing: list[Path] = []
    current = path
    while not current.exists():
        missing.append(current)
        if current.parent == current:
            break
        current = current.parent
    for directory in reversed(missing):
        try:
            os.mkdir(directory, 0o755)
        except FileExistsError:
            pass


def _group_is_only_this_user(gid: int) -> bool:
    """A user-private group: the account's own group, with no other member."""
    try:
        import grp
        import pwd
    except ImportError:  # pragma: no cover - POSIX only
        return False
    if gid != os.getgid():
        return False
    try:
        members = set(grp.getgrgid(gid).gr_mem)
        name = pwd.getpwuid(os.geteuid()).pw_name
    except KeyError:
        return False
    return members <= {name}


def _untrusted_cache_reason(path: Path) -> str | None:
    """Why another account could place an executable under ``path``, if it can.

    The directory and every ancestor must belong to this user or root, and no
    other account may write to them: not the world (a root-owned sticky
    ancestor such as ``/tmp`` excepted) and not a group with another member.
    """
    import stat

    if not hasattr(os, "geteuid"):  # pragma: no cover - POSIX only
        return None
    uid = os.geteuid()
    leaf = Path(os.path.realpath(path))
    for directory in (leaf, *leaf.parents):
        info = os.stat(directory)
        if not stat.S_ISDIR(info.st_mode):
            return f"{directory} is not a directory"
        if info.st_uid not in (uid, 0):
            return f"{directory} belongs to another user"
        mode = stat.S_IMODE(info.st_mode)
        sticky_root = (
            info.st_uid == 0 and bool(mode & stat.S_ISVTX) and directory != leaf
        )
        if mode & stat.S_IWOTH and not sticky_root:
            return f"{directory} is world-writable"
        if (
            mode & stat.S_IWGRP
            and not sticky_root
            and not _group_is_only_this_user(info.st_gid)
        ):
            return f"{directory} is writable by a group with other members"
    return None


def trusted_compile_cache_dir(directory: str | os.PathLike[str]) -> Path | None:
    """Prepare ``directory`` and return it when its executables can be trusted.

    Returns ``None``, with one warning per directory, when another account can
    write into the cache or its ancestors; the caller then compiles without a
    persistent cache. ``FOLDJAX_TRUST_SHARED_COMPILE_CACHE=1`` skips the check
    for a deliberately shared store.
    """
    path = Path(directory).expanduser()
    if os.environ.get(TRUST_SHARED_COMPILE_CACHE_ENV, "").strip().lower() in _TRUE:
        path.mkdir(parents=True, exist_ok=True)
        return path
    try:
        _make_cache_directories(path)
        reason = _untrusted_cache_reason(path)
    except OSError as error:
        reason = f"it cannot be prepared ({error})"
    if reason is None:
        return path
    if str(path) not in _WARNED_CACHE_DIRS:
        _WARNED_CACHE_DIRS.add(str(path))
        import warnings

        warnings.warn(
            f"not using the compile cache {path}: {reason}, so another account "
            "could plant an executable there. Compiling without a persistent "
            f"cache; set {TRUST_SHARED_COMPILE_CACHE_ENV}=1 to trust a shared "
            "store, or use a cache directory only you can write.",
            RuntimeWarning,
            stacklevel=2,
        )
    return None


@contextmanager
def compilation_cache_scope(
    directory: Path | None,
    *,
    min_entry_size_bytes: int | None = None,
):
    """Apply one request's JAX cache setting and restore the host's afterwards.

    JAX exposes the persistent cache through process-wide config. FoldJAX
    predictions serialize while that setting is in force, and an embedding
    application gets its original config back even when prediction raises.
    Unrelated JAX compilation in another thread cannot be isolated from a
    process-global setting; mixed workloads must use a separate process.

    ``min_entry_size_bytes`` overrides XLA's size floor for what is worth
    writing. Only OpenFold3 passes it, as ``-1``: that port's slowest graphs to
    compile are small enough that the default floor skips exactly them. Left
    unset the floor is the host's, so no other backend's cache policy moves.
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
            if directory is not None:
                directory = trusted_compile_cache_dir(directory)
            if directory is None:
                jax.config.update("jax_compilation_cache_dir", None)
            else:
                jax.config.update("jax_compilation_cache_dir", str(directory))
                jax.config.update(
                    "jax_persistent_cache_min_compile_time_secs",
                    PERSISTENT_CACHE_MIN_COMPILE_SECS,
                )
                if min_entry_size_bytes is not None:
                    jax.config.update(
                        "jax_persistent_cache_min_entry_size_bytes",
                        min_entry_size_bytes,
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
