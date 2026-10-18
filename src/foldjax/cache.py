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
import sys
from collections.abc import Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any

from foldjax._cache_tiers import (
    CacheLocations,
    cache_locations,
    current_cache_locations,
    open_directory,
    prepare_private_directory,
)

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

# These models changed their graph ABI and must be warmed again.
# Keep other model namespaces stable; revision changes are model-specific.
_GRAPH_REVISIONS = {
    "esmfold2": "2026-10-18-compact-lm-chain64",
    "protenix": "2026-10-18-chain64",
    "opendde": "2026-10-18-chain64",
    "openfold3": "2026-10-18-chain64",
}


def cache_revision(model: str) -> str | None:
    """The declared graph ABI that an operator must warm for this model."""
    return _GRAPH_REVISIONS.get(model)


def private_compile_cache_root() -> Path:
    """A user cache independent of a shared FoldJAX home or source checkout."""
    explicit = os.environ.get("FOLDJAX_PRIVATE_COMPILE_CACHE")
    if explicit:
        return Path(explicit).expanduser()
    root = os.environ.get("XDG_CACHE_HOME")
    return (
        (Path(root).expanduser() if root else Path.home() / ".cache")
        / "foldjax"
        / "compile"
    )


def clear_record() -> None:
    module = sys.modules.get("foldjax._jax_cache_tiers")
    if module is not None:
        module.clear_record()


def recorded() -> dict[str, Any] | None:
    module = sys.modules.get("foldjax._jax_cache_tiers")
    return module.recorded() if module is not None else None


@dataclass(frozen=True, slots=True)
class CacheSnapshot:
    """Cheap, JSON-friendly size summary for one compilation namespace."""

    files: int = 0
    bytes: int = 0

    def summary(self) -> dict[str, int]:
        return {"files": self.files, "bytes": self.bytes}


def cache_snapshot(directory: Path, *, warn_unshareable: bool = False) -> CacheSnapshot:
    """Count regular files, optionally checking shared publication modes.

    Private cache snapshots leave the optional sharing check disabled. Like
    the size count, it skips symlinks and never repairs existing permissions.
    """

    directory = Path(directory)
    if not directory.is_dir() or directory.is_symlink():
        return CacheSnapshot()
    files = 0
    bytes_ = 0
    unshareable: tuple[Path, int] | None = None
    trusted_group = warn_unshareable and shared_compile_cache_trusted()

    def inspect_mode(path: Path, info: os.stat_result, required: int) -> None:
        nonlocal unshareable
        if (
            warn_unshareable
            and unshareable is None
            and info.st_mode & required != required
        ):
            unshareable = path, info.st_mode & 0o7777

    for root, directories, names in os.walk(directory, followlinks=False):
        parent = Path(root)
        if warn_unshareable:
            try:
                inspect_mode(parent, parent.stat(), 0o050 if trusted_group else 0o005)
            except OSError:
                pass
        directories[:] = [
            name for name in directories if not (parent / name).is_symlink()
        ]
        for name in names:
            path = parent / name
            try:
                if path.is_symlink() or not path.is_file():
                    continue
                info = path.stat()
            except OSError:
                # JAX may atomically replace a cache entry while it is counted.
                # A snapshot is telemetry, not a readiness gate, so skip the
                # transient path rather than turning a successful warm into an
                # unrelated filesystem race.
                continue
            files += 1
            bytes_ += info.st_size
            inspect_mode(path, info, 0o040 if trusted_group else 0o004)
    if unshareable is not None:
        import warnings

        path, mode = unshareable
        if trusted_group:
            audience = "members of the trusted shared group"
            repair = (
                "Give that group directory read/search permissions (g+rx) "
                "and entry read permissions (g+r)."
            )
        else:
            audience = "other accounts"
            repair = (
                "Make shared directories readable/searchable (0755) and "
                "entries readable (0644), or run foldjax setup --shared "
                "--apply for a managed installation."
            )
        warnings.warn(
            f"the warmed shared compile cache {directory} contains {path} "
            f"with mode {mode:04o}; {audience} may not be able to read "
            f"its entries. {repair}",
            RuntimeWarning,
            stacklevel=2,
        )
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
    identity = _profile_value(dict(profile))
    revision = cache_revision(model)
    if revision is not None:
        identity = {"graph_revision": revision, "profile": identity}
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":"))
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


def device_identity(device: Any) -> dict[str, Any]:
    """Return stable, public device attributes without retaining a JAX object.

    The one description of a device. It is digested into every compile-cache
    namespace (`runtime_profile`), so its output is an on-disk contract;
    in-process session keys take :func:`device_key` instead.
    """

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


def device_key(device: Any) -> tuple[str, str]:
    """A hashable, process-local key naming one concrete device.

    For retaining a runner or parameter tree bound to a device: its type plus
    :func:`device_identity`. A device whose attributes cannot be read gets a
    key of its own object, so an opaque device splits a session rather than
    sharing one with another device.
    """
    kind = f"{type(device).__module__}.{type(device).__qualname__}"
    try:
        return kind, json.dumps(device_identity(device), sort_keys=True)
    except Exception:  # noqa: BLE001 - an opaque device must split the session
        return kind, f"unavailable:{id(device)}"


def _topology_identity(
    devices: Any,
    *,
    process_count: int,
    local_device_count: int,
) -> dict[str, Any]:
    """Normalize JAX's selected topology into deterministic JSON."""
    identities = [device_identity(device) for device in devices]
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

    from foldjax._gpu import default_device

    selected = default_device()
    devices = tuple(jax.devices(backend=selected.platform))
    topology = _topology_identity(
        devices,
        process_count=jax.process_count(),
        local_device_count=jax.local_device_count(backend=selected.platform),
    )
    return {
        "jax": jax.__version__,
        "jaxlib": jaxlib.__version__,
        "platform": selected.platform,
        "device_kind": selected.device_kind,
        "selected_device": json.dumps(
            device_identity(selected), sort_keys=True, separators=(",", ":")
        ),
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
    every new namespace 0775, which the check after it would then refuse.
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


def _untrusted_cache_reason(path: Path, *, leaf: bool = True) -> str | None:
    """Why another account could place an executable under ``path``, if it can.

    The directory and every ancestor must belong to this user or root, and no
    other account may write to them: not the world (a root-owned sticky
    ancestor such as ``/tmp`` excepted) and not a group with another member.
    """
    if not hasattr(os, "geteuid"):  # pragma: no cover - POSIX only
        return None
    resolved = Path(os.path.realpath(path))
    for directory in (resolved, *resolved.parents):
        reason = _untrusted_directory_reason(
            directory, leaf=leaf and directory == resolved
        )
        if reason is not None:
            return reason
    return None


def _untrusted_directory_reason(directory: Path, *, leaf: bool) -> str | None:
    """`_untrusted_cache_reason` for one directory of the walk, if it fails."""
    import stat

    info = os.stat(directory)
    if not stat.S_ISDIR(info.st_mode):
        return f"{directory} is not a directory"
    if info.st_uid not in (os.geteuid(), 0):
        return f"{directory} belongs to another user"
    mode = stat.S_IMODE(info.st_mode)
    sticky_root = info.st_uid == 0 and bool(mode & stat.S_ISVTX) and not leaf
    if mode & stat.S_IWOTH and not sticky_root:
        return f"{directory} is world-writable"
    if (
        mode & stat.S_IWGRP
        and not sticky_root
        and not _group_is_only_this_user(info.st_gid)
    ):
        return f"{directory} is writable by a group with other members"
    return None


def shared_compile_cache_trusted() -> bool:
    """Whether `TRUST_SHARED_COMPILE_CACHE_ENV` opts in to a shared store."""
    value = os.environ.get(TRUST_SHARED_COMPILE_CACHE_ENV, "")
    return value.strip().lower() in _TRUE


def _deepest_existing(path: Path) -> Path:
    """``path`` itself, or the nearest ancestor of it that exists."""
    current = path
    while not current.exists() and current.parent != current:
        current = current.parent
    return current


def compile_cache_untrusted_reason(
    directory: str | os.PathLike[str],
) -> str | None:
    """Why `trusted_compile_cache_dir` would refuse ``directory``, if it would.

    Read-only: nothing is created and nothing is warned, so a report such as
    `foldjax doctor` can ask before any run has made the namespace.
    """
    if shared_compile_cache_trusted():
        return None
    try:
        path = Path(directory).expanduser()
        existing = _deepest_existing(path)
        return _untrusted_cache_reason(existing, leaf=existing == path)
    except OSError as error:
        return f"it cannot be inspected ({error})"


def _warn_once(path: Path, message: str) -> None:
    if str(path) in _WARNED_CACHE_DIRS:
        return
    _WARNED_CACHE_DIRS.add(str(path))
    import warnings

    warnings.warn(message, RuntimeWarning, stacklevel=3)


def _warn_readonly_cache(path: Path) -> Path:
    if not os.access(path, os.W_OK | os.X_OK):
        _warn_once(
            path,
            f"the compile cache {path} is not writable by this account: "
            "existing entries load read-only; new entries require a writable "
            "cache tier. Its owner can repair a shared store with "
            f"chmod -R g+w {path}.",
        )
    return path


def trusted_compile_cache_dir(directory: str | os.PathLike[str]) -> Path | None:
    """Prepare ``directory`` and return it when its executables can be trusted.

    Returns ``None``, with one warning per directory, when another account can
    write into the cache or its ancestors; the caller can use another trusted
    tier or compile without persistence. The trust decision uses the nearest
    existing ancestor before creation, so a refused cache leaves no empty
    namespace behind. ``FOLDJAX_TRUST_SHARED_COMPILE_CACHE=1`` skips the check
    for a deliberately shared store, whose new directories then take the
    umask's mode (and a setgid parent's group) like the rest of it.
    """
    path = Path(directory).expanduser()
    if shared_compile_cache_trusted():
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            _warn_once(
                path,
                f"not using the compile cache {path}: it cannot be created "
                f"({error}). Skipping this namespace; make the "
                "shared store group-writable (chmod -R g+w) to share it.",
            )
            return None
        return _warn_readonly_cache(path)
    try:
        existing = _deepest_existing(path)
        reason = _untrusted_cache_reason(existing, leaf=existing == path)
        if reason is None:
            _make_cache_directories(path)
            reason = _untrusted_cache_reason(path)
    except OSError as error:
        reason = f"it cannot be prepared ({error})"
    if reason is None:
        return _warn_readonly_cache(path)
    _warn_once(
        path,
        f"not using the compile cache {path}: {reason}. Its executables could "
        "not be verified as trusted. Skipping this namespace; "
        f"set {TRUST_SHARED_COMPILE_CACHE_ENV}=1 to trust a shared "
        "store, or use a cache directory only you can write.",
    )
    return None


def _layered_shared_directory(directory: Path) -> Path | None:
    path = Path(directory).expanduser()
    reason = compile_cache_untrusted_reason(path)
    if reason is not None:
        _warn_once(path, f"not using the shared compile cache {path}: {reason}")
        return None
    try:
        # Create through held descriptors too: checking after Path.mkdir would
        # already have created namespaces through an unsafe symlink ancestor.
        # Shared reads need traversal without granting other accounts write
        # access; the separately prepared private tier keeps mode 0700.
        with open_directory(
            path,
            private=False,
            create=True,
            mode=0o777 if shared_compile_cache_trusted() else 0o755,
        ):
            pass
    except OSError as error:
        _warn_once(
            path,
            f"not using the shared compile cache {path}: "
            f"safe directory access failed ({error})",
        )
        return None
    return _warn_readonly_cache(path)


@contextmanager
def compilation_cache_scope(
    directory: Path | None,
    *,
    private_directory: Path | None = None,
    min_entry_size_bytes: int | None = None,
):
    """Apply one request's JAX cache setting and restore the host's afterwards.

    JAX exposes the persistent cache through process-wide config. FoldJAX
    predictions serialize while that setting is in force, and an embedding
    application gets its original config back even when prediction raises.
    Unrelated JAX compilation in another thread cannot be isolated from a
    process-global setting; mixed workloads must use a separate process.

    ``private_directory`` is the same profile under a user-owned cache root.
    Reads prefer shared entries; writes fall back to private. Equal namespaces
    share one tier, and nested backend scopes inherit the active private tier.
    Layered scopes disable XLA's separate native caches because their writes
    bypass the recoverable CacheInterface path, including quota/mount errors.

    ``min_entry_size_bytes`` overrides XLA's size floor for what is worth
    writing. Only OpenFold3 passes it, as ``-1``: that port's slowest graphs to
    compile are small enough that the default floor skips exactly them. Left
    unset the floor is the host's, so no other backend's cache policy moves.
    """
    import jax
    from jax.experimental.compilation_cache import compilation_cache

    active = current_cache_locations()
    if (
        directory is not None
        and active is not None
        and private_directory is None
        and active.requested_shared is not None
        and os.path.abspath(directory) == os.path.abspath(active.requested_shared)
    ):
        with _CONFIG_LOCK:
            previous_floor = jax.config.jax_persistent_cache_min_entry_size_bytes
            try:
                if min_entry_size_bytes is not None:
                    jax.config.update(
                        "jax_persistent_cache_min_entry_size_bytes",
                        min_entry_size_bytes,
                    )
                yield
            finally:
                jax.config.update(
                    "jax_persistent_cache_min_entry_size_bytes", previous_floor
                )
        return

    names = (
        "jax_compilation_cache_dir",
        "jax_persistent_cache_min_compile_time_secs",
        "jax_persistent_cache_min_entry_size_bytes",
        "jax_compilation_cache_max_size",
        "jax_persistent_cache_enable_xla_caches",
    )
    with _CONFIG_LOCK:
        previous = {name: getattr(jax.config, name) for name in names}
        requested_shared = Path(directory) if directory is not None else None
        layered = private_directory is not None
        try:
            # JAX constructs its file-cache object at most once. Updating the
            # config alone after the first compilation leaves that object
            # pointed at the old directory, so reset before selecting this
            # request's namespace (or disabling it).
            compilation_cache.reset_cache()
            if directory is not None:
                directory = (
                    _layered_shared_directory(directory)
                    if layered
                    else trusted_compile_cache_dir(directory)
                )
            private = None
            if private_directory is not None and (
                directory is None
                or os.path.realpath(directory) != os.path.realpath(private_directory)
            ):
                try:
                    private = prepare_private_directory(private_directory)
                except OSError as error:
                    _warn_once(
                        Path(private_directory),
                        f"not using the private compile cache {private_directory}: "
                        f"{error}",
                    )
            locations = CacheLocations(requested_shared, directory, private)
            # XLA auxiliary caches bypass CacheInterface and its recoverable
            # write fallback, including quota/mount failures after access().
            selected = private or locations.write_directory or directory
            if layered:
                jax.config.update("jax_persistent_cache_enable_xla_caches", "none")
            if selected is None:
                jax.config.update("jax_compilation_cache_dir", None)
            else:
                readonly = locations.write_directory is None
                jax.config.update("jax_compilation_cache_dir", str(selected))
                jax.config.update(
                    "jax_persistent_cache_min_compile_time_secs",
                    float("inf") if readonly else PERSISTENT_CACHE_MIN_COMPILE_SECS,
                )
                if readonly:
                    # Bounded JAX caches write access times and acquire a
                    # writable lock even on hits. XLA's auxiliary caches also
                    # write directly, outside JAX's recoverable cache errors.
                    jax.config.update("jax_compilation_cache_max_size", -1)
                    jax.config.update("jax_persistent_cache_enable_xla_caches", "none")
                if min_entry_size_bytes is not None:
                    jax.config.update(
                        "jax_persistent_cache_min_entry_size_bytes",
                        min_entry_size_bytes,
                    )
            with ExitStack() as stack:
                stack.enter_context(cache_locations(locations))
                if layered:
                    from foldjax import _jax_cache_tiers

                    def trusted_file(info: os.stat_result) -> bool:
                        if shared_compile_cache_trusted():
                            return True
                        return info.st_uid in (os.geteuid(), 0) and not (
                            info.st_mode & 0o002
                            or info.st_mode & 0o020
                            and not _group_is_only_this_user(info.st_gid)
                        )

                    stack.enter_context(
                        _jax_cache_tiers.installed(
                            locations,
                            shared_validator=compile_cache_untrusted_reason,
                            shared_file_validator=trusted_file,
                            shared_mode=0o666
                            if shared_compile_cache_trusted()
                            else 0o644,
                        )
                    )
                    if requested_shared is not None and directory is None:
                        _jax_cache_tiers.note("lookups", "shared", "unavailable")
                    if (
                        private_directory is not None
                        and private is None
                        and (
                            directory is None
                            or os.path.realpath(directory)
                            != os.path.realpath(private_directory)
                        )
                    ):
                        _jax_cache_tiers.note("writes", "private", "unavailable")
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
