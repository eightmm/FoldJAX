"""JAX-free resolution of OpenFold3 compile-time choices.

This lives outside ``foldjax.models.openfold3`` deliberately.  Importing a
model submodule first executes that package's ``__init__`` and pulls in the
JAX inference runtime, while the common backend needs these helpers for cache
planning and capability discovery without paying that import cost.
"""

from __future__ import annotations

import os
import stat
import threading
import time
from collections import OrderedDict
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, NamedTuple

TRIANGLE_BACKEND_ENV = "OPENFOLD3_TRIANGLE_BACKEND"
_TRIANGLE_BACKEND_LOCK = threading.RLock()
_CACHE_SCOPE_LIMIT = 128
_CACHE_SCOPE_SNAPSHOTS: OrderedDict[str, tuple[Any, ...]] = OrderedDict()


class CacheScopeToken(NamedTuple):
    """One pre-call cache observation used to close mutation races."""

    key: str
    previous: tuple[Any, ...] | None
    current: tuple[Any, ...]
    invalidated: bool


def resolve_triangle_kernel(value: str | None, *, cp_shards: int) -> str:
    """Return the triangle kernel the next trace will actually build.

    With no explicit or ambient choice, serial OpenFold3 uses cuEquivariance.
    Context parallelism instead defaults to XLA because its distributed
    triangle-attention path cannot assume the fused extension is available.
    These are the existing dispatch semantics, made explicit for cache keys.
    """

    # ``triangle_backend`` temporarily changes a process-global variable while
    # tracing. Read it under the same lock, otherwise an unrelated direct call
    # can resolve the transient value and cache a graph under the wrong key.
    with _TRIANGLE_BACKEND_LOCK:
        selected = value
        if selected is None:
            selected = os.environ.get(TRIANGLE_BACKEND_ENV)
        if selected is None:
            return "xla" if cp_shards > 1 else "cueq"
        return str(selected).lower()


def _cache_scope_key(scope: str) -> str:
    # Preserve the lexical path. If its final component or any parent symlink
    # is retargeted, the key must remain the same so the changed target is
    # compared against the generation previously observed at this spelling.
    return str(Path(scope).expanduser().absolute())


def canonical_cache_scope(scope: str) -> str:
    """Freeze a cache directory before later calls can change working directory."""

    return _cache_scope_key(scope)


def _cache_scope_snapshot(scope: str) -> tuple[Any, ...]:
    """Cheap identity of a JAX cache directory and its direct entries.

    JAX stores persistent executable entries directly below the configured
    directory. A shallow metadata snapshot catches deletion, replacement and
    in-place corruption without recursively walking (or reading) caches whose
    payloads can be hundreds of megabytes.
    """

    key = _cache_scope_key(scope)
    path = Path(key)
    try:
        lexical = path.lstat()
        root = path.stat()
        if not path.is_dir():
            return (
                key,
                (
                    "not-directory",
                    lexical.st_mode,
                    lexical.st_dev,
                    lexical.st_ino,
                    lexical.st_size,
                    lexical.st_mtime_ns,
                    lexical.st_ctime_ns,
                ),
                (),
            )
        link_target = os.readlink(path) if path.is_symlink() else None
        entries = []
        with os.scandir(path) as iterator:
            for entry in iterator:
                # JAX's size-bounded LRU rewrites access-time sidecars on a
                # normal cache hit, and lockfiles are transient coordination
                # metadata. The corresponding ``*-cache`` payload remains in
                # the snapshot, so excluding these avoids false invalidation
                # without hiding executable deletion or corruption.
                if entry.name.endswith(".lockfile"):
                    continue
                if entry.name.endswith("-atime"):
                    info = entry.stat(follow_symlinks=False)
                    entries.append(
                        (
                            entry.name,
                            "volatile-presence",
                            info.st_mode,
                            info.st_dev,
                            info.st_ino,
                        )
                    )
                    continue
                info = entry.stat(follow_symlinks=False)
                entries.append(
                    (
                        entry.name,
                        info.st_mode,
                        info.st_dev,
                        info.st_ino,
                        info.st_size,
                        info.st_mtime_ns,
                        info.st_ctime_ns,
                    )
                )
        return (
            key,
            (
                "directory",
                lexical.st_mode,
                lexical.st_dev,
                lexical.st_ino,
                link_target,
                root.st_mode,
                root.st_dev,
                root.st_ino,
            ),
            tuple(sorted(entries)),
        )
    except (OSError, RuntimeError):
        return (key, ("missing-or-unreadable",), ())


def _cache_scope_invalidated(
    previous: tuple[Any, ...], current: tuple[Any, ...]
) -> bool:
    """Whether content that existed in ``previous`` disappeared or changed."""

    if len(previous) == 3 and previous[1][0] == "population-missing":
        return True
    if previous == current:
        return False
    if len(previous) != 3 or len(current) != 3 or previous[:2] != current[:2]:
        return True
    previous_entries = {entry[0]: entry[1:] for entry in previous[2]}
    current_entries = {entry[0]: entry[1:] for entry in current[2]}
    # Another process may legitimately populate additional JAX entries. Keep
    # the warm in-memory program when every entry we observed is unchanged;
    # deletions, replacements and in-place corruption still force a recompile.
    return any(
        current_entries.get(name) != identity
        for name, identity in previous_entries.items()
    )


def _cache_scope_population_ready(
    snapshot: tuple[Any, ...], *, require_atime: bool
) -> bool:
    """Whether a successful compile left a usable persistent entry behind."""

    if len(snapshot) != 3:
        return False
    entries = {entry[0]: entry[1:] for entry in snapshot[2]}
    payloads = {name for name in entries if name.endswith("-cache")}
    if not payloads:
        return False
    if not all(_cache_entry_is_regular(entries[name]) for name in payloads):
        return False
    if not require_atime:
        return True
    return all(
        _cache_entry_is_regular(entries.get(f"{name[: -len('-cache')]}-atime", ()))
        for name in payloads
    )


def _cache_entry_is_regular(identity: tuple[Any, ...]) -> bool:
    if not identity:
        return False
    mode_index = 1 if identity[0] == "volatile-presence" else 0
    return len(identity) > mode_index and stat.S_ISREG(identity[mode_index])


@contextmanager
def _cache_scope_file_lock(directory: Path, *, directory_fd: int) -> Iterator[bool]:
    """Acquire the advisory lock used by JAX's bounded file cache."""

    lock_path = directory / ".lockfile"
    try:
        import fcntl
    except ImportError:  # pragma: no cover - non-POSIX JAX uses filelock
        try:
            import filelock
        except ImportError:
            yield False
            return
        lock = filelock.FileLock(str(lock_path))
        try:
            lock.acquire(timeout=10)
        except (filelock.Timeout, OSError):
            yield False
            return
        try:
            yield True
        finally:
            lock.release()
        return

    while True:
        try:
            descriptor = os.open(
                ".lockfile",
                os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0),
                0o600,
                dir_fd=directory_fd,
            )
            break
        except InterruptedError:
            continue
        except OSError:
            yield False
            return
    acquired = False
    try:
        deadline = time.monotonic() + 10.0
        while True:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    break
                time.sleep(0.05)
            except InterruptedError:
                continue
            except OSError:
                break
        if not acquired:
            yield False
            return
        yield True
    finally:
        if acquired:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            except OSError:
                pass
        os.close(descriptor)


def _cache_scope_entries_from_fd(directory_fd: int) -> dict[str, tuple[Any, ...]]:
    """Return direct-entry identities from one already-anchored directory."""

    entries: dict[str, tuple[Any, ...]] = {}
    for name in os.listdir(directory_fd):
        if name.endswith(".lockfile"):
            continue
        try:
            info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        except OSError:
            continue
        if name.endswith("-atime"):
            entries[name] = (
                "volatile-presence",
                info.st_mode,
                info.st_dev,
                info.st_ino,
            )
        else:
            entries[name] = (
                info.st_mode,
                info.st_dev,
                info.st_ino,
                info.st_size,
                info.st_mtime_ns,
                info.st_ctime_ns,
            )
    return entries


def _remove_invalid_cache_artifact(name: str, *, directory_fd: int) -> None:
    """Remove a cache file, quarantining directories so removal stays shallow."""

    try:
        mode = os.stat(name, dir_fd=directory_fd, follow_symlinks=False).st_mode
    except OSError:
        return
    if stat.S_ISDIR(mode):
        quarantine = f".foldjax-invalid-{name}-{os.getpid()}-{time.time_ns()}"
        os.rename(
            name,
            quarantine,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
        )
    else:
        try:
            os.unlink(name, dir_fd=directory_fd)
        except FileNotFoundError:
            pass


def _repair_cache_scope_atime_pairs(
    scope: str, snapshot: tuple[Any, ...]
) -> tuple[Any, ...]:
    """Remove bounded-cache payloads whose required atime sidecar vanished."""

    directory = Path(_cache_scope_key(scope))
    try:
        directory_fd = os.open(
            directory,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
    except OSError:
        return snapshot
    try:
        with _cache_scope_file_lock(directory, directory_fd=directory_fd) as acquired:
            if not acquired:
                return snapshot
            # List and mutate through the same directory fd. Retargeting a
            # lexical symlink or replacing its directory cannot redirect a
            # repair into a different cache generation.
            entries = _cache_scope_entries_from_fd(directory_fd)
            for name, identity in tuple(entries.items()):
                if not name.endswith("-cache"):
                    continue
                atime_name = f"{name[: -len('-cache')]}-atime"
                atime_identity = entries.get(atime_name, ())
                if _cache_entry_is_regular(identity) and _cache_entry_is_regular(
                    atime_identity
                ):
                    continue
                try:
                    _remove_invalid_cache_artifact(name, directory_fd=directory_fd)
                    if atime_identity:
                        _remove_invalid_cache_artifact(
                            atime_name, directory_fd=directory_fd
                        )
                except OSError:
                    # Keep a raced or unreadable entry visible to the readiness
                    # gate; it must not become a trusted baseline.
                    pass
    finally:
        os.close(directory_fd)
    return _cache_scope_snapshot(scope)


def inspect_cache_scope(
    scope: str | None, *, repair_atime: bool = False
) -> CacheScopeToken | None:
    """Inspect a persistent cache immediately before a JIT operation.

    Callers hold :data:`_TRIANGLE_BACKEND_LOCK`, the same lock that serialises
    OpenFold compilation. Extra entries from another writer are normal shared
    cache growth; a previously observed entry disappearing or changing is a
    generation change that requires repopulation.
    """

    if scope is None:
        return None
    key = _cache_scope_key(scope)
    current = _cache_scope_snapshot(key)
    if repair_atime:
        current = _repair_cache_scope_atime_pairs(key, current)
    previous = _CACHE_SCOPE_SNAPSHOTS.get(key)
    if previous is not None:
        _CACHE_SCOPE_SNAPSHOTS.move_to_end(key)
    return CacheScopeToken(
        key=key,
        previous=previous,
        current=current,
        invalidated=(
            previous is not None and _cache_scope_invalidated(previous, current)
        ),
    )


def cache_scope_changed(scope: str | None) -> bool:
    """Whether a previously observed persistent cache changed externally."""

    token = inspect_cache_scope(scope)
    return token is not None and token.invalidated


def observe_cache_scope(
    scope: str | None,
    *,
    token: CacheScopeToken | None = None,
    require_payload: bool = False,
    require_atime: bool = False,
) -> None:
    """Remember a successful operation without absorbing an in-call deletion.

    With an established, non-invalidated generation, retain the pre-call
    observation. Pure additions already present at the start are absorbed, but
    deleting entries during an in-memory hit remains visible on the next call.
    The first observation and a deliberate invalidation instead take a fresh
    post-call snapshot so a successful compilation establishes its baseline.
    """

    if scope is None:
        return
    key = _cache_scope_key(scope)
    if token is not None and token.key != key:
        raise ValueError("cache scope token does not match the observed scope")
    if token is not None and token.previous is not None and not token.invalidated:
        snapshot = token.current
    else:
        snapshot = _cache_scope_snapshot(key)
        if require_atime:
            snapshot = _repair_cache_scope_atime_pairs(key, snapshot)
        population_ready = _cache_scope_population_ready(
            snapshot, require_atime=require_atime
        )
        already_retried = (
            token is not None
            and token.previous is not None
            and len(token.previous) == 3
            and token.previous[1][0] == "population-missing"
        )
        if require_payload and not population_ready and not already_retried:
            # ``cache_scope`` is an explicit promise that persistent caching is
            # enabled (the backend and CLI pass the path returned by their
            # cache enabler). If a first/recovery compilation leaves no payload
            # -- or a bounded cache leaves an orphan payload without its access
            # metadata -- do not bless it as stable. The next call retries.
            # Some valid JAX configurations intentionally write no entry, so a
            # second miss (including an unrepairable cache) is treated as a
            # stable unavailable namespace rather than retracing forever.
            snapshot = (
                snapshot[0],
                ("population-missing", snapshot[1]),
                snapshot[2],
            )
    _CACHE_SCOPE_SNAPSHOTS[key] = snapshot
    _CACHE_SCOPE_SNAPSHOTS.move_to_end(key)
    while len(_CACHE_SCOPE_SNAPSHOTS) > _CACHE_SCOPE_LIMIT:
        _CACHE_SCOPE_SNAPSHOTS.popitem(last=False)


def reset_cache_scope_tracking() -> None:
    """Forget process-local cache observations when executable state clears."""

    _CACHE_SCOPE_SNAPSHOTS.clear()


@contextmanager
def triangle_backend(value: str | None) -> Iterator[None]:
    """Pin one resolved kernel without leaking it across calls or threads."""

    # The model reads a process-global environment variable during tracing.
    # Serialise that small Python scope so two first-use compilations cannot
    # trace kernel A under the static identity for kernel B. An RLock permits
    # the backend's outer eager scope to contain compile_predict's trace scope.
    with _TRIANGLE_BACKEND_LOCK:
        previous = os.environ.get(TRIANGLE_BACKEND_ENV)
        if value is not None:
            os.environ[TRIANGLE_BACKEND_ENV] = value
        try:
            yield
        finally:
            if value is not None:
                if previous is None:
                    os.environ.pop(TRIANGLE_BACKEND_ENV, None)
                else:
                    os.environ[TRIANGLE_BACKEND_ENV] = previous
