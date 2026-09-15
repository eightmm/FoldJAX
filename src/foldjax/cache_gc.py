"""Reclaim space in the persistent compilation cache.

`foldjax cache gc` is the only command that deletes from the FoldJAX store, so
it is kept apart from the rest of the CLI: the scan pins a directory
descriptor and walks it with `os.fwalk`, and every removal is made relative to
that descriptor, so a path swapped underneath the command cannot widen what it
touches. The argument surface and the dispatch stay in `foldjax.cli`; this
module is the body they call.

Nothing here imports JAX, directly or transitively. The allocator and
rendezvous state the CLI sets up has to be in place before the runtime is
imported, and a discovery command must not be what imports it.
"""

from __future__ import annotations

import argparse
import dataclasses
import errno
import json
import math
import os
import sys
from pathlib import Path

from foldjax import paths


def _parse_size(text: str) -> int:
    """Accept 20G / 500M / 1024 the way every other disk tool does."""
    value = text.strip().upper().rstrip("B")
    scale = {"K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}
    factor = 1
    if value and value[-1] in scale:
        factor, value = scale[value[-1]], value[:-1]
    try:
        size = float(value)
    except ValueError as error:
        raise ValueError(
            f"--max-size must look like 20G or 500M; got {text!r}"
        ) from error
    if not math.isfinite(size):
        raise ValueError(f"--max-size must look like 20G or 500M; got {text!r}")
    if size <= 0:
        raise ValueError("--max-size must be positive")
    return int(size * factor)


def _open_cache_root(root: Path) -> int | None:
    """Pin a real cache directory so later path swaps cannot widen GC scope."""
    required = ("O_DIRECTORY", "O_NOFOLLOW")
    if any(not hasattr(os, name) for name in required) or not hasattr(os, "fwalk"):
        raise RuntimeError(
            "cache gc requires directory-descriptor and no-follow filesystem support"
        )
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    flags |= getattr(os, "O_CLOEXEC", 0)
    try:
        return os.open(root, flags)
    except OSError as error:
        if error.errno in {errno.ENOENT, errno.ENOTDIR, errno.ELOOP}:
            return None
        raise


_TOKAMAX_AUTOTUNE_DIRECTORIES = frozenset(
    (".tokamax-autotuning-v1", ".tokamax-autotuning-v2")
)
_CacheGcEntry = tuple[Path, float, int, int, int, int, str | None]


@dataclasses.dataclass(frozen=True)
class _CacheGcScan:
    entries: tuple[_CacheGcEntry, ...]
    protected_files: int
    protected_bytes: int

    @property
    def total_files(self) -> int:
        return len(self.entries) + self.protected_files

    @property
    def total_bytes(self) -> int:
        return sum(item[2] for item in self.entries) + self.protected_bytes


def _tokamax_temporary_lock_name(name: str) -> str | None:
    """Return the exact lock name for a v2 atomic-write temporary."""
    if not (name.startswith(".") and name.endswith(".tmp")):
        return None
    try:
        result_name, process_id, timestamp = name[1:-4].rsplit(".", maxsplit=2)
    except ValueError:
        return None
    if not result_name.endswith(".json"):
        return None
    signature = result_name.removesuffix(".json")
    if (
        len(signature) != 64
        or any(character not in "0123456789abcdef" for character in signature)
        or not process_id.isascii()
        or not process_id.isdecimal()
        or not timestamp.isascii()
        or not timestamp.isdecimal()
    ):
        return None
    return f"{signature}.lockfile"


def _acquire_tokamax_gc_lock(directory_fd: int, lock_name: str) -> int | None:
    """Non-blockingly pin an existing regular Tokamax lock for safe GC."""
    import fcntl
    import stat

    flags = os.O_RDWR | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    try:
        lock_fd = os.open(lock_name, flags, dir_fd=directory_fd)
    except OSError:
        return None
    try:
        if not stat.S_ISREG(os.fstat(lock_fd).st_mode):
            os.close(lock_fd)
            return None
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (OSError, ValueError):
        os.close(lock_fd)
        return None
    return lock_fd


def _release_tokamax_gc_lock(lock_fd: int) -> None:
    import fcntl

    try:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
    finally:
        os.close(lock_fd)


def _raise_cache_gc_walk_error(error: OSError) -> None:
    """Never turn an incomplete filesystem scan into a successful budget report."""
    raise error


def _cache_gc_usage(root_fd: int) -> tuple[int, int]:
    """Measure regular-file usage below an already pinned cache root."""
    import stat

    files = 0
    size = 0
    for _directory, _directories, names, directory_fd in os.fwalk(
        ".",
        topdown=True,
        onerror=_raise_cache_gc_walk_error,
        follow_symlinks=False,
        dir_fd=root_fd,
    ):
        for name in names:
            try:
                info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            except FileNotFoundError:
                continue
            if stat.S_ISREG(info.st_mode):
                files += 1
                size += info.st_size
    return files, size


def _cache_gc_entries(
    root_fd: int,
) -> _CacheGcScan:
    """List stable regular-file identities under a pinned cache root.

    Tokamax lock files and unrecognised temporaries remain protected but count
    toward the real cache footprint. An exactly named v2 temporary is eligible
    only when its existing per-signature lock can be acquired without waiting;
    apply acquires that same lock again and holds it through unlink.
    """
    import stat

    entries: list[_CacheGcEntry] = []
    protected_files = 0
    protected_bytes = 0
    for directory, _directories, names, directory_fd in os.fwalk(
        ".",
        topdown=True,
        onerror=_raise_cache_gc_walk_error,
        follow_symlinks=False,
        dir_fd=root_fd,
    ):
        parent = Path(directory)
        for name in names:
            try:
                info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            except FileNotFoundError:
                continue
            if not stat.S_ISREG(info.st_mode):
                continue
            lock_name = None
            if not _TOKAMAX_AUTOTUNE_DIRECTORIES.isdisjoint(parent.parts):
                if name.endswith(".lockfile"):
                    protected_files += 1
                    protected_bytes += info.st_size
                    continue
                if name.endswith(".tmp"):
                    lock_name = (
                        _tokamax_temporary_lock_name(name)
                        if ".tokamax-autotuning-v2" in parent.parts
                        else None
                    )
                    lock_fd = (
                        None
                        if lock_name is None
                        else _acquire_tokamax_gc_lock(directory_fd, lock_name)
                    )
                    if lock_fd is None:
                        protected_files += 1
                        protected_bytes += info.st_size
                        continue
                    _release_tokamax_gc_lock(lock_fd)
            entries.append(
                (
                    parent / name,
                    info.st_mtime,
                    info.st_size,
                    info.st_dev,
                    info.st_ino,
                    info.st_mtime_ns,
                    lock_name,
                )
            )
    return _CacheGcScan(
        entries=tuple(entries),
        protected_files=protected_files,
        protected_bytes=protected_bytes,
    )


def _apply_cache_gc(
    root: Path,
    root_fd: int,
    doomed: list[_CacheGcEntry],
) -> tuple[int, int, int, int, int]:
    """Delete selected files through pinned parent descriptors.

    Returns ``(removed_files, removed_bytes, failed_files,
    already_absent_files, changed_files)``. A concurrent collector removing the
    same entry is success for the desired state, while a path replaced since
    the report was planned is left for a later GC pass.
    """
    pending = {
        relative: (size, device, inode, mtime_ns, lock_name)
        for relative, _mtime, size, device, inode, mtime_ns, lock_name in doomed
    }
    removed_files = 0
    removed_bytes = 0
    failed_files = 0
    already_absent_files = 0
    changed_files = 0
    for directory, directories, names, directory_fd in os.fwalk(
        ".",
        topdown=False,
        onerror=_raise_cache_gc_walk_error,
        follow_symlinks=False,
        dir_fd=root_fd,
    ):
        parent = Path(directory)
        for name in names:
            relative = parent / name
            if relative not in pending:
                continue
            size, device, inode, mtime_ns, lock_name = pending.pop(relative)
            lock_fd = (
                None
                if lock_name is None
                else _acquire_tokamax_gc_lock(directory_fd, lock_name)
            )
            if lock_name is not None and lock_fd is None:
                changed_files += 1
                continue
            try:
                try:
                    current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                except FileNotFoundError:
                    already_absent_files += 1
                    continue
                except OSError as error:
                    failed_files += 1
                    print(
                        f"[cache] could not inspect {root / relative}: {error}",
                        file=sys.stderr,
                    )
                    continue
                if (
                    current.st_dev,
                    current.st_ino,
                    current.st_mtime_ns,
                    current.st_size,
                ) != (device, inode, mtime_ns, size):
                    changed_files += 1
                    continue
                try:
                    os.unlink(name, dir_fd=directory_fd)
                except FileNotFoundError:
                    already_absent_files += 1
                except OSError as error:
                    failed_files += 1
                    print(
                        f"[cache] could not remove {root / relative}: {error}",
                        file=sys.stderr,
                    )
                else:
                    removed_files += 1
                    removed_bytes += size
            finally:
                if lock_fd is not None:
                    _release_tokamax_gc_lock(lock_fd)

        # Children have already been visited. ``rmdir`` is itself an atomic
        # emptiness and no-follow check; a symlink or concurrent writer simply
        # makes it fail without escaping this pinned parent descriptor.
        for name in directories:
            try:
                os.rmdir(name, dir_fd=directory_fd)
            except OSError:
                continue

    # Entries removed or renamed before the second descriptor walk no longer
    # need collection and are not failures of this invocation.
    already_absent_files += len(pending)
    return (
        removed_files,
        removed_bytes,
        failed_files,
        already_absent_files,
        changed_files,
    )


def run_cache_gc(args: argparse.Namespace) -> int:
    """Reclaim compile-cache space, reporting first and deleting only on request.

    The cache is derived data keyed by accelerator, runtime, weight identity and
    shapes. Removing entries preserves the model and shape semantics, but costs
    a recompile or Tokamax retune; a retune may select a different valid
    floating-point schedule and therefore change last bits. It is still
    someone's disk, and a command that deletes gigabytes because it was run to
    see what was there is not a good trade, so the report is the default and
    `--apply` is the verb.
    """
    from foldjax.cli import _format_bytes

    if args.older_than is None and args.max_size is None:
        raise ValueError("cache gc needs --older-than DAYS, --max-size SIZE, or both")
    # Parsed before the store is inspected, so a typo is reported the same way
    # whether or not a cache happens to exist yet.
    budget = None if args.max_size is None else _parse_size(args.max_size)
    if args.older_than is not None and args.older_than < 0:
        raise ValueError("--older-than must be a non-negative number of days")
    root = paths.compile_cache_dir()
    root_fd = _open_cache_root(root)
    if root_fd is None:
        print(f"[cache] nothing at {root}", file=sys.stderr)
        print(
            json.dumps(
                {
                    "root": str(root),
                    "applied": bool(args.apply),
                    "total_files": 0,
                    "total_bytes": 0,
                    "protected_files": 0,
                    "protected_bytes": 0,
                    "planned_removed_files": 0,
                    "planned_removed_bytes": 0,
                    "removed_files": 0,
                    "removed_bytes": 0,
                    "failed_files": 0,
                    "already_absent_files": 0,
                    "changed_files": 0,
                    "remaining_bytes": 0,
                    "budget_satisfied": True if budget is not None else None,
                }
            )
        )
        return 0

    import time

    try:
        scan = _cache_gc_entries(root_fd)
        entries = list(scan.entries)
        entries.sort(key=lambda item: item[1], reverse=True)

        doomed: list[_CacheGcEntry] = []
        if args.older_than is not None:
            cutoff = time.time() - args.older_than * 86400
            doomed = [item for item in entries if item[1] < cutoff]
        if budget is not None:
            kept = scan.protected_bytes
            over: list[_CacheGcEntry] = []
            for item in entries:
                if kept + item[2] <= budget:
                    kept += item[2]
                else:
                    over.append(item)
            chosen = {item[0] for item in doomed}
            doomed.extend(item for item in over if item[0] not in chosen)

        planned_removed_files = len(doomed)
        planned_removed_bytes = sum(item[2] for item in doomed)
        removed_files = planned_removed_files
        removed_bytes = planned_removed_bytes
        failed_files = 0
        already_absent_files = 0
        changed_files = 0
        if args.apply:
            (
                removed_files,
                removed_bytes,
                failed_files,
                already_absent_files,
                changed_files,
            ) = _apply_cache_gc(root, root_fd, doomed)
            _remaining_files, remaining_bytes = _cache_gc_usage(root_fd)
        else:
            remaining_bytes = scan.total_bytes - planned_removed_bytes
    finally:
        os.close(root_fd)
    action = "removed" if args.apply else "would remove"
    outcome = f"{removed_files} file(s), {_format_bytes(removed_bytes)}"
    if args.apply and failed_files:
        outcome += (
            f" (planned {planned_removed_files} file(s), "
            f"{_format_bytes(planned_removed_bytes)}; {failed_files} failed)"
        )
    elif args.apply and already_absent_files:
        outcome += f" ({already_absent_files} already absent)"
    if args.apply and changed_files:
        outcome += f" ({changed_files} changed since planning; kept)"
    print(
        f"[cache] {action} {outcome} of "
        f"{_format_bytes(scan.total_bytes)} under {root}"
        + ("" if args.apply else "; pass --apply to do it"),
        file=sys.stderr,
    )
    print(
        json.dumps(
            {
                "root": str(root),
                "applied": bool(args.apply),
                "total_files": scan.total_files,
                "total_bytes": scan.total_bytes,
                "protected_files": scan.protected_files,
                "protected_bytes": scan.protected_bytes,
                "planned_removed_files": planned_removed_files,
                "planned_removed_bytes": planned_removed_bytes,
                "removed_files": removed_files,
                "removed_bytes": removed_bytes,
                "failed_files": failed_files,
                "already_absent_files": already_absent_files,
                "changed_files": changed_files,
                "remaining_bytes": remaining_bytes,
                "budget_satisfied": (
                    None if budget is None else remaining_bytes <= budget
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0
