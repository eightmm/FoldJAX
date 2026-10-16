"""Request-local cache locations and no-follow private filesystem access."""

from __future__ import annotations

import errno
import os
import stat
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path


class CachePathError(OSError):
    """A cache path cannot be used without crossing its trust boundary."""


@dataclass(frozen=True)
class CacheLocations:
    requested_shared: Path | None
    shared: Path | None
    private: Path | None

    @property
    def write_directory(self) -> Path | None:
        for directory in (self.shared, self.private):
            if directory is not None and os.access(directory, os.W_OK | os.X_OK):
                return directory
        return None


_LOCATIONS: ContextVar[CacheLocations | None] = ContextVar(
    "foldjax_cache_locations", default=None
)


def current_cache_locations() -> CacheLocations | None:
    return _LOCATIONS.get()


@contextmanager
def cache_locations(locations: CacheLocations):
    token = _LOCATIONS.set(locations)
    try:
        yield locations
    finally:
        _LOCATIONS.reset(token)


@contextmanager
def open_directory(
    path: Path, *, private: bool, create: bool = False, mode: int = 0o777
):
    """Walk from a held root fd; private trust never inherits shared opt-in."""
    if not all(hasattr(os, flag) for flag in ("O_DIRECTORY", "O_NOFOLLOW")):
        raise CachePathError(errno.ENOTSUP, "safe directory access is unavailable")
    path = Path(os.path.abspath(path))
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    fd = os.open(path.anchor, flags)
    try:
        for index, component in enumerate(path.parts[1:]):
            try:
                child = os.open(component, flags, dir_fd=fd)
            except FileNotFoundError:
                if not create:
                    raise
                try:
                    os.mkdir(component, 0o700 if private else mode, dir_fd=fd)
                except FileExistsError:
                    pass
                child = os.open(component, flags, dir_fd=fd)
            os.close(fd)
            fd = child
            if private:
                info = os.fstat(fd)
                leaf = index == len(path.parts) - 2
                sticky_root = info.st_uid == 0 and info.st_mode & stat.S_ISVTX
                if info.st_uid not in (os.geteuid(), 0) or (
                    info.st_mode & 0o022 and not (sticky_root and not leaf)
                ):
                    raise CachePathError(errno.EACCES, "unsafe private cache ancestor")
        if private:
            info = os.fstat(fd)
            if info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) != 0o700:
                raise CachePathError(
                    errno.EACCES,
                    "private cache must be owned by this uid with mode 0700",
                )
        yield fd
    finally:
        os.close(fd)


def prepare_private_directory(path: Path) -> Path:
    path = Path(os.path.abspath(path))
    with open_directory(path, private=True, create=True):
        return path
