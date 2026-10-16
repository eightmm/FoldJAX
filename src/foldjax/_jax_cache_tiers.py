"""JAX 0.11.1 cache adapter; imported only inside a compilation scope."""

from __future__ import annotations

import errno
import json
import os
import stat
import time
import warnings
from collections.abc import Callable
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

from jax._src import compilation_cache as jax_cache
from jax._src import compiler, config
from jax._src.compilation_cache_interface import CacheInterface

from foldjax._cache_tiers import CacheLocations, CachePathError, open_directory

try:
    import fcntl
except ImportError:  # pragma: no cover - fd-based persistence requires POSIX
    fcntl = None

_RECORD: ContextVar[dict | None] = ContextVar(
    "foldjax_compilation_cache_record", default=None
)


def clear_record() -> None:
    _RECORD.set(None)


def recorded() -> dict | None:
    record = _RECORD.get()
    if record is None:
        return None
    return json.loads(json.dumps({k: v for k, v in record.items() if k != "_warnings"}))


def begin_record(locations: CacheLocations) -> None:
    if _RECORD.get() is None:
        _RECORD.set(
            {
                "mode": "layered",
                "locations": {
                    "shared": str(locations.shared) if locations.shared else None,
                    "private": str(locations.private) if locations.private else None,
                },
                "lookups": {},
                "loads": {},
                "writes": {},
                "compiles": {},
                "in_memory_hits": "unobserved",
                "_warnings": set(),
            }
        )


def note(stage: str, tier: str, event: str) -> None:
    record = _RECORD.get()
    if record is not None:
        counters = record[stage].setdefault(tier, {})
        counters[event] = counters.get(event, 0) + 1


def _warn(stage: str, tier: str, error: Exception) -> None:
    reason = errno.errorcode.get(getattr(error, "errno", None), type(error).__name__)
    key = (stage, tier, reason)
    record = _RECORD.get()
    if record is not None:
        if key in record["_warnings"]:
            return
        record["_warnings"].add(key)
    warnings.warn(
        f"{tier} compilation cache {stage} failed ({reason}); "
        "trying another cache tier or compiling without persistence",
        RuntimeWarning,
        stacklevel=3,
    )


def _name(key: str) -> str:
    if not key or "/" in key or "\\" in key or "\0" in key:
        raise CachePathError(errno.EINVAL, "unsafe compilation cache key")
    return f"{key}-cache"


class LayeredCache(CacheInterface):
    def __init__(
        self,
        locations: CacheLocations,
        *,
        max_size: int,
        shared_validator: Callable[[Path], str | None],
        shared_file_validator: Callable[[os.stat_result], bool],
        shared_mode: int = 0o644,
    ):
        self.locations = locations
        self.max_size = max_size
        self._path = locations.write_directory or locations.shared or locations.private
        self.shared_validator = shared_validator
        self.shared_file_validator = shared_file_validator
        self.shared_mode = shared_mode
        self.last_read: ContextVar[str | None] = ContextVar(
            "cache_last_read", default=None
        )
        self.read_override: ContextVar[str | None] = ContextVar(
            "cache_read_tier", default=None
        )
        self.bad: dict[str, set[str]] = {"shared": set(), "private": set()}

    @contextmanager
    def directory(self, tier: str):
        path = getattr(self.locations, tier)
        if path is None:
            raise FileNotFoundError(errno.ENOENT, "cache tier is unavailable")
        if tier == "shared" and self.shared_validator(path) is not None:
            raise CachePathError(errno.EACCES, "shared cache is no longer trusted")
        with open_directory(path, private=tier == "private") as fd:
            yield fd

    def validate_file(self, fd: int, tier: str) -> None:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise CachePathError(
                errno.EACCES, "cache entry is not a regular single-link file"
            )
        safe = (
            info.st_uid == os.geteuid() and not info.st_mode & 0o077
            if tier == "private"
            else self.shared_file_validator(info)
        )
        if not safe:
            raise CachePathError(errno.EACCES, "unsafe cache entry ownership or mode")

    @contextmanager
    def lock(self, directory_fd: int, name: str, tier: str):
        if fcntl is None:
            raise CachePathError(errno.ENOTSUP, "safe cache locking is unavailable")
        fd = os.open(
            name,
            os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC,
            0o600 if tier == "private" else self.shared_mode,
            dir_fd=directory_fd,
        )
        try:
            self.validate_file(fd, tier)
            deadline = time.monotonic() + 10
            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise CachePathError(
                            errno.ETIMEDOUT, "cache lock timed out"
                        ) from None
                    time.sleep(0.025)
            try:
                yield
            finally:
                fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)

    def get(self, key: str) -> bytes | None:
        self.last_read.set(None)
        name = _name(key)
        override = self.read_override.get()
        for tier in ("shared", "private"):
            if override is not None and tier != override:
                continue
            if getattr(self.locations, tier) is None or key in self.bad[tier]:
                continue
            try:
                with self.directory(tier) as directory_fd:
                    fd = os.open(
                        name,
                        os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
                        dir_fd=directory_fd,
                    )
                    try:
                        self.validate_file(fd, tier)
                        with os.fdopen(fd, "rb", closefd=False) as handle:
                            payload = handle.read()
                    finally:
                        os.close(fd)
                    if self.max_size > 0 and os.access(
                        getattr(self.locations, tier), os.W_OK | os.X_OK
                    ):
                        try:
                            now = time.time_ns()
                            os.utime(
                                name,
                                ns=(now, now),
                                dir_fd=directory_fd,
                                follow_symlinks=False,
                            )
                        except OSError:
                            pass
                note("lookups", tier, "hit")
                self.last_read.set(tier)
                return payload
            except FileNotFoundError:
                note("lookups", tier, "miss")
            except OSError as error:
                note("lookups", tier, "error")
                _warn("read", tier, error)
        return None

    def _evict(self, directory_fd: int, name: str, additional: int) -> None:
        if self.max_size < 0:
            return
        entries = []
        total = 0
        for candidate in os.listdir(directory_fd):
            if not candidate.endswith("-cache") or candidate == name:
                continue
            try:
                info = os.stat(candidate, dir_fd=directory_fd, follow_symlinks=False)
            except FileNotFoundError:
                continue
            if stat.S_ISREG(info.st_mode) and info.st_nlink == 1:
                entries.append((info.st_mtime_ns, candidate, info.st_size))
                total += info.st_size
        for _, candidate, size in sorted(entries):
            if total + additional <= self.max_size:
                break
            try:
                os.unlink(candidate, dir_fd=directory_fd)
            except FileNotFoundError:
                pass
            total -= size

    def _put(self, tier: str, key: str, value: bytes) -> str:
        name = _name(key)
        with (
            self.directory(tier) as directory_fd,
            self.lock(
                directory_fd,
                ".foldjax-cache.lock" if self.max_size >= 0 else f".{name}.lock",
                tier,
            ),
        ):
            try:
                existing = os.open(
                    name,
                    os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
                    dir_fd=directory_fd,
                )
            except FileNotFoundError:
                pass
            else:
                try:
                    self.validate_file(existing, tier)
                    if key not in self.bad[tier]:
                        return "exists"
                finally:
                    os.close(existing)
            self._evict(directory_fd, name, len(value))
            temporary = f".foldjax-{os.getpid()}-{time.time_ns()}.tmp"
            fd = os.open(
                temporary,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                0o600 if tier == "private" else self.shared_mode,
                dir_fd=directory_fd,
            )
            try:
                self.validate_file(fd, tier)
                with os.fdopen(fd, "wb", closefd=False) as handle:
                    handle.write(value)
                    handle.flush()
                os.fsync(fd)
                os.replace(
                    temporary, name, src_dir_fd=directory_fd, dst_dir_fd=directory_fd
                )
                os.fsync(directory_fd)
            finally:
                os.close(fd)
                try:
                    os.unlink(temporary, dir_fd=directory_fd)
                except FileNotFoundError:
                    pass
            self.bad[tier].discard(key)
            return "success"

    def put(self, key: str, value: bytes) -> None:
        if self.max_size >= 0 and len(value) > self.max_size:
            note("writes", "policy", "oversized")
            return
        for tier in ("shared", "private"):
            path = getattr(self.locations, tier)
            if path is None:
                continue
            if tier == "shared" and key in self.bad[tier] and self.locations.private:
                continue
            if not os.access(path, os.W_OK | os.X_OK):
                note("writes", tier, "unwritable")
                continue
            try:
                event = self._put(tier, key, value)
                note("writes", tier, event)
                return
            except OSError as error:
                note("writes", tier, "error")
                _warn("write", tier, error)


@contextmanager
def installed(
    locations: CacheLocations,
    *,
    shared_validator,
    shared_file_validator,
    shared_mode: int = 0o644,
):
    """Install and restore the pinned JAX factory and deserialization hooks."""
    original_factory = jax_cache.get_file_cache
    original_load = jax_cache.get_executable_and_time
    original_compile = compiler.backend_compile_and_load
    expected_path = config.compilation_cache_dir.value
    adapter = LayeredCache(
        locations,
        max_size=config.compilation_cache_max_size.value,
        shared_validator=shared_validator,
        shared_file_validator=shared_file_validator,
        shared_mode=shared_mode,
    )

    def factory(path):
        if expected_path is None or os.path.abspath(path) != os.path.abspath(
            expected_path
        ):
            return original_factory(path)
        result = (
            jax_cache.VerificationCache(adapter)
            if config.compilation_cache_check_contents.value
            else adapter
        )
        return result, path

    def load(cache_key, *args, **kwargs):
        adapter.last_read.set(None)
        try:
            result = original_load(cache_key, *args, **kwargs)
        except Exception as error:
            tier = adapter.last_read.get()
            if tier is not None:
                note("loads", tier, "error")
                adapter.bad[tier].add(cache_key)
            if (
                tier != "shared"
                or locations.private is None
                or config.raise_persistent_cache_errors.value
                or isinstance(error, jax_cache.CacheVerificationError)
            ):
                raise
            _warn("deserialize", "shared", error)
            token = adapter.read_override.set("private")
            try:
                result = original_load(cache_key, *args, **kwargs)
            except Exception:
                adapter.bad["private"].add(cache_key)
                note("loads", "private", "error")
                raise
            finally:
                adapter.read_override.reset(token)
        if result[0] is not None and (tier := adapter.last_read.get()) is not None:
            note("loads", tier, "success")
        return result

    def compile_and_load(*args, **kwargs):
        try:
            result = original_compile(*args, **kwargs)
        except Exception:
            record = _RECORD.get()
            if record is not None:
                counts = record["compiles"]
                counts["error"] = counts.get("error", 0) + 1
            raise
        record = _RECORD.get()
        if record is not None:
            counts = record["compiles"]
            counts["success"] = counts.get("success", 0) + 1
        return result

    compile_and_load._foldjax_cache_observer = True

    begin_record(locations)
    jax_cache.get_file_cache = factory
    jax_cache.get_executable_and_time = load
    if not getattr(original_compile, "_foldjax_cache_observer", False):
        compiler.backend_compile_and_load = compile_and_load
    try:
        yield
    finally:
        jax_cache.get_executable_and_time = original_load
        jax_cache.get_file_cache = original_factory
        compiler.backend_compile_and_load = original_compile
