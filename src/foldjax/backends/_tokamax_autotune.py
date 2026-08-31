"""Persistent, process-safe Tokamax autotuning results for AlphaFold 3.

Tokamax 0.0.13 exposes serialisable autotuning results, but keeps online
autotuning results in process memory.  This module adds the missing persistence
layer without reading Tokamax's private cache or changing the vendored model
runner.  Imports of JAX and Tokamax stay lazy so ordinary backend discovery
remains lightweight.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import importlib.metadata as metadata
import json
import logging
import math
import operator
import os
import stat
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any

try:
    import fcntl
except ImportError:  # pragma: no cover - AlphaFold 3 GPU support is Linux-only
    fcntl = None  # type: ignore[assignment]


_LOGGER = logging.getLogger(__name__)
_CACHE_DIRECTORY = ".tokamax-autotuning-v2"
_FORMAT = "foldjax-tokamax-autotuning"
_SCHEMA = 2
_MAX_RESULT_BYTES = 64 << 20
_MAX_DISCOVERY_ROUNDS = 4
_NONDEFAULT_AUTOTUNE_MESSAGE = (
    "Tokamax autotuning cache miss on a non-default GPU; warm this exact "
    "AlphaFold 3 profile on the process-default first local GPU, or explicitly "
    "select kernel_autotuning=heuristics"
)
_MANIFEST_KEYS = frozenset(
    {
        "format",
        "schema",
        "signature",
        "call_sha256",
        "identity",
        "coverage",
        "result_sha256",
        "result",
    }
)


class _CacheError(RuntimeError):
    """Persistence failed before model execution began."""


class _UnsafeCachePathError(_CacheError):
    """A cache path failed a no-follow, ownership, or mode check."""


class _InvalidManifestError(_CacheError):
    """A regular cache file is incomplete or belongs to another identity."""


class _IncompleteCoverageError(_CacheError):
    """Explicit tuning did not produce one successful config for every op."""


def _tokamax_api():
    import tokamax

    return tokamax


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _canonical_bytes(value: Any) -> bytes:
    return _canonical_json(value).encode("ascii")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _distribution_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "unavailable"


def _identity_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, (list, tuple)):
        return [_identity_value(item) for item in value]
    return str(value)


def _device_attribute(device: Any, name: str) -> Any:
    try:
        value = getattr(device, name)
        value = value() if callable(value) else value
    except (AttributeError, RuntimeError, TypeError):
        return "unavailable"
    return _identity_value(value)


def _platform_version(platform: str) -> str:
    try:
        from jax.extend import backend as jax_backend

        client = jax_backend.get_backend(platform)
        return str(getattr(client, "platform_version", "unavailable"))
    except (AttributeError, RuntimeError, ValueError):
        return "unavailable"


def _runtime_identity(
    *,
    device: Any,
    source_snapshot: Sequence[Any],
    config_identity: Sequence[Any],
    buckets: Sequence[int] | None,
    padding: bool,
) -> dict[str, Any]:
    import jax
    import jaxlib

    from foldjax import __version__ as foldjax_version

    tokamax = _tokamax_api()
    source_sha256 = _sha256(_canonical_bytes(source_snapshot))
    identity = {
        "model": "alphafold3",
        "foldjax": foldjax_version,
        "source_sha256": source_sha256,
        "config": config_identity,
        "buckets": buckets,
        "padding": bool(padding),
        "versions": {
            "tokamax": str(tokamax.__version__),
            "jax": str(jax.__version__),
            "jaxlib": str(jaxlib.__version__),
            "triton": _distribution_version("triton"),
            "jax-cuda12-plugin": _distribution_version("jax-cuda12-plugin"),
            "jax-cuda12-pjrt": _distribution_version("jax-cuda12-pjrt"),
            "jax-cuda13-plugin": _distribution_version("jax-cuda13-plugin"),
            "jax-cuda13-pjrt": _distribution_version("jax-cuda13-pjrt"),
        },
        "device": {
            "platform": _device_attribute(device, "platform"),
            "device_kind": _device_attribute(device, "device_kind"),
            "compute_capability": _device_attribute(device, "compute_capability"),
            "core_count": _device_attribute(device, "core_count"),
            "platform_version": _platform_version(str(device.platform)),
        },
    }
    # Round-trip once so tuple/list distinctions do not depend on whether the
    # identity came from the live process or a JSON manifest.
    return json.loads(_canonical_json(identity))


def _same_device(left: Any, right: Any) -> bool:
    names = ("platform", "id", "process_index", "local_hardware_id")
    return all(
        _device_attribute(left, name) == _device_attribute(right, name)
        for name in names
    )


def _can_tune_on_device(device: Any) -> bool:
    """Whether Tokamax's worker threads inherit the intended default device.

    Tokamax 0.0.13 initialises abstract benchmark arguments in worker threads,
    which do not inherit ``jax.default_device``'s thread-local context.  Only
    the process default platform's first local device is therefore safe for a
    new persisted measurement.  A result already measured on identical
    hardware can still be read and used on another ordinal.
    """

    import jax

    try:
        platform = str(device.platform)
        devices = jax.local_devices(backend=platform)
        return (
            bool(devices)
            and jax.default_backend() == platform
            and _same_device(device, devices[0])
        )
    except (AttributeError, RuntimeError, ValueError):
        return False


def _raise_nondefault_autotune_miss() -> None:
    raise RuntimeError(_NONDEFAULT_AUTOTUNE_MESSAGE)


def ensure_safe_autotuning_route(
    *, device: Any, strategy: str, persistent_installed: bool
) -> None:
    """Reject a process-local Tokamax worker on a non-default GPU.

    A successfully installed persistent store may still satisfy the call from a
    result measured on an identical first local GPU.  Every other route would
    expose Tokamax 0.0.13's worker threads to GPU 0 instead of the device bound
    into the model runner.
    """

    if (
        strategy == "autotune"
        and not persistent_installed
        and str(getattr(device, "platform", "")) == "gpu"
        and not _can_tune_on_device(device)
    ):
        _raise_nondefault_autotune_miss()


def _call_descriptor(rng_key: Any, batch: Any) -> dict[str, Any]:
    import jax

    leaves_with_paths, tree = jax.tree_util.tree_flatten_with_path((rng_key, batch))
    leaves = []
    for path, leaf in leaves_with_paths:
        try:
            shape = [operator.index(dimension) for dimension in leaf.shape]
            dtype = str(leaf.dtype)
        except (AttributeError, TypeError, ValueError) as error:
            raise _CacheError(
                "Tokamax persistence requires concrete array model inputs"
            ) from error
        leaves.append(
            {
                "path": jax.tree_util.keystr(path),
                "shape": shape,
                "dtype": dtype,
                "weak_type": bool(getattr(leaf, "weak_type", False)),
            }
        )
    return {"tree": str(tree), "leaves": leaves}


def _filesystem_supported() -> bool:
    return fcntl is not None and all(
        hasattr(os, name)
        for name in ("O_CLOEXEC", "O_DIRECTORY", "O_NOFOLLOW", "fchmod")
    )


def _fstat(fd: int, *, subject: str) -> os.stat_result:
    try:
        return os.fstat(fd)
    except OSError as error:
        raise _CacheError(f"cannot inspect {subject}") from error


def _close_fd(fd: int, *, subject: str) -> None:
    try:
        os.close(fd)
    except OSError as error:
        raise _CacheError(f"cannot close {subject}") from error


def _validate_directory(fd: int, *, private: bool) -> None:
    info = _fstat(fd, subject="cache directory")
    if not stat.S_ISDIR(info.st_mode):
        raise _UnsafeCachePathError("cache path is not a directory")
    if hasattr(os, "geteuid") and info.st_uid != os.geteuid():
        raise _UnsafeCachePathError("cache directory is owned by another user")
    forbidden = 0o077 if private else 0o022
    if stat.S_IMODE(info.st_mode) & forbidden:
        raise _UnsafeCachePathError("cache directory permissions are unsafe")


def _validate_regular_file(fd: int) -> os.stat_result:
    info = _fstat(fd, subject="cache entry")
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise _UnsafeCachePathError("cache entry is not a singly linked regular file")
    if hasattr(os, "geteuid") and info.st_uid != os.geteuid():
        raise _UnsafeCachePathError("cache entry is owned by another user")
    if stat.S_IMODE(info.st_mode) & 0o077:
        raise _UnsafeCachePathError("cache entry permissions are unsafe")
    return info


@contextmanager
def _open_cache_directory(cache_dir: Path) -> Iterator[int]:
    if not _filesystem_supported():
        raise _CacheError("safe file-descriptor cache operations are unavailable")
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    try:
        root_fd = os.open(cache_dir, directory_flags)
    except OSError as error:
        raise _UnsafeCachePathError(
            "cannot safely open the compilation cache"
        ) from error
    try:
        # Path.mkdir follows the caller's umask; a common collaborative umask
        # of 0002 makes the otherwise private per-profile namespace 0775.  Use
        # the already no-follow-opened descriptor to remove write access from
        # group/other before trusting it.  This is race-safe and also repairs
        # namespaces created by older FoldJAX processes without resolving the
        # path a second time.
        root_info = _fstat(root_fd, subject="compilation cache")
        if stat.S_ISDIR(root_info.st_mode) and (
            not hasattr(os, "geteuid") or root_info.st_uid == os.geteuid()
        ):
            root_mode = stat.S_IMODE(root_info.st_mode)
            if root_mode & 0o022:
                try:
                    os.fchmod(root_fd, root_mode & ~0o022)
                except OSError as error:
                    raise _UnsafeCachePathError(
                        "cannot secure the compilation cache permissions"
                    ) from error
        _validate_directory(root_fd, private=False)
        try:
            os.mkdir(_CACHE_DIRECTORY, 0o700, dir_fd=root_fd)
        except FileExistsError:
            pass
        except OSError as error:
            raise _UnsafeCachePathError(
                "cannot safely create the Tokamax cache"
            ) from error
        try:
            cache_fd = os.open(_CACHE_DIRECTORY, directory_flags, dir_fd=root_fd)
        except OSError as error:
            raise _UnsafeCachePathError(
                "cannot safely open the Tokamax cache"
            ) from error
        try:
            _validate_directory(cache_fd, private=True)
            yield cache_fd
        finally:
            _close_fd(cache_fd, subject="Tokamax cache directory")
    finally:
        _close_fd(root_fd, subject="compilation cache directory")


def _read_bytes(directory_fd: int, name: str) -> bytes | None:
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC
    try:
        fd = os.open(name, flags, dir_fd=directory_fd)
    except FileNotFoundError:
        return None
    except OSError as error:
        raise _UnsafeCachePathError(
            "cannot safely open a Tokamax cache entry"
        ) from error
    try:
        info = _validate_regular_file(fd)
        if not 0 < info.st_size <= _MAX_RESULT_BYTES:
            raise _InvalidManifestError("Tokamax cache entry has an invalid size")
        chunks = []
        remaining = info.st_size
        while remaining:
            try:
                chunk = os.read(fd, min(remaining, 1 << 20))
            except InterruptedError:
                continue
            except OSError as error:
                raise _CacheError("cannot read a Tokamax cache entry") from error
            if not chunk:
                raise _InvalidManifestError("Tokamax cache entry was truncated")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)
    finally:
        _close_fd(fd, subject="Tokamax cache entry")


@contextmanager
def _exclusive_lock(directory_fd: int, signature: str) -> Iterator[None]:
    if fcntl is None:
        raise _CacheError("process-safe Tokamax cache locking is unavailable")
    name = f"{signature}.lockfile"
    flags = os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC
    try:
        fd = os.open(name, flags, 0o600, dir_fd=directory_fd)
    except OSError as error:
        raise _UnsafeCachePathError(
            "cannot safely open the Tokamax cache lock"
        ) from error
    try:
        _validate_regular_file(fd)
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX)
                break
            except InterruptedError:
                continue
            except OSError as error:
                raise _CacheError("cannot lock a Tokamax cache entry") from error
        try:
            yield
        finally:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError as error:
                raise _CacheError("cannot unlock a Tokamax cache entry") from error
    finally:
        _close_fd(fd, subject="Tokamax cache lock")


def _atomic_write(directory_fd: int, name: str, payload: bytes) -> None:
    if not 0 < len(payload) <= _MAX_RESULT_BYTES:
        raise _InvalidManifestError("Tokamax cache entry has an invalid size")
    temporary = f".{name}.{os.getpid()}.{time.time_ns()}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC
    fd: int | None = None
    replaced = False
    try:
        fd = os.open(temporary, flags, 0o600, dir_fd=directory_fd)
        _validate_regular_file(fd)
        view = memoryview(payload)
        while view:
            try:
                written = os.write(fd, view)
            except InterruptedError:
                continue
            if written <= 0:
                raise _CacheError("Tokamax cache write made no progress")
            view = view[written:]
        os.fsync(fd)
        closing_fd = fd
        fd = None
        _close_fd(closing_fd, subject="Tokamax cache temporary")
        os.replace(
            temporary,
            name,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
        )
        replaced = True
        os.fsync(directory_fd)
    except OSError as error:
        raise _CacheError("cannot atomically publish the Tokamax cache") from error
    finally:
        if fd is not None:
            _close_fd(fd, subject="Tokamax cache temporary")
        if not replaced:
            try:
                os.unlink(temporary, dir_fd=directory_fd)
            except OSError:
                pass


def _validate_complete_result(
    result: Any,
    *,
    expected_count: int,
    device_kind: str,
    tokamax: Any,
) -> None:
    if str(result.tokamax_version) != str(tokamax.__version__):
        raise _InvalidManifestError("Tokamax cache version does not match the runtime")
    if str(result.device_kind) != device_kind:
        raise _InvalidManifestError("Tokamax cache device kind does not match")
    if expected_count <= 0 or len(result.data) != expected_count:
        raise _InvalidManifestError("Tokamax cache coverage is incomplete")
    identities: set[tuple[str, Any]] = set()
    for bound_arg, data in result.data:
        try:
            identity = _bound_arg_identity(bound_arg)
        except _IncompleteCoverageError as error:
            raise _InvalidManifestError(
                "Tokamax cache entry has no stable operation identity"
            ) from error
        if identity in identities:
            raise _InvalidManifestError(
                "Tokamax cache contains a duplicate operation entry"
            )
        identities.add(identity)
        if len(data) != 1:
            raise _InvalidManifestError("Tokamax cache entry is not pruned")
        try:
            data.fastest_config
        except Exception as error:  # noqa: BLE001 - invalid derived cache data
            raise _InvalidManifestError(
                "Tokamax cache entry has no successful config"
            ) from error


def _decode_manifest(
    payload: bytes,
    *,
    signature: str,
    call_sha256: str,
    identity: Mapping[str, Any],
    device_kind: str,
) -> Any:
    try:
        manifest = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise _InvalidManifestError("Tokamax cache entry is not valid JSON") from error
    if not isinstance(manifest, dict) or set(manifest) != _MANIFEST_KEYS:
        raise _InvalidManifestError("Tokamax cache manifest fields do not match")
    if manifest["format"] != _FORMAT or manifest["schema"] != _SCHEMA:
        raise _InvalidManifestError("Tokamax cache manifest version does not match")
    if manifest["signature"] != signature or manifest["call_sha256"] != call_sha256:
        raise _InvalidManifestError("Tokamax cache signature does not match")
    try:
        if _canonical_bytes(manifest["identity"]) != _canonical_bytes(identity):
            raise _InvalidManifestError("Tokamax cache identity does not match")
        result_payload = _canonical_bytes(manifest["result"])
    except (TypeError, ValueError) as error:
        raise _InvalidManifestError(
            "Tokamax cache manifest is not canonical JSON"
        ) from error
    if manifest["result_sha256"] != _sha256(result_payload):
        raise _InvalidManifestError("Tokamax cache result checksum does not match")
    coverage = manifest["coverage"]
    if not isinstance(coverage, dict) or set(coverage) != {
        "bound_arguments",
        "entries",
        "configs_per_entry",
    }:
        raise _InvalidManifestError("Tokamax cache coverage record does not match")
    values = tuple(coverage.values())
    if any(type(value) is not int for value in values):
        raise _InvalidManifestError("Tokamax cache coverage values are not integers")
    if (
        coverage["bound_arguments"] <= 0
        or coverage["entries"] != coverage["bound_arguments"]
        or coverage["configs_per_entry"] != 1
    ):
        raise _InvalidManifestError("Tokamax cache coverage is incomplete")
    tokamax = _tokamax_api()
    try:
        result = tokamax.AutotuningResult.loads(result_payload.decode("ascii"))
    except Exception as error:  # noqa: BLE001 - invalid derived cache data
        raise _InvalidManifestError("Tokamax cache result cannot be loaded") from error
    _validate_complete_result(
        result,
        expected_count=coverage["entries"],
        device_kind=device_kind,
        tokamax=tokamax,
    )
    return result


def _manifest_bytes(
    result: Any,
    *,
    signature: str,
    call_sha256: str,
    identity: Mapping[str, Any],
    expected_count: int,
    device_kind: str,
) -> bytes:
    tokamax = _tokamax_api()
    _validate_complete_result(
        result,
        expected_count=expected_count,
        device_kind=device_kind,
        tokamax=tokamax,
    )
    try:
        result_value = json.loads(result.dumps(prune_errors=True))
        result_payload = _canonical_bytes(result_value)
        payload = _canonical_bytes(
            {
                "format": _FORMAT,
                "schema": _SCHEMA,
                "signature": signature,
                "call_sha256": call_sha256,
                "identity": identity,
                "coverage": {
                    "bound_arguments": expected_count,
                    "entries": len(result.data),
                    "configs_per_entry": 1,
                },
                "result_sha256": _sha256(result_payload),
                "result": result_value,
            }
        )
    except Exception as error:  # noqa: BLE001 - persistence must fail closed
        raise _InvalidManifestError("Tokamax result is not canonical JSON") from error
    if len(payload) > _MAX_RESULT_BYTES:
        raise _InvalidManifestError("Tokamax cache manifest is too large")
    return payload


def _complete_result(
    bound_args: Sequence[Any],
    tuned: Any,
    device_kind: str,
    *,
    base: Any | None = None,
) -> Any:
    tokamax = _tokamax_api()
    if str(tuned.device_kind) != device_kind:
        raise _IncompleteCoverageError("Tokamax tuned the wrong device kind")
    entries = []
    with ExitStack() as stack:
        if base is not None:
            if str(base.device_kind) != device_kind:
                raise _IncompleteCoverageError(
                    "Tokamax base result has the wrong device kind"
                )
            stack.enter_context(base)
        stack.enter_context(tuned)
        for bound_arg in bound_args:
            data = bound_arg.cached_autotuning_data
            if data is None:
                raise _IncompleteCoverageError("Tokamax omitted an operation")
            try:
                pruned = data.prune()
                if len(pruned) != 1:
                    raise _IncompleteCoverageError(
                        "Tokamax produced no successful config"
                    )
                pruned.fastest_config
            except _IncompleteCoverageError:
                raise
            except Exception as error:  # noqa: BLE001 - coverage is fail-closed
                raise _IncompleteCoverageError(
                    "Tokamax produced no successful config"
                ) from error
            entries.append((bound_arg, pruned))
    result = tokamax.AutotuningResult(device_kind, tuple(entries))
    _validate_complete_result(
        result,
        expected_count=len(bound_args),
        device_kind=device_kind,
        tokamax=tokamax,
    )
    return result


def _bound_arg_identity(bound_arg: Any) -> tuple[str, Any]:
    """Tokamax's own uniqueness key, retaining the implementation class."""

    try:
        operation = bound_arg.op
        key = bound_arg.autotuning_cache_key
        hash(key)
    except Exception as error:  # noqa: BLE001 - opaque specs cannot be persisted
        raise _IncompleteCoverageError(
            "Tokamax produced an unidentifiable operation"
        ) from error
    operation_type = type(operation)
    return f"{operation_type.__module__}.{operation_type.__qualname__}", key


def _pre_vmap_variants(bound_arg: Any) -> tuple[Any, ...]:
    """Reconstruct lookups made before Tokamax's custom-vmap rules run.

    ``get_bound_args(lowered)`` sees only the post-vmap op specs retained in
    StableHLO. Tokamax's public ``capture_batched_args`` wrapper asks for a
    config before every custom-vmap layer. Because each layer prepends one axis
    (including ``None`` for broadcast leaves), those hidden lookups are exactly
    the coherent suffixes of the retained ``vmap_axes`` tuples.
    """

    arguments = getattr(bound_arg, "arguments", None)
    if not isinstance(arguments, Mapping) or not dataclasses.is_dataclass(bound_arg):
        return ()
    import jax

    def is_shape(value: Any) -> bool:
        return isinstance(value, jax.ShapeDtypeStruct)

    leaves = jax.tree_util.tree_leaves(dict(arguments), is_leaf=is_shape)
    batched = [
        value for value in leaves if is_shape(value) and hasattr(value, "vmap_axes")
    ]
    if not batched:
        return ()
    depths = {len(value.vmap_axes) for value in batched}
    if depths == {0}:
        return ()
    if len(depths) != 1 or 0 in depths:
        raise _IncompleteCoverageError(
            "Tokamax produced incoherent nested vmap operation metadata"
        )
    (depth,) = tuple(depths)
    variants = []
    for removed in range(1, depth + 1):

        def remove_outer_axes(value: Any) -> Any:
            if not (is_shape(value) and hasattr(value, "vmap_axes")):
                return value
            remaining = tuple(value.vmap_axes[removed:])
            if not remaining:
                return jax.ShapeDtypeStruct(value.shape, value.dtype)
            try:
                return type(value)(value.shape, value.dtype, vmap_axes=remaining)
            except (TypeError, ValueError) as error:
                raise _IncompleteCoverageError(
                    "Tokamax's nested pre-vmap operation could not be reconstructed"
                ) from error

        replaced = jax.tree_util.tree_map(
            remove_outer_axes,
            dict(arguments),
            is_leaf=is_shape,
        )
        try:
            variants.append(dataclasses.replace(bound_arg, arguments=replaced))
        except (TypeError, ValueError) as error:
            raise _IncompleteCoverageError(
                "Tokamax's pre-vmap operation could not be reconstructed"
            ) from error
    return tuple(variants)


def _include_pre_vmap_variants(bound_args: Sequence[Any]) -> tuple[Any, ...]:
    unique: dict[tuple[str, Any], Any] = {}
    for bound_arg in bound_args:
        unique.setdefault(_bound_arg_identity(bound_arg), bound_arg)
        for variant in _pre_vmap_variants(bound_arg):
            unique.setdefault(_bound_arg_identity(variant), variant)
    return tuple(unique.values())


def _capture_bound_args(
    lower: Callable[[], Any],
    *,
    result: Any | None,
    fallback: str,
    tokamax: Any,
) -> tuple[Any, ...]:
    """Lower once under an optional overlay and return Tokamax op specs."""

    with ExitStack() as stack:
        if result is not None:
            stack.enter_context(result)
        stack.enter_context(tokamax.config.autotuning_cache_miss_fallback(fallback))
        lowered = lower()
        return tuple(tokamax.autotuning.get_bound_args(lowered))


def _missing_bound_args(
    bound_args: Sequence[Any], result: Any | None
) -> tuple[Any, ...]:
    with ExitStack() as stack:
        if result is not None:
            stack.enter_context(result)
        return tuple(
            bound_arg
            for bound_arg in bound_args
            if bound_arg.cached_autotuning_data is None
        )


def _is_missing_config_error(error: BaseException) -> bool:
    """Recognise Tokamax's public error-policy miss without parsing a traceback."""

    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(
            current, ValueError
        ) and "No config found for BoundArguments(" in str(current):
            return True
        current = current.__cause__ or current.__context__
    return False


def _discover_complete_result(
    lower: Callable[[], Any], *, device_kind: str, tokamax: Any
) -> Any | None:
    """Tune to a fixed point and prove the exact error-policy lowering succeeds.

    The operation set exposed by a heuristics lowering can differ from the set
    exposed after its first configs are overlaid.  Persisting that first set is
    therefore insufficient.  Discovery is monotonic and bounded: each round
    retains every observed operation, tunes only uncovered keys, and then
    lowers the same call with the strict error policy.  Only that strict
    lowering is accepted as complete.
    """

    discovered: dict[tuple[str, Any], Any] = {}
    previous_results: list[Any] = []
    result = None
    for _round in range(_MAX_DISCOVERY_ROUNDS):
        current = _include_pre_vmap_variants(
            _capture_bound_args(
                lower,
                result=result,
                fallback="heuristics",
                tokamax=tokamax,
            )
        )
        if not current:
            return None
        for bound_arg in current:
            discovered.setdefault(_bound_arg_identity(bound_arg), bound_arg)

        missing = _missing_bound_args(current, result)
        if missing:
            with ExitStack() as stack:
                if result is not None:
                    stack.enter_context(result)
                tuned = tokamax.autotune(missing, progress_bar=False)
        else:
            tuned = tokamax.AutotuningResult(device_kind, ())
        previous = result
        result = _complete_result(
            tuple(discovered.values()),
            tuned,
            device_kind,
            base=previous,
        )
        if previous is not None:
            # Tokamax includes ``id(result)`` in JAX's tracing context.  Keep
            # every superseded overlay alive until discovery finishes so a
            # later round cannot accidentally reuse an earlier object ID.
            previous_results.append(previous)

        try:
            verified = _capture_bound_args(
                lower,
                result=result,
                fallback="error",
                tokamax=tokamax,
            )
        except Exception as error:  # noqa: BLE001 - only a cache miss is iterative
            if _is_missing_config_error(error):
                continue
            raise _IncompleteCoverageError(
                "strict Tokamax coverage lowering failed"
            ) from error

        discovered_before = len(discovered)
        for bound_arg in verified:
            discovered.setdefault(_bound_arg_identity(bound_arg), bound_arg)
        if len(discovered) != discovered_before:
            continue
        if _missing_bound_args(verified, result):
            continue
        return result

    raise _IncompleteCoverageError(
        "Tokamax operation discovery did not reach complete strict coverage"
    )


class _PersistentTokamaxStore:
    def __init__(
        self,
        cache_dir: Path,
        *,
        identity: Mapping[str, Any],
        strategy: str,
        device_kind: str,
        can_tune: bool,
    ) -> None:
        self._cache_dir = Path(cache_dir)
        self._identity = json.loads(_canonical_json(identity))
        self._strategy = strategy
        self._device_kind = device_kind
        self._can_tune = can_tune
        self._results: dict[str, Any] = {}
        self._warned: set[str] = set()
        self._thread_lock = threading.RLock()
        self.compatibility_key = (
            str(self._cache_dir.absolute()),
            _canonical_json(self._identity),
            strategy,
        )

    def _warn_once(self, reason: str) -> None:
        if reason in self._warned:
            return
        self._warned.add(reason)
        _LOGGER.warning(
            "persistent Tokamax autotuning %s; using the %s cache-miss policy",
            reason,
            self._strategy,
        )

    def _load_result(
        self, directory_fd: int, signature: str, call_sha256: str
    ) -> Any | None:
        try:
            payload = _read_bytes(directory_fd, f"{signature}.json")
            if payload is None:
                return None
            return _decode_manifest(
                payload,
                signature=signature,
                call_sha256=call_sha256,
                identity=self._identity,
                device_kind=self._device_kind,
            )
        except _InvalidManifestError:
            self._warn_once("ignored an invalid cache entry")
            return None

    def _invoke_with_result(self, result: Any, invoke: Callable[[], Any]) -> Any:
        tokamax = _tokamax_api()
        with result, tokamax.config.autotuning_cache_miss_fallback("error"):
            return invoke()

    def _invoke_fallback(self, invoke: Callable[[], Any]) -> Any:
        if self._strategy == "autotune" and not self._can_tune:
            _raise_nondefault_autotune_miss()
        return invoke()

    def call(
        self,
        *,
        rng_key: Any,
        batch: Any,
        lower: Callable[[], Any],
        invoke: Callable[[], Any],
    ) -> Any:
        """Run one model call under a complete persisted Tokamax overlay."""

        with self._thread_lock:
            try:
                descriptor = _call_descriptor(rng_key, batch)
                call_sha256 = _sha256(_canonical_bytes(descriptor))
                signature = _sha256(
                    _canonical_bytes({"identity": self._identity, "call": descriptor})
                )
            except (TypeError, ValueError, _CacheError):
                self._warn_once("could not identify this model call")
                return self._invoke_fallback(invoke)

            if (result := self._results.get(signature)) is not None:
                return self._invoke_with_result(result, invoke)

            try:
                with _open_cache_directory(self._cache_dir) as directory_fd:
                    result = self._load_result(directory_fd, signature, call_sha256)
            except _CacheError:
                self._warn_once("is unavailable on this filesystem")
                return self._invoke_fallback(invoke)
            if result is not None:
                self._results[signature] = result
                return self._invoke_with_result(result, invoke)

            if self._strategy == "autotune" and not self._can_tune:
                # Tokamax 0.0.13 compiles candidates in worker threads. JAX's
                # selected/default-device scope is thread-local, so allowing its
                # ordinary inline fallback here would silently benchmark GPU 0
                # even when this model is bound to another (possibly different)
                # GPU. A compatible result warmed on the first local device is
                # safe to read above; an uncached measurement is not.
                _raise_nondefault_autotune_miss()

            if self._strategy != "autotune":
                return self._invoke_fallback(invoke)

            tokamax = _tokamax_api()
            result = None
            try:
                with _open_cache_directory(self._cache_dir) as directory_fd:
                    with _exclusive_lock(directory_fd, signature):
                        result = self._load_result(directory_fd, signature, call_sha256)
                        if result is None:
                            try:
                                result = _discover_complete_result(
                                    lower,
                                    device_kind=self._device_kind,
                                    tokamax=tokamax,
                                )
                            except Exception:  # noqa: BLE001 - preserve inline fallback
                                self._warn_once("did not obtain complete coverage")
                                result = None
                            if result is not None:
                                try:
                                    payload = _manifest_bytes(
                                        result,
                                        signature=signature,
                                        call_sha256=call_sha256,
                                        identity=self._identity,
                                        expected_count=len(result.data),
                                        device_kind=self._device_kind,
                                    )
                                    _atomic_write(
                                        directory_fd,
                                        f"{signature}.json",
                                        payload,
                                    )
                                except _CacheError:
                                    # The complete in-memory result is still safe
                                    # for this call even when persistence failed.
                                    self._warn_once("could not publish its result")
            except _CacheError:
                self._warn_once("could not lock its cache entry")
                return self._invoke_fallback(invoke)

            if result is None:
                return self._invoke_fallback(invoke)
            self._results[signature] = result
            return self._invoke_with_result(result, invoke)


class _PersistentModelCall:
    def __init__(self, original: functools.partial, store: _PersistentTokamaxStore):
        self.original = original
        self.store = store

    def __call__(self, rng_key: Any, batch: Any) -> Any:
        original = self.original
        return self.store.call(
            rng_key=rng_key,
            batch=batch,
            lower=lambda: original.func.lower(
                *original.args,
                rng_key,
                batch,
                **(original.keywords or {}),
            ),
            invoke=lambda: original(rng_key, batch),
        )


def create_store(
    *,
    managed: bool,
    cache_dir: Path | None,
    device: Any,
    source_snapshot: Sequence[Any] | None,
    config_identity: Sequence[Any],
    buckets: Sequence[int] | None,
    padding: bool,
    strategy: str,
) -> _PersistentTokamaxStore | None:
    """Create a store only for a verifiable managed GPU cache namespace."""

    if (
        not managed
        or cache_dir is None
        or source_snapshot is None
        or str(getattr(device, "platform", "")) != "gpu"
        or not _filesystem_supported()
    ):
        return None
    try:
        identity = _runtime_identity(
            device=device,
            source_snapshot=source_snapshot,
            config_identity=config_identity,
            buckets=buckets,
            padding=padding,
        )
    except Exception:  # noqa: BLE001 - persistence must not break inference
        _LOGGER.warning(
            "persistent Tokamax autotuning identity is unavailable; using the %s "
            "cache-miss policy",
            strategy,
        )
        return None
    return _PersistentTokamaxStore(
        Path(cache_dir),
        identity=identity,
        strategy=strategy,
        device_kind=str(device.device_kind),
        can_tune=_can_tune_on_device(device),
    )


def install_store(model_runner: Any, store: _PersistentTokamaxStore) -> bool:
    """Wrap the vendored runner's lowerable cached model callable."""

    try:
        current = model_runner._model  # noqa: SLF001 - checked vendored ABI
    except AttributeError:
        _LOGGER.warning(
            "AlphaFold 3 runner has no lowerable _model; using process-local "
            "Tokamax autotuning"
        )
        return False
    if isinstance(current, _PersistentModelCall):
        if current.store.compatibility_key == store.compatibility_key:
            return True
        current = current.original
    if not isinstance(current, functools.partial) or not callable(
        getattr(current.func, "lower", None)
    ):
        _LOGGER.warning(
            "AlphaFold 3 runner _model is not a lowerable partial; using "
            "process-local Tokamax autotuning"
        )
        return False
    try:
        model_runner._model = _PersistentModelCall(current, store)  # noqa: SLF001
    except (AttributeError, TypeError):
        _LOGGER.warning(
            "AlphaFold 3 runner _model cannot be wrapped; using process-local "
            "Tokamax autotuning"
        )
        return False
    return True


__all__ = ["create_store", "ensure_safe_autotuning_route", "install_store"]
