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
from contextlib import ExitStack, contextmanager, nullcontext
from contextvars import ContextVar
from pathlib import Path
from typing import Any

from foldjax._gpu import compute_capability
from foldjax.cache import shared_compile_cache_trusted

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


class _ReadOnlyStoreUnavailableError(_CacheError):
    """A trusted read-only namespace has no Tokamax store yet."""


class _InvalidManifestError(_CacheError):
    """A regular cache file is incomplete or belongs to another identity."""


class _IncompleteCoverageError(_CacheError):
    """Explicit tuning did not produce one successful config for every op."""


class _IncompleteMeasurementsError(_IncompleteCoverageError):
    def __init__(self, message: str, *, partial: Any, missing: tuple[Any, ...]):
        super().__init__(message)
        self.partial = partial
        self.missing = missing


def _tokamax_api():
    import tokamax

    return tokamax


# ---------------------------------------------------------------------------
# Tokamax's HLO payload, and why the executed AlphaFold 3 program drops it.
#
# Every configurable Tokamax op wraps its body in
# ``set_xla_metadata(xla_metadata_payload="tokamax:<json>")`` so offline tools
# can find it in the HLO; the payload becomes ``OpMetadata.metadata_payload``.
# On the GPU, a persistent-cache entry of the AlphaFold 3 program cannot be read
# back: XLA's ``InlineMetadataPayloadsFromProtoPayloadTable`` refuses it with
# ``RET_CHECK ... Invalid metadata payload id 201 with payloads size 192`` and
# the warm process recompiles (jctc-v3 E9: 143 of 160 AlphaFold 3 warm
# processes -- all 70 of DeepMind's own runner, 73 of 90 FoldJAX ones; the
# bucket-padded FoldJAX arm read its entries back; no other port's compiled
# program). That check only reads instructions that carry a payload, so a
# program with none cannot fail it. The payload has no effect on the compiled
# code; the one reader this backend has is the discovery lowering below
# (``get_bound_args`` finds the ops by it), which keeps it.
#
# Protenix's eager guided sampler hit the same check from outside any compiled
# program: an eager `pallas_call` binds a fresh `jit` per call, so every tokamax
# denoiser attention persisted its own `jit__jit_run` entry and failed to read
# it back on the next call (4,147 times in one 60-minute TFG run, 2026-10-07).
# That path resolves its attention to `xla_jit` instead
# (`backends/protenix.py`, `_GUIDED_DIFFUSION_ATTENTION_BACKEND`) rather than
# dropping the payload here, because a readable entry still re-traces and
# re-lowers on every call.
#
# Discovery and execution lower the same jitted callable, and JAX shares one
# trace between ``lower`` and the call when the context at the jit boundary is
# the same -- so a cold process would otherwise execute (and cache) the trace
# discovery made, payloads included, under a key no warm process asks for.
# Discovery therefore lowers under an extra metadata entry, which is part of
# the trace context and keeps the two traces apart.
# ---------------------------------------------------------------------------

_HLO_PAYLOAD_SUPPRESSED: ContextVar[bool] = ContextVar(
    "foldjax_tokamax_hlo_payload_suppressed", default=False
)
_DISCOVERY_MARKER = "foldjax_tokamax_discovery"
_PAYLOAD_GATE_LOCK = threading.Lock()
_payload_gate_installed: bool | None = None


def _install_payload_gate() -> bool:
    """Route Tokamax's payload context through ``_HLO_PAYLOAD_SUPPRESSED``."""

    global _payload_gate_installed
    with _PAYLOAD_GATE_LOCK:
        if _payload_gate_installed is not None:
            return _payload_gate_installed
        try:
            from tokamax._src.ops import op as op_module
        except ImportError:
            _payload_gate_installed = False
            return False
        original = getattr(op_module, "_tokamax_metadata", None)
        active = getattr(op_module, "_ACTIVE_TOKAMAX_PAYLOAD", None)
        if not callable(original) or not isinstance(active, ContextVar):
            _LOGGER.warning(
                "this Tokamax has no recognised HLO payload hook; AlphaFold 3 "
                "persistent-cache entries keep the payload and may not load"
            )
            _payload_gate_installed = False
            return False

        @contextmanager
        def gated(json_data: str) -> Iterator[None]:
            if not _HLO_PAYLOAD_SUPPRESSED.get():
                with original(json_data):
                    yield
                return
            # Keep Tokamax's own nesting record; only the HLO attribute goes.
            previous = active.get()
            payload = (
                f"{previous}/tokamax:{json_data}"
                if previous
                else f"tokamax:{json_data}"
            )
            token = active.set(payload)
            try:
                yield
            finally:
                active.reset(token)

        gated.__wrapped__ = original  # type: ignore[attr-defined]
        op_module._tokamax_metadata = gated
        _payload_gate_installed = True
        return True


@contextmanager
def without_tokamax_hlo_payload() -> Iterator[None]:
    """Trace Tokamax ops without their HLO payload inside this block."""

    _install_payload_gate()
    token = _HLO_PAYLOAD_SUPPRESSED.set(True)
    try:
        yield
    finally:
        _HLO_PAYLOAD_SUPPRESSED.reset(token)


@contextmanager
def _discovery_lowering() -> Iterator[None]:
    """Lower with payloads, in a trace the executed call cannot reuse."""

    from jax.experimental import xla_metadata

    token = _HLO_PAYLOAD_SUPPRESSED.set(False)
    try:
        with xla_metadata.set_xla_metadata(**{_DISCOVERY_MARKER: "1"}):
            yield
    finally:
        _HLO_PAYLOAD_SUPPRESSED.reset(token)


# ---------------------------------------------------------------------------
# Tokamax's heuristic GLU tile on small-shared-memory GPUs.
#
# On a cache miss under `kernel_autotuning=heuristics`, Tokamax sizes its
# Pallas-Triton GLU tile for a data-centre GPU: block 128 x 64, block_k 32,
# four pipeline stages. On sm_120 (RTX PRO 6000 Blackwell, 99 KiB of shared
# memory per block) that tile asks for 108 KiB (`alphafold3._tokamax_kernel_
# fallback`) and the launch fails with RESOURCE_EXHAUSTED after compiling,
# which is why the backend's default is to autotune -- and autotuning is most
# of the cold compile. The pipeline holds up to `num_stages` copies of one x
# tile and two weight tiles; fewer stages hold fewer. Clamping the stage count
# (then the tile) until that bound fits leaves `block_k`, the K step every
# output accumulates over, where it was: only the schedule changes. That the
# bound is conservative enough, and the result unchanged, is what the GPU
# check in foldjax-bench/v011-perf-check measures; the default stays autotune.
# ---------------------------------------------------------------------------

#: Per-block opt-in limits, not per-SM totals. Ampere 8.0/8.7 have larger
#: budgets than consumer Ampere 8.6; a major-only lookup conflates them.
_SMALL_SHARED_MEMORY_BYTES = {
    (8, 6): 99 * 1024,
    (8, 9): 99 * 1024,
    (12, 0): 99 * 1024,
    (12, 1): 99 * 1024,
}
_GLU_HEURISTICS_LOCK = threading.RLock()
_GLU_PATCHES: dict[type, tuple[Any, int]] = {}
_GLU_BUDGET: ContextVar[int | None] = ContextVar("foldjax_glu_budget", default=None)
_GLU_DEVICE: ContextVar[Any | None] = ContextVar("foldjax_glu_device", default=None)


def _selected_glu_config(ba: Any, *, device: Any, config_cls: Any) -> Any:
    """Tokamax 0.0.13's tile rule, using the runner's GPU SM count."""
    x, weights = ba.args
    m = math.prod(x.shape[:-1])
    n = weights[0].shape[-1] if isinstance(weights, tuple) else weights.shape[2]
    rows, columns = (m, n) if n >= m else (n, m)
    block_m = min(max(32, 1 << (max(rows, 1) - 1).bit_length()), 128)
    block_n = min(max(32, 1 << (max(columns, 1) - 1).bit_length()), 256)
    block_n = min(block_n, (128 * 128) // block_m)
    while (
        math.ceil(rows / block_m) * math.ceil(columns / block_n) < device.core_count
        and block_m > 32
    ):
        block_m //= 2
    if n < m:
        block_m, block_n = block_n, block_m
    return config_cls(
        block_m=block_m,
        block_n=block_n // 2,
        block_k=32,
        num_warps=4,
        num_stages=4,
    )


def _glu_stage_bytes(config: Any, itemsize: int) -> int:
    x_tile = config.block_m * config.block_k
    weight_tiles = 2 * config.block_k * config.block_n
    return (x_tile + weight_tiles) * itemsize


def clamp_glu_config(config: Any, *, itemsize: int, budget_bytes: int) -> Any:
    """Fewest changes to ``config`` that fit ``num_stages`` tiles in the budget."""

    stages = config.num_stages
    while stages > 1 and stages * _glu_stage_bytes(config, itemsize) > budget_bytes:
        stages -= 1
    config = dataclasses.replace(config, num_stages=stages)
    while config.num_stages * _glu_stage_bytes(config, itemsize) > budget_bytes:
        if config.block_n > 32:
            config = dataclasses.replace(config, block_n=config.block_n // 2)
        elif config.block_m > 32:
            config = dataclasses.replace(config, block_m=config.block_m // 2)
        else:
            break
    return config


def _shared_memory_budget(device: Any) -> int | None:
    return _SMALL_SHARED_MEMORY_BYTES.get(compute_capability(device))


@contextmanager
def glu_heuristics_within_shared_memory(device: Any) -> Iterator[None]:
    """Size Tokamax's heuristic tile for the selected GPU and its memory."""

    token = _GLU_BUDGET.set(_shared_memory_budget(device))
    selected_gpu = getattr(device, "platform", None) in {"gpu", "cuda"}
    device_token = _GLU_DEVICE.set(device if selected_gpu else None)
    try:
        if not selected_gpu:
            yield
            return
        try:
            from tokamax._src.ops.gated_linear_unit import pallas_triton
        except ImportError:
            yield
            return
        glu_op = pallas_triton.PallasTritonGatedLinearUnit
        # Only installation/restoration is global. Device and memory limits
        # stay context-local while overlapping model calls share the wrapper.
        with _GLU_HEURISTICS_LOCK:
            original = glu_op.__dict__.get("_get_heuristics_config")
            if original is None:
                yield
                return
            if glu_op in _GLU_PATCHES:
                original, users = _GLU_PATCHES[glu_op]
            else:
                users = 0

                def clamped(self: Any, ba: Any) -> Any:
                    selected = _GLU_DEVICE.get()
                    config = (
                        original(self, ba)
                        if selected is None
                        else _selected_glu_config(
                            ba, device=selected, config_cls=pallas_triton.Config
                        )
                    )
                    budget = _GLU_BUDGET.get()
                    if budget is None:
                        return config
                    itemsize = int(getattr(ba.args[0].dtype, "itemsize", 4))
                    return clamp_glu_config(
                        config, itemsize=itemsize, budget_bytes=budget
                    )

                glu_op._get_heuristics_config = clamped
            _GLU_PATCHES[glu_op] = original, users + 1
        try:
            yield
        finally:
            with _GLU_HEURISTICS_LOCK:
                original, users = _GLU_PATCHES[glu_op]
                if users == 1:
                    glu_op._get_heuristics_config = original
                    del _GLU_PATCHES[glu_op]
                else:
                    _GLU_PATCHES[glu_op] = original, users - 1
    finally:
        _GLU_DEVICE.reset(device_token)
        _GLU_BUDGET.reset(token)


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
    """Whether parallel Tokamax workers use the intended device.

    Tokamax 0.0.13 initialises abstract benchmark arguments in worker threads,
    which do not inherit ``jax.default_device``'s thread-local context.  Only
    the process default platform's first local device is therefore safe for a
    parallel persisted measurement. Other devices use synchronous measurement
    in the caller's device context. Inline worker fallbacks remain unsafe there.
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


def _shared_store(fd: int) -> bool:
    return (
        shared_compile_cache_trusted()
        or _fstat(fd, subject="cache directory").st_uid == 0
    )


def _directory_writable(fd: int) -> bool:
    return os.access(".", os.W_OK | os.X_OK, dir_fd=fd)


def _validate_directory(fd: int, *, private: bool, shared: bool = False) -> None:
    info = _fstat(fd, subject="cache directory")
    if not stat.S_ISDIR(info.st_mode):
        raise _UnsafeCachePathError("cache path is not a directory")
    if (
        not shared_compile_cache_trusted()
        and hasattr(os, "geteuid")
        and info.st_uid not in (os.geteuid(), 0)
    ):
        raise _UnsafeCachePathError("cache directory is owned by another user")
    forbidden = (
        0
        if shared_compile_cache_trusted()
        else (0o077 if private and not shared else 0o022)
    )
    if stat.S_IMODE(info.st_mode) & forbidden:
        raise _UnsafeCachePathError("cache directory permissions are unsafe")


def _validate_regular_file(fd: int, *, shared: bool = False) -> os.stat_result:
    info = _fstat(fd, subject="cache entry")
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise _UnsafeCachePathError("cache entry is not a singly linked regular file")
    if (
        not shared_compile_cache_trusted()
        and hasattr(os, "geteuid")
        and info.st_uid not in (os.geteuid(), 0)
    ):
        raise _UnsafeCachePathError("cache entry is owned by another user")
    forbidden = 0 if shared_compile_cache_trusted() else (0o022 if shared else 0o077)
    if stat.S_IMODE(info.st_mode) & forbidden:
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
        #
        # A store shared on purpose (FOLDJAX_TRUST_SHARED_COMPILE_CACHE) keeps
        # its namespace as the group made it: stripping group write there
        # locked every other member out of the XLA entries beside this one.
        root_info = _fstat(root_fd, subject="compilation cache")
        shared = _shared_store(root_fd)
        if shared_compile_cache_trusted():
            if not stat.S_ISDIR(root_info.st_mode):
                raise _UnsafeCachePathError("cache path is not a directory")
        else:
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
            try:
                cache_fd = os.open(_CACHE_DIRECTORY, directory_flags, dir_fd=root_fd)
            except FileNotFoundError:
                if not _directory_writable(root_fd):
                    raise _ReadOnlyStoreUnavailableError(
                        "read-only namespace has no store"
                    )
                try:
                    os.mkdir(
                        _CACHE_DIRECTORY, 0o777 if shared else 0o700, dir_fd=root_fd
                    )
                except FileExistsError:
                    pass
                cache_fd = os.open(_CACHE_DIRECTORY, directory_flags, dir_fd=root_fd)
        except OSError as error:
            raise _UnsafeCachePathError(
                "cannot safely open the Tokamax cache"
            ) from error
        try:
            _validate_directory(cache_fd, private=True, shared=shared)
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
        info = _validate_regular_file(fd, shared=_shared_store(directory_fd))
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
        shared = _shared_store(directory_fd)
        fd = os.open(name, flags, 0o666 if shared else 0o600, dir_fd=directory_fd)
    except OSError as error:
        raise _UnsafeCachePathError(
            "cannot safely open the Tokamax cache lock"
        ) from error
    try:
        _validate_regular_file(fd, shared=shared)
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
        shared = _shared_store(directory_fd)
        fd = os.open(temporary, flags, 0o666 if shared else 0o600, dir_fd=directory_fd)
        _validate_regular_file(fd, shared=shared)
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


def _operation_label(bound_arg: Any) -> str:
    operation, key = _bound_arg_identity(bound_arg)
    return f"{operation} key={str(key)[:240]}"


def _candidate_failure_summary(data: Any) -> str:
    if data is None:
        return "no measurements returned"
    failures: dict[str, int] = {}
    for value in data.values():
        if isinstance(value, BaseException):
            message = (str(value).splitlines() or [""])[0][:180]
            detail = f"{type(value).__name__}: {message}"
            failures[detail] = failures.get(detail, 0) + 1
    return "; ".join(
        f"{count} x {detail}" for detail, count in list(failures.items())[:4]
    ) or "no successful benchmark measurements"


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
    missing = []
    reasons = []
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
            try:
                if data is None:
                    raise ValueError("no measurements returned")
                # Timeouts and compilation errors are failed candidates, not
                # failed operations when another candidate was measured.
                successful = (
                    data.prune_errors()
                    if callable(getattr(data, "prune_errors", None))
                    else {
                        k: v
                        for k, v in data.items()
                        if not isinstance(v, BaseException)
                    }
                )
                pruned = type(data)(successful).prune()
                if len(pruned) != 1:
                    raise ValueError("no successful config")
                pruned.fastest_config
            except Exception:  # noqa: BLE001 - failed operations cannot be persisted
                missing.append(bound_arg)
                reasons.append(
                    f"{_operation_label(bound_arg)}: {_candidate_failure_summary(data)}"
                )
                continue
            entries.append((bound_arg, pruned))
    if missing:
        raise _IncompleteMeasurementsError(
            f"{len(missing)} Tokamax operations have no successful config: "
            + " | ".join(reasons[:8]),
            partial=tokamax.AutotuningResult(device_kind, tuple(entries)),
            missing=tuple(missing),
        )
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


def _tune_selected_device(
    bound_args: Sequence[Any], *, device_kind: str, tokamax: Any
) -> Any:
    from tokamax._src.autotuning.autotuner import Autotuner, AutotuningData

    tuner = Autotuner(compile_executor_fn=None)
    # JAX's device context is thread-local. Keep compilation and measurement
    # here, and preserve completed operations when another operation raises.
    entries = []
    for bound_arg in bound_args:
        try:
            data = bound_arg.autotune(
                cache_results=False, autotuner=tuner, timeout=600.0
            )
        except Exception as error:  # noqa: BLE001 - retain per-operation evidence
            data = AutotuningData({"unmeasured": error})
        entries.append((bound_arg, data))
    return tokamax.AutotuningResult(device_kind, tuple(entries))


def _retry_missing_heuristics(
    bound_args: Sequence[Any], *, device_kind: str, tokamax: Any
) -> Any:
    """Measure one heuristic candidate per missing op in the caller's context."""
    from tokamax._src.autotuning.autotuner import Autotuner

    tuner = Autotuner(compile_executor_fn=None)
    entries = []
    for bound_arg in bound_args:
        try:
            config = bound_arg.heuristics_config
            data = bound_arg.autotune(
                configs={config}, cache_results=False, autotuner=tuner, timeout=60.0
            )
        except Exception as error:  # noqa: BLE001 - strict coverage still decides
            _LOGGER.warning(
                "Tokamax heuristic retry failed for %s: %s: %s",
                _operation_label(bound_arg),
                type(error).__name__,
                str(error)[:240],
            )
            continue
        entries.append((bound_arg, data))
    return tokamax.AutotuningResult(device_kind, tuple(entries))


def _discover_complete_result(
    lower: Callable[[], Any], *, device_kind: str, tokamax: Any, parallel: bool = True
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
    retried: set[tuple[str, Any]] = set()
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
                tuned = (
                    tokamax.autotune(missing, progress_bar=False)
                    if parallel
                    else _tune_selected_device(
                        missing, device_kind=device_kind, tokamax=tokamax
                    )
                )
        else:
            tuned = tokamax.AutotuningResult(device_kind, ())
        previous = result
        try:
            result = _complete_result(
                tuple(discovered.values()), tuned, device_kind, base=previous
            )
        except _IncompleteMeasurementsError as error:
            retry = tuple(
                ba for ba in error.missing if _bound_arg_identity(ba) not in retried
            )
            if not retry:
                raise
            retried.update(_bound_arg_identity(ba) for ba in retry)
            previous_results.append(error.partial)
            with error.partial:
                recovered = _retry_missing_heuristics(
                    retry, device_kind=device_kind, tokamax=tokamax
                )
            try:
                result = _complete_result(
                    tuple(discovered.values()),
                    recovered,
                    device_kind,
                    base=error.partial,
                )
            except _IncompleteMeasurementsError as retry_error:
                raise _IncompleteMeasurementsError(
                    f"{error}; single-heuristic retry remains incomplete: "
                    f"{retry_error}",
                    partial=retry_error.partial,
                    missing=retry_error.missing,
                ) from retry_error
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


# Where each model call's Tokamax configurations came from, for the run
# manifest. ``kernel_autotuning`` alone cannot say: it is Tokamax's cache-miss
# fallback and this store is a cache in front of it, so a ``heuristics`` run on
# a namespace an ``autotune`` run already filled reads the measured
# configurations and executes the autotuned program. The two values do compile
# different programs on a miss; the persistent compilation cache keys them by
# their HLO, so neither reuses the other's executable -- but which one ran
# depends on the store's history, and this is where that is written down. Read
# from a context variable rather than the store: `install_store` keeps an
# earlier prediction's store when the runner is reused. `foldjax.api` clears it
# before each prediction, as it does `memory_policy`'s record.
_KERNEL_RECORD: ContextVar[dict[str, Any] | None] = ContextVar(
    "foldjax_tokamax_kernel_record", default=None
)


def start_record(*, strategy: str, persistent_installed: bool) -> None:
    """Begin this prediction's record; ``sources`` stays null without a store,
    where nothing sees the calls and Tokamax resolves every one itself."""

    _KERNEL_RECORD.set(
        {
            "kernel_autotuning": str(strategy),
            "persistent_store": bool(persistent_installed),
            "sources": {} if persistent_installed else None,
        }
    )


def _note_source(source: str) -> None:
    record = _KERNEL_RECORD.get()
    if record is not None and record["sources"] is not None:
        record["sources"][source] = record["sources"].get(source, 0) + 1


def recorded() -> dict[str, Any] | None:
    """This prediction's record, or ``None`` when it ran no AlphaFold 3 model."""

    record = _KERNEL_RECORD.get()
    if record is None:
        return None
    sources = record["sources"]
    return {**record, "sources": dict(sources) if sources is not None else None}


def clear_record() -> None:
    """Forget the last record, so the next prediction reports only its own."""

    _KERNEL_RECORD.set(None)


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
        # Partial overlays aid inline fallback but are never complete or
        # persisted. Retaining their identity also stabilizes JAX's trace key.
        self._partial_results: dict[str, Any] = {}
        # Signatures whose result is on disk: read from it, or written to it.
        self._persisted: set[str] = set()
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

    def _invoke_with_result(
        self, result: Any, invoke: Callable[[], Any], *, source: str
    ) -> Any:
        tokamax = _tokamax_api()
        _note_source(source)
        with result, tokamax.config.autotuning_cache_miss_fallback("error"):
            return invoke()

    def _invoke_fallback(
        self, invoke: Callable[[], Any], *, partial: Any | None = None
    ) -> Any:
        if self._strategy == "autotune" and not self._can_tune:
            _raise_nondefault_autotune_miss()
        _note_source("partial" if partial is not None else "tokamax")
        with partial if partial is not None else nullcontext():
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
                return self._invoke_with_result(
                    result,
                    invoke,
                    source="store" if signature in self._persisted else "measured",
                )
            if (partial := self._partial_results.get(signature)) is not None:
                return self._invoke_fallback(invoke, partial=partial)

            store_unavailable = False
            try:
                with _open_cache_directory(self._cache_dir) as directory_fd:
                    writable = _directory_writable(directory_fd)
                    if not writable:
                        self._warn_once(
                            "cache is read-only; new results will not be saved"
                        )
                    result = self._load_result(directory_fd, signature, call_sha256)
            except _ReadOnlyStoreUnavailableError:
                self._warn_once("cache is read-only; new results will not be saved")
                store_unavailable = True
                result = None
            except _CacheError:
                self._warn_once("is unavailable on this filesystem")
                return self._invoke_fallback(invoke)
            if result is not None:
                self._results[signature] = result
                self._persisted.add(signature)
                return self._invoke_with_result(result, invoke, source="store")

            if self._strategy != "autotune":
                return self._invoke_fallback(invoke)

            tokamax = _tokamax_api()
            result = None
            coverage_error = None
            source = "store"
            try:
                with (
                    nullcontext(None)
                    if store_unavailable
                    else _open_cache_directory(self._cache_dir)
                ) as directory_fd:
                    writable = directory_fd is not None and _directory_writable(
                        directory_fd
                    )
                    with (
                        _exclusive_lock(directory_fd, signature)
                        if writable
                        else nullcontext()
                    ):
                        result = (
                            self._load_result(directory_fd, signature, call_sha256)
                            if directory_fd is not None
                            else None
                        )
                        if result is None:
                            source = "measured"
                            try:
                                result = _discover_complete_result(
                                    lower,
                                    device_kind=self._device_kind,
                                    tokamax=tokamax,
                                    parallel=self._can_tune,
                                )
                            except Exception as error:  # noqa: BLE001 - preserve fallback
                                self._warn_once(
                                    f"did not obtain complete coverage: {error}"
                                )
                                coverage_error = error
                                if (
                                    self._can_tune
                                    and isinstance(error, _IncompleteMeasurementsError)
                                    and error.partial.data
                                ):
                                    self._partial_results[signature] = error.partial
                                result = None
                            if result is not None and writable:
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
                                    self._persisted.add(signature)
                                except _CacheError:
                                    # The complete in-memory result is still safe
                                    # for this call even when persistence failed.
                                    self._warn_once("could not publish its result")
            except _CacheError:
                self._warn_once("could not lock its cache entry")
                return self._invoke_fallback(invoke)

            if result is None:
                if coverage_error is not None and not self._can_tune:
                    raise RuntimeError(
                        "Tokamax selected-device tuning could not obtain complete "
                        f"kernel coverage: {coverage_error}"
                    ) from coverage_error
                return self._invoke_fallback(
                    invoke, partial=self._partial_results.get(signature)
                )
            self._results[signature] = result
            if source == "store":
                self._persisted.add(signature)
            return self._invoke_with_result(result, invoke, source=source)


class _PersistentModelCall:
    def __init__(self, original: functools.partial, store: _PersistentTokamaxStore):
        self.original = original
        self.store = store

    def __call__(self, rng_key: Any, batch: Any) -> Any:
        original = self.original

        def lower() -> Any:
            with _discovery_lowering():
                return original.func.lower(
                    *original.args,
                    rng_key,
                    batch,
                    **(original.keywords or {}),
                )

        return self.store.call(
            rng_key=rng_key,
            batch=batch,
            lower=lower,
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


__all__ = [
    "clear_record",
    "create_store",
    "ensure_safe_autotuning_route",
    "install_store",
    "recorded",
    "start_record",
]
