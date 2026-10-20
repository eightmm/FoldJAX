"""Development-only observation of the pinned JAX compiler and cache loader.

Imports remain lazy: callers enter this context only after native CLI runtime
configuration. Timings describe inclusive compiler/load calls; a cache-load
residual includes deserialization, CUDA module loading and adapter bookkeeping,
and must not be presented as a measured CUDA-only interval.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from functools import wraps
from pathlib import Path
from typing import Any
from unittest.mock import patch


def write_report(path: Path, report: dict[str, Any]) -> None:
    """Retain a prediction failure if the diagnostic filesystem also fails."""
    pending = sys.exception()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    except OSError as error:
        if pending is None and report.get("exit_code", 0) == 0:
            raise
        try:
            print(
                f"diagnostic report could not be saved: {type(error).__name__}: "
                f"{error}; retaining the original prediction failure",
                file=sys.stderr,
            )
        except OSError:
            pass


def device_memory(request=None) -> dict[str, int]:
    from foldjax._gpu import default_device

    stats = default_device(request).memory_stats() or {}
    return {
        name: int(stats[name])
        for name in ("bytes_in_use", "peak_bytes_in_use", "bytes_limit")
        if name in stats
    }


def executable_memory(executable: Any) -> dict[str, int] | None:
    method = getattr(executable, "get_compiled_memory_stats", None)
    if method is None:
        return None
    try:
        stats = method()
    except (NotImplementedError, RuntimeError):
        return None
    return {
        name: int(getattr(stats, name))
        for name in (
            "argument_size_in_bytes",
            "output_size_in_bytes",
            "alias_size_in_bytes",
            "temp_size_in_bytes",
            "generated_code_size_in_bytes",
        )
        if hasattr(stats, name)
    }


def _digest(value: str | bytes) -> str:
    return hashlib.sha256(
        value.encode() if isinstance(value, str) else value
    ).hexdigest()


def _input_fingerprints(computation, args, kwargs):
    fingerprints = {}
    try:
        fingerprints["pre_compile_mlir_sha256"] = _digest(
            computation.operation.get_asm(enable_debug_info=False)
        )
        options = kwargs.get("compile_options", args[1] if len(args) > 1 else None)
        fingerprints["compile_options_sha256"] = _digest(options.SerializeAsString())
        fingerprints["compile_options_scope"] = (
            "raw JAX CompileOptions, including debug/dump settings"
        )
    except Exception as error:
        fingerprints["input_unavailable"] = type(error).__name__
    return fingerprints


def _optimized_fingerprints(executable):
    from jaxlib import _hlo

    options = _hlo.HloPrintOptions.canonical()
    options.print_metadata = False
    options.print_backend_config = True
    options.print_large_constants = True
    options.include_layout_in_shapes = True
    return [_digest(module.to_string(options)) for module in executable.hlo_modules()]


def _memory_observation():
    try:
        return device_memory()
    except Exception as error:
        return {"unavailable": type(error).__name__}


@contextmanager
def observe_compilation(report: dict[str, Any], *, fingerprint_hlo: bool = False):
    """Observe cache reads and compilation without changing keys or backends."""
    from jax._src import compilation_cache, compiler

    from foldjax._jax_cache_tiers import LayeredCache

    executables = report.setdefault("executables", [])
    current: ContextVar[dict | None] = ContextVar("profile_executable", default=None)
    reading: ContextVar[dict | None] = ContextVar("profile_cache_read", default=None)
    original_compile = compiler.compile_or_get_cached
    original_read = compiler._cache_read
    original_get = LayeredCache.get
    original_decompress = compilation_cache.decompress_executable
    original_backend_compile = compiler.backend_compile_and_load

    @wraps(original_get)
    def get(adapter, key):
        started = time.perf_counter()
        try:
            return original_get(adapter, key)
        finally:
            if (entry := reading.get()) is not None:
                entry["layered_get_calls"] += 1
                entry["layered_get_seconds"] += time.perf_counter() - started

    @wraps(original_decompress)
    def decompress(*args, **kwargs):
        started = time.perf_counter()
        try:
            return original_decompress(*args, **kwargs)
        finally:
            if (entry := reading.get()) is not None:
                entry["decompress_seconds"] += time.perf_counter() - started

    @wraps(original_read)
    def read(module_name, cache_key, *args, **kwargs):
        entry = {
            "key": cache_key,
            "layered_get_calls": 0,
            "layered_get_seconds": 0.0,
            "decompress_seconds": 0.0,
        }
        token = reading.set(entry)
        started = time.perf_counter()
        try:
            result = original_read(module_name, cache_key, *args, **kwargs)
            entry["hit"] = result[0] is not None
            entry["status"] = "ok"
            return result
        except BaseException as error:
            entry.update(status="error", error_type=type(error).__name__)
            raise
        finally:
            reading.reset(token)
            entry["seconds"] = time.perf_counter() - started
            entry["load_residual_seconds"] = max(
                0.0,
                entry["seconds"]
                - entry["layered_get_seconds"]
                - entry["decompress_seconds"],
            )
            if (owner := current.get()) is not None:
                owner.setdefault("cache_reads", []).append(entry)

    @wraps(original_backend_compile)
    def backend_compile(*args, **kwargs):
        started = time.perf_counter()
        try:
            return original_backend_compile(*args, **kwargs)
        finally:
            if (entry := current.get()) is not None:
                entry["backend_compile_and_load_calls"] = (
                    entry.get("backend_compile_and_load_calls", 0) + 1
                )
                entry["backend_compile_and_load_seconds"] = (
                    entry.get("backend_compile_and_load_seconds", 0.0)
                    + time.perf_counter()
                    - started
                )

    @wraps(original_compile)
    def compile_or_load(backend, computation, *args, **kwargs):
        entry = {
            "module": str(computation.operation.attributes["sym_name"]).strip('"'),
        }
        if fingerprint_hlo:
            entry["hlo_fingerprints"] = _input_fingerprints(computation, args, kwargs)
            entry["device_before_compile_or_load"] = _memory_observation()
        token = current.set(entry)
        started = time.perf_counter()
        try:
            executable = original_compile(backend, computation, *args, **kwargs)
            entry.update(memory=executable_memory(executable), status="ok")
            if fingerprint_hlo:
                try:
                    entry["hlo_fingerprints"]["optimized_hlo_sha256"] = (
                        _optimized_fingerprints(executable)
                    )
                except Exception as error:
                    entry["hlo_fingerprints"]["optimized_unavailable"] = type(
                        error
                    ).__name__
            return executable
        except BaseException as error:
            entry.update(status="error", error_type=type(error).__name__)
            raise
        finally:
            current.reset(token)
            entry["compile_or_load_seconds"] = time.perf_counter() - started
            if fingerprint_hlo:
                entry["device_after_compile_or_load"] = _memory_observation()
            executables.append(entry)

    with ExitStack() as stack:
        for owner, name, observer in (
            (compiler, "compile_or_get_cached", compile_or_load),
            (compiler, "_cache_read", read),
            (compiler, "backend_compile_and_load", backend_compile),
            (compilation_cache, "decompress_executable", decompress),
            (LayeredCache, "get", get),
        ):
            stack.enter_context(patch.object(owner, name, observer))
        yield
