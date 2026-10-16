"""Bind Tokamax's implicit hardware queries to one prediction's device.

Tokamax 0.0.13 queries the process-default backend in several kernel helpers.
A JAX default-device context controls array placement but does not change those
queries. Only Tokamax's module-local JAX bindings are adapted here; unrelated
JAX callers and explicit backend queries retain their usual behavior.
"""

from __future__ import annotations

import importlib
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from types import ModuleType
from typing import Any

_MODULES = (
    "tokamax._src.pallas.grid",
    "tokamax._src.ops.normalization.pallas_triton_config",
    "tokamax._src.ops.gated_linear_unit.pallas_triton",
    "tokamax._src.precision",
    "tokamax._src.benchmarking",
)
_DEVICE: ContextVar[Any | None] = ContextVar("foldjax_tokamax_device", default=None)
_LOCK = threading.RLock()
_PATCHES: dict[ModuleType, tuple[Any, int]] = {}


class _SelectedJax:
    def __init__(self, original: Any) -> None:
        self._original = original

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original, name)

    def devices(self, backend: Any = None) -> list[Any]:
        selected = _DEVICE.get()
        if selected is not None and backend is None:
            return [selected]
        return self._original.devices(backend=backend)

    def default_backend(self) -> str:
        selected = _DEVICE.get()
        if selected is None:
            return self._original.default_backend()
        platform = str(selected.platform)
        return "gpu" if platform == "cuda" else platform


@contextmanager
def selected_tokamax_device(device: Any) -> Iterator[None]:
    """Use this device in Tokamax helpers, restoring nested and concurrent calls.

    Worker threads do not inherit this context, just as they do not inherit
    JAX's default-device context. FoldJAX permits parallel tuning only on the
    process-default device; selected-device tuning runs synchronously.
    """
    token = _DEVICE.set(device)
    modules = []
    try:
        for name in _MODULES:
            try:
                module = importlib.import_module(name)
            except ModuleNotFoundError as error:
                if error.name == "tokamax":
                    break
                raise
            with _LOCK:
                if module in _PATCHES:
                    original, users = _PATCHES[module]
                else:
                    original, users = module.jax, 0
                    module.jax = _SelectedJax(original)
                _PATCHES[module] = original, users + 1
                modules.append(module)
        yield
    finally:
        with _LOCK:
            for module in reversed(modules):
                original, users = _PATCHES[module]
                if users == 1:
                    module.jax = original
                    del _PATCHES[module]
                else:
                    _PATCHES[module] = original, users - 1
        _DEVICE.reset(token)
