"""Bounded ownership for whole-model JAX executables.

Module-level ``jax.jit`` wrappers retain every shape and static configuration
they have ever compiled. That is convenient for small kernels, but a serving
process can otherwise retain an unbounded number of multi-gigabyte model
executables. This module gives each graph identity its own JIT owner so the
least-recently-used owner can be cleared explicitly.
"""

from __future__ import annotations

import functools
import inspect
import threading
from collections import OrderedDict
from collections.abc import Callable, Mapping
from typing import Any

import jax


def _static_identity(value: Any) -> Any:
    """Return a recursively hashable identity for one static argument."""

    if isinstance(value, Mapping):
        return tuple(
            sorted((key, _static_identity(item)) for key, item in value.items())
        )
    if isinstance(value, (tuple, list)):
        return tuple(_static_identity(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return tuple(sorted(_static_identity(item) for item in value))
    try:
        hash(value)
    except TypeError:
        return (type(value).__module__, type(value).__qualname__, repr(value))
    return value


def _sharding_identity(value: Any) -> tuple[str, str] | None:
    sharding = getattr(value, "sharding", None)
    if sharding is None:
        return None
    return (type(sharding).__qualname__, repr(sharding))


def _dynamic_identity(tree: Any) -> tuple[Any, tuple[Any, ...]]:
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    signatures = []
    for leaf in leaves:
        aval = jax.typeof(leaf)
        signatures.append(
            (
                tuple(int(size) for size in aval.shape),
                str(aval.dtype),
                bool(getattr(aval, "weak_type", False)),
                _sharding_identity(leaf),
            )
        )
    return treedef, tuple(signatures)


class BoundedJitPool:
    """A callable ``jax.jit`` facade with an LRU bound on compiled programs."""

    def __init__(
        self,
        function: Callable[..., Any],
        *,
        static_argnames: tuple[str, ...] = (),
        limit: int = 8,
    ) -> None:
        if limit < 1:
            raise ValueError("a bounded JIT pool must retain at least one entry")
        self._function = function
        self._static_argnames = tuple(static_argnames)
        self._static_argname_set = frozenset(static_argnames)
        signature = inspect.signature(function)
        self._static_position_names: dict[int, str] = {}
        position = 0
        for name, parameter in signature.parameters.items():
            if parameter.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            ):
                if name in self._static_argname_set:
                    self._static_position_names[position] = name
                position += 1
            elif parameter.kind is inspect.Parameter.VAR_POSITIONAL:
                # Named static arguments cannot identify individual *args.
                break
        self._limit = int(limit)
        self._entries: OrderedDict[Any, Any] = OrderedDict()
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._active: dict[Any, int] = {}
        self._new_succeeded: dict[Any, bool] = {}
        self._clearing = False
        self.__wrapped__ = function
        self.__name__ = getattr(function, "__name__", type(function).__name__)
        self.__doc__ = function.__doc__

    def _identity(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        # Preserve JAX's call-style cache semantics: omitted defaults, explicit
        # keyword defaults and positional defaults are distinct jit cache keys
        # in JAX 0.11.1. Applying signature defaults only to this outer key
        # would hide multiple executables inside one owner instead of deduping
        # them. We normalize only which explicitly supplied values are static.
        static_values = {
            self._static_position_names[position]: _static_identity(value)
            for position, value in enumerate(args)
            if position in self._static_position_names
        }
        dynamic_args = tuple(
            value
            for position, value in enumerate(args)
            if position not in self._static_position_names
        )
        dynamic_kwargs = {
            name: value
            for name, value in kwargs.items()
            if name not in self._static_argname_set
        }
        static_values.update(
            (name, _static_identity(value))
            for name, value in kwargs.items()
            if name in self._static_argname_set
        )
        static = tuple(sorted(static_values.items()))
        runtime = _static_identity(jax.config.values)
        return runtime, static, _dynamic_identity((dynamic_args, dynamic_kwargs))

    def _new(self):
        # A fresh partial is a distinct JAX cache owner. Wrapping the same
        # function object repeatedly would make the wrappers share JAX's
        # underlying cache and defeat per-identity eviction.
        owned_function = functools.partial(self._function)
        return jax.jit(
            owned_function,
            static_argnames=self._static_argnames,
        )

    @staticmethod
    def _entry_size(compiled: Any) -> int:
        size = getattr(compiled, "_cache_size", None)
        return int(size()) if callable(size) else 1

    @staticmethod
    def _drop(compiled: Any) -> None:
        # JAX's public clear_cache() also flushes process-wide parameter
        # inference metadata. Prefer the owner's cache primitive so evicting a
        # Protenix graph does not perturb unrelated JIT wrappers. The public
        # method remains the compatibility fallback for another JAX version.
        clear = getattr(compiled, "_clear_cache", None)
        if not callable(clear):
            clear = getattr(compiled, "clear_cache", None)
        if callable(clear):
            clear()

    def _evict_one_idle_unlocked(self) -> bool:
        for identity, compiled in tuple(self._entries.items()):
            if self._active[identity] == 0:
                del self._entries[identity]
                del self._active[identity]
                self._new_succeeded.pop(identity, None)
                self._drop(compiled)
                return True
        return False

    def _acquire(self, identity: Any) -> Any:
        with self._condition:
            while True:
                while self._clearing:
                    self._condition.wait()

                compiled = self._entries.pop(identity, None)
                if compiled is not None:
                    self._entries[identity] = compiled
                    self._active[identity] += 1
                    return compiled

                if len(self._entries) < self._limit:
                    compiled = self._new()
                    self._entries[identity] = compiled
                    self._active[identity] = 1
                    self._new_succeeded[identity] = False
                    return compiled

                if self._evict_one_idle_unlocked():
                    continue

                # Every retained owner is tracing or dispatching. Waiting here
                # preserves the hard entry bound, while existing warm
                # identities can still acquire a lease and dispatch.
                self._condition.wait()

    def _cache_size_unlocked(self) -> int:
        return sum(self._entry_size(entry) for entry in self._entries.values())

    def _trim_unlocked(self) -> None:
        # An identity should normally own one executable. If JAX discovers an
        # additional trace context that the public identity cannot observe,
        # evicting even the current owner is deliberate: the executable bound
        # must fail closed instead of silently becoming unbounded.
        while self._entries and self._cache_size_unlocked() > self._limit:
            if not self._evict_one_idle_unlocked():
                return

    def _release(
        self,
        identity: Any,
        compiled: Any,
        *,
        succeeded: bool,
    ) -> None:
        with self._condition:
            if succeeded and identity in self._new_succeeded:
                self._new_succeeded[identity] = True

            active = self._active[identity] - 1
            self._active[identity] = active
            if (
                active == 0
                and identity in self._new_succeeded
                and not self._new_succeeded[identity]
                and self._entries.get(identity) is compiled
            ):
                del self._entries[identity]
                del self._active[identity]
                del self._new_succeeded[identity]
                self._drop(compiled)
            else:
                if active == 0:
                    self._new_succeeded.pop(identity, None)
                self._trim_unlocked()
            self._condition.notify_all()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        identity = self._identity(args, kwargs)
        compiled = self._acquire(identity)
        try:
            result = compiled(*args, **kwargs)
        except BaseException:
            self._release(
                identity,
                compiled,
                succeeded=False,
            )
            raise
        self._release(
            identity,
            compiled,
            succeeded=True,
        )
        return result

    def lower(self, *args: Any, **kwargs: Any) -> Any:
        # A lowered/AOT executable is owned by its caller and never enters the
        # jit wrapper's dispatch cache. Give it a fresh owner so inspection or
        # export cannot evict a warm inference entry while pretending to be
        # covered by this pool's runtime bound.
        return self._new().lower(*args, **kwargs)

    def clear_cache(self) -> None:
        with self._condition:
            self._clearing = True
            try:
                while any(self._active.values()):
                    self._condition.wait()
                for compiled in self._entries.values():
                    self._drop(compiled)
                self._entries.clear()
                self._active.clear()
                self._new_succeeded.clear()
            finally:
                self._clearing = False
                self._condition.notify_all()

    def _cache_size(self) -> int:
        with self._lock:
            return self._cache_size_unlocked()

    def _entry_count(self) -> int:
        with self._lock:
            return len(self._entries)
