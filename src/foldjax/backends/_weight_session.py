"""Request-scoped ownership for one verifiable native weight source."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from foldjax.manifest import path_stat_identity
from foldjax.schema import PredictionError


def _file_snapshot(path: Path) -> tuple[str, str, str] | None:
    """Lexical path, resolved target and complete cheap stat identity."""

    identity = path_stat_identity(path)
    if identity is None:
        return None
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError):
        return None
    return (
        str(path.absolute()),
        str(resolved),
        json.dumps(identity, sort_keys=True, separators=(",", ":")),
    )


class PreparedWeightSession:
    """Retain one prepared tree only for the lifetime of one request batch."""

    def __init__(self, model: str) -> None:
        self._model = model
        self._open = False
        self._active = False
        self._poisoned: str | None = None
        self._anchors: dict[str, tuple[str, str, str] | None] = {}
        self._cached: tuple[tuple[Any, ...], Any] | None = None

    @property
    def active(self) -> bool:
        return self._active

    @contextmanager
    def session(self, requests: Sequence[Any]) -> Iterator[None]:
        if self._open:
            raise RuntimeError(
                f"nested {self._model} backend sessions are not supported"
            )
        self._open = True
        self._active = sum(len(request.resolved_seeds) for request in requests) > 1
        try:
            yield
        finally:
            self.invalidate()
            self._anchors.clear()
            self._poisoned = None
            self._active = False
            self._open = False

    def invalidate(self) -> None:
        self._cached = None

    def _poison(self, message: str) -> None:
        self.invalidate()
        self._poisoned = message
        raise PredictionError(message)

    def _anchor(
        self,
        path: Path,
        *,
        require_verifiable: bool = False,
    ) -> tuple[str, tuple[str, str, str] | None]:
        if self._poisoned is not None:
            raise PredictionError(self._poisoned)
        source = str(path)
        snapshot = _file_snapshot(path)
        missing = object()
        expected = self._anchors.get(source, missing)
        if expected is missing:
            self._anchors[source] = snapshot
        elif expected is None:
            snapshot = None
        elif snapshot != expected:
            self._poison(
                f"{self._model} weights changed while a prediction batch was active"
            )
        if require_verifiable and snapshot is None:
            self._poison(
                f"{self._model} cannot verify weights used by a resumed prediction"
            )
        return source, snapshot

    def validate(self, path: Path, *, resumed: bool = False) -> None:
        if self._active:
            self._anchor(path, require_verifiable=resumed)

    def load(
        self,
        path: Path,
        loader: Callable[[Path], Any],
        *,
        prepare_key: tuple[Any, ...] = (),
    ) -> Any:
        """Load once, pre-evicting an incompatible tree before replacement."""

        if not self._active:
            return loader(path)
        source, snapshot = self._anchor(path)
        key = (source, snapshot, prepare_key)
        cached = self._cached
        if cached is not None:
            if snapshot is not None and cached[0] == key:
                return cached[1]
            self._cached = None
            # Do not hold a multi-gigabyte old generation in this frame while
            # the replacement loader constructs another one.
            del cached

        value = loader(path)
        after = _file_snapshot(path)
        if snapshot is None or after is None:
            return value
        if after != snapshot:
            self._poison(f"{self._model} weights changed while they were loading")
        self._cached = (key, value)
        return value
