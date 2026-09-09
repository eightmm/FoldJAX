"""Session plumbing shared by the backends that lease chemistry lazily.

Three of the ports keep one managed-memory ``ExitStack`` per session and lease a
component-dictionary cache into it the first time a prediction needs chemistry;
three anchor their weights through :class:`PreparedWeightSession` with the same
three hooks. Both blocks were copied verbatim between backends, so what a port
still writes for itself is only what genuinely differs: the lease name and the
release callable, supplied by :meth:`ManagedCcdMemory._ccd_lease`.

That hook returns a whole context manager rather than the ``(name, release)``
pair because the release function must be imported late -- naming it at class
definition would pull a model package into cache planning -- and because each
port's ``managed_memory_lease`` stays resolved in its own module.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import AbstractContextManager, ExitStack, contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from foldjax.backends._weight_session import PreparedWeightSession
    from foldjax.backends.base import Backend
    from foldjax.schema import PredictionRequest


class WeightSessionHooks:
    """Bind the dispatcher's failure-isolation hooks to a weight session."""

    _weights: PreparedWeightSession

    def invalidate_session(self) -> None:
        self._weights.invalidate()

    def validate_session(self, request: PredictionRequest) -> None:
        if self._weights.active and request.weights is not None:
            self._weights.validate(Path(request.weights))

    def observe_resumed(self, request: PredictionRequest) -> None:
        if self._weights.active and request.weights is not None:
            self._weights.validate(Path(request.weights), resumed=True)


class ManagedCcdMemory:
    """Hold one chemistry lease per session, acquired on first use."""

    _managed_memory: ExitStack | None
    _ccd_memory_leased: bool

    def _ccd_lease(self) -> AbstractContextManager[None]:
        """Return this backend's chemistry lease, importing its release late."""

    @contextmanager
    def _ccd_memory_scope(self) -> Iterator[None]:
        memory = self._managed_memory
        if memory is not None:
            if not self._ccd_memory_leased:
                memory.enter_context(self._ccd_lease())
                self._ccd_memory_leased = True
            yield
        else:
            with self._ccd_lease():
                yield


class ManagedCcdSession(ManagedCcdMemory, WeightSessionHooks):
    """A weight session that also owns the managed-memory stack it leases into."""

    @contextmanager
    def session(self, requests: Sequence[PredictionRequest]) -> Iterator[Backend]:
        memory = ExitStack()
        try:
            with self._weights.session(requests):
                self._managed_memory = memory
                try:
                    yield self
                finally:
                    self._managed_memory = None
                    self._ccd_memory_leased = False
        finally:
            try:
                memory.close()
            except BaseException:
                pass
