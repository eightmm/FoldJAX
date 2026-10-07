"""What the run is doing right now, and what each part of it cost.

A prediction is minutes of silence. Weight fetching has had a progress line
since the beginning (`foldjax.cli._WeightReporter`), but the part that actually
takes the time -- featurize, compile, sample, write -- printed nothing at all,
and at short token counts most of that time is one XLA compile. The honest
reading of a silent terminal is "it has hung", and that is what people concluded.

The same measurement answers a second question. `foldjax_run.json` recorded one
`seconds` for the whole run, so "it took eleven minutes" could not be split into
the parts a person can do something about: a slow alignment search, a cold
compile, and a long sample schedule call for three different responses. A
`Timeline` records each phase once and hands it to both consumers -- the live
line on stderr and the `cost.phases` mapping in the manifest -- so the number
you watch and the number you keep are the same number.

Printing is off by default. A library that wrote to stderr because it was
imported would be deciding for its host application; `foldjax predict` turns it
on while the command runs and puts the host's setting back when it returns, and
`FOLDJAX_PROGRESS=0` turns it off again for people piping stderr somewhere that
should stay clean.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, TextIO

_ENV = "FOLDJAX_PROGRESS"

_enabled = False

#: Only set when a caller named a stream. Otherwise ``sys.stderr`` is resolved
#: at write time, not at enable time: holding the object means writing to
#: whatever stderr *was*, which in a host application that swapped it -- or a
#: test harness that captured and then closed it -- is a stream that no longer
#: exists.
_stream: TextIO | None = None


def enable(stream: TextIO | None = None) -> None:
    """Send progress lines to ``stream`` (default stderr) for this process."""
    global _enabled, _stream
    if os.environ.get(_ENV, "").strip() in {"0", "false", "no", "off"}:
        _enabled, _stream = False, None
        return
    _enabled, _stream = True, stream


def disable() -> None:
    global _enabled, _stream
    _enabled, _stream = False, None


def enabled() -> bool:
    return _enabled


def _write(text: str) -> None:
    if not _enabled:
        return
    target = _stream if _stream is not None else sys.stderr
    try:
        print(text, file=target, flush=True)
    except (ValueError, OSError):
        # A progress line must never be the reason a prediction fails. If the
        # destination has gone away, stop trying rather than raise into a run
        # that is otherwise fine.
        disable()


def _duration(seconds: float) -> str:
    """Durations people read at a glance: 4.1s, 1m42s, 2h03m."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, remainder = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m{remainder:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


def message(text: str) -> None:
    """One free-form progress line: a wait, a retry, a search that failed."""
    _write(text)


def header(model: str, target: str, seed: int) -> None:
    """Announce which run the following stage lines belong to."""
    _write(f"[foldjax] {model} · {target} · seed {seed}")


class Timeline:
    """Phase durations for one prediction, reported live and recorded after.

    Repeated labels accumulate rather than overwrite: a multi-sample backend may
    enter the same phase more than once, and the sum is what the phase cost.
    """

    def __init__(self) -> None:
        self._phases: dict[str, float] = {}
        self._parts: dict[str, float] = {}
        self._counts: dict[str, int] = {}
        self._lock = threading.Lock()

    @contextmanager
    def stage(self, label: str, *, detail: str = "") -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - started
            self._phases[label] = self._phases.get(label, 0.0) + elapsed
            suffix = f"  {detail}" if detail else ""
            _write(f"  {label:<18s}{_duration(elapsed):>8s}{suffix}")

    def note(self, label: str, detail: str) -> None:
        """A stage that took no measurable time but is worth seeing."""
        _write(f"  {label:<18s}{detail:>8s}")

    def summary(self) -> dict[str, Any]:
        return {label: round(value, 2) for label, value in self._phases.items()}

    @contextmanager
    def recording(self) -> Iterator[None]:
        """Attribute :func:`part` blocks and JAX compile events to this run.

        The parts nest inside the top-level phases rather than beside them, so
        they are kept apart from :meth:`summary`, whose phases sum to the run.
        """
        _install_jax_listeners()
        token = _ACTIVE.set(self)
        try:
            yield
        finally:
            _ACTIVE.reset(token)

    def _add_part(self, label: str, seconds: float) -> None:
        with self._lock:
            self._parts[label] = self._parts.get(label, 0.0) + seconds

    def _count(self, label: str) -> None:
        with self._lock:
            self._counts[label] = self._counts.get(label, 0) + 1

    def breakdown(self) -> dict[str, Any] | None:
        """Where the time inside the phases went, or None if nothing was seen.

        ``seconds`` holds what was measured directly: the parts a backend
        marked with :func:`part` (weight load, featurize, language model) and
        the compiler's own events -- ``trace``, ``lower``, ``compile`` (XLA's
        compile time with the persistent-cache reads taken out) and ``cache
        restore`` (those reads). ``execute and host`` is the ``predict`` phase
        less every part recorded inside it: device execution plus whatever host
        work no part names. It is derived, not measured.
        """
        with self._lock:
            parts = dict(self._parts)
            counts = dict(self._counts)
        if not parts and not counts:
            return None
        compile_total = parts.pop(_COMPILE_OR_RESTORE, 0.0)
        if compile_total:
            parts["compile"] = max(0.0, compile_total - parts.get("cache restore", 0.0))
        seconds = {label: round(value, 2) for label, value in parts.items()}
        predict = self._phases.get("predict")
        if predict is not None:
            inside = sum(parts.values())
            seconds["execute and host"] = round(max(0.0, predict - inside), 2)
        return {"seconds": seconds, "counts": counts}


#: The timeline :func:`part` and the JAX listeners report into, if any.
_ACTIVE: ContextVar[Timeline | None] = ContextVar("foldjax_timeline", default=None)

#: XLA's compile event spans the persistent-cache read too; the listener files
#: it here and :meth:`Timeline.breakdown` splits the read back out.
_COMPILE_OR_RESTORE = "_compile_or_restore"

#: JAX monitoring event -> breakdown label, for durations and for counts.
_JAX_DURATIONS = {
    "/jax/core/compile/jaxpr_trace_duration": "trace",
    "/jax/core/compile/jaxpr_to_mlir_module_duration": "lower",
    "/jax/core/compile/backend_compile_duration": _COMPILE_OR_RESTORE,
    "/jax/compilation_cache/cache_retrieval_time_sec": "cache restore",
}
_JAX_COUNTS = {
    "/jax/compilation_cache/cache_hits": "cache hits",
    "/jax/compilation_cache/cache_misses": "cache misses",
}
_listeners_lock = threading.Lock()
_listeners_installed = False


#: Whether a :func:`part` is open in this context. A compile inside one (the
#: conversion programs a weight loader builds, say) is already inside that
#: part's time, so it is counted but not timed again: parts stay disjoint and
#: the ``execute and host`` remainder is not subtracted twice.
_IN_PART: ContextVar[bool] = ContextVar("foldjax_in_part", default=False)


def _on_duration(event: str, seconds: float, **_kwargs: Any) -> None:
    label = _JAX_DURATIONS.get(event)
    timeline = _ACTIVE.get()
    if label is None or timeline is None:
        return
    if label == _COMPILE_OR_RESTORE:
        timeline._count("programs")
    if not _IN_PART.get():
        timeline._add_part(label, float(seconds))


def _on_event(event: str, **_kwargs: Any) -> None:
    label = _JAX_COUNTS.get(event)
    timeline = _ACTIVE.get()
    if label is not None and timeline is not None:
        timeline._count(label)


def _install_jax_listeners() -> None:
    """Register the two process-wide listeners once.

    They are inert outside :meth:`Timeline.recording`: an event is filed under
    the timeline active in the thread that compiled and dropped when there is
    none, so compiles a backend runs on its own worker threads go unattributed
    rather than attributed to the wrong run.
    """
    global _listeners_installed
    with _listeners_lock:
        if _listeners_installed:
            return
        from jax import monitoring

        monitoring.register_event_duration_secs_listener(_on_duration)
        monitoring.register_event_listener(_on_event)
        _listeners_installed = True


@contextmanager
def part(label: str) -> Iterator[None]:
    """Time one named part of a phase, when a recording timeline is active."""
    timeline = _ACTIVE.get()
    if timeline is None or _IN_PART.get():
        yield
        return
    token = _IN_PART.set(True)
    started = time.perf_counter()
    try:
        yield
    finally:
        timeline._add_part(label, time.perf_counter() - started)
        _IN_PART.reset(token)


__all__ = ["Timeline", "disable", "enable", "enabled", "header", "message", "part"]
