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
imported would be deciding for its host application; the CLI turns it on for
its own process, and `FOLDJAX_PROGRESS=0` turns it off again for people piping
stderr somewhere that should stay clean.
"""

from __future__ import annotations

import os
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
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


__all__ = ["Timeline", "disable", "enable", "enabled", "header"]
