"""Take a named intermediate out of a traced program, and only that one.

A model here is one ``jax.jit`` program. Inside it every value is a tracer,
so the only way to see an intermediate from outside is for the traced
function to return it -- which is why this cannot be a plain read, and why
the set of names has to be static: a different set is a different program.

The rules that make it safe to leave the calls in place:

*Off is free.* ``capture`` checks the active name set at trace time and
returns its argument unchanged, so an un-requested tap leaves no trace in
the jaxpr and costs nothing at run time.

*On costs liveness, not bytes.* A captured value becomes an entry output.
Measured on a small program, the total does not grow -- a temporary becomes
an output instead (temp 64 KiB + output 64 KiB, against temp 0 + output 128).
What changes is that the temp arena is reused across a run and an entry
output is not, which is the whole of the 15.2 GiB peak that returning
Protenix's confidence logits used to cost at 3,012 tokens.

*Inside a loop it multiplies.* ``lax.scan`` can only hand back a per-iteration
value by stacking it, so a tap in the pairformer costs the loop length times
the array: OpenDDE's structural pair state is 1.37 GiB at 488 residues and
the stack is 48 blocks, which is 63 GiB for one careless name. Loop call
sites go through `capture_in_loop`, which refuses that unless the caller has
said how much it is willing to spend.
"""

from __future__ import annotations

import contextlib
import math
from collections.abc import Iterator, Sequence
from typing import Any

#: Names the active trace should hand back. Read at trace time only.
_WANTED: frozenset[str] = frozenset()
#: What this trace has recorded so far, in the order the model built it.
_COLLECTED: dict[str, Any] = {}

#: A stacked loop capture larger than this is refused unless the caller
#: raises the budget explicitly. Two gigabytes is enough for a per-block
#: single stream at any size this package runs, and far short of a per-block
#: pair stack at any size worth capturing.
DEFAULT_LOOP_BUDGET_BYTES = 2 * 2**30


@contextlib.contextmanager
def capturing(names: Sequence[str] | None) -> Iterator[dict[str, Any]]:
    """Ask this trace for ``names``; yields the dict they land in.

    The dict fills during tracing, so the model reads it inside its own body
    -- while the values are still tracers of the program being built -- and
    returns what it finds. Reading it after the trace gives concrete arrays
    only because the caller returned them.
    """
    global _WANTED, _COLLECTED
    previous_wanted, previous_collected = _WANTED, _COLLECTED
    _WANTED = frozenset(names or ())
    _COLLECTED = {}
    try:
        yield _COLLECTED
    finally:
        _WANTED, _COLLECTED = previous_wanted, previous_collected


def wanted() -> frozenset[str]:
    """The names the active trace was asked for."""
    return _WANTED


def is_wanted(name: str) -> bool:
    """Whether ``name`` would be recorded, for call sites that can skip work."""
    return name in _WANTED


def capture(name: str, value):
    """Record ``value`` under ``name`` if it was asked for; return it either way.

    Always returns its argument, so a call site reads as an annotation rather
    than a branch: ``z = capture("trunk.pair", z)``.
    """
    if name in _WANTED:
        _COLLECTED[name] = value
    return value


def collected() -> dict[str, Any]:
    """What the active trace recorded, in build order."""
    return dict(_COLLECTED)


def stacked_bytes(value, length: int) -> int:
    """Bytes a ``lax.scan`` would need to stack ``value`` over ``length`` steps."""
    itemsize = getattr(getattr(value, "dtype", None), "itemsize", 4)
    return int(math.prod(getattr(value, "shape", ()))) * itemsize * int(length)


def capture_in_loop(
    name: str,
    value,
    *,
    length: int,
    budget_bytes: int = DEFAULT_LOOP_BUDGET_BYTES,
) -> bool:
    """Whether a scanned call site should hand ``value`` back each iteration.

    Returns False when the name was not asked for, so the loop keeps its
    ordinary shape. Raises when it was asked for and stacking it would cost
    more than the budget -- with the number in the message, because "it OOMed
    after twenty minutes" is a much worse way to learn this.
    """
    if name not in _WANTED:
        return False
    cost = stacked_bytes(value, length)
    if cost > budget_bytes:
        raise ValueError(
            f"capturing {name!r} on every one of {length} loop steps would "
            f"stack {cost / 2**30:.1f} GiB, over the {budget_bytes / 2**30:.1f} "
            "GiB budget. Capture one step by name, or raise the budget "
            "deliberately if that is really what you want."
        )
    return True
