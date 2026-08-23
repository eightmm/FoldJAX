"""Turn an out-of-memory failure into something that says what actually happened.

XLA reports the allocation it could not satisfy and nothing about the ceiling it
hit. On a 96 GiB card that reads as "this model does not fit", and for Boltz-2 at
3012 tokens it is not: the run needs 73.32 GiB against a pool of 71.7 GiB, so it
died 1.6 GiB short with 24 GiB of the card unused. JAX preallocates
``XLA_PYTHON_CLIENT_MEM_FRACTION`` of the device -- 0.75 by default -- and that
fraction, not the card, is what a job runs out of.

The distinction matters because the two failures need opposite responses. A pool
limit is one flag away. A capacity limit means the model genuinely does not fit,
and raising the fraction only wastes another hour before failing again.
"""

from __future__ import annotations

import os
import re

#: JAX's own default when the variable is unset.
DEFAULT_MEM_FRACTION = 0.75

#: What `foldjax predict` asks for instead, when the caller has not said.
#:
#: JAX's 0.75 suits a process that shares a card. `foldjax predict` does not: it
#: owns the process for one prediction, and on a 95.6 GiB device 0.75 caps it at
#: 71.7 GiB -- which Boltz-2 at 3012 tokens misses by 1.6 GiB while a quarter of
#: the card sits reserved and unused. Its measured need is 73.33 GiB, so the
#: fraction it wanted was 0.77.
#:
#: 0.9 rather than 0.95 because the pool is not everything on the card: cuBLAS,
#: cuDNN and the cuEquivariance kernel take workspaces outside it, and a fraction
#: chosen to just fit the pool moves the failure there instead. This leaves
#: 9.6 GiB on that device against a measured need of 73.33.
PREDICT_MEM_FRACTION = 0.9

FRACTION_ENV = "XLA_PYTHON_CLIENT_MEM_FRACTION"
_PREALLOCATE_ENV = "XLA_PYTHON_CLIENT_PREALLOCATE"
_REQUESTED = re.compile(r"allocate ([0-9.]+)([KMG]i?B)")
_UNITS = {"KiB": 2**10, "MiB": 2**20, "GiB": 2**30, "B": 1}


def is_out_of_memory(error: BaseException) -> bool:
    """Whether this is an allocator failure rather than any other runtime error."""
    return "RESOURCE_EXHAUSTED" in str(error) or "Out of memory" in str(error)


def mem_fraction() -> float:
    """The configured pool fraction, or JAX's default."""
    raw = os.environ.get(FRACTION_ENV)
    if raw is None:
        return DEFAULT_MEM_FRACTION
    try:
        return float(raw)
    except ValueError:
        return DEFAULT_MEM_FRACTION


def _requested_bytes(message: str) -> int | None:
    match = _REQUESTED.search(message)
    if match is None:
        return None
    value, unit = match.groups()
    scale = _UNITS.get(unit)
    return int(float(value) * scale) if scale else None


def _pool_card_and_used() -> tuple[int | None, int | None, int | None]:
    """The allocator's ceiling, the device's capacity and what is held, in bytes.

    ``bytes_limit`` is the pool and ``bytes_in_use`` is what it currently holds.
    JAX exposes no device total, so the card is recovered from the fraction that
    produced the pool -- exact when preallocation is on, which is the case this
    diagnosis is about. With preallocation off there is no pool to be short of,
    so everything comes back ``None`` and the caller says nothing.
    """
    if os.environ.get(_PREALLOCATE_ENV, "").lower() in {"false", "0"}:
        return None, None, None
    try:
        import jax

        devices = jax.devices()
    except Exception:  # noqa: BLE001 - a broken runtime is not what we are reporting
        return None, None, None
    if not devices:
        return None, None, None
    stats = devices[0].memory_stats() or {}
    limit = stats.get("bytes_limit")
    if limit is None:
        return None, None, None
    fraction = mem_fraction()
    card = int(limit / fraction) if fraction > 0 else None
    used = stats.get("bytes_in_use")
    return int(limit), card, (int(used) if used is not None else None)


def device_budget() -> tuple[int | None, int | None]:
    """The allocator's pool and the device's capacity in bytes, or ``None``s.

    A public read of the same numbers :func:`diagnose` uses after a failure, so
    a caller can ask *before* one. Returns ``None`` when preallocation is off:
    there is no pool to be measured against, and a preflight that guessed one
    would be inventing its own ceiling.
    """
    pool, card, _ = _pool_card_and_used()
    return pool, card


def diagnose(error: BaseException) -> str | None:
    """Explain an OOM against the pool, or return ``None`` if it cannot be read.

    Silent about anything it cannot establish: a wrong explanation of a failure
    costs more than no explanation.
    """
    if not is_out_of_memory(error):
        return None
    pool, card, used = _pool_card_and_used()
    if pool is None or card is None:
        return None

    gib = 2**30
    lines = [
        f"the allocator's pool is {pool / gib:.1f} GiB -- "
        f"{mem_fraction():g} of a {card / gib:.1f} GiB device, set by "
        f"{FRACTION_ENV}."
    ]
    requested = _requested_bytes(str(error))
    # The allocation that failed, on top of what the pool already holds, is the
    # smallest total this run needed. If that total fits the device, the pool is
    # what stopped it -- and only the pool. Comparing the failed allocation
    # against the device on its own proves nothing: it is a fraction of the need.
    if requested is not None and used is not None:
        need = used + requested
        if need <= card:
            lines.append(
                f"It already held {used / gib:.1f} GiB, so it needed "
                f"{need / gib:.1f} GiB at that moment -- inside the "
                f"{card / gib:.1f} GiB device but past the pool. This is the "
                "pool's limit and not the card's."
            )
        else:
            lines.append(
                f"With the {used / gib:.1f} GiB already held that is "
                f"{need / gib:.1f} GiB, past the device itself."
            )
    lines.append(
        f"Raise it with {FRACTION_ENV}=0.95, or lower the job's cost "
        "(--num-samples, --max-msa-depth). If the run needs more than the device "
        "has, the fraction will not help."
    )
    return " ".join(lines)
