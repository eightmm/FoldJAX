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

An OOM that never arrives cannot be explained at all, which is the second half
of this module. Under context parallelism one device's allocator can fail while
the others wait at XLA's collective rendezvous for as long as the scheduler
allows -- see ``CP_RENDEZVOUS_SECONDS``, which bounds that wait, and
``record_mesh``, which is how the explanation comes to say *which* device ran
out.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from contextvars import ContextVar
from typing import Any

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

#: jaxlib renamed this knob. Both spellings still work *alone*, and setting
#: both together is a hard error inside the CUDA plugin's initialize(): it
#: raises, the plugin does not load, and JAX then reports "an NVIDIA GPU may be
#: present on this machine, but a CUDA-enabled jaxlib is not installed" and
#: falls back to the CPU. The jaxlib is installed; nothing about that message
#: points at the pair of variables that caused it. So FoldJAX must never add
#: its spelling next to one the caller already chose.
CURRENT_FRACTION_ENV = "XLA_CLIENT_MEM_FRACTION"
FRACTION_ENV = "XLA_PYTHON_CLIENT_MEM_FRACTION"


def configured_fraction_env() -> str | None:
    """Whichever spelling of the pool fraction this process already carries."""
    for name in (CURRENT_FRACTION_ENV, FRACTION_ENV):
        if name in os.environ:
            return name
    return None


def set_mem_fraction(value: float, *, override: bool = False) -> None:
    """Ask XLA for a pool fraction without costing the caller their GPU.

    ``override`` replaces whichever spelling is already present rather than
    adding a second one. With no spelling present the current name is written,
    because that is the one jaxlib documents.
    """
    existing = configured_fraction_env()
    if existing is None:
        os.environ[CURRENT_FRACTION_ENV] = str(value)
    elif override:
        os.environ[existing] = str(value)


# ---------------------------------------------------------------------------
# A rank that runs out of memory must not leave the other ranks waiting
# ---------------------------------------------------------------------------

#: Seconds XLA may wait at a collective rendezvous before ending the process.
#:
#: Under context parallelism an OOM is an expected outcome -- CP is for targets
#: that do not fit one card -- so it has to end the job with the diagnosis
#: below. What happened instead, three times on 2- and 4-GPU runs: one device's
#: allocator failed, the other devices' threads kept waiting at the rendezvous
#: XLA holds before each execution, and the job hung until the scheduler killed
#: it two hours later.
#:
#: That wait is unbounded by default. At the XLA revision this jaxlib pins
#: (jax 0.11.1 -> openxla/xla dcf304bc),
#: `xla/backends/gpu/collectives/gpu_cliques.cc:87` warns after a hardcoded ten
#: seconds -- the `rendezvous.cc` "This thread has been waiting for `... Acquire
#: clique ...`" line in the log -- and `:89-93` takes the *terminate* timeout
#: from `xla_gpu_nccl_termination_timeout_seconds`, whose default is `-1`
#: (`xla/debug_options_flags.cc:355`), which it reads as
#: `absl::InfiniteDuration()`.
#:
#: Ten minutes rather than a tighter bound because the rendezvous is keyed by
#: `run_id` (`gpu_cliques.cc:864`): it is entered on every dispatch, not once
#: per process, so the bound has to clear any legitimate skew between ranks
#: reaching the same execution. It is short enough that a failure is a failure
#: within the hour, which is the point.
CP_RENDEZVOUS_SECONDS = 600

#: The flag that carries it. XLA reads it through `GetDebugOptionsFromFlags()`
#: into a function-local `static const` (`gpu_cliques.cc:89-93`), and
#: `XLA_FLAGS` itself is parsed when the backend initialises. Both facts say
#: the same thing: after JAX has a backend, setting this changes nothing.
RENDEZVOUS_FLAG = "xla_gpu_nccl_termination_timeout_seconds"

#: Overrides the bound, in seconds. A negative value means to XLA what it
#: always meant -- wait forever -- and is how a caller declines the bound.
#: Zero is refused rather than read as "no timeout": to XLA it is a
#: zero-second timeout, which would end the first dispatch of every CP run.
#:
#: It tunes the value FoldJAX writes, so it only means anything while FoldJAX
#: still gets to write one -- before a backend exists. A process that is
#: already up is past that, and only `XLA_FLAGS` carrying the flag itself
#: reaches XLA at all; that is what the late warning in `models/_cp` says.
RENDEZVOUS_ENV = "FOLDJAX_CP_RENDEZVOUS_TIMEOUT"

_XLA_FLAGS_ENV = "XLA_FLAGS"

#: Platform names that can reach a GPU collective rendezvous at all.
_GPU_PLATFORMS = frozenset({"cuda", "gpu", "rocm"})

#: In JAX's own precedence order; the first one set decides.
_PLATFORM_ENVS = ("JAX_PLATFORMS", "JAX_PLATFORM_NAME")


def rendezvous_timeout_seconds() -> int | None:
    """The bound to ask XLA for, or ``None`` when the caller declined it.

    Raises ``ValueError`` on a value that cannot mean a number of seconds, and
    on ``0``. Both are startup mistakes worth failing on: a run that silently
    kept the unbounded default is the defect this exists to remove.
    """
    raw = os.environ.get(RENDEZVOUS_ENV)
    if raw is None or not raw.strip():
        return CP_RENDEZVOUS_SECONDS
    try:
        seconds = int(raw.strip())
    except ValueError:
        raise ValueError(
            f"{RENDEZVOUS_ENV} must be a whole number of seconds; got {raw!r}"
        ) from None
    if seconds == 0:
        raise ValueError(
            f"{RENDEZVOUS_ENV}=0 is a zero-second timeout to XLA, which would "
            "end every context-parallel dispatch; pass a negative value to "
            "wait forever instead"
        )
    return seconds if seconds > 0 else None


def rendezvous_timeout_is_set() -> bool:
    """Whether ``XLA_FLAGS`` already carries the terminate timeout."""
    return RENDEZVOUS_FLAG in os.environ.get(_XLA_FLAGS_ENV, "")


def set_rendezvous_timeout() -> str | None:
    """Bound XLA's collective rendezvous; return the composed ``XLA_FLAGS``.

    ``None`` when nothing was written, which is every case where writing would
    be wrong: the flag is already there -- a value the caller chose wins, as it
    does for the pool fraction -- or the caller declined the bound. Calling it
    twice composes it once, because the second call reads the flag the first
    one wrote.

    Whether this is early enough is the caller's to know, and it is not
    knowable from here: `XLA_FLAGS` is parsed when the backend initialises.
    `foldjax.cli` calls this while resolving arguments, before anything imports
    JAX; `models/_cp.context_parallel` calls it only while no backend exists
    and says so in a warning when one already does.
    """
    if rendezvous_timeout_is_set():
        return None
    seconds = rendezvous_timeout_seconds()
    if seconds is None:
        return None
    existing = os.environ.get(_XLA_FLAGS_ENV, "")
    composed = f"{existing} --{RENDEZVOUS_FLAG}={seconds}".strip()
    os.environ[_XLA_FLAGS_ENV] = composed
    return composed


def is_gpu_platform(name: str) -> bool:
    """Whether a `jax.Device.platform` string is one with a NCCL rendezvous."""
    return name.strip().lower() in _GPU_PLATFORMS


def gpu_is_possible() -> bool:
    """Whether this process can still end up on a GPU.

    A ``--xla_gpu_*`` flag is inert off GPU, but ``XLA_FLAGS`` is process-wide
    and inherited by every child a run spawns, so a process pinned to the CPU
    is left alone. The question has to be answered from the environment: the
    platform is only knowable from ``jax.devices()``, and asking that
    initialises the backend -- after which the flag can no longer be set.
    """
    for name in _PLATFORM_ENVS:
        raw = os.environ.get(name)
        if raw is None or not raw.strip():
            continue
        wanted = {item.strip().lower() for item in raw.split(",") if item.strip()}
        return bool(wanted & _GPU_PLATFORMS)
    return True


#: The mesh the most recent context-parallel entry built, or ``None``.
#:
#: `models/_cp` keeps the *active* mesh in a ContextVar it resets on exit,
#: which is the right lifetime for sharding and the wrong one here: an OOM is
#: diagnosed in `foldjax.api`, by which point the context has unwound and the
#: mesh that produced the failure is gone. This record survives the unwinding,
#: and `foldjax.api` clears it before each prediction so that no run inherits
#: the previous one's topology -- the same pattern, for the same reason, as
#: `memory_policy._RECORDED`.
_MESH: ContextVar[dict[str, Any] | None] = ContextVar("foldjax_cp_mesh", default=None)


def record_mesh(*, layout: str, devices: int, grid: tuple[int, int]) -> None:
    """Keep this topology for whatever failure comes next. JSON-native only."""
    _MESH.set(
        {
            "layout": str(layout),
            "devices": int(devices),
            "grid": [int(grid[0]), int(grid[1])],
        }
    )


def recorded_mesh() -> dict[str, Any] | None:
    """The topology this prediction entered, or ``None`` if it ran serially."""
    return _MESH.get()


def clear_mesh_record() -> None:
    """Forget the last topology, so the next run reports only its own."""
    _MESH.set(None)


def _mesh_sentences(mesh: Mapping[str, Any]) -> str:
    """What an OOM means when it is one device of a mesh rather than the job."""
    rows, columns = mesh["grid"]
    grid = f"{rows}x{columns} grid" if columns > 1 else f"{rows} rows"
    return (
        f"This is one device of a {mesh['devices']}-device context-parallel mesh "
        f"({mesh['layout']} layout, {grid}), so every figure here is one "
        "device's and not the job's: each device gets the same fraction, and "
        "the allocator that failed need not be the device these statistics "
        "were read from. The levers are cp_devices, which divides what the "
        "layout shards and nothing else -- replicated state sits on every "
        "device whatever the count -- and cp_layout, which decides whether the "
        "pair state is split by rows alone or by rows and columns. What one "
        "device is expected to need is per model: foldjax.memory_policy holds "
        "the fitted laws."
    )


_PREALLOCATE_ENV = "XLA_PYTHON_CLIENT_PREALLOCATE"
_REQUESTED = re.compile(r"allocate ([0-9.]+)([KMG]i?B)")
_UNITS = {"KiB": 2**10, "MiB": 2**20, "GiB": 2**30, "B": 1}


def is_out_of_memory(error: BaseException) -> bool:
    """Whether this is an allocator failure rather than any other runtime error."""
    return "RESOURCE_EXHAUSTED" in str(error) or "Out of memory" in str(error)


def mem_fraction() -> float:
    """The configured pool fraction, or JAX's default."""
    name = configured_fraction_env()
    raw = os.environ.get(name) if name else None
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


def _pool_card_and_used(
    *, require_preallocated: bool = True
) -> tuple[int | None, int | None, int | None]:
    """The allocator's ceiling, the device's capacity and what is held, in bytes.

    ``bytes_limit`` is the pool and ``bytes_in_use`` is what it currently holds.
    JAX exposes no device total, so the card is recovered from the fraction that
    produced the pool -- exact when preallocation is on, which is the case this
    diagnosis is about.

    ``require_preallocated`` is what separates the two readers. Measured with
    both settings, ``bytes_limit`` is the fraction times the card either way
    (0.9 -> 91,779,760,128 B; 0.25 -> 25,495,076,864 B), so the *ceiling* is
    always readable and a preflight can compare against it. What preallocation
    off removes is the reserved pool: nothing is held up front, so
    ``bytes_in_use`` after a failure no longer says how much of the ceiling the
    run had already taken, which is the number :func:`diagnose` reasons with.
    """
    if require_preallocated and os.environ.get(_PREALLOCATE_ENV, "").lower() in {
        "false",
        "0",
    }:
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
    """The allocator's ceiling and the device's capacity in bytes, or ``None``s.

    A public read of the same numbers :func:`diagnose` uses after a failure, so
    a caller can ask *before* one -- which is what `foldjax.memory_policy`
    admits runs against.

    Read whenever ``bytes_limit`` is present, preallocation on or off. This
    used to return ``None`` with preallocation off, on the reasoning that there
    is no pool to be short of. The ceiling is there either way; only the
    reserved pool is not. Both are ``None`` on a platform that reports no
    memory statistics at all, CPU among them.
    """
    pool, card, _ = _pool_card_and_used(require_preallocated=False)
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
        f"{configured_fraction_env() or CURRENT_FRACTION_ENV}."
    ]
    requested = _requested_bytes(str(error))
    # The allocation that failed, on top of what the pool already holds, is the
    # smallest total this run needed. Three outcomes, and they take different
    # actions: past the device, no setting helps; past the pool but inside the
    # device, the fraction is the lever; inside both, the total fits and a
    # contiguous block of it does not, and raising the fraction is wasted work.
    # Comparing the failed allocation against the device on its own proves
    # nothing: it is a fraction of the need.
    if requested is not None and used is not None:
        need = used + requested
        if need > card:
            lines.append(
                f"With the {used / gib:.1f} GiB already held that is "
                f"{need / gib:.1f} GiB, past the device itself."
            )
        elif need > pool:
            lines.append(
                f"It already held {used / gib:.1f} GiB, so it needed "
                f"{need / gib:.1f} GiB at that moment -- inside the "
                f"{card / gib:.1f} GiB device but past the pool. This is the "
                "pool's limit and not the card's."
            )
        else:
            # Fits the pool and the device, and still failed. The pool is
            # assembled piecewise, so it can hold the total and be unable to
            # serve one contiguous block of it. Saying "past the pool" here --
            # which this branch used to, because it only compared against the
            # card -- sends the reader to raise a fraction that cannot help:
            # measured, a 90.5 GiB need failed inside a 93.1 GiB pool and
            # failed again at 0.98.
            lines.append(
                f"It already held {used / gib:.1f} GiB, so it needed "
                f"{need / gib:.1f} GiB at that moment -- inside both the "
                f"{pool / gib:.1f} GiB pool and the {card / gib:.1f} GiB "
                "device, so a larger fraction will not help: the allocator "
                "could not place this block, not find the bytes. The live "
                "figure above does not say why -- a nearly empty pool can fail "
                "a half-pool request because the program\'s own co-live set "
                "leaves no room to lay one out. XLA\'s rematerialization lines "
                "in this log report the total it could not get the program "
                "under; compare that with the pool."
            )
    mesh = recorded_mesh()
    if mesh is not None:
        lines.append(_mesh_sentences(mesh))
    lines.append(
        f"Raise it with {configured_fraction_env() or CURRENT_FRACTION_ENV}"
        "=0.95 if the total is past the pool, or lower the job's cost with "
        "--num-samples. If the run needs more than the device has, the fraction "
        "will not help. On a model that takes it, "
        "--option diffusion_chunk_size=1 narrows the same sample axis without "
        "dropping a prediction, but whether that touches the buffer that failed "
        "is model-specific: measured at 4,100 tokens it turned an OpenFold3 "
        "failure into a 76.4 GiB run and moved Protenix by one mebibyte. "
        "--max-msa-depth is not a memory lever here: measured at "
        "488 and at 4,100 tokens it moved the peak by under 1%, because the "
        "alignment tensor is not co-live with the pair stack that sets it."
    )
    return " ".join(lines)
