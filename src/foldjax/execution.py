"""One vocabulary for the knobs every port spells differently.

`sampling_options` already does this for "how many structures" and its three
siblings: the request carries one neutral name and each adapter translates it.
Everything below that line was left native, so the same idea had four names and
two value vocabularies --

    boltz2     compute_dtype=bfloat16  triangle_backend=cueq
    protenix   trunk_dtype=bf16        trunk_triangle_attention_backend=cueq_jit
    opendde    trunk_dtype=bf16        (no triangle kernel option)
    openfold3  (no dtype option)       OPENFOLD3_TRIANGLE_BACKEND=cueq

-- which means a script that runs one model cannot run another without knowing
which port it is talking to, and a benchmark comparing them has to encode all
four spellings to say one thing.

This module holds the neutral names, the values they take, and the aliases the
old spellings keep. A backend declares `execution_options` mapping each neutral
name to its own name and value vocabulary; `Backend.apply_options` does the
translation, in the same place and with the same rules as sampling.

What this does **not** do is decide defaults. A neutral name means the same
thing everywhere; it does not follow that the same value is right everywhere,
and for `dtype` it measurably is not -- see `DEFAULTS` in each backend and the
measurements recorded beside them.

`matmul_precision` is the exception to the rename model, and was the hole in
it. Naming it here bought nothing for a long time: every one of the six
backends answered `does not support matmul_precision`, because there was
nothing to rename it *to*. Four ports open their own
`jax.default_matmul_precision` scope inside the call, so a scope an adapter
opened around them was simply overridden -- and the other two opened none, so
there was nothing to reach. It is now delivered by `matmul_precision_scope`
below: the adapter records the request, and the port that pins reads it where
it used to name its own constant. With no request in flight each port keeps
its own pinned value, which is upstream parity and measured -- Boltz-2 is
`highest` because `main.py:1096` asks for it, OpenFold3 and Protenix are
`high` because their upstreams select TF32.
"""

from __future__ import annotations

import contextvars
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

#: Neutral knob -> the values it accepts, in the neutral vocabulary.
#:
#: `auto` means "the fastest path this build can actually run". It is not a
#: synonym for "try the fast one and fall back": a silent fallback makes two
#: machines run two different programs under one command, which is how a
#: benchmark ends up comparing kernels instead of models. A backend that cannot
#: reach its fast path says so.
KNOBS: dict[str, tuple[str, ...]] = {
    "dtype": ("float32", "bfloat16"),
    "matmul_precision": ("highest", "high"),
    "triangle_kernel": ("auto", "cueq", "xla"),
    "attention_kernel": ("auto", "cueq", "tokamax", "xla"),
}

#: Old spellings, kept working. The new name is the documented one; these are
#: what the scripts in this repository, `EXPERIMENT_LOG.md`'s reproduction
#: commands and anyone's shell history already say.
ALIASES: dict[str, str] = {
    "compute_dtype": "dtype",
    "trunk_dtype": "dtype",
    "triangle_backend": "triangle_kernel",
    "trunk_triangle_attention_backend": "triangle_kernel",
    "attention_backend": "attention_kernel",
    "trunk_single_attention_backend": "attention_kernel",
}


#: Diffusion samples a rollout denoises at once, once it chunks at all.
#:
#: One constant, because the alternative is what this module exists to stop:
#: Protenix resolved it as `PROTENIX_SAMPLE_DIFFUSION_CHUNK_SIZE`, and the
#: ports that grew the same knob later would each have named their own. The
#: knob itself is `diffusion_chunk_size` everywhere, spelled the same in the
#: CLI, the Python API and each port's own signature.
#:
#: Five is the released sample count of every schedule here, so the shipped run
#: takes the single unchunked rollout it always took and only a caller who
#: raises the count meets the loop. That is the whole reason the rule is
#: `> chunk` rather than `>= chunk`.
DIFFUSION_CHUNK_SIZE = 5


def auto_diffusion_chunk_size(num_samples: int) -> int | None:
    """The chunk width for `num_samples`, or None to denoise them together."""
    return DIFFUSION_CHUNK_SIZE if num_samples > DIFFUSION_CHUNK_SIZE else None


class Alias(UserWarning):
    """An option was named the way it used to be named."""


def normalize(options: dict[str, Any], *, native: set[str]) -> dict[str, Any]:
    """Rewrite alias keys to their neutral names, warning once each.

    `native` is the set of names this backend still takes directly, so a port
    whose own option happens to share a name with an alias -- boltz2's
    `compute_dtype` is also its native spelling -- is not rewritten out from
    under itself when the caller meant the native one. The two agree on value
    either way; the distinction matters for the warning, not the result.
    """
    out: dict[str, Any] = {}
    for key, value in options.items():
        neutral = ALIASES.get(key)
        if neutral is None or key in native:
            out[key] = value
            continue
        if neutral in options:
            raise ValueError(
                f"{key!r} and {neutral!r} were both set; {key!r} is the old "
                f"name for {neutral!r}, so pass one of them"
            )
        warnings.warn(
            f"{key!r} is the old name for {neutral!r} and still works; the "
            "neutral name is the documented one and means the same thing on "
            "every model",
            Alias,
            stacklevel=3,
        )
        out[neutral] = value
    return out


def translate(
    options: dict[str, Any],
    table: dict[str, tuple[str, dict[str, str]]],
    *,
    model: str,
) -> dict[str, Any]:
    """Turn neutral knobs into one backend's own names and values.

    `table` maps a neutral knob to `(native name, {neutral value: native
    value})`. A knob this model cannot express is an error rather than a
    silently dropped argument -- the same rule `apply_sampling` uses, and for
    the same reason: a request that quietly did something else is worse than
    one that failed.
    """
    out = dict(options)
    for knob, value in list(options.items()):
        if knob not in KNOBS:
            continue
        entry = table.get(knob)
        if entry is None:
            raise ValueError(f"{model} does not support {knob}")
        native_name, values = entry
        allowed = KNOBS[knob]
        if value not in allowed:
            raise ValueError(
                f"{knob} must be one of {allowed}, got {value!r}"
            )
        native_value = values.get(value)
        if native_value is None:
            raise ValueError(
                f"{model} does not support {knob}={value!r}; it takes "
                f"{tuple(values)}"
            )
        del out[knob]
        if native_name in out:
            raise ValueError(
                f"{knob} and the native option {native_name!r} were both set "
                f"for {model}; pass one of them"
            )
        out[native_name] = native_value
    return out


#: The float32 matmul precision a caller asked for, or `None` for "whatever
#: this model pins".
#:
#: `matmul_precision` is the one neutral knob that cannot be delivered by
#: translating a name. Four of the six ports open their own
#: `jax.default_matmul_precision` scope inside the call, so a scope the adapter
#: opened around it is simply overridden -- the knob would accept a value,
#: change nothing, and report success. The port that pins has to be the port
#: that reads.
#:
#: A ContextVar rather than a module global because two predictions can be in
#: flight in one process, and because the in-process native CLIs run several
#: frames below the adapter that set it.
_REQUESTED_MATMUL_PRECISION: contextvars.ContextVar[str | None] = (
    contextvars.ContextVar("foldjax_matmul_precision", default=None)
)


@contextmanager
def matmul_precision_scope(value: str | None) -> Iterator[None]:
    """Ask every port in this call to run float32 matmuls at `value`.

    `None` is not "default" -- it is "do not ask", and leaves each port on the
    precision it pins for itself. Those pins are upstream parity, measured and
    recorded next to each one, so they stay the default and this only overrides
    them when a caller says so.

    The JAX scope is opened here as well as read by the ports, so that the
    three backends which pin nothing get the setting too.
    """
    if value is None:
        yield
        return
    if value not in KNOBS["matmul_precision"]:
        raise ValueError(
            f"matmul_precision must be one of {KNOBS['matmul_precision']}, "
            f"got {value!r}"
        )
    import jax

    token = _REQUESTED_MATMUL_PRECISION.set(value)
    try:
        with jax.default_matmul_precision(value):
            yield
    finally:
        _REQUESTED_MATMUL_PRECISION.reset(token)


def resolved_matmul_precision(default: str) -> str:
    """What this port should pin: the caller's request, or its own default.

    Ports call this where they used to name a module constant. With no request
    in flight the constant still wins, so an unasked run is bit-for-bit the run
    it was before.
    """
    return _REQUESTED_MATMUL_PRECISION.get() or default
