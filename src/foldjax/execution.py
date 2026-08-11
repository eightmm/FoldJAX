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
"""

from __future__ import annotations

import warnings
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
