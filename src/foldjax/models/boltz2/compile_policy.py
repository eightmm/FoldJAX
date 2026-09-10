"""Boltz-only preservation of publisher mixed-precision rounding boundaries.

XLA's default excess-precision pass may erase FP32 -> BF16 -> FP32
conversions. Native autocast materializes that rounding, including before
sparse lookup sums and FP32 bias additions. Disable this optimization on the
outer low-precision executable, without globally changing XLA or preventing
fusion with optimization barriers. This is not a promise of IEEE operation
ordering or upstream parity for every kernel.

A run that asks for repeatable reduction orders composes with this rather
than replacing it: the shared constant in ``foldjax.models._compile_policy``
is merged on top of the dtype policy, so the BF16 rounding boundary survives.

Callers composing the pure ``boltz2_predict`` graph must apply
``compiler_options(compute_dtype)`` to their own outer ``jax.jit``. An eager
steering execution has no outer executable and is outside this policy, and so
must refuse ``deterministic`` rather than run without it.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from foldjax.models._compile_policy import compiler_options as _shared_options


def compiler_options(
    compute_dtype: Any, *, deterministic: bool = False
) -> dict[str, bool]:
    """Return a fresh policy mapping; retain the historical FP32 defaults.

    An empty mapping rather than the shared helper's ``None``: this port's
    callers report the answer (``dict(compile_options)``) and key their
    retained graph owners on it, and ``jit`` below is what turns "nothing
    asked for" back into a ``jax.jit`` call with no ``compiler_options`` at
    all -- which is the compile every recorded Boltz-2 measurement describes.
    """
    name = str(
        getattr(
            compute_dtype, "name", getattr(compute_dtype, "__name__", compute_dtype)
        )
    )
    if name == "float32":
        base: dict[str, bool] = {}
    elif name == "bfloat16":
        base = {"xla_allow_excess_precision": False}
    else:
        raise ValueError(f"unsupported Boltz compute dtype: {name!r}")
    return _shared_options(deterministic=deterministic, base=base) or {}


def jit(
    function: Callable[..., Any],
    *,
    compute_dtype: Any,
    deterministic: bool = False,
) -> Any:
    """Compile only this Boltz graph under its resolved dtype policy."""
    import jax

    options = compiler_options(compute_dtype, deterministic=deterministic)
    return jax.jit(function, compiler_options=options) if options else jax.jit(function)
