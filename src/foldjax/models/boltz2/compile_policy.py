"""Boltz-only preservation of publisher mixed-precision rounding boundaries.

XLA's default excess-precision pass may erase FP32 -> BF16 -> FP32
conversions. Native autocast materializes that rounding, including before
sparse lookup sums and FP32 bias additions. Disable this optimization on the
outer low-precision executable, without globally changing XLA or preventing
fusion with optimization barriers. This is not a promise of IEEE operation
ordering or upstream parity for every kernel.

Callers composing the pure ``boltz2_predict`` graph must apply
``compiler_options(compute_dtype)`` to their own outer ``jax.jit``. An eager
steering execution has no outer executable and is outside this policy.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def compiler_options(compute_dtype: Any) -> dict[str, bool]:
    """Return a fresh policy mapping; retain the historical FP32 defaults."""
    name = str(
        getattr(
            compute_dtype, "name", getattr(compute_dtype, "__name__", compute_dtype)
        )
    )
    if name == "float32":
        return {}
    if name == "bfloat16":
        return {"xla_allow_excess_precision": False}
    raise ValueError(f"unsupported Boltz compute dtype: {name!r}")


def jit(function: Callable[..., Any], *, compute_dtype: Any) -> Any:
    """Compile only this Boltz graph under its resolved dtype policy."""
    import jax

    options = compiler_options(compute_dtype)
    return jax.jit(function, compiler_options=options) if options else jax.jit(function)
