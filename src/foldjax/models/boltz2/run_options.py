"""Upstream `boltz predict` run options, checked without importing JAX.

Shared by the native `predict` and the `foldjax` adapter, so `foldjax plan`
refuses a malformed value with the same message the run would.
"""

from __future__ import annotations

import math


def validate_upstream_run_options(
    *,
    step_scale: object,
    subsample_msa: object,
    num_subsampled_msa: object,
    method: object,
    use_potentials: object,
) -> str | None:
    """Refuse a malformed upstream run option; return the lowercased method."""

    if (
        isinstance(step_scale, bool)
        or not isinstance(step_scale, (int, float))
        or not math.isfinite(float(step_scale))
        or float(step_scale) <= 0
    ):
        raise ValueError(f"step_scale must be a positive number; got {step_scale!r}")
    for name, value in (
        ("subsample_msa", subsample_msa),
        ("use_potentials", use_potentials),
    ):
        if not isinstance(value, bool):
            raise ValueError(f"{name} must be a boolean; got {value!r}")
    if (
        isinstance(num_subsampled_msa, bool)
        or not isinstance(num_subsampled_msa, int)
        or num_subsampled_msa < 1
    ):
        raise ValueError(
            f"num_subsampled_msa must be a positive integer; got {num_subsampled_msa!r}"
        )
    if method is None:
        return None
    from foldjax.models.boltz2.data import const

    if not isinstance(method, str) or method.lower() not in const.method_types_ids:
        raise ValueError(
            f"method {method!r} is not supported; choose one of "
            f"{sorted(const.method_types_ids)}"
        )
    return method.lower()
