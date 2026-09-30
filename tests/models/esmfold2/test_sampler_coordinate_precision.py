"""ESMFold2's sampler carries coordinates through full-FP32 products only.

Upstream runs the augmentation rotation and the Kabsch align at FP32 (torch
leaves `allow_tf32` off; the align is under `autocast(enabled=False)`). This
port pins no global matmul precision, and JAX's GPU default rounds FP32 dot
operands to TF32, which made bond lengths noisier with distance from the
origin. A TF32 scope stands in for that default here, since the CPU ignores it.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from foldjax.models.esmfold2.models.diffusion import (
    center_random_augmentation,
    weighted_rigid_align,
)


def _dot_precisions(function, *args) -> list[object]:
    with jax.default_matmul_precision("tensorfloat32"):
        jaxpr = jax.make_jaxpr(function)(*args)
    return [
        eqn.params["precision"]
        for eqn in jaxpr.jaxpr.eqns
        if eqn.primitive.name == "dot_general"
    ]


def _is_highest(precision) -> bool:
    highest = jax.lax.Precision.HIGHEST
    return precision == highest or precision == (highest, highest)


def test_augmentation_and_align_products_run_at_full_fp32() -> None:
    x = jax.random.normal(jax.random.PRNGKey(0), (2, 64, 3)) * 40.0
    mask = jnp.ones((2, 64), dtype=jnp.float32)

    augment = _dot_precisions(
        lambda key, a, b, m: center_random_augmentation(key, a, m, b),
        jax.random.PRNGKey(1),
        x,
        x + 0.1,
        mask,
    )
    align = _dot_precisions(weighted_rigid_align, x, x + 0.1, mask, mask)

    assert len(augment) == 2 and align
    assert all(_is_highest(p) for p in augment + align), augment + align
