"""The TF32 operand grid, and the promise that switching it off changes nothing.

The correction only works because rounding to the round-nearest-even grid is a
*fixed point* of the ties-away rounding JAX's ``high`` then applies: quantize
first and the second rounding has nothing left to do.  That is the property
worth testing, more than any particular value, because it is what separates a
correction from a second approximation stacked on the first.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.openfold3.models.primitives import LinearParams, linear
from foldjax.models.openfold3.models.tf32_rounding import (
    round_operands,
    round_to_tf32_rne,
    tf32_operand_rounding,
)

_STEP_BITS = 13
_GRID_MASK = np.uint32(0xFFFFE000)


def _bits(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float32).view(np.uint32)


def _on_grid(x) -> bool:
    return bool(((_bits(x) & np.uint32(0x1FFF)) == 0).all())


def test_results_land_on_the_grid_and_within_half_a_step() -> None:
    """Nearest, not merely coarser: no value moves more than half a step."""
    rng = np.random.default_rng(0)
    values = np.concatenate(
        [
            rng.normal(scale=10.0, size=4096).astype(np.float32),
            rng.normal(scale=1e-6, size=1024).astype(np.float32),
            rng.normal(scale=1e6, size=1024).astype(np.float32),
        ]
    )
    rounded = np.asarray(round_to_tf32_rne(jnp.asarray(values)))
    assert _on_grid(rounded)

    # Half a step is half the value of the lowest retained mantissa bit, which
    # scales with the exponent; comparing in float64 keeps that exact.
    step = np.ldexp(1.0, np.frexp(values.astype(np.float64))[1] - 11)
    assert np.all(np.abs(rounded.astype(np.float64) - values) <= step / 2 + 0.0)


def test_exact_ties_go_to_the_even_neighbour_in_both_signs() -> None:
    """The whole finding is the tie direction, so pin it explicitly."""
    # A tie is exactly half a step above a grid point: low 13 bits == 0x1000.
    base_even = np.uint32(0x3F800000)  # 1.0; retained mantissa even
    base_odd = base_even | np.uint32(1 << _STEP_BITS)  # next grid point, odd
    for base in (base_even, base_odd):
        for sign in (np.uint32(0), np.uint32(0x80000000)):
            tie = np.uint32((base | np.uint32(1 << (_STEP_BITS - 1))) | sign)
            value = tie.view(np.float32)
            got = _bits(round_to_tf32_rne(jnp.asarray(value)))
            # Ties-to-even: the retained mantissa's lowest bit must end at zero.
            assert (got >> _STEP_BITS) & 1 == 0, (hex(int(tie)), hex(int(got)))
            # And it must be one of the two neighbours, not somewhere else.
            assert int(got) in {
                int(np.uint32(tie) & _GRID_MASK),
                int((np.uint32(tie) & _GRID_MASK) + np.uint32(1 << _STEP_BITS)),
            }


def test_signed_zero_and_non_finite_values_survive_the_bit_arithmetic() -> None:
    """Adding half a step to an infinity's bits would manufacture a NaN."""
    values = np.array([0.0, -0.0, np.inf, -np.inf, np.nan], dtype=np.float32)
    rounded = np.asarray(round_to_tf32_rne(jnp.asarray(values)))
    # Bit comparison, so +0.0 and -0.0 are distinguished and the NaN payload is
    # checked rather than skipped by an equality that NaN never satisfies.
    np.testing.assert_array_equal(rounded.view(np.uint32), values.view(np.uint32))


def test_the_grid_is_a_fixed_point_of_ties_away_rounding() -> None:
    """Why prequantizing corrects instead of approximating twice.

    JAX's ``high`` rounds operands to the same grid with ties away from zero.
    An operand already on the grid is unchanged by it, so the port's rounding
    is the only one that decides the result.
    """
    rng = np.random.default_rng(1)
    values = rng.normal(scale=10.0, size=8192).astype(np.float32)
    rounded = np.asarray(round_to_tf32_rne(jnp.asarray(values)))

    ties_away = ((rounded.view(np.uint32) + np.uint32(0x1000)) & _GRID_MASK).view(
        np.float32
    )
    np.testing.assert_array_equal(ties_away.view(np.uint32), rounded.view(np.uint32))


def test_rounding_is_idempotent() -> None:
    rng = np.random.default_rng(2)
    values = jnp.asarray(rng.normal(size=2048).astype(np.float32))
    once = round_to_tf32_rne(values)
    np.testing.assert_array_equal(np.asarray(round_to_tf32_rne(once)), np.asarray(once))


@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float16, jnp.int32])
def test_only_float32_is_touched(dtype) -> None:
    """A BF16 activation is already coarser than this grid."""
    values = jnp.asarray(np.arange(8), dtype=dtype)
    assert round_to_tf32_rne(values) is values


def test_the_policy_is_off_by_default(monkeypatch) -> None:
    """An unasked run must be the run it was before this module existed.

    The correction is only correct on hardware that rounds to TF32; on CPU,
    where ``high`` is plain FP32, applying it would introduce the very error it
    removes. So the default has to be inert, not merely small.
    """
    monkeypatch.delenv("OPENFOLD3_TF32_ROUNDING", raising=False)
    assert tf32_operand_rounding() == "none"

    values = jnp.asarray(np.random.default_rng(3).normal(size=(4, 4)), jnp.float32)
    assert round_operands(values) == (values,)

    params = LinearParams(
        weight=jnp.asarray(np.random.default_rng(4).normal(size=(4, 4)), jnp.float32),
        bias=None,
    )
    np.testing.assert_array_equal(
        np.asarray(linear(values, params, match_native_tf32=True)),
        np.asarray(linear(values, params)),
    )


def test_the_policy_reaches_the_projection_when_asked(monkeypatch) -> None:
    """And when it is asked for, both operands are on the grid."""
    monkeypatch.setenv("OPENFOLD3_TF32_ROUNDING", "rne")
    assert tf32_operand_rounding() == "rne"

    rng = np.random.default_rng(5)
    values = jnp.asarray(rng.normal(size=(6, 4)), jnp.float32)
    weight = jnp.asarray(rng.normal(size=(3, 4)), jnp.float32)
    params = LinearParams(weight=weight, bias=None)

    matched = np.asarray(linear(values, params, match_native_tf32=True))
    expected = np.asarray(
        jnp.matmul(
            round_to_tf32_rne(values),
            jnp.swapaxes(round_to_tf32_rne(weight), -1, -2),
        )
    )
    np.testing.assert_array_equal(matched, expected)
    # The unasked projection stays on the untouched operands even now.
    assert not np.array_equal(matched, np.asarray(linear(values, params)))
