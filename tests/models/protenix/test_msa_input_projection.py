from __future__ import annotations

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.protenix.models.primitives.primitives import LinearParams
from foldjax.models.protenix.models.trunk_blocks.msa import (
    _dense_msa_input_projection,
    _msa_input_projection,
)


def _assert_ieee_equal(actual: jax.Array, expected: jax.Array) -> None:
    actual_np = np.asarray(actual)
    expected_np = np.asarray(expected)
    assert actual_np.dtype == expected_np.dtype
    np.testing.assert_array_equal(np.isnan(actual_np), np.isnan(expected_np))
    np.testing.assert_array_equal(np.isposinf(actual_np), np.isposinf(expected_np))
    np.testing.assert_array_equal(np.isneginf(actual_np), np.isneginf(expected_np))
    finite = np.isfinite(actual_np) & np.isfinite(expected_np)
    item_dtype = np.dtype(f"u{actual_np.dtype.itemsize}")
    np.testing.assert_array_equal(
        actual_np[finite].view(item_dtype), expected_np[finite].view(item_dtype)
    )


@pytest.mark.parametrize("shape", [(7, 11), (2, 3, 5)])
def test_compact_bfloat16_projection_matches_dense_bits(shape: tuple[int, ...]) -> None:
    rng = np.random.default_rng(sum(shape))
    # Include the clipping and negative-index normalization of the historical
    # eye-table indexing, not just the valid production range.
    choices = np.asarray([-100, -33, -32, -1, 0, 1, 17, 31, 32, 100])
    msa = jnp.asarray(rng.choice(choices, size=shape), dtype=jnp.int32)
    has_deletion = jnp.asarray(rng.integers(0, 2, size=shape), dtype=jnp.bool_)
    deletion_value = jnp.asarray(
        rng.normal(size=shape).astype(np.float32), dtype=jnp.bfloat16
    )
    params = LinearParams(
        weight=jnp.asarray(
            rng.normal(size=(13, 34)).astype(np.float32), dtype=jnp.bfloat16
        ),
        bias=None,
    )

    with jax.numpy_dtype_promotion("strict"):
        expected = _dense_msa_input_projection(
            msa,
            has_deletion,
            deletion_value,
            params,
            activation_dtype=jnp.bfloat16,
        )
        actual = _msa_input_projection(
            msa,
            has_deletion,
            deletion_value,
            params,
            activation_dtype=jnp.bfloat16,
        )

    _assert_ieee_equal(actual, expected)


def test_compact_bfloat16_projection_preserves_nonfinite_and_signed_zero() -> None:
    values = np.asarray(
        [0.0, -0.0, 1.0, -1.0, np.nan, np.inf, -np.inf], dtype=np.float32
    )
    # Each output channel covers one combination of selected-category,
    # unselected-category, deletion-indicator, and deletion-value weights.
    weight_cases = np.asarray(
        list(itertools.product(range(values.size), repeat=4)), dtype=np.int16
    )
    selected, unselected, indicator_weight, value_weight = (
        values[weight_cases[:, index]] for index in range(4)
    )
    weight = np.broadcast_to(unselected[:, None], (weight_cases.shape[0], 34)).copy()
    weight[:, 3] = selected
    weight[:, 32] = indicator_weight
    weight[:, 33] = value_weight

    # The two dynamic deletion inputs span the same seven IEEE edge values,
    # yielding 117,649 dense-vs-compact output comparisons in one program.
    deletion_cases = np.asarray(
        list(itertools.product(range(values.size), repeat=2)), dtype=np.int16
    )
    shape = (values.size, values.size)
    has_deletion = jnp.asarray(
        values[deletion_cases[:, 0]].reshape(shape), jnp.bfloat16
    )
    deletion_value = jnp.asarray(
        values[deletion_cases[:, 1]].reshape(shape), jnp.bfloat16
    )
    msa = jnp.full(shape, 3, dtype=jnp.int32)
    params = LinearParams(jnp.asarray(weight, jnp.bfloat16), None)

    expected = jax.jit(
        lambda a, b, c, w: _dense_msa_input_projection(
            a,
            b,
            c,
            LinearParams(w, None),
            activation_dtype=jnp.bfloat16,
        )
    )(msa, has_deletion, deletion_value, params.weight)
    actual = jax.jit(
        lambda a, b, c, w: _msa_input_projection(
            a,
            b,
            c,
            LinearParams(w, None),
            activation_dtype=jnp.bfloat16,
        )
    )(msa, has_deletion, deletion_value, params.weight)

    _assert_ieee_equal(actual, expected)


@pytest.mark.parametrize(
    ("activation_dtype", "feature_dtype", "weight_dtype", "with_bias"),
    [
        (jnp.float32, jnp.float32, jnp.float32, False),
        (jnp.bfloat16, jnp.float32, jnp.bfloat16, False),
        (jnp.bfloat16, jnp.bfloat16, jnp.bfloat16, True),
    ],
)
def test_generic_projection_contract_keeps_the_dense_dot(
    activation_dtype: jnp.dtype,
    feature_dtype: jnp.dtype,
    weight_dtype: jnp.dtype,
    with_bias: bool,
) -> None:
    msa = jnp.arange(15, dtype=jnp.int32).reshape(3, 5)
    has_deletion = jnp.zeros((3, 5), dtype=feature_dtype)
    deletion_value = jnp.ones((3, 5), dtype=feature_dtype)
    weight = jnp.arange(7 * 34, dtype=weight_dtype).reshape(7, 34) / 100
    bias = jnp.zeros((7,), dtype=weight_dtype) if with_bias else None

    fn = jax.jit(
        lambda a, b, c, w, d: _msa_input_projection(
            a,
            b,
            c,
            LinearParams(w, d),
            activation_dtype=activation_dtype,
        )
    )
    lowered = fn.lower(msa, has_deletion, deletion_value, weight, bias)
    stablehlo = str(lowered.compiler_ir(dialect="stablehlo"))

    assert stablehlo.count("stablehlo.dot_general") == 1


def test_compact_projection_removes_wide_dot_and_temp_arena() -> None:
    if jax.default_backend() != "cpu":
        pytest.skip("CPU executable memory accounting regression")

    shape = (256, 128)
    msa = jnp.zeros(shape, dtype=jnp.int32)
    has_deletion = jnp.zeros(shape, dtype=jnp.bool_)
    deletion_value = jnp.zeros(shape, dtype=jnp.bfloat16)
    weight = jnp.zeros((64, 34), dtype=jnp.bfloat16)

    dense = jax.jit(
        lambda a, b, c, w: _dense_msa_input_projection(
            a,
            b,
            c,
            LinearParams(w, None),
            activation_dtype=jnp.bfloat16,
        )
    ).lower(msa, has_deletion, deletion_value, weight)
    compact = jax.jit(
        lambda a, b, c, w: _msa_input_projection(
            a,
            b,
            c,
            LinearParams(w, None),
            activation_dtype=jnp.bfloat16,
        )
    ).lower(msa, has_deletion, deletion_value, weight)

    dense_hlo = str(dense.compiler_ir(dialect="stablehlo"))
    compact_hlo = str(compact.compiler_ir(dialect="stablehlo"))
    assert dense_hlo.count("stablehlo.dot_general") == 1
    assert compact_hlo.count("stablehlo.dot_general") == 0
    assert compact_hlo.count("stablehlo.gather") >= 1

    dense_memory = dense.compile().memory_analysis()
    compact_memory = compact.compile().memory_analysis()
    assert dense_memory.output_size_in_bytes == compact_memory.output_size_in_bytes
    assert compact_memory.temp_size_in_bytes * 8 < dense_memory.temp_size_in_bytes
