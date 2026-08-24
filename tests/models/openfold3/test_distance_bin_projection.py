"""Exact and compiler gates for compact confidence distance projection."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.openfold3.models.heads import _project_distance_bins
from foldjax.models.openfold3.models.primitives import LinearParams, linear

MIN_BIN = 3.25
MAX_BIN = 50.75
NO_BIN = 39
INF = 1e9
OUT_CHANNELS = 128


def _squared_distance(coordinates: jax.Array) -> jax.Array:
    return jnp.sum(
        (coordinates[..., :, None, :] - coordinates[..., None, :, :]) ** 2,
        axis=-1,
    )


def _dense_legacy_projection(
    coordinates: jax.Array,
    weight: jax.Array,
    bias: jax.Array | None = None,
) -> jax.Array:
    """The historical one-hot plus linear, retained only as a test oracle."""

    return _dense_squared_projection(
        _squared_distance(coordinates), weight, bias, dtype=coordinates.dtype
    )


def _dense_squared_projection(
    squared_distance: jax.Array,
    weight: jax.Array,
    bias: jax.Array | None = None,
    *,
    dtype: jnp.dtype,
) -> jax.Array:
    bins = jnp.linspace(MIN_BIN, MAX_BIN, NO_BIN, dtype=dtype)
    squared = bins**2
    upper = jnp.concatenate([squared[1:], jnp.asarray([INF], dtype=dtype)])
    one_hot = (
        (squared_distance[..., None] > squared)
        & (squared_distance[..., None] < upper)
    ).astype(dtype)
    return linear(one_hot, LinearParams(weight=weight, bias=bias))


def _indexed_projection(
    coordinates: jax.Array,
    weight: jax.Array,
    bias: jax.Array | None = None,
) -> jax.Array:
    return _indexed_squared_projection(
        _squared_distance(coordinates), weight, bias, dtype=coordinates.dtype
    )


def _indexed_squared_projection(
    squared_distance: jax.Array,
    weight: jax.Array,
    bias: jax.Array | None = None,
    *,
    dtype: jnp.dtype,
) -> jax.Array:
    return _project_distance_bins(
        squared_distance,
        LinearParams(weight=weight, bias=bias),
        dtype=dtype,
        min_bin=MIN_BIN,
        max_bin=MAX_BIN,
        no_bin=NO_BIN,
        inf=INF,
    )


def _bits(value: np.ndarray) -> np.ndarray:
    unsigned = {2: np.uint16, 4: np.uint32, 8: np.uint64}[value.dtype.itemsize]
    return np.ascontiguousarray(value).view(unsigned)


def _assert_same_nonfinite_semantics(actual: jax.Array, expected: jax.Array) -> None:
    actual_host = np.asarray(actual)
    expected_host = np.asarray(expected)
    np.testing.assert_array_equal(np.isnan(actual_host), np.isnan(expected_host))
    np.testing.assert_array_equal(np.isposinf(actual_host), np.isposinf(expected_host))
    np.testing.assert_array_equal(np.isneginf(actual_host), np.isneginf(expected_host))
    finite = np.isfinite(expected_host)
    np.testing.assert_array_equal(np.isfinite(actual_host), finite)
    np.testing.assert_array_equal(
        _bits(actual_host[finite]), _bits(expected_host[finite])
    )


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_indexed_projection_is_bitwise_equal_for_finite_values(dtype) -> None:
    rng = np.random.default_rng(20260824)
    coordinates = jnp.asarray(rng.normal(size=(2, 33, 3)) * 20.0, dtype=dtype)
    weight = jnp.asarray(
        rng.normal(size=(OUT_CHANNELS, NO_BIN)), dtype=dtype
    )
    bias = jnp.asarray(rng.normal(size=(OUT_CHANNELS,)), dtype=dtype)

    dense = jax.jit(_dense_legacy_projection)(coordinates, weight, bias)
    indexed = jax.jit(_indexed_projection)(coordinates, weight, bias)
    np.testing.assert_array_equal(_bits(np.asarray(indexed)), _bits(np.asarray(dense)))


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_exact_bin_edges_select_no_weight_row(dtype) -> None:
    weight = jnp.ones((4, NO_BIN), dtype=dtype)

    dense = jax.jit(
        lambda weights: _dense_squared_projection(
            jnp.linspace(MIN_BIN, MAX_BIN, NO_BIN, dtype=dtype) ** 2,
            weights,
            dtype=dtype,
        )
    )(weight)
    indexed = jax.jit(
        lambda weights: _indexed_squared_projection(
            jnp.linspace(MIN_BIN, MAX_BIN, NO_BIN, dtype=dtype) ** 2,
            weights,
            dtype=dtype,
        )
    )(weight)
    np.testing.assert_array_equal(_bits(np.asarray(indexed)), _bits(np.asarray(dense)))
    assert not np.any(_bits(np.asarray(indexed)))


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_nonfinite_coordinates_and_weights_keep_dense_dot_semantics(dtype) -> None:
    coordinates = jnp.asarray(
        [
            [
                [0.0, 0.0, 0.0],
                [4.0, 0.0, 0.0],
                [8.0, 0.0, 0.0],
                [MAX_BIN + 1.0, 0.0, 0.0],
                [40000.0, 0.0, 0.0],
                [jnp.inf, 0.0, 0.0],
                [jnp.nan, 0.0, 0.0],
            ]
        ],
        dtype=dtype,
    )
    rng = np.random.default_rng(1977)
    dense = jax.jit(_dense_legacy_projection)
    indexed = jax.jit(_indexed_projection)

    weights = []
    base = rng.normal(size=(8, NO_BIN)).astype(np.float32)
    weights.append(base)
    negative_zero = np.full_like(base, -0.0)
    weights.append(negative_zero)
    for _ in range(30):
        weight = base.copy()
        replace = rng.random(weight.shape) < 0.15
        weight[replace] = rng.choice(
            np.asarray([np.nan, np.inf, -np.inf, -0.0], dtype=np.float32),
            int(replace.sum()),
        )
        weights.append(weight)

    for weight in weights:
        weight_array = jnp.asarray(weight, dtype=dtype)
        expected = dense(coordinates, weight_array)
        actual = indexed(coordinates, weight_array)
        _assert_same_nonfinite_semantics(actual, expected)

    # The all-negative-zero case is a signed-zero regression, not a vacuous
    # ordinary equality check: dense dot accumulation yields positive zero.
    negative_zero_result = np.asarray(
        indexed(coordinates, jnp.asarray(negative_zero, dtype=dtype))
    )
    assert not np.any(np.signbit(negative_zero_result))


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_single_bin_layout_keeps_dense_signed_zero_semantics(dtype) -> None:
    squared_distance = jnp.asarray(
        [-0.0, MIN_BIN**2, jnp.nan, jnp.inf], dtype=dtype
    )
    weight = jnp.asarray([[-0.0], [-1.0], [0.0], [1.0]], dtype=dtype)
    params = LinearParams(weight=weight, bias=None)
    bins = jnp.linspace(MIN_BIN, MAX_BIN, 1, dtype=dtype)
    squared = bins**2
    upper = jnp.asarray([INF], dtype=dtype)
    one_hot = (
        (squared_distance[..., None] > squared)
        & (squared_distance[..., None] < upper)
    ).astype(dtype)

    expected = jax.jit(linear)(one_hot, params)
    actual = jax.jit(
        lambda distance, projection: _project_distance_bins(
            distance,
            projection,
            dtype=dtype,
            min_bin=MIN_BIN,
            max_bin=MAX_BIN,
            no_bin=1,
            inf=INF,
        )
    )(squared_distance, params)
    np.testing.assert_array_equal(
        _bits(np.asarray(actual)), _bits(np.asarray(expected))
    )


def _lowered_pair(n_token: int):
    coordinates = jax.ShapeDtypeStruct((1, n_token, 3), jnp.float32)
    weight = jax.ShapeDtypeStruct((OUT_CHANNELS, NO_BIN), jnp.float32)
    return (
        jax.jit(_dense_legacy_projection).lower(coordinates, weight),
        jax.jit(_indexed_projection).lower(coordinates, weight),
    )


def test_indexed_projection_removes_the_39_wide_dot_from_stablehlo() -> None:
    dense, indexed = _lowered_pair(32)
    dense_hlo = str(dense.compiler_ir())
    indexed_hlo = str(indexed.compiler_ir())
    assert "stablehlo.dot_general" in dense_hlo
    assert "stablehlo.dot_general" not in indexed_hlo


def test_indexed_projection_reduces_compiled_cpu_temporary_memory() -> None:
    dense, indexed = _lowered_pair(128)
    dense_temp = dense.compile().memory_analysis().temp_size_in_bytes
    indexed_temp = indexed.compile().memory_analysis().temp_size_in_bytes
    assert dense_temp > 4 * indexed_temp, (dense_temp, indexed_temp)
