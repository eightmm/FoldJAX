"""Exactness and compiler gates for compact Protenix/OpenDDE distance bins."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.protenix.models.heads.confidence import (
    ConfidenceDistanceEmbeddingParams,
    ConfidenceHeadParams,
    ConfidenceOutputParams,
    _compact_confidence_bin_projection,
    can_compact_confidence_distance_embedding,
    confidence_distance_embedding,
    confidence_head_single_sample,
    confidence_one_hot,
)
from foldjax.models.protenix.models.primitives.primitives import (
    LayerNormParams,
    LinearParams,
    linear,
)
from foldjax.models.protenix.models.trunk_blocks.pairformer import (
    PairformerStackParams,
)
from tests.models.cp_probe_env import inherited_environment


def _params(
    *,
    dtype: jnp.dtype = jnp.float32,
    bins: int = 39,
    channels: int = 128,
    seed: int = 20260824,
) -> ConfidenceDistanceEmbeddingParams:
    rng = np.random.default_rng(seed)
    lower = jnp.arange(bins, dtype=dtype)
    upper = jnp.concatenate(
        [lower[1:], jnp.asarray([float(bins)], dtype=dtype)]
    )
    return ConfidenceDistanceEmbeddingParams(
        lower_bins=lower,
        upper_bins=upper,
        linear_d=LinearParams(
            weight=jnp.asarray(rng.normal(size=(channels, bins)), dtype=dtype),
            bias=jnp.asarray(rng.normal(size=(channels,)), dtype=dtype),
        ),
        linear_d_wo_onehot=LinearParams(
            weight=jnp.asarray(rng.normal(size=(channels, 1)), dtype=dtype),
            bias=None,
        ),
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
def test_compact_embedding_is_bitwise_equal_for_finite_batched_inputs(dtype) -> None:
    rng = np.random.default_rng(1989)
    coordinates = jnp.asarray(rng.normal(size=(2, 33, 3)) * 10.0, dtype=dtype)
    params = _params(dtype=dtype)

    dense = jax.jit(lambda value: confidence_distance_embedding(value, params))(
        coordinates
    )
    compact = jax.jit(
        lambda value: confidence_distance_embedding(
            value, params, compact_bins=True
        )
    )(coordinates)

    np.testing.assert_array_equal(_bits(np.asarray(compact)), _bits(np.asarray(dense)))
    assert can_compact_confidence_distance_embedding(params)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_strict_open_edges_and_signed_zero_match_dense_projection(dtype) -> None:
    lower = jnp.asarray([0.0, 1.0, 2.0], dtype=dtype)
    upper = jnp.asarray([1.0, 2.0, 4.0], dtype=dtype)
    weight = jnp.asarray(
        [[-0.0, 2.0, 3.0], [1.0, -0.0, -4.0]], dtype=dtype
    )
    params = ConfidenceDistanceEmbeddingParams(
        lower_bins=lower,
        upper_bins=upper,
        linear_d=LinearParams(weight=weight, bias=None),
        linear_d_wo_onehot=LinearParams(
            weight=jnp.zeros((2, 1), dtype=dtype), bias=None
        ),
    )
    distances = jnp.asarray(
        [-jnp.inf, -0.0, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, jnp.inf, jnp.nan],
        dtype=dtype,
    )

    expected = linear(confidence_one_hot(distances, lower, upper), params.linear_d)
    actual = _compact_confidence_bin_projection(distances, params)
    _assert_same_nonfinite_semantics(actual, expected)
    actual_host = np.asarray(actual)
    for edge in (0, 1, 2, 4, 6, 8, 9, 10, 11):
        assert not np.any(_bits(actual_host[edge]))
    # Dense multi-term dot turns an active -0 weight into positive zero.
    assert not np.signbit(actual_host[3, 0])
    assert not np.signbit(actual_host[5, 1])


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_nonfinite_coordinates_keep_dense_ieee_classes_and_finite_bits(dtype) -> None:
    coordinates = jnp.asarray(
        [
            [0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [jnp.inf, 0.0, 0.0],
            [-jnp.inf, 0.0, 0.0],
            [jnp.nan, 0.0, 0.0],
        ],
        dtype=dtype,
    )
    params = _params(dtype=dtype, channels=8)

    dense = jax.jit(lambda value: confidence_distance_embedding(value, params))(
        coordinates
    )
    compact = jax.jit(
        lambda value: confidence_distance_embedding(
            value, params, compact_bins=True
        )
    )(coordinates)
    _assert_same_nonfinite_semantics(compact, dense)


def test_nonfinite_projection_parameters_force_dense_default() -> None:
    coordinates = jnp.asarray(
        [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [jnp.nan, 0.0, 0.0]],
        dtype=jnp.float32,
    )
    base = _params(bins=3, channels=4)
    weight = np.asarray(base.linear_d.weight).copy()
    weight[0, 0] = np.nan
    weight[1, 1] = np.inf
    weight[2, 2] = -np.inf
    params = base._replace(
        linear_d=base.linear_d._replace(weight=jnp.asarray(weight))
    )

    assert not can_compact_confidence_distance_embedding(params)
    coords = coordinates.astype(jnp.float32)
    distance = jnp.sqrt(
        jnp.sum(
            jnp.square(coords[..., :, None, :] - coords[..., None, :, :]),
            axis=-1,
        )
    )
    expected = linear(
        confidence_one_hot(distance, params.lower_bins, params.upper_bins),
        params.linear_d,
    ) + linear(distance[..., None], params.linear_d_wo_onehot)
    # The public/direct call deliberately stays on this dense IEEE path.
    actual = confidence_distance_embedding(coordinates, params)
    _assert_same_nonfinite_semantics(actual, expected)

    bad_bias = base._replace(
        linear_d=base.linear_d._replace(
            bias=base.linear_d.bias.at[0].set(jnp.nan)
        )
    )
    assert not can_compact_confidence_distance_embedding(bad_bias)


def test_single_sample_confidence_head_threads_compact_projection_bitwise() -> None:
    rng = np.random.default_rng(17)
    tokens, single_channels, pair_channels = 7, 5, 8

    def arr(*shape):
        return jnp.asarray(rng.normal(size=shape), dtype=jnp.float32)

    def norm(channels):
        return LayerNormParams(
            weight=jnp.ones((channels,), dtype=jnp.float32),
            bias=jnp.zeros((channels,), dtype=jnp.float32),
        )

    distance = _params(bins=5, channels=pair_channels)
    head_params = ConfidenceHeadParams(
        input_strunk_ln=norm(single_channels),
        linear_s1=LinearParams(weight=arr(pair_channels, single_channels)),
        linear_s2=LinearParams(weight=arr(pair_channels, single_channels)),
        distance_embedding=distance,
        pairformer_stack=PairformerStackParams(blocks=()),
        output=ConfidenceOutputParams(
            pae_ln=norm(pair_channels),
            pde_ln=norm(pair_channels),
            plddt_ln=norm(single_channels),
            resolved_ln=norm(single_channels),
            linear_pae=LinearParams(weight=arr(4, pair_channels)),
            linear_pde=LinearParams(weight=arr(4, pair_channels)),
            plddt_weight=arr(1, single_channels, 4),
            resolved_weight=arr(1, single_channels, 2),
        ),
    )
    inputs = (
        arr(tokens, single_channels),
        arr(tokens, single_channels),
        arr(tokens, tokens, pair_channels),
        None,
        arr(tokens, 3),
        jnp.arange(tokens, dtype=jnp.int32),
        jnp.zeros((tokens,), dtype=jnp.int32),
        head_params,
    )

    dense = jax.jit(lambda *values: confidence_head_single_sample(*values))(*inputs)
    compact = jax.jit(
        lambda *values: confidence_head_single_sample(
            *values, compact_distance_bins=True
        )
    )(*inputs)
    for name in dense:
        np.testing.assert_array_equal(
            _bits(np.asarray(compact[name])), _bits(np.asarray(dense[name]))
        )


@pytest.mark.parametrize(
    ("lower", "upper"),
    [
        ([0.0], [1.0]),
        ([0.0, 2.0, 1.0], [2.0, 1.0, 3.0]),
        ([0.0, 1.0, 2.0], [0.5, 2.0, 3.0]),
        ([0.0, 1.0, 2.0], [1.5, 2.0, 3.0]),
        ([0.0, 1.0, np.nan], [1.0, np.nan, 3.0]),
    ],
)
def test_host_gate_rejects_noncanonical_interval_layouts(lower, upper) -> None:
    bins = len(lower)
    params = _params(bins=bins)._replace(
        lower_bins=jnp.asarray(lower, dtype=jnp.float32),
        upper_bins=jnp.asarray(upper, dtype=jnp.float32),
    )
    assert not can_compact_confidence_distance_embedding(params)


def test_host_gate_rejects_wrong_projection_layout() -> None:
    params = _params(bins=4, channels=3)
    wrong = params._replace(
        linear_d=params.linear_d._replace(weight=jnp.ones((4, 3)))
    )
    assert not can_compact_confidence_distance_embedding(wrong)


@partial(jax.jit, static_argnames=("compact_bins",))
def _cached_embedding(coordinates, params, *, compact_bins):
    return confidence_distance_embedding(
        coordinates, params, compact_bins=compact_bins
    )


def _lowered_pair(n_token: int):
    coordinates = jax.ShapeDtypeStruct((n_token, 3), jnp.float32)
    params = _params()
    return (
        _cached_embedding.lower(coordinates, params, compact_bins=False),
        _cached_embedding.lower(coordinates, params, compact_bins=True),
    )


def test_static_compact_flag_changes_cache_identity_and_removes_dense_dot() -> None:
    _cached_embedding.clear_cache()
    coordinates = jnp.zeros((8, 3), dtype=jnp.float32)
    params = _params()
    _cached_embedding(coordinates, params, compact_bins=False).block_until_ready()
    _cached_embedding(coordinates, params, compact_bins=True).block_until_ready()
    assert _cached_embedding._cache_size() == 2

    dense, compact = _lowered_pair(32)
    dense_hlo = str(dense.compiler_ir())
    compact_hlo = str(compact.compiler_ir())
    # The unchanged scalar-distance projection is the one remaining dot; the
    # compact graph removes only the wide one-hot projection.
    assert dense_hlo.count("stablehlo.dot_general") == 2
    assert compact_hlo.count("stablehlo.dot_general") == 1


def test_compact_projection_reduces_compiled_cpu_temporary_memory() -> None:
    dense, compact = _lowered_pair(128)
    dense_temp = dense.compile().memory_analysis().temp_size_in_bytes
    compact_temp = compact.compile().memory_analysis().temp_size_in_bytes
    assert dense_temp > 4 * compact_temp, (dense_temp, compact_temp)


_CP_PROBE = textwrap.dedent(
    r"""
    import os

    import jax
    import jax.numpy as jnp
    import numpy as np

    from foldjax.models._cp import context_parallel, shard_pair_rows
    from foldjax.models.protenix.models.heads.confidence import (
        ConfidenceDistanceEmbeddingParams,
        confidence_distance_embedding,
    )
    from foldjax.models.protenix.models.primitives.primitives import LinearParams

    assert jax.device_count() == 4, jax.devices()
    layout = os.environ["FOLDJAX_CP_PROBE_LAYOUT"]
    tokens, bins, channels = 16, 7, 8
    rng = np.random.default_rng(20260824)

    lower = jnp.arange(bins, dtype=jnp.float32)
    params = ConfidenceDistanceEmbeddingParams(
        lower_bins=lower,
        upper_bins=jnp.concatenate([lower[1:], jnp.asarray([float(bins)])]),
        linear_d=LinearParams(
            weight=jnp.asarray(rng.normal(size=(channels, bins)), jnp.float32),
            bias=jnp.asarray(rng.normal(size=(channels,)), jnp.float32),
        ),
        linear_d_wo_onehot=LinearParams(
            weight=jnp.asarray(rng.normal(size=(channels, 1)), jnp.float32),
            bias=None,
        ),
    )
    coordinates = jnp.asarray(rng.normal(size=(tokens, 3)), jnp.float32)

    def build(compact):
        def run(value):
            return shard_pair_rows(
                confidence_distance_embedding(
                    value, params, compact_bins=compact
                )
            )
        return run

    serial_dense = np.asarray(jax.jit(build(False))(coordinates))
    serial_compact = np.asarray(jax.jit(build(True))(coordinates))
    np.testing.assert_array_equal(serial_compact, serial_dense)

    jax.clear_caches()
    with context_parallel(4, layout=layout):
        dense_executable = jax.jit(build(False)).lower(coordinates).compile()
        compact_executable = jax.jit(build(True)).lower(coordinates).compile()
        cp_dense = np.asarray(dense_executable(coordinates))
        cp_compact = np.asarray(compact_executable(coordinates))
        dense_hlo = dense_executable.as_text().lower()
        compact_hlo = compact_executable.as_text().lower()

    np.testing.assert_array_equal(cp_dense, serial_dense)
    np.testing.assert_array_equal(cp_compact, serial_dense)
    for marker in (
        " all-gather(",
        " all-reduce(",
        " collective-permute(",
        " reduce-scatter(",
    ):
        assert dense_hlo.count(marker) == compact_hlo.count(marker), (
            marker,
            dense_hlo.count(marker),
            compact_hlo.count(marker),
        )
    print("CONFIDENCE_DISTANCE_CP_OK")
    """
)


@pytest.mark.parametrize("layout", ["1d", "2d"])
def test_compact_projection_preserves_cp_parity_and_collectives(layout: str) -> None:
    completed = subprocess.run(
        [sys.executable, "-c", _CP_PROBE],
        capture_output=True,
        text=True,
        env={
            "JAX_PLATFORMS": "cpu",
            "XLA_FLAGS": "--xla_force_host_platform_device_count=4",
            "FOLDJAX_CP_PROBE_LAYOUT": layout,
            **inherited_environment(),
        },
        timeout=180,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "CONFIDENCE_DISTANCE_CP_OK" in completed.stdout
