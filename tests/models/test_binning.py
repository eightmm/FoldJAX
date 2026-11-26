"""Parity and edge-case tests for shared distance and open-interval binning."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models._binning import distance_bins, open_interval_bin


@pytest.mark.parametrize(
    ("site", "boundaries"),
    [
        # Boltz-2 confidence head: num_dist_bins=64, min_dist=2.0, max_dist=22.0
        ("boltz2_confidence", jnp.linspace(2.0, 22.0, 64 - 1, dtype=jnp.float32)),
        # Boltz-2 template module: min_dist=3.25, max_dist=50.75, num_bins=38
        ("boltz2_template", jnp.linspace(3.25, 50.75, 38 - 1, dtype=jnp.float32)),
        # Boltz-2 affinity head: num_dist_bins=64, min_dist=2.0, max_dist=22.0
        ("boltz2_affinity", jnp.linspace(2.0, 22.0, 64 - 1, dtype=jnp.float32)),
        # Edge cases: single boundary and empty boundary array
        ("single_boundary", jnp.asarray([2.0], dtype=jnp.float32)),
        ("empty_boundaries", jnp.zeros((0,), dtype=jnp.float32)),
    ],
)
def test_distance_bins_matches_compare_and_sum_for_boltz2_boundaries(
    site: str,
    boundaries: jnp.ndarray,
) -> None:
    edges = np.asarray(boundaries, dtype=np.float32)
    rng = np.random.default_rng(0)
    parts = [
        rng.uniform(-10.0, 65.0, 4096).astype(np.float32),
        np.asarray(
            [-100.0, -1.0, -0.0, 0.0, np.inf, -np.inf, np.nan], dtype=np.float32
        ),
    ]
    if edges.size:
        parts.extend(
            [
                edges,
                np.nextafter(edges, np.float32(-np.inf)),
                np.nextafter(edges, np.float32(np.inf)),
            ]
        )
    distances = jnp.asarray(np.concatenate(parts), dtype=jnp.float32)

    expected = jnp.sum(distances[..., None] > boundaries, axis=-1)
    got = distance_bins(distances, boundaries)
    assert got.dtype == expected.dtype, site
    np.testing.assert_array_equal(got, expected)

    with jax.enable_x64(True):
        wide_expected = jnp.sum(distances[..., None] > boundaries, axis=-1)
        wide_got = distance_bins(distances, boundaries)
        assert wide_got.dtype == wide_expected.dtype, site
        np.testing.assert_array_equal(wide_got, wide_expected)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_open_interval_bin_matches_strict_dense_membership(dtype: jnp.dtype) -> None:
    lower = jnp.linspace(3.25, 50.75, 39, dtype=dtype) ** 2
    upper = jnp.concatenate([lower[1:], jnp.asarray([1e9], dtype=dtype)])
    edges = np.asarray(lower, dtype=np.float32)
    rng = np.random.default_rng(1)
    values = jnp.asarray(
        np.concatenate(
            [
                rng.uniform(-5.0, 3000.0, 512),
                edges,
                [float(upper[-1])],
                np.nextafter(edges, np.float32(-np.inf)),
                np.nextafter(edges, np.float32(np.inf)),
                [-np.inf, -1.0, -0.0, 0.0, np.inf, np.nan],
            ]
        ),
        dtype=dtype,
    )

    safe_index, valid = open_interval_bin(values, lower, upper)
    dense = (values[..., None] > lower) & (values[..., None] < upper)

    assert int(jnp.min(safe_index)) >= 0
    assert int(jnp.max(safe_index)) < lower.shape[0]
    np.testing.assert_array_equal(valid, jnp.any(dense, axis=-1))
    active = np.asarray(valid)
    np.testing.assert_array_equal(
        np.asarray(safe_index)[active],
        np.argmax(np.asarray(dense)[active], axis=-1),
    )


def test_distance_bins_ignores_boundary_order() -> None:
    # Checkpoint mappers pass boundaries in their stored order; the count of
    # thresholds below a distance does not depend on it.
    boundaries = jnp.asarray([2.0, 1.0, 0.0, 3.5], dtype=jnp.float32)
    distances = jnp.asarray([-1.0, 0.0, 0.5, 1.0, 2.5, 3.5, 9.0, np.nan])
    expected = jnp.sum(distances[..., None] > boundaries, axis=-1)
    np.testing.assert_array_equal(distance_bins(distances, boundaries), expected)


@pytest.mark.parametrize("shape", [(1, 6), (1, 1, 6), (1, 1, 1, 6), (3, 1, 6)])
def test_distance_bins_broadcasts_boundary_buffers_as_compare_and_sum(
    shape: tuple[int, ...],
) -> None:
    distances = jnp.asarray(
        [[0.5, 2.0, 7.5], [jnp.nan, 3.0, 11.0], [4.0, 4.5, 9.0]], jnp.float32
    )
    base = jnp.asarray([1.0, 2.0, 3.0, 5.0, 8.0, 10.0], jnp.float32)
    boundaries = jnp.broadcast_to(base, shape)
    expected = jnp.sum(distances[..., None] > boundaries, axis=-1)
    got = distance_bins(distances, boundaries)
    assert got.shape == expected.shape
    assert got.dtype == expected.dtype
    np.testing.assert_array_equal(np.asarray(got), np.asarray(expected))
