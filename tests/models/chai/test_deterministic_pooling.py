from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from foldjax.models.chai.models.primitives import contiguous_segment_mean


def test_contiguous_segment_mean_ignores_padding_and_empty_tokens() -> None:
    values = jnp.asarray([[[1.0, 2.0], [3.0, 6.0], [8.0, 4.0], [99.0, 99.0]]])
    mask = jnp.asarray([[True, True, True, False]])
    segment_ids = jnp.asarray([[0, 0, 2, 0]])

    actual = contiguous_segment_mean(values, mask, segment_ids, num_segments=4)

    np.testing.assert_array_equal(
        actual,
        [[[2.0, 4.0], [0.0, 0.0], [8.0, 4.0], [0.0, 0.0]]],
    )


def test_contiguous_segment_mean_is_repeatable_when_compiled() -> None:
    values = jnp.arange(2 * 32 * 7, dtype=jnp.bfloat16).reshape(2, 32, 7)
    mask = jnp.asarray([[True] * 23 + [False] * 9] * 2)
    segment_ids = jnp.asarray([list(range(8)) for _ in range(2)]).repeat(4, axis=1)
    segment_ids = segment_ids.at[:, 23:].set(0)
    compiled = jax.jit(
        lambda x: contiguous_segment_mean(x, mask, segment_ids, num_segments=8)
    )

    first = compiled(values)
    second = compiled(values)

    np.testing.assert_array_equal(first, second)


def test_contiguous_segment_mean_handles_masked_hole_before_valid_tokens() -> None:
    """Glycan leaving-atom tokens are masked between later valid atom tokens."""
    values = jnp.asarray([[[2.0], [99.0], [6.0], [8.0], [123.0]]])
    mask = jnp.asarray([[True, False, True, True, False]])
    segment_ids = jnp.asarray([[0, 1, 2, 2, 0]])

    actual = contiguous_segment_mean(values, mask, segment_ids, num_segments=4)

    np.testing.assert_array_equal(actual, [[[2.0], [0.0], [7.0], [0.0]]])


def test_contiguous_segment_mean_handles_leading_and_trailing_masked_atoms() -> None:
    values = jnp.asarray([[[99.0], [4.0], [8.0], [123.0]]])
    mask = jnp.asarray([[False, True, True, False]])
    segment_ids = jnp.asarray([[0, 1, 1, 0]])

    actual = contiguous_segment_mean(values, mask, segment_ids, num_segments=3)

    np.testing.assert_array_equal(actual, [[[0.0], [6.0], [0.0]]])


def test_masked_hole_segment_mean_is_repeatable_when_compiled() -> None:
    values = jnp.arange(20, dtype=jnp.bfloat16).reshape(1, 10, 2)
    mask = jnp.asarray(
        [[True, True, False, True, True, True, False, False, False, False]]
    )
    segment_ids = jnp.asarray([[0, 0, 1, 2, 2, 3, 0, 0, 0, 0]])
    compiled = jax.jit(
        lambda x: contiguous_segment_mean(x, mask, segment_ids, num_segments=5)
    )

    first = compiled(values)
    second = compiled(values)

    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first[:, 1], 0)
    np.testing.assert_array_equal(first[:, 4], 0)
