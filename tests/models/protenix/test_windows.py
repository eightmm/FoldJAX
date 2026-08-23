"""Overlapping atom windows stay exact while lowering as one gather."""

from __future__ import annotations

import math
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.protenix.models.diffusion import atom
from foldjax.models.protenix.models.primitives import attention
from foldjax.models.protenix.models.primitives.windows import (
    gather_overlapping_windows,
)


def _slice_reference(array, *, axis, window_size, stride):
    axis %= array.ndim
    n_windows = (array.shape[axis] - window_size) // stride + 1
    windows = []
    for index in range(n_windows):
        slices = [slice(None)] * array.ndim
        slices[axis] = slice(index * stride, index * stride + window_size)
        windows.append(array[tuple(slices)])
    return jnp.stack(windows, axis=axis)


@pytest.mark.parametrize(
    ("shape", "axis"),
    [((11, 3), 0), ((2, 11, 3), 1), ((2, 4, 11, 3), -2)],
)
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_gather_windows_matches_the_slice_stack(shape, axis, dtype) -> None:
    values = jnp.arange(np.prod(shape), dtype=jnp.float32).reshape(shape).astype(dtype)
    expected = _slice_reference(values, axis=axis, window_size=5, stride=3)
    actual = gather_overlapping_windows(
        values,
        axis=axis,
        n_windows=3,
        window_size=5,
        stride=3,
    )
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


@pytest.mark.parametrize(
    ("raw_bits", "storage_dtype", "value_dtype"),
    [
        (
            [
                0x00000000,
                0x80000000,
                0x7FC01234,
                0x7F800000,
                0xFF800000,
                0x00000001,
                0x3F800000,
                0xBF800000,
            ],
            np.uint32,
            np.float32,
        ),
        (
            [
                0x0000,
                0x8000,
                0x7FC0,
                0x7F80,
                0xFF80,
                0x0001,
                0x3F80,
                0xBF80,
            ],
            np.uint16,
            jnp.bfloat16,
        ),
    ],
)
def test_gather_windows_preserves_nonfinite_raw_bits(
    raw_bits, storage_dtype, value_dtype
) -> None:
    bits = np.asarray(raw_bits, dtype=storage_dtype)
    if value_dtype is jnp.bfloat16:
        import ml_dtypes

        host_values = bits.view(ml_dtypes.bfloat16)
    else:
        host_values = bits.view(value_dtype)
    values = jnp.asarray(host_values)
    expected = _slice_reference(values, axis=0, window_size=4, stride=2)
    actual = gather_overlapping_windows(
        values,
        axis=0,
        n_windows=3,
        window_size=4,
        stride=2,
    )
    np.testing.assert_array_equal(
        np.asarray(actual).view(storage_dtype),
        np.asarray(expected).view(storage_dtype),
    )


@pytest.mark.parametrize("dtype", [jnp.bool_, jnp.int32])
def test_gather_windows_preserves_non_floating_dtypes(dtype) -> None:
    values = jnp.arange(8, dtype=jnp.int32).astype(dtype)
    expected = _slice_reference(values, axis=0, window_size=4, stride=2)
    actual = gather_overlapping_windows(
        values,
        axis=0,
        n_windows=3,
        window_size=4,
        stride=2,
    )
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


def _leading_reference(value, *, n_queries, n_keys, compute_mask):
    n = value.shape[0]
    n_trunks = math.ceil(n / n_queries)
    q_pad = n_trunks * n_queries - n
    pad_left = (n_keys - n_queries) // 2
    pad_right = int((n_trunks - 0.5) * n_queries + n_keys / 2 - n + 0.5)
    q_padded = jnp.pad(value, ((0, q_pad), (0, 0)))
    k_padded = jnp.pad(value, ((pad_left, pad_right), (0, 0)))
    q_trunked = q_padded.reshape(n_trunks, n_queries, value.shape[-1])
    k_trunked = _slice_reference(
        k_padded,
        axis=0,
        window_size=n_keys,
        stride=n_queries,
    )
    mask = None
    if compute_mask:
        q_abs = jnp.arange(n_trunks * n_queries).reshape(n_trunks, n_queries)
        k_abs = (
            jnp.arange(n_keys)[None]
            + jnp.arange(n_trunks)[:, None] * n_queries
            - pad_left
        )
        mask = (q_abs[..., None] < n) & (k_abs[:, None] >= 0) & (
            k_abs[:, None] < n
        )
    return q_trunked, k_trunked, mask, (q_pad, pad_left, pad_right)


@pytest.mark.parametrize("n_atoms", [1, 32, 33])
@pytest.mark.parametrize("compute_mask", [False, True])
def test_public_windows_preserve_padding_and_masks(n_atoms, compute_mask) -> None:
    values = jnp.arange(n_atoms * 2, dtype=jnp.float32).reshape(n_atoms, 2)
    expected_q, expected_k, expected_mask, expected_padding = _leading_reference(
        values,
        n_queries=32,
        n_keys=128,
        compute_mask=compute_mask,
    )
    actual_q, actual_k, padding = atom.rearrange_qk_to_dense_trunk(
        values,
        values,
        n_queries=32,
        n_keys=128,
        compute_mask=compute_mask,
    )

    np.testing.assert_array_equal(np.asarray(actual_q), np.asarray(expected_q))
    np.testing.assert_array_equal(np.asarray(actual_k), np.asarray(expected_k))
    if expected_mask is None:
        assert padding["mask_trunked"] is None
    else:
        np.testing.assert_array_equal(
            np.asarray(padding["mask_trunked"]), np.asarray(expected_mask)
        )
    assert (
        padding["q_pad"],
        padding["k_pad_left"],
        padding["k_pad_right"],
    ) == expected_padding


@pytest.mark.parametrize(
    "function",
    [
        lambda value, q, k: atom.rearrange_qk_to_dense_trunk(
            value, value, n_queries=q, n_keys=k
        ),
        lambda value, q, k: atom._rearrange_qk_to_dense_trunk_atom_axis(
            value, value, n_queries=q, n_keys=k
        ),
        lambda value, q, k: attention._local_qk_trunks(
            value, value, n_queries=q, n_keys=k
        ),
    ],
)
@pytest.mark.parametrize(("n_queries", "n_keys"), [(0, 4), (-2, 4), (2, 0)])
def test_production_windows_reject_nonpositive_sizes(
    function, n_queries, n_keys
) -> None:
    with pytest.raises(ValueError):
        function(jnp.zeros((1, 4), dtype=jnp.float32), n_queries, n_keys)


@pytest.mark.parametrize(
    "function",
    [
        lambda value: atom.rearrange_qk_to_dense_trunk(value, value),
        lambda value: atom._rearrange_qk_to_dense_trunk_atom_axis(
            value, value, n_queries=32, n_keys=128
        ),
        lambda value: attention._local_qk_trunks(
            value, value, n_queries=32, n_keys=128
        ),
    ],
)
def test_production_windows_reject_empty_atom_axes(function) -> None:
    with pytest.raises(ValueError, match="at least one atom"):
        function(jnp.zeros((0, 4), dtype=jnp.float32))


@pytest.mark.parametrize(
    ("array", "axis", "n_windows", "window_size", "stride", "message"),
    [
        (jnp.asarray(1), 0, 1, 1, 1, "array axis"),
        (jnp.zeros((4,)), 1, 1, 1, 1, "out of bounds"),
        (jnp.zeros((4,)), 0, 1, 0, 1, "window_size"),
        (jnp.zeros((4,)), 0, 1, 1, 0, "stride"),
        (jnp.zeros((4,)), 0, 0, 1, 1, "n_windows"),
        (jnp.zeros((3,)), 0, 1, 4, 1, "does not match"),
        (jnp.zeros((8,)), 0, 3, 4, 3, "expected 10"),
    ],
)
def test_gather_windows_rejects_ambiguous_layouts(
    array, axis, n_windows, window_size, stride, message
) -> None:
    with pytest.raises(ValueError, match=message):
        gather_overlapping_windows(
            array,
            axis=axis,
            n_windows=n_windows,
            window_size=window_size,
            stride=stride,
        )


def _leading_windows(value):
    return atom.rearrange_qk_to_dense_trunk(
        value,
        value,
        n_queries=32,
        n_keys=128,
        compute_mask=False,
    )[1]


def _atom_axis_windows(value):
    return atom._rearrange_qk_to_dense_trunk_atom_axis(
        value,
        value,
        n_queries=32,
        n_keys=128,
    )[1]


def _attention_windows(value):
    return attention._local_qk_trunks(
        value,
        value,
        n_queries=32,
        n_keys=128,
    )[1]


@pytest.mark.parametrize(
    ("function", "shape"),
    [
        (_leading_windows, (2048, 8)),
        (_atom_axis_windows, (2, 2048, 8)),
        (_attention_windows, (2, 4, 2048, 8)),
    ],
)
def test_every_production_window_lowers_to_one_gather(
    function: Callable[[jnp.ndarray], jnp.ndarray], shape: tuple[int, ...]
) -> None:
    lowered = jax.jit(function).lower(jnp.zeros(shape, dtype=jnp.float32))
    hlo = str(lowered.compiler_ir(dialect="stablehlo"))

    assert hlo.count('"stablehlo.gather"(') == 1
    assert "stablehlo.concatenate" not in hlo


def test_atom_axis_window_does_not_allocate_an_output_sized_temporary() -> None:
    value = jnp.zeros((2, 4, 2048, 8), dtype=jnp.float32)
    compiled = jax.jit(_attention_windows).lower(value).compile()
    memory = compiled.memory_analysis()
    output_bytes = 2 * 4 * 64 * 128 * 8 * np.dtype(np.float32).itemsize

    assert memory.temp_size_in_bytes < output_bytes
