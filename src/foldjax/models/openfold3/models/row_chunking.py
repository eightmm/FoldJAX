"""Evaluate a row-independent function on a pair tensor in row blocks.

The pair transition is where a long target's memory goes. It widens
``[..., N, N, C_z]`` to ``[..., N, N, n * C_z]`` twice -- SwiGLU is two projections
multiplied together -- so at the released ``C_z = 128`` and ``n = 4`` a single
transition holds roughly eight times the pair representation. At 2000 tokens the
pair representation is 2 GiB, which puts one transition well past what fits
alongside everything else.

Every row of that computation is independent: the layer norm is over the channel
axis and the projections are per position. So it can be evaluated a block of rows at
a time, which caps the widened intermediate at ``chunk / N`` of its full size while
computing exactly the same values. This is the same lever upstream calls
``settings.memory.eval.chunk_size`` and the sibling Chai port applies to its own pair
transition.

Triangle attention in the same block is chunked by the same ``chunk_size``, and at
the sizes that matter it dominates: its scores are ``[rows, heads, N, N]``, so one
unchunked pair block needs 27 GiB of temporaries at 966 tokens and 267 GiB at 2076,
against 9 GiB and 41 GiB at 256 rows. Both are measured on the released widths.

An earlier note in this port said chunking changed neither peak nor speed. That was
wrong, and wrong in an instructive way: peak was read from the allocator's pool,
which had been preallocated and therefore did not move whatever the program did.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp


def map_row_chunks(
    function: Callable[..., jnp.ndarray],
    *arrays: jnp.ndarray,
    chunk_size: int | None,
    row_axes: tuple[int, ...] | int = -3,
    out_row_axis: int | None = None,
) -> jnp.ndarray:
    """Apply ``function`` to row blocks of ``arrays`` and rejoin the results.

    Args:
        function: called with one row block of each array, in order. It must treat
            rows independently: block ``k`` sees only rows ``k`` of the inputs, so a
            reduction across the row axis would silently be computed per block
            instead of across all of them.
        *arrays: the operands. They need not have equal rank, which is why the axis
            is given per array rather than once: a pair tensor is
            ``[..., N, N, C]`` and its mask is ``[..., N, N]``, so the same rows
            live at ``-3`` in one and ``-2`` in the other. Passing a single axis for
            both would chunk the mask along its channel-less second token axis and
            silently compute the wrong thing.
        chunk_size: rows per block. ``None`` or a value at least the row count
            applies ``function`` once, which is the fastest and highest-peak path.
        row_axes: the row axis of each array, or one axis used for all of them.
        out_row_axis: the row axis of the result. Defaults to ``row_axes[0]``.

    Returns:
        The same tensor ``function`` would have returned unchunked.
    """
    if not arrays:
        raise ValueError("map_row_chunks needs at least one array")
    if isinstance(row_axes, int):
        row_axes = (row_axes,) * len(arrays)
    if len(row_axes) != len(arrays):
        raise ValueError(
            f"got {len(row_axes)} row axes for {len(arrays)} arrays"
        )
    rows = arrays[0].shape[row_axes[0]]
    for array, axis in zip(arrays[1:], row_axes[1:], strict=True):
        if array.shape[axis] != rows:
            raise ValueError(
                "every array must have the same length along its row axis; got "
                f"{rows} and {array.shape[axis]}"
            )
    if chunk_size is None or chunk_size >= rows or chunk_size <= 0:
        return function(*arrays)

    blocks = -(-rows // chunk_size)
    padded = blocks * chunk_size

    def to_blocks(array: jnp.ndarray, row_axis: int) -> jnp.ndarray:
        axis = row_axis % array.ndim
        if padded != rows:
            # Padding rather than a ragged last block, so the body is emitted once.
            # The pad rows are computed and then discarded; they cannot affect the
            # real rows because the function is row-independent, which is the
            # precondition stated above.
            width = [(0, 0)] * array.ndim
            width[axis] = (0, padded - rows)
            array = jnp.pad(array, width)
        shape = (
            array.shape[:axis] + (blocks, chunk_size) + array.shape[axis + 1 :]
        )
        return jnp.moveaxis(array.reshape(shape), axis, 0)

    stacked = [
        to_blocks(array, axis) for array, axis in zip(arrays, row_axes, strict=True)
    ]
    result = jax.lax.map(lambda block: function(*block), tuple(stacked))
    # ``result`` is ``[blocks, ...leading..., chunk, ...]``: undo the move, fold the
    # block axis back into the row axis, and drop the padding.
    axis = (row_axes[0] if out_row_axis is None else out_row_axis) % (result.ndim - 1)
    result = jnp.moveaxis(result, 0, axis)
    merged = result.shape[:axis] + (padded,) + result.shape[axis + 2 :]
    result = result.reshape(merged)
    if padded != rows:
        result = jax.lax.slice_in_dim(result, 0, rows, axis=axis)
    return result
