"""Native cyclic token ordering, including mixed chains and padding."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.openfold3.models.relpos import relpos_complex


@pytest.mark.parametrize("length", [1, 3, 5, 6, 7])
def test_cyclic_offsets_follow_native_ties_to_even(length):
    n = length + 2
    batch = {
        "residue_index": jnp.arange(n)[None] * 3,
        "token_index": jnp.arange(n)[None],
        "asym_id": jnp.asarray([[1] * length + [2, 2]]),
        "entity_id": jnp.ones((1, n), dtype=jnp.int32),
        "sym_id": jnp.asarray([[1] * length + [2, 2]]),
        "cyclic_mask": jnp.asarray([[True] * length + [False, False]]),
    }
    run = jax.jit(
        lambda x: relpos_complex(x, max_relative_idx=16, max_relative_chain=8)
    )
    actual = np.asarray(run(batch))
    linear = np.asarray(run({k: v for k, v in batch.items() if k != "cyclic_mask"}))
    # The released implementation builds a row and rolls it; notably n=3
    # rounds its centre to 2 and does not use a generic shortest-path formula.
    row = np.arange(0, -length, -1)
    centre = round(length / 2)
    row[centre + 1 :] = np.arange(len(row[centre + 1 :]), 0, -1)
    offsets = np.stack([np.roll(row, i) for i in range(length)])
    np.testing.assert_array_equal(
        actual[0, :length, :length, :34].argmax(-1), offsets + 16
    )
    np.testing.assert_array_equal(actual[0, length:], linear[0, length:])
    np.testing.assert_array_equal(actual[0, :, length:], linear[0, :, length:])
    # Native applies the same cyclic substitution to its sym_id block too.
    np.testing.assert_array_equal(
        actual[0, :length, :length, 69:].argmax(-1), offsets + 8
    )


def test_cyclic_masks_are_per_sample_and_select_ordered_subsets():
    batch = {
        "residue_index": jnp.zeros((2, 6), dtype=jnp.int32),
        "token_index": jnp.arange(6)[None].repeat(2, 0),
        "asym_id": jnp.ones((2, 6), dtype=jnp.int32),
        "entity_id": jnp.ones((2, 6), dtype=jnp.int32),
        "sym_id": jnp.ones((2, 6), dtype=jnp.int32),
        "cyclic_mask": jnp.asarray([[1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1]], bool),
    }
    actual = np.asarray(relpos_complex(batch, max_relative_idx=4, max_relative_chain=2))
    for sample, indices in enumerate(([0, 2, 4], [1, 3, 5])):
        expected = np.asarray([[0, -1, -2], [-2, 0, -1], [-1, -2, 0]])
        np.testing.assert_array_equal(
            actual[sample, :, :, :10].argmax(-1)[np.ix_(indices, indices)],
            expected + 4,
        )
