from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.chai.data.collate import (
    AVAILABLE_MODEL_SIZES,
    block_atom_pair_mask,
    get_pad_sizes,
    qkv_indices_for_blocks,
)


@pytest.mark.parametrize(
    ("tokens", "bucket"),
    [
        (1, 256),
        (256, 256),
        (257, 384),
        (384, 384),
        (385, 512),
        (512, 512),
        (513, 768),
        (768, 768),
        (769, 1024),
        (1024, 1024),
        (1025, 1536),
        (1536, 1536),
        (1537, 2048),
        (2048, 2048),
    ],
)
def test_chai_token_buckets_and_atom_padding(tokens: int, bucket: int) -> None:
    padded = get_pad_sizes(tokens, tokens * 7)
    assert padded.n_tokens == bucket
    assert padded.n_atoms == 23 * bucket
    assert padded.n_atoms % 32 == 0


def test_padding_rejects_unsupported_sizes() -> None:
    with pytest.raises(ValueError, match="2049"):
        get_pad_sizes(2049, 1)
    with pytest.raises(ValueError, match="atoms"):
        get_pad_sizes(10, 23 * 256 + 1)
    assert AVAILABLE_MODEL_SIZES == (256, 384, 512, 768, 1024, 1536, 2048)


def test_qkv_indices_and_wrapped_mask_match_chai_boundary_behavior() -> None:
    query, keys, valid = qkv_indices_for_blocks(256, 32, 128)
    assert query.shape == (8, 32)
    assert keys.shape == (8, 128)
    np.testing.assert_array_equal(np.asarray(query[0]), np.arange(32))
    np.testing.assert_array_equal(np.asarray(keys[0, :48]), np.arange(208, 256))
    np.testing.assert_array_equal(np.asarray(keys[0, 48:]), np.arange(80))
    assert not np.asarray(valid[0, :48]).any()
    assert np.asarray(valid[0, 48:]).all()
    assert np.asarray(valid[3]).all()


def test_block_pair_mask_combines_atom_and_wrap_masks() -> None:
    query, keys, valid = qkv_indices_for_blocks(256, 32, 128)
    atom_mask = jnp.ones((1, 256), dtype=bool).at[:, 0].set(False)
    pair_mask = block_atom_pair_mask(atom_mask, query, keys, valid)
    assert pair_mask.shape == (1, 8, 32, 128)
    assert not bool(pair_mask[0, 0, 0].any())
    assert not bool(pair_mask[0, 0, 1, :48].any())
    assert bool(pair_mask[0, 0, 1, 49])
