"""Row chunking must change peak memory and nothing else.

``chunk_size`` exists to make long targets fit: the pair transition widens
``[..., N, N, C_z]`` to four times its channel count twice over, and at 2000 tokens
that single intermediate is larger than everything else in the block put together.
Rows are independent, so evaluating them in blocks is exact -- and "exact" is the
whole claim, so it needs a test that would notice a chunk boundary handled wrongly.

The cases that matter are the ones a hand-written loop gets wrong: a chunk size that
does not divide the token count, a chunk size of one, and a chunk size larger than
the tensor.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.openfold3.models.primitives import (
    LayerNormParams,
    LinearParams,
    SwiGLUParams,
    SwiGLUTransitionParams,
    swiglu_transition,
)
from foldjax.models.openfold3.models.row_chunking import map_row_chunks

N_TOKEN, C_Z, HIDDEN = 13, 6, 24


def _params(key: jax.Array) -> SwiGLUTransitionParams:
    keys = jax.random.split(key, 3)
    return SwiGLUTransitionParams(
        layer_norm=LayerNormParams(
            weight=jnp.ones(C_Z, dtype=jnp.float32),
            bias=jnp.zeros(C_Z, dtype=jnp.float32),
        ),
        swiglu=SwiGLUParams(
            linear_a=LinearParams(
                weight=jax.random.normal(keys[0], (HIDDEN, C_Z), jnp.float32) * 0.3,
                bias=None,
            ),
            linear_b=LinearParams(
                weight=jax.random.normal(keys[1], (HIDDEN, C_Z), jnp.float32) * 0.3,
                bias=None,
            ),
        ),
        linear_out=LinearParams(
            weight=jax.random.normal(keys[2], (C_Z, HIDDEN), jnp.float32) * 0.3,
            bias=None,
        ),
    )


@pytest.fixture(scope="module")
def case():
    keys = jax.random.split(jax.random.key(0), 3)
    z = jax.random.normal(keys[0], (1, N_TOKEN, N_TOKEN, C_Z), jnp.float32)
    # A real mask, not all-ones: masking is applied per row, so an all-ones mask
    # would hide a chunked mask that had been sliced along the wrong axis.
    mask = (jax.random.uniform(keys[1], (1, N_TOKEN, N_TOKEN)) > 0.3).astype(
        jnp.float32
    )
    return z, mask, _params(keys[2])


# 13 tokens: 5 and 8 leave a partial final block, 13 is exactly one block, 64 is
# larger than the tensor, and 1 is the degenerate case.
@pytest.mark.parametrize("chunk_size", [1, 2, 5, 8, 12, 13, 14, 64, None])
def test_chunked_transition_is_exact(case, chunk_size) -> None:
    z, mask, params = case
    expected = swiglu_transition(z, params, mask=mask)
    actual = map_row_chunks(
        lambda rows, rows_mask: swiglu_transition(rows, params, mask=rows_mask),
        z,
        mask,
        chunk_size=chunk_size,
        row_axes=(-3, -2),
    )
    assert actual.shape == expected.shape
    np.testing.assert_allclose(
        np.asarray(actual, dtype=np.float64),
        np.asarray(expected, dtype=np.float64),
        rtol=1e-6,
        atol=1e-6,
        err_msg=f"chunk_size={chunk_size} changed the transition's output",
    )


def test_chunking_is_not_trivially_satisfied(case) -> None:
    """The mask must actually do something, or the test above proves little.

    If every mask entry were one, a chunked mask sliced along the wrong axis would
    still produce the right answer and the parametrization above would pass.
    """
    _, mask, _ = case
    fraction = float(np.asarray(mask).mean())
    assert 0.2 < fraction < 0.95, f"mask is nearly constant ({fraction:.2f})"


def test_mismatched_row_lengths_are_refused() -> None:
    """Silently broadcasting a wrong-length operand would corrupt the rows."""
    with pytest.raises(ValueError, match="same length along its row axis"):
        map_row_chunks(
            lambda a, b: a + b,
            jnp.zeros((4, 4, 2)),
            jnp.zeros((3, 3, 2)),
            chunk_size=2,
        )


def test_row_axis_count_must_match_arrays() -> None:
    with pytest.raises(ValueError, match="row axes for"):
        map_row_chunks(
            lambda a, b: a + b,
            jnp.zeros((4, 4, 2)),
            jnp.zeros((4, 4, 2)),
            chunk_size=2,
            row_axes=(-3,),
        )


def test_pair_block_chunking_is_exact(openfold3_source, randomized) -> None:
    """End to end through ``pair_block``, where the wiring could still be wrong.

    ``map_row_chunks`` being exact does not prove ``pair_block`` passes it the right
    axes; this runs the real block both ways on the same parameters.
    """
    pytest.importorskip("torch")
    from foldjax.models.openfold3.models.pair_block import pair_block

    from .test_inference_end_to_end import _params, _torch

    torch = _torch()
    params = _params(torch, randomized)
    block = params.trunk.pairformer_stack.blocks[0].pair_stack

    keys = jax.random.split(jax.random.key(1), 2)
    tokens = 11
    channels = block.pair_transition.layer_norm.weight.shape[0]
    z = jax.random.normal(keys[0], (1, tokens, tokens, channels), jnp.float32) * 0.1
    mask = jnp.ones((1, tokens, tokens), dtype=jnp.float32)

    unchunked = pair_block(z, block, pair_mask=mask, no_heads_pair=2, chunk_size=None)
    chunked = pair_block(z, block, pair_mask=mask, no_heads_pair=2, chunk_size=4)
    np.testing.assert_allclose(
        np.asarray(chunked, dtype=np.float64),
        np.asarray(unchunked, dtype=np.float64),
        rtol=1e-5,
        atol=1e-5,
        err_msg="pair_block's chunked transition disagrees with the unchunked one",
    )


def test_auto_chunk_leaves_small_targets_unchunked() -> None:
    """Chunking costs speed, so it must not engage where it is not needed."""
    from foldjax.models.openfold3.inference import auto_pair_chunk_size

    assert auto_pair_chunk_size(76, no_heads=4) is None
    assert auto_pair_chunk_size(574, no_heads=4) is None


def test_auto_chunk_caps_the_score_tensor() -> None:
    """Whatever the token count, the score tensor stays inside the budget.

    This is the property that makes long targets runnable: triangle attention's
    scores are ``[rows, heads, N, N]``, cubic in token count, so an unchunked
    2076-token block needs 267 GiB while a chunked one needs 41.
    """
    from foldjax.models.openfold3.inference import (
        PAIR_SCORE_BUDGET_BYTES,
        auto_pair_chunk_size,
    )

    for n_token in (832, 966, 1494, 1928, 2076, 4096):
        rows = auto_pair_chunk_size(n_token, no_heads=4)
        assert rows is not None, f"{n_token} tokens should be chunked"
        assert 1 <= rows <= n_token
        assert rows * 4 * n_token * n_token * 4 <= PAIR_SCORE_BUDGET_BYTES


@pytest.mark.parametrize("n_token", [832, 966, 1494, 1928, 2076, 3000])
def test_auto_chunk_uses_the_fewest_blocks_that_fit(n_token) -> None:
    """No block count below this one fits, so no fewer passes are possible.

    The chunk itself is *not* the largest that fits, deliberately: the rows are
    spread evenly over the minimum number of blocks instead. A chunk that does not
    divide the token count is padded to a whole block and the padding is computed and
    discarded, so 575 of 966 costs a fifth more work than 483 for the same two
    passes. What has to be minimal is the block count.
    """
    from foldjax.models.openfold3.inference import (
        PAIR_SCORE_BUDGET_BYTES,
        auto_pair_chunk_size,
    )

    heads = 4
    rows = auto_pair_chunk_size(n_token, no_heads=heads)
    per_row = heads * n_token * n_token * 4
    blocks = -(-n_token // rows)
    assert blocks * rows - n_token < rows, "a whole block of pure padding"
    fewer = blocks - 1
    assert fewer < 1 or (-(-n_token // fewer)) * per_row > PAIR_SCORE_BUDGET_BYTES, (
        f"{n_token} tokens could have run in {fewer} blocks instead of {blocks}"
    )


def test_auto_chunk_spreads_rows_evenly() -> None:
    """The padding a chunk forces must be under one block, and small."""
    from foldjax.models.openfold3.inference import auto_pair_chunk_size

    for n_token in (832, 966, 1494, 1928, 2076, 3000):
        rows = auto_pair_chunk_size(n_token, no_heads=4)
        padded = -(-n_token // rows) * rows
        assert (padded - n_token) / n_token < 0.02, (
            f"{n_token} tokens padded to {padded}, wasting "
            f"{100 * (padded - n_token) / n_token:.1f}% of the chunked work"
        )


def test_auto_chunk_never_returns_zero() -> None:
    """A budget too small for one row must still yield a runnable chunk."""
    from foldjax.models.openfold3.inference import auto_pair_chunk_size

    assert auto_pair_chunk_size(4096, no_heads=4, budget_bytes=1) == 1


def test_released_config_chunks_by_size_and_honours_an_override() -> None:
    from foldjax.models.openfold3.inference import released_config

    assert released_config(n_token=76, n_atom=601).pair_chunk_size is None
    assert released_config(n_token=2076, n_atom=16176).pair_chunk_size is not None
    # An explicit value, including None, must win over the automatic choice.
    assert (
        released_config(n_token=2076, n_atom=1, pair_chunk_size=None).pair_chunk_size
        is None
    )
    assert (
        released_config(n_token=2076, n_atom=1, pair_chunk_size=64).pair_chunk_size
        == 64
    )
