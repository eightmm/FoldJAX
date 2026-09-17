import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.esmfold2.models import model as structure_model
from foldjax.models.esmfold2.models.primitives import _cuda_profile_divide


@pytest.mark.parametrize("rows", [1, 2, 5, 36])
def test_the_blocked_msa_profile_totals_are_bit_identical(rows):
    """Blocking the alignment rows may be asserted bitwise, not to a tolerance.

    Both the one-hot and `msa_attention_mask` are 0.0/1.0, so every partial sum
    is a whole number no larger than the alignment depth -- which float32
    represents exactly, and exact integers add in any order to the same bits.
    The last assertion is what makes that a measurement rather than a claim:
    the counts really are whole numbers, so a fractional mask would show up
    here rather than as a quiet drift.
    """
    rng = np.random.default_rng(0)
    depth, tokens = 37, 11
    msa = jnp.asarray(rng.integers(0, structure_model.NUM_RES_TYPES,
                                   (1, depth, tokens)), jnp.int32)
    mask = jnp.asarray(rng.random((1, depth, tokens)) > 0.2)

    def totals(width):
        return np.asarray(
            jax.jit(
                lambda a, b: structure_model._masked_one_hot_totals(
                    a, b, rows=width
                )
            )(msa, mask)
        )

    whole = totals(None)
    # The spelling the unblocked code used, kept here so the block is compared
    # against the arithmetic it replaced rather than against itself.
    one_hot = jax.nn.one_hot(msa.astype(jnp.int32), structure_model.NUM_RES_TYPES)
    one_hot = one_hot * mask[..., None].astype(jnp.float32)
    np.testing.assert_array_equal(whole, np.asarray(jnp.sum(one_hot, axis=1)))

    np.testing.assert_array_equal(totals(rows), whole)
    assert whole.max() > 1.0, whole.max()
    assert (whole == np.floor(whole)).all()


def test_the_msa_profile_block_is_off_under_the_budget():
    """A small alignment takes the original single-call route."""
    assert structure_model._msa_profile_rows(jnp.zeros((1, 37, 11), jnp.int32)) is None
    # 2,096 tokens x 13,280 rows is the shape the block was measured on.
    assert (
        structure_model._msa_profile_rows(jnp.zeros((1, 13280, 2096), jnp.int32))
        == 1940
    )


@pytest.mark.parametrize("rows", [0, 1, 7, 33])
def test_profile_division_shape_and_tail(rows):
    result = jax.eval_shape(
        _cuda_profile_divide,
        jax.ShapeDtypeStruct((1, rows, 33), jnp.float32),
        jax.ShapeDtypeStruct((1, rows, 1), jnp.float32),
    )
    assert result.shape == (1, rows, 33)
    assert result.dtype == jnp.float32


def test_profile_division_rejects_low_precision_operands():
    with pytest.raises(ValueError, match="FP32"):
        _cuda_profile_divide(jnp.ones((2, 33), jnp.bfloat16), jnp.ones((2, 1)))


@pytest.mark.skipif(
    os.environ.get("FOLDJAX_TEST_PROFILE_CUDA") != "1",
    reason="queued CUDA operator gate",
)
@pytest.mark.parametrize("rows", [1, 7, 33])
def test_profile_division_cuda_rounds_integer_count_ratios(rows):
    assert jax.default_backend() == "gpu"
    rng = np.random.default_rng(27)
    denominator = rng.integers(1, 4097, size=(1, rows, 1)).astype(np.float32)
    numerator = np.minimum(
        rng.integers(0, 4097, size=(1, rows, 33)), denominator
    ).astype(np.float32)
    expected = (numerator.astype(np.float64) / denominator.astype(np.float64)).astype(
        np.float32
    )
    actual = jax.jit(_cuda_profile_divide)(
        jnp.asarray(numerator), jnp.asarray(denominator)
    )
    np.testing.assert_array_equal(np.asarray(actual), expected)
