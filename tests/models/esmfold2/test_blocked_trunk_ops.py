"""Blocking the trunk's two widest operations changes bytes, not arithmetic.

`swiglu` widens `[..., C]` to `[..., 2 * hidden]` and holds the split halves
and their product at once; `outer_product_mean` builds `[B, N, N, c, d]` before
a projection narrows it to `[B, N, N, C_z]`. At 1,003 tokens XLA's arena
accounting named seven of the first at 3,930 MiB and four of the second at
1,965 MiB -- the top two tenants of this model's peak.

Both are blocked along an axis nothing reduces over, so the result is the same
value computed in smaller pieces. These tests pin that, and pin that the block
only engages when it is worth engaging.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.esmfold2.models import primitives, trunk


def _swiglu_params(channels: int, hidden: int, key) -> dict:
    ka, kb = jax.random.split(key)
    return {
        "ffn.w12.weight": jax.random.normal(ka, (2 * hidden, channels)) * 0.05,
        "ffn.w3.weight": jax.random.normal(kb, (channels, hidden)) * 0.05,
    }


def test_blocked_swiglu_matches_the_whole_one():
    """Same weights, same input, one call versus several."""
    key = jax.random.key(0)
    x = jax.random.normal(key, (1, 24, 24, 16))
    params = _swiglu_params(16, 32, jax.random.key(1))

    whole = primitives.swiglu(x, params, "ffn")
    # A budget small enough that the rows have to be divided.
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(primitives, "_SWIGLU_WIDE_BUDGET_BYTES", 4096)
        blocked = primitives.swiglu(x, params, "ffn")

    assert blocked.shape == whole.shape
    np.testing.assert_allclose(blocked, whole, rtol=1e-6, atol=1e-6)


def test_blocked_swiglu_survives_an_axis_that_does_not_divide():
    """The trailing block is shorter, and that is the case that got this wrong.

    The first version sliced `start + rows` unconditionally. `slice_in_dim`
    rejects an overrun instead of clamping, so it raised on any axis the block
    size did not divide -- which the original test missed by choosing a length
    the block size happened to divide. 23 rows in blocks of 5 leaves 3.
    """
    x = jax.random.normal(jax.random.key(8), (1, 23, 7, 16))
    params = _swiglu_params(16, 32, jax.random.key(9))

    whole = primitives.swiglu(x, params, "ffn")
    with pytest.MonkeyPatch.context() as patch:
        # 5 rows: 32 * 2 channels * 4 bytes * 7 = 1792 per row.
        patch.setattr(primitives, "_SWIGLU_WIDE_BUDGET_BYTES", 1792 * 5)
        blocked = primitives.swiglu(x, params, "ffn")

    assert blocked.shape == whole.shape
    np.testing.assert_allclose(blocked, whole, rtol=1e-6, atol=1e-6)


def test_chunked_outer_product_survives_a_token_count_that_does_not_divide():
    """Same trailing-block case on the other blocked operation."""
    msa = jax.random.normal(jax.random.key(10), (1, 13, 3, 12))
    msa_mask = jnp.ones((1, 13, 3))
    params = _opm_params(12, 4, 8, jax.random.key(11))

    whole = trunk.outer_product_mean(msa, params, "opm", msa_mask=msa_mask)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(trunk, "_OPM_OUTER_BUDGET_BYTES", 3 * 16 * 4 * 5)
        chunked = trunk.outer_product_mean(msa, params, "opm", msa_mask=msa_mask)

    assert chunked.shape == whole.shape
    np.testing.assert_allclose(chunked, whole, rtol=1e-5, atol=1e-6)


def test_the_swiglu_block_is_off_under_the_budget():
    """A small input takes the original single-call route.

    Asserted through the row helper rather than by timing: the point is that
    nothing below the budget pays for the machinery.
    """
    x = jnp.zeros((1, 8, 8, 16))
    assert primitives._swiglu_rows(x, axis=1, wide=32) is None


def test_the_swiglu_row_axis_skips_a_batch_of_one():
    """`[1, N, N, C]` divides along N; there is nothing to divide at axis 0."""
    assert primitives._swiglu_row_axis(jnp.zeros((1, 12, 12, 8))) == 1
    assert primitives._swiglu_row_axis(jnp.zeros((12, 12, 8))) == 0
    assert primitives._swiglu_row_axis(jnp.zeros((1, 1, 8))) is None


def _opm_params(c_m: int, c: int, c_z: int, key) -> dict:
    ka, kb = jax.random.split(key)
    return {
        "opm.norm.weight": jnp.ones((c_m,)),
        "opm.norm.bias": jnp.zeros((c_m,)),
        "opm.W.weight": jax.random.normal(ka, (2 * c, c_m)) * 0.05,
        "opm.Wout.weight": jax.random.normal(kb, (c_z, c * c)) * 0.05,
        "opm.Wout.bias": jax.random.normal(jax.random.key(7), (c_z,)) * 0.05,
    }


def test_chunked_outer_product_mean_matches_the_whole_one():
    """The projection moves inside the block; the value does not move."""
    key = jax.random.key(2)
    msa = jax.random.normal(key, (1, 6, 20, 12))
    msa_mask = jnp.ones((1, 6, 20))
    params = _opm_params(12, 4, 8, jax.random.key(3))

    whole = trunk.outer_product_mean(msa, params, "opm", msa_mask=msa_mask)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(trunk, "_OPM_OUTER_BUDGET_BYTES", 512)
        chunked = trunk.outer_product_mean(msa, params, "opm", msa_mask=msa_mask)

    assert chunked.shape == whole.shape
    np.testing.assert_allclose(chunked, whole, rtol=1e-5, atol=1e-6)


def test_the_outer_product_division_still_follows_the_projection():
    """The divide lands after `Wout`, so the bias is scaled by it too.

    Pinned against the arrangement written out here rather than against a
    property of the output, because the two orders differ only in whether the
    bias participates -- and a masked alignment is what makes them differ at
    all. Both the whole and the chunked route are checked against it, so the
    block cannot quietly move the divide inside.

    Axes are `[batch, token, sequence, channel]`: the contraction is over the
    sequence axis, which is what `bimc,bjmd->bijcd` says.
    """
    msa = jax.random.normal(jax.random.key(4), (1, 10, 4, 12))
    # Three of the four alignment rows are empty, so `valid` is not all ones.
    mask = jnp.array([[[1.0, 0.0, 0.0, 0.0]] * 10])
    params = _opm_params(12, 4, 8, jax.random.key(5))

    normalised = primitives.layer_norm(
        msa, params["opm.norm.weight"], params["opm.norm.bias"]
    )
    projected = primitives.linear(normalised, params, "opm.W")
    projected = projected * mask[..., None].astype(projected.dtype)
    half = projected.shape[-1] // 2
    a, b = projected[..., :half], projected[..., half:]
    valid = jnp.maximum(jnp.einsum("bim,bjm->bij", mask, mask)[..., None], 1.0)
    outer = jnp.einsum("bimc,bjmd->bijcd", a, b)
    outer = outer.reshape(outer.shape[:-2] + (half * half,))
    reference = primitives.linear(outer, params, "opm.Wout") / valid

    whole = trunk.outer_product_mean(msa, params, "opm", msa_mask=mask)
    np.testing.assert_allclose(whole, reference, rtol=1e-6, atol=1e-6)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(trunk, "_OPM_OUTER_BUDGET_BYTES", 512)
        chunked = trunk.outer_product_mean(msa, params, "opm", msa_mask=mask)
    np.testing.assert_allclose(chunked, reference, rtol=1e-5, atol=1e-6)

    # The bias is what the two orders disagree about, so it has to be present.
    assert not jnp.allclose(params["opm.Wout.bias"], 0.0)


def test_triangle_multiplication_keeps_its_operand_width():
    """The contraction's operands stay in the trunk's dtype.

    `routed` used to be promoted to float32 immediately after the mask was
    cast down to avoid exactly that. Asserted on the realized dtype of the
    lowered program rather than on the source, so a re-introduced cast fails
    here even if it is spelled differently.
    """
    pair = jnp.zeros((1, 8, 8, 4), dtype=jnp.bfloat16)
    params = {
        "t._engine.norm_start.weight": jnp.ones((4,)),
        "t._engine.norm_start.bias": jnp.zeros((4,)),
        "t._engine.proj_bundle.weight": jnp.zeros((16, 4), dtype=jnp.bfloat16),
        "t._engine.norm_mix.weight": jnp.ones((4,)),
        "t._engine.norm_mix.bias": jnp.zeros((4,)),
        "t._engine.proj_emit.weight": jnp.zeros((4, 4), dtype=jnp.bfloat16),
        "t._engine.proj_gate.weight": jnp.zeros((4, 4), dtype=jnp.bfloat16),
    }
    text = jax.jit(
        lambda p: trunk.triangle_multiplicative(p, params, "t", outgoing=True)
    ).lower(pair).as_text()
    # The triangular contraction is the one dot_general with batching dims.
    contraction = [
        line
        for line in text.splitlines()
        if "dot_general" in line and "batching_dims" in line
    ]
    assert len(contraction) == 1, contraction
    signature = contraction[0].split(":")[-1]
    operands, result = signature.split("->")
    assert operands.count("bf16") == 2, signature
    assert "f32" in result, signature
