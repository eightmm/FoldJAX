"""Blocking the trunk's two widest operations changes bytes, not arithmetic.

`swiglu` widens `[..., C]` to `[..., 2 * hidden]` and holds the split halves
and their product at once; `outer_product_mean` builds `[B, N, N, c, d]` before
a projection narrows it to `[B, N, N, C_z]`. At 1,003 tokens, and at the
2026-08-28 arrangement these two were measured in, XLA's arena accounting named
seven of the first at 3,930 MiB and four of the second at 1,965 MiB -- the top
two tenants of that model's peak.

Both are blocked along an axis nothing reduces over, so the result is the same
value computed in smaller pieces. These tests pin that, and pin that the block
only engages when it is worth engaging.

Only one of the two is still on the released path. `trunk_dtype="bfloat16"`
routes every transition to `trunk._autocast_transition`, whose own 64-row chunk
loop replaces `swiglu` entirely; `outer_product_mean` blocks on both arms of
its `native_autocast` branch and is untouched. The last test here pins that
routing, because it is the fact the `swiglu` docstring's arena numbers no
longer describe.
"""

from __future__ import annotations

import collections

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


def _triangle_params(channels: int) -> dict:
    """One `TriangleMultiplicativeBlock`'s weights, at `latent == channels`."""
    return {
        "t._engine.norm_start.weight": jnp.ones((channels,)),
        "t._engine.norm_start.bias": jnp.zeros((channels,)),
        "t._engine.proj_bundle.weight": jnp.zeros(
            (4 * channels, channels), dtype=jnp.bfloat16
        ),
        "t._engine.norm_mix.weight": jnp.ones((channels,)),
        "t._engine.norm_mix.bias": jnp.zeros((channels,)),
        "t._engine.proj_emit.weight": jnp.zeros((channels, channels), jnp.bfloat16),
        "t._engine.proj_gate.weight": jnp.zeros((channels, channels), jnp.bfloat16),
    }


def _pair_shaped_float32(jaxpr, tokens: int, channels: int) -> collections.Counter:
    """Which primitives emit a float32 `[b, N, N, channels]`, and how many.

    Width-keyed rather than name-keyed: the two lines this guards promote a
    `[b, N, N, 2 * latent]` tensor, and the point is which operations produce
    that width in float32, whatever spelling produced them.
    """
    found: collections.Counter = collections.Counter()

    def walk(eqns):
        for eqn in eqns:
            for var in eqn.outvars:
                aval = getattr(var, "aval", None)
                shape = getattr(aval, "shape", ())
                if (
                    len(shape) == 4
                    and shape[1] == shape[2] == tokens
                    and shape[3] == channels
                    and aval.dtype == jnp.float32
                ):
                    found[eqn.primitive.name] += 1
            for value in eqn.params.values():
                for inner in _sub_jaxprs(value):
                    walk(inner.eqns)

    walk(jaxpr.jaxpr.eqns)
    return found


def _sub_jaxprs(value):
    """Open jaxprs reachable from an equation parameter, at any nesting.

    `_autocast_linear` hides its arithmetic in a `platform_dependent`, whose
    branches are a tuple of closed jaxprs rather than a `.jaxpr` attribute.
    """
    inner = getattr(value, "jaxpr", None)
    if inner is not None:
        yield inner if not hasattr(inner, "jaxpr") else inner.jaxpr
    elif hasattr(value, "eqns"):
        yield value
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _sub_jaxprs(item)


def test_the_autocast_triangle_keeps_its_operand_width():
    """The same guard as above, on the branch the released default takes.

    `test_triangle_multiplication_keeps_its_operand_width` calls
    `triangle_multiplicative` without `native_autocast`, so it only ever
    exercised the body at the bottom of that function. The released default
    is `trunk_dtype="bfloat16"` off a mesh, which dispatches to
    `_autocast_triangle` instead -- and that copy kept the promotion for
    another two weeks because no test reached it.

    Two lines are pinned here and they fail differently, so neither can rot
    behind the other:

    * the mask multiply. `pair_mask` is float32 (`model.py:1310`) and `routed`
      is bfloat16, so a multiply without the down-cast promotes the widest
      tensor the block owns. Dropping the cast leaves a float32 `mul` and a
      float32 `split` at `2 * latent`.
    * the explicit `routed.astype(jnp.float32)` that used to follow it.
      Reinstating it leaves a float32 `convert_element_type` at the same
      width, with the `mul` still bfloat16.
    """
    tokens, channels = 8, 4
    pair = jnp.zeros((1, tokens, tokens, channels), dtype=jnp.bfloat16)
    # Float32 and 0/1, exactly as `model.py:1310-1312` builds it.
    mask = jnp.ones((1, tokens, tokens), dtype=jnp.float32)
    params = _triangle_params(channels)

    jaxpr = jax.make_jaxpr(
        lambda p, m: trunk.triangle_multiplicative(
            p, params, "t", outgoing=True, mask=m, native_autocast=True
        )
    )(pair, mask)

    # Exactly one float32 tenant of that width survives, and it is the gate:
    # `logits` is the other half of the same `proj_bundle` output, and an
    # exponential is never fed a rounded input. An exact census rather than
    # emptiness, because emptiness would be wrong here and a bare "fewer than
    # before" would let either line come back alone.
    #
    #   mask multiply un-cast   -> a `mul` joins this census
    #   `.astype(float32)` back -> a second `convert_element_type` joins it
    assert _pair_shaped_float32(jaxpr, tokens, 2 * channels) == {
        "convert_element_type": 1,  # logits -> float32, for the sigmoid
        "logistic": 1,  # the sigmoid itself
    }

    text = jax.jit(
        lambda p, m: trunk.triangle_multiplicative(
            p, params, "t", outgoing=True, mask=m, native_autocast=True
        )
    ).lower(pair, mask).as_text()
    contraction = [
        line
        for line in text.splitlines()
        if "dot_general" in line and "batching_dims" in line
    ]
    assert len(contraction) == 1, contraction
    operands, result = contraction[0].split(":")[-1].split("->")
    assert operands.count("bf16") == 2, contraction[0]
    assert "f32" in result, contraction[0]


def test_the_down_cast_mask_is_bit_identical_to_the_promotion():
    """Why the narrowing above may be asserted bitwise rather than to a tolerance.

    The einsum operand used to be `(routed * mask).astype(bfloat16)` with the
    multiply in float32, and is now `routed * mask.astype(bfloat16)`. Those
    agree bit for bit on two facts, both of which this checks rather than
    assumes: float32 represents every bfloat16 value exactly, so the
    round-trip is the identity; and `pair_mask` is 0.0/1.0, so the multiply
    is exact at either width.

    The second is a property of the caller, not of this function, so the last
    assertion shows what a mask that broke it would cost -- without it this
    test would pass on arithmetic that is merely close.
    """
    key = jax.random.key(0)
    routed = (jax.random.normal(key, (1, 8, 8, 6)) * 3.0).astype(jnp.bfloat16)
    mask = jnp.asarray(np.random.default_rng(0).integers(0, 2, (1, 8, 8)), jnp.float32)

    promoted = (routed * mask[..., None]).astype(jnp.float32)
    narrowed = routed * mask[..., None].astype(routed.dtype)

    assert promoted.dtype == jnp.float32 and narrowed.dtype == jnp.bfloat16
    np.testing.assert_array_equal(
        np.asarray(promoted.astype(jnp.bfloat16)), np.asarray(narrowed)
    )

    # A mask that is not 0/1 separates them, so the equality above is carried
    # by the caller's mask values rather than by bfloat16 happening to round
    # both orders the same way. Every weight, not one: a single off-grid
    # value often does round the same, which is exactly how a weak version of
    # this assertion would pass while proving nothing.
    weighted = jax.random.uniform(jax.random.key(1), mask.shape, jnp.float32)
    differing = np.count_nonzero(
        np.asarray((routed * weighted[..., None]).astype(jnp.bfloat16))
        != np.asarray(routed * weighted[..., None].astype(routed.dtype))
    )
    assert differing > 0, "a weighted mask must separate the two orders"


def test_the_released_pair_trunk_never_reaches_the_blocked_swiglu(monkeypatch):
    """`trunk_dtype="bfloat16"` sends every transition past `swiglu`.

    The `swiglu` docstring's arena numbers were measured before the
    native-autocast redirect existed, and a reader who trusts them picks this
    function as the place to change the trunk's memory. It is not: at the
    released default `transition` dispatches to `_autocast_transition`, whose
    own 64-row chunk loop owns that tenant. Asserted on which function runs,
    not on the value of `trunk_dtype`, so a redirect that moves keeps failing
    here rather than leaving the prose to rot again.
    """
    from foldjax.models.esmfold2.models import model as structure_model

    settings = structure_model.ModelSettings()
    # The condition `predict` and `confidence_head` both spell for themselves.
    assert jnp.dtype(settings.trunk_dtype) == jnp.bfloat16

    reached: list[str] = []

    def record(name):
        def stand_in(x, *args, **kwargs):
            reached.append(name)
            return x

        return stand_in

    monkeypatch.setattr(trunk, "swiglu", record("swiglu"))
    monkeypatch.setattr(trunk, "_autocast_transition", record("autocast"))

    params = {
        "t.norm.weight": jnp.ones((4,)),
        "t.norm.bias": jnp.zeros((4,)),
        "t.ffn.w12.weight": jnp.zeros((8, 4)),
        "t.ffn.w3.weight": jnp.zeros((4, 4)),
    }
    x = jnp.zeros((1, 6, 6, 4))

    trunk.transition(x, params, "t", residual=True, native_autocast=True)
    assert reached == ["autocast"]

    trunk.transition(x, params, "t", residual=True, native_autocast=False)
    assert reached == ["autocast", "swiglu"]
