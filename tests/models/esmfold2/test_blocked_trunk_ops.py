"""Blocking the trunk's widest operations changes bytes, not arithmetic.

`swiglu` widens `[..., C]` to `[..., 2 * hidden]` and holds the split halves
and their product at once; `outer_product_mean` builds `[B, N, N, c, d]` before
a projection narrows it to `[B, N, N, C_z]`. At 1,003 tokens, and at the
2026-08-28 arrangement these two were measured in, XLA's arena accounting named
seven of the first at 3,930 MiB and four of the second at 1,965 MiB -- the top
two tenants of that model's peak.

Both are blocked along an axis nothing reduces over, so the result is the same
value computed in smaller pieces. These tests pin that, and pin that the block
only engages when it is worth engaging.

Only one of those two is still on the released path. `trunk_dtype="bfloat16"`
routes every transition to `trunk._autocast_transition`, whose own 64-row chunk
loop replaces `swiglu` entirely; `outer_product_mean` blocks on both arms of
its `native_autocast` branch and is untouched. The last test here pins that
routing, because it is the fact the `swiglu` docstring's arena numbers no
longer describe.

The third blocked stage is the native triangle itself. Its prologue --
`trunk._triangle_prologue`, everything from `norm_start` to the split -- was
the widest unblocked pair region left on that released path: the `proj_bundle`
output alone is four times the pair width and is pinned by
`_native_bf16_linear`'s barrier. Blocking the prologue alone only moved those
bytes, because assembling `left`, `right` and `output_gate` for a whole
contraction is three full-width destinations that did not exist before, so the
serial path now streams the contraction instead
(`trunk._autocast_triangle_streamed`): one operand whole, everything else --
the other operand, the gate, the contraction's destination and the whole
epilogue -- a block of the output at a time. Its arms here are a value
comparison against the unblocked form and a census counting what survives at
full width; the sharded half lives in `test_context_parallel`, because a mesh
keeps the whole-operand arrangement and blocks the prologue inside a
`shard_map` instead.
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


def _random_triangle_params(channels: int, seed: int) -> dict:
    """`_triangle_params` with weights, which the zero ones cannot replace.

    The shared fixture zeroes every projection because the tests that use it
    read the *program*, not its output. A value comparison run on it agrees
    perfectly and proves nothing, which is what the non-zero tripwire in each
    test below is there to say.
    """
    rng = np.random.default_rng(seed)

    def arr(*shape, dtype=jnp.float32):
        return jnp.asarray(rng.normal(size=shape, scale=0.5), dtype)

    params = _triangle_params(channels)
    engine = "t._engine"
    params[f"{engine}.norm_start.weight"] = arr(channels) * 0.1 + 1.0
    params[f"{engine}.norm_start.bias"] = arr(channels) * 0.1
    params[f"{engine}.norm_mix.weight"] = arr(channels) * 0.1 + 1.0
    params[f"{engine}.norm_mix.bias"] = arr(channels) * 0.1
    params[f"{engine}.proj_bundle.weight"] = arr(
        4 * channels, channels, dtype=jnp.bfloat16
    )
    params[f"{engine}.proj_emit.weight"] = arr(channels, channels, dtype=jnp.bfloat16)
    params[f"{engine}.proj_gate.weight"] = arr(channels, channels, dtype=jnp.bfloat16)
    return params


def _ulp(reference: np.ndarray) -> float:
    """One bfloat16 ULP at the magnitude of `reference`.

    A blocked native block rounds to bfloat16 at every linear, so its floor is
    the format and a decimal tolerance would be either unmeetable or
    meaningless -- the same unit `tests/models/esmfold2/test_context_parallel`
    states its grid tolerance in.
    """
    scale = float(np.abs(reference).max())
    return 2.0 ** (np.floor(np.log2(scale)) - 7)


@pytest.mark.parametrize("outgoing", [True, False])
@pytest.mark.parametrize("tokens", [11, 12])
def test_the_blocked_triangle_prologue_matches_the_whole_one(outgoing, tokens):
    """The streamed native triangle against the same call in one piece.

    11 rows in blocks of 4 leaves 3, which is the trailing-block case every
    other blocked path in this file also keeps an arm for. Both contraction
    directions are run because the streamed form is not symmetric: it cuts
    the output on `i` outgoing and on `j` incoming, and holds the other
    operand whole, so a direction whose cut and whose whole operand disagree
    fails on exactly one of the two.

    Asserted to a bfloat16 ULP rather than bitwise: a blocked shape reaches a
    different GEMM tiling, which is the caveat every blocked path in this
    repository carries. Measured 0.0 ULP against the unblocked arm at these
    sizes and 1.0 at 65 tokens in blocks of 64, where the tiling does change.
    """
    channels = 8
    params = _random_triangle_params(channels, 0)
    rng = np.random.default_rng(0)
    pair = jnp.asarray(
        rng.normal(size=(1, tokens, tokens, channels), scale=0.5), jnp.bfloat16
    )
    keep = rng.random(tokens) > 0.15
    mask = jnp.asarray((keep[:, None] & keep[None, :])[None].astype(np.float32))

    def run():
        return np.asarray(
            trunk.triangle_multiplicative(
                pair,
                params,
                "t",
                outgoing=outgoing,
                mask=mask,
                native_autocast=True,
            ),
            dtype=np.float32,
        )

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(trunk, "_AUTOCAST_ROWS", 10**6)
        whole = run()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(trunk, "_AUTOCAST_ROWS", 4)
        blocked = run()

    assert blocked.shape == whole.shape
    assert float(np.abs(whole).max()) > 0.0
    assert float(np.abs(whole - blocked).max()) <= 8 * _ulp(whole)


def _full_width_values(rows, tokens, channels, params, outgoing=True):
    """Every pair-shaped `[b, N, N, *]` value one native triangle call emits.

    `_autocast_linear` and `_autocast_norm` hide their native arms inside a
    `platform_dependent`, so a compile on any one platform prunes exactly the
    branch whose buffers this is about; the jaxpr carries both, which makes
    this an upper bound on what any one platform allocates and a sound basis
    for a before/after comparison.
    """
    pair = jax.ShapeDtypeStruct((1, tokens, tokens, channels), jnp.bfloat16)
    mask = jax.ShapeDtypeStruct((1, tokens, tokens), jnp.float32)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(trunk, "_AUTOCAST_ROWS", rows)
        jaxpr = jax.make_jaxpr(
            lambda p, m: trunk.triangle_multiplicative(
                p, params, "t", outgoing=outgoing, mask=m, native_autocast=True
            )
        )(pair, mask)
    found: collections.Counter = collections.Counter()

    def walk(eqns):
        for eqn in eqns:
            for var in eqn.outvars:
                aval = getattr(var, "aval", None)
                shape = getattr(aval, "shape", ())
                if len(shape) == 4 and shape[1] == shape[2] == tokens:
                    found[(aval.dtype, shape[3])] += 1
            for value in eqn.params.values():
                for inner in _sub_jaxprs(value):
                    walk(inner.eqns)

    walk(jaxpr.jaxpr.eqns)
    return found


@pytest.mark.parametrize("outgoing", [True, False])
def test_the_streamed_triangle_leaves_two_full_width_values(outgoing):
    """The census the block exists for, read off the jaxpr.

    The contraction sums over `k`, so one operand is read whole by every
    block of the output and cannot be cut; the count this pins is therefore
    two and not one. They are that whole operand -- `right` outgoing, `left`
    incoming -- and the pair update the epilogue writes, both bfloat16 and
    both the pair's own width.

    Everything else the call used to leave at full width is gone: the other
    operand, the output gate, the contraction's concatenated destination, the
    float32 `norm_mix` output and the `proj_emit` chain that read it. At
    2,112 tokens and 256 channels the same census reads 11 bfloat16 values
    (23,958 MiB) and 10 float32 (43,560 MiB) before, 2 bfloat16
    (4,356 MiB) and no float32 after.
    """
    tokens, channels = 12, 8
    params = _triangle_params(channels)
    whole = _full_width_values(10**6, tokens, channels, params, outgoing)
    streamed = _full_width_values(4, tokens, channels, params, outgoing)

    # Unblocked, the prologue's own widths are there to be removed.
    bf16 = jnp.dtype(jnp.bfloat16)
    assert {(bf16, 2 * channels), (bf16, 4 * channels)} <= set(whole)
    # Streamed, two values of the pair's own width and nothing else at all --
    # no float32 pair tensor, and nothing wider than the pair.
    assert streamed == {(bf16, channels): 2}, streamed
    assert sum(whole.values()) > sum(streamed.values())


@pytest.mark.parametrize("channels", [8, 256])
@pytest.mark.parametrize("rows", [10**6, 4])
def test_the_narrow_pair_norms_change_no_value(channels, rows):
    """`norm_start` and `norm_mix` stored bfloat16 is the same block, bit for bit.

    Their only consumers are `proj_bundle`, `proj_gate` and `proj_emit`, and
    `_autocast_linear` rounds its input to bfloat16 before the GEMM, so the
    narrowed store differs from the float32 one by which side of one
    round-nearest-even convert it happens on. Both selection branches of
    `_autocast_norm` are run -- 256 channels is the width that reaches the
    pinned CUDA reduction, 8 the one that keeps the ESM formula -- and both
    the streamed and the whole arrangement, since they normalise different
    shapes.

    A fresh closure and cleared caches per arm: `jax.jit` keyed on one shared
    function would answer the second arm from the first arm's program, and the
    comparison would be of that program with itself. The tripwire arm is what
    says the patch fires at all.
    """
    params = _random_triangle_params(channels, 3)
    rng = np.random.default_rng(4)
    pair = jnp.asarray(
        rng.normal(size=(1, 12, 12, channels), scale=0.5), jnp.bfloat16
    )
    mask = jnp.ones((1, 12, 12), jnp.float32)
    original = trunk._autocast_norm

    def arm(rewrite):
        def run(z, m):
            return trunk.triangle_multiplicative(
                z, params, "t", outgoing=True, mask=m, native_autocast=True
            )

        jax.clear_caches()
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(trunk, "_AUTOCAST_ROWS", rows)
            patch.setattr(trunk, "_autocast_norm", rewrite)
            value = jax.jit(run)(pair, mask)
        return np.asarray(value, np.float32)

    def widened(x, p, name, eps=1e-5, *, out_dtype=jnp.float32):
        del out_dtype
        return original(x, p, name, eps)

    def doubled(x, p, name, eps=1e-5, *, out_dtype=jnp.float32):
        result = original(x, p, name, eps, out_dtype=out_dtype)
        return result * 2 if name.endswith(".norm_start") else result

    shipped = arm(original)
    assert float(np.abs(shipped).max()) > 0.0
    np.testing.assert_array_equal(shipped, arm(widened))
    assert not np.array_equal(shipped, arm(doubled))


def test_the_output_gate_is_taken_from_the_normalised_input():
    """`proj_gate` reads `norm_start`'s output, not the contraction's.

    Moving `proj_gate` into the prologue is what lets the float32
    normalisation stop straddling the contraction, and it is only sound
    because the gate never depended on the contraction in the first place --
    which is one of the three facts this module's docstring opens with.
    Changing the contraction must therefore not change the gate.
    """
    channels = 8
    params = _random_triangle_params(channels, 1)
    pair = jnp.asarray(
        np.random.default_rng(2).normal(size=(1, 6, 6, channels), scale=0.5),
        jnp.bfloat16,
    )

    engine = "t._engine"
    normalized = trunk._autocast_norm(pair, params, f"{engine}.norm_start")
    expected = jax.nn.sigmoid(
        trunk._autocast_linear(normalized, params, f"{engine}.proj_gate").astype(
            jnp.float32
        )
    ).astype(jnp.bfloat16)

    _, _, gate = trunk._triangle_prologue(pair, params, engine, None, 1e-5)
    np.testing.assert_array_equal(np.asarray(gate), np.asarray(expected))
    assert gate.dtype == jnp.bfloat16
    assert float(np.abs(np.asarray(gate, np.float32) - 0.5).max()) > 0.0


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
