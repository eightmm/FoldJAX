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


def _full_width_values(rows, tokens, channels, params, outgoing=True, threaded=False):
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
            lambda p, m: trunk._triangle_multiplicative(
                p,
                params,
                "t",
                outgoing=outgoing,
                mask=m,
                eps=1e-5,
                native_autocast=True,
                workspace=_workspace(p, params) if threaded else None,
            )[0]
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

    This is the arm with no buffer to write into, where both blocked stages
    end in a `concatenate` and each full-width value is written once. The
    rolled arm is `test_the_rolled_triangle_writes_into_the_two_it_was_lent`
    below: the same two buffers, spelled six times, because a loop names its
    destination as the value it starts from, the slice write its body
    returns, and its own carry output.
    """
    tokens, channels = 12, 8
    params = _triangle_params(channels)
    whole = _full_width_values(10**6, tokens, channels, params, outgoing)
    streamed = _full_width_values(4, tokens, channels, params, outgoing)

    # Unblocked, the prologue's own widths are there to be removed.
    bf16 = jnp.dtype(jnp.bfloat16)
    assert {(bf16, 2 * channels), (bf16, 4 * channels)} <= set(whole)
    # Streamed, the pair's own width and nothing else at all -- no float32
    # pair tensor, and nothing wider than the pair.
    assert set(streamed) == {(bf16, channels)}, streamed
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


# --- the blocks are a loop, not thirty-three copies of the body -------------
#
# Every blocked stage above used to spell its blocks as a Python `for`, so
# each was traced and compiled separately. At 2,096 tokens and 64 rows that is
# 33 copies of the body per call site per layer: the released trunk program
# reached 969,694 HLO lines and 62 minutes of GPU compile, against well under
# 20 before the blocks were introduced. `trunk._assemble_row_blocks` runs them
# as a `fori_loop` whose body is traced once, with the trailing block -- which
# is shorter whenever the axis does not divide -- traced once more after it.
#
# It only does so for a stage that has a buffer to write into, because a
# `while`'s initial value, loop parameter and result are one colocated
# allocation XLA never lets another value reuse: a loop that allocates its own
# destination keeps a full-width buffer for the whole program, and five of
# those a layer is 10,730 MiB at 2,096 tokens against the 1,573 MiB a layer
# the separate slices cost. The tests below pin both halves: the value is the
# value the separate blocks produced, and the program grows with neither the
# token count nor the layer count.


def _workspace(pair, params):
    """The two block-loop destinations `folding_trunk` threads, for one call.

    Spelled here rather than called through `trunk._streamed_workspace`
    because the arms below lend the same buffers to the rolled and to the
    separate-slice program, and the trunk's helper declines to make them for
    the second on purpose. That the two agree is asserted on its own in
    `test_the_trunk_lends_exactly_the_two_buffers_the_loops_write`.
    """
    latent = params["t._engine.proj_bundle.weight"].shape[0] // 4
    return (
        jnp.zeros(pair.shape[:-1] + (latent,), jnp.bfloat16),
        jnp.zeros(pair.shape, jnp.bfloat16),
    )


def test_the_trunk_lends_exactly_the_two_buffers_the_loops_write():
    """`_streamed_workspace` makes the operand's and the update's, or none."""
    params = _random_triangle_params(8, 254)
    pair = jnp.zeros((1, 254, 254, 8), jnp.bfloat16)
    made = trunk._streamed_workspace(pair, params, "t._engine")
    assert made is not None
    assert [(part.shape, part.dtype) for part in made] == [
        (part.shape, part.dtype) for part in _workspace(pair, params)
    ]
    # Nothing to lend where there is nothing to roll: no native projection to
    # read the operand width off, and no mesh-free local axis to roll on.
    assert trunk._streamed_workspace(pair, {}, "t._engine") is None
    with pytest.MonkeyPatch.context() as patch:
        _unrolled(patch)
        assert trunk._streamed_workspace(pair, params, "t._engine") is None


def _unrolled(patch):
    """The arm that keeps the separate static slices and the `concatenate`.

    It is the shipped code's own other branch -- the one a globally sharded
    blocked axis takes -- rather than a reimplementation beside it, so the
    comparison is against the arrangement this change replaced and not
    against a second copy of it that could drift from it.
    """
    patch.setattr(trunk, "blocks_are_local", lambda: False)


def _bits(value):
    array = np.asarray(jax.device_get(value))
    return array.view(np.uint16) if array.dtype == jnp.bfloat16 else array


def _arm(run, *args, unrolled=False):
    jax.clear_caches()
    with pytest.MonkeyPatch.context() as patch:
        if unrolled:
            _unrolled(patch)
        return jax.jit(run)(*args)


@pytest.mark.parametrize("tokens", [254, 499])
def test_the_rolled_transition_is_bit_identical_to_the_separate_blocks(tokens):
    """64-row transition blocks, rolled and unrolled, on an axis 64 does not
    divide -- 254 is three blocks and 62, 499 is seven and 51.

    Bitwise and not to a tolerance: the loop runs the same block sizes in the
    same order over the same slices, and writing each block back over the rows
    it was read from changes where the result lands, not what it is.
    """
    rng = np.random.default_rng(tokens)
    x = jnp.asarray(rng.normal(size=(1, tokens, 12, 8), scale=0.5), jnp.bfloat16)
    params = {
        "t.norm.weight": jnp.asarray(rng.normal(size=8) * 0.1 + 1.0, jnp.float32),
        "t.norm.bias": jnp.asarray(rng.normal(size=8) * 0.1, jnp.float32),
        "t.ffn.w12.weight": jnp.asarray(rng.normal(size=(16, 8)), jnp.bfloat16),
        "t.ffn.w3.weight": jnp.asarray(rng.normal(size=(8, 8)), jnp.bfloat16),
    }

    def run(value):
        return trunk._autocast_transition(value, params, "t", True, 1e-5)

    rolled = _arm(run, x)
    separate = _arm(run, x, unrolled=True)
    assert float(np.abs(np.asarray(separate, np.float32)).max()) > 0.0
    np.testing.assert_array_equal(_bits(rolled), _bits(separate))


@pytest.mark.parametrize("outgoing", [True, False])
@pytest.mark.parametrize("tokens", [254, 499])
def test_the_rolled_streamed_triangle_agrees_to_the_format(tokens, outgoing):
    """One native block, rolled against the separate blocks, to the format.

    Bitwise in the incoming direction at both sizes, and at most two `_ulp`
    -- four bfloat16 steps at the magnitude of the output -- in the outgoing
    one. The blocks, their sizes, their order and their operands are the
    same; what differs is XLA:CPU's choice of whether to contract a multiply
    and an add into an FMA, which it makes differently for straight-line code
    and for a loop body. Measured on the smallest program that shows it,
    `x * g + b` over one array with no blocking of any kind: 16,550 of 65,536
    float32 elements move by one rounding of the multiply-add, and the loop's
    answer is the closer of the two to the float64 value.

    The tolerance is this file's own unit, and the reference for its size is
    the port's existing sensitivity to the block itself: on the unmodified
    tree, a two-layer native trunk at 254 tokens moves by 12 of these units
    when `_AUTOCAST_ROWS` goes from 64 to 63, and by 9 between blocked and
    unblocked. End to end it moves nothing measurable -- the 1UBQ CPU parity
    residual reads 0.025820 A on both trees.
    """
    params = _random_triangle_params(8, tokens)
    rng = np.random.default_rng(tokens + 1)
    pair = jnp.asarray(rng.normal(size=(1, tokens, tokens, 8), scale=0.5), jnp.bfloat16)
    keep = rng.random(tokens) > 0.15
    mask = jnp.asarray((keep[:, None] & keep[None, :])[None].astype(np.float32))

    def run(p, m):
        return trunk._triangle_multiplicative(
            p,
            params,
            "t",
            outgoing=outgoing,
            mask=m,
            eps=1e-5,
            native_autocast=True,
            workspace=_workspace(p, params),
        )[0]

    rolled = np.asarray(_arm(run, pair, mask), np.float32)
    separate = np.asarray(_arm(run, pair, mask, unrolled=True), np.float32)
    assert float(np.abs(separate).max()) > 0.0
    assert float(np.abs(rolled - separate).max()) <= 2 * _ulp(separate)


@pytest.mark.parametrize("outgoing", [True, False])
def test_the_rolled_triangle_writes_into_the_two_it_was_lent(outgoing):
    """The rolled census: still two buffers, and still only the pair's width.

    Six jaxpr values, two allocations. A loop names its destination three
    times -- the value it starts from, the slice write its body returns, and
    its own carry output -- and there are two loops: the whole operand's and
    the pair update's. Both start from a buffer the caller lent rather than
    from one they made, which is the whole point: a `while`'s allocation is
    one no other value may reuse, so the trunk lends the same two to every
    layer.

    Asserted at two token counts because a census that grew with them would
    be the unrolled program wearing the rolled program's name.
    """
    channels = 8
    params = _triangle_params(channels)
    bf16 = jnp.dtype(jnp.bfloat16)
    small = _full_width_values(4, 12, channels, params, outgoing, threaded=True)
    large = _full_width_values(4, 20, channels, params, outgoing, threaded=True)
    assert small == large, (small, large)
    assert set(small) == {(bf16, channels)}, small
    assert small == {(bf16, channels): 6}, small


@pytest.mark.parametrize("lend,rolled", [(True, True), (False, False)])
def test_the_loop_is_taken_only_when_it_is_lent_somewhere_to_write(lend, rolled):
    """The tripwire for every comparison above, and the memory contract.

    A loop that allocates its own destination keeps that buffer for the whole
    program, which is more than the separate slices cost rather than less, so
    the blocks are a loop only where the caller lends a buffer they are done
    with. Without it a rolled-versus-unrolled arm that took the
    separate-slice branch on both sides would be the same program twice, and
    would agree perfectly while proving nothing.
    """
    tokens = 254
    params = _random_triangle_params(8, tokens)
    pair = jax.ShapeDtypeStruct((1, tokens, tokens, 8), jnp.bfloat16)
    mask = jax.ShapeDtypeStruct((1, tokens, tokens), jnp.float32)

    def run(p, m):
        return trunk._triangle_multiplicative(
            p,
            params,
            "t",
            outgoing=True,
            mask=m,
            eps=1e-5,
            native_autocast=True,
            workspace=_workspace(p, params) if lend else None,
        )[0]

    jax.clear_caches()
    assert ("stablehlo.while" in jax.jit(run).lower(pair, mask).as_text()) is rolled
    jax.clear_caches()
    with pytest.MonkeyPatch.context() as patch:
        _unrolled(patch)
        assert "stablehlo.while" not in jax.jit(run).lower(pair, mask).as_text()


def test_two_blocks_are_not_worth_a_loop():
    """Below three blocks the separate slices *are* the rolled program.

    127 tokens is one block of 64 and a tail of 63; a one-trip `fori_loop` is
    the same body inside a `while` XLA has to prove runs once. The trunk's
    own helper declines to make the buffers at that size for the same reason.
    """
    params = _random_triangle_params(8, 127)
    pair = jnp.zeros((1, 127, 127, 8), jnp.bfloat16)
    assert trunk._streamed_workspace(pair, params, "t._engine") is None


def _trunk_params(channels):
    """One `pair_update_block`'s weights, spelled the way the stack reads them."""
    params = {}
    for direction in ("tri_mul_out", "tri_mul_in"):
        engine = f"blocks.0.{direction}._engine"
        for name, value in _random_triangle_params(channels, 0).items():
            params[engine + name[len("t._engine") :]] = value
    dot = "blocks.0.pair_transition"
    params[f"{dot}.norm.weight"] = jnp.ones((channels,))
    params[f"{dot}.norm.bias"] = jnp.zeros((channels,))
    params[f"{dot}.ffn.w12.weight"] = jnp.zeros((2 * channels, channels), jnp.bfloat16)
    params[f"{dot}.ffn.w3.weight"] = jnp.zeros((channels, channels), jnp.bfloat16)
    return params


def _dots(tokens, channels=8, unrolled=False):
    """`dot_general`s in the lowered native trunk at this token count."""
    pair = jax.ShapeDtypeStruct((1, tokens, tokens, channels), jnp.bfloat16)
    mask = jax.ShapeDtypeStruct((1, tokens, tokens), jnp.float32)
    params = _trunk_params(channels)
    jax.clear_caches()
    with pytest.MonkeyPatch.context() as patch:
        if unrolled:
            _unrolled(patch)
        text = (
            jax.jit(
                lambda p, m: trunk.folding_trunk(
                    p, params, n_layers=1, mask=m, native_autocast=True
                )
            )
            .lower(pair, mask)
            .as_text()
        )
    return text.count("stablehlo.dot_general")


def test_the_block_body_is_traced_once_however_many_blocks_there_are():
    """The program does not grow with the token count any more.

    Both sizes leave a trailing block -- 254 is three blocks of 64 and one of
    62, 499 is seven and 51 -- so both carry the loop body and the tail body,
    and the census is the same number. The unrolled arm is the tripwire: it
    is the arrangement this replaced, and there the count grows with the
    tokens, which is what a census that measured nothing would fail to show.
    """
    rolled = (_dots(254), _dots(499))
    assert rolled[0] == rolled[1], rolled
    # Two bodies per blocked stage rather than one per block: a constant, and
    # a small one. The exact number is the program's, so it is bounded here
    # rather than spelled, but it may not drift into the hundreds.
    assert rolled[0] <= 40, rolled

    separate = (_dots(254, unrolled=True), _dots(499, unrolled=True))
    assert separate[1] > separate[0], separate
    assert separate[0] > rolled[0], (separate, rolled)




# --- one pair of buffers for the whole forward, not one per trunk call ------
#
# The section above ends at a single `folding_trunk` call, where the chain of
# lent buffers begins and ends. A program that calls the trunk more than once
# began a chain at each: a `while`'s initial value, loop parameter and result
# are one colocated allocation XLA never lets another value reuse, so every
# call held its own pair for the whole program. This model makes four calls --
# the LM encoder and the trunk inside the recycle body, the parcae coda, the
# confidence head -- which is eight full-width buffers, 2,145 MiB each at
# 2,096 tokens.
#
# `trunk.LentBuffers` is the slot that carries one pair across them, and the
# tests below are its two halves: the arena law on a bare chain, where the
# buffers *are* the peak and the saving is exactly two pair widths a call, and
# the census on the released `predict`, where they are not and only the count
# says whether the threading reached every site.


def _pair_fills(text: str, shape: str) -> int:
    """Zero-fills of one full-width pair shape anywhere in a lowered program.

    A lent buffer is `jnp.zeros`, which lowers to a scalar constant broadcast
    with no dimensions mapped; a broadcast that *does* map dimensions is some
    operand being widened and is not one. Counted across every function in the
    module, because two of this model's four trunk calls are inside the
    recycle scan's body and one is inside the confidence sample loop's.
    """
    return sum(
        1
        for line in text.splitlines()
        if "stablehlo.broadcast_in_dim" in line
        and "dims = []" in line
        and line.rstrip().endswith(f"-> tensor<{shape}>")
    )


def _chain(calls: int, params, threaded: bool):
    def run(pair, mask):
        slot = trunk.LentBuffers() if threaded else None
        for _ in range(calls):
            pair = trunk.folding_trunk(
                pair,
                params,
                n_layers=1,
                mask=mask,
                native_autocast=True,
                workspace=slot,
            )
        return pair

    return run


@pytest.mark.parametrize("calls", [1, 2, 4])
def test_a_chain_of_trunk_calls_fills_two_buffers_however_many_it_is(calls):
    """Two zero-fills threaded, two per call unthreaded.

    The count is the whole contract: a lent buffer is the one value in this
    stack that is *allocated* rather than computed, so a program that fills
    more of them than the threading allows is a program holding more of them.
    The unthreaded arm is the tripwire -- it grows, which is what a census
    that measured nothing would fail to do.
    """
    channels, tokens = 8, 12
    params = _trunk_params(channels)
    pair = jax.ShapeDtypeStruct((1, tokens, tokens, channels), jnp.bfloat16)
    mask = jax.ShapeDtypeStruct((1, tokens, tokens), jnp.float32)
    shape = f"1x{tokens}x{tokens}x{channels}xbf16"
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(trunk, "_AUTOCAST_ROWS", 4)
        jax.clear_caches()
        threaded = jax.jit(_chain(calls, params, True)).lower(pair, mask).as_text()
        jax.clear_caches()
        alone = jax.jit(_chain(calls, params, False)).lower(pair, mask).as_text()
    assert _pair_fills(threaded, shape) == 2
    assert _pair_fills(alone, shape) == 2 * calls


def test_the_chain_costs_two_pair_widths_once_rather_than_once_a_call():
    """The arena law behind the count, on a program where the buffers are it.

    Measured at 128 tokens and 128 channels, one full-width pair being
    4.0 MiB. Unthreaded the temporary arena grows by two pair widths for
    every added call -- 5.51, 7.51, 9.51, 11.51 widths -- and threaded it
    grows by one width once, when the pair first has to outlive the call that
    made it, and then by nothing at all: 5.51, 6.51, 6.51, 6.51. That is the
    colocated allocation -- an initial value, loop parameter and result XLA
    never lets another value reuse -- being paid once for the whole chain
    instead of once at each call that starts one.

    The few hundred bytes either side of each step are the small per-call
    temporaries an extra layer carries; they are the same on both arms, which
    is why the law is asserted as a band around the pair width rather than as
    an equality.

    On the released `predict` this saving does not reach the arena, because
    six of the eight buffers it removes live inside the recycle scan's body
    and the confidence sample loop's, whose space XLA:CPU reuses after the
    loop, and the two that remain are co-tenants of offsets other values
    already hold. The count is therefore the contract that travels; the
    arena is this program, where nothing else is live to absorb it.
    """
    channels, tokens = 128, 128
    width = tokens * tokens * channels * 2
    slack = width // 1000
    params = _trunk_params(channels)
    pair = jax.ShapeDtypeStruct((1, tokens, tokens, channels), jnp.bfloat16)
    mask = jax.ShapeDtypeStruct((1, tokens, tokens), jnp.float32)

    def arena(calls, threaded):
        jax.clear_caches()
        compiled = jax.jit(_chain(calls, params, threaded)).lower(pair, mask).compile()
        return compiled.memory_analysis().temp_size_in_bytes

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(trunk, "_AUTOCAST_ROWS", 32)
        alone = [arena(calls, False) for calls in (1, 2, 3, 4)]
        threaded = [arena(calls, True) for calls in (1, 2, 3, 4)]
    assert threaded[0] == alone[0], (threaded, alone)
    steps_alone = [b - a for a, b in zip(alone[:-1], alone[1:], strict=True)]
    steps_threaded = [b - a for a, b in zip(threaded[:-1], threaded[1:], strict=True)]
    for step in steps_alone:
        assert 2 * width <= step < 2 * width + slack, (steps_alone, width)
    assert steps_threaded[0] < width + slack, (steps_threaded, width)
    for step in steps_threaded[1:]:
        assert step < slack, (steps_threaded, width)


def _released_predict_text(tokens: int, rows: int) -> str:
    """The whole released `predict`, lowered, with every trunk call reached.

    Real weights: the widths the lent buffers take come off the checkpoint's
    own `proj_bundle`, and a synthetic tree that happened to disagree between
    two call sites would make them separate buffers for a reason this test is
    not about. Layer counts are cut to one apiece because the census counts
    call sites and not layers, and the released 48 would only make it slow.

    The MSA stack is on. It is the fifth call site that runs the streamed
    contraction at the pair's own width, and it is reached only when the
    features carry an alignment, so a fixture without one would leave it out
    of the count it is here to be in.
    """
    import dataclasses

    from foldjax.models.esmfold2.bridge import checkpoint
    from foldjax.models.esmfold2.models import model as structure_model
    from foldjax.paths import weights_dir

    directory = weights_dir("esmfold2")
    if not (directory / checkpoint.WEIGHTS_NAME).exists():
        pytest.skip("esmfold2 weights are not in the store")
    parameters = checkpoint.load_parameters(directory)
    settings = dataclasses.replace(
        checkpoint.load_settings(directory),
        trunk_n_layers=1,
        lm_encoder_n_layers=1,
        coda_n_layers=1,
        confidence_n_layers=1,
        msa_n_layers=1,
        num_recycles=1,
        # Above one so the confidence head takes its sequential sample loop,
        # which is the fourth call site and the only one inside a second scan.
        num_samples=2,
    )
    settings = dataclasses.replace(
        settings, diffusion=dataclasses.replace(settings.diffusion, num_steps=2)
    )
    atoms = tokens * 3
    rng = np.random.default_rng(0)
    index = np.arange(tokens, dtype=np.int64)[None]
    features = {
        "token_index": index,
        "residue_index": index.copy(),
        "asym_id": (np.arange(tokens, dtype=np.int64) // (tokens // 2))[None],
        "sym_id": np.zeros((1, tokens), dtype=np.int64),
        "entity_id": (np.arange(tokens, dtype=np.int64) // (tokens // 2))[None],
        "mol_type": np.zeros((1, tokens), dtype=np.int64),
        "res_type": rng.integers(0, 20, (1, tokens)).astype(np.int64),
        "token_bonds": np.zeros((1, tokens, tokens, 1), dtype=np.float32),
        "token_attention_mask": np.ones((1, tokens), dtype=np.float32),
        "ref_pos": rng.standard_normal((1, atoms, 3)).astype(np.float32),
        "ref_element": np.full((1, atoms), 6, dtype=np.int64),
        "ref_charge": np.zeros((1, atoms), dtype=np.float32),
        "ref_atom_name_chars": np.zeros((1, atoms, 4), dtype=np.int64),
        "ref_space_uid": (np.arange(atoms, dtype=np.int64) // 3)[None],
        "atom_attention_mask": np.ones((1, atoms), dtype=np.float32),
        "atom_to_token": (np.arange(atoms, dtype=np.int64) // 3)[None],
        "distogram_atom_idx": (np.arange(tokens, dtype=np.int64) * 3)[None],
        "msa": rng.integers(0, 20, (1, 3, tokens)).astype(np.int64),
        "msa_attention_mask": np.ones((1, 3, tokens), dtype=np.float32),
        "has_deletion": np.zeros((1, 3, tokens), dtype=np.float32),
        "deletion_value": np.zeros((1, 3, tokens), dtype=np.float32),
    }
    hidden = np.zeros((1, tokens, 81, 2560), dtype=np.float32)

    def run(key, arrays, weights):
        return structure_model.predict(
            key,
            arrays,
            weights,
            settings=settings,
            lm_hidden_states=hidden,
            n_chains=2,
        )

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(trunk, "_AUTOCAST_ROWS", rows)
        jax.clear_caches()
        return jax.jit(run).lower(jax.random.key(0), features, parameters).as_text()


@pytest.mark.slow
def test_the_released_forward_fills_two_pair_buffers_and_not_eight():
    """The count on the model itself, at every site the threading has to reach.

    Five call sites -- the LM encoder, the MSA stack and the trunk inside the
    recycle scan's body, the parcae coda in the entry function, the confidence
    head inside the sequential sample loop -- and one pair of buffers between
    them. Eight is what this read before `predict` threaded a
    `trunk.LentBuffers` through the four `folding_trunk` calls, and it is what
    it reads again if any one call stops being handed the previous one's; four
    is what the MSA stack makes it if it allocates its own rather than reusing
    the ones the slot already holds.

    The three loops matter as much as the four calls: buffers a scan body
    allocates are that body's own, so the pair is made before the recycle scan
    and carried through it, and the confidence sample loop is spelled as the
    `scan` its `lax.map` already lowers to so that the pair can be its carry
    as well. A buffer closed over from outside a loop instead is invariant
    across its trips and cannot be written in place.
    """
    tokens, rows = 8, 4
    text = _released_predict_text(tokens, rows)
    assert _pair_fills(text, f"1x{tokens}x{tokens}x256xbf16") == 2


def test_a_lent_pair_is_reused_only_where_it_is_the_shape_the_loops_write():
    """The shape check is load-bearing, not a formality.

    `confidence_sample_sequential` is on by default and the head's trunk then
    sees the same `[1, N, N, C]` pair the rest of the stack does. Turned off,
    the head batches every structure through at once and its pair carries a
    sample axis -- a buffer lent by the trunk is the wrong destination there,
    and this is where that is noticed.
    """
    params = _random_triangle_params(8, 254)
    pair = jnp.zeros((1, 254, 254, 8), jnp.bfloat16)
    made = trunk._streamed_workspace(pair, params, "t._engine")
    again = trunk._streamed_workspace(pair, params, "t._engine", made)
    assert all(a is b for a, b in zip(made, again, strict=True))

    spread = jnp.zeros((2, 254, 254, 8), jnp.bfloat16)
    fresh = trunk._streamed_workspace(spread, params, "t._engine", made)
    assert trunk.buffer_spec(fresh) == trunk.buffer_spec(
        trunk._streamed_workspace(spread, params, "t._engine")
    )
    assert trunk.buffer_spec(fresh) != trunk.buffer_spec(made)


# --- the fifth call site: the MSA encoder's own triangle updates -----------
#
# `embedders.msa_encoder_block` runs the same streamed contraction on the same
# `[1, N, N, 256]` pair the folding trunk does, and it was the one stack the
# `trunk.LentBuffers` slot never reached. Without a buffer to write into,
# `_assemble_row_blocks` keeps the separate slices and the `concatenate`, so
# every block held its row blocks and their assembled result live at once --
# at 2,096 tokens that is what put 33 live `bf16[1, 64, 2096, 256]` blocks
# (65.5 MiB each) beside a `bf16[1, 256, 2096, 2096]` concatenate result in
# the GPU peak-live set, under `embedders.py:167`.
#
# Lent the slot the trunk already carries, the stack writes into those two
# buffers instead. Measured on the shipped program, compiled by the GPU
# probe's own recipe on CPU: `memory_analysis().temp_size_in_bytes` reads
# 4,634,957,824 -> 4,036,853,696 bytes at 254 tokens and
# 107,319,837,680 -> 101,398,769,456 at 2,096, the latter 5,647.0 MiB, which
# is 2.63 of that size's 2,145.1 MiB pair widths.


def _msa_stack(tokens: int, layers: int, separate: bool) -> dict[str, int]:
    """The released MSA stack, lowered, with its block loops rolled or not.

    Real weights for the same reason `_released_predict_text` uses them: the
    lent buffers take their width from the checkpoint's own `proj_bundle`.
    One block, which is `is_final` and so has no MSA-side submodules; the
    census is about the pair-side block loops and not about how many of them
    there are.
    """
    from foldjax.models.esmfold2.bridge import checkpoint
    from foldjax.models.esmfold2.models import embedders
    from foldjax.paths import weights_dir

    directory = weights_dir("esmfold2")
    if not (directory / checkpoint.WEIGHTS_NAME).exists():
        pytest.skip("esmfold2 weights are not in the store")
    params = {
        name: value
        for name, value in checkpoint.load_parameters(directory).items()
        if name.startswith("msa_encoder.")
    }
    depth = 4
    pair = jax.ShapeDtypeStruct((1, tokens, tokens, 256), jnp.bfloat16)

    def run(p):
        return embedders.msa_encoder(
            p,
            jnp.zeros((1, tokens, 451), jnp.bfloat16),
            jnp.zeros((1, tokens, depth, 33), jnp.bfloat16),
            jnp.zeros((1, tokens, depth), jnp.bfloat16),
            jnp.zeros((1, tokens, depth), jnp.bfloat16),
            jnp.ones((1, tokens, depth), jnp.float32),
            params,
            "msa_encoder",
            n_layers=layers,
            native_opm_params=params,
            workspace=trunk.LentBuffers(),
        )

    jax.clear_caches()
    with pytest.MonkeyPatch.context() as patch:
        if separate:
            _unrolled(patch)
        text = jax.jit(run).lower(pair).as_text()
    return {
        "while": text.count("stablehlo.while"),
        "concatenate": text.count("stablehlo.concatenate"),
        "dot_general": text.count("stablehlo.dot_general"),
    }


@pytest.mark.slow
def test_the_msa_stack_writes_into_the_buffers_instead_of_concatenating():
    """The census at two token counts, with the arrangement it replaced beside it.

    Rolled, the stack carries four `while`s -- the whole operand's loop and
    the update's loop, in each of the block's two directions -- and its
    `concatenate` count does not move between 254 and 499 tokens, because the
    body is traced once whatever the block count is. Separate, there is no
    `while` at all and the count grows with the tokens, which is the
    arrangement whose live row blocks the GPU peak-live set named.

    The separate arm is the tripwire and it is the shipped code's own other
    branch -- the one a globally sharded blocked axis takes -- rather than a
    reimplementation, so a census that measured nothing would show it by
    reading the same number twice.
    """
    rolled = (_msa_stack(254, 1, False), _msa_stack(499, 1, False))
    separate = (_msa_stack(254, 1, True), _msa_stack(499, 1, True))

    assert rolled[0]["while"] == rolled[1]["while"] > 0, rolled
    assert rolled[0]["concatenate"] == rolled[1]["concatenate"], rolled

    assert separate[0]["while"] == separate[1]["while"] == 0, separate
    assert separate[1]["concatenate"] > separate[0]["concatenate"], separate
    for size in (0, 1):
        assert rolled[size]["concatenate"] < separate[size]["concatenate"], (
            rolled,
            separate,
        )
        assert rolled[size]["dot_general"] < separate[size]["dot_general"], (
            rolled,
            separate,
        )
