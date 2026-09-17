import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.esmfold2.models import trunk


def test_native_linear_retains_bf16_output_boundary():
    x = jnp.ones((2, 4), jnp.float32)
    weight = jnp.ones((3, 4), jnp.float32)
    traced = jax.make_jaxpr(trunk._native_bf16_linear)(x, weight)
    dot = next(e for e in traced.jaxpr.eqns if e.primitive.name == "dot_general")
    assert dot.outvars[0].aval.dtype == jnp.bfloat16
    assert dot.params["precision"] == (jax.lax.Precision.DEFAULT,) * 2
    assert traced.jaxpr.eqns[-1].primitive.name == "optimization_barrier"
    np.testing.assert_array_equal(
        jax.jit(trunk._native_bf16_linear)(x, weight), jnp.full((2, 3), 4)
    )


@pytest.mark.parametrize("rows,width", [(1, 1), (1, 7), (3, 1)])
def test_native_vector_linear_does_not_round_each_product(rows, width):
    x = jnp.broadcast_to(jnp.asarray([1.0078125, 1.0]), (rows, 2))
    weight = jnp.broadcast_to(jnp.asarray([1.0078125, -1.015625]), (width, 2))
    traced = jax.make_jaxpr(trunk._native_bf16_linear)(x, weight)
    dot = next(e for e in traced.jaxpr.eqns if e.primitive.name == "dot_general")
    assert dot.outvars[0].aval.dtype == jnp.float32
    actual = jax.jit(trunk._native_bf16_linear)(x, weight)
    np.testing.assert_array_equal(actual, jnp.full((rows, width), 2**-14))
    # Native autocast must first round both inputs, even when the rewritten
    # dot can otherwise keep their original FP32 values through multiplication.
    x = jnp.full((rows, 1), 1.0035)
    weight = jnp.full((width, 1), 1.0035)
    actual = jax.jit(trunk._native_bf16_linear)(x, weight)
    np.testing.assert_array_equal(actual, jnp.ones((rows, width)))


@pytest.mark.parametrize("cp,bias", [(False, False), (True, False), (False, True)])
def test_native_linear_cuda_route_is_scoped(monkeypatch, cp, bias):
    monkeypatch.setattr(trunk, "cp_mesh", lambda: object() if cp else None)
    calls = []

    def cuda(x, weight):
        calls.append("cuda")
        return jnp.full((*x.shape[:-1], weight.shape[0]), 7, jnp.bfloat16)

    monkeypatch.setattr(trunk, "_native_bf16_linear", cuda)
    monkeypatch.setattr(
        jax.lax, "platform_dependent", lambda *args, **routes: routes["cuda"](*args)
    )
    x = jnp.ones((2, 4))
    params = {"p.weight": jnp.ones((3, 4))}
    if bias:
        params["p.bias"] = jnp.ones(3)
    actual = trunk._autocast_linear(x, params, "p")
    selected = not cp and not bias
    assert calls == (["cuda"] if selected else [])
    expected = (
        jnp.full((2, 3), 7)
        if selected
        else trunk._autocast_linear_fallback(x, params, "p")
    )
    np.testing.assert_array_equal(actual, expected)


def test_native_linear_cpu_fallback_is_unchanged():
    rng = np.random.default_rng(41)
    with jax.default_device(jax.devices("cpu")[0]):
        x = jnp.asarray(rng.normal(size=(2, 7)), jnp.float32)
        params = {"p.weight": jnp.asarray(rng.normal(size=(3, 7)), jnp.float32)}
        actual = jax.jit(lambda a: trunk._autocast_linear(a, params, "p"))(x)
        expected = jax.jit(lambda a: trunk._autocast_linear_fallback(a, params, "p"))(x)
    assert actual.devices() == {jax.devices("cpu")[0]}
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "width,cp", [(256, False), (256, True), (128, False), (128, True)]
)
@pytest.mark.parametrize(
    "prefix", ["n", "msa_encoder.blocks.0.outer_product_mean.norm",
               "msa_encoder.blocks.0.msa_pair_weighted_averaging.norm_single",
               "msa_encoder.blocks.0.msa_transition.norm"]
)
def test_native_norm_cuda_selection_preserves_other_routes(
    monkeypatch, width, cp, prefix
):
    from foldjax.models import _cp
    from foldjax.models.boltz2.models.primitives import native_amp_norm

    monkeypatch.setattr(_cp, "cp_mesh", lambda: object() if cp else None)
    calls = []

    def cuda(x, weight, bias, eps, out_dtype=jnp.float32):
        assert x.dtype == weight.dtype == bias.dtype == jnp.float32
        assert eps == 1e-5
        calls.append("cuda")
        return (jnp.full_like(x, 7, out_dtype), None, None)

    monkeypatch.setattr(native_amp_norm, "_cuda_layer_norm", cuda)
    monkeypatch.setattr(
        jax.lax, "platform_dependent", lambda *args, **routes: routes["cuda"](*args)
    )
    x = jnp.arange(width, dtype=jnp.bfloat16)[None]
    params = {prefix + ".weight": jnp.ones(width), prefix + ".bias": jnp.zeros(width)}
    result = trunk._autocast_norm(x, params, prefix)
    assert result.dtype == jnp.float32
    if (width == 256 or prefix != "n") and not cp:
        assert calls == ["cuda"]
        np.testing.assert_array_equal(result, jnp.full_like(result, 7))
    else:
        assert not calls
        expected = trunk.layer_norm(
            x.astype(jnp.float32), params[prefix + ".weight"], params[prefix + ".bias"]
        )
        np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("width", [4, 256])
@pytest.mark.parametrize("jit", [False, True])
def test_the_narrowed_norm_is_one_rounding_and_not_two(width, jit):
    """`out_dtype` moves the rounding into the norm; it does not add one.

    The reduction and the affine are float32 on every route and `out_dtype`
    only chooses the width the result is stored at, so float32 storage
    followed by one round-nearest-even convert and a direct bfloat16 store
    are the same value -- the argument `amp_affine` records for its own
    parameter. Asserted bitwise, because "the same value" is the whole claim
    that lets the triangle's two pair norms hand `_autocast_linear` the
    bfloat16 it would otherwise round to itself.

    Both widths: 256 reaches the pinned CUDA reduction (whose CPU arm is the
    ESM formula), and 4 reaches the formula directly.
    """
    rng = np.random.default_rng(7)
    params = {
        "n.weight": jnp.asarray(rng.normal(size=width) * 0.1 + 1.0, jnp.float32),
        "n.bias": jnp.asarray(rng.normal(size=width) * 0.1, jnp.float32),
    }
    x = jnp.asarray(rng.normal(size=(3, width), scale=2.0), jnp.bfloat16)

    def wrap(f):
        return jax.jit(f) if jit else f

    direct = wrap(
        lambda a: trunk._autocast_norm(a, params, "n", out_dtype=jnp.bfloat16)
    )(x)
    twice = wrap(
        lambda a: trunk._autocast_norm(a, params, "n").astype(jnp.bfloat16)
    )(x)

    assert direct.dtype == jnp.bfloat16
    np.testing.assert_array_equal(direct, twice)
    # Not a comparison of two zeros, and not of two saturated stores either.
    assert 0.0 < float(np.abs(np.asarray(direct, np.float32)).max()) < np.inf


def test_the_narrow_out_dtype_is_stored_by_the_cuda_kernel_itself():
    """The Pallas store, read off the kernel's jaxpr rather than run.

    A `pallas_call` output is a real buffer and a following convert cannot be
    fused into it, so asking the kernel for bfloat16 is what removes the
    float32 buffer rather than shortening its life. Two things are pinned:
    the normalised output is bfloat16 while `mean` and `rstd` stay float32,
    and the affine FMA is still float32 with the convert after it -- a kernel
    that narrowed its operands instead would be a different reduction.
    """
    from foldjax.models.boltz2.models.primitives import native_amp_norm

    traced = jax.make_jaxpr(
        lambda x, s, b: native_amp_norm._cuda_layer_norm(
            x, s, b, 1e-5, out_dtype=jnp.bfloat16
        )
    )(jnp.zeros((2, 256), jnp.float32), jnp.ones(256), jnp.zeros(256))
    call = next(e for e in traced.jaxpr.eqns if e.primitive.name == "pallas_call")
    assert [v.aval.dtype for v in call.outvars] == [
        jnp.dtype(jnp.bfloat16),
        jnp.dtype(jnp.float32),
        jnp.dtype(jnp.float32),
    ]

    kernel = call.params["jaxpr"]
    kernel = getattr(kernel, "jaxpr", kernel)
    affine = [
        e for e in kernel.eqns if e.params.get("asm") == "fma.rn.f32 $0, $1, $2, $3;"
    ][-1]
    assert affine.outvars[0].aval.dtype == jnp.float32
    consumers = [e for e in kernel.eqns if affine.outvars[0] in e.invars]
    assert [e.primitive.name for e in consumers] == ["convert_element_type"]
    assert consumers[0].params["new_dtype"] == jnp.bfloat16


def test_native_norm_cpu_keeps_esm_formula():
    with jax.default_device(jax.devices("cpu")[0]):
        x = jnp.arange(256, dtype=jnp.bfloat16)[None]
        params = {"n.weight": jnp.ones(256), "n.bias": jnp.zeros(256)}
        expected = trunk.layer_norm(
            x.astype(jnp.float32), params["n.weight"], params["n.bias"]
        )
        result = jax.jit(lambda a: trunk._autocast_norm(a, params, "n"))(x)
    assert result.devices() == {jax.devices("cpu")[0]}
    np.testing.assert_allclose(result, expected, atol=1e-6, rtol=1e-6)


def _params():
    rng = np.random.default_rng(1)
    params = {}
    for direction in ("out", "in"):
        p = f"b.tri_mul_{direction}._engine"
        for norm in ("norm_start", "norm_mix"):
            params[f"{p}.{norm}.weight"] = jnp.full(4, 1.0031)
            params[f"{p}.{norm}.bias"] = jnp.full(4, 0.00037)
        for name, width in (("proj_bundle", 16), ("proj_gate", 4), ("proj_emit", 4)):
            params[f"{p}.{name}.weight"] = jnp.asarray(rng.normal(size=(width, 4)) / 4)
    params["b.pair_transition.norm.weight"] = jnp.full(4, 1.0031)
    params["b.pair_transition.norm.bias"] = jnp.zeros(4)
    for name, shape in (("w12", (32, 4)), ("w3", (4, 16))):
        params[f"b.pair_transition.ffn.{name}.weight"] = jnp.asarray(
            rng.normal(size=shape) / 4
        )
    return params


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("length", [3, 65])
def test_native_block_boundaries_and_chunk_tail(monkeypatch, compiled, length):
    params = _params()
    pair = jnp.asarray(
        np.random.default_rng(2).normal(size=(1, length, length, 4)), jnp.bfloat16
    )
    observed = []
    contractions = []
    original = trunk._autocast_norm
    einsum = jnp.einsum

    def contract(equation, left, right, **kwargs):
        contractions.append(
            (equation, left.shape, right.shape, left.dtype, right.dtype)
        )
        return einsum(equation, left, right, **kwargs)

    def norm(x, p, name, eps=1e-5, **kwargs):
        result = original(x, p, name, eps, **kwargs)
        observed.append((name, x.shape, x.dtype, result.dtype))
        assert p[name + ".weight"].dtype == jnp.float32
        return result

    monkeypatch.setattr(trunk, "_autocast_norm", norm)
    monkeypatch.setattr(jnp, "einsum", contract)

    def run(x):
        return trunk.pair_update_block(
            x, params, "b", mask=jnp.ones(x.shape[:-1]), native_autocast=True
        )

    out = (jax.jit(run) if compiled else run)(pair)
    assert out.dtype == jnp.bfloat16
    assert np.isfinite(np.asarray(out.astype(jnp.float32))).all()
    # Every norm here reads the bfloat16 pair state, and the width it *emits*
    # is a per-site choice rather than a rule: the two pair norms of the
    # triangle block hand their result to one linear that rounds it anyway and
    # are asked for bfloat16, and the transition's norm keeps the float32
    # native autocast leaves a LayerNorm at. A blanket assertion here is what
    # let the narrowing look like a policy change it is not.
    narrowed = (".norm_start", ".norm_mix")
    for name, _, inp, out in observed:
        assert inp == jnp.bfloat16, (name, inp)
        assert out == (jnp.bfloat16 if name.endswith(narrowed) else jnp.float32), (
            name,
            out,
        )
    assert any(name.endswith(narrowed) for name, _, _, _ in observed)
    transition = [
        shape[1] for name, shape, _, _ in observed if name.endswith("transition.norm")
    ]
    assert transition == ([64, 1] if length == 65 else [3])
    # Which operand carries the block differs by direction, because the
    # streamed contraction cuts the output on `i` outgoing and on `j`
    # incoming, and the operand the other index reads is passed whole: `left`
    # blocked on axis 1 against a whole `right` for `bikd,bjkd`, a whole
    # `left` against `right` blocked on axis 2 for `bkid,bkjd`. Reading the
    # left operand for both is how this test read 65 where it wanted 64.
    for equation, operand, axis in (
        ("bikd,bjkd->bijd", 0, 1),
        ("bkid,bkjd->bijd", 1, 2),
    ):
        selected = [
            (shapes[operand][axis], a, b)
            for eq, *shapes, a, b in contractions
            if eq == equation
        ]
        assert [size for size, _, _ in selected] == transition
        assert all(a == b == jnp.bfloat16 for _, a, b in selected)
        # The whole operand really is whole: a block on both would be a
        # partial contraction, and nothing here would say so.
        whole = [
            shapes[1 - operand][axis]
            for eq, *shapes, _, _ in contractions
            if eq == equation
        ]
        assert whole == [length] * len(selected)


def test_affine_is_not_prematurely_rounded():
    params = _params()
    x = jnp.arange(12, dtype=jnp.bfloat16).reshape(3, 4)
    name = "b.tri_mul_out._engine.norm_start"
    actual = trunk._autocast_norm(x, params, name)
    rounded = {k: v.astype(jnp.bfloat16) for k, v in params.items()}
    wrong = trunk._autocast_norm(x, rounded, name)
    assert actual.dtype == jnp.float32
    assert not np.array_equal(actual, wrong)


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("residual", [False, True])
def test_transition_empty_prefix_matches_named(compiled, residual):
    params = _params()
    prefix = "b.pair_transition"
    bare = {
        k.removeprefix(prefix + "."): v
        for k, v in params.items()
        if k.startswith(prefix + ".")
    }
    x = jnp.arange(24, dtype=jnp.bfloat16).reshape(1, 2, 3, 4) / 8

    def run(value, weights, name):
        return trunk.transition(
            value, weights, name, residual=residual, native_autocast=True
        )

    fn = jax.jit(run, static_argnums=2) if compiled else run
    np.testing.assert_array_equal(fn(x, bare, ""), fn(x, params, prefix))
