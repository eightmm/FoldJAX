"""Private module formula/scheduling tests, not native GPU admission."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.openfold3.bridge.torch_mapping import (
    map_triangle_attention,
    map_triangle_multiplication,
)
from foldjax.models.openfold3.models import native_triangle_ops as ops


def weights(width=64, attention=False):
    from bench.openbind_triangle_ops_probe import Case, parameter_shapes

    case = Case(
        "template" if width == 64 else "pairformer",
        "tri_att_start" if attention else "tri_mul_out",
        3,
    )
    rng = np.random.default_rng(771)
    state = {}
    for key, shape in parameter_shapes(case).items():
        x = (rng.normal(size=shape) * 0.1).astype(np.float32)
        if len(shape) == 1 and key.endswith("weight"):
            x += 1
        state[key] = x
    mapper = map_triangle_attention if attention else map_triangle_multiplication
    return mapper(state)


def inputs(n=3, c=64):
    rng = np.random.default_rng(816)
    z = rng.normal(size=(1, n, n, c)).astype(np.float32)
    i, j = np.indices((n, n))
    mask = (((i + 2 * j) % 5) != 2)[None].astype(np.float32)
    return jnp.asarray(z), jnp.asarray(mask)


def norm(x, p, native):
    mean = x.mean(-1, keepdims=True)
    var = (
        np.maximum((x * x).mean(-1, keepdims=True) - mean * mean, 1e-5)
        if native
        else ((x - mean) ** 2).mean(-1, keepdims=True) + 1e-5
    )
    return ((x - mean) / np.sqrt(var)) * np.asarray(p.weight) + np.asarray(p.bias)


def rne(x):
    # Independent arithmetic oracle: normalized binary significands have 11
    # retained bits. np.rint resolves ties to even for either sign.
    value = np.asarray(x, dtype=np.float32)
    significand, exponent = np.frexp(value.astype(np.float64))
    return np.ldexp(np.rint(significand * 2048) / 2048, exponent).astype(np.float32)


def lin(x, p):
    return x @ np.asarray(p.weight).T


def sigmoid(x, clamped=False):
    return 1 / (1 + np.exp(-np.clip(x, -20, 20) if clamped else -x))


def mul_formula(z, mask, p, outgoing):
    x = norm(np.asarray(z), p.layer_norm_in, True)
    a = (
        lin(x, p.linear_a_p)
        * sigmoid(lin(x, p.linear_a_g), True)
        * np.asarray(mask)[..., None]
    )
    b = (
        lin(x, p.linear_b_p)
        * sigmoid(lin(x, p.linear_b_g), True)
        * np.asarray(mask)[..., None]
    )
    contraction = np.einsum(
        "...ikc,...jkc->...ijc" if outgoing else "...kic,...kjc->...ijc", a, b
    )
    update = lin(norm(contraction, p.layer_norm_out, True), p.linear_z)
    return update * sigmoid(lin(x, p.linear_g), True)


@pytest.mark.parametrize(
    "n,chunk,half,expected",
    [
        (1, 256, True, ((0, 1),)),
        (17, 256, True, ((0, 9), (9, 17))),
        (437, 256, True, ((0, 219), (219, 437))),
        (513, 256, True, ((0, 256), (256, 257), (257, 513))),
        (17, 7, False, ((0, 7), (7, 14), (14, 17))),
    ],
)
def test_native_exact_chunk_ranges(n, chunk, half, expected):
    assert ops.chunk_ranges(n, chunk, split_half=half) == expected


@pytest.mark.parametrize("length,chunk", [(0, 1), (1, 0), (-1, 2), (1, True), (2, 1.5)])
def test_invalid_chunk_ranges(length, chunk):
    with pytest.raises(ValueError, match="positive integers"):
        ops.chunk_ranges(length, chunk)


@pytest.mark.parametrize("outgoing", [True, False])
@pytest.mark.parametrize("width", [64, 128])
def test_multiplication_interpret_formula_and_residual_contract(outgoing, width):
    z, mask = inputs(c=width)
    p = weights(width)
    call = functools.partial(
        ops.native_triangle_multiplication_update,
        outgoing=outgoing,
        interpret=True,
        chunk_size=2,
    )
    update = call(z, p, mask=mask)
    expected = mul_formula(z, mask, p, outgoing)
    np.testing.assert_allclose(update, expected, rtol=2e-4, atol=2e-5)
    residual = ops.native_triangle_multiplication_residual(
        z, p, outgoing=outgoing, mask=mask, interpret=True, chunk_size=2
    )
    np.testing.assert_allclose(residual, np.asarray(z) + expected, rtol=2e-4, atol=2e-5)
    np.testing.assert_allclose(
        jax.jit(call)(z, p, mask=mask), update, rtol=1e-5, atol=2e-6
    )
    np.testing.assert_array_equal(z, inputs(c=width)[0])


def attention_formula(z, mask, p, transpose_bias=False):
    x = norm(np.asarray(z), p.layer_norm, False)
    q, k, v = [
        lin(x, pp).reshape(*x.shape[:-1], 4, -1)
        for pp in (p.mha.linear_q, p.mha.linear_k, p.mha.linear_v)
    ]
    q = q / np.sqrt(np.float32(q.shape[-1]))
    scores = np.einsum("brqhd,brkhd->brhqk", q, k)
    scores += (1e9 * (np.asarray(mask) - 1))[:, :, None, None, :]
    bias = (
        lin(x, p.linear_z).transpose(0, 3, 2, 1)
        if transpose_bias
        else lin(x, p.linear_z).transpose(0, 3, 1, 2)
    )
    scores += bias[:, None]
    probability = np.exp(scores - scores.max(-1, keepdims=True))
    probability /= probability.sum(-1, keepdims=True)
    out = np.einsum("brhqk,brkhd->brqhd", probability, v).reshape(x.shape)
    return lin(out * sigmoid(lin(x, p.mha.linear_g)), p.mha.linear_o)


@pytest.mark.parametrize("n", [16, 17])
@pytest.mark.parametrize("transpose_bias", [False, True])
@pytest.mark.parametrize("width", [64, 128])
def test_attention_formula_exact_tails_and_asymmetric_bias(n, transpose_bias, width):
    z, mask = inputs(n, width)
    p = weights(width, attention=True)
    out = ops.native_triangle_attention_update(
        z, p, mask=mask, transpose_bias=transpose_bias, chunk_size=7, interpret=True
    )
    np.testing.assert_allclose(
        out, attention_formula(z, mask, p, transpose_bias), rtol=2e-4, atol=3e-6
    )


def test_attention_standard_norm_and_projection_route_not_native_fused(monkeypatch):
    z, mask = inputs(17)
    p = weights(attention=True)
    chunks = []

    def forbidden(*args, **kwargs):
        raise AssertionError("attention must not use native triangle-mul LN/linear")

    monkeypatch.setattr(ops, "native_layer_norm", forbidden)
    monkeypatch.setattr(ops, "native_linear_fused", forbidden)
    original = ops.native_attention_core

    def core(q, *args, **kwargs):
        chunks.append(q.shape[1])
        return original(q, *args, **kwargs)

    monkeypatch.setattr(ops, "native_attention_core", core)
    ops.native_triangle_attention_update(z, p, mask=mask, chunk_size=7, interpret=True)
    assert chunks == [7, 7, 3]
    monkeypatch.setattr(ops, "native_attention_core", forbidden)
    ops.native_triangle_attention_update(
        z[:, :16, :16], p, mask=mask[:, :16, :16], interpret=True
    )


def test_attention_starting_false_and_pairblock_end_are_explicit():
    z, mask = inputs(3)
    p = weights(attention=True)
    transposed_z, transposed_mask = z.swapaxes(1, 2), mask.swapaxes(1, 2)
    expected = attention_formula(transposed_z, transposed_mask, p).swapaxes(1, 2)
    actual = ops.native_triangle_attention_update(
        z, p, mask=mask, starting=False, interpret=True
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=3e-6)
    end = ops.native_triangle_attention_update(
        transposed_z, p, mask=transposed_mask, transpose_bias=True, interpret=True
    ).swapaxes(1, 2)
    assert np.max(np.abs(np.asarray(actual - end))) > 0.01


@pytest.mark.parametrize("attention", [True, False])
@pytest.mark.parametrize(
    "bad", ["batch", "rectangular", "dtype", "mask", "params", "eps", "cp"]
)
def test_unadmitted_contracts_rejected(attention, bad, monkeypatch):
    z, mask = inputs()
    p = weights(attention=attention)
    kwargs = {"mask": mask}
    if bad == "batch":
        z = jnp.concatenate([z, z])
    if bad == "rectangular":
        z = z[:, :, :2]
    if bad == "dtype":
        z = z.astype(jnp.bfloat16)
    if bad == "mask":
        kwargs["mask"] = mask.astype(jnp.bool_)
    if bad == "params":
        p = jax.tree.map(lambda x: x.astype(jnp.bfloat16), p)
    if bad == "eps":
        kwargs["eps"] = float("nan")
    if bad == "cp":
        monkeypatch.setattr(ops, "cp_mesh", lambda: object())
    fn = (
        ops.native_triangle_attention_update
        if attention
        else functools.partial(ops.native_triangle_multiplication_update, outgoing=True)
    )
    with pytest.raises(ValueError):
        fn(z, p, **kwargs)


def test_attention_does_not_change_global_matmul_precision():
    z, mask = inputs(3)
    with jax.default_matmul_precision("highest"):
        ops.native_triangle_attention_update(z, weights(attention=True), mask=mask)
        assert jax.config.jax_default_matmul_precision == "highest"


@pytest.mark.parametrize(
    "operator", ["mul_out", "mul_in", "attention_stock", "attention_native"]
)
def test_default_mask_matches_explicit_ones(operator):
    is_attention = operator.startswith("attention")
    z, _ = inputs(17 if operator == "attention_native" else 3)
    p = weights(attention=is_attention)
    call = (
        ops.native_triangle_attention_update
        if is_attention
        else functools.partial(
            ops.native_triangle_multiplication_update, outgoing=operator == "mul_out"
        )
    )
    implicit = call(z, p, interpret=True)
    explicit = call(z, p, mask=jnp.ones(z.shape[:-1], jnp.float32), interpret=True)
    np.testing.assert_array_equal(implicit, explicit)


@pytest.mark.parametrize("inf", [float("nan"), float("inf"), 0.0, -1.0])
def test_attention_rejects_nonfinite_or_nonpositive_mask_constant(inf):
    z, mask = inputs()
    with pytest.raises(ValueError, match="finite positive mask inf"):
        ops.native_triangle_attention_update(
            z, weights(attention=True), mask=mask, inf=inf, interpret=True
        )


@pytest.mark.parametrize("outgoing", [True, False])
def test_multiplication_numeric_halfway_split_inside_chunk(outgoing, monkeypatch):
    z, mask = inputs(5)
    p = weights()
    rows = []
    original = ops.native_linear_fused

    def capture(x, weight, *args, **kwargs):
        if weight is p.linear_b_g.weight:
            rows.append(x.shape[1 if outgoing else 2])
        return original(x, weight, *args, **kwargs)

    monkeypatch.setattr(ops, "native_linear_fused", capture)
    actual = ops.native_triangle_multiplication_update(
        z, p, outgoing=outgoing, mask=mask, chunk_size=2, interpret=True
    )
    assert rows == [2, 1, 2]
    np.testing.assert_allclose(
        actual, mul_formula(z, mask, p, outgoing), rtol=2e-4, atol=2e-5
    )


def test_negative_gate_tail_distinguishes_mul_clamp_from_ordinary_attention():
    from foldjax.models.openfold3.models.primitives import LayerNormParams, LinearParams

    # Positive tails already round to one in FP32. At -30 both probabilities
    # remain normal FP32 values, separated by exp(10), so atol must not hide it.
    identity = LinearParams(jnp.eye(64, dtype=jnp.float32))
    zero = LinearParams(jnp.zeros((64, 64), jnp.float32))
    negative_gate = LinearParams(-30 * identity.weight)
    constant_norm = LayerNormParams(jnp.zeros(64), jnp.ones(64))
    mul = weights()._replace(
        layer_norm_in=constant_norm,
        layer_norm_out=constant_norm,
        linear_a_p=identity,
        linear_b_p=identity,
        linear_a_g=zero,
        linear_b_g=zero,
        linear_g=negative_gate,
        linear_z=identity,
    )
    z, _ = inputs()
    multiplied = ops.native_triangle_multiplication_update(
        z, mul, outgoing=True, interpret=True
    )
    clamped = np.float32(1 / (1 + np.exp(np.float64(20))))
    np.testing.assert_allclose(multiplied, clamped, rtol=2e-6, atol=0)

    attention = weights(attention=True)
    attention = attention._replace(
        layer_norm=constant_norm,
        linear_z=LinearParams(jnp.zeros((4, 64), jnp.float32)),
        mha=attention.mha._replace(
            linear_q=zero,
            linear_k=zero,
            linear_v=identity,
            linear_g=negative_gate,
            linear_o=identity,
        ),
    )
    unclamped = np.float32(1 / (1 + np.exp(np.float64(30))))
    for length in (3, 17):
        z, _ = inputs(length)
        attended = ops.native_triangle_attention_update(z, attention, interpret=True)
        np.testing.assert_allclose(attended, unclamped, rtol=2e-6, atol=0)
        assert np.max(np.asarray(attended)) < clamped / 20_000


def test_runtime_rne_ties_grid_and_nonfinite_payloads_eager_and_jit():
    bits = np.array(
        [
            0x3F801000,
            0x3F803000,
            0xBF801000,
            0xBF803000,
            0x3F800FFF,
            0x3F801001,
            0x3F802000,
            0xBF802000,
            0,
            0x80000000,
            0x7F800000,
            0xFF800000,
            0x7F800001,
            0xFFC01234,
        ],
        np.uint32,
    )
    expected = np.array(
        [
            0x3F800000,
            0x3F804000,
            0xBF800000,
            0xBF804000,
            0x3F800000,
            0x3F802000,
            0x3F802000,
            0xBF802000,
            0,
            0x80000000,
            0x7F800000,
            0xFF800000,
            0x7F800001,
            0xFFC01234,
        ],
        np.uint32,
    )

    def output_bits(x):
        return jax.lax.bitcast_convert_type(ops._tf32_rne(x), jnp.uint32)

    value = jnp.asarray(bits.view(np.float32))
    np.testing.assert_array_equal(output_bits(value), expected)
    np.testing.assert_array_equal(jax.jit(output_bits)(value), expected)
    np.testing.assert_array_equal(output_bits(ops._tf32_rne(value)), expected)


@pytest.mark.parametrize("interpret", [False, True])
def test_ordinary_norm_selects_cuda_welford_only_outside_interpret(
    monkeypatch, interpret
):
    from foldjax.models.boltz2.models.primitives import native_amp_norm
    from foldjax.models.openfold3.models.primitives import LayerNormParams

    x = jnp.arange(128, dtype=jnp.float32)[None, None, None]
    params = LayerNormParams(jnp.ones(128), jnp.zeros(128))
    calls = []

    def cuda(value, weight, bias, eps):
        assert value is x and weight is params.weight and bias is params.bias
        assert eps == 1e-4
        calls.append(True)
        return jnp.full_like(value, 7), None, None

    monkeypatch.setattr(native_amp_norm, "_cuda_layer_norm", cuda)
    monkeypatch.setattr(
        jax.lax, "platform_dependent", lambda *args, **routes: routes["cuda"](*args)
    )
    actual = ops._ordinary_norm(x, params, 1e-4, interpret=interpret)
    expected = ops.layer_norm(x, params, eps=1e-4) if interpret else jnp.full_like(x, 7)
    np.testing.assert_array_equal(actual, expected)
    assert len(calls) == int(not interpret)


@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16, jnp.int32])
def test_runtime_rne_rejects_non_fp32(dtype):
    with pytest.raises(TypeError, match="requires FP32"):
        ops._tf32_rne(jnp.ones(3, dtype))


def test_runtime_rne_fp64_is_not_silently_narrowed():
    with pytest.raises(TypeError, match="requires FP32"):
        ops._tf32_rne(np.ones(3, np.float64))


def test_ordinary_linear_rounds_only_dot_inputs_not_bias_and_stays_high():
    from jax.extend import core

    from foldjax.models.openfold3.models.primitives import LinearParams

    values = np.array([0x3F801000, 0xBF803000], np.uint32).view(np.float32)
    x = jnp.asarray(values[None])
    p = LinearParams(jnp.asarray(np.diag(values)), jnp.asarray(values))
    expected = rne(np.asarray(x)) @ rne(np.asarray(p.weight)).T + values
    np.testing.assert_array_equal(ops._ordinary_linear(x, p), expected)
    np.testing.assert_array_equal(jax.jit(ops._ordinary_linear)(x, p), expected)
    formula = np.asarray(x) @ np.asarray(p.weight).T + values
    np.testing.assert_array_equal(ops._ordinary_linear(x, p, interpret=True), formula)
    assert np.max(np.abs(expected - formula)) > 1e-4
    graph = jax.make_jaxpr(ops._ordinary_linear)(x, p).jaxpr
    integer_bitcasts = [
        eq
        for eq in graph.eqns
        if eq.primitive.name == "bitcast_convert_type"
        and eq.params["new_dtype"] == np.dtype(np.uint32)
    ]
    assert [eq.invars[0] for eq in integer_bitcasts] == list(graph.invars[:2])

    def collect(g):
        dots = []
        for eq in g.eqns:
            if eq.primitive.name == "dot_general":
                dots.append(eq)
            for value in eq.params.values():
                if isinstance(value, core.ClosedJaxpr):
                    dots += collect(value.jaxpr)
                elif isinstance(value, core.Jaxpr):
                    dots += collect(value)
        return dots

    dots = collect(graph)
    assert len(dots) == 1
    assert dots[0].params["precision"] == (jax.lax.Precision.HIGH,) * 2


def test_stock_core_rounds_qk_and_pv_separately_without_rounding_bias(monkeypatch):
    rng = np.random.default_rng(450)
    q, k, v = [
        jnp.asarray(rng.normal(size=(1, 4, 3, 32)), jnp.float32) for _ in range(3)
    ]
    biases = (
        jnp.full((1, 1, 1, 3), -3.1251),
        jnp.asarray(rng.normal(size=(1, 4, 3, 3)), jnp.float32),
    )
    captured = []
    original = ops._ordinary_einsum

    def capture(equation, lhs, rhs, **kwargs):
        captured.append((equation, np.asarray(lhs), np.asarray(rhs)))
        return original(equation, lhs, rhs, **kwargs)

    monkeypatch.setattr(ops, "_ordinary_einsum", capture)
    actual = ops._stock_attention(q, k, v, biases)
    scores = np.einsum("...qc,...kc->...qk", rne(q), rne(k))
    for bias in biases:
        scores += np.asarray(bias)
    probability = np.exp(scores - scores.max(-1, keepdims=True))
    probability /= probability.sum(-1, keepdims=True)
    expected = np.einsum("...qk,...kc->...qc", rne(probability), rne(v))
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=2e-6)
    assert [x[0] for x in captured] == ["...qc,...kc->...qk", "...qk,...kc->...qc"]
    np.testing.assert_array_equal(captured[0][1], q)
    np.testing.assert_array_equal(captured[0][2], k)
    np.testing.assert_allclose(captured[1][1], probability, rtol=1e-5, atol=2e-6)
    np.testing.assert_array_equal(captured[1][2], v)


def test_stock_q_scaling_precedes_rne_dot_inputs(monkeypatch):
    z, mask = inputs(3, 128)
    p = weights(128, attention=True)
    captured = []

    def capture(q, k, v, biases, *, interpret):
        assert interpret is False
        captured.append(np.asarray(q))
        return jnp.zeros_like(v)

    monkeypatch.setattr(ops, "_stock_attention", capture)
    ops.native_triangle_attention_update(z, p, mask=mask)
    projected = ops._ordinary_linear(ops.layer_norm(z, p.layer_norm), p.mha.linear_q)
    expected = projected.reshape(1, 3, 3, 4, 32).swapaxes(-2, -3) / np.sqrt(32)
    np.testing.assert_array_equal(captured[0], expected)


@pytest.mark.parametrize(
    "operator,n,count", [("mul", 3, 4), ("stock", 3, 16), ("native", 17, 12)]
)
def test_rne_stays_out_of_existing_custom_triton_kernels(
    operator, n, count, monkeypatch
):
    z, mask = inputs(n)
    captured = []
    original = ops._tf32_rne

    def capture(value):
        captured.append(value.shape)
        return original(value)

    monkeypatch.setattr(ops, "_tf32_rne", capture)
    if operator == "mul":
        jax.make_jaxpr(
            lambda x, p: ops.native_triangle_multiplication_update(
                x, p, outgoing=True, mask=mask
            )
        )(z, weights())
        assert captured == [(1, 64, 3, 3), (1, 64, 3, 2), (1, 64, 3, 3), (1, 64, 3, 1)]
    else:
        jax.make_jaxpr(
            lambda x, p: ops.native_triangle_attention_update(x, p, mask=mask)
        )(z, weights(attention=True))
    assert len(captured) == count


@pytest.mark.parametrize("attention", [False, True])
def test_interpret_mode_explicitly_omits_runtime_rne(attention, monkeypatch):
    z, mask = inputs()

    def forbidden(*args):
        raise AssertionError("formula-only interpretation must omit RNE")

    monkeypatch.setattr(ops, "_tf32_rne", forbidden)
    if attention:
        ops.native_triangle_attention_update(
            z, weights(attention=True), mask=mask, interpret=True
        )
    else:
        ops.native_triangle_multiplication_update(
            z, weights(), outgoing=True, mask=mask, interpret=True
        )


@pytest.mark.parametrize(
    "operator", ["tri_mul_out", "tri_mul_in", "tri_att_start", "tri_att_end"]
)
def test_real_weight_probe_traces_default_runtime_not_interpretation(
    operator, monkeypatch
):
    from bench import openbind_triangle_ops_probe as probe

    case = probe.Case("template", operator, 17)
    z, mask = inputs(17)
    called = []
    for name in (
        "native_layer_norm",
        "native_linear",
        "native_linear_fused",
        "native_attention_core",
    ):
        original = getattr(ops, name)

        def capture(*args, _fn=original, **kwargs):
            called.append(kwargs["interpret"])
            return _fn(*args, **kwargs)

        monkeypatch.setattr(ops, name, capture)
    jax.make_jaxpr(lambda data, p: probe.call_candidate(ops, case, data, p))(
        {"z": z, "mask": mask}, weights(attention=not case.multiplication)
    )
    assert called and all(value is False for value in called)
