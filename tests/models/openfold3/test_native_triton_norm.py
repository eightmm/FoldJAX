"""CPU interpretation evidence, not native CUDA operator parity."""

from __future__ import annotations

import functools
import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.openfold3.models import native_triton_norm as norm
from foldjax.models.openfold3.models.primitives import LayerNormParams, layer_norm


def _native_formula(x, weight, bias, eps=1e-5):
    """Independent FP32 formula oracle, with the publisher's block boundaries."""
    shape = x.shape
    x = np.asarray(x, np.float32).reshape(-1, shape[-1])
    weight, bias = np.asarray(weight, np.float32), np.asarray(bias, np.float32)
    width = shape[-1]
    block_size = min(1024, 1 << (width - 1).bit_length())
    total = np.zeros((len(x), 1), np.float32)
    total_squared = np.zeros_like(total)
    for start in range(0, width, block_size):
        block = x[:, start : start + block_size]
        total += np.sum(block, axis=-1, keepdims=True)
        total_squared += np.sum(block * block, axis=-1, keepdims=True)
    mean = total / np.float32(width)
    variance = total_squared / np.float32(width) - mean * mean
    rstd = np.float32(1) / np.sqrt(np.maximum(variance, np.float32(eps)))
    return ((x - mean) * rstd * weight + bias).reshape(shape)


@pytest.mark.parametrize(
    "shape",
    [
        (1,),
        (1, 3),
        (7, 64),
        (8, 128),
        (9, 64),
        (17, 128),
        (2, 3, 129),
        (9, 1025),
        (2, 2049),
    ],
)
def test_interpret_native_formula_and_launch_tails(shape):
    rng = np.random.default_rng(8)
    x = jnp.asarray(rng.normal(size=shape), jnp.float32)
    weight = jnp.asarray(rng.normal(size=shape[-1]), jnp.float32)
    bias = jnp.asarray(rng.normal(size=shape[-1]), jnp.float32)
    actual = norm.native_layer_norm(x, weight, bias, interpret=True)
    expected = _native_formula(x, weight, bias)
    assert actual.shape == x.shape
    assert actual.dtype == x.dtype
    np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=3e-6)


@pytest.mark.parametrize("variance_ratio", [0.0, 0.25, 1.0, 4.0])
def test_epsilon_is_a_variance_floor_not_an_additive_term(variance_ratio):
    eps = 1e-5
    x = jnp.tile(jnp.array([-1, 1], jnp.float32), 64)[None]
    x *= np.float32(np.sqrt(eps * variance_ratio))
    weight, bias = jnp.ones(128), jnp.zeros(128)
    actual = norm.native_layer_norm(x, weight, bias, eps, interpret=True)
    expected = _native_formula(x, weight, bias, eps)
    np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=3e-6)
    if variance_ratio == 1.0:
        ordinary = layer_norm(x, LayerNormParams(weight, bias), eps=eps)
        assert float(jnp.max(jnp.abs(actual - ordinary))) > 0.29


def test_second_moment_cancellation_is_not_replaced_with_centered_variance():
    x = 10000 + jnp.tile(jnp.array([-0.03125, 0.03125]), 64)[None]
    weight, bias = jnp.ones(128), jnp.zeros(128)
    actual = norm.native_layer_norm(x, weight, bias, interpret=True)
    expected = _native_formula(x, weight, bias)
    np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=3e-6)
    ordinary = layer_norm(x, LayerNormParams(weight, bias))
    assert float(jnp.max(jnp.abs(actual - ordinary))) > 8


@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16, jnp.float32])
@pytest.mark.parametrize("affine_dtype", [jnp.float16, jnp.bfloat16, jnp.float32])
def test_storage_dtypes_preserve_fp32_moments_and_affine(dtype, affine_dtype):
    rng = np.random.default_rng(71)
    # Squaring these FP16 inputs without the native FP32 island overflows.
    x = jnp.asarray(rng.normal(size=(9, 64)) * 1000, dtype)
    weight = jnp.asarray(rng.normal(size=64), affine_dtype)
    bias = jnp.asarray(rng.normal(size=64), affine_dtype)
    actual = norm.native_layer_norm(x, weight, bias, interpret=True)
    expected = jnp.asarray(_native_formula(x, weight, bias), dtype)
    assert actual.dtype == dtype
    assert np.isfinite(np.asarray(actual, np.float32)).all()
    np.testing.assert_allclose(
        np.asarray(actual, np.float32),
        np.asarray(expected, np.float32),
        rtol=3e-6,
        atol=3e-6,
    )


def test_fp32_affine_is_not_downcast_to_input_storage_dtype():
    rng = np.random.default_rng(4)
    x = jnp.asarray(rng.normal(size=(9, 128)), jnp.bfloat16)
    weight = jnp.asarray(1 + rng.normal(size=128) * 0.001, jnp.float32)
    bias = jnp.asarray(rng.normal(size=128) * 0.0001, jnp.float32)
    actual = norm.native_layer_norm(x, weight, bias, interpret=True)
    expected = jnp.asarray(_native_formula(x, weight, bias), x.dtype)
    downcast = jnp.asarray(
        _native_formula(x, weight.astype(x.dtype), bias.astype(x.dtype)), x.dtype
    )
    np.testing.assert_array_equal(actual, expected)
    assert np.any(np.asarray(actual, np.float32) != np.asarray(downcast, np.float32))


def test_jit_interpret_uses_new_values_without_changing_program():
    call = jax.jit(functools.partial(norm.native_layer_norm, interpret=True))
    weight, bias = jnp.linspace(0.5, 1.5, 64), jnp.linspace(-0.2, 0.2, 64)
    for offset in (0.0, 2.0):
        x = jnp.arange(9 * 64, dtype=jnp.float32).reshape(9, 64) / 128 + offset
        actual = call(x, weight, bias)
        np.testing.assert_allclose(
            actual, _native_formula(x, weight, bias), rtol=3e-4, atol=3e-4
        )


def test_launch_configuration_and_separate_interpret_flag(monkeypatch):
    calls = []

    def pallas_call(kernel, **kwargs):
        calls.append((kernel, kwargs))
        return lambda x, *args: jnp.zeros_like(x)

    monkeypatch.setattr(norm.pl, "pallas_call", pallas_call)
    x, weight, bias = jnp.ones((17, 128)), jnp.ones(128), jnp.zeros(128)
    norm.native_layer_norm(x, weight, bias)
    kernel, kwargs = calls.pop()
    assert kernel.keywords["block_size"] == 128
    assert kwargs["grid"] == (3,)
    assert kwargs["compiler_params"].num_warps == 4
    assert kwargs["compiler_params"].num_stages == 3
    assert kwargs["interpret"] is False
    assert kwargs["name"] == "openbind_native_triangle_layer_norm"


def test_empty_rows_do_not_launch(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("empty arrays must not launch a kernel")

    monkeypatch.setattr(norm.pl, "pallas_call", forbidden)
    actual = norm.native_layer_norm(jnp.ones((2, 0, 64)), jnp.ones(64), jnp.zeros(64))
    assert actual.shape == (2, 0, 64)


@pytest.mark.parametrize("eps", [0, -1e-5, np.nan, np.inf, -np.inf, 1e-50, 1e50])
def test_rejects_invalid_fp32_epsilon(eps):
    with pytest.raises(ValueError, match="finite and positive in FP32"):
        norm.native_layer_norm(jnp.ones((2, 64)), jnp.ones(64), jnp.zeros(64), eps)


@pytest.mark.parametrize("eps", [True, None, "1e-5", jnp.array(1e-5)])
def test_rejects_nonstatic_epsilon(eps):
    with pytest.raises(TypeError, match="static real scalar"):
        norm.native_layer_norm(jnp.ones((2, 64)), jnp.ones(64), jnp.zeros(64), eps)


@pytest.mark.parametrize(
    "x,weight,bias,error",
    [
        (jnp.array(1.0), jnp.ones(1), jnp.zeros(1), "positive last dimension"),
        (jnp.ones((2, 0)), jnp.ones(0), jnp.zeros(0), "positive last dimension"),
        (jnp.ones((2, 64)), jnp.ones((1, 64)), jnp.zeros(64), "both have shape"),
        (jnp.ones((2, 64)), jnp.ones(64), jnp.zeros(63), "both have shape"),
    ],
)
def test_rejects_invalid_shapes(x, weight, bias, error):
    with pytest.raises(ValueError, match=error):
        norm.native_layer_norm(x, weight, bias)


@pytest.mark.parametrize("index", [0, 1, 2])
def test_rejects_integer_storage_before_launch(index):
    args = [jnp.ones((2, 64)), jnp.ones(64), jnp.zeros(64)]
    args[index] = args[index].astype(jnp.int32)
    with pytest.raises(TypeError, match="FP16, BF16 or FP32"):
        norm.native_layer_norm(*args)


def test_missing_pallas_does_not_silently_use_ordinary_norm(monkeypatch):
    monkeypatch.setattr(norm, "pt", None)
    with pytest.raises(RuntimeError, match="Pallas/Triton is required"):
        norm.native_layer_norm(jnp.ones((2, 64)), jnp.ones(64), jnp.zeros(64))


def test_nonfinite_input_is_not_sanitized():
    x = jnp.ones((9, 64)).at[0, 0].set(jnp.nan).at[8, 1].set(jnp.inf)
    actual = norm.native_layer_norm(x, jnp.ones(64), jnp.zeros(64), interpret=True)
    assert np.isnan(np.asarray(actual)[[0, 8]]).all()
    np.testing.assert_array_equal(np.asarray(actual)[1:8], 0)


def _emulate_fp32_asm(instruction, *values):
    """FP64 exactly represents these bounded FP32 products/sums before rounding."""
    values = [np.asarray(value, np.float64) for value in values]
    if instruction == "add.rn.f32":
        result = values[0] + values[1]
    elif instruction == "mul.rn.f32":
        result = values[0] * values[1]
    elif instruction == "fma.rn.f32":
        result = values[0] * values[1] + values[2]
    else:
        raise AssertionError(instruction)
    return jnp.asarray(np.asarray(result, np.float32))


@pytest.mark.parametrize(
    "width,expected_variance",
    [
        (
            64,
            [
                -11.531250953674316,
                3.5312490463256836,
                15.062496185302734,
                4.468749046325684,
                -4.468750953674316,
                3.5312490463256836,
                10.593741416931152,
                0.0,
                -7.062503814697266,
            ],
        ),
        (
            128,
            [
                7.062496185302734,
                -8.0,
                -3.5312509536743164,
                0.0,
                0.0,
                -12.468750953674316,
                23.062496185302734,
                7.062496185302734,
                8.0,
            ],
        ),
    ],
)
def test_explicit_fma_tree_matches_native_ptx_counterfactual(
    monkeypatch, width, expected_variance
):
    # Fixed operands from GPU jobs414/415's large_mean profile. These values
    # distinguish BOTH missing FMA boundaries; a centered-variance test cannot.
    rng = np.random.default_rng(7301 + width)
    x = np.float32(10000) + rng.normal(size=(9, width)).astype(np.float32) * np.float32(
        0.01
    )
    monkeypatch.setattr(norm, "_fp32_asm", _emulate_fp32_asm)
    actual = []
    unfused = []
    for row in x:
        total, squared = norm._native_block_moments(jnp.asarray(row))
        mean = np.float32(total) / np.float32(width)
        second = np.float32(squared) / np.float32(width)
        actual.append(float(_emulate_fp32_asm("fma.rn.f32", -mean, mean, second)))
        unfused.append(np.float32(second - np.float32(mean * mean)))
    np.testing.assert_array_equal(np.asarray(actual, np.float32), expected_variance)
    assert np.any(np.asarray(actual, np.float32) != np.asarray(unfused))


@pytest.mark.parametrize(
    "width,interpret,has_asm",
    [
        (64, False, True),
        (128, False, True),
        (64, True, False),
        (128, True, False),
        (129, False, False),
    ],
)
def test_only_observed_gpu_widths_trace_explicit_fma(width, interpret, has_asm):
    # Abstract tracing proves branch selection without launching or compiling
    # a CUDA program. Interpreter coverage remains explicitly formula-only.
    traced = jax.make_jaxpr(
        functools.partial(norm.native_layer_norm, interpret=interpret)
    )(jnp.ones((9, width)), jnp.ones(width), jnp.zeros(width))
    text = str(traced)
    assert ("fma.rn.f32" in text) is has_asm
    assert ("mul.rn.f32" in text) is has_asm
    assert ("add.rn.f32" in text) is has_asm
    assert ("sqrt.approx.ftz.f32" in text) is has_asm
    assert ("div.full.f32" in text) is has_asm


@pytest.mark.parametrize("width", [64, 128, 129])
@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16, jnp.float32])
def test_actual_triton_ir_preserves_sqrt_division_affine_without_compiler_patch(
    width, dtype
):
    from jax._src import sharding_impls
    from jax._src.interpreters import mlir
    from jax._src.pallas.triton import lowering, primitives

    registry = lowering.triton_lowering_rules
    original = registry[primitives.elementwise_inline_asm_p]
    graph = jax.make_jaxpr(norm.native_layer_norm)(
        jax.ShapeDtypeStruct((9, width), dtype),
        jax.ShapeDtypeStruct((width,), jnp.float32),
        jax.ShapeDtypeStruct((width,), jnp.float32),
    )
    call = next(e for e in graph.jaxpr.eqns if e.primitive.name == "pallas_call")
    context = mlir.ModuleContext(
        platforms=("cuda",),
        backend=None,
        axis_context=sharding_impls.ShardingContext(num_devices=1),
        keepalives=[],
        channel_iterator=itertools.count(1),
        host_callbacks=[],
        lowering_parameters=mlir.LoweringParameters(),
    )
    result = lowering.lower_jaxpr_to_triton_module(
        call.params["jaxpr"], call.params["grid_mapping"], "cuda", 120, context
    )
    assert result.module.operation.verify()
    assert result.grid == [2]
    text = str(result.module)
    if width in (64, 128):
        assert text.count('"sqrt.approx.ftz.f32 $0, $1;"') == 1
        assert text.count('"div.full.f32 $0, $1, $2;"') == 1
        assert "__nv_sqrtf" not in text

        def walk(op):
            yield op
            for region in op.regions:
                for block in region.blocks:
                    for child in block.operations:
                        yield from walk(child.operation)

        assembly = [
            op
            for op in walk(result.module.operation)
            if op.name == "tt.elementwise_inline_asm"
        ]
        root = next(op for op in assembly if "sqrt.approx.ftz.f32" in str(op))
        divide = next(op for op in assembly if "div.full.f32" in str(op))
        # The scalar's tensor reshape must not break the direct sqrt -> div
        # data dependency or let a reciprocal-sqrt rewrite replace it.
        assert str(root.results[0].type) == "tensor<1xf32>"
        assert root.operands[0].owner.name == "tt.splat"
        assert root.operands[0].owner.operands[0].owner.name == "arith.maxnumf"
        denominator = divide.operands[1].owner
        assert denominator.name == "tt.splat"
        assert denominator.operands[0].owner.name == "tt.reduce"
        assert denominator.operands[0].owner.operands[0] == root.results[0]

        affine = [
            op
            for op in assembly
            if "fma.rn.f32" in str(op)
            and str(op.results[0].type) == f"tensor<{width}xf32>"
        ]
        assert len(affine) == 1
        affine = affine[0]
        normalized = affine.operands[0].owner
        assert normalized.name == "arith.mulf"
        assert normalized.operands[0].owner.name == "arith.subf"
        reciprocal = normalized.operands[1].owner
        assert reciprocal.name == "tt.splat"
        assert reciprocal.operands[0].owner.name == "tt.reduce"
        assert reciprocal.operands[0].owner.operands[0] == divide.results[0]

        function = next(iter(result.module.body.operations))
        arguments = function.regions[0].blocks[0].arguments
        # Bind the final FMA's two affine operands to the real weight/bias
        # parameters; the moment and variance FMAs cannot satisfy this test.
        for operand, index in zip(affine.operands[1:], (1, 2), strict=True):
            load = operand.owner
            assert load.name == "tt.load"
            pointer = load.operands[0].owner
            assert pointer.name == "tt.addptr"
            assert pointer.operands[0].owner.name == "tt.splat"
            assert pointer.operands[0].owner.operands[0] == arguments[index]
        store = next(
            op for op in walk(result.module.operation) if op.name == "tt.store"
        )
        stored = store.operands[1]
        if dtype != jnp.float32:
            assert stored.owner.name == "arith.truncf"
            stored = stored.owner.operands[0]
        assert stored == affine.results[0]
    else:
        assert "sqrt.approx.ftz.f32" not in text
        assert "div.full.f32" not in text
        assert "__nv_sqrtf" in text
    assert not context.host_callbacks
    assert registry[primitives.elementwise_inline_asm_p] is original
    assert original is primitives._elementwise_inline_asm_lowering
