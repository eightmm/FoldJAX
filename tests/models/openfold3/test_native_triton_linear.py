"""CPU formula/storage checks; actual native GPU parity is a separate gate."""

from __future__ import annotations

import functools
import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.openfold3.models import native_triton_linear as linear


def test_tf32_rtz_exact_bits_preserve_sign_grid_and_special_payloads():
    bits = np.array(
        [
            0x3F801FFF,
            0xBF801FFF,
            0x3F801000,
            0xBF801000,
            0x3F802000,
            0,
            0x80000000,
            1,
            0x80000001,
            0x7F7FFFFF,
            0x7F800000,
            0xFF800000,
            0x7F800001,
            0xFFC01234,
        ],
        np.uint32,
    )
    value = jnp.asarray(bits.view(np.float32))
    expected = np.where(
        (bits & np.uint32(0x7F800000)) == np.uint32(0x7F800000),
        bits,
        bits & np.uint32(0xFFFFE000),
    )

    def output_bits(x):
        return jax.lax.bitcast_convert_type(linear._tf32_rtz(x), jnp.uint32)

    np.testing.assert_array_equal(output_bits(value), expected)
    np.testing.assert_array_equal(jax.jit(output_bits)(value), expected)
    np.testing.assert_array_equal(output_bits(linear._tf32_rtz(value)), expected)


@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float16])
def test_tf32_rtz_helper_rejects_half_storage(dtype):
    with pytest.raises(TypeError, match="requires FP32"):
        linear._tf32_rtz(jnp.ones(2, dtype))


def test_fp32_truncation_is_inside_loaded_dot_tiles_not_host_or_epilogue():
    graph = jax.make_jaxpr(linear.native_linear_fused)(
        jnp.ones((9, 128)),
        jnp.ones((128, 128)),
        jnp.ones(128),
        other=jnp.ones((9, 128)),
        mask=jnp.ones((9, 1)),
        add_tensor=jnp.ones((9, 128)),
    ).jaxpr
    kernel = next(eq for eq in graph.eqns if eq.primitive.name == "pallas_call")
    kernel = kernel.params["jaxpr"]
    loop = next(eq for eq in kernel.eqns if eq.primitive.name == "while")
    body = loop.params["body_jaxpr"]
    dot = next(eq for eq in body.eqns if eq.primitive.name == "dot_general")
    back_to_float = [
        eq
        for eq in body.eqns
        if eq.primitive.name == "bitcast_convert_type"
        and eq.params["new_dtype"] == np.dtype(np.float32)
    ]
    assert len(back_to_float) == 2
    assert dot.invars == [eq.outvars[0] for eq in back_to_float]
    assert str(body).count("4294959104:u32[]") == 2
    # Both original dot loads are upstream of their integer bitcasts; bias
    # and fused operands are loaded after this loop and never truncated.
    loads = [eq for eq in body.eqns if eq.primitive.name == "masked_load"]
    to_bits = [
        eq
        for eq in body.eqns
        if eq.primitive.name == "bitcast_convert_type"
        and eq.params["new_dtype"] == np.dtype(np.uint32)
    ]
    assert len(loads) == 2
    assert [eq.invars[0] for eq in to_bits] == [eq.outvars[0] for eq in loads]
    assert all(
        eq.primitive.name != "bitcast_convert_type"
        for eq in [*graph.eqns, *kernel.eqns]
    )


def _formula(
    x, weight, bias=None, other=None, mask=None, add_tensor=None, apply_sigmoid=False
):
    dtype, shape = x.dtype, (*x.shape[:-1], weight.shape[0])
    weight = np.asarray(jnp.asarray(weight, dtype), np.float32)
    inputs = np.asarray(x, np.float32).reshape(-1, weight.shape[1])
    result = np.zeros((len(inputs), len(weight)), np.float32)
    for start in range(0, weight.shape[1], 64):
        result += inputs[:, start : start + 64] @ weight[:, start : start + 64].T
    if bias is not None:
        result += np.asarray(jnp.asarray(bias, dtype), np.float32)
    if apply_sigmoid:
        result = 1 / (1 + np.exp(-np.clip(result, -20, 20)))
    if other is not None:
        result *= np.asarray(other, np.float32).reshape(result.shape)
    if mask is not None:
        result *= np.asarray(mask, np.float32).reshape(-1, mask.shape[-1])
    if add_tensor is not None:
        result += np.asarray(add_tensor, np.float32).reshape(result.shape)
    return jnp.asarray(result.reshape(shape), dtype)


@pytest.mark.parametrize("flags", list(itertools.product((False, True), repeat=5)))
def test_all_native_epilogue_flag_combinations(flags):
    has_bias, sigmoid, has_other, has_mask, has_add = flags
    rng = np.random.default_rng(401)
    x = jnp.asarray(rng.normal(size=(3, 64)) * 0.1, jnp.float32)
    weight = jnp.asarray(rng.normal(size=(64, 64)) * 0.1, jnp.float32)
    kwargs = {
        "bias": jnp.asarray(rng.normal(size=64), jnp.float32) if has_bias else None,
        "other": jnp.asarray(rng.normal(size=(3, 64)), jnp.float32)
        if has_other
        else None,
        "mask": jnp.array([[0.0], [0.25], [-1.0]]) if has_mask else None,
        "add_tensor": jnp.asarray(rng.normal(size=(3, 64)), jnp.float32)
        if has_add
        else None,
        "apply_sigmoid": sigmoid,
    }
    actual = linear.native_linear_fused(x, weight, **kwargs, interpret=True)
    np.testing.assert_allclose(
        actual, _formula(x, weight, **kwargs), rtol=3e-6, atol=3e-6
    )


@pytest.mark.parametrize("inputs,outputs", list(itertools.product((64, 128), repeat=2)))
@pytest.mark.parametrize("shape", [(1,), (7,), (8,), (9,), (129,), (2, 3)])
def test_released_widths_shape_preservation_and_row_tile_tails(inputs, outputs, shape):
    rng = np.random.default_rng(40)
    x = jnp.asarray(rng.normal(size=(*shape, inputs)) * 0.1, jnp.float32)
    weight = jnp.asarray(rng.normal(size=(outputs, inputs)) * 0.1, jnp.float32)
    actual = linear.native_linear(x, weight, interpret=True)
    assert actual.shape == (*shape, outputs)
    np.testing.assert_allclose(actual, _formula(x, weight), rtol=3e-6, atol=3e-6)


def test_vector_input_has_one_output_row():
    actual = linear.native_linear(jnp.ones(128), jnp.ones((64, 128)), interpret=True)
    np.testing.assert_array_equal(actual, np.full(64, 128, np.float32))


@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16, jnp.float32])
def test_fp32_accumulator_is_not_stored_before_multiply(dtype):
    x, weight = jnp.full((3, 128), 1000, dtype), jnp.ones((64, 128), dtype)
    other = jnp.full((3, 64), 0.001, jnp.float32)
    result = linear.native_linear_fused(x, weight, other=other, interpret=True)
    assert result.dtype == dtype
    np.testing.assert_array_equal(result, jnp.full((3, 64), 128, dtype))


def test_only_final_store_rounds_other_and_residual_to_input_dtype():
    x = jnp.ones((3, 64), jnp.bfloat16)
    weight = jnp.full((64, 64), 1 / 64, jnp.float32)
    other = jnp.full((3, 64), 1.0039, jnp.float32)
    add = jnp.full((3, 64), 2e-5, jnp.float32)
    actual = linear.native_linear_fused(
        x, weight, other=other, add_tensor=add, interpret=True
    )
    expected = jnp.asarray(np.float32(1.0039) + np.float32(2e-5), jnp.bfloat16)
    np.testing.assert_array_equal(actual, jnp.full((3, 64), expected))
    assert float(expected) != float(jnp.asarray(1.0039, jnp.bfloat16))


def test_mask_before_residual_and_native_flattened_other_contract():
    x, weight = jnp.ones((2, 3, 64)), jnp.ones((128, 64))
    other = jnp.arange(6 * 128, dtype=jnp.float32)
    add = jnp.arange(6 * 128, dtype=jnp.float32).reshape(6, 128)
    actual = linear.native_linear_fused(
        x,
        weight,
        other=other,
        mask=jnp.zeros((1,)),
        add_tensor=add,
        interpret=True,
    )
    np.testing.assert_array_equal(actual, add.reshape(2, 3, 128))


@pytest.mark.parametrize(
    "mask_shape", [(1,), (1, 1), (6, 1), (2, 3, 1), (6, 64), (2, 3, 64)]
)
def test_native_mask_accepted_shapes(mask_shape):
    x, weight = jnp.ones((2, 3, 64)), jnp.ones((64, 64))
    mask = jnp.ones(mask_shape, jnp.bool_)
    actual = linear.native_linear_fused(x, weight, mask=mask, interpret=True)
    np.testing.assert_array_equal(actual, jnp.full((2, 3, 64), 64, jnp.float32))


def test_sigmoid_clamps_values_at_twenty_before_exponentiation():
    x = jnp.array([-jnp.inf, -100, -20, -19, 0, 19, 20, 100, jnp.inf, jnp.nan])
    actual = linear._sigmoid(x, interpret=True)
    assert float(actual[0]) > 0
    np.testing.assert_array_equal(actual[:3], jnp.full((3,), actual[2]))
    np.testing.assert_array_equal(actual[6:9], jnp.ones(3))
    assert float(actual[4]) == 0.5
    assert np.isnan(np.asarray(actual[-1]))


def test_launch_contract_and_only_weight_bias_storage_cast(monkeypatch):
    calls = []

    def pallas_call(kernel, **kwargs):
        def run(*args):
            calls.append((kernel, kwargs, args))
            return jnp.zeros(kwargs["out_shape"].shape, args[0].dtype)

        return run

    monkeypatch.setattr(linear.pl, "pallas_call", pallas_call)
    linear.native_linear_fused(
        jnp.ones((129, 128), jnp.bfloat16),
        jnp.ones((128, 128)),
        jnp.zeros(128),
        other=jnp.ones((129, 128)),
        mask=jnp.ones((129, 1), jnp.bool_),
        add_tensor=jnp.ones((129, 128)),
        apply_sigmoid=True,
    )
    kernel, kwargs, args = calls.pop()
    assert [value.dtype for value in args[:6]] == [
        jnp.bfloat16,
        jnp.bfloat16,
        jnp.bfloat16,
        jnp.float32,
        jnp.bool_,
        jnp.float32,
    ]
    assert [kernel.keywords[key] for key in ("block_m", "block_n", "block_k")] == [
        128,
        64,
        64,
    ]
    assert kwargs["grid"] == (2, 2)
    assert kwargs["compiler_params"].num_warps == 4
    assert kwargs["compiler_params"].num_stages == 3
    assert kwargs["interpret"] is False
    assert kwargs["name"] == "openbind_native_triangle_linear_fused"


@pytest.mark.parametrize(
    "dtype,preset",
    [
        (jnp.float32, "TF32_TF32_F32"),
        (jnp.bfloat16, "BF16_BF16_F32"),
        (jnp.float16, "F16_F16_F32"),
    ],
)
def test_gpu_precision_and_fast_epilogue_are_explicit_without_launch(dtype, preset):
    def call(x, weight, other, add):
        return linear.native_linear_fused(
            x, weight, other=other, add_tensor=add, apply_sigmoid=True
        )

    with jax.default_matmul_precision("highest"):
        graph = str(
            jax.make_jaxpr(call)(
                jnp.ones((9, 128), dtype),
                jnp.ones((128, 128), dtype),
                jnp.ones((9, 128)),
                jnp.ones((9, 128)),
            )
        )
    assert preset in graph
    assert "ex2.approx.f32" in graph
    assert "div.full.f32" in graph
    assert "fma.rn.f32" in graph
    assert ("4294959104:u32[]" in graph) == (dtype == jnp.float32)


def test_interpret_is_formula_only_and_plain_wrapper_disables_epilogue():
    graph = str(
        jax.make_jaxpr(functools.partial(linear.native_linear, interpret=True))(
            jnp.ones((9, 64)), jnp.ones((128, 64))
        )
    )
    assert "TF32_TF32_F32" not in graph
    assert "ex2.approx.f32" not in graph
    assert "fma.rn.f32" not in graph
    assert "4294959104:u32[]" not in graph


def test_empty_rows_do_not_launch(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("empty arrays must not launch")

    monkeypatch.setattr(linear.pl, "pallas_call", forbidden)
    actual = linear.native_linear(jnp.ones((2, 0, 64)), jnp.ones((128, 64)))
    assert actual.shape == (2, 0, 128)


@pytest.mark.parametrize("mask_shape", [(), (64,), (1, 64), (2, 1), (3, 63)])
def test_rejects_non_native_mask_broadcast(mask_shape):
    with pytest.raises(ValueError, match="mask"):
        linear.native_linear_fused(
            jnp.ones((3, 64)), jnp.ones((64, 64)), mask=jnp.ones(mask_shape)
        )


@pytest.mark.parametrize("name", ["other", "add_tensor"])
def test_rejects_operand_broadcast_instead_of_silent_repeat(name):
    with pytest.raises(ValueError, match="exactly M\\*N"):
        linear.native_linear_fused(
            jnp.ones((3, 64)), jnp.ones((64, 64)), **{name: jnp.ones((1, 64))}
        )


@pytest.mark.parametrize(
    "shape,weight_shape",
    [
        ((), (64, 64)),
        ((3, 64), (64,)),
        ((3, 64), (64, 128)),
        ((3, 32), (64, 32)),
        ((3, 128), (32, 128)),
    ],
)
def test_rejects_invalid_or_unimplemented_dimensions(shape, weight_shape):
    with pytest.raises(ValueError):
        linear.native_linear(jnp.ones(shape), jnp.ones(weight_shape))


@pytest.mark.parametrize("name", ["bias", "other", "add_tensor", "mask"])
def test_rejects_integer_optional_operands(name):
    value = jnp.ones((64,) if name == "bias" else (3, 64), jnp.int32)
    with pytest.raises(TypeError, match="dtype"):
        linear.native_linear_fused(
            jnp.ones((3, 64)), jnp.ones((64, 64)), **{name: value}
        )


def test_rejects_bias_broadcast_and_nonstatic_flags():
    x, weight = jnp.ones((3, 64)), jnp.ones((64, 64))
    with pytest.raises(ValueError, match="bias"):
        linear.native_linear(x, weight, bias=jnp.zeros((1, 64)))
    for kwargs in ({"apply_sigmoid": 1}, {"interpret": jnp.array(True)}):
        with pytest.raises(TypeError, match="static booleans"):
            linear.native_linear_fused(x, weight, **kwargs)


def test_missing_pallas_does_not_select_ordinary_jax(monkeypatch):
    monkeypatch.setattr(linear, "pt", None)
    with pytest.raises(RuntimeError, match="Pallas/Triton"):
        linear.native_linear(jnp.ones((3, 64)), jnp.ones((64, 64)))
