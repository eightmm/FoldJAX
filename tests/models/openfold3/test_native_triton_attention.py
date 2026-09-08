"""CPU formula and IR contracts, never native CUDA numerical admission."""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.openfold3.models import native_triton_attention as attention


def _inputs(length=17, dim=16, batch=1, rows=1):
    rng = np.random.default_rng(1803)
    shape = (batch, rows, length, 4, dim)
    qkv = [rng.normal(size=shape).astype(np.float32) * 0.4 for _ in range(3)]
    mask = np.zeros((batch, rows, 1, 1, length), np.float32)
    mask[..., 1::3] = -12
    bias = rng.normal(size=(batch, 1, 4, length, length)).astype(np.float32) * 0.7
    return (*qkv, mask, bias)


def _online_formula(query, key, value, mask, bias):
    # Independent NumPy oracle retains native mask-before-bias and KV16 state
    # updates. Omitting padded zero probabilities is formula-level equivalence.
    batch, rows, length, heads, dim = query.shape
    out = np.empty_like(query)
    for b in range(batch):
        for r in range(rows):
            for h in range(heads):
                q = query[b, r, :, h] * np.float32(dim**-0.5)
                maximum = np.full((length,), -np.inf, np.float32)
                normalizer = np.ones((length,), np.float32)
                output = np.zeros((length, dim), np.float32)
                for start in range(0, length, 16):
                    scores = q @ key[b, r, start : start + 16, h].T
                    scores += mask[b, r, 0, 0, start : start + 16]
                    scores += bias[b, 0, h, :, start : start + 16]
                    new_maximum = np.maximum(maximum, scores.max(axis=-1))
                    p = np.exp(scores - new_maximum[:, None])
                    alpha = np.exp(maximum - new_maximum)
                    normalizer = normalizer * alpha + p.sum(axis=-1)
                    output = output * alpha[:, None]
                    output += p @ value[b, r, start : start + 16, h]
                    maximum = new_maximum
                out[b, r, :, h] = output / normalizer[:, None]
    return out


@pytest.mark.parametrize("length", [17, 31, 64, 65])
@pytest.mark.parametrize("dim", [16, 32])
def test_interpret_online_formula_query_kv_and_head_dimension_tails(length, dim):
    values = _inputs(length, dim)
    actual = attention.native_triangle_attention(
        *map(jnp.asarray, values), interpret=True
    )
    assert actual.shape == values[0].shape and actual.dtype == jnp.float32
    np.testing.assert_allclose(actual, _online_formula(*values), rtol=1e-5, atol=2e-6)


def test_native_layout_row_broadcast_and_asymmetric_pair_bias():
    values = _inputs(batch=2, rows=2)
    actual = attention.native_triangle_attention(
        *map(jnp.asarray, values), interpret=True
    )
    expected = _online_formula(*values)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=2e-6)
    transposed = (*values[:4], values[4].swapaxes(-1, -2))
    assert np.max(np.abs(expected - _online_formula(*transposed))) > 0.05


def test_additive_finite_all_masked_rows_are_not_zero_or_safe_softmax():
    q, k, v, mask, bias = _inputs()
    mask.fill(-1e9)
    bias.fill(0)
    actual = attention.native_triangle_attention(
        *map(jnp.asarray, (q, k, v, mask, bias)), interpret=True
    )
    expected = np.broadcast_to(v.mean(axis=2, keepdims=True), q.shape)
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-7)
    assert np.isfinite(actual).all() and np.max(np.abs(actual)) > 0.01


@pytest.mark.parametrize("only_first_tile", [False, True])
def test_true_infinite_all_masked_tile_keeps_native_nan_failure(only_first_tile):
    q, k, v, mask, bias = _inputs()
    mask.fill(0)
    mask[..., : 16 if only_first_tile else 17] = -np.inf
    actual = attention.native_triangle_attention(
        *map(jnp.asarray, (q, k, v, mask, bias)), interpret=True
    )
    assert np.isnan(actual).all()


def test_mask_is_added_before_pair_bias_not_precombined():
    q, k, v, mask, bias = _inputs()
    mask.fill(-1e9)
    bias.fill(1e9)
    actual = attention.native_triangle_attention(
        *map(jnp.asarray, (q, k, v, mask, bias)), interpret=True
    )
    expected = np.broadcast_to(v.mean(axis=2, keepdims=True), q.shape)
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-7)
    combined_mask = np.zeros_like(mask)
    combined_bias = np.zeros_like(bias)
    assert (
        np.max(
            np.abs(expected - _online_formula(q, k, v, combined_mask, combined_bias))
        )
        > 0.01
    )


def test_private_carried_dot_lowering_uses_live_accumulator_in_actual_triton_ir():
    from jax._src.lib.mlir import ir
    from jax._src.lib.mlir.dialects import func

    with attention.triton_lowering._new_ir_context(), ir.Location.unknown():
        module = ir.Module.create()
        tensor = ir.RankedTensorType.get((16, 16), ir.F32Type.get())
        with ir.InsertionPoint(module.body):
            function = func.FuncOp(
                "carried", ir.FunctionType.get([tensor] * 3, [tensor])
            )
            block = function.add_entry_block()
            with ir.InsertionPoint(block):
                output = attention._carried_dot_lowering(None, *block.arguments)
                func.ReturnOp([output])
        assert module.operation.verify()
        dot = output.owner
        assert dot.name == "tt.dot"
        assert list(dot.operands) == list(block.arguments)
        assert "tf32" in str(dot)
        assert "arith.addf" not in str(module)
    assert attention.triton_lowering.triton_lowering_rules[jax.lax.dot_general_p] is (
        attention.triton_lowering._dot_general_lowering
    )


def test_gpu_graph_has_private_carried_dot_and_native_arithmetic_not_formula_fallback():
    values = tuple(map(jnp.asarray, _inputs()))
    with jax.default_matmul_precision("highest"):
        graph = str(jax.make_jaxpr(attention.native_triangle_attention)(*values))
    assert graph.count("openbind_triangle_carried_dot") == 2
    assert graph.count("4294959104:u32[]") == 4
    assert "fma.rn.f32" in graph and "ex2.approx.f32" in graph
    assert "div.full.f32" in graph
    assert "dot_general" not in graph
    interpreted = str(
        jax.make_jaxpr(
            functools.partial(attention.native_triangle_attention, interpret=True)
        )(*values)
    )
    assert "openbind_triangle_carried_dot" not in interpreted
    assert "4294959104:u32[]" not in interpreted
    assert "ex2.approx.f32" not in interpreted
    assert "precision=(Precision.HIGHEST, Precision.HIGHEST)" in interpreted


def test_native_launch_tiles_warps_stages_and_original_layout(monkeypatch):
    seen = []

    def call(kernel, **kwargs):
        def run(*args):
            seen.append((kernel, kwargs, args))
            return jnp.zeros(kwargs["out_shape"].shape, jnp.float32)

        return run

    monkeypatch.setattr(attention.pl, "pallas_call", call)
    values = tuple(map(jnp.asarray, _inputs(length=65, dim=16, batch=2, rows=3)))
    actual = attention.native_triangle_attention(*values)
    kernel, kwargs, args = seen.pop()
    assert actual.shape == values[0].shape
    assert all(a is b for a, b in zip(args, values))
    assert kernel.keywords == {"rows": 3, "length": 65, "dim": 16, "interpret": False}
    assert kwargs["grid"] == (2, 24, 1)
    assert kwargs["compiler_params"].num_warps == 4
    assert kwargs["compiler_params"].num_stages == 1
    assert kwargs["interpret"] is False
    assert kwargs["name"] == "openbind_native_triangle_attention"


@pytest.mark.parametrize("length", [0, 1, 16])
def test_small_inputs_explicitly_require_native_stock_dispatch(length):
    with pytest.raises(ValueError, match="native stock attention"):
        attention.native_triangle_attention(*map(jnp.asarray, _inputs(length)))


@pytest.mark.parametrize("index", range(5))
def test_no_implicit_operand_dtype_conversion(index):
    values = list(map(jnp.asarray, _inputs()))
    values[index] = values[index].astype(jnp.bfloat16)
    with pytest.raises(TypeError, match="FP32"):
        attention.native_triangle_attention(*values)


@pytest.mark.parametrize(
    "index,shape",
    [
        (0, (1, 17, 4, 16)),
        (1, (1, 2, 17, 4, 16)),
        (3, (1, 1, 1, 17, 1)),
        (4, (1, 4, 17, 17)),
    ],
)
def test_shape_contract_does_not_silently_reshape_or_broadcast(index, shape):
    values = list(map(jnp.asarray, _inputs()))
    values[index] = jnp.ones(shape)
    with pytest.raises(ValueError):
        attention.native_triangle_attention(*values)


@pytest.mark.parametrize(
    "batch,rows,heads,dim",
    [
        (0, 1, 4, 16),
        (1, 0, 4, 16),
        (1, 1, 2, 16),
        (1, 1, 4, 64),
    ],
)
def test_unimplemented_axes_fail_closed(batch, rows, heads, dim):
    q = jnp.ones((batch, rows, 17, heads, dim))
    with pytest.raises(ValueError, match="positive B,R"):
        attention.native_triangle_attention(
            q,
            q,
            q,
            jnp.zeros((batch, rows, 1, 1, 17)),
            jnp.zeros((batch, 1, heads, 17, 17)),
        )


def test_private_primitive_rejects_bad_shapes_dtype_and_eager_fallback():
    x = jnp.ones((16, 16))
    with pytest.raises(RuntimeError, match="Pallas Triton-only"):
        attention._carried_dot_p.bind(x, x, x)
    for arrays in (
        (x, x, jnp.ones((16, 32))),
        (jnp.ones((8, 16)), x, jnp.ones((8, 16))),
    ):
        with pytest.raises(ValueError):
            jax.eval_shape(attention._carried_dot_p.bind, *arrays)
    with pytest.raises(TypeError, match="FP32"):
        jax.eval_shape(attention._carried_dot_p.bind, x.astype(jnp.bfloat16), x, x)


def test_missing_backend_or_nonstatic_flag_has_no_silent_fallback(monkeypatch):
    values = tuple(map(jnp.asarray, _inputs()))
    with pytest.raises(TypeError, match="static boolean"):
        attention.native_triangle_attention(*values, interpret=1)
    monkeypatch.setattr(attention, "_carried_dot_p", None)
    with pytest.raises(RuntimeError, match="carried-dot lowering"):
        attention.native_triangle_attention(*values)
