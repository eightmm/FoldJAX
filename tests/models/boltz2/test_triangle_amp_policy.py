"""CPU witnesses for Boltz's native mixed triangle operator boundaries."""

from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.boltz2.models.triangle import triangle, triangle_attention
from foldjax.models.boltz2.models.triangle import triangle_cueq as cueq


def _params(dtype=jnp.bfloat16):
    rng = np.random.default_rng(42)

    def kernel(rows, cols):
        return jnp.asarray(rng.normal(size=(rows, cols)) * 0.3, dtype=dtype)

    return {
        "norm_in": {
            "scale": jnp.asarray([0.8113, 1.0137, 1.4123, 0.7311]),
            "bias": jnp.asarray([0.0313, -0.2191, 0.5511, 0.1411]),
        },
        "norm_out": {
            "scale": jnp.asarray([0.5913, 1.3147, 0.8111, 1.6211]),
            "bias": jnp.asarray([-0.1313, 0.1191, 0.3511, -0.2411]),
        },
        "g_in": {"kernel": kernel(4, 8)},
        "p_in": {"kernel": kernel(4, 8)},
        "g_out": {"kernel": kernel(4, 4)},
        "p_out": {"kernel": kernel(4, 4)},
    }


def _norm(x, params):
    x = x.astype(jnp.float32)
    centered = x - jnp.mean(x, axis=-1, keepdims=True)
    normed = centered * jax.lax.rsqrt(
        jnp.mean(centered * centered, axis=-1, keepdims=True) + 1e-5
    )
    return normed * params["scale"].astype(jnp.float32) + params["bias"].astype(
        jnp.float32
    )


@pytest.mark.parametrize("direction", ["incoming", "outgoing"])
@pytest.mark.parametrize("batch_shape", [(), (1,), (2, 1)])
def test_cueq_mixed_path_has_native_internal_casts(monkeypatch, direction, batch_shape):
    events = []
    params = _params()
    x = (
        jnp.arange(np.prod((*batch_shape, 3, 3, 4)), dtype=jnp.float32).reshape(
            *batch_shape, 3, 3, 4
        )
        / 17
    )
    mask = jnp.ones(x.shape[:-1], dtype=jnp.float32)

    def norm(value, scale, bias, *, eps, layout, fallback):
        events.append(("norm", value.dtype, scale.dtype, bias.dtype, layout))
        assert fallback is False
        assert eps == 1e-5
        if layout == "dbij->bijd":
            value = jnp.transpose(value, (1, 2, 3, 0))
        return _norm(value, {"scale": scale, "bias": bias}).astype(value.dtype)

    def gemm(value, w1, w2, *, mask, transpose_out, precision, fallback):
        events.append(("gemm", value.dtype, w1.dtype, w2.dtype))
        assert precision == "DEFAULT"
        assert transpose_out is True
        assert fallback is False
        out = jax.nn.sigmoid(value @ w1.T) * (value @ w2.T)
        return jnp.transpose(out * mask[..., None].astype(out.dtype), (3, 0, 1, 2))

    def dual(value1, value2, w1, w2, *, precision, fallback):
        events.append(("dual", value1.dtype, value2.dtype, w1.dtype, w2.dtype))
        assert precision == "DEFAULT"
        assert fallback is False
        return jax.nn.sigmoid(value1 @ w1.T) * (value2 @ w2.T)

    def unexpected(**kwargs):
        pytest.fail("the homogeneous public API cannot express mixed native AMP")

    monkeypatch.setattr(
        cueq,
        "load_cueq",
        lambda: SimpleNamespace(
            TriMulPrecision=SimpleNamespace(DEFAULT="DEFAULT", IEEE="IEEE"),
            triangle_multiplicative_update=unexpected,
        ),
    )
    monkeypatch.setattr(
        cueq, "_load_cueq_amp_primitives", lambda: (norm, gemm, dual), raising=False
    )
    with jax.default_matmul_precision("highest"):
        result = cueq.cueq_triangle_multiplication_forward(params, x, mask, direction)

    assert result.shape == x.shape
    assert result.dtype == jnp.bfloat16
    assert [event[0] for event in events] == ["norm", "gemm", "norm", "dual"]
    assert events[0][1:4] == (jnp.float32, jnp.float32, jnp.float32)
    assert events[1][1:] == (jnp.bfloat16,) * 3
    assert events[2][1:4] == (jnp.bfloat16, jnp.float32, jnp.float32)
    assert events[3][1:] == (jnp.bfloat16,) * 4


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_cueq_homogeneous_path_still_uses_public_api(monkeypatch, dtype):
    x = jnp.ones((1, 3, 3, 4), dtype=dtype)
    params = jax.tree.map(lambda value: value.astype(dtype), _params(dtype))
    calls = []

    def public(**kwargs):
        calls.append(kwargs)
        return kwargs["x"]

    monkeypatch.setattr(
        cueq,
        "load_cueq",
        lambda: SimpleNamespace(
            TriMulPrecision=SimpleNamespace(DEFAULT="DEFAULT", IEEE="IEEE"),
            triangle_multiplicative_update=public,
        ),
    )
    result = cueq.cueq_triangle_multiplication_forward(
        params, x, jnp.ones(x.shape[:-1]), "outgoing"
    )
    assert result is x
    assert len(calls) == 1


@pytest.mark.parametrize("direction", ["incoming", "outgoing"])
@pytest.mark.parametrize("chunk_size", [0, 2])
def test_xla_mixed_triangle_matches_native_operator_order(
    monkeypatch, direction, chunk_size
):
    monkeypatch.setenv("BOLTZ_JAX_TRIANGLE_MULTIPLICATION_BACKEND", "xla")
    params = _params()
    x = jnp.asarray(np.random.default_rng(8).normal(size=(1, 3, 3, 4)), jnp.float32)
    mask = jnp.asarray([[[1, 1, 0], [1, 0, 1], [1, 1, 1]]], jnp.float32)
    normalized = _norm(x, params["norm_in"])
    projected = jax.nn.sigmoid(
        (normalized.astype(jnp.bfloat16) @ params["g_in"]["kernel"]).astype(jnp.float32)
    ).astype(jnp.bfloat16)
    projected = projected * (normalized.astype(jnp.bfloat16) @ params["p_in"]["kernel"])
    a, b = jnp.split((projected * mask[..., None]).astype(jnp.bfloat16), 2, axis=-1)
    equation = "bikd,bjkd->bijd" if direction == "outgoing" else "bkid,bkjd->bijd"
    contracted = jnp.einsum(equation, a, b)
    out = (
        _norm(contracted, params["norm_out"]).astype(jnp.bfloat16)
        @ params["p_out"]["kernel"]
    )
    expected = out * jax.nn.sigmoid(
        (normalized.astype(jnp.bfloat16) @ params["g_out"]["kernel"]).astype(
            jnp.float32
        )
    ).astype(jnp.bfloat16)
    actual = triangle.triangle_multiplication_forward(
        params, x, mask, direction, chunk_size=chunk_size
    )
    assert actual.dtype == jnp.bfloat16
    np.testing.assert_array_equal(actual, expected)


def test_triangle_attention_linear_uses_kernel_amp_dtype():
    x = jnp.asarray([[1.001, 2.003, -3.007]], dtype=jnp.float32)
    kernel = jnp.asarray([[1.1], [0.9], [0.7]], dtype=jnp.bfloat16)
    actual = triangle_attention._linear(x, kernel)
    assert actual.dtype == jnp.bfloat16
    np.testing.assert_array_equal(actual, x.astype(jnp.bfloat16) @ kernel)


@pytest.mark.parametrize("jit", [False, True])
def test_cueq_attention_output_gate_uses_native_single_rounding(monkeypatch, jit):
    # Isolate the post-cuEq gate from the attention kernel itself.
    monkeypatch.setattr(
        cueq, "cueq_attention_core", lambda q, *args, **kwargs: jnp.ones_like(q)
    )
    x = jnp.asarray(np.linspace(-8, 8, 16).reshape(1, 2, 2, 4), jnp.float32)
    params = {
        name: {"kernel": jnp.eye(4, dtype=jnp.bfloat16)}
        for name in ("linear_q", "linear_k", "linear_v", "linear_g", "linear_o")
    }
    bias = jnp.zeros((1, 1, 1, 2, 2), jnp.bfloat16)
    mask = jnp.zeros((1, 2, 1, 1, 2), jnp.float32)

    def run(value):
        return triangle_attention._attention(
            params, value, value, bias, mask, triangle_backend="cueq"
        )

    function = jax.jit(run) if jit else run
    rounded = np.asarray(x.astype(jnp.bfloat16), np.float64)
    expected = jnp.asarray(1 / (1 + np.exp(-rounded)), jnp.bfloat16)
    np.testing.assert_array_equal(function(x), expected)


def test_triangle_attention_bf16_input_narrows_affine_before_native_norm():
    x = jnp.asarray([[0.17, 0.83, -0.23, 1.49]], dtype=jnp.bfloat16)
    params = _params()["norm_in"]
    rounded_affine = jax.tree.map(lambda value: value.astype(jnp.bfloat16), params)
    expected = _norm(x, rounded_affine).astype(jnp.bfloat16)
    actual = triangle_attention._layer_norm(x, params["scale"], params["bias"], 1e-5)
    assert actual.dtype == jnp.bfloat16
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("chunk_size,q_chunk_size", [(0, None), (2, None), (2, 2)])
def test_xla_mixed_attention_keeps_native_bf16_inplace_scores(chunk_size, q_chunk_size):
    rng = np.random.default_rng(14)
    x = jnp.asarray(rng.normal(size=(1, 3, 3, 4)), dtype=jnp.float32)
    params = {
        name: {"kernel": jnp.asarray(rng.normal(size=shape), dtype=jnp.bfloat16)}
        for name, shape in {
            "linear_q": (4, 3),
            "linear_k": (4, 3),
            "linear_v": (4, 3),
            "linear_g": (4, 3),
            "linear_o": (3, 4),
        }.items()
    }
    bias = jnp.asarray(rng.normal(size=(1, 1, 1, 3, 3)), dtype=jnp.bfloat16)
    mask = jnp.asarray(
        [[[[[0, -1e9, 0]]], [[[0, 0, 0]]], [[[-1e9, 0, 0]]]]], dtype=jnp.float32
    )

    def project(name):
        return (x.astype(jnp.bfloat16) @ params[name]["kernel"])[:, :, None]

    q = (project("linear_q").astype(jnp.float32) / float(3**0.5)).astype(jnp.bfloat16)
    k, v = project("linear_k"), project("linear_v")
    scores = q @ jnp.swapaxes(k, -1, -2)
    scores = (scores.astype(jnp.float32) + mask).astype(jnp.bfloat16)
    scores = (scores.astype(jnp.float32) + bias.astype(jnp.float32)).astype(
        jnp.bfloat16
    )
    probabilities = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(
        jnp.bfloat16
    )
    out = probabilities @ v
    out = out[:, :, 0] * jax.nn.sigmoid(
        project("linear_g")[:, :, 0].astype(jnp.float32)
    ).astype(jnp.bfloat16)
    expected = out @ params["linear_o"]["kernel"]
    actual = triangle_attention._attention(
        params, x, x, bias, mask, chunk_size=chunk_size, q_chunk_size=q_chunk_size
    )
    assert actual.dtype == jnp.bfloat16
    np.testing.assert_array_equal(actual, expected)
