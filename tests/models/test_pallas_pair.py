"""The Pallas pair kernels against float32 references, in interpret mode.

The kernels are GPU programs; interpret mode runs the same Pallas bodies on the
CPU. That catches wrong indexing, masking, slicing and direction. It does not
catch Triton-specific failures, and the timings mean nothing. Sizes are chosen
so no grid divides its blocks, which exercises the masked edge tiles.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models import _pallas_pair


@pytest.fixture
def interpret(monkeypatch):
    monkeypatch.setattr(_pallas_pair, "INTERPRET", True)


def _ln(x, w, b, eps=1e-5):
    m = x.mean(-1, keepdims=True)
    v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / jnp.sqrt(v + eps) * w + b


def _rel(a, b):
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    return float(np.abs(a - b).max() / np.abs(b).max())


def _weights(rng, c, hidden, dtype=jnp.bfloat16):
    w = lambda o, i: jnp.asarray(rng.normal(size=(o, i)) / np.sqrt(i), dtype)  # noqa: E731
    affine = lambda n: (  # noqa: E731
        jnp.asarray(1 + 0.1 * rng.normal(size=n), jnp.float32),
        jnp.asarray(0.1 * rng.normal(size=n), jnp.float32),
    )
    return {
        "norm_in": affine(c),
        "p_in": (w(2 * hidden, c), None),
        "g_in": (w(2 * hidden, c), None),
        "norm_out": affine(hidden),
        "p_out": (w(c, hidden), None),
        "g_out": (w(c, c), None),
    }


def _reference_multiplication(x, mask, p, direction):
    f = lambda t: jnp.asarray(t, jnp.float32)  # noqa: E731
    xn = _ln(f(x), *p["norm_in"])
    ab = jax.nn.sigmoid(xn @ f(p["g_in"][0]).T) * (xn @ f(p["p_in"][0]).T)
    ab = ab * mask[..., None]
    hidden = ab.shape[-1] // 2
    a, b = ab[..., :hidden], ab[..., hidden:]
    equation = (
        "...ikc,...jkc->...ijc" if direction == "outgoing" else "...kic,...kjc->...ijc"
    )
    e = jnp.einsum(equation, a, b, precision="highest")
    out = _ln(e, *p["norm_out"]) @ f(p["p_out"][0]).T
    return out * jax.nn.sigmoid(xn @ f(p["g_out"][0]).T)


@pytest.mark.parametrize("direction", ["outgoing", "incoming"])
@pytest.mark.parametrize(
    "c, hidden",
    # (64, 128) is a template stack's shape: OPENFOLD3_TRIANGLE_BACKEND set in
    # the environment forces the fused path whatever the widths.
    [(32, 32), (256, 256), (64, 128)],
)
def test_triangle_multiplication_matches_float32(interpret, direction, c, hidden):
    rng = np.random.default_rng(c + hidden)
    n = 21  # 441 pixels: no 32-pixel block divides it
    x = jnp.asarray(rng.normal(size=(1, n, n, c)), jnp.bfloat16)
    valid = np.arange(n) < n - 2  # a padded tail: prefix keys, fully masked rows
    mask = jnp.asarray((valid[:, None] & valid[None, :])[None], jnp.float32)
    p = _weights(rng, c, hidden)
    out = _pallas_pair.triangle_multiplication(
        x, direction=direction, mask=mask, eps=1e-5, **p
    )
    assert out.shape == x.shape and out.dtype == x.dtype
    assert _rel(out, _reference_multiplication(x, mask, p, direction)) < 2e-2


def test_triangle_multiplication_folds_leading_axes(interpret):
    rng = np.random.default_rng(1)
    n, c = 9, 32
    x = jnp.asarray(rng.normal(size=(2, 3, n, n, c)), jnp.bfloat16)
    mask = jnp.ones((2, 3, n, n), jnp.float32)
    p = _weights(rng, c, c)
    out = _pallas_pair.triangle_multiplication(
        x, direction="outgoing", mask=mask, eps=1e-5, **p
    )
    for i in range(2):
        for j in range(3):
            single = _pallas_pair.triangle_multiplication(
                x[i, j], direction="outgoing", mask=mask[i, j], eps=1e-5, **p
            )
            np.testing.assert_array_equal(np.asarray(out[i, j]), np.asarray(single))


def test_triangle_multiplication_refuses_a_bias(interpret):
    rng = np.random.default_rng(2)
    p = _weights(rng, 32, 32)
    p["p_out"] = (p["p_out"][0], jnp.zeros(32, jnp.bfloat16))
    x = jnp.zeros((1, 4, 4, 32), jnp.bfloat16)
    with pytest.raises(ValueError, match="bias-free"):
        _pallas_pair.triangle_multiplication(
            x, direction="outgoing", mask=jnp.ones((1, 4, 4)), eps=1e-5, **p
        )


@pytest.mark.parametrize("c, hidden", [(32, 128), (256, 512)])
def test_transition_matches_float32(interpret, c, hidden):
    rng = np.random.default_rng(hidden)
    x = jnp.asarray(rng.normal(size=(1, 13, 11, c)), jnp.bfloat16)
    w = lambda i, o: jnp.asarray(rng.normal(size=(i, o)) / np.sqrt(i), jnp.bfloat16)  # noqa: E731
    norm = (
        jnp.asarray(1 + 0.1 * rng.normal(size=c), jnp.float32),
        jnp.asarray(0.1 * rng.normal(size=c), jnp.float32),
    )
    w1, w2, w3 = w(c, hidden), w(c, hidden), w(hidden, c)
    out = _pallas_pair.transition(x, norm, w1, w2, w3, eps=1e-5)
    f = lambda t: jnp.asarray(t, jnp.float32)  # noqa: E731
    xn = _ln(f(x), *norm)
    expected = (jax.nn.silu(xn @ f(w1)) * (xn @ f(w2))) @ f(w3)
    assert out.shape == x.shape and out.dtype == x.dtype
    assert _rel(out, expected) < 2e-2


@pytest.mark.parametrize(
    "x_dtype, out_dtype",
    [(jnp.float32, jnp.bfloat16), (jnp.bfloat16, jnp.float32), (jnp.bfloat16, None)],
)
def test_transition_rounds_through_the_input_dtype_then_the_callers(
    interpret, x_dtype, out_dtype
):
    # Storing the caller's dtype directly must not change a bit: the result is
    # the float32 accumulator rounded to x's dtype, then to out_dtype.
    rng = np.random.default_rng(7)
    c, hidden = 32, 128
    x = jnp.asarray(rng.normal(size=(1, 13, 11, c)), x_dtype)
    w = lambda i, o: jnp.asarray(rng.normal(size=(i, o)) / np.sqrt(i), jnp.bfloat16)  # noqa: E731
    norm = (
        jnp.asarray(1 + 0.1 * rng.normal(size=c), jnp.float32),
        jnp.asarray(0.1 * rng.normal(size=c), jnp.float32),
    )
    w1, w2, w3 = w(c, hidden), w(c, hidden), w(hidden, c)
    out = _pallas_pair.transition(x, norm, w1, w2, w3, eps=1e-5, out_dtype=out_dtype)
    staged = _pallas_pair._transition(
        x, *norm, w1, w2, w3, eps=1e-5, store_dtype=x.dtype
    ).astype(out_dtype or x_dtype)
    assert out.dtype == staged.dtype
    np.testing.assert_array_equal(
        np.asarray(out, np.float32), np.asarray(staged, np.float32)
    )


@pytest.mark.skipif(jax.default_backend() == "gpu", reason="checks the off-GPU refusal")
def test_off_a_gpu_the_kernels_refuse_and_name_the_alternative():
    x = jnp.ones((1, 4, 4, 32), jnp.bfloat16)
    w = jnp.ones((32, 32), jnp.bfloat16)
    norm = (jnp.ones(32), jnp.zeros(32))
    with pytest.raises(ValueError, match="glu_backend=xla"):
        _pallas_pair.transition(x, norm, w, w, w, eps=1e-5)
    p = _weights(np.random.default_rng(0), 32, 32)
    with pytest.raises(ValueError, match="'cueq' or 'xla'"):
        _pallas_pair.triangle_multiplication(
            x, direction="outgoing", mask=jnp.ones((1, 4, 4)), eps=1e-5, **p
        )


@pytest.mark.parametrize(
    "x_dtype, out_dtype",
    # float32 MSA + bfloat16 update (Boltz-2's MSA transition), bfloat16 pair +
    # bfloat16 update (its pair transitions), and float32 throughout
    [
        (jnp.float32, jnp.bfloat16),
        (jnp.bfloat16, jnp.bfloat16),
        (jnp.float32, jnp.float32),
    ],
)
def test_residual_mode_is_the_unfused_add_bit_for_bit(interpret, x_dtype, out_dtype):
    rng = np.random.default_rng(11)
    c, hidden = 64, 256
    x = jnp.asarray(rng.normal(size=(1, 13, 11, c)), x_dtype)
    w = lambda i, o: jnp.asarray(rng.normal(size=(i, o)) / np.sqrt(i), jnp.bfloat16)  # noqa: E731
    norm = (
        jnp.asarray(1 + 0.1 * rng.normal(size=c), jnp.float32),
        jnp.asarray(0.1 * rng.normal(size=c), jnp.float32),
    )
    w1, w2, w3 = w(c, hidden), w(c, hidden), w(hidden, c)
    update = _pallas_pair.transition(x, norm, w1, w2, w3, eps=1e-5, out_dtype=out_dtype)
    expected = x + update
    fused = _pallas_pair.transition(
        x, norm, w1, w2, w3, eps=1e-5, out_dtype=out_dtype, residual=True
    )
    assert fused.dtype == expected.dtype
    np.testing.assert_array_equal(
        np.asarray(fused, np.float32), np.asarray(expected, np.float32)
    )
