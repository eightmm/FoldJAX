"""Each port's ``pallas`` spelling reaches the Pallas kernels and agrees with XLA.

Run in Pallas interpret mode on the CPU, at widths every kernel slice divides.
Each case checks two things. The kernel fires: a counter wrapped around the
shared entry point proves the port routed there, and did not quietly run its
XLA body under the new name. The result also matches that port's own XLA path
to within bf16 rounding. The shootout measured speed and error on the GPU
(``foldjax-bench/kernel-shootout-20260924``); this is only about wiring.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models import _pallas_pair

C = 32
N = 11


@pytest.fixture
def calls(monkeypatch):
    """Interpret mode, and a count of every entry into the shared kernels."""
    monkeypatch.setattr(_pallas_pair, "INTERPRET", True)
    counts = {"triangle_multiplication": 0, "transition": 0, "gated_linear_unit": 0}
    for name in counts:
        original = getattr(_pallas_pair, name)

        def counted(*args, _name=name, _original=original, **kwargs):
            counts[_name] += 1
            return _original(*args, **kwargs)

        monkeypatch.setattr(_pallas_pair, name, counted)
    return counts


def _rel(a, b):
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    return float(np.abs(a - b).max() / np.abs(b).max())


def _pair(rng, dtype=jnp.bfloat16):
    x = jnp.asarray(rng.normal(size=(1, N, N, C)), dtype)
    valid = np.arange(N) < N - 2
    mask = jnp.asarray((valid[:, None] & valid[None, :])[None], jnp.float32)
    return x, mask


def _w(rng, fan_in, fan_out, dtype=jnp.bfloat16):
    return jnp.asarray(rng.normal(size=(fan_in, fan_out)) / np.sqrt(fan_in), dtype)


def _affine(rng, width):
    return (
        jnp.asarray(1 + 0.1 * rng.normal(size=width), jnp.float32),
        jnp.asarray(0.1 * rng.normal(size=width), jnp.float32),
    )


# --------------------------------------------------------------------------- #
# Boltz-2
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("direction", ["outgoing", "incoming"])
def test_boltz2_triangle_multiplication_pallas(calls, monkeypatch, direction):
    from foldjax.models.boltz2.models.triangle.triangle import (
        triangle_multiplication_forward,
    )

    rng = np.random.default_rng(0)
    x, mask = _pair(rng)
    norm = lambda: dict(zip(("scale", "bias"), _affine(rng, C), strict=True))  # noqa: E731
    params = {
        "norm_in": norm(),
        "norm_out": norm(),
        "p_in": {"kernel": _w(rng, C, 2 * C)},
        "g_in": {"kernel": _w(rng, C, 2 * C)},
        "p_out": {"kernel": _w(rng, C, C)},
        "g_out": {"kernel": _w(rng, C, C)},
    }
    monkeypatch.setenv("BOLTZ_JAX_TRIANGLE_MULTIPLICATION_BACKEND", "pallas")
    out = triangle_multiplication_forward(params, x, mask, direction, native_amp=True)
    assert calls["triangle_multiplication"] == 1
    monkeypatch.setenv("BOLTZ_JAX_TRIANGLE_MULTIPLICATION_BACKEND", "xla")
    expected = triangle_multiplication_forward(
        params, x, mask, direction, native_amp=True
    )
    assert out.shape == expected.shape and out.dtype == expected.dtype
    assert _rel(out, expected) < 3e-2


def test_boltz2_transition_pallas(calls):
    from foldjax.models.boltz2.models.primitives.transition import transition_forward

    rng = np.random.default_rng(1)
    x, _ = _pair(rng)
    params = {
        "norm": dict(zip(("scale", "bias"), _affine(rng, C), strict=True)),
        "fc1": {"kernel": _w(rng, C, 4 * C)},
        "fc2": {"kernel": _w(rng, C, 4 * C)},
        "fc3": {"kernel": _w(rng, 4 * C, C)},
    }
    out = transition_forward(params, x, glu_backend="pallas", native_amp_norm=True)
    assert calls["transition"] == 1
    expected = transition_forward(params, x, glu_backend="xla", native_amp_norm=True)
    assert out.shape == expected.shape and out.dtype == expected.dtype
    assert _rel(out, expected) < 3e-2


# --------------------------------------------------------------------------- #
# Protenix (and OpenDDE, which runs Protenix's modules)
# --------------------------------------------------------------------------- #


def _torch_linear(rng, fan_in, fan_out):
    from foldjax.models.protenix.models.primitives.primitives import LinearParams

    return LinearParams(weight=_w(rng, fan_in, fan_out).T, bias=None)


@pytest.mark.parametrize("direction", ["outgoing", "incoming"])
def test_protenix_triangle_multiplication_pallas(calls, monkeypatch, direction):
    from foldjax.models.protenix.models.primitives.primitives import LayerNormParams
    from foldjax.models.protenix.models.triangle.triangle import (
        TriangleMultiplicationParams,
        triangle_multiplication,
    )

    rng = np.random.default_rng(2)
    x, mask = _pair(rng)
    params = TriangleMultiplicationParams(
        layer_norm_in=LayerNormParams(*_affine(rng, C)),
        layer_norm_out=LayerNormParams(*_affine(rng, C)),
        **{
            name: _torch_linear(rng, C, C)
            for name in (
                "linear_a_p",
                "linear_a_g",
                "linear_b_p",
                "linear_b_g",
                "linear_z",
                "linear_g",
            )
        },
    )
    monkeypatch.setenv("PROTENIX_TRIANGLE_MULTIPLICATION_BACKEND", "pallas")
    out = triangle_multiplication(x[0], mask[0], params, direction)
    assert calls["triangle_multiplication"] == 1
    monkeypatch.setenv("PROTENIX_TRIANGLE_MULTIPLICATION_BACKEND", "xla")
    expected = triangle_multiplication(x[0], mask[0], params, direction)
    assert out.shape == expected.shape and out.dtype == expected.dtype
    assert _rel(out, expected) < 3e-2


def test_protenix_transition_pallas(calls):
    from foldjax.models.protenix.models.primitives.primitives import (
        LayerNormParams,
        TransitionParams,
        _transition_block,
    )

    rng = np.random.default_rng(3)
    x, _ = _pair(rng)
    params = TransitionParams(
        layer_norm=LayerNormParams(*_affine(rng, C)),
        linear_a=_torch_linear(rng, C, 4 * C),
        linear_b=_torch_linear(rng, C, 4 * C),
        linear_out=_torch_linear(rng, 4 * C, C),
    )
    out = _transition_block(x, params, glu_backend="pallas")
    assert calls["transition"] == 1
    expected = _transition_block(x, params, glu_backend="xla")
    assert out.shape == expected.shape and out.dtype == expected.dtype
    assert _rel(out, expected) < 3e-2


# --------------------------------------------------------------------------- #
# OpenFold3
# --------------------------------------------------------------------------- #


def _of3_linear(rng, fan_in, fan_out):
    from foldjax.models.openfold3.models.primitives import LinearParams

    return LinearParams(weight=_w(rng, fan_in, fan_out).T, bias=None)


@pytest.mark.parametrize("outgoing", [True, False])
def test_openfold3_cueq_pallas_multiplication(calls, monkeypatch, outgoing):
    from foldjax.models.openfold3.models.primitives import LayerNormParams
    from foldjax.models.openfold3.models.triangle import (
        TriangleMultiplicationParams,
        triangle_multiplication,
    )

    rng = np.random.default_rng(4)
    x, mask = _pair(rng)
    params = TriangleMultiplicationParams(
        layer_norm_in=LayerNormParams(*_affine(rng, C)),
        layer_norm_out=LayerNormParams(*_affine(rng, C)),
        **{
            name: _of3_linear(rng, C, C)
            for name in (
                "linear_a_p",
                "linear_a_g",
                "linear_b_p",
                "linear_b_g",
                "linear_g",
                "linear_z",
            )
        },
    )
    monkeypatch.setenv("OPENFOLD3_TRIANGLE_BACKEND", "cueq-pallas")
    out = triangle_multiplication(x, params, outgoing=outgoing, mask=mask)
    assert calls["triangle_multiplication"] == 1
    # The XLA path widens its result to float32; the fused kernels return the
    # pair's width, which is the contract `cueq-full` already ships.
    monkeypatch.setenv("OPENFOLD3_TRIANGLE_BACKEND", "xla")
    expected = triangle_multiplication(x, params, outgoing=outgoing, mask=mask)
    assert out.shape == expected.shape and out.dtype == x.dtype
    assert _rel(out, expected) < 3e-2


def test_openfold3_cueq_pallas_keeps_cueq_attention(monkeypatch):
    """The suffix changes the multiplication only; attention stays cuEquivariance."""
    from foldjax.models.openfold3.models import triangle_attention

    seen = []
    monkeypatch.setattr(
        triangle_attention,
        "_cueq_attention",
        lambda *args, **kwargs: seen.append("cueq") or args[0],
    )
    monkeypatch.setenv("OPENFOLD3_TRIANGLE_BACKEND", "cueq-pallas")
    rng = np.random.default_rng(5)
    x, mask = _pair(rng, jnp.float32)
    from foldjax.models.openfold3.models.attention import AttentionParams
    from foldjax.models.openfold3.models.primitives import LayerNormParams

    params = triangle_attention.TriangleAttentionParams(
        layer_norm=LayerNormParams(*_affine(rng, C)),
        linear_z=_of3_linear(rng, C, 4),
        mha=AttentionParams(
            **{name: _of3_linear(rng, C, C) for name in AttentionParams._fields}
        ),
    )
    triangle_attention.triangle_attention(x, params, no_heads=4, mask=mask)
    assert seen == ["cueq"]


def test_openfold3_transition_pallas(calls):
    from foldjax.models.openfold3.models.primitives import (
        LayerNormParams,
        SwiGLUParams,
        SwiGLUTransitionParams,
        swiglu_transition,
    )

    rng = np.random.default_rng(6)
    x, _ = _pair(rng)
    params = SwiGLUTransitionParams(
        layer_norm=LayerNormParams(*_affine(rng, C)),
        swiglu=SwiGLUParams(_of3_linear(rng, C, 4 * C), _of3_linear(rng, C, 4 * C)),
        linear_out=_of3_linear(rng, 4 * C, C),
    )
    out = swiglu_transition(x, params, glu_backend="pallas")
    assert calls["transition"] == 1
    expected = swiglu_transition(x, params, glu_backend="xla")
    assert out.shape == expected.shape and out.dtype == expected.dtype
    assert _rel(out, expected) < 3e-2


# --------------------------------------------------------------------------- #
# The GLU-only form, through the shared vocabulary
# --------------------------------------------------------------------------- #


def test_the_shared_glu_routes_pallas_to_the_unit_kernel(calls):
    from foldjax.models._glu import gated_linear_unit, gated_linear_unit_packed

    rng = np.random.default_rng(7)
    x = jnp.asarray(rng.normal(size=(5, 23, C)), jnp.bfloat16)
    wg, wv = _w(rng, C, 2 * C), _w(rng, C, 2 * C)
    out = gated_linear_unit(x, wg, wv, jax.nn.sigmoid, backend="pallas")
    expected = gated_linear_unit(x, wg, wv, jax.nn.sigmoid, backend="xla")
    assert out.shape == expected.shape and out.dtype == expected.dtype
    assert _rel(out, expected) < 3e-2
    packed = jnp.concatenate([wg, wv], axis=-1)
    out = gated_linear_unit_packed(x, packed, jax.nn.silu, backend="pallas")
    expected = gated_linear_unit_packed(x, packed, jax.nn.silu, backend="xla")
    assert _rel(out, expected) < 3e-2
    assert calls["gated_linear_unit"] == 2
