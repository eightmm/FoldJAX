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
    counts = {"triangle_multiplication": 0, "transition": 0}
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
# Which sites `pallas` reaches
# --------------------------------------------------------------------------- #


def test_site_backend_names_the_sites():
    from foldjax.models._glu import site_backend

    assert site_backend("pallas", released="tokamax", width=128) == "pallas"
    assert site_backend("pallas", released="tokamax", width=64) == "pallas"
    assert site_backend("pallas", released="tokamax", width=384) == "tokamax"
    assert site_backend("pallas", released="xla", width=None) == "xla"
    for value in ("xla", "tokamax"):
        assert site_backend(value, released="xla", width=128) == value
    with pytest.raises(ValueError, match="must be one of"):
        site_backend("triton", released="xla", width=128)


def test_an_unresolved_glu_site_refuses_pallas():
    from foldjax.models._glu import gated_linear_unit, gated_linear_unit_packed

    x = jnp.ones((3, C), jnp.bfloat16)
    w = jnp.ones((C, C), jnp.bfloat16)
    with pytest.raises(ValueError, match="site_backend"):
        gated_linear_unit(x, w, w, jax.nn.silu, backend="pallas")
    with pytest.raises(ValueError, match="site_backend"):
        gated_linear_unit_packed(
            x, jnp.ones((C, 2 * C), jnp.bfloat16), jax.nn.silu, backend="pallas"
        )


def test_protenix_wide_and_conditioned_transitions_stay_on_xla(calls):
    """A 256-wide transition and a conditioned one run their XLA arithmetic."""
    from foldjax.models.protenix.models.primitives.primitives import (
        LayerNormParams,
        TransitionParams,
        _transition_block,
        _transition_for_runtime,
    )

    rng = np.random.default_rng(8)
    wide = 256
    x = jnp.asarray(rng.normal(size=(1, N, N, wide)), jnp.bfloat16)
    params = TransitionParams(
        layer_norm=LayerNormParams(*_affine(rng, wide)),
        linear_a=_torch_linear(rng, wide, 2 * wide),
        linear_b=_torch_linear(rng, wide, 2 * wide),
        linear_out=_torch_linear(rng, 2 * wide, wide),
    )
    identity = ("serial", 1, (1, 1), ())
    out = _transition_for_runtime(
        x, params, chunk_size=None, runtime_identity=identity, glu_backend="pallas"
    )
    expected = _transition_for_runtime(
        x, params, chunk_size=None, runtime_identity=identity, glu_backend="xla"
    )
    np.testing.assert_array_equal(np.asarray(out), np.asarray(expected))
    np.testing.assert_array_equal(
        np.asarray(_transition_block(x, params, glu_backend="pallas")),
        np.asarray(_transition_block(x, params, glu_backend="xla")),
    )
    assert calls["transition"] == 0


def test_openfold3_bare_swiglu_stays_on_xla(calls):
    from foldjax.models.openfold3.models.primitives import SwiGLUParams, swiglu

    rng = np.random.default_rng(9)
    x = jnp.asarray(rng.normal(size=(5, C)), jnp.bfloat16)
    params = SwiGLUParams(_of3_linear(rng, C, 2 * C), _of3_linear(rng, C, 2 * C))
    np.testing.assert_array_equal(
        np.asarray(swiglu(x, params, glu_backend="pallas")),
        np.asarray(swiglu(x, params, glu_backend="xla")),
    )
    assert calls["transition"] == 0


def test_boltz2_single_transition_keeps_tokamax(monkeypatch):
    """384 wide under `pallas`: the released tokamax GLU, not the fused kernel."""
    from foldjax.models.boltz2.models.primitives import transition as module

    seen = []
    monkeypatch.setattr(
        module,
        "gated_linear_unit",
        lambda x, w1, w2, act, backend: (
            seen.append(backend) or jnp.zeros(x.shape[:-1] + (w1.shape[-1],), x.dtype)
        ),
    )
    rng = np.random.default_rng(10)
    wide = 384
    params = {
        "norm": dict(zip(("scale", "bias"), _affine(rng, wide), strict=True)),
        "fc1": {"kernel": _w(rng, wide, 2 * wide)},
        "fc2": {"kernel": _w(rng, wide, 2 * wide)},
        "fc3": {"kernel": _w(rng, 2 * wide, wide)},
    }
    x = jnp.asarray(rng.normal(size=(1, N, wide)), jnp.bfloat16)
    module.transition_forward(params, x, glu_backend="pallas")
    assert seen == ["tokamax"]


def test_esmfold2_refuses_pallas_it_has_no_pair_stack():
    from foldjax.backends.esmfold2 import _checked_glu_backend

    with pytest.raises(ValueError, match="ESMFold2 has none"):
        _checked_glu_backend("pallas")
    assert _checked_glu_backend("tokamax") == "tokamax"


def test_boltz2_msa_transition_stays_on_tokamax_under_pallas(monkeypatch):
    """The MSA transition keeps the released row-chunked tokamax GLU under
    `pallas`; the pair transitions of the MSA layer still take `pallas`."""
    from foldjax.models.boltz2.models.trunk_blocks import msa

    seen = {}
    m = jnp.ones((1, 2, 3, 4), jnp.bfloat16)
    monkeypatch.setattr(
        msa, "pair_weighted_averaging_forward", lambda *a, **k: jnp.zeros_like(m)
    )

    def transition(params, value, **kwargs):
        seen["msa"] = kwargs["glu_backend"]
        return jnp.zeros_like(value)

    def noseq(params, z, *args, **kwargs):
        seen["pair"] = kwargs["glu_backend"]
        return z

    monkeypatch.setattr(msa, "transition_forward", transition)
    monkeypatch.setattr(
        msa,
        "outer_product_mean_forward",
        lambda *a, **k: jnp.zeros((1, 3, 3, 2), jnp.float32),
    )
    monkeypatch.setattr(msa, "pairformer_no_seq_layer_forward", noseq)
    params = {
        "pair_weighted_averaging": {},
        "msa_transition": {"fc1": {"kernel": jnp.zeros((4, 4), jnp.bfloat16)}},
        "outer_product_mean": {},
        "pairformer_layer": {},
    }
    msa.msa_layer_forward(
        params,
        jnp.zeros((1, 3, 3, 2)),
        m,
        jnp.ones((1, 3, 3)),
        jnp.ones((1, 2, 3)),
        glu_backend="pallas",
    )
    assert seen == {"msa": "tokamax", "pair": "pallas"}


@pytest.mark.parametrize("asked, runs", [("pallas", "tokamax"), ("xla", "xla")])
def test_boltz2_msa_module_pair_transition_stays_on_tokamax_under_pallas(
    monkeypatch, asked, runs
):
    """The MSA module sets the prediction's peak at 3k, where the fused kernel's
    operand is one more pair tensor; its pair transition keeps tokamax."""
    from foldjax.models.boltz2.models.trunk_blocks import msa

    seen = []
    z = jnp.ones((1, 3, 3, 4), jnp.bfloat16)
    zero = lambda *a, **k: jnp.zeros_like(z)  # noqa: E731
    monkeypatch.setattr(msa, "triangle_multiplication_forward", zero)
    monkeypatch.setattr(msa, "triangle_attention_forward", zero)

    def transition(params, value, **kwargs):
        seen.append(kwargs["glu_backend"])
        return jnp.zeros_like(value)

    monkeypatch.setattr(msa, "transition_forward", transition)
    params = {
        name: {}
        for name in ("tri_mul_out", "tri_mul_in", "tri_att_start", "tri_att_end")
    }
    params["transition_z"] = {"fc1": {"kernel": jnp.zeros((4, 4), jnp.bfloat16)}}
    msa.pairformer_no_seq_layer_forward(
        params, z, jnp.ones((1, 3, 3)), glu_backend=asked
    )
    assert seen == [runs]
