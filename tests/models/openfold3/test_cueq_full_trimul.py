"""The ``cueq-full`` kernel routes triangle multiplication through cuEquivariance.

Upstream packs the ``a``/``b`` gates and projections into one kernel call and
adds the residual outside it. These tests pin that contract with a fake
kernel on CPU; the GPU test checks the real kernel against the XLA path.
"""

from __future__ import annotations

import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax._openfold3_compile import triangle_backend
from foldjax.models.openfold3.models import triangle as triangle_module
from foldjax.models.openfold3.models.primitives import LayerNormParams, LinearParams
from foldjax.models.openfold3.models.triangle import (
    TriangleMultiplicationParams,
    triangle_multiplication,
)


def _params(key, c_z=8, c_hidden=8):
    keys = jax.random.split(key, 8)
    scale = 0.2

    def lin(k, out, inp):
        return LinearParams(scale * jax.random.normal(k, (out, inp), jnp.float32), None)

    return TriangleMultiplicationParams(
        layer_norm_in=LayerNormParams(
            1.0 + scale * jax.random.normal(keys[0], (c_z,)),
            scale * jax.random.normal(keys[1], (c_z,)),
        ),
        layer_norm_out=LayerNormParams(
            1.0 + scale * jax.random.normal(keys[2], (c_hidden,)),
            scale * jax.random.normal(keys[3], (c_hidden,)),
        ),
        linear_a_p=lin(keys[4], c_hidden, c_z),
        linear_a_g=lin(keys[5], c_hidden, c_z),
        linear_b_p=lin(keys[6], c_hidden, c_z),
        linear_b_g=lin(keys[7], c_hidden, c_z),
        linear_g=lin(keys[0], c_z, c_z),
        linear_z=lin(keys[1], c_z, c_hidden),
    )


class _FakeCueq:
    class TriMulPrecision:
        DEFAULT = "default"
        TF32 = "tf32"
        IEEE = "ieee"
        TF32x3 = "tf32x3"

    def __init__(self):
        self.calls = []

    def triangle_multiplicative_update(self, **kwargs):
        self.calls.append(kwargs)
        x = kwargs["x"]
        return jnp.full((*x.shape[:-1], kwargs["p_out_weight"].shape[0]), 7.0, x.dtype)


@pytest.fixture
def fake_cueq(monkeypatch):
    fake = _FakeCueq()
    monkeypatch.setattr(
        "foldjax.models._cueq.load_cueq", lambda: fake, raising=True
    )
    return fake


def test_cueq_full_packs_weights_like_upstream_and_returns_the_update(fake_cueq):
    params = _params(jax.random.PRNGKey(0))
    z = jax.random.normal(jax.random.PRNGKey(1), (2, 3, 5, 5, 8), jnp.float32)
    mask = jnp.ones((2, 3, 5, 5), jnp.float32).at[0, 0, 4].set(0.0)
    with triangle_backend("cueq-full"):
        out = triangle_multiplication(z, params, outgoing=False, mask=mask)
    assert out.shape == (2, 3, 5, 5, 8)
    np.testing.assert_array_equal(out, 7.0)  # the update alone, no residual
    (call,) = fake_cueq.calls
    assert call["direction"] == "incoming"
    assert call["x"].shape == (6, 5, 5, 8) and call["mask"].shape == (6, 5, 5)
    assert call["mask"].dtype == jnp.float32
    np.testing.assert_array_equal(call["mask"][0, 4], 0.0)
    np.testing.assert_array_equal(
        call["p_in_weight"],
        jnp.concatenate((params.linear_a_p.weight, params.linear_b_p.weight), 0),
    )
    np.testing.assert_array_equal(
        call["g_in_weight"],
        jnp.concatenate((params.linear_a_g.weight, params.linear_b_g.weight), 0),
    )
    assert call["p_in_bias"] is None and call["g_in_bias"] is None
    assert call["p_out_weight"] is params.linear_z.weight
    assert call["g_out_weight"] is params.linear_g.weight
    assert call["fallback"] is False
    assert call["eps"] == 1e-5


def test_cueq_full_outgoing_direction_and_default_mask(fake_cueq):
    params = _params(jax.random.PRNGKey(0))
    z = jnp.zeros((4, 4, 8), jnp.float32)
    with triangle_backend("cueq-full"):
        triangle_multiplication(z, params, outgoing=True)
    (call,) = fake_cueq.calls
    assert call["direction"] == "outgoing"
    np.testing.assert_array_equal(call["mask"], jnp.ones((1, 4, 4)))


def test_cueq_alone_keeps_the_xla_multiplication(fake_cueq):
    params = _params(jax.random.PRNGKey(0))
    z = jax.random.normal(jax.random.PRNGKey(2), (4, 4, 8), jnp.float32)
    with triangle_backend("cueq"):
        fused = triangle_multiplication(z, params, outgoing=True)
    with triangle_backend("xla"):
        blocked = triangle_multiplication(z, params, outgoing=True)
    assert fake_cueq.calls == []
    np.testing.assert_array_equal(fused, blocked)


def test_cueq_full_refuses_context_parallelism(fake_cueq, monkeypatch):
    params = _params(jax.random.PRNGKey(0))
    monkeypatch.setattr(triangle_module, "cp_mesh", lambda: types.SimpleNamespace())
    with triangle_backend("cueq-full"), pytest.raises(ValueError, match="context"):
        triangle_multiplication(jnp.zeros((4, 4, 8)), params, outgoing=True)


def test_cueq_full_requires_affine_norms(fake_cueq):
    params = _params(jax.random.PRNGKey(0))
    bare = params._replace(layer_norm_out=LayerNormParams(None, None))
    with triangle_backend("cueq-full"), pytest.raises(ValueError, match="affine"):
        triangle_multiplication(jnp.zeros((4, 4, 8)), bare, outgoing=True)


@pytest.mark.skipif(
    jax.default_backend() != "gpu", reason="real cuEquivariance kernel needs a GPU"
)
def test_cueq_full_matches_xla_multiplication_on_gpu():
    pytest.importorskip("cuequivariance_jax")
    params = _params(jax.random.PRNGKey(0), c_z=64, c_hidden=64)
    z = jax.random.normal(jax.random.PRNGKey(3), (2, 48, 48, 64), jnp.float32)
    mask = jnp.ones((2, 48, 48), jnp.float32).at[1, 40:].set(0.0)
    with jax.default_matmul_precision("highest"):
        with triangle_backend("xla"):
            blocked = triangle_multiplication(z, params, outgoing=True, mask=mask)
            blocked_in = triangle_multiplication(z, params, outgoing=False, mask=mask)
        with triangle_backend("cueq-full"):
            fused = triangle_multiplication(z, params, outgoing=True, mask=mask)
            fused_in = triangle_multiplication(z, params, outgoing=False, mask=mask)
    for a, b in ((blocked, fused), (blocked_in, fused_in)):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=2e-3, atol=2e-3)
