"""Measured native PWA precision boundaries, independent of GPU availability."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.esmfold2.models import trunk


def _pwa_params():
    params = {}
    for name in ("norm_single", "compute_bias.0"):
        params[name + ".weight"] = jnp.array([1.0035, 0.9965], jnp.float32)
        params[name + ".bias"] = jnp.zeros(2, jnp.float32)
    for name in ("compute_bias.1", "Wv", "Wgate", "Wout"):
        params[name + ".weight"] = jnp.eye(2, dtype=jnp.float32)
    return params


@pytest.mark.parametrize("compiled", [False, True])
def test_native_pwa_retains_fp32_norm_until_projection(monkeypatch, compiled):
    """The MSA norm stays float32; the pair norm is stored bfloat16.

    `norm_single` feeds two projections and a float32 sigmoid, and keeps the
    width native autocast leaves a LayerNorm at. `compute_bias.0` is the one
    norm here that reads a *pair* tensor -- the widest value this stack owns
    -- and `compute_bias.1` is its only consumer and rounds to bfloat16
    itself, so it is stored at that width instead. The value is unchanged,
    which the test below asserts bit for bit rather than inferring from the
    dtype.
    """
    params = _pwa_params()
    seen = []
    narrow = {"compute_bias.1", "Wout"}

    def project(x, selected, prefix):
        assert selected is params
        assert x.dtype == (jnp.bfloat16 if prefix in narrow else jnp.float32)
        seen.append(prefix)
        return trunk._autocast_linear_fallback(x, selected, prefix)

    monkeypatch.setattr(trunk, "_autocast_linear", project)
    original_sigmoid = jax.nn.sigmoid

    def sigmoid(x):
        assert x.dtype == jnp.float32
        return original_sigmoid(x)

    monkeypatch.setattr(jax.nn, "sigmoid", sigmoid)
    msa = jnp.arange(12, dtype=jnp.float32).reshape(1, 3, 2, 2).astype(jnp.bfloat16)
    pair = jnp.arange(18, dtype=jnp.float32).reshape(1, 3, 3, 2).astype(jnp.bfloat16)

    def run(m, z):
        return trunk.msa_pair_weighted_averaging(
            m, z, params, "", pair_mask=jnp.zeros((1, 3, 3)), native_autocast=True
        )

    result = (jax.jit(run) if compiled else run)(msa, pair)
    assert result.dtype == jnp.bfloat16 and result.shape == msa.shape
    assert np.isfinite(np.asarray(result.astype(jnp.float32))).all()
    assert seen == ["compute_bias.1", "Wv", "Wgate", "Wout"]


@pytest.mark.parametrize("compiled", [False, True])
def test_the_pair_norm_width_changes_no_value(monkeypatch, compiled):
    """Storing `compute_bias.0` narrow is the same value, bit for bit.

    The reduction and the affine are float32 whatever the result is stored
    at, and the only consumer rounds to bfloat16, so the two spellings differ
    by which side of one round-nearest-even convert the store happens on.
    Run against a float32-storing `_autocast_norm` rather than against a
    recorded array, with a tripwire arm so a patch that never fired could not
    report agreement.
    """
    params = _pwa_params()
    rng = np.random.default_rng(5)
    msa = jnp.asarray(rng.normal(size=(1, 3, 2, 2), scale=2.0), jnp.bfloat16)
    pair = jnp.asarray(rng.normal(size=(1, 3, 3, 2), scale=2.0), jnp.bfloat16)
    original = trunk._autocast_norm

    def arm(rewrite):
        # A fresh closure per arm, and the caches cleared: `jax.jit` keyed on
        # one shared `run` would answer the second and third arms from the
        # first arm's traced program, and every comparison below would be of
        # that program with itself.
        def run(m, z):
            return trunk.msa_pair_weighted_averaging(
                m, z, params, "", pair_mask=jnp.ones((1, 3, 3)), native_autocast=True
            )

        jax.clear_caches()
        monkeypatch.setattr(trunk, "_autocast_norm", rewrite)
        try:
            value = (jax.jit(run) if compiled else run)(msa, pair)
        finally:
            monkeypatch.setattr(trunk, "_autocast_norm", original)
        return np.asarray(value, np.float32)

    def widened(x, p, name, eps=1e-5, *, out_dtype=jnp.float32):
        del out_dtype
        return original(x, p, name, eps)

    def doubled(x, p, name, eps=1e-5, *, out_dtype=jnp.float32):
        result = original(x, p, name, eps, out_dtype=out_dtype)
        return result * 2 if name == "compute_bias.0" else result

    shipped = arm(original)
    assert float(np.abs(shipped).max()) > 0.0
    np.testing.assert_array_equal(shipped, arm(widened))
    assert not np.array_equal(shipped, arm(doubled))
