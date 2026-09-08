"""Measured native PWA precision boundaries, independent of GPU availability."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.esmfold2.models import trunk


@pytest.mark.parametrize("compiled", [False, True])
def test_native_pwa_retains_fp32_norm_until_projection(monkeypatch, compiled):
    params = {}
    for name in ("norm_single", "compute_bias.0"):
        params[name + ".weight"] = jnp.array([1.0035, 0.9965], jnp.float32)
        params[name + ".bias"] = jnp.zeros(2, jnp.float32)
    for name in ("compute_bias.1", "Wv", "Wgate", "Wout"):
        params[name + ".weight"] = jnp.eye(2, dtype=jnp.float32)
    seen = []

    def project(x, selected, prefix):
        assert selected is params
        assert x.dtype == (jnp.bfloat16 if prefix == "Wout" else jnp.float32)
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
