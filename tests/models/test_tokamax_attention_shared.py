"""Preserve the native pre-scaled triangle attention boundary across both ports."""

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from foldjax.models import _tokamax_attention as shared
from foldjax.models.boltz2.models.triangle import triangle_attention_tokamax as boltz
from foldjax.models.protenix.models.triangle import (
    triangle_attention_tokamax as protenix,
)


def test_native_imports_share_the_same_attention_contract(monkeypatch):
    seen = {}

    def attention(q, k, v, **kwargs):
        seen.update(q=q, k=k, v=v, **kwargs)
        return v

    monkeypatch.setattr(shared, "_TOKAMAX_AVAILABLE", True)
    monkeypatch.setattr(
        shared, "tokamax", SimpleNamespace(dot_product_attention=attention)
    )
    q = jnp.arange(48, dtype=jnp.float32).reshape(1, 2, 2, 3, 4)
    bias = jnp.zeros((1, 1, 2, 3, 3))
    mask = jnp.array([0., -1e9, 0.]).reshape(1, 1, 1, 1, 3)
    assert boltz.tokamax_attention_core is protenix.tokamax_attention_core
    result = boltz.tokamax_attention_core(q, q, q, bias, mask)
    np.testing.assert_array_equal(result, q)
    np.testing.assert_array_equal(seen["q"], q.swapaxes(-3, -2))
    np.testing.assert_array_equal(seen["mask"], mask >= 0)
    assert seen["bias"] is bias
    assert seen["scale"] == 1.0
    assert seen["implementation"] == "triton"
