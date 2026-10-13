"""Direct attention calls register their selected CUDA target before lowering."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from foldjax import _pallas_gpu


@pytest.mark.parametrize("site", ["triangle", "boltz2", "protenix", "pallas"])
def test_direct_attention_registers_before_dispatch(monkeypatch, site):
    import tokamax

    from foldjax.models import _tokamax_attention
    from foldjax.models.boltz2.models.primitives import attention_backend
    from foldjax.models.boltz2.models.triangle import triangle_attention_pallas
    from foldjax.models.protenix.models.primitives import attention_tokamax

    events = []
    monkeypatch.setattr(_pallas_gpu, "install", lambda: events.append("registered"))

    def dispatch(q, *args, **kwargs):
        assert events == ["registered"]
        events.append("dispatch")
        return q

    monkeypatch.setattr(tokamax, "dot_product_attention", dispatch)
    monkeypatch.setattr(triangle_attention_pallas, "_flash_attention", dispatch)
    q = jnp.ones((1, 2, 1, 4), dtype=jnp.bfloat16)
    if site in {"triangle", "pallas"}:
        q = jnp.ones((1, 2, 1, 2, 4), dtype=jnp.bfloat16)
        bias = jnp.zeros((1, 1, 1, 2, 2), dtype=q.dtype)
        mask = jnp.zeros((1, 2, 1, 1, 2), dtype=q.dtype)
        call = (
            _tokamax_attention.tokamax_attention_core
            if site == "triangle"
            else triangle_attention_pallas.pallas_attention_core
        )
        out = call(q, q, q, bias, mask)
    elif site == "boltz2":
        out = attention_backend.tokamax_dot_product_attention(
            q,
            q,
            q,
            jnp.zeros((1, 1, 2, 2), dtype=q.dtype),
            jnp.ones((1, 2), dtype=bool),
            scale=1.0,
            backend="triton",
        )
    else:
        out = attention_tokamax.tokamax_attention(q, q, q)
    np.testing.assert_array_equal(out, q)
    assert events == ["registered", "dispatch"]
