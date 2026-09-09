"""Explicit experimental dispatch; these CPU controls do not admit GPU parity."""

import importlib
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax._openfold3_compile import resolve_triangle_kernel, triangle_backend
from foldjax.models.openfold3.models import native_triangle_ops as ops

block = importlib.import_module("foldjax.models.openfold3.models.pair_block")


@pytest.mark.parametrize("batch", [1, 5])
def test_native_dispatch_preserves_residual_and_sample_order(monkeypatch, batch):
    def residual(z, params, *, outgoing, **kwargs):
        return z * 2 + (3 if outgoing else 7)

    monkeypatch.setattr(ops, "native_triangle_multiplication_residual", residual)
    params = SimpleNamespace(tri_mul_out=None, tri_mul_in=None)
    z = jnp.arange(batch * 8, dtype=jnp.float32).reshape(batch, 2, 2, 2)
    mask = jnp.ones(z.shape[:-1])
    original = block.tri_mul_out_in
    with triangle_backend("native-private"):
        assert resolve_triangle_kernel(None, cp_shards=1) == "native-private"
        result = jax.jit(lambda x: block.tri_mul_out_in(x, params, pair_mask=mask))(z)
    np.testing.assert_array_equal(result, (z * 2 + 3) * 2 + 7)
    assert block.tri_mul_out_in is original


def test_attention_dispatch_preserves_transpose_and_update_contract(monkeypatch):
    seen = []

    def attention(z, params, *, mask, **kwargs):
        seen.append((np.asarray(z), np.asarray(mask), params, kwargs))
        return jnp.ones_like(z) * (2 if params == "start" else 5)

    monkeypatch.setattr(ops, "native_triangle_attention_update", attention)
    z = jnp.arange(8, dtype=jnp.float32).reshape(1, 2, 2, 2)
    mask = jnp.array([[[1.0, 0.0], [1.0, 1.0]]])
    params = SimpleNamespace(tri_att_start="start", tri_att_end="end")
    with triangle_backend("native-private"):
        result = block.tri_att_start_end(z, params, pair_mask=mask, no_heads_pair=4)
    np.testing.assert_array_equal(result, z + 7)
    np.testing.assert_array_equal(seen[1][0], np.swapaxes(z + 2, -2, -3))
    np.testing.assert_array_equal(seen[1][1], np.swapaxes(mask, -1, -2))
    assert seen[1][3]["transpose_bias"] is True
    assert seen[0][3]["chunk_size"] == 1024


def test_native_rejects_context_parallelism_and_wrong_heads(monkeypatch):
    monkeypatch.setattr(ops, "cp_mesh", lambda: object())
    with pytest.raises(ValueError, match="context parallelism"):
        ops.map_native_samples(None, None, None)
    with triangle_backend("native-private"):
        with pytest.raises(ValueError, match="four heads"):
            block._pair_attention(None, None, no_heads=8, mask=None)


def test_default_route_does_not_select_native(monkeypatch):
    monkeypatch.delenv("OPENFOLD3_TRIANGLE_BACKEND", raising=False)
    assert not block._native_pair_backend()
    assert resolve_triangle_kernel(None, cp_shards=1) == "cueq"
    with triangle_backend("xla"):
        assert not block._native_pair_backend()
