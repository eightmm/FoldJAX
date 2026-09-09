"""Transition control formula and boundaries, not native GPU admission."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.openfold3.models import native_triangle_ops as ops
from foldjax.models.openfold3.models.primitives import (
    LayerNormParams,
    LinearParams,
    SwiGLUParams,
    SwiGLUTransitionParams,
    swiglu_transition,
)


def parameters(width):
    rng = np.random.default_rng(619)

    def linear(out, size):
        return LinearParams(jnp.asarray(rng.normal(0, 0.03, (out, size)), jnp.float32))

    return SwiGLUTransitionParams(
        LayerNormParams(jnp.ones(width), jnp.zeros(width)),
        SwiGLUParams(linear(4 * width, width), linear(4 * width, width)),
        linear(width, 4 * width),
    )


@pytest.mark.parametrize("width", [64, 128])
def test_native_transition_interpret_formula_on_rectangular_rows(width):
    p = parameters(width)
    x = jnp.arange(6 * width, dtype=jnp.float32).reshape(2, 3, width) / 100
    mask = jnp.array([[1, 0, 1], [0, 1, 1]], dtype=jnp.float32)
    expected = swiglu_transition(x, p, mask=mask)
    actual = jax.jit(
        lambda a: ops.native_swiglu_transition_update(a, p, mask=mask, interpret=True)
    )(x)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
    np.testing.assert_array_equal(np.asarray(actual)[np.asarray(mask) == 0], 0)


def test_transition_precision_mask_and_cp_fail_closed(monkeypatch):
    p = parameters(64)
    x = jnp.zeros((2, 3, 64), jnp.float32)
    with pytest.raises(ValueError, match="FP32"):
        ops.native_swiglu_transition_update(x.astype(jnp.bfloat16), p)
    with pytest.raises(ValueError, match="mask"):
        ops.native_swiglu_transition_update(x, p, mask=jnp.ones((3,)))
    monkeypatch.setattr(ops, "cp_mesh", lambda: object())
    with pytest.raises(ValueError, match="context parallelism"):
        ops.native_swiglu_transition_update(x, p)


def test_rejected_transition_control_is_not_dispatched(monkeypatch):
    import importlib
    from types import SimpleNamespace

    from foldjax._openfold3_compile import triangle_backend

    block = importlib.import_module("foldjax.models.openfold3.models.pair_block")
    p = SimpleNamespace(pair_transition=None)
    x = jnp.zeros((1, 2, 2, 64), jnp.float32)
    calls = []

    def controlled(value, params, *, mask, eps):
        calls.append((mask, eps))
        return value + 7

    def forbidden(*args, **kwargs):
        pytest.fail("rejected transition must not enter a full model")

    monkeypatch.setattr(ops, "native_swiglu_transition_update", forbidden)
    monkeypatch.setattr(block, "swiglu_transition", controlled)
    monkeypatch.setattr(block, "tri_mul_out_in", lambda z, *a, **k: z)
    monkeypatch.setattr(block, "tri_att_start_end", lambda z, *a, **k: z)
    for backend in ("xla", "native-private"):
        with triangle_backend(backend):
            np.testing.assert_array_equal(
                block.pair_block(
                    x, p, pair_mask=jnp.ones(x.shape[:-1]), no_heads_pair=4
                ),
                x + 7,
            )
    assert len(calls) == 2
