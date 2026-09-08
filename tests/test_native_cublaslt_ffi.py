import jax
import jax.numpy as jnp
import pytest

from bench.native_cublaslt_ffi import linear


def test_ffi_boundary_has_bf16_buffers_and_xla_owned_scratch():
    traced = jax.make_jaxpr(lambda x, w: linear(x, w, target="abstract_test"))(
        jnp.zeros((2, 3, 4)), jnp.zeros((5, 4))
    )
    call = next(e for e in traced.jaxpr.eqns if e.primitive.name == "ffi_call")
    assert all(v.aval.dtype == jnp.bfloat16 for v in call.invars)
    assert call.outvars[0].aval.shape == (6, 5)
    assert call.outvars[0].aval.dtype == jnp.bfloat16
    assert call.outvars[1].aval.shape == (32 * 1024 * 1024,)
    assert call.outvars[1].aval.dtype == jnp.uint8
    assert len(traced.out_avals) == 1
    assert traced.out_avals[0].shape == (2, 3, 5)


@pytest.mark.parametrize("x,w", [((4,), (5, 4)), ((2, 4), (5, 3)), ((0, 4), (5, 4))])
def test_ffi_rejects_invalid_shapes_without_loading_library(x, w):
    with pytest.raises(ValueError):
        linear(jnp.zeros(x), jnp.zeros(w), target="unused")
