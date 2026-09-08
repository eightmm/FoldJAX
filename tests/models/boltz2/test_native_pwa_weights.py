import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bench.boltz_pwa_logits_probe import strided_four_forward
from bench.boltz_pwa_softmax_probe import warp_softmax as reference_softmax
from foldjax.models.boltz2.models.primitives import native_pwa_weights as weights


@pytest.mark.parametrize("size", [0, 31])
def test_native_division_rejects_invalid_block_size(size):
    with pytest.raises(ValueError, match="32-element"):
        weights._cuda_rn_divide(jnp.ones(size), jnp.ones(size))


def test_native_division_rejects_low_precision():
    with pytest.raises(ValueError, match="FP32"):
        weights._cuda_rn_divide(
            jnp.ones(32, jnp.bfloat16), jnp.ones(32, jnp.bfloat16)
        )


def test_projection_preserves_verified_arithmetic():
    x = jnp.asarray(np.random.default_rng(13).normal(size=(2, 128)), jnp.float32)
    kernel = jnp.asarray(np.random.default_rng(14).normal(size=(128, 1)), jnp.bfloat16)
    actual = jax.jit(weights.four_lane_projection)(x, kernel)
    expected = jax.jit(strided_four_forward)(x, kernel)
    np.testing.assert_array_equal(actual, expected)


def test_softmax_preserves_verified_arithmetic():
    x = jnp.asarray(np.random.default_rng(15).normal(size=(3, 437)), jnp.float32)
    np.testing.assert_array_equal(
        jax.jit(weights.warp_softmax)(x), jax.jit(reference_softmax)(x)
    )


def test_128_softmax_matches_masked_extension():
    x = jnp.asarray(np.random.default_rng(16).normal(size=(3, 128)), jnp.float32)
    extended = jnp.pad(x, ((0, 0), (0, 309)), constant_values=-jnp.inf)
    np.testing.assert_array_equal(
        jax.jit(weights.warp_softmax)(x),
        jax.jit(weights.warp_softmax)(extended)[..., :128],
    )


@pytest.mark.parametrize("columns,dtype", [(127, jnp.float32), (128, jnp.bfloat16)])
def test_warp_softmax_rejects_unsupported_profiles(columns, dtype):
    with pytest.raises(ValueError, match="128 or 437 FP32"):
        weights.warp_softmax(jnp.zeros((2, columns), dtype))


def test_profile_excludes_other_shapes_dtypes_and_cp(monkeypatch):
    z = jax.ShapeDtypeStruct((1, 437, 437, 128), jnp.float32)
    kernel = jax.ShapeDtypeStruct((128, 1), jnp.bfloat16)
    assert weights.observed_profile(z, kernel)
    assert not weights.observed_profile(
        jax.ShapeDtypeStruct((1, 436, 436, 128), jnp.float32), kernel
    )
    assert not weights.observed_profile(z, jax.ShapeDtypeStruct((128, 1), jnp.float32))
    monkeypatch.setattr(weights, "cp_mesh", lambda: object())
    assert not weights.observed_profile(z, kernel)


def test_other_softmax_shape_retains_jax():
    x = jnp.asarray([[1., 2., 3.]], jnp.float32)
    np.testing.assert_array_equal(weights.pwa_softmax(x), jax.nn.softmax(x))
