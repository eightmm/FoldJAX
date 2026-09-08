import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bench.boltz_pwa_softmax_probe import warp_softmax


def test_warp_softmax_matches_sequential_lane_reference():
    x = np.random.default_rng(17).normal(size=(1, 1, 3, 437)).astype(np.float32)
    lanes = np.pad(x, [(0, 0)] * 3 + [(0, 75)], constant_values=-np.inf)
    lanes = lanes.reshape(1, 1, 3, 16, 32)
    maximum = np.max(lanes, axis=(-2, -1), keepdims=True)
    exponentials = np.exp(lanes - maximum)
    total = np.zeros((1, 1, 3, 32), np.float32)
    for i in range(16):
        total = total + exponentials[..., i, :]
    for offset in (16, 8, 4, 2, 1):
        total = total + total[..., np.arange(32) ^ offset]
    expected = (exponentials / total[..., None, :]).reshape(1, 1, 3, 512)[..., :437]
    actual = np.asarray(jax.jit(warp_softmax)(jnp.asarray(x)))
    # NumPy and XLA exp are not a cross-runtime bitwise oracle.
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize(
    "shape,dtype", [((2, 436), jnp.float32), ((2, 437), jnp.bfloat16)]
)
def test_warp_softmax_rejects_unobserved_boundary(shape, dtype):
    with pytest.raises(ValueError, match="437 FP32"):
        warp_softmax(jnp.zeros(shape, dtype))
def test_explicit_division_rejects_non_fp32():
    import jax.numpy as jnp
    import pytest

    from bench.boltz_pwa_softmax_probe import rn_divide

    with pytest.raises(ValueError, match="FP32"):
        rn_divide(jnp.ones(32, jnp.bfloat16), jnp.ones(32, jnp.bfloat16))


def test_explicit_division_rejects_partial_blocks():
    import jax.numpy as jnp
    import pytest

    from bench.boltz_pwa_softmax_probe import rn_divide

    with pytest.raises(ValueError, match="32-element"):
        rn_divide(jnp.ones(31), jnp.ones(31))

