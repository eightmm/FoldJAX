import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.esmfold2.models import native_softmax


@pytest.mark.parametrize("cp", [False, True])
@pytest.mark.parametrize("shape", [(1, 437, 437, 8), (1, 3, 3, 8)])
def test_profile_dispatch_and_fallback(monkeypatch, cp, shape):
    monkeypatch.setattr(native_softmax, "cp_mesh", lambda: object() if cp else None)
    calls = []

    def dispatch(x, *, cuda, default):
        assert cuda is native_softmax.spatial_softmax
        calls.append(True)
        return default(x)

    monkeypatch.setattr(jax.lax, "platform_dependent", dispatch)
    x = jnp.full(shape, -1e5, jnp.float32)
    result = native_softmax.pwa_softmax(x)
    np.testing.assert_array_equal(result, jax.nn.softmax(x, axis=-2))
    assert len(calls) == int(not cp and shape == (1, 437, 437, 8))
    assert np.isfinite(result).all()


@pytest.mark.parametrize("shape", [(1, 3, 3, 8), (1, 437, 437, 4)])
def test_spatial_rejects_unverified_shapes(shape):
    with pytest.raises(ValueError, match="437 by 8"):
        native_softmax.spatial_softmax(jnp.zeros(shape, jnp.float32))
