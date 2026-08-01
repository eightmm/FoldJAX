from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from foldjax.models.protenix.bridge.torch_mapping import map_distogram_state_dict
from foldjax.models.protenix.models.heads.head import distogram_head


def test_distogram_head_matches_reference_formula() -> None:
    rng = np.random.default_rng(0)
    z = rng.normal(size=(3, 3, 5)).astype(np.float32)
    state = {
        "linear.weight": rng.normal(size=(4, 5)).astype(np.float32),
        "linear.bias": rng.normal(size=(4,)).astype(np.float32),
    }
    params = map_distogram_state_dict(state)

    actual = np.asarray(distogram_head(jnp.asarray(z), params))
    projected = np.matmul(z, state["linear.weight"].T) + state["linear.bias"]
    expected = projected + np.swapaxes(projected, -2, -3)

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
