"""Input extraction stops before pair initialization and recycling."""

import jax
import jax.numpy as jnp
import numpy as np

from foldjax.models.boltz2.models import predict
from foldjax.models.boltz2.models.trunk_blocks import trunk


def test_inputs_stop_before_pair_initialization(monkeypatch):
    expected = jnp.arange(12, dtype=jnp.float32).reshape(1, 3, 4)
    monkeypatch.setattr(trunk, "input_embedder_forward", lambda *a, **k: expected)
    # No s_init, pairformer, diffusion or confidence parameters exist. Any
    # execution past the input embedder therefore fails instead of passing silently.
    output = jax.jit(
        lambda: predict.boltz2_predict(
            {"trunk": {"input_embedder": {}}},
            {"token_pad_mask": jnp.ones((1, 3))},
            jax.random.PRNGKey(0),
            stop_after_inputs=True,
            return_representations=("single_inputs",),
        )
    )()
    assert set(output) == {"single_inputs"}
    np.testing.assert_array_equal(output["single_inputs"], expected)
