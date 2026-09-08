"""Input extraction must not execute recycling or downstream heads."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from foldjax.models.opendde.models import model


def test_inputs_stop_before_pairformer(monkeypatch):
    monkeypatch.setattr(
        model, "_validate_static_structural_features", lambda *a, **k: (3, 6, 9)
    )
    expected = jnp.arange(12, dtype=jnp.float32).reshape(3, 4)
    monkeypatch.setattr(model, "input_feature_embedder", lambda *a, **k: expected)

    def forbidden(*args, **kwargs):
        raise AssertionError("input extraction entered trunk")

    monkeypatch.setattr(model, "pairformer_output_from_s_inputs", forbidden)
    output = jax.jit(
        lambda: model.opendde_infer_static(
            {"restype": jnp.zeros((3, 32)), "ref_element": jnp.ones((9, 1))},
            SimpleNamespace(input_embedder=None),
            jnp.ones(2),
            key=None,
            num_samples=1,
            stop_after_inputs=True,
        )
    )()
    assert set(output) == {"single_inputs"}
    np.testing.assert_array_equal(output["single_inputs"], expected)
