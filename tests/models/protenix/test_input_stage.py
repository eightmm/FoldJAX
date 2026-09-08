"""Input extraction must not execute recycling or downstream heads."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from foldjax.models.protenix.models import model


def test_inputs_stop_before_pairformer(monkeypatch):
    expected = jnp.arange(12, dtype=jnp.float32).reshape(3, 4)
    monkeypatch.setattr(model, "input_feature_embedder", lambda *a, **k: expected)

    def forbidden(*args, **kwargs):
        raise AssertionError("input extraction entered trunk")

    monkeypatch.setattr(model, "pairformer_output_from_s_inputs", forbidden)
    output = jax.jit(
        lambda: model.protenix_infer_static(
            {"restype": jnp.zeros((3, 32))},
            SimpleNamespace(input_embedder=None),
            jnp.ones(2),
            key=None,
            num_samples=1,
            stop_after_inputs=True,
        )
    )()
    assert set(output) == {"single_inputs"}
    np.testing.assert_array_equal(output["single_inputs"], expected)


def test_real_toy_embedder_matches_trunk_capture():
    from foldjax.models.protenix.models.predict import protenix_predict_static

    from .test_model import _toy_features, _toy_params

    kwargs = dict(
        key=None,
        num_samples=1,
        num_sampling_steps=1,
        recycling_steps=1,
        input_atom_heads=1,
        atom_encoder_heads=1,
        token_heads=1,
        atom_decoder_heads=1,
        n_queries=2,
        n_keys=4,
        sigma_data=4.0,
        capture_names=("single_inputs",),
        graph_jit=True,
    )
    params, features = _toy_params(), _toy_features()
    expected = protenix_predict_static(
        params, features, stop_after_trunk=True, **kwargs
    )
    actual = protenix_predict_static(params, features, stop_after_inputs=True, **kwargs)
    np.testing.assert_array_equal(actual["single_inputs"], expected["single_inputs"])
