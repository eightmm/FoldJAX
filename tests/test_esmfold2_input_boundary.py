from dataclasses import replace
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bench.esmfold2_input_boundary import INPUT_NAMES, candidate_prefix
from foldjax.models.esmfold2.models import model
from tests.models.esmfold2.test_distogram_output_flag import _cheap_features


@pytest.mark.parametrize("compiled", [False, True])
def test_real_prefix_stops_before_parameters_and_restores_function(compiled):
    original = model.inputs_embedding
    settings = replace(model.ModelSettings(), trunk_dtype="bfloat16")

    def run(f):
        return candidate_prefix(model, f, settings)

    fields = (jax.jit(run) if compiled else run)(_cheap_features())
    assert set(fields) == set(INPUT_NAMES)
    assert model.inputs_embedding is original
    assert fields["profile"].shape == (1, 2, 33)
    assert fields["atom_to_token"].shape == (1, 2)
    assert fields["ref_pos"].dtype == jnp.float32


@pytest.mark.parametrize("behavior", ["not_reached", "exception", "bad_signature"])
def test_prefix_rejects_missing_boundary_and_restores_on_failure(behavior):
    def embedding(*a, **kw):
        raise AssertionError("must not execute learned embedding")

    module = SimpleNamespace(inputs_embedding=embedding)

    def predict(*a, **kw):
        if behavior == "exception":
            raise RuntimeError("upstream failure")
        if behavior == "bad_signature":
            module.inputs_embedding()

    module.predict = predict
    with pytest.raises(RuntimeError if behavior == "exception" else ValueError):
        candidate_prefix(module, {}, None)
    assert module.inputs_embedding is embedding


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("cp", [False, True])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_predict_sequence_features_are_not_prematurely_rounded(
    monkeypatch, compiled, cp, dtype
):
    monkeypatch.setattr(model, "cp_mesh", lambda: object() if cp else None)
    features = _cheap_features()
    features["msa"] = jnp.array([[[0, 0], [0, 1], [1, 1]]], jnp.int32)
    features["msa_attention_mask"] = jnp.ones((1, 3, 2), bool)
    features["deletion_mean"] = jnp.array([[0.1037, 1.0035]], jnp.float32)
    features["ref_pos"] = jnp.full((1, 2, 3), 0.1037, jnp.float32)
    settings = replace(model.ModelSettings(), trunk_dtype=dtype)

    def run(f):
        return candidate_prefix(model, f, settings)

    fields = (jax.jit(run) if compiled else run)(features)
    expected_dtype = jnp.bfloat16 if dtype == "bfloat16" and cp else jnp.float32
    for key in (
        "aatype",
        "profile",
        "deletion_mean",
        "ref_pos",
        "ref_element",
        "ref_atom_name_chars",
    ):
        assert fields[key].dtype == expected_dtype
    np.testing.assert_array_equal(
        fields["ref_pos"], features["ref_pos"].astype(expected_dtype)
    )
    expected_profile = jnp.mean(jax.nn.one_hot(features["msa"], 33), axis=1)
    np.testing.assert_array_equal(
        fields["profile"], expected_profile.astype(expected_dtype)
    )
    np.testing.assert_array_equal(
        fields["deletion_mean"], features["deletion_mean"].astype(expected_dtype)
    )
    if expected_dtype == jnp.float32:
        assert not np.array_equal(
            fields["profile"], expected_profile.astype(jnp.bfloat16)
        )
