from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.esmfold2 import inference
from foldjax.models.esmfold2.models import model as structure_model


def _parameters(seed: int = 0) -> dict[str, jnp.ndarray]:
    rng = np.random.default_rng(seed)
    layers, hidden, combined, down, pair_hidden = 5, 8, 6, 4, 7
    shapes = {
        "language_model.base_z_linear.0.weight": (hidden,),
        "language_model.base_z_linear.0.bias": (hidden,),
        "language_model.base_z_linear.1.weight": (combined, hidden),
        "language_model.base_z_linear.1.bias": (combined,),
        "language_model.base_z_combine": (layers,),
        "language_model.base_z_mlp.0.downproject.weight": (down, combined),
        "language_model.base_z_mlp.0.downproject.bias": (down,),
        "language_model.base_z_mlp.0.output_mlp.0.weight": (
            pair_hidden,
            2 * down,
        ),
        "language_model.base_z_mlp.0.output_mlp.0.bias": (pair_hidden,),
        "language_model.base_z_mlp.0.output_mlp.2.weight": (
            combined,
            pair_hidden,
        ),
        "language_model.base_z_mlp.0.output_mlp.2.bias": (combined,),
        "language_model.base_z_mlp.1.weight": (combined,),
        "language_model.base_z_mlp.1.bias": (combined,),
    }
    return {
        name: jnp.asarray(rng.normal(size=shape).astype(np.float32))
        for name, shape in shapes.items()
    }


@pytest.mark.parametrize("compute_dtype", ["float32", "bfloat16"])
def test_compiled_embedding_boundary_preserves_the_language_model_pair(
    compute_dtype: str,
) -> None:
    rng = np.random.default_rng(1)
    hidden = jnp.asarray(
        rng.normal(size=(1, 9, 5, 8)).astype(np.float32)
    )
    parameters = _parameters()
    compute = jnp.dtype(compute_dtype)

    def cast(params):
        return structure_model._cast(  # noqa: SLF001
            params, structure_model.TRUNK_PREFIXES, compute
        )

    historical = jax.jit(
        lambda states, params: structure_model.language_model_pair(
            states.astype(compute), cast(params)
        )
    )(hidden, parameters)
    loaded = inference.LoadedModel(
        parameters=parameters,
        settings=structure_model.ModelSettings(trunk_dtype=compute_dtype),
    )
    embedding = inference._language_model_embedding_from_states(hidden, loaded)
    compact = jax.jit(
        lambda value, params: structure_model.language_model_pair_from_embedding(
            value.astype(compute), cast(params)
        )
    )(embedding, parameters)

    np.testing.assert_array_equal(np.asarray(compact), np.asarray(historical))
    assert embedding.shape == (1, 9, 6)
    assert embedding.size * 5 * 8 == hidden.size * 6
    assert embedding.nbytes < hidden.nbytes


def test_inference_marks_the_compact_input_in_the_static_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ESMFOLD2_ATOM_ATTENTION_BACKEND", raising=False)
    monkeypatch.delenv("ESMFOLD2_ATOM_ROWS_PER_BLOCK", raising=False)
    settings = structure_model.ModelSettings()
    features = {
        "asym_id": np.zeros((1, 2), dtype=np.int32),
        "token_attention_mask": np.ones((1, 2), dtype=bool),
        "token_bonds": np.ones((1, 2, 2, 1), dtype=np.float32),
        "msa": np.asarray([[[0, 32], [1, 2]]], dtype=np.int64),
        "msa_attention_mask": np.ones((1, 2, 2), dtype=np.float32),
        "has_deletion": np.asarray([[[0.0, 1.0], [1.0, 0.0]]], dtype=np.float32),
        "deletion_value": np.zeros((1, 2, 2), dtype=np.float32),
    }
    loaded = inference.LoadedModel(
        parameters={
            "token_bonds.weight": jnp.ones(
                (settings.d_pair, 1), dtype=jnp.bfloat16
            )
        },
        settings=settings,
    )
    identities: list[bool] = []

    def fake_compiled_predict(*identity):
        identities.append(identity[-2])
        assert identity[-1] == 256

        def run(*args):
            assert args[-2] is identity[-2]
            assert args[-1] == identity[-1]
            arrays = args[1]
            assert arrays["msa"].dtype == jnp.uint8
            assert arrays["msa_attention_mask"].dtype == jnp.bool_
            assert arrays["has_deletion"].dtype == jnp.bool_
            assert arrays["deletion_value"].dtype == jnp.float32
            return {}

        return run

    monkeypatch.setattr(inference, "compiled_predict", fake_compiled_predict)

    inference.predict(
        jax.random.key(0),
        features,
        loaded,
        precomputed_lm_embedding=jnp.zeros((1, 2, 256), jnp.bfloat16),
    )
    inference.predict(
        jax.random.key(0),
        features,
        loaded,
        precomputed_lm_states=jnp.zeros((1, 2, 81, 2560), jnp.bfloat16),
    )

    assert identities == [True, False]


def test_inference_rejects_two_precomputed_language_model_inputs() -> None:
    settings = structure_model.ModelSettings()
    loaded = inference.LoadedModel(parameters={}, settings=settings)
    with pytest.raises(ValueError, match="not both"):
        inference.predict(
            jax.random.key(0),
            {
                "asym_id": np.zeros((1, 1), dtype=np.int32),
                "token_attention_mask": np.ones((1, 1), dtype=bool),
            },
            loaded,
            precomputed_lm_states=jnp.zeros((1, 1, 1, 1)),
            precomputed_lm_embedding=jnp.zeros((1, 1, 1)),
        )


def test_compiled_embedding_pool_is_bounded_by_input_signature() -> None:
    parameters = _parameters()
    loaded = inference.LoadedModel(
        parameters=parameters,
        settings=structure_model.ModelSettings(trunk_dtype="float32"),
    )
    inference._compiled_language_model_embedding.cache_clear()
    try:
        for n_token in range(1, 13):
            inference._language_model_embedding_from_states(
                jnp.zeros((1, n_token, 5, 8), jnp.float32), loaded
            )
        assert inference._compiled_language_model_embedding.cache_info().currsize == 8
    finally:
        inference._compiled_language_model_embedding.cache_clear()
