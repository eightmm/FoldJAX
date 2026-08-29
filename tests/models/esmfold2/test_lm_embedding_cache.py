from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from safetensors.numpy import save_file

from foldjax.models.esmfold2 import inference
from foldjax.models.esmfold2.bridge import checkpoint as structure_checkpoint
from foldjax.models.esmfold2.models import esmc as esmc_model
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
@pytest.mark.parametrize("output_bias", [True, False])
def test_compiled_embedding_boundary_preserves_the_language_model_pair(
    compute_dtype: str, output_bias: bool
) -> None:
    rng = np.random.default_rng(1)
    hidden = jnp.asarray(
        rng.normal(size=(1, 9, 5, 8)).astype(np.float32)
    )
    parameters = _parameters()
    if not output_bias:
        # The released ESMFold2 checkpoint uses this bias-free projection.
        del parameters["language_model.base_z_linear.1.bias"]
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


def test_language_model_stage_reads_only_the_bias_optional_projection(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parameters = _parameters()
    del parameters["language_model.base_z_linear.1.bias"]
    parameters["large.unrelated.weight"] = jnp.ones((64, 64), jnp.float32)
    save_file(
        {name: np.asarray(value) for name, value in parameters.items()},
        tmp_path / structure_checkpoint.WEIGHTS_NAME,
    )
    (tmp_path / structure_checkpoint.CONFIG_NAME).write_text("{}")
    esmc_path = tmp_path / "esmc"
    esmc_path.mkdir()
    sentinel = {"embed.weight": jnp.ones((2, 2), jnp.bfloat16)}
    monkeypatch.setattr(
        inference.esmc_checkpoint, "load_parameters", lambda *args, **kwargs: sentinel
    )
    monkeypatch.setattr(
        inference.esmc_checkpoint,
        "load_settings",
        lambda *args, **kwargs: esmc_model_settings(),
    )

    staged = inference.load_language_model_stage(tmp_path)

    assert staged.esmc_parameters is sentinel
    assert set(staged.parameters) == {
        "language_model.base_z_linear.0.weight",
        "language_model.base_z_linear.0.bias",
        "language_model.base_z_linear.1.weight",
        "language_model.base_z_combine",
    }


def esmc_model_settings() -> esmc_model.ESMCSettings:
    return esmc_model.ESMCSettings(d_model=2, n_heads=1, n_layers=1, vocab_size=2)


def test_releasing_staged_language_model_waits_before_deleting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class Buffer:
        def delete(self) -> None:
            events.append("delete")

    result = object()
    monkeypatch.setattr(
        inference.jax,
        "block_until_ready",
        lambda value: events.append("ready") if value is result else None,
    )
    loaded = inference.LoadedModel(
        parameters={},
        settings=structure_model.ModelSettings(),
        esmc_parameters={"a": Buffer(), "b": Buffer()},
        esmc_settings=esmc_model_settings(),
    )

    inference.release_language_model_parameters(loaded, after=result)

    assert events == ["ready", "delete", "delete"]
