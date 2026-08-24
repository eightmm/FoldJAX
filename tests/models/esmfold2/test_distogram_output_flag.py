"""The common backend omits ESMFold2's otherwise-public native distogram."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.esmfold2 import inference
from foldjax.models.esmfold2 import output as output_module
from foldjax.models.esmfold2.data.features import build_features, pad_features
from foldjax.models.esmfold2.models import model as structure_model
from foldjax.models.esmfold2.output import SAMPLE_SCORES


def _cheap_features() -> dict[str, jax.Array]:
    token = jnp.arange(2, dtype=jnp.int32)[None]
    atom = jnp.arange(2, dtype=jnp.int32)[None]
    return {
        "token_index": token,
        "residue_index": token,
        "asym_id": jnp.zeros_like(token),
        "sym_id": jnp.zeros_like(token),
        "entity_id": jnp.ones_like(token),
        "mol_type": jnp.zeros_like(token),
        "res_type": token,
        "token_bonds": jnp.zeros((1, 2, 2, 1), dtype=jnp.float32),
        "token_attention_mask": jnp.ones((1, 2), dtype=bool),
        "ref_pos": jnp.zeros((1, 2, 3), dtype=jnp.float32),
        "ref_element": jnp.ones((1, 2), dtype=jnp.int32),
        "ref_charge": jnp.zeros((1, 2), dtype=jnp.float32),
        "ref_atom_name_chars": jnp.zeros((1, 2, 4), dtype=jnp.int32),
        "ref_space_uid": atom,
        "atom_attention_mask": jnp.ones((1, 2), dtype=bool),
        "atom_to_token": atom,
        "distogram_atom_idx": atom,
    }


def test_direct_eager_default_retains_distogram_and_false_skips_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep the public low-level result while proving the dead branch is skipped."""

    settings = dataclasses.replace(
        structure_model.ModelSettings(),
        d_pair=2,
        d_inputs=2,
        trunk_n_layers=0,
        lm_encoder_n_layers=None,
        coda_n_layers=0,
        confidence_n_layers=0,
        msa_n_layers=None,
        num_loops=0,
        num_samples=1,
        confidence_sample_sequential=False,
        trunk_dtype="float32",
    )
    calls: list[str] = []

    monkeypatch.setattr(
        structure_model,
        "one_hot_atom_features",
        lambda *args, **kwargs: (
            jnp.zeros((1, 2, 128), dtype=jnp.float32),
            jnp.zeros((1, 2, 4, 64), dtype=jnp.float32),
        ),
    )
    monkeypatch.setattr(
        structure_model,
        "inputs_embedding",
        lambda *args, **kwargs: jnp.asarray(
            [[[1.0, -2.0], [3.0, -4.0]]], dtype=jnp.float32
        ),
    )
    monkeypatch.setattr(
        structure_model,
        "relative_position_encoding",
        lambda *args, **kwargs: jnp.zeros((1, 2, 2, 2), dtype=jnp.float32),
    )
    monkeypatch.setattr(
        structure_model,
        "_token_bonds_encoding",
        lambda *args, **kwargs: jnp.zeros((1, 2, 2, 2), dtype=jnp.float32),
    )
    monkeypatch.setattr(
        structure_model,
        "run_loops",
        lambda key, z, z_init, *args, **kwargs: z_init,
    )
    monkeypatch.setattr(
        structure_model, "folding_trunk", lambda value, *args, **kwargs: value
    )

    def fake_linear(value, params, prefix):
        del params
        if prefix in {"z_init_1", "z_init_2", "parcae_readout"}:
            return value
        if prefix == "distogram_head":
            calls.append(prefix)
            return jnp.sum(value, axis=-1, keepdims=True)
        raise AssertionError(f"unexpected linear {prefix}")

    monkeypatch.setattr(structure_model, "linear", fake_linear)
    monkeypatch.setattr(
        structure_model.diffusion,
        "build_cache",
        lambda *args, **kwargs: {"pair": args[7]},
    )

    def fake_sample(key, single, cache, *args, **kwargs):
        del key, single, args, kwargs
        signal = jnp.sum(cache["pair"])
        return jnp.broadcast_to(signal, (1, 2, 3)), None

    monkeypatch.setattr(structure_model.diffusion, "sample", fake_sample)

    def fake_confidence(single, pair, coords, *args, **kwargs):
        del single, args, kwargs
        signal = jnp.sum(pair) + jnp.sum(coords)
        token = jnp.broadcast_to(signal, (1, 2))
        return {
            "plddt": token,
            "plddt_per_atom": token,
            "complex_plddt": jnp.reshape(signal, (1,)),
            "ptm": jnp.reshape(signal, (1,)),
        }

    monkeypatch.setattr(structure_model, "confidence_head", fake_confidence)

    params = {"token_bonds.weight": jnp.ones((2, 1), dtype=jnp.float32)}
    initial = jnp.zeros((1, 2, 2, 2), dtype=jnp.float32)
    with_distogram = structure_model.predict(
        jax.random.key(0),
        _cheap_features(),
        params,
        settings=settings,
        initial_pair_state=initial,
        n_chains=1,
        return_representations=("single", "pair"),
    )
    without_distogram = structure_model.predict(
        jax.random.key(0),
        _cheap_features(),
        params,
        settings=settings,
        initial_pair_state=initial,
        n_chains=1,
        return_representations=("single", "pair"),
        return_distogram_logits=False,
    )

    assert "distogram_logits" in with_distogram
    assert "distogram_logits" not in without_distogram
    assert calls == ["distogram_head"]
    assert set(without_distogram) == set(with_distogram) - {"distogram_logits"}
    for name, value in without_distogram.items():
        assert np.asarray(value).tobytes() == np.asarray(
            with_distogram[name]
        ).tobytes(), name

    def compiled_predict(*, emit: bool):
        return jax.jit(
            lambda key, features, dynamic_params, initial_pair: (
                structure_model.predict(
                    key,
                    features,
                    dynamic_params,
                    settings=settings,
                    initial_pair_state=initial_pair,
                    n_chains=1,
                    return_representations=("single", "pair"),
                    return_distogram_logits=emit,
                )
            )
        )

    compiled_with = compiled_predict(emit=True)(
        jax.random.key(0), _cheap_features(), params, initial
    )
    compiled_without = compiled_predict(emit=False)(
        jax.random.key(0), _cheap_features(), params, initial
    )

    assert calls == ["distogram_head", "distogram_head"]
    assert set(compiled_without) == set(compiled_with) - {"distogram_logits"}
    for name, value in compiled_without.items():
        assert np.asarray(value).tobytes() == np.asarray(
            compiled_with[name]
        ).tobytes(), name


def test_padded_inference_routes_omission_into_the_compiled_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    features = pad_features(
        build_features([("AG", "A", 0, 0)]),
        n_token=8,
        n_atom=64,
        n_msa=4,
    )
    settings = structure_model.ModelSettings()
    model = SimpleNamespace(
        settings=settings,
        parameters={
            "token_bonds.weight": jnp.ones(
                (settings.d_pair, 1), dtype=jnp.bfloat16
            )
        },
        esmc_parameters=None,
        esmc_settings=None,
    )
    seen: list[tuple[bool, bool]] = []

    def fake_compiled_predict(*identity):
        def run(*args):
            seen.append((identity[-1], args[-1]))
            return {}

        return run

    monkeypatch.setattr(inference, "compiled_predict", fake_compiled_predict)
    inference.predict(
        jax.random.key(0),
        features,
        model,
        preserve_prefix_rng=True,
        return_distogram_logits=False,
    )

    assert seen == [(False, False)]


def test_distogram_choice_has_distinct_cached_jit_owners() -> None:
    settings = dataclasses.replace(
        structure_model.ModelSettings(), trunk_n_layers=0, coda_n_layers=0
    )
    inference.compiled_predict.cache_clear()
    try:
        retained = inference.compiled_predict(
            settings, 1, False, 1, (), False, False, False, True
        )
        omitted = inference.compiled_predict(
            settings, 1, False, 1, (), False, False, False, False
        )

        assert retained is inference.compiled_predict(
            settings, 1, False, 1, (), False, False, False, True
        )
        assert omitted is not retained
        assert inference.compiled_predict.cache_info().currsize == 2
    finally:
        inference.compiled_predict.cache_clear()


def test_omitted_distogram_stage_has_no_pair_projection_in_hlo() -> None:
    pair = jnp.zeros((1, 32, 32, 16), dtype=jnp.float32)
    weight = jnp.ones((64, 16), dtype=jnp.float32)
    bias = jnp.zeros((64,), dtype=jnp.float32)

    def stage(pair, weight, bias, *, emit):
        result = {"carry": pair[:, 0, 0]}
        if emit:
            result["distogram_logits"] = structure_model._distogram_logits(
                pair,
                {
                    "distogram_head.weight": weight,
                    "distogram_head.bias": bias,
                },
            )
        return result

    emitted = jax.jit(
        lambda pair, weight, bias: stage(pair, weight, bias, emit=True)
    ).lower(pair, weight, bias)
    omitted = jax.jit(
        lambda pair, weight, bias: stage(pair, weight, bias, emit=False)
    ).lower(pair, weight, bias)

    assert emitted.as_text().count("stablehlo.dot_general") == 1
    assert omitted.as_text().count("stablehlo.dot_general") == 0


def test_writer_accepts_the_backend_output_without_a_distogram(tmp_path) -> None:
    features = build_features([("AG", "A", 0, 0)])
    n_atoms = features["ref_pos"].shape[1]
    prediction = {
        "sample_atom_coords": np.zeros((1, n_atoms, 3), dtype=np.float32),
        "plddt_per_atom": np.full((1, n_atoms), 0.75, dtype=np.float32),
        "plddt": np.full((1, 2), 0.75, dtype=np.float32),
        **{
            name: np.asarray([0.5], dtype=np.float32)
            for name in SAMPLE_SCORES
        },
    }

    written = output_module.write_prediction_outputs(
        prediction, features, tmp_path, name="without-distogram"
    )

    assert len(written["structures"]) == 1
    assert written["structures"][0].is_file()
    assert written["scores"].is_file()
    assert written["summary"][0]["plddt"] == pytest.approx(0.75)
