"""Managed warming and prediction must hand the native graph the same ABI."""

from __future__ import annotations

import dataclasses
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax import api, warmup
from foldjax.backends.esmfold2 import ESMFold2Backend
from foldjax.executable_cache_verify import (
    UnsupportedCacheVerificationError,
    lowered_cache_key,
)
from foldjax.models._jit_pool import BoundedJitPool
from foldjax.models.esmfold2 import inference
from foldjax.models.esmfold2.data.features import build_features
from foldjax.models.esmfold2.models import heads
from foldjax.models.esmfold2.models import model as structure_model
from foldjax.schema import PaddingConfig, PredictionRequest, PredictionResult
from tests.models.esmfold2.test_confidence_dtype import _inputs, _params
from tests.test_esmfold2_backend import _fake_session_weights


@pytest.fixture
def native_graph(monkeypatch):
    """Keep `_run` and its real owner, replacing the expensive model body."""
    settings = dataclasses.replace(
        structure_model.ModelSettings(),
        msa_n_layers=None,
        max_msa_depth=4,
        trunk_dtype=jnp.dtype(jnp.float32),
    )
    model = inference.LoadedModel(
        {
            "token_bonds.weight": jnp.ones((settings.d_pair, 1), dtype=jnp.float32),
            "language_model.base_z_linear.0.weight": jnp.ones(2),
            "language_model.base_z_linear.0.bias": jnp.zeros(2),
            "language_model.base_z_linear.1.weight": jnp.ones((4, 2)),
            "language_model.base_z_combine": jnp.ones(3),
        },
        settings,
        esmc_parameters={},
    )
    pool = BoundedJitPool(
        inference._run, static_argnames=inference._COMPILED_PREDICT_STATIC_ARGNAMES
    )
    monkeypatch.setattr(inference, "_compiled_predict_pool", pool)
    inference._compiled_predict_factory.cache_clear()
    lowered = []
    keys = []
    original_factory = inference.compiled_predict

    def recorded_factory(*args, **kwargs):
        runner = original_factory(*args, **kwargs)

        def run(*dynamic, **options):
            graph = runner.lower(*dynamic, **options)
            lowered.append(graph.as_text())
            keys.append(lowered_cache_key(graph))
            return runner(*dynamic, **options)

        run.lower = runner.lower
        return run

    def body(
        key,
        features,
        parameters,
        *,
        n_chains,
        lm_embedding=None,
        lm_hidden_states=None,
        **kwargs,
    ):
        del parameters, kwargs
        hidden = lm_embedding if lm_embedding is not None else lm_hidden_states
        signal = jnp.sum(hidden) if hidden is not None else jnp.float32(0)
        signal += jax.random.uniform(key)
        membership = jax.nn.one_hot(features["asym_id"], n_chains)
        membership *= features["token_attention_mask"][..., None]
        matrix = jnp.einsum("btc,btd->bcd", membership, membership)
        return {
            "pair_chains_iptm": matrix + signal,
            "sample_atom_coords": features["ref_pos"] + signal,
        }

    monkeypatch.setattr(inference, "compiled_predict", recorded_factory)
    monkeypatch.setattr(structure_model, "predict", body)
    yield model, lowered, pool, keys
    pool.clear_cache()
    inference._compiled_predict_factory.cache_clear()


@pytest.mark.parametrize("keep_output", [False, True])
@pytest.mark.parametrize(
    "variant", ["released", "no_lm", "exact_no_lm", "inputs", "trunk"]
)
def test_warm_and_single_batch_multiseed_prediction_share_native_lowering(
    tmp_path, monkeypatch, native_graph, keep_output, variant
):
    model, lowered, pool, keys = native_graph
    weights = _fake_session_weights(tmp_path)
    events = []

    def load(_weights, *, language_model, **kwargs):
        del kwargs
        events.append("complete" if language_model else "structure")
        return (
            model
            if language_model
            else dataclasses.replace(model, esmc_parameters=None)
        )

    def stage(*args, **kwargs):
        del args, kwargs
        events.append("stage")
        return model

    def embedding(features, loaded, **kwargs):
        del kwargs
        assert loaded.has_language_model
        events.append("embedding")
        return jnp.broadcast_to(
            jnp.asarray(features["input_ids"], dtype=jnp.float32)[..., None],
            (*features["input_ids"].shape, 4),
        )

    def states(features, loaded, **kwargs):
        if not loaded.has_language_model:
            return None
        events.append("states")
        return jnp.broadcast_to(
            embedding(features, loaded, **kwargs)[..., None, :],
            (*features["input_ids"].shape, 3, 4),
        )

    def features(document, **kwargs):
        del kwargs
        entity = document["entities"][0]
        return build_features([(entity["sequence"], "A", 0, 0)])

    monkeypatch.setattr(inference, "load", load)
    monkeypatch.setattr(inference, "load_language_model_stage", stage)
    monkeypatch.setattr(
        inference,
        "release_language_model_parameters",
        lambda *_args, **_kwargs: events.append("release"),
    )
    monkeypatch.setattr(inference, "language_model_embedding", embedding)
    monkeypatch.setattr(inference, "language_model_states", states)
    monkeypatch.setattr(inference, "build_common_job_features", features)
    from foldjax.models.esmfold2 import output

    monkeypatch.setattr(
        output,
        "write_prediction_outputs",
        lambda *args, **kwargs: {
            "structures": [],
            "summary": [],
        },
    )
    for name, sequence in (
        ("warm", "AG"),
        ("predict", "AG" if variant == "exact_no_lm" else "AGG"),
    ):
        (tmp_path / f"{name}.json").write_text(
            json.dumps(
                {
                    "name": name,
                    "entities": [
                        {"type": "protein", "id": ["A"], "sequence": sequence}
                    ],
                }
            )
        )

    request = PredictionRequest(
        model="esmfold2",
        input=tmp_path / "warm.json",
        weights=weights,
        output_dir=tmp_path / "out",
        cache_dir=tmp_path / "compile",
        seed=7,
        padding=None
        if variant == "exact_no_lm"
        else PaddingConfig(
            tokens=8,
            atoms=192,
            msa=4,
            language_model_tokens=None if variant == "no_lm" else 16,
        ),
        options={
            "memory_check": "warn",
            **(
                {"no_language_model": True}
                if variant in {"no_lm", "exact_no_lm"}
                else {}
            ),
        },
        stop_after=variant if variant in {"inputs", "trunk"} else "full",
        representations=("single_inputs",) if variant in {"inputs", "trunk"} else None,
    )

    # Native model/feature/staging dispatch stays real. Output and the common
    # manifest plumbing are irrelevant to the structure executable's inputs.
    def predict_resolved(run, *, backend=None, **kwargs):
        del kwargs
        active = backend if backend is not None else ESMFold2Backend()
        for seed in run.resolved_seeds:
            result = active.predict(
                dataclasses.replace(
                    run,
                    seed=seed,
                    seeds=None,
                    num_seeds=None,
                )
            )
        return result

    monkeypatch.setattr(warmup, "_predict_resolved", predict_resolved)
    monkeypatch.setattr(api, "_predict_resolved", predict_resolved)
    monkeypatch.setattr(warmup, "get_backend", lambda _: ESMFold2Backend())
    monkeypatch.setattr(api, "get_backend", lambda _: ESMFold2Backend())
    monkeypatch.setattr(warmup, "resolve_cache_dir", lambda *_: tmp_path / "compile")
    monkeypatch.setattr(warmup, "runtime_dir", lambda *_: tmp_path / "runtime")
    monkeypatch.setattr(api, "_preflight_or_record", lambda *_: True)
    warmup._warm_resolved(request, keep_output=keep_output, output_root=None)
    if variant in {"released", "trunk"}:
        assert events[:4] == ["stage", "embedding", "release", "structure"]
    else:
        assert events == ["structure"]
    assert "states" not in events
    expected = lowered[0]

    predict = dataclasses.replace(request, input=tmp_path / "predict.json")
    scenarios = (
        (predict,),
        (predict, request),
        (dataclasses.replace(predict, seed=None, seeds=(7, 11)),),
    )
    for requests in scenarios:
        monkeypatch.setattr(api, "resolve_requests", lambda *a, **k: requests)
        report = api.predict_batch(requests[0])
        assert all(isinstance(result, PredictionResult) for result in report.results)
    assert len(lowered) == 6
    assert all(text == expected for text in lowered)
    assert len(set(keys)) == 1
    assert pool._cache_size() == 1
    assert "states" not in events

    def forbidden(*args, **kwargs):
        del args, kwargs
        pytest.fail("lowering-only verification dispatched or executed a model")

    from jax._src import compiler, dispatch

    for name in (
        "load_language_model_stage",
        "release_language_model_parameters",
        "language_model_states",
        "language_model_embedding",
    ):
        monkeypatch.setattr(inference, name, forbidden)
    monkeypatch.setattr(output, "write_prediction_outputs", forbidden)
    monkeypatch.setattr(BoundedJitPool, "__call__", forbidden)
    monkeypatch.setattr(compiler, "compile_or_get_cached", forbidden)
    monkeypatch.setattr(dispatch, "apply_primitive", forbidden)
    backend = ESMFold2Backend()
    with backend.session((predict,)):
        ((name, graph),) = backend.lower_cache_graphs(predict)
    assert name == "jit(_run)"
    assert graph.as_text() == expected
    assert lowered_cache_key(graph) == keys[0]


def test_chain_extents_share_native_graph_and_retain_public_matrix_shape(native_graph):
    model, lowered, pool, keys = native_graph
    model = dataclasses.replace(model, esmc_parameters=None)
    features = build_features([("A" * 12, "A", 0, 0)])
    original = inference.structure_model.predict
    for count in (1, 5, 12):
        current = dict(features, asym_id=(np.arange(12) % count)[None])
        expected = original(
            jax.random.key(7), current, model.parameters, n_chains=count
        )
        actual = inference.predict(
            jax.random.key(7), current, model, memory_check="off"
        )
        assert isinstance(actual["pair_chains_iptm"], jax.Array)
        assert actual["pair_chains_iptm"].shape[-2:] == (count, count)
        for name in expected:
            np.testing.assert_allclose(actual[name], expected[name], atol=1e-6)
    assert len(set(lowered)) == 1
    assert len(set(keys)) == 1
    assert pool._cache_size() == 1


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("count", [1, 2, 3, 4, 5, 12])
def test_real_confidence_head_fixed_slots_match_exact_extent(dtype, count):
    inputs = _inputs()
    inputs["s_inputs"] = jnp.tile(inputs["s_inputs"], (1, 3, 1))
    inputs["z"] = jnp.tile(inputs["z"], (1, 3, 3, 1))
    inputs["x_pred"] = jnp.tile(inputs["x_pred"], (1, 3, 1))
    for name in ("token_mask", "atom_mask", "mol_type"):
        inputs[name] = jnp.tile(inputs[name], (1, 3))
    inputs["distogram_atom_idx"] = jnp.arange(12)[None]
    inputs["atom_to_token"] = jnp.arange(12)[None]
    inputs["asym_id"] = (jnp.arange(12) % count)[None]
    params = _params()
    expected = heads.confidence_head(
        **inputs,
        params=params,
        n_layers=0,
        n_chains=count,
        trunk_dtype=dtype,
        confidence_dtype=dtype,
    )
    actual = heads.confidence_head(
        **inputs,
        params=params,
        n_layers=0,
        n_chains=64,
        trunk_dtype=dtype,
        confidence_dtype=dtype,
    )
    for name in expected:
        value = actual[name]
        if name == "pair_chains_iptm":
            value = value[..., :count, :count]
        np.testing.assert_allclose(value, expected[name], atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("max_msa_depth", [None, 1, 2, 4])
def test_lowering_msa_normalization_keeps_the_prediction_signature(max_msa_depth):
    features = build_features([("AGG", "A", 0, 0)])
    for name in ("msa", "msa_attention_mask", "has_deletion", "deletion_value"):
        features[name] = np.repeat(features[name], 3, axis=1)
    kwargs = dict(n_msa=4, max_msa_depth=max_msa_depth, total_steps=4)
    real = inference.normalize_msa_features(jax.random.key(7), features, **kwargs)
    abstract = inference.normalize_msa_features(
        None,
        features,
        **kwargs,
        _shape_only=True,
    )
    assert real.keys() == abstract.keys()
    for name in real:
        assert np.asarray(real[name]).shape == np.asarray(abstract[name]).shape, name
        assert np.asarray(real[name]).dtype == np.asarray(abstract[name]).dtype, name


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_abstract_compact_embedding_matches_the_real_projection(native_graph, dtype):
    model, _, _, _ = native_graph
    model = dataclasses.replace(
        model,
        settings=dataclasses.replace(model.settings, trunk_dtype=jnp.dtype(dtype)),
    )
    features = build_features([("AGG", "A", 0, 0)])
    hidden = jnp.ones((1, 3, 3, 2), dtype=jnp.float32)
    expected = inference._language_model_embedding_from_states(hidden, model)
    actual = inference.abstract_language_model_embedding(features, model)
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype


def test_lowering_refuses_unverifiable_singleton_lm_route(tmp_path, monkeypatch):
    path = tmp_path / "job.json"
    path.write_text(json.dumps({"entities": []}))
    request = PredictionRequest(
        model="esmfold2",
        input=path,
        weights=_fake_session_weights(tmp_path),
        seed=7,
    )
    monkeypatch.setattr(
        "foldjax.backends.esmfold2._model_asset_snapshot", lambda *a, **k: None
    )
    monkeypatch.setattr(
        inference,
        "load",
        lambda *a, **k: pytest.fail("unsupported route loaded weights"),
    )
    backend = ESMFold2Backend()
    with (
        backend.session((request,)),
        pytest.raises(UnsupportedCacheVerificationError, match="raw-state fallback"),
    ):
        backend.lower_cache_graphs(request)


def test_lowering_refuses_legacy_prediction_wrapper(tmp_path, monkeypatch):
    path = tmp_path / "job.json"
    path.write_text(json.dumps({"entities": []}))
    request = PredictionRequest(
        model="esmfold2",
        input=path,
        weights=_fake_session_weights(tmp_path),
        seed=7,
    )
    monkeypatch.setattr(inference, "MANAGED_CACHE_LOWERING_API", False)
    backend = ESMFold2Backend()
    with (
        backend.session((request,)),
        pytest.raises(
            UnsupportedCacheVerificationError, match="wrapper does not provide"
        ),
    ):
        backend.lower_cache_graphs(request)
