"""ESMFold2 managed inference releases input-only owners after dispatch."""

from __future__ import annotations

import dataclasses
import gc
import json
import subprocess
import sys
import textwrap
import weakref
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from foldjax.backends.esmfold2 import ESMFold2Backend
from foldjax.models import _representations
from foldjax.models.esmfold2 import output as real_output
from foldjax.models.esmfold2.data.features import build_features
from foldjax.schema import PredictionRequest
from tests.models.cp_probe_env import inherited_environment


def _job(tmp_path, *, all_atom: bool) -> object:
    path = tmp_path / ("ligand.json" if all_atom else "protein.json")
    entities = (
        [{"type": "ligand", "id": "L", "ccd": "ATP"}]
        if all_atom
        else [{"type": "protein", "id": "A", "sequence": "AG"}]
    )
    path.write_text(json.dumps({"name": "test", "entities": entities}))
    return path


def _weights(tmp_path, *, session: bool = False) -> object:
    weights = tmp_path / "weights"
    weights.mkdir()
    if session:
        esmc = weights / "esmc"
        esmc.mkdir()
        (weights / "model.safetensors").write_bytes(b"structure")
        (weights / "config.json").write_text("{}")
        (esmc / "model.safetensors").write_bytes(b"language-model")
        (esmc / "config.json").write_text("{}")
    return weights


def _feature_tree(
    feature_refs: list[weakref.ReferenceType[np.ndarray]],
    *,
    all_biomolecule: bool,
) -> dict[str, np.ndarray]:
    features = build_features([("AG", "A", 0, 0)])
    tokens = features["token_attention_mask"].shape[-1]
    if all_biomolecule:
        features["token_chain_id_chars"] = np.full(
            (1, tokens, 1), ord("A"), dtype=np.uint8
        )
        features["token_residue_name_chars"] = np.asarray(
            [[[ord(char) for char in "ALA"], [ord(char) for char in "GLY"]]],
            dtype=np.uint8,
        )
    large_temporary = np.ones(1 << 18, dtype=np.float32)
    assert large_temporary.nbytes == 1 << 20
    feature_refs.append(weakref.ref(large_temporary))
    features["input_only_large_temporary"] = large_temporary
    return features


def _model() -> SimpleNamespace:
    return SimpleNamespace(
        has_language_model=True,
        settings=SimpleNamespace(max_msa_depth=16, msa_n_layers=1, num_recycles=3),
    )


def _written(tmp_path) -> dict[str, object]:
    return {
        "structures": [tmp_path / "sample_0.cif"],
        "summary": [{"sample": 0, "plddt": 0.75}],
    }


def _install_modules(monkeypatch, inference, writer) -> None:
    modules = {
        "foldjax.models.esmfold2.inference": inference,
        "foldjax.models.esmfold2.output": SimpleNamespace(
            project_generated_output_features=(
                real_output.project_generated_output_features
            ),
            write_prediction_outputs=writer,
        ),
    }
    monkeypatch.setattr(
        "foldjax.backends.esmfold2.import_module", lambda name: modules[name]
    )


def test_split_releases_raw_lm_and_full_features_before_export_and_writer(
    tmp_path, monkeypatch
) -> None:
    feature_refs: list[weakref.ReferenceType[np.ndarray]] = []
    lm_refs: list[weakref.ReferenceType[np.ndarray]] = []
    events: list[str] = []
    model = _model()
    prediction_key = object()

    def build_common(*args, **kwargs):
        del args, kwargs
        return _feature_tree(feature_refs, all_biomolecule=True)

    def language_model_states(features, loaded, *, packed_length):
        assert "input_only_large_temporary" in features
        assert loaded is model
        assert packed_length is None
        state = np.ones((1, 4, 81, 1024), dtype=np.float32)
        assert state.nbytes == 1_327_104
        lm_refs.append(weakref.ref(state))
        return state

    def predict(key, features, loaded, **kwargs):
        events.append("predict")
        assert key is prediction_key
        assert loaded is model
        assert "input_only_large_temporary" in features
        assert kwargs["precomputed_lm_states"] is lm_refs[-1]()
        assert "precomputed_lm_embedding" not in kwargs
        assert kwargs["return_distogram_logits"] is False
        assert kwargs["return_auxiliary_outputs"] is False
        return {"single": np.ones((1, 2, 3), dtype=np.float32)}

    inference = SimpleNamespace(
        COMPACT_LANGUAGE_MODEL_API=True,
        MANAGED_AUXILIARY_OUTPUT_API=True,
        LANGUAGE_MODEL_FEATURES=(
            "input_ids",
            "asym_id",
            "residue_index",
            "mol_type",
            "token_attention_mask",
        ),
        load=lambda *args, **kwargs: model,
        seed_key=lambda seed: prediction_key,
        build_common_job_features=build_common,
        build_job_features=lambda *args: pytest.fail("legacy builder was used"),
        language_model_states=language_model_states,
        language_model_embedding=lambda *args, **kwargs: pytest.fail(
            "single split was changed to the compact graph boundary"
        ),
        predict=predict,
    )

    original_save = _representations.save

    def checking_save(*args, **kwargs):
        gc.collect()
        assert feature_refs[-1]() is None
        assert lm_refs[-1]() is None
        events.append("representations")
        return original_save(*args, **kwargs)

    def writer(prediction, features, *args, **kwargs):
        del prediction, args, kwargs
        gc.collect()
        assert feature_refs[-1]() is None
        assert lm_refs[-1]() is None
        assert set(features) == real_output.ESMFOLD2_GENERATED_OUTPUT_FEATURE_FIELDS
        events.append("writer")
        return _written(tmp_path)

    monkeypatch.setattr(_representations, "save", checking_save)
    _install_modules(monkeypatch, inference, writer)
    backend = ESMFold2Backend()
    monkeypatch.setattr(backend, "_ccd_memory_scope", nullcontext)

    result = backend.predict(
        PredictionRequest(
            model="esmfold2",
            input=_job(tmp_path, all_atom=True),
            weights=_weights(tmp_path),
            output_dir=tmp_path / "out",
            representations=("single",),
        )
    )

    assert events == ["predict", "representations", "writer"]
    assert result.samples[0].scores == {"plddt": 0.75}
    assert lm_refs[0]() is None
    assert feature_refs[0]() is None


def test_scalar_predict_job_graph_is_preserved_and_full_features_are_released(
    tmp_path, monkeypatch
) -> None:
    feature_refs: list[weakref.ReferenceType[np.ndarray]] = []
    seen: list[tuple[object, dict[str, object]]] = []
    model = _model()
    prediction_key = object()

    def predict_job(key, chains, alignments, loaded, **kwargs):
        del chains, alignments
        assert loaded is model
        seen.append((key, dict(kwargs)))
        return {}, _feature_tree(feature_refs, all_biomolecule=False)

    inference = SimpleNamespace(
        MANAGED_AUXILIARY_OUTPUT_API=True,
        LANGUAGE_MODEL_FEATURES=("input_ids",),
        load=lambda *args, **kwargs: model,
        seed_key=lambda seed: prediction_key,
        build_job_features=lambda *args: pytest.fail("split builder was used"),
        language_model_states=lambda *args, **kwargs: pytest.fail(
            "split LM was used"
        ),
        predict=lambda *args, **kwargs: pytest.fail("split predictor was used"),
        predict_job=predict_job,
    )

    def writer(prediction, features, *args, **kwargs):
        del prediction, args, kwargs
        gc.collect()
        assert feature_refs[-1]() is None
        assert "input_only_large_temporary" not in features
        return _written(tmp_path)

    _install_modules(monkeypatch, inference, writer)
    ESMFold2Backend().predict(
        PredictionRequest(
            model="esmfold2",
            input=_job(tmp_path, all_atom=False),
            weights=_weights(tmp_path),
            output_dir=tmp_path / "out",
        )
    )

    assert seen == [
        (
            prediction_key,
            {
                "return_distogram_logits": False,
                "return_auxiliary_outputs": False,
            },
        )
    ]


def test_writer_exception_observes_released_input_only_features(
    tmp_path, monkeypatch
) -> None:
    feature_refs: list[weakref.ReferenceType[np.ndarray]] = []
    model = _model()
    error = OSError("writer failed")
    inference = SimpleNamespace(
        load=lambda *args, **kwargs: model,
        seed_key=lambda seed: seed,
        predict_job=lambda *args, **kwargs: (
            {},
            _feature_tree(feature_refs, all_biomolecule=False),
        ),
    )

    def fail_writer(prediction, features, *args, **kwargs):
        del prediction, args, kwargs
        gc.collect()
        assert feature_refs[-1]() is None
        assert "input_only_large_temporary" not in features
        raise error

    _install_modules(monkeypatch, inference, fail_writer)
    with pytest.raises(OSError, match="writer failed") as caught:
        ESMFold2Backend().predict(
            PredictionRequest(
                model="esmfold2",
                input=_job(tmp_path, all_atom=False),
                weights=_weights(tmp_path),
                output_dir=tmp_path / "out",
            )
        )

    assert caught.value is error


def test_predict_exception_rebinds_every_caller_owner(
    tmp_path, monkeypatch
) -> None:
    feature_refs: list[weakref.ReferenceType[np.ndarray]] = []
    lm_refs: list[weakref.ReferenceType[np.ndarray]] = []
    model = _model()

    def language_model_states(*args, **kwargs):
        del args, kwargs
        state = np.ones(1 << 18, dtype=np.float32)
        lm_refs.append(weakref.ref(state))
        return state

    def fail_predict(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("compiled inference failed")

    inference = SimpleNamespace(
        LANGUAGE_MODEL_FEATURES=("input_ids",),
        load=lambda *args, **kwargs: model,
        seed_key=lambda seed: seed,
        build_common_job_features=lambda *args, **kwargs: _feature_tree(
            feature_refs, all_biomolecule=True
        ),
        build_job_features=lambda *args: pytest.fail("legacy builder was used"),
        language_model_states=language_model_states,
        predict=fail_predict,
    )
    _install_modules(
        monkeypatch,
        inference,
        lambda *args, **kwargs: pytest.fail("writer ran after prediction failure"),
    )
    backend = ESMFold2Backend()
    monkeypatch.setattr(backend, "_ccd_memory_scope", nullcontext)

    with pytest.raises(RuntimeError, match="compiled inference failed") as caught:
        backend.predict(
            PredictionRequest(
                model="esmfold2",
                input=_job(tmp_path, all_atom=True),
                weights=_weights(tmp_path),
                output_dir=tmp_path / "out",
            )
        )

    traceback = caught.value.__traceback__
    backend_locals = None
    while traceback is not None:
        if (
            traceback.tb_frame.f_code.co_name == "predict"
            and traceback.tb_frame.f_globals.get("__name__")
            == "foldjax.backends.esmfold2"
        ):
            backend_locals = dict(traceback.tb_frame.f_locals)
            break
        traceback = traceback.tb_next
    assert backend_locals is not None
    assert backend_locals["lm_input"] is None
    assert backend_locals["model_features"] is None
    assert backend_locals["prebuilt_features"] is None
    assert backend_locals["output_features"] is None

    del traceback, backend_locals, caught
    gc.collect()
    assert lm_refs[0]() is None
    assert feature_refs[0]() is None


def test_trunk_only_releases_inputs_before_representation_export(
    tmp_path, monkeypatch
) -> None:
    feature_refs: list[weakref.ReferenceType[np.ndarray]] = []
    lm_refs: list[weakref.ReferenceType[np.ndarray]] = []
    model = _model()

    def language_model_states(*args, **kwargs):
        del args, kwargs
        state = np.ones(1 << 18, dtype=np.float32)
        lm_refs.append(weakref.ref(state))
        return state

    inference = SimpleNamespace(
        MANAGED_AUXILIARY_OUTPUT_API=True,
        LANGUAGE_MODEL_FEATURES=("input_ids",),
        load=lambda *args, **kwargs: model,
        seed_key=lambda seed: seed,
        build_common_job_features=lambda *args, **kwargs: _feature_tree(
            feature_refs, all_biomolecule=True
        ),
        build_job_features=lambda *args: pytest.fail("legacy builder was used"),
        language_model_states=language_model_states,
        predict=lambda *args, **kwargs: {
            "single": np.ones((1, 2, 3), dtype=np.float32)
        },
    )
    _install_modules(
        monkeypatch,
        inference,
        lambda *args, **kwargs: pytest.fail("trunk-only called the writer"),
    )
    backend = ESMFold2Backend()
    monkeypatch.setattr(backend, "_ccd_memory_scope", nullcontext)
    original_save = _representations.save

    def checking_save(*args, **kwargs):
        gc.collect()
        assert feature_refs[-1]() is None
        assert lm_refs[-1]() is None
        return original_save(*args, **kwargs)

    monkeypatch.setattr(_representations, "save", checking_save)
    result = backend.predict(
        PredictionRequest(
            model="esmfold2",
            input=_job(tmp_path, all_atom=True),
            weights=_weights(tmp_path),
            output_dir=tmp_path / "out",
            stop_after="trunk",
            representations=("single",),
        )
    )

    assert result.samples == ()
    assert result.representations is not None


def test_multi_seed_session_keeps_one_compact_embedding_not_feature_trees(
    tmp_path, monkeypatch
) -> None:
    feature_refs: list[weakref.ReferenceType[np.ndarray]] = []
    embedding_refs: list[weakref.ReferenceType[np.ndarray]] = []
    predict_embedding_ids: list[int] = []
    prediction_keys: list[tuple[str, int]] = []
    model = _model()

    def language_model_embedding(*args, **kwargs):
        del args, kwargs
        embedding = np.ones((1, 2, 256), dtype=np.float32)
        embedding_refs.append(weakref.ref(embedding))
        return embedding

    def predict(key, features, loaded, **kwargs):
        assert loaded is model
        assert "input_only_large_temporary" in features
        embedding = kwargs["precomputed_lm_embedding"]
        predict_embedding_ids.append(id(embedding))
        prediction_keys.append(key)
        return {}

    inference = SimpleNamespace(
        COMPACT_LANGUAGE_MODEL_API=True,
        LANGUAGE_MODEL_FEATURES=(
            "input_ids",
            "asym_id",
            "residue_index",
            "mol_type",
            "token_attention_mask",
        ),
        load=lambda *args, **kwargs: model,
        seed_key=lambda seed: ("seed", seed),
        build_job_features=lambda *args: _feature_tree(
            feature_refs, all_biomolecule=False
        ),
        language_model_states=lambda *args, **kwargs: pytest.fail(
            "session used raw LM states"
        ),
        language_model_embedding=language_model_embedding,
        predict=predict,
    )

    def writer(*args, **kwargs):
        del args, kwargs
        gc.collect()
        assert feature_refs[-1]() is None
        assert embedding_refs[0]() is not None
        return _written(tmp_path)

    _install_modules(monkeypatch, inference, writer)
    weights = _weights(tmp_path, session=True)
    request = PredictionRequest(
        model="esmfold2",
        input=_job(tmp_path, all_atom=False),
        weights=weights,
        output_dir=tmp_path / "out",
        num_seeds=2,
    )
    backend = ESMFold2Backend()

    with backend.session((request,)):
        for seed in request.resolved_seeds:
            backend.predict(
                dataclasses.replace(
                    request,
                    seed=seed,
                    num_seeds=None,
                    output_dir=tmp_path / f"out-{seed}",
                )
            )
        assert backend._lm_embedding is embedding_refs[0]()

    gc.collect()
    assert len(embedding_refs) == 1
    assert embedding_refs[0]() is None
    assert all(reference() is None for reference in feature_refs)
    assert len(set(predict_embedding_ids)) == 1
    assert prediction_keys == [("seed", 0), ("seed", 1)]


_ASYNC_OWNERSHIP_PROBE = textwrap.dedent(
    r"""
    import gc
    import weakref

    import jax
    import numpy as np
    from jax.sharding import NamedSharding, PartitionSpec

    from foldjax.models._cp import context_parallel

    assert jax.device_count() == 4, jax.devices()
    expected = np.arange(4096, dtype=np.float32).reshape(1024, 4)

    serial_input = jax.device_put(expected)
    serial_reference = weakref.ref(serial_input)
    serial_output = jax.jit(lambda value: value * 3 + 1)(serial_input)
    del serial_input
    gc.collect()
    assert serial_reference() is None
    np.testing.assert_array_equal(np.asarray(serial_output), expected * 3 + 1)

    with context_parallel(4, layout="1d") as mesh:
        sharded_input = jax.device_put(
            expected,
            NamedSharding(mesh, PartitionSpec("cp", None)),
        )
        sharded_reference = weakref.ref(sharded_input)
        sharded_output = jax.jit(lambda value: value * 5 - 2)(sharded_input)
        del sharded_input
        gc.collect()
        assert sharded_reference() is None
        np.testing.assert_array_equal(np.asarray(sharded_output), expected * 5 - 2)

    print("ESMFOLD2_ASYNC_INPUT_OWNERSHIP_OK")
    """
)


def test_async_dispatch_owns_inputs_on_serial_and_four_cpu_devices() -> None:
    completed = subprocess.run(
        [sys.executable, "-c", _ASYNC_OWNERSHIP_PROBE],
        capture_output=True,
        text=True,
        env={
            "JAX_PLATFORMS": "cpu",
            "XLA_FLAGS": "--xla_force_host_platform_device_count=4",
            **inherited_environment(),
        },
        timeout=240,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "ESMFOLD2_ASYNC_INPUT_OWNERSHIP_OK" in completed.stdout
