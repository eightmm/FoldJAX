"""The ESMFold2 adapter's contract, without loading 25 GB of weights.

Everything here is the part FoldJAX owns: how a neutral request becomes
upstream's argument names, what the adapter refuses rather than folds wrongly,
and how a job document turns into chains. The model itself is upstream's and
is exercised by the benchmark, not by the suite.
"""

from __future__ import annotations

import dataclasses
import json
import os
from types import SimpleNamespace

import numpy as np
import pytest

from foldjax.backends.esmfold2 import (
    ESMFold2Backend,
    _job_chains,
    _model_asset_snapshot,
    managed_asset_profile,
)
from foldjax.schema import PredictionError, PredictionRequest


def _job(tmp_path, entities) -> str:
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "job.json"
    path.write_text(json.dumps({"name": "t", "entities": entities}))
    return path


def test_the_neutral_schedule_reaches_the_ports_own_names(tmp_path) -> None:
    """`num_recycles` is `num_loops` here; the other three keep their names."""
    job = _job(tmp_path, [{"type": "protein", "id": ["A"], "sequence": "ACDEF"}])
    options = ESMFold2Backend().apply_sampling(
        PredictionRequest(
            model="esmfold2",
            input=job,
            num_samples=5,
            num_steps=200,
            num_recycles=10,
            max_msa_depth=1024,
        )
    )
    assert options == {
        "num_samples": 5,
        "num_steps": 200,
        "num_loops": 10,
        "msa_max_depth": 1024,
    }


def test_the_torch_kernel_knob_is_gone_rather_than_ignored(tmp_path) -> None:
    """The port's attention is XLA's; `fused` and `cueq` no longer exist here.

    Accepting the knob and doing nothing with it would report a kernel choice
    that never happened, which is worse than refusing it.
    """
    job = _job(tmp_path, [{"type": "protein", "id": ["A"], "sequence": "ACDEF"}])
    with pytest.raises(ValueError, match="attention_kernel"):
        ESMFold2Backend().apply_sampling(
            PredictionRequest(
                model="esmfold2", input=job, options={"attention_kernel": "auto"}
            )
        )


def test_managed_asset_profile_tracks_the_language_model_variant() -> None:
    assert managed_asset_profile({}) == "released"
    assert managed_asset_profile({"no_language_model": False}) == "released"
    assert managed_asset_profile({"no_language_model": True}) == "structure-only"
    assert managed_asset_profile({"esmc_weights": "/tmp/esmc"}) == "structure-only"
    assert (
        managed_asset_profile(
            {"no_language_model": False, "esmc_weights": "/tmp/esmc"}
        )
        == "structure-only"
    )
    with pytest.raises(ValueError, match="no_language_model must be a boolean"):
        managed_asset_profile({"no_language_model": "true"})
    with pytest.raises(ValueError, match="esmc_weights cannot be combined"):
        managed_asset_profile(
            {"no_language_model": True, "esmc_weights": "/tmp/esmc"}
        )


def test_external_esmc_keeps_the_released_language_model_branch(
    tmp_path, monkeypatch
) -> None:
    """A smaller managed download must not silently select a smaller model."""
    job = _job(tmp_path, [{"type": "protein", "id": ["A"], "sequence": "ACDEF"}])
    weights = tmp_path / "weights"
    weights.mkdir()
    external_esmc = tmp_path / "external-esmc"
    external_esmc.mkdir()
    seen = {}
    model = SimpleNamespace(has_language_model=True)

    def fake_load(path, *, esmc, language_model):
        seen.update(path=path, esmc=esmc, language_model=language_model)
        return model

    modules = {
        "foldjax.models.esmfold2.inference": SimpleNamespace(
            load=fake_load,
            seed_key=lambda seed: seed,
            predict_job=lambda *args, **kwargs: (object(), object()),
        ),
        "foldjax.models.esmfold2.output": SimpleNamespace(
            write_prediction_outputs=lambda *args, **kwargs: {
                "structures": [tmp_path / "sample_0.cif"],
                "summary": [{"sample": 0, "plddt": 91.0}],
            }
        ),
    }
    monkeypatch.setattr(
        "foldjax.backends.esmfold2.import_module", lambda name: modules[name]
    )

    result = ESMFold2Backend().predict(
        PredictionRequest(
            model="esmfold2",
            input=job,
            weights=weights,
            output_dir=tmp_path / "out",
            options={"esmc_weights": external_esmc},
        )
    )

    assert seen == {
        "path": weights,
        "esmc": external_esmc,
        "language_model": True,
    }
    assert result.raw["language_model"] is True


def test_a_ligand_job_is_refused_rather_than_folded_as_protein(tmp_path) -> None:
    """`forward` expresses ligands; this adapter does not build their features.

    Accepting the job and folding the protein alone would return a structure
    that answers a different question than the one asked.
    """
    assert "ligand" not in ESMFold2Backend().capabilities().entity_types


def test_chain_copies_become_separate_chains_of_one_entity(tmp_path) -> None:
    """A homodimer names one sequence twice: two chains, one entity, two copies."""
    job = _job(tmp_path, [{"type": "protein", "id": ["A", "B"], "sequence": "ACDEF"}])
    chains, alignments = _job_chains(job)
    assert chains == [("ACDEF", "A", 0, 0), ("ACDEF", "B", 0, 1)]
    assert alignments == {}


def test_an_alignment_is_resolved_against_the_job_file(tmp_path) -> None:
    """A job names its a3m the way it names any path: relative to itself."""
    (tmp_path / "chain.a3m").write_text(">q\nACDEF\n")
    job = _job(
        tmp_path,
        [
            {
                "type": "protein",
                "id": ["A"],
                "sequence": "ACDEF",
                "unpaired_msa": "chain.a3m",
            }
        ],
    )
    _, alignments = _job_chains(job)
    assert alignments == {0: tmp_path / "chain.a3m"}


def test_a_native_dialect_is_not_invented(tmp_path) -> None:
    """Upstream takes a sequence, not a job file, so there is nothing to claim."""
    assert ESMFold2Backend().capabilities().input_formats == ("foldjax",)


def test_a_non_foldjax_document_says_so(tmp_path) -> None:
    path = tmp_path / "other.json"
    path.write_text(json.dumps({"sequences": ["ACDEF"]}))
    with pytest.raises(ValueError, match="FoldJAX job document"):
        _job_chains(path)


def _fake_session_modules(tmp_path, calls):
    model = SimpleNamespace(
        has_language_model=True,
        settings=SimpleNamespace(msa_max_depth=16, msa_n_layers=1, num_loops=3),
    )

    def load(path, *, esmc, language_model):
        calls["load"].append((path, esmc, language_model))
        return model

    def build_job_features(chains, alignments):
        del alignments
        tokens = sum(len(sequence) for sequence, *_rest in chains)
        row = np.arange(tokens, dtype=np.int64)[None]
        return {
            "input_ids": row + 4,
            "asym_id": np.zeros_like(row),
            "residue_index": row,
            "mol_type": np.zeros_like(row),
            "token_attention_mask": np.ones_like(row, dtype=bool),
        }

    def language_model_states(features, loaded, *, packed_length):
        del loaded
        state = np.asarray(features["input_ids"], dtype=np.float32)[..., None]
        calls["lm"].append((packed_length, state.copy()))
        return state

    def predict(key, features, loaded, **kwargs):
        del features, loaded
        calls["predict"].append((key, kwargs["precomputed_lm_states"]))
        return {}

    inference = SimpleNamespace(
        LANGUAGE_MODEL_FEATURES=(
            "input_ids",
            "asym_id",
            "residue_index",
            "mol_type",
            "token_attention_mask",
        ),
        load=load,
        seed_key=lambda seed: seed,
        build_job_features=build_job_features,
        language_model_length=lambda features: int(features["input_ids"].shape[-1]) + 2,
        language_model_states=language_model_states,
        predict=predict,
    )
    output = SimpleNamespace(
        write_prediction_outputs=lambda *args, **kwargs: {
            "structures": [tmp_path / "sample_0.cif"],
            "summary": [{"sample": 0, "plddt": 91.0}],
        }
    )
    return model, {
        "foldjax.models.esmfold2.inference": inference,
        "foldjax.models.esmfold2.output": output,
    }


def _fake_session_weights(tmp_path):
    weights = tmp_path / "weights"
    esmc = weights / "esmc"
    esmc.mkdir(parents=True)
    (weights / "model.safetensors").write_bytes(b"structure")
    (weights / "config.json").write_text("{}")
    (esmc / "config.json").write_text("{}")
    (esmc / "model.safetensors").write_bytes(b"language-model")
    return weights


def test_request_session_loads_once_and_runs_esmc_once_per_input(
    tmp_path, monkeypatch
) -> None:
    calls = {"load": [], "lm": [], "predict": []}
    _model, modules = _fake_session_modules(tmp_path, calls)
    monkeypatch.setattr(
        "foldjax.backends.esmfold2.import_module", lambda name: modules[name]
    )
    weights = _fake_session_weights(tmp_path)
    first_job = _job(
        tmp_path / "first", [{"type": "protein", "id": ["A"], "sequence": "ACD"}]
    )
    second_job = _job(
        tmp_path / "second",
        [{"type": "protein", "id": ["A"], "sequence": "ACDE"}],
    )
    first = PredictionRequest(
        model="esmfold2",
        input=first_job,
        weights=weights,
        output_dir=tmp_path / "out-first",
        num_seeds=3,
    )
    second = dataclasses.replace(
        first, input=second_job, output_dir=tmp_path / "out-second"
    )
    backend = ESMFold2Backend()

    with backend.session((first, second)):
        for request in (first, second):
            for seed in request.resolved_seeds:
                backend.predict(
                    dataclasses.replace(
                        request,
                        seed=seed,
                        seeds=None,
                        num_seeds=None,
                        output_dir=request.output_dir / f"seed_{seed}",
                    )
                )

    assert len(calls["load"]) == 1
    assert len(calls["lm"]) == 2
    assert len(calls["predict"]) == 6
    assert calls["predict"][0][1] is calls["predict"][1][1]
    assert calls["predict"][1][1] is calls["predict"][2][1]
    assert calls["predict"][2][1] is not calls["predict"][3][1]
    assert calls["predict"][3][1] is calls["predict"][4][1]
    assert calls["predict"][4][1] is calls["predict"][5][1]
    assert backend._loaded_model is None
    assert backend._lm_state is None


def test_session_is_lazy_when_every_run_is_resumed(tmp_path, monkeypatch) -> None:
    calls = {"load": [], "lm": [], "predict": []}
    _model, modules = _fake_session_modules(tmp_path, calls)
    monkeypatch.setattr(
        "foldjax.backends.esmfold2.import_module", lambda name: modules[name]
    )
    weights = _fake_session_weights(tmp_path)
    request = PredictionRequest(
        model="esmfold2",
        input=_job(tmp_path, [{"type": "protein", "id": ["A"], "sequence": "ACD"}]),
        weights=weights,
        output_dir=tmp_path / "out",
        num_seeds=2,
    )
    backend = ESMFold2Backend()

    with backend.session((request,)):
        pass

    assert calls == {"load": [], "lm": [], "predict": []}


def test_session_refuses_to_mix_replaced_weights(tmp_path, monkeypatch) -> None:
    calls = {"load": [], "lm": [], "predict": []}
    _model, modules = _fake_session_modules(tmp_path, calls)
    monkeypatch.setattr(
        "foldjax.backends.esmfold2.import_module", lambda name: modules[name]
    )
    weights = _fake_session_weights(tmp_path)
    request = PredictionRequest(
        model="esmfold2",
        input=_job(tmp_path, [{"type": "protein", "id": ["A"], "sequence": "ACD"}]),
        weights=weights,
        output_dir=tmp_path / "out",
        num_seeds=2,
    )
    backend = ESMFold2Backend()

    with backend.session((request,)):
        backend.predict(dataclasses.replace(request, num_seeds=None))
        replacement = weights / "config.new"
        replacement.write_text("{}")
        os.replace(replacement, weights / "config.json")
        with pytest.raises(RuntimeError, match="weights changed"):
            backend.predict(
                dataclasses.replace(
                    request, seed=1, num_seeds=None, output_dir=tmp_path / "out-1"
                )
            )

    assert len(calls["load"]) == 1


def test_unowned_model_never_reuses_another_models_lm_state() -> None:
    calls = []
    inference = SimpleNamespace(
        LANGUAGE_MODEL_FEATURES=(
            "input_ids",
            "asym_id",
            "residue_index",
            "mol_type",
            "token_attention_mask",
        ),
        language_model_states=lambda features, model, *, packed_length: (
            calls.append(model.tag) or np.asarray([model.tag], dtype=np.float32)
        ),
    )
    features = {
        "input_ids": np.asarray([[4, 5]]),
        "asym_id": np.asarray([[0, 0]]),
        "residue_index": np.asarray([[0, 1]]),
        "mol_type": np.asarray([[0, 0]]),
        "token_attention_mask": np.asarray([[1, 1]], dtype=bool),
    }
    backend = ESMFold2Backend()
    backend._session_active = True
    first = SimpleNamespace(has_language_model=True, tag=1)
    second = SimpleNamespace(has_language_model=True, tag=2)

    first_state = backend._language_model_states(
        inference, features, first, packed_length=None
    )
    second_state = backend._language_model_states(
        inference, features, second, packed_length=None
    )

    assert calls == [1, 2]
    np.testing.assert_array_equal(first_state, [1])
    np.testing.assert_array_equal(second_state, [2])


def test_hidden_cache_releases_the_previous_input_before_computing_next() -> None:
    backend = ESMFold2Backend()
    model = SimpleNamespace(has_language_model=True)
    backend._session_active = True
    backend._loaded_model = model
    backend._lm_state = np.ones((4, 4), dtype=np.float32)
    backend._lm_state_key = "previous-input"
    features = {
        "input_ids": np.asarray([[4, 5]]),
        "asym_id": np.asarray([[0, 0]]),
        "residue_index": np.asarray([[0, 1]]),
        "mol_type": np.asarray([[0, 0]]),
        "token_attention_mask": np.asarray([[1, 1]], dtype=bool),
    }

    def compute(features, loaded, *, packed_length):
        del features, loaded, packed_length
        assert backend._lm_state is None
        assert backend._lm_state_key is None
        return np.asarray([42.0], dtype=np.float32)

    inference = SimpleNamespace(
        LANGUAGE_MODEL_FEATURES=tuple(features),
        language_model_states=compute,
    )

    state = backend._language_model_states(
        inference, features, model, packed_length=None
    )

    np.testing.assert_array_equal(state, [42.0])
    assert backend._lm_state is state


def test_resumed_seed_anchors_weights_before_the_first_load(
    tmp_path, monkeypatch
) -> None:
    calls = {"load": [], "lm": [], "predict": []}
    _model, modules = _fake_session_modules(tmp_path, calls)
    monkeypatch.setattr(
        "foldjax.backends.esmfold2.import_module", lambda name: modules[name]
    )
    weights = _fake_session_weights(tmp_path)
    request = PredictionRequest(
        model="esmfold2",
        input=_job(
            tmp_path, [{"type": "protein", "id": ["A"], "sequence": "ACD"}]
        ),
        weights=weights,
        output_dir=tmp_path / "out",
        num_seeds=2,
    )
    backend = ESMFold2Backend()

    with backend.session((request,)):
        backend.observe_resumed(request)
        replacement = weights / "config.new"
        replacement.write_text("{}")
        os.replace(replacement, weights / "config.json")
        with pytest.raises(PredictionError, match="weights changed"):
            backend.predict(dataclasses.replace(request, num_seeds=None))
        with pytest.raises(PredictionError, match="weights changed"):
            backend.predict(
                dataclasses.replace(
                    request, seed=1, num_seeds=None, output_dir=tmp_path / "out-1"
                )
            )

    assert calls["load"] == []


def test_huggingface_style_symlinked_esmc_shards_are_verifiable(tmp_path) -> None:
    weights = tmp_path / "weights"
    esmc = weights / "esmc"
    blobs = weights / "blobs"
    esmc.mkdir(parents=True)
    blobs.mkdir()
    (weights / "model.safetensors").write_bytes(b"structure")
    (weights / "config.json").write_text("{}")
    (esmc / "config.json").write_text("{}")
    shard = blobs / "abcdef"
    shard.write_bytes(b"language-model")
    (esmc / "model-00001-of-00001.safetensors").symlink_to(shard)
    (esmc / "model.safetensors.index.json").write_text(
        json.dumps(
            {"weight_map": {"esmc.embed.weight": "model-00001-of-00001.safetensors"}}
        )
    )

    snapshot = _model_asset_snapshot(weights, esmc=None, language_model=True)

    assert snapshot is not None
    assert any(str(shard.resolve()) == path for path, _identity in snapshot)


def test_multi_seed_session_keeps_predict_job_only_wrappers_compatible(
    tmp_path, monkeypatch
) -> None:
    weights = _fake_session_weights(tmp_path)
    job = _job(
        tmp_path, [{"type": "protein", "id": ["A"], "sequence": "ACD"}]
    )
    calls = {"load": 0, "predict_job": 0}
    model = SimpleNamespace(has_language_model=True)

    def load(*args, **kwargs):
        calls["load"] += 1
        return model

    def predict_job(*args, **kwargs):
        calls["predict_job"] += 1
        return {}, {"asym_id": np.asarray([[0]])}

    modules = {
        "foldjax.models.esmfold2.inference": SimpleNamespace(
            load=load,
            seed_key=lambda seed: seed,
            predict_job=predict_job,
        ),
        "foldjax.models.esmfold2.output": SimpleNamespace(
            write_prediction_outputs=lambda *args, **kwargs: {
                "structures": [tmp_path / "sample_0.cif"],
                "summary": [{"sample": 0, "plddt": 91.0}],
            }
        ),
    }
    monkeypatch.setattr(
        "foldjax.backends.esmfold2.import_module", lambda name: modules[name]
    )
    request = PredictionRequest(
        model="esmfold2",
        input=job,
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

    assert calls == {"load": 1, "predict_job": 2}
