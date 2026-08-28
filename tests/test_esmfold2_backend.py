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
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest

from foldjax.backends.esmfold2 import (
    DEFAULTS,
    ESMFold2Backend,
    _job_chains,
    _model_asset_snapshot,
    _requires_all_atom_features,
    managed_asset_profile,
)
from foldjax.paths import weights_dir
from foldjax.schema import PredictionError, PredictionRequest


def _job(tmp_path, entities) -> str:
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "job.json"
    path.write_text(json.dumps({"name": "t", "entities": entities}))
    return path


def test_the_neutral_schedule_reaches_the_ports_own_names(tmp_path) -> None:
    """`num_recycles` is `num_recycles` here; the other three keep their names."""
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
        "num_recycles": 10,
        "max_msa_depth": 1024,
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


def test_scalar_backend_withholds_distogram_without_exposing_an_override(
    tmp_path, monkeypatch
) -> None:
    job = _job(tmp_path, [{"type": "protein", "id": ["A"], "sequence": "ACD"}])
    weights = tmp_path / "weights"
    weights.mkdir()
    seen: dict[str, object] = {}
    model = SimpleNamespace(has_language_model=False)

    def predict_job(*args, **kwargs):
        del args
        seen.update(kwargs)
        return {}, {"asym_id": np.asarray([[0]])}

    modules = {
        "foldjax.models.esmfold2.inference": SimpleNamespace(
            load=lambda *args, **kwargs: model,
            seed_key=lambda seed: seed,
            predict_job=predict_job,
        ),
        "foldjax.models.esmfold2.output": SimpleNamespace(
            write_prediction_outputs=lambda *args, **kwargs: {
                "structures": [tmp_path / "sample_0.cif"],
                "summary": [{"sample": 0, "plddt": 0.75}],
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
            options={"no_language_model": True},
        )
    )

    assert seen == {"return_distogram_logits": False}
    assert "return_distogram_logits" not in result.raw["overrides"]


def test_all_released_biomolecule_types_are_advertised() -> None:
    assert ESMFold2Backend().capabilities().entity_types == (
        "protein",
        "dna",
        "rna",
        "ligand",
    )


@pytest.mark.parametrize(
    "document",
    [
        {"entities": [{"type": "dna", "id": "D", "sequence": "AT"}]},
        {"entities": [{"type": "rna", "id": "R", "sequence": "AU"}]},
        {"entities": [{"type": "ligand", "id": "L", "ccd": "ATP"}]},
        {
            "entities": [
                {
                    "type": "protein",
                    "id": "P",
                    "sequence": "AS",
                    "modifications": [{"ccd": "SEP", "position": 2}],
                }
            ]
        },
        {
            "entities": [{"type": "protein", "id": "P", "sequence": "AS"}],
            "bonds": [[['P', 1, 'CA'], ['P', 2, 'CA']]],
        },
    ],
)
def test_noncanonical_chemistry_selects_all_atom_features(document) -> None:
    assert _requires_all_atom_features(document)


def test_plain_protein_keeps_the_historical_feature_path() -> None:
    assert not _requires_all_atom_features(
        {"entities": [{"type": "protein", "id": "P", "sequence": "AS"}]}
    )


def test_all_biomolecule_job_uses_common_feature_builder(tmp_path, monkeypatch) -> None:
    job = tmp_path / "mixed.json"
    document = {
        "name": "mixed",
        "entities": [
            {"type": "protein", "id": "P", "sequence": "AS"},
            {"type": "dna", "id": "D", "sequence": "AT"},
            {"type": "rna", "id": "R", "sequence": "AU"},
            {"type": "ligand", "id": "L", "ccd": "ATP"},
        ],
        "bonds": [[['P', 2, 'CA'], ['L', 1, 'PA']]],
    }
    job.write_text(json.dumps(document))
    weights = tmp_path / "weights"
    weights.mkdir()
    features = {
        "input_ids": np.asarray([[4]]),
        "asym_id": np.asarray([[0]]),
        "residue_index": np.asarray([[0]]),
        "mol_type": np.asarray([[0]]),
        "token_attention_mask": np.asarray([[True]]),
    }
    seen = {}
    memory_events = []
    model = SimpleNamespace(
        has_language_model=False,
        settings=SimpleNamespace(max_msa_depth=16, msa_n_layers=1, num_recycles=3),
    )

    def build_common(value, **kwargs):
        memory_events.append("build")
        seen["document"] = value
        seen.update(kwargs)
        return features

    def predict(key, value, loaded, **kwargs):
        seen.update(key=key, features=value, model=loaded, predict=kwargs)
        return {}

    modules = {
        "foldjax.models.esmfold2.inference": SimpleNamespace(
            load=lambda *args, **kwargs: model,
            seed_key=lambda seed: seed,
            build_common_job_features=build_common,
            build_job_features=lambda *_args: pytest.fail("legacy builder was used"),
            predict=predict,
        ),
        "foldjax.models.esmfold2.output": SimpleNamespace(
            write_prediction_outputs=lambda *args, **kwargs: {
                "structures": [tmp_path / "sample_0.cif"],
                "summary": [{"sample": 0, "plddt": 0.75}],
            }
        ),
    }
    monkeypatch.setattr(
        "foldjax.backends.esmfold2.import_module", lambda name: modules[name]
    )

    @contextmanager
    def recording_lease(key, release):
        del release
        memory_events.append(("enter", key))
        try:
            yield
        finally:
            memory_events.append(("exit", key))

    monkeypatch.setattr(
        "foldjax.backends.esmfold2.managed_memory_lease", recording_lease
    )

    ESMFold2Backend().predict(
        PredictionRequest(
            model="esmfold2",
            input=job,
            weights=weights,
            output_dir=tmp_path / "out",
            options={"no_language_model": True},
        )
    )

    assert seen["document"] == document
    assert seen["base_dir"] == tmp_path
    assert seen["ccd_path"] == weights / "ccd.pkl"
    assert "msa_depth" not in seen
    assert seen["features"] is features
    assert seen["predict"]["return_distogram_logits"] is False
    assert memory_events == [
        ("enter", "esmfold2_ccd"),
        "build",
        ("exit", "esmfold2_ccd"),
    ]


def _all_atom_session_fixture(tmp_path, monkeypatch, *, build_error=None):
    job = _job(tmp_path / "job", [{"type": "ligand", "id": "L", "ccd": "ATP"}])
    weights = tmp_path / "weights"
    weights.mkdir()
    (weights / "model.safetensors").write_bytes(b"structure")
    (weights / "config.json").write_text("{}")
    features = {
        "input_ids": np.asarray([[4]]),
        "asym_id": np.asarray([[0]]),
        "residue_index": np.asarray([[0]]),
        "mol_type": np.asarray([[3]]),
        "token_attention_mask": np.asarray([[True]]),
    }
    calls = {"build": 0, "predict": 0}
    model = SimpleNamespace(
        has_language_model=False,
        settings=SimpleNamespace(max_msa_depth=16, msa_n_layers=1, num_recycles=3),
    )

    def build_common(*args, **kwargs):
        del args, kwargs
        calls["build"] += 1
        if build_error is not None:
            raise build_error
        return features

    def predict(*args, **kwargs):
        del args, kwargs
        calls["predict"] += 1
        return {}

    modules = {
        "foldjax.models.esmfold2.inference": SimpleNamespace(
            load=lambda *args, **kwargs: model,
            seed_key=lambda seed: seed,
            build_common_job_features=build_common,
            predict=predict,
        ),
        "foldjax.models.esmfold2.output": SimpleNamespace(
            write_prediction_outputs=lambda *args, **kwargs: {
                "structures": [tmp_path / "sample_0.cif"],
                "summary": [{"sample": 0, "plddt": 0.75}],
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
        options={"no_language_model": True},
    )
    return request, calls


def test_all_atom_session_holds_one_ccd_lease_across_seeds(
    tmp_path, monkeypatch
) -> None:
    request, calls = _all_atom_session_fixture(tmp_path, monkeypatch)
    events = []

    @contextmanager
    def recording_lease(key, release):
        del release
        events.append(("enter", key))
        try:
            yield
        finally:
            events.append(("exit", key))

    monkeypatch.setattr(
        "foldjax.backends.esmfold2.managed_memory_lease", recording_lease
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
        assert events == [("enter", "esmfold2_ccd")]

    assert calls == {"build": 2, "predict": 2}
    assert events == [
        ("enter", "esmfold2_ccd"),
        ("exit", "esmfold2_ccd"),
    ]


@pytest.mark.parametrize("error", [ValueError("bad chemistry"), KeyboardInterrupt()])
def test_scalar_all_atom_failure_releases_ccd_lease(
    tmp_path, monkeypatch, error
) -> None:
    request, _calls = _all_atom_session_fixture(
        tmp_path, monkeypatch, build_error=error
    )
    events = []

    @contextmanager
    def recording_lease(key, release):
        del release
        events.append(("enter", key))
        try:
            yield
        finally:
            events.append(("exit", key))

    monkeypatch.setattr(
        "foldjax.backends.esmfold2.managed_memory_lease", recording_lease
    )

    with pytest.raises(type(error)):
        ESMFold2Backend().predict(dataclasses.replace(request, num_seeds=None))

    assert events == [
        ("enter", "esmfold2_ccd"),
        ("exit", "esmfold2_ccd"),
    ]


def test_session_keyboard_interrupt_releases_ccd_and_resets_state(
    tmp_path, monkeypatch
) -> None:
    request, _calls = _all_atom_session_fixture(
        tmp_path, monkeypatch, build_error=KeyboardInterrupt()
    )
    events = []

    @contextmanager
    def recording_lease(key, release):
        del release
        events.append(("enter", key))
        try:
            yield
        finally:
            events.append(("exit", key))

    monkeypatch.setattr(
        "foldjax.backends.esmfold2.managed_memory_lease", recording_lease
    )
    backend = ESMFold2Backend()

    with pytest.raises(KeyboardInterrupt), backend.session((request,)):
        backend.predict(dataclasses.replace(request, num_seeds=None))

    assert events == [
        ("enter", "esmfold2_ccd"),
        ("exit", "esmfold2_ccd"),
    ]
    assert backend._session_open is False
    assert backend._managed_memory is None
    assert backend._ccd_memory_leased is False


def test_poison_after_all_atom_seed_still_releases_session_ccd(
    tmp_path, monkeypatch
) -> None:
    request, _calls = _all_atom_session_fixture(tmp_path, monkeypatch)
    events = []

    @contextmanager
    def recording_lease(key, release):
        del release
        events.append(("enter", key))
        try:
            yield
        finally:
            events.append(("exit", key))

    monkeypatch.setattr(
        "foldjax.backends.esmfold2.managed_memory_lease", recording_lease
    )
    backend = ESMFold2Backend()

    with backend.session((request,)):
        backend.predict(dataclasses.replace(request, num_seeds=None))
        replacement = request.weights / "config.new"
        replacement.write_text("{}")
        os.replace(replacement, request.weights / "config.json")
        with pytest.raises(PredictionError, match="weights changed"):
            backend.predict(
                dataclasses.replace(
                    request,
                    seed=1,
                    num_seeds=None,
                    output_dir=tmp_path / "out-1",
                )
            )

    assert events == [
        ("enter", "esmfold2_ccd"),
        ("exit", "esmfold2_ccd"),
    ]
    assert backend._session_poisoned is None


def test_all_atom_session_without_execution_never_acquires_ccd(
    tmp_path, monkeypatch
) -> None:
    request, calls = _all_atom_session_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "foldjax.backends.esmfold2.managed_memory_lease",
        lambda *_args, **_kwargs: pytest.fail("resumed session acquired CCD"),
    )

    with ESMFold2Backend().session((request,)):
        pass

    assert calls == {"build": 0, "predict": 0}


def test_plain_protein_prediction_never_acquires_ccd(tmp_path, monkeypatch) -> None:
    job = _job(tmp_path, [{"type": "protein", "id": ["A"], "sequence": "ACD"}])
    weights = tmp_path / "weights"
    weights.mkdir()
    model = SimpleNamespace(has_language_model=False)
    modules = {
        "foldjax.models.esmfold2.inference": SimpleNamespace(
            load=lambda *args, **kwargs: model,
            seed_key=lambda seed: seed,
            predict_job=lambda *args, **kwargs: (
                {},
                {"asym_id": np.asarray([[0]])},
            ),
        ),
        "foldjax.models.esmfold2.output": SimpleNamespace(
            write_prediction_outputs=lambda *args, **kwargs: {
                "structures": [tmp_path / "sample_0.cif"],
                "summary": [{"sample": 0, "plddt": 0.75}],
            }
        ),
    }
    monkeypatch.setattr(
        "foldjax.backends.esmfold2.import_module", lambda name: modules[name]
    )
    monkeypatch.setattr(
        "foldjax.backends.esmfold2.managed_memory_lease",
        lambda *_args, **_kwargs: pytest.fail("protein-only path acquired CCD"),
    )

    ESMFold2Backend().predict(
        PredictionRequest(
            model="esmfold2",
            input=job,
            weights=weights,
            output_dir=tmp_path / "out",
            options={"no_language_model": True},
        )
    )


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
        settings=SimpleNamespace(max_msa_depth=16, msa_n_layers=1, num_recycles=3),
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

    def language_model_embedding(features, loaded, *, packed_length):
        state = language_model_states(
            features, loaded, packed_length=packed_length
        )
        embedding = state + np.float32(100.0)
        calls.setdefault("embedding", []).append(embedding.copy())
        return embedding

    def predict(key, features, loaded, **kwargs):
        del features, loaded
        lm_input = kwargs.get(
            "precomputed_lm_embedding", kwargs.get("precomputed_lm_states")
        )
        calls["predict"].append(
            (key, lm_input, dict(kwargs))
        )
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
        load=load,
        seed_key=lambda seed: seed,
        build_job_features=build_job_features,
        language_model_length=lambda features: int(features["input_ids"].shape[-1]) + 2,
        language_model_states=language_model_states,
        language_model_embedding=language_model_embedding,
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
    assert len(calls["embedding"]) == 2
    assert len(calls["predict"]) == 6
    assert all(
        call[2]["return_distogram_logits"] is False for call in calls["predict"]
    )
    assert all(
        "precomputed_lm_embedding" in call[2]
        and "precomputed_lm_states" not in call[2]
        for call in calls["predict"]
    )
    assert calls["predict"][0][1] is calls["predict"][1][1]
    assert calls["predict"][1][1] is calls["predict"][2][1]
    assert calls["predict"][2][1] is not calls["predict"][3][1]
    assert calls["predict"][3][1] is calls["predict"][4][1]
    assert calls["predict"][4][1] is calls["predict"][5][1]
    assert backend._loaded_model is None
    assert backend._lm_embedding is None


def test_legacy_split_wrapper_recomputes_raw_states_without_retaining_them(
    tmp_path, monkeypatch
) -> None:
    calls = {"load": [], "lm": [], "predict": []}
    _model, modules = _fake_session_modules(tmp_path, calls)
    inference = modules["foldjax.models.esmfold2.inference"]
    del inference.language_model_embedding
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
        for seed in request.resolved_seeds:
            backend.predict(
                dataclasses.replace(
                    request,
                    seed=seed,
                    num_seeds=None,
                    output_dir=tmp_path / f"out-{seed}",
                )
            )

    assert len(calls["lm"]) == 2
    assert all(
        "precomputed_lm_states" in call[2]
        and "precomputed_lm_embedding" not in call[2]
        for call in calls["predict"]
    )
    assert backend._lm_embedding is None


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


def test_embedding_cache_releases_the_previous_input_before_computing_next() -> None:
    backend = ESMFold2Backend()
    model = SimpleNamespace(has_language_model=True)
    backend._session_active = True
    backend._loaded_model = model
    backend._lm_embedding = np.ones((4, 4), dtype=np.float32)
    backend._lm_embedding_key = "previous-input"
    features = {
        "input_ids": np.asarray([[4, 5]]),
        "asym_id": np.asarray([[0, 0]]),
        "residue_index": np.asarray([[0, 1]]),
        "mol_type": np.asarray([[0, 0]]),
        "token_attention_mask": np.asarray([[1, 1]], dtype=bool),
    }

    def compute(features, loaded, *, packed_length):
        del features, loaded, packed_length
        assert backend._lm_embedding is None
        assert backend._lm_embedding_key is None
        return np.asarray([42.0], dtype=np.float32)

    inference = SimpleNamespace(
        LANGUAGE_MODEL_FEATURES=tuple(features),
        language_model_embedding=compute,
    )

    state = backend._language_model_embedding(
        inference, features, model, packed_length=None
    )

    np.testing.assert_array_equal(state, [42.0])
    assert backend._lm_embedding is state


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


def test_available_ccd_is_part_of_the_model_asset_snapshot(tmp_path) -> None:
    weights = tmp_path / "weights"
    weights.mkdir()
    (weights / "model.safetensors").write_bytes(b"structure")
    (weights / "config.json").write_text("{}")
    ccd = weights / "ccd.pkl"
    ccd.write_bytes(b"chemistry")

    before = _model_asset_snapshot(weights, esmc=None, language_model=False)
    assert before is not None
    assert any(str(ccd.resolve()) == path for path, _identity in before)

    replacement = weights / "ccd.new"
    replacement.write_bytes(b"different chemistry")
    os.replace(replacement, ccd)
    after = _model_asset_snapshot(weights, esmc=None, language_model=False)

    assert after is not None
    assert after != before


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


def test_the_defaults_are_the_released_config_not_what_fits_on_a_card() -> None:
    """`DEFAULTS` must stay upstream's released values, especially the hard one.

    ESMFold2's release ships `num_diffusion_samples = 32`. The confidence head
    builds a `bf16[num_samples, N, N, 2048]` arena, linear in that number, so a
    card that cannot hold it invites one obvious and wrong fix: lower the
    default. That would be a silent divergence -- the port would keep the name
    and stop being the configuration upstream released, and every benchmark
    taken afterwards would be at a sample count nobody chose. The lever for the
    memory is `confidence_sample_sequential`; it is not the sample count.

    The first three assertions hold in any clone. The rest run only where the
    weight store is present, and they are the ones that matter: they compare
    against the release artifact instead of against a restatement of it, which
    is the difference between pinning a value and pinning the *source* of a
    value. `max_msa_depth` is deliberately not among them -- it appears nowhere
    in the released config, so it is the port's own choice and this test would
    be asserting our opinion back to ourselves.
    """
    assert DEFAULTS["num_diffusion_samples"] == 32
    assert DEFAULTS["num_recycles"] == 3
    assert DEFAULTS["num_sampling_steps"] == 14

    config = weights_dir("esmfold2") / "config.json"
    if not config.is_file():
        return

    document = json.loads(config.read_text(encoding="utf-8"))
    assert document["type"] == "release", document.get("type")
    assert document["num_diffusion_samples"] == DEFAULTS["num_diffusion_samples"]
    # The release spells this `num_loops`; `num_recycles` is only this port's
    # internal name for it. Unifying internal names never renamed a checkpoint
    # key, and asserting the internal spelling against the release artifact is
    # how this test read a KeyError as a missing default for three weeks --
    # invisible in CI, which has no weight store and returns above.
    assert document["num_loops"] == DEFAULTS["num_recycles"]
    assert (
        document["structure_head"]["inference_num_steps"]
        == DEFAULTS["num_sampling_steps"]
    )
