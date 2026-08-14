"""The ESMFold2 adapter's contract, without loading 25 GB of weights.

Everything here is the part FoldJAX owns: how a neutral request becomes
upstream's argument names, what the adapter refuses rather than folds wrongly,
and how a job document turns into chains. The model itself is upstream's and
is exercised by the benchmark, not by the suite.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from foldjax.backends.esmfold2 import (
    ESMFold2Backend,
    _job_chains,
    managed_asset_profile,
)
from foldjax.schema import PredictionRequest


def _job(tmp_path, entities) -> str:
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
