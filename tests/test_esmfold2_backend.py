"""The ESMFold2 adapter's contract, without loading 25 GB of weights.

Everything here is the part FoldJAX owns: how a neutral request becomes
upstream's argument names, what the adapter refuses rather than folds wrongly,
and how a job document turns into chains. The model itself is upstream's and
is exercised by the benchmark, not by the suite.
"""

from __future__ import annotations

import json

import pytest

from foldjax.backends.esmfold2 import ESMFold2Backend, _job_chains
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
