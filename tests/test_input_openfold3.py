"""Materializing OpenFold3's query document from a FoldJAX job.

OpenFold3 can express everything the shared format carries, but with two
constraints the other backends do not have, and both fail silently upstream:
alignment files are selected by *stem* and any other name is skipped, and a paired
MSA that cannot actually be paired collapses the MSA to the query sequence alone.
The first is checkable here and is therefore refused rather than passed through.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from foldjax.input import materialize_native_input
from foldjax.registry import capabilities


def _materialize(source: Path, output_dir: Path, *, seed: int = 9) -> dict:
    path = materialize_native_input(
        source, capabilities("openfold3"), output_dir, seed=seed
    )
    assert path.suffix == ".json"
    return json.loads(path.read_text())


def _write(path: Path, job: dict) -> Path:
    path.write_text(json.dumps(job))
    return path


@pytest.fixture
def job(tmp_path: Path) -> dict:
    (tmp_path / "colabfold_main.a3m").write_text(">q\nACD\n")
    return {
        "name": "complex",
        "entities": [
            {
                "type": "protein",
                "id": ["A", "C"],
                "sequence": "ACD",
                "unpaired_msa": "colabfold_main.a3m",
                "modifications": [{"ccd": "SEP", "position": 2}],
            },
            {"type": "dna", "id": ["B"], "sequence": "ACGT"},
            {"type": "ligand", "id": ["L"], "ccd": "ATP"},
        ],
    }


def test_every_entity_becomes_a_chain(tmp_path: Path, job: dict) -> None:
    document = _materialize(_write(tmp_path / "job.json", job), tmp_path / "out")
    chains = document["queries"]["complex"]["chains"]
    assert [chain["molecule_type"] for chain in chains] == ["protein", "dna", "ligand"]
    # Repeated chains stay one entity with several ids, which is how OpenFold3
    # expresses a homomer.
    assert chains[0]["chain_ids"] == ["A", "C"]
    assert chains[1]["sequence"] == "ACGT"
    assert chains[2]["ccd_codes"] == ["ATP"]
    assert "sequence" not in chains[2]


def test_alignment_paths_are_absolute(tmp_path: Path, job: dict) -> None:
    """The document is written to a different directory than the job, so a relative
    path would resolve against the wrong base."""
    document = _materialize(_write(tmp_path / "job.json", job), tmp_path / "out")
    paths = document["queries"]["complex"]["chains"][0]["main_msa_file_paths"]
    assert len(paths) == 1
    assert Path(paths[0]).is_absolute()
    assert Path(paths[0]).name == "colabfold_main.a3m"


def test_an_unrecognized_alignment_stem_is_refused(tmp_path: Path, job: dict) -> None:
    """Upstream skips it and then dies inside the MSA parser with an IndexError."""
    (tmp_path / "msa.a3m").write_text(">q\nACD\n")
    job["entities"][0]["unpaired_msa"] = "msa.a3m"
    with pytest.raises(ValueError, match="would be ignored"):
        _materialize(_write(tmp_path / "job.json", job), tmp_path / "out")


def test_a_recognized_stem_in_any_directory_is_accepted(
    tmp_path: Path, job: dict
) -> None:
    nested = tmp_path / "aln"
    nested.mkdir()
    (nested / "uniref90_hits.a3m").write_text(">q\nACD\n")
    job["entities"][0]["unpaired_msa"] = "aln/uniref90_hits.a3m"
    document = _materialize(_write(tmp_path / "job.json", job), tmp_path / "out")
    paths = document["queries"]["complex"]["chains"][0]["main_msa_file_paths"]
    assert Path(paths[0]).name == "uniref90_hits.a3m"


def test_paired_alignments_are_carried(tmp_path: Path, job: dict) -> None:
    (tmp_path / "colabfold_paired.a3m").write_text(">q\nACD\n")
    job["entities"][0]["paired_msa"] = "colabfold_paired.a3m"
    document = _materialize(_write(tmp_path / "job.json", job), tmp_path / "out")
    chain = document["queries"]["complex"]["chains"][0]
    assert Path(chain["paired_msa_file_paths"][0]).name == "colabfold_paired.a3m"


def test_modifications_become_non_canonical_residues(tmp_path: Path, job: dict) -> None:
    """OpenFold3 names these by residue position, not by PTM type."""
    document = _materialize(_write(tmp_path / "job.json", job), tmp_path / "out")
    chain = document["queries"]["complex"]["chains"][0]
    assert chain["non_canonical_residues"] == {"2": "SEP"}


def test_smiles_ligands_are_carried(tmp_path: Path, job: dict) -> None:
    job["entities"][2] = {"type": "ligand", "id": ["L"], "smiles": "CCO"}
    document = _materialize(_write(tmp_path / "job.json", job), tmp_path / "out")
    ligand = document["queries"]["complex"]["chains"][2]
    assert ligand["smiles"] == "CCO"
    assert "ccd_codes" not in ligand


def test_bonds_are_carried_as_endpoint_pairs(tmp_path: Path, job: dict) -> None:
    job["bonds"] = [[["A", 1, "SG"], ["L", 1, "C1"]]]
    document = _materialize(_write(tmp_path / "job.json", job), tmp_path / "out")
    assert document["queries"]["complex"]["bonds"] == [[["A", 1, "SG"], ["L", 1, "C1"]]]


def test_no_bonds_key_when_there_are_none(tmp_path: Path, job: dict) -> None:
    document = _materialize(_write(tmp_path / "job.json", job), tmp_path / "out")
    assert "bonds" not in document["queries"]["complex"]


def test_the_model_is_registered() -> None:
    from foldjax.registry import available_models, normalize_model_name

    assert "openfold3" in available_models()
    for alias in ("of3", "openfold-3", "openfold3-jax", "OpenFold3"):
        assert normalize_model_name(alias) == "openfold3"
