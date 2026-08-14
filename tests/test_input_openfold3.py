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


def test_an_unrecognized_alignment_stem_is_linked_not_refused(
    tmp_path: Path, job: dict
) -> None:
    """Upstream selects a source by stem and skips a name it does not know.

    This used to raise, which is honest -- nothing was silently dropped -- but
    it left the backend unable to take the one file users actually have, an
    `.a3m` named after the target. Every other backend here accepts that name.
    Linking it under a stem upstream reads keeps the guarantee and lets the run
    happen; the stem also picks the row cap, so the choice is not cosmetic.
    """
    (tmp_path / "msa.a3m").write_text(">q\nACD\n")
    job["entities"][0]["unpaired_msa"] = "msa.a3m"

    document = _materialize(_write(tmp_path / "job.json", job), tmp_path / "out")

    (main,) = document["queries"]["complex"]["chains"][0]["main_msa_file_paths"]
    assert Path(main).stem == "colabfold_main"
    assert Path(main).read_text() == ">q\nACD\n"


@pytest.mark.parametrize(
    ("contents", "suffix"),
    [(">q\nACD\n", ".a3m"), ("# STOCKHOLM 1.0\nq ACD\n//\n", ".sto")],
)
def test_extensionless_alignment_is_published_with_its_detected_format(
    tmp_path: Path, job: dict, contents: str, suffix: str
) -> None:
    (tmp_path / "alignment").write_text(contents)
    job["entities"][0]["unpaired_msa"] = "alignment"

    document = _materialize(_write(tmp_path / "job.json", job), tmp_path / "out")

    (main,) = document["queries"]["complex"]["chains"][0][
        "main_msa_file_paths"
    ]
    assert Path(main).name == f"colabfold_main{suffix}"
    assert Path(main).read_text() == contents


def test_unknown_extensionless_alignment_fails_at_materialization(
    tmp_path: Path, job: dict
) -> None:
    (tmp_path / "alignment").write_text("not an alignment\n")
    job["entities"][0]["unpaired_msa"] = "alignment"

    with pytest.raises(ValueError, match="cannot infer MSA format"):
        _materialize(_write(tmp_path / "job.json", job), tmp_path / "out")


def test_an_unpaired_alignment_never_lands_on_the_paired_stem(
    tmp_path: Path, job: dict
) -> None:
    """`colabfold_paired` caps at 8,192 rows where `colabfold_main` caps at 16,384.

    Written as `"paired" in field`, which is true of `"unpaired_msa"`, this
    halved the depth of every renamed unpaired alignment and failed nothing.
    """
    (tmp_path / "msa.a3m").write_text(">q\nACD\n")
    (tmp_path / "pair.a3m").write_text(">p\nACD\n")
    job["entities"][0]["unpaired_msa"] = "msa.a3m"
    job["entities"][0]["paired_msa"] = "pair.a3m"

    chain = _materialize(_write(tmp_path / "job.json", job), tmp_path / "out")[
        "queries"
    ]["complex"]["chains"][0]

    (main,) = chain["main_msa_file_paths"]
    (paired,) = chain["paired_msa_file_paths"]
    assert Path(main).stem == "colabfold_main"
    assert Path(paired).stem == "colabfold_paired"
    assert Path(main).read_text() != Path(paired).read_text()


def test_two_chains_do_not_overwrite_each_others_alignment(tmp_path: Path) -> None:
    """Both want the name `colabfold_main`; only a per-chain directory saves them."""
    (tmp_path / "one.a3m").write_text(">a\nACD\n")
    (tmp_path / "two.a3m").write_text(">b\nWYF\n")
    job = {
        "name": "complex",
        "entities": [
            {
                "type": "protein",
                "id": ["A"],
                "sequence": "ACD",
                "unpaired_msa": "one.a3m",
            },
            {
                "type": "protein",
                "id": ["B"],
                "sequence": "WYF",
                "unpaired_msa": "two.a3m",
            },
        ],
    }

    chains = _materialize(_write(tmp_path / "job.json", job), tmp_path / "out")[
        "queries"
    ]["complex"]["chains"]

    (left,) = chains[0]["main_msa_file_paths"]
    (right,) = chains[1]["main_msa_file_paths"]
    assert left != right
    assert Path(left).read_text() == ">a\nACD\n"
    assert Path(right).read_text() == ">b\nWYF\n"


def test_chain_ids_never_become_alignment_path_components(
    tmp_path: Path, job: dict
) -> None:
    """Chain IDs are document data and may contain path traversal syntax."""
    (tmp_path / "msa.a3m").write_text(">q\nACD\n")
    job["entities"][0]["id"] = ["../../outside"]
    job["entities"][0]["unpaired_msa"] = "msa.a3m"
    output = tmp_path / "generated"

    document = _materialize(_write(tmp_path / "job.json", job), output)

    (main,) = document["queries"]["complex"]["chains"][0][
        "main_msa_file_paths"
    ]
    path = Path(main)
    assert path.is_relative_to(output)
    assert path.parent.name == "entity_0000"
    assert "outside" not in path.parts


def test_alignment_directory_symlink_is_refused_before_writing(
    tmp_path: Path, job: dict
) -> None:
    (tmp_path / "msa.a3m").write_text(">q\nACD\n")
    job["entities"][0]["unpaired_msa"] = "msa.a3m"
    output = tmp_path / "generated"
    output.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (output / "msa").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="MSA directory is a symlink"):
        _materialize(_write(tmp_path / "job.json", job), output)

    assert not list(outside.iterdir())


def test_existing_alignment_symlink_is_replaced_without_touching_target(
    tmp_path: Path, job: dict
) -> None:
    (tmp_path / "msa.a3m").write_text(">q\nACD\n")
    job["entities"][0]["unpaired_msa"] = "msa.a3m"
    output = tmp_path / "generated"
    directory = output / "msa" / "entity_0000"
    directory.mkdir(parents=True)
    external = tmp_path / "user-owned.a3m"
    external.write_text("do not replace\n")
    generated = directory / "colabfold_main.a3m"
    generated.symlink_to(external)

    document = _materialize(_write(tmp_path / "job.json", job), output)

    (main,) = document["queries"]["complex"]["chains"][0][
        "main_msa_file_paths"
    ]
    assert Path(main).read_text() == ">q\nACD\n"
    assert external.read_text() == "do not replace\n"


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


def test_bonds_are_rejected_until_the_native_pipeline_applies_them(
    tmp_path: Path, job: dict
) -> None:
    job["bonds"] = [[["A", 1, "SG"], ["L", 1, "C1"]]]
    with pytest.raises(ValueError, match="openfold3 cannot express bonds"):
        _materialize(_write(tmp_path / "job.json", job), tmp_path / "out")


def test_no_bonds_key_when_there_are_none(tmp_path: Path, job: dict) -> None:
    document = _materialize(_write(tmp_path / "job.json", job), tmp_path / "out")
    assert "bonds" not in document["queries"]["complex"]


def test_the_model_is_registered() -> None:
    from foldjax.registry import available_models, normalize_model_name

    assert "openfold3" in available_models()
    for alias in ("of3", "openfold-3", "openfold3-jax", "OpenFold3"):
        assert normalize_model_name(alias) == "openfold3"
