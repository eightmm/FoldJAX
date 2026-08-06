"""Building a job in Python produces the same document a job file holds."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from foldjax import Bond, Dna, Job, Ligand, Modification, Protein, Rna, capabilities
from foldjax.input import materialize_native_input

SEQUENCE = "ACDEFGHIKLMN"


def _job(base: Path) -> Job:
    (base / "a.a3m").write_text(f">query\n{SEQUENCE}\n", encoding="utf-8")
    return Job(
        "demo",
        [
            Protein("A", SEQUENCE, unpaired_msa="a.a3m"),
            Dna("B", "ACGT"),
            Rna("C", "ACGU"),
            Ligand("L", ccd="ATP"),
        ],
    )


@pytest.mark.parametrize(
    "model", ["alphafold3", "boltz2", "opendde", "openfold3", "protenix"]
)
def test_every_backend_materializes_a_job_built_in_python(
    tmp_path: Path, model: str
) -> None:
    """The class is only useful if it feeds the path that already exists."""
    source = _job(tmp_path).write(tmp_path / "job.json")
    native = materialize_native_input(
        source, capabilities(model), tmp_path / model, seed=0
    )
    assert native.is_file() and native.stat().st_size > 0


def test_a_job_survives_the_round_trip_through_a_file(tmp_path: Path) -> None:
    job = _job(tmp_path)
    assert Job.read(job.write(tmp_path / "job.json")) == job


def test_reading_accepts_the_yaml_people_already_write(tmp_path: Path) -> None:
    path = tmp_path / "job.yaml"
    path.write_text(
        "name: demo\nentities:\n  - type: protein\n    id: A\n"
        f"    sequence: {SEQUENCE}\n",
        encoding="utf-8",
    )
    job = Job.read(path)
    assert job == Job("demo", (Protein("A", SEQUENCE),))


def test_optional_fields_are_absent_rather_than_null(tmp_path: Path) -> None:
    """A written job reads like one a person would write, not like a form.

    It also has to: the document is checked against a fixed key set per entity
    type, so emitting every field always would put `paired_msa: null` in front
    of a backend that rejects that field's presence.
    """
    document = Job("demo", (Protein("A", SEQUENCE),)).to_document()
    assert document["entities"][0] == {
        "type": "protein",
        "id": "A",
        "sequence": SEQUENCE,
    }
    assert "bonds" not in document


def test_the_rules_stay_in_one_place(tmp_path: Path) -> None:
    """A ligand naming both representations is refused, but not by this class.

    Whether a ligand may carry a CCD code, a SMILES string, or either is a
    question about the backend, and `foldjax.input` answers it once with the
    model in hand. Re-checking it at construction would be a second rule set to
    keep in step, and would reject a document the CLI accepts.
    """
    job = Job("demo", (Protein("A", SEQUENCE), Ligand("L", ccd="ATP", smiles="CCO")))
    source = job.write(tmp_path / "job.json")
    with pytest.raises(ValueError, match="either ccd or smiles"):
        materialize_native_input(
            source, capabilities("protenix"), tmp_path / "out", seed=0
        )


def test_an_unknown_field_is_refused_on_read(tmp_path: Path) -> None:
    """Silently dropping a field someone wrote changes the job they asked for."""
    path = tmp_path / "job.json"
    path.write_text(
        json.dumps(
            {
                "name": "demo",
                "entities": [
                    {"type": "protein", "id": "A", "sequence": SEQUENCE, "msa": "x"}
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unsupported protein entity fields"):
        Job.read(path)


def test_msa_paths_stay_relative_and_resolve_against_the_document(
    tmp_path: Path,
) -> None:
    """A job and its alignments move together; the written file keeps it that way."""
    workdir = tmp_path / "work"
    workdir.mkdir()
    (workdir / "a.a3m").write_text(f">query\n{SEQUENCE}\n", encoding="utf-8")
    job = Job("demo", (Protein("A", SEQUENCE, unpaired_msa="a.a3m"),))
    source = job.write(workdir / "job.json")

    assert json.loads(source.read_text())["entities"][0]["unpaired_msa"] == "a.a3m"
    native = materialize_native_input(
        source, capabilities("protenix"), tmp_path / "out", seed=0
    )
    assert str(workdir / "a.a3m") in native.read_text()


def test_bonds_and_modifications_reach_the_native_dialect(tmp_path: Path) -> None:
    (tmp_path / "a.a3m").write_text(f">query\n{SEQUENCE}\n", encoding="utf-8")
    job = Job(
        "demo",
        (
            Protein("A", SEQUENCE, modifications=(Modification("MSE", 3),)),
            Ligand("L", ccd="ATP"),
        ),
        bonds=(Bond(("A", 3, "SG"), ("L", 1, "C1")),),
    )
    source = job.write(tmp_path / "job.json")
    native = materialize_native_input(
        source, capabilities("protenix"), tmp_path / "out", seed=0
    ).read_text()
    assert "MSE" in native and "SG" in native
