"""One layout and one CIF header, whichever backend produced the run."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from foldjax.output import best_sample, normalize, safe_job_name
from foldjax.schema import PredictionOutputError, PredictionResult, PredictionSample

# Minimal but real: gemmi has to parse it, and the atom loop has to survive the
# header rewrite untouched.
CIF = """data_whatever_the_backend_called_it
_entry.id whatever
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_seq_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.B_iso_or_equiv
ATOM 1 N N MET A 1 1.000 2.000 3.000 90.00
ATOM 2 C CA MET A 1 2.000 3.000 4.000 91.00
"""


def _result(tmp_path: Path, model: str = "protenix", **kwargs) -> PredictionResult:
    native = tmp_path / "job_sample_0.cif"
    native.write_text(CIF, encoding="utf-8")
    sample = PredictionSample(
        seed=3,
        structure_path=native,
        scores={"ranking_score": 0.5, "plddt": 80.0},
        metadata={"sample": 0},
        **kwargs,
    )
    return PredictionResult(model=model, samples=(sample,), output_dir=tmp_path, raw={})


def test_a_structure_lands_where_its_name_says_it_is(tmp_path: Path) -> None:
    result = normalize(_result(tmp_path), job="1abc")
    path = result.samples[0].structure_path

    assert path == tmp_path / "seed-3_sample-00" / "1abc_seed-3_sample-00.cif"
    assert path.is_file()
    assert not (tmp_path / "job_sample_0.cif").exists(), "moved, not copied"


def test_a_job_name_cannot_move_a_structure_outside_the_run(tmp_path: Path) -> None:
    result = normalize(_result(tmp_path), job="../../escaped\\target")
    path = result.samples[0].structure_path

    assert path.resolve().is_relative_to(tmp_path.resolve())
    assert path.parent == tmp_path / "seed-3_sample-00"
    assert "/" not in path.name and "\\" not in path.name
    assert not (tmp_path.parent / "escaped_seed-3_sample-00.cif").exists()


def test_an_external_backend_path_is_copied_without_deleting_the_original(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    source = tmp_path / "user-owned.cif"
    original = CIF.encode()
    source.write_bytes(original)
    result = PredictionResult(
        model="protenix",
        samples=(PredictionSample(seed=0, structure_path=source),),
        output_dir=run,
    )

    placed = normalize(result, job="safe", root=run).samples[0].structure_path

    assert source.read_bytes() == original
    assert placed.is_file()
    assert placed != source


def test_a_canonical_target_symlink_cannot_rewrite_an_external_file(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    directory = run / "seed-0_sample-00"
    directory.mkdir(parents=True)
    external = tmp_path / "external.cif"
    original = CIF.encode()
    external.write_bytes(original)
    target = directory / "safe_seed-0_sample-00.cif"
    target.symlink_to(external)
    result = PredictionResult(
        model="protenix",
        samples=(PredictionSample(seed=0, structure_path=target),),
        output_dir=run,
    )

    placed = normalize(result, job="safe", root=run).samples[0].structure_path

    assert external.read_bytes() == original
    assert placed.is_file() and not placed.is_symlink()


def test_a_target_symlink_to_an_external_directory_is_replaced(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    source = run / "native.cif"
    source.parent.mkdir()
    source.write_text(CIF)
    directory = run / "seed-0_sample-00"
    directory.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    target = directory / "safe_seed-0_sample-00.cif"
    target.symlink_to(outside, target_is_directory=True)
    result = PredictionResult(
        model="protenix",
        samples=(PredictionSample(seed=0, structure_path=source),),
        output_dir=run,
    )

    placed = normalize(result, job="safe", root=run).samples[0].structure_path

    assert placed.is_file() and not placed.is_symlink()
    assert not (outside / source.name).exists()


def test_a_canonical_directory_symlink_cannot_escape_the_run(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (run / "seed-0_sample-00").symlink_to(outside, target_is_directory=True)
    source = run / "native.cif"
    source.write_text(CIF)
    result = PredictionResult(
        model="protenix",
        samples=(PredictionSample(seed=0, structure_path=source),),
        output_dir=run,
    )

    with pytest.raises(PredictionOutputError, match="is a symlink"):
        normalize(result, job="safe", root=run)


def test_a_canonical_directory_symlink_inside_the_run_is_rejected(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    destination = run / "other"
    destination.mkdir(parents=True)
    (run / "seed-0_sample-00").symlink_to(destination, target_is_directory=True)
    source = run / "native.cif"
    source.write_text(CIF)
    result = PredictionResult(
        model="protenix",
        samples=(PredictionSample(seed=0, structure_path=source),),
        output_dir=run,
    )

    with pytest.raises(PredictionOutputError, match="is a symlink"):
        normalize(result, job="safe", root=run)


def test_an_internal_hard_link_is_detached_before_cif_normalization(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    run.mkdir()
    external = tmp_path / "user-owned.cif"
    original = CIF.encode()
    external.write_bytes(original)
    source = run / "native.cif"
    source.hardlink_to(external)
    result = PredictionResult(
        model="protenix",
        samples=(PredictionSample(seed=0, structure_path=source),),
        output_dir=run,
    )

    placed = normalize(result, job="safe", root=run).samples[0].structure_path

    assert external.read_bytes() == original
    assert placed.stat().st_ino != external.stat().st_ino


def test_safe_job_name_keeps_readable_unicode() -> None:
    assert safe_job_name("  단백질 복합체 α/β  ") == "단백질_복합체_α_β"
    assert safe_job_name("../..") == "prediction"


def test_safe_job_name_limits_utf8_bytes_not_only_characters() -> None:
    safe = safe_job_name("단" * 80)
    assert len(safe.encode("utf-8")) <= 120
    assert safe.endswith(tuple("0123456789abcdef"))


def test_a_pdb_keeps_its_true_extension(tmp_path: Path) -> None:
    source = tmp_path / "native.pdb"
    source.write_text("ATOM      1  CA  GLY A   1       0.0  0.0  0.0\n")
    result = PredictionResult(
        model="boltz2",
        samples=(PredictionSample(seed=3, structure_path=source),),
        output_dir=tmp_path,
    )

    placed = normalize(result, job="target").samples[0].structure_path
    assert placed.suffix == ".pdb"
    assert placed.read_text().startswith("ATOM")


def test_a_backend_reported_job_name_is_preserved_per_sample(tmp_path: Path) -> None:
    samples = []
    for index, job in enumerate(("first", "second")):
        source = tmp_path / f"native-{index}.cif"
        source.write_text(CIF)
        samples.append(
            PredictionSample(
                seed=3,
                structure_path=source,
                metadata={"job": job, "sample": index},
            )
        )
    result = normalize(
        PredictionResult(
            model="alphafold3", samples=tuple(samples), output_dir=tmp_path
        ),
        job="container",
    )

    assert [sample.structure_path.name for sample in result.samples] == [
        "first_seed-3_sample-00.cif",
        "second_seed-3_sample-01.cif",
    ]


def test_the_sample_index_is_padded_so_ten_sorts_after_two(tmp_path: Path) -> None:
    """`sample-10` sorting before `sample-2` is the one flaw in the layout this
    is modelled on, and it is a directory listing away from anyone."""
    samples = []
    for index in (2, 10):
        native = tmp_path / f"s{index}.cif"
        native.write_text(CIF, encoding="utf-8")
        samples.append(
            PredictionSample(
                seed=0,
                structure_path=native,
                scores={},
                metadata={"sample": index},
            )
        )
    result = normalize(
        PredictionResult(
            model="protenix", samples=tuple(samples), output_dir=tmp_path, raw={}
        ),
        job="j",
    )
    names = sorted(p.parent.name for p in (s.structure_path for s in result.samples))
    assert names == ["seed-0_sample-02", "seed-0_sample-10"]


def test_the_header_says_what_the_file_is_without_touching_the_atoms(
    tmp_path: Path,
) -> None:
    from gemmi import cif

    result = normalize(_result(tmp_path, model="boltz2"), job="1abc")
    document = cif.read(str(result.samples[0].structure_path))
    block = document.sole_block()

    assert block.name == "1abc_seed-3_sample-00"
    assert block.find_pair("_entry.id")[1].strip("'\"") == "1abc_seed-3_sample-00"
    title = block.find_pair("_struct.title")[1]
    assert "1abc" in title and "boltz2" in title and "seed 3" in title
    # The claim the docstring makes: coordinates are not re-serialized.
    rows = list(block.find("_atom_site.", ["Cartn_x", "B_iso_or_equiv"]))
    assert [(row[0], row[1]) for row in rows] == [
        ("1.000", "90.00"),
        ("2.000", "91.00"),
    ]


def test_confidence_is_written_as_the_model_reported_it(tmp_path: Path) -> None:
    """No unified score is invented: the numbers keep the model's own names.

    Confidence scales differ per model -- a Boltz confidence score and an
    AlphaFold 3 ranking score are different quantities -- so the file says which
    model produced them and stops there.
    """
    result = normalize(_result(tmp_path, model="opendde"), job="1abc")
    payload = json.loads(
        (result.samples[0].structure_path.parent / "confidence.json").read_text()
    )
    assert payload["model"] == "opendde"
    assert payload["seed"] == 3 and payload["sample"] == 0
    assert payload["scores"] == {"ranking_score": 0.5, "plddt": 80.0}
    assert payload["scores_are_model_specific"] is True


def test_a_confidence_symlink_cannot_rewrite_an_external_file(
    tmp_path: Path,
) -> None:
    result = _result(tmp_path, model="opendde")
    directory = tmp_path / "seed-3_sample-00"
    directory.mkdir()
    external = tmp_path / "user-owned.json"
    external.write_text("do not replace\n")
    confidence = directory / "confidence.json"
    confidence.symlink_to(external)

    normalized = normalize(result, job="1abc")

    assert external.read_text() == "do not replace\n"
    assert confidence.is_file() and not confidence.is_symlink()
    payload = json.loads(confidence.read_text())
    assert payload["model"] == normalized.model


def test_an_unparsable_cif_is_still_placed(tmp_path: Path) -> None:
    """A header FoldJAX cannot rewrite must not lose a run that succeeded."""
    native = tmp_path / "job_sample_0.cif"
    native.write_text("this is not a CIF at all", encoding="utf-8")
    result = PredictionResult(
        model="protenix",
        samples=(
            PredictionSample(
                seed=0, structure_path=native, scores={}, metadata={"sample": 0}
            ),
        ),
        output_dir=tmp_path,
        raw={},
    )
    placed = normalize(result, job="j").samples[0].structure_path
    assert placed.is_file()
    assert placed.read_text() == "this is not a CIF at all"


def test_a_sample_without_a_file_passes_through(tmp_path: Path) -> None:
    result = PredictionResult(
        model="boltz2",
        samples=(PredictionSample(seed=0, structure_path=None, scores={}),),
        output_dir=tmp_path,
        raw={},
    )
    assert normalize(result, job="j").samples[0].structure_path is None


def test_normalizing_twice_leaves_the_file_alone(tmp_path: Path) -> None:
    once = normalize(_result(tmp_path), job="1abc")
    twice = normalize(once, job="1abc")
    assert twice.samples[0].structure_path == once.samples[0].structure_path
    assert twice.samples[0].structure_path.is_file()


def test_best_names_the_score_it_ranked_by(tmp_path: Path) -> None:
    result = PredictionResult(
        model="protenix",
        samples=(
            PredictionSample(
                seed=0, structure_path=None, scores={"ranking_score": 0.2}
            ),
            PredictionSample(
                seed=1, structure_path=None, scores={"ranking_score": 0.9}
            ),
        ),
        output_dir=tmp_path,
        raw={},
    )
    best = best_sample(result)
    assert best == {
        "score": "ranking_score",
        "value": 0.9,
        "seed": 1,
        "sample": 1,
        "structure_path": None,
    }


def test_best_is_absent_rather_than_guessed(tmp_path: Path) -> None:
    """Ranking by a quantity the model does not rank with is a different claim."""
    result = PredictionResult(
        model="protenix",
        samples=(
            PredictionSample(seed=0, structure_path=None, scores={"plddt": 80.0}),
        ),
        output_dir=tmp_path,
        raw={},
    )
    assert best_sample(result) is None


def test_a_model_whose_ranking_score_is_not_reported_gets_no_best(
    tmp_path: Path,
) -> None:
    """Boltz-2 is the case: upstream ranks by a score this port does not compute.

    The fields it does report are the components of that score, not the score,
    so electing one would publish a ranking under a rule Boltz does not use.
    """
    result = PredictionResult(
        model="boltz2",
        samples=(
            PredictionSample(
                seed=0,
                structure_path=None,
                scores={"complex_plddt": 0.9, "iptm": 0.5, "ptm": 0.6},
            ),
        ),
        output_dir=tmp_path,
        raw={},
    )
    assert best_sample(result) is None


def test_openfold3_is_ranked_only_by_its_complete_score(tmp_path: Path) -> None:
    scored = PredictionResult(
        model="openfold3",
        samples=(
            PredictionSample(
                seed=0, structure_path=None, scores={"sample_ranking_score": 0.4}
            ),
            PredictionSample(
                seed=0, structure_path=None, scores={"sample_ranking_score": 0.7}
            ),
        ),
        output_dir=tmp_path,
        raw={},
    )
    assert best_sample(scored)["value"] == 0.7
    assert best_sample(scored)["score"] == "sample_ranking_score"
    assert best_sample(scored)["sample"] == 1

    protein = PredictionResult(
        model="openfold3",
        samples=(
            PredictionSample(
                seed=0,
                structure_path=None,
                scores={"sample_ranking_score_no_disorder": 0.9},
            ),
        ),
        output_dir=tmp_path,
        raw={},
    )
    assert best_sample(protein) is None


@pytest.mark.parametrize("incomplete", [None, float("nan"), float("inf")])
def test_best_requires_a_finite_exact_score_from_every_sample(
    tmp_path: Path, incomplete: float | None
) -> None:
    second_scores = {} if incomplete is None else {"ranking_score": incomplete}
    result = PredictionResult(
        model="protenix",
        samples=(
            PredictionSample(seed=0, scores={"ranking_score": 0.8}),
            PredictionSample(seed=1, scores=second_scores),
        ),
        output_dir=tmp_path,
    )
    assert best_sample(result) is None


def test_best_ties_are_stable_and_report_the_backend_sample_index(
    tmp_path: Path,
) -> None:
    result = PredictionResult(
        model="opendde",
        samples=(
            PredictionSample(
                seed=7,
                scores={"ranking_score": 0.8},
                metadata={"sample": 12},
            ),
            PredictionSample(
                seed=8,
                scores={"ranking_score": 0.8},
                metadata={"sample": 3},
            ),
        ),
        output_dir=tmp_path,
    )
    assert best_sample(result) == {
        "score": "ranking_score",
        "value": 0.8,
        "seed": 7,
        "sample": 12,
        "structure_path": None,
    }
