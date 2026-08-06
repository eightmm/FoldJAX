"""One layout and one CIF header, whichever backend produced the run."""

from __future__ import annotations

import json
from pathlib import Path

from foldjax.output import best_sample, normalize
from foldjax.schema import PredictionResult, PredictionSample

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


def test_openfold3_is_ranked_by_the_key_it_actually_reports(tmp_path: Path) -> None:
    """`ranking_score_no_clash`, not `ranking_score`.

    OpenFold3 names it that way on purpose: a Prediction carries no clash term,
    so the veto the real AF3 ranking score applies never fired. It is also only
    emitted for a job with more than one chain, which is why a single-chain run
    legitimately has no best.
    """
    scored = PredictionResult(
        model="openfold3",
        samples=(
            PredictionSample(
                seed=0, structure_path=None, scores={"ranking_score_no_clash": 0.4}
            ),
            PredictionSample(
                seed=0, structure_path=None, scores={"ranking_score_no_clash": 0.7}
            ),
        ),
        output_dir=tmp_path,
        raw={},
    )
    assert best_sample(scored)["value"] == 0.7
    assert best_sample(scored)["score"] == "ranking_score_no_clash"

    single_chain = PredictionResult(
        model="openfold3",
        samples=(
            PredictionSample(
                seed=0, structure_path=None, scores={"mean_plddt": 0.9, "ptm": 0.3}
            ),
        ),
        output_dir=tmp_path,
        raw={},
    )
    assert best_sample(single_chain) is None
