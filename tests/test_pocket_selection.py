"""``--option pocket_sampling=select``: FoldJAX's own pocket route on all six.

The route never changes sampling. After prediction every sample is scored
with Boltz-2's pocket rule (`foldjax.pocket_selection`) and ``best`` prefers
a sample that satisfied every pocket. On AlphaFold 3, ESMFold2, OpenDDE and
the released Protenix checkpoint -- which refuse or drop a common pocket --
the option makes the job run with the pocket read by the selection alone;
``off`` (the default) leaves every backend, file and compile namespace as
it was.
"""

from __future__ import annotations

import dataclasses
import json
import warnings
from pathlib import Path

import pytest

import foldjax
from foldjax import Job, Pocket
from foldjax.api import preflight, resolve_cache_dir, resolve_request
from foldjax.input import IGNORE_CONSTRAINTS, materialize_native_input
from foldjax.manifest import MANIFEST_NAME
from foldjax.output import best_sample
from foldjax.pocket_selection import (
    METADATA_KEY,
    POCKET_SAMPLING,
    SELECTION,
    JobChain,
    annotate,
    job_chains,
    requested,
    score_structure,
)
from foldjax.portspec import PORTS, provider
from foldjax.registry import backend_override, capabilities, get_backend
from foldjax.schema import PredictionRequest, PredictionResult, PredictionSample
from foldjax.summary import load_schema
from tests._schema_lite import errors, undeclared

MODELS = ("alphafold3", "boltz2", "esmfold2", "opendde", "openfold3", "protenix")
#: Which models' native input conditions on a common pocket at the released
#: default (Protenix: only the constraint checkpoint).
NATIVE = {"boltz2", "openfold3"}
#: Whose upstream has a pocket distance of its own when the job omits it.
OWN_DEFAULT = {"boltz2", "openfold3"}

_POCKET = {"binder": "L", "contacts": [["A", 1]], "max_distance": 6.0}


# --- synthetic structures ----------------------------------------------------

_HEADER = """data_test
#
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.B_iso_or_equiv
_atom_site.auth_seq_id
_atom_site.auth_asym_id
_atom_site.pdbx_PDB_model_num
"""


def _atom(
    serial: int,
    element: str,
    name: str,
    comp: str,
    chain: str,
    seq: int,
    xyz: tuple[float, float, float],
    *,
    hetatm: bool = False,
    label_seq: bool = True,
) -> str:
    x, y, z = xyz
    return (
        f"{'HETATM' if hetatm else 'ATOM'} {serial} {element} {name} . {comp} "
        f"{chain} 1 {seq if label_seq else '.'} ? {x:.3f} {y:.3f} {z:.3f} 1.00 "
        f"90.00 {seq} {chain} 1\n"
    )


_THREE = {
    "A": "ALA", "C": "CYS", "D": "ASP", "E": "GLU", "F": "PHE", "G": "GLY",
    "H": "HIS", "I": "ILE", "K": "LYS",
}  # fmt: skip


def _protein(
    chain: str, xs: list[float], *, sequence: str | None = None, label_seq: bool = True
) -> list[str]:
    """One residue per ``xs`` entry, N and CA on the x axis (alanines by default)."""
    rows = []
    for index, x in enumerate(xs, start=1):
        name = _THREE[sequence[index - 1]] if sequence else "ALA"
        rows.append(
            _atom(0, "N", "N", name, chain, index, (x, 1.0, 0.0), label_seq=label_seq)
        )
        rows.append(
            _atom(0, "C", "CA", name, chain, index, (x, 0.0, 0.0), label_seq=label_seq)
        )
    return rows


def _ligand(
    chain: str, atoms: list[tuple[str, tuple[float, float, float]]]
) -> list[str]:
    return [
        _atom(0, element, f"{element}{index}", "LIG", chain, 1, xyz, hetatm=True)
        for index, (element, xyz) in enumerate(atoms, start=1)
    ]


def _cif(path: Path, *blocks: list[str]) -> Path:
    rows = [row for block in blocks for row in block]
    text = _HEADER + "".join(
        row.replace(" 0 ", f" {serial} ", 1) for serial, row in enumerate(rows, start=1)
    )
    path.write_text(text)
    return path


#: Residues of chain A at x = 0, 10, 20, 9; binder L has heavy atoms at
#: x = 3 and a hydrogen 1 A from residue 2; chain M, not the binder, sits
#: on residue 3.
def _structure(path: Path, *, chains: tuple[str, str, str] = ("A", "L", "M")) -> Path:
    protein, binder, other = chains
    return _cif(
        path,
        _protein(protein, [0.0, 10.0, 20.0, 9.0]),
        _ligand(
            binder,
            [("C", (3.0, 0.0, 0.0)), ("O", (3.0, 2.0, 0.0)), ("H", (10.0, 1.0, 0.0))],
        ),
        _ligand(other, [("C", (20.0, 0.5, 0.0))]),
    )


_JOB = (
    JobChain("A", "protein", 4),
    JobChain("L", "ligand", 1),
    JobChain("M", "ligand", 1),
)


def _pocket(*contacts: tuple[str, int], binder: str = "L", max_distance: float = 6.0):
    return {
        "kind": "pocket",
        "binder": binder,
        "contacts": [list(contact) for contact in contacts],
        "max_distance": max_distance,
        "max_distance_source": "job",
    }


# --- the rule ----------------------------------------------------------------


def test_the_rule_measures_heavy_atoms_of_the_binder_chain_only(tmp_path) -> None:
    score = score_structure(
        _structure(tmp_path / "s.cif"), [_pocket(("A", 1), ("A", 2), ("A", 3))], _JOB
    )
    assert score["satisfied"] is False
    (pocket,) = score["pockets"]
    by_residue = {residue["residue"]: residue for residue in pocket["residues"]}
    # Residue 1: binder carbon 3 A away.
    assert by_residue[1] == {
        "chain": "A",
        "residue": 1,
        "min_distance": 3.0,
        "satisfied": True,
    }
    # Residue 2: the binder's hydrogen is 1 A away and does not count; the
    # nearest heavy atom is 7 A.
    assert by_residue[2]["min_distance"] == 7.0 and not by_residue[2]["satisfied"]
    # Residue 3: chain M touches it, but M is not the binder.
    assert by_residue[3]["min_distance"] == 17.0 and not by_residue[3]["satisfied"]


def test_a_pocket_is_satisfied_when_every_listed_residue_is(tmp_path) -> None:
    path = _structure(tmp_path / "s.cif")
    assert score_structure(path, [_pocket(("A", 1))], _JOB)["satisfied"] is True
    # Several pockets: all of them must hold (Boltz-2 allows more than one).
    score = score_structure(path, [_pocket(("A", 1)), _pocket(("A", 2))], _JOB)
    assert score["satisfied"] is False
    assert [pocket["satisfied"] for pocket in score["pockets"]] == [True, False]
    # Chain M as the binder: residue 3 is 0.5 A from it.
    assert score_structure(path, [_pocket(("A", 3), binder="M")], _JOB)["satisfied"]


def test_the_threshold_is_strict_as_in_boltz2s_featurizer(tmp_path) -> None:
    # Residue 4 sits exactly 6 A from the binder's carbon.
    path = _structure(tmp_path / "s.cif")
    assert not score_structure(path, [_pocket(("A", 4))], _JOB)["satisfied"]
    assert score_structure(path, [_pocket(("A", 4), max_distance=6.001)], _JOB)[
        "satisfied"
    ]


def test_chains_are_matched_by_name_else_by_the_jobs_order(tmp_path) -> None:
    # OpenFold3 labels output chains A, B, C in the job's chain order.
    path = _structure(tmp_path / "s.cif", chains=("A", "B", "C"))
    score = score_structure(path, [_pocket(("A", 1))], _JOB)
    assert score["satisfied"] is True
    # The order only stands in when every polymer chain has the job's length.
    wrong = (JobChain("A", "protein", 5), *_JOB[1:])
    score = score_structure(path, [_pocket(("A", 1))], wrong)
    assert score["satisfied"] is None and "5" in score["reason"]
    # A chain count mismatch cannot be mapped either.
    score = score_structure(path, [_pocket(("A", 1))], _JOB[:2])
    assert score["satisfied"] is None and "chain" in score["reason"]


def test_residue_numbers_come_from_auth_seq_id_when_label_seq_id_is_empty(
    tmp_path,
) -> None:
    # ESMFold2's writer leaves label_seq_id '.' and numbers auth_seq_id.
    path = _cif(
        tmp_path / "esm.cif",
        _protein("A", [0.0, 10.0], label_seq=False),
        _ligand("L", [("C", (3.0, 0.0, 0.0))]),
    )
    job = (JobChain("A", "protein", 2), JobChain("L", "ligand", 1))
    score = score_structure(path, [_pocket(("A", 1), ("A", 2))], job)
    assert [r["min_distance"] for r in score["pockets"][0]["residues"]] == [3.0, 7.0]


def test_a_missing_residue_or_unreadable_file_is_unscored_not_fatal(tmp_path) -> None:
    path = _structure(tmp_path / "s.cif")
    score = score_structure(path, [_pocket(("A", 9))], _JOB)
    assert score["satisfied"] is None and "residue 9" in score["reason"]
    missing = score_structure(tmp_path / "none.cif", [_pocket(("A", 1))], _JOB)
    assert missing["satisfied"] is None and "cannot read" in missing["reason"]


def test_job_chains_flatten_entities_and_copies_in_order() -> None:
    job = Job.from_document(
        {
            "name": "j",
            "entities": [
                {"type": "protein", "id": ["A", "B"], "sequence": "ACDE"},
                {"type": "ligand", "id": "L", "ccd": ["NAG", "NAG"]},
                {"type": "ligand", "id": "S", "smiles": "CCO"},
            ],
        }
    )
    assert job_chains(job) == (
        JobChain("A", "protein", 4, "ACDE"),
        JobChain("B", "protein", 4, "ACDE"),
        JobChain("L", "ligand", 2),
        JobChain("S", "ligand", 1),
    )


def test_a_label_that_merely_coincides_with_another_chains_id_is_not_it(
    tmp_path,
) -> None:
    # The job names its protein B and its ligand A; OpenFold3 labels the
    # written chains A (the protein, first) and B (the ligand). Matching by
    # name would swap them; the sequence check sends it to the job's order.
    path = _cif(
        tmp_path / "swapped.cif",
        _protein("A", [0.0, 10.0]),
        _ligand("B", [("C", (3.0, 0.0, 0.0))]),
    )
    job = (JobChain("B", "protein", 2, "AA"), JobChain("A", "ligand", 1))
    score = score_structure(path, [_pocket(("B", 1), binder="A")], job)
    assert score["pockets"][0]["residues"][0]["min_distance"] == 3.0
    # A sequence the structure does not spell is refused on either route.
    job = (JobChain("A", "protein", 2, "GG"), JobChain("B", "ligand", 1))
    score = score_structure(path, [_pocket(("A", 1), binder="B")], job)
    assert score["satisfied"] is None and "is not job chain" in score["reason"]


def test_the_option_takes_off_or_select() -> None:
    assert (
        requested(None) == "off" and requested({POCKET_SAMPLING: "select"}) == "select"
    )
    with pytest.raises(ValueError, match="pocket_sampling must be one of"):
        requested({POCKET_SAMPLING: "on"})


# --- best-sample selection ---------------------------------------------------


def _sample(index: int, score: float, satisfied: bool | None = "unscored"):
    metadata: dict = {"sample": index}
    if satisfied != "unscored":
        metadata[METADATA_KEY] = {"mode": "select", "satisfied": satisfied}
    return PredictionSample(
        seed=1, scores={"confidence_score": score}, metadata=metadata
    )


def test_best_prefers_a_satisfied_sample_by_the_models_own_score() -> None:
    result = PredictionResult(
        model="boltz2",
        samples=(_sample(0, 0.9, False), _sample(1, 0.5, True), _sample(2, 0.7, True)),
    )
    best = best_sample(result)
    assert best["sample"] == 2 and best["value"] == 0.7
    assert best["selection"] == SELECTION and best["pocket_satisfied"] is True


def test_best_falls_back_to_the_models_ranking_when_no_sample_satisfied() -> None:
    result = PredictionResult(
        model="boltz2", samples=(_sample(0, 0.9, False), _sample(1, 0.5, None))
    )
    best = best_sample(result)
    assert best["sample"] == 0
    assert best["selection"] == "within-model confidence ranking"
    assert best["pocket_satisfied"] is False


def test_best_is_unchanged_for_a_run_that_never_scored() -> None:
    result = PredictionResult(
        model="boltz2", samples=(_sample(0, 0.5), _sample(1, 0.9))
    )
    best = best_sample(result)
    assert best["sample"] == 1 and "pocket_satisfied" not in best
    assert best["selection"] == "within-model confidence ranking"


def test_annotate_scores_each_sample_and_leaves_coordinates_alone(tmp_path) -> None:
    near = _structure(tmp_path / "near.cif")
    result = PredictionResult(
        model="boltz2",
        samples=(
            PredictionSample(
                seed=1, structure_path=near, scores={"confidence_score": 1.0}
            ),
            PredictionSample(seed=1, scores={"confidence_score": 1.0}),
        ),
    )
    scored = annotate(result, pockets=[_pocket(("A", 1))], job=_JOB)
    first, second = scored.samples
    assert first.structure_path == near and first.metadata[METADATA_KEY]["satisfied"]
    assert second.metadata[METADATA_KEY]["satisfied"] is None
    assert near.read_text() == _structure(tmp_path / "again.cif").read_text()


# --- per-backend admission ----------------------------------------------------


def _job(
    tmp_path: Path, *pockets: dict, contacts: list | None = None, name: str = "job"
) -> Path:
    (tmp_path / "protein.a3m").write_text(">query\nACDEFGHIK\n>hit\nACDEFGHIY\n")
    document = {
        "name": "pocketed",
        "entities": [
            {
                "type": "protein",
                "id": "A",
                "sequence": "ACDEFGHIK",
                "unpaired_msa": "protein.a3m",
            },
            {"type": "ligand", "id": "L", "smiles": "CCO"},
        ],
        "constraints": [{"pocket": pocket} for pocket in pockets]
        + [{"contact": contact} for contact in contacts or []],
    }
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(document))
    return path


_WEIGHTS = {"protenix": "protenix_base_default_v1.0.0.jax"}


def _request(tmp_path: Path, model: str, source: Path, **kwargs) -> PredictionRequest:
    weights = tmp_path / _WEIGHTS.get(model, "weights.bin")
    weights.write_bytes(b"not really weights")
    fields = {
        "model": model,
        "input": source,
        "weights": weights,
        "output_dir": tmp_path / "out",
        "seed": 1,
        "msa": "none",
        "use_compile_cache": False,
    }
    fields.update(kwargs)
    return PredictionRequest(**fields)


def _plan(tmp_path: Path, model: str, source: Path, **kwargs) -> None:
    preflight(resolve_request(_request(tmp_path, model, source, **kwargs)))


@pytest.mark.parametrize("model", MODELS)
def test_select_lets_every_model_plan_a_pocket_job(tmp_path, model) -> None:
    source = _job(tmp_path, _POCKET)
    _plan(tmp_path, model, source, options={POCKET_SAMPLING: "select"})


@pytest.mark.parametrize("model", sorted(set(MODELS) - OWN_DEFAULT))
def test_select_requires_max_distance_where_upstream_has_no_default(
    tmp_path, model
) -> None:
    source = _job(tmp_path, {"binder": "L", "contacts": [["A", 1]]})
    with pytest.raises(ValueError, match="set max_distance"):
        _plan(tmp_path, model, source, options={POCKET_SAMPLING: "select"})


@pytest.mark.parametrize("model", sorted(OWN_DEFAULT))
def test_select_keeps_the_upstream_default_distance_where_there_is_one(
    tmp_path, model
) -> None:
    source = _job(tmp_path, {"binder": "L", "contacts": [["A", 1]]})
    records: list = []
    materialize_native_input(
        source,
        capabilities(model),
        tmp_path / "out",
        seed=1,
        msa="none",
        options={POCKET_SAMPLING: "select"},
        constraints=records,
    )
    (record,) = records
    assert record["max_distance_source"] == "upstream" and record["route"] == "native"


@pytest.mark.parametrize("spelling", [None, "off"])
@pytest.mark.parametrize("model", MODELS)
def test_off_keeps_todays_refusal_or_drop(tmp_path, model, spelling) -> None:
    source = _job(tmp_path, _POCKET)
    options = {} if spelling is None else {POCKET_SAMPLING: spelling}
    if model in NATIVE:
        _plan(tmp_path, model, source, options=options)
        return
    if model == "opendde":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _plan(tmp_path, model, source, options=options)
        with pytest.raises(ValueError, match=f"{IGNORE_CONSTRAINTS}=false"):
            _plan(
                tmp_path, model, source, options={**options, IGNORE_CONSTRAINTS: False}
            )
        return
    match = (
        "no constraint embedder"
        if model == "protenix"
        else "no pocket or restraint field"
    )
    with pytest.raises(ValueError, match=match):
        _plan(tmp_path, model, source, options=options)


def test_off_and_omitted_translate_a_job_byte_for_byte_the_same(tmp_path) -> None:
    for model in ("boltz2", "opendde"):
        texts = []
        records = []
        for spelling in (None, "off"):
            options = {} if spelling is None else {POCKET_SAMPLING: spelling}
            constraints: list = []
            dropped: list = []
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                path = materialize_native_input(
                    _job(tmp_path, _POCKET),
                    capabilities(model),
                    tmp_path / f"out-{model}-{spelling}",
                    seed=1,
                    msa="none",
                    options=options,
                    constraints=constraints,
                    ignored_constraints=dropped,
                )
            texts.append(path.read_text())
            records.append((constraints, dropped))
        assert texts[0] == texts[1]
        assert records[0] == records[1]
        for record in records[0][0]:
            assert "route" not in record


def test_select_is_refused_on_native_input(tmp_path) -> None:
    native = tmp_path / "native.yaml"
    native.write_text(
        "sequences:\n  - protein:\n      id: A\n      sequence: ACDEFGHIK\n"
    )
    with pytest.raises(ValueError, match="native boltz2 input is passed through"):
        resolve_request(
            _request(tmp_path, "boltz2", native, options={POCKET_SAMPLING: "select"})
        )


def test_an_unknown_value_is_refused_while_planning(tmp_path) -> None:
    with pytest.raises(ValueError, match="pocket_sampling must be one of"):
        resolve_request(
            _request(
                tmp_path,
                "boltz2",
                _job(tmp_path, _POCKET),
                options={POCKET_SAMPLING: "yes"},
            )
        )


def test_opendde_select_reads_the_pocket_and_still_drops_a_contact(tmp_path) -> None:
    contact = {"token1": ["A", 1], "token2": ["A", 9], "max_distance": 8.0}
    source = _job(tmp_path, _POCKET, contacts=[contact], name="with-contact")
    records: list = []
    dropped: list = []
    with pytest.warns(UserWarning, match="contact constraint"):
        path = materialize_native_input(
            source,
            capabilities("opendde"),
            tmp_path / "out",
            seed=1,
            msa="none",
            options={POCKET_SAMPLING: "select"},
            constraints=records,
            ignored_constraints=dropped,
        )
    (native,) = json.loads(path.read_text())
    assert "constraint" not in native
    (record,) = records
    assert record["kind"] == "pocket" and record["route"] == "selection"
    assert record["max_distance"] == 6.0 and record["max_distance_source"] == "job"
    (drop,) = dropped
    assert drop["keys"] == ["contact"] and drop["binders"] == []
    # The pocket is read, so refusing an ignored constraint no longer names it...
    _plan(
        tmp_path,
        "opendde",
        _job(tmp_path, _POCKET),
        options={POCKET_SAMPLING: "select", IGNORE_CONSTRAINTS: False},
    )
    # ... but a contact is still dropped, and still refused on request.
    with pytest.raises(ValueError, match="contact constraint"):
        _plan(
            tmp_path,
            "opendde",
            source,
            options={POCKET_SAMPLING: "select", IGNORE_CONSTRAINTS: False},
        )


def test_models_without_a_pocket_field_keep_a_contact_refusal_under_select(
    tmp_path,
) -> None:
    contact = {"token1": ["A", 1], "token2": ["A", 9], "max_distance": 8.0}
    source = _job(tmp_path, _POCKET, contacts=[contact])
    for model in ("alphafold3", "esmfold2"):
        with pytest.raises(ValueError, match="contact"):
            _plan(tmp_path, model, source, options={POCKET_SAMPLING: "select"})


def test_openfold3_keeps_its_own_pocket_rules_under_select(tmp_path) -> None:
    (tmp_path / "protein.a3m").write_text(">query\nACDEFGHIK\n")
    document = {
        "name": "polymer-binder",
        "entities": [
            {
                "type": "protein",
                "id": "A",
                "sequence": "ACDEFGHIK",
                "unpaired_msa": "protein.a3m",
            },
            {
                "type": "protein",
                "id": "B",
                "sequence": "MKV",
                "unpaired_msa": "protein.a3m",
            },
        ],
        "constraints": [
            {"pocket": {"binder": "B", "contacts": [["A", 1]], "max_distance": 6.0}}
        ],
    }
    source = tmp_path / "job.json"
    source.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="polymer binder"):
        _plan(tmp_path, "openfold3", source, options={POCKET_SAMPLING: "select"})


def test_protenix_routes_by_checkpoint(tmp_path) -> None:
    source = _job(tmp_path, _POCKET)
    backend = get_backend("protenix")
    default = resolve_request(
        _request(tmp_path, "protenix", source, options={POCKET_SAMPLING: "select"})
    )
    assert backend.pocket_conditioning(default) is False
    weights = tmp_path / "protenix_base_constraint_v0.5.0.jax"
    weights.write_bytes(b"not really weights")
    conditioned = resolve_request(
        dataclasses.replace(
            default, weights=weights, options={POCKET_SAMPLING: "select"}
        )
    )
    assert backend.pocket_conditioning(conditioned) is True
    # The constraint checkpoint keeps writing the native pocket.
    records: list = []
    path = materialize_native_input(
        source,
        capabilities("protenix"),
        tmp_path / "native",
        seed=1,
        msa="none",
        options={POCKET_SAMPLING: "select"},
        constraints=records,
        pocket_conditioning=True,
    )
    (native,) = json.loads(path.read_text())
    assert "pocket" in native["constraint"] and records[0]["route"] == "native"
    # The default one leaves it to the selection.
    records = []
    path = materialize_native_input(
        source,
        capabilities("protenix"),
        tmp_path / "selection",
        seed=1,
        msa="none",
        options={POCKET_SAMPLING: "select"},
        constraints=records,
        pocket_conditioning=False,
    )
    (native,) = json.loads(path.read_text())
    assert "constraint" not in native and records[0]["route"] == "selection"


@pytest.mark.parametrize("model", MODELS)
def test_the_option_never_changes_the_compile_namespace(tmp_path, model) -> None:
    source = _job(tmp_path, _POCKET)
    backend = get_backend(model)
    omitted = _request(
        tmp_path, model, source, use_compile_cache=True, cache_dir=tmp_path / "cache"
    )
    namespace = resolve_cache_dir(omitted, backend)
    profile = backend.cache_profile(omitted)
    for value in ("off", "select"):
        spelled = dataclasses.replace(omitted, options={POCKET_SAMPLING: value})
        assert backend.cache_profile(spelled) == profile
        assert resolve_cache_dir(spelled, backend) == namespace


# --- capabilities ---------------------------------------------------------------


@pytest.mark.parametrize("model", MODELS)
def test_capabilities_list_the_route_as_foldjax_only(model, capsys) -> None:
    from foldjax import cli
    from foldjax.registry import model_info

    assert cli.main(["capabilities", "--model", model, "--json"]) == 0
    described = json.loads(capsys.readouterr().out)
    assert described["foldjax_only_features"] == ["pocket_selection"]
    assert "pocket_selection" not in described["common_schema_features"]
    assert "pocket_selection" not in described["native_only_features"]
    assert model_info(model).summary()["foldjax_only_features"] == ["pocket_selection"]


# --- end to end, through foldjax.predict -------------------------------------


def _recorder(model: str, placements: list[float], seen: list | None = None):
    """A backend that writes one structure per entry of ``placements``.

    Each sample puts the ligand ``x`` angstrom from residue 1 of chain A; the
    samples' ranking score falls with the index, so the model's own ranking
    always prefers the first one.
    """
    from foldjax.output import _RANKING_SCORE

    base = provider(PORTS[model].backend)
    key = _RANKING_SCORE[model]

    class Recorder(base):
        def predict(self, request):
            if seen is not None:
                seen.append(request)
            request.output_dir.mkdir(parents=True, exist_ok=True)
            samples = []
            for index, x in enumerate(placements):
                path = request.output_dir / f"s{request.seed}_{index}.cif"
                _cif(
                    path,
                    _protein(
                        "A",
                        [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
                        sequence="ACDEFGHIK",
                    ),
                    _ligand("L", [("C", (x, 0.0, 0.0)), ("O", (x, 1.0, 0.0))]),
                )
                samples.append(
                    PredictionSample(
                        seed=request.seed,
                        structure_path=path,
                        scores={key: 0.9 - 0.1 * index, "ptm": 0.5},
                        metadata={"sample": index},
                    )
                )
            return PredictionResult(
                model=model, samples=tuple(samples), output_dir=request.output_dir
            )

    return Recorder


def _predict(tmp_path: Path, model: str, placements: list[float], **kwargs):
    source = _job(tmp_path, _POCKET)
    request = _request(tmp_path, model, source, **kwargs)
    with (
        backend_override(model, _recorder(model, placements)),
        warnings.catch_warnings(),
    ):
        warnings.simplefilter("ignore")
        result = foldjax.predict(request)
    out = request.output_dir
    manifest = json.loads((out / MANIFEST_NAME).read_text())
    confidences = [
        json.loads(path.read_text())
        for path in sorted(out.glob("seed-*/confidence.json"))
    ]
    return result, manifest, confidences


@pytest.mark.parametrize("model", MODELS)
def test_select_ranks_the_satisfied_sample_first_and_records_the_route(
    tmp_path, model
) -> None:
    # Sample 0 ranks higher but leaves the ligand 9 A away; sample 1 has it 3 A away.
    result, manifest, confidences = _predict(
        tmp_path, model, [9.0, 3.0], options={POCKET_SAMPLING: "select"}
    )
    assert manifest["best"]["sample"] == 1
    assert manifest["best"]["selection"] == SELECTION
    assert manifest["best"]["pocket_satisfied"] is True
    block = manifest["pocket_sampling"]
    assert block["mode"] == "select" and block["route"] == "foldjax"
    assert block["native_conditioning"] is (model in NATIVE)
    assert (block["samples"], block["satisfied"], block["unscored"]) == (2, 1, 0)
    (pocket,) = block["pockets"]
    assert pocket["route"] == ("native" if model in NATIVE else "selection")
    assert manifest["constraints"] == [pocket]
    assert manifest["options"][POCKET_SAMPLING] == "select"
    assert [c[METADATA_KEY]["satisfied"] for c in confidences] == [False, True]
    assert confidences[1][METADATA_KEY]["pockets"][0]["residues"] == [
        {"chain": "A", "residue": 1, "min_distance": 3.0, "satisfied": True}
    ]
    if model == "opendde":
        assert manifest["ignored_constraints"] == []
    # Both files keep to the published contracts, with nothing undeclared.
    assert errors(manifest, load_schema("run")) == []
    assert undeclared(manifest, load_schema("run")) == []
    for confidence in confidences:
        assert errors(confidence, load_schema("confidence")) == []
        assert undeclared(confidence, load_schema("confidence")) == []
    # The samples themselves are untouched: the recorder's files, in order.
    assert [
        sample.metadata[METADATA_KEY]["satisfied"] for sample in result.samples
    ] == [
        False,
        True,
    ]
    # Every reader of the run directory takes the widened manifest: the
    # results loader, its tables and the comparison carry the pocket-aware
    # `best`; the group aggregate ranks by score alone, as it documents.
    from foldjax.compare import compare_rows
    from foldjax.results import aggregate_table, load_results, results_table, to_csv

    (run,) = load_results(tmp_path / "out").runs
    assert run.best["pocket_satisfied"] is True and run.best["sample"] == 1
    assert run.constraints[0]["route"] == pocket["route"]
    rows = results_table(load_results(tmp_path / "out"))
    assert len(rows) == 2 and to_csv(rows)
    (group,) = aggregate_table(rows)
    assert group["best_within_model"]["sample"] == 0
    (entry,) = compare_rows(tmp_path / "out")["inputs"]
    assert len(entry["structures"]) == 2


def test_select_with_no_satisfied_sample_keeps_the_models_ranking(tmp_path) -> None:
    _result, manifest, _confidences = _predict(
        tmp_path, "boltz2", [9.0, 12.0], options={POCKET_SAMPLING: "select"}
    )
    assert manifest["best"]["sample"] == 0
    assert manifest["best"]["selection"] == "within-model confidence ranking"
    assert manifest["best"]["pocket_satisfied"] is False
    assert manifest["pocket_sampling"]["satisfied"] == 0


def test_off_writes_the_same_outputs_as_an_omitted_option(tmp_path) -> None:
    runs = {}
    for spelling in (None, "off"):
        options = {} if spelling is None else {POCKET_SAMPLING: spelling}
        root = tmp_path / str(spelling)
        root.mkdir()
        runs[spelling] = _predict(root, "boltz2", [9.0, 3.0], options=options)
    (_, omitted, omitted_confidences), (_, off, off_confidences) = (
        runs[None],
        runs["off"],
    )
    # The model's own ranking stands and nothing about the pocket is recorded.
    for manifest in (omitted, off):
        assert manifest["best"]["sample"] == 0
        assert "pocket_satisfied" not in manifest["best"]
        assert "pocket_sampling" not in manifest
        for sample in manifest["samples"]:
            assert METADATA_KEY not in sample["metadata"]
        assert "route" not in manifest["constraints"][0]
    assert [c.keys() for c in omitted_confidences] == [
        c.keys() for c in off_confidences
    ]
    assert all(METADATA_KEY not in c for c in omitted_confidences)
    # The manifests differ only where they must: the typed option, the clock,
    # the cost and the paths under two roots.
    differing = {
        key for key in set(omitted) | set(off) if omitted.get(key) != off.get(key)
    }
    assert differing <= {
        "options",
        "finished",
        "cost",
        "input",
        "weights",
        "native_input",
        "input_dependencies",
        "msa_stats",
        "samples",
        "best",
        "output_dir",
        "confidence_arrays",
    }
    assert {k: v for k, v in off["options"].items() if k != POCKET_SAMPLING} == omitted[
        "options"
    ]
    assert off["options"][POCKET_SAMPLING] == "off"
    for left, right in zip(omitted["samples"], off["samples"], strict=True):
        assert left["structure_sha256"] == right["structure_sha256"]
        assert (
            left["scores"] == right["scores"] and left["metadata"] == right["metadata"]
        )
    for left, right in zip(omitted_confidences, off_confidences, strict=True):
        assert left == right


def test_several_seeds_combine_the_counts(tmp_path) -> None:
    source = _job(tmp_path, _POCKET)
    request = _request(
        tmp_path,
        "boltz2",
        source,
        seed=None,
        seeds=(1, 2),
        options={POCKET_SAMPLING: "select"},
    )
    with (
        backend_override("boltz2", _recorder("boltz2", [9.0, 3.0])),
        warnings.catch_warnings(),
    ):
        warnings.simplefilter("ignore")
        foldjax.predict(request)
    manifest = json.loads((request.output_dir / MANIFEST_NAME).read_text())
    block = manifest["pocket_sampling"]
    assert (block["samples"], block["satisfied"], block["unscored"]) == (4, 2, 0)
    assert "pockets" not in block and "native_conditioning" not in block
    assert manifest["best"]["pocket_satisfied"] is True
    assert manifest["best"]["sample"] == 1
    assert errors(manifest, load_schema("run")) == []
    assert undeclared(manifest, load_schema("run")) == []
    per_seed = json.loads((request.output_dir / "seed_1" / MANIFEST_NAME).read_text())
    assert per_seed["pocket_sampling"]["pockets"][0]["route"] == "native"


def test_the_job_builder_round_trips_a_pocket_meant_for_selection() -> None:
    job = Job(
        "j",
        entities=(foldjax.Protein("A", "ACDE"), foldjax.Ligand("L", smiles="CCO")),
        pockets=(Pocket("L", (("A", 2),), max_distance=5.0),),
    )
    assert job_chains(Job.from_document(job.to_document())) == (
        JobChain("A", "protein", 4, "ACDE"),
        JobChain("L", "ligand", 1),
    )


# --- the six ports' own files --------------------------------------------------

_FIXTURES = Path(__file__).parent / "fixtures" / "outputs"


@pytest.mark.parametrize(
    "case", sorted(path.name for path in _FIXTURES.iterdir() if path.is_dir())
)
def test_the_rule_reads_what_each_port_actually_writes(tmp_path, case) -> None:
    """Numeric label_seq_id, ESMFold2's '.' with auth_seq_id, HETATM ligands."""
    import gzip

    import gemmi

    fixture = json.loads((_FIXTURES / case / "sample.json").read_text())
    path = tmp_path / "structure.cif"
    with gzip.open(_FIXTURES / case / "structure.cif.gz", "rb") as source:
        path.write_bytes(source.read())
    structure = gemmi.read_structure(str(path))
    structure.setup_entities()
    chains = []
    for chain in structure[0]:
        polymer = any(
            residue.entity_type == gemmi.EntityType.Polymer for residue in chain
        )
        chains.append(
            JobChain(chain.name, "protein" if polymer else "ligand", len(chain))
        )
    binder = chains[-1].id
    pocket = _pocket(("A", 1), (chains[0].id, chains[0].residues), binder=binder)
    score = score_structure(path, [pocket], chains, model=fixture["model"])
    assert score["satisfied"] in (True, False), score
    residues = score["pockets"][0]["residues"]
    assert [r["residue"] for r in residues] == [1, chains[0].residues]
    assert all(r["min_distance"] >= 0.0 for r in residues)
    if binder == "A":
        # A chain against itself: the first residue is its own binder atom.
        assert residues[0]["min_distance"] == 0.0 and residues[0]["satisfied"]
    else:
        assert all(r["min_distance"] > 0.0 for r in residues)


def test_openfold3s_positional_labels_tell_two_ligands_apart(tmp_path) -> None:
    # Job order: ligand B, ligand A, protein C. OpenFold3 writes A=job B,
    # B=job A, C=protein: by name the two ligands would swap.
    path = _cif(
        tmp_path / "of3.cif",
        _ligand("A", [("C", (3.0, 0.0, 0.0))]),
        _ligand("B", [("C", (30.0, 0.0, 0.0))]),
        _protein("C", [0.0, 10.0]),
    )
    job = (
        JobChain("B", "ligand", 1),
        JobChain("A", "ligand", 1),
        JobChain("C", "protein", 2, "AA"),
    )
    near = score_structure(
        path, [_pocket(("C", 1), binder="B")], job, model="openfold3"
    )
    far = score_structure(path, [_pocket(("C", 1), binder="A")], job, model="openfold3")
    assert near["pockets"][0]["residues"][0]["min_distance"] == 3.0
    assert far["pockets"][0]["residues"][0]["min_distance"] == 30.0
    # A writer that keeps the job's ids reads the same file by name.
    by_name = score_structure(
        path, [_pocket(("C", 1), binder="A")], job, model="boltz2"
    )
    assert by_name["pockets"][0]["residues"][0]["min_distance"] == 3.0


def test_a_polymer_chain_cannot_stand_in_for_a_ligand(tmp_path) -> None:
    path = _cif(
        tmp_path / "s.cif",
        _protein("A", [0.0]),
        _protein("L", [3.0]),
    )
    job = (JobChain("A", "protein", 1, "A"), JobChain("L", "ligand", 1))
    score = score_structure(path, [_pocket(("A", 1))], job)
    assert score["satisfied"] is None and "is not job chain 'L'" in score["reason"]
