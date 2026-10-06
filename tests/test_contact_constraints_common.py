"""The common ``constraints: [{contact: ...}]`` field, translated per backend.

A contact names two residues, ``[chain_id, residue_index]`` each, and an
optional ``max_distance``. Boltz-2 takes its own ``contact`` constraint with
upstream's default 6.0 when the job omits the distance; Protenix a token
``constraint.contact`` addressed by entity/copy/position, with no default and
only across chains, as upstream's ``_canonicalize_contact_format`` requires.
A ligand residue is refused for both: Boltz-2 addresses a ligand only by atom
name, and Protenix draws a random atom token of a multi-token residue (a
modified residue too). OpenDDE's upstream ignores a constraint, so the field
is dropped and recorded as a pocket is; AlphaFold 3, ESMFold2 and OpenFold3
have no contact field upstream and refuse it.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import pytest

import foldjax
from foldjax import Contact, Job, Pocket
from foldjax.input import (
    IGNORE_CONSTRAINTS,
    common_schema_features,
    materialize_native_input,
    native_only_features,
)
from foldjax.manifest import MANIFEST_NAME
from foldjax.registry import backend_override, capabilities
from foldjax.schema import PredictionRequest
from tests.test_pocket_constraints_common import _find, _recorder

_CONTACT = {"token1": ["A", 2], "token2": ["B", 3]}


def _job(
    tmp_path: Path, *constraints: dict, modified: bool = False, ligand=("NAG", "BMA")
) -> Path:
    (tmp_path / "protein.a3m").write_text(">query\nACDEFGHIK\n>hit\nACDEFGHIY\n")
    (tmp_path / "b.a3m").write_text(">query\nMKVLS\n")
    protein_a = {
        "type": "protein",
        "id": "A",
        "sequence": "ACDEFGHIK",
        "unpaired_msa": "protein.a3m",
    }
    if modified:
        protein_a["modifications"] = [{"ccd": "SEP", "position": 2}]
    document = {
        "name": "contacted",
        "entities": [
            protein_a,
            {
                "type": "protein",
                "id": "B",
                "sequence": "MKVLS",
                "unpaired_msa": "b.a3m",
            },
            {"type": "ligand", "id": "L", "ccd": list(ligand)},
        ],
        "constraints": list(constraints),
    }
    path = tmp_path / "job.json"
    path.write_text(json.dumps(document))
    return path


def _contact(**body) -> dict:
    return {"contact": {**_CONTACT, **body}}


def _materialize(source: Path, model: str, *, options=None, records=None, dropped=None):
    return materialize_native_input(
        source,
        capabilities(model),
        source.parent / f"out-{model}",
        seed=1,
        msa="none",
        options=options,
        constraints=records,
        ignored_constraints=dropped,
    )


def test_boltz2_takes_its_own_contact_form_and_default(tmp_path) -> None:
    records: list = []
    native = json.loads(
        _materialize(
            _job(tmp_path, _contact(), _contact(token2=["B", 5], max_distance=9)),
            "boltz2",
            records=records,
        ).read_text()
    )
    assert native["constraints"] == [
        {"contact": {"token1": ["A", 2], "token2": ["B", 3], "max_distance": 6.0}},
        {"contact": {"token1": ["A", 2], "token2": ["B", 5], "max_distance": 9.0}},
    ]
    assert records == [
        {
            "kind": "contact",
            "token1": ["A", 2],
            "token2": ["B", 3],
            "max_distance": 6.0,
            "max_distance_source": "upstream",
        },
        {
            "kind": "contact",
            "token1": ["A", 2],
            "token2": ["B", 5],
            "max_distance": 9.0,
            "max_distance_source": "job",
        },
    ]


def test_boltz2_keeps_a_pocket_and_a_contact_together(tmp_path) -> None:
    pocket = {"pocket": {"binder": "L", "contacts": [["A", 4]]}}
    records: list = []
    native = json.loads(
        _materialize(
            _job(tmp_path, _contact(), pocket), "boltz2", records=records
        ).read_text()
    )
    assert [next(iter(item)) for item in native["constraints"]] == [
        "pocket",
        "contact",
    ]
    assert [record["kind"] for record in records] == ["pocket", "contact"]


def test_boltz2_allows_a_contact_within_one_chain(tmp_path) -> None:
    native = json.loads(
        _materialize(_job(tmp_path, _contact(token2=["A", 7])), "boltz2").read_text()
    )
    assert native["constraints"][0]["contact"]["token2"] == ["A", 7]


def test_protenix_addresses_the_contact_by_entity_copy_and_position(tmp_path) -> None:
    pocket = {"pocket": {"binder": "L", "contacts": [["A", 4]], "max_distance": 5}}
    records: list = []
    (native,) = json.loads(
        _materialize(
            _job(tmp_path, _contact(max_distance=7.5), pocket),
            "protenix",
            records=records,
        ).read_text()
    )
    assert native["constraint"]["contact"] == [
        {
            "entity1": 1,
            "copy1": 1,
            "position1": 2,
            "entity2": 2,
            "copy2": 1,
            "position2": 3,
            "max_distance": 7.5,
        }
    ]
    assert native["constraint"]["pocket"]["max_distance"] == 5.0
    assert records[1] == {
        "kind": "contact",
        "token1": ["A", 2],
        "token2": ["B", 3],
        "max_distance": 7.5,
        "max_distance_source": "job",
    }


def test_protenix_featurizer_puts_the_contact_in_its_channel(tmp_path) -> None:
    import numpy as np

    from foldjax.models.protenix.data.featurize_json import featurize_protein_json

    # The glycan needs Protenix's full CCD for leaving atoms; polymers do not.
    source = _job(tmp_path, _contact(max_distance=7.5))
    document = json.loads(source.read_text())
    document["entities"].pop()
    source.write_text(json.dumps(document))
    (native,) = json.loads(_materialize(source, "protenix").read_text())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        features = featurize_protein_json(
            native, base_dir=tmp_path / "out-protenix", seed=1
        )
    channel = np.asarray(_find(features, "constraint_feature")["contact"])
    # A2 is token 1; B starts at token 9, so B3 is token 11.
    assert {tuple(index) for index in np.argwhere(channel[..., 1])} == {
        (1, 11),
        (11, 1),
    }
    assert channel[1, 11].tolist() == [0.0, 7.5]


@pytest.mark.parametrize(
    ("constraint", "match"),
    [
        (_contact(), "contact constraint without max_distance"),
        (_contact(token2=["A", 5], max_distance=5), "contact within one chain"),
        (_contact(token2=["L", 1], max_distance=5), "random atom token of a ligand"),
    ],
)
def test_protenix_refuses_what_upstream_cannot_read_exactly(
    tmp_path, constraint, match
) -> None:
    with pytest.raises(ValueError, match=match):
        _materialize(_job(tmp_path, constraint), "protenix")


def test_protenix_refuses_a_contact_on_a_modified_residue(tmp_path) -> None:
    with pytest.raises(ValueError, match="modified residue A:2"):
        _materialize(
            _job(tmp_path, _contact(max_distance=5), modified=True), "protenix"
        )
    # Boltz-2 matches the first token of that residue, deterministically.
    native = json.loads(
        _materialize(_job(tmp_path, _contact(), modified=True), "boltz2").read_text()
    )
    assert native["constraints"][0]["contact"]["token1"] == ["A", 2]


def test_boltz2_refuses_a_ligand_residue(tmp_path) -> None:
    with pytest.raises(ValueError, match="by atom name, not by residue"):
        _materialize(_job(tmp_path, _contact(token1=["L", 2])), "boltz2")


def test_opendde_drops_and_records_the_contact_unless_refused(tmp_path) -> None:
    dropped: list = []
    records: list = []
    with pytest.warns(UserWarning, match="1 contact constraint"):
        path = _materialize(
            _job(tmp_path, _contact()), "opendde", records=records, dropped=dropped
        )
    (native,) = json.loads(path.read_text())
    assert "constraint" not in native
    assert records == []
    (record,) = dropped
    assert record["keys"] == ["contact"] and record["binders"] == []
    assert record["contacts"] == [[["A", 2], ["B", 3]]]
    with pytest.raises(ValueError, match=f"contact constraint.*{IGNORE_CONSTRAINTS}"):
        _materialize(
            _job(tmp_path, _contact()),
            "opendde",
            options={IGNORE_CONSTRAINTS: False},
        )


def test_opendde_records_a_pocket_and_a_contact_in_one_drop(tmp_path) -> None:
    pocket = {"pocket": {"binder": "L", "contacts": [["A", 4]]}}
    dropped: list = []
    with pytest.warns(UserWarning, match="pocket constraint .* and 1 contact"):
        _materialize(_job(tmp_path, pocket, _contact()), "opendde", dropped=dropped)
    (record,) = dropped
    assert record["keys"] == ["pocket", "contact"]
    assert record["binders"] == ["L"]


@pytest.mark.parametrize("model", ["alphafold3", "esmfold2", "openfold3"])
def test_models_without_a_contact_field_refuse_it(tmp_path, model) -> None:
    with pytest.raises(ValueError, match="no contact restraint field"):
        _materialize(_job(tmp_path, _contact(), ligand=["ATP"]), model)


@pytest.mark.parametrize(
    ("body", "match"),
    [
        ({"token1": ["A", 2]}, "token2 must be"),
        ({**_CONTACT, "token2": ["Q", 1]}, "unknown chain id"),
        ({**_CONTACT, "token2": ["B", 6]}, "outside chain 'B'"),
        ({**_CONTACT, "token2": ["L", 3]}, "outside chain 'L', which has 2"),
        ({**_CONTACT, "token1": ["A", 0]}, "1-based"),
        ({**_CONTACT, "token1": ["A", True]}, "must be an integer"),
        ({**_CONTACT, "token2": ["A", 2]}, "name the same residue"),
        ({**_CONTACT, "max_distance": -1}, "positive and finite"),
        ({**_CONTACT, "max_distance": "6"}, "must be a number"),
        ({**_CONTACT, "force": True}, "contact fields"),
        ({**_CONTACT, "min_distance": 1}, "contact fields"),
    ],
)
def test_malformed_contacts_are_refused_for_every_model(tmp_path, body, match) -> None:
    with pytest.raises(ValueError, match=match):
        _materialize(_job(tmp_path, {"contact": body}), "boltz2")


def test_a_constraint_item_names_one_kind(tmp_path) -> None:
    both = {"pocket": {"binder": "L", "contacts": [["A", 4]]}, "contact": _CONTACT}
    for item in (both, {"distance": _CONTACT}):
        with pytest.raises(ValueError, match="exactly one field, pocket or contact"):
            _materialize(_job(tmp_path, item), "boltz2")


def test_the_capability_record_moves_contacts_to_the_common_schema() -> None:
    for model in ("boltz2", "protenix"):
        assert "contact_constraints" in common_schema_features(model), model
        assert "contact_constraints" not in native_only_features(
            model, capabilities(model)
        )
    for model in ("alphafold3", "esmfold2", "opendde", "openfold3"):
        assert "contact_constraints" not in common_schema_features(model), model
        assert "contact_constraints" not in native_only_features(
            model, capabilities(model)
        )


def test_job_builder_round_trips_contacts() -> None:
    document = {
        "name": "j",
        "entities": [
            {"type": "protein", "id": "A", "sequence": "ACDEF"},
            {"type": "protein", "id": "B", "sequence": "MKV"},
            {"type": "ligand", "id": "L", "ccd": ["NAG", "BMA"]},
        ],
        "constraints": [
            {"pocket": {"binder": "L", "contacts": [["A", 2]]}},
            {"contact": {"token1": ["A", 2], "token2": ["B", 3], "max_distance": 5.0}},
        ],
    }
    job = Job.from_document(document)
    assert job.pockets == (Pocket("L", (("A", 2),)),)
    assert job.contacts == (Contact(("A", 2), ("B", 3), max_distance=5.0),)
    assert job.to_document()["constraints"] == document["constraints"]
    # Residues of a ligand are counted by its CCD codes.
    with pytest.raises(ValueError, match="which has 2"):
        Job.from_document(
            {
                **document,
                "constraints": [{"contact": {"token1": ["A", 2], "token2": ["L", 3]}}],
            }
        )


@pytest.mark.parametrize("model", ["boltz2", "opendde"])
def test_the_manifest_records_what_ran(tmp_path, model) -> None:
    weights = tmp_path / "weights.bin"
    weights.write_bytes(b"not really weights")
    seen: list = []
    with backend_override(model, _recorder(model, seen)), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        foldjax.predict(
            PredictionRequest(
                model=model,
                input=_job(tmp_path, _contact()),
                weights=weights,
                output_dir=tmp_path / "out",
                seed=3,
                msa="none",
                use_compile_cache=False,
            )
        )
    manifest = json.loads((tmp_path / "out" / MANIFEST_NAME).read_text())
    if model == "boltz2":
        (record,) = manifest["constraints"]
        assert record["kind"] == "contact"
        assert record["max_distance"] == 6.0
        assert record["max_distance_source"] == "upstream"
        assert manifest["ignored_constraints"] is None
    else:
        assert manifest["constraints"] == []
        (record,) = manifest["ignored_constraints"]
        assert record["keys"] == ["contact"]


def test_results_and_compare_carry_the_contact_record(tmp_path) -> None:
    from foldjax.compare import compare_rows
    from foldjax.results import load_results, results_table
    from foldjax.summary import load_schema
    from tests._schema_lite import errors
    from tests.test_output_contract import _run

    (tmp_path / "jobs").mkdir()
    job = _job(tmp_path / "jobs", _contact(max_distance=8))
    root = tmp_path / "batch"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _run(tmp_path / "scratch", "boltz2_e9_8reh", out=root / "boltz2" / "c", job=job)
    (run,) = load_results(root).runs
    (record,) = run.constraints
    assert record["kind"] == "contact" and record["max_distance"] == 8.0
    assert record["max_distance_source"] == "job"
    (row,) = results_table(load_results(root))
    assert row["constraints"] == [dict(record)]
    (entry,) = compare_rows(root)["inputs"]
    (structure,) = entry["structures"]
    assert structure["constraints"] == [dict(record)]
    assert not errors(dict(run.manifest), load_schema("run"))
