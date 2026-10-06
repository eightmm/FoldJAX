"""The common ``constraints: [{pocket: ...}]`` field, translated per backend.

A pocket names a binder chain and the polymer residues it should sit near.
Boltz-2, OpenFold3 and Protenix carry it into their own native field, with
each upstream's own ``max_distance`` when the job omits one (Boltz-2 6.0,
OpenFold3 4.0; Protenix has no default and is refused without one). OpenDDE's
upstream ignores a constraint, so the field is dropped and recorded as a
native OpenDDE constraint is. AlphaFold 3 and ESMFold2 have no such field
upstream and refuse it.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import pytest

import foldjax
from foldjax import Job, Pocket
from foldjax.input import (
    IGNORE_CONSTRAINTS,
    common_schema_features,
    materialize_native_input,
    native_only_features,
)
from foldjax.manifest import MANIFEST_NAME
from foldjax.portspec import PORTS, provider
from foldjax.registry import backend_override, capabilities
from foldjax.schema import PredictionRequest, PredictionResult, PredictionSample

_POCKET = {"binder": "L", "contacts": [["A", 2], ["A", 5]]}


def _job(tmp_path: Path, *pockets: dict, binder_ligand: bool = True) -> Path:
    (tmp_path / "protein.a3m").write_text(">query\nACDEFGHIK\n>hit\nACDEFGHIY\n")
    entities: list[dict] = [
        {
            "type": "protein",
            "id": "A",
            "sequence": "ACDEFGHIK",
            "unpaired_msa": "protein.a3m",
        },
        {"type": "ligand", "id": "L", "smiles": "CCO"},
    ]
    if not binder_ligand:
        entities.append(
            {
                "type": "protein",
                "id": "B",
                "sequence": "MKV",
                "unpaired_msa": "protein.a3m",
            }
        )
    document = {"name": "pocketed", "entities": entities}
    document["constraints"] = [{"pocket": pocket} for pocket in pockets]
    path = tmp_path / "job.json"
    path.write_text(json.dumps(document))
    return path


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


def test_boltz2_takes_its_own_pocket_form_and_default(tmp_path) -> None:
    records: list = []
    native = json.loads(
        _materialize(_job(tmp_path, _POCKET), "boltz2", records=records).read_text()
    )
    assert native["constraints"] == [
        {
            "pocket": {
                "binder": "L",
                "contacts": [["A", 2], ["A", 5]],
                "max_distance": 6.0,
            }
        }
    ]
    assert records == [
        {
            "kind": "pocket",
            "binder": "L",
            "contacts": [["A", 2], ["A", 5]],
            "max_distance": 6.0,
            "max_distance_source": "upstream",
        }
    ]


def test_boltz2_keeps_several_pockets_and_an_explicit_distance(tmp_path) -> None:
    second = {"binder": "L", "contacts": [["A", 9]], "max_distance": 8}
    records: list = []
    native = json.loads(
        _materialize(
            _job(tmp_path, _POCKET, second), "boltz2", records=records
        ).read_text()
    )
    assert [item["pocket"]["max_distance"] for item in native["constraints"]] == [
        6.0,
        8.0,
    ]
    assert [record["max_distance_source"] for record in records] == [
        "upstream",
        "job",
    ]


def test_openfold3_takes_a_query_pocket_constraint(tmp_path) -> None:
    from foldjax.models.openfold3.data.featurize import _query_set

    records: list = []
    path = _materialize(_job(tmp_path, _POCKET), "openfold3", records=records)
    document = json.loads(path.read_text())
    (query,) = document["queries"].values()
    assert query["pocket_constraint"] == {
        "ligand_chain_id": "L",
        "pocket_residues": [["A", 2], ["A", 5]],
        "max_distance": 4.0,
    }
    assert records[0]["max_distance_source"] == "upstream"
    # Upstream's own query schema accepts what was written.
    parsed = _query_set(document).queries["pocketed"].pocket_constraint
    assert parsed.ligand_chain_id == "L" and parsed.max_distance == 4.0
    # ... and the port featurizes it into upstream's pocket sampling features.
    from foldjax.models.openfold3.data import featurize_query

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        features = featurize_query(document)
    assert features["pocket_sampling_ligand_atom_mask"].sum() == 3
    assert features["pocket_sampling_contact_distance"].item() == 4.0


def test_openfold3_refuses_what_its_single_ligand_pocket_cannot_express(
    tmp_path,
) -> None:
    with pytest.raises(ValueError, match="more than one pocket"):
        _materialize(_job(tmp_path, _POCKET, _POCKET), "openfold3")
    polymer = {"binder": "B", "contacts": [["A", 2]]}
    with pytest.raises(ValueError, match="polymer binder"):
        _materialize(_job(tmp_path, polymer, binder_ligand=False), "openfold3")


def test_protenix_addresses_the_pocket_by_entity_and_copy(tmp_path) -> None:
    pocket = {**_POCKET, "max_distance": 6.5}
    records: list = []
    (native,) = json.loads(
        _materialize(_job(tmp_path, pocket), "protenix", records=records).read_text()
    )
    assert native["constraint"] == {
        "pocket": {
            "binder_chain": {"entity": 2, "copy": 1},
            "contact_residues": [
                {"entity": 1, "copy": 1, "position": 2},
                {"entity": 1, "copy": 1, "position": 5},
            ],
            "max_distance": 6.5,
        }
    }
    assert records[0]["max_distance_source"] == "job"
    # The port's Protenix featurizer turns it into the pocket channel: the
    # three binder tokens against residues 2 and 5 (tokens 1 and 4).
    import numpy as np

    from foldjax.models.protenix.data.featurize_json import featurize_protein_json

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        features = featurize_protein_json(
            native, base_dir=tmp_path / "out-protenix", seed=1
        )
    channel = np.asarray(_find(features, "constraint_feature")["pocket"])[..., 0]
    assert {tuple(index) for index in np.argwhere(channel)} == {
        (binder, contact) for binder in (9, 10, 11) for contact in (1, 4)
    }
    assert channel.max() == 6.5


def _find(tree, name):
    if isinstance(tree, dict):
        if name in tree:
            return tree[name]
        for value in tree.values():
            found = _find(value, name)
            if found is not None:
                return found
    return None


def test_protenix_has_no_default_distance(tmp_path) -> None:
    with pytest.raises(ValueError, match="without max_distance"):
        _materialize(_job(tmp_path, _POCKET), "protenix")


def test_opendde_drops_and_records_the_pocket_unless_refused(tmp_path) -> None:
    dropped: list = []
    records: list = []
    with pytest.warns(UserWarning, match="pocket constraint"):
        path = _materialize(
            _job(tmp_path, _POCKET), "opendde", records=records, dropped=dropped
        )
    (native,) = json.loads(path.read_text())
    assert "constraint" not in native
    assert records == []
    (record,) = dropped
    assert record["field"] == "constraints" and record["binders"] == ["L"]
    assert "ignored, as upstream does" in record["reason"]
    with pytest.raises(ValueError, match=f"{IGNORE_CONSTRAINTS}=false"):
        _materialize(
            _job(tmp_path, _POCKET),
            "opendde",
            options={IGNORE_CONSTRAINTS: False},
        )


@pytest.mark.parametrize("model", ["alphafold3", "esmfold2"])
def test_models_without_a_pocket_field_refuse_it(tmp_path, model) -> None:
    with pytest.raises(ValueError, match="no pocket or restraint field"):
        _materialize(_job(tmp_path, _POCKET), model)


@pytest.mark.parametrize(
    ("pocket", "match"),
    [
        ({"binder": "Z", "contacts": [["A", 2]]}, "binder is not a chain"),
        ({"binder": "L", "contacts": []}, "contacts must be a non-empty list"),
        ({"binder": "L", "contacts": [["Q", 2]]}, "unknown chain id"),
        ({"binder": "L", "contacts": [["A", 10]]}, "outside chain 'A' length 9"),
        ({"binder": "L", "contacts": [["A", 0]]}, "1-based"),
        ({"binder": "L", "contacts": [["A", True]]}, "must be an integer"),
        ({"binder": "L", "contacts": [["L", 1]]}, "is a ligand"),
        ({"binder": "A", "contacts": [["A", 2]]}, "on the binder chain"),
        ({**_POCKET, "max_distance": 0}, "positive and finite"),
        ({**_POCKET, "max_distance": "6"}, "must be a number"),
        ({**_POCKET, "force": True}, "pocket fields"),
    ],
)
def test_malformed_pockets_are_refused_for_every_model(tmp_path, pocket, match) -> None:
    with pytest.raises(ValueError, match=match):
        _materialize(_job(tmp_path, pocket), "boltz2")


def test_the_capability_record_moves_pockets_to_the_common_schema() -> None:
    for model in ("boltz2", "openfold3", "protenix"):
        assert "pocket_constraints" in common_schema_features(model), model
        assert "pocket_constraints" not in native_only_features(
            model, capabilities(model)
        )
    for model in ("alphafold3", "esmfold2", "opendde"):
        assert "pocket_constraints" not in common_schema_features(model), model
    assert "contact_constraints" in native_only_features(
        "boltz2", capabilities("boltz2")
    )


def test_job_builder_round_trips_pockets() -> None:
    document = {
        "name": "j",
        "entities": [
            {"type": "protein", "id": "A", "sequence": "ACDEF"},
            {"type": "ligand", "id": "L", "ccd": "ATP"},
        ],
        "constraints": [
            {"pocket": {"binder": "L", "contacts": [["A", 2]], "max_distance": 5.0}}
        ],
    }
    job = Job.from_document(document)
    assert job.pockets == (Pocket("L", (("A", 2),), max_distance=5.0),)
    assert job.to_document()["constraints"] == document["constraints"]
    with pytest.raises(ValueError, match="outside chain"):
        Job.from_document(
            {
                **document,
                "constraints": [{"pocket": {"binder": "L", "contacts": [["A", 9]]}}],
            }
        )


def _recorder(model: str, seen: list):
    base = provider(PORTS[model].backend)

    class Recorder(base):
        def predict(self, request):
            seen.append(request)
            path = request.output_dir / f"s{request.seed}.cif"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("data_mock\n#\n", encoding="utf-8")
            return PredictionResult(
                model=model,
                samples=(
                    PredictionSample(
                        seed=request.seed, structure_path=path, scores={"ptm": 0.5}
                    ),
                ),
                output_dir=request.output_dir,
            )

    return Recorder


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
                input=_job(tmp_path, _POCKET),
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
        assert record["max_distance"] == 6.0
        assert record["max_distance_source"] == "upstream"
        assert manifest["ignored_constraints"] is None
    else:
        assert manifest["constraints"] == []
        (record,) = manifest["ignored_constraints"]
        assert record["field"] == "constraints"


def test_results_and_compare_carry_the_constraints_record(tmp_path) -> None:
    from foldjax.compare import compare_rows
    from foldjax.results import load_results, results_table, to_csv
    from tests.test_output_contract import _run

    (tmp_path / "jobs").mkdir()
    job = _job(tmp_path / "jobs", _POCKET)
    root = tmp_path / "batch"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _run(tmp_path / "scratch", "boltz2_e9_8reh", out=root / "boltz2" / "p", job=job)
    (run,) = load_results(root).runs
    (record,) = run.constraints
    assert record["max_distance"] == 6.0
    assert record["max_distance_source"] == "upstream"
    (row,) = results_table(load_results(root))
    assert row["constraints"] == [dict(record)]
    assert "constraints" in to_csv([row]).splitlines()[0].split(",")
    (entry,) = compare_rows(root)["inputs"]
    (structure,) = entry["structures"]
    assert structure["constraints"] == [dict(record)]
