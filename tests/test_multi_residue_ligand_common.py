"""The common ``ccd`` list: one ligand chain of several CCD components.

A glycan is ``ccd: [NAG, NAG, BMA]``, residue i being code i - 1, and the
common ``bonds`` address its residues 1, 2, ... as AlphaFold 3's
``bondedAtomPairs`` do. Each writer emits its upstream's own form: AlphaFold 3
``ccdCodes``, Boltz-2 a ``ccd`` list, Protenix/OpenDDE ``CCD_A_B`` with
``covalent_bonds`` positions numbering the codes from 1, ESMFold2 the list
itself. OpenFold3 v0.5.0 raises NotImplementedError for more than one code,
so it refuses one here.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from foldjax import Job, Ligand, Protein
from foldjax.input import (
    common_schema_features,
    materialize_native_input,
    native_only_features,
)
from foldjax.registry import capabilities

_GLYCAN = ["NAG", "NAG", "BMA"]
_BONDS = [
    [["A", 3, "ND2"], ["G", 1, "C1"]],
    [["G", 1, "O4"], ["G", 2, "C1"]],
    [["G", 2, "O4"], ["G", 3, "C1"]],
]


def _job(tmp_path: Path, ccd=None, *, bonds=_BONDS) -> Path:
    document = {
        "name": "glyco",
        "entities": [
            {"type": "protein", "id": "A", "sequence": "MKNGST"},
            {"type": "ligand", "id": "G", "ccd": _GLYCAN if ccd is None else ccd},
        ],
    }
    if bonds:
        document["bonds"] = bonds
    path = tmp_path / "job.json"
    path.write_text(json.dumps(document))
    return path


def _native(source: Path, model: str):
    path = materialize_native_input(
        source,
        capabilities(model),
        source.parent / f"out-{model}",
        seed=1,
        msa="single",
    )
    return json.loads(path.read_text())


def test_alphafold3_lists_the_codes_and_keeps_the_residue_numbering(
    tmp_path,
) -> None:
    native = _native(_job(tmp_path), "alphafold3")
    assert native["sequences"][1] == {"ligand": {"id": ["G"], "ccdCodes": _GLYCAN}}
    assert native["bondedAtomPairs"] == _BONDS


def test_boltz2_takes_a_ccd_list_and_bond_constraints(tmp_path) -> None:
    native = _native(_job(tmp_path), "boltz2")
    assert native["sequences"][1] == {"ligand": {"id": ["G"], "ccd": _GLYCAN}}
    assert native["constraints"] == [
        {"bond": {"atom1": left, "atom2": right}} for left, right in _BONDS
    ]


@pytest.mark.parametrize("model", ["protenix", "opendde"])
def test_protenix_dialect_joins_codes_and_numbers_them_from_one(
    tmp_path, model
) -> None:
    (native,) = _native(_job(tmp_path), model)
    assert native["sequences"][1]["ligand"]["ligand"] == "CCD_NAG_NAG_BMA"
    # build_ligand gives code i res_id i + 1; covalent_bonds' position is it.
    assert [
        (
            bond["entity1"],
            bond["position1"],
            bond["atom1"],
            bond["entity2"],
            bond["position2"],
            bond["atom2"],
        )
        for bond in native["covalent_bonds"]
    ] == [
        (1, 3, "ND2", 2, 1, "C1"),
        (2, 1, "O4", 2, 2, "C1"),
        (2, 2, "O4", 2, 3, "C1"),
    ]


def _protenix_glycan(tmp_path: Path, bonds):
    """The port's featurizer on the glycan job, or a skip without the CCD.

    Leaving atoms come from the full CCD, components.cif: the asset store's
    copy, or the file ``PROTENIX_CCD_COMPONENTS_FILE`` names (as for a run).
    """
    import warnings

    import numpy as np

    from foldjax.models.protenix.data.featurize_json import featurize_protein_json

    (native,) = _native(_job(tmp_path, bonds=bonds), "protenix")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            features = featurize_protein_json(
                native, base_dir=tmp_path / "out-protenix", seed=1
            )
    except ValueError as error:
        if "requires components.cif" in str(error):
            pytest.skip(
                "the Protenix components.cif asset is not available; set "
                "PROTENIX_CCD_COMPONENTS_FILE"
            )
        raise
    asym = np.asarray(features["asym_id"])
    residue = np.asarray(features["residue_index"])
    names = np.asarray(features["ref_atom_name_chars"]).reshape(-1, 4, 64)
    atoms: dict[int, list[str]] = {}
    tokens: dict[tuple[int, str], int] = {}
    for atom, token in enumerate(np.asarray(features["atom_to_token_idx"])):
        if asym[token] != asym.max():
            continue
        name = "".join(chr(int(c) + 32) for c in names[atom].argmax(-1)).strip()
        atoms.setdefault(int(residue[token]), []).append(name)
        tokens.setdefault((int(residue[token]), name), int(token))
    return features, atoms, tokens


def test_protenix_featurizer_builds_the_glycan_and_its_inter_residue_bonds(
    tmp_path,
) -> None:
    """The port's own reader, representative-atom check included.

    NAG has 15 heavy atoms and BMA 12. A covalent bond at C1 removes that
    residue's leaving O1 (upstream ``remove_leaving_atoms``), so the N-linked
    chain NAG-NAG-BMA, bonded at every C1, keeps 14/14/11; unbonded, 15/15/12.
    """
    import numpy as np

    features, atoms, tokens = _protenix_glycan(tmp_path, _BONDS)
    assert {key: len(value) for key, value in atoms.items()} == {1: 14, 2: 14, 3: 11}
    assert all("O1" not in names for names in atoms.values())
    bonds = np.asarray(features["token_bonds"])
    assert bonds[tokens[(1, "O4")], tokens[(2, "C1")]] == 1
    assert bonds[tokens[(2, "O4")], tokens[(3, "C1")]] == 1

    (tmp_path / "free").mkdir()
    _, atoms, _ = _protenix_glycan(tmp_path / "free", None)
    assert {key: len(value) for key, value in atoms.items()} == {1: 15, 2: 15, 3: 12}
    assert all("O1" in names for names in atoms.values())


@pytest.mark.parametrize(
    "model", ["alphafold3", "boltz2", "esmfold2", "opendde", "protenix"]
)
def test_a_bond_counts_one_residue_per_code(tmp_path, model) -> None:
    past = [*_BONDS, [["G", 3, "O4"], ["G", 4, "C1"]]]
    with pytest.raises(ValueError, match="outside chain 'G', which has 3 residue"):
        _native(_job(tmp_path, bonds=past), model)


def test_boltz2_refuses_affinity_for_a_multi_residue_ligand(tmp_path) -> None:
    source = _job(tmp_path, bonds=None)
    document = json.loads(source.read_text())
    document["properties"] = [{"affinity": {"binder": "G"}}]
    source.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="affinity for the multi-residue ligand 'G'"):
        _native(source, "boltz2")
    # A one-code list is a single residue, which upstream scores.
    document["entities"][1]["ccd"] = ["NAG"]
    source.write_text(json.dumps(document))
    assert _native(source, "boltz2")["properties"] == [{"affinity": {"binder": "G"}}]


def test_esmfold2_reads_the_list_itself(tmp_path) -> None:
    native = _native(_job(tmp_path), "esmfold2")
    assert native["entities"][1]["ccd"] == _GLYCAN
    assert native["bonds"] == _BONDS


def test_openfold3_refuses_more_than_one_code(tmp_path) -> None:
    with pytest.raises(ValueError, match="several CCD codes.*NotImplementedError"):
        _native(_job(tmp_path, bonds=None), "openfold3")
    # One code in a list is the single-code ligand upstream does build.
    native = _native(_job(tmp_path, ["NAG"], bonds=None), "openfold3")
    (query,) = native["queries"].values()
    assert query["chains"][1]["ccd_codes"] == ["NAG"]


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("alphafold3", {"ligand": {"id": ["G"], "ccdCodes": ["NAG"]}}),
        ("boltz2", {"ligand": {"id": ["G"], "ccd": "NAG"}}),
    ],
)
def test_a_single_code_string_is_written_as_before(tmp_path, model, expected) -> None:
    native = _native(_job(tmp_path, "NAG", bonds=None), model)
    assert native["sequences"][1] == expected


@pytest.mark.parametrize(
    ("ccd", "match"),
    [
        ([], "one or more non-empty CCD codes"),
        (["NAG", ""], "one or more non-empty CCD codes"),
        (["NAG", 7], "one or more non-empty CCD codes"),
        (123, "ligand ccd must be a non-empty string"),
    ],
)
def test_malformed_ccd_lists_are_refused(tmp_path, ccd, match) -> None:
    with pytest.raises(ValueError, match=match):
        _native(_job(tmp_path, ccd, bonds=None), "alphafold3")


@pytest.mark.parametrize("model", ["protenix", "opendde"])
def test_protenix_dialect_refuses_a_code_its_separator_would_split(
    tmp_path, model
) -> None:
    with pytest.raises(ValueError, match="containing '_'"):
        _native(_job(tmp_path, ["NAG", "A_B"], bonds=None), model)


def test_boltz2_validates_every_code_in_the_list(tmp_path) -> None:
    with pytest.raises(ValueError, match="ligand CCD code"):
        _native(_job(tmp_path, ["NAG", "../x"], bonds=None), "boltz2")


def test_the_capability_record_moves_multi_residue_ligands_to_the_common_schema() -> (
    None
):
    for model in ("alphafold3", "boltz2", "esmfold2", "opendde", "protenix"):
        assert "multi_residue_ligand" in common_schema_features(model), model
        assert "multi_residue_ligand" not in native_only_features(
            model, capabilities(model)
        )
    assert "multi_residue_ligand" not in common_schema_features("openfold3")
    assert "multi_residue_ligand" not in native_only_features(
        "openfold3", capabilities("openfold3")
    )


def test_job_builder_round_trips_a_ccd_list() -> None:
    job = Job(
        "glyco",
        [Protein("A", "MKNGST"), Ligand("G", ccd=("NAG", "NAG", "BMA"))],
    )
    document = job.to_document()
    assert document["entities"][1]["ccd"] == _GLYCAN
    assert Job.from_document(document) == job
    assert Ligand("G", ccd=["NAG", "BMA"]).ccd == ("NAG", "BMA")
    with pytest.raises(ValueError, match="one or more non-empty CCD codes"):
        Job.from_document(
            {"entities": [{"type": "ligand", "id": "G", "ccd": ["NAG", None]}]}
        )
