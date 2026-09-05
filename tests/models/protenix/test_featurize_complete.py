from __future__ import annotations

import hashlib
import os
import pickle
import threading
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from foldjax.models.protenix.data import featurize_json as featurize_impl
from foldjax.models.protenix.data.featurize_json import featurize_protein_json


def _job(*sequences, **extra):
    return {"name": "complete-input", "sequences": list(sequences), **extra}


def _ligand(value: str):
    return {"ligand": {"ligand": value, "count": 1, "id": ["L"]}}


def _use_official_ccd_assets(monkeypatch: pytest.MonkeyPatch) -> None:
    asset_root = Path(__file__).resolve().parents[4] / "protenix" / "common"
    components = asset_root / "components.cif"
    rdkit_cache = asset_root / "components.cif.rdkit_mol.pkl"
    if not components.is_file() or not rdkit_cache.is_file():
        pytest.skip("official components.cif/RDKit CCD assets are unavailable")
    monkeypatch.setenv("PROTENIX_CCD_COMPONENTS_FILE", str(components))
    monkeypatch.setenv("PROTENIX_CCD_RDKIT_MOL_FILE", str(rdkit_cache))


def _write_ligand_files(tmp_path: Path) -> dict[str, Path]:
    mol = Chem.AddHs(Chem.MolFromSmiles("F[C@H](Cl)Br"))
    assert AllChem.EmbedMolecule(mol, randomSeed=7) == 0
    mol = Chem.RemoveHs(mol)
    paths = {
        "mol": tmp_path / "stereo.mol",
        "sdf": tmp_path / "stereo.sdf",
        "pdb": tmp_path / "stereo.pdb",
        "mol2": tmp_path / "stereo.mol2",
    }
    Chem.MolToMolFile(mol, str(paths["mol"]))
    writer = Chem.SDWriter(str(paths["sdf"]))
    writer.write(mol)
    writer.close()
    Chem.MolToPDBFile(mol, str(paths["pdb"]))
    coords = mol.GetConformer().GetPositions()
    atoms = list(mol.GetAtoms())
    bonds = list(mol.GetBonds())
    mol2 = [
        "@<TRIPOS>MOLECULE",
        "stereo",
        f"{len(atoms)} {len(bonds)} 0 0 0",
        "SMALL",
        "USER_CHARGES",
        "",
        "@<TRIPOS>ATOM",
    ]
    for i, (atom, xyz) in enumerate(zip(atoms, coords), 1):
        mol2.append(
            f"{i:7d} {atom.GetSymbol()}{i} {xyz[0]:.6f} {xyz[1]:.6f} "
            f"{xyz[2]:.6f} {atom.GetSymbol()} 1 LIG {atom.GetFormalCharge():.4f}"
        )
    mol2.append("@<TRIPOS>BOND")
    for i, bond in enumerate(bonds, 1):
        order = {Chem.BondType.SINGLE: "1", Chem.BondType.DOUBLE: "2"}.get(
            bond.GetBondType(), "1"
        )
        mol2.append(
            f"{i:6d} {bond.GetBeginAtomIdx() + 1} {bond.GetEndAtomIdx() + 1} {order}"
        )
    paths["mol2"].write_text("\n".join(mol2) + "\n")
    return paths


def test_external_ccd_uses_official_atom_identity_and_leaving_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mol = Chem.AddHs(Chem.MolFromSmiles("C[O-]"))
    assert AllChem.EmbedMolecule(mol, randomSeed=11) == 0
    mol = Chem.RemoveHs(mol)
    components = tmp_path / "components.cif"
    components.write_text(
        """data_TST
#
loop_
_chem_comp_atom.comp_id
_chem_comp_atom.atom_id
_chem_comp_atom.type_symbol
_chem_comp_atom.charge
_chem_comp_atom.pdbx_leaving_atom_flag
TST CX C 0 N
TST "O'X" O -1 Y
TST HX H 0 N
#
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("PROTENIX_CCD_COMPONENTS_FILE", str(components))
    monkeypatch.setattr(featurize_impl, "_EXTERNAL_CCD_MOLS", {"TST": mol})

    entry = featurize_impl._external_ccd_component("TST")

    assert entry["names"].tolist() == ["CX", "O'X"]
    assert entry["elem"].tolist() == ["C", "O"]
    assert entry["charge"].tolist() == [0.0, -1.0]
    assert entry["leaving_atom_flag"].tolist() == [False, True]


def test_external_ccd_release_preserves_exact_feature_arrays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def assert_exact(left, right, name: str) -> None:
        if isinstance(left, dict):
            assert isinstance(right, dict) and left.keys() == right.keys(), name
            for child in left:
                assert_exact(left[child], right[child], f"{name}.{child}")
            return
        left_array = np.asarray(left)
        right_array = np.asarray(right)
        assert left_array.dtype == right_array.dtype, name
        assert left_array.shape == right_array.shape, name
        np.testing.assert_array_equal(left_array, right_array, err_msg=name)

    mol = Chem.AddHs(Chem.MolFromSmiles("C[O-]"))
    assert AllChem.EmbedMolecule(mol, randomSeed=11) == 0
    mol = Chem.RemoveHs(mol)
    components = tmp_path / "components.cif"
    components.write_text(
        """data_TST
#
loop_
_chem_comp_atom.comp_id
_chem_comp_atom.atom_id
_chem_comp_atom.type_symbol
_chem_comp_atom.charge
_chem_comp_atom.pdbx_leaving_atom_flag
TST CX C 0 N
TST OX O -1 Y
#
""",
        encoding="utf-8",
    )
    cache = tmp_path / "components.cif.rdkit_mol.pkl"
    cache.touch()
    monkeypatch.setenv("PROTENIX_CCD_COMPONENTS_FILE", str(components))
    monkeypatch.setenv("PROTENIX_CCD_RDKIT_MOL_FILE", str(cache))
    monkeypatch.setattr(featurize_impl, "_EXTERNAL_CCD_MOLS", {"TST": mol})
    monkeypatch.setattr(featurize_impl, "_EXTERNAL_CCD_ATOMS", OrderedDict())
    monkeypatch.setattr(
        featurize_impl,
        "_load_verified_rdkit_cache",
        lambda _path: {"TST": mol},
    )

    before = featurize_protein_json(_job(_ligand("CCD_TST")))
    assert featurize_impl._release_external_ccd_cache() is True
    after = featurize_protein_json(_job(_ligand("CCD_TST")))

    assert before.keys() == after.keys()
    for name in before:
        assert_exact(before[name], after[name], name)


def test_external_ccd_clear_waits_for_a_direct_load_and_keeps_the_local_mol(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mol = Chem.AddHs(Chem.MolFromSmiles("C[O-]"))
    assert AllChem.EmbedMolecule(mol, randomSeed=13) == 0
    mol = Chem.RemoveHs(mol)
    components = tmp_path / "components.cif"
    components.write_text(
        """data_TST
#
loop_
_chem_comp_atom.comp_id
_chem_comp_atom.atom_id
_chem_comp_atom.type_symbol
_chem_comp_atom.charge
_chem_comp_atom.pdbx_leaving_atom_flag
TST CX C 0 N
TST OX O -1 Y
#
""",
        encoding="utf-8",
    )
    cache = tmp_path / "components.cif.rdkit_mol.pkl"
    cache.touch()
    monkeypatch.setenv("PROTENIX_CCD_COMPONENTS_FILE", str(components))
    monkeypatch.setenv("PROTENIX_CCD_RDKIT_MOL_FILE", str(cache))
    monkeypatch.setattr(featurize_impl, "_EXTERNAL_CCD_MOLS", None)
    monkeypatch.setattr(featurize_impl, "_EXTERNAL_CCD_ATOMS", OrderedDict())
    load_started = threading.Event()
    finish_load = threading.Event()
    clear_finished = threading.Event()
    results = []
    errors = []

    def load(_path):
        load_started.set()
        if not finish_load.wait(5):
            raise TimeoutError("CCD load was not resumed")
        return {"TST": mol}

    def read_component() -> None:
        try:
            results.append(featurize_impl._external_ccd_component("TST"))
        except BaseException as error:
            errors.append(error)

    def clear_cache() -> None:
        try:
            results.append(featurize_impl._release_external_ccd_cache())
        except BaseException as error:
            errors.append(error)
        finally:
            clear_finished.set()

    monkeypatch.setattr(featurize_impl, "_load_verified_rdkit_cache", load)
    reader = threading.Thread(target=read_component)
    reader.start()
    assert load_started.wait(5)
    clearer = threading.Thread(target=clear_cache)
    clearer.start()
    assert not clear_finished.wait(0.05)
    finish_load.set()
    reader.join(5)
    clearer.join(5)

    assert not reader.is_alive() and not clearer.is_alive()
    assert errors == []
    component = next(value for value in results if isinstance(value, dict))
    assert component["names"].tolist() == ["CX", "OX"]
    assert True in results
    assert featurize_impl._EXTERNAL_CCD_MOLS is None
    featurize_impl._release_external_ccd_cache()


def test_external_ccd_rejects_unverified_rdkit_pickle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "components.cif.rdkit_mol.pkl"
    cache.write_bytes(b"not the pinned official cache")
    monkeypatch.setenv("PROTENIX_CCD_RDKIT_MOL_FILE", str(cache))
    monkeypatch.setattr(featurize_impl, "_EXTERNAL_CCD_MOLS", None)

    with pytest.raises(ValueError, match="SHA-256"):
        featurize_impl._external_ccd_component("TST")


def test_verified_rdkit_pickle_is_loaded_from_the_hashed_immutable_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "components.cif.rdkit_mol.pkl"
    payload = pickle.dumps({"sentinel": 7})
    cache.write_bytes(payload)
    monkeypatch.setattr(
        featurize_impl,
        "_TRUSTED_CCD_RDKIT_SHA256",
        frozenset({hashlib.sha256(payload).hexdigest()}),
    )
    monkeypatch.setattr(
        featurize_impl.pickle,
        "loads",
        lambda *_args, **_kwargs: pytest.fail(
            "the verified cache must not be retained as one in-memory bytes object"
        ),
    )

    assert featurize_impl._load_verified_rdkit_cache(cache) == {"sentinel": 7}


def test_ccd_block_lookup_binary_searches_sorted_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    components = tmp_path / "components.cif"
    components.write_text(
        "".join(f"data_{index:04d}\n_value {index}\n#\n" for index in range(1000)),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        featurize_impl,
        "_linear_ccd_block_bounds",
        lambda *_args, **_kwargs: pytest.fail(
            "a sorted CCD lookup must not scan from the beginning"
        ),
    )

    assert featurize_impl._read_ccd_block(components, "0999").startswith(
        "data_0999\n_value 999\n"
    )


def test_ccd_block_lookup_preserves_unsorted_custom_files(tmp_path: Path) -> None:
    components = tmp_path / "components.cif"
    components.write_text(
        "data_ZZZ\n_value last\n#\n"
        "data_AAA\n_value first\n#\n"
        "data_MMM\n_value middle\n#\n",
        encoding="utf-8",
    )

    assert featurize_impl._read_ccd_block(components, "AAA").startswith(
        "data_AAA\n_value first\n"
    )


def test_external_ccd_metadata_cache_is_lru_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    components = tmp_path / "components.cif"
    components.write_text("cache identity", encoding="utf-8")
    calls: list[str] = []
    monkeypatch.setenv("PROTENIX_CCD_COMPONENTS_FILE", str(components))
    monkeypatch.setattr(featurize_impl, "_EXTERNAL_CCD_ATOMS", OrderedDict())
    monkeypatch.setattr(featurize_impl, "_EXTERNAL_CCD_ATOM_CACHE_LIMIT", 3)
    monkeypatch.setattr(
        featurize_impl,
        "_read_ccd_block",
        lambda _path, code: calls.append(code) or f"block {code}",
    )
    monkeypatch.setattr(
        featurize_impl,
        "_parse_ccd_atom_metadata",
        lambda block, _code: {"block": block},
    )

    for code in ("AAA", "BBB", "CCC", "AAA", "DDD"):
        featurize_impl._external_ccd_atom_metadata(code)

    assert calls == ["AAA", "BBB", "CCC", "DDD"]
    assert [key[-1] for key in featurize_impl._EXTERNAL_CCD_ATOMS] == [
        "CCC",
        "AAA",
        "DDD",
    ]


def test_external_ccd_metadata_cache_tracks_file_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    components = tmp_path / "components.cif"
    components.write_text("first", encoding="utf-8")
    replacement = tmp_path / "replacement.cif"
    replacement.write_text("second", encoding="utf-8")
    monkeypatch.setenv("PROTENIX_CCD_COMPONENTS_FILE", str(components))
    monkeypatch.setattr(featurize_impl, "_EXTERNAL_CCD_ATOMS", OrderedDict())
    monkeypatch.setattr(
        featurize_impl,
        "_read_ccd_block",
        lambda path, _code: path.read_text(encoding="utf-8"),
    )
    monkeypatch.setattr(
        featurize_impl,
        "_parse_ccd_atom_metadata",
        lambda block, _code: {"block": block},
    )

    assert featurize_impl._external_ccd_atom_metadata("TST") == {"block": "first"}
    os.replace(replacement, components)
    assert featurize_impl._external_ccd_atom_metadata("TST") == {"block": "second"}
    assert len(featurize_impl._EXTERNAL_CCD_ATOMS) == 2


def test_smiles_preserves_charge_stereo_and_bond_graph():
    features = featurize_protein_json(_job(_ligand("C[C@H]([NH3+])C(=O)[O-]")))
    assert features["restype"].shape[0] == 6
    assert sorted(features["ref_charge"].tolist()) == [-1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    assert int(features["token_bonds"].sum()) == 10
    assert features["ligand_stereo"].max() > 0
    assert features["atom_input_index"].tolist() == list(range(6))
    assert 2.0 in features["chemical_bond_order"]
    assert features["chemical_bond_atom_indices"].shape == (5, 2)
    assert features["output_atom_name"].tolist() == ["C1", "C2", "N1", "C3", "O1", "O2"]
    assert set(features["output_atom_element"]) == {"C", "N", "O"}
    assert set(features["output_atom_res_name"]) == {"l01"}
    assert set(features["output_atom_chain_id"]) == {"L"}
    assert set(features["output_atom_res_id"]) == {1}


@pytest.mark.parametrize("suffix", ["mol", "sdf", "pdb", "mol2"])
def test_file_ligand_formats_require_and_keep_3d(tmp_path: Path, suffix: str):
    path = _write_ligand_files(tmp_path)[suffix]
    features = featurize_protein_json(
        _job(_ligand(f"FILE_{path.name}")), base_dir=tmp_path
    )
    assert features["restype"].shape[0] == 4
    assert features["ref_mask"].tolist() == [1.0] * 4
    assert np.ptp(features["ref_pos"], axis=0).max() > 0
    assert int(features["token_bonds"].sum()) == 6


def test_mse_is_normalized_to_standard_met(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _use_official_ccd_assets(monkeypatch)
    features = featurize_protein_json(
        _job(
            {
                "proteinChain": {
                    "sequence": "MA",
                    "modifications": [{"ptmType": "CCD_MSE", "ptmPosition": 1}],
                }
            },
        )
    )

    assert features["restype"].shape == (2, 32)
    np.testing.assert_array_equal(np.argmax(features["restype"], axis=-1), [12, 0])
    np.testing.assert_array_equal(features["token_is_modified"], [0, 0])
    np.testing.assert_array_equal(features["token_is_standard_polymer"], [1, 1])
    first = np.flatnonzero(features["atom_to_token_idx"] == 0)
    assert features["output_atom_name"][first].tolist() == [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "SD",
        "CE",
    ]
    assert features["output_atom_element"][first].tolist() == [
        "N",
        "C",
        "C",
        "O",
        "C",
        "C",
        "S",
        "C",
    ]
    assert set(features["output_atom_res_name"][first]) == {"MET"}
    np.testing.assert_array_equal(features["atom_to_tokatom_idx"][first], range(8))


def test_c_terminal_mse_uses_met_atom_slots_and_normalized_distogram_rep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _use_official_ccd_assets(monkeypatch)
    features = featurize_protein_json(
        _job(
            {
                "proteinChain": {
                    # The biological sequence deliberately says glycine: the
                    # normalized CCD identity, MET, must control the atom slot
                    # and representative atom after applying MSE.
                    "sequence": "AG",
                    "modifications": [{"ptmType": "CCD_MSE", "ptmPosition": 2}],
                }
            },
        )
    )

    second = np.flatnonzero(features["atom_to_token_idx"] == 1)
    assert features["output_atom_name"][second].tolist() == [
        "N",
        "CA",
        "C",
        "O",
        "OXT",
        "CB",
        "CG",
        "SD",
        "CE",
    ]
    np.testing.assert_array_equal(
        features["atom_to_tokatom_idx"][second], [0, 1, 2, 3, 8, 4, 5, 6, 7]
    )
    representative = second[features["distogram_rep_atom_mask"][second].astype(bool)]
    assert features["output_atom_name"][representative].tolist() == ["CB"]


@pytest.mark.parametrize(
    ("entity", "restype_index", "residue_name"),
    [
        (
            {
                "proteinChain": {
                    "sequence": "G",
                    "modifications": [{"ptmType": "CCD_MET", "ptmPosition": 1}],
                }
            },
            12,
            "MET",
        ),
        (
            {
                "dnaSequence": {
                    "sequence": "G",
                    "modifications": [
                        {"modificationType": "CCD_DA", "basePosition": 1}
                    ],
                }
            },
            26,
            "DA",
        ),
        (
            {
                "rnaSequence": {
                    "sequence": "C",
                    "modifications": [{"modificationType": "CCD_A", "basePosition": 1}],
                }
            },
            21,
            "A",
        ),
    ],
)
def test_explicit_standard_ccd_substitution_remains_a_residue_token(
    monkeypatch: pytest.MonkeyPatch,
    entity: dict[str, object],
    restype_index: int,
    residue_name: str,
) -> None:
    _use_official_ccd_assets(monkeypatch)
    features = featurize_protein_json(_job(entity))

    assert features["restype"].shape[0] == 1
    assert int(np.argmax(features["restype"][0])) == restype_index
    np.testing.assert_array_equal(features["token_is_standard_polymer"], [1])
    np.testing.assert_array_equal(features["token_is_modified"], [0])
    assert set(features["output_atom_res_name"]) == {residue_name}


def test_ligand_mse_normalizes_identity_but_remains_atom_tokenized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _use_official_ccd_assets(monkeypatch)
    features = featurize_protein_json(
        _job(
            {"proteinChain": {"sequence": "C"}},
            {"ligand": {"ligand": "CCD_MSE"}},
            covalent_bonds=[
                {
                    "entity1": 1,
                    "copy1": 1,
                    "position1": 1,
                    "atom1": "SG",
                    "entity2": 2,
                    "copy2": 1,
                    "position2": 1,
                    "atom2": "SE",
                }
            ],
        )
    )

    ligand_atoms = features["atom_entity_id"] == 2
    assert set(features["output_atom_res_name"][ligand_atoms]) == {"MET"}
    assert "SD" in features["output_atom_name"][ligand_atoms]
    assert "SE" not in features["output_atom_name"][ligand_atoms]
    sd = ligand_atoms & (features["output_atom_name"] == "SD")
    assert features["output_atom_element"][sd].tolist() == ["S"]
    assert features["token_polymer_type"][features["atom_to_token_idx"][sd][0]] == 0
    assert features["covalent_atom_indices"].shape == (1, 2)


def test_sep_atom_tokenization_bonds_and_biological_column_gather(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _use_official_ccd_assets(monkeypatch)
    template_path = tmp_path / "template.json"
    template_path.write_text("[]", encoding="utf-8")

    def fake_template_dense(_path, *, sequence: str, skip: bool):
        del skip
        width = len(sequence)
        aatype = np.tile(np.arange(width, dtype=np.int32), (4, 1))
        positions = np.zeros((4, width, 24, 3), dtype=np.float32)
        mask = np.zeros((4, width, 24), dtype=np.int32)
        return aatype, positions, mask

    monkeypatch.setattr(featurize_impl, "chain_template_dense", fake_template_dense)
    features = featurize_protein_json(
        _job(
            {
                "proteinChain": {
                    "sequence": "SA",
                    "templatesPath": str(template_path),
                    "modifications": [{"ptmType": "CCD_SEP", "ptmPosition": 1}],
                }
            }
        )
    )

    assert features["restype"].shape == (11, 32)
    np.testing.assert_array_equal(
        np.argmax(features["restype"], axis=-1), [15] * 10 + [0]
    )
    np.testing.assert_array_equal(features["residue_index"], [1] * 10 + [2])
    np.testing.assert_array_equal(
        features["atom_to_token_idx"], list(range(10)) + [10] * 6
    )
    np.testing.assert_array_equal(
        features["atom_to_tokatom_idx"], [0] * 10 + list(range(6))
    )
    np.testing.assert_array_equal(features["token_is_modified"], [1] * 10 + [0])
    np.testing.assert_array_equal(features["token_is_standard_polymer"], [0] * 10 + [1])
    expected_edges = {
        (0, 1),
        (1, 2),
        (1, 4),
        (2, 3),
        (3, 6),
        (4, 5),
        (6, 7),
        (6, 8),
        (6, 9),
    }
    actual_edges = {
        tuple(edge) for edge in np.argwhere(np.triu(features["token_bonds"], k=1) == 1)
    }
    assert actual_edges == expected_edges
    np.testing.assert_array_equal(
        features["msa"],
        np.tile([15] * 10 + [0], (features["msa"].shape[0], 1)),
    )
    np.testing.assert_array_equal(features["template_aatype"][0], [0] * 10 + [1])


def test_modified_nucleic_acids_are_atom_tokenized_with_terminal_rules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _use_official_ccd_assets(monkeypatch)
    dna = featurize_protein_json(
        _job(
            {
                "dnaSequence": {
                    "sequence": "AG",
                    "modifications": [
                        {"modificationType": "CCD_6MA", "basePosition": 1}
                    ],
                }
            }
        )
    )
    assert dna["restype"].shape == (24, 32)
    np.testing.assert_array_equal(np.argmax(dna["restype"], axis=-1), [26] * 23 + [27])
    first = np.flatnonzero(dna["atom_residue_index"] == 1)
    assert "OP3" in dna["output_atom_name"][first]

    rna = featurize_protein_json(
        _job(
            {
                "rnaSequence": {
                    "sequence": "AC",
                    "modifications": [
                        {"modificationType": "CCD_5MC", "basePosition": 2}
                    ],
                }
            }
        )
    )
    assert rna["restype"].shape == (22, 32)
    np.testing.assert_array_equal(np.argmax(rna["restype"], axis=-1), [21] + [23] * 21)
    second = np.flatnonzero(rna["atom_residue_index"] == 2)
    assert "OP3" not in rna["output_atom_name"][second]


@pytest.mark.parametrize("kind", ["dnaSequence", "rnaSequence"])
def test_standard_nucleotide_atom_slots_keep_canonical_op3_offset(kind: str) -> None:
    features = featurize_protein_json(_job({kind: {"sequence": "AA", "id": ["N"]}}))

    second = np.flatnonzero(features["atom_residue_index"] == 2)
    names = features["output_atom_name"][second]
    slots = features["atom_to_tokatom_idx"][second]
    code = "DA" if kind == "dnaSequence" else "A"
    canonical = featurize_impl._ccd_nucleotides()[code]["names"].astype(str)
    expected = [int(np.flatnonzero(canonical == name)[0]) for name in names]

    assert "OP3" not in names
    assert int(slots[names == "P"][0]) == 1
    np.testing.assert_array_equal(slots, expected)


def test_molecule_ids_follow_native_inference_without_merging_ligand_bonds() -> None:
    features = featurize_protein_json(
        _job(
            {"proteinChain": {"sequence": "C", "id": ["Z"]}},
            {"ion": {"ion": "MG", "id": ["M"]}},
            {"ligand": {"ligand": "[O:7]", "id": ["A"]}},
            covalent_bonds=[
                {
                    "entity1": 1,
                    "copy1": 1,
                    "position1": 1,
                    "atom1": "SG",
                    "entity2": 3,
                    "copy2": 1,
                    "position2": 1,
                    "atom2": 7,
                }
            ],
        )
    )

    chain_ids = features["output_atom_chain_id"]
    mol_ids = features["mol_id"]
    np.testing.assert_array_equal(np.unique(mol_ids[chain_ids == "A"]), [0])
    np.testing.assert_array_equal(np.unique(mol_ids[chain_ids == "Z"]), [2])
    np.testing.assert_array_equal(np.unique(mol_ids[chain_ids == "M"]), [1])
    assert features["covalent_atom_indices"].shape == (1, 2)


def _assert_modified_polymer_identity() -> None:
    features = featurize_protein_json(
        _job(
            {
                "proteinChain": {
                    "sequence": "AS",
                    "id": ["A"],
                    "modifications": [{"ptmPosition": 2, "ptmType": "CCD_SEP"}],
                }
            },
            {"ligand": {"ligand": "CCD_ATP", "id": ["L"]}},
        )
    )
    atoms = features["output_atom_chain_id"] == "L"
    np.testing.assert_array_equal(features["is_ligand"], atoms.astype(np.int64))
    np.testing.assert_array_equal(
        features["token_is_ligand"][features["atom_to_token_idx"]], atoms
    )
    assert np.any(features["token_is_modified"])
    assert atoms.any() and (~atoms).any()
    sep_atoms = features["output_atom_res_name"] == "SEP"
    assert sep_atoms.any()
    assert not np.any(features["is_ligand"][sep_atoms])


def test_modified_polymer_is_not_a_ligand_for_confidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A small synthetic phosphoserine fixture exercises the real CCD parser,
    # modification expansion and identity builder, without a publisher database.
    mol = Chem.AddHs(Chem.MolFromSmiles("N[C@@H](COP(=O)(O)O)C(=O)O"))
    assert AllChem.EmbedMolecule(mol, randomSeed=7) == 0
    mol = Chem.RemoveHs(mol)
    names = ["N", "CA", "CB", "OG", "P", "O1P", "O2P", "O3P", "C", "O", "OXT"]
    assert len(names) == mol.GetNumAtoms()
    rows = "\n".join(
        f"SEP {name} {atom.GetSymbol()} {'Y' if name == 'OXT' else 'N'}"
        for name, atom in zip(names, mol.GetAtoms(), strict=True)
    )
    components = tmp_path / "components.cif"
    components.write_text(
        "data_SEP\n_chem_comp.type 'L-PEPTIDE LINKING'\n"
        "_chem_comp.one_letter_code S\nloop_\n"
        "_chem_comp_atom.comp_id\n_chem_comp_atom.atom_id\n"
        "_chem_comp_atom.type_symbol\n_chem_comp_atom.pdbx_leaving_atom_flag\n"
        + rows + "\n#\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("PROTENIX_CCD_COMPONENTS_FILE", str(components))
    monkeypatch.setenv("PROTENIX_CCD_RDKIT_MOL_FILE", str(tmp_path / "absent.pkl"))
    monkeypatch.setattr(featurize_impl, "_EXTERNAL_CCD_MOLS", {"SEP": mol})
    monkeypatch.setattr(featurize_impl, "_EXTERNAL_CCD_ATOMS", OrderedDict())
    monkeypatch.setattr(
        featurize_impl, "_managed_ccd_asset", lambda name: tmp_path / "absent" / name
    )
    _assert_modified_polymer_identity()


@pytest.mark.official_parity
def test_modified_polymer_identity_with_official_ccd(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _use_official_ccd_assets(monkeypatch)
    _assert_modified_polymer_identity()


def test_polymer_modification_without_canonical_type_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _use_official_ccd_assets(monkeypatch)
    with pytest.raises(ValueError, match="canonical polymer type"):
        featurize_protein_json(
            _job(
                {
                    "proteinChain": {
                        "sequence": "A",
                        "modifications": [{"ptmType": "CCD_ACE", "ptmPosition": 1}],
                    }
                }
            )
        )


def test_covalent_bonds_add_token_bonds_and_atom_indices():
    features = featurize_protein_json(
        _job(
            {"proteinChain": {"sequence": "C", "count": 1}},
            _ligand("[C:7]O"),
            covalent_bonds=[
                {
                    "entity1": "1",
                    "copy1": 1,
                    "position1": "1",
                    "atom1": "SG",
                    "entity2": "2",
                    "copy2": 1,
                    "position2": "1",
                    "atom2": 7,
                }
            ],
        )
    )
    assert features["covalent_atom_indices"].shape == (1, 2)
    a, b = features["covalent_token_indices"][0]
    assert features["token_bonds"][a, b] == features["token_bonds"][b, a] == 1


def test_adjacent_standard_polymer_covalent_bond_is_not_a_token_bond() -> None:
    features = featurize_protein_json(
        _job(
            {"proteinChain": {"sequence": "AA"}},
            covalent_bonds=[
                {
                    "entity1": 1,
                    "copy1": 1,
                    "position1": 1,
                    "atom1": "C",
                    "entity2": 1,
                    "copy2": 1,
                    "position2": 2,
                    "atom2": "N",
                }
            ],
        )
    )

    assert features["covalent_atom_indices"].shape == (1, 2)
    np.testing.assert_array_equal(features["token_bonds"], np.zeros((2, 2)))


def test_conflicting_covalent_endpoint_aliases_are_rejected() -> None:
    with pytest.raises(ValueError, match="conflicting covalent bond aliases"):
        featurize_protein_json(
            _job(
                {"proteinChain": {"sequence": "C"}},
                _ligand("[O:7]"),
                covalent_bonds=[
                    {
                        "left_entity": 1,
                        "entity1": 2,
                        "left_copy": 1,
                        "copy1": 1,
                        "left_position": 1,
                        "position1": 1,
                        "left_atom": "SG",
                        "atom1": "SG",
                        "right_entity": 2,
                        "right_copy": 1,
                        "right_position": 1,
                        "right_atom": 7,
                    }
                ],
            )
        )


def test_identical_input_entities_stay_distinct_and_auto_ids_skip_reserved_ids():
    features = featurize_protein_json(
        _job(
            {"proteinChain": {"sequence": "C", "count": 1}},
            {"proteinChain": {"sequence": "C", "count": 1, "id": ["A"]}},
            {
                "proteinChain": {
                    "sequence": "C",
                    "count": 2,
                    "id": ["X", "Y"],
                }
            },
        )
    )

    # Input entries define entities in the upstream contract.  Identical
    # sequences are not merged; only copies within one entry share an entity.
    assert features["entity_id"].tolist() == [0, 1, 2, 2]
    assert features["sym_id"].tolist() == [0, 0, 0, 1]
    assert features["token_entity_id"].tolist() == [1, 2, 3, 3]
    assert list(dict.fromkeys(features["output_atom_chain_id"])) == [
        "B",
        "A",
        "X",
        "Y",
    ]
    assert list(dict.fromkeys(features["atom_entity_id"])) == [1, 2, 3]


def test_contact_and_pocket_constraint_features():
    features = featurize_protein_json(
        _job(
            {"proteinChain": {"sequence": "AC", "count": 1}},
            _ligand("CO"),
            constraint={
                "contact": [
                    {
                        "entity1": 1,
                        "copy1": 1,
                        "position1": 2,
                        "entity2": 2,
                        "copy2": 1,
                        "position2": 1,
                        "max_distance": 6,
                    },
                    {
                        "entity1": 1,
                        "copy1": 1,
                        "position1": 1,
                        "atom1": "CA",
                        "entity2": 2,
                        "copy2": 1,
                        "position2": 1,
                        "atom2": "O1",
                        "min_distance": 2,
                        "max_distance": 5,
                    },
                ],
                "pocket": {
                    "binder_chain": {"entity": 2, "copy": 1},
                    "contact_residues": [{"entity": 1, "copy": 1, "position": 2}],
                    "max_distance": 7,
                },
            },
        )
    )
    c = features["constraint_feature"]
    assert c["contact"].shape == (4, 4, 2)
    assert np.count_nonzero(c["contact"][..., 1] == 6) == 2
    assert np.count_nonzero(c["contact_atom"][..., 1] == 5) == 2
    assert np.count_nonzero(c["contact_atom"][..., 0] == 2) == 2
    assert np.count_nonzero(c["pocket"][..., 0] == 7) == 2


def test_polymer_atom_contacts_preserve_atoms_and_aggregate_token_interval():
    features = featurize_protein_json(
        _job(
            {"proteinChain": {"sequence": "C", "count": 1}},
            _ligand("CO"),
            constraint={
                "contact": [
                    {
                        "entity1": 1,
                        "copy1": 1,
                        "position1": 1,
                        "atom1": "CA",
                        "entity2": 2,
                        "copy2": 1,
                        "position2": 1,
                        "atom2": "O1",
                        "min_distance": 1,
                        "max_distance": 7,
                    },
                    {
                        "entity1": 1,
                        "copy1": 1,
                        "position1": 1,
                        "atom1": "SG",
                        "entity2": 2,
                        "copy2": 1,
                        "position2": 1,
                        "atom2": "O1",
                        "min_distance": 2,
                        "max_distance": 6,
                    },
                ]
            },
        )
    )
    atom = features["constraint_feature"]
    assert atom["contact_atom_index_pairs"].shape == (2, 2)
    assert atom["contact_atom_token_pairs"].tolist() == [[0, 2], [0, 2]]
    assert atom["contact_atom"][0, 2].tolist() == [2.0, 6.0]
    assert atom["contact_atom"][2, 0].tolist() == [2.0, 6.0]


def test_incompatible_atom_contacts_on_one_token_pair_fail():
    job = _job(
        {"proteinChain": {"sequence": "C"}},
        _ligand("O"),
        constraint={
            "contact": [
                {
                    "entity1": 1,
                    "copy1": 1,
                    "position1": 1,
                    "atom1": "CA",
                    "entity2": 2,
                    "copy2": 1,
                    "position2": 1,
                    "atom2": "O1",
                    "min_distance": 1,
                    "max_distance": 2,
                },
                {
                    "entity1": 1,
                    "copy1": 1,
                    "position1": 1,
                    "atom1": "SG",
                    "entity2": 2,
                    "copy2": 1,
                    "position2": 1,
                    "atom2": "O1",
                    "min_distance": 4,
                    "max_distance": 5,
                },
            ]
        },
    )
    with pytest.raises(ValueError, match="incompatible atom-contact"):
        featurize_protein_json(job)


def test_vendored_atp_covalent_bond_uses_official_empty_leaving_map(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _use_official_ccd_assets(monkeypatch)
    features = featurize_protein_json(
        _job(
            {"proteinChain": {"sequence": "C"}},
            {"ligand": {"ligand": "CCD_ATP"}},
            covalent_bonds=[
                {
                    "entity1": 1,
                    "copy1": 1,
                    "position1": 1,
                    "atom1": "SG",
                    "entity2": 2,
                    "copy2": 1,
                    "position2": 1,
                    "atom2": "PG",
                }
            ],
        )
    )

    assert features["covalent_atom_indices"].shape == (1, 2)
    ligand_atoms = features["output_atom_name"][features["atom_entity_id"] == 2]
    assert "PG" in ligand_atoms


def test_vendored_nag_covalent_bond_removes_official_o1_leaving_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _use_official_ccd_assets(monkeypatch)
    features = featurize_protein_json(
        _job(
            {"proteinChain": {"sequence": "C"}},
            {"ligand": {"ligand": "CCD_NAG"}},
            covalent_bonds=[
                {
                    "entity1": 1,
                    "copy1": 1,
                    "position1": 1,
                    "atom1": "SG",
                    "entity2": 2,
                    "copy2": 1,
                    "position2": 1,
                    "atom2": "C1",
                }
            ],
        )
    )

    ligand_atoms = features["output_atom_name"][features["atom_entity_id"] == 2]
    assert "C1" in ligand_atoms
    assert "O1" not in ligand_atoms
    assert features["covalent_atom_indices"].shape == (1, 2)


@pytest.mark.parametrize(
    ("polymer", "centre", "terminal_atom"),
    [
        ({"proteinChain": {"sequence": "A"}}, "C", "OXT"),
        ({"dnaSequence": {"sequence": "A"}}, "P", "OP3"),
    ],
)
def test_standard_polymer_covalent_bond_removes_terminal_leaving_atom(
    polymer: dict[str, object], centre: str, terminal_atom: str
) -> None:
    features = featurize_protein_json(
        _job(
            polymer,
            _ligand("[O:7]"),
            covalent_bonds=[
                {
                    "entity1": 1,
                    "copy1": 1,
                    "position1": 1,
                    "atom1": centre,
                    "entity2": 2,
                    "copy2": 1,
                    "position2": 1,
                    "atom2": 7,
                }
            ],
        )
    )

    polymer_atoms = features["output_atom_name"][features["atom_entity_id"] == 1]
    assert terminal_atom not in polymer_atoms


def test_one_covalent_bond_removes_only_one_independent_leaving_group(monkeypatch):
    from foldjax.models.protenix.data import featurize_json as module

    monkeypatch.setattr(
        module,
        "_LIGAND_TABLE",
        {
            "TST": {
                "names": np.array(["C1", "O1", "O2", "N1"]),
                "coord": np.array(
                    [[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0]], np.float32
                ),
                "charge": np.zeros(4, np.float32),
                "mask": np.ones(4, np.float32),
                "elem": np.array(["C", "O", "O", "N"]),
                "bonds": np.array([[0, 1], [0, 2], [0, 3]], np.int64),
                "leaving_atom_flag": np.array([False, True, True, False]),
            }
        },
    )
    features = featurize_protein_json(
        _job(
            {"proteinChain": {"sequence": "C"}},
            {"ligand": {"ligand": "CCD_TST"}},
            covalent_bonds=[
                {
                    "entity1": 1,
                    "copy1": 1,
                    "position1": 1,
                    "atom1": "SG",
                    "entity2": 2,
                    "copy2": 1,
                    "position2": 1,
                    "atom2": "C1",
                }
            ],
        ),
        seed=0,
    )

    ligand_atoms = features["output_atom_name"][features["atom_entity_id"] == 2]
    assert ligand_atoms.tolist() == ["C1", "O1", "N1"]


def test_seeded_leaving_group_selection_is_independent_for_each_copy(monkeypatch):
    from foldjax.models.protenix.data import featurize_json as module

    monkeypatch.setattr(
        module,
        "_LIGAND_TABLE",
        {
            "TST": {
                "names": np.array(["C1", "O1", "O2", "N1"]),
                "coord": np.array(
                    [[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0]], np.float32
                ),
                "charge": np.zeros(4, np.float32),
                "mask": np.ones(4, np.float32),
                "elem": np.array(["C", "O", "O", "N"]),
                "bonds": np.array([[0, 1], [0, 2], [0, 3]], np.int64),
                "leaving_atom_flag": np.array([False, True, True, False]),
            }
        },
    )
    features = featurize_protein_json(
        _job(
            {"proteinChain": {"sequence": "C", "count": 2}},
            {"ligand": {"ligand": "CCD_TST", "count": 2}},
            covalent_bonds=[
                {
                    "entity1": 1,
                    "position1": 1,
                    "atom1": "SG",
                    "entity2": 2,
                    "position2": 1,
                    "atom2": "C1",
                }
            ],
        ),
        seed=101,
    )

    ligand = features["atom_entity_id"] == 2
    names = features["output_atom_name"]
    copies = features["atom_copy_id"]
    assert names[ligand & (copies == 1)].tolist() == ["C1", "O2", "N1"]
    assert names[ligand & (copies == 2)].tolist() == ["C1", "O1", "N1"]


@pytest.mark.parametrize(
    ("length", "retained"),
    [
        (1, ["O1", "O1", "O2"]),
        (2, ["O1", "O2", "O1"]),
        (3, ["O2", "O1", "O1"]),
        (4, ["O1", "O1", "O1"]),
    ],
)
def test_polymer_link_rng_prefix_matches_upstream_group_identity(
    monkeypatch: pytest.MonkeyPatch, length: int, retained: list[str]
) -> None:
    from foldjax.models.protenix.data import featurize_json as module

    monkeypatch.setattr(
        module,
        "_LIGAND_TABLE",
        {
            "TST": {
                "names": np.array(["C1", "O1", "O2", "N1"]),
                "coord": np.array(
                    [[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0]], np.float32
                ),
                "charge": np.zeros(4, np.float32),
                "mask": np.ones(4, np.float32),
                "elem": np.array(["C", "O", "O", "N"]),
                "bonds": np.array([[0, 1], [0, 2], [0, 3]], np.int64),
                "leaving_atom_flag": np.array([False, True, True, False]),
            }
        },
    )
    features = featurize_protein_json(
        _job(
            {"proteinChain": {"sequence": "A" * (length - 1) + "C", "count": 3}},
            {"ligand": {"ligand": "CCD_TST", "count": 3}},
            covalent_bonds=[
                {
                    "entity1": 1,
                    "position1": length,
                    "atom1": "SG",
                    "entity2": 2,
                    "position2": 1,
                    "atom2": "C1",
                }
            ],
        ),
        seed=0,
    )

    observed = []
    for copy in (1, 2, 3):
        copy_atoms = (features["atom_entity_id"] == 2) & (
            features["atom_copy_id"] == copy
        )
        observed.append(
            next(
                name
                for name in features["output_atom_name"][copy_atoms]
                if name in {"O1", "O2"}
            )
        )
    assert observed == retained


def test_covalent_ccd_removes_metadata_defined_leaving_group(monkeypatch):
    from foldjax.models.protenix.data import featurize_json as module

    monkeypatch.setattr(
        module,
        "_LIGAND_TABLE",
        {
            "TST": {
                "names": np.array(["C1", "O1", "N1"]),
                "coord": np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0]], np.float32),
                "charge": np.zeros(3, np.float32),
                "mask": np.ones(3, np.float32),
                "elem": np.array(["C", "O", "N"]),
                "bonds": np.array([[0, 1], [0, 2]], np.int64),
                "leaving_atom_flag": np.array([False, True, False]),
            }
        },
    )
    features = featurize_protein_json(
        _job(
            {"proteinChain": {"sequence": "C"}},
            {"ligand": {"ligand": "CCD_TST"}},
            covalent_bonds=[
                {
                    "entity1": 1,
                    "copy1": 1,
                    "position1": 1,
                    "atom1": "SG",
                    "entity2": 2,
                    "copy2": 1,
                    "position2": 1,
                    "atom2": "C1",
                }
            ],
        )
    )
    ligand_atoms = features["output_atom_name"][features["atom_entity_id"] == 2]
    assert ligand_atoms.tolist() == ["C1", "N1"]
    assert features["covalent_atom_indices"].shape == (1, 2)


@pytest.mark.parametrize(
    "job,match",
    [
        (_job(_ligand("this is not smiles")), "invalid SMILES"),
        (_job(_ligand("FILE_missing.sdf")), "does not exist"),
        (
            _job(
                {
                    "proteinChain": {
                        "sequence": "A",
                        "modifications": [{"ptmType": "CCD_ACE", "ptmPosition": 0}],
                    }
                }
            ),
            "position",
        ),
        (
            {
                "name": "x",
                "sequences": [{"proteinChain": {"sequence": "A"}}],
                "mystery": 1,
            },
            "unsupported top-level",
        ),
    ],
)
def test_invalid_or_unhandled_inputs_fail_explicitly(job, match):
    with pytest.raises((ValueError, FileNotFoundError), match=match):
        featurize_protein_json(job)
