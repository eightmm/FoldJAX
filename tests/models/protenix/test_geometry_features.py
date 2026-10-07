from __future__ import annotations

from itertools import combinations

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem.rdDistGeom import GetExperimentalTorsions, GetMoleculeBoundsMatrix

from foldjax.models.protenix.data.geometry import prepare_tfg_features
from foldjax.models.protenix.tfg.config import parse_tfg_config, validate_features


def _input_ligand(n_atom: int, res_name: str = "l01") -> dict[str, np.ndarray]:
    """Per-atom annotations of one SMILES/FILE_ ligand residue."""

    return {
        "output_atom_name": np.asarray([f"C{i + 1}" for i in range(n_atom)]),
        "output_atom_res_name": np.full((n_atom,), res_name),
        "output_atom_res_id": np.ones((n_atom,), dtype=np.int64),
        "output_atom_chain_id": np.full((n_atom,), "B"),
        "output_atom_polymer_type": np.full((n_atom,), "non-polymer"),
    }


def _features(
    coords: np.ndarray,
    bonds: list[tuple[int, int]] | None = None,
    orders: list[float] | None = None,
    stereos: list[int] | None = None,
) -> dict[str, np.ndarray]:
    n_atom = len(coords)
    elements = np.zeros((n_atom, 128), dtype=np.float32)
    elements[:, 5] = 1.0  # carbon
    bond_array = np.asarray(bonds or [], dtype=np.int64).reshape((-1, 2))
    return {
        "ref_pos": np.asarray(coords, dtype=np.float32),
        "ref_element": elements,
        "atom_to_token_idx": np.arange(n_atom, dtype=np.int64),
        "asym_id": np.zeros((n_atom,), dtype=np.int64),
        "chemical_bond_atom_indices": bond_array,
        "chemical_bond_order": np.asarray(
            orders if orders is not None else np.ones(len(bond_array)),
            dtype=np.float32,
        ),
        "chemical_bond_stereo": np.asarray(
            stereos if stereos is not None else np.zeros(len(bond_array)),
            dtype=np.int64,
        ),
        "ligand_stereo": np.zeros((n_atom,), dtype=np.int64),
        "covalent_atom_indices": np.empty((0, 2), dtype=np.int64),
        **_input_ligand(n_atom),
    }


def test_prepare_tfg_features_shapes_and_all_term_validation() -> None:
    features = _features(
        np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]], dtype=np.float32),
        bonds=[(0, 1), (1, 2)],
    )

    result = prepare_tfg_features(features)

    expected_shapes = {
        "interchain_bond_index": (2, 0),
        "pairwise_distance_index": (2, 3),
        "pairwise_distance_upper_bound": (3,),
        "pairwise_distance_lower_bound": (3,),
        "pairwise_distance_is_bond": (3,),
        "pairwise_distance_is_angle": (3,),
        "experimental_torsion_index": (4, 0),
        "experimental_torsion_force_constant": (0, 6),
        "experimental_torsion_sign": (0, 6),
        "linear_triple_bond_index": (3, 0),
        "chiral_index": (4, 0),
        "chiral_orientation": (0,),
        "stereo_bond_index": (4, 0),
        "stereo_bond_orientation": (0,),
        "planar_improper_index": (4, 0),
        "planar_improper_is_carbonyl": (0,),
    }
    for key, shape in expected_shapes.items():
        assert result[key].shape == shape
    assert result["pairwise_distance_index"].dtype == np.int64
    assert result["pairwise_distance_is_bond"].dtype == np.int64
    assert result["pairwise_distance_lower_bound"].dtype == np.float32
    assert result["experimental_torsion_force_constant"].dtype == np.float32

    cfg = parse_tfg_config(
        {
            "enable": True,
            "terms": {
                name: {"weight": 1.0}
                for name in (
                    "InterchainBondPotential",
                    "PairwiseDistancePotential",
                    "StereoBondPotential",
                    "ChiralAtomPotential",
                    "PlanarImproperPotential",
                    "LinearBondPotential",
                    "ExperimentalTorsionPotential",
                    "VinaStericPotential",
                )
            },
        }
    )
    validate_features(result, cfg.terms)


def test_interchain_bonds_filter_same_chain_covalent_pairs() -> None:
    features = _features(np.zeros((4, 3), dtype=np.float32))
    features["asym_id"] = np.array([0, 1], dtype=np.int64)
    features["atom_to_token_idx"] = np.array([0, 0, 1, 1], dtype=np.int64)
    features["covalent_atom_indices"] = np.array(
        [[0, 2], [0, 1], [2, 3]], dtype=np.int64
    )

    result = prepare_tfg_features(features)

    np.testing.assert_array_equal(
        result["interchain_bond_index"], np.array([[0], [2]], dtype=np.int64)
    )


def test_double_triple_planar_and_chiral_annotations_from_graph_and_reference() -> None:
    stereo = _features(
        np.array([[0, 1, 0], [0, 0, 0], [1, 0, 0], [1, -1, 0]], np.float32),
        bonds=[(0, 1), (1, 2), (2, 3)],
        orders=[1.0, 2.0, 1.0],
        stereos=[0, 3, 0],
    )
    stereo_result = prepare_tfg_features(stereo)
    np.testing.assert_array_equal(
        stereo_result["stereo_bond_index"],
        np.array([[0], [1], [2], [3]], dtype=np.int64),
    )
    assert stereo_result["stereo_bond_orientation"].shape == (1,)

    triple = _features(
        np.array([[-1, 0, 0], [0, 0, 0], [1, 0, 0], [2, 0, 0]], np.float32),
        bonds=[(0, 1), (1, 2), (2, 3)],
        orders=[1.0, 3.0, 1.0],
    )
    triple_result = prepare_tfg_features(triple)
    np.testing.assert_array_equal(
        triple_result["linear_triple_bond_index"],
        np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int64),
    )

    planar = _features(
        np.array([[0, 1, 0], [0, 0, 0], [1, 0, 0], [-1, -1, 0]], np.float32),
        bonds=[(1, 0), (1, 2), (1, 3)],
        orders=[2.0, 1.0, 1.0],
    )
    planar["output_atom_element"] = np.array(["O", "C", "C", "C"])
    planar_result = prepare_tfg_features(planar)
    assert planar_result["planar_improper_index"].shape == (4, 3)
    np.testing.assert_array_equal(
        planar_result["planar_improper_is_carbonyl"], np.ones(3, np.float32)
    )

    chiral = _features(
        np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32),
        bonds=[(0, 1), (0, 2), (0, 3)],
    )
    chiral["output_atom_element"] = np.asarray(["C", "F", "Cl", "Br"])
    chiral["ligand_stereo"][0] = 1
    chiral_result = prepare_tfg_features(chiral)
    assert chiral_result["chiral_index"].shape == (4, 1)
    assert abs(float(chiral_result["chiral_orientation"][0])) == 1.0


def _polymer(res_names: list[str], atoms_per_residue: int = 3) -> dict[str, np.ndarray]:
    """A bonded chain of standard residues, as the featurizer annotates it."""

    n_atom = len(res_names) * atoms_per_residue
    coords = np.stack([[1.4 * i, float(i % 2), 0.0] for i in range(n_atom)]).astype(
        np.float32
    )
    features = _features(coords, bonds=[(i, i + 1) for i in range(n_atom - 1)])
    features.update(
        {
            "output_atom_name": np.asarray(["N", "CA", "C"] * len(res_names)),
            "output_atom_res_name": np.repeat(res_names, atoms_per_residue),
            "output_atom_res_id": np.repeat(
                np.arange(1, len(res_names) + 1), atoms_per_residue
            ),
            "output_atom_chain_id": np.full((n_atom,), "A"),
            "output_atom_polymer_type": np.full((n_atom,), "polypeptide(L)"),
        }
    )
    return features


def _no_ccd_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(code, **_kwargs):
        raise AssertionError(f"the CCD cache was opened for {code!r}")

    monkeypatch.setattr(
        "foldjax.models.protenix.data.featurize_json._external_ccd_molecule", refuse
    )


def test_standard_polymer_residues_get_no_geometry(monkeypatch) -> None:
    """Upstream's `exclude_std_residue=True`: no constraint, no CCD lookup."""

    _no_ccd_cache(monkeypatch)
    result = prepare_tfg_features(_polymer(["ALA", "GLY", "SER"]))

    for key in (
        "pairwise_distance_index",
        "experimental_torsion_index",
        "linear_triple_bond_index",
        "chiral_index",
        "stereo_bond_index",
        "planar_improper_index",
    ):
        assert result[key].shape[-1] == 0, key
    assert result["experimental_torsion_force_constant"].shape == (0, 6)
    assert result["geometry_provenance"]["residues"] == []


def test_a_metal_component_is_skipped_without_its_reference(monkeypatch) -> None:
    _no_ccd_cache(monkeypatch)
    features = _features(np.zeros((1, 3), dtype=np.float32))
    features.update(_input_ligand(1, res_name="ZN"))
    features["output_atom_element"] = np.asarray(["ZN"])

    result = prepare_tfg_features(features)

    assert result["pairwise_distance_index"].shape == (2, 0)
    (record,) = result["geometry_provenance"]["residues"]
    assert (record["res_name"], record["status"]) == ("ZN", "metal")


def test_mse_is_methionine_even_as_a_ligand(monkeypatch) -> None:
    """Upstream's `mse_to_met` renames every MSE a non-hetero MET first."""

    _no_ccd_cache(monkeypatch)
    features = _features(np.zeros((2, 3), dtype=np.float32), bonds=[(0, 1)])
    features.update(_input_ligand(2, res_name="MSE"))

    result = prepare_tfg_features(features)

    assert result["pairwise_distance_index"].shape == (2, 0)
    assert result["geometry_provenance"]["residues"] == []


def test_a_ccd_component_without_the_cache_names_it(monkeypatch) -> None:
    def absent(code, **_kwargs):
        raise ValueError(f"CCD code {code!r} is not vendored; set ...")

    monkeypatch.setattr(
        "foldjax.models.protenix.data.featurize_json._external_ccd_molecule", absent
    )
    features = _features(np.zeros((2, 3), dtype=np.float32), bonds=[(0, 1)])
    features.update(_input_ligand(2, res_name="ATP"))

    with pytest.raises(ValueError, match="'ATP' needs the official components.cif"):
        prepare_tfg_features(features)


def test_missing_residue_annotations_are_refused() -> None:
    features = _features(np.zeros((2, 3), dtype=np.float32), bonds=[(0, 1)])
    del features["output_atom_res_name"]

    with pytest.raises(ValueError, match="output_atom_res_name"):
        prepare_tfg_features(features)


def _features_from_rdkit(mol: Chem.Mol) -> dict[str, np.ndarray]:
    n_atom = mol.GetNumAtoms()
    coords = np.stack(
        [[float(i), float(i % 2), float((i * 2) % 3)] for i in range(n_atom)]
    ).astype(np.float32)
    elements = np.zeros((n_atom, 128), dtype=np.float32)
    for atom in mol.GetAtoms():
        elements[atom.GetIdx(), atom.GetAtomicNum() - 1] = 1.0
    bonds = [(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) for bond in mol.GetBonds()]
    return {
        "ref_pos": coords,
        "ref_element": elements,
        "output_atom_element": np.asarray(
            [atom.GetSymbol() for atom in mol.GetAtoms()]
        ),
        "ref_charge": np.asarray(
            [atom.GetFormalCharge() for atom in mol.GetAtoms()], dtype=np.float32
        ),
        "atom_to_token_idx": np.arange(n_atom, dtype=np.int64),
        "asym_id": np.zeros((n_atom,), dtype=np.int64),
        "chemical_bond_atom_indices": np.asarray(bonds, dtype=np.int64),
        "chemical_bond_order": np.asarray(
            [bond.GetBondTypeAsDouble() for bond in mol.GetBonds()], dtype=np.float32
        ),
        "chemical_bond_stereo": np.asarray(
            [int(bond.GetStereo()) for bond in mol.GetBonds()], dtype=np.int64
        ),
        "ligand_stereo": np.asarray(
            [int(atom.GetChiralTag()) for atom in mol.GetAtoms()], dtype=np.int64
        ),
        "covalent_atom_indices": np.empty((0, 2), dtype=np.int64),
        **_input_ligand(n_atom),
    }


def test_rdkit_all_pair_bounds_match_direct_fixture() -> None:
    """Bounds of the hydrogenated molecule, as upstream's SMILES ligand has."""

    mol = Chem.MolFromSmiles("CCCO")
    features = _features_from_rdkit(mol)
    result = prepare_tfg_features(features)
    direct = GetMoleculeBoundsMatrix(Chem.AddHs(mol))
    pairs = np.asarray(list(combinations(range(mol.GetNumAtoms()), 2)), dtype=np.int64)

    np.testing.assert_array_equal(result["pairwise_distance_index"], pairs.T)
    np.testing.assert_allclose(
        result["pairwise_distance_upper_bound"],
        direct[pairs[:, 0], pairs[:, 1]],
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        result["pairwise_distance_lower_bound"],
        direct[pairs[:, 1], pairs[:, 0]],
        rtol=1e-6,
    )
    assert result["geometry_unsupported"] is False
    (record,) = result["geometry_provenance"]["residues"]
    assert record["status"] == "featurized"


def test_rdkit_experimental_torsions_keep_only_heavy_atom_matches() -> None:
    """Torsions are matched with hydrogens present, then kept if all-heavy.

    N-cyclohexylacetamide's secondary amide matches a hydrogen-bearing
    pattern; without explicit hydrogens RDKit returns one torsion more, which
    upstream's SMILES ligand never sees.
    """

    mol = Chem.MolFromSmiles("CC(=O)NC1CCCCC1")
    result = prepare_tfg_features(_features_from_rdkit(mol))
    n_heavy = mol.GetNumAtoms()
    direct = [
        item
        for item in GetExperimentalTorsions(Chem.AddHs(mol), useSmallRingTorsions=True)
        if max(item["atomIndices"]) < n_heavy
    ]

    np.testing.assert_array_equal(
        result["experimental_torsion_index"].T,
        np.asarray([item["atomIndices"] for item in direct], dtype=np.int64),
    )
    np.testing.assert_allclose(
        result["experimental_torsion_force_constant"],
        np.asarray([item["V"] for item in direct], dtype=np.float32),
    )
    np.testing.assert_allclose(
        result["experimental_torsion_sign"],
        np.asarray([item["signs"] for item in direct], dtype=np.float32),
    )
    heavy_only = GetExperimentalTorsions(mol, useSmallRingTorsions=True)
    assert len(heavy_only) == len(direct) + 1


def test_geometry_is_scoped_by_residue() -> None:
    features = _features(
        np.asarray([[0, 0, 0], [1, 0, 0], [5, 0, 0], [6, 0, 0]], np.float32),
        bonds=[(0, 1), (2, 3)],
    )
    features["output_atom_res_name"] = np.asarray(["l01", "l01", "l02", "l02"])

    result = prepare_tfg_features(features)

    np.testing.assert_array_equal(
        result["pairwise_distance_index"],
        np.asarray([[0, 2], [1, 3]], dtype=np.int64),
    )
    assert [
        record["res_name"] for record in result["geometry_provenance"]["residues"]
    ] == ["l01", "l02"]


def test_a_component_rdkit_rejects_warns_and_gets_no_constraints() -> None:
    """Upstream warns and returns empty features for it; the run goes on."""

    features = _features(
        np.asarray([[0, 0, 0], [1, 0, 0], [5, 0, 0], [6, 0, 0]], np.float32),
        bonds=[(0, 1), (2, 3)],
    )
    features["output_atom_res_name"] = np.asarray(["l01", "l01", "l02", "l02"])
    features["output_atom_element"] = np.asarray(["NotAnElement", "C", "C", "C"])

    with pytest.warns(UserWarning, match="l01 .* no geometry constraints"):
        result = prepare_tfg_features(features)

    assert result["geometry_unsupported"] is True
    failed, kept = result["geometry_provenance"]["residues"]
    assert failed["status"] == "failed" and "unknown atom element" in failed["error"]
    assert kept["status"] == "featurized"
    np.testing.assert_array_equal(
        result["pairwise_distance_index"], np.asarray([[2], [3]], dtype=np.int64)
    )
