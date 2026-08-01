from __future__ import annotations

from itertools import combinations

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem.rdDistGeom import GetExperimentalTorsions, GetMoleculeBoundsMatrix

from foldjax.models.protenix.data.geometry import (
    prepare_tfg_features,
    require_supported_geometry,
)
from foldjax.models.protenix.tfg.config import parse_tfg_config, validate_features


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


def test_empty_polymer_geometry_is_safe_and_does_not_fabricate_annotations() -> None:
    features = {
        "ref_pos": np.zeros((2, 3), dtype=np.float32),
        "ref_element": np.zeros((2, 128), dtype=np.float32),
        "atom_to_token_idx": np.array([0, 1], dtype=np.int64),
        "asym_id": np.array([0, 0], dtype=np.int64),
    }

    result = prepare_tfg_features(features)

    for key in (
        "interchain_bond_index",
        "pairwise_distance_index",
        "experimental_torsion_index",
        "linear_triple_bond_index",
        "chiral_index",
        "stereo_bond_index",
        "planar_improper_index",
    ):
        assert result[key].shape[-1] == 0
    assert result["experimental_torsion_force_constant"].shape == (0, 6)
    assert result["experimental_torsion_sign"].shape == (0, 6)


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
        "mol_id": np.zeros((n_atom,), dtype=np.int64),
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
    }


def test_rdkit_all_pair_bounds_match_direct_fixture() -> None:
    mol = Chem.MolFromSmiles("CCCO")
    features = _features_from_rdkit(mol)
    result = prepare_tfg_features(features)
    direct = GetMoleculeBoundsMatrix(mol)
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
    assert result["geometry_provenance"]["molecules"][0]["supported"] is True


def test_rdkit_experimental_torsions_match_direct_fixture() -> None:
    mol = Chem.MolFromSmiles("CCCCC")
    result = prepare_tfg_features(_features_from_rdkit(mol))
    direct = GetExperimentalTorsions(mol, useSmallRingTorsions=True)

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


def test_rdkit_reconstruction_is_scoped_by_mol_id() -> None:
    features = _features(
        np.asarray([[0, 0, 0], [1, 0, 0], [5, 0, 0], [6, 0, 0]], np.float32),
        bonds=[(0, 1), (2, 3)],
    )
    features["mol_id"] = np.asarray([10, 10, 20, 20], dtype=np.int64)

    result = prepare_tfg_features(features)

    np.testing.assert_array_equal(
        result["pairwise_distance_index"],
        np.asarray([[0, 2], [1, 3]], dtype=np.int64),
    )
    assert [item["mol_id"] for item in result["geometry_provenance"]["molecules"]] == [
        10,
        20,
    ]


def test_unsanitizable_molecule_has_explicit_unsupported_provenance() -> None:
    features = _features(np.zeros((2, 3), dtype=np.float32), bonds=[(0, 1)])
    features["output_atom_element"] = np.asarray(["NotAnElement", "C"])
    features["mol_id"] = np.zeros((2,), dtype=np.int64)

    result = prepare_tfg_features(features)

    assert result["geometry_unsupported"] is True
    assert result["geometry_unsupported_mol_ids"].tolist() == [0]
    assert result["geometry_provenance"]["molecules"][0]["supported"] is False
    assert result["pairwise_distance_index"].shape == (2, 0)
    with pytest.raises(ValueError, match="must not run with empty fallback"):
        require_supported_geometry(result)
