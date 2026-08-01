"""Reconstruct RDKit geometry guidance features from static chemical topology."""

from __future__ import annotations

from collections.abc import Mapping
from itertools import combinations
from typing import Any

import numpy as np
import rdkit
from rdkit import Chem
from rdkit.Chem.rdDistGeom import GetExperimentalTorsions, GetMoleculeBoundsMatrix
from rdkit.Chem.rdMolTransforms import GetDihedralRad


def prepare_tfg_features(features: Mapping[str, Any]) -> dict[str, Any]:
    """Return ``features`` augmented with every JAX TFG feature contract.

    Reference positions are interpreted in Angstrom, matching the featurizer.
    Exact reference distances are used only for graph bonds and two-bond angle
    endpoints. They are not treated as general conformer bounds.
    """

    result = dict(features)
    coordinates = np.asarray(features["ref_pos"], dtype=np.float32)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("ref_pos must have shape (n_atom, 3)")
    n_atom = coordinates.shape[0]
    atom_to_token = np.asarray(features["atom_to_token_idx"], dtype=np.int64)
    if atom_to_token.shape != (n_atom,):
        raise ValueError("atom_to_token_idx must have shape (n_atom,)")
    token_asym = np.asarray(features["asym_id"], dtype=np.int64)
    if atom_to_token.size and (
        np.any(atom_to_token < 0) or np.any(atom_to_token >= len(token_asym))
    ):
        raise ValueError("atom_to_token_idx contains an invalid token index")
    atom_asym = token_asym[atom_to_token]

    bonds = _pairs(features.get("chemical_bond_atom_indices"), n_atom)
    orders = _vector(
        features.get("chemical_bond_order"), len(bonds), np.float32, default=1.0
    )
    stereos = _vector(
        features.get("chemical_bond_stereo"), len(bonds), np.int64, default=0
    )
    covalent = _pairs(features.get("covalent_atom_indices"), n_atom)
    interchain = [pair for pair in covalent if atom_asym[pair[0]] != atom_asym[pair[1]]]
    result["interchain_bond_index"] = _index(interchain, 2)

    mol_ids = np.asarray(features.get("mol_id", np.zeros(n_atom)), dtype=np.int64)
    if mol_ids.shape != (n_atom,):
        raise ValueError("mol_id must have shape (n_atom,)")
    accumulated = _empty_rdkit_lists()
    provenance: list[dict[str, Any]] = []
    unsupported_ids: list[int] = []
    for mol_id in sorted(np.unique(mol_ids).tolist()):
        atom_indices = np.flatnonzero(mol_ids == mol_id).astype(np.int64)
        bond_mask = np.isin(bonds[:, 0], atom_indices) & np.isin(
            bonds[:, 1], atom_indices
        )
        try:
            mol = _reconstruct_rdkit_mol(
                features,
                atom_indices,
                bonds[bond_mask],
                orders[bond_mask],
                stereos[bond_mask],
                coordinates,
            )
            local = _extract_rdkit_geometry(mol)
            _accumulate_rdkit_geometry(accumulated, local, atom_indices)
            provenance.append(
                {
                    "mol_id": int(mol_id),
                    "atom_indices": atom_indices.tolist(),
                    "supported": True,
                    "error": None,
                }
            )
        except Exception as exc:
            unsupported_ids.append(int(mol_id))
            provenance.append(
                {
                    "mol_id": int(mol_id),
                    "atom_indices": atom_indices.tolist(),
                    "supported": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    result.update(_finalize_rdkit_geometry(accumulated))
    result["geometry_unsupported"] = bool(unsupported_ids)
    result["geometry_unsupported_mol_ids"] = np.asarray(unsupported_ids, dtype=np.int64)
    result["geometry_provenance"] = {
        "backend": "rdkit",
        "rdkit_version": rdkit.__version__,
        "molecules": provenance,
    }
    return result


def require_supported_geometry(features: Mapping[str, Any]) -> None:
    """Fail before guided sampling when any molecule lacked RDKit geometry."""
    if bool(features.get("geometry_unsupported", False)):
        molecule_ids = np.asarray(
            features.get("geometry_unsupported_mol_ids", []), dtype=np.int64
        ).tolist()
        raise ValueError(
            "TFG geometry is unavailable for molecule IDs "
            f"{molecule_ids}; active geometry terms must not run with empty fallback"
        )


_INDEX_WIDTHS = {
    "pairwise_distance_index": 2,
    "experimental_torsion_index": 4,
    "linear_triple_bond_index": 3,
    "chiral_index": 4,
    "stereo_bond_index": 4,
    "planar_improper_index": 4,
}


def _empty_rdkit_lists() -> dict[str, list[Any]]:
    return {
        "pairwise_distance_index": [],
        "pairwise_distance_upper_bound": [],
        "pairwise_distance_lower_bound": [],
        "pairwise_distance_is_bond": [],
        "pairwise_distance_is_angle": [],
        "experimental_torsion_index": [],
        "experimental_torsion_force_constant": [],
        "experimental_torsion_sign": [],
        "linear_triple_bond_index": [],
        "chiral_index": [],
        "chiral_orientation": [],
        "stereo_bond_index": [],
        "stereo_bond_orientation": [],
        "planar_improper_index": [],
        "planar_improper_is_carbonyl": [],
    }


def _bond_type(order: float) -> Chem.BondType:
    if np.isclose(order, 1.0):
        return Chem.BondType.SINGLE
    if np.isclose(order, 1.5):
        return Chem.BondType.AROMATIC
    if np.isclose(order, 2.0):
        return Chem.BondType.DOUBLE
    if np.isclose(order, 3.0):
        return Chem.BondType.TRIPLE
    raise ValueError(f"unsupported RDKit bond order: {order}")


def _reconstruct_rdkit_mol(
    features: Mapping[str, Any],
    atom_indices: np.ndarray,
    bonds: np.ndarray,
    orders: np.ndarray,
    stereos: np.ndarray,
    coordinates: np.ndarray,
) -> Chem.Mol:
    symbols = _elements(features, len(coordinates))[atom_indices]
    charges = np.asarray(
        features.get("ref_charge", np.zeros(len(coordinates))), dtype=np.float32
    )
    if charges.shape != (len(coordinates),):
        raise ValueError("ref_charge must have shape (n_atom,)")
    chiral_tags = np.asarray(
        features.get("ligand_stereo", np.zeros(len(coordinates))), dtype=np.int64
    )
    if chiral_tags.shape != (len(coordinates),):
        raise ValueError("ligand_stereo must have shape (n_atom,)")
    periodic_table = Chem.GetPeriodicTable()
    editable = Chem.RWMol()
    global_to_local = {int(global_idx): i for i, global_idx in enumerate(atom_indices)}
    for global_idx, symbol in zip(atom_indices, symbols, strict=True):
        symbol = str(symbol).strip()
        symbol = symbol[:1].upper() + symbol[1:].lower()
        try:
            atomic_number = int(periodic_table.GetAtomicNumber(symbol))
        except RuntimeError as exc:
            raise ValueError(f"unknown atom element {symbol!r}") from exc
        if atomic_number <= 0:
            raise ValueError(f"unknown atom element {symbol!r}")
        atom = Chem.Atom(atomic_number)
        atom.SetFormalCharge(int(round(float(charges[global_idx]))))
        tag = int(chiral_tags[global_idx])
        if tag in Chem.ChiralType.values:
            atom.SetChiralTag(Chem.ChiralType.values[tag])
        editable.AddAtom(atom)
    for (left, right), order in zip(bonds, orders, strict=True):
        editable.AddBond(
            global_to_local[int(left)],
            global_to_local[int(right)],
            _bond_type(float(order)),
        )
    mol = editable.GetMol()
    for bond, order in zip(mol.GetBonds(), orders, strict=True):
        if np.isclose(order, 1.5):
            bond.SetIsAromatic(True)
            bond.GetBeginAtom().SetIsAromatic(True)
            bond.GetEndAtom().SetIsAromatic(True)
    for bond, stereo in zip(mol.GetBonds(), stereos, strict=True):
        stereo = int(stereo)
        if stereo not in Chem.BondStereo.values or stereo == 0:
            continue
        begin, end = bond.GetBeginAtom(), bond.GetEndAtom()
        begin_neighbors = sorted(
            atom.GetIdx()
            for atom in begin.GetNeighbors()
            if atom.GetIdx() != end.GetIdx()
        )
        end_neighbors = sorted(
            atom.GetIdx()
            for atom in end.GetNeighbors()
            if atom.GetIdx() != begin.GetIdx()
        )
        if begin_neighbors and end_neighbors:
            bond.SetStereoAtoms(begin_neighbors[0], end_neighbors[0])
            bond.SetStereo(Chem.BondStereo.values[stereo])
    conformer = Chem.Conformer(len(atom_indices))
    for local_idx, global_idx in enumerate(atom_indices):
        x, y, z = (float(value) for value in coordinates[global_idx])
        conformer.SetAtomPosition(local_idx, (x, y, z))
    mol.AddConformer(conformer, assignId=True)
    Chem.SanitizeMol(mol)
    Chem.AssignStereochemistry(mol, force=True, cleanIt=True)
    # Bond directions are not part of the static feature contract, so restore
    # the explicit RDKit stereo enum and its substituent atoms after cleaning.
    for bond, stereo in zip(mol.GetBonds(), stereos, strict=True):
        stereo = int(stereo)
        if stereo not in Chem.BondStereo.values or stereo == 0:
            continue
        begin, end = bond.GetBeginAtom(), bond.GetEndAtom()
        begin_neighbors = sorted(
            atom.GetIdx()
            for atom in begin.GetNeighbors()
            if atom.GetIdx() != end.GetIdx()
        )
        end_neighbors = sorted(
            atom.GetIdx()
            for atom in end.GetNeighbors()
            if atom.GetIdx() != begin.GetIdx()
        )
        if begin_neighbors and end_neighbors:
            bond.SetStereoAtoms(begin_neighbors[0], end_neighbors[0])
            bond.SetStereo(Chem.BondStereo.values[stereo])
    Chem.GetSymmSSSR(mol)
    return mol


def _extract_rdkit_geometry(mol: Chem.Mol) -> dict[str, list[Any]]:
    output = _empty_rdkit_lists()
    n_atom = mol.GetNumAtoms()
    bounds = GetMoleculeBoundsMatrix(mol)
    bond_pairs = {
        tuple(sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())))
        for bond in mol.GetBonds()
    }
    bond_array = np.asarray(sorted(bond_pairs), dtype=np.int64).reshape((-1, 2))
    adjacency = _adjacency(n_atom, bond_array)
    angle_pairs = {
        tuple(sorted(pair))
        for neighbors in adjacency
        for pair in combinations(neighbors, 2)
    }
    for left, right in combinations(range(n_atom), 2):
        output["pairwise_distance_index"].append([left, right])
        output["pairwise_distance_upper_bound"].append(float(bounds[left, right]))
        output["pairwise_distance_lower_bound"].append(float(bounds[right, left]))
        output["pairwise_distance_is_bond"].append(int((left, right) in bond_pairs))
        output["pairwise_distance_is_angle"].append(int((left, right) in angle_pairs))

    marked_bonds: set[tuple[int, int]] = set()
    for torsion in GetExperimentalTorsions(mol, useSmallRingTorsions=True):
        atoms = list(torsion["atomIndices"])
        output["experimental_torsion_index"].append(atoms)
        output["experimental_torsion_force_constant"].append(list(torsion["V"]))
        output["experimental_torsion_sign"].append(list(torsion["signs"]))
        marked_bonds.add(tuple(sorted((atoms[1], atoms[2]))))
    for ring in mol.GetRingInfo().AtomRings():
        if not 3 < len(ring) < 7:
            continue
        for position in range(len(ring)):
            atoms = [ring[(position + offset) % len(ring)] for offset in range(4)]
            center_bond = tuple(sorted((atoms[1], atoms[2])))
            if center_bond in marked_bonds:
                continue
            if all(
                mol.GetAtomWithIdx(atom).GetHybridization()
                == Chem.HybridizationType.SP2
                for atom in atoms
            ):
                output["experimental_torsion_index"].append(atoms)
                output["experimental_torsion_force_constant"].append(
                    [0.0, 100.0, 0.0, 0.0, 0.0, 0.0]
                )
                output["experimental_torsion_sign"].append([1, -1, 1, 1, 1, 1])
                marked_bonds.add(center_bond)

    conformer = mol.GetConformer(0)
    for atom in mol.GetAtoms():
        center = atom.GetIdx()
        neighbors = sorted(neighbor.GetIdx() for neighbor in atom.GetNeighbors())
        if (
            atom.GetChiralTag()
            in {
                Chem.ChiralType.CHI_TETRAHEDRAL_CCW,
                Chem.ChiralType.CHI_TETRAHEDRAL_CW,
            }
            and 3 <= len(neighbors) <= 4
        ):
            for selected in combinations(neighbors, 3):
                atoms = [*selected, center]
                output["chiral_index"].append(atoms)
                output["chiral_orientation"].append(
                    1.0 if GetDihedralRad(conformer, *atoms) >= 0 else -1.0
                )

        if (
            atom.GetSymbol() in {"C", "N", "O"}
            and atom.GetHybridization() == Chem.HybridizationType.SP2
            and len(neighbors) == 3
        ):
            first, second, third = neighbors
            output["planar_improper_index"].extend(
                (
                    [first, second, center, third],
                    [third, first, center, second],
                    [second, third, center, first],
                )
            )
            carbonyl = atom.GetSymbol() == "C" and any(
                neighbor.GetSymbol() == "O"
                and neighbor.GetHybridization() == Chem.HybridizationType.SP2
                for neighbor in atom.GetNeighbors()
            )
            output["planar_improper_is_carbonyl"].extend([float(carbonyl)] * 3)

    for bond in mol.GetBonds():
        begin, end = bond.GetBeginAtom(), bond.GetEndAtom()
        begin_idx, end_idx = begin.GetIdx(), end.GetIdx()
        begin_neighbors = sorted(
            atom.GetIdx() for atom in begin.GetNeighbors() if atom.GetIdx() != end_idx
        )
        end_neighbors = sorted(
            atom.GetIdx() for atom in end.GetNeighbors() if atom.GetIdx() != begin_idx
        )
        if (
            bond.GetBondType() == Chem.BondType.TRIPLE
            and not bond.GetIsAromatic()
            and not begin.GetIsAromatic()
            and not end.GetIsAromatic()
            and begin.GetHybridization() == Chem.HybridizationType.SP
            and end.GetHybridization() == Chem.HybridizationType.SP
        ):
            output["linear_triple_bond_index"].extend(
                [neighbor, begin_idx, end_idx] for neighbor in begin_neighbors
            )
            output["linear_triple_bond_index"].extend(
                [begin_idx, end_idx, neighbor] for neighbor in end_neighbors
            )
        if (
            bond.GetStereo()
            not in {
                Chem.BondStereo.STEREOE,
                Chem.BondStereo.STEREOZ,
            }
            or not begin_neighbors
            or not end_neighbors
        ):
            continue
        stereo_atoms = [
            begin_neighbors[0],
            begin_idx,
            end_idx,
            end_neighbors[0],
        ]
        output["stereo_bond_index"].append(stereo_atoms)
        output["stereo_bond_orientation"].append(
            float(abs(GetDihedralRad(conformer, *stereo_atoms)) >= np.pi / 2)
        )
        if len(begin_neighbors) == 2 and len(end_neighbors) == 2:
            stereo_atoms = [
                begin_neighbors[1],
                begin_idx,
                end_idx,
                end_neighbors[1],
            ]
            output["stereo_bond_index"].append(stereo_atoms)
            output["stereo_bond_orientation"].append(
                float(abs(GetDihedralRad(conformer, *stereo_atoms)) >= np.pi / 2)
            )
    return output


def _accumulate_rdkit_geometry(
    accumulated: dict[str, list[Any]],
    local: dict[str, list[Any]],
    atom_indices: np.ndarray,
) -> None:
    for key, values in local.items():
        if key in _INDEX_WIDTHS:
            accumulated[key].extend(
                atom_indices[np.asarray(values, dtype=np.int64)].tolist()
                if values
                else []
            )
        else:
            accumulated[key].extend(values)


def _finalize_rdkit_geometry(values: dict[str, list[Any]]) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    for key, width in _INDEX_WIDTHS.items():
        output[key] = _index(values[key], width)
    for key in (
        "pairwise_distance_is_bond",
        "pairwise_distance_is_angle",
    ):
        output[key] = np.asarray(values[key], dtype=np.int64)
    for key in (
        "pairwise_distance_upper_bound",
        "pairwise_distance_lower_bound",
        "chiral_orientation",
        "stereo_bond_orientation",
        "planar_improper_is_carbonyl",
    ):
        output[key] = np.asarray(values[key], dtype=np.float32)
    for key in (
        "experimental_torsion_force_constant",
        "experimental_torsion_sign",
    ):
        array = np.asarray(values[key], dtype=np.float32)
        output[key] = array.reshape((-1, 6))
    return output


def _pairs(value: Any, n_atom: int) -> np.ndarray:
    if value is None:
        return np.empty((0, 2), dtype=np.int64)
    array = np.asarray(value, dtype=np.int64)
    if array.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    if array.ndim != 2:
        raise ValueError("bond atom indices must be rank-2")
    if array.shape[1] == 2:
        pairs = array
    elif array.shape[0] == 2:
        pairs = array.T
    else:
        raise ValueError("bond atom indices must have shape (n_bond, 2) or (2, n_bond)")
    if np.any(pairs < 0) or np.any(pairs >= n_atom):
        raise ValueError("bond atom indices contain an invalid atom index")
    if np.any(pairs[:, 0] == pairs[:, 1]):
        raise ValueError("self bonds are not valid geometry annotations")
    return pairs.astype(np.int64, copy=False)


def _vector(value: Any, length: int, dtype: Any, *, default: float) -> np.ndarray:
    if value is None:
        return np.full((length,), default, dtype=dtype)
    array = np.asarray(value, dtype=dtype)
    if array.shape != (length,):
        raise ValueError("chemical bond annotation length does not match bond indices")
    return array


def _adjacency(n_atom: int, bonds: np.ndarray) -> list[list[int]]:
    neighbors: list[set[int]] = [set() for _ in range(n_atom)]
    for left, right in bonds:
        neighbors[int(left)].add(int(right))
        neighbors[int(right)].add(int(left))
    return [sorted(values) for values in neighbors]


def _index(rows: Any, width: int) -> np.ndarray:
    array = np.asarray(list(rows), dtype=np.int64)
    if array.size == 0:
        return np.empty((width, 0), dtype=np.int64)
    return array.reshape((-1, width)).T


def _distance_features(
    coordinates: np.ndarray,
    bonds: np.ndarray,
    adjacency: list[list[int]],
) -> dict[str, np.ndarray]:
    categories: dict[tuple[int, int], list[int]] = {}
    for left, right in bonds:
        categories.setdefault(tuple(sorted((int(left), int(right)))), [0, 0])[0] = 1
    for neighbors in adjacency:
        for left, right in combinations(neighbors, 2):
            categories.setdefault(tuple(sorted((left, right))), [0, 0])[1] = 1
    pairs = sorted(categories)
    if not pairs:
        return {
            "pairwise_distance_index": np.empty((2, 0), dtype=np.int64),
            "pairwise_distance_upper_bound": np.empty((0,), dtype=np.float32),
            "pairwise_distance_lower_bound": np.empty((0,), dtype=np.float32),
            "pairwise_distance_is_bond": np.empty((0,), dtype=np.int64),
            "pairwise_distance_is_angle": np.empty((0,), dtype=np.int64),
        }
    index = np.asarray(pairs, dtype=np.int64)
    distance = np.linalg.norm(
        coordinates[index[:, 0]] - coordinates[index[:, 1]], axis=-1
    ).astype(np.float32)
    labels = np.asarray([categories[pair] for pair in pairs], dtype=np.int64)
    return {
        "pairwise_distance_index": index.T,
        "pairwise_distance_upper_bound": distance.copy(),
        "pairwise_distance_lower_bound": distance.copy(),
        "pairwise_distance_is_bond": labels[:, 0],
        "pairwise_distance_is_angle": labels[:, 1],
    }


def _signed_dihedral(coordinates: np.ndarray, atoms: list[int]) -> float:
    first, second, third, fourth = coordinates[np.asarray(atoms)]
    ij = second - first
    kj = second - third
    kl = fourth - third
    m = np.cross(ij, kj)
    n = np.cross(kj, kl)
    phi = np.arctan2(np.linalg.norm(np.cross(m, n)), np.dot(m, n) + 1.0e-8)
    return float(-phi * np.sign(np.dot(ij, n)))


def _stereo_features(
    coordinates: np.ndarray,
    bonds: np.ndarray,
    stereos: np.ndarray,
    adjacency: list[list[int]],
) -> dict[str, np.ndarray]:
    indices: list[list[int]] = []
    orientations: list[float] = []
    for (left_raw, right_raw), stereo in zip(bonds, stereos, strict=True):
        if int(stereo) not in {2, 3, 4, 5}:  # RDKit Z/E/CIS/TRANS
            continue
        left, right = int(left_raw), int(right_raw)
        left_neighbors = [atom for atom in adjacency[left] if atom != right]
        right_neighbors = [atom for atom in adjacency[right] if atom != left]
        if not left_neighbors or not right_neighbors:
            continue
        candidates = [(left_neighbors[0], right_neighbors[0])]
        if len(left_neighbors) == 2 and len(right_neighbors) == 2:
            candidates.append((left_neighbors[1], right_neighbors[1]))
        for outer_left, outer_right in candidates:
            atoms = [outer_left, left, right, outer_right]
            indices.append(atoms)
            orientations.append(
                float(abs(_signed_dihedral(coordinates, atoms)) >= np.pi / 2)
            )
    return {
        "stereo_bond_index": _index(indices, 4),
        "stereo_bond_orientation": np.asarray(orientations, dtype=np.float32),
    }


def _chiral_features(
    coordinates: np.ndarray,
    features: Mapping[str, Any],
    adjacency: list[list[int]],
    n_atom: int,
) -> dict[str, np.ndarray]:
    tags = np.asarray(
        features.get("ligand_stereo", np.zeros((n_atom,), dtype=np.int64)),
        dtype=np.int64,
    )
    if tags.shape != (n_atom,):
        raise ValueError("ligand_stereo must have shape (n_atom,)")
    indices: list[list[int]] = []
    orientations: list[float] = []
    for center, tag in enumerate(tags):
        if int(tag) not in {1, 2}:  # RDKit tetrahedral CW/CCW
            continue
        neighbors = adjacency[center]
        if len(neighbors) < 3 or len(neighbors) > 4:
            continue
        for selected in combinations(neighbors, 3):
            atoms = [*selected, center]
            indices.append(atoms)
            orientations.append(
                1.0 if _signed_dihedral(coordinates, atoms) >= 0.0 else -1.0
            )
    return {
        "chiral_index": _index(indices, 4),
        "chiral_orientation": np.asarray(orientations, dtype=np.float32),
    }


def _elements(features: Mapping[str, Any], n_atom: int) -> np.ndarray:
    if "output_atom_element" in features:
        elements = np.asarray(features["output_atom_element"], dtype=str)
        if elements.shape != (n_atom,):
            raise ValueError("output_atom_element must have shape (n_atom,)")
        return elements
    encoded = np.asarray(features["ref_element"])
    if encoded.ndim != 2 or encoded.shape[0] != n_atom:
        raise ValueError("ref_element must have shape (n_atom, n_element)")
    periodic_table = Chem.GetPeriodicTable()
    indices = np.argmax(encoded, axis=-1) + 1
    return np.asarray(
        [
            periodic_table.GetElementSymbol(int(index))
            if np.any(encoded[row] != 0)
            else "X"
            for row, index in enumerate(indices)
        ]
    )


def _planar_features(
    features: Mapping[str, Any],
    bonds: np.ndarray,
    orders: np.ndarray,
    adjacency: list[list[int]],
    n_atom: int,
) -> dict[str, np.ndarray]:
    elements = _elements(features, n_atom)
    order_by_pair = {
        tuple(sorted((int(left), int(right)))): float(order)
        for (left, right), order in zip(bonds, orders, strict=True)
    }
    indices: list[list[int]] = []
    carbonyl: list[float] = []
    for center, neighbors in enumerate(adjacency):
        if elements[center] not in {"C", "N", "O"} or len(neighbors) != 3:
            continue
        has_double = any(
            order_by_pair[tuple(sorted((center, neighbor)))] >= 1.75
            for neighbor in neighbors
        )
        if not has_double:
            continue
        first, second, third = neighbors
        indices.extend(
            (
                [first, second, center, third],
                [third, first, center, second],
                [second, third, center, first],
            )
        )
        is_carbonyl = elements[center] == "C" and any(
            elements[neighbor] == "O"
            and order_by_pair[tuple(sorted((center, neighbor)))] >= 1.75
            for neighbor in neighbors
        )
        carbonyl.extend([float(is_carbonyl)] * 3)
    return {
        "planar_improper_index": _index(indices, 4),
        "planar_improper_is_carbonyl": np.asarray(carbonyl, dtype=np.float32),
    }


def _linear_triples(
    bonds: np.ndarray, orders: np.ndarray, adjacency: list[list[int]]
) -> np.ndarray:
    triples: list[list[int]] = []
    for (left_raw, right_raw), order in zip(bonds, orders, strict=True):
        if float(order) < 2.75:
            continue
        left, right = int(left_raw), int(right_raw)
        triples.extend(
            [neighbor, left, right] for neighbor in adjacency[left] if neighbor != right
        )
        triples.extend(
            [left, right, neighbor] for neighbor in adjacency[right] if neighbor != left
        )
    return _index(triples, 3)
