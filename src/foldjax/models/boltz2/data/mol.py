import itertools
import pickle
import random

import numpy as np
from foldjax.models.boltz2.data._torch import torch
from rdkit.Chem import Mol

from foldjax.models.boltz2.data import const
from foldjax.models.boltz2.data.identifiers import resolve_molecule_pickle


def load_molecules(moldir: str, molecules: list[str]) -> dict[str, Mol]:
    """Load the given input data.

    Parameters
    ----------
    moldir : str
        The path to the molecules directory.
    molecules : list[str]
        The molecules to load.

    Returns
    -------
    dict[str, Mol]
        The loaded molecules.
    """
    loaded_mols = {}
    for molecule in molecules:
        path = resolve_molecule_pickle(moldir, molecule)
        with path.open("rb") as f:
            loaded_mols[molecule] = pickle.load(f)  # noqa: S301
    return loaded_mols


def load_canonicals(moldir: str) -> dict[str, Mol]:
    """Load the given input data.

    Parameters
    ----------
    moldir : str
        The molecules to load.

    Returns
    -------
    dict[str, Mol]
        The loaded molecules.

    """
    return load_molecules(moldir, const.canonical_tokens)


def get_symmetries(mols: dict[str, Mol]) -> dict:  # noqa: PLR0912
    """Create a dictionary for the ligand symmetries.

    Parameters
    ----------
    path : str
        The path to the ligand symmetries.

    Returns
    -------
    dict
        The ligand symmetries.

    """
    symmetries = {}
    for key, mol in mols.items():
        try:
            sym = pickle.loads(bytes.fromhex(mol.GetProp("symmetries")))  # noqa: S301

            if mol.HasProp("pb_edge_index"):
                edge_index = pickle.loads(
                    bytes.fromhex(mol.GetProp("pb_edge_index"))
                ).astype(np.int64)  # noqa: S301
                lower_bounds = pickle.loads(
                    bytes.fromhex(mol.GetProp("pb_lower_bounds"))
                )  # noqa: S301
                upper_bounds = pickle.loads(
                    bytes.fromhex(mol.GetProp("pb_upper_bounds"))
                )  # noqa: S301
                bond_mask = pickle.loads(bytes.fromhex(mol.GetProp("pb_bond_mask")))  # noqa: S301
                angle_mask = pickle.loads(bytes.fromhex(mol.GetProp("pb_angle_mask")))  # noqa: S301
            else:
                edge_index = np.empty((2, 0), dtype=np.int64)
                lower_bounds = np.array([], dtype=np.float32)
                upper_bounds = np.array([], dtype=np.float32)
                bond_mask = np.array([], dtype=np.float32)
                angle_mask = np.array([], dtype=np.float32)

            if mol.HasProp("chiral_atom_index"):
                chiral_atom_index = pickle.loads(
                    bytes.fromhex(mol.GetProp("chiral_atom_index"))
                ).astype(np.int64)
                chiral_check_mask = pickle.loads(
                    bytes.fromhex(mol.GetProp("chiral_check_mask"))
                ).astype(np.int64)
                chiral_atom_orientations = pickle.loads(
                    bytes.fromhex(mol.GetProp("chiral_atom_orientations"))
                )
            else:
                chiral_atom_index = np.empty((4, 0), dtype=np.int64)
                chiral_check_mask = np.array([], dtype=bool)
                chiral_atom_orientations = np.array([], dtype=bool)

            if mol.HasProp("stereo_bond_index"):
                stereo_bond_index = pickle.loads(
                    bytes.fromhex(mol.GetProp("stereo_bond_index"))
                ).astype(np.int64)
                stereo_check_mask = pickle.loads(
                    bytes.fromhex(mol.GetProp("stereo_check_mask"))
                ).astype(np.int64)
                stereo_bond_orientations = pickle.loads(
                    bytes.fromhex(mol.GetProp("stereo_bond_orientations"))
                )
            else:
                stereo_bond_index = np.empty((4, 0), dtype=np.int64)
                stereo_check_mask = np.array([], dtype=bool)
                stereo_bond_orientations = np.array([], dtype=bool)

            if mol.HasProp("aromatic_5_ring_index"):
                aromatic_5_ring_index = pickle.loads(
                    bytes.fromhex(mol.GetProp("aromatic_5_ring_index"))
                ).astype(np.int64)
            else:
                aromatic_5_ring_index = np.empty((5, 0), dtype=np.int64)
            if mol.HasProp("aromatic_6_ring_index"):
                aromatic_6_ring_index = pickle.loads(
                    bytes.fromhex(mol.GetProp("aromatic_6_ring_index"))
                ).astype(np.int64)
            else:
                aromatic_6_ring_index = np.empty((6, 0), dtype=np.int64)
            if mol.HasProp("planar_double_bond_index"):
                planar_double_bond_index = pickle.loads(
                    bytes.fromhex(mol.GetProp("planar_double_bond_index"))
                ).astype(np.int64)
            else:
                planar_double_bond_index = np.empty((6, 0), dtype=np.int64)

            atom_names = [atom.GetProp("name") for atom in mol.GetAtoms()]
            symmetries[key] = (
                sym,
                atom_names,
                edge_index,
                lower_bounds,
                upper_bounds,
                bond_mask,
                angle_mask,
                chiral_atom_index,
                chiral_check_mask,
                chiral_atom_orientations,
                stereo_bond_index,
                stereo_check_mask,
                stereo_bond_orientations,
                aromatic_5_ring_index,
                aromatic_6_ring_index,
                planar_double_bond_index,
            )
        except Exception:  # noqa: BLE001, PERF203, S110
            pass

    return symmetries


def all_different_after_swap(l):
    final = [s[-1] for s in l]
    return len(final) == len(set(final))


def get_chain_symmetries(cropped, max_n_symmetries=100):
    # get all coordinates and resolved mask
    structure = cropped.structure
    all_coords = []
    all_resolved_mask = []
    original_atom_idx = []
    chain_atom_idx = []
    chain_atom_num = []
    chain_in_crop = []
    chain_asym_id = []
    new_atom_idx = 0

    for chain in structure.chains:
        atom_idx, atom_num = (
            chain["atom_idx"],  # Global index of first atom in the chain
            chain["atom_num"],  # Number of atoms in the chain
        )

        # compute coordinates and resolved mask
        resolved_mask = structure.atoms["is_present"][
            atom_idx : atom_idx + atom_num
        ]  # Whether each atom in the chain is actually resolved

        # ensemble_atom_starts = [structure.ensemble[idx]["atom_coord_idx"] for idx in cropped.ensemble_ref_idxs]
        # coords = np.array(
        #    [structure.coords[ensemble_atom_start + atom_idx: ensemble_atom_start + atom_idx + atom_num]["coords"] for
        #     ensemble_atom_start in ensemble_atom_starts])

        coords = structure.atoms["coords"][atom_idx : atom_idx + atom_num]

        in_crop = False
        for token in cropped.tokens:
            if token["asym_id"] == chain["asym_id"]:
                in_crop = True
                break

        all_coords.append(coords)
        all_resolved_mask.append(resolved_mask)
        original_atom_idx.append(atom_idx)
        chain_atom_idx.append(new_atom_idx)
        chain_atom_num.append(atom_num)
        chain_in_crop.append(in_crop)
        chain_asym_id.append(chain["asym_id"])

        new_atom_idx += atom_num

    all_coords = np.concatenate(all_coords, axis=0)
    # Compute backmapping from token to all coords
    crop_to_all_atom_map = []
    for token in cropped.tokens:
        chain_idx = chain_asym_id.index(token["asym_id"])
        start = (
            chain_atom_idx[chain_idx] - original_atom_idx[chain_idx] + token["atom_idx"]
        )
        crop_to_all_atom_map.append(np.arange(start, start + token["atom_num"]))
    crop_to_all_atom_map = np.concatenate(crop_to_all_atom_map, axis=0)

    # Compute the connections edge index for covalent bonds
    all_atom_to_crop_map = np.zeros(all_coords.shape[0], dtype=np.int64)
    all_atom_to_crop_map[crop_to_all_atom_map.astype(np.int64)] = np.arange(
        crop_to_all_atom_map.shape[0]
    )
    connections_edge_index = []
    for connection in structure.bonds:
        if (connection["chain_1"] == connection["chain_2"]) and (
            connection["res_1"] == connection["res_2"]
        ):
            continue
        connections_edge_index.append([connection["atom_1"], connection["atom_2"]])
    if len(connections_edge_index) > 0:
        connections_edge_index = np.array(connections_edge_index, dtype=np.int64).T
        connections_edge_index = all_atom_to_crop_map[connections_edge_index]
    else:
        connections_edge_index = np.empty((2, 0))

    # Compute the symmetries between chains
    symmetries = []
    swaps = []
    for i, chain in enumerate(structure.chains):
        start = chain_atom_idx[i]
        end = start + chain_atom_num[i]

        if chain_in_crop[i]:
            possible_swaps = []
            for j, chain2 in enumerate(structure.chains):
                start2 = chain_atom_idx[j]
                end2 = start2 + chain_atom_num[j]
                if (
                    chain["entity_id"] == chain2["entity_id"]
                    and end - start == end2 - start2
                ):
                    possible_swaps.append((start, end, start2, end2, i, j))
            swaps.append(possible_swaps)

        found = False
        for symmetry_idx, symmetry in enumerate(symmetries):
            j = symmetry[0][0]
            chain2 = structure.chains[j]
            start2 = chain_atom_idx[j]
            end2 = start2 + chain_atom_num[j]
            if (
                chain["entity_id"] == chain2["entity_id"]
                and end - start == end2 - start2
            ):
                symmetries[symmetry_idx].append(
                    (i, start, end, chain_in_crop[i], chain["mol_type"])
                )
                found = True
        if not found:
            symmetries.append([(i, start, end, chain_in_crop[i], chain["mol_type"])])

    combinations = itertools.product(*swaps)
    # to avoid combinatorial explosion, bound the number of combinations even considered
    combinations = list(itertools.islice(combinations, max_n_symmetries * 10))
    # filter for all chains getting a different assignment
    combinations = [c for c in combinations if all_different_after_swap(c)]

    if len(combinations) > max_n_symmetries:
        combinations = random.sample(combinations, max_n_symmetries)

    if len(combinations) == 0:
        combinations.append([])

    for i in range(len(symmetries) - 1, -1, -1):
        if not any(chain[3] for chain in symmetries[i]):
            symmetries.pop(i)

    features = {}
    features["all_coords"] = torch.Tensor(all_coords)  # axis=1 with ensemble

    features["all_resolved_mask"] = torch.Tensor(
        np.concatenate(all_resolved_mask, axis=0)
    )
    features["crop_to_all_atom_map"] = torch.Tensor(crop_to_all_atom_map)
    features["chain_symmetries"] = symmetries
    features["connections_edge_index"] = torch.tensor(connections_edge_index)
    features["chain_swaps"] = combinations

    return features


def get_amino_acids_symmetries(cropped):
    # Compute standard amino-acids symmetries
    swaps = []
    start_index_crop = 0
    for token in cropped.tokens:
        symmetries = const.ref_symmetries.get(const.tokens[token["res_type"]], [])
        if len(symmetries) > 0:
            residue_swaps = []
            for sym in symmetries:
                sym_new_idx = [
                    (i + start_index_crop, j + start_index_crop) for i, j in sym
                ]
                residue_swaps.append(sym_new_idx)
            swaps.append(residue_swaps)
        start_index_crop += token["atom_num"]

    features = {"amino_acids_symmetries": swaps}
    return features


def slice_valid_index(index, ccd_to_valid_id_array, args=None):
    index = ccd_to_valid_id_array[index]
    valid_index_mask = (~np.isnan(index)).all(axis=0)
    index = index[:, valid_index_mask]
    if args is None:
        return index
    args = (arg[valid_index_mask] for arg in args)
    return index, args


def get_ligand_symmetries(cropped, symmetries, return_physical_metrics=False):
    # Compute ligand and non-standard amino-acids symmetries
    structure = cropped.structure

    added_molecules = {}
    index_mols = []
    atom_count = 0

    for token in cropped.tokens:
        # check if molecule is already added by identifying it through asym_id and res_idx
        atom_count += token["atom_num"]
        mol_id = (token["asym_id"], token["res_idx"])
        if mol_id in added_molecules:
            added_molecules[mol_id] += token["atom_num"]
            continue
        added_molecules[mol_id] = token["atom_num"]

        # get the molecule type and indices
        residue_idx = token["res_idx"] + structure.chains[token["asym_id"]]["res_idx"]
        mol_name = structure.residues[residue_idx]["name"]
        atom_idx = structure.residues[residue_idx]["atom_idx"]
        mol_atom_names = structure.atoms[
            atom_idx : atom_idx + structure.residues[residue_idx]["atom_num"]
        ]["name"]
        if mol_name not in const.ref_symmetries:
            index_mols.append(
                (mol_name, atom_count - token["atom_num"], mol_id, mol_atom_names)
            )

    # for each molecule, get the symmetries
    molecule_symmetries = []
    all_edge_index = []
    all_lower_bounds, all_upper_bounds = [], []
    all_bond_mask, all_angle_mask = [], []
    all_chiral_atom_index, all_chiral_check_mask, all_chiral_atom_orientations = (
        [],
        [],
        [],
    )
    all_stereo_bond_index, all_stereo_check_mask, all_stereo_bond_orientations = (
        [],
        [],
        [],
    )
    (
        all_aromatic_5_ring_index,
        all_aromatic_6_ring_index,
        all_planar_double_bond_index,
    ) = (
        [],
        [],
        [],
    )
    for mol_name, start_mol, mol_id, mol_atom_names in index_mols:
        if mol_name not in symmetries:
            continue
        else:
            swaps = []
            (
                syms_ccd,
                mol_atom_names_ccd,
                edge_index,
                lower_bounds,
                upper_bounds,
                bond_mask,
                angle_mask,
                chiral_atom_index,
                chiral_check_mask,
                chiral_atom_orientations,
                stereo_bond_index,
                stereo_check_mask,
                stereo_bond_orientations,
                aromatic_5_ring_index,
                aromatic_6_ring_index,
                planar_double_bond_index,
            ) = symmetries[mol_name]
            # Get indices of mol_atom_names_ccd that are in mol_atom_names
            ccd_to_valid_ids = {
                mol_atom_names_ccd.index(name): i
                for i, name in enumerate(mol_atom_names)
            }
            ccd_to_valid_id_array = np.array(
                [
                    float("nan") if i not in ccd_to_valid_ids else ccd_to_valid_ids[i]
                    for i in range(len(mol_atom_names_ccd))
                ]
            )
            ccd_valid_ids = set(ccd_to_valid_ids.keys())
            syms = []
            # Get syms
            for sym_ccd in syms_ccd:
                sym_dict = {}
                bool_add = True
                for i, j in enumerate(sym_ccd):
                    if i in ccd_valid_ids:
                        if j in ccd_valid_ids:
                            i_true = ccd_to_valid_ids[i]
                            j_true = ccd_to_valid_ids[j]
                            sym_dict[i_true] = j_true
                        else:
                            bool_add = False
                            break
                if bool_add:
                    syms.append([sym_dict[i] for i in range(len(ccd_valid_ids))])
            for sym in syms:
                if len(sym) != added_molecules[mol_id]:
                    raise Exception(
                        f"Symmetry length mismatch {len(sym)} {added_molecules[mol_id]}"
                    )
                # assert (
                #     len(sym) == added_molecules[mol_id]
                # ), f"Symmetry length mismatch {len(sym)} {added_molecules[mol_id]}"
                sym_new_idx = []
                for i, j in enumerate(sym):
                    if i != int(j):
                        sym_new_idx.append((i + start_mol, int(j) + start_mol))
                if len(sym_new_idx) > 0:
                    swaps.append(sym_new_idx)

            if len(swaps) > 0:
                molecule_symmetries.append(swaps)

            if return_physical_metrics:
                edge_index, (lower_bounds, upper_bounds, bond_mask, angle_mask) = (
                    slice_valid_index(
                        edge_index,
                        ccd_to_valid_id_array,
                        (lower_bounds, upper_bounds, bond_mask, angle_mask),
                    )
                )
                all_edge_index.append(edge_index + start_mol)
                all_lower_bounds.append(lower_bounds)
                all_upper_bounds.append(upper_bounds)
                all_bond_mask.append(bond_mask)
                all_angle_mask.append(angle_mask)

                chiral_atom_index, (chiral_check_mask, chiral_atom_orientations) = (
                    slice_valid_index(
                        chiral_atom_index,
                        ccd_to_valid_id_array,
                        (chiral_check_mask, chiral_atom_orientations),
                    )
                )
                all_chiral_atom_index.append(chiral_atom_index + start_mol)
                all_chiral_check_mask.append(chiral_check_mask)
                all_chiral_atom_orientations.append(chiral_atom_orientations)

                stereo_bond_index, (stereo_check_mask, stereo_bond_orientations) = (
                    slice_valid_index(
                        stereo_bond_index,
                        ccd_to_valid_id_array,
                        (stereo_check_mask, stereo_bond_orientations),
                    )
                )
                all_stereo_bond_index.append(stereo_bond_index + start_mol)
                all_stereo_check_mask.append(stereo_check_mask)
                all_stereo_bond_orientations.append(stereo_bond_orientations)

                aromatic_5_ring_index = slice_valid_index(
                    aromatic_5_ring_index, ccd_to_valid_id_array
                )
                aromatic_6_ring_index = slice_valid_index(
                    aromatic_6_ring_index, ccd_to_valid_id_array
                )
                planar_double_bond_index = slice_valid_index(
                    planar_double_bond_index, ccd_to_valid_id_array
                )
                all_aromatic_5_ring_index.append(aromatic_5_ring_index + start_mol)
                all_aromatic_6_ring_index.append(aromatic_6_ring_index + start_mol)
                all_planar_double_bond_index.append(
                    planar_double_bond_index + start_mol
                )

    if return_physical_metrics:
        if len(all_edge_index) > 0:
            all_edge_index = np.concatenate(all_edge_index, axis=1)
            all_lower_bounds = np.concatenate(all_lower_bounds, axis=0)
            all_upper_bounds = np.concatenate(all_upper_bounds, axis=0)
            all_bond_mask = np.concatenate(all_bond_mask, axis=0)
            all_angle_mask = np.concatenate(all_angle_mask, axis=0)

            all_chiral_atom_index = np.concatenate(all_chiral_atom_index, axis=1)
            all_chiral_check_mask = np.concatenate(all_chiral_check_mask, axis=0)
            all_chiral_atom_orientations = np.concatenate(
                all_chiral_atom_orientations, axis=0
            )

            all_stereo_bond_index = np.concatenate(all_stereo_bond_index, axis=1)
            all_stereo_check_mask = np.concatenate(all_stereo_check_mask, axis=0)
            all_stereo_bond_orientations = np.concatenate(
                all_stereo_bond_orientations, axis=0
            )

            all_aromatic_5_ring_index = np.concatenate(
                all_aromatic_5_ring_index, axis=1
            )
            all_aromatic_6_ring_index = np.concatenate(
                all_aromatic_6_ring_index, axis=1
            )
            all_planar_double_bond_index = np.empty(
                (6, 0), dtype=np.int64
            )  # TODO remove np.concatenate(all_planar_double_bond_index, axis=1)
        else:
            all_edge_index = np.empty((2, 0), dtype=np.int64)
            all_lower_bounds = np.array([], dtype=np.float32)
            all_upper_bounds = np.array([], dtype=np.float32)
            all_bond_mask = np.array([], dtype=bool)
            all_angle_mask = np.array([], dtype=bool)

            all_chiral_atom_index = np.empty((4, 0), dtype=np.int64)
            all_chiral_check_mask = np.array([], dtype=bool)
            all_chiral_atom_orientations = np.array([], dtype=bool)

            all_stereo_bond_index = np.empty((4, 0), dtype=np.int64)
            all_stereo_check_mask = np.array([], dtype=bool)
            all_stereo_bond_orientations = np.array([], dtype=bool)

            all_aromatic_5_ring_index = np.empty((5, 0), dtype=np.int64)
            all_aromatic_6_ring_index = np.empty((6, 0), dtype=np.int64)
            all_planar_double_bond_index = np.empty((6, 0), dtype=np.int64)

        features = {
            "ligand_symmetries": molecule_symmetries,
            "ligand_edge_index": torch.tensor(all_edge_index).long(),
            "ligand_edge_lower_bounds": torch.tensor(all_lower_bounds),
            "ligand_edge_upper_bounds": torch.tensor(all_upper_bounds),
            "ligand_edge_bond_mask": torch.tensor(all_bond_mask),
            "ligand_edge_angle_mask": torch.tensor(all_angle_mask),
            "ligand_chiral_atom_index": torch.tensor(all_chiral_atom_index).long(),
            "ligand_chiral_check_mask": torch.tensor(all_chiral_check_mask),
            "ligand_chiral_atom_orientations": torch.tensor(
                all_chiral_atom_orientations
            ),
            "ligand_stereo_bond_index": torch.tensor(all_stereo_bond_index).long(),
            "ligand_stereo_check_mask": torch.tensor(all_stereo_check_mask),
            "ligand_stereo_bond_orientations": torch.tensor(
                all_stereo_bond_orientations
            ),
            "ligand_aromatic_5_ring_index": torch.tensor(
                all_aromatic_5_ring_index
            ).long(),
            "ligand_aromatic_6_ring_index": torch.tensor(
                all_aromatic_6_ring_index
            ).long(),
            "ligand_planar_double_bond_index": torch.tensor(
                all_planar_double_bond_index
            ).long(),
        }
    else:
        features = {
            "ligand_symmetries": molecule_symmetries,
        }
    return features
