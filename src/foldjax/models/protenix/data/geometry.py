"""RDKit geometry guidance features, built per residue as upstream builds them.

Upstream's guided run featurizes with ``GeometryFeaturizer(atom_array,
exclude_std_residue=True)`` (Protenix ``data/core/geometry_featurizer.py``,
``data/inference/json_to_feature.py:396-402``; OpenDDE ships the same file).
That walks residues, not chains: standard polymer residues get no geometry
constraints at all, a component holding a metal atom is skipped, and every
other residue -- a ligand, an ion, a modified residue -- is featurized from
its own reference molecule, with the constraints kept only when all of their
atoms are present in the structure.
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Mapping
from itertools import chain, combinations
from typing import Any

import numpy as np
import rdkit
from rdkit import Chem
from rdkit.Chem.rdDistGeom import GetExperimentalTorsions, GetMoleculeBoundsMatrix
from rdkit.Chem.rdMolTransforms import GetDihedralRad

#: Protenix ``data/constants.py`` ``STD_RESIDUES``: the 20 amino acids and
#: UNK, then RNA and DNA (with their unknown N/DN).
STD_RESIDUES = frozenset(
    {
        "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
        "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
        "UNK", "A", "G", "C", "U", "N", "DA", "DG", "DC", "DT", "DN",
    }
)  # fmt: skip
#: Protenix ``geometry_featurizer.py`` ``METAL_ATOMIC_NUMBERS``.
METAL_ATOMIC_NUMBERS = frozenset(
    chain(
        (3, 4),
        range(11, 14),
        range(19, 32),
        range(37, 52),
        range(55, 85),
        range(87, 119),
    )
)
#: Upstream names a SMILES or FILE_ ligand ``l01``..``l99`` and featurizes it
#: from the input molecule; every other residue name is a CCD code.
_INPUT_LIGAND = re.compile(r"l\d\d")
_ANNOTATIONS = (
    "output_atom_name",
    "output_atom_res_name",
    "output_atom_res_id",
    "output_atom_chain_id",
    "output_atom_polymer_type",
)
#: Each constraint family's atom-index key and the per-constraint values that
#: travel with it (upstream's ``RDKIT_GEOMETRY_FEATURES`` grouping).
_FAMILIES = {
    "pairwise_distance": (
        "pairwise_distance_upper_bound",
        "pairwise_distance_lower_bound",
        "pairwise_distance_is_bond",
        "pairwise_distance_is_angle",
    ),
    "experimental_torsion": (
        "experimental_torsion_force_constant",
        "experimental_torsion_sign",
    ),
    "linear_triple_bond": (),
    "chiral": ("chiral_orientation",),
    "stereo_bond": ("stereo_bond_orientation",),
    "planar_improper": ("planar_improper_is_carbonyl",),
}


def prepare_tfg_features(
    features: Mapping[str, Any], *, assets: Any = None
) -> dict[str, Any]:
    """Return ``features`` augmented with every JAX TFG feature contract.

    ``assets`` is the run's ``FeaturizerAssets``: a CCD component (a CCD
    ligand or a modified residue) reads its reference molecule from the same
    ``components.cif.rdkit_mol.pkl`` the featurizer used. A job of standard
    polymer residues and SMILES/FILE_ ligands never opens that cache.
    """

    from foldjax.models.protenix.data.featurize_json import _assets_in_force

    with _assets_in_force(assets):
        return _prepare_tfg_features(features)


def _prepare_tfg_features(features: Mapping[str, Any]) -> dict[str, Any]:
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

    missing = [key for key in _ANNOTATIONS if key not in features]
    if missing:
        raise ValueError(
            "TFG geometry needs the featurizer's per-atom residue annotations; "
            f"missing {missing}"
        )
    annotations = {key: np.asarray(features[key]) for key in _ANNOTATIONS}
    for key, value in annotations.items():
        if value.shape != (n_atom,):
            raise ValueError(f"{key} must have shape (n_atom,)")
    atom_names = annotations["output_atom_name"].astype(str)
    res_names = annotations["output_atom_res_name"].astype(str)
    symbols = _elements(features, n_atom)
    periodic_table = Chem.GetPeriodicTable()

    accumulated = _empty_rdkit_lists()
    provenance: list[dict[str, Any]] = []
    ccd_geometry: dict[str, tuple[dict[str, Any], dict[str, int]] | None] = {}
    for start, stop in _residue_spans(annotations):
        res_name = str(res_names[start])
        hetero = str(annotations["output_atom_polymer_type"][start]) == "non-polymer"
        # Upstream's `mse_to_met` makes every MSE a non-hetero MET before this.
        if res_name == "MSE" or (res_name in STD_RESIDUES and not hetero):
            continue
        record: dict[str, Any] = {
            "chain_id": str(annotations["output_atom_chain_id"][start]),
            "res_id": int(annotations["output_atom_res_id"][start]),
            "res_name": res_name,
            "atom_indices": [start, stop],
            "status": "featurized",
            "error": None,
        }
        provenance.append(record)
        # Upstream checks the reference molecule's atoms; a metal present here
        # is one of them, and finding it first spares a CCD cache load.
        if any(_atomic_number(periodic_table, symbol) in METAL_ATOMIC_NUMBERS
               for symbol in symbols[start:stop]):  # fmt: skip
            record["status"] = "metal"
            continue
        if _INPUT_LIGAND.fullmatch(res_name):
            atom_indices = np.arange(start, stop, dtype=np.int64)
            local_to_global = {
                local: int(global_idx) for local, global_idx in enumerate(atom_indices)
            }
            within = np.isin(bonds[:, 0], atom_indices) & np.isin(
                bonds[:, 1], atom_indices
            )
            try:
                mol = _reconstruct_rdkit_mol(
                    features,
                    atom_indices,
                    bonds[within],
                    orders[within],
                    stereos[within],
                    coordinates,
                )
                # Upstream's SMILES molecule carries explicit hydrogens
                # (`json_parser.py` `smiles_to_atom_info`), and some torsion
                # patterns -- a secondary amide's -- match differently with them.
                mol = Chem.AddHs(mol, addCoords=True)
                Chem.GetSymmSSSR(mol)
                local = _extract_rdkit_geometry(mol)
            except Exception as exc:
                local = _failed(record, exc)
        else:
            if res_name not in ccd_geometry:
                ccd_geometry[res_name] = _ccd_geometry(res_name)
            cached = ccd_geometry[res_name]
            if cached is None:
                record["status"] = "unknown component"
                continue
            local, atom_map = cached
            if local.get("_metal"):
                record["status"] = "metal"
                continue
            if local.get("_error") is not None:
                local = _failed(record, local["_error"])
            local_to_global = {}
            for global_idx in range(start, stop):
                name = str(atom_names[global_idx])
                if name not in atom_map:
                    raise ValueError(
                        f"atom {name!r} is not in CCD component {res_name!r}"
                    )
                local_to_global[int(atom_map[name])] = global_idx
        _accumulate_rdkit_geometry(accumulated, local, local_to_global)
    result.update(_finalize_rdkit_geometry(accumulated))
    result["geometry_unsupported"] = any(
        record["status"] == "failed" for record in provenance
    )
    result["geometry_provenance"] = {
        "backend": "rdkit",
        "rdkit_version": rdkit.__version__,
        "residues": provenance,
    }
    return result


def _residue_spans(annotations: Mapping[str, np.ndarray]) -> list[tuple[int, int]]:
    """Contiguous atom runs of one residue, as biotite's residue starts."""

    keys = (
        annotations["output_atom_chain_id"].astype(str),
        annotations["output_atom_res_id"].astype(np.int64),
        annotations["output_atom_res_name"].astype(str),
    )
    n_atom = len(keys[0])
    if n_atom == 0:
        return []
    change = np.zeros((n_atom,), dtype=bool)
    change[0] = True
    for key in keys:
        change[1:] |= key[1:] != key[:-1]
    starts = np.flatnonzero(change).tolist()
    return list(zip(starts, [*starts[1:], n_atom], strict=True))


def _atomic_number(periodic_table: Any, symbol: str) -> int:
    symbol = str(symbol).strip()
    try:
        return int(
            periodic_table.GetAtomicNumber(symbol[:1].upper() + symbol[1:].lower())
        )
    except RuntimeError:
        return 0


def _ccd_geometry(code: str) -> tuple[dict[str, Any], dict[str, int]] | None:
    """Upstream's ``get_ccd_geometry_features`` on the CCD RDKit cache's molecule.

    The cached molecule keeps its hydrogens and leaving atoms and is not
    re-sanitized; conformer 0 orients chirality and E/Z, as upstream reads it.
    ``None`` is an unknown code, which upstream skips.
    """

    from foldjax.models.protenix.data.featurize_json import _external_ccd_molecule

    try:
        source = _external_ccd_molecule(code, missing_ok=True)
    except ValueError as exc:
        # A vendored ligand featurizes without the cache; its geometry cannot.
        raise ValueError(
            f"TFG geometry for CCD component {code!r} needs the official "
            "components.cif.rdkit_mol.pkl (fetched with the model; or set "
            f"PROTENIX_CCD_RDKIT_MOL_FILE, or OpenDDE's --ccd-rdkit-cache): {exc}"
        ) from exc
    if source is None or source.GetNumAtoms() == 0:
        return None
    atom_map = {str(name): int(index) for name, index in source.atom_map.items()}
    if any(atom.GetAtomicNum() in METAL_ATOMIC_NUMBERS for atom in source.GetAtoms()):
        return {"_metal": True}, atom_map
    mol = Chem.Mol(source)
    try:
        mol.UpdatePropertyCache(strict=False)
        Chem.AssignStereochemistry(mol, force=True, cleanIt=True)
        Chem.GetSymmSSSR(mol)
        return _extract_rdkit_geometry(mol), atom_map
    except Exception as exc:
        return {"_error": exc}, atom_map


def _failed(record: dict[str, Any], exc: BaseException) -> dict[str, list[Any]]:
    """Upstream's fallback: warn, and give the component no geometry terms."""

    record["status"] = "failed"
    record["error"] = f"{type(exc).__name__}: {exc}"
    warnings.warn(
        f"TFG geometry for {record['res_name']} (chain {record['chain_id']}, "
        f"residue {record['res_id']}) failed ({record['error']}); it gets no "
        "geometry constraints",
        stacklevel=4,
    )
    return _empty_rdkit_lists()


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
    local_to_global: Mapping[int, int],
) -> None:
    """Keep the constraints whose atoms are all present, in global indices."""

    for family, value_keys in _FAMILIES.items():
        index_key = f"{family}_index"
        kept = []
        for row, atoms in enumerate(local[index_key]):
            if all(int(atom) in local_to_global for atom in atoms):
                kept.append(row)
                accumulated[index_key].append(
                    [local_to_global[int(atom)] for atom in atoms]
                )
        for key in value_keys:
            accumulated[key].extend(local[key][row] for row in kept)


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
