"""Features for a query-level ligand ``pocket_constraint``, without PyTorch.

A NumPy transcription of OpenFold3 v0.5.0
``core/data/pipelines/featurization/pocket_constraints.py`` (commit c4771653):
the same feature names, values, dtypes and RDKit conformer recipe, returned as
NumPy arrays instead of tensors. The sampler that reads them is
:mod:`foldjax.models.openfold3.models.pocket_constraints`.

Upstream runs this for every query that carries ``pocket_constraint``
(``PocketSamplingSettings.enabled`` defaults to true and FoldJAX exposes no
switch for it), so the presence of these features is what turns pocket-guided
sampling on.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

VDW_RADII = {
    "H": 1.20,
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "F": 1.47,
    "P": 1.80,
    "S": 1.80,
    "CL": 1.75,
    "BR": 1.85,
    "I": 1.98,
}
DEFAULT_VDW_RADIUS = VDW_RADII["C"]

#: Per-atom arrays the sampler reads inside the compiled program.
POCKET_SAMPLING_ARRAY_FEATURES = (
    "pocket_sampling_ligand_atom_mask",
    "pocket_sampling_pocket_atom_mask",
    "pocket_sampling_vdw_radii",
)
#: Present only when RDKit produced at least one ligand conformer.
POCKET_SAMPLING_CONFORMERS = "pocket_sampling_conformer_rels"
#: One-element settings. They fix loop bounds and the refinement start step,
#: so the sampler reads them on the host into a static
#: :class:`~foldjax.models.openfold3.models.pocket_constraints.PocketSamplingConfig`.
POCKET_SAMPLING_SCALAR_FEATURES = (
    "pocket_sampling_enabled",
    "pocket_sampling_contact_distance",
    "pocket_sampling_num_parents",
    "pocket_sampling_candidates",
    "pocket_sampling_start_frac",
    "pocket_sampling_ligand_jitter",
    "pocket_sampling_center_jitter",
    "pocket_sampling_surface_jitter",
    "pocket_sampling_vdw_buffer",
    "pocket_sampling_diversity_rmsd",
)
POCKET_SAMPLING_FEATURES = (
    *POCKET_SAMPLING_SCALAR_FEATURES,
    *POCKET_SAMPLING_ARRAY_FEATURES,
    POCKET_SAMPLING_CONFORMERS,
)

_INTEGER_SCALARS = ("pocket_sampling_num_parents", "pocket_sampling_candidates")


def _molecule_type():
    from foldjax.models.openfold3._upstream.openfold3.core.data.resources.residues import (  # noqa: E501
        MoleculeType,
    )

    return MoleculeType


def _resolve_ligand_reference_molecule(
    query: Any,
    processed_reference_molecules: list[Any],
    ligand_chain_id: str,
) -> Any | None:
    """Find the processed reference molecule matching a query ligand chain."""
    molecule_type = _molecule_type()
    ref_mol_idx = 0
    for chain in query.chains:
        for chain_id in chain.chain_ids:
            match chain.molecule_type:
                case molecule_type.PROTEIN | molecule_type.DNA | molecule_type.RNA:
                    if chain.sequence is None:
                        raise ValueError(
                            f"Chain {chain_id} has no sequence but is "
                            "required to resolve pocket sampling reference molecules"
                        )
                    ref_mol_idx += len(chain.sequence)
                case molecule_type.LIGAND:
                    if ref_mol_idx >= len(processed_reference_molecules):
                        raise ValueError(
                            "Not enough processed reference molecules to resolve "
                            f"ligand chain {ligand_chain_id!r}"
                        )
                    if chain_id == ligand_chain_id:
                        return processed_reference_molecules[ref_mol_idx]
                    ref_mol_idx += 1
                case _:
                    raise ValueError(
                        f"Unsupported molecule type: {chain.molecule_type}"
                    )
    return None


def _atom_order_from_reference_molecule(
    processed_reference_molecule: Any,
    ligand_atom_array: Any,
) -> list[int]:
    """Map OF3 ligand atom order onto annotated reference-molecule atom indices."""
    from foldjax.models.openfold3._upstream.openfold3.core.data.primitives.structure.labels import (  # noqa: E501
        uniquify_ids,
    )

    mol = processed_reference_molecule.mol
    atom_names = np.asarray(
        [atom.GetProp("annot_atom_name") for atom in mol.GetAtoms()], dtype=object
    )
    atom_elements = np.asarray(
        [atom.GetSymbol().upper() for atom in mol.GetAtoms()], dtype=object
    )
    in_crop_mask = np.asarray(processed_reference_molecule.in_crop_mask, dtype=bool)

    ref_keys = np.asarray(uniquify_ids(atom_names[in_crop_mask].tolist()))
    ligand_keys = np.asarray(uniquify_ids(ligand_atom_array.atom_name.tolist()))

    if len(ref_keys) != len(ligand_keys) or set(ref_keys) != set(ligand_keys):
        raise ValueError(
            "Reference molecule atom names do not match OF3 ligand atom names"
        )

    ref_indices_by_key = {
        key: int(idx) for key, idx in zip(ref_keys, np.flatnonzero(in_crop_mask))
    }
    ordered_indices = [ref_indices_by_key[key] for key in ligand_keys]

    ref_ordered_elements = atom_elements[ordered_indices]
    ligand_elements = np.asarray(
        [str(element).upper() for element in ligand_atom_array.element], dtype=object
    )
    if not np.array_equal(ref_ordered_elements, ligand_elements):
        raise ValueError(
            "Reference molecule atom elements do not match OF3 ligand atom order"
        )

    return ordered_indices


def create_pocket_sampling_features(
    query: Any,
    atom_array: Any,
    processed_reference_molecules: list[Any] | None = None,
    settings: Any | None = None,
) -> dict[str, np.ndarray]:
    """Create sampler features for pocket proposal and partial-diffusion refinement.

    Unbatched, like the rest of the raw featurizer; ``{}`` when the query has
    no ``pocket_constraint`` or ``settings.enabled`` is false. ``settings`` is
    upstream's ``PocketSamplingSettings``; ``None`` takes its released
    defaults. RDKit conformer failures are logged and leave the sampler with
    the parent ligand conformations only, as upstream does.
    """
    from foldjax.models.openfold3._upstream.openfold3.core.config.pocket_sampling_config import (  # noqa: E501
        PocketSamplingSettings,
    )

    molecule_type = _molecule_type()
    settings = settings if settings is not None else PocketSamplingSettings()
    if query.pocket_constraint is None or not settings.enabled:
        return {}

    constraint = query.pocket_constraint
    lig_mask = (atom_array.chain_id == constraint.ligand_chain_id) & (
        atom_array.molecule_type_id == molecule_type.LIGAND
    )
    pocket_mask = np.zeros(len(atom_array), dtype=bool)
    for residue in constraint.pocket_residues:
        residue_mask = (atom_array.chain_id == residue.chain_id) & (
            atom_array.res_id == residue.residue_id
        )
        if not residue_mask.any():
            raise ValueError(
                "Pocket constraint residue "
                f"{residue.chain_id}:{residue.residue_id} does not match any atoms"
            )
        pocket_mask |= residue_mask
    if not lig_mask.any() or not pocket_mask.any():
        raise ValueError("Pocket sampling requested but ligand or pocket mask is empty")

    def _vdw_radius(element: object) -> float:
        if not isinstance(element, str):
            return DEFAULT_VDW_RADIUS
        return VDW_RADII.get(element.upper(), DEFAULT_VDW_RADIUS)

    features = {
        "pocket_sampling_enabled": np.asarray([True], dtype=bool),
        "pocket_sampling_ligand_atom_mask": lig_mask.astype(np.float32),
        "pocket_sampling_pocket_atom_mask": pocket_mask.astype(np.float32),
        "pocket_sampling_vdw_radii": np.asarray(
            [_vdw_radius(e) for e in atom_array.element], dtype=np.float32
        ),
        "pocket_sampling_contact_distance": np.asarray(
            [float(constraint.max_distance)], dtype=np.float32
        ),
        "pocket_sampling_num_parents": np.asarray(
            [settings.num_parents], dtype=np.int64
        ),
        "pocket_sampling_candidates": np.asarray([settings.candidates], dtype=np.int64),
        "pocket_sampling_start_frac": np.asarray(
            [settings.noise_frac], dtype=np.float32
        ),
        "pocket_sampling_ligand_jitter": np.asarray(
            [settings.ligand_jitter], dtype=np.float32
        ),
        "pocket_sampling_center_jitter": np.asarray(
            [settings.center_jitter], dtype=np.float32
        ),
        "pocket_sampling_surface_jitter": np.asarray(
            [settings.surface_jitter], dtype=np.float32
        ),
        "pocket_sampling_vdw_buffer": np.asarray(
            [settings.vdw_buffer], dtype=np.float32
        ),
        "pocket_sampling_diversity_rmsd": np.asarray(
            [settings.diversity_rmsd], dtype=np.float32
        ),
    }

    n_conformers = settings.rdkit_num_conformers
    if n_conformers > 0 and processed_reference_molecules is not None:
        conformer_rng = settings.rdkit_conformer_rng
        conformer_prune_rmsd = settings.rdkit_conformer_prune_rmsd
        conformer_max_iters = settings.rdkit_conformer_max_iters
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem

            processed_ligand_mol = _resolve_ligand_reference_molecule(
                query=query,
                processed_reference_molecules=processed_reference_molecules,
                ligand_chain_id=constraint.ligand_chain_id,
            )
            if processed_ligand_mol is None:
                raise ValueError(
                    f"No processed reference molecule found for ligand chain "
                    f"{constraint.ligand_chain_id!r}"
                )
            ligand_atom_array = atom_array[lig_mask]
            heavy_indices = _atom_order_from_reference_molecule(
                processed_reference_molecule=processed_ligand_mol,
                ligand_atom_array=ligand_atom_array,
            )

            mol = Chem.Mol(processed_ligand_mol.mol)
            mol.RemoveAllConformers()
            mol_h = Chem.AddHs(mol)
            for atom_idx in heavy_indices:
                atom = mol_h.GetAtomWithIdx(atom_idx)
                if atom.GetAtomicNum() <= 1 or not atom.HasProp("annot_atom_name"):
                    raise ValueError(
                        "RDKit hydrogen expansion changed reference atom indices"
                    )
            params = AllChem.ETKDGv3()
            params.randomSeed = conformer_rng
            params.pruneRmsThresh = conformer_prune_rmsd
            conf_ids = list(
                AllChem.EmbedMultipleConfs(
                    mol_h,
                    numConfs=n_conformers,
                    params=params,
                )
            )
            if AllChem.MMFFHasAllMoleculeParams(mol_h):
                for conf_id in conf_ids:
                    AllChem.MMFFOptimizeMolecule(
                        mol_h, confId=int(conf_id), maxIters=conformer_max_iters
                    )
            else:
                for conf_id in conf_ids:
                    AllChem.UFFOptimizeMolecule(
                        mol_h, confId=int(conf_id), maxIters=conformer_max_iters
                    )

            conformer_rels = []
            for conf_id in conf_ids:
                conf = mol_h.GetConformer(int(conf_id))
                conf_coords = np.asarray(
                    [
                        [
                            conf.GetAtomPosition(idx).x,
                            conf.GetAtomPosition(idx).y,
                            conf.GetAtomPosition(idx).z,
                        ]
                        for idx in heavy_indices
                    ],
                    dtype=np.float32,
                )
                conformer_rels.append(
                    conf_coords - conf_coords.mean(axis=0, keepdims=True)
                )
            if conformer_rels:
                features[POCKET_SAMPLING_CONFORMERS] = np.stack(
                    conformer_rels, axis=0
                ).astype(np.float32)
            logger.info(
                "[pocket_sampling_build] rdkit_conformers=%s/%s",
                len(conformer_rels),
                n_conformers,
            )
        except Exception as exc:
            logger.warning(
                "[pocket_sampling_build] RDKit conformer generation failed; "
                "using parent ligand conformations only: %s: %s",
                type(exc).__name__,
                exc,
            )

    return features


def has_pocket_sampling_features(features: Mapping[str, Any]) -> bool:
    """Whether a feature mapping asks for pocket-guided sampling."""
    return any(name in features for name in POCKET_SAMPLING_FEATURES)


def validate_pocket_sampling_features(features: Mapping[str, Any]) -> None:
    """Check batched pocket features against the atom axis they index.

    Upstream builds them as a complete set and its sampler trusts that
    (``core/model/structure/pocket_constraints.py`` module docstring). An
    archive is a trust boundary, so the set, shapes and dtypes are checked
    here instead of failing as a shape error inside the compiled sampler.
    """
    if not has_pocket_sampling_features(features):
        return
    required = (*POCKET_SAMPLING_SCALAR_FEATURES, *POCKET_SAMPLING_ARRAY_FEATURES)
    missing = [name for name in required if name not in features]
    if missing:
        raise ValueError(f"OpenFold3 pocket sampling features are missing: {missing}")
    atoms = np.asarray(features["atom_mask"]).shape[-1]
    for name in POCKET_SAMPLING_SCALAR_FEATURES:
        value = np.asarray(features[name])
        if value.shape != (1, 1):
            raise ValueError(
                f"OpenFold3 feature {name!r} must have shape (1, 1); got {value.shape}"
            )
    if not bool(np.asarray(features["pocket_sampling_enabled"]).item()):
        raise ValueError(
            "OpenFold3 pocket_sampling_enabled is false; drop the pocket "
            "sampling features instead of disabling them"
        )
    for name in _INTEGER_SCALARS:
        value = np.asarray(features[name])
        if not np.issubdtype(value.dtype, np.integer) or int(value.item()) < 1:
            raise ValueError(f"OpenFold3 feature {name!r} must be a positive integer")
    for name in POCKET_SAMPLING_ARRAY_FEATURES:
        value = np.asarray(features[name])
        if value.shape != (1, atoms) or value.dtype != np.dtype(np.float32):
            raise ValueError(
                f"OpenFold3 feature {name!r} must be float32 with shape "
                f"(1, {atoms}); got {value.dtype} {value.shape}"
            )
    atom_mask = np.asarray(features["atom_mask"]).astype(bool)
    for name in (
        "pocket_sampling_ligand_atom_mask",
        "pocket_sampling_pocket_atom_mask",
    ):
        mask = np.asarray(features[name])
        if not np.isin(mask, (0.0, 1.0)).all():
            raise ValueError(f"OpenFold3 feature {name!r} must be a 0/1 mask")
        if not mask.astype(bool).any() or np.any(mask.astype(bool) & ~atom_mask):
            raise ValueError(
                f"OpenFold3 feature {name!r} must select at least one real atom"
            )
    ligand_atoms = int(np.asarray(features["pocket_sampling_ligand_atom_mask"]).sum())
    if POCKET_SAMPLING_CONFORMERS in features:
        conformers = np.asarray(features[POCKET_SAMPLING_CONFORMERS])
        if (
            conformers.ndim != 4
            or conformers.shape[0] != 1
            or conformers.shape[1] < 1
            or conformers.shape[2:] != (ligand_atoms, 3)
            or conformers.dtype != np.dtype(np.float32)
        ):
            raise ValueError(
                f"OpenFold3 feature {POCKET_SAMPLING_CONFORMERS!r} must be float32 "
                f"with shape (1, conformers, {ligand_atoms}, 3); got "
                f"{conformers.dtype} {conformers.shape}"
            )


def pocket_sampling_config(features: Mapping[str, Any]):
    """Read the static sampler settings off concrete batched features.

    ``None`` when the features carry no pocket constraint. The result is a
    field of ``InferenceConfig``: it fixes loop bounds and gather sizes, so it
    must be known when the program is traced.
    """
    if not has_pocket_sampling_features(features):
        return None
    validate_pocket_sampling_features(features)
    from foldjax.models.openfold3.models.pocket_constraints import (
        PocketSamplingConfig,
    )

    def scalar(name: str):
        return np.asarray(features[name]).reshape(())

    conformers = features.get(POCKET_SAMPLING_CONFORMERS)
    return PocketSamplingConfig(
        n_ligand_atoms=int(
            np.asarray(features[POCKET_SAMPLING_ARRAY_FEATURES[0]]).sum()
        ),
        n_pocket_atoms=int(
            np.asarray(features[POCKET_SAMPLING_ARRAY_FEATURES[1]]).sum()
        ),
        n_conformers=0 if conformers is None else int(np.asarray(conformers).shape[1]),
        num_parents=int(scalar("pocket_sampling_num_parents")),
        candidates=int(scalar("pocket_sampling_candidates")),
        start_frac=float(scalar("pocket_sampling_start_frac")),
        ligand_jitter=float(scalar("pocket_sampling_ligand_jitter")),
        center_jitter=float(scalar("pocket_sampling_center_jitter")),
        surface_jitter=float(scalar("pocket_sampling_surface_jitter")),
        vdw_buffer=float(scalar("pocket_sampling_vdw_buffer")),
        diversity_rmsd=float(scalar("pocket_sampling_diversity_rmsd")),
        contact_distance=float(scalar("pocket_sampling_contact_distance")),
    )
