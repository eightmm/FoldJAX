"""Add OpenDDE structural-token metadata to Protenix-JAX JSON features.

This module ports the structural tokenization and frame rules from pinned
OpenDDE without importing Torch or the upstream ``opendde`` package. Standard
polymer residues use residue tokens, while CCD polymer modifications follow
upstream's atom-token rules (with MSE normalized to standard MET).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from copy import deepcopy
from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from foldjax.models.protenix.data.featurize_json import (
    _local_atom_geometry,
    featurize_protein_json,
)

STRUCTURAL_TOKEN_ROLES = {
    "atom": 0,
    "protein_bb": 1,
    "protein_sc": 2,
    "dna_bb": 3,
    "dna_base": 4,
    "rna_bb": 5,
    "rna_base": 6,
}

#: Fields OpenDDE's shipped inference defaults discard, and the flag that does
#: it. `use_template=False` and `use_rna_msa=False`
#: (OpenDDE/opendde/config/inference_defaults.py:28-29) mean upstream's
#: featurizer never reads these, so a job that carries one is predicted from the
#: sequence alone. The Protenix featurizer this port shares does read them, so
#: without this the two implementations quietly diverge on exactly the inputs a
#: user went to the trouble of supplying -- different pair representation, and a
#: template stack running every recycle. Dropped rather than refused, because
#: the goal is upstream's answer; said out loud, because upstream's silence
#: about it is the part worth not copying.
_OPENDDE_IGNORED_FIELDS: dict[str, tuple[tuple[str, str], ...]] = {
    "proteinChain": (("templatesPath", "use_template=False"),),
    "rnaSequence": (("unpairedMsaPath", "use_rna_msa=False"),),
}

#: One warning per field per document, not per chain.
_warned_dropped: set[str] = set()


def _drop_fields_opendde_ignores(
    kind: str,
    info: dict[str, Any],
    *,
    use_template: bool,
    use_rna_msa: bool,
) -> None:
    """Remove what upstream's defaults would have ignored, and say which."""
    import warnings

    for key, flag in _OPENDDE_IGNORED_FIELDS.get(kind, ()):
        if key == "templatesPath" and use_template:
            continue
        if key == "unpairedMsaPath" and use_rna_msa:
            continue
        if not info.pop(key, None):
            continue
        if key in _warned_dropped:
            continue
        _warned_dropped.add(key)
        warnings.warn(
            f"OpenDDE ignores {key!r} on {kind}: its shipped inference default "
            f"is {flag} (config/inference_defaults.py). Dropping it here so the "
            "prediction matches upstream's; pass it to Protenix instead if you "
            "want it used.",
            RuntimeWarning,
            stacklevel=3,
        )


_NO_TWIN_TOKEN_IDX = -1
_PROTEIN_BACKBONE_ATOMS = frozenset({"N", "CA", "C", "O", "OXT"})
_NUCLEIC_BACKBONE_ATOMS = frozenset(
    {
        "P",
        "OP1",
        "OP2",
        "OP3",
        "O1P",
        "O2P",
        "O3P",
        "O5'",
        "C5'",
        "C4'",
        "O4'",
        "C3'",
        "O3'",
        "C2'",
        "O2'",
        "C1'",
        "O5*",
        "C5*",
        "C4*",
        "O4*",
        "C3*",
        "O3*",
        "C2*",
        "O2*",
        "C1*",
        "O5T",
        "O3T",
    }
)
_PURINE_RESTYPE_INDICES = frozenset({21, 22, 26, 27})
_PYRIMIDINE_RESTYPE_INDICES = frozenset({23, 24, 28, 29})
_CONCRETE_PROTEIN_ALPHABET = frozenset("ARNDCQEGHILKMFPSTWYV")
_CONCRETE_DNA_ALPHABET = frozenset("ACGT")
_CONCRETE_RNA_ALPHABET = frozenset("ACGU")


def load_jobs(path: str | Path) -> list[dict[str, Any]]:
    """Load and validate all jobs from an OpenDDE inference JSON file."""

    with Path(path).open("r", encoding="utf-8") as handle:
        jobs = json.load(handle)
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("input JSON must be a non-empty top-level list")
    if not all(isinstance(job, dict) for job in jobs):
        raise ValueError("every input JSON job must be an object")
    return jobs


def featurize_opendde_json(
    job: dict[str, Any],
    *,
    base_dir: str | Path | None = None,
    n_queries: int = 32,
    n_keys: int = 128,
    max_msa_depth: int = 16384,
    seed: int | None = None,
    augment_reference: bool = True,
    use_template: bool = False,
    use_rna_msa: bool = False,
) -> dict[str, Any]:
    """Build Torch-free residue and OpenDDE structural-token input features.

    The underlying sequence/reference/MSA features come from Protenix-JAX.  This
    wrapper adds the exact metadata needed by ``opendde_infer_static`` for
    proteins, DNA, RNA, CCD polymer modifications, and atom-tokenized ligands.
    """

    prepared_job = _prepare_job(
        job,
        base_dir=base_dir,
        use_template=use_template,
        use_rna_msa=use_rna_msa,
    )
    _validate_supported_polymers(prepared_job)
    resolved_seed = _resolve_seed(job, seed)
    features = featurize_protein_json(
        prepared_job,
        base_dir=base_dir,
        n_queries=n_queries,
        n_keys=n_keys,
        max_msa_depth=max_msa_depth,
        seed=resolved_seed,
        center_reference=False,
        augment_reference=False,
    )
    _prepare_reference_features(
        features,
        seed=resolved_seed,
        augment=augment_reference,
        n_queries=n_queries,
        n_keys=n_keys,
    )
    return _add_open_dde_metadata(features)


def _prepare_job(
    job: dict[str, Any],
    *,
    base_dir: str | Path | None,
    use_template: bool = False,
    use_rna_msa: bool = False,
) -> dict[str, Any]:
    prepared = deepcopy(job)
    prepared.pop("assembly_id", None)

    sequences = prepared.get("sequences")
    if not isinstance(sequences, list):
        return prepared
    _warned_dropped.clear()
    for entry in sequences:
        if not isinstance(entry, dict) or len(entry) != 1:
            continue
        kind, info = next(iter(entry.items()))
        if not isinstance(info, dict):
            continue
        path_keys = {
            "proteinChain": (
                "pairedMsaPath",
                "unpairedMsaPath",
                "templatesPath",
            ),
            "rnaSequence": ("unpairedMsaPath",),
        }.get(kind, ())
        if base_dir is not None:
            for key in path_keys:
                value = info.get(key)
                if isinstance(value, (str, Path)) and str(value):
                    info[key] = str(_resolve_asset_path(value, base_dir=base_dir))
        _drop_fields_opendde_ignores(
            kind,
            info,
            use_template=use_template,
            use_rna_msa=use_rna_msa,
        )
        if kind == "proteinChain" and isinstance(info.get("msa"), dict):
            directory = info["msa"].get("precomputed_msa_dir")
            if isinstance(directory, (str, Path)) and str(directory):
                if base_dir is not None:
                    info["msa"]["precomputed_msa_dir"] = str(
                        _resolve_asset_path(directory, base_dir=base_dir)
                    )
        if kind == "ligand" and isinstance(info.get("ligand"), str):
            ligand = info["ligand"]
            if ligand.startswith("FILE_") and base_dir is not None:
                info["ligand"] = "FILE_" + str(
                    _resolve_asset_path(ligand[5:], base_dir=base_dir)
                )
    return prepared


def _resolve_asset_path(value: str | Path, *, base_dir: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path

    json_parent = Path(base_dir).expanduser().resolve()
    preferred = json_parent / path
    if preferred.exists():
        return preferred

    if path.parts and path.parts[0] == "examples":
        for ancestor in json_parent.parents:
            candidate = ancestor / path
            if candidate.exists():
                return candidate
    return preferred


def _resolve_seed(job: Mapping[str, Any], seed: int | None) -> int:
    if seed is not None:
        return int(seed)
    model_seeds = job.get("modelSeeds")
    if isinstance(model_seeds, list) and model_seeds:
        return int(model_seeds[0])
    return 101


@lru_cache(maxsize=1)
def _standard_reference_tables() -> dict[str, np.ndarray]:
    resource = files("foldjax.models.opendde.data").joinpath(
        "opendde_std_reference.npz"
    )
    with resource.open("rb") as handle, np.load(handle, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def _prepare_reference_features(
    features: dict[str, Any],
    *,
    seed: int,
    augment: bool,
    n_queries: int,
    n_keys: int,
) -> None:
    tables = _standard_reference_tables()
    atom_names = np.asarray(features["output_atom_name"], dtype=str)
    residue_names = np.asarray(features["output_atom_res_name"], dtype=str)
    ref_space_uid = np.asarray(features["ref_space_uid"], dtype=np.int64)
    ref_pos = np.asarray(features["ref_pos"], dtype=np.float32).copy()
    ref_mask = np.asarray(features["ref_mask"], dtype=np.int64).copy()
    ref_charge = np.asarray(features["ref_charge"], dtype=np.int64).copy()
    atom_to_token = np.asarray(features["atom_to_token_idx"], dtype=np.int64)
    reference_is_mse = np.asarray(
        features.get("token_reference_is_mse", np.zeros(features["restype"].shape[0])),
        dtype=np.int64,
    )
    random_state = np.random.RandomState(seed)

    for uid in np.unique(ref_space_uid):
        atom_indices = np.flatnonzero(ref_space_uid == uid)
        code = str(residue_names[atom_indices[0]])
        names_key = f"{code}_names"
        use_standard_table = names_key in tables and not np.any(
            reference_is_mse[atom_to_token[atom_indices]]
        )
        if use_standard_table:
            reference_names = np.asarray(tables[names_key], dtype=str)
            name_to_index = {name: index for index, name in enumerate(reference_names)}
            try:
                selected = np.asarray(
                    [name_to_index[name] for name in atom_names[atom_indices]],
                    dtype=np.int64,
                )
            except KeyError as exc:
                raise ValueError(
                    f"official reference table for {code} lacks atom {exc.args[0]!r}"
                ) from exc
            points = np.asarray(tables[f"{code}_coord"], dtype=np.float32)[selected]
            ref_mask[atom_indices] = np.asarray(tables[f"{code}_mask"], dtype=np.int64)[
                selected
            ]
            ref_charge[atom_indices] = np.asarray(
                tables[f"{code}_charge"], dtype=np.int64
            )[selected]
        else:
            points = ref_pos[atom_indices]
        points = points - points.mean(axis=0)
        if augment:
            translation = random_state.uniform(-1.0, 1.0, size=3)
            rotation = Rotation.random(random_state=random_state).as_matrix()
            points = points @ rotation.T + translation
        ref_pos[atom_indices] = points.astype(np.float32)

    d_lm, v_lm, pad_info = _local_atom_geometry(
        ref_pos,
        ref_space_uid,
        n_queries=n_queries,
        n_keys=n_keys,
    )
    features.update(
        {
            "ref_pos": ref_pos,
            "ref_mask": ref_mask,
            "ref_charge": ref_charge,
            "ref_element": np.asarray(features["ref_element"], dtype=np.int64),
            "ref_atom_name_chars": np.asarray(
                features["ref_atom_name_chars"], dtype=np.int64
            ),
            "distogram_rep_atom_mask": np.asarray(
                features["distogram_rep_atom_mask"], dtype=np.int64
            ),
            "d_lm": d_lm,
            "v_lm": v_lm,
            "pad_info": pad_info,
        }
    )


def _validate_supported_polymers(job: Mapping[str, Any]) -> None:
    sequences = job.get("sequences", ())
    if not isinstance(sequences, list):
        return
    for entity in sequences:
        if not isinstance(entity, Mapping) or len(entity) != 1:
            continue
        kind, value = next(iter(entity.items()))
        if kind not in {"proteinChain", "dnaSequence", "rnaSequence"}:
            continue
        if not isinstance(value, Mapping):
            continue
        sequence = value.get("sequence")
        if not isinstance(sequence, str):
            continue
        alphabet = {
            "proteinChain": _CONCRETE_PROTEIN_ALPHABET,
            "dnaSequence": _CONCRETE_DNA_ALPHABET,
            "rnaSequence": _CONCRETE_RNA_ALPHABET,
        }[kind]
        unsupported = sorted(set(sequence.upper()) - alphabet)
        if unsupported:
            raise NotImplementedError(
                f"{kind} contains residue(s) without an exact vendored "
                f"reference conformer: {unsupported}"
            )


def _add_open_dde_metadata(features: dict[str, Any]) -> dict[str, Any]:
    restype = np.asarray(features["restype"])
    atom_to_parent = np.asarray(features["atom_to_token_idx"], dtype=np.int64)
    atom_names = np.asarray(features["output_atom_name"], dtype=str)
    ref_pos = np.asarray(features["ref_pos"], dtype=np.float32)
    ref_mask = np.asarray(features["ref_mask"])
    ref_space_uid = np.asarray(features["ref_space_uid"], dtype=np.int64)
    residue_index = np.asarray(features["residue_index"], dtype=np.int64)
    asym_id = np.asarray(features["asym_id"], dtype=np.int64)
    token_polymer_type = np.asarray(features["token_polymer_type"], dtype=np.int64)
    standard_polymer_parent = np.asarray(
        features["token_is_standard_polymer"], dtype=np.int64
    ).astype(bool)
    modified_parent = np.asarray(features["token_is_modified"], dtype=np.int64)

    if restype.ndim != 2 or restype.shape[-1] != 32:
        raise ValueError("restype must have shape [N_token, 32]")
    n_parent = restype.shape[0]
    n_atom = ref_pos.shape[0]
    if atom_to_parent.shape != (n_atom,):
        raise ValueError("atom_to_token_idx must have shape [N_atom]")
    if np.any((atom_to_parent < 0) | (atom_to_parent >= n_parent)):
        raise ValueError("atom_to_token_idx contains an out-of-range token index")
    if token_polymer_type.shape != (n_parent,):
        raise ValueError("token_polymer_type must have shape [N_token]")
    if standard_polymer_parent.shape != (n_parent,):
        raise ValueError("token_is_standard_polymer must have shape [N_token]")
    if modified_parent.shape != (n_parent,):
        raise ValueError("token_is_modified must have shape [N_token]")
    if np.any(~np.isin(token_polymer_type, [0, 1, 2, 3])):
        raise ValueError("token_polymer_type contains an unsupported value")

    restype_index = np.argmax(restype, axis=-1)
    protein_parent = token_polymer_type == 1
    dna_parent = token_polymer_type == 2
    rna_parent = token_polymer_type == 3
    polymer_parent = protein_parent | rna_parent | dna_parent
    ligand_parent = token_polymer_type == 0
    if np.any(standard_polymer_parent & ~polymer_parent):
        raise ValueError("only polymer tokens may be marked as standard polymer")
    if np.any(~np.isin(restype_index, np.r_[0:20, 20, 21:25, 26:30])):
        raise ValueError("unsupported residue-token restype in OpenDDE metadata")

    parent_atoms = [np.flatnonzero(atom_to_parent == i) for i in range(n_parent)]
    if any(atom_indices.size == 0 for atom_indices in parent_atoms):
        raise ValueError("every residue token must contain at least one atom")

    pae_rep_atom_mask = np.zeros(n_atom, dtype=np.int64)
    plddt_m_rep_atom_mask = np.zeros(n_atom, dtype=np.int64)
    has_frame = np.zeros(n_parent, dtype=np.int64)
    frame_atom_index = np.full((n_parent, 3), -1, dtype=np.int64)

    for parent_idx, atom_indices in enumerate(parent_atoms):
        names = atom_names[atom_indices]
        if standard_polymer_parent[parent_idx] and protein_parent[parent_idx]:
            centre = _named_atom(atom_indices, names, "CA", parent_idx)
            frame = _polymer_frame(
                atom_indices,
                names,
                ("N", "CA", "C"),
                parent_idx,
            )
            plddt_m_rep_atom_mask[centre] = 1
        elif standard_polymer_parent[parent_idx] and (
            rna_parent[parent_idx] or dna_parent[parent_idx]
        ):
            centre = _named_atom(atom_indices, names, "C1'", parent_idx)
            frame = _polymer_frame(
                atom_indices,
                names,
                ("C1'", "C3'", "C4'"),
                parent_idx,
            )
            plddt_m_rep_atom_mask[centre] = 1
        else:
            if atom_indices.size != 1:
                raise ValueError(
                    "ligand residue tokens must be atom-level (one atom per token)"
                )
            centre = int(atom_indices[0])
            frame, frame_valid = _ligand_frame(
                centre,
                ref_pos=ref_pos,
                ref_mask=ref_mask,
                ref_space_uid=ref_space_uid,
            )
            has_frame[parent_idx] = frame_valid
        pae_rep_atom_mask[centre] = 1
        frame_atom_index[parent_idx] = frame
        if standard_polymer_parent[parent_idx]:
            has_frame[parent_idx] = 1

    atom_is_protein = protein_parent[atom_to_parent].astype(np.int64)
    atom_is_rna = rna_parent[atom_to_parent].astype(np.int64)
    atom_is_dna = dna_parent[atom_to_parent].astype(np.int64)
    atom_is_ligand = ligand_parent[atom_to_parent].astype(np.int64)

    structural = _structural_metadata(
        restype_index=restype_index,
        parent_atoms=parent_atoms,
        atom_names=atom_names,
        residue_frames=frame_atom_index,
        residue_has_frame=has_frame,
        protein_parent=protein_parent,
        dna_parent=dna_parent,
        rna_parent=rna_parent,
        standard_polymer_parent=standard_polymer_parent,
        residue_index=residue_index,
        asym_id=asym_id,
        ref_pos=ref_pos,
        ref_mask=ref_mask,
        ref_space_uid=ref_space_uid,
        atom_to_parent=atom_to_parent,
        covalent_atom_indices=np.asarray(
            features.get("covalent_atom_indices", np.zeros((0, 2))),
            dtype=np.int64,
        ).reshape((-1, 2)),
    )

    features.update(
        {
            "has_frame": has_frame,
            "frame_atom_index": frame_atom_index,
            "pae_rep_atom_mask": pae_rep_atom_mask,
            "plddt_m_rep_atom_mask": plddt_m_rep_atom_mask,
            "modified_res_mask": modified_parent[atom_to_parent],
            "is_protein": atom_is_protein,
            "is_ligand": atom_is_ligand,
            "is_dna": atom_is_dna,
            "is_rna": atom_is_rna,
            **structural,
        }
    )
    return features


def _structural_metadata(
    *,
    restype_index: np.ndarray,
    parent_atoms: list[np.ndarray],
    atom_names: np.ndarray,
    residue_frames: np.ndarray,
    residue_has_frame: np.ndarray,
    protein_parent: np.ndarray,
    dna_parent: np.ndarray,
    rna_parent: np.ndarray,
    standard_polymer_parent: np.ndarray,
    residue_index: np.ndarray,
    asym_id: np.ndarray,
    ref_pos: np.ndarray,
    ref_mask: np.ndarray,
    ref_space_uid: np.ndarray,
    atom_to_parent: np.ndarray,
    covalent_atom_indices: np.ndarray,
) -> dict[str, np.ndarray]:
    n_atom = atom_names.shape[0]
    parents: list[int] = []
    roles: list[int] = []
    twins: list[int] = []
    groups: list[int] = []
    structural_atoms: list[np.ndarray] = []
    centres: list[int] = []

    for parent_idx, atom_indices in enumerate(parent_atoms):
        names = atom_names[atom_indices]
        token_offset = len(parents)
        if standard_polymer_parent[parent_idx] and protein_parent[parent_idx]:
            if restype_index[parent_idx] == 7:
                token_groups = [("protein_bb", atom_indices)]
            else:
                backbone_mask = np.isin(names, list(_PROTEIN_BACKBONE_ATOMS))
                backbone_atoms = atom_indices[backbone_mask]
                sidechain_atoms = atom_indices[~backbone_mask]
                token_groups = [("protein_bb", backbone_atoms)]
                if sidechain_atoms.size:
                    token_groups.append(("protein_sc", sidechain_atoms))
        elif standard_polymer_parent[parent_idx] and (
            dna_parent[parent_idx] or rna_parent[parent_idx]
        ):
            backbone_mask = np.isin(names, list(_NUCLEIC_BACKBONE_ATOMS))
            prefix = "dna" if dna_parent[parent_idx] else "rna"
            token_groups = [(f"{prefix}_bb", atom_indices[backbone_mask])]
            base_atoms = atom_indices[~backbone_mask]
            if base_atoms.size:
                token_groups.append((f"{prefix}_base", base_atoms))
        else:
            token_groups = [("atom", atom_indices)]

        if not token_groups[0][1].size:
            token_groups = [(token_groups[0][0], atom_indices)]
        for role_name, group_atoms in token_groups:
            parents.append(parent_idx)
            roles.append(STRUCTURAL_TOKEN_ROLES[role_name])
            groups.append(parent_idx)
            structural_atoms.append(group_atoms)
            centres.append(
                _structural_centre(
                    role_name,
                    group_atoms,
                    atom_names,
                    restype_index[parent_idx],
                )
            )
            twins.append(_NO_TWIN_TOKEN_IDX)
        if len(token_groups) == 2:
            twins[token_offset] = token_offset + 1
            twins[token_offset + 1] = token_offset

    n_structural = len(parents)
    atom_to_structural = np.full(n_atom, -1, dtype=np.int64)
    atom_to_structural_slot = np.full(n_atom, -1, dtype=np.int64)
    structural_rep = np.zeros(n_atom, dtype=np.int64)
    structural_frames = np.full((n_structural, 3), -1, dtype=np.int64)
    structural_has_frame = np.zeros(n_structural, dtype=np.int64)
    for token_idx, (parent_idx, atom_indices, centre) in enumerate(
        zip(parents, structural_atoms, centres, strict=True)
    ):
        atom_to_structural[atom_indices] = token_idx
        atom_to_structural_slot[atom_indices] = np.arange(atom_indices.size)
        structural_rep[centre] = 1
        if (
            protein_parent[parent_idx]
            or dna_parent[parent_idx]
            or rna_parent[parent_idx]
        ):
            structural_frames[token_idx] = residue_frames[parent_idx]
            structural_has_frame[token_idx] = residue_has_frame[parent_idx]
        else:
            frame, valid = _ligand_frame(
                centre,
                ref_pos=ref_pos,
                ref_mask=ref_mask,
                ref_space_uid=ref_space_uid,
            )
            structural_frames[token_idx] = frame
            structural_has_frame[token_idx] = valid

    if np.any(atom_to_structural < 0) or np.any(atom_to_structural_slot < 0):
        raise ValueError("some atoms were not assigned to a structural token")

    parent_polymer_type = np.zeros(len(parent_atoms), dtype=np.int64)
    parent_polymer_type[protein_parent] = 1
    parent_polymer_type[dna_parent] = 2
    parent_polymer_type[rna_parent] = 3
    graph_polymer_type = parent_polymer_type * standard_polymer_parent
    parent_prev, parent_next = _polymer_residue_graph(
        graph_polymer_type,
        asym_id,
        residue_index,
        atom_to_parent=atom_to_parent,
        atom_names=atom_names,
        covalent_atom_indices=covalent_atom_indices,
    )
    parent_array = np.asarray(parents, dtype=np.int64)
    role_array = np.asarray(roles, dtype=np.int64)
    return {
        "structural_token_index": np.arange(n_structural, dtype=np.int64),
        "residue_token_group_id": np.asarray(groups, dtype=np.int64),
        "subtoken_role": role_array.copy(),
        "subtoken_role_id": role_array,
        "twin_token_idx": np.asarray(twins, dtype=np.int64),
        "parent_residue_idx": parent_array,
        "atom_to_structural_token_idx": atom_to_structural,
        "atom_to_structural_tokatom_idx": atom_to_structural_slot,
        "structural_distogram_rep_atom_mask": structural_rep.copy(),
        "structural_pae_rep_atom_mask": structural_rep,
        "structural_has_frame": structural_has_frame,
        "structural_frame_atom_index": structural_frames,
        "prev_parent_residue_idx": parent_prev[parent_array],
        "next_parent_residue_idx": parent_next[parent_array],
        "structural_is_polymer": (parent_polymer_type[parent_array] > 0).astype(
            np.int64
        ),
        "structural_polymer_type": parent_polymer_type[parent_array],
        "structural_seq_pos": residue_index[parent_array],
    }


def _structural_centre(
    role: str,
    atom_indices: np.ndarray,
    atom_names: np.ndarray,
    restype_index: int,
) -> int:
    preferences = {
        "protein_bb": ("CA", "N", "C"),
        "protein_sc": ("CB",),
        "dna_bb": ("C4'", "C4*", "C1'", "C1*"),
        "rna_bb": ("C4'", "C4*", "C1'", "C1*"),
    }.get(role)
    if role in {"dna_base", "rna_base"}:
        if restype_index in _PURINE_RESTYPE_INDICES:
            preferences = ("N9", "C4", "C8", "N7", "C5")
        elif restype_index in _PYRIMIDINE_RESTYPE_INDICES:
            preferences = ("N1", "C2", "C6", "C5", "C4")
        else:
            preferences = ("C1'", "C1*", "N9", "N1")
    if preferences is not None:
        names = atom_names[atom_indices]
        for preferred in preferences:
            match = np.flatnonzero(names == preferred)
            if match.size:
                return int(atom_indices[match[0]])
    return int(atom_indices[0])


def _named_atom(
    atom_indices: np.ndarray,
    names: np.ndarray,
    target: str,
    parent_idx: int,
) -> int:
    matches = np.flatnonzero(names == target)
    if matches.size != 1:
        raise ValueError(
            f"residue token {parent_idx} must contain exactly one {target!r} atom"
        )
    return int(atom_indices[matches[0]])


def _polymer_frame(
    atom_indices: np.ndarray,
    names: np.ndarray,
    frame_names: tuple[str, str, str],
    parent_idx: int,
) -> np.ndarray:
    return np.asarray(
        [_named_atom(atom_indices, names, name, parent_idx) for name in frame_names],
        dtype=np.int64,
    )


def _ligand_frame(
    centre: int,
    *,
    ref_pos: np.ndarray,
    ref_mask: np.ndarray,
    ref_space_uid: np.ndarray,
) -> tuple[np.ndarray, int]:
    atom_ids = np.flatnonzero(ref_space_uid == ref_space_uid[centre])
    if atom_ids.size < 3:
        return np.asarray([-1, centre, -1], dtype=np.int64), 0
    distances = np.linalg.norm(ref_pos[atom_ids] - ref_pos[centre], axis=-1)
    order = np.argsort(distances, kind="stable")
    neighbours = atom_ids[order[atom_ids[order] != centre]][:2]
    if neighbours.size != 2:
        return np.asarray([-1, centre, -1], dtype=np.int64), 0
    frame = np.asarray([neighbours[0], centre, neighbours[1]], dtype=np.int64)
    valid = bool(np.all(ref_mask[frame]))
    ab = ref_pos[frame[1]] - ref_pos[frame[0]]
    bc = ref_pos[frame[2]] - ref_pos[frame[1]]
    denominator = np.linalg.norm(ab) * np.linalg.norm(bc)
    if np.isclose(denominator, 0.0):
        valid = False
    elif valid:
        cosine = np.clip(np.dot(ab, bc) / denominator, -1.0, 1.0)
        angle = float(np.degrees(np.arccos(cosine)))
        valid = 25.0 < angle < 155.0
    return frame, int(valid)


def _polymer_residue_graph(
    polymer_type: np.ndarray,
    asym_id: np.ndarray,
    residue_index: np.ndarray,
    *,
    atom_to_parent: np.ndarray,
    atom_names: np.ndarray,
    covalent_atom_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n_parent = polymer_type.shape[0]
    prev_parent = np.full(n_parent, -1, dtype=np.int64)
    next_parent = np.full(n_parent, -1, dtype=np.int64)
    graph_edges: set[tuple[int, int]] = set()
    for atom_a, atom_b in covalent_atom_indices:
        parent_a = int(atom_to_parent[atom_a])
        parent_b = int(atom_to_parent[atom_b])
        if parent_a == parent_b:
            continue
        if (
            polymer_type[parent_a] <= 0
            or polymer_type[parent_a] != polymer_type[parent_b]
            or asym_id[parent_a] != asym_id[parent_b]
            or not _is_polymer_backbone_bond(
                int(polymer_type[parent_a]),
                str(atom_names[atom_a]),
                str(atom_names[atom_b]),
            )
        ):
            continue
        first_parent, second_parent = sorted((parent_a, parent_b))
        next_parent[first_parent] = second_parent
        prev_parent[second_parent] = first_parent
        graph_edges.add((first_parent, second_parent))

    for parent_idx in range(n_parent - 1):
        next_idx = parent_idx + 1
        if (parent_idx, next_idx) in graph_edges:
            continue
        if next_parent[parent_idx] >= 0 or prev_parent[next_idx] >= 0:
            continue
        if (
            polymer_type[parent_idx] > 0
            and polymer_type[parent_idx] == polymer_type[next_idx]
            and asym_id[parent_idx] == asym_id[next_idx]
            and residue_index[next_idx] - residue_index[parent_idx] == 1
        ):
            next_parent[parent_idx] = next_idx
            prev_parent[next_idx] = parent_idx
    return prev_parent, next_parent


def _is_polymer_backbone_bond(
    polymer_type: int, atom_name_a: str, atom_name_b: str
) -> bool:
    atom_pair = {atom_name_a, atom_name_b}
    if polymer_type == 1:
        return atom_pair == {"C", "N"}
    if polymer_type in (2, 3):
        return "P" in atom_pair and bool(atom_pair & {"O3'", "O3*", "O3T"})
    return False
