"""Sequence-only Protenix JSON to static feature conversion."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import mmap
import os
import pickle
import random
import re
import string
import tempfile
from collections import OrderedDict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import gemmi
import numpy as np

from foldjax.models.protenix.data.static_io import save_static_feature_npz
from foldjax.models.protenix.data.template_features import (
    assemble_template_features,
    chain_template_dense,
)

RESTYPE_INDEX = {
    "A": 0,
    "R": 1,
    "N": 2,
    "D": 3,
    "C": 4,
    "Q": 5,
    "E": 6,
    "G": 7,
    "H": 8,
    "I": 9,
    "L": 10,
    "K": 11,
    "M": 12,
    "F": 13,
    "P": 14,
    "S": 15,
    "T": 16,
    "W": 17,
    "Y": 18,
    "V": 19,
    "X": 20,
}
MSA_PROTEIN_INDEX = {
    **RESTYPE_INDEX,
    "B": RESTYPE_INDEX["D"],
    "J": RESTYPE_INDEX["X"],
    "O": RESTYPE_INDEX["X"],
    "U": RESTYPE_INDEX["C"],
    "Z": RESTYPE_INDEX["E"],
    "-": 31,
}
# Periodic table order used by torch ``get_all_elems`` (index == Z - 1).
_PERIODIC_TABLE = (
    "H HE LI BE B C N O F NE NA MG AL SI P S CL AR K CA SC TI V CR MN FE CO "
    "NI CU ZN GA GE AS SE BR KR RB SR Y ZR NB MO TC RU RH PD AG CD IN SN SB "
    "TE I XE CS BA LA CE PR ND PM SM EU GD TB DY HO ER TM YB LU HF TA W RE OS "
    "IR PT AU HG TL PB BI PO AT RN FR RA AC TH PA U NP PU AM CM BK CF ES FM "
    "MD NO LR RF DB SG BH HS MT DS RG CN NH FL MC LV TS OG"
).split()
ELEMENT_INDEX = {elem: i for i, elem in enumerate(_PERIODIC_TABLE)}
for _i in range(119, 129):
    ELEMENT_INDEX[f"UNK_ELEM_{_i}"] = _i - 1
# Nucleotide restype indices (torch STD_RESIDUES); RNA then DNA.
RNA_RESTYPE_INDEX = {"A": 21, "G": 22, "C": 23, "U": 24, "N": 25}
DNA_RESTYPE_INDEX = {"DA": 26, "DG": 27, "DC": 28, "DT": 29, "DN": 30}
MSA_RNA_INDEX = {
    **{char: 25 for char in string.ascii_uppercase},
    **RNA_RESTYPE_INDEX,
    "-": 31,
}
RNA_CODES = {"A": "A", "G": "G", "C": "C", "U": "U", "N": "N"}
DNA_CODES = {"A": "DA", "G": "DG", "C": "DC", "T": "DT", "N": "DN"}
# Distogram representative atom: purine -> C4, pyrimidine -> C2.
_PURINE_CODES = {"DA", "DG", "A", "G"}
_PYRIMIDINE_CODES = {"DC", "DT", "C", "U"}
AA1_TO_AA3 = {
    "A": "ALA",
    "R": "ARG",
    "N": "ASN",
    "D": "ASP",
    "C": "CYS",
    "Q": "GLN",
    "E": "GLU",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "L": "LEU",
    "K": "LYS",
    "M": "MET",
    "F": "PHE",
    "P": "PRO",
    "S": "SER",
    "T": "THR",
    "W": "TRP",
    "Y": "TYR",
    "V": "VAL",
    "X": "UNK",
}
_CCD_TABLE_PATH = Path(__file__).with_name("ccd_std_residues.npz")
_CCD_TABLE: dict[str, dict[str, np.ndarray]] | None = None
_LIGAND_TABLE_PATH = Path(__file__).with_name("ccd_ligands.npz")
_LIGAND_TABLE: dict[str, dict[str, np.ndarray]] | None = None
_NUCLEIC_TABLE_PATH = Path(__file__).with_name("ccd_nucleotides.npz")
_NUCLEIC_TABLE: dict[str, dict[str, np.ndarray]] | None = None
_EXTERNAL_CCD_MOLS: dict[str, Any] | None = None
_EXTERNAL_CCD_ATOM_CACHE_LIMIT = 256
_EXTERNAL_CCD_ATOMS: OrderedDict[
    tuple[Path, tuple[int, int, int, int, int], str], dict[str, Any]
] = OrderedDict()
_TRUSTED_CCD_RDKIT_SHA256 = frozenset(
    {"d1cfb71f5993a3ebea7c47877022d7f597bbfbaf86e28a4770e957da6c50cd35"}
)
_CCD_HASH_CHUNK = 1 << 20
_TOKEN_POLYMER_TYPE = {"ligand": 0, "protein": 1, "dna": 2, "rna": 3}


def _ccd_ligands() -> dict[str, dict[str, np.ndarray]]:
    """Lazy-load the vendored ligand CCD reference table.

    Each CCD code maps to ``names``/``coord``/``charge``/``mask``/``elem``
    arrays in the order produced by torch ``get_component_atom_array``
    (keep_leaving_atoms=True, keep_hydrogens=False).
    """

    global _LIGAND_TABLE
    if _LIGAND_TABLE is None:
        raw = np.load(_LIGAND_TABLE_PATH, allow_pickle=False)
        codes = [str(c) for c in raw["_codes"]]
        table: dict[str, dict[str, np.ndarray]] = {}
        for code in codes:
            table[code] = {
                "names": raw[f"{code}/names"],
                "coord": raw[f"{code}/coord"].astype(np.float32),
                "charge": raw[f"{code}/charge"].astype(np.float32),
                "mask": raw[f"{code}/mask"].astype(np.float32),
                "elem": raw[f"{code}/elem"],
                # intra-ligand bond edges (atom-index pairs in this entry's
                # atom order); empty if absent in an older npz.
                "bonds": (
                    raw[f"{code}/bonds"].astype(np.int64)
                    if f"{code}/bonds" in raw.files
                    else np.zeros((0, 2), dtype=np.int64)
                ),
            }
        _LIGAND_TABLE = table
    return _LIGAND_TABLE


def _ccd_nucleotides() -> dict[str, dict[str, np.ndarray]]:
    """Lazy-load the vendored nucleotide CCD reference table.

    Each CCD code (DA/DC/DG/DT, A/C/G/U) maps to ``names``/``coord``/
    ``charge``/``mask``/``elem`` arrays in RES_ATOMS order (OP3 first). OP3
    is the 5'-terminal leaving atom: kept only for the first residue.
    """

    global _NUCLEIC_TABLE
    if _NUCLEIC_TABLE is None:
        raw = np.load(_NUCLEIC_TABLE_PATH, allow_pickle=False)
        codes = [str(c) for c in raw["_codes"]]
        table: dict[str, dict[str, np.ndarray]] = {}
        for code in codes:
            table[code] = {
                "names": raw[f"{code}/names"],
                "coord": raw[f"{code}/coord"].astype(np.float32),
                "charge": raw[f"{code}/charge"].astype(np.float32),
                "mask": raw[f"{code}/mask"].astype(np.float32),
                "elem": raw[f"{code}/elem"],
            }
        _NUCLEIC_TABLE = table
    return _NUCLEIC_TABLE


def _ccd_std_residues() -> dict[str, dict[str, np.ndarray]]:
    """Lazy-load the vendored CCD reference-conformer table (20 std residues).

    Each residue maps to ``names``/``coord``/``charge``/``elem`` arrays in the
    canonical RES_ATOMS order (N, CA, C, O, sidechain..., OXT last). OXT is
    kept only for the C-terminal residue of a chain.
    """

    global _CCD_TABLE
    if _CCD_TABLE is None:
        raw = np.load(_CCD_TABLE_PATH, allow_pickle=False)
        table: dict[str, dict[str, np.ndarray]] = {}
        for aa3 in AA1_TO_AA3.values():
            table[aa3] = {
                "names": raw[f"{aa3}/names"],
                "coord": raw[f"{aa3}/coord"].astype(np.float32),
                "charge": raw[f"{aa3}/charge"].astype(np.float32),
                "elem": raw[f"{aa3}/elem"],
            }
        _CCD_TABLE = table
    return _CCD_TABLE


def load_first_job(path: str | Path) -> dict[str, Any]:
    """Load the first Protenix JSON job."""

    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list) or not data:
        raise ValueError("input JSON must be a non-empty top-level list")
    job = data[0]
    if not isinstance(job, dict):
        raise ValueError("first input JSON entry must be an object")
    return job


def _resolve_featurization_seed(job: dict[str, Any], seed: int | None) -> int:
    """Resolve the local chemistry RNG without mutating Python's global RNG."""

    if seed is not None:
        return int(seed)
    model_seeds = job.get("modelSeeds")
    if isinstance(model_seeds, list) and model_seeds:
        return int(model_seeds[0])
    return 101


def featurize_protein_json(
    job: dict[str, Any],
    *,
    base_dir: str | Path | None = None,
    n_queries: int = 32,
    n_keys: int = 128,
    max_msa_depth: int = 16384,
    seed: int | None = None,
) -> dict[str, Any]:
    """Build static features for proteinChain inputs."""

    if not isinstance(job, dict):
        raise ValueError("job must be an object")
    unknown_job_keys = set(job) - {
        "name",
        "modelSeeds",
        "dialect",
        "version",
        "sequences",
        "covalent_bonds",
        "constraint",
    }
    if unknown_job_keys:
        raise ValueError(
            f"unsupported top-level inference fields: {sorted(unknown_job_keys)}"
        )

    if n_keys < n_queries or n_queries % 2 or n_keys % 2:
        raise ValueError("n_keys must be >= n_queries and both must be even")
    if max_msa_depth <= 0:
        raise ValueError("max_msa_depth must be positive")
    chains = _expand_chains(job, base_dir=base_dir)
    chemistry_rng = random.Random(_resolve_featurization_seed(job, seed))
    _remove_polymer_link_leaving_groups(chains, rng=chemistry_rng)
    _remove_covalent_leaving_groups(
        chains,
        job.get("covalent_bonds", []),
        rng=chemistry_rng,
    )
    _prepare_polymer_tokens(chains)
    n_token = sum(_chain_token_count(chain) for chain in chains)
    if n_token <= 0:
        raise ValueError("at least one token is required")

    restype = np.zeros((n_token, 32), dtype=np.float32)
    residue_index = np.zeros((n_token,), dtype=np.int64)
    # torch sets token_index = arange(N_token) (featurizer.py:329). jax left it
    # all-zeros, which is invariant for polymers (relp token term is masked to
    # same-chain-same-residue → diagonal) but wrong for ligands where many tokens
    # share one residue, breaking the ligand relp block.
    token_index = np.arange(n_token, dtype=np.int64)
    asym_id = np.zeros((n_token,), dtype=np.int64)
    entity_id = np.zeros((n_token,), dtype=np.int64)
    sym_id = np.zeros((n_token,), dtype=np.int64)
    has_frame = np.ones((n_token,), dtype=np.int64)
    profile = np.zeros((n_token, 32), dtype=np.float32)
    deletion_mean = np.zeros((n_token,), dtype=np.float32)

    atom_to_token_idx: list[int] = []
    atom_to_tokatom_idx: list[int] = []
    ref_pos: list[tuple[float, float, float]] = []
    ref_space_uid: list[int] = []
    ref_charge: list[float] = []
    ref_element: list[str] = []
    ref_atom_names: list[str] = []
    distogram_rep_atom_mask: list[int] = []
    ref_mask_list: list[float] = []
    mol_id: list[int] = []
    atom_entity_id: list[int] = []
    atom_copy_id: list[int] = []
    atom_residue_index: list[int] = []
    atom_input_index: list[int] = []
    ligand_stereo_atom: list[int] = []

    state = {
        "token_i": 0,
        "restype": restype,
        "profile": profile,
        "deletion_mean": deletion_mean,
        "residue_index": residue_index,
        "asym_id": asym_id,
        "entity_id": entity_id,
        "sym_id": sym_id,
        "atom_to_token_idx": atom_to_token_idx,
        "atom_to_tokatom_idx": atom_to_tokatom_idx,
        "ref_pos": ref_pos,
        "ref_space_uid": ref_space_uid,
        "next_ref_space_uid": 0,
        "ref_charge": ref_charge,
        "ref_element": ref_element,
        "ref_atom_names": ref_atom_names,
        "distogram_rep_atom_mask": distogram_rep_atom_mask,
        "ref_mask": ref_mask_list,
        "mol_id": mol_id,
        "atom_entity_id": atom_entity_id,
        "atom_copy_id": atom_copy_id,
        "atom_residue_index": atom_residue_index,
        "atom_input_index": atom_input_index,
        "ligand_stereo_atom": ligand_stereo_atom,
        "output_atom_res_name": [],
        "output_atom_chain_id": [],
        "output_atom_res_id": [],
        "output_atom_polymer_type": [],
        "chemical_bond_atom_indices": [],
        "chemical_bond_order": [],
        "chemical_bond_stereo": [],
        "token_entity_number": np.zeros((n_token,), dtype=np.int64),
        "token_copy_id": np.zeros((n_token,), dtype=np.int64),
        "token_ccd_codes": [""] * n_token,
        "token_is_modified": np.zeros((n_token,), dtype=np.int64),
        "token_is_standard_polymer": np.zeros((n_token,), dtype=np.int64),
        "token_polymer_type": np.zeros((n_token,), dtype=np.int64),
        "token_reference_is_mse": np.zeros((n_token,), dtype=np.int64),
        "token_bond_edges": [],
    }
    for chain in chains:
        if chain["kind"] == "protein":
            _emit_protein_tokens(chain, state)
        elif chain["kind"] == "nucleic":
            _emit_nucleic_tokens(chain, state)
        else:
            _emit_ligand_tokens(chain, state)

    covalent_atom_indices, covalent_token_indices = _apply_covalent_bonds(
        job.get("covalent_bonds", []), state
    )

    atom_to_token = np.asarray(atom_to_token_idx, dtype=np.int64)
    ref_pos_arr = np.asarray(ref_pos, dtype=np.float32)
    ref_space_uid_arr = np.asarray(ref_space_uid, dtype=np.int64)
    for uid in np.unique(ref_space_uid_arr):
        in_ref_space = ref_space_uid_arr == uid
        ref_pos_arr[in_ref_space] -= ref_pos_arr[in_ref_space].mean(axis=0)
    d_lm, v_lm, pad_info = _local_atom_geometry(
        ref_pos_arr,
        ref_space_uid_arr,
        n_queries=n_queries,
        n_keys=n_keys,
    )
    relp = _relative_position_features(
        asym_id=asym_id,
        residue_index=residue_index,
        entity_id=entity_id,
        sym_id=sym_id,
        token_index=token_index,
    )
    msa, deletion_matrix, assembled_profile, assembled_deletion_mean = (
        _assemble_msa_features(chains, max_msa_depth=max_msa_depth)
    )
    profile[:] = assembled_profile
    deletion_mean[:] = assembled_deletion_mean
    template_features = _assemble_chain_templates(chains)
    # token_bonds: intra-ligand bond adjacency (torch keeps only non-standard
    # bonds = ligand-internal; polymer backbone bonds are excluded, so this is
    # zero for polymer-only inputs). Symmetric.
    token_bonds = np.zeros((n_token, n_token), dtype=np.float32)
    for ti, tj in state["token_bond_edges"]:
        token_bonds[ti, tj] = 1.0
        token_bonds[tj, ti] = 1.0
    constraint_feature = _build_constraint_features(
        job.get("constraint", {}), state, n_token=n_token
    )
    out = {
        "atom_to_token_idx": atom_to_token,
        "ref_pos": ref_pos_arr,
        "ref_space_uid": ref_space_uid_arr,
        "ref_charge": np.asarray(ref_charge, dtype=np.float32),
        "ref_mask": np.asarray(ref_mask_list, dtype=np.float32),
        "ref_atom_name_chars": _encode_atom_name_chars(ref_atom_names),
        "ref_element": _encode_elements(ref_element),
        "d_lm": d_lm,
        "v_lm": v_lm,
        "pad_info": pad_info,
        "restype": restype,
        "profile": profile,
        "deletion_mean": deletion_mean,
        "msa": msa,
        "has_deletion": np.clip(deletion_matrix, 0.0, 1.0).astype(np.float32),
        "deletion_value": (
            np.arctan(deletion_matrix.astype(np.float32) / 3.0) * (2.0 / np.pi)
        ).astype(np.float32),
        "relp": relp,
        "token_bonds": token_bonds,
        "residue_index": residue_index,
        "token_index": token_index,
        "asym_id": asym_id,
        "entity_id": entity_id,
        "sym_id": sym_id,
        "has_frame": has_frame,
        "distogram_rep_atom_mask": np.asarray(
            distogram_rep_atom_mask,
            dtype=np.float32,
        ),
        "atom_to_tokatom_idx": np.asarray(atom_to_tokatom_idx, dtype=np.int64),
        "mol_id": np.asarray(mol_id, dtype=np.int64),
        "atom_entity_id": np.asarray(atom_entity_id, dtype=np.int64),
        "atom_copy_id": np.asarray(atom_copy_id, dtype=np.int64),
        "atom_residue_index": np.asarray(atom_residue_index, dtype=np.int64),
        "atom_input_index": np.asarray(atom_input_index, dtype=np.int64),
        "ligand_stereo": np.asarray(ligand_stereo_atom, dtype=np.int64),
        "output_atom_name": np.asarray(ref_atom_names, dtype=str),
        "output_atom_element": np.asarray(ref_element, dtype=str),
        "output_atom_res_name": np.asarray(state["output_atom_res_name"], dtype=str),
        "output_atom_chain_id": np.asarray(state["output_atom_chain_id"], dtype=str),
        "output_atom_res_id": np.asarray(state["output_atom_res_id"], dtype=np.int64),
        "output_atom_polymer_type": np.asarray(
            state["output_atom_polymer_type"], dtype=str
        ),
        "chemical_bond_atom_indices": np.asarray(
            state["chemical_bond_atom_indices"], dtype=np.int64
        ).reshape((-1, 2)),
        "chemical_bond_order": np.asarray(
            state["chemical_bond_order"], dtype=np.float32
        ),
        "chemical_bond_stereo": np.asarray(
            state["chemical_bond_stereo"], dtype=np.int64
        ),
        "token_entity_id": state["token_entity_number"],
        "token_copy_id": state["token_copy_id"],
        "token_is_modified": state["token_is_modified"],
        "token_is_standard_polymer": state["token_is_standard_polymer"],
        "token_polymer_type": state["token_polymer_type"],
        "token_reference_is_mse": state["token_reference_is_mse"],
        "token_ccd_code_chars": _encode_fixed_strings(
            state["token_ccd_codes"], width=8
        ),
        "covalent_atom_indices": covalent_atom_indices,
        "covalent_token_indices": covalent_token_indices,
    }
    if "constraint" in job:
        out["constraint_feature"] = constraint_feature
    if template_features is not None:
        out.update(template_features)
    return out


def _assemble_chain_templates(
    chains: list[dict[str, Any]],
) -> dict[str, np.ndarray] | None:
    """Build per-token template features in chain/token order (torch parity)."""

    chain_dense = []
    for chain in chains:
        num_res = _chain_biological_count(chain)
        if chain["kind"] == "protein":
            sequence = chain["sequence"]
            dense = chain_template_dense(
                chain.get("templates_path"),
                sequence=sequence,
                skip=len(sequence) <= 4,
            )
        else:
            dense = chain_template_dense(
                None,
                sequence="X" * num_res,
                skip=True,
            )
        gather = np.asarray(chain["token_to_sequence_idx"], dtype=np.int64)
        chain_dense.append(tuple(np.take(value, gather, axis=1) for value in dense))
    return assemble_template_features(chain_dense)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-queries", type=int, default=32)
    parser.add_argument("--n-keys", type=int, default=128)
    parser.add_argument(
        "--max-msa-depth",
        "--max-msa-rows",
        dest="max_msa_depth",
        type=int,
        default=16384,
    )
    args = parser.parse_args(argv)

    features = featurize_protein_json(
        load_first_job(args.input),
        base_dir=args.input.parent,
        n_queries=args.n_queries,
        n_keys=args.n_keys,
        max_msa_depth=args.max_msa_depth,
    )
    save_static_feature_npz(args.out, features)
    print(f"wrote: {args.out}")


def parse_a3m_profile(
    query_sequence: str,
    a3m: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Parse protein A3M content into Protenix profile and deletion mean."""

    msa, deletion_matrix = parse_a3m_rows(query_sequence, a3m)
    profile = (msa[..., None] == np.arange(32)).sum(axis=0) / msa.shape[0]
    return profile.astype(np.float32), deletion_matrix.mean(axis=0)


def parse_a3m_rows(
    query_sequence: str,
    a3m: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Parse protein A3M content into Protenix MSA rows and deletion matrix."""

    records, _descs = _parse_a3m_records(a3m)
    if not records:
        records = [query_sequence]
    rows = []
    deletions = []
    for sequence in records:
        row, deletion = _aligned_protein_row(sequence)
        if len(row) != len(query_sequence):
            raise ValueError(
                "A3M aligned length must match query length: "
                f"{len(row)} != {len(query_sequence)}"
            )
        rows.append(row)
        deletions.append(deletion)
    msa = np.asarray(rows, dtype=np.int64)
    deletion_matrix = np.asarray(deletions, dtype=np.float32)
    return msa, deletion_matrix


def _expand_chains(
    job: dict[str, Any],
    *,
    base_dir: str | Path | None,
) -> list[dict[str, Any]]:
    sequences = job.get("sequences")
    if not isinstance(sequences, list) or not sequences:
        raise ValueError("job must contain a non-empty sequences list")
    built_entities: list[tuple[int, dict[str, Any]]] = []
    reserved_chain_ids: set[str] = set()
    for entity_number, entry in enumerate(sequences, start=1):
        if not isinstance(entry, dict) or len(entry) != 1:
            raise ValueError("each sequence entry must have one entity key")
        (kind,) = entry.keys()
        if kind == "proteinChain":
            built = _build_protein_chain(entry[kind], base_dir=base_dir)
        elif kind in ("dnaSequence", "rnaSequence"):
            built = _build_nucleic_chain(entry[kind], kind=kind, base_dir=base_dir)
        elif kind in ("ligand", "ion"):
            built = _build_ligand_chain(entry[kind], kind=kind, base_dir=base_dir)
        else:
            raise ValueError(f"unsupported entity kind: {kind}")
        built["polymer_type"] = {
            "proteinChain": "polypeptide(L)",
            "dnaSequence": "polydeoxyribonucleotide",
            "rnaSequence": "polyribonucleotide",
            "ligand": "non-polymer",
            "ion": "non-polymer",
        }[kind]
        ids = built.get("ids")
        if ids:
            duplicate = reserved_chain_ids.intersection(ids)
            if duplicate:
                raise ValueError(f"duplicate chain id: {sorted(duplicate)[0]!r}")
            reserved_chain_ids.update(ids)
        built_entities.append((entity_number, built))

    chains: list[dict[str, Any]] = []
    used_chain_ids: set[str] = set()
    next_auto_chain_index = 0
    for entity_number, built in built_entities:
        entity = entity_number - 1
        ids = built.get("ids")
        for copy_id in range(1, built["count"] + 1):
            if ids:
                chain_id = ids[copy_id - 1]
            else:
                while True:
                    chain_id = _default_chain_id(next_auto_chain_index)
                    next_auto_chain_index += 1
                    if (
                        chain_id not in reserved_chain_ids
                        and chain_id not in used_chain_ids
                    ):
                        break
            if chain_id in used_chain_ids:
                raise ValueError(f"duplicate chain id: {chain_id!r}")
            used_chain_ids.add(chain_id)
            chains.append(
                {
                    **built["chain"],
                    "entity_id": entity,
                    "entity_number": entity_number,
                    "copy_id": copy_id,
                    "chain_id": chain_id,
                    "asym_id": len(chains),
                    "sym_id": copy_id - 1,
                    "polymer_type": built["polymer_type"],
                }
            )
    return chains


def _build_protein_chain(
    chain: dict[str, Any],
    *,
    base_dir: str | Path | None,
) -> dict[str, Any]:
    if not isinstance(chain, dict):
        raise ValueError("proteinChain entry must be an object")
    templates_path = chain.get("templatesPath") or None
    if templates_path:
        templates_path = str(_resolve_path(templates_path, base_dir=base_dir))
    sequence = _normalize_sequence(chain.get("sequence"))
    entries = [_ccd_component(AA1_TO_AA3[aa]) for aa in sequence]
    codes = [AA1_TO_AA3[aa] for aa in sequence]
    modified = np.zeros((len(sequence),), dtype=np.int64)
    for mod in _validate_modifications(
        chain.get("modifications", []),
        len(sequence),
        type_key="ptmType",
        position_key="ptmPosition",
    ):
        pos, code = mod
        entries[pos - 1] = _modified_ccd_component(code)
        codes[pos - 1] = code
        modified[pos - 1] = 1
    paired_a3m, unpaired_a3m = _read_chain_a3m(chain, base_dir=base_dir)
    count = int(chain.get("count", 1))
    if count <= 0:
        raise ValueError("proteinChain count must be positive")
    ids = chain.get("id")
    _validate_ids(ids, count, "proteinChain")
    return {
        "entity_key": f"protein:{','.join(codes)}",
        "count": count,
        "ids": ids,
        "chain": {
            "kind": "protein",
            "sequence": sequence,
            "paired_a3m": paired_a3m,
            "unpaired_a3m": unpaired_a3m,
            "templates_path": templates_path,
            "entries": entries,
            "codes": codes,
            "modified": modified,
        },
    }


def _build_ligand_chain(
    info: dict[str, Any], *, kind: str, base_dir: str | Path | None
) -> dict[str, Any]:
    """Build a ligand/ion chain from CCD codes (one token per heavy atom)."""

    if not isinstance(info, dict):
        raise ValueError(f"{kind} entry must be an object")
    if kind == "ion":
        ligand_str = f"CCD_{info['ion']}"
    else:
        ligand_str = info["ligand"]
        if not isinstance(ligand_str, str) or not ligand_str:
            raise ValueError("ligand string must be non-empty")
    residues = []
    if ligand_str.startswith("CCD_"):
        codes = ligand_str[4:].split("_")
        for res_id, code in enumerate(codes, start=1):
            residues.append((res_id, _ccd_component(code)))
    else:
        residues.append((1, _rdkit_ligand_entry(ligand_str, base_dir=base_dir)))
    count = int(info.get("count", 1))
    if count <= 0:
        raise ValueError(f"{kind} count must be positive")
    n_tok = sum(len(entry["names"]) for _res_id, entry in residues)
    gap = MSA_PROTEIN_INDEX["-"]
    msa = np.full((1, n_tok), gap, dtype=np.int64)
    deletion_matrix = np.zeros((1, n_tok), dtype=np.float32)
    ids = info.get("id")
    _validate_ids(ids, count, kind)
    return {
        "entity_key": f"ligand:{ligand_str}",
        "count": count,
        "ids": ids,
        "chain": {
            "kind": "ligand",
            "residues": residues,
            "msa": msa,
            "deletion_matrix": deletion_matrix,
        },
    }


def _build_nucleic_chain(
    info: dict[str, Any], *, kind: str, base_dir: str | Path | None
) -> dict[str, Any]:
    """Build a DNA/RNA chain (one per-residue token like protein)."""

    if not isinstance(info, dict):
        raise ValueError(f"{kind} entry must be an object")
    sequence = info.get("sequence")
    if not isinstance(sequence, str) or not sequence:
        raise ValueError(f"{kind} sequence must be a non-empty string")
    sequence = sequence.upper()
    code_map = DNA_CODES if kind == "dnaSequence" else RNA_CODES
    restype_map = DNA_RESTYPE_INDEX if kind == "dnaSequence" else RNA_RESTYPE_INDEX
    table = _ccd_nucleotides()
    codes = []
    entries = []
    for base in sequence:
        if base not in code_map:
            raise ValueError(f"unsupported {kind} base: {base}")
        codes.append(code_map[base])
        entries.append(_ccd_component(code_map[base]))
    modified = np.zeros((len(sequence),), dtype=np.int64)
    for pos, code in _validate_modifications(
        info.get("modifications", []),
        len(sequence),
        type_key="modificationType",
        position_key="basePosition",
    ):
        codes[pos - 1] = code
        entries[pos - 1] = _modified_ccd_component(code)
        modified[pos - 1] = 1
    count = int(info.get("count", 1))
    if count <= 0:
        raise ValueError(f"{kind} count must be positive")
    label = "dna" if kind == "dnaSequence" else "rna"
    ids = info.get("id")
    _validate_ids(ids, count, kind)
    unpaired_a3m = ""
    if kind == "rnaSequence":
        unpaired_a3m = info.get("unpairedMsa") or ""
        if not unpaired_a3m and info.get("unpairedMsaPath"):
            unpaired_a3m = _resolve_path(
                info["unpairedMsaPath"], base_dir=base_dir
            ).read_text(encoding="utf-8")
    n_tok = len(sequence)
    gap = MSA_PROTEIN_INDEX["-"]
    msa = np.full((1, n_tok), gap, dtype=np.int64)
    deletion_matrix = np.zeros((1, n_tok), dtype=np.float32)
    return {
        "entity_key": f"{label}:{','.join(codes)}",
        "count": count,
        "ids": ids,
        "chain": {
            "kind": "nucleic",
            "nucleic_type": label,
            "sequence": sequence,
            "codes": codes,
            "restype_map": restype_map,
            "table": table,
            "entries": entries,
            "restype_indices": [restype_map[code_map[b]] for b in sequence],
            "biological_restype_indices": [restype_map[code_map[b]] for b in sequence],
            "modified": modified,
            "unpaired_a3m": unpaired_a3m,
            "msa": msa,
            "deletion_matrix": deletion_matrix,
        },
    }


def _is_standard_polymer_code(code: str, *, expected_type: str) -> bool:
    if expected_type == "protein":
        return code in _ccd_std_residues()
    if expected_type == "dna":
        return code in DNA_RESTYPE_INDEX
    if expected_type == "rna":
        return code in RNA_RESTYPE_INDEX
    return False


def _prepare_polymer_tokens(chains: list[dict[str, Any]]) -> None:
    """Resolve canonical modified-residue semantics and token expansion."""

    for chain in chains:
        if chain["kind"] not in {"protein", "nucleic"}:
            residues = []
            for residue_id, entry in chain["residues"]:
                if str(entry.get("code", "")) == "MSE":
                    entry = _normalize_mse_entry(entry)
                residues.append((residue_id, entry))
            chain["residues"] = residues
            chain["token_to_sequence_idx"] = np.arange(
                _chain_biological_count(chain), dtype=np.int64
            )
            continue

        entries = list(chain["entries"])
        codes = list(chain["codes"])
        reference_codes = list(codes)
        modified = np.asarray(chain["modified"], dtype=np.int64).copy()
        if chain["kind"] == "protein":
            expected_type = "protein"
            restype_indices = [RESTYPE_INDEX[aa] for aa in chain["sequence"]]
        else:
            expected_type = chain["nucleic_type"]
            restype_indices = list(chain["restype_indices"])
        standard = np.ones((len(entries),), dtype=np.int64)

        for index, entry in enumerate(entries):
            was_modified = bool(modified[index])
            if was_modified:
                restype_indices[index] = _canonical_modified_restype(
                    entry,
                    expected_type=expected_type,
                )
                if expected_type == "protein" and codes[index] == "MSE":
                    entries[index] = _normalize_mse_entry(entry)
                    codes[index] = "MET"
                if _is_standard_polymer_code(codes[index], expected_type=expected_type):
                    modified[index] = 0
                else:
                    standard[index] = 0

                terminal_atoms = []
                if chain["kind"] == "protein" and index + 1 < len(entries):
                    terminal_atoms.append("OXT")
                if chain["kind"] == "nucleic" and index > 0:
                    terminal_atoms.append("OP3")
                if terminal_atoms:
                    entries[index] = _entry_without_atoms(
                        entries[index], terminal_atoms
                    )

        token_to_sequence_idx = []
        for sequence_idx, (entry, is_standard) in enumerate(
            zip(entries, standard, strict=True)
        ):
            repeat = 1 if is_standard else len(entry["names"])
            token_to_sequence_idx.extend([sequence_idx] * repeat)

        chain["entries"] = entries
        chain["codes"] = codes
        chain["reference_codes"] = reference_codes
        chain["modified"] = modified
        chain["restype_indices"] = restype_indices
        chain["is_standard_polymer"] = standard
        chain["token_to_sequence_idx"] = np.asarray(
            token_to_sequence_idx, dtype=np.int64
        )


def _chain_biological_count(chain: dict[str, Any]) -> int:
    if chain["kind"] == "protein":
        return len(chain["sequence"])
    if chain["kind"] == "nucleic":
        return len(chain["codes"])
    return sum(len(entry["names"]) for _res_id, entry in chain["residues"])


def _chain_token_count(chain: dict[str, Any]) -> int:
    return len(chain["token_to_sequence_idx"])


def _emit_protein_tokens(chain: dict[str, Any], state: dict[str, Any]) -> None:
    sequence = chain["sequence"]
    for pos, _aa in enumerate(sequence, start=1):
        entry = chain["entries"][pos - 1]
        if not chain["is_standard_polymer"][pos - 1]:
            _emit_polymer_atom_tokens(
                chain,
                state,
                position=pos,
                entry=entry,
                restype_index=chain["restype_indices"][pos - 1],
                polymer_type="protein",
            )
            continue
        token_i = state["token_i"]
        ref_space_uid = state["next_ref_space_uid"]
        state["next_ref_space_uid"] += 1
        state["restype"][token_i, chain["restype_indices"][pos - 1]] = 1.0
        state["residue_index"][token_i] = pos
        state["asym_id"][token_i] = chain["asym_id"]
        state["entity_id"][token_i] = chain["entity_id"]
        state["sym_id"][token_i] = chain["sym_id"]
        _append_entry_bonds(state, entry)
        state["token_entity_number"][token_i] = chain["entity_number"]
        state["token_copy_id"][token_i] = chain["copy_id"]
        state["token_ccd_codes"][token_i] = chain["codes"][pos - 1]
        state["token_is_modified"][token_i] = chain["modified"][pos - 1]
        state["token_is_standard_polymer"][token_i] = 1
        state["token_polymer_type"][token_i] = _TOKEN_POLYMER_TYPE["protein"]
        state["token_reference_is_mse"][token_i] = int(
            chain["reference_codes"][pos - 1] == "MSE"
        )
        names = entry["names"]
        is_cterm = pos == len(sequence)
        normalized_code = chain["codes"][pos - 1]
        rep = "CA" if normalized_code == "GLY" else "CB"
        canonical_names = np.asarray(
            _ccd_std_residues()[normalized_code]["names"]
        ).astype(str)
        canonical_slots = {
            atom_name: index for index, atom_name in enumerate(canonical_names)
        }
        for j, atom_name in enumerate(names):
            atom_name = str(atom_name)
            if atom_name == "OXT" and not is_cterm:
                continue
            if atom_name not in canonical_slots:
                raise ValueError(
                    f"standard residue {normalized_code!r} contains unknown atom "
                    f"{atom_name!r}"
                )
            state["atom_to_token_idx"].append(token_i)
            state["atom_to_tokatom_idx"].append(canonical_slots[atom_name])
            xyz = entry["coord"][j]
            state["ref_pos"].append((float(xyz[0]), float(xyz[1]), float(xyz[2])))
            state["ref_space_uid"].append(ref_space_uid)
            state["ref_charge"].append(float(entry["charge"][j]))
            state["ref_element"].append(str(entry["elem"][j]))
            state["ref_atom_names"].append(atom_name)
            state["distogram_rep_atom_mask"].append(int(atom_name == rep))
            state["ref_mask"].append(float(entry["mask"][j]))
            state["mol_id"].append(chain["asym_id"])
            _append_atom_metadata(state, chain, pos, entry, j)
        state["token_i"] = token_i + 1


def _emit_polymer_atom_tokens(
    chain: dict[str, Any],
    state: dict[str, Any],
    *,
    position: int,
    entry: dict[str, Any],
    restype_index: int,
    polymer_type: str,
) -> None:
    """Emit one token per atom for a non-standard polymer component."""

    _append_entry_bonds(state, entry)
    ref_space_uid = state["next_ref_space_uid"]
    state["next_ref_space_uid"] += 1
    res_start = state["token_i"]
    for left, right in np.asarray(entry["bonds"], dtype=np.int64).reshape((-1, 2)):
        state["token_bond_edges"].append(
            (res_start + int(left), res_start + int(right))
        )

    for atom_index, atom_name in enumerate(entry["names"]):
        token_i = state["token_i"]
        state["restype"][token_i, restype_index] = 1.0
        state["residue_index"][token_i] = position
        state["asym_id"][token_i] = chain["asym_id"]
        state["entity_id"][token_i] = chain["entity_id"]
        state["sym_id"][token_i] = chain["sym_id"]
        state["token_entity_number"][token_i] = chain["entity_number"]
        state["token_copy_id"][token_i] = chain["copy_id"]
        state["token_ccd_codes"][token_i] = str(entry["code"])
        state["token_is_modified"][token_i] = 1
        state["token_polymer_type"][token_i] = _TOKEN_POLYMER_TYPE[polymer_type]
        state["token_reference_is_mse"][token_i] = int(entry["code"] == "MSE")

        xyz = entry["coord"][atom_index]
        state["atom_to_token_idx"].append(token_i)
        state["atom_to_tokatom_idx"].append(0)
        state["ref_pos"].append((float(xyz[0]), float(xyz[1]), float(xyz[2])))
        state["ref_space_uid"].append(ref_space_uid)
        state["ref_charge"].append(float(entry["charge"][atom_index]))
        state["ref_element"].append(str(entry["elem"][atom_index]))
        state["ref_atom_names"].append(str(atom_name))
        state["distogram_rep_atom_mask"].append(1)
        state["ref_mask"].append(float(entry["mask"][atom_index]))
        state["mol_id"].append(chain["asym_id"])
        _append_atom_metadata(state, chain, position, entry, atom_index)
        state["token_i"] = token_i + 1


def _emit_ligand_tokens(chain: dict[str, Any], state: dict[str, Any]) -> None:
    """One token per ligand heavy atom; restype=UNK(20), tokatom_idx=0."""

    for res_id, entry in chain["residues"]:
        names = entry["names"]
        _append_entry_bonds(state, entry)
        ref_space_uid = state["next_ref_space_uid"]
        state["next_ref_space_uid"] += 1
        res_start = state["token_i"]
        for a, b in entry.get("bonds", np.zeros((0, 2), dtype=np.int64)):
            # ligand: one token per atom, so atom-index == token offset
            state["token_bond_edges"].append((res_start + int(a), res_start + int(b)))
        for j in range(len(names)):
            token_i = state["token_i"]
            state["restype"][token_i, RESTYPE_INDEX["X"]] = 1.0
            state["residue_index"][token_i] = res_id
            state["asym_id"][token_i] = chain["asym_id"]
            state["entity_id"][token_i] = chain["entity_id"]
            state["sym_id"][token_i] = chain["sym_id"]
            state["token_entity_number"][token_i] = chain["entity_number"]
            state["token_copy_id"][token_i] = chain["copy_id"]
            state["token_ccd_codes"][token_i] = str(entry.get("code", "UNL"))
            state["token_polymer_type"][token_i] = _TOKEN_POLYMER_TYPE["ligand"]
            state["token_reference_is_mse"][token_i] = int(
                entry.get("reference_is_mse", False)
            )
            xyz = entry["coord"][j]
            state["atom_to_token_idx"].append(token_i)
            state["atom_to_tokatom_idx"].append(0)
            state["ref_pos"].append((float(xyz[0]), float(xyz[1]), float(xyz[2])))
            state["ref_space_uid"].append(ref_space_uid)
            state["ref_charge"].append(float(entry["charge"][j]))
            state["ref_element"].append(str(entry["elem"][j]))
            state["ref_atom_names"].append(str(names[j]))
            state["distogram_rep_atom_mask"].append(1)
            state["ref_mask"].append(float(entry["mask"][j]))
            state["mol_id"].append(chain["asym_id"])
            _append_atom_metadata(state, chain, res_id, entry, j)
            state["token_i"] = token_i + 1


def _emit_nucleic_tokens(chain: dict[str, Any], state: dict[str, Any]) -> None:
    """One token per nucleotide; OP3 kept only on the 5'-terminal residue."""

    codes = chain["codes"]
    for pos, code in enumerate(codes, start=1):
        entry = chain["entries"][pos - 1]
        if not chain["is_standard_polymer"][pos - 1]:
            _emit_polymer_atom_tokens(
                chain,
                state,
                position=pos,
                entry=entry,
                restype_index=chain["restype_indices"][pos - 1],
                polymer_type=chain["nucleic_type"],
            )
            continue
        token_i = state["token_i"]
        ref_space_uid = state["next_ref_space_uid"]
        state["next_ref_space_uid"] += 1
        state["restype"][token_i, chain["restype_indices"][pos - 1]] = 1.0
        state["residue_index"][token_i] = pos
        state["asym_id"][token_i] = chain["asym_id"]
        state["entity_id"][token_i] = chain["entity_id"]
        state["sym_id"][token_i] = chain["sym_id"]
        _append_entry_bonds(state, entry)
        state["token_entity_number"][token_i] = chain["entity_number"]
        state["token_copy_id"][token_i] = chain["copy_id"]
        state["token_ccd_codes"][token_i] = code
        state["token_is_modified"][token_i] = chain["modified"][pos - 1]
        state["token_is_standard_polymer"][token_i] = 1
        state["token_polymer_type"][token_i] = _TOKEN_POLYMER_TYPE[
            chain["nucleic_type"]
        ]
        names = entry["names"]
        is_first = pos == 1
        rep = "C4" if code in _PURINE_CODES else "C2"
        for j, atom_name in enumerate(names):
            atom_name = str(atom_name)
            if atom_name == "OP3" and not is_first:
                continue
            state["atom_to_token_idx"].append(token_i)
            state["atom_to_tokatom_idx"].append(j)
            xyz = entry["coord"][j]
            state["ref_pos"].append((float(xyz[0]), float(xyz[1]), float(xyz[2])))
            state["ref_space_uid"].append(ref_space_uid)
            state["ref_charge"].append(float(entry["charge"][j]))
            state["ref_element"].append(str(entry["elem"][j]))
            state["ref_atom_names"].append(atom_name)
            state["distogram_rep_atom_mask"].append(int(atom_name == rep))
            state["ref_mask"].append(float(entry["mask"][j]))
            state["mol_id"].append(chain["asym_id"])
            _append_atom_metadata(state, chain, pos, entry, j)
        state["token_i"] = token_i + 1


_GAP_IDX = MSA_PROTEIN_INDEX["-"]
_MSA_PAD_VALUES = {"msa": _GAP_IDX, "deletion_matrix": 0}
_UNIPROT_REGEX = re.compile(
    r"(?:tr|sp)\|[A-Z0-9]{6,10}(?:_\d+)?\|"
    r"(?:[A-Z0-9]{1,10}_)(?P<SpeciesId>[A-Z0-9]{1,5})"
)
_UNIREF_REGEX = re.compile(r"^UniRef100_[^_]+_([^_/]+)")


def _read_chain_a3m(
    chain: dict[str, Any],
    *,
    base_dir: str | Path | None,
) -> tuple[str, str]:
    """Read paired/unpaired A3M for a protein chain (inline or precomputed)."""

    def _resolve(inline_key: str, path_key: str) -> str:
        inline = chain.get(inline_key)
        if inline:
            return inline
        path = chain.get(path_key)
        if path:
            return _resolve_path(path, base_dir=base_dir).read_text(encoding="utf-8")
        return ""

    paired = _resolve("pairedMsa", "pairedMsaPath")
    unpaired = _resolve("unpairedMsa", "unpairedMsaPath")
    legacy = chain.get("msa")
    if legacy is not None and not paired and not unpaired:
        if not isinstance(legacy, dict):
            raise ValueError("deprecated protein msa field must be an object")
        unknown = set(legacy) - {"precomputed_msa_dir", "pairing_db"}
        if unknown:
            raise ValueError(f"unsupported deprecated msa fields: {sorted(unknown)}")
        directory_value = legacy.get("precomputed_msa_dir")
        if not isinstance(directory_value, (str, Path)) or not str(directory_value):
            raise ValueError("deprecated msa.precomputed_msa_dir is required")
        directory = Path(directory_value)
        if not directory.is_absolute() and base_dir is not None:
            directory = Path(base_dir) / directory
        if not directory.is_dir():
            raise ValueError(f"legacy MSA directory does not exist: {directory}")
        paired_path = directory / "pairing.a3m"
        unpaired_path = directory / "non_pairing.a3m"
        if not paired_path.is_file() and not unpaired_path.is_file():
            raise ValueError(
                "legacy MSA directory contains no pairing.a3m or "
                f"non_pairing.a3m: {directory}"
            )
        if paired_path.is_file():
            paired = paired_path.read_text(encoding="utf-8")
        if unpaired_path.is_file():
            unpaired = unpaired_path.read_text(encoding="utf-8")
    return paired, unpaired


def _species_id(description: str) -> str:
    """Extract a species identifier from a UniProt/UniRef description line."""

    desc = description.strip()
    m = _UNIPROT_REGEX.match(desc) or _UNIREF_REGEX.match(desc)
    if not m:
        return ""
    return m.group("SpeciesId") if "SpeciesId" in m.groupdict() else m.group(1)


def _ensure_trailing_newline(a3m: str) -> str:
    """Match torch ``ensure_ends_with_newline`` so concatenated A3M records
    don't merge the last sequence of one block with the first header of the
    next."""
    if a3m and not a3m.endswith("\n"):
        return a3m + "\n"
    return a3m


def _featurize_a3m(
    query: str,
    a3m: str,
    *,
    dedup: bool,
) -> dict[str, np.ndarray]:
    """Port of torch ``RawMsa(...).featurize`` for protein chains."""

    seqs, descs = _parse_a3m_records(a3m)
    if dedup:
        seqs, descs = _dedup_sequences(seqs, descs)
    if not seqs:
        seqs, descs = [query], ["Original query"]
    cols = len(query)
    msa = np.zeros((len(seqs), cols), dtype=np.int64)
    deletion = np.zeros((len(seqs), cols), dtype=np.int64)
    for i, sequence in enumerate(seqs):
        row, dels = _aligned_protein_row(sequence)
        n = min(len(row), cols)
        msa[i, :n] = row[:n]
        deletion[i, :n] = dels[:n]
    return {
        "msa": msa,
        "deletion_matrix": deletion,
        "species": np.array([_species_id(d) for d in descs], dtype=object),
    }


def _featurize_rna_a3m(
    query: str,
    a3m: str,
    *,
    dedup: bool,
) -> dict[str, np.ndarray]:
    """Parse RNA A3M using Protenix RNA residue indices."""

    seqs, descs = _parse_a3m_records(a3m)
    if dedup:
        seqs, descs = _dedup_sequences(seqs, descs)
    if not seqs:
        seqs, descs = [query], ["Original query"]
    rows = []
    deletions = []
    for sequence in seqs:
        row, deletion = _aligned_rna_row(sequence)
        if len(row) != len(query):
            raise ValueError(
                "RNA MSA aligned length does not match query: "
                f"{len(row)} != {len(query)}"
            )
        rows.append(row)
        deletions.append(deletion)
    return {
        "msa": np.asarray(rows, dtype=np.int64),
        "deletion_matrix": np.asarray(deletions, dtype=np.int64),
        "species": np.array([_species_id(d) for d in descs], dtype=object),
    }


def _dedup_sequences(seqs: list[str], descs: list[str]) -> tuple[list[str], list[str]]:
    """Remove duplicate sequences ignoring insertion (lowercase) columns."""

    table = str.maketrans("", "", string.ascii_lowercase)
    u_seqs: list[str] = []
    u_descs: list[str] = []
    seen: set[str] = set()
    for s, d in zip(seqs, descs):
        stripped = s.translate(table)
        if stripped not in seen:
            seen.add(stripped)
            u_seqs.append(s)
            u_descs.append(d)
    return u_seqs, u_descs


def _gap_only_chain_features(width: int) -> dict[str, np.ndarray]:
    """Single gap row used for ligand chains (torch placeholder)."""

    return {
        "msa": np.full((1, width), _GAP_IDX, dtype=np.int64),
        "deletion_matrix": np.zeros((1, width), dtype=np.int64),
        "species": np.array([""], dtype=object),
    }


def _ligand_query_features(width: int) -> dict[str, np.ndarray]:
    """Single query row for a ligand chain = UNK (restype X = 20) per token.

    Torch builds ligand MSA via ``RawMsa(seq="X"*n, PROTEIN_CHAIN, [], [])`` whose
    query fallback encodes each token as UNK (index 20), so the per-token profile
    is a one-hot of UNK, not a gap. jax previously gap-filled ligand chains.
    Mirrors msa_featurizer.py:225 (ligand placeholder)."""

    unk = RESTYPE_INDEX["X"]
    return {
        "msa": np.full((1, width), unk, dtype=np.int64),
        "deletion_matrix": np.zeros((1, width), dtype=np.int64),
        "species": np.array([""], dtype=object),
    }


def _nucleic_query_features(chain: dict[str, Any]) -> dict[str, np.ndarray]:
    """Single query row for a DNA/RNA chain in MSA-restype space.

    Torch builds nucleic MSA via ``RawMsa.from_a3m(seq, ctype, "")`` whose
    empty-A3M fallback yields the query sequence as the only row (encoded with
    DNA/RNA STD_RESIDUES indices), so the per-token profile is a one-hot of the
    base, not a gap. jax previously gap-filled nucleic chains, corrupting the
    profile for those tokens. Mirrors msa_featurizer.py:204 + RawMsa.__init__.
    """

    row = np.asarray([chain["biological_restype_indices"]], dtype=np.int64)
    return {
        "msa": row,
        "deletion_matrix": np.zeros((1, row.shape[1]), dtype=np.int64),
        "species": np.array([""], dtype=object),
    }


def _align_species(
    all_species: list[str],
    chain_species_map: list[dict[str, np.ndarray]],
    species_min_hits: dict[str, int],
) -> np.ndarray:
    blocks = []
    for species in all_species:
        rows = []
        for s2r in chain_species_map:
            n = species_min_hits[species]
            if species not in s2r:
                rows.append(np.full(n, -1, dtype=np.int32))
            else:
                rows.append(s2r[species][:n])
        blocks.append(np.stack(rows, axis=1))
    return np.concatenate(blocks, axis=0)


def _pair_chains_by_species(
    chains: list[dict[str, np.ndarray]],
    max_paired: int,
    active: set[int],
    max_per_species: int,
) -> list[dict[str, np.ndarray]]:
    """Port of torch ``MSAPairingEngine.pair_chains_by_species``."""

    chain_species_map: list[dict[str, np.ndarray]] = []
    all_counts: dict[str, int] = {}
    min_hits: dict[str, int] = {}
    for c in chains:
        ids = c.get("species_all_seq", np.array([], dtype=object))
        no_species = ids.size == 0 or (ids.size == 1 and not ids[0])
        if no_species or c["asym_id"] not in active:
            chain_species_map.append({})
            continue
        row_idx = np.arange(len(ids))
        order = ids.argsort(kind="stable")
        ids_s = ids[order]
        row_idx = row_idx[order]
        species, uniq = np.unique(ids_s, return_index=True)
        grouped = np.split(row_idx, uniq[1:])
        mapping = dict(zip(species, grouped))
        chain_species_map.append(mapping)
        for s in species:
            all_counts[s] = all_counts.get(s, 0) + 1
        for s, idxs in mapping.items():
            min_hits[s] = min(min_hits.get(s, max_per_species), len(idxs))

    ranked: dict[int, list[str]] = {}
    for s, count in all_counts.items():
        if not s or count <= 1:
            continue
        ranked.setdefault(count, []).append(s)

    pair_idxs = [np.zeros((1, len(chains)), dtype=np.int32)]
    total = 0
    for count in sorted(ranked.keys(), reverse=True):
        rows = _align_species(ranked[count], chain_species_map, min_hits)
        rank = np.sum(np.log(np.abs(rows.astype(np.float32)) + 1e-10), axis=1)
        pair_idxs.append(rows[np.argsort(rank)])
        total += rows.shape[0]
        if total >= max_paired:
            break
    final_idxs = np.concatenate(pair_idxs, axis=0)[:max_paired]

    new_chains = []
    for i, c in enumerate(chains):
        nc = {k: v for k, v in c.items() if "all_seq" not in k}
        sel = final_idxs[:, i]
        for f in ["msa", "deletion_matrix"]:
            src = c[f"{f}_all_seq"]
            pad = np.full((1, src.shape[1]), _MSA_PAD_VALUES[f], src.dtype)
            padded = np.concatenate([src, pad], axis=0)
            nc[f"{f}_all_seq"] = padded[sel]
        new_chains.append(nc)
    return new_chains


def _cleanup_unpaired(
    chains: list[dict[str, np.ndarray]],
) -> list[dict[str, np.ndarray]]:
    """Drop unpaired rows already present in the paired MSA (torch port)."""

    for c in chains:
        paired_bytes = {row.tobytes() for row in c["msa_all_seq"].astype(np.int8)}
        keep = [
            i
            for i, row in enumerate(c["msa"].astype(np.int8))
            if row.tobytes() not in paired_bytes
        ]
        c["msa"] = c["msa"][keep]
        c["deletion_matrix"] = c["deletion_matrix"][keep]
    return chains


def _filter_all_gapped_rows(
    chains: list[dict[str, np.ndarray]],
    active: set[int],
) -> list[dict[str, np.ndarray]]:
    """Remove all-gap rows from the paired MSA across active chains."""

    subset = [c["msa_all_seq"] for c in chains if c["asym_id"] in active]
    if not subset:
        return chains
    non_gap = np.any(np.concatenate(subset, axis=1) != _GAP_IDX, axis=1)
    for c in chains:
        c["msa_all_seq"] = c["msa_all_seq"][non_gap]
        c["deletion_matrix_all_seq"] = c["deletion_matrix_all_seq"][non_gap]
    return chains


def _merge_feature(chains: list[dict[str, np.ndarray]], key: str) -> np.ndarray:
    if "_all_seq" in key:
        return np.concatenate([c[key] for c in chains], axis=1)
    max_d = max(c[key].shape[0] for c in chains)
    base = key
    pads = [
        np.pad(
            c[key],
            ((0, max_d - c[key].shape[0]), (0, 0)),
            constant_values=_MSA_PAD_VALUES.get(base, 0),
        )
        for c in chains
    ]
    return np.concatenate(pads, axis=1)


def _assemble_msa_features(
    chains: list[dict[str, Any]],
    *,
    max_msa_depth: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Port of torch ``FeatureAssemblyLine.assemble`` for inference inputs."""

    unique_prot_seqs = {c["sequence"] for c in chains if c["kind"] == "protein"}
    need_pairing = len(unique_prot_seqs) > 1
    active = {c["asym_id"] for c in chains}

    raw_chains: list[dict[str, np.ndarray]] = []
    for c in chains:
        width = _chain_biological_count(c)
        if c["kind"] == "protein":
            skip = len(c["sequence"]) <= 4
            # torch inference sets msa_pair_as_unpair=True (configs_inference.py):
            # the paired A3M is folded into the unpaired stack (pairing rows
            # first), then deduped together. Without this jax drops pairing.a3m
            # for single-protein complexes, halving MSA depth and corrupting
            # profile/deletion_mean. Mirrors msa_featurizer.py:654.
            # Scope: only when NOT pairing (single unique protein). There the
            # paired A3M would otherwise be discarded entirely (verified vs torch
            # capture on 7r6r). For multimers the paired stack is consumed by the
            # cross-chain pairing engine, and the assemble-level golden test
            # encodes that path, so leave it untouched.
            unpaired_a3m = c["unpaired_a3m"]
            if not skip and not need_pairing and c["paired_a3m"]:
                unpaired_a3m = (
                    _ensure_trailing_newline(c["paired_a3m"]) + c["unpaired_a3m"]
                )
            up = _featurize_a3m(
                c["sequence"],
                "" if skip else unpaired_a3m,
                dedup=True,
            )
            p = _featurize_a3m(
                c["sequence"],
                c["paired_a3m"] if (need_pairing and not skip) else "",
                dedup=False,
            )
        elif c["kind"] == "nucleic":
            # torch from_a3m("") fallback puts the query row in both stacks.
            if c["unpaired_a3m"] and width > 4:
                query = c["sequence"]
                up = _featurize_rna_a3m(
                    query,
                    c["unpaired_a3m"],
                    dedup=True,
                )
            else:
                up = _nucleic_query_features(c)
            p = _nucleic_query_features(c)
        else:
            up = _ligand_query_features(width)
            p = _ligand_query_features(width)
        feat: dict[str, np.ndarray] = dict(up)
        feat.update({f"{k}_all_seq": v for k, v in p.items()})
        feat["asym_id"] = c["asym_id"]
        msa = feat["msa"]
        prof = (msa[..., None] == np.arange(32)).sum(axis=0) / msa.shape[0]
        feat["profile"] = prof.astype(np.float32)
        feat["deletion_mean"] = np.mean(feat["deletion_matrix"], axis=0)
        raw_chains.append(feat)

    max_p = max_msa_depth // 2
    if need_pairing:
        raw_chains = _pair_chains_by_species(raw_chains, max_p, active, 600)
        raw_chains = _cleanup_unpaired(raw_chains)
    if "msa_all_seq" in raw_chains[0]:
        raw_chains = _filter_all_gapped_rows(raw_chains, active)

    cropped = []
    for c in raw_chains:
        p_msa = c.get("msa_all_seq")
        ps = min(p_msa.shape[0], max_p) if p_msa is not None else 0
        us = max(0, min(c["msa"].shape[0], max_msa_depth - ps))
        cr: dict[str, np.ndarray] = {
            "asym_id": c["asym_id"],
            "profile": c["profile"],
            "deletion_mean": c["deletion_mean"],
        }
        for k in ("msa", "deletion_matrix"):
            cr[k] = c[k][:us]
            if f"{k}_all_seq" in c:
                cr[f"{k}_all_seq"] = c[f"{k}_all_seq"][:ps]
        cropped.append(cr)

    merged: dict[str, np.ndarray] = {}
    for base in ("msa", "deletion_matrix"):
        for f in (base, f"{base}_all_seq"):
            if f in cropped[0]:
                merged[f] = _merge_feature(cropped, f)
    profile = np.concatenate([c["profile"] for c in cropped], axis=0)
    deletion_mean = np.concatenate([c["deletion_mean"] for c in cropped], axis=0)

    max_u = max(c["msa"].shape[0] for c in cropped if c["asym_id"] in active)
    merged["msa"] = merged["msa"][:max_u]
    merged["deletion_matrix"] = merged["deletion_matrix"][:max_u]
    if "msa_all_seq" in merged:
        max_pa = max(
            c["msa_all_seq"].shape[0] for c in cropped if c["asym_id"] in active
        )
        merged["msa_all_seq"] = merged["msa_all_seq"][:max_pa]
        merged["deletion_matrix_all_seq"] = merged["deletion_matrix_all_seq"][:max_pa]
        for base in ("msa", "deletion_matrix"):
            merged[base] = np.concatenate(
                [merged[f"{base}_all_seq"], merged[base]], axis=0
            )

    msa = merged["msa"].astype(np.int64)
    deletion_matrix = merged["deletion_matrix"].astype(np.float32)

    # Forward-compat patch for non-protein entities (torch msa_featurizer.py:335).
    # Rows that are all-gap across a DNA/RNA/ligand chain's columns are filled
    # from the query row (row 0), so those columns carry the query restype in
    # every MSA row rather than gap padding. Without this every protein-MSA row
    # keeps gaps in the nucleic columns and z_trunk diverges from torch.
    col = 0
    for c in chains:
        width = _chain_biological_count(c)
        if c["kind"] != "protein":
            gap_rows = np.all(msa[:, col : col + width] == _GAP_IDX, axis=1)
            msa[gap_rows, col : col + width] = msa[0, col : col + width]
        col += width

    gather = []
    biological_offset = 0
    for c in chains:
        gather.extend(
            biological_offset + np.asarray(c["token_to_sequence_idx"], dtype=np.int64)
        )
        biological_offset += _chain_biological_count(c)
    gather_idx = np.asarray(gather, dtype=np.int64)
    msa = msa[:, gather_idx]
    deletion_matrix = deletion_matrix[:, gather_idx]
    profile = profile[gather_idx]
    deletion_mean = deletion_mean[gather_idx]

    return (
        msa,
        deletion_matrix,
        profile.astype(np.float32),
        deletion_mean.astype(np.float32),
    )


def _default_chain_id(index: int) -> str:
    """Return Excel-like chain IDs (A..Z, AA..), matching upstream ordering."""

    value = index + 1
    chars = []
    while value:
        value, remainder = divmod(value - 1, 26)
        chars.append(chr(ord("A") + remainder))
    return "".join(reversed(chars))


def _validate_ids(ids: Any, count: int, kind: str) -> None:
    if ids is None:
        return
    if (
        not isinstance(ids, list)
        or len(ids) != count
        or not all(isinstance(value, str) and value for value in ids)
        or len(set(ids)) != len(ids)
    ):
        raise ValueError(f"{kind} id must contain {count} unique non-empty strings")


def _validate_modifications(
    modifications: Any,
    length: int,
    *,
    type_key: str,
    position_key: str,
) -> list[tuple[int, str]]:
    if modifications is None:
        return []
    if not isinstance(modifications, list):
        raise ValueError("modifications must be a list")
    result = []
    seen = set()
    for modification in modifications:
        if not isinstance(modification, dict):
            raise ValueError("each modification must be an object")
        unknown = set(modification) - {type_key, position_key}
        if unknown:
            raise ValueError(f"unsupported modification fields: {sorted(unknown)}")
        try:
            position = int(modification[position_key])
            value = modification[type_key]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("modification type and position are required") from exc
        if position < 1 or position > length:
            raise ValueError(
                f"modification position {position} outside sequence length {length}"
            )
        if position in seen:
            raise ValueError(f"duplicate modification position: {position}")
        if not isinstance(value, str) or not value.startswith("CCD_"):
            raise ValueError(f"unknown modification type: {value!r}")
        seen.add(position)
        result.append((position, value[4:]))
    return result


def _ccd_component(code: str) -> dict[str, np.ndarray]:
    """Return a normalized heavy-atom CCD entry from the vendored tables."""

    entry = None
    if code in _ccd_ligands():
        entry = _ccd_ligands()[code]
    elif code in _ccd_nucleotides():
        entry = _ccd_nucleotides()[code]
    elif code in _ccd_std_residues():
        entry = _ccd_std_residues()[code]
    elif code in ("N", "DN"):
        # CCD N/DN are the RNA/DNA phosphate-sugar backbone without a base.
        # Derive them from the vendored C/DC entries by retaining atoms through
        # C1'. This matches upstream's N/DN heavy-atom layout without requiring
        # the optional 490 MB full CCD database for a documented input symbol.
        source = _ccd_nucleotides()["C" if code == "N" else "DC"]
        names = np.asarray(source["names"]).astype(str)
        c1_index = int(np.flatnonzero(names == "C1'")[0])
        keep = slice(0, c1_index + 1)
        entry = {
            key: np.asarray(source[key])[keep]
            for key in ("names", "coord", "charge", "mask", "elem")
        }
    if entry is None:
        entry = _external_ccd_component(code)
    normalized = dict(entry)
    size = len(normalized["names"])
    normalized.setdefault("mask", np.ones((size,), dtype=np.float32))
    normalized.setdefault("bonds", np.zeros((0, 2), dtype=np.int64))
    normalized.setdefault(
        "bond_order", np.ones((len(normalized["bonds"]),), dtype=np.float32)
    )
    normalized.setdefault(
        "bond_stereo", np.zeros((len(normalized["bonds"]),), dtype=np.int64)
    )
    normalized.setdefault("input_index", np.arange(size, dtype=np.int64))
    normalized.setdefault("stereo", np.zeros((size,), dtype=np.int64))
    normalized["code"] = code
    return normalized


def _modified_ccd_component(code: str) -> dict[str, Any]:
    """Load a CCD modification with authoritative polymer annotations."""

    metadata = _external_ccd_atom_metadata(code)
    if metadata["canonical_polymer_type"] is None:
        raise ValueError(
            f"CCD modification {code!r} does not establish a canonical polymer type"
        )
    if metadata["one_letter_code"] is None:
        raise ValueError(
            f"CCD modification {code!r} does not establish a canonical restype"
        )
    entry = _ccd_component(code)
    names = np.asarray(entry["names"]).astype(str)
    metadata_names = np.asarray(metadata["names"]).astype(str)
    if not np.array_equal(names, metadata_names):
        raise ValueError(
            f"CCD code {code!r} has inconsistent atom identity between its "
            "component and components.cif"
        )
    annotated: dict[str, Any] = dict(entry)
    annotated["leaving_atom_flag"] = np.asarray(
        metadata["leaving_atom_flag"], dtype=bool
    )
    annotated["canonical_polymer_type"] = metadata["canonical_polymer_type"]
    annotated["one_letter_code"] = metadata["one_letter_code"]
    return annotated


def _canonical_modified_restype(entry: dict[str, Any], *, expected_type: str) -> int:
    code = str(entry["code"])
    canonical_type = entry.get("canonical_polymer_type")
    if canonical_type != expected_type:
        raise ValueError(
            f"CCD modification {code!r} does not establish a canonical polymer "
            f"type compatible with {expected_type!r}"
        )
    one_letter = entry.get("one_letter_code")
    if not isinstance(one_letter, str) or len(one_letter) != 1:
        raise ValueError(
            f"CCD modification {code!r} does not establish a canonical restype"
        )
    if expected_type == "protein":
        if one_letter not in RESTYPE_INDEX or one_letter == "X":
            raise ValueError(
                f"CCD modification {code!r} has unsupported canonical restype "
                f"{one_letter!r}"
            )
        return RESTYPE_INDEX[one_letter]
    if expected_type == "dna":
        residue_code = DNA_CODES.get(one_letter)
        if residue_code is not None and residue_code != "DN":
            return DNA_RESTYPE_INDEX[residue_code]
    elif expected_type == "rna":
        residue_code = RNA_CODES.get(one_letter)
        if residue_code is not None and residue_code != "N":
            return RNA_RESTYPE_INDEX[residue_code]
    raise ValueError(
        f"CCD modification {code!r} has unsupported canonical restype "
        f"{one_letter!r} for {expected_type!r}"
    )


def _normalize_mse_entry(entry: dict[str, Any]) -> dict[str, Any]:
    """Apply upstream's MSE-to-MET identity normalization after CCD loading."""

    normalized = dict(entry)
    names = np.asarray(entry["names"]).astype(str).copy()
    elements = np.asarray(entry["elem"]).astype(str).copy()
    selenium = np.flatnonzero(names == "SE")
    if selenium.size != 1 or elements[selenium[0]].upper() != "SE":
        raise ValueError("CCD MSE must contain exactly one selenium atom named 'SE'")
    names[selenium[0]] = "SD"
    elements[selenium[0]] = "S"
    normalized["names"] = names
    normalized["elem"] = elements
    normalized["code"] = "MET"
    normalized["reference_is_mse"] = True
    return normalized


def _managed_ccd_asset(name: str) -> Path:
    """Where `foldjax weights fetch` puts a shared CCD asset.

    Both CCD files are declared `shared=True` in the asset registry, so the
    store keeps one copy under `assets/` for Protenix and OpenDDE together.
    Resolving it here is what makes a fetched asset actually reachable: the
    OpenDDE CLI exports these paths into the environment, but the Protenix
    backend never did, so a fetched components.cif still produced "set
    PROTENIX_CCD_COMPONENTS_FILE" on the first modified residue or ligand.
    """

    from foldjax.paths import assets_dir

    return assets_dir() / name


def _external_ccd_component(code: str) -> dict[str, np.ndarray]:
    """Load an arbitrary CCD component from the official Protenix assets."""

    global _EXTERNAL_CCD_MOLS
    if _EXTERNAL_CCD_MOLS is None:
        candidates = []
        configured = os.environ.get("PROTENIX_CCD_RDKIT_MOL_FILE")
        if configured:
            candidates.append(Path(configured))
        candidates.append(_managed_ccd_asset("components.cif.rdkit_mol.pkl"))
        cache_path = next((path for path in candidates if path.is_file()), None)
        if cache_path is None:
            raise ValueError(
                f"CCD code {code!r} is not vendored; set "
                "PROTENIX_CCD_RDKIT_MOL_FILE to components.cif.rdkit_mol.pkl"
            )
        try:
            _EXTERNAL_CCD_MOLS = _load_verified_rdkit_cache(cache_path)
        except (ImportError, ModuleNotFoundError) as exc:
            raise RuntimeError(
                "RDKit is required to load arbitrary CCD components"
            ) from exc
    source_mol = _EXTERNAL_CCD_MOLS.get(code)
    if source_mol is None:
        raise ValueError(f"unknown CCD code: {code!r}")
    try:
        from rdkit import Chem
    except ImportError as exc:
        raise RuntimeError(
            "RDKit is required to load arbitrary CCD components"
        ) from exc
    atom_map = getattr(source_mol, "atom_map", None)
    inverse_atom_map = (
        {int(index): str(name) for name, index in atom_map.items()}
        if isinstance(atom_map, dict)
        else {}
    )
    ref_conf_id = int(getattr(source_mol, "ref_conf_id", 0))
    source_mask = np.asarray(
        getattr(
            source_mol,
            "ref_mask",
            np.ones((source_mol.GetNumAtoms(),), dtype=bool),
        ),
        dtype=np.float32,
    )
    heavy_source_indices = np.asarray(
        [
            atom.GetIdx()
            for atom in source_mol.GetAtoms()
            if atom.GetAtomicNum() not in (1,)
        ],
        dtype=np.int64,
    )
    if source_mask.shape != (source_mol.GetNumAtoms(),):
        raise ValueError(f"CCD code {code!r} has an invalid reference mask")

    mol = Chem.RemoveHs(Chem.Mol(source_mol), sanitize=False)
    if mol.GetNumConformers() == 0:
        raise ValueError(f"CCD code {code!r} has no reference conformer")
    try:
        conf = mol.GetConformer(ref_conf_id)
    except ValueError as exc:
        raise ValueError(
            f"CCD code {code!r} has no reference conformer {ref_conf_id}"
        ) from exc

    metadata = _external_ccd_atom_metadata(code)
    elements = np.asarray(
        [atom.GetSymbol().upper() for atom in mol.GetAtoms()], dtype=str
    )
    metadata_elements = np.asarray(metadata["elem"]).astype(str)
    if len(metadata_elements) != mol.GetNumAtoms():
        raise ValueError(
            f"CCD code {code!r} has {len(metadata_elements)} heavy atoms in "
            f"components.cif but {mol.GetNumAtoms()} in the RDKit cache"
        )
    if not np.array_equal(metadata_elements, elements):
        raise ValueError(
            f"CCD code {code!r} has inconsistent atom order between "
            "components.cif and the RDKit cache"
        )
    mapped_names = [inverse_atom_map.get(int(i)) for i in heavy_source_indices]
    names = np.asarray(metadata["names"]).astype(str)
    if all(name is not None for name in mapped_names):
        rdkit_names = np.asarray(mapped_names, dtype=str)
        if not np.array_equal(rdkit_names, names):
            raise ValueError(
                f"CCD code {code!r} has inconsistent atom identity between "
                "components.cif and the RDKit cache"
            )
        names = rdkit_names
    return {
        "names": names,
        "coord": np.asarray(conf.GetPositions(), dtype=np.float32),
        "charge": np.asarray(
            [atom.GetFormalCharge() for atom in mol.GetAtoms()], dtype=np.float32
        ),
        "mask": source_mask[heavy_source_indices],
        "elem": elements,
        "leaving_atom_flag": np.asarray(metadata["leaving_atom_flag"], dtype=bool),
        "bonds": np.asarray(
            [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()],
            dtype=np.int64,
        ).reshape((-1, 2)),
        "bond_order": np.asarray(
            [b.GetBondTypeAsDouble() for b in mol.GetBonds()], dtype=np.float32
        ),
        "bond_stereo": np.asarray(
            [int(b.GetStereo()) for b in mol.GetBonds()], dtype=np.int64
        ),
        "input_index": np.arange(mol.GetNumAtoms(), dtype=np.int64),
        "stereo": np.asarray(
            [int(atom.GetChiralTag()) for atom in mol.GetAtoms()], dtype=np.int64
        ),
        "code": code,
    }


def _verify_official_rdkit_cache(digest: str, *, path: Path) -> None:
    """Verify the digest of the immutable snapshot to be deserialized."""

    actual = digest
    if actual not in _TRUSTED_CCD_RDKIT_SHA256:
        expected = ", ".join(sorted(_TRUSTED_CCD_RDKIT_SHA256))
        raise ValueError(
            f"refusing unverified CCD RDKit pickle {path}: SHA-256 {actual}; "
            f"expected one of {expected}"
        )


def _load_verified_rdkit_cache(path: Path) -> Any:
    """Verify a bounded-RSS snapshot, then deserialize those exact bytes.

    The publisher cache is 136 MiB.  Keeping it in one Python ``bytes`` while
    unpickling retained that complete copy beside the much larger RDKit object
    tree.  A temporary file keeps the trust boundary unchanged -- unpickling
    still starts only after the complete immutable snapshot has been hashed --
    without charging the snapshot itself to process RSS.
    """

    digest = hashlib.sha256()
    with tempfile.TemporaryFile() as snapshot, path.open("rb") as source:
        while chunk := source.read(_CCD_HASH_CHUNK):
            digest.update(chunk)
            snapshot.write(chunk)
        _verify_official_rdkit_cache(digest.hexdigest(), path=path)
        snapshot.seek(0)
        return pickle.load(snapshot)


def _external_ccd_atom_metadata(code: str) -> dict[str, Any]:
    """Read exact heavy-atom identity from one block of components.cif."""

    candidates = []
    configured = os.environ.get("PROTENIX_CCD_COMPONENTS_FILE")
    if configured:
        candidates.append(Path(configured))
    configured_rdkit = os.environ.get("PROTENIX_CCD_RDKIT_MOL_FILE")
    if configured_rdkit:
        candidates.append(Path(configured_rdkit).with_name("components.cif"))
    candidates.append(_managed_ccd_asset("components.cif"))
    components_path = next((path for path in candidates if path.is_file()), None)
    if components_path is None:
        raise ValueError(
            f"CCD code {code!r} requires components.cif. Run "
            "`foldjax weights fetch --model protenix`, which puts it in the "
            "shared asset store, or set PROTENIX_CCD_COMPONENTS_FILE"
        )
    resolved = components_path.resolve()
    stat = resolved.stat()
    identity = (
        stat.st_dev,
        stat.st_ino,
        stat.st_size,
        stat.st_mtime_ns,
        stat.st_ctime_ns,
    )
    cache_key = (resolved, identity, code)
    cached = _EXTERNAL_CCD_ATOMS.get(cache_key)
    if cached is not None:
        _EXTERNAL_CCD_ATOMS.move_to_end(cache_key)
        return cached
    block = _read_ccd_block(resolved, code)
    metadata = _parse_ccd_atom_metadata(block, code)
    _EXTERNAL_CCD_ATOMS[cache_key] = metadata
    _EXTERNAL_CCD_ATOMS.move_to_end(cache_key)
    while len(_EXTERNAL_CCD_ATOMS) > _EXTERNAL_CCD_ATOM_CACHE_LIMIT:
        _EXTERNAL_CCD_ATOMS.popitem(last=False)
    return metadata


def _read_ccd_block(path: Path, code: str) -> str:
    """Extract one CCD data block without parsing the full 490 MB dictionary."""

    with path.open("rb") as handle:
        if path.stat().st_size == 0:
            raise ValueError(f"components.cif is empty: {path}")
        with mmap.mmap(handle.fileno(), length=0, access=mmap.ACCESS_READ) as mapped:
            bounds = _sorted_ccd_block_bounds(mapped, code.encode())
            if bounds is None:
                # Environment overrides are allowed to point at another
                # components.cif.  The official file is sorted by data-block
                # code; preserve the historical linear lookup for a custom
                # file that is not.
                bounds = _linear_ccd_block_bounds(mapped, code.encode())
            if bounds is None:
                raise ValueError(f"unknown CCD code in components.cif: {code!r}")
            start, end = bounds
            return mapped[start:end].decode("utf-8")


def _sorted_ccd_block_bounds(
    mapped: mmap.mmap, code: bytes
) -> tuple[int, int] | None:
    """Binary-search a code-sorted CCD without first building a 49k-row index."""

    low = 0
    high = len(mapped)
    # Byte offsets, rather than record indices, make the search independent of
    # each component's highly variable atom/bond-table size. ``rfind`` starts
    # at the midpoint and normally meets the containing block a few KiB away.
    while low < high:
        midpoint = (low + high) // 2
        marker = mapped.rfind(b"\ndata_", 0, midpoint + 1)
        start = 0 if marker < 0 else marker + 1
        line_end = mapped.find(b"\n", start + len(b"data_"))
        if line_end < 0:
            return None
        current = mapped[start + len(b"data_") : line_end].rstrip(b"\r")
        if current == code:
            end = mapped.find(b"\ndata_", line_end)
            return start, len(mapped) if end < 0 else end
        if current < code:
            next_marker = mapped.find(b"\ndata_", line_end)
            if next_marker < 0:
                return None
            next_start = next_marker + 1
            if next_start <= low:
                return None
            low = next_start
        else:
            next_high = midpoint if start >= high else start
            if next_high >= high:
                return None
            high = next_high
    return None


def _linear_ccd_block_bounds(
    mapped: mmap.mmap, code: bytes
) -> tuple[int, int] | None:
    """Historical exact lookup for an unsorted custom components.cif."""

    marker = b"data_" + code + b"\n"
    start = mapped.find(marker)
    while start > 0 and mapped[start - 1 : start] != b"\n":
        start = mapped.find(marker, start + 1)
    if start < 0:
        return None
    end = mapped.find(b"\ndata_", start + len(marker))
    return start, len(mapped) if end < 0 else end


def _parse_ccd_atom_metadata(block: str, code: str) -> dict[str, Any]:
    """Parse exact atom identity and canonical polymer annotations."""

    try:
        cif_block = gemmi.cif.read_string(block).sole_block()
        rows = cif_block.find(
            [
                "_chem_comp_atom.atom_id",
                "_chem_comp_atom.type_symbol",
                "_chem_comp_atom.pdbx_leaving_atom_flag",
            ]
        )
    except RuntimeError as exc:
        raise ValueError(f"invalid components.cif block for CCD {code!r}") from exc
    if len(rows) == 0:
        raise ValueError(f"CCD code {code!r} has no atom records")
    names = []
    elements = []
    leaving = []
    for atom_id, element, leaving_flag in rows:
        atom_id = gemmi.cif.as_string(str(atom_id))
        element = gemmi.cif.as_string(str(element)).upper()
        leaving_flag = gemmi.cif.as_string(str(leaving_flag)).upper()
        if element in ("H", "D"):
            continue
        names.append(atom_id)
        elements.append(element)
        leaving.append(leaving_flag == "Y")
    if not names:
        raise ValueError(f"CCD code {code!r} has no heavy atoms")
    if len(set(names)) != len(names):
        raise ValueError(f"CCD code {code!r} has duplicate atom identifiers")
    component_type = _cif_string(cif_block, "_chem_comp.type").upper()
    if "PEPTIDE LINKING" in component_type:
        canonical_polymer_type = "protein"
    elif "DNA LINKING" in component_type:
        canonical_polymer_type = "dna"
    elif "RNA LINKING" in component_type:
        canonical_polymer_type = "rna"
    else:
        canonical_polymer_type = None
    one_letter_code = _cif_string(cif_block, "_chem_comp.one_letter_code").upper()
    if one_letter_code in {"", ".", "?"}:
        one_letter_code = None
    return {
        "names": np.asarray(names),
        "elem": np.asarray(elements),
        "leaving_atom_flag": np.asarray(leaving, dtype=bool),
        "canonical_polymer_type": canonical_polymer_type,
        "one_letter_code": one_letter_code,
    }


def _cif_string(block: gemmi.cif.Block, tag: str) -> str:
    value = block.find_value(tag)
    if value is None:
        return ""
    try:
        return gemmi.cif.as_string(str(value))
    except RuntimeError:
        return ""


def _rdkit_ligand_entry(
    ligand: str, *, base_dir: str | Path | None
) -> dict[str, np.ndarray]:
    """Parse/sanitize a SMILES or a supported 3D file into static arrays."""

    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError as exc:  # optional until chemical inputs are requested
        raise RuntimeError(
            "RDKit is required for SMILES and FILE_ ligand inputs"
        ) from exc

    if ligand.startswith("FILE_"):
        path = Path(ligand[5:])
        if not path.is_absolute() and base_dir is not None:
            path = Path(base_dir) / path
        if not path.exists():
            raise FileNotFoundError(f"ligand file does not exist: {path}")
        suffix = path.suffix.lower()
        if suffix == ".mol":
            mol = Chem.MolFromMolFile(str(path), sanitize=True, removeHs=False)
        elif suffix == ".sdf":
            supplier = Chem.SDMolSupplier(str(path), sanitize=True, removeHs=False)
            mol = next(iter(supplier), None)
        elif suffix == ".pdb":
            mol = Chem.MolFromPDBFile(str(path), sanitize=True, removeHs=False)
        elif suffix == ".mol2":
            mol = Chem.MolFromMol2File(str(path), sanitize=True, removeHs=False)
        else:
            raise ValueError(
                f"unsupported ligand file type {suffix!r}; expected PDB/SDF/MOL/MOL2"
            )
        if mol is None:
            raise ValueError(f"invalid or unsanitizable ligand file: {path}")
        if mol.GetNumConformers() == 0 or not mol.GetConformer().Is3D():
            raise ValueError(f"3D conformer not found in ligand file: {path}")
    else:
        mol = Chem.MolFromSmiles(ligand, sanitize=True)
        if mol is None:
            raise ValueError(f"invalid SMILES ligand: {ligand!r}")
        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 0
        status = AllChem.EmbedMolecule(mol, params)
        if status != 0:
            params.useRandomCoords = True
            status = AllChem.EmbedMolecule(mol, params)
        if status != 0:
            raise ValueError(f"3D conformer generation failed for SMILES: {ligand!r}")

    # Preserve the user's heavy-atom index/map through hydrogen removal.
    for atom in mol.GetAtoms():
        atom.SetIntProp("_input_index", atom.GetIdx())
    mol = Chem.RemoveHs(mol, sanitize=True)
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    conf = mol.GetConformer()
    coords = np.asarray(conf.GetPositions(), dtype=np.float32)
    element_counts: dict[str, int] = {}
    names = []
    elements = []
    charges = []
    input_indices = []
    stereo = []
    for atom in mol.GetAtoms():
        elem = atom.GetSymbol().upper()
        element_counts[elem] = element_counts.get(elem, 0) + 1
        names.append(f"{elem}{element_counts[elem]}")
        elements.append(elem)
        charges.append(atom.GetFormalCharge())
        input_indices.append(
            atom.GetAtomMapNum()
            if atom.GetAtomMapNum()
            else atom.GetIntProp("_input_index")
        )
        stereo.append(int(atom.GetChiralTag()))
    bonds = np.asarray(
        [(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) for bond in mol.GetBonds()],
        dtype=np.int64,
    ).reshape((-1, 2))
    return {
        "names": np.asarray(names),
        "coord": coords,
        "charge": np.asarray(charges, dtype=np.float32),
        "mask": np.ones((len(names),), dtype=np.float32),
        "elem": np.asarray(elements),
        "bonds": bonds,
        "bond_order": np.asarray(
            [bond.GetBondTypeAsDouble() for bond in mol.GetBonds()],
            dtype=np.float32,
        ),
        "bond_stereo": np.asarray(
            [int(bond.GetStereo()) for bond in mol.GetBonds()], dtype=np.int64
        ),
        "input_index": np.asarray(input_indices, dtype=np.int64),
        "stereo": np.asarray(stereo, dtype=np.int64),
        "code": "UNL",
    }


def _append_atom_metadata(
    state: dict[str, Any],
    chain: dict[str, Any],
    position: int,
    entry: dict[str, Any],
    atom_index: int,
) -> None:
    state["atom_entity_id"].append(chain["entity_number"])
    state["atom_copy_id"].append(chain["copy_id"])
    state["atom_residue_index"].append(position)
    state["atom_input_index"].append(
        int(entry.get("input_index", np.arange(len(entry["names"])))[atom_index])
    )
    state["ligand_stereo_atom"].append(
        int(entry.get("stereo", np.zeros(len(entry["names"])))[atom_index])
    )
    state["output_atom_res_name"].append(str(entry.get("code", "UNK")))
    state["output_atom_chain_id"].append(chain["chain_id"])
    state["output_atom_res_id"].append(position)
    state["output_atom_polymer_type"].append(chain["polymer_type"])


def _append_entry_bonds(state: dict[str, Any], entry: dict[str, Any]) -> None:
    atom_offset = len(state["atom_to_token_idx"])
    bonds = np.asarray(entry.get("bonds", []), dtype=np.int64).reshape((-1, 2))
    orders = np.asarray(
        entry.get("bond_order", np.ones((len(bonds),))), dtype=np.float32
    )
    stereos = np.asarray(
        entry.get("bond_stereo", np.zeros((len(bonds),))), dtype=np.int64
    )
    for bond, order, stereo in zip(bonds, orders, stereos):
        state["chemical_bond_atom_indices"].append(
            (atom_offset + int(bond[0]), atom_offset + int(bond[1]))
        )
        state["chemical_bond_order"].append(float(order))
        state["chemical_bond_stereo"].append(int(stereo))


def _canonical_bond_endpoint(
    bond: dict[str, Any], side: int
) -> tuple[int, int | None, int, Any]:
    old = "left" if side == 1 else "right"

    def aliased_value(
        old_key: str,
        new_key: str,
        *,
        optional: bool = False,
        atom: bool = False,
    ) -> Any:
        present = [(key, bond[key]) for key in (old_key, new_key) if key in bond]
        if not present:
            if optional:
                return None
            raise ValueError(f"covalent_bonds field {old_key!r} is required")

        def normalized(value: Any) -> Any:
            if atom:
                if isinstance(value, str) and value.isdigit():
                    return int(value)
                return value
            if value is None and optional:
                return None
            return int(value)

        try:
            normalized_values = [normalized(value) for _key, value in present]
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "covalent_bonds entity, copy, and position must be integers"
            ) from exc
        if any(value != normalized_values[0] for value in normalized_values[1:]):
            raise ValueError(
                f"conflicting covalent bond aliases {old_key!r} and {new_key!r}"
            )
        return normalized_values[0]

    try:
        entity = aliased_value(f"{old}_entity", f"entity{side}")
        position = aliased_value(f"{old}_position", f"position{side}")
        atom_identifier = aliased_value(f"{old}_atom", f"atom{side}", atom=True)
        copy = aliased_value(f"{old}_copy", f"copy{side}", optional=True)
    except KeyError as exc:
        raise ValueError("covalent bond endpoint fields are required") from exc
    if atom_identifier is None:
        raise ValueError("covalent bond atom identifiers are required")
    return entity, copy, position, atom_identifier


def _chain_residue_entry(
    chain: dict[str, Any], position: int
) -> tuple[dict[str, Any], tuple[str, int]]:
    if chain["kind"] in ("protein", "nucleic"):
        if position < 1 or position > len(chain["entries"]):
            raise ValueError(f"covalent residue position is out of range: {position}")
        return chain["entries"][position - 1], ("polymer", position - 1)
    for index, (residue_id, entry) in enumerate(chain["residues"]):
        if residue_id == position:
            return entry, ("ligand", index)
    raise ValueError(f"covalent ligand position is out of range: {position}")


def _replace_chain_residue_entry(
    chain: dict[str, Any], location: tuple[str, int], entry: dict[str, Any]
) -> None:
    kind, index = location
    if kind == "polymer":
        entries = list(chain["entries"])
        entries[index] = entry
        chain["entries"] = entries
    else:
        residues = list(chain["residues"])
        residue_id, _ = residues[index]
        residues[index] = (residue_id, entry)
        chain["residues"] = residues


def _leaving_groups_for_atom(entry: dict[str, Any], atom_name: str) -> list[list[str]]:
    explicit = entry.get("central_to_leaving_groups")
    if explicit is not None:
        return [list(group) for group in explicit.get(atom_name, [])]
    flags = entry.get("leaving_atom_flag")
    if flags is None:
        code = entry.get("code", "UNKNOWN")
        raise ValueError(f"leaving-atom metadata is unavailable for CCD {code}")
    names = np.asarray(entry["names"]).astype(str)
    central = np.flatnonzero(names == atom_name)
    if central.size != 1:
        raise ValueError(f"covalent centre atom {atom_name!r} is not unique")
    flags = np.asarray(flags, dtype=bool)
    bonds = np.asarray(entry.get("bonds", []), dtype=np.int64).reshape((-1, 2))
    adjacency: dict[int, list[int]] = {index: [] for index in range(len(names))}
    for left, right in bonds:
        adjacency[int(left)].append(int(right))
        adjacency[int(right)].append(int(left))
    seeds = sorted(
        neighbor for neighbor in adjacency[int(central[0])] if flags[neighbor]
    )
    if not seeds:
        return []
    # Each connected component in the leaving-only subgraph is an independent
    # leaving group. A single inter-residue bond removes one group, not the
    # union of every leaving atom adjacent to the same centre.
    remaining = set(seeds)
    groups: list[list[str]] = []
    while remaining:
        seed = min(remaining)
        component = {seed}
        stack = [seed]
        while stack:
            current = stack.pop()
            for neighbor in adjacency[current]:
                if flags[neighbor] and neighbor not in component:
                    component.add(neighbor)
                    stack.append(neighbor)
        remaining.difference_update(component)
        groups.append(names[sorted(component)].tolist())
    return groups


def _entry_with_official_leaving_metadata(
    entry: dict[str, Any],
) -> dict[str, Any]:
    """Attach official CCD leaving flags to a compact vendored component."""

    if (
        entry.get("central_to_leaving_groups") is not None
        or entry.get("leaving_atom_flag") is not None
    ):
        return entry
    code = str(entry.get("code", ""))
    if not code:
        raise ValueError("CCD component has no code for leaving-atom lookup")
    metadata = _external_ccd_atom_metadata(code)
    names = np.asarray(entry["names"]).astype(str)
    metadata_names = np.asarray(metadata["names"]).astype(str)
    if not np.array_equal(names, metadata_names):
        raise ValueError(
            f"CCD code {code!r} has inconsistent atom identity between its "
            "vendored component and components.cif"
        )
    annotated = dict(entry)
    annotated["leaving_atom_flag"] = np.asarray(
        metadata["leaving_atom_flag"], dtype=bool
    )
    return annotated


def _entry_without_atoms(
    entry: dict[str, Any], atom_names: list[str]
) -> dict[str, Any]:
    names = np.asarray(entry["names"]).astype(str)
    keep = ~np.isin(names, atom_names)
    if keep.all():
        return entry
    old_to_new = np.full((len(names),), -1, dtype=np.int64)
    old_to_new[keep] = np.arange(int(keep.sum()))
    bonds = np.asarray(entry.get("bonds", []), dtype=np.int64).reshape((-1, 2))
    keep_bonds = (
        keep[bonds[:, 0]] & keep[bonds[:, 1]] if len(bonds) else np.zeros(0, dtype=bool)
    )
    result = dict(entry)
    for key in (
        "names",
        "coord",
        "charge",
        "mask",
        "elem",
        "input_index",
        "stereo",
        "leaving_atom_flag",
    ):
        if key in result:
            result[key] = np.asarray(result[key])[keep]
    result["bonds"] = (
        old_to_new[bonds[keep_bonds]]
        if len(bonds)
        else np.zeros((0, 2), dtype=np.int64)
    )
    for key in ("bond_order", "bond_stereo"):
        if key in result:
            result[key] = np.asarray(result[key])[keep_bonds]
    if "central_to_leaving_groups" in result:
        removed = set(atom_names)
        updated_groups = {}
        for centre, groups in result["central_to_leaving_groups"].items():
            retained = [
                [name for name in group if name not in removed] for group in groups
            ]
            retained = [group for group in retained if group]
            if retained:
                updated_groups[centre] = retained
        result["central_to_leaving_groups"] = updated_groups
    return result


def _entry_atom_name(entry: dict[str, Any], atom: Any) -> str:
    if isinstance(atom, str) and atom.isdigit():
        atom = int(atom)
    if isinstance(atom, int):
        matches = np.flatnonzero(np.asarray(entry.get("input_index", [])) == atom)
        if matches.size != 1:
            raise ValueError(f"covalent atom index not found: {atom}")
        return str(np.asarray(entry["names"])[matches[0]])
    return str(atom)


def _endpoint_leaving_groups(
    chain: dict[str, Any],
    *,
    position: int,
    entry: dict[str, Any],
    atom_name: str,
) -> tuple[dict[str, Any], list[list[str]]]:
    is_polymer = chain["kind"] in ("protein", "nucleic")
    is_modified_polymer = is_polymer and bool(chain["modified"][position - 1])
    code = str(entry.get("code", ""))
    is_ccd_ligand = chain["kind"] == "ligand" and code != "UNL"
    if is_polymer and not is_modified_polymer:
        if chain["kind"] == "protein" and atom_name == "C":
            return entry, [["OXT"]]
        if chain["kind"] == "nucleic" and atom_name == "P":
            return entry, [["OP3"]]
        return entry, []
    if is_modified_polymer or is_ccd_ligand:
        annotated = _entry_with_official_leaving_metadata(entry)
        return annotated, _leaving_groups_for_atom(annotated, atom_name)
    return entry, []


def _sample_leaving_atoms(
    groups: list[list[str]], bond_count: int, *, rng: random.Random
) -> list[str]:
    if bond_count <= 0 or not groups:
        return []
    selected = groups if bond_count > len(groups) else rng.sample(groups, bond_count)
    return [atom_name for group in selected for atom_name in group]


def _remove_polymer_link_leaving_groups(
    chains: list[dict[str, Any]], *, rng: random.Random
) -> None:
    """Apply upstream's seeded implicit polymer-link leaving-group pass."""

    processed_entities: dict[int, list[dict[str, Any]]] = {}
    for chain in chains:
        if chain["kind"] not in ("protein", "nucleic"):
            continue
        entity_number = int(chain["entity_number"])
        if entity_number in processed_entities:
            chain["entries"] = processed_entities[entity_number]
            continue

        entries = list(chain["entries"])
        connector = ("C", "N") if chain["kind"] == "protein" else ("O3'", "P")
        for left_index in range(len(entries) - 1):
            right_index = left_index + 1
            left_names = np.asarray(entries[left_index]["names"]).astype(str)
            right_names = np.asarray(entries[right_index]["names"]).astype(str)
            if connector[0] not in left_names or connector[1] not in right_names:
                continue
            for residue_index, atom_name in (
                (left_index, connector[0]),
                (right_index, connector[1]),
            ):
                entry, groups = _endpoint_leaving_groups(
                    chain,
                    position=residue_index + 1,
                    entry=entries[residue_index],
                    atom_name=atom_name,
                )
                entries[residue_index] = entry
                leaving_atoms = _sample_leaving_atoms(groups, 1, rng=rng)
                if leaving_atoms:
                    entries[residue_index] = _entry_without_atoms(entry, leaving_atoms)
        chain["entries"] = entries
        processed_entities[entity_number] = entries


def _matching_chain_bond_centres(
    chains: list[dict[str, Any]],
    endpoint: tuple[int, int | None, int, Any],
) -> list[tuple[int, tuple[str, int], str]]:
    entity, copy, position, atom = endpoint
    records: list[tuple[int, tuple[str, int], str]] = []
    for chain_index, chain in enumerate(chains):
        if chain["entity_number"] != entity or (
            copy is not None and chain["copy_id"] != copy
        ):
            continue
        entry, location = _chain_residue_entry(chain, position)
        records.append((chain_index, location, _entry_atom_name(entry, atom)))
    if not records:
        raise ValueError(f"covalent entity/copy was not found: {entity}/{copy}")
    return records


def _remove_covalent_leaving_groups(
    chains: list[dict[str, Any]],
    bonds: Any,
    *,
    rng: random.Random,
) -> None:
    """Remove seeded, bond-count-selected CCD leaving groups."""

    if not bonds:
        return
    if not isinstance(bonds, list):
        raise ValueError("covalent_bonds must be a list")
    pending: dict[tuple[int, str, int, str], int] = {}
    for bond in bonds:
        if not isinstance(bond, dict):
            raise ValueError("each covalent bond must be an object")
        left = _matching_chain_bond_centres(chains, _canonical_bond_endpoint(bond, 1))
        right = _matching_chain_bond_centres(chains, _canonical_bond_endpoint(bond, 2))
        if len(left) != len(right):
            raise ValueError("covalent bond endpoints have unequal copy counts")
        for pair in zip(left, right, strict=True):
            for chain_index, location, atom_name in pair:
                key = (chain_index, location[0], location[1], atom_name)
                pending[key] = pending.get(key, 0) + 1

    for (
        chain_index,
        location_kind,
        location_index,
        atom_name,
    ), count in pending.items():
        chain = chains[chain_index]
        location = (location_kind, location_index)
        position = (
            location_index + 1
            if location_kind == "polymer"
            else int(chain["residues"][location_index][0])
        )
        entry, _ = _chain_residue_entry(chain, position)
        if len(entry["names"]) <= 1:
            continue
        entry, groups = _endpoint_leaving_groups(
            chain,
            position=position,
            entry=entry,
            atom_name=atom_name,
        )
        _replace_chain_residue_entry(chain, location, entry)
        leaving_atoms = _sample_leaving_atoms(groups, count, rng=rng)
        if leaving_atoms:
            _replace_chain_residue_entry(
                chain,
                location,
                _entry_without_atoms(entry, leaving_atoms),
            )


def _matching_atoms(
    state: dict[str, Any], endpoint: tuple[int, int | None, int, Any]
) -> np.ndarray:
    entity, copy, position, atom = endpoint
    scope = (np.asarray(state["atom_entity_id"]) == entity) & (
        np.asarray(state["atom_residue_index"]) == position
    )
    if copy is not None:
        scope &= np.asarray(state["atom_copy_id"]) == copy
    if isinstance(atom, str) and atom.isdigit():
        atom = int(atom)
    if isinstance(atom, int):
        mask = scope & (np.asarray(state["atom_input_index"]) == atom)
    else:
        mask = scope & (np.asarray(state["ref_atom_names"]) == str(atom))
    indices = np.flatnonzero(mask)
    if indices.size == 0 and atom == "SE":
        atom_to_token = np.asarray(state["atom_to_token_idx"], dtype=np.int64)
        mse_atom = state["token_reference_is_mse"][atom_to_token].astype(bool)
        indices = np.flatnonzero(
            scope & mse_atom & (np.asarray(state["ref_atom_names"]) == "SD")
        )
    if indices.size == 0:
        raise ValueError(
            "no atom found for "
            f"entity={entity}, copy={copy}, position={position}, atom={atom!r}"
        )
    return indices


def _apply_covalent_bonds(
    bonds: Any, state: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    if bonds is None:
        bonds = []
    if not isinstance(bonds, list):
        raise ValueError("covalent_bonds must be a list")
    atom_pairs = []
    token_pairs = []
    atom_to_token = np.asarray(state["atom_to_token_idx"], dtype=np.int64)
    for bond in bonds:
        if not isinstance(bond, dict):
            raise ValueError("each covalent bond must be an object")
        left = _matching_atoms(state, _canonical_bond_endpoint(bond, 1))
        right = _matching_atoms(state, _canonical_bond_endpoint(bond, 2))
        if len(left) != len(right):
            raise ValueError("covalent bond endpoints have unequal copy counts")
        for atom1, atom2 in zip(left, right):
            token1, token2 = int(atom_to_token[atom1]), int(atom_to_token[atom2])
            atom_pairs.append((int(atom1), int(atom2)))
            token_pairs.append((token1, token2))
            if _keep_covalent_token_bond(
                state,
                atom1=int(atom1),
                atom2=int(atom2),
                token1=token1,
                token2=token2,
            ):
                state["token_bond_edges"].append((token1, token2))
    return (
        np.asarray(atom_pairs, dtype=np.int64).reshape((-1, 2)),
        np.asarray(token_pairs, dtype=np.int64).reshape((-1, 2)),
    )


def _keep_covalent_token_bond(
    state: dict[str, Any],
    *,
    atom1: int,
    atom2: int,
    token1: int,
    token2: int,
) -> bool:
    """Match upstream token-bond filtering for explicit covalent bonds."""

    polymer_type = np.asarray(state["token_polymer_type"], dtype=np.int64)
    standard = np.asarray(state["token_is_standard_polymer"], dtype=np.int64).astype(
        bool
    )
    polymer1 = polymer_type[token1] > 0
    polymer2 = polymer_type[token2] > 0
    if not (polymer1 and polymer2):
        return True

    standard1 = bool(standard[token1])
    standard2 = bool(standard[token2])
    unstandard1 = not standard1
    unstandard2 = not standard2
    ref_space_uid = np.asarray(state["ref_space_uid"], dtype=np.int64)
    excluded = (
        (standard1 and standard2)
        or (standard1 and unstandard2)
        or (standard2 and unstandard1)
        or (
            unstandard1 and unstandard2 and ref_space_uid[atom1] != ref_space_uid[atom2]
        )
    )
    if not excluded:
        return True

    residue_index = np.asarray(state["atom_residue_index"], dtype=np.int64)
    chain_id = np.asarray(state["output_atom_chain_id"], dtype=str)
    return bool(
        abs(int(residue_index[atom1]) - int(residue_index[atom2])) > 1
        or chain_id[atom1] != chain_id[atom2]
    )


def _tokens_for_identifier(
    state: dict[str, Any], identifier: dict[str, Any], suffix: str = ""
) -> np.ndarray:
    try:
        entity = int(identifier[f"entity{suffix}"])
        copy = int(identifier[f"copy{suffix}"])
        position = int(identifier[f"position{suffix}"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "constraint entity/copy/position identifiers are required"
        ) from exc
    atom = identifier.get(f"atom{suffix}")
    if atom is not None:
        atoms = _matching_atoms(state, (entity, copy, position, atom))
        return np.unique(np.asarray(state["atom_to_token_idx"])[atoms])
    mask = (
        (state["token_entity_number"] == entity)
        & (state["token_copy_id"] == copy)
        & (state["residue_index"] == position)
    )
    tokens = np.flatnonzero(mask)
    if tokens.size == 0:
        raise ValueError(
            f"no token found for entity={entity}, copy={copy}, position={position}"
        )
    return tokens


def _build_constraint_features(
    constraint: Any, state: dict[str, Any], *, n_token: int
) -> dict[str, np.ndarray]:
    if constraint is None:
        constraint = {}
    if not isinstance(constraint, dict):
        raise ValueError("constraint must be an object")
    unknown = set(constraint) - {"contact", "pocket"}
    if unknown:
        raise ValueError(f"unsupported constraint fields: {sorted(unknown)}")
    contact = np.zeros((n_token, n_token, 2), dtype=np.float32)
    contact_atom = np.zeros_like(contact)
    contact_atom_is_set = np.zeros((n_token, n_token), dtype=bool)
    contact_atom_index_pairs: list[tuple[int, int]] = []
    contact_atom_token_pairs: list[tuple[int, int]] = []
    pocket_feature = np.zeros((n_token, n_token, 1), dtype=np.float32)
    pairs = constraint.get("contact", [])
    if not isinstance(pairs, list):
        raise ValueError("constraint.contact must be a list")
    for pair in pairs:
        if not isinstance(pair, dict):
            raise ValueError("each contact constraint must be an object")
        is_atom_contact = (
            pair.get("atom1") is not None and pair.get("atom2") is not None
        )
        if is_atom_contact:
            atoms1 = _matching_atoms(
                state,
                (
                    int(pair["entity1"]),
                    int(pair["copy1"]),
                    int(pair["position1"]),
                    pair["atom1"],
                ),
            )
            atoms2 = _matching_atoms(
                state,
                (
                    int(pair["entity2"]),
                    int(pair["copy2"]),
                    int(pair["position2"]),
                    pair["atom2"],
                ),
            )
            if atoms1.size != 1 or atoms2.size != 1:
                raise ValueError("atom-contact identifiers must resolve to one atom")
            atom1, atom2 = int(atoms1[0]), int(atoms2[0])
            atom_to_token = np.asarray(state["atom_to_token_idx"], dtype=np.int64)
            tokens1 = np.asarray([atom_to_token[atom1]])
            tokens2 = np.asarray([atom_to_token[atom2]])
            target = contact_atom
            minimum = float(pair.get("min_distance", 0))
        else:
            tokens1 = _tokens_for_identifier(state, pair, "1")
            tokens2 = _tokens_for_identifier(state, pair, "2")
            target = contact
            minimum = 0.0
        maximum = float(pair["max_distance"])
        if minimum < 0 or maximum < minimum:
            raise ValueError("constraint distances must satisfy 0 <= min <= max")
        token1, token2 = int(tokens1[0]), int(tokens2[0])
        if state["asym_id"][token1] == state["asym_id"][token2]:
            raise ValueError("a contact pair cannot be on the same chain")
        if is_atom_contact:
            # ConstraintEmbedder consumes [N_token, N_token, 2], not atom-pair
            # tensors. Preserve the exact atom pair below, and combine multiple
            # atom constraints landing on one residue-token pair by interval
            # intersection. This is the strongest lossless representation the
            # current model input contract permits; contradictory intervals fail.
            if contact_atom_is_set[token1, token2]:
                minimum = max(minimum, float(target[token1, token2, 0]))
                maximum = min(maximum, float(target[token1, token2, 1]))
                if maximum < minimum:
                    raise ValueError(
                        "incompatible atom-contact intervals aggregate to an "
                        "empty token-pair interval"
                    )
            contact_atom_is_set[token1, token2] = True
            contact_atom_is_set[token2, token1] = True
            contact_atom_index_pairs.append((atom1, atom2))
            contact_atom_token_pairs.append((token1, token2))
        target[token1, token2] = (minimum, maximum)
        target[token2, token1] = (minimum, maximum)

    pocket = constraint.get("pocket")
    if pocket:
        if not isinstance(pocket, dict):
            raise ValueError("constraint.pocket must be an object")
        binder = pocket.get("binder_chain")
        if not isinstance(binder, dict):
            raise ValueError("pocket binder_chain is required")
        try:
            binder_entity = int(binder["entity"])
            binder_copy = int(binder["copy"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("pocket binder entity/copy are required") from exc
        binder_tokens = np.flatnonzero(
            (state["token_entity_number"] == binder_entity)
            & (state["token_copy_id"] == binder_copy)
        )
        if binder_tokens.size == 0:
            raise ValueError("pocket binder chain was not found")
        maximum = float(pocket["max_distance"])
        for residue in pocket.get("contact_residues", []):
            pocket_tokens = _tokens_for_identifier(state, residue)
            pocket_token = int(pocket_tokens[0])
            if state["asym_id"][binder_tokens[0]] == state["asym_id"][pocket_token]:
                raise ValueError("pocket residues cannot be on the binder chain")
            pocket_feature[binder_tokens, pocket_token, 0] = maximum
    return {
        "contact": contact,
        "pocket": pocket_feature,
        "contact_atom": contact_atom,
        "contact_atom_index_pairs": np.asarray(
            contact_atom_index_pairs, dtype=np.int64
        ).reshape((-1, 2)),
        "contact_atom_token_pairs": np.asarray(
            contact_atom_token_pairs, dtype=np.int64
        ).reshape((-1, 2)),
        "substructure": np.zeros((n_token, n_token, 4), dtype=np.float32),
    }


def _encode_fixed_strings(values: list[str], *, width: int) -> np.ndarray:
    encoded = np.zeros((len(values), width), dtype=np.int64)
    for row, value in enumerate(values):
        for col, char in enumerate(str(value)[:width]):
            encoded[row, col] = ord(char)
    return encoded


def _resolve_path(path: str | Path, *, base_dir: str | Path | None) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute() and base_dir is not None:
        resolved = Path(base_dir) / resolved
    if not resolved.exists():
        raise ValueError(f"MSA path does not exist: {resolved}")
    return resolved


def _parse_a3m_records(a3m: str) -> tuple[list[str], list[str]]:
    """Parse FASTA/A3M into (sequences, descriptions); torch ``parse_fasta``."""

    sequences: list[str] = []
    descriptions: list[str] = []
    index = -1
    for raw_line in a3m.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(">"):
            index += 1
            descriptions.append(line[1:])
            sequences.append("")
        elif index >= 0:
            sequences[index] += line
    return sequences, descriptions


def _aligned_protein_row(sequence: str) -> tuple[list[int], list[int]]:
    row: list[int] = []
    deletion: list[int] = []
    pending_insertions = 0
    for char in sequence:
        if char.islower() or char == ".":
            pending_insertions += 1
            continue
        upper = char.upper()
        if upper not in MSA_PROTEIN_INDEX:
            raise ValueError(f"unsupported MSA residue: {char}")
        row.append(MSA_PROTEIN_INDEX[upper])
        deletion.append(pending_insertions)
        pending_insertions = 0
    return row, deletion


def _aligned_rna_row(sequence: str) -> tuple[list[int], list[int]]:
    row: list[int] = []
    deletion: list[int] = []
    pending_insertions = 0
    for char in sequence:
        if char.islower() or char == ".":
            pending_insertions += 1
            continue
        if char not in MSA_RNA_INDEX:
            raise ValueError(f"unsupported RNA MSA residue: {char}")
        row.append(MSA_RNA_INDEX[char])
        deletion.append(pending_insertions)
        pending_insertions = 0
    return row, deletion


def _normalize_sequence(sequence: Any) -> str:
    if not isinstance(sequence, str) or not sequence:
        raise ValueError("proteinChain sequence must be a non-empty string")
    sequence = sequence.upper()
    for aa in sequence:
        if aa not in RESTYPE_INDEX:
            raise ValueError(f"unsupported residue: {aa}")
    return sequence


def _distogram_rep_atom_name(aa: str) -> str:
    if aa == "G":
        return "CA"
    return "CB"


def _encode_elements(elements: list[str]) -> np.ndarray:
    encoded = np.zeros((len(elements), 128), dtype=np.float32)
    for i, element in enumerate(elements):
        encoded[i, ELEMENT_INDEX[element]] = 1.0
    return encoded


def _encode_atom_name_chars(atom_names: list[str]) -> np.ndarray:
    encoded = np.zeros((len(atom_names), 4, 64), dtype=np.float32)
    for i, name in enumerate(atom_names):
        for j, char in enumerate(name.ljust(4)[:4]):
            encoded[i, j, min(max(ord(char) - 32, 0), 63)] = 1.0
    return encoded


def _local_atom_geometry(
    ref_pos: np.ndarray,
    ref_space_uid: np.ndarray,
    *,
    n_queries: int,
    n_keys: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    n_atom = ref_pos.shape[0]
    n_trunks = int(math.ceil(n_atom / n_queries))
    q_pad = n_trunks * n_queries - n_atom
    pad_left = (n_keys - n_queries) // 2
    pad_right = int((n_trunks - 0.5) * n_queries + n_keys / 2 - n_atom + 0.5)

    q_padded = np.pad(ref_pos, ((0, q_pad), (0, 0)))
    k_padded = np.pad(ref_pos, ((pad_left, pad_right), (0, 0)))
    q_uid_padded = np.pad(ref_space_uid, (0, q_pad))
    k_uid_padded = np.pad(ref_space_uid, (pad_left, pad_right))
    q = q_padded.reshape(n_trunks, n_queries, 3)
    k = np.stack(
        [k_padded[i * n_queries : i * n_queries + n_keys] for i in range(n_trunks)],
        axis=0,
    )
    q_uid = q_uid_padded.reshape(n_trunks, n_queries)
    k_uid = np.stack(
        [k_uid_padded[i * n_queries : i * n_queries + n_keys] for i in range(n_trunks)],
        axis=0,
    )
    q_abs = np.arange(n_trunks * n_queries).reshape(n_trunks, n_queries)
    k_abs = (
        np.arange(n_keys)[None, :] + np.arange(n_trunks)[:, None] * n_queries - pad_left
    )
    mask = (
        (q_abs[..., None] < n_atom)
        & (k_abs[:, None, :] >= 0)
        & (k_abs[:, None, :] < n_atom)
    )
    d_lm = q[:, :, None, :] - k[:, None, :, :]
    d_lm = d_lm.astype(np.float32)
    v_lm = (q_uid[:, :, None] == k_uid[:, None, :])[..., None].astype(np.float32)
    return d_lm, v_lm, {"mask_trunked": mask}


def _relative_position_features(
    *,
    asym_id: np.ndarray,
    residue_index: np.ndarray,
    entity_id: np.ndarray,
    sym_id: np.ndarray,
    token_index: np.ndarray,
    r_max: int = 32,
    s_max: int = 2,
) -> np.ndarray:
    same_chain = asym_id[:, None] == asym_id[None, :]
    same_residue = residue_index[:, None] == residue_index[None, :]
    same_entity = entity_id[:, None] == entity_id[None, :]

    residue_delta = np.clip(
        residue_index[:, None] - residue_index[None, :] + r_max,
        0,
        2 * r_max,
    )
    residue_bins = np.where(same_chain, residue_delta, 2 * r_max + 1)
    token_delta = np.clip(
        token_index[:, None] - token_index[None, :] + r_max,
        0,
        2 * r_max,
    )
    token_bins = np.where(same_chain & same_residue, token_delta, 2 * r_max + 1)
    chain_delta = np.clip(sym_id[:, None] - sym_id[None, :] + s_max, 0, 2 * s_max)
    chain_bins = np.where(same_entity, chain_delta, 2 * s_max + 1)

    # This [N_token, N_token, 139] feature is three 0/1 one-hot blocks plus a
    # 0/1 same-entity indicator. Keep those exact values compact while they
    # cross the host/device boundary. Each projection promotes them through its
    # weights: bfloat16 in the default trunk and float32 in diffusion, matching
    # the dtypes that the former float32 host feature reached at those sites.
    rel_pos = np.eye(2 * (r_max + 1), dtype=np.int8)[residue_bins]
    rel_token = np.eye(2 * (r_max + 1), dtype=np.int8)[token_bins]
    rel_chain = np.eye(2 * (s_max + 1), dtype=np.int8)[chain_bins]
    return np.concatenate(
        [rel_pos, rel_token, same_entity[..., None].astype(np.int8), rel_chain],
        axis=-1,
    )


if __name__ == "__main__":
    main()
