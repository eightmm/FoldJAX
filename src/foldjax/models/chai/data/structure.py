"""Torch-free Chai all-atom structure tokenization for inference inputs."""

from __future__ import annotations

import secrets
from dataclasses import dataclass, fields, replace
from typing import Any

import numpy as np

from foldjax.models.chai.bridge.conformer_io import ConformerData
from foldjax.models.chai.data.chemistry import (
    CovalentBond,
    GlycosidicBond,
    generate_ligand_conformer,
    parse_glycan,
)
from foldjax.models.chai.data.input import (
    EntityType,
    Input,
    constituents_of_modified_fasta,
    synthetic_chain_id,
)

_RESTYPES = "ARNDCQEGHILKMFPSTWYV"
_RESTYPE_INDEX = {name: index for index, name in enumerate(_RESTYPES)}
_RESTYPE_INDEX.update(
    {
        "X": 20,
        "RA": 21,
        "RC": 22,
        "RG": 23,
        "RU": 24,
        "RX": 25,
        "DA": 26,
        "DC": 27,
        "DG": 28,
        "DT": 29,
        "DX": 30,
    }
)
_AA_1_TO_3 = dict(
    zip(
        _RESTYPES,
        (
            "ALA",
            "ARG",
            "ASN",
            "ASP",
            "CYS",
            "GLN",
            "GLU",
            "GLY",
            "HIS",
            "ILE",
            "LEU",
            "LYS",
            "MET",
            "PHE",
            "PRO",
            "SER",
            "THR",
            "TRP",
            "TYR",
            "VAL",
        ),
        strict=True,
    )
)
_AA_3_TO_1 = {value: key for key, value in _AA_1_TO_3.items()}
_ATOM37 = {
    name: index
    for index, name in enumerate(
        (
            "N",
            "CA",
            "C",
            "CB",
            "O",
            "CG",
            "CG1",
            "CG2",
            "OG",
            "OG1",
            "SG",
            "CD",
            "CD1",
            "CD2",
            "ND1",
            "ND2",
            "OD1",
            "OD2",
            "SD",
            "CE",
            "CE1",
            "CE2",
            "CE3",
            "NE",
            "NE1",
            "NE2",
            "OE1",
            "OE2",
            "CH2",
            "NH1",
            "NH2",
            "OH",
            "CZ",
            "CZ2",
            "CZ3",
            "NZ",
            "OXT",
        )
    )
}
_NA_ATOM_INDEX = {
    "A": {
        "O4'": 0,
        "C1'": 1,
        "C2'": 2,
        "OP1": 3,
        "P": 4,
        "OP2": 5,
        "O5'": 6,
        "C5'": 7,
        "C4'": 8,
        "C3'": 9,
        "O3'": 10,
        "O2'": 11,
        "N1": 12,
        "C2": 13,
        "N3": 14,
        "C4": 15,
        "C5": 16,
        "C6": 17,
        "N6": 18,
        "N7": 19,
        "C8": 20,
        "N9": 21,
    },
    "C": {
        "O4'": 0,
        "C1'": 1,
        "C2'": 2,
        "OP1": 3,
        "P": 4,
        "OP2": 5,
        "O5'": 6,
        "C5'": 7,
        "C4'": 8,
        "C3'": 9,
        "O3'": 10,
        "O2'": 11,
        "N1": 12,
        "C2": 13,
        "O2": 14,
        "N3": 15,
        "C4": 16,
        "N4": 17,
        "C5": 18,
        "C6": 19,
    },
    "G": {
        "O4'": 0,
        "C1'": 1,
        "C2'": 2,
        "OP1": 3,
        "P": 4,
        "OP2": 5,
        "O5'": 6,
        "C5'": 7,
        "C4'": 8,
        "C3'": 9,
        "O3'": 10,
        "O2'": 11,
        "N1": 12,
        "C2": 13,
        "N2": 14,
        "N3": 15,
        "C4": 16,
        "C5": 17,
        "C6": 18,
        "O6": 19,
        "N7": 20,
        "C8": 21,
        "N9": 22,
    },
    "U": {
        "O4'": 0,
        "C1'": 1,
        "C2'": 2,
        "OP1": 3,
        "P": 4,
        "OP2": 5,
        "O5'": 6,
        "C5'": 7,
        "C4'": 8,
        "C3'": 9,
        "O3'": 10,
        "O2'": 11,
        "N1": 12,
        "C2": 13,
        "O2": 14,
        "N3": 15,
        "C4": 16,
        "O4": 17,
        "C5": 18,
        "C6": 19,
    },
    "DA": {
        "O4'": 0,
        "C1'": 1,
        "C2'": 2,
        "OP1": 3,
        "P": 4,
        "OP2": 5,
        "O5'": 6,
        "C5'": 7,
        "C4'": 8,
        "C3'": 9,
        "O3'": 10,
        "N9": 11,
        "C4": 12,
        "N3": 13,
        "C2": 14,
        "N1": 15,
        "C6": 16,
        "C5": 17,
        "N7": 18,
        "C8": 19,
        "N6": 20,
    },
    "DC": {
        "O4'": 0,
        "C1'": 1,
        "C2'": 2,
        "OP1": 3,
        "P": 4,
        "OP2": 5,
        "O5'": 6,
        "C5'": 7,
        "C4'": 8,
        "C3'": 9,
        "O3'": 10,
        "N1": 11,
        "C2": 12,
        "O2": 13,
        "N3": 14,
        "C4": 15,
        "N4": 16,
        "C5": 17,
        "C6": 18,
    },
    "DG": {
        "O4'": 0,
        "C1'": 1,
        "C2'": 2,
        "OP1": 3,
        "P": 4,
        "OP2": 5,
        "O5'": 6,
        "C5'": 7,
        "C4'": 8,
        "C3'": 9,
        "O3'": 10,
        "N9": 11,
        "C4": 12,
        "N3": 13,
        "C2": 14,
        "N1": 15,
        "C6": 16,
        "C5": 17,
        "N7": 18,
        "C8": 19,
        "N2": 20,
        "O6": 21,
    },
    "DT": {
        "O4'": 0,
        "C1'": 1,
        "C2'": 2,
        "OP1": 3,
        "P": 4,
        "OP2": 5,
        "O5'": 6,
        "C5'": 7,
        "C4'": 8,
        "C3'": 9,
        "O3'": 10,
        "N1": 11,
        "C2": 12,
        "O2": 13,
        "N3": 14,
        "C4": 15,
        "O4": 16,
        "C5": 17,
        "C7": 18,
        "C6": 19,
    },
}
_STANDARD = set(_AA_3_TO_1) | set(_NA_ATOM_INDEX) | {"UNK"}


def _random_rigid_augment(
    conformer: ConformerData, rng: np.random.Generator
) -> ConformerData:
    """Match Chai's centering, uniform rotation, and normal translation law."""
    position = np.asarray(conformer.position, np.float32)
    quaternion = rng.standard_normal(4).astype(np.float32)
    quaternion /= np.copysign(np.linalg.norm(quaternion), quaternion[0])
    real, i, j, k = quaternion
    scale = np.float32(2.0) / np.dot(quaternion, quaternion)
    rotation = np.asarray(
        [
            [
                1 - scale * (j * j + k * k),
                scale * (i * j - k * real),
                scale * (i * k + j * real),
            ],
            [
                scale * (i * j + k * real),
                1 - scale * (i * i + k * k),
                scale * (j * k - i * real),
            ],
            [
                scale * (i * k - j * real),
                scale * (j * k + i * real),
                1 - scale * (i * i + j * j),
            ],
        ],
        np.float32,
    )
    centered = position - position.mean(axis=0, keepdims=True)
    transformed = centered @ rotation.T + rng.standard_normal((1, 3)).astype(np.float32)
    return conformer._replace(position=transformed)


def _tensorcode(value: str, length: int) -> np.ndarray:
    if not value.isascii() or len(value) > length:
        raise ValueError(f"ASCII value {value!r} does not fit length {length}")
    return np.asarray(
        [*(ord(char) for char in value), *([255] * (length - len(value)))], np.uint8
    )


def _residue_names(item: Input) -> list[str]:
    parts = constituents_of_modified_fasta(item.sequence)
    if parts is None:
        raise ValueError(f"invalid polymer FASTA: {item.sequence!r}")
    entity_type = EntityType(item.entity_type)
    if entity_type is EntityType.PROTEIN:
        return [
            _AA_1_TO_3.get(part, "UNK") if len(part) == 1 else part for part in parts
        ]
    mapping = (
        {key: key for key in "ACGU"}
        if entity_type is EntityType.RNA
        else {key: f"D{key}" for key in "ACGT"}
    )
    unknown = "X" if entity_type is EntityType.RNA else "DX"
    return [mapping.get(part, unknown) if len(part) == 1 else part for part in parts]


def _restype(residue_name: str, entity_type: EntityType) -> int:
    if entity_type is EntityType.PROTEIN:
        return _RESTYPE_INDEX.get(_AA_3_TO_1.get(residue_name, "X"), 20)
    if entity_type is EntityType.RNA:
        return _RESTYPE_INDEX.get(f"R{residue_name}", 25)
    if entity_type is EntityType.DNA:
        return _RESTYPE_INDEX.get(residue_name, 30)
    return 20


@dataclass(frozen=True)
class StructureContext:
    """NumPy equivalent of Chai's inference ``AllAtomStructureContext``."""

    token_residue_type: np.ndarray
    token_residue_index: np.ndarray
    token_index: np.ndarray
    token_centre_atom_index: np.ndarray
    token_ref_atom_index: np.ndarray
    token_exists_mask: np.ndarray
    token_backbone_frame_mask: np.ndarray
    token_backbone_frame_index: np.ndarray
    token_asym_id: np.ndarray
    token_entity_id: np.ndarray
    token_sym_id: np.ndarray
    token_entity_type: np.ndarray
    token_residue_name: np.ndarray
    token_b_factor_or_plddt: np.ndarray
    atom_token_index: np.ndarray
    atom_within_token_index: np.ndarray
    atom_ref_pos: np.ndarray
    atom_ref_mask: np.ndarray
    atom_ref_element: np.ndarray
    atom_ref_charge: np.ndarray
    atom_ref_name: tuple[str, ...]
    atom_ref_name_chars: np.ndarray
    atom_ref_space_uid: np.ndarray
    atom_is_not_padding_mask: np.ndarray
    atom_gt_coords: np.ndarray
    atom_exists_mask: np.ndarray
    pdb_id: np.ndarray
    source_pdb_chain_id: np.ndarray
    subchain_id: np.ndarray
    resolution: np.ndarray
    is_distillation: np.ndarray
    symmetries: np.ndarray
    atom_covalent_bond_indices: tuple[np.ndarray, np.ndarray]

    @property
    def num_tokens(self) -> int:
        return int(self.token_index.shape[0])

    @property
    def num_atoms(self) -> int:
        return int(self.atom_token_index.shape[0])

    def to_dict(self) -> dict[str, Any]:
        return {field.name: getattr(self, field.name) for field in fields(self)}

    def pad(self, n_tokens: int, n_atoms: int) -> StructureContext:
        if n_tokens < self.num_tokens or n_atoms < self.num_atoms:
            raise ValueError("padding target is smaller than the structure context")

        def pad(value: np.ndarray, size: int, fill: int = 0) -> np.ndarray:
            return np.pad(
                value,
                [(0, size - value.shape[0]), *([(0, 0)] * (value.ndim - 1))],
                constant_values=fill,
            )

        values = self.to_dict()
        token_names = {name for name in values if name.startswith("token_")} | {
            "pdb_id",
            "source_pdb_chain_id",
            "subchain_id",
        }
        atom_names = {name for name in values if name.startswith("atom_")} | {
            "symmetries"
        }
        atom_names -= {"atom_ref_name", "atom_covalent_bond_indices"}
        for name in token_names:
            values[name] = pad(values[name], n_tokens)
        for name in atom_names:
            values[name] = pad(
                values[name],
                n_atoms,
                -1 if name in {"atom_ref_space_uid", "symmetries"} else 0,
            )
        return StructureContext(**values)


def _make_sym_ids(entity_ids: list[int]) -> list[int]:
    counts: dict[int, int] = {}
    output = []
    for entity_id in entity_ids:
        output.append(counts.get(entity_id, 0))
        counts[entity_id] = output[-1] + 1
    return output


def _tokenize_chain(
    residue_names: list[str],
    entity_type: EntityType,
    conformers: dict[str, ConformerData],
    *,
    chain_id: int,
    entity_id: int,
    sym_id: int,
    subchain_id: str,
    identifier: str,
    glycosidic_bonds: list[GlycosidicBond] | None = None,
    rng: np.random.Generator | None = None,
) -> StructureContext:
    token_restype: list[int] = []
    token_residue_index: list[int] = []
    centre: list[int] = []
    reference: list[int] = []
    frame_mask: list[bool] = []
    frame_index: list[list[int]] = []
    token_names: list[str] = []
    atom_token: list[int] = []
    atom_within: list[int] = []
    atom_pos: list[np.ndarray] = []
    atom_element: list[np.ndarray] = []
    atom_charge: list[np.ndarray] = []
    atom_names: list[str] = []
    atom_uid: list[int] = []
    symmetries: list[np.ndarray] = []
    atom_offset = 0
    token_offset = 0
    max_symmetries = 1

    for residue_index, residue_name in enumerate(residue_names):
        try:
            conformer = conformers[residue_name]
        except KeyError as error:
            raise ValueError(
                f"reference conformer is missing for {residue_name}"
            ) from error
        if residue_name not in _STANDARD and entity_type is not EntityType.LIGAND:
            if rng is None:
                raise ValueError("modified-residue augmentation requires an RNG")
            conformer = _random_rigid_augment(conformer, rng)
        names = list(conformer.atom_names)
        n_atoms = len(names)
        if n_atoms == 0:
            continue
        standard = residue_name in _STANDARD
        n_tokens = 1 if standard else n_atoms
        token_restype.extend([_restype(residue_name, entity_type)] * n_tokens)
        token_residue_index.extend([residue_index] * n_tokens)
        token_names.extend([residue_name] * n_tokens)
        if standard:
            centre_name = "C1'" if residue_name in _NA_ATOM_INDEX else "CA"
            if residue_name == "GLY":
                reference_name = "CA"
            elif residue_name in {"A", "G", "DA", "DG"}:
                reference_name = "C4"
            elif residue_name in {"C", "U", "DC", "DT"}:
                reference_name = "C2"
            else:
                reference_name = "CB"
            try:
                centre.append(atom_offset + names.index(centre_name))
                reference.append(atom_offset + names.index(reference_name))
            except ValueError as error:
                raise ValueError(
                    f"standard conformer {residue_name} misses frame atom"
                ) from error
            frame_names = (
                ("C1'", "C3'", "C4'")
                if residue_name in _NA_ATOM_INDEX
                else ("N", "CA", "C")
            )
            complete = all(name in names for name in frame_names)
            frame_mask.append(complete)
            frame_index.append(
                [atom_offset + names.index(name) for name in frame_names]
                if complete
                else [atom_offset] * 3
            )
            atom_token.extend([token_offset] * n_atoms)
            index_map = (
                {name: index for index, name in enumerate(names)}
                if residue_name == "UNK"
                else _NA_ATOM_INDEX.get(residue_name, _ATOM37)
            )
            try:
                atom_within.extend(index_map[name] for name in names)
            except KeyError as error:
                raise ValueError(
                    f"unknown atom {error.args[0]!r} in {residue_name}"
                ) from error
        else:
            centre.extend(range(atom_offset, atom_offset + n_atoms))
            reference.extend(range(atom_offset, atom_offset + n_atoms))
            frame_mask.extend([False] * n_atoms)
            frame_index.extend([[atom_offset + index] * 3 for index in range(n_atoms)])
            atom_token.extend(range(token_offset, token_offset + n_atoms))
            atom_within.extend([0] * n_atoms)
        atom_pos.append(np.asarray(conformer.position, np.float32))
        atom_element.append(np.asarray(conformer.element, np.int32))
        atom_charge.append(np.asarray(conformer.charge, np.int32))
        atom_names.extend(names)
        atom_uid.extend([residue_index] * n_atoms)
        symm = np.asarray(conformer.symmetries, np.int64)
        symmetries.append(symm)
        max_symmetries = max(max_symmetries, symm.shape[1])
        atom_offset += n_atoms
        token_offset += n_tokens

    if not token_restype:
        raise ValueError("input contains no tokenizable residues")
    padded_symmetries = [
        np.pad(
            value, ((0, 0), (0, max_symmetries - value.shape[1])), constant_values=-1
        )
        for value in symmetries
    ]
    n_tokens = len(token_restype)
    n_atoms = len(atom_token)
    token_index = np.arange(n_tokens, dtype=np.int32)
    token_exists = np.bincount(np.asarray(atom_token), minlength=n_tokens) > 0
    position = np.concatenate(atom_pos).astype(np.float32, copy=False)
    left_bonds: list[int] = []
    right_bonds: list[int] = []
    for bond in glycosidic_bonds or []:
        left = [
            index
            for index, (residue, name) in enumerate(
                zip(atom_uid, atom_names, strict=True)
            )
            if residue == bond.src_sugar_index and name == bond.src_atom_name
        ]
        right = [
            index
            for index, (residue, name) in enumerate(
                zip(atom_uid, atom_names, strict=True)
            )
            if residue == bond.dst_sugar_index and name == bond.dst_atom_name
        ]
        if len(left) != 1 or len(right) != 1:
            raise ValueError(
                "glycosidic bond atoms must resolve uniquely: "
                f"{bond.src_atom_name}->{bond.dst_atom_name}"
            )
        left_bonds.append(left[0])
        right_bonds.append(right[0])
    local_bonds = (
        np.asarray(left_bonds, np.int64),
        np.asarray(right_bonds, np.int64),
    )
    return StructureContext(
        token_residue_type=np.asarray(token_restype, np.int32),
        token_residue_index=np.asarray(token_residue_index, np.int32),
        token_index=token_index,
        token_centre_atom_index=np.asarray(centre, np.int32),
        token_ref_atom_index=np.asarray(reference, np.int32),
        token_exists_mask=token_exists,
        token_backbone_frame_mask=np.asarray(frame_mask),
        token_backbone_frame_index=np.asarray(frame_index, np.int32),
        token_asym_id=np.full(n_tokens, chain_id, np.int32),
        token_entity_id=np.full(n_tokens, entity_id, np.int32),
        token_sym_id=np.full(n_tokens, sym_id, np.int32),
        token_entity_type=np.full(n_tokens, entity_type.value, np.int32),
        token_residue_name=np.stack([_tensorcode(name, 8) for name in token_names]),
        token_b_factor_or_plddt=np.ones(n_tokens, np.float32),
        atom_token_index=np.asarray(atom_token, np.int32),
        atom_within_token_index=np.asarray(atom_within, np.int32),
        atom_ref_pos=position,
        atom_ref_mask=np.ones(n_atoms, bool),
        atom_ref_element=np.concatenate(atom_element).astype(np.int32, copy=False),
        atom_ref_charge=np.concatenate(atom_charge).astype(np.int32, copy=False),
        atom_ref_name=tuple(atom_names),
        atom_ref_name_chars=np.asarray(
            [[ord(char) - 32 for char in name.ljust(4)[:4]] for name in atom_names],
            np.int32,
        ),
        atom_ref_space_uid=np.asarray(atom_uid, np.int32),
        atom_is_not_padding_mask=np.ones(n_atoms, bool),
        atom_gt_coords=position.copy(),
        atom_exists_mask=np.ones(n_atoms, bool),
        pdb_id=np.repeat(_tensorcode(identifier, 32)[None], n_tokens, axis=0),
        source_pdb_chain_id=np.repeat(
            _tensorcode(subchain_id, 4)[None], n_tokens, axis=0
        ),
        subchain_id=np.repeat(_tensorcode(subchain_id, 4)[None], n_tokens, axis=0),
        resolution=np.asarray([0.0], np.float32),
        is_distillation=np.asarray([False]),
        symmetries=np.concatenate(padded_symmetries),
        atom_covalent_bond_indices=local_bonds,
    )


def _decode_tensorcode(value: np.ndarray) -> str:
    return "".join(chr(int(code)) for code in value if code != 255)


def _residue_position(value: str) -> tuple[str, int]:
    if not value:
        return "", 0
    name = value[0]
    suffix = value[1:]
    try:
        position = int(suffix) - 1 if suffix else 0
    except ValueError as error:
        raise ValueError(f"invalid covalent residue selector: {value!r}") from error
    if position < 0:
        raise ValueError(f"invalid covalent residue selector: {value!r}")
    return name, position


def _resolve_covalent_atom(
    context: StructureContext,
    chain: str,
    residue: str,
    atom_name: str,
) -> int:
    chain_mask = np.asarray(
        [_decode_tensorcode(value) == chain for value in context.subchain_id], bool
    )
    if not np.any(chain_mask):
        raise ValueError(f"unknown covalent-bond chain: {chain!r}")
    residue_name, residue_index = _residue_position(residue)
    token_mask = chain_mask & (context.token_residue_index == residue_index)
    if residue_name:
        expected = _AA_1_TO_3.get(residue_name, "UNK")
        token_mask &= np.asarray(
            [
                _decode_tensorcode(value) == expected
                for value in context.token_residue_name
            ],
            bool,
        )
    token_indices = np.flatnonzero(token_mask)
    if token_indices.size == 0:
        raise ValueError(f"covalent residue does not resolve: {chain}:{residue or '1'}")
    atom_mask = np.isin(context.atom_token_index, token_indices) & np.asarray(
        [name == atom_name for name in context.atom_ref_name], bool
    )
    atom_indices = np.flatnonzero(atom_mask)
    if atom_indices.size != 1:
        raise ValueError(
            f"covalent atom must resolve uniquely: {chain}:{residue or '1'}@{atom_name}"
        )
    return int(atom_indices[0])


def add_covalent_bonds(
    context: StructureContext, bonds: list[CovalentBond]
) -> StructureContext:
    """Resolve declared Chai covalent bonds against a merged structure context."""
    if not bonds:
        return context
    left, right = context.atom_covalent_bond_indices
    new_left = [
        _resolve_covalent_atom(context, bond.chain_a, bond.residue_a, bond.atom_a)
        for bond in bonds
    ]
    new_right = [
        _resolve_covalent_atom(context, bond.chain_b, bond.residue_b, bond.atom_b)
        for bond in bonds
    ]
    return replace(
        context,
        atom_covalent_bond_indices=(
            np.concatenate([left, np.asarray(new_left, np.int64)]),
            np.concatenate([right, np.asarray(new_right, np.int64)]),
        ),
    )


def drop_glycan_leaving_atoms(context: StructureContext) -> StructureContext:
    """Match Chai's terminal-oxygen removal after glycosidic bond formation."""
    exists = context.atom_exists_mask.copy()

    def nearby(atom_index: int, allowed_elements: tuple[int, ...] | None) -> np.ndarray:
        token = int(context.atom_token_index[atom_index])
        if context.token_entity_type[token] != EntityType.MANUAL_GLYCAN.value:
            return np.empty(0, np.int64)
        residue_index = context.token_residue_index[token]
        asym_id = context.token_asym_id[token]
        atom_tokens = context.atom_token_index
        same_residue = (
            (context.token_residue_index[atom_tokens] == residue_index)
            & (context.token_asym_id[atom_tokens] == asym_id)
            & exists
        )
        delta = context.atom_gt_coords - context.atom_gt_coords[atom_index]
        distance = np.sqrt(np.sum(delta * delta, axis=-1, dtype=np.float32))
        candidates = same_residue & (distance < np.float32(1.5))
        candidates[atom_index] = False
        if allowed_elements is not None:
            candidates &= np.isin(context.atom_ref_element, allowed_elements)
        return np.flatnonzero(candidates)

    left, right = context.atom_covalent_bond_indices
    for atom_a, atom_b in zip(left, right, strict=True):
        removed = False
        for bonded_atom in (int(atom_b), int(atom_a)):
            oxygen_candidates = nearby(bonded_atom, (8,))
            terminal = [
                int(candidate)
                for candidate in oxygen_candidates
                if nearby(int(candidate), None).size == 1
            ]
            if len(terminal) == 1:
                exists[terminal[0]] = False
                removed = True
            if removed:
                break
    return replace(context, atom_exists_mask=exists)


def merge_structure_contexts(contexts: list[StructureContext]) -> StructureContext:
    """Merge chains using the released Chai index and symmetry conventions."""
    if not contexts:
        raise ValueError("at least one structure context is required")
    token_offsets = np.cumsum([0, *(ctx.num_tokens for ctx in contexts[:-1])])
    atom_offsets = np.cumsum([0, *(ctx.num_atoms for ctx in contexts[:-1])])
    max_symmetries = max(ctx.symmetries.shape[1] for ctx in contexts)

    def cat(name: str) -> np.ndarray:
        return np.concatenate([getattr(ctx, name) for ctx in contexts])

    values = contexts[0].to_dict()
    token_arrays = {name for name in values if name.startswith("token_")} | {
        "pdb_id",
        "source_pdb_chain_id",
        "subchain_id",
    }
    atom_arrays = {name for name in values if name.startswith("atom_")} | {"symmetries"}
    atom_arrays -= {"atom_ref_name", "atom_covalent_bond_indices"}
    for name in token_arrays | atom_arrays:
        if name in {
            "token_index",
            "token_centre_atom_index",
            "token_ref_atom_index",
            "token_backbone_frame_index",
            "atom_token_index",
            "atom_ref_space_uid",
            "symmetries",
        }:
            continue
        values[name] = cat(name)
    values["token_index"] = np.arange(
        sum(ctx.num_tokens for ctx in contexts), dtype=np.int32
    )
    values["atom_token_index"] = np.concatenate(
        [
            ctx.atom_token_index + offset
            for ctx, offset in zip(contexts, token_offsets, strict=True)
        ]
    ).astype(np.int32)
    values["token_centre_atom_index"] = np.concatenate(
        [
            ctx.token_centre_atom_index + offset
            for ctx, offset in zip(contexts, atom_offsets, strict=True)
        ]
    ).astype(np.int32)
    values["token_ref_atom_index"] = np.concatenate(
        [
            ctx.token_ref_atom_index + offset
            for ctx, offset in zip(contexts, atom_offsets, strict=True)
        ]
    ).astype(np.int32)
    # Preserve the upstream Torch preprocessing convention for compatibility.
    values["token_backbone_frame_index"] = np.concatenate(
        [
            ctx.token_backbone_frame_index + offset
            for ctx, offset in zip(contexts, token_offsets, strict=True)
        ]
    ).astype(np.int32)
    values["atom_ref_space_uid"] = np.concatenate(
        [
            np.unique(ctx.atom_ref_space_uid, return_inverse=True)[1] + offset
            for ctx, offset in zip(contexts, atom_offsets, strict=True)
        ]
    ).astype(np.int64)
    values["symmetries"] = np.concatenate(
        [
            np.pad(
                ctx.symmetries,
                ((0, 0), (0, max_symmetries - ctx.symmetries.shape[1])),
                constant_values=-1,
            )
            for ctx in contexts
        ]
    )
    values["atom_ref_name"] = tuple(
        name for ctx in contexts for name in ctx.atom_ref_name
    )
    left, right = [], []
    for ctx, offset in zip(contexts, atom_offsets, strict=True):
        left.append(ctx.atom_covalent_bond_indices[0] + offset)
        right.append(ctx.atom_covalent_bond_indices[1] + offset)
    values["atom_covalent_bond_indices"] = (np.concatenate(left), np.concatenate(right))
    values["resolution"] = np.max(
        np.stack([ctx.resolution for ctx in contexts]), axis=0
    )
    values["is_distillation"] = np.max(
        np.stack([ctx.is_distillation for ctx in contexts]), axis=0
    )
    return replace(contexts[0], **values)


def tokenize_inputs(
    inputs: list[Input],
    conformers: dict[str, ConformerData],
    *,
    identifier: str = "test",
    entity_name_as_subchain: bool = False,
    covalent_bonds: list[CovalentBond] | None = None,
    rng: np.random.Generator | None = None,
) -> StructureContext:
    """Tokenize and merge Chai public inputs without Torch or Gemmi."""
    rng = np.random.default_rng(secrets.randbits(128)) if rng is None else rng
    entity_keys: dict[tuple[int, tuple[str, ...]], int] = {}
    chain_data: list[
        tuple[
            Input,
            EntityType,
            list[str],
            int,
            dict[str, ConformerData],
            list[GlycosidicBond],
        ]
    ] = []
    for item in inputs:
        entity_type = EntityType(item.entity_type)
        chain_conformers = conformers
        glycosidic_bonds: list[GlycosidicBond] = []
        if entity_type in {EntityType.PROTEIN, EntityType.RNA, EntityType.DNA}:
            names = _residue_names(item)
            entity_sequence = tuple(names)
        elif entity_type is EntityType.LIGAND:
            names = ["LIG"]
            entity_sequence = (item.sequence,)
            chain_conformers = {
                **conformers,
                "LIG": generate_ligand_conformer(item.sequence),
            }
        elif entity_type is EntityType.MANUAL_GLYCAN:
            names, glycosidic_bonds = parse_glycan(item.sequence)
            entity_sequence = tuple(names)
        else:
            raise NotImplementedError(
                f"native tokenization is not implemented for {entity_type.name}"
            )
        key = (entity_type.value, entity_sequence)
        entity_id = entity_keys.setdefault(key, len(entity_keys))
        chain_data.append(
            (
                item,
                entity_type,
                names,
                entity_id,
                chain_conformers,
                glycosidic_bonds,
            )
        )
    sym_ids = _make_sym_ids([data[3] for data in chain_data])
    contexts = []
    for index, (
        (item, entity_type, names, entity_id, chain_conformers, glycosidic_bonds),
        sym_id,
    ) in enumerate(zip(chain_data, sym_ids, strict=True)):
        subchain = (
            item.entity_name if entity_name_as_subchain else synthetic_chain_id(index)
        )
        contexts.append(
            _tokenize_chain(
                names,
                entity_type,
                chain_conformers,
                chain_id=index + 1,
                entity_id=entity_id,
                sym_id=sym_id,
                subchain_id=subchain,
                identifier=identifier,
                glycosidic_bonds=glycosidic_bonds,
                rng=rng,
            )
        )
    merged = add_covalent_bonds(
        merge_structure_contexts(contexts), covalent_bonds or []
    )
    return drop_glycan_leaving_atoms(merged)
