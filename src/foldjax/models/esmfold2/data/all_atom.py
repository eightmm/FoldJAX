"""Torch-free all-biomolecule ESMFold2 featurization.

This is the NumPy adaptation of Biohub's released ``prepare_input.py`` input
contract: standard proteins, DNA and RNA are residue-tokenized; modified
residues and ligands are atom-tokenized; CCD/SMILES bonds and caller-supplied
covalent bonds populate the dense token-bond feature.  Publisher chemistry is
read through :class:`CCDStore`; no publisher Python package or torch runtime is
needed.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from foldjax.models.esmfold2.data import chemistry
from foldjax.models.esmfold2.data.all_atom_constants import (
    CHARGED_ATOMS,
    DNA_1TO3,
    DNA_BACKBONE_ATOMS,
    DNA_HEAVY_ATOMS,
    DNA_RESIDUE_TO_RES_TYPE,
    DNA_RNA_LIGAND_INPUT_ID,
    DNA_UNK_RES_TYPE,
    ELEMENT_TO_ATOMIC_NUM,
    ESM_PROTEIN_VOCAB,
    MOL_TYPE_DNA,
    MOL_TYPE_NONPOLYMER,
    MOL_TYPE_PROTEIN,
    MOL_TYPE_RNA,
    MSA_GAP_TOKEN_ID,
    PROTEIN_1TO3,
    PROTEIN_HEAVY_ATOMS,
    PROTEIN_RESIDUE_TO_RES_TYPE,
    PROTEIN_UNK_RES_TYPE,
    RNA_1TO3,
    RNA_BACKBONE_ATOMS,
    RNA_HEAVY_ATOMS,
    RNA_RESIDUE_TO_RES_TYPE,
    RNA_UNK_RES_TYPE,
)
from foldjax.models.esmfold2.data.ccd import CCDStore, get_ccd_store

ATOM_BLOCK = 32
_ZERO = np.zeros(3, dtype=np.float32)
OUTPUT_METADATA_FEATURES = frozenset(
    {"token_chain_id_chars", "token_residue_name_chars"}
)


@dataclass(slots=True)
class Atom:
    name: str
    element: str
    charge: int
    ref_pos: np.ndarray
    token_index: int
    atom_index: int
    space_uid: int


@dataclass(slots=True)
class Token:
    token_index: int
    residue_index: int
    residue_name: str
    mol_type: int
    res_type: int
    input_id: int
    asym_id: int
    sym_id: int
    entity_id: int
    atom_start: int
    atom_count: int


@dataclass(slots=True)
class Chain:
    chain_id: str
    asym_id: int
    entity_index: int
    entity_id: int
    sym_id: int
    kind: str
    sequence: str | None
    tokens: list[Token] = field(default_factory=list)
    ligand_bonds: list[tuple[str, str]] = field(default_factory=list)


def _text_bytes(value: str) -> bytes:
    try:
        return value.encode("ascii")
    except UnicodeEncodeError as error:
        raise ValueError(
            f"ESMFold2 mmCIF identifiers must be ASCII: {value!r}"
        ) from error


def _encode_text(value: str, *, width: int) -> np.ndarray:
    encoded = _text_bytes(value)
    result = np.zeros(width, dtype=np.uint8)
    result[: len(encoded)] = np.frombuffer(encoded, dtype=np.uint8)
    return result


def _ids(entity: Mapping[str, Any]) -> list[str]:
    value = entity.get("id")
    return [str(item) for item in value] if isinstance(value, list) else [str(value)]


def _modifications(entity: Mapping[str, Any]) -> dict[int, str]:
    return {
        int(item["position"]) - 1: str(item["ccd"]).upper()
        for item in entity.get("modifications", ())
    }


def _entity_key(entity: Mapping[str, Any]) -> tuple[object, ...]:
    """Biohub's canonical chemical identity, independent of common input index."""

    kind = str(entity["type"])
    if kind == "ligand":
        if entity.get("ccd"):
            return ("NONPOLYMER", None, (str(entity["ccd"]).upper(),))
        return ("NONPOLYMER", entity.get("smiles"), ())
    return (
        kind.upper(),
        str(entity["sequence"]),
        frozenset(
            (int(item["position"]) - 1, str(item["ccd"]).upper(), None)
            for item in entity.get("modifications", ())
        ),
    )


def _element(atom_name: str) -> str:
    name = atom_name.strip().upper()
    if not name:
        return "C"
    if name[0].isdigit():
        return name[1] if len(name) > 1 else "H"
    # Match Biohub's atom-name inference exactly. In particular ``CA`` is an
    # alpha carbon, not calcium; arbitrary CCD/SMILES atoms carry their real
    # element separately and do not pass through this heuristic.
    if len(name) == 2 and name in {
        "FE",
        "ZN",
        "MG",
        "MN",
        "CO",
        "NI",
        "CU",
        "SE",
        "BR",
    }:
        return name
    return name[0]


def _position(value: np.ndarray | None) -> np.ndarray:
    return _ZERO.copy() if value is None else np.asarray(value, dtype=np.float32).copy()


def _next_space_uid(atoms: Sequence[Atom]) -> int:
    """Next residue-local geometry group in the monotonically built atom list."""

    return 0 if not atoms else atoms[-1].space_uid + 1


def _append_standard_residue(
    *,
    tokens: list[Token],
    atoms: list[Atom],
    atom_names: Sequence[str],
    residue_index: int,
    residue_name: str,
    mol_type: int,
    res_type: int,
    input_id: int,
    asym_id: int,
    sym_id: int,
    entity_id: int,
    space_uid: int,
    positions: Mapping[str, np.ndarray] | None,
) -> None:
    token_index = len(tokens)
    atom_start = len(atoms)
    for name in atom_names:
        atoms.append(
            Atom(
                name=name,
                element=_element(name),
                charge=CHARGED_ATOMS.get((residue_name, name), 0),
                ref_pos=_position(None if positions is None else positions.get(name)),
                token_index=token_index,
                atom_index=len(atoms),
                space_uid=space_uid,
            )
        )
    tokens.append(
        Token(
            token_index=token_index,
            residue_index=residue_index,
            residue_name=residue_name,
            mol_type=mol_type,
            res_type=res_type,
            input_id=input_id,
            asym_id=asym_id,
            sym_id=sym_id,
            entity_id=entity_id,
            atom_start=atom_start,
            atom_count=len(atom_names),
        )
    )


def _append_atom_tokenized_residue(
    *,
    tokens: list[Token],
    atoms: list[Atom],
    records: Sequence[tuple[str, str, int]],
    residue_index: int,
    residue_name: str,
    mol_type: int,
    asym_id: int,
    sym_id: int,
    entity_id: int,
    space_uid: int,
    ccd: CCDStore,
) -> None:
    single = len(records) == 1
    for name, element, charge in records:
        token_index = len(tokens)
        atom_index = len(atoms)
        atoms.append(
            Atom(
                name=name,
                element=element,
                charge=int(charge),
                ref_pos=_ZERO.copy()
                if single
                else _position(ccd.idealized_position(residue_name, name)),
                token_index=token_index,
                atom_index=atom_index,
                space_uid=space_uid,
            )
        )
        tokens.append(
            Token(
                token_index=token_index,
                residue_index=residue_index,
                residue_name=residue_name,
                mol_type=mol_type,
                res_type=PROTEIN_UNK_RES_TYPE,
                input_id=DNA_RNA_LIGAND_INPUT_ID,
                asym_id=asym_id,
                sym_id=sym_id,
                entity_id=entity_id,
                atom_start=atom_index,
                atom_count=1,
            )
        )


def _protein(
    entity: Mapping[str, Any],
    *,
    asym_id: int,
    sym_id: int,
    entity_id: int,
    tokens: list[Token],
    atoms: list[Atom],
    ccd: CCDStore,
) -> None:
    sequence = str(entity["sequence"])
    modifications = _modifications(entity)
    for residue_index, letter in enumerate(sequence):
        residue_name = modifications.get(
            residue_index, PROTEIN_1TO3.get(letter, "UNK")
        )
        if residue_index not in modifications:
            corrected = "MET" if residue_name == "MSE" else residue_name
            names = PROTEIN_HEAVY_ATOMS.get(corrected, PROTEIN_HEAVY_ATOMS["UNK"])
            conformer = ccd.conformer(corrected)
            _append_standard_residue(
                tokens=tokens,
                atoms=atoms,
                atom_names=names,
                residue_index=residue_index,
                residue_name=corrected,
                mol_type=MOL_TYPE_PROTEIN,
                res_type=PROTEIN_RESIDUE_TO_RES_TYPE.get(
                    corrected, PROTEIN_UNK_RES_TYPE
                ),
                input_id=ESM_PROTEIN_VOCAB.get(letter, ESM_PROTEIN_VOCAB["X"]),
                asym_id=asym_id,
                sym_id=sym_id,
                entity_id=entity_id,
                space_uid=_next_space_uid(atoms),
                positions=conformer,
            )
            continue
        records = ccd.atoms(residue_name)
        if not records:
            records = [(name, _element(name), 0) for name in ("N", "CA", "C", "O")]
        if residue_index != len(sequence) - 1:
            leaving = ccd.leaving_atoms(residue_name)
            records = [record for record in records if record[0] not in leaving]
        _append_atom_tokenized_residue(
            tokens=tokens,
            atoms=atoms,
            records=records,
            residue_index=residue_index,
            residue_name=residue_name,
            mol_type=MOL_TYPE_PROTEIN,
            asym_id=asym_id,
            sym_id=sym_id,
            entity_id=entity_id,
            space_uid=_next_space_uid(atoms),
            ccd=ccd,
        )


def _nucleic_acid(
    entity: Mapping[str, Any],
    *,
    kind: str,
    asym_id: int,
    sym_id: int,
    entity_id: int,
    tokens: list[Token],
    atoms: list[Atom],
    ccd: CCDStore,
) -> None:
    sequence = str(entity["sequence"])
    modifications = _modifications(entity)
    dna = kind == "dna"
    mol_type = MOL_TYPE_DNA if dna else MOL_TYPE_RNA
    names_by_residue = DNA_HEAVY_ATOMS if dna else RNA_HEAVY_ATOMS
    backbone = DNA_BACKBONE_ATOMS if dna else RNA_BACKBONE_ATOMS
    mapping = DNA_1TO3 if dna else RNA_1TO3
    residue_types = DNA_RESIDUE_TO_RES_TYPE if dna else RNA_RESIDUE_TO_RES_TYPE
    unknown = DNA_UNK_RES_TYPE if dna else RNA_UNK_RES_TYPE
    for residue_index, letter in enumerate(sequence):
        residue_name = modifications.get(residue_index, mapping.get(letter, "UNK"))
        if residue_index not in modifications:
            atom_names = names_by_residue.get(residue_name, backbone)
            res_type = residue_types.get(residue_name, unknown)
            conformer = None if residue_name == "UNK" else ccd.conformer(residue_name)
            _append_standard_residue(
                tokens=tokens,
                atoms=atoms,
                atom_names=atom_names,
                residue_index=residue_index,
                residue_name=residue_name,
                mol_type=mol_type,
                res_type=res_type,
                input_id=DNA_RNA_LIGAND_INPUT_ID,
                asym_id=asym_id,
                sym_id=sym_id,
                entity_id=entity_id,
                space_uid=_next_space_uid(atoms),
                positions=conformer,
            )
            continue
        records = ccd.atoms(residue_name)
        if not records:
            records = [(name, _element(name), 0) for name in backbone]
        if residue_index != len(sequence) - 1:
            leaving = ccd.leaving_atoms(residue_name)
            records = [record for record in records if record[0] not in leaving]
        _append_atom_tokenized_residue(
            tokens=tokens,
            atoms=atoms,
            records=records,
            residue_index=residue_index,
            residue_name=residue_name,
            mol_type=mol_type,
            asym_id=asym_id,
            sym_id=sym_id,
            entity_id=entity_id,
            space_uid=_next_space_uid(atoms),
            ccd=ccd,
        )


def _smiles(
    value: str,
    *,
    asym_id: int,
    sym_id: int,
    entity_id: int,
    tokens: list[Token],
    atoms: list[Atom],
    seed: int,
) -> list[tuple[str, str]]:
    from rdkit import Chem
    from rdkit.Chem import AllChem

    molecule = Chem.MolFromSmiles(value)
    if molecule is None:
        raise ValueError(f"ESMFold2 could not parse ligand SMILES: {value}")
    molecule = Chem.AddHs(molecule)
    ranks = AllChem.CanonicalRankAtoms(molecule)
    for atom, rank in zip(molecule.GetAtoms(), ranks, strict=True):
        name = atom.GetSymbol().upper() + str(rank + 1)
        if len(name) > 4:
            raise ValueError(f"ESMFold2 SMILES atom name exceeds four chars: {name}")
        atom.SetProp("name", name)
    options = AllChem.ETKDGv3()
    options.clearConfs = False
    options.randomSeed = int(seed)
    conformer_id = AllChem.EmbedMolecule(molecule, options)
    if conformer_id == -1:
        options.useRandomCoords = True
        conformer_id = AllChem.EmbedMolecule(molecule, options)
    if conformer_id == -1:
        raise ValueError(f"ESMFold2 could not generate a conformer for {value}")
    try:
        AllChem.UFFOptimizeMolecule(molecule, confId=conformer_id, maxIters=1000)
    except (RuntimeError, ValueError):
        pass
    molecule = Chem.RemoveHs(molecule)
    if molecule.GetNumConformers() == 0:
        raise ValueError(f"ESMFold2 could not retain a conformer for {value}")
    conformer = molecule.GetConformer(0)
    space_uid = _next_space_uid(atoms)
    for rd_atom in molecule.GetAtoms():
        name = rd_atom.GetProp("name")
        position = conformer.GetAtomPosition(rd_atom.GetIdx())
        token_index = len(tokens)
        atom_index = len(atoms)
        atoms.append(
            Atom(
                name=name,
                element=rd_atom.GetSymbol(),
                charge=rd_atom.GetFormalCharge(),
                ref_pos=np.asarray((position.x, position.y, position.z), np.float32),
                token_index=token_index,
                atom_index=atom_index,
                space_uid=space_uid,
            )
        )
        tokens.append(
            Token(
                token_index=token_index,
                residue_index=0,
                residue_name="LIG",
                mol_type=MOL_TYPE_NONPOLYMER,
                res_type=PROTEIN_UNK_RES_TYPE,
                input_id=DNA_RNA_LIGAND_INPUT_ID,
                asym_id=asym_id,
                sym_id=sym_id,
                entity_id=entity_id,
                atom_start=atom_index,
                atom_count=1,
            )
        )
    return [
        (bond.GetBeginAtom().GetProp("name"), bond.GetEndAtom().GetProp("name"))
        for bond in molecule.GetBonds()
    ]


def _ligand(
    entity: Mapping[str, Any],
    *,
    asym_id: int,
    sym_id: int,
    entity_id: int,
    tokens: list[Token],
    atoms: list[Atom],
    ccd: CCDStore,
    seed: int,
    bonded: bool,
) -> list[tuple[str, str]]:
    if entity.get("smiles"):
        return _smiles(
            str(entity["smiles"]),
            asym_id=asym_id,
            sym_id=sym_id,
            entity_id=entity_id,
            tokens=tokens,
            atoms=atoms,
            seed=seed,
        )
    component = str(entity["ccd"]).upper()
    records = ccd.atoms(component)
    if not records:
        raise ValueError(f"ESMFold2 CCD component {component!r} was not found")
    if bonded:
        leaving = ccd.leaving_atoms(component)
        records = [record for record in records if record[0] not in leaving]
    _append_atom_tokenized_residue(
        tokens=tokens,
        atoms=atoms,
        records=records,
        residue_index=0,
        residue_name=component,
        mol_type=MOL_TYPE_NONPOLYMER,
        asym_id=asym_id,
        sym_id=sym_id,
        entity_id=entity_id,
        space_uid=_next_space_uid(atoms),
        ccd=ccd,
    )
    return []


def _build_chains(
    document: Mapping[str, Any], *, ccd: CCDStore, seed: int
) -> tuple[list[Chain], list[Token], list[Atom]]:
    chains: list[Chain] = []
    tokens: list[Token] = []
    atoms: list[Atom] = []
    bonded_ids = {
        str(endpoint[0])
        for bond in document.get("bonds", ())
        for endpoint in bond
    }
    entity_ids: dict[tuple[object, ...], int] = {}
    entity_sym_counts: dict[int, int] = {}
    for entity_index, entity in enumerate(document["entities"]):
        kind = str(entity["type"])
        key = _entity_key(entity)
        entity_id = entity_ids.setdefault(key, len(entity_ids))
        for chain_id in _ids(entity):
            sym_id = entity_sym_counts.get(entity_id, 0)
            entity_sym_counts[entity_id] = sym_id + 1
            asym_id = len(chains)
            start = len(tokens)
            ligand_bonds: list[tuple[str, str]] = []
            common = dict(
                asym_id=asym_id,
                sym_id=sym_id,
                entity_id=entity_id,
                tokens=tokens,
                atoms=atoms,
            )
            if kind == "protein":
                _protein(entity, ccd=ccd, **common)
            elif kind in {"dna", "rna"}:
                _nucleic_acid(entity, kind=kind, ccd=ccd, **common)
            elif kind == "ligand":
                ligand_bonds = _ligand(
                    entity,
                    ccd=ccd,
                    seed=seed,
                    bonded=chain_id in bonded_ids,
                    **common,
                )
            else:
                raise ValueError(f"unsupported ESMFold2 entity type: {kind!r}")
            if len(tokens) == start:
                raise ValueError(f"ESMFold2 entity {chain_id!r} produced no tokens")
            chain_tokens = tokens[start:]
            chains.append(
                Chain(
                    chain_id=chain_id,
                    asym_id=asym_id,
                    entity_index=entity_index,
                    entity_id=entity_id,
                    sym_id=sym_id,
                    kind=kind,
                    sequence=None if kind == "ligand" else str(entity["sequence"]),
                    tokens=chain_tokens,
                    ligand_bonds=ligand_bonds,
                )
            )
    return chains, tokens, atoms


def _token_bonds(
    document: Mapping[str, Any],
    chains: Sequence[Chain],
    tokens: Sequence[Token],
    atoms: Sequence[Atom],
    ccd: CCDStore,
) -> np.ndarray:
    edges: set[tuple[int, int]] = set()

    def add(left: int, right: int) -> None:
        if left != right:
            edges.add((min(left, right), max(left, right)))

    residue_atoms: dict[tuple[int, int], list[Atom]] = defaultdict(list)
    for atom in atoms:
        token = tokens[atom.token_index]
        residue_atoms[(token.asym_id, token.residue_index)].append(atom)

    explicit = {
        (chain.asym_id, 0): chain.ligand_bonds
        for chain in chains
        if chain.ligand_bonds
    }
    for key, members in residue_atoms.items():
        token_members = [tokens[atom.token_index] for atom in members]
        if not any(
            token.mol_type == MOL_TYPE_NONPOLYMER
            or token.res_type == PROTEIN_UNK_RES_TYPE
            for token in token_members
        ):
            continue
        names = {atom.name: atom.token_index for atom in members}
        bonds = explicit.get(key)
        if bonds is None:
            bonds = ccd.bonds(token_members[0].residue_name)
        if bonds:
            for left, right in bonds:
                if left in names and right in names:
                    add(names[left], names[right])
        else:
            indices = sorted(set(names.values()))
            for left in indices:
                for right in indices:
                    add(left, right)

    chain_by_id = {chain.chain_id: chain for chain in chains}
    for bond in document.get("bonds", ()):
        resolved: list[int] = []
        for chain_id, residue_number, atom_name in bond:
            chain = chain_by_id[str(chain_id)]
            members = residue_atoms.get((chain.asym_id, int(residue_number) - 1), ())
            matching = [atom for atom in members if atom.name == str(atom_name)]
            if len(matching) != 1:
                raise ValueError(
                    "ESMFold2 covalent bond atom was not found uniquely: "
                    f"{chain_id}:{residue_number}:{atom_name}"
                )
            resolved.append(matching[0].token_index)
        add(resolved[0], resolved[1])

    protein_residues: dict[tuple[int, int], list[Token]] = defaultdict(list)
    for token in tokens:
        if token.mol_type == MOL_TYPE_PROTEIN:
            protein_residues[(token.asym_id, token.residue_index)].append(token)

    def backbone(members: Sequence[Token], name: str) -> int | None:
        if len(members) == 1 and members[0].res_type != PROTEIN_UNK_RES_TYPE:
            return members[0].token_index
        for token in members:
            for atom in atoms[token.atom_start : token.atom_start + token.atom_count]:
                if atom.name == name:
                    return token.token_index
        return members[0].token_index if members else None

    for (asym_id, residue), members in protein_residues.items():
        if not any(token.res_type == PROTEIN_UNK_RES_TYPE for token in members):
            continue
        previous = protein_residues.get((asym_id, residue - 1), ())
        following = protein_residues.get((asym_id, residue + 1), ())
        n_token = backbone(members, "N")
        c_token = backbone(members, "C")
        previous_c = backbone(previous, "C")
        following_n = backbone(following, "N")
        if n_token is not None and previous_c is not None:
            add(previous_c, n_token)
        if c_token is not None and following_n is not None:
            add(c_token, following_n)

    matrix = np.zeros((len(tokens), len(tokens), 1), dtype=np.float32)
    for left, right in edges:
        matrix[left, right, 0] = 1
        matrix[right, left, 0] = 1
    return matrix


def _representative_atoms(tokens: Sequence[Token], atoms: Sequence[Atom]) -> np.ndarray:
    by_token: dict[int, dict[str, int]] = defaultdict(dict)
    for atom in atoms:
        by_token[atom.token_index][atom.name] = atom.atom_index
    result = np.zeros(len(tokens), dtype=np.int64)
    for token in tokens:
        names = by_token[token.token_index]
        fallback = next(iter(names.values()), 0)
        if token.mol_type == MOL_TYPE_PROTEIN:
            selected = names.get("CB", names.get("CA", fallback))
        elif token.mol_type in {MOL_TYPE_DNA, MOL_TYPE_RNA}:
            if token.res_type in {27, 32}:
                selected = names.get("C1'", fallback)
            elif token.res_type in {23, 24, 28, 29}:
                selected = names.get("C4", names.get("C1'", fallback))
            else:
                selected = names.get("C2", names.get("C1'", fallback))
        else:
            selected = fallback
        result[token.token_index] = selected
    return result


def _msa(
    chains: Sequence[Chain],
    tokens: Sequence[Token],
    alignments: Mapping[int, Path],
    *,
    msa_depth: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from foldjax.models.esmfold2.data.paired_msa import (
        MSA,
        MSAEntry,
        construct_paired_msa,
        read_a3m,
    )

    chain_msas = {}
    chain_queries = {}
    for chain in chains:
        path = alignments.get(chain.entity_index)
        chain_queries[chain.asym_id] = np.asarray(
            [token.res_type for token in chain.tokens], dtype=np.int64
        )
        if chain.kind == "protein" and chain.sequence is not None:
            chain_msas[chain.asym_id] = (
                read_a3m(path, expected_columns=len(chain.sequence))
                if path is not None
                else MSA((MSAEntry("query", chain.sequence),))
            )
        else:
            chain_msas[chain.asym_id] = None
    max_seqs = 16384 if msa_depth is None else msa_depth
    msa, raw_deletions, _paired = construct_paired_msa(
        chain_msas,
        chain_queries,
        np.asarray([token.asym_id for token in tokens], dtype=np.int64),
        np.asarray([token.residue_index for token in tokens], dtype=np.int64),
        max_seqs=max_seqs,
    )
    for chain in chains:
        if chain_msas[chain.asym_id] is None:
            indices = [token.token_index for token in chain.tokens]
            msa[:, indices] = MSA_GAP_TOKEN_ID
            msa[0, indices] = chain_queries[chain.asym_id]
    deletion_value = np.arctan(raw_deletions / np.float32(3)) * np.float32(
        np.pi / 2
    )
    return msa, np.ones_like(msa, dtype=bool), raw_deletions > 0, deletion_value


def build_job_features(
    document: Mapping[str, Any],
    *,
    base_dir: str | Path,
    ccd_path: str | Path,
    seed: int,
    msa_depth: int | None = None,
) -> dict[str, np.ndarray]:
    """Build one validated common-schema job, including the batch axis."""

    base = Path(base_dir)
    alignments = {}
    for entity_index, entity in enumerate(document["entities"]):
        if entity.get("unpaired_msa"):
            candidate = Path(str(entity["unpaired_msa"]))
            alignments[entity_index] = (
                candidate if candidate.is_absolute() else base / candidate
            )

    ccd = get_ccd_store(ccd_path)
    chains, tokens, atoms = _build_chains(document, ccd=ccd, seed=seed)
    if not tokens or not atoms:
        raise ValueError("ESMFold2 job produced no tokens or atoms")
    n_atoms = max(ATOM_BLOCK, -(-len(atoms) // ATOM_BLOCK) * ATOM_BLOCK)

    ref_pos = np.zeros((n_atoms, 3), dtype=np.float32)
    ref_element = np.zeros(n_atoms, dtype=np.int64)
    ref_charge = np.zeros(n_atoms, dtype=np.int8)
    ref_atom_name_chars = np.zeros((n_atoms, 4), dtype=np.int64)
    ref_space_uid = np.zeros(n_atoms, dtype=np.int64)
    atom_mask = np.zeros(n_atoms, dtype=bool)
    atom_to_token = np.zeros(n_atoms, dtype=np.int64)
    for atom in atoms:
        index = atom.atom_index
        ref_pos[index] = atom.ref_pos
        ref_element[index] = ELEMENT_TO_ATOMIC_NUM.get(atom.element.upper(), 0)
        ref_charge[index] = atom.charge
        ref_atom_name_chars[index] = chemistry.encode_atom_name(atom.name)
        ref_space_uid[index] = atom.space_uid
        atom_mask[index] = True
        atom_to_token[index] = atom.token_index

    msa, msa_mask, has_deletion, deletion_value = _msa(
        chains, tokens, alignments, msa_depth=msa_depth
    )
    chain_id_by_asym = {chain.asym_id: chain.chain_id for chain in chains}
    chain_width = max(
        1, *(len(_text_bytes(value)) for value in chain_id_by_asym.values())
    )
    residue_width = max(
        1, *(len(_text_bytes(token.residue_name)) for token in tokens)
    )
    token_chain_id_chars = np.stack(
        [
            _encode_text(chain_id_by_asym[token.asym_id], width=chain_width)
            for token in tokens
        ]
    )
    token_residue_name_chars = np.stack(
        [_encode_text(token.residue_name, width=residue_width) for token in tokens]
    )
    values = {
        "token_index": np.arange(len(tokens), dtype=np.int64),
        "residue_index": np.asarray(
            [token.residue_index for token in tokens], np.int64
        ),
        "asym_id": np.asarray([token.asym_id for token in tokens], np.int64),
        "sym_id": np.asarray([token.sym_id for token in tokens], np.int64),
        "entity_id": np.asarray([token.entity_id for token in tokens], np.int64),
        "mol_type": np.asarray([token.mol_type for token in tokens], np.int64),
        "res_type": np.asarray([token.res_type for token in tokens], np.int64),
        "input_ids": np.asarray([token.input_id for token in tokens], np.int64),
        "token_bonds": _token_bonds(document, chains, tokens, atoms, ccd),
        "token_attention_mask": np.ones(len(tokens), dtype=bool),
        "ref_pos": ref_pos,
        "ref_element": ref_element,
        "ref_charge": ref_charge,
        "ref_atom_name_chars": ref_atom_name_chars,
        "ref_space_uid": ref_space_uid,
        "atom_attention_mask": atom_mask,
        "atom_to_token": atom_to_token,
        "distogram_atom_idx": _representative_atoms(tokens, atoms),
        "msa": msa,
        "msa_attention_mask": msa_mask,
        "has_deletion": has_deletion,
        "deletion_value": deletion_value,
        "deletion_mean": deletion_value.mean(axis=0),
        "token_chain_id_chars": token_chain_id_chars,
        "token_residue_name_chars": token_residue_name_chars,
    }
    return {name: value[None] for name, value in values.items()}


__all__ = ["OUTPUT_METADATA_FEATURES", "build_job_features"]
