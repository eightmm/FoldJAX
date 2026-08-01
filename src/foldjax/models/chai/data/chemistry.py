"""Native chemistry helpers for Chai public inference inputs."""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from foldjax.models.chai.bridge.conformer_io import ConformerData


@dataclass(frozen=True)
class GlycosidicBond:
    src_sugar_index: int
    dst_sugar_index: int
    src_atom: int
    dst_atom: int

    @property
    def src_atom_name(self) -> str:
        return f"O{self.src_atom}"

    @property
    def dst_atom_name(self) -> str:
        return f"C{self.dst_atom}"


def parse_glycan(glycan: str) -> tuple[list[str], list[GlycosidicBond]]:
    """Parse Chai's manual glycan grammar and preserve its residue ordering."""
    glycan = glycan.strip()
    if not glycan:
        raise ValueError("manual glycan must contain at least one sugar")
    sugars: list[str] = []
    parents: list[int] = []
    bonds: list[GlycosidicBond] = []
    open_count = 0
    closed_count = 0
    index = 0
    while index < len(glycan):
        char = glycan[index]
        if char == " ":
            index += 1
        elif char == "(":
            if not parents:
                raise ValueError(f"Invalid glycan string: {glycan}")
            open_count += 1
            index += 1
        elif char == ")":
            if not parents or closed_count >= open_count:
                raise ValueError(f"Invalid glycan string: {glycan}")
            parents.pop()
            closed_count += 1
            index += 1
        else:
            chunk = glycan[index : index + 3]
            if re.fullmatch(r"[1-6]-[1-6]", chunk):
                if not parents:
                    raise ValueError(f"Invalid glycan string: {glycan}")
                source, destination = chunk.split("-")
                bonds.append(
                    GlycosidicBond(
                        src_sugar_index=parents[-1],
                        dst_sugar_index=len(sugars),
                        src_atom=int(source),
                        dst_atom=int(destination),
                    )
                )
                index += 3
            elif re.fullmatch(r"[0-9A-Z]{3}", chunk):
                sugars.append(chunk)
                parents.append(len(sugars) - 1)
                index += 3
            else:
                raise ValueError(f"Invalid glycan string: {glycan}")
    if not sugars or open_count != closed_count:
        raise ValueError(f"Invalid glycan string: {glycan}")
    if any(bond.dst_sugar_index >= len(sugars) for bond in bonds):
        raise ValueError(f"Invalid glycan string: {glycan}")
    return sugars, bonds


@dataclass(frozen=True)
class CovalentBond:
    chain_a: str
    residue_a: str
    atom_a: str
    chain_b: str
    residue_b: str
    atom_b: str


def _parse_residue_atom(value: str) -> tuple[str, str]:
    if value.endswith("@"):
        raise ValueError(f"Invalid residue index: {value}")
    parts = value.split("@")
    if len(parts) == 1:
        parts.append("")
    if len(parts) != 2 or not any(parts):
        raise ValueError(f"Invalid residue index: {value}")
    return parts[0], parts[1]


def parse_covalent_bonds_csv(path: str | Path) -> list[CovalentBond]:
    """Read the covalent subset of Chai's public restraints CSV contract."""
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = csv.DictReader(handle)
        required = {"chainA", "res_idxA", "chainB", "res_idxB", "connection_type"}
        if rows.fieldnames is None or not required.issubset(rows.fieldnames):
            raise ValueError("covalent restraints CSV is missing required columns")
        output = []
        for row in rows:
            if row["connection_type"].strip().lower() != "covalent":
                continue
            residue_a, atom_a = _parse_residue_atom(row["res_idxA"].strip())
            residue_b, atom_b = _parse_residue_atom(row["res_idxB"].strip())
            chain_a = row["chainA"].strip()
            chain_b = row["chainB"].strip()
            if not chain_a or not chain_b or not atom_a or not atom_b:
                raise ValueError("covalent bonds require two chains and two atom names")
            output.append(
                CovalentBond(chain_a, residue_a, atom_a, chain_b, residue_b, atom_b)
            )
    return output


def generate_ligand_conformer(smiles: str) -> ConformerData:
    """Generate Chai's deterministic ETKDGv3 heavy-atom reference conformer."""
    try:
        from rdkit import Chem
        from rdkit.Chem import rdDistGeom
    except ImportError as error:  # pragma: no cover - installation contract
        raise RuntimeError("RDKit is required for ligand SMILES inputs") from error

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    molecule = Chem.AddHs(molecule)
    params = rdDistGeom.ETKDGv3()
    params.useSmallRingTorsions = True
    params.randomSeed = 123
    params.enforceChirality = True
    params.maxIterations = 100
    params.useRandomCoords = True
    params.numThreads = -1
    conformer_ids = rdDistGeom.EmbedMultipleConfs(molecule, numConfs=1, params=params)
    if not conformer_ids:
        raise ValueError(f"RDKit could not embed SMILES: {smiles}")
    molecule = Chem.RemoveAllHs(molecule)

    names: list[str] = []
    element_counts: dict[str, int] = defaultdict(int)
    for atom in molecule.GetAtoms():
        symbol = atom.GetSymbol()
        element_counts[symbol] += 1
        names.append(f"{symbol}{element_counts[symbol]}".upper())
    positions = np.asarray(molecule.GetConformer().GetPositions(), np.float32)
    elements = np.asarray(
        [atom.GetAtomicNum() for atom in molecule.GetAtoms()], np.int32
    )
    charges = np.asarray(
        [atom.GetFormalCharge() for atom in molecule.GetAtoms()], np.int32
    )
    bonds = tuple(
        (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) for bond in molecule.GetBonds()
    )
    matches = molecule.GetSubstructMatches(
        molecule, uniquify=False, maxMatches=1000, useChirality=False
    ) or (tuple(range(molecule.GetNumAtoms())),)
    symmetries = np.stack([np.asarray(match, np.int64) for match in matches], axis=-1)
    return ConformerData(positions, elements, charges, tuple(names), bonds, symmetries)
