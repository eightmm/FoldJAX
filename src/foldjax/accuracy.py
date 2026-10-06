"""Accuracy of a predicted structure against a deposited one.

`foldjax compare DIR --reference X.cif --metrics lddt,tm,dockq,lig_rmsd`
scores every structure of a finished run against one reference. Unlike the
confidence scores a run reports, these are measurements against an external
answer, so they *are* comparable across models -- as long as each prediction
is scored against the same reference with the same chain assignment rule.

Chains are assigned the way a homomer needs: every injective map between
predicted and reference chains of the same polymer kind whose paired residues
are at least 90% identical (over at least 20 residues, or the whole chain if
shorter) is tried, and the one with the lowest complex RMSD over representative
atoms (CA; C4' for nucleic acids) after one rigid fit wins. Residues are paired
by sequence index -- ``label_seq_id`` on the reference, and on the prediction
when it carries one (ESMFold2 writes the index in ``auth_seq_id``) -- with a
global sequence alignment as the fallback when the numbering does not agree.

Metrics:

- ``lddt_ca``: global CA-lDDT (C4' for nucleic acids) -- pairs whose reference
  distance is under 15 A, preserved within 0.5, 1, 2 and 4 A (strictly), the
  four fractions averaged, pair-weighted over the whole complex. A reference
  pair whose atom the prediction lacks counts as not preserved.
- ``lddt``: all-heavy-atom lDDT over polymer residues -- pairs of atoms in
  different residues under 15 A in the reference, preserved within each
  threshold (inclusive), only atoms present in both; a residue pair of
  different names contributes N, CA, C, O only. Symmetric side-chain atoms
  (Asp OD1/OD2, Glu OE1/OE2, Arg NH1/NH2, Phe and Tyr CD1/CD2 + CE1/CE2) are
  renamed per residue when that preserves more distances to non-symmetric
  atoms, as OpenStructure's lDDT scorer does.
- ``tm``: TM-score of the protein CA atoms on the fixed residue correspondence,
  normalized by the reference's protein residue count, with the TM-score
  program's superposition search (fragment seeds of L, L/2, ... 4 residues,
  iterative extension within d0_search). Equal to US-align ``-TMscore 1`` on
  the paired residues when the prediction covers every reference residue.
- ``rmsd_ca``: RMSD over the paired representative atoms after the rigid fit.
- ``dockq``: DockQ v2 (Mirabello and Wallner, Bioinformatics 40:btae586, 2024)
  through the ``DockQ`` package, per native protein-protein interface under
  the chain map DockQ prefers, and their mean. DockQ 2.1.3 pins ``numpy<2``
  and so cannot share FoldJAX's environment; it is run as an installed tool
  (`DOCKQ_INSTALL_HINT`) or imported when it can be.
- ``lig_rmsd``: heavy-atom RMSD of each predicted ligand after superposing the
  predicted protein on the reference pocket (reference CA atoms within 10 A
  of the reference ligand), minimized over the ligand's graph automorphisms
  (bonds inferred from coordinates, element-labelled) and over the reference's
  copies of the same ligand.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

METRICS = ("lddt", "lddt_ca", "tm", "rmsd_ca", "dockq", "lig_rmsd")
DEFAULT_METRICS = ("lddt", "lddt_ca", "tm", "rmsd_ca")
LDDT_RADIUS = 15.0
LDDT_THRESHOLDS = (0.5, 1.0, 2.0, 4.0)
POCKET_RADIUS = 10.0
ASSIGNMENT_LIMIT = 50_000
MIN_IDENTITY = 0.9
_WATER = frozenset({"HOH", "DOD", "WAT", "H2O"})
_BACKBONE = ("N", "CA", "C", "O")
SYMMETRIC = {
    "ASP": (("OD1", "OD2"),),
    "GLU": (("OE1", "OE2"),),
    "ARG": (("NH1", "NH2"),),
    "PHE": (("CD1", "CD2"), ("CE1", "CE2")),
    "TYR": (("CD1", "CD2"), ("CE1", "CE2")),
}
DOCKQ_INSTALL_HINT = (
    "DockQ is not available. DockQ 2.1.3 pins numpy<2, so it cannot be "
    "installed into FoldJAX's environment (numpy>=2.1); install it as a "
    "separate tool instead -- `uv tool install --python 3.12 DockQ==2.1.3` puts "
    "a `DockQ` executable on PATH -- or point FOLDJAX_DOCKQ at one"
)


class AccuracyError(ValueError):
    """The structures cannot be put in correspondence for this metric."""


@dataclass(slots=True)
class _Residue:
    key: int
    name: str
    letter: str
    atoms: dict[str, np.ndarray]


@dataclass(slots=True)
class _Chain:
    name: str
    kind: str
    residues: list[_Residue]

    @property
    def sequence(self) -> str:
        return "".join(residue.letter for residue in self.residues)


@dataclass(slots=True)
class _Ligand:
    chain: str
    name: str
    number: int
    atoms: dict[str, tuple[str, np.ndarray]]


@dataclass(slots=True)
class _Structure:
    path: Path
    chains: dict[str, _Chain]
    ligands: list[_Ligand] = field(default_factory=list)


def _representative(kind: str) -> str:
    return "C4'" if kind == "nucleic" else "CA"


def read_structure(path: str | os.PathLike[str], *, reference: bool = False) -> Any:
    """Polymer chains (heavy atoms, first model, first conformer) and ligands."""
    import gemmi

    path = Path(path)
    try:
        structure = gemmi.read_structure(str(path))
    except Exception as error:  # noqa: BLE001 - normalized into a domain error
        raise AccuracyError(f"cannot read {path}: {error}") from error
    if len(structure) == 0:
        raise AccuracyError(f"{path} has no model")
    structure.setup_entities()
    structure.remove_alternative_conformations()
    structure.remove_hydrogens()
    model = structure[0]
    use_label = reference or any(
        residue.label_seq is not None
        for chain in model
        for residue in chain
        if residue.entity_type == gemmi.EntityType.Polymer
    )
    chains: dict[str, _Chain] = {}
    ligands: list[_Ligand] = []
    for chain in model:
        residues: list[_Residue] = []
        kinds: set[str] = set()
        seen: set[int] = set()
        for residue in chain:
            if residue.name in _WATER:
                continue
            info = gemmi.find_tabulated_residue(residue.name)
            if residue.entity_type != gemmi.EntityType.Polymer:
                ligands.append(
                    _Ligand(
                        chain.name,
                        residue.name,
                        int(residue.seqid.num),
                        {
                            atom.name: (
                                atom.element.name.upper(),
                                np.array(atom.pos.tolist(), dtype=float),
                            )
                            for atom in residue
                        },
                    )
                )
                continue
            if info is not None and info.is_amino_acid():
                kind = "protein"
            elif info is not None and info.is_nucleic_acid():
                kind = "nucleic"
            else:
                kind = "protein" if residue.find_atom("CA", "*") else "nucleic"
            key = (
                residue.label_seq
                if use_label and residue.label_seq is not None
                else residue.seqid.num
            )
            if key is None or key in seen:
                continue
            seen.add(int(key))
            kinds.add(kind)
            letter = (info.one_letter_code.strip().upper() if info else "") or "X"
            atoms: dict[str, np.ndarray] = {}
            for atom in residue:
                atoms.setdefault(atom.name, np.array(atom.pos.tolist(), dtype=float))
            residues.append(_Residue(int(key), residue.name, letter, atoms))
        if residues:
            name = chain.name
            while name in chains:
                name += "'"
            kind = "protein" if "protein" in kinds else "nucleic"
            chains[name] = _Chain(name, kind, sorted(residues, key=lambda r: r.key))
    return _Structure(path=path, chains=chains, ligands=ligands)


def _pair_residues(
    predicted: _Chain, reference: _Chain
) -> tuple[list[tuple[int, int]], float]:
    """Residue index pairs (pred, ref) and their name identity."""
    by_key = {residue.key: k for k, residue in enumerate(predicted.residues)}
    keyed = [
        (by_key[residue.key], j)
        for j, residue in enumerate(reference.residues)
        if residue.key in by_key
    ]

    def identity(pairs: Sequence[tuple[int, int]]) -> float:
        if not pairs:
            return 0.0
        same = sum(
            predicted.residues[i].name == reference.residues[j].name for i, j in pairs
        )
        return same / len(pairs)

    if keyed and identity(keyed) >= MIN_IDENTITY:
        return keyed, identity(keyed)
    from foldjax.alignment import _align_sequence

    aligned = list(_align_sequence(predicted.sequence, reference.sequence))
    if identity(aligned) * len(aligned) > identity(keyed) * len(keyed):
        return aligned, identity(aligned)
    return keyed, identity(keyed)


def kabsch(mobile: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rotation and translation with ``mobile @ rotation + translation ~ target``."""
    mobile_center = mobile.mean(axis=0)
    target_center = target.mean(axis=0)
    left, _, right = np.linalg.svd(
        (mobile - mobile_center).T @ (target - target_center)
    )
    correction = np.eye(3)
    correction[-1, -1] = math.copysign(1.0, np.linalg.det(left @ right))
    rotation = left @ correction @ right
    return rotation, target_center - mobile_center @ rotation


def rmsd_after_fit(mobile: np.ndarray, target: np.ndarray) -> float:
    rotation, translation = kabsch(mobile, target)
    moved = mobile @ rotation + translation
    return float(np.sqrt(((moved - target) ** 2).sum(axis=1).mean()))


@dataclass(slots=True)
class Assignment:
    #: predicted chain -> reference chain
    chain_map: dict[str, str]
    pairs: dict[tuple[str, str], list[tuple[int, int]]]
    rmsd_ca: float
    tried: int
    truncated: bool


def _representative_points(
    predicted: _Structure,
    reference: _Structure,
    pairs: Mapping[tuple[str, str], Sequence[tuple[int, int]]],
    *,
    kinds: Sequence[str] = ("protein", "nucleic"),
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    mobile, target, names = [], [], []
    for (p, r), indices in pairs.items():
        chain = reference.chains[r]
        if chain.kind not in kinds:
            continue
        atom = _representative(chain.kind)
        for i, j in indices:
            source = predicted.chains[p].residues[i].atoms.get(atom)
            dest = chain.residues[j].atoms.get(atom)
            if source is None or dest is None:
                continue
            mobile.append(source)
            target.append(dest)
            names.append(chain.residues[j].name)
    return (
        np.asarray(mobile, dtype=float).reshape(-1, 3),
        np.asarray(target, dtype=float).reshape(-1, 3),
        names,
    )


def assign_chains(predicted: _Structure, reference: _Structure) -> Assignment:
    """The homomer-aware chain map with the lowest complex representative RMSD."""
    options: dict[str, list[str]] = {}
    paired: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for r_name, r_chain in reference.chains.items():
        options[r_name] = []
        for p_name, p_chain in predicted.chains.items():
            if p_chain.kind != r_chain.kind:
                continue
            pairs, identity = _pair_residues(p_chain, r_chain)
            needed = min(20, len(r_chain.residues), len(p_chain.residues))
            if len(pairs) >= needed and identity >= MIN_IDENTITY:
                options[r_name].append(p_name)
                paired[(p_name, r_name)] = pairs
    keys = sorted(name for name in options if options[name])
    if not keys:
        raise AccuracyError(
            "no predicted chain matches any reference chain (90% identity over "
            "the paired residues)"
        )
    best: tuple[float, dict[str, str]] | None = None
    tried = 0
    truncated = False

    def walk(index: int, used: set[str], current: dict[str, str]) -> None:
        nonlocal best, tried, truncated
        if tried >= ASSIGNMENT_LIMIT:
            truncated = True
            return
        if index == len(keys):
            tried += 1
            chosen = {(p, r): paired[(p, r)] for r, p in current.items()}
            mobile, target, _ = _representative_points(predicted, reference, chosen)
            if len(mobile) < 3:
                return
            value = rmsd_after_fit(mobile, target)
            if best is None or value < best[0]:
                best = (value, dict(current))
            return
        reference_chain = keys[index]
        candidates = [name for name in options[reference_chain] if name not in used]
        if not candidates:
            walk(index + 1, used, current)
            return
        for name in candidates:
            current[reference_chain] = name
            used.add(name)
            walk(index + 1, used, current)
            used.discard(name)
            del current[reference_chain]

    walk(0, set(), {})
    if best is None:
        raise AccuracyError("no chain assignment has three paired atoms to fit")
    rmsd, by_reference = best
    chain_map = {p: r for r, p in sorted(by_reference.items())}
    pairs = {(p, r): paired[(p, r)] for p, r in chain_map.items()}
    return Assignment(chain_map, pairs, rmsd, tried, truncated)


# --------------------------------------------------------------------- lDDT


def lddt_representative(
    predicted: _Structure, reference: _Structure, assignment: Assignment
) -> float | None:
    """Global CA-lDDT (C4' for nucleic acids) over the whole reference."""
    ref_points, model_points = [], []
    lookup = {r: p for p, r in assignment.chain_map.items()}
    for r_name, r_chain in reference.chains.items():
        atom = _representative(r_chain.kind)
        p_name = lookup.get(r_name)
        mapping = (
            {j: i for i, j in assignment.pairs[(p_name, r_name)]} if p_name else {}
        )
        for j, residue in enumerate(r_chain.residues):
            position = residue.atoms.get(atom)
            if position is None:
                continue
            ref_points.append(position)
            i = mapping.get(j)
            source = (
                predicted.chains[p_name].residues[i].atoms.get(atom)
                if p_name is not None and i is not None
                else None
            )
            model_points.append(source if source is not None else np.full(3, np.nan))
    if len(ref_points) < 2:
        return None
    q = np.asarray(ref_points)
    p = np.asarray(model_points)
    preserved = np.zeros(len(LDDT_THRESHOLDS))
    total = 0
    for start in range(0, len(q), 1024):
        stop = min(len(q), start + 1024)
        dq = np.linalg.norm(q[start:stop, None, :] - q[None, :, :], axis=-1)
        dp = np.linalg.norm(p[start:stop, None, :] - p[None, :, :], axis=-1)
        mask = dq < LDDT_RADIUS
        mask[np.arange(stop - start), np.arange(start, stop)] = False
        diff = np.abs(dp - dq)[mask]
        total += diff.size
        for k, threshold in enumerate(LDDT_THRESHOLDS):
            # NaN (a missing model atom) compares False: not preserved.
            preserved[k] += np.count_nonzero(diff < threshold)
    return float((preserved / max(total, 1)).mean())


def _preserved(model_d: np.ndarray, reference_d: np.ndarray) -> np.ndarray:
    diff = np.abs(model_d - reference_d)
    return sum((diff <= t).astype(np.int64) for t in LDDT_THRESHOLDS)


def lddt_all_atom(
    predicted: _Structure, reference: _Structure, assignment: Assignment
) -> dict[str, Any]:
    """All-heavy-atom lDDT with symmetric side-chain naming resolved."""
    from scipy.spatial import cKDTree

    ref_xyz, model_xyz, resid, resname, aname = [], [], [], [], []
    residue_index = 0
    for (p_name, r_name), pairs in assignment.pairs.items():
        p_chain = predicted.chains[p_name]
        r_chain = reference.chains[r_name]
        for i, j in pairs:
            source = p_chain.residues[i]
            target = r_chain.residues[j]
            if source.name == target.name:
                names = [name for name in target.atoms if name in source.atoms]
            else:
                names = [
                    name
                    for name in _BACKBONE
                    if name in target.atoms and name in source.atoms
                ]
            for name in names:
                ref_xyz.append(target.atoms[name])
                model_xyz.append(source.atoms[name])
                resid.append(residue_index)
                resname.append(target.name if source.name == target.name else "")
                aname.append(name)
            residue_index += 1
    if len(ref_xyz) < 2:
        raise AccuracyError("no common atoms for all-atom lDDT")
    ref = np.asarray(ref_xyz)
    model = np.asarray(model_xyz)
    residues = np.asarray(resid)
    pairs_ij = cKDTree(ref).query_pairs(LDDT_RADIUS, output_type="ndarray")
    if len(pairs_ij) == 0:
        raise AccuracyError("no reference contacts within 15 A")
    i, j = pairs_ij[:, 0], pairs_ij[:, 1]
    keep = residues[i] != residues[j]
    i, j = i[keep], j[keep]
    reference_d = np.linalg.norm(ref[i] - ref[j], axis=1)
    keep = reference_d < LDDT_RADIUS
    i, j, reference_d = i[keep], j[keep], reference_d[keep]

    index = {(r, a): k for k, (r, a) in enumerate(zip(resid, aname, strict=True))}
    symmetric = np.zeros(len(aname), dtype=bool)
    groups: list[list[tuple[int, int]]] = []
    seen: set[int] = set()
    for r, name in zip(resid, resname, strict=True):
        if name not in SYMMETRIC or r in seen:
            continue
        seen.add(r)
        group = []
        for a, b in SYMMETRIC[name]:
            ia, ib = index.get((r, a)), index.get((r, b))
            for x in (ia, ib):
                if x is not None:
                    symmetric[x] = True
            if ia is not None and ib is not None:
                group.append((ia, ib))
        if group:
            groups.append(group)
    resolved = model.copy()
    swapped = 0
    if groups:
        group_of = np.full(len(model), -1)
        partner = np.arange(len(model))
        for g, group in enumerate(groups):
            for a, b in group:
                group_of[a] = group_of[b] = g
                partner[a], partner[b] = b, a
        one = symmetric[i] & ~symmetric[j]
        two = symmetric[j] & ~symmetric[i]
        s = np.concatenate([i[one], j[two]])
        o = np.concatenate([j[one], i[two]])
        d = np.concatenate([reference_d[one], reference_d[two]])
        keep = group_of[s] >= 0
        s, o, d = s[keep], o[keep], d[keep]
        as_is = _preserved(np.linalg.norm(model[s] - model[o], axis=1), d)
        flipped = _preserved(np.linalg.norm(model[partner[s]] - model[o], axis=1), d)
        score_as_is = np.bincount(group_of[s], weights=as_is, minlength=len(groups))
        score_flip = np.bincount(group_of[s], weights=flipped, minlength=len(groups))
        for g in np.nonzero(score_flip > score_as_is)[0]:
            for a, b in groups[g]:
                resolved[[a, b]] = resolved[[b, a]]
            swapped += 1
    total = 4 * len(i)
    value = _preserved(np.linalg.norm(resolved[i] - resolved[j], axis=1), reference_d)
    return {
        "lddt": float(value.sum() / total),
        "atoms": int(len(ref)),
        "pairs": int(len(i)),
        "symmetry_swaps": swapped,
    }


# ----------------------------------------------------------------- TM-score


def _tm_d0(length: float) -> float:
    d0 = 1.24 * (length - 15) ** (1.0 / 3.0) - 1.8 if length > 21 else 0.5
    return max(d0, 0.5)


def tm_score(
    mobile: np.ndarray, target: np.ndarray, *, length: int | None = None
) -> float:
    """TM-score of ``mobile`` on ``target`` over a fixed correspondence.

    The TM-score program's search (``TMscore8_search`` with a step of one and
    the full score sum): Kabsch fits seeded on every fragment of L, L/2, ...
    down to 4 paired residues, each extended for up to 20 iterations over the
    residues within d0_search, keeping the best score. ``length`` normalizes
    (the reference's residue count); it defaults to the paired count.
    """
    mobile = np.asarray(mobile, dtype=float)
    target = np.asarray(target, dtype=float)
    n = len(mobile)
    if n < 3:
        return 0.0
    norm = float(length if length is not None else n)
    d0 = _tm_d0(norm)
    d0_search = min(max(d0, 4.5), 8.0)
    d02 = d0 * d0

    def score(rotation: np.ndarray, translation: np.ndarray, cutoff: float):
        moved = mobile @ rotation + translation
        dist2 = ((moved - target) ** 2).sum(axis=1)
        total = float((1.0 / (1.0 + dist2 / d02)).sum()) / norm
        threshold = cutoff
        selected = np.flatnonzero(dist2 < threshold * threshold)
        increment = 0
        while len(selected) < 3 and n > 3:
            increment += 1
            threshold = cutoff + increment * 0.5
            selected = np.flatnonzero(dist2 < threshold * threshold)
        return total, selected

    minimum = min(4, n)
    lengths = []
    for k in range(5):
        fragment = int(n / 2**k)
        if fragment <= minimum:
            lengths.append(minimum)
            break
        lengths.append(fragment)
    else:
        lengths.append(minimum)
    best = -1.0
    for fragment in lengths:
        last = n - fragment
        start = 0
        while True:
            chosen = np.arange(start, start + fragment)
            rotation, translation = kabsch(mobile[chosen], target[chosen])
            value, selected = score(rotation, translation, d0_search - 1)
            best = max(best, value)
            for _ in range(20):
                previous = selected
                rotation, translation = kabsch(mobile[previous], target[previous])
                value, selected = score(rotation, translation, d0_search + 1)
                best = max(best, value)
                if len(selected) == len(previous) and np.array_equal(
                    selected, previous
                ):
                    break
            if start < last:
                start = min(start + 1, last)
            else:
                break
    return best


# ------------------------------------------------------------------ ligands


def _ligand_mol(atoms: Mapping[str, tuple[str, np.ndarray]]):
    from rdkit import Chem
    from rdkit.Chem import rdDetermineBonds
    from rdkit.Geometry import Point3D

    names = list(atoms)
    mol = Chem.RWMol()
    conformer = Chem.Conformer(len(names))
    for k, name in enumerate(names):
        element = atoms[name][0].capitalize()
        atom = Chem.Atom(element)
        atom.SetNoImplicit(True)
        mol.AddAtom(atom)
        conformer.SetAtomPosition(k, Point3D(*map(float, atoms[name][1])))
    mol.AddConformer(conformer, assignId=True)
    rdDetermineBonds.DetermineConnectivity(mol)
    mol.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(mol)
    return mol.GetMol(), names


def _composition(atoms: Mapping[str, tuple[str, np.ndarray]]) -> tuple:
    return tuple(sorted(element for element, _ in atoms.values()))


def _ligand_rmsd(
    predicted: _Ligand,
    reference: _Ligand,
    move,
    *,
    max_matches: int = 1000,
) -> tuple[float, int, str] | None:
    """(RMSD, atoms, how matched) of ``predicted`` (already moved) vs ``reference``."""
    ref_mol, ref_names = _ligand_mol(reference.atoms)
    ref_xyz = np.asarray([reference.atoms[name][1] for name in ref_names])
    pred_xyz = move(np.asarray([atom[1] for atom in predicted.atoms.values()]))
    pred_names = list(predicted.atoms)
    if set(ref_names) <= set(pred_names) and all(
        predicted.atoms[name][0] == reference.atoms[name][0] for name in ref_names
    ):
        position = {name: k for k, name in enumerate(pred_names)}
        automorphisms = ref_mol.GetSubstructMatches(
            ref_mol, uniquify=False, useChirality=False, maxMatches=max_matches
        ) or (tuple(range(len(ref_names))),)
        best = None
        for perm in automorphisms:
            # Reference atom k corresponds to the predicted atom named like
            # reference atom perm[k].
            order = [position[ref_names[perm[k]]] for k in range(len(ref_names))]
            value = float(
                np.sqrt(((pred_xyz[order] - ref_xyz) ** 2).sum(axis=1).mean())
            )
            best = value if best is None else min(best, value)
        return best, len(ref_names), "name"
    pred_mol, _ = _ligand_mol(predicted.atoms)
    matches = pred_mol.GetSubstructMatches(
        ref_mol, uniquify=False, useChirality=False, maxMatches=max_matches
    )
    if not matches:
        return None
    best = None
    for match in matches:
        value = float(
            np.sqrt(((pred_xyz[list(match)] - ref_xyz) ** 2).sum(axis=1).mean())
        )
        best = value if best is None else min(best, value)
    return best, len(ref_names), "graph"


def ligand_rmsds(
    predicted: _Structure, reference: _Structure, assignment: Assignment
) -> list[dict[str, Any]]:
    """One record per predicted ligand residue with at least two heavy atoms."""
    lookup = {r: p for p, r in assignment.chain_map.items()}
    out = []
    for ligand in predicted.ligands:
        if len(ligand.atoms) < 2:
            continue
        record: dict[str, Any] = {
            "chain": ligand.chain,
            "ligand": ligand.name,
            "residue": ligand.number,
        }
        copies = [item for item in reference.ligands if item.name == ligand.name]
        if not copies:
            copies = [
                item
                for item in reference.ligands
                if _composition(item.atoms) == _composition(ligand.atoms)
            ]
        if not copies:
            record["error"] = "no reference ligand of the same name or composition"
            out.append(record)
            continue
        best: dict[str, Any] | None = None
        for copy in copies:
            ligand_xyz = np.asarray([atom[1] for atom in copy.atoms.values()])
            mobile, target = [], []
            for r_name, r_chain in reference.chains.items():
                if r_chain.kind != "protein" or r_name not in lookup:
                    continue
                p_chain = predicted.chains[lookup[r_name]]
                for i, j in assignment.pairs[(lookup[r_name], r_name)]:
                    ca_ref = r_chain.residues[j].atoms.get("CA")
                    ca_pred = p_chain.residues[i].atoms.get("CA")
                    if ca_ref is None or ca_pred is None:
                        continue
                    if (
                        np.linalg.norm(ligand_xyz - ca_ref, axis=1).min()
                        <= POCKET_RADIUS
                    ):
                        mobile.append(ca_pred)
                        target.append(ca_ref)
            if len(mobile) < 3:
                continue
            rotation, translation = kabsch(np.asarray(mobile), np.asarray(target))

            def move(xyz, rotation=rotation, translation=translation):
                return xyz @ rotation + translation

            result = _ligand_rmsd(ligand, copy, move)
            if result is None:
                continue
            value, atoms, how = result
            candidate = {
                "lig_rmsd": value,
                "atoms": atoms,
                "atom_map": how,
                "pocket_ca": len(mobile),
                "reference": f"{copy.chain}/{copy.name}{copy.number}",
            }
            if best is None or (candidate["atoms"], -value) > (
                best["atoms"],
                -best["lig_rmsd"],
            ):
                best = candidate
        if best is None:
            record["error"] = (
                "no reference copy with a 3-CA pocket and an atom correspondence"
            )
        else:
            record.update(best)
        out.append(record)
    return out


# -------------------------------------------------------------------- DockQ


def _dockq_executable() -> str | None:
    configured = os.environ.get("FOLDJAX_DOCKQ")
    if configured:
        return configured
    return shutil.which("DockQ")


def _protein_only(source: Path, target: Path, chains: set[str] | None) -> Path:
    """First model, no altlocs or hydrogens, amino-acid residues only.

    DockQ's mmCIF reader needs occupancy and would read a deposit's ligands
    and waters as chain residues.
    """
    import gemmi

    structure = gemmi.read_structure(str(source))
    structure.remove_alternative_conformations()
    structure.remove_hydrogens()
    while len(structure) > 1:
        del structure[len(structure) - 1]
    model = structure[0]
    for k in reversed(range(len(model))):
        chain = model[k]
        if chains is not None and chain.name not in chains:
            del model[k]
            continue
        for r in reversed(range(len(chain))):
            info = gemmi.find_tabulated_residue(chain[r].name)
            if info is None or not info.is_amino_acid():
                del chain[r]
    structure.remove_empty_chains()
    structure.setup_entities()
    structure.make_mmcif_document().write_file(str(target))
    return target


def dockq(
    predicted: str | os.PathLike[str],
    reference: str | os.PathLike[str],
    *,
    chain_map: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """DockQ v2 per native protein interface, and their mean.

    With ``chain_map`` (predicted -> reference) DockQ scores that map; without
    it, DockQ searches the map that maximizes the summed DockQ itself.
    """
    executable = _dockq_executable()
    try:
        import DockQ.DockQ  # type: ignore  # noqa: F401

        in_process = True
    except ImportError:
        in_process = False
    if executable is None and not in_process:
        raise ModuleNotFoundError(DOCKQ_INSTALL_HINT, name="DockQ")
    with tempfile.TemporaryDirectory(prefix="foldjax-dockq-") as scratch:
        model = _protein_only(Path(predicted), Path(scratch) / "model.cif", None)
        native = _protein_only(Path(reference), Path(scratch) / "native.cif", None)
        output = Path(scratch) / "dockq.json"
        command = [
            executable or "DockQ",
            str(model),
            str(native),
            "--short",
            "--json",
            str(output),
        ]
        if chain_map:
            pairs = sorted(chain_map.items(), key=lambda item: item[1])
            command += [
                "--mapping",
                "".join(p for p, _ in pairs) + ":" + "".join(r for _, r in pairs),
            ]
        if executable is not None:
            completed = subprocess.run(
                command, capture_output=True, text=True, check=False
            )
            if completed.returncode != 0 or not output.is_file():
                raise AccuracyError(
                    "DockQ failed: "
                    + (completed.stderr.strip() or completed.stdout.strip())[-500:]
                )
        else:  # pragma: no cover - only where DockQ imports alongside numpy>=2
            import sys

            from DockQ.DockQ import main as dockq_main  # type: ignore

            saved = sys.argv
            sys.argv = command
            try:
                dockq_main()
            finally:
                sys.argv = saved
        document = json.loads(output.read_text())
    interfaces = document.get("best_result") or {}
    values = {
        name: {
            "dockq": float(entry["DockQ"]),
            "fnat": float(entry["fnat"]),
            "irmsd": float(entry["iRMSD"]),
            "lrmsd": float(entry["LRMSD"]),
        }
        for name, entry in interfaces.items()
    }
    if not values:
        raise AccuracyError("DockQ found no native interface")
    best_map = document.get("best_mapping") or document.get("best_mapping_str")
    return {
        "dockq": float(np.mean([item["dockq"] for item in values.values()])),
        "interfaces": values,
        "chain_map": best_map,
    }


# -------------------------------------------------------------------- driver


def parse_metrics(text: str | Sequence[str] | None) -> tuple[str, ...]:
    if text is None:
        return DEFAULT_METRICS
    items = text.split(",") if isinstance(text, str) else list(text)
    chosen = []
    for item in items:
        name = item.strip().lower()
        if not name:
            continue
        if name == "all":
            chosen.extend(METRICS)
            continue
        if name not in METRICS:
            raise ValueError(
                f"unknown metric {name!r}; choose from {', '.join(METRICS)}"
            )
        chosen.append(name)
    return tuple(dict.fromkeys(chosen)) or DEFAULT_METRICS


def score_structure(
    predicted: str | os.PathLike[str],
    reference: str | os.PathLike[str],
    *,
    metrics: Sequence[str] = DEFAULT_METRICS,
    reference_parsed: Any = None,
) -> dict[str, Any]:
    """Every requested metric of one predicted structure, with any per-metric error.

    One failing metric (DockQ absent, no ligand) is recorded under
    ``errors`` and does not hide the others.
    """
    pred = read_structure(predicted)
    ref = reference_parsed or read_structure(reference, reference=True)
    assignment = assign_chains(pred, ref)
    out: dict[str, Any] = {
        "chain_map": assignment.chain_map,
        "assignments_tried": assignment.tried,
        "assignment_truncated": assignment.truncated,
        "errors": {},
    }
    mobile, target, _ = _representative_points(pred, ref, assignment.pairs)
    out["paired_residues"] = int(len(mobile))
    if "rmsd_ca" in metrics:
        out["rmsd_ca"] = assignment.rmsd_ca
    if "lddt_ca" in metrics:
        out["lddt_ca"] = lddt_representative(pred, ref, assignment)
    if "lddt" in metrics:
        try:
            result = lddt_all_atom(pred, ref, assignment)
            out["lddt"] = result["lddt"]
            out["lddt_atoms"] = result["atoms"]
        except AccuracyError as error:
            out["errors"]["lddt"] = str(error)
    if "tm" in metrics:
        p_ca, r_ca, _ = _representative_points(
            pred, ref, assignment.pairs, kinds=("protein",)
        )
        length = sum(
            1
            for chain in ref.chains.values()
            if chain.kind == "protein"
            for residue in chain.residues
            if "CA" in residue.atoms
        )
        if len(p_ca) >= 3 and length:
            out["tm"] = tm_score(p_ca, r_ca, length=length)
        else:
            out["errors"]["tm"] = "no paired protein CA atoms"
    if "lig_rmsd" in metrics:
        ligands = ligand_rmsds(pred, ref, assignment)
        out["ligands"] = ligands
        scored = [item for item in ligands if "lig_rmsd" in item]
        out["lig_rmsd"] = scored[0]["lig_rmsd"] if scored else None
        if not scored:
            out["errors"]["lig_rmsd"] = (
                ligands[0]["error"] if ligands else "the prediction has no ligand"
            )
    if "dockq" in metrics:
        protein_chains = [c for c in ref.chains.values() if c.kind == "protein"]
        if len(protein_chains) < 2:
            out["errors"]["dockq"] = "the reference has fewer than two protein chains"
        else:
            try:
                result = dockq(predicted, reference)
                out["dockq"] = result["dockq"]
                out["dockq_interfaces"] = result["interfaces"]
                out["dockq_chain_map"] = result["chain_map"]
            except (ModuleNotFoundError, AccuracyError, OSError) as error:
                out["errors"]["dockq"] = str(error)
    if not out["errors"]:
        out["errors"] = None
    return out


def describe() -> dict[str, Any]:
    """The definitions written beside the numbers."""
    return {
        "chain_assignment": (
            "injective map over same-kind chains with >=90% residue identity; "
            "lowest complex RMSD over CA/C4' after one rigid fit"
        ),
        "lddt": "all heavy polymer atoms, inter-residue pairs <15 A, thresholds "
        "0.5/1/2/4 A inclusive, symmetric side chains resolved",
        "lddt_ca": "CA (C4' nucleic) pairs <15 A, thresholds 0.5/1/2/4 A strict",
        "tm": "protein CA, fixed correspondence, TM-score search, normalized by "
        "the reference protein residue count",
        "rmsd_ca": "CA/C4' RMSD after one rigid fit under the chosen chain map",
        "dockq": "DockQ v2 (Mirabello & Wallner 2024), mean over native "
        "protein-protein interfaces, DockQ's own chain map",
        "lig_rmsd": "ligand heavy-atom RMSD after a pocket CA fit (10 A), "
        "minimized over graph automorphisms and reference copies",
        "comparable_across_models": True,
    }


__all__ = [
    "DEFAULT_METRICS",
    "METRICS",
    "AccuracyError",
    "assign_chains",
    "describe",
    "dockq",
    "lddt_all_atom",
    "lddt_representative",
    "ligand_rmsds",
    "parse_metrics",
    "read_structure",
    "score_structure",
    "tm_score",
]
