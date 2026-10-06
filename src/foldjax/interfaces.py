"""Chain-pair interface scores derived from a finished run's PAE and structure.

`foldjax interfaces DIR` reads each sample's mmCIF and ``confidence_full.npz``
and reports, per ordered chain pair and per unordered pair:

- **ipSAE** (Dunbrack, "Res ipSAE loquunt: What's wrong with AlphaFold's ipTM
  score and how to fix it", bioRxiv 2025.02.10.637595), with its d0chn and
  d0dom variants and the PAE-only ipTM it is contrasted with;
- **pDockQ** (Bryant, Pozzati and Elofsson, Nat. Commun. 13:1265, 2022);
- **pDockQ2** (Zhu, Shenoy, Kundrotas and Elofsson, Bioinformatics 39:btad424,
  2023);
- **LIS** (Kim et al., bioRxiv 2024.02.19.580970).

Every definition is a port of the reference implementation, ``ipsae.py``
version 4 (DunbrackLab/IPSAE, 2026-01-03), down to its edges: the scalar d0
used for d0chn/d0dom floors ``L`` at 27 while the per-residue d0res array
floors it at 26; residues are one token per polymer residue (its CA, or C1'
for a nucleotide; ligand tokens are dropped and a modified residue keeps its
CA token); distances are between CB atoms (CA for glycine, C3' for a
nucleotide); pDockQ's pLDDT is the one at that CB atom. Defaults are the
reference's recommended ``pae_cutoff`` of 10 A; ``dist_cutoff`` only changes
the two interface-residue counts (``dist1``/``dist2``), never a score.

These numbers are **derived by FoldJAX** from what the model wrote. They are
kept apart from the model's own scores: ``native`` holds only what the model
itself returned (its chain-pair ipTM, where it has one), ``derived`` holds the
rest with the method, cutoffs and citation. The inputs differ by model -- each
PAE is over that model's own tokens and has that model's own calibration -- so
these are within-model quantities like any other confidence score, and nothing
here pools or ranks across models.
"""

from __future__ import annotations

import csv
import io
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from foldjax import confidence_arrays

INTERFACES_SCHEMA = "foldjax-interfaces-v1"
DEFAULT_PAE_CUTOFF = 10.0
DEFAULT_DIST_CUTOFF = 10.0
#: pDockQ's contact distance between CB atoms, fixed by its definition.
PDOCKQ_CUTOFF = 8.0
#: LIS's PAE threshold, fixed by its definition.
LIS_CUTOFF = 12.0

#: ipsae.py's residue_set and nucleic subset.
_NUCLEIC = frozenset({"DA", "DC", "DT", "DG", "A", "C", "U", "G"})

METHOD = {
    "reference_implementation": (
        "ipsae.py version 4 (DunbrackLab/IPSAE, 2026-01-03), ported to NumPy"
    ),
    "citations": {
        "ipsae": "Dunbrack RL. bioRxiv 2025.02.10.637595",
        "pdockq": "Bryant P, Pozzati G, Elofsson A. Nat Commun 13:1265 (2022)",
        "pdockq2": "Zhu W et al. Bioinformatics 39(7):btad424 (2023)",
        "lis": "Kim AR et al. bioRxiv 2024.02.19.580970",
    },
    "tokens": (
        "one per polymer residue (CA; C1' for nucleotides); ligand tokens are "
        "dropped; distances between CB atoms (CA for Gly, C3' for nucleotides)"
    ),
    "d0": (
        "1.24*(L-15)^(1/3)-1.8, at least 1.0 (2.0 if either chain is nucleic); "
        "L floored at 27 for d0chn/d0dom and at 26 for d0res"
    ),
    "pdockq_distance_angstrom": PDOCKQ_CUTOFF,
    "lis_pae_cutoff_angstrom": LIS_CUTOFF,
    "note": (
        "derived by FoldJAX from the model's PAE, pLDDT and structure; not a "
        "score the model reported, and not comparable across models"
    ),
}

#: Columns of one interface row, in order.
ROW_FIELDS = (
    "model",
    "input",
    "configuration",
    "job",
    "seed",
    "sample",
    "chain1",
    "chain2",
    "direction",
    "pair_type",
    "native.chain_pair_iptm",
    "derived.ipsae",
    "derived.ipsae_d0chn",
    "derived.ipsae_d0dom",
    "derived.iptm_d0chn",
    "derived.pdockq",
    "derived.pdockq2",
    "derived.lis",
    "n0res",
    "n0chn",
    "n0dom",
    "d0res",
    "d0chn",
    "d0dom",
    "nres1",
    "nres2",
    "dist1",
    "dist2",
    "pae_cutoff",
    "dist_cutoff",
    "structure_path",
    "skipped",
)


class InterfaceInputError(ValueError):
    """A sample lacks what the derived scores need (PAE, token maps, atoms)."""


def calc_d0(length: float, pair_type: str) -> float:
    """ipsae.py ``calc_d0``: d0 for a scalar length (floor 27)."""
    length = float(length)
    minimum = 2.0 if pair_type == "nucleic_acid" else 1.0
    d0 = 1.24 * (length - 15) ** (1.0 / 3.0) - 1.8 if length > 27 else 1.0
    return max(minimum, d0)


def calc_d0_array(length: Any, pair_type: str) -> np.ndarray:
    """ipsae.py ``calc_d0_array``: d0 per residue (floor 26)."""
    length = np.maximum(26.0, np.asarray(length, dtype=float))
    minimum = 2.0 if pair_type == "nucleic_acid" else 1.0
    return np.maximum(minimum, 1.24 * (length - 15) ** (1.0 / 3.0) - 1.8)


def _ptm(pae: np.ndarray, d0: Any) -> np.ndarray:
    return 1.0 / (1.0 + (pae / d0) ** 2.0)


@dataclass(frozen=True, slots=True)
class Residues:
    """The residues the scores run over, one entry per polymer residue."""

    chain: np.ndarray
    resnum: np.ndarray
    resname: np.ndarray
    #: CB (CA for Gly, C3' for nucleotides) coordinates, [n, 3].
    coordinates: np.ndarray
    #: The model's token for each residue (its CA/C1' token).
    token: np.ndarray
    #: pLDDT at the CB atom (or of the token), 0-100; None when unavailable.
    plddt: np.ndarray | None
    plddt_source: str | None


def _structure_residues(path: Path) -> list[dict[str, Any]]:
    """Polymer residues with their CA and CB atoms, and every atom's index."""
    import gemmi

    structure = gemmi.read_structure(str(path))
    structure.setup_entities()
    if len(structure) == 0:
        raise InterfaceInputError(f"{path} has no model")
    found: list[dict[str, Any]] = []
    atom_index = 0
    for chain in structure[0]:
        for residue in chain:
            names = [atom.name for atom in residue]
            heavy = [
                position
                for position, atom in enumerate(residue)
                if atom.element.name not in ("H", "D")
            ]
            start = atom_index
            atom_index += len(names)
            if residue.entity_type != gemmi.EntityType.Polymer:
                continue
            ca = next(
                (k for k, name in enumerate(names) if name == "CA" or "C1" in name),
                None,
            )
            cb = next(
                (
                    k
                    for k, name in enumerate(names)
                    if name == "CB"
                    or "C3" in name
                    or (residue.name == "GLY" and name == "CA")
                ),
                None,
            )
            if ca is None or cb is None:
                continue
            found.append(
                {
                    "chain": chain.name,
                    "resnum": int(residue.seqid.num),
                    "resname": residue.name,
                    "ca_atom": start + ca,
                    "cb_atom": start + cb,
                    "ca_heavy_rank": heavy.index(ca) if ca in heavy else None,
                    "xyz": np.array(residue[cb].pos.tolist(), dtype=float),
                }
            )
    return found


def _scale(arrays: confidence_arrays.ConfidenceArrays, name: str) -> float:
    scale = arrays.describe(name).get("scale")
    return 100.0 if scale == "0-1" else 1.0


def residues_for(
    structure: str | os.PathLike[str],
    arrays: confidence_arrays.ConfidenceArrays,
) -> Residues:
    """Map each polymer residue of ``structure`` to its token in ``arrays``."""
    structure = Path(structure)
    entries = _structure_residues(structure)
    if not entries:
        raise InterfaceInputError(f"{structure} has no polymer residue with a CA/C1'")
    token_of: list[int] = []
    atom_tokens = arrays.get("atom_token_index")
    n_atoms = max(entry["ca_atom"] for entry in entries) + 1
    if atom_tokens is not None and len(atom_tokens) >= n_atoms:
        token_of = [int(atom_tokens[entry["ca_atom"]]) for entry in entries]
    else:
        chains = arrays.get("token_chain_id")
        numbers = arrays.get("token_residue_index")
        if chains is None or numbers is None:
            raise InterfaceInputError(
                "confidence_full.npz has neither atom_token_index nor "
                "token_chain_id/token_residue_index, so its PAE cannot be "
                "placed on the structure's residues"
            )
        index: dict[tuple[str, int], list[int]] = {}
        for position, (chain, number) in enumerate(
            zip(chains.astype(str), numbers.astype(int), strict=True)
        ):
            index.setdefault((chain, int(number)), []).append(position)
        for entry in entries:
            tokens = index.get((entry["chain"], entry["resnum"]))
            if not tokens:
                raise InterfaceInputError(
                    f"no token for chain {entry['chain']} residue {entry['resnum']} "
                    "in confidence_full.npz"
                )
            if len(tokens) == 1:
                token_of.append(tokens[0])
                continue
            # A modified residue tokenized per atom: its tokens follow its heavy
            # atoms in file order, and the CA's token is the one ipsae.py uses.
            rank = entry["ca_heavy_rank"]
            if rank is None or rank >= len(tokens):
                raise InterfaceInputError(
                    f"chain {entry['chain']} residue {entry['resnum']} has "
                    f"{len(tokens)} tokens and no CA among its first heavy atoms"
                )
            token_of.append(tokens[rank])
    plddt = None
    source = None
    atom_plddt = arrays.get("atom_plddt")
    token_plddt = arrays.get("token_plddt")
    if atom_plddt is not None and len(atom_plddt) >= n_atoms:
        factor = _scale(arrays, "atom_plddt")
        plddt = np.asarray(
            [float(atom_plddt[entry["cb_atom"]]) * factor for entry in entries]
        )
        source = "atom_plddt at the CB atom"
    elif token_plddt is not None:
        factor = _scale(arrays, "token_plddt")
        plddt = np.asarray([float(token_plddt[t]) * factor for t in token_of])
        source = "token_plddt"
    return Residues(
        chain=np.asarray([entry["chain"] for entry in entries]),
        resnum=np.asarray([entry["resnum"] for entry in entries], dtype=int),
        resname=np.asarray([entry["resname"] for entry in entries]),
        coordinates=np.stack([entry["xyz"] for entry in entries]),
        token=np.asarray(token_of, dtype=int),
        plddt=plddt,
        plddt_source=source,
    )


def _unique_chains(chains: np.ndarray) -> list[str]:
    return list(dict.fromkeys(str(chain) for chain in chains))


def pair_scores(
    residues: Residues,
    pae: np.ndarray,
    *,
    pae_cutoff: float = DEFAULT_PAE_CUTOFF,
    dist_cutoff: float = DEFAULT_DIST_CUTOFF,
) -> dict[tuple[str, str], dict[str, Any]]:
    """Every score of ipsae.py for each ordered chain pair.

    ``pae`` is the residue-by-residue PAE (already restricted to
    ``residues``), in angstroms.
    """
    pae = np.asarray(pae, dtype=np.float64)
    chains = residues.chain
    coordinates = residues.coordinates
    distances = np.sqrt(
        ((coordinates[:, None, :] - coordinates[None, :, :]) ** 2).sum(axis=2)
    )
    names = _unique_chains(chains)
    nucleic = {
        chain: bool(np.isin(residues.resname[chains == chain], list(_NUCLEIC)).any())
        for chain in names
    }
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for chain1 in names:
        for chain2 in names:
            if chain1 == chain2:
                continue
            pair_type = (
                "nucleic_acid" if nucleic[chain1] or nucleic[chain2] else "protein"
            )
            rows = np.flatnonzero(chains == chain1)
            cols = np.flatnonzero(chains == chain2)
            block = pae[np.ix_(rows, cols)]
            dist = distances[np.ix_(rows, cols)]
            entry: dict[str, Any] = {"pair_type": pair_type}

            # pDockQ / pDockQ2: CB contacts within 8 A.
            contact = dist <= PDOCKQ_CUTOFF
            npairs = int(contact.sum())
            interface = sorted(
                set(rows[contact.any(axis=1)].tolist())
                | set(cols[contact.any(axis=0)].tolist())
            )
            if npairs > 0 and residues.plddt is not None:
                mean_plddt = float(residues.plddt[interface].mean())
                x = mean_plddt * math.log10(npairs)
                entry["pdockq"] = 0.724 / (1 + math.exp(-0.052 * (x - 152.611))) + 0.018
                mean_ptm = float(_ptm(block[contact], 10.0).sum()) / npairs
                x2 = mean_plddt * mean_ptm
                entry["pdockq2"] = 1.31 / (1 + math.exp(-0.075 * (x2 - 84.733))) + 0.005
            elif residues.plddt is None:
                entry["pdockq"] = entry["pdockq2"] = None
            else:
                entry["pdockq"] = entry["pdockq2"] = 0.0

            # LIS: mean of (12 - PAE)/12 over inter-chain PAE below 12 A.
            below = block[block < LIS_CUTOFF]
            entry["lis"] = (
                float(np.mean((LIS_CUTOFF - below) / LIS_CUTOFF)) if below.size else 0.0
            )

            # ipTM (PAE-derived) and ipSAE.
            n0chn = int(len(rows) + len(cols))
            d0chn = calc_d0(n0chn, pair_type)
            ptm_chn = _ptm(block, d0chn)
            valid = block < pae_cutoff
            counts = valid.sum(axis=1)
            iptm_byres = ptm_chn.mean(axis=1) if len(cols) else np.zeros(len(rows))
            with np.errstate(invalid="ignore", divide="ignore"):
                ipsae_chn_byres = np.where(
                    counts > 0, (ptm_chn * valid).sum(axis=1) / counts, 0.0
                )
            residues_1 = set(residues.resnum[rows[counts > 0]].tolist())
            residues_2 = set(residues.resnum[cols[valid.any(axis=0)]].tolist())
            n0dom = len(residues_1) + len(residues_2)
            d0dom = calc_d0(n0dom, pair_type)
            ptm_dom = _ptm(block, d0dom)
            d0res_byres = calc_d0_array(counts, pair_type)
            ptm_res = _ptm(block, d0res_byres[:, None])
            with np.errstate(invalid="ignore", divide="ignore"):
                ipsae_dom_byres = np.where(
                    counts > 0, (ptm_dom * valid).sum(axis=1) / counts, 0.0
                )
                ipsae_res_byres = np.where(
                    counts > 0, (ptm_res * valid).sum(axis=1) / counts, 0.0
                )
            close = valid & (dist < dist_cutoff)
            best = int(np.argmax(ipsae_res_byres)) if len(rows) else 0
            entry.update(
                {
                    "ipsae": float(ipsae_res_byres.max()) if len(rows) else 0.0,
                    "ipsae_d0chn": float(ipsae_chn_byres.max()) if len(rows) else 0.0,
                    "ipsae_d0dom": float(ipsae_dom_byres.max()) if len(rows) else 0.0,
                    "iptm_d0chn": float(iptm_byres.max()) if len(rows) else 0.0,
                    "n0res": int(counts[best]) if len(rows) else 0,
                    "d0res": float(d0res_byres[best]) if len(rows) else 1.0,
                    "n0chn": n0chn,
                    "d0chn": d0chn,
                    "n0dom": n0dom,
                    "d0dom": d0dom,
                    "nres1": len(residues_1),
                    "nres2": len(residues_2),
                    "dist1": len(
                        set(residues.resnum[rows[close.any(axis=1)]].tolist())
                    ),
                    "dist2": len(
                        set(residues.resnum[cols[close.any(axis=0)]].tolist())
                    ),
                    "_residues_1": residues_1,
                    "_residues_2": residues_2,
                    "_dist_1": set(residues.resnum[rows[close.any(axis=1)]].tolist()),
                    "_dist_2": set(residues.resnum[cols[close.any(axis=0)]].tolist()),
                }
            )
            out[(chain1, chain2)] = entry
    return out


def _max_row(one: Mapping[str, Any], other: Mapping[str, Any]) -> dict[str, Any]:
    """ipsae.py's "max" line for an unordered pair (``one`` is chain1 > chain2)."""
    row: dict[str, Any] = {"pair_type": one["pair_type"], "n0chn": one["n0chn"]}
    row["d0chn"] = one["d0chn"]
    for key in ("ipsae_d0chn", "iptm_d0chn"):
        row[key] = max(one[key], other[key])
    dom = one if one["ipsae_d0dom"] >= other["ipsae_d0dom"] else other
    row["ipsae_d0dom"] = dom["ipsae_d0dom"]
    row["n0dom"], row["d0dom"] = dom["n0dom"], dom["d0dom"]
    res = one if one["ipsae"] >= other["ipsae"] else other
    row["ipsae"] = res["ipsae"]
    row["n0res"], row["d0res"] = res["n0res"], res["d0res"]
    row["pdockq"] = one["pdockq"]
    row["pdockq2"] = (
        None
        if one["pdockq2"] is None or other["pdockq2"] is None
        else max(one["pdockq2"], other["pdockq2"])
    )
    row["lis"] = (one["lis"] + other["lis"]) / 2.0
    row["nres1"] = max(len(one["_residues_2"]), len(other["_residues_1"]))
    row["nres2"] = max(len(one["_residues_1"]), len(other["_residues_2"]))
    row["dist1"] = max(len(one["_dist_2"]), len(other["_dist_1"]))
    row["dist2"] = max(len(one["_dist_1"]), len(other["_dist_2"]))
    return row


def _native_chain_pair_iptm(
    arrays: confidence_arrays.ConfidenceArrays,
) -> tuple[dict[tuple[str, str], float] | None, str | None]:
    matrix = arrays.get("chain_pair_iptm")
    if matrix is None:
        reason = arrays.unavailable.get(
            "chain_pair_iptm", "the model wrote no chain_pair_iptm"
        )
        return None, reason
    chain_ids = arrays.get("chain_id")
    if chain_ids is None:
        token_chains = arrays.get("token_chain_id")
        if token_chains is None:
            return None, "chain_pair_iptm has no chain_id to index it"
        chain_ids = np.asarray(_unique_chains(token_chains))
    labels = [str(chain) for chain in chain_ids]
    if matrix.shape != (len(labels), len(labels)):
        return None, "chain_pair_iptm does not match chain_id"
    return {
        (a, b): float(matrix[i, j])
        for i, a in enumerate(labels)
        for j, b in enumerate(labels)
        if a != b
    }, None


def sample_interfaces(
    sample_dir: str | os.PathLike[str],
    structure: str | os.PathLike[str] | None = None,
    *,
    pae_cutoff: float = DEFAULT_PAE_CUTOFF,
    dist_cutoff: float = DEFAULT_DIST_CUTOFF,
) -> dict[str, Any]:
    """Interface scores of one canonical sample directory.

    Returns ``{"native": ..., "derived": ..., "skipped": reason-or-None}``.
    A sample without PAE, token maps or a second chain is reported with the
    reason and no numbers rather than raised: one such sample must not hide
    the others of a batch.
    """
    sample_dir = Path(sample_dir)
    if structure is None:
        found = sorted(
            path
            for path in sample_dir.iterdir()
            if path.suffix.lower() in {".cif", ".mmcif"}
        )
        if not found:
            return {"native": None, "derived": None, "skipped": "no mmCIF structure"}
        structure = found[0]
    try:
        arrays = confidence_arrays.load_confidence_arrays(sample_dir)
    except FileNotFoundError:
        return {
            "native": None,
            "derived": None,
            "skipped": f"no {confidence_arrays.FILENAME}; the run wrote no "
            "confidence arrays",
        }
    native, native_reason = _native_chain_pair_iptm(arrays)
    native_block = {
        "chain_pair_iptm": (
            None
            if native is None
            else [
                {"chain1": a, "chain2": b, "value": v} for (a, b), v in native.items()
            ]
        ),
        "source": "confidence_full.npz chain_pair_iptm (the model's own)",
        "reason": native_reason,
    }
    if "pae" not in arrays:
        reason = arrays.unavailable.get("pae", "the model wrote no PAE")
        return {
            "native": native_block,
            "derived": None,
            "skipped": f"no PAE in {confidence_arrays.FILENAME}: {reason}",
        }
    try:
        residues = residues_for(structure, arrays)
    except InterfaceInputError as error:
        return {"native": native_block, "derived": None, "skipped": str(error)}
    if len(_unique_chains(residues.chain)) < 2:
        return {
            "native": native_block,
            "derived": None,
            "skipped": "fewer than two polymer chains",
        }
    pae = np.asarray(arrays["pae"], dtype=np.float64)
    if residues.token.max() >= pae.shape[0]:
        return {
            "native": native_block,
            "derived": None,
            "skipped": "token maps point past the PAE matrix",
        }
    sub = pae[np.ix_(residues.token, residues.token)]
    scores = pair_scores(residues, sub, pae_cutoff=pae_cutoff, dist_cutoff=dist_cutoff)
    pairs = []
    names = _unique_chains(residues.chain)
    public = (
        "pair_type",
        "ipsae",
        "ipsae_d0chn",
        "ipsae_d0dom",
        "iptm_d0chn",
        "pdockq",
        "pdockq2",
        "lis",
        "n0res",
        "n0chn",
        "n0dom",
        "d0res",
        "d0chn",
        "d0dom",
        "nres1",
        "nres2",
        "dist1",
        "dist2",
    )
    for chain_a in names:
        for chain_b in names:
            if chain_a >= chain_b:
                continue
            for chain1, chain2 in ((chain_a, chain_b), (chain_b, chain_a)):
                entry = scores[(chain1, chain2)]
                record = {key: entry[key] for key in public}
                record.update(
                    {
                        "chain1": chain1,
                        "chain2": chain2,
                        "direction": "asym",
                        "native_chain_pair_iptm": (
                            None if native is None else native.get((chain1, chain2))
                        ),
                    }
                )
                pairs.append(record)
            record = _max_row(scores[(chain_b, chain_a)], scores[(chain_a, chain_b)])
            record.update(
                {
                    "chain1": chain_a,
                    "chain2": chain_b,
                    "direction": "max",
                    "native_chain_pair_iptm": (
                        None
                        if native is None
                        or (chain_a, chain_b) not in native
                        or (chain_b, chain_a) not in native
                        else max(native[(chain_a, chain_b)], native[(chain_b, chain_a)])
                    ),
                }
            )
            pairs.append(record)
    return {
        "native": native_block,
        "derived": {
            "method": METHOD,
            "pae_cutoff": pae_cutoff,
            "dist_cutoff": dist_cutoff,
            "plddt_source": residues.plddt_source,
            "pdockq_missing": (
                None
                if residues.plddt is not None
                else "no atom_plddt or token_plddt in confidence_full.npz"
            ),
            "residues": int(len(residues.chain)),
            "pairs": pairs,
        },
        "skipped": None,
    }


def interface_rows(
    root: str | os.PathLike[str],
    *,
    pae_cutoff: float = DEFAULT_PAE_CUTOFF,
    dist_cutoff: float = DEFAULT_DIST_CUTOFF,
) -> list[dict[str, Any]]:
    """One row per sample, chain pair and direction (``asym`` and ``max``).

    A sample that cannot be scored is one row with ``skipped`` set.
    """
    from foldjax.results import load_results

    report = load_results(root)
    rows: list[dict[str, Any]] = []
    for run, sample in report.samples():
        base = {
            "model": run.model,
            "input": run.input,
            "configuration": run.configuration,
            "job": sample.job,
            "seed": sample.seed,
            "sample": sample.sample,
            "structure_path": str(sample.structure_path)
            if sample.structure_path
            else None,
            "pae_cutoff": pae_cutoff,
            "dist_cutoff": dist_cutoff,
        }
        if sample.structure_path is None or not sample.structure_verified:
            rows.append({**base, "skipped": "structure missing or changed"})
            continue
        result = sample_interfaces(
            sample.structure_path.parent,
            sample.structure_path,
            pae_cutoff=pae_cutoff,
            dist_cutoff=dist_cutoff,
        )
        if result["derived"] is None:
            rows.append({**base, "skipped": result["skipped"]})
            continue
        for pair in result["derived"]["pairs"]:
            row = dict(base)
            row.update(
                {
                    "chain1": pair["chain1"],
                    "chain2": pair["chain2"],
                    "direction": pair["direction"],
                    "pair_type": pair["pair_type"],
                    "native.chain_pair_iptm": pair["native_chain_pair_iptm"],
                    "skipped": None,
                }
            )
            for key in (
                "ipsae",
                "ipsae_d0chn",
                "ipsae_d0dom",
                "iptm_d0chn",
                "pdockq",
                "pdockq2",
                "lis",
            ):
                row[f"derived.{key}"] = pair[key]
            for key in (
                "n0res",
                "n0chn",
                "n0dom",
                "d0res",
                "d0chn",
                "d0dom",
                "nres1",
                "nres2",
                "dist1",
                "dist2",
            ):
                row[key] = pair[key]
            rows.append(row)
    return rows


def interface_columns(rows: Sequence[Mapping[str, Any]]) -> dict[tuple, dict]:
    """The ``max`` rows folded into per-sample columns for ``show --interfaces``.

    Keyed by (run_dir-free identity: model, input, configuration, job, seed,
    sample); each value maps ``derived.<metric>.<A>-<B>`` and
    ``native.chain_pair_iptm.<A>-<B>`` to the unordered pair's value.
    """
    folded: dict[tuple, dict] = {}
    for row in rows:
        key = (
            row.get("model"),
            row.get("input"),
            row.get("configuration"),
            row.get("job"),
            row.get("seed"),
            row.get("sample"),
        )
        target = folded.setdefault(key, {})
        if row.get("skipped"):
            target["derived.interfaces_skipped"] = row["skipped"]
            continue
        if row.get("direction") != "max":
            continue
        pair = f"{row['chain1']}-{row['chain2']}"
        for metric in ("ipsae", "pdockq", "pdockq2", "lis"):
            target[f"derived.{metric}.{pair}"] = row.get(f"derived.{metric}")
        target[f"native.chain_pair_iptm.{pair}"] = row.get("native.chain_pair_iptm")
    return folded


def to_csv(rows: Sequence[Mapping[str, Any]]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(ROW_FIELDS), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {name: "" if row.get(name) is None else row[name] for name in ROW_FIELDS}
        )
    return buffer.getvalue()


def render_table(rows: Sequence[Mapping[str, Any]]) -> str:
    """The unordered-pair (``max``) rows for reading; skipped samples named."""
    lines = [
        "derived by FoldJAX from each model's own PAE (ipsae.py v4 definitions); "
        "within-model only",
        f"{'model':<11s}{'seed':>5s}{'smp':>4s}  {'pair':<7s}{'ipSAE':>7s}"
        f"{'pDockQ':>8s}{'pDockQ2':>9s}{'LIS':>7s}{'ipTM*':>7s}  input",
    ]

    def number(value: Any) -> str:
        return "-" if value is None else f"{float(value):.3f}"

    for row in rows:
        head = (
            f"{str(row.get('model')):<11s}{str(row.get('seed')):>5s}"
            f"{str(row.get('sample')):>4s}  "
        )
        stem = Path(str(row.get("input") or "")).stem
        if row.get("skipped"):
            lines.append(f"{head}skipped: {row['skipped']}  {stem}")
            continue
        if row.get("direction") != "max":
            continue
        lines.append(
            f"{head}{row['chain1'] + '-' + row['chain2']:<7s}"
            f"{number(row.get('derived.ipsae')):>7s}"
            f"{number(row.get('derived.pdockq')):>8s}"
            f"{number(row.get('derived.pdockq2')):>9s}"
            f"{number(row.get('derived.lis')):>7s}"
            f"{number(row.get('native.chain_pair_iptm')):>7s}  {stem}"
        )
    lines.append("ipTM* = the model's own chain-pair ipTM (max of both directions)")
    return "\n".join(lines)


def interfaces_document(
    root: str | os.PathLike[str],
    *,
    pae_cutoff: float = DEFAULT_PAE_CUTOFF,
    dist_cutoff: float = DEFAULT_DIST_CUTOFF,
) -> dict[str, Any]:
    rows = interface_rows(root, pae_cutoff=pae_cutoff, dist_cutoff=dist_cutoff)
    return {
        "schema": INTERFACES_SCHEMA,
        "root": str(root),
        "method": METHOD,
        "pae_cutoff": pae_cutoff,
        "dist_cutoff": dist_cutoff,
        "rows": rows,
    }


def to_json(document: Mapping[str, Any]) -> str:
    return json.dumps(document, indent=2, sort_keys=True, default=str)
