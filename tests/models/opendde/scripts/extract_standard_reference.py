"""Extract small standard-component reference tables from the official CCD cache."""

from __future__ import annotations

import argparse
import hashlib
import pickle
from pathlib import Path

import numpy as np

EXPECTED_SHA256 = "d1cfb71f5993a3ebea7c47877022d7f597bbfbaf86e28a4770e957da6c50cd35"
STANDARD_CODES = (
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
    "A",
    "C",
    "G",
    "U",
    "DA",
    "DC",
    "DG",
    "DT",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    digest = hashlib.sha256(args.cache.read_bytes()).hexdigest()
    if digest != EXPECTED_SHA256:
        raise RuntimeError(
            f"official CCD cache SHA-256 mismatch: expected {EXPECTED_SHA256}, "
            f"got {digest}"
        )
    with args.cache.open("rb") as handle:
        molecules = pickle.load(handle)

    arrays: dict[str, np.ndarray] = {}
    for code in STANDARD_CODES:
        molecule = molecules[code]
        conformer = molecule.GetConformer(molecule.ref_conf_id)
        positions = np.asarray(conformer.GetPositions(), dtype=np.float32)
        ordered_names = sorted(molecule.atom_map, key=molecule.atom_map.__getitem__)
        atom_indices = np.asarray(
            [molecule.atom_map[name] for name in ordered_names], dtype=np.int64
        )
        atoms = list(molecule.GetAtoms())
        arrays[f"{code}_names"] = np.asarray(ordered_names, dtype="U8")
        arrays[f"{code}_coord"] = positions[atom_indices]
        arrays[f"{code}_mask"] = np.asarray(molecule.ref_mask, dtype=np.int64)[
            atom_indices
        ]
        arrays[f"{code}_charge"] = np.asarray(
            [atoms[index].GetFormalCharge() for index in atom_indices],
            dtype=np.int64,
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **arrays)
    print(f"wrote {len(STANDARD_CODES)} standard components: {args.out}")


if __name__ == "__main__":
    main()
