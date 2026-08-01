"""Build the vendored ligand CCD reference table (``ccd_ligands.npz``).

Run with the upstream torch Protenix venv, which provides biotite + the CCD
component dictionary::

    LAYERNORM_TYPE=torch CUDA_VISIBLE_DEVICES="" \
        /path/to/protenix/.venv/bin/python \
        scripts/build_ccd_ligands_npz.py --out \
        src/foldjax/models/protenix/data/ccd_ligands.npz

For each CCD code it mirrors ``json_parser.build_ligand`` exactly:
``get_component_atom_array(code, keep_leaving_atoms=True, keep_hydrogens=False)``
for the per-atom order/names/elements, and ``get_ccd_ref_info`` (looked up by
atom name) for ref_pos/charge/mask.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from protenix.data.core import ccd

# Curated ligand/ion set (examples + common cofactors and ions).
DEFAULT_CODES = [
    # ions
    "NA", "K", "MG", "CA", "ZN", "FE", "FE2", "MN", "CU", "CU1",
    "CO", "NI", "CD", "HG", "CL", "BR", "IOD", "F",
    # common ligands / cofactors
    "ATP", "ADP", "AMP", "GTP", "GDP", "GNP", "ANP",
    "NAD", "NAP", "FAD", "FMN", "HEM", "PLP", "SAM", "SAH",
    "PCG", "PO4", "SO4", "GOL", "EDO", "ACT", "PEG",
    "NAG", "MAN", "BMA", "GAL", "FUC", "BGC",
    "CIT", "MES", "EPE", "TRS", "DMS", "ACE",
    # common non-standard protein residue
    "MSE",
]


def build(codes: list[str]) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for code in codes:
        atom_array = ccd.get_component_atom_array(
            code, keep_leaving_atoms=True, keep_hydrogens=False
        )
        if atom_array is None or len(atom_array) == 0:
            raise ValueError(f"no component atom array for CCD code {code!r}")
        ref = ccd.get_ccd_ref_info(code, return_atomic_number=True)
        if not ref:
            raise ValueError(f"no ref info for CCD code {code!r}")
        names = [str(n) for n in atom_array.atom_name]
        sub = [ref["atom_map"][n] for n in names]
        arrays[f"{code}/names"] = np.asarray(names)
        arrays[f"{code}/elem"] = np.asarray(
            [str(e) for e in atom_array.element]
        )
        arrays[f"{code}/coord"] = ref["coord"][sub].astype(np.float32)
        arrays[f"{code}/charge"] = ref["charge"][sub].astype(np.float32)
        arrays[f"{code}/mask"] = ref["mask"][sub].astype(np.float32)
        bond_array = (
            atom_array.bonds.as_array()
            if atom_array.bonds is not None
            else np.zeros((0, 3), dtype=np.int64)
        )
        arrays[f"{code}/bonds"] = (
            bond_array[:, :2].astype(np.int64)
            if len(bond_array)
            else np.zeros((0, 2), dtype=np.int64)
        )
    arrays["_codes"] = np.asarray(codes)
    return arrays


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--codes", nargs="*", default=None)
    args = parser.parse_args()
    codes = args.codes or DEFAULT_CODES
    arrays = build(codes)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **arrays)
    print(f"wrote {args.out} with {len(codes)} ligand codes")


if __name__ == "__main__":
    main()
