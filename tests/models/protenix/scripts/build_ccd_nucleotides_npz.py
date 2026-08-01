"""Build the vendored nucleotide CCD reference table (``ccd_nucleotides.npz``).

Run with the upstream torch Protenix venv, which provides biotite + the CCD
component dictionary::

    LAYERNORM_TYPE=torch CUDA_VISIBLE_DEVICES="" \
        /path/to/protenix/.venv/bin/python \
        scripts/build_ccd_nucleotides_npz.py --out \
        src/foldjax/models/protenix/data/ccd_nucleotides.npz

Mirrors the ligand builder exactly:
``get_component_atom_array(code, keep_leaving_atoms=True, keep_hydrogens=False)``
for the per-atom order/names/elements (== RES_ATOMS_DICT order, OP3 first), and
``get_ccd_ref_info`` (looked up by atom name) for ref_pos/charge/mask. OP3 is
the 5'-terminal leaving atom: kept only for the first residue of a chain.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from protenix.data.core import ccd

# Standard DNA (DA/DC/DG/DT) and RNA (A/C/G/U) residues.
DEFAULT_CODES = ["DA", "DC", "DG", "DT", "A", "C", "G", "U"]


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
    print(f"wrote {args.out} with {len(codes)} nucleotide codes")


if __name__ == "__main__":
    main()
