"""Turn predicted coordinates into a written structure file.

Mirrors the coords -> structure plumbing of boltz's BoltzWriter
(write_on_batch_end), but standalone: loads the processed StructureV2 from
``{id}.npz``, unpads the predicted coords with the atom pad mask, replaces the
atom/residue/coords tables, and writes a PDB or mmCIF via the vendored writers.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from foldjax.models.boltz2.data.types import Coords, Interface, StructureV2
from foldjax.models.boltz2.data.write.mmcif import to_mmcif
from foldjax.models.boltz2.data.write.pdb import to_pdb


def write_prediction(
    structure_npz: Path,
    coords: np.ndarray,
    atom_pad_mask: np.ndarray,
    out_path: Path,
    plddts: np.ndarray | None = None,
    fmt: str = "pdb",
) -> Path:
    """Write a predicted structure to disk.

    Parameters
    ----------
    structure_npz
        Path to the processed StructureV2 ``{id}.npz``.
    coords
        Predicted atom coordinates, shape ``[n_atoms_padded, 3]`` (or any
        leading singleton/model dims that squeeze to that).
    atom_pad_mask
        Boolean atom pad mask, shape ``[n_atoms_padded]``; True = real atom.
    out_path
        Output file path (extension overridden by ``fmt``).
    plddts
        Optional per-atom / per-token pLDDT, passed to the writer.
    fmt
        ``"pdb"`` or ``"cif"`` / ``"mmcif"``.

    Returns
    -------
    Path
        The written file path.
    """
    structure = StructureV2.load(structure_npz)

    # Drop masked chains exactly like the reference writer.
    structure = structure.remove_invalid_chains()

    coords = np.asarray(coords)
    coords = np.squeeze(coords)
    if coords.ndim != 2 or coords.shape[-1] != 3:
        coords = coords.reshape(-1, 3)

    mask = np.asarray(atom_pad_mask).astype(bool).reshape(-1)
    coord_unpad = coords[mask]

    n_atoms = structure.atoms.shape[0]
    assert coord_unpad.shape[0] == n_atoms, (
        f"unpadded coords ({coord_unpad.shape[0]}) != "
        f"structure atoms ({n_atoms})"
    )

    atoms = structure.atoms.copy()
    atoms["coords"] = coord_unpad
    atoms["is_present"] = True

    residues = structure.residues.copy()
    residues["is_present"] = True

    coords_struct = np.array([(x,) for x in coord_unpad], dtype=Coords)
    interfaces = np.array([], dtype=Interface)

    new_structure = replace(
        structure,
        atoms=atoms,
        residues=residues,
        interfaces=interfaces,
        coords=coords_struct,
    )

    if plddts is not None:
        plddts = np.asarray(plddts).reshape(-1)

    if fmt in ("cif", "mmcif"):
        out_path = out_path.with_suffix(".cif")
        text = to_mmcif(new_structure, plddts=plddts, boltz2=True)
    elif fmt == "pdb":
        out_path = out_path.with_suffix(".pdb")
        text = to_pdb(new_structure, plddts=plddts, boltz2=True)
    else:
        msg = f"Invalid output format: {fmt}"
        raise ValueError(msg)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text)
    return out_path
