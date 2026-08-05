"""Build the representative-atom table from upstream's constants.

This lives in ``bridge`` because it imports OpenFold3: the table encodes chemistry
that must not be transcribed by hand, but the inference path must not depend on
upstream either. Build it once with torch installed, then carry it as data.
"""

from __future__ import annotations

import numpy as np

from foldjax.models.openfold3.models.representative_atoms import RepresentativeAtomTable


def representative_atom_table() -> RepresentativeAtomTable:
    """Read the lookups out of upstream's own tables."""
    from foldjax.models.openfold3.upstream import ensure_importable

    ensure_importable()
    from openfold3.core.data.resources.residues import (
        STANDARD_PROTEIN_RESIDUES_ORDER,
        STANDARD_RESIDUES_WITH_GAP_3,
    )
    from openfold3.core.utils.atomize_utils import atom_name_to_index_by_restype

    n_restype = len(STANDARD_RESIDUES_WITH_GAP_3)

    def indicator(*names: str) -> np.ndarray:
        vector = np.zeros(n_restype, dtype=np.float32)
        for name in names:
            vector[STANDARD_RESIDUES_WITH_GAP_3.index(name)] = 1.0
        return vector

    def lookup(atom_name: str, field: str) -> np.ndarray:
        return np.asarray(
            atom_name_to_index_by_restype[atom_name][field], dtype=np.float32
        )

    # Glycine is indexed in the protein-only ordering upstream uses at this one
    # site, not in STANDARD_RESIDUES_WITH_GAP_3; reproducing that is the point of
    # building the table here rather than by hand.
    glycine = np.zeros(n_restype, dtype=np.float32)
    glycine[STANDARD_PROTEIN_RESIDUES_ORDER["G"]] = 1.0

    return RepresentativeAtomTable(
        cb_offset=lookup("CB", "index"),
        cb_mask=lookup("CB", "mask"),
        ca_offset=lookup("CA", "index"),
        ca_mask=lookup("CA", "mask"),
        c4_offset=lookup("C4", "index"),
        c4_mask=lookup("C4", "mask"),
        c2_offset=lookup("C2", "index"),
        c2_mask=lookup("C2", "mask"),
        glycine=glycine,
        purine_dna=indicator("DA", "DG"),
        purine_rna=indicator("A", "G"),
        pyrimidine_dna=indicator("DC", "DT"),
        pyrimidine_rna=indicator("C", "U"),
    )
