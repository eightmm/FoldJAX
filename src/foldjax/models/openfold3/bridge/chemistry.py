"""Build the representative-atom table from upstream's constants.

This lives in ``bridge`` because it imports OpenFold3's vendored constant tables:
the chemistry must not be transcribed by hand, but feature-archive inference must
not import the preprocessing pipeline. Build it once while featurizing, then carry
it as data. The constants themselves are NumPy-only and do not import Torch.
"""

from __future__ import annotations

import numpy as np

from foldjax.models.openfold3.models.representative_atoms import RepresentativeAtomTable


def representative_atom_table() -> RepresentativeAtomTable:
    """Read the lookups out of upstream's own tables, vendored at `.._upstream`.

    These are constants rather than a pipeline -- residue order and the atom-name
    index -- so reading them from upstream's own definition is what keeps them
    from drifting into a second, hand-copied truth.
    """
    from foldjax.models.openfold3._upstream.openfold3.core.data.resources.residues import (  # noqa: E501
        STANDARD_PROTEIN_RESIDUES_ORDER,
        STANDARD_RESIDUES_WITH_GAP_3,
    )
    from foldjax.models.openfold3._upstream.openfold3.core.data.resources.token_atom_constants import (  # noqa: E501
        atom_name_to_index_by_restype,
    )

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
