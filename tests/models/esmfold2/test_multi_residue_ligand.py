"""ESMFold2 multi-CCD ligands against Biohub's own ``prepare_input``.

The expected arrays below were produced by running Biohub esm 3.3.0
(``esm/models/esmfold2/prepare_input.py``, source tree esm-26b0bc2b, torch, CPU)
on this exact job, with ``esm.models.esmfold2.conformers`` replaced by the
same fake chemistry ``_FakeCCD`` serves the port. Upstream addresses a
covalent bond by 0-based residue and an index into that residue's kept atoms;
the job's ``bonds`` name the same atoms (ND2 of ASN 2 to C1 of residue 1, O4
of residue 1 to C1 of residue 2). With the released CCD the port and upstream
were also compared on NAG-NAG-BMA N-linked to a protein: all 18 feature
arrays equal, ``ref_pos`` bit for bit. That run needs the registered CCD and
torch, so it is not repeated here.

What upstream does, per ``tokenize_ligand_ccd`` and ``build_chains_from_input``:
residue i of the ligand is code i, one token per atom, its own
``ref_space_uid``; a chain in any covalent bond loses every residue's leaving
atoms (O1 here), while the unbonded NAG chain keeps its own; and a code list
is a different entity from the single code.
"""

from __future__ import annotations

import numpy as np
import pytest

from foldjax.models.esmfold2.data import all_atom
from foldjax.models.esmfold2.data.all_atom_constants import PROTEIN_HEAVY_ATOMS

_ATOMS = {
    "NAG": [("C1", "C", 0), ("C2", "C", 0), ("O4", "O", 0), ("O1", "O", 0)],
    "BMA": [("C1", "C", 0), ("O4", "O", 0), ("O1", "O", 0)],
}
_BONDS = {
    "NAG": [("C1", "C2"), ("C2", "O4"), ("C1", "O1")],
    "BMA": [("C1", "O4"), ("C1", "O1")],
}


class _FakeCCD:
    def __init__(self, _path) -> None:
        pass

    def atoms(self, component):
        return _ATOMS.get(component)

    def bonds(self, component):
        return _BONDS.get(component)

    def leaving_atoms(self, component):
        return {"O1"} if component in _ATOMS else set()

    def conformer(self, component):
        names = (
            [record[0] for record in _ATOMS[component]]
            if component in _ATOMS
            else PROTEIN_HEAVY_ATOMS.get(component)
        )
        if names is None:
            return None
        offset = 10.0 * (sum(map(ord, component)) % 7)
        return {
            name: np.asarray((offset + i, i + 1.5, i + 2.25), dtype=np.float32)
            for i, name in enumerate(names)
        }

    def idealized_position(self, component, atom_name):
        found = self.conformer(component)
        return None if found is None else found.get(atom_name)


_DOCUMENT = {
    "entities": [
        {"type": "protein", "id": "A", "sequence": "ANA"},
        {"type": "ligand", "id": "G", "ccd": ["NAG", "BMA"]},
        {"type": "ligand", "id": "H", "ccd": "NAG"},
    ],
    "bonds": [
        [["A", 2, "ND2"], ["G", 1, "C1"]],
        [["G", 1, "O4"], ["G", 2, "C1"]],
    ],
}

# Upstream prepare_esmfold2_input on _DOCUMENT (see the module docstring).
_UPSTREAM = {
    "residue_index": [0, 1, 2, 0, 0, 0, 1, 1, 0, 0, 0, 0],
    "asym_id": [0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2],
    "entity_id": [0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2],
    "sym_id": [0] * 12,
    "mol_type": [0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3],
    "ref_space_uid": [0] * 5 + [1] * 8 + [2] * 5 + [3] * 3 + [4] * 2 + [5] * 4,
    "atom_to_token": [0] * 5 + [1] * 8 + [2] * 5 + list(range(3, 12)),
}
_UPSTREAM_RESIDUE_NAMES = (
    ["ALA", "ASN", "ALA"] + ["NAG"] * 3 + ["BMA"] * 2 + ["NAG"] * 4
)
_UPSTREAM_LIGAND_ATOMS = ["C1", "C2", "O4", "C1", "O4", "C1", "C2", "O4", "O1"]
_UPSTREAM_BOND_EDGES = {
    (1, 3), (3, 4), (4, 5), (5, 6), (6, 7), (8, 9), (8, 11), (9, 10),
}  # fmt: skip
_UPSTREAM_LIGAND_REF_POS = [
    [40.0, 1.5, 2.25], [41.0, 2.5, 3.25], [42.0, 3.5, 4.25],
    [50.0, 1.5, 2.25], [51.0, 2.5, 3.25],
    [40.0, 1.5, 2.25], [41.0, 2.5, 3.25], [42.0, 3.5, 4.25], [43.0, 4.5, 5.25],
]  # fmt: skip


@pytest.fixture
def built(monkeypatch):
    monkeypatch.setattr(all_atom, "get_ccd_store", _FakeCCD)
    features = all_atom.build_job_features(
        _DOCUMENT, base_dir=".", ccd_path="unused", seed=7
    )
    return {name: np.asarray(value)[0] for name, value in features.items()}


def _decode(rows) -> list[str]:
    return [bytes(int(value) for value in row if value).decode() for row in rows]


def test_multi_ccd_ligand_tokens_match_upstream(built) -> None:
    atoms = int(built["atom_attention_mask"].sum())
    assert atoms == 27
    for name, expected in _UPSTREAM.items():
        actual = built[name]
        if name in {"ref_space_uid", "atom_to_token"}:
            actual = actual[:atoms]
        assert actual.tolist() == expected, name
    assert _decode(built["token_residue_name_chars"]) == _UPSTREAM_RESIDUE_NAMES
    ligand_atoms = [
        "".join(chr(int(code) + 32) for code in row if code)
        for row in built["ref_atom_name_chars"][18:atoms]
    ]
    assert ligand_atoms == _UPSTREAM_LIGAND_ATOMS
    np.testing.assert_array_equal(
        built["ref_pos"][18:atoms], np.asarray(_UPSTREAM_LIGAND_REF_POS, np.float32)
    )


def test_multi_ccd_ligand_bonds_match_upstream(built) -> None:
    edges = {
        (int(left), int(right))
        for left, right in np.argwhere(np.triu(built["token_bonds"][..., 0]))
    }
    assert edges == _UPSTREAM_BOND_EDGES
