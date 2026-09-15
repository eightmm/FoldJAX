"""Emit `data/chemistry.py` from the installed upstream tables.

Generated rather than transcribed: 350 lines of reference-conformer
coordinates copied by hand is 350 chances to be quietly wrong.

Seven of upstream's tables are *not* emitted as literals: `data/
all_atom_constants.py` already vendors them from Biohub's own ESMFold2
checkout, so the generated module derives those from it and this script
asserts here that the two sources still agree -- MSE-free on this side,
because `all_atom_constants` maps `MSE` onto methionine and upstream's
featuriser tables have no such row. A disagreement anywhere else fails the
generation instead of landing a second copy that has drifted.
"""

import importlib.util
import pprint
import subprocess

from transformers.models.esmfold2 import protein_utils as up

CONSTANTS_PATH = "src/foldjax/models/esmfold2/data/all_atom_constants.py"
PATH = "src/foldjax/models/esmfold2/data/chemistry.py"

HEADER = '''"""ESMFold2's protein chemistry tables, vendored so nothing needs torch.

The reference conformers, charges and element numbers below are copied verbatim
from Biohub's `transformers.models.esmfold2.protein_utils` (Apache-2.0,
Copyright 2026 Biohub) by `scripts/generate_esmfold2_chemistry.py`, which reads
the installed module rather than transcribing it -- the conformer block alone is
three hundred lines of coordinates, and a hand copy is three hundred chances to
be silently wrong. Upstream's file imports torch for its featuriser; only the
tables are taken.

The residue vocabularies are not vendored twice. `all_atom_constants` already
carries the same seven, from Biohub's own ESMFold2 checkout, and the two sources
agree on every entry except one row: the canonical tables map `MSE`
(selenomethionine) onto methionine, and upstream's featuriser tables have no
such row. This module keeps them MSE-free, which is what its consumers are
written against -- `features.py` builds three-letter codes from a one-letter
sequence, so `PROTEIN_1TO3` never yields `MSE` and the bare `PROTEIN_HEAVY_ATOMS`
subscript beside it cannot miss, while the structure path normalises `MSE` to
`MET` before it looks anything up (`all_atom.py:287`). Keeping the row out is
also what keeps `RES_TYPE_TO_3LETTER[14]` at `MET`: `MSE` and `MET` share that
index, so inverting a table that carried `MSE` last would name selenomethionine
in every methionine of every output PDB (`pdb.py:117`).

`tests/models/esmfold2/test_chemistry.py` re-derives every entry from the
installed module when it is present, so a checkout that has upstream cannot
drift from it and one that does not still runs.

The literal tables below are generated. Do not edit those by hand. Regenerate.
"""

from __future__ import annotations

from foldjax.models.esmfold2.data import all_atom_constants as _constants

#: `mol_type`'s protein code, the unknown-residue index, the MSA gap.
MOL_TYPE_PROTEIN = _constants.MOL_TYPE_PROTEIN
PROTEIN_UNK_RES_TYPE = _constants.PROTEIN_UNK_RES_TYPE
MSA_GAP_TOKEN_ID = _constants.MSA_GAP_TOKEN_ID

PROTEIN_1TO3 = _constants.PROTEIN_1TO3
ESM_PROTEIN_VOCAB = _constants.ESM_PROTEIN_VOCAB

#: The two canonical tables that carry a row this path never reaches, without it.
PROTEIN_RESIDUE_TO_RES_TYPE = {
    name: res_type
    for name, res_type in _constants.PROTEIN_RESIDUE_TO_RES_TYPE.items()
    if name != "MSE"
}
PROTEIN_HEAVY_ATOMS = {
    name: atoms
    for name, atoms in _constants.PROTEIN_HEAVY_ATOMS.items()
    if name != "MSE"
}

'''


def load_constants():
    """Read the canonical tables from the file, not through the package.

    `all_atom_constants` imports nothing, so loading it by path keeps this
    script independent of whether `foldjax` is installed in the environment
    that has upstream.
    """
    spec = importlib.util.spec_from_file_location("_all_atom_constants", CONSTANTS_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def without_mse(table):
    return {name: value for name, value in table.items() if name != "MSE"}


constants = load_constants()
# The derivation the generated module performs, checked against upstream here.
for name in ("MOL_TYPE_PROTEIN", "PROTEIN_UNK_RES_TYPE", "MSA_GAP_TOKEN_ID"):
    assert getattr(up, name) == getattr(constants, name), name
for name in ("PROTEIN_1TO3", "ESM_PROTEIN_VOCAB"):
    assert getattr(up, name) == getattr(constants, name), name
for name in ("PROTEIN_RESIDUE_TO_RES_TYPE", "PROTEIN_HEAVY_ATOMS"):
    assert getattr(up, name) == without_mse(getattr(constants, name)), name
    assert "MSE" not in getattr(up, name), name
print(f"the seven shared tables agree with {CONSTANTS_PATH}")


def emit(name, value):
    return f"{name} = {pprint.pformat(value, width=84, sort_dicts=False)}\n\n"


parts = [HEADER]
for name in ("PROTEIN_REF_POS", "PROTEIN_CHARGED_ATOMS"):
    parts.append(emit(name, getattr(up, name)))
parts.append(emit("PROTEIN_ELEMENT_TO_ATOMIC_NUM", up._PROTEIN_ELEMENT_TO_ATOMIC_NUM))
parts.append('''#: res_type index -> three-letter name, and the one-letter reverse map.
#: Built from this module's MSE-free table: `MSE` and `MET` share index 14, so
#: inverting one that carried `MSE` last would rename every methionine.
RES_TYPE_TO_3LETTER = {
    residue_type: name for name, residue_type in PROTEIN_RESIDUE_TO_RES_TYPE.items()
}
RES_TYPE_TO_3LETTER[PROTEIN_UNK_RES_TYPE] = "UNK"

#: res_type index -> one-letter code, for recognising an a3m's query row.
RES_TYPE_TO_LETTER: dict[int, str] = {}
for _letter, _residue in PROTEIN_1TO3.items():
    _index = PROTEIN_RESIDUE_TO_RES_TYPE.get(_residue)
    if _index is not None:
        RES_TYPE_TO_LETTER.setdefault(_index, _letter)


def encode_atom_name(name: str) -> list[int]:
    """Upstream's four-character atom-name encoding: `ord(c) - 32`, zero-padded."""
    return [ord(character) - 32 for character in name.ljust(4)[:4]]
''')
out = "".join(parts)
open(PATH, "w").write(out)
# pprint's line breaks are its own; the repository's are ruff's.
subprocess.run(["ruff", "format", PATH], check=True)
subprocess.run(["ruff", "check", "--fix", PATH], check=True)
print(PATH, len(open(PATH).read().splitlines()), "lines")
# The encoding must agree with upstream's private helper.
assert [up._encode_atom_name(n) for n in ("CA", "CB", "OXT", "N")] == [
    [ord(c) - 32 for c in n.ljust(4)[:4]] for n in ("CA", "CB", "OXT", "N")
]
print("atom-name encoding agrees with upstream")
