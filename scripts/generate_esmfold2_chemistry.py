"""Emit `data/chemistry.py` from the installed upstream tables.

Generated rather than transcribed: 350 lines of reference-conformer
coordinates copied by hand is 350 chances to be quietly wrong.
"""
import pprint
import subprocess

from transformers.models.esmfold2 import protein_utils as up

HEADER = '''"""ESMFold2's protein chemistry tables, vendored so nothing needs torch.

Copied verbatim from Biohub's `transformers.models.esmfold2.protein_utils`
(Apache-2.0, Copyright 2026 Biohub) by `scripts/generate_esmfold2_chemistry.py`,
which reads the installed module rather than transcribing it -- the reference
conformer block alone is three hundred lines of coordinates, and a hand copy is
three hundred chances to be silently wrong. Upstream's file imports torch for
its featuriser; only the tables are taken, so this one imports nothing.

`tests/models/esmfold2/test_chemistry.py` re-derives every entry from the
installed module when it is present, so a checkout that has upstream cannot
drift from it and one that does not still runs.

Do not edit by hand. Regenerate.
"""

from __future__ import annotations

'''

def emit(name, value):
    return f"{name} = {pprint.pformat(value, width=84, sort_dicts=False)}\n\n"

parts = [HEADER]
parts.append("#: `mol_type`'s protein code, the unknown-residue index, the MSA gap.\n")
parts.append(f"MOL_TYPE_PROTEIN = {up.MOL_TYPE_PROTEIN}\n")
parts.append(f"PROTEIN_UNK_RES_TYPE = {up.PROTEIN_UNK_RES_TYPE}\n")
parts.append(f"MSA_GAP_TOKEN_ID = {up.MSA_GAP_TOKEN_ID}\n\n")
for name in (
    "PROTEIN_1TO3",
    "PROTEIN_RESIDUE_TO_RES_TYPE",
    "ESM_PROTEIN_VOCAB",
    "PROTEIN_HEAVY_ATOMS",
    "PROTEIN_REF_POS",
    "PROTEIN_CHARGED_ATOMS",
):
    parts.append(emit(name, getattr(up, name)))
parts.append(emit("PROTEIN_ELEMENT_TO_ATOMIC_NUM", up._PROTEIN_ELEMENT_TO_ATOMIC_NUM))
parts.append('''#: res_type index -> three-letter name, and the one-letter reverse map.
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
path = "src/foldjax/models/esmfold2/data/chemistry.py"
open(path, "w").write(out)
# pprint's line breaks are its own; the repository's are ruff's.
subprocess.run(["ruff", "format", path], check=True)
subprocess.run(["ruff", "check", "--fix", path], check=True)
print(path, len(open(path).read().splitlines()), "lines")
# The encoding must agree with upstream's private helper.
assert [up._encode_atom_name(n) for n in ("CA", "CB", "OXT", "N")] == [
    [ord(c) - 32 for c in n.ljust(4)[:4]] for n in ("CA", "CB", "OXT", "N")
]
print("atom-name encoding agrees with upstream")
