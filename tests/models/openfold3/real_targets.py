"""Real RCSB targets, featurized once per session.

Featurization of a real entry costs seconds and every test that needs one needs the
same one, so this caches per (pdb_id, ligands) rather than refeaturizing. Nothing
here is synthetic: the point of using depositions is that they carry the gaps,
repeated chains, metals and mixed molecule types that a hand-built batch does not.
"""

from __future__ import annotations

import functools
from importlib.util import find_spec

import pytest

# The loader lives in `tests/_foldbench` now, so it is always importable and the
# old "is the sibling package installed?" gate would never fire again -- a check
# that cannot fail is worse than no check, because it reads as coverage.
#
# What these tests actually need is what that gate was standing in for: biotite,
# to parse a deposition, and the network, to fetch one. Entries are cached under
# `$FOLDBENCH_CACHE` (else XDG) after the first run, so a warm machine needs only
# the first.
requires_real_targets = pytest.mark.skipif(
    find_spec("biotite") is None,
    reason="needs biotite to parse a deposition: "
    "uv sync --extra openfold3-preprocess",
)

# Smallest useful ladder for the default suite: a DNA duplex, a protein monomer, an
# RNA with metals, and a protein with an organic ligand. The full set (up to ~850
# tokens) is exercised by scripts/compare_real.py, which is too slow for a suite.
SUITE_LADDER = ("1BNA", "1UBQ", "1EHZ", "1CBS")


@functools.cache
def featurized(pdb_id: str, include_ligands: bool = True) -> dict:
    """Fetch, parse and featurize one deposition."""
    from foldjax.models.openfold3.data import featurize_query
    from tests._foldbench import load_target, openfold3_query

    target = load_target(pdb_id)
    return featurize_query(openfold3_query(target, include_ligands=include_ligands))


@functools.cache
def described(pdb_id: str) -> str:
    from tests._foldbench import load_target

    return load_target(pdb_id).describe()
