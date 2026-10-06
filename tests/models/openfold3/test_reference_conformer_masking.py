"""Masked reference atoms are zeroed before centring, as upstream does.

Upstream OpenFold3 (`featurization/conformer.py:141-156`) zeroes an unused
atom's position -- NaN when the conformer gave it none -- before the random
centring, and skips the augmentation for a molecule with no used atom. A NaN
left in place reaches the masked mean (NaN * 0 is NaN) and poisons every atom
of the molecule.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("rdkit")

from rdkit import Chem  # noqa: E402
from rdkit.Geometry import Point3D  # noqa: E402

from foldjax.models.openfold3.data._numpy_featurization import (  # noqa: E402
    _conformer_features,
)


def _molecule(positions, used) -> SimpleNamespace:
    mol = Chem.RWMol()
    for index in range(len(positions)):
        atom = Chem.Atom(6)
        atom.SetBoolProp("annot_used_atom_mask", bool(used[index]))
        atom.SetProp("annot_atom_name", f"C{index}")
        mol.AddAtom(atom)
    conformer = Chem.Conformer(len(positions))
    for index, point in enumerate(positions):
        conformer.SetAtomPosition(index, Point3D(*point))
    mol = mol.GetMol()
    mol.AddConformer(conformer, assignId=True)
    return SimpleNamespace(mol=mol, in_crop_mask=[1] * len(positions))


def test_a_masked_nan_atom_does_not_poison_its_molecule() -> None:
    nan = float("nan")
    molecule = _molecule(
        [(0.0, 0.0, 0.0), (1.5, 0.0, 0.0), (nan, nan, nan)], used=[1, 1, 0]
    )
    features = _conformer_features([molecule], seed=0)
    ref_pos = features["ref_pos"]
    assert np.isfinite(ref_pos).all()
    np.testing.assert_array_equal(features["ref_mask"], [1, 1, 0])
    np.testing.assert_array_equal(ref_pos[2], np.zeros(3, np.float32))
    # The rigid augmentation keeps the used atoms' bond length.
    assert np.linalg.norm(ref_pos[0] - ref_pos[1]) == pytest.approx(1.5, abs=1e-5)


def test_an_all_nan_molecule_is_zeros_and_skips_augmentation() -> None:
    nan = float("nan")
    unused = _molecule([(nan, nan, nan), (nan, nan, nan)], used=[0, 0])
    used = _molecule([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)], used=[1, 1])
    alone = _conformer_features([used], seed=3)
    features = _conformer_features([unused, used], seed=3)
    np.testing.assert_array_equal(features["ref_pos"][:2], np.zeros((2, 3)))
    # No draw is spent on the skipped molecule: the next one is augmented
    # exactly as it would be on its own.
    np.testing.assert_array_equal(features["ref_pos"][2:], alone["ref_pos"])


def test_a_used_atom_without_a_position_is_refused() -> None:
    nan = float("nan")
    molecule = _molecule([(0.0, 0.0, 0.0), (nan, nan, nan)], used=[1, 1])
    with pytest.raises(ValueError, match="valid atoms"):
        _conformer_features([molecule], seed=0)
