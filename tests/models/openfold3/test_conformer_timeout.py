"""Conformer embeds keep their timeout without `func_timeout`'s thread per call."""

from __future__ import annotations

import importlib
import random

import pytest

pytest.importorskip("rdkit")
pytest.importorskip("func_timeout")

from rdkit import Chem  # noqa: E402
from rdkit.Chem import AllChem  # noqa: E402

conformer = importlib.import_module(
    "foldjax.models.openfold3._upstream.openfold3.core.data.primitives.structure.conformer"
)

_RESIDUES = ("NCC(=O)O", "CC(C)C(N)C(=O)O", "OC(=O)C(N)CC1=CNC2=CC=CC=C12")


@pytest.mark.parametrize("smiles", _RESIDUES)
def test_a_timeout_leaves_the_conformer_unchanged(smiles) -> None:
    mol = Chem.MolFromSmiles(smiles)
    for seed in (0, 7, 2026):
        random.seed(seed)
        timed, timed_id = conformer._compute_conformer(Chem.Mol(mol), timeout=120)
        random.seed(seed)
        free, free_id = conformer._compute_conformer(Chem.Mol(mol), timeout=None)
        assert timed_id == free_id
        assert (
            timed.GetConformer(timed_id).GetPositions().tobytes()
            == free.GetConformer(free_id).GetPositions().tobytes()
        )


def test_rdkit_enforces_the_timeout_and_a_timeout_falls_through(monkeypatch) -> None:
    seen, real = [], AllChem.EmbedMolecule

    def embed(mol, strategy):
        seen.append(strategy.timeout)
        return -1 if len(seen) == 1 else real(mol, strategy)

    monkeypatch.setattr(conformer.AllChem, "EmbedMolecule", embed)
    result = conformer.multistrategy_compute_conformer(
        Chem.MolFromSmiles("CC(N)C(=O)O"), timeouts={"default": 0.4}
    )
    assert seen[0] == 1
    assert result.strategy == "small_ring_torsions"
