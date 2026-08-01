from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from foldjax.models.chai.bridge.conformer_io import ConformerData
from foldjax.models.chai.data.chemistry import (
    CovalentBond,
    generate_ligand_conformer,
    parse_covalent_bonds_csv,
    parse_glycan,
)
from foldjax.models.chai.data.input import EntityType, Input
from foldjax.models.chai.data.structure import tokenize_inputs


def _conformer(names: tuple[str, ...]) -> ConformerData:
    n = len(names)
    return ConformerData(
        position=np.arange(n * 3, dtype=np.float32).reshape(n, 3),
        element=np.full(n, 6, np.int32),
        charge=np.zeros(n, np.int32),
        atom_names=names,
        bonds=tuple((i, i + 1) for i in range(n - 1)),
        symmetries=np.arange(n, dtype=np.int64)[:, None],
    )


def test_parse_branched_glycan_matches_chai_order_and_bonds() -> None:
    sugars, bonds = parse_glycan("MAN(6-1 FUC)(4-1 MAN)")

    assert sugars == ["MAN", "FUC", "MAN"]
    assert [(b.src_sugar_index, b.dst_sugar_index) for b in bonds] == [(0, 1), (0, 2)]
    assert [(b.src_atom_name, b.dst_atom_name) for b in bonds] == [
        ("O6", "C1"),
        ("O4", "C1"),
    ]


@pytest.mark.parametrize("value", ["", "NAG(", "NAG(7-1 MAN)", "NA"])
def test_invalid_glycan_fails_explicitly(value: str) -> None:
    with pytest.raises(ValueError):
        parse_glycan(value)


def test_parse_covalent_csv_without_pandas(tmp_path: Path) -> None:
    path = tmp_path / "bonds.csv"
    path.write_text(
        "chainA,res_idxA,chainB,res_idxB,connection_type,confidence,"
        "min_distance_angstrom,max_distance_angstrom,comment,restraint_id\n"
        "A,C2@SG,B,@S1,covalent,1.0,0.0,0.0,test,bond1\n",
        encoding="utf-8",
    )

    assert parse_covalent_bonds_csv(path) == [
        CovalentBond("A", "C2", "SG", "B", "", "S1")
    ]


def test_modified_ligand_glycan_and_declared_bond_are_tokenized() -> None:
    conformers = {
        "ALA": _conformer(("N", "CA", "C", "CB")),
        "CYS": _conformer(("N", "CA", "C", "CB", "SG")),
        "SEP": _conformer(("N", "CA", "C", "CB", "P")),
        "NAG": _conformer(("C1", "O4", "C2")),
    }
    inputs = [
        Input("AC(SEP)", EntityType.PROTEIN.value, "protein"),
        Input("CS", EntityType.LIGAND.value, "ligand"),
        Input("NAG(4-1 NAG)", EntityType.MANUAL_GLYCAN.value, "glycan"),
    ]
    context = tokenize_inputs(
        inputs,
        conformers,
        covalent_bonds=[CovalentBond("A", "C2", "SG", "B", "", "S1")],
    )

    # ALA and CYS are residue tokens; SEP, ligand, and sugars are per-atom tokens.
    assert np.count_nonzero(context.token_entity_type == EntityType.PROTEIN.value) == 7
    assert np.count_nonzero(context.token_entity_type == EntityType.LIGAND.value) == 2
    assert (
        np.count_nonzero(context.token_entity_type == EntityType.MANUAL_GLYCAN.value)
        == 6
    )
    left, right = context.atom_covalent_bond_indices
    assert left.shape == right.shape == (2,)
    assert context.atom_ref_name[left[0]] == "O4"
    assert context.atom_ref_name[right[0]] == "C1"
    assert context.atom_ref_name[left[1]] == "SG"
    assert context.atom_ref_name[right[1]] == "S1"


def test_glycosidic_bond_drops_terminal_oxygen_leaving_group() -> None:
    conformer = ConformerData(
        position=np.asarray(
            [
                [0.0, 0.0, 0.0],  # C1
                [1.4, 0.0, 0.0],  # terminal O1 leaving group
                [0.0, 1.4, 0.0],  # C2
                [0.0, 2.8, 0.0],  # O4, away from C1
            ],
            np.float32,
        ),
        element=np.asarray([6, 8, 6, 8], np.int32),
        charge=np.zeros(4, np.int32),
        atom_names=("C1", "O1", "C2", "O4"),
        bonds=((0, 1), (0, 2), (2, 3)),
        symmetries=np.arange(4, dtype=np.int64)[:, None],
    )

    context = tokenize_inputs(
        [
            Input(
                "NAG(4-1 NAG)",
                EntityType.MANUAL_GLYCAN.value,
                "glycan",
            )
        ],
        {"NAG": conformer},
        rng=np.random.default_rng(0),
    )

    second_o1 = next(
        atom
        for atom, (token, name) in enumerate(
            zip(context.atom_token_index, context.atom_ref_name, strict=True)
        )
        if context.token_residue_index[token] == 1 and name == "O1"
    )
    assert not context.atom_exists_mask[second_o1]
    assert context.atom_is_not_padding_mask[second_o1]


def test_ligand_generation_is_deterministic_and_preserves_chemistry() -> None:
    first = generate_ligand_conformer("C[C@H](O)Cl")
    second = generate_ligand_conformer("C[C@H](O)Cl")

    np.testing.assert_allclose(first.position, second.position, atol=0, rtol=0)
    assert first.atom_names == ("C1", "C2", "O1", "CL1")
    np.testing.assert_array_equal(first.element, [6, 6, 8, 17])
    assert len(first.bonds) == 3
    assert first.symmetries.shape[0] == 4


def test_invalid_smiles_fails_explicitly() -> None:
    with pytest.raises(ValueError, match="Invalid SMILES"):
        generate_ligand_conformer("C1(not-smiles")
