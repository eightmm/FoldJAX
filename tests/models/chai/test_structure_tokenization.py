from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest

from foldjax.models.chai.bridge.conformer_io import ConformerData
from foldjax.models.chai.data.input import EntityType, Input
from foldjax.models.chai.data.structure import tokenize_inputs


def _conformer(atom_names: tuple[str, ...]) -> ConformerData:
    count = len(atom_names)
    return ConformerData(
        position=np.arange(count * 3, dtype=np.float32).reshape(count, 3),
        element=np.full(count, 6, np.int32),
        charge=np.zeros(count, np.int32),
        atom_names=atom_names,
        bonds=(),
        symmetries=np.arange(count, dtype=np.int64)[:, None],
    )


def test_standard_protein_chains_match_chai_index_contract() -> None:
    conformers = {
        "ALA": _conformer(("N", "CA", "C", "O", "CB")),
        "GLY": _conformer(("N", "CA", "C", "O")),
    }
    inputs = [
        Input("AG", EntityType.PROTEIN.value, "first"),
        Input("AG", EntityType.PROTEIN.value, "second"),
    ]

    context = tokenize_inputs(inputs, conformers, identifier="test")

    np.testing.assert_array_equal(context.token_residue_type, [0, 7, 0, 7])
    np.testing.assert_array_equal(context.token_residue_index, [0, 1, 0, 1])
    np.testing.assert_array_equal(context.token_asym_id, [1, 1, 2, 2])
    np.testing.assert_array_equal(context.token_entity_id, [0, 0, 0, 0])
    np.testing.assert_array_equal(context.token_sym_id, [0, 0, 1, 1])
    np.testing.assert_array_equal(context.token_centre_atom_index, [1, 6, 10, 15])
    np.testing.assert_array_equal(context.token_ref_atom_index, [4, 6, 13, 15])
    np.testing.assert_array_equal(
        context.token_backbone_frame_index,
        [[0, 1, 2], [5, 6, 7], [2, 3, 4], [7, 8, 9]],
    )
    np.testing.assert_array_equal(
        context.atom_token_index, [0] * 5 + [1] * 4 + [2] * 5 + [3] * 4
    )
    np.testing.assert_array_equal(
        context.atom_ref_space_uid, [0] * 5 + [1] * 4 + [9] * 5 + [10] * 4
    )
    assert context.atom_ref_space_uid.dtype == np.int64
    assert context.symmetries.dtype == np.int64
    assert context.atom_ref_name[:5] == ("N", "CA", "C", "O", "CB")


def test_model_chain_names_reject_more_than_four_ascii_bytes() -> None:
    with pytest.raises(ValueError, match="does not fit length 4"):
        tokenize_inputs(
            [Input("A", EntityType.PROTEIN.value, "receptor")],
            {"ALA": _conformer(("N", "CA", "C", "O", "CB"))},
            entity_name_as_subchain=True,
        )


def test_rna_and_dna_use_chai_restype_and_atom36_indices() -> None:
    conformers = {
        "A": _conformer(("P", "C1'", "C3'", "C4'", "C4")),
        "DT": _conformer(("P", "C1'", "C3'", "C4'", "C2")),
    }

    context = tokenize_inputs(
        [
            Input("A", EntityType.RNA.value, "rna"),
            Input("T", EntityType.DNA.value, "dna"),
        ],
        conformers,
    )

    np.testing.assert_array_equal(context.token_residue_type, [21, 29])
    np.testing.assert_array_equal(
        context.atom_within_token_index, [4, 1, 9, 8, 15, 4, 1, 9, 8, 12]
    )
    np.testing.assert_array_equal(context.token_backbone_frame_mask, [True, True])


def test_padding_matches_structure_mask_sentinels() -> None:
    context = tokenize_inputs(
        [Input("A", EntityType.PROTEIN.value, "protein")],
        {"ALA": _conformer(("N", "CA", "C", "O", "CB"))},
    )

    padded = context.pad(4, 8)

    assert padded.num_tokens == 4
    assert padded.num_atoms == 8
    np.testing.assert_array_equal(padded.token_exists_mask, [True, False, False, False])
    np.testing.assert_array_equal(
        padded.atom_is_not_padding_mask, [True] * 5 + [False] * 3
    )
    np.testing.assert_array_equal(padded.atom_ref_space_uid[-3:], [-1, -1, -1])
    np.testing.assert_array_equal(padded.symmetries[-3:], -1)


def test_modified_residue_uses_chai_per_atom_tokenization() -> None:
    context = tokenize_inputs(
        [Input("A(MSE)", EntityType.PROTEIN.value, "modified")],
        {
            "ALA": _conformer(("N", "CA", "C", "O", "CB")),
            "MSE": _conformer(("N", "CA", "C", "O", "CB", "SE")),
        },
        rng=np.random.default_rng(0),
    )

    np.testing.assert_array_equal(context.token_residue_type, [0] + [20] * 6)
    np.testing.assert_array_equal(context.token_residue_index, [0] + [1] * 6)
    np.testing.assert_array_equal(context.atom_token_index, [0] * 5 + list(range(1, 7)))
    np.testing.assert_array_equal(
        context.atom_within_token_index, [0, 1, 2, 4, 3] + [0] * 6
    )


@pytest.mark.official_parity
def test_standard_polymer_structure_matches_official_chai(
    tmp_path: Path, upstream_chai_dir: Path, upstream_chai_python: Path
) -> None:
    conformers = {
        "ALA": _conformer(("N", "CA", "C", "O", "CB")),
        "GLY": _conformer(("N", "CA", "C", "O")),
        "A": _conformer(("P", "C1'", "C3'", "C4'", "C4")),
        "DT": _conformer(("P", "C1'", "C3'", "C4'", "C2")),
    }
    inputs = [
        Input("AG", EntityType.PROTEIN.value, "first"),
        Input("AG", EntityType.PROTEIN.value, "second"),
        Input("A", EntityType.RNA.value, "rna"),
        Input("T", EntityType.DNA.value, "dna"),
    ]
    actual = tokenize_inputs(inputs, conformers)
    output_path = tmp_path / "official.npz"
    script = r"""
import sys
import numpy as np
import torch
from chai_lab.data.dataset.inference_dataset import Input, raw_inputs_to_entitites_data
from chai_lab.data.dataset.structure.all_atom_residue_tokenizer import (
    AllAtomResidueTokenizer, _make_sym_ids,
)
from chai_lab.data.dataset.structure.all_atom_structure_context import (
    AllAtomStructureContext,
)
from chai_lab.data.parsing.structure.residue import ConformerData

def conf(names):
    n = len(names)
    return ConformerData(
        torch.arange(n * 3, dtype=torch.float32).reshape(n, 3),
        torch.full((n,), 6, dtype=torch.int32),
        torch.zeros(n, dtype=torch.int32),
        names, [], torch.arange(n, dtype=torch.int64)[:, None],
    )

class FakeGenerator:
    conformers = {
        "ALA": conf(["N", "CA", "C", "O", "CB"]),
        "GLY": conf(["N", "CA", "C", "O"]),
        "A": conf(["P", "C1'", "C3'", "C4'", "C4"]),
        "DT": conf(["P", "C1'", "C3'", "C4'", "C2"]),
    }
    def get(self, name):
        return self.conformers.get(name)

inputs = [Input("AG", 0, "first"), Input("AG", 0, "second"),
          Input("A", 1, "rna"), Input("T", 2, "dna")]
entities = raw_inputs_to_entitites_data(inputs)
tokenizer = AllAtomResidueTokenizer(FakeGenerator())
sym_ids = _make_sym_ids([entity.entity_id for entity in entities])
contexts = [tokenizer._tokenize_entity(entity, chain_id=i + 1, sym_id=sym)
            for i, (entity, sym) in enumerate(zip(entities, sym_ids))]
ctx = AllAtomStructureContext.merge(contexts)
names = sys.argv[2:]
np.savez(sys.argv[1], **{name: getattr(ctx, name).numpy() for name in names})
"""
    names = [
        name
        for name, value in actual.to_dict().items()
        if isinstance(value, np.ndarray)
        and name not in {"resolution", "is_distillation"}
    ]
    subprocess.run(
        [str(upstream_chai_python), "-c", script, str(output_path), *names],
        cwd=upstream_chai_dir,
        check=True,
        capture_output=True,
        text=True,
    )
    with np.load(output_path) as expected:
        for name in names:
            assert getattr(actual, name).dtype == expected[name].dtype, name
            np.testing.assert_array_equal(getattr(actual, name), expected[name])
