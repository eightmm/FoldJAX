from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest

from foldjax.models.chai.bridge.conformer_io import load_native_conformers
from foldjax.models.chai.data.chemistry import CovalentBond
from foldjax.models.chai.data.input import EntityType, Input
from foldjax.models.chai.data.structure import tokenize_inputs

pytestmark = pytest.mark.official_parity

def _export_upstream(
    path: Path, upstream_chai_dir: Path, upstream_chai_python: Path
) -> None:
    script = r"""
import sys
import numpy as np
import torch
from chai_lab.data.dataset.inference_dataset import Input, load_chains_from_raw
from chai_lab.data.dataset.structure.all_atom_structure_context import (
    AllAtomStructureContext,
)
from chai_lab.data.dataset.structure.bond_utils import (
    get_atom_covalent_bond_pairs_from_constraints,
)
from chai_lab.data.parsing.restraints import (
    PairwiseInteraction,
    PairwiseInteractionType,
)
from chai_lab.data.parsing.structure.entity_type import EntityType

torch.manual_seed(0)
inputs = [
    Input("AC(SEP)", EntityType.PROTEIN.value, "protein"),
    Input("CS", EntityType.LIGAND.value, "ligand"),
    Input("NAG(4-1 NAG)", EntityType.MANUAL_GLYCAN.value, "glycan"),
]
chains = load_chains_from_raw(inputs)
ctx = AllAtomStructureContext.merge([chain.structure_context for chain in chains])
bond = PairwiseInteraction(
    chainA="A", res_idxA="C2", atom_nameA="SG",
    chainB="B", res_idxB="", atom_nameB="S1",
    connection_type=PairwiseInteractionType.COVALENT,
)
manual = get_atom_covalent_bond_pairs_from_constraints(
    [bond], ctx.token_residue_index, ctx.token_residue_name,
    ctx.subchain_id, ctx.token_asym_id, ctx.atom_token_index, ctx.atom_ref_name,
)
left = torch.cat([ctx.atom_covalent_bond_indices[0], manual[0]])
right = torch.cat([ctx.atom_covalent_bond_indices[1], manual[1]])
names = np.asarray(ctx.atom_ref_name, dtype="U8")
arrays = {
    key: value.detach().cpu().numpy()
    for key, value in vars(ctx).items()
    if torch.is_tensor(value)
}
arrays["atom_ref_name"] = names
arrays["bond_left"] = left.numpy()
arrays["bond_right"] = right.numpy()
np.savez(sys.argv[1], **arrays)
"""
    subprocess.run(
        [str(upstream_chai_python), "-c", script, str(path)],
        cwd=upstream_chai_dir,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_public_chemistry_inputs_match_upstream_chai(
    tmp_path: Path,
    native_conformer_path: Path,
    upstream_chai_dir: Path,
    upstream_chai_python: Path,
) -> None:
    expected_path = tmp_path / "upstream.npz"
    _export_upstream(expected_path, upstream_chai_dir, upstream_chai_python)
    conformers = load_native_conformers(native_conformer_path)
    actual = tokenize_inputs(
        [
            Input("AC(SEP)", EntityType.PROTEIN.value, "protein"),
            Input("CS", EntityType.LIGAND.value, "ligand"),
            Input("NAG(4-1 NAG)", EntityType.MANUAL_GLYCAN.value, "glycan"),
        ],
        conformers,
        covalent_bonds=[CovalentBond("A", "C2", "SG", "B", "", "S1")],
    )

    exact_fields = (
        "token_residue_type",
        "token_residue_index",
        "token_index",
        "token_centre_atom_index",
        "token_ref_atom_index",
        "token_exists_mask",
        "token_backbone_frame_mask",
        "token_backbone_frame_index",
        "token_asym_id",
        "token_entity_id",
        "token_sym_id",
        "token_entity_type",
        "token_residue_name",
        "atom_token_index",
        "atom_within_token_index",
        "atom_ref_mask",
        "atom_ref_element",
        "atom_ref_charge",
        "atom_ref_name_chars",
        "atom_ref_space_uid",
        "atom_is_not_padding_mask",
        "symmetries",
    )
    with np.load(expected_path, allow_pickle=False) as expected:
        for field in exact_fields:
            np.testing.assert_array_equal(getattr(actual, field), expected[field])
        np.testing.assert_array_equal(
            np.asarray(actual.atom_ref_name), expected["atom_ref_name"]
        )
        np.testing.assert_array_equal(
            actual.atom_covalent_bond_indices[0], expected["bond_left"]
        )
        np.testing.assert_array_equal(
            actual.atom_covalent_bond_indices[1], expected["bond_right"]
        )

        # Cached non-standard conformers are randomly rigid-transformed upstream.
        # Pairwise distances prove that the exact reference geometry is preserved.
        for entity_type in (EntityType.PROTEIN.value, EntityType.MANUAL_GLYCAN.value):
            token_mask = actual.token_entity_type == entity_type
            entity_atom_mask = np.isin(
                actual.atom_token_index, np.flatnonzero(token_mask)
            )
            for reference_space in np.unique(
                actual.atom_ref_space_uid[entity_atom_mask]
            ):
                atom_mask = entity_atom_mask & (
                    actual.atom_ref_space_uid == reference_space
                )
                got = actual.atom_ref_pos[atom_mask]
                want = expected["atom_ref_pos"][atom_mask]
                np.testing.assert_allclose(
                    np.linalg.norm(got[:, None] - got[None, :], axis=-1),
                    np.linalg.norm(want[:, None] - want[None, :], axis=-1),
                    atol=2e-4,
                    rtol=2e-5,
                )

        ligand_tokens = np.flatnonzero(
            actual.token_entity_type == EntityType.LIGAND.value
        )
        ligand_atoms = np.isin(actual.atom_token_index, ligand_tokens)
        np.testing.assert_allclose(
            actual.atom_ref_pos[ligand_atoms],
            expected["atom_ref_pos"][ligand_atoms],
            atol=1e-6,
            rtol=1e-6,
        )
