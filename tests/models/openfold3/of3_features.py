"""A featurized batch for the composite parity tests.

Upstream has ``openfold3.tests.utils.data_utils.random_of3_features``, but
importing it drags in the CCD/cache layer (pdbeccdutils -> rdkit -> lmdb ->
pandas -> memory_profiler), none of which random feature generation uses.
Installing that stack into a test-only extra is a large, pin-sensitive change,
and stubbing it turned out to require stubs that swallow module-level *calls* --
which would leave upstream's data code silently inert.

So the batch is built here instead, from the key/shape/dtype spec in upstream's
generator. That is safe because this helper is not the authority: upstream's own
model consumes the batch, so a malformed feature surfaces as a crash or a NaN in
the reference output rather than as a passing test. The two constants that encode
real chemistry are imported from upstream rather than copied, and the atom/token
index mapping uses upstream's own ``broadcast_token_feat_to_atoms``.
"""

from __future__ import annotations

from typing import Any


def make_batch(
    *,
    n_token: int = 12,
    n_msa: int = 6,
    n_templ: int = 2,
    n_chain: int = 2,
    batch_size: int = 1,
    seed: int = 0,
) -> dict[str, Any]:
    """Build one batch. Returns torch tensors, matching upstream's generator."""
    import torch
    from openfold3.core.data.resources.residues import (
        STANDARD_PROTEIN_RESIDUES_3,
        STANDARD_RESIDUES_3,
        STANDARD_RESIDUES_WITH_GAP_3,
    )
    from openfold3.core.data.resources.token_atom_constants import (
        TOKEN_NAME_TO_ATOM_NAMES,
    )
    from openfold3.core.utils.atomize_utils import broadcast_token_feat_to_atoms

    generator = torch.Generator().manual_seed(seed)
    restype_idx = torch.randint(
        0, len(STANDARD_RESIDUES_3), (n_token,), generator=generator
    )
    names = [STANDARD_RESIDUES_3[index] for index in restype_idx]
    restype_one_hot = torch.nn.functional.one_hot(
        restype_idx, len(STANDARD_RESIDUES_WITH_GAP_3)
    )
    num_atoms_per_token = torch.tensor(
        [float(len(TOKEN_NAME_TO_ATOM_NAMES[name])) for name in names]
    )
    is_protein = torch.tensor(
        [1.0 if name in STANDARD_PROTEIN_RESIDUES_3 else 0.0 for name in names]
    )
    n_atom = int(num_atoms_per_token.sum().item())

    token_mask = torch.ones(n_token)
    # Chains split evenly; asym_id is 1-based, as upstream's random_asym_ids is.
    asym_id = (torch.arange(n_token) * n_chain // n_token + 1).float()
    start_atom_index = torch.cat(
        (torch.zeros(1), torch.cumsum(num_atoms_per_token, dim=-1)[:-1])
    )
    atom_to_token_index = broadcast_token_feat_to_atoms(
        token_mask=token_mask,
        num_atoms_per_token=num_atoms_per_token,
        token_feat=torch.arange(n_token).float(),
    ).int()

    def batched(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.unsqueeze(0).repeat((batch_size, *([1] * tensor.dim())))

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(shape, generator=generator)

    return {
        "residue_index": batched(torch.arange(n_token)).int(),
        "token_index": batched(torch.arange(n_token)).int(),
        "asym_id": batched(asym_id).int(),
        "entity_id": batched(asym_id).int(),
        "sym_id": torch.ones((batch_size, n_token)).int(),
        "restype": batched(restype_one_hot).int(),
        "is_protein": batched(is_protein).int(),
        "is_dna": torch.zeros((batch_size, n_token)).int(),
        "is_rna": torch.zeros((batch_size, n_token)).int(),
        "is_ligand": torch.zeros((batch_size, n_token)).int(),
        "is_atomized": torch.zeros((batch_size, n_token)).int(),
        # Reference conformer features.
        "ref_pos": randn(batch_size, n_atom, 3),
        "ref_mask": torch.ones((batch_size, n_atom)).int(),
        "ref_element": torch.ones((batch_size, n_atom, 119)).int(),
        "ref_charge": torch.ones((batch_size, n_atom)),
        "ref_atom_name_chars": torch.ones((batch_size, n_atom, 4, 64)).int(),
        "ref_space_uid": batched(atom_to_token_index),
        # MSA features.
        "msa": torch.ones((batch_size, n_msa, n_token, 32)).int(),
        "has_deletion": torch.ones((batch_size, n_msa, n_token)),
        "deletion_value": torch.ones((batch_size, n_msa, n_token)),
        "profile": torch.ones((batch_size, n_token, 32)),
        "deletion_mean": torch.ones((batch_size, n_token)),
        "msa_mask": torch.ones((batch_size, n_msa, n_token)),
        "num_paired_seqs": torch.full((batch_size,), max(1, n_msa // 4)),
        # Template features.
        "template_restype": torch.ones((batch_size, n_templ, n_token, 32)).int(),
        "template_pseudo_beta_mask": torch.ones((batch_size, n_templ, n_token)),
        "template_backbone_frame_mask": torch.ones((batch_size, n_templ, n_token)),
        "template_distogram": torch.ones(
            (batch_size, n_templ, n_token, n_token, 39)
        ),
        "template_unit_vector": torch.ones((batch_size, n_templ, n_token, n_token, 3)),
        # Bonds.
        "token_bonds": torch.ones((batch_size, n_token, n_token)).int(),
        # Shapes and masks.
        "token_mask": batched(token_mask),
        "atom_mask": torch.ones((batch_size, n_atom)),
        "start_atom_index": batched(start_atom_index).int(),
        "num_atoms_per_token": batched(num_atoms_per_token).int(),
        "atom_to_token_index": batched(atom_to_token_index),
    }
