"""Small, internally consistent OpenFold3 feature batches for boundary tests."""

from __future__ import annotations

import numpy as np


def minimal_features(
    *, tokens: int = 4, atoms: int = 7, msa_rows: int = 2, templates: int = 1
) -> dict[str, np.ndarray]:
    if not (1 <= tokens <= atoms <= tokens * 23):
        raise ValueError("fixture needs 1 <= tokens <= atoms <= tokens * 23")
    base, extra = divmod(atoms, tokens)
    counts = np.full(tokens, base, dtype=np.int32)
    counts[:extra] += 1
    starts = np.concatenate(
        (np.zeros(1, dtype=np.int32), np.cumsum(counts[:-1], dtype=np.int32))
    )
    owners = np.repeat(np.arange(tokens, dtype=np.int32), counts)
    token = (1, tokens)
    atom = (1, atoms)

    restype = np.zeros((1, tokens, 32), dtype=np.int32)
    restype[..., 0] = 1
    ref_element = np.zeros((1, atoms, 119), dtype=np.int32)
    ref_element[..., 5] = 1
    ref_names = np.zeros((1, atoms, 4, 64), dtype=np.int32)
    for position, character in enumerate("C".ljust(4)):
        ref_names[:, :, position, ord(character) - 32] = 1
    msa = np.zeros((1, msa_rows, tokens, 32), dtype=np.int32)
    msa[..., 0] = 1
    profile = np.zeros((1, tokens, 32), dtype=np.float32)
    profile[..., 0] = 1
    template_restype = np.zeros((1, templates, tokens, 32), dtype=np.int32)
    template_restype[..., 0] = 1
    slots = (
        np.arange(23, dtype=np.int32)[None, None, :] < counts[None, :, None]
    ).reshape(1, tokens * 23)

    return {
        "asym_id": np.ones(token, dtype=np.int32),
        "atom_mask": np.ones(atom, dtype=np.float32),
        "atom_to_token_index": owners[None],
        "deletion_mean": np.zeros(token, dtype=np.float32),
        "deletion_value": np.zeros((1, msa_rows, tokens), dtype=np.float32),
        "entity_id": np.ones(token, dtype=np.int32),
        "has_deletion": np.zeros((1, msa_rows, tokens), dtype=np.float32),
        "is_atomized": np.zeros(token, dtype=np.int32),
        "is_dna": np.zeros(token, dtype=np.int32),
        "is_protein": np.ones(token, dtype=np.int32),
        "is_rna": np.zeros(token, dtype=np.int32),
        "max_atom_per_token_mask": slots.astype(np.float32),
        "msa": msa,
        "msa_mask": np.ones((1, msa_rows, tokens), dtype=np.float32),
        "num_atoms_per_token": counts[None],
        "profile": profile,
        "ref_atom_name_chars": ref_names,
        "ref_charge": np.zeros(atom, dtype=np.float32),
        "ref_element": ref_element,
        "ref_mask": np.ones(atom, dtype=np.int32),
        "ref_pos": np.zeros((1, atoms, 3), dtype=np.float32),
        "ref_space_uid": owners[None],
        "residue_index": np.arange(1, tokens + 1, dtype=np.int32)[None],
        "restype": restype,
        "start_atom_index": starts[None],
        "sym_id": np.ones(token, dtype=np.int32),
        "template_backbone_frame_mask": np.ones(
            (1, templates, tokens), dtype=np.float32
        ),
        "template_distogram": np.zeros(
            (1, templates, tokens, tokens, 39), dtype=np.float32
        ),
        "template_pseudo_beta_mask": np.ones(
            (1, templates, tokens), dtype=np.float32
        ),
        "template_restype": template_restype,
        "template_unit_vector": np.zeros(
            (1, templates, tokens, tokens, 3), dtype=np.float32
        ),
        "token_bonds": np.zeros((1, tokens, tokens), dtype=np.int32),
        "token_index": np.arange(tokens, dtype=np.int32)[None],
        "token_mask": np.ones(token, dtype=np.float32),
    }
