"""Validation for the portable OpenFold3 feature-archive ABI."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

_EXPECTED_DTYPES = {
    "asym_id": np.dtype(np.int32),
    "atom_mask": np.dtype(np.float32),
    "atom_to_token_index": np.dtype(np.int32),
    "deletion_mean": np.dtype(np.float32),
    "deletion_value": np.dtype(np.float32),
    "entity_id": np.dtype(np.int32),
    "has_deletion": np.dtype(np.float32),
    "is_atomized": np.dtype(np.int32),
    "is_dna": np.dtype(np.int32),
    "is_protein": np.dtype(np.int32),
    "is_rna": np.dtype(np.int32),
    "max_atom_per_token_mask": np.dtype(np.float32),
    "msa": np.dtype(np.int32),
    "msa_mask": np.dtype(np.float32),
    "num_atoms_per_token": np.dtype(np.int32),
    "profile": np.dtype(np.float32),
    "ref_atom_name_chars": np.dtype(np.int32),
    "ref_charge": np.dtype(np.float32),
    "ref_element": np.dtype(np.int32),
    "ref_mask": np.dtype(np.int32),
    "ref_pos": np.dtype(np.float32),
    "ref_space_uid": np.dtype(np.int32),
    "residue_index": np.dtype(np.int32),
    "restype": np.dtype(np.int32),
    "start_atom_index": np.dtype(np.int32),
    "sym_id": np.dtype(np.int32),
    "template_backbone_frame_mask": np.dtype(np.float32),
    "template_distogram": np.dtype(np.float32),
    "template_pseudo_beta_mask": np.dtype(np.float32),
    "template_restype": np.dtype(np.int32),
    "template_unit_vector": np.dtype(np.float32),
    "token_bonds": np.dtype(np.int32),
    "token_index": np.dtype(np.int32),
    "token_mask": np.dtype(np.float32),
}


def _expected_shapes(
    *, tokens: int, atoms: int, msa_rows: int, templates: int
) -> dict[str, tuple[int, ...]]:
    token = (1, tokens)
    atom = (1, atoms)
    return {
        "asym_id": token,
        "atom_mask": atom,
        "atom_to_token_index": atom,
        "deletion_mean": token,
        "deletion_value": (1, msa_rows, tokens),
        "entity_id": token,
        "has_deletion": (1, msa_rows, tokens),
        "is_atomized": token,
        "is_dna": token,
        "is_protein": token,
        "is_rna": token,
        "max_atom_per_token_mask": (1, tokens * 23),
        "msa": (1, msa_rows, tokens, 32),
        "msa_mask": (1, msa_rows, tokens),
        "num_atoms_per_token": token,
        "profile": (1, tokens, 32),
        "ref_atom_name_chars": (1, atoms, 4, 64),
        "ref_charge": atom,
        "ref_element": (1, atoms, 119),
        "ref_mask": atom,
        "ref_pos": (1, atoms, 3),
        "ref_space_uid": atom,
        "residue_index": token,
        "restype": (1, tokens, 32),
        "start_atom_index": token,
        "sym_id": token,
        "template_backbone_frame_mask": (1, templates, tokens),
        "template_distogram": (1, templates, tokens, tokens, 39),
        "template_pseudo_beta_mask": (1, templates, tokens),
        "template_restype": (1, templates, tokens, 32),
        "template_unit_vector": (1, templates, tokens, tokens, 3),
        "token_bonds": (1, tokens, tokens),
        "token_index": token,
        "token_mask": token,
    }


def _prefix_mask(value: np.ndarray, *, name: str) -> int:
    flat = np.asarray(value).reshape(-1)
    if not np.isin(flat, (0, 1)).all():
        raise ValueError(f"OpenFold3 feature {name!r} must be a binary mask")
    count = int(flat.sum())
    expected = np.arange(flat.size) < count
    if not np.array_equal(flat.astype(bool), expected):
        raise ValueError(
            f"OpenFold3 feature {name!r} must contain real entries before padding"
        )
    if count == 0:
        raise ValueError(f"OpenFold3 feature {name!r} contains no real entries")
    return count


def validate_features(features: Mapping[str, np.ndarray]) -> None:
    """Reject malformed archives before weights are loaded or JAX is initialized.

    OpenFold3's compiled functions assume an exact batched feature ABI. A bad
    archive otherwise fails deep in tracing (or silently associates atoms with
    the wrong tokens), after expensive checkpoint loading. This boundary checks
    every model feature's rank, dimensions and dtype plus the core mask/index
    invariants established by the upstream featurizer.
    """
    from foldjax.models.openfold3.data.featurize import MODEL_FEATURES

    missing = [name for name in MODEL_FEATURES if name not in features]
    if missing:
        raise ValueError(f"OpenFold3 feature archive is missing: {missing}")

    arrays: dict[str, np.ndarray] = {}
    for name, value in features.items():
        array = np.asarray(value)
        if not np.issubdtype(array.dtype, np.number):
            raise ValueError(
                f"OpenFold3 feature {name!r} must be numeric; got {array.dtype}"
            )
        if not np.isfinite(array).all():
            raise ValueError(f"OpenFold3 feature {name!r} contains NaN or infinity")
        arrays[name] = array

    token_mask = arrays["token_mask"]
    atom_mask = arrays["atom_mask"]
    if token_mask.ndim != 2 or token_mask.shape[0] != 1:
        raise ValueError(
            "OpenFold3 feature 'token_mask' must have shape (1, num_tokens); "
            f"got {token_mask.shape}"
        )
    if atom_mask.ndim != 2 or atom_mask.shape[0] != 1:
        raise ValueError(
            "OpenFold3 feature 'atom_mask' must have shape (1, num_atoms); "
            f"got {atom_mask.shape}"
        )
    tokens = token_mask.shape[1]
    atoms = atom_mask.shape[1]
    msa = arrays["msa"]
    templates = arrays["template_restype"]
    if msa.ndim != 4 or msa.shape[0] != 1 or msa.shape[1] < 1:
        raise ValueError(
            "OpenFold3 feature 'msa' must have shape "
            f"(1, positive_rows, num_tokens, 32); got {msa.shape}"
        )
    if templates.ndim != 4 or templates.shape[0] != 1 or templates.shape[1] < 1:
        raise ValueError(
            "OpenFold3 feature 'template_restype' must have shape "
            f"(1, positive_templates, num_tokens, 32); got {templates.shape}"
        )

    expected_shapes = _expected_shapes(
        tokens=tokens,
        atoms=atoms,
        msa_rows=msa.shape[1],
        templates=templates.shape[1],
    )
    for name in MODEL_FEATURES:
        array = arrays[name]
        expected_shape = expected_shapes[name]
        if array.shape != expected_shape:
            raise ValueError(
                f"OpenFold3 feature {name!r} has shape {array.shape}; "
                f"expected {expected_shape}"
            )
        expected_dtype = _EXPECTED_DTYPES[name]
        if array.dtype != expected_dtype:
            raise ValueError(
                f"OpenFold3 feature {name!r} has dtype {array.dtype}; "
                f"expected {expected_dtype}"
            )

    real_tokens = _prefix_mask(token_mask, name="token_mask")
    real_atoms = _prefix_mask(atom_mask, name="atom_mask")
    counts = arrays["num_atoms_per_token"][0].astype(np.int64)
    starts = arrays["start_atom_index"][0].astype(np.int64)
    if (counts < 0).any() or (counts > 23).any():
        raise ValueError("OpenFold3 num_atoms_per_token must be between 0 and 23")
    if (counts[real_tokens:] != 0).any():
        raise ValueError("OpenFold3 padded tokens must have zero atoms")
    if int(counts[:real_tokens].sum()) != real_atoms:
        raise ValueError(
            "OpenFold3 num_atoms_per_token does not sum to the real atom count"
        )
    expected_starts = np.zeros(tokens, dtype=np.int64)
    if real_tokens > 1:
        expected_starts[1:real_tokens] = np.cumsum(counts[: real_tokens - 1])
    if not np.array_equal(starts, expected_starts):
        raise ValueError("OpenFold3 start_atom_index is inconsistent with atom counts")

    owners = arrays["atom_to_token_index"][0].astype(np.int64)
    expected_owners = np.repeat(np.arange(real_tokens), counts[:real_tokens])
    if not np.array_equal(owners[:real_atoms], expected_owners):
        raise ValueError(
            "OpenFold3 atom_to_token_index is inconsistent with atom counts"
        )
    if (owners[real_atoms:] != 0).any():
        raise ValueError("OpenFold3 padded atoms must map to token zero")

    slot_mask = arrays["max_atom_per_token_mask"].reshape(tokens, 23)
    expected_slots = np.arange(23)[None, :] < counts[:, None]
    expected_slots &= token_mask[0, :, None].astype(bool)
    if not np.array_equal(slot_mask.astype(bool), expected_slots):
        raise ValueError(
            "OpenFold3 max_atom_per_token_mask is inconsistent with atom counts"
        )
