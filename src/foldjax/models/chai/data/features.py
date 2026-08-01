"""Deterministic NumPy port of Chai-1's inference feature factory.

The public :func:`generate_features` entry point accepts the already batched and
padded ``inputs`` mapping produced immediately before Chai's ``FeatureFactory``.
It returns the 32 arrays consumed by the port's ``models.feature_embedding``.

Upstream generators retain a trailing singleton feature dimension for scalar
categorical/identity features.  The released feature embedder accepts those
scalars without that dimension, so this module removes it for model-facing
features.  The non-scalar dimensions of RBF restraints, template residue type,
and inverse-distance features are retained exactly.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

import numpy as np

from foldjax.models.chai.data.msa import NO_PAIRING_KEY, RESIDUE_TYPES
from foldjax.models.chai.data.restraints import manual_restraint_features

FEATURE_NAMES: Final[tuple[str, ...]] = (
    "RelativeSequenceSeparation",
    "RelativeTokenSeparation",
    "RelativeEntity",
    "RelativeChain",
    "ResidueType",
    "ESMEmbeddings",
    "BlockedAtomPairDistogram",
    "InverseSquaredBlockedAtomPairDistances",
    "AtomRefPos",
    "AtomRefCharge",
    "AtomRefMask",
    "AtomRefElement",
    "AtomNameOneHot",
    "TemplateMask",
    "TemplateUnitVector",
    "TemplateResType",
    "TemplateDistogram",
    "TokenDistanceRestraint",
    "DockingConstraintGenerator",
    "TokenPairPocketRestraint",
    "MSAProfile",
    "MSADeletionMean",
    "IsDistillation",
    "TokenBFactor",
    "TokenPLDDT",
    "ChainIsCropped",
    "MissingChainContact",
    "MSAOneHot",
    "MSAHasDeletion",
    "MSADeletionValue",
    "IsPairedMSA",
    "MSADataSource",
)

_SEPARATION_BINS: Final[np.ndarray] = np.arange(-32, 33, dtype=np.float32)
_ATOM_DISTANCE_BINS: Final[np.ndarray] = np.asarray(
    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 12.0, 16.0],
    dtype=np.float32,
)
_TEMPLATE_DISTANCE_BINS: Final[np.ndarray] = np.linspace(
    3.25, 50.75, 38, dtype=np.float32
)[1:]


def _array(inputs: Mapping[str, Any], name: str) -> np.ndarray:
    try:
        value = np.asarray(inputs[name])
    except KeyError as error:
        raise KeyError(f"missing padded Chai input: {name}") from error
    if value.dtype == np.dtype("O"):
        raise TypeError(f"input {name} must be a numeric array")
    return value


def _batched(inputs: Mapping[str, Any], name: str, ndim: int) -> np.ndarray:
    value = _array(inputs, name)
    if value.ndim != ndim:
        raise ValueError(f"input {name} must have {ndim} dimensions, got {value.shape}")
    return value


def _same_shape(name: str, value: np.ndarray, shape: tuple[int, ...]) -> None:
    if value.shape != shape:
        raise ValueError(f"input {name} must have shape {shape}, got {value.shape}")


def _remap_sorted(values: np.ndarray) -> np.ndarray:
    """Match ``torch.unique(sorted=True, return_inverse=True)`` globally."""
    _, inverse = np.unique(values, return_inverse=True)
    return inverse.reshape(values.shape).astype(np.int64, copy=False)


def _blocked_distances(
    inputs: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    positions = _batched(inputs, "atom_ref_pos", 3).astype(np.float32, copy=False)
    space_uid = _batched(inputs, "atom_ref_space_uid", 2)
    q_indices = _batched(inputs, "block_atom_pair_q_idces", 2).astype(
        np.int64, copy=False
    )
    kv_indices = _batched(inputs, "block_atom_pair_kv_idces", 2).astype(
        np.int64, copy=False
    )
    pair_mask = _batched(inputs, "block_atom_pair_mask", 4).astype(np.bool_, copy=False)
    batch, atoms, coordinate_dim = positions.shape
    if coordinate_dim != 3:
        raise ValueError("input atom_ref_pos must end in three coordinates")
    _same_shape("atom_ref_space_uid", space_uid, (batch, atoms))
    if q_indices.shape[0] != kv_indices.shape[0]:
        raise ValueError("query and key atom indices must have the same block count")
    expected = (batch, q_indices.shape[0], q_indices.shape[1], kv_indices.shape[1])
    _same_shape("block_atom_pair_mask", pair_mask, expected)
    if (
        np.any(q_indices < 0)
        or np.any(q_indices >= atoms)
        or np.any(kv_indices < 0)
        or np.any(kv_indices >= atoms)
    ):
        raise ValueError("blocked atom indices are outside atom_ref_pos")

    q_positions = positions[:, q_indices]
    kv_positions = positions[:, kv_indices]
    delta = q_positions[..., :, None, :] - kv_positions[..., None, :, :]
    distances = np.sqrt(np.sum(delta * delta, axis=-1, dtype=np.float32))
    same_space = (
        space_uid[:, q_indices][..., :, None] == space_uid[:, kv_indices][..., None, :]
    )
    return distances, pair_mask & same_space


def _missing_chain_contact(inputs: Mapping[str, Any]) -> np.ndarray:
    coords = _batched(inputs, "atom_gt_coords", 3).astype(np.float32, copy=False)
    atom_mask = _batched(inputs, "atom_exists_mask", 2).astype(np.bool_, copy=False)
    token_mask = _batched(inputs, "token_exists_mask", 2).astype(np.bool_, copy=False)
    asym_id = _batched(inputs, "token_asym_id", 2)
    atom_token = _batched(inputs, "atom_token_index", 2).astype(np.int64, copy=False)
    batch, tokens = asym_id.shape
    atoms = coords.shape[1]
    _same_shape("atom_gt_coords", coords, (batch, atoms, 3))
    _same_shape("atom_exists_mask", atom_mask, (batch, atoms))
    _same_shape("atom_token_index", atom_token, (batch, atoms))
    _same_shape("token_exists_mask", token_mask, (batch, tokens))
    if np.any(atom_token < 0) or np.any(atom_token >= tokens):
        raise ValueError("atom_token_index is outside the padded token axis")

    result = np.zeros((batch, tokens), dtype=np.float32)
    for batch_index in range(batch):
        chain_ids = np.unique(asym_id[batch_index, token_mask[batch_index]])
        if chain_ids.size <= 1:
            continue
        valid_atom = atom_mask[batch_index]
        valid_coords = coords[batch_index, valid_atom]
        valid_asym = asym_id[batch_index, atom_token[batch_index, valid_atom]]
        contacted: set[int] = set()
        # Keep peak temporary memory bounded for full inference buckets.
        for start in range(0, valid_coords.shape[0], 512):
            query = valid_coords[start : start + 512]
            query_asym = valid_asym[start : start + 512]
            delta = query[:, None, :] - valid_coords[None, :, :]
            dist_sq = np.sum(delta * delta, axis=-1, dtype=np.float32)
            contacts = (dist_sq < 36.0) & (query_asym[:, None] != valid_asym[None, :])
            if np.any(contacts):
                query_rows, key_rows = np.nonzero(contacts)
                contacted.update(int(value) for value in query_asym[query_rows])
                contacted.update(int(value) for value in valid_asym[key_rows])
        missing = np.asarray(
            [chain for chain in chain_ids.tolist() if int(chain) not in contacted]
        )
        if missing.size:
            result[batch_index] = np.isin(asym_id[batch_index], missing)
    return result


def _template_features(
    inputs: Mapping[str, Any], asym_id: np.ndarray
) -> dict[str, np.ndarray]:
    backbone = _batched(inputs, "template_backbone_frame_mask", 3).astype(
        np.bool_, copy=False
    )
    pseudo_beta = _batched(inputs, "template_pseudo_beta_mask", 3).astype(
        np.bool_, copy=False
    )
    distances = _batched(inputs, "template_distances", 4).astype(np.float32, copy=False)
    unit_vector = _batched(inputs, "template_unit_vector", 5).astype(
        np.float32, copy=False
    )
    restype = _batched(inputs, "template_restype", 3).astype(np.uint8, copy=False)
    batch, templates, tokens = backbone.shape
    shape = (batch, templates, tokens)
    _same_shape("template_pseudo_beta_mask", pseudo_beta, shape)
    _same_shape("template_restype", restype, shape)
    _same_shape("template_distances", distances, shape + (tokens,))
    _same_shape("template_unit_vector", unit_vector, shape + (tokens, 3))
    _same_shape("token_asym_id", asym_id, (batch, tokens))

    same_asym = asym_id[:, None, :, None] == asym_id[:, None, None, :]
    backbone_pair = backbone[..., :, None] & backbone[..., None, :]
    pseudo_pair = pseudo_beta[..., :, None] & pseudo_beta[..., None, :]
    template_mask = np.stack((backbone_pair, pseudo_pair), axis=-1).astype(np.float32)
    template_mask *= same_asym[..., None]
    discretized = np.searchsorted(_TEMPLATE_DISTANCE_BINS, distances).astype(np.int64)
    discretized = np.where(same_asym, discretized, 38)
    return {
        "TemplateMask": template_mask,
        "TemplateUnitVector": unit_vector * same_asym[..., None],
        "TemplateResType": restype[..., None],
        "TemplateDistogram": discretized,
    }


def _msa_features(inputs: Mapping[str, Any]) -> dict[str, np.ndarray]:
    msa_tokens = _batched(inputs, "msa_tokens", 3).astype(np.uint8, copy=False)
    msa_mask = _batched(inputs, "msa_mask", 3).astype(np.bool_, copy=False)
    deletion = _batched(inputs, "msa_deletion_matrix", 3).astype(np.uint8, copy=False)
    pairkey = _batched(inputs, "msa_pairkey", 3)
    source = _batched(inputs, "msa_sequence_source", 3).astype(np.uint8, copy=True)
    shape = msa_tokens.shape
    for name, value in (
        ("msa_mask", msa_mask),
        ("msa_deletion_matrix", deletion),
        ("msa_pairkey", pairkey),
        ("msa_sequence_source", source),
    ):
        _same_shape(name, value, shape)

    main_tokens = _batched(inputs, "main_msa_tokens", 3).astype(np.uint8, copy=False)
    main_mask = _batched(inputs, "main_msa_mask", 3).astype(np.bool_, copy=False)
    main_deletion = _batched(inputs, "main_msa_deletion_matrix", 3).astype(
        np.uint8, copy=False
    )
    _same_shape("main_msa_mask", main_mask, main_tokens.shape)
    _same_shape("main_msa_deletion_matrix", main_deletion, main_tokens.shape)
    if main_tokens.shape[0] != shape[0] or main_tokens.shape[2] != shape[2]:
        raise ValueError("main MSA and sampled MSA must share batch and token axes")
    residue_count = len(RESIDUE_TYPES)
    if np.any(main_tokens >= residue_count):
        raise ValueError("main_msa_tokens contains an unknown residue class")

    batch, _, tokens = main_tokens.shape
    counts = np.zeros((batch, tokens, residue_count), dtype=np.uint8)
    batch_index = np.arange(batch)[:, None, None]
    token_index = np.arange(tokens)[None, None, :]
    np.add.at(
        counts,
        (batch_index, token_index, main_tokens),
        main_mask.astype(np.uint8),
    )
    denominator = np.maximum(counts.sum(axis=-1, keepdims=True), 1)
    profile = counts.astype(np.float32) / denominator.astype(np.float32)
    deletion_sum = np.sum(
        main_mask * main_deletion.astype(np.float32), axis=1, dtype=np.float32
    )
    deletion_denominator = np.maximum(main_mask.sum(axis=1), 1)
    deletion_mean = deletion_sum / deletion_denominator.astype(np.float32)

    can_pair = msa_mask & (pairkey != NO_PAIRING_KEY)
    paired = (pairkey == pairkey[..., :1]) & can_pair
    source[source == 5] = 4  # Chai-1 remaps QUERY to NONE.
    source[~msa_mask] = 4
    return {
        "MSAProfile": profile,
        "MSADeletionMean": deletion_mean.astype(np.float32, copy=False),
        "MSAOneHot": msa_tokens,
        "MSAHasDeletion": (deletion > 0).astype(np.float32),
        "MSADeletionValue": (
            np.float32(2.0 / np.pi)
            * np.arctan(deletion.astype(np.float32) / np.float32(3.0))
        ).astype(np.float32),
        "IsPairedMSA": paired.astype(np.float32),
        "MSADataSource": source,
    }


def generate_features(
    inputs: Mapping[str, Any], *, rng: np.random.Generator | None = None
) -> dict[str, np.ndarray]:
    """Generate Chai-1 model features from batched, padded inference inputs.

    Manual contact, docking, and pocket restraints use Chai's released feature
    semantics; absent/null restraints produce the released inference defaults.
    """
    residue_index = _batched(inputs, "token_residue_index", 2).astype(
        np.int64, copy=False
    )
    asym_id = _batched(inputs, "token_asym_id", 2)
    token_index = _batched(inputs, "token_index", 2)
    entity_id = _batched(inputs, "token_entity_id", 2).astype(np.int64, copy=False)
    sym_id = _batched(inputs, "token_sym_id", 2).astype(np.int64, copy=False)
    batch, tokens = residue_index.shape
    for name, value in (
        ("token_asym_id", asym_id),
        ("token_index", token_index),
        ("token_entity_id", entity_id),
        ("token_sym_id", sym_id),
    ):
        _same_shape(name, value, (batch, tokens))

    relative_residue = residue_index[..., :, None] - residue_index[..., None, :]
    relative_asym = asym_id[..., :, None] - asym_id[..., None, :]
    relative_sequence = np.searchsorted(
        _SEPARATION_BINS, relative_residue.astype(np.float32) + np.float32(1e-4)
    ).astype(np.int64)
    relative_sequence[relative_asym != 0] = 66

    relative_token = token_index[..., :, None] - token_index[..., None, :]
    same_residue_and_chain = (relative_residue == 0) & (relative_asym == 0)
    relative_token = np.clip(relative_token + 32, 0, 65)
    relative_token = np.where(same_residue_and_chain, relative_token, 66)

    remapped_entity = _remap_sorted(entity_id)
    relative_entity = remapped_entity[..., :, None] - remapped_entity[..., None, :]
    relative_entity = np.clip(relative_entity + 1, 0, 2).astype(np.int64)

    remapped_sym = _remap_sorted(sym_id)
    relative_chain = remapped_sym[..., :, None] - remapped_sym[..., None, :]
    relative_chain = np.clip(relative_chain + 2, 0, 4).astype(np.int64)
    relative_chain[(entity_id[..., :, None] - entity_id[..., None, :]) != 0] = 5

    atom_ref_pos = _batched(inputs, "atom_ref_pos", 3).astype(np.float32, copy=False)
    atom_ref_charge = _batched(inputs, "atom_ref_charge", 2).astype(
        np.float32, copy=False
    )
    atom_ref_mask = _batched(inputs, "atom_ref_mask", 2).astype(np.float32, copy=False)
    atoms = atom_ref_pos.shape[1]
    _same_shape("atom_ref_pos", atom_ref_pos, (batch, atoms, 3))
    _same_shape("atom_ref_charge", atom_ref_charge, (batch, atoms))
    _same_shape("atom_ref_mask", atom_ref_mask, (batch, atoms))
    atom_element = _batched(inputs, "atom_ref_element", 2)
    atom_name = _batched(inputs, "atom_ref_name_chars", 3)
    _same_shape("atom_ref_element", atom_element, (batch, atoms))
    _same_shape("atom_ref_name_chars", atom_name, (batch, atoms, 4))

    blocked_distances, blocked_mask = _blocked_distances(inputs)
    blocked_distogram = np.searchsorted(_ATOM_DISTANCE_BINS, blocked_distances).astype(
        np.int64
    )
    blocked_distogram[~blocked_mask] = 11
    inverse_distances = np.concatenate(
        (
            (np.float32(1.0) / (np.float32(1.0) + blocked_distances**2))[..., None],
            blocked_mask.astype(np.float32)[..., None],
        ),
        axis=-1,
    )

    residue_type = _batched(inputs, "token_residue_type", 2).astype(
        np.int64, copy=False
    )
    _same_shape("token_residue_type", residue_type, (batch, tokens))
    esm = np.asarray(
        inputs.get("esm_embeddings", np.zeros((batch, tokens, 2560), np.float32))
    ).astype(np.float32, copy=False)
    _same_shape("esm_embeddings", esm, (batch, tokens, 2560))

    token_mask = _batched(inputs, "token_exists_mask", 2).astype(np.bool_, copy=False)
    _same_shape("token_exists_mask", token_mask, (batch, tokens))
    is_distillation = np.asarray(
        inputs.get("is_distillation", np.zeros((batch, 1), np.bool_))
    ).astype(np.bool_, copy=False)
    _same_shape("is_distillation", is_distillation, (batch, 1))

    contact_restraint, docking_restraint, pocket_restraint = manual_restraint_features(
        inputs, batch=batch, tokens=tokens, rng=rng
    )
    features: dict[str, np.ndarray] = {
        "RelativeSequenceSeparation": relative_sequence,
        "RelativeTokenSeparation": relative_token,
        "RelativeEntity": relative_entity,
        "RelativeChain": relative_chain,
        "ResidueType": residue_type,
        "ESMEmbeddings": esm,
        "BlockedAtomPairDistogram": blocked_distogram,
        "InverseSquaredBlockedAtomPairDistances": inverse_distances.astype(
            np.float32, copy=False
        ),
        "AtomRefPos": atom_ref_pos / np.float32(10.0),
        "AtomRefCharge": atom_ref_charge,
        "AtomRefMask": atom_ref_mask,
        "AtomRefElement": np.minimum(atom_element, 129),
        "AtomNameOneHot": atom_name,
        "TokenDistanceRestraint": contact_restraint,
        "DockingConstraintGenerator": docking_restraint,
        "TokenPairPocketRestraint": pocket_restraint,
        "IsDistillation": np.broadcast_to(
            is_distillation.astype(np.uint8)[:, None, 0], (batch, tokens)
        ).copy(),
        "TokenBFactor": np.full((batch, tokens), 2, dtype=np.int64),
        "TokenPLDDT": np.full((batch, tokens), 3, dtype=np.int64),
        "ChainIsCropped": np.zeros((batch, tokens), dtype=np.float32),
        "MissingChainContact": _missing_chain_contact(inputs),
    }
    features.update(_template_features(inputs, asym_id))
    features.update(_msa_features(inputs))
    if tuple(features) != FEATURE_NAMES:
        # Dict insertion order is part of fixture reproducibility, not model math.
        features = {name: features[name] for name in FEATURE_NAMES}
    return features


__all__ = ["FEATURE_NAMES", "generate_features"]
