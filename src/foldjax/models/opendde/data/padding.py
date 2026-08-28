"""Schema-aware padding for OpenDDE's residue and structural token spaces."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from foldjax.models.opendde.models.msa_sampling import (
    pad_opendde_msa_cycle_features,
)
from foldjax.models.protenix.data.padding import (
    crop_protenix_outputs,
    pad_protenix_features,
)
from foldjax.models.protenix.relative_position import COMPACT_RELP_FIELDS
from foldjax.padding import PaddingPlan, resolve_axis
from foldjax.schema import PaddingConfig

_OPENDDE_TOKEN_FIELDS = {
    "frame_atom_index",
    "is_protein_token",
}
_OPENDDE_ATOM_FIELDS = {
    "pae_rep_atom_mask",
    "plddt_m_rep_atom_mask",
    "modified_res_mask",
    "is_protein",
    "is_ligand",
    "is_dna",
    "is_rna",
    "atom_to_structural_token_idx",
    "atom_to_structural_tokatom_idx",
    "structural_distogram_rep_atom_mask",
    "structural_pae_rep_atom_mask",
}
_OPENDDE_STRUCTURAL_FIELDS = {
    "structural_token_index",
    "residue_token_group_id",
    "subtoken_role",
    "subtoken_role_id",
    "twin_token_idx",
    "parent_residue_idx",
    "structural_has_frame",
    "structural_frame_atom_index",
    "prev_parent_residue_idx",
    "next_parent_residue_idx",
    "structural_is_polymer",
    "structural_polymer_type",
    "structural_seq_pos",
    "structural_is_protein_token",
}

# Numeric features that actually enter the OpenDDE graph.  Generated feature
# dictionaries also contain variable-length bond lists and writer metadata.
# Passing those unused arrays to ``jax.jit`` would make their shapes part of the
# cache key even though the graph never reads them.
_MODEL_FEATURES = {
    "atom_padding_mask",
    "atom_to_structural_token_idx",
    "atom_to_structural_tokatom_idx",
    "atom_to_token_idx",
    "atom_to_tokatom_idx",
    "asym_id",
    "d_lm",
    "deletion_mean",
    "distogram_rep_atom_mask",
    "entity_id",
    "esm_token_embedding",
    "frame_atom_index",
    "has_frame",
    "is_dna",
    "is_protein",
    "is_rna",
    "next_parent_residue_idx",
    "pad_info",
    "pae_rep_atom_mask",
    "parent_residue_idx",
    "prev_parent_residue_idx",
    "profile",
    "ref_atom_name_chars",
    "ref_charge",
    "ref_element",
    "ref_mask",
    "ref_pos",
    "relp",
    "residue_index",
    "restype",
    "structural_distogram_rep_atom_mask",
    "structural_frame_atom_index",
    "structural_has_frame",
    "structural_pae_rep_atom_mask",
    "structural_polymer_type",
    "structural_token_index",
    "structural_token_padding_mask",
    "subtoken_role_id",
    "sym_id",
    "token_bonds",
    "token_index",
    "token_padding_mask",
    "v_lm",
} | set(COMPACT_RELP_FIELDS)


def pad_opendde_features(
    features: Mapping[str, Any],
    cycle_msa_features: tuple[dict[str, np.ndarray], ...],
    config: PaddingConfig,
    *,
    n_queries: int,
    n_keys: int,
) -> tuple[dict[str, Any], tuple[dict[str, np.ndarray], ...], PaddingPlan]:
    """Pad every semantic OpenDDE model axis and return its concrete plan.

    MSA cycles must already have been sampled from the unpadded alignment.  The
    helper therefore cannot let synthetic rows alter a seeded permutation, and
    it can right-pad the sampled rows and columns with one exact mask contract.
    """

    unsupported = sorted(
        set(config.explicit_axes)
        - {"tokens", "atoms", "msa", "structural_tokens"}
    )
    if unsupported:
        raise ValueError(
            "OpenDDE does not support explicit padding axes: "
            + ", ".join(unsupported)
        )

    required = {
        "frame_atom_index",
        "pae_rep_atom_mask",
        "parent_residue_idx",
        "subtoken_role_id",
        "structural_token_index",
        "atom_to_structural_token_idx",
        "atom_to_structural_tokatom_idx",
        "structural_distogram_rep_atom_mask",
        "structural_pae_rep_atom_mask",
        "structural_has_frame",
        "structural_frame_atom_index",
    }
    missing = sorted(required - features.keys())
    if missing:
        raise ValueError(
            "padding requires the complete generated OpenDDE feature schema; "
            "missing: " + ", ".join(missing)
        )

    restype = np.asarray(features["restype"])
    ref_pos = np.asarray(features["ref_pos"])
    parent = np.asarray(features["parent_residue_idx"])
    if restype.ndim != 2 or restype.shape[-1] != 32:
        raise ValueError("OpenDDE padding requires restype [N_token, 32]")
    if ref_pos.ndim != 2 or ref_pos.shape[-1] != 3:
        raise ValueError("OpenDDE padding requires ref_pos [N_atom, 3]")
    if parent.ndim != 1:
        raise ValueError(
            "OpenDDE padding requires parent_residue_idx [N_structural_token]"
        )
    storage_token = int(restype.shape[0])
    storage_atom = int(ref_pos.shape[0])
    storage_structural = int(parent.shape[0])

    structural_mask = _existing_prefix_mask(
        features,
        "structural_token_padding_mask",
        storage_structural,
    )
    actual_structural = int(np.count_nonzero(structural_mask))
    if actual_structural < 1:
        raise ValueError(
            "structural_token_padding_mask must retain at least one token"
        )
    target_structural = resolve_axis(
        actual_structural,
        config,
        "structural_tokens",
        minimum=storage_structural,
    )

    # The raw MSA is not a graph input once sampled cycles are supplied.  Do
    # not duplicate a potentially 16k-row alignment across the padded token
    # width merely to discard it before jit: give the shared schema helper one
    # inert row, then remove those placeholder leaves.  The sampled cycles are
    # padded independently below and are the only MSA arrays the graph sees.
    common_source = dict(features)
    for name in ("msa", "has_deletion", "deletion_value"):
        value = np.asarray(features[name])
        if value.ndim != 2 or value.shape[0] < 1:
            raise ValueError(
                f"OpenDDE padding requires non-empty {name} [N_msa, N_token]"
            )
        common_source[name] = value[:1]
    common_source["msa_mask"] = np.ones_like(
        np.asarray(common_source["msa"]),
        dtype=np.float32,
    )
    # Generated templates have a fixed four-slot averaging contract.  Keep
    # that depth unchanged while the shared helper pads all residue/atom
    # schema fields and the templates' token axes.
    common_config = PaddingConfig(
        tokens=config.tokens,
        atoms=config.atoms,
        msa=1,
        templates=4,
        overflow=config.overflow,
    )
    padded, common_plan = pad_protenix_features(
        common_source,
        common_config,
        n_queries=n_queries,
        n_keys=n_keys,
    )
    for name in ("msa", "has_deletion", "deletion_value", "msa_mask"):
        padded.pop(name, None)
    target_token = common_plan.target["tokens"]
    target_atom = common_plan.target["atoms"]

    _pad_named_axis(
        padded,
        features,
        _OPENDDE_TOKEN_FIELDS,
        storage_token,
        target_token,
    )
    _pad_named_axis(
        padded,
        features,
        _OPENDDE_ATOM_FIELDS,
        storage_atom,
        target_atom,
    )
    _pad_named_axis(
        padded,
        features,
        _OPENDDE_STRUCTURAL_FIELDS,
        storage_structural,
        target_structural,
        constants={
            "twin_token_idx": -1,
            "prev_parent_residue_idx": -1,
            "next_parent_residue_idx": -1,
        },
    )

    # Dummy structural tokens are fully masked.  Pointing their parent and
    # frame indices at zero keeps every gather in-bounds before the mask is
    # applied.  Give their own token index a unique storage position so unused
    # relative-position values remain well-formed too.
    if target_structural > storage_structural:
        structural_index = np.asarray(padded["structural_token_index"]).copy()
        structural_index[storage_structural:] = np.arange(
            storage_structural,
            target_structural,
            dtype=structural_index.dtype,
        )
        padded["structural_token_index"] = structural_index
    padded["structural_token_padding_mask"] = _pad_array(
        structural_mask.astype(np.float32),
        target_structural,
    )

    padded_cycles, msa_plan = pad_opendde_msa_cycle_features(
        cycle_msa_features,
        config,
        token_target=target_token,
    )

    plan = PaddingPlan(
        actual={
            "tokens": common_plan.actual["tokens"],
            "atoms": common_plan.actual["atoms"],
            "msa": msa_plan.actual["msa"],
            "structural_tokens": actual_structural,
        },
        storage={
            "tokens": common_plan.storage["tokens"],
            "atoms": common_plan.storage["atoms"],
            "msa": msa_plan.storage["msa"],
            "structural_tokens": storage_structural,
        },
        target={
            "tokens": target_token,
            "atoms": target_atom,
            "msa": msa_plan.target["msa"],
            "structural_tokens": target_structural,
        },
    )
    return padded, padded_cycles, plan


def select_opendde_model_features(features: Mapping[str, Any]) -> dict[str, Any]:
    """Keep only normalized arrays consumed by the compiled OpenDDE graph."""

    return {
        name: value
        for name, value in features.items()
        if name in _MODEL_FEATURES or name.startswith("template_")
    }


def crop_opendde_outputs(
    output: Mapping[str, Any],
    plan: PaddingPlan,
) -> dict[str, Any]:
    """Crop residue, atom, and structural representations to real sizes."""

    cropped = crop_protenix_outputs(output, plan)
    n_structural = plan.actual["structural_tokens"]
    for name in {"structural_s_inputs", "structural_s_trunk"} & cropped.keys():
        cropped[name] = _slice_axis(cropped[name], -2, n_structural)
    if "structural_z_trunk" in cropped:
        value = _slice_axis(cropped["structural_z_trunk"], -2, n_structural)
        cropped["structural_z_trunk"] = _slice_axis(value, -3, n_structural)
    return cropped


def _existing_prefix_mask(
    features: Mapping[str, Any],
    name: str,
    size: int,
) -> np.ndarray:
    value = features.get(name)
    mask = (
        np.ones((size,), dtype=bool)
        if value is None
        else np.asarray(value).astype(bool)
    )
    if mask.shape != (size,):
        raise ValueError(f"{name} must have shape [{size}]")
    count = int(np.count_nonzero(mask))
    if not np.all(mask[:count]) or np.any(mask[count:]):
        raise ValueError(f"{name} must describe one contiguous real-token prefix")
    return mask


def _pad_named_axis(
    output: dict[str, Any],
    source: Mapping[str, Any],
    names: set[str],
    storage: int,
    target: int,
    *,
    constants: Mapping[str, Any] | None = None,
) -> None:
    for name in names & source.keys():
        value = np.asarray(source[name])
        if value.ndim < 1 or value.shape[0] != storage:
            raise ValueError(
                f"{name} padding axis expected {storage}, got {value.shape}"
            )
        widths = [(0, 0)] * value.ndim
        widths[0] = (0, target - storage)
        output[name] = np.pad(
            value,
            widths,
            mode="constant",
            constant_values=(constants or {}).get(name, 0),
        )


def _pad_array(value: np.ndarray, target: int) -> np.ndarray:
    return np.pad(value, (0, target - int(value.shape[0])))


def _slice_axis(value: Any, axis: int, size: int) -> Any:
    index = [slice(None)] * value.ndim
    index[axis] = slice(0, size)
    return value[tuple(index)]


__all__ = [
    "crop_opendde_outputs",
    "pad_opendde_features",
    "select_opendde_model_features",
]
