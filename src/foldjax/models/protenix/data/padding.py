"""Schema-aware padding and output cropping for Protenix-family features.

The featurizer emits unbatched arrays with several independent semantic axes.
Padding them by matching dimensions is unsafe (a bond list of length ``N`` is
not a token feature merely because the job also has ``N`` tokens), so this
module names every supported field and fails when its expected layout drifts.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from foldjax.padding import PaddingPlan, resolve_axis
from foldjax.schema import PaddingConfig

_TOKEN_FIELDS = {
    "restype",
    "profile",
    "deletion_mean",
    "residue_index",
    "token_index",
    "asym_id",
    "entity_id",
    "sym_id",
    "has_frame",
    "token_entity_id",
    "token_copy_id",
    "token_is_modified",
    "token_is_standard_polymer",
    "token_polymer_type",
    "token_reference_is_mse",
    "token_ccd_code_chars",
    "esm_token_embedding",
}
_ATOM_FIELDS = {
    "ref_pos",
    "ref_charge",
    "ref_mask",
    "ref_atom_name_chars",
    "ref_element",
    "distogram_rep_atom_mask",
    "atom_to_tokatom_idx",
    "mol_id",
    "atom_entity_id",
    "atom_copy_id",
    "atom_residue_index",
    "atom_input_index",
    "ligand_stereo",
    "output_atom_name",
    "output_atom_element",
    "output_atom_res_name",
    "output_atom_chain_id",
    "output_atom_res_id",
    "output_atom_polymer_type",
}
_TEMPLATE_FIELDS = {
    "template_aatype",
    "template_atom_positions",
    "template_atom_mask",
    "template_pseudo_beta_mask",
    "template_distogram",
    "template_unit_vector",
    "template_backbone_frame_mask",
}
_MSA_FIELDS = ("msa", "has_deletion", "deletion_value")


def pad_protenix_features(
    features: Mapping[str, Any],
    config: PaddingConfig,
    *,
    n_queries: int,
    n_keys: int,
) -> tuple[dict[str, Any], PaddingPlan]:
    """Pad one generated, unbatched Protenix feature dictionary.

    Static feature archives are deliberately handled by the CLI rather than
    accepted here: an archive may contain publisher/private leaves whose axes
    are not part of this schema, and silently leaving one exact-sized would
    defeat both masking and compile-shape normalization.
    """

    if "constraint_feature" in features:
        raise ValueError(
            "padding Protenix constraint features is not yet supported; "
            "remove padding or the constraint block"
        )
    required = {
        "asym_id",
        "atom_to_tokatom_idx",
        "restype",
        "profile",
        "deletion_mean",
        "ref_pos",
        "ref_charge",
        "ref_element",
        "ref_atom_name_chars",
        "atom_to_token_idx",
        "ref_space_uid",
        "ref_mask",
        "distogram_rep_atom_mask",
        "entity_id",
        "sym_id",
        "residue_index",
        "token_index",
        "has_frame",
        "msa",
        "has_deletion",
        "deletion_value",
        "relp",
        "token_bonds",
    }
    missing = sorted(required - features.keys())
    if missing:
        raise ValueError(
            "padding requires the complete generated Protenix feature schema; "
            "missing: " + ", ".join(missing)
        )

    restype = np.asarray(features["restype"])
    ref_pos = np.asarray(features["ref_pos"])
    msa = np.asarray(features["msa"])
    if restype.ndim != 2 or restype.shape[1] != 32:
        raise ValueError("padding requires unbatched restype [N_token, 32]")
    if ref_pos.ndim != 2 or ref_pos.shape[1] != 3:
        raise ValueError("padding requires unbatched ref_pos [N_atom, 3]")
    if msa.ndim != 2 or msa.shape[1] != restype.shape[0]:
        raise ValueError("padding requires unbatched msa [N_msa, N_token]")
    if n_queries <= 0 or n_keys < n_queries or n_queries % 2 or n_keys % 2:
        raise ValueError(
            "padding requires positive even n-queries/n-keys with n-keys >= n-queries"
        )

    storage_token = int(restype.shape[0])
    storage_atom = int(ref_pos.shape[0])
    storage_msa = int(msa.shape[0])
    token_mask = _existing_mask(features, "token_padding_mask", storage_token)
    atom_mask = _existing_mask(features, "atom_padding_mask", storage_atom)
    actual_token = int(np.count_nonzero(token_mask))
    actual_atom = int(np.count_nonzero(atom_mask))
    _require_real_prefix(token_mask, "token_padding_mask")
    _require_real_prefix(atom_mask, "atom_padding_mask")
    if actual_token < 1 or actual_atom < 1:
        raise ValueError("padding masks must retain at least one token and one atom")

    source_msa_mask = features.get("msa_mask")
    if source_msa_mask is None:
        source_msa_mask = np.ones((storage_msa, storage_token), dtype=np.float32)
    else:
        source_msa_mask = np.asarray(source_msa_mask)
        if source_msa_mask.shape != (storage_msa, storage_token):
            raise ValueError("msa_mask must share [N_msa, N_token] with msa")
    msa_row_mask = np.any(source_msa_mask[:, :actual_token] != 0, axis=-1)
    _require_real_prefix(msa_row_mask, "msa_mask rows")
    actual_msa = int(np.count_nonzero(msa_row_mask))
    if actual_msa < 1:
        raise ValueError("padding requires at least one real MSA row")

    template_keys = sorted(_TEMPLATE_FIELDS & features.keys())
    if set(template_keys) != _TEMPLATE_FIELDS:
        missing_template = sorted(_TEMPLATE_FIELDS - features.keys())
        raise ValueError(
            "padding requires the complete generated template schema; missing: "
            + ", ".join(missing_template)
        )
    storage_template = int(np.asarray(features["template_aatype"]).shape[0])
    if storage_template != 4:
        raise ValueError(
            "padding currently supports generated Protenix templates only "
            f"(expected the native fixed depth 4, got {storage_template})"
        )
    template_mask = np.asarray(features["template_pseudo_beta_mask"])
    if template_mask.shape[:3] != (storage_template, storage_token, storage_token):
        raise ValueError(
            "template_pseudo_beta_mask must have [N_template, N_token, N_token]"
        )
    template_row_mask = np.any(template_mask != 0, axis=(1, 2))
    _require_real_prefix(template_row_mask, "template mask rows")
    actual_template = int(np.count_nonzero(template_row_mask))

    target_token = resolve_axis(actual_token, config, "tokens", minimum=storage_token)
    target_atom = resolve_axis(actual_atom, config, "atoms", minimum=storage_atom)
    target_msa = resolve_axis(actual_msa, config, "msa", minimum=storage_msa)
    target_template = resolve_axis(
        actual_template, config, "templates", minimum=storage_template
    )
    # The native featurizer always materializes four templates, including
    # masked empty slots. Adding a fifth slot is not neutral: the template
    # module divides by its storage depth. Refuse it instead of changing the
    # released averaging rule behind an "exact" public pin.
    if target_template != storage_template:
        raise ValueError(
            "Protenix template padding target must be exactly 4; the native "
            "template module has a fixed four-slot averaging contract"
        )

    output = dict(features)
    _pad_named_axis(output, _TOKEN_FIELDS, storage_token, target_token, axis=0)
    _pad_named_axis(output, _ATOM_FIELDS, storage_atom, target_atom, axis=0)

    relp = np.asarray(features["relp"])
    token_bonds = np.asarray(features["token_bonds"])
    if relp.shape[:2] != (storage_token, storage_token):
        raise ValueError("relp must start with [N_token, N_token]")
    if token_bonds.shape != (storage_token, storage_token):
        raise ValueError("token_bonds must have [N_token, N_token]")
    token_width = (0, target_token - storage_token)
    output["relp"] = _pad_array(
        relp,
        (token_width, token_width) + ((0, 0),) * (relp.ndim - 2),
    )
    output["token_bonds"] = _pad_array(
        token_bonds, ((0, target_token - storage_token),) * 2
    )

    for name in _MSA_FIELDS:
        value = np.asarray(features[name])
        if value.shape != (storage_msa, storage_token):
            raise ValueError(f"{name} must have [N_msa, N_token]")
        constant = 31 if name == "msa" else 0
        output[name] = _pad_array(
            value,
            (
                (0, target_msa - storage_msa),
                (0, target_token - storage_token),
            ),
            constant=constant,
        )
    padded_msa_mask = _pad_array(
        source_msa_mask,
        (
            (0, target_msa - storage_msa),
            (0, target_token - storage_token),
        ),
    ).astype(np.float32, copy=False)

    _pad_templates(output, features, storage_token, target_token)

    output["token_padding_mask"] = _pad_array(
        token_mask.astype(np.float32), ((0, target_token - storage_token),)
    )
    output["atom_padding_mask"] = _pad_array(
        atom_mask.astype(np.float32), ((0, target_atom - storage_atom),)
    )
    output["msa_mask"] = (
        padded_msa_mask * output["token_padding_mask"][None, :]
    ).astype(np.float32, copy=False)

    atom_to_token = np.asarray(features["atom_to_token_idx"])
    if atom_to_token.shape != (storage_atom,):
        raise ValueError("atom_to_token_idx must have [N_atom]")
    dummy_token = actual_token if target_token > actual_token else 0
    output["atom_to_token_idx"] = _pad_array(
        atom_to_token,
        ((0, target_atom - storage_atom),),
        constant=dummy_token,
    )

    ref_space_uid = np.asarray(features["ref_space_uid"])
    if ref_space_uid.shape != (storage_atom,):
        raise ValueError("ref_space_uid must have [N_atom]")
    padded_uid = _pad_array(
        ref_space_uid,
        ((0, target_atom - storage_atom),),
    )
    if target_atom > storage_atom:
        start = int(np.max(ref_space_uid)) + 1 if storage_atom else 0
        padded_uid[storage_atom:] = np.arange(
            start, start + target_atom - storage_atom, dtype=padded_uid.dtype
        )
    output["ref_space_uid"] = padded_uid

    # Local atom geometry is storage-layout dependent. Rebuild it from the
    # padded arrays, then intersect its boundary mask with the semantic atom
    # mask so a real query can never normalize attention over a dummy key.
    from foldjax.models.protenix.data.featurize_json import _local_atom_geometry

    d_lm, v_lm, pad_info = _local_atom_geometry(
        np.asarray(output["ref_pos"]),
        np.asarray(output["ref_space_uid"]),
        n_queries=n_queries,
        n_keys=n_keys,
    )
    local_mask = np.asarray(pad_info["mask_trunked"], dtype=bool)
    q_mask, k_mask = _local_qk_masks(
        np.asarray(output["atom_padding_mask"], dtype=bool),
        n_queries=n_queries,
        n_keys=n_keys,
    )
    pad_info["mask_trunked"] = local_mask & q_mask[..., None] & k_mask[:, None, :]
    output["d_lm"] = d_lm
    output["v_lm"] = v_lm
    output["pad_info"] = pad_info

    plan = PaddingPlan(
        actual={
            "tokens": actual_token,
            "atoms": actual_atom,
            "msa": actual_msa,
            "templates": actual_template,
        },
        storage={
            "tokens": storage_token,
            "atoms": storage_atom,
            "msa": storage_msa,
            "templates": storage_template,
        },
        target={
            "tokens": target_token,
            "atoms": target_atom,
            "msa": target_msa,
            "templates": target_template,
        },
    )
    return output, plan


def crop_protenix_outputs(
    output: Mapping[str, Any],
    plan: PaddingPlan,
) -> dict[str, Any]:
    """Crop every public/raw Protenix output back to real cardinalities."""

    n_token = plan.actual["tokens"]
    n_atom = plan.actual["atoms"]
    cropped = dict(output)
    atom_second_last = {"coordinate", "plddt", "resolved"}
    atom_last = {"atom_plddt"}
    token_pair_last = {
        "token_pair_pde",
        "token_pair_pae",
        "contact_probs",
    }
    token_pair_with_channels = {"distogram_logits", "pae", "pde"}
    token_single = {"s_inputs", "s_trunk"}
    token_pair_repr = {"z_trunk"}
    for name in atom_second_last & cropped.keys():
        cropped[name] = _slice_axis(cropped[name], -2, n_atom)
    for name in atom_last & cropped.keys():
        cropped[name] = _slice_axis(cropped[name], -1, n_atom)
    for name in token_pair_last & cropped.keys():
        cropped[name] = _slice_axis(
            _slice_axis(cropped[name], -1, n_token), -2, n_token
        )
    for name in token_pair_with_channels & cropped.keys():
        cropped[name] = _slice_axis(
            _slice_axis(cropped[name], -2, n_token), -3, n_token
        )
    for name in token_single & cropped.keys():
        cropped[name] = _slice_axis(cropped[name], -2, n_token)
    for name in token_pair_repr & cropped.keys():
        cropped[name] = _slice_axis(
            _slice_axis(cropped[name], -2, n_token), -3, n_token
        )
    return cropped


def pad_atom_noise(value: Any, *, actual: int, target: int) -> Any:
    """Right-pad a random tape's atom axis without changing its real prefix."""

    if target == actual:
        return value
    import jax.numpy as jnp

    value = jnp.asarray(value)
    if value.shape[-2] != actual:
        raise ValueError(
            f"random tape atom axis is {value.shape[-2]}, expected {actual}"
        )
    widths = [(0, 0)] * value.ndim
    widths[-2] = (0, target - actual)
    return jnp.pad(value, widths)


def _existing_mask(features: Mapping[str, Any], name: str, size: int) -> np.ndarray:
    value = features.get(name)
    if value is None:
        return np.ones((size,), dtype=bool)
    mask = np.asarray(value)
    if mask.shape != (size,):
        raise ValueError(f"{name} must have shape [{size}]")
    return mask.astype(bool)


def _require_real_prefix(mask: np.ndarray, name: str) -> None:
    count = int(np.count_nonzero(mask))
    if not np.all(mask[:count]) or np.any(mask[count:]):
        raise ValueError(f"{name} must describe one contiguous real-value prefix")


def _pad_named_axis(
    output: dict[str, Any],
    names: set[str],
    storage: int,
    target: int,
    *,
    axis: int,
) -> None:
    for name in names & output.keys():
        value = np.asarray(output[name])
        if value.ndim <= axis or value.shape[axis] != storage:
            raise ValueError(
                f"{name} padding axis {axis} expected {storage}, got {value.shape}"
            )
        widths = [(0, 0)] * value.ndim
        widths[axis] = (0, target - storage)
        output[name] = _pad_array(value, tuple(widths))


def _pad_templates(
    output: dict[str, Any],
    source: Mapping[str, Any],
    storage_token: int,
    target_token: int,
) -> None:
    pair_fields = {
        "template_pseudo_beta_mask",
        "template_distogram",
        "template_unit_vector",
        "template_backbone_frame_mask",
    }
    for name in _TEMPLATE_FIELDS:
        value = np.asarray(source[name])
        if value.shape[0] != 4:
            raise ValueError(f"{name} must have native template depth 4")
        widths = [(0, 0)] * value.ndim
        if name in pair_fields:
            if value.shape[1:3] != (storage_token, storage_token):
                raise ValueError(
                    f"{name} must start [4, N_token, N_token], got {value.shape}"
                )
            widths[1] = (0, target_token - storage_token)
            widths[2] = (0, target_token - storage_token)
        else:
            if value.shape[1] != storage_token:
                raise ValueError(
                    f"{name} token axis expected {storage_token}, got {value.shape}"
                )
            widths[1] = (0, target_token - storage_token)
        constant = 31 if name == "template_aatype" else 0
        output[name] = _pad_array(value, tuple(widths), constant=constant)


def _local_qk_masks(
    mask: np.ndarray,
    *,
    n_queries: int,
    n_keys: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_atom = int(mask.shape[0])
    n_trunks = (n_atom + n_queries - 1) // n_queries
    q_pad = n_trunks * n_queries - n_atom
    pad_left = (n_keys - n_queries) // 2
    pad_right = int((n_trunks - 0.5) * n_queries + n_keys / 2 - n_atom + 0.5)
    q = np.pad(mask, (0, q_pad)).reshape(n_trunks, n_queries)
    padded_k = np.pad(mask, (pad_left, pad_right))
    k = np.stack(
        [padded_k[i * n_queries : i * n_queries + n_keys] for i in range(n_trunks)],
        axis=0,
    )
    return q, k


def _pad_array(
    value: np.ndarray,
    widths: tuple[tuple[int, int], ...],
    *,
    constant: Any = 0,
) -> np.ndarray:
    if all(before == 0 and after == 0 for before, after in widths):
        return value.copy()
    return np.pad(value, widths, mode="constant", constant_values=constant)


def _slice_axis(value: Any, axis: int, size: int) -> Any:
    index = [slice(None)] * value.ndim
    index[axis] = slice(0, size)
    return value[tuple(index)]


__all__ = [
    "crop_protenix_outputs",
    "pad_atom_noise",
    "pad_protenix_features",
]
