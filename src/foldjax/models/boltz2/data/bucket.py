"""Schema-aware feature bucketing for persistent JAX compile caches."""

from __future__ import annotations

from collections.abc import Mapping

import jax.numpy as jnp
import numpy as np

TOKEN_BUCKETS = (256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096)
MSA_BUCKETS = (1, 128, 256, 512, 768, 1024)

_TOKEN = "token"
_ATOM = "atom"
_MSA = "msa"

# Axes are for the batched numpy features returned by ``featurize_yaml``.
# Unknown features are intentionally left unchanged: guessing from a matching
# dimension corrupts inputs when token/atom counts equal channel sizes.
_FEATURE_AXES: dict[str, tuple[tuple[int, str], ...]] = {
    # Per-token features.
    **{
        key: ((1, _TOKEN),)
        for key in (
            "token_index", "residue_index", "asym_id", "entity_id", "sym_id",
            "mol_type", "res_type", "disto_center", "token_pad_mask",
            "token_resolved_mask", "token_disto_mask", "method_feature",
            "modified", "cyclic_period", "affinity_token_mask", "deletion_mean",
            "profile", "deletion_mean_affinity", "profile_affinity",
        )
    },
    # Per-token-pair features.
    **{
        key: ((1, _TOKEN), (2, _TOKEN))
        for key in (
            "token_bonds", "type_bonds", "contact_conditioning", "contact_threshold",
        )
    },
    # Per-atom features.
    **{
        key: ((1, _ATOM),)
        for key in (
            "ref_pos", "atom_resolved_mask", "ref_atom_name_chars", "ref_element",
            "ref_charge", "ref_chirality", "atom_backbone_feat", "ref_space_uid",
            "atom_pad_mask", "bfactor", "plddt",
        )
    },
    "coords": ((2, _ATOM),),
    "atom_to_token": ((1, _ATOM), (2, _TOKEN)),
    "token_to_rep_atom": ((1, _TOKEN), (2, _ATOM)),
    # The first r_set axis is not necessarily the token count.
    "r_set_to_rep_atom": ((2, _ATOM),),
    "token_to_center_atom": ((1, _TOKEN), (2, _ATOM)),
    "disto_target": ((1, _TOKEN), (2, _TOKEN)),
    "disto_coords_ensemble": ((2, _TOKEN),),
    "frames_idx": ((2, _TOKEN),),
    "frame_resolved_mask": ((2, _TOKEN),),
    # MSA features: (batch, depth, token).
    **{
        key: ((1, _MSA), (2, _TOKEN))
        for key in ("msa", "msa_paired", "deletion_value", "has_deletion", "msa_mask")
    },
    # Template features: (batch, template, token, ...).
    **{
        key: ((2, _TOKEN),)
        for key in (
            "template_restype", "template_frame_rot", "template_frame_t",
            "template_cb", "template_ca", "template_mask_cb",
            "template_mask_frame", "template_mask", "query_to_template",
            "visibility_ids",
        )
    },
}


def _pad_to(arr: np.ndarray, axis: int, target: int) -> np.ndarray:
    cur = arr.shape[axis]
    if cur >= target:
        return arr
    pad = [(0, 0)] * arr.ndim
    pad[axis] = (0, target - cur)
    return np.pad(arr, pad, mode="constant", constant_values=0)


def _truncate_to(arr: np.ndarray, axis: int, target: int) -> np.ndarray:
    if arr.shape[axis] <= target:
        return arr
    index = [slice(None)] * arr.ndim
    index[axis] = slice(0, target)
    return arr[tuple(index)]


def resolve_bucket_shape(feats: Mapping[str, object]) -> tuple[int, int, int]:
    """Return the shared ``(token, atom, MSA-depth)`` bucket for features."""
    tokens = int(np.shape(feats["token_pad_mask"])[-1])
    atoms = int(np.shape(feats["atom_pad_mask"])[-1])
    msa_depth = int(np.shape(feats["msa"])[1]) if "msa" in feats else 1
    target_tokens = next((size for size in TOKEN_BUCKETS if size >= tokens), tokens)
    target_atoms = ((atoms + 31) // 32) * 32
    target_msa = next((size for size in MSA_BUCKETS if size >= msa_depth), 1024)
    return target_tokens, target_atoms, target_msa


def pad_feats(
    feats: Mapping[str, object],
    target_tokens: int,
    target_atoms: int,
    *,
    target_msa: int | None = None,
) -> tuple[dict[str, jnp.ndarray], list[str]]:
    """Pad known semantic axes and optionally truncate/pad MSA depth.

    Token and atom targets may only grow. MSA depth may be truncated because
    production inference already caps it at 1024 rows deterministically.
    """
    tokens = int(np.shape(feats["token_pad_mask"])[-1])
    atoms = int(np.shape(feats["atom_pad_mask"])[-1])
    if target_tokens < tokens:
        raise ValueError(
            f"target_tokens {target_tokens} is smaller than input {tokens}"
        )
    if target_atoms < atoms:
        raise ValueError(f"target_atoms {target_atoms} is smaller than input {atoms}")
    if target_msa is not None and target_msa <= 0:
        raise ValueError("target_msa must be positive")

    targets = {_TOKEN: target_tokens, _ATOM: target_atoms, _MSA: target_msa}
    out: dict[str, jnp.ndarray] = {}
    log: list[str] = []
    for key, value in feats.items():
        arr = np.asarray(value)
        new = arr
        actions: list[str] = []
        for axis, kind in _FEATURE_AXES.get(key, ()):
            target = targets[kind]
            if target is None:
                continue
            before = new.shape[axis]
            if kind == _MSA:
                new = _truncate_to(new, axis, target)
            new = _pad_to(new, axis, target)
            if before != new.shape[axis]:
                actions.append(f"{kind}@{axis}:{before}->{new.shape[axis]}")
        out[key] = jnp.asarray(new.astype(arr.dtype, copy=False))
        if actions:
            log.append(f"{key}: {arr.shape} -> {new.shape} [{', '.join(actions)}]")
        else:
            log.append(f"{key}: {arr.shape} unchanged")
    return out, log


_TOKEN_OUTPUT_AXES: dict[str, tuple[int, ...]] = {
    "plddt": (1,),
    "plddt_logits": (1,),
    "resolved_logits": (1,),
    "pbfactor": (1,),
    "pde": (1, 2),
    "pde_logits": (1, 2),
    "pae": (1, 2),
    "pae_logits": (1, 2),
    "pdistogram": (1, 2),
}


def crop_prediction_outputs(
    outputs: Mapping[str, object], original_tokens: int, original_atoms: int
) -> dict[str, object]:
    """Remove bucket-only padding from public model outputs."""
    cropped: dict[str, object] = {}
    for key, value in outputs.items():
        if key == "sample_atom_coords":
            cropped[key] = value[..., :original_atoms, :]
            continue
        axes = _TOKEN_OUTPUT_AXES.get(key)
        if not axes:
            cropped[key] = value
            continue
        index = [slice(None)] * value.ndim
        for axis in axes:
            index[axis] = slice(0, original_tokens)
        cropped[key] = value[tuple(index)]
    return cropped
