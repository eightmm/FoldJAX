"""Schema-aware feature bucketing for persistent JAX compile caches."""

from __future__ import annotations

from collections.abc import Mapping

import jax.numpy as jnp
import numpy as np

from foldjax.padding import (
    ATOM_BUCKETS as STANDARD_ATOM_BUCKETS,
)
from foldjax.padding import (
    PaddingPlan,
    resolve_axis,
)
from foldjax.schema import PaddingConfig

TOKEN_BUCKETS = (256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096)
ATOM_BUCKETS = STANDARD_ATOM_BUCKETS
MSA_BUCKETS = (1, 128, 256, 512, 768, 1024)

_TOKEN = "token"
_ATOM = "atom"
_MSA = "msa"

_TEMPLATE_FEATURES = (
    "template_restype",
    "template_frame_rot",
    "template_frame_t",
    "template_cb",
    "template_ca",
    "template_mask_cb",
    "template_mask_frame",
    "template_mask",
    "query_to_template",
    "visibility_ids",
)

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
        for key in _TEMPLATE_FEATURES
    },
}

# These variable-length index lists are consumed only by the eager steering
# path.  Keeping them in the default jitted feature pytree would fragment its
# abstract signature even though the graph never reads them.
_STEERING_FEATURES = frozenset(
    {
        "rdkit_bounds_index",
        "rdkit_bounds_bond_mask",
        "rdkit_bounds_angle_mask",
        "rdkit_upper_bounds",
        "rdkit_lower_bounds",
        "chiral_atom_index",
        "chiral_reference_mask",
        "chiral_atom_orientations",
        "stereo_bond_index",
        "stereo_reference_mask",
        "stereo_bond_orientations",
        "planar_bond_index",
        "planar_ring_5_index",
        "planar_ring_6_index",
        "connected_chain_index",
        "connected_atom_index",
        "symmetric_chain_index",
        "contact_pair_index",
        "contact_union_index",
        "contact_negation_mask",
        "contact_thresholds",
        "template_force",
        "template_force_threshold",
    }
)

# Every array read by the vendored non-steering forward graph.  This is narrower
# than ``_FEATURE_AXES``: the latter also describes archive/writer fields such as
# ``r_set_to_rep_atom`` that must be padded in legacy feature utilities but must
# not enter the jitted prediction pytree.  Keeping this explicit prevents those
# host-only dimensions from fragmenting the executable cache.
_MODEL_FEATURES = frozenset(
    {
        "affinity_mw",
        "affinity_token_mask",
        "asym_id",
        "atom_pad_mask",
        "atom_to_token",
        "contact_conditioning",
        "contact_threshold",
        "cyclic_period",
        "deletion_mean",
        "deletion_mean_affinity",
        "deletion_value",
        "entity_id",
        "frames_idx",
        "has_deletion",
        "method_feature",
        "modified",
        "mol_type",
        "msa",
        "msa_mask",
        "msa_paired",
        "profile",
        "profile_affinity",
        "ref_atom_name_chars",
        "ref_charge",
        "ref_element",
        "ref_pos",
        "ref_space_uid",
        "res_type",
        "residue_index",
        "sym_id",
        "template_ca",
        "template_cb",
        "template_frame_rot",
        "template_frame_t",
        "template_mask",
        "template_mask_cb",
        "template_mask_frame",
        "template_restype",
        "token_bonds",
        "token_index",
        "token_pad_mask",
        "token_to_rep_atom",
        "type_bonds",
        "visibility_ids",
    }
)


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


def _real_mask_size(value: object, *, name: str) -> int:
    """Count real entries in the first production batch's one-dimensional mask."""

    mask = np.asarray(value)
    first = mask.reshape((-1, mask.shape[-1]))[0].astype(bool)
    size = int(np.count_nonzero(first))
    expected = np.arange(first.size) < size
    if not np.array_equal(first, expected):
        raise ValueError(
            f"{name} real entries must form a contiguous prefix for output cropping"
        )
    return size


def select_model_features_for_padding(
    feats: Mapping[str, object], *, steering_active: bool
) -> dict[str, object]:
    """Return a shape-audited pytree for neutral padded inference.

    Active steering consumes variable-length constraint lists and executes
    eagerly, so it has no stable JIT profile to warm.  Template tensors have a
    separate row dimension that is not part of Boltz2's advertised neutral
    token/atom/MSA contract; reject multiple rows instead of claiming two such
    jobs share a profile when they do not.
    """

    if steering_active:
        raise ValueError("neutral Boltz2 padding does not support active steering")
    template_rows = {
        int(np.shape(value)[1])
        for key, value in feats.items()
        if key in _TEMPLATE_FEATURES
        and np.ndim(value) >= 2
    }
    if len(template_rows) > 1:
        raise ValueError("Boltz2 template features disagree on their row dimension")
    if template_rows and next(iter(template_rows)) != 1:
        raise ValueError(
            "neutral Boltz2 padding currently supports exactly one template row"
        )
    return {
        key: value
        for key, value in feats.items()
        if key in _MODEL_FEATURES and key not in _STEERING_FEATURES
    }


def _feature_sizes(
    feats: Mapping[str, object],
) -> tuple[dict[str, int], dict[str, int]]:
    """Return biological and already-materialized token/atom/MSA sizes."""

    storage = {
        "tokens": int(np.shape(feats["token_pad_mask"])[-1]),
        "atoms": int(np.shape(feats["atom_pad_mask"])[-1]),
        "msa": int(np.shape(feats["msa"])[1]) if "msa" in feats else 1,
    }
    actual = {
        "tokens": _real_mask_size(feats["token_pad_mask"], name="token_pad_mask"),
        "atoms": _real_mask_size(feats["atom_pad_mask"], name="atom_pad_mask"),
        "msa": storage["msa"],
    }
    if "msa_mask" in feats:
        msa_mask = np.asarray(feats["msa_mask"])[0]
        row_axes = tuple(range(1, msa_mask.ndim))
        real_rows = np.any(msa_mask != 0, axis=row_axes) if row_axes else msa_mask != 0
        actual["msa"] = max(1, int(np.count_nonzero(real_rows)))
    return actual, storage


def resolve_padding_plan(
    feats: Mapping[str, object], config: PaddingConfig
) -> PaddingPlan:
    """Resolve the neutral Boltz shape profile over token, atom and MSA axes."""

    actual, storage = _feature_sizes(feats)
    target = {
        axis: resolve_axis(actual[axis], config, axis, minimum=storage[axis])
        for axis in ("tokens", "atoms", "msa")
    }
    if target["atoms"] % 32:
        raise ValueError(
            "padding.atoms must be a multiple of 32 for Boltz2 atom windows"
        )
    return PaddingPlan(actual=actual, storage=storage, target=target)


def resolve_legacy_padding_plan(feats: Mapping[str, object]) -> PaddingPlan:
    """Resolve ``bucket=True`` while retaining its historic overflow policy.

    Token and MSA grids remain the port's legacy grids.  Atom storage now uses
    real buckets instead of merely rounding an already 32-aligned featurizer
    output, so two jobs in one atom bucket can actually share an executable.
    """

    actual, storage = _feature_sizes(feats)
    target_tokens = next(
        (size for size in TOKEN_BUCKETS if size >= storage["tokens"]),
        storage["tokens"],
    )
    target_atoms = next(
        (size for size in ATOM_BUCKETS if size >= storage["atoms"]),
        storage["atoms"],
    )
    # Legacy bucket mode intentionally caps/truncates exceptionally deep MSAs.
    target_msa = next(
        (size for size in MSA_BUCKETS if size >= storage["msa"]),
        MSA_BUCKETS[-1],
    )
    return PaddingPlan(
        actual=actual,
        storage=storage,
        target={
            "tokens": target_tokens,
            "atoms": target_atoms,
            "msa": target_msa,
        },
    )


def resolve_bucket_shape(feats: Mapping[str, object]) -> tuple[int, int, int]:
    """Return the legacy ``(token, atom, MSA-depth)`` bucket for features."""

    plan = resolve_legacy_padding_plan(feats)
    return plan.target["tokens"], plan.target["atoms"], plan.target["msa"]


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
