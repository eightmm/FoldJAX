"""OpenDDE's cross-chain shape complementarity, as a pure function of geometry.

Upstream computes this on every prediction -- `alpha_shape_comp` is 3e-2 and the
three sub-weights are non-zero, so `_should_compute_shape_comp()` is true by
default -- and emits six confidence fields from it. This port had none of them.

Three properties make it portable without a checkpoint:

* It reads no learned weights. Every number here comes from the predicted
  coordinates and the token features.
* It is scored strictly across chains: `cross_chain` below is
  `asym_id[i] != asym_id[j]`, so a single-chain job scores identically zero.
  That is why the absence never showed in this project's benchmark, whose four
  cases are all single-chain -- and why a test on one chain proves nothing.
* It never enters `ranking_score`, which upstream builds from
  `0.8*iptm + 0.2*ptm + 0.5*disorder - 100*has_clash`
  (`sample_confidence.py:163`). These are reported fields, not a ranking input.

Upstream chunks the two quadratic passes over tokens and optionally
gradient-checkpoints them. Both exist for training: `use_checkpoint` is
`checkpoint_chunks and coord.requires_grad`, which is false under inference, and
the chunk loop then only changes the order of a reduction. The chunking is kept
here anyway, because the pair map is `[N_token, N_token]` and the density pass
is `[N_token, N_atom]` -- at OpenDDE's structural token count those are the two
largest arrays in postprocessing.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from foldjax.models.opendde.models.structural_tokens import STRUCTURAL_TOKEN_ROLES

#: Upstream's `confidence.shape_comp` block (`opendde/config/model_base.py`).
#: Repeated rather than imported because this port does not load upstream's
#: config objects, and a silently different default here would move every
#: reported score.
SHAPE_COMP_DEFAULTS: dict[str, float | int | bool] = {
    "density_sigma": 1.5,
    "interface_cutoff": 16.0,
    "gap_mean": 6.0,
    "gap_scale": 3.0,
    "clash_distance": 2.0,
    "clash_scale": 0.5,
    "pool_temperature": 16.0,
    "normal_strength_min": 1e-4,
    "pair_chunk_size": 128,
    "eps": 1e-6,
}

#: `build_shape_comp_pred_outputs` summarizes the pair map over its best 32.
PAIR_SUMMARY_TOPK = 32


class ShapeCompTokenFeatures(NamedTuple):
    """What the geometry needs, resolved out of a feature dict."""

    atom_to_token_idx: np.ndarray
    rep_atom_indices: np.ndarray
    rep_atom_valid: np.ndarray
    token_asym_id: np.ndarray
    token_role_id: np.ndarray
    is_structural: bool
    is_protein_token: np.ndarray


def _as_numpy(value: Any) -> np.ndarray:
    return np.asarray(value)


def _select_rep_atom_mask(
    features: Mapping[str, Any],
    atom_to_token_idx: np.ndarray,
    n_token: int,
    is_structural: bool,
) -> np.ndarray:
    """The rep-atom mask that actually indexes this token space, one per token.

    Upstream tries both keys in an order that depends on the branch and accepts
    the first whose selected atoms hit every token exactly once. Taking the
    structural key on a residue-token job -- or the reverse -- yields a mask of
    the wrong length that would otherwise scatter into the wrong tokens, so the
    check is the point rather than a formality.
    """
    candidates = (
        ("structural_distogram_rep_atom_mask", "distogram_rep_atom_mask")
        if is_structural
        else ("distogram_rep_atom_mask", "structural_distogram_rep_atom_mask")
    )
    expected = np.arange(n_token)
    checked: list[str] = []
    for key in candidates:
        if key not in features:
            continue
        mask = _as_numpy(features[key]).astype(bool).reshape(-1)
        checked.append(key)
        if mask.shape[0] != atom_to_token_idx.shape[0]:
            continue
        rep_atom_idx = np.flatnonzero(mask)
        if rep_atom_idx.shape[0] != n_token:
            continue
        if np.array_equal(np.sort(atom_to_token_idx[rep_atom_idx]), expected):
            return mask
    raise ValueError(
        "could not resolve a representative atom mask for shape complementarity "
        f"with n_token={n_token}; checked={checked}"
    )


def _token_any(
    features: Mapping[str, Any],
    name: str,
    atom_to_token_idx: np.ndarray,
    n_token: int,
) -> np.ndarray:
    flag = _as_numpy(features[name]).astype(np.float32).reshape(-1)
    summed = np.zeros(n_token, dtype=np.float32)
    np.add.at(summed, atom_to_token_idx, flag)
    return summed > 0.5


def _residue_protein_token_mask(
    features: Mapping[str, Any],
    atom_to_token_idx: np.ndarray,
    n_token: int,
) -> np.ndarray:
    if "is_protein_token" in features:
        candidate = _as_numpy(features["is_protein_token"]).reshape(-1)
        if candidate.shape[0] == n_token:
            return candidate.astype(bool)
    atom_count = np.zeros(n_token, dtype=np.float32)
    np.add.at(atom_count, atom_to_token_idx, 1.0)
    return (
        _token_any(features, "is_protein", atom_to_token_idx, n_token)
        & ~_token_any(features, "is_ligand", atom_to_token_idx, n_token)
        & ~_token_any(features, "is_dna", atom_to_token_idx, n_token)
        & ~_token_any(features, "is_rna", atom_to_token_idx, n_token)
        # A single-atom "protein" token is an ion or a modified residue stub;
        # it has no surface to complement.
        & (atom_count > 1)
    )


def resolve_shape_comp_token_features(
    features: Mapping[str, Any],
    atom_mask: np.ndarray,
    n_token: int | None = None,
) -> ShapeCompTokenFeatures:
    """Resolve the token-level inputs, failing loudly on a mismatched space."""

    if n_token is None:
        n_token = int(_as_numpy(features["token_index"]).reshape(-1).shape[0])
    atom_to_token_idx = _as_numpy(features["atom_to_token_idx"]).astype(np.int64)
    atom_to_token_idx = atom_to_token_idx.reshape(-1)
    if atom_to_token_idx.size == 0:
        raise ValueError("shape complementarity requires at least one atom")
    if int(atom_to_token_idx.max()) + 1 != n_token:
        raise ValueError(
            "atom_to_token_idx does not match the active token space: "
            f"max+1={int(atom_to_token_idx.max()) + 1}, n_token={n_token}"
        )
    token_asym_id = _as_numpy(features["asym_id"]).astype(np.int64).reshape(-1)
    if token_asym_id.shape[0] != n_token:
        raise ValueError(
            f"asym_id does not match the active token space: "
            f"{token_asym_id.shape} vs ({n_token},)"
        )

    role = features.get("subtoken_role_id")
    role_array = None if role is None else _as_numpy(role).reshape(-1)
    is_structural = role_array is not None and role_array.shape[0] == n_token

    rep_atom_mask = _select_rep_atom_mask(
        features, atom_to_token_idx, n_token, is_structural
    )
    # Sorted by token, so row `t` of the gathered coordinates is token `t`.
    rep_atom_indices = np.flatnonzero(rep_atom_mask)
    rep_atom_indices = rep_atom_indices[np.argsort(atom_to_token_idx[rep_atom_indices])]

    if is_structural:
        token_role_id = role_array.astype(np.int64)
        structural_protein = features.get("structural_is_protein_token")
        if (
            structural_protein is not None
            and _as_numpy(structural_protein).reshape(-1).shape[0] == n_token
        ):
            is_protein_token = _as_numpy(structural_protein).reshape(-1).astype(bool)
        else:
            is_protein_token = (
                token_role_id == STRUCTURAL_TOKEN_ROLES["protein_bb"]
            ) | (token_role_id == STRUCTURAL_TOKEN_ROLES["protein_sc"])
    else:
        token_role_id = np.full(n_token, -1, dtype=np.int64)
        is_protein_token = _residue_protein_token_mask(
            features, atom_to_token_idx, n_token
        )

    return ShapeCompTokenFeatures(
        atom_to_token_idx=atom_to_token_idx,
        rep_atom_indices=rep_atom_indices,
        rep_atom_valid=np.asarray(atom_mask).astype(bool).reshape(-1)[rep_atom_indices],
        token_asym_id=token_asym_id,
        token_role_id=token_role_id,
        is_structural=is_structural,
        is_protein_token=is_protein_token,
    )


def _masked_softmax(
    logits: jnp.ndarray, mask: jnp.ndarray, *, eps: float
) -> jnp.ndarray:
    """Softmax over the last axis, with an all-masked row returning zeros.

    `where` before the exponential, not after: a masked entry can hold a
    distance of zero against a partner it never had, and `exp` of the shifted
    logit would contribute to the denominator.
    """
    large_negative = jnp.finfo(logits.dtype).min
    masked = jnp.where(mask, logits, large_negative)
    peak = jnp.max(masked, axis=-1, keepdims=True)
    peak = jnp.where(jnp.any(mask, axis=-1, keepdims=True), peak, 0.0)
    weights = jnp.where(mask, jnp.exp(masked - peak), 0.0)
    denominator = jnp.maximum(jnp.sum(weights, axis=-1, keepdims=True), eps)
    return jnp.where(mask, weights / denominator, 0.0)


def _token_centers_and_normals(
    coordinate: jnp.ndarray,
    atom_mask: jnp.ndarray,
    resolved: ShapeCompTokenFeatures,
    *,
    n_token: int,
    density_sigma: float,
    chunk_size: int,
    eps: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Per-token surface point and outward normal.

    The normal is the gradient of a Gaussian atom density restricted to the
    token's *own* chain, which points away from that chain's bulk -- the
    direction a complementary surface has to face.
    """
    atom_to_token = jnp.asarray(resolved.atom_to_token_idx)
    atom_mask_float = atom_mask.astype(coordinate.dtype)
    rep_center = coordinate[jnp.asarray(resolved.rep_atom_indices)]
    rep_valid = jnp.asarray(resolved.rep_atom_valid)

    if resolved.is_structural:
        # A sidechain subtoken's centre is the mean of the atoms it supervises,
        # not its representative atom: the representative is shared with the
        # backbone subtoken and would give the two the same surface point.
        supervised_count = jnp.zeros(n_token, coordinate.dtype).at[atom_to_token].add(
            atom_mask_float
        )
        supervised_sum = jnp.zeros((n_token, 3), coordinate.dtype).at[
            atom_to_token
        ].add(coordinate * atom_mask_float[:, None])
        supervised_center = supervised_sum / jnp.maximum(supervised_count, 1.0)[:, None]
        sidechain = jnp.asarray(
            resolved.token_role_id == STRUCTURAL_TOKEN_ROLES["protein_sc"]
        )
        token_center = jnp.where(sidechain[:, None], supervised_center, rep_center)
        center_valid = jnp.where(sidechain, supervised_count > 0, rep_valid)
    else:
        token_center = rep_center
        center_valid = rep_valid

    token_asym_id = jnp.asarray(resolved.token_asym_id)
    atom_asym_id = jnp.asarray(
        resolved.token_asym_id[resolved.atom_to_token_idx]
    )

    def gradient_chunk(start: int, size: int) -> jnp.ndarray:
        centers = jax.lax.dynamic_slice_in_dim(token_center, start, size, axis=0)
        center_asym_id = jax.lax.dynamic_slice_in_dim(
            token_asym_id, start, size, axis=0
        )
        delta = centers[:, None, :] - coordinate[None, :, :]
        weight = jnp.exp(-jnp.sum(delta * delta, -1) / (2.0 * density_sigma**2))
        same_chain = (center_asym_id[:, None] == atom_asym_id[None, :]) & atom_mask[
            None, :
        ]
        weight = weight * same_chain.astype(coordinate.dtype)
        return jnp.sum(weight[..., None] * delta, axis=-2) / (density_sigma**2)

    gradient = jnp.concatenate(
        [
            gradient_chunk(start, min(chunk_size, n_token - start))
            for start in range(0, n_token, chunk_size)
        ],
        axis=0,
    )
    strength = jnp.linalg.norm(gradient, axis=-1)
    normal = gradient / jnp.maximum(strength, eps)[:, None]
    return token_center, normal, strength, center_valid


def compute_shape_complementarity(
    coordinate: jnp.ndarray,
    features: Mapping[str, Any],
    atom_mask: jnp.ndarray,
    **overrides: Any,
) -> dict[str, jnp.ndarray]:
    """Return upstream's six reported shape-complementarity fields.

    `coordinate` is `[N_atom, 3]` for one sample; callers with a sample axis map
    over it, because the pair map is quadratic in tokens and stacking samples
    would multiply the largest array in postprocessing by the sample count.
    """
    settings = {**SHAPE_COMP_DEFAULTS, **overrides}
    eps = float(settings["eps"])
    if coordinate.ndim != 2 or coordinate.shape[-1] != 3:
        raise ValueError(
            f"coordinate must be [N_atom, 3]; got {tuple(coordinate.shape)}"
        )

    n_token = int(np.asarray(features["token_index"]).reshape(-1).shape[0])
    resolved = resolve_shape_comp_token_features(
        features, np.asarray(atom_mask), n_token
    )
    coordinate = coordinate.astype(jnp.float32)
    atom_mask = atom_mask.astype(bool)
    chunk = int(settings["pair_chunk_size"] or n_token)

    center, normal, strength, center_valid = _token_centers_and_normals(
        coordinate,
        atom_mask,
        resolved,
        n_token=n_token,
        density_sigma=float(settings["density_sigma"]),
        chunk_size=chunk,
        eps=eps,
    )
    token_valid = (
        center_valid
        & jnp.asarray(resolved.is_protein_token)
        & (strength > float(settings["normal_strength_min"]))
    )
    token_asym_id = jnp.asarray(resolved.token_asym_id)

    token_scores: list[jnp.ndarray] = []
    token_masks: list[jnp.ndarray] = []
    pair_sum = jnp.zeros((), coordinate.dtype)
    pair_count = jnp.zeros((), coordinate.dtype)
    topk_pool: list[jnp.ndarray] = []

    for start in range(0, n_token, chunk):
        size = min(chunk, n_token - start)
        centers = jax.lax.dynamic_slice_in_dim(center, start, size, axis=0)
        normals = jax.lax.dynamic_slice_in_dim(normal, start, size, axis=0)
        valid = jax.lax.dynamic_slice_in_dim(token_valid, start, size, axis=0)
        row_asym_id = jax.lax.dynamic_slice_in_dim(
            token_asym_id, start, size, axis=0
        )

        delta = center[None, :, :] - centers[:, None, :]
        distance = jnp.linalg.norm(delta, axis=-1)
        unit = delta / jnp.maximum(distance, eps)[..., None]

        # Both surfaces must point at each other, their normals must oppose,
        # the gap must be near `gap_mean`, and they must not interpenetrate.
        facing = jax.nn.relu(jnp.sum(normals[:, None, :] * unit, -1)) * jax.nn.relu(
            jnp.sum(normal[None, :, :] * -unit, -1)
        )
        opposite = 0.5 * (1.0 - jnp.sum(normals[:, None, :] * normal[None, :, :], -1))
        gap = jnp.exp(
            -(
                ((distance - float(settings["gap_mean"]))
                 / float(settings["gap_scale"])) ** 2
            )
        )
        anti_clash = 1.0 - jax.nn.sigmoid(
            (float(settings["clash_distance"]) - distance)
            / float(settings["clash_scale"])
        )

        pair_mask = (
            valid[:, None]
            & token_valid[None, :]
            & (row_asym_id[:, None] != token_asym_id[None, :])
            & (distance <= float(settings["interface_cutoff"]))
        )
        pair_score = jnp.where(pair_mask, facing * opposite * gap * anti_clash, 0.0)

        partner = _masked_softmax(
            -(distance * distance) / float(settings["pool_temperature"]),
            pair_mask,
            eps=eps,
        )
        row_has_partner = jnp.any(pair_mask, axis=-1)
        token_scores.append(
            jnp.where(row_has_partner, jnp.sum(partner * pair_score, -1), 0.0)
        )
        token_masks.append(row_has_partner)

        mask_float = pair_mask.astype(coordinate.dtype)
        pair_sum = pair_sum + jnp.sum(pair_score * mask_float)
        pair_count = pair_count + jnp.sum(mask_float)
        # Upstream keeps a running top-32 seeded with -inf and merges each
        # chunk's own top-32 into it, so a chunk with fewer than 32 valid pairs
        # contributes only what it has. Scoring masked entries as -inf rather
        # than their stored 0.0 is what reproduces that: a valid pair can score
        # exactly 0.0, and without this an all-masked chunk would donate 32
        # zeros to the mean.
        flat = jnp.where(pair_mask, pair_score, -jnp.inf).reshape(-1)
        topk_pool.append(jax.lax.top_k(flat, min(PAIR_SUMMARY_TOPK, flat.shape[0]))[0])

    token_score = jnp.concatenate(token_scores)
    token_mask = jnp.concatenate(token_masks)
    denominator = jnp.maximum(jnp.sum(token_mask.astype(coordinate.dtype)), 1.0)
    global_score = jnp.where(
        jnp.any(token_mask), jnp.sum(token_score) / denominator, 0.0
    )

    candidates = jnp.concatenate(topk_pool)
    keep = jax.lax.top_k(candidates, min(PAIR_SUMMARY_TOPK, candidates.shape[0]))[0]
    finite = jnp.isfinite(keep)
    topk_mean = jnp.where(
        jnp.any(finite),
        jnp.sum(jnp.where(finite, keep, 0.0))
        / jnp.maximum(jnp.sum(finite.astype(coordinate.dtype)), 1.0),
        0.0,
    )

    pair_mean = jnp.where(pair_count > 0, pair_sum / jnp.maximum(pair_count, 1.0), 0.0)
    total_pairs = float(max(n_token * n_token, 1))
    return {
        "shape_comp_token_pred": token_score,
        "shape_comp_token_mask": token_mask,
        "shape_comp_global_pred": global_score,
        "shape_comp_pair_mean_pred": pair_mean,
        "shape_comp_pair_topk_mean_pred": topk_mean,
        "shape_comp_valid_pair_frac_pred": pair_count / total_pairs,
    }
