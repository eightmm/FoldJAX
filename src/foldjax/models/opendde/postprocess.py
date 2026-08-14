"""OpenDDE-specific confidence and ranking postprocessing."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax.nn as jnn
import jax.numpy as jnp

from foldjax.models.protenix.models.heads.confidence import (
    calculate_chain_based_gpde,
    confidence_scores_from_logits,
)

_OPENDDE_SCORE_KEYS = frozenset(
    {
        "atom_plddt",
        "token_pair_pde",
        "token_pair_pae",
        "contact_probs",
        "summary_plddt",
        "summary_gpde",
        "summary_ptm",
        "summary_iptm",
        "chain_gpde",
        "chain_pair_gpde",
        "chain_ptm",
        "chain_iptm",
        "chain_pair_iptm",
        "chain_pair_iptm_global",
        "chain_plddt",
        "chain_pair_plddt",
        "has_clash",
        "num_recycles",
        "disorder",
        "summary_ranking_score",
        "ranking_score",
        "final_score",
        # Cross-chain shape complementarity. Reported, never ranked -- upstream
        # builds `ranking_score` from iptm, ptm, disorder and clash alone.
        "shape_comp_token_pred",
        "shape_comp_token_mask",
        "shape_comp_global_pred",
        "shape_comp_pair_mean_pred",
        "shape_comp_pair_topk_mean_pred",
        "shape_comp_valid_pair_frac_pred",
        "shape_comp_uses_structural_tokens",
    }
)


def compute_contact_prob(
    distogram_logits: jnp.ndarray,
    *,
    min_bin: float = 2.25,
    max_bin: float = 25.75,
    no_bins: int = 96,
    threshold: float = 8.0,
) -> jnp.ndarray:
    """Compute contacts using OpenDDE's inclusive distogram bin tops."""

    if int(distogram_logits.shape[-1]) != no_bins:
        raise ValueError(
            f"distogram logits expected {no_bins} bins, "
            f"got {distogram_logits.shape[-1]}"
        )
    breaks = jnp.linspace(
        min_bin,
        max_bin,
        no_bins - 1,
        dtype=jnp.float32,
    )
    bin_tops = jnp.concatenate((breaks, jnp.asarray([jnp.inf], dtype=breaks.dtype)))
    probabilities = jnn.softmax(distogram_logits.astype(jnp.float32), axis=-1)
    return jnp.sum(
        probabilities * (bin_tops <= threshold).astype(probabilities.dtype),
        axis=-1,
    )


def opendde_confidence_scores(
    output: Mapping[str, Any],
    features: Mapping[str, Any],
    *,
    num_recycles: int,
    n_chain: int | None = None,
    include_shape_complementarity: bool = True,
) -> dict[str, jnp.ndarray]:
    """Convert raw OpenDDE logits to released confidence summaries.

    When the program already computed them in-graph (`run_confidence_scores`
    in `opendde_infer_static`), pass its output straight through: recomputing
    on the host would need the raw logits, which that path deliberately no
    longer returns.
    """

    if "summary_ranking_score" in output:
        scores = {
            key: jnp.asarray(value)
            for key, value in output.items()
            if key in _OPENDDE_SCORE_KEYS
        }
        # Shape complementarity is numpy-based and cannot trace, so the
        # in-graph path leaves it out and it is finished here, on concrete
        # arrays. It is a reported field only -- never part of ranking_score.
        scores.update(
            _shape_complementarity_scores(
                output,
                features,
                n_token=int(jnp.asarray(features["asym_id"]).shape[0]),
            )
        )
        return {
            key: value
            for key, value in scores.items()
            if key in _OPENDDE_SCORE_KEYS
        }

    required_output = {"coordinate", "plddt", "pae", "pde", "distogram_logits"}
    missing_output = sorted(required_output - output.keys())
    if missing_output:
        raise KeyError("missing raw OpenDDE output(s): " + ", ".join(missing_output))
    required_features = {"has_frame", "asym_id", "atom_to_token_idx"}
    missing_features = sorted(required_features - features.keys())
    if missing_features:
        raise KeyError(
            "missing OpenDDE confidence feature(s): " + ", ".join(missing_features)
        )

    atom_to_token_idx = jnp.asarray(features["atom_to_token_idx"], dtype=jnp.int32)
    token_asym_id = jnp.asarray(features["asym_id"], dtype=jnp.int32)
    n_token = int(token_asym_id.shape[0])
    atom_is_polymer = jnp.zeros(atom_to_token_idx.shape, dtype=bool)
    found_polymer_mask = False
    for name in ("is_protein", "is_dna", "is_rna"):
        if name in features:
            atom_is_polymer |= jnp.asarray(features[name]).astype(bool)
            found_polymer_mask = True
    if not found_polymer_mask:
        raise KeyError(
            "OpenDDE confidence requires per-atom is_protein/is_dna/is_rna masks"
        )
    atom_padding_mask = features.get("atom_padding_mask")
    if atom_padding_mask is None:
        real_atom_mask = jnp.ones(atom_to_token_idx.shape, dtype=bool)
    else:
        real_atom_mask = jnp.asarray(atom_padding_mask).astype(bool)
        if real_atom_mask.shape != atom_to_token_idx.shape:
            raise ValueError(
                "atom_padding_mask must share shape [N_atom] with atom_to_token_idx"
            )
    token_ligand_count = (
        jnp.zeros((n_token,), dtype=jnp.int32)
        .at[atom_to_token_idx]
        .add(((~atom_is_polymer) & real_atom_mask).astype(jnp.int32))
    )
    token_is_ligand = token_ligand_count > 0

    scores = confidence_scores_from_logits(
        plddt_logits=jnp.asarray(output["plddt"]),
        pae_logits=jnp.asarray(output["pae"]),
        pde_logits=jnp.asarray(output["pde"]),
        distogram_logits=jnp.asarray(output["distogram_logits"]),
        plddt_min_bin=0.0,
        plddt_max_bin=1.0,
        plddt_no_bins=50,
        pae_min_bin=0.0,
        pae_max_bin=32.0,
        pae_no_bins=64,
        pde_min_bin=0.0,
        pde_max_bin=32.0,
        pde_no_bins=64,
        distogram_min_bin=2.25,
        distogram_max_bin=25.75,
        distogram_no_bins=96,
        contact_threshold=8.0,
        token_has_frame=jnp.asarray(features["has_frame"]),
        token_asym_id=token_asym_id,
        atom_to_token_idx=atom_to_token_idx,
        atom_coordinate=jnp.asarray(output["coordinate"]),
        atom_is_polymer=atom_is_polymer,
        elements_one_hot=None,
        mol_id=None,
        token_is_ligand=token_is_ligand,
        num_recycles=num_recycles,
        n_chain=n_chain,
        include_chain_pair_pae=False,
        token_mask=features.get("token_padding_mask"),
        atom_mask=features.get("atom_padding_mask"),
    )

    contact_probs = compute_contact_prob(jnp.asarray(output["distogram_logits"]))
    token_pair_pde = scores["token_pair_pde"]
    token_mask = features.get("token_padding_mask")
    if token_mask is not None:
        token_mask = jnp.asarray(token_mask).astype(bool)
        pair_mask = token_mask[:, None] & token_mask[None, :]
        contact_probs = contact_probs * pair_mask.astype(contact_probs.dtype)
    gpde_denominator = jnp.sum(contact_probs, axis=(-1, -2))
    scores["contact_probs"] = contact_probs.astype(jnp.float32)
    gpde_numerator = jnp.sum(token_pair_pde * contact_probs, axis=(-1, -2))
    scores["summary_gpde"] = jnp.where(
        gpde_denominator > 0,
        gpde_numerator / gpde_denominator,
        0.0,
    ).astype(jnp.float32)
    scores.update(
        calculate_chain_based_gpde(
            token_pair_pde,
            contact_probs,
            token_asym_id,
            n_chain=n_chain,
            token_mask=token_mask,
        )
    )
    if include_shape_complementarity:
        scores.update(
            _shape_complementarity_scores(output, features, n_token=n_token)
        )
    return {key: value for key, value in scores.items() if key in _OPENDDE_SCORE_KEYS}


def _shape_complementarity_scores(
    output: Mapping[str, Any],
    features: Mapping[str, Any],
    *,
    n_token: int,
) -> dict[str, jnp.ndarray]:
    """Upstream's six cross-chain shape-complementarity fields, per sample.

    Upstream computes these on every prediction -- `alpha_shape_comp` is 3e-2
    and the three sub-weights are non-zero, so `_should_compute_shape_comp()` is
    true by default -- and this port reported none of them.

    Mapped over the sample axis rather than batched: the pair map is
    `[N_token, N_token]` and stacking five samples of it is the largest array
    postprocessing would hold. Skipped rather than raised when the features it
    needs are absent, because the trunk-only and component tests build partial
    dicts and this is a reported field, not a structural one -- it never enters
    `ranking_score`, which upstream builds from iptm, ptm, disorder and clash
    alone (`sample_confidence.py:163`).
    """
    from foldjax.models.opendde.models.shape_complementarity import (
        compute_shape_complementarity,
    )

    needed = {"token_index", "atom_to_token_idx", "asym_id"}
    if not needed <= features.keys():
        return {}
    if not (
        {"distogram_rep_atom_mask", "structural_distogram_rep_atom_mask"}
        & features.keys()
    ):
        return {}

    coordinate = jnp.asarray(output["coordinate"])
    atom_mask = features.get("atom_exists_mask")
    if atom_mask is None:
        atom_mask = features.get("atom_padding_mask")
    atom_mask = (
        jnp.ones(coordinate.shape[-2], dtype=bool)
        if atom_mask is None
        else jnp.asarray(atom_mask).astype(bool).reshape(-1)
    )

    samples = [coordinate] if coordinate.ndim == 2 else list(coordinate)
    per_sample = [
        compute_shape_complementarity(sample, features, atom_mask)
        for sample in samples
    ]
    stacked = {
        key: jnp.stack([entry[key] for entry in per_sample])
        for key in per_sample[0]
    }
    if coordinate.ndim == 2:
        stacked = {key: value[0] for key, value in stacked.items()}
    stacked["shape_comp_uses_structural_tokens"] = jnp.asarray(
        "subtoken_role_id" in features
        and jnp.asarray(features["subtoken_role_id"]).reshape(-1).shape[0] == n_token
    )
    return stacked


__all__ = ["compute_contact_prob", "opendde_confidence_scores"]
