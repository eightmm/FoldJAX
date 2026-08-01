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
) -> dict[str, jnp.ndarray]:
    """Convert raw OpenDDE logits to released confidence summaries."""

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
    token_ligand_count = (
        jnp.zeros((n_token,), dtype=jnp.int32)
        .at[atom_to_token_idx]
        .add((~atom_is_polymer).astype(jnp.int32))
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
        include_chain_pair_pae=False,
    )

    contact_probs = compute_contact_prob(jnp.asarray(output["distogram_logits"]))
    token_pair_pde = scores["token_pair_pde"]
    gpde_denominator = jnp.sum(contact_probs, axis=(-1, -2))
    scores["contact_probs"] = contact_probs.astype(jnp.float32)
    scores["summary_gpde"] = (
        jnp.sum(token_pair_pde * contact_probs, axis=(-1, -2)) / gpde_denominator
    ).astype(jnp.float32)
    scores.update(
        calculate_chain_based_gpde(
            token_pair_pde,
            contact_probs,
            token_asym_id,
        )
    )
    return {key: value for key, value in scores.items() if key in _OPENDDE_SCORE_KEYS}


__all__ = ["compute_contact_prob", "opendde_confidence_scores"]
