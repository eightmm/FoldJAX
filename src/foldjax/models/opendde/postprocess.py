"""OpenDDE-specific confidence and ranking postprocessing."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax.nn as jnn
import jax.numpy as jnp
import numpy as np

from foldjax.models._output_features import has_complete_output_atom_metadata
from foldjax.models.protenix.models.heads.confidence import (
    calculate_chain_based_gpde,
    confidence_scores_from_logits,
)

# OpenDDE's managed CLI uses the shared structure writer and then consumes a
# separate feature subset for shape complementarity.  This is intentionally an
# explicit union rather than an import of Protenix's allowlist: the two readers
# have separate contracts and should fail review independently if either grows.
OPENDDE_GENERATED_OUTPUT_FEATURE_FIELDS = frozenset(
    {
        # Shared structure writer.
        "output_atom_name",
        "output_atom_element",
        "output_atom_res_name",
        "output_atom_chain_id",
        "output_atom_res_id",
        "atom_entity_id",
        "output_atom_polymer_type",
        "covalent_atom_indices",
        # OpenDDE 1.1.1 repairs invalid terminal OXT coordinates from the CCD
        # reference frame immediately before serialization.
        "ref_pos",
        "ref_mask",
        # Shape-complementarity resolution.
        "token_index",
        "atom_to_token_idx",
        "asym_id",
        "distogram_rep_atom_mask",
        "structural_distogram_rep_atom_mask",
        "atom_exists_mask",
        "atom_padding_mask",
        "subtoken_role_id",
        "structural_is_protein_token",
        "is_protein_token",
        "is_protein",
        "is_ligand",
        "is_dna",
        "is_rna",
        # Historical host confidence fallback.  The managed full graph emits
        # summaries, but retaining these linear masks keeps an unexpected raw
        # output on the same scoring path instead of silently narrowing it.
        "has_frame",
        "token_padding_mask",
    }
)

SHAPE_COMPLEMENTARITY_SCORE_KEYS = frozenset(
    {
        "shape_comp_token_pred",
        "shape_comp_token_mask",
        "shape_comp_global_pred",
        "shape_comp_pair_mean_pred",
        "shape_comp_pair_topk_mean_pred",
        "shape_comp_valid_pair_frac_pred",
        "shape_comp_uses_structural_tokens",
    }
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
) | SHAPE_COMPLEMENTARITY_SCORE_KEYS

CONFIDENCE_DETAIL_KEYS = frozenset(
    {"token_pair_pae", "token_pair_pde", "contact_probs"}
)

_MIN_TERMINAL_C_OXT_DISTANCE = 1.0
_MAX_TERMINAL_C_OXT_DISTANCE = 1.7
_MIN_TERMINAL_C_CA_DISTANCE = 1.3
_MAX_TERMINAL_C_CA_DISTANCE = 1.7
_MIN_TERMINAL_C_O_DISTANCE = 1.0
_MAX_TERMINAL_C_O_DISTANCE = 1.5
_MIN_TERMINAL_O_OXT_DISTANCE = 1.8


def _local_coordinate_frame(
    carbon: np.ndarray,
    alpha_carbon: np.ndarray,
    oxygen: np.ndarray,
) -> np.ndarray | None:
    ca_axis = alpha_carbon - carbon
    ca_norm = np.linalg.norm(ca_axis)
    if not np.isfinite(ca_norm) or ca_norm < 1e-6:
        return None
    ca_axis = ca_axis / ca_norm

    oxygen_axis = oxygen - carbon
    oxygen_axis -= np.dot(oxygen_axis, ca_axis) * ca_axis
    oxygen_norm = np.linalg.norm(oxygen_axis)
    if not np.isfinite(oxygen_norm) or oxygen_norm < 1e-6:
        return None
    oxygen_axis = oxygen_axis / oxygen_norm
    return np.column_stack((ca_axis, oxygen_axis, np.cross(ca_axis, oxygen_axis)))


def repair_terminal_oxt_coordinates(
    coordinates: Any,
    features: Mapping[str, Any],
) -> tuple[np.ndarray, int]:
    """Apply OpenDDE 1.1.1's free C-terminal OXT serialization repair.

    This is deliberately a writer-side operation: model coordinates and
    confidence calculations retain the network output, while serialized CIFs
    avoid a collapsed/chemically invalid OXT.  Caller-owned arrays are never
    mutated.
    """

    result = np.asarray(coordinates).copy()
    if result.ndim != 3 or result.shape[-1] != 3:
        return result, 0
    required = {
        "output_atom_name",
        "output_atom_chain_id",
        "output_atom_res_id",
        "output_atom_polymer_type",
        "ref_pos",
    }
    if not required <= features.keys():
        return result, 0

    names = np.asarray(features["output_atom_name"], dtype=str)
    chain_ids = np.asarray(features["output_atom_chain_id"], dtype=str)
    residue_ids = np.asarray(features["output_atom_res_id"], dtype=np.int64)
    polymer_types = np.asarray(features["output_atom_polymer_type"], dtype=str)
    reference = np.asarray(features["ref_pos"], dtype=np.float64)
    n_atom = result.shape[-2]
    if any(
        value.shape != (n_atom,)
        for value in (names, chain_ids, residue_ids, polymer_types)
    ) or reference.shape != (n_atom, 3):
        return result, 0
    ref_mask = np.asarray(
        features.get("ref_mask", np.ones(n_atom, dtype=bool)), dtype=bool
    )
    if ref_mask.shape != (n_atom,):
        return result, 0
    bonds = np.asarray(
        features.get("covalent_atom_indices", np.empty((0, 2), dtype=np.int64))
    )
    if bonds.size:
        if bonds.ndim != 2 or bonds.shape[1] != 2:
            return result, 0
        bonds = bonds.astype(np.int64, copy=False)
    else:
        bonds = np.empty((0, 2), dtype=np.int64)

    repaired = 0
    protein = polymer_types == "polypeptide(L)"
    for chain_id in np.unique(chain_ids[protein]):
        chain_mask = protein & (chain_ids == chain_id)
        terminal_residue = np.max(residue_ids[chain_mask])
        terminal_mask = chain_mask & (residue_ids == terminal_residue)
        matches = {
            name: np.flatnonzero(terminal_mask & (names == name))
            for name in ("CA", "C", "O", "OXT")
        }
        if any(len(indices) != 1 for indices in matches.values()):
            continue
        indices = {name: int(value[0]) for name, value in matches.items()}

        if len(bonds):
            terminal_indices = {indices["C"], indices["O"], indices["OXT"]}
            if any(
                (int(left) in terminal_indices and not terminal_mask[int(right)])
                or (int(right) in terminal_indices and not terminal_mask[int(left)])
                for left, right in bonds
                if 0 <= int(left) < n_atom and 0 <= int(right) < n_atom
            ):
                continue

        ordered = np.asarray(
            [indices["C"], indices["CA"], indices["O"], indices["OXT"]]
        )
        if not np.all(ref_mask[ordered]) or not np.all(np.isfinite(reference[ordered])):
            continue
        reference_coordinates = reference[ordered]
        for sample in range(result.shape[0]):
            predicted = np.asarray(result[sample, ordered], dtype=np.float64)
            if not np.all(np.isfinite(predicted[:3])):
                continue
            c_oxt = np.linalg.norm(predicted[3] - predicted[0])
            o_oxt = np.linalg.norm(predicted[3] - predicted[2])
            if (
                np.isfinite(c_oxt)
                and _MIN_TERMINAL_C_OXT_DISTANCE
                <= c_oxt
                <= _MAX_TERMINAL_C_OXT_DISTANCE
                and np.isfinite(o_oxt)
                and o_oxt >= _MIN_TERMINAL_O_OXT_DISTANCE
            ):
                continue
            c_ca = np.linalg.norm(predicted[1] - predicted[0])
            c_o = np.linalg.norm(predicted[2] - predicted[0])
            if not (
                _MIN_TERMINAL_C_CA_DISTANCE <= c_ca <= _MAX_TERMINAL_C_CA_DISTANCE
                and _MIN_TERMINAL_C_O_DISTANCE <= c_o <= _MAX_TERMINAL_C_O_DISTANCE
            ):
                continue
            predicted_frame = _local_coordinate_frame(*predicted[:3])
            reference_frame = _local_coordinate_frame(*reference_coordinates[:3])
            if predicted_frame is None or reference_frame is None:
                continue
            reference_c_oxt = reference_coordinates[3] - reference_coordinates[0]
            reference_c_oxt_distance = np.linalg.norm(reference_c_oxt)
            reference_o_oxt_distance = np.linalg.norm(
                reference_coordinates[3] - reference_coordinates[2]
            )
            if not (
                _MIN_TERMINAL_C_OXT_DISTANCE
                <= reference_c_oxt_distance
                <= _MAX_TERMINAL_C_OXT_DISTANCE
                and reference_o_oxt_distance >= _MIN_TERMINAL_O_OXT_DISTANCE
            ):
                continue
            rotation = predicted_frame @ reference_frame.T
            result[sample, indices["OXT"]] = (
                predicted[0] + rotation @ reference_c_oxt
            )
            repaired += 1
    return result, repaired


def project_generated_output_features(
    features: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Project a managed OpenDDE output snapshot after exact schema proof."""

    if not has_complete_output_atom_metadata(features):
        return features
    return {
        name: features[name]
        for name in OPENDDE_GENERATED_OUTPUT_FEATURE_FIELDS
        if name in features
    }


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
    return_confidence_details: bool = True,
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
        filtered = {
            key: value
            for key, value in scores.items()
            if key in _OPENDDE_SCORE_KEYS
        }
        if not return_confidence_details:
            for key in CONFIDENCE_DETAIL_KEYS:
                filtered.pop(key, None)
        return filtered

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
    filtered = {
        key: value for key, value in scores.items() if key in _OPENDDE_SCORE_KEYS
    }
    if not return_confidence_details:
        for key in CONFIDENCE_DETAIL_KEYS:
            filtered.pop(key, None)
    return filtered


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
        compute_shape_complementarity_batched,
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

    stacked = compute_shape_complementarity_batched(
        coordinate, features, atom_mask, n_token=n_token
    )
    stacked["shape_comp_uses_structural_tokens"] = jnp.asarray(
        "subtoken_role_id" in features
        and jnp.asarray(features["subtoken_role_id"]).reshape(-1).shape[0] == n_token
    )
    return stacked


__all__ = [
    "CONFIDENCE_DETAIL_KEYS",
    "OPENDDE_GENERATED_OUTPUT_FEATURE_FIELDS",
    "SHAPE_COMPLEMENTARITY_SCORE_KEYS",
    "compute_contact_prob",
    "opendde_confidence_scores",
    "project_generated_output_features",
    "repair_terminal_oxt_coordinates",
]
