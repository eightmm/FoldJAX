"""Official 256-crop confidence-head through score-NPZ parity gate."""

from __future__ import annotations

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from foldjax.models.chai.ranking import (
    confidence_logits_to_scores,
    midpoint_bin_centers,
    rank,
)
from foldjax.models.chai.ranking.rank import get_scores

_FIXTURE = Path(__file__).with_name("fixtures") / "official_confidence_256.npz"
_SCORE_FIELDS = {
    "aggregate_score",
    "ptm",
    "iptm",
    "per_chain_ptm",
    "per_chain_pair_iptm",
    "has_inter_chain_clashes",
    "chain_chain_clashes",
}


def _nrmse(actual: np.ndarray, expected: np.ndarray) -> float:
    delta = np.asarray(actual, np.float64) - np.asarray(expected, np.float64)
    denominator = max(float(np.linalg.norm(expected)), 1e-12)
    return float(np.linalg.norm(delta) / denominator)


def test_real_prepared_official_confidence_pipeline_matches_torch_npz(
    tmp_path: Path,
) -> None:
    with np.load(_FIXTURE, allow_pickle=False) as archive:
        fixture = {name: archive[name] for name in archive.files}
    metadata = json.loads(str(fixture["metadata_json"]))
    assert metadata["crop_size"] == 256
    assert metadata["fixture_id"] == "protein_ligand"
    assert metadata["source"] == "official-chai-torch"
    assert metadata["fasta_sha256"] == (
        "338231696a9384f7f4eaa629ad9e89933a0abd2926027943e81f672f1099b742"
    )
    assert metadata["confidence_component_sha256"] == (
        "82a4ac9f934fdd0e73870150f508cecbe010c1383234bdadcaa70d57323deb6b"
    )
    assert int(fixture["token_mask"].sum()) == metadata["active_tokens"]
    assert int(fixture["atom_mask"].sum()) == metadata["active_atoms"]

    # This joins the official 256-crop head gate to public postprocessing:
    # compare the active logits produced from the same real prepared context.
    for name, limit in (("pae", 0.03), ("pde", 0.03), ("plddt", 0.04)):
        actual = fixture[f"jax_{name}_logits"]
        expected = fixture[f"torch_{name}_logits"]
        assert _nrmse(actual, expected) < limit, name

    token_mask = jnp.asarray(fixture["token_mask"])
    atom_mask = jnp.asarray(fixture["atom_mask"])
    atom_token_index = jnp.asarray(fixture["atom_token_index"])
    pae_logits = jnp.asarray(fixture["jax_pae_logits"])
    pde_logits = jnp.asarray(fixture["jax_pde_logits"])
    plddt_logits = jnp.asarray(fixture["jax_plddt_logits"])
    confidence = confidence_logits_to_scores(
        pae_logits,
        pde_logits,
        plddt_logits,
        token_mask=token_mask,
        atom_mask=atom_mask,
        atom_token_index=atom_token_index,
    )
    for name in ("pae", "pde", "plddt"):
        np.testing.assert_allclose(
            np.asarray(getattr(confidence, name)),
            fixture[f"torch_{name}_scores"],
            rtol=2e-2,
            atol=0.08 if name != "plddt" else 0.01,
            err_msg=name,
        )

    ranking = rank(
        atom_coords=jnp.asarray(fixture["coords"]),
        atom_mask=atom_mask[None],
        atom_token_index=atom_token_index[None],
        token_exists_mask=token_mask[None],
        token_asym_id=jnp.asarray(fixture["token_asym_id"])[None],
        token_entity_type=jnp.asarray(fixture["token_entity_type"])[None],
        token_valid_frames_mask=jnp.asarray(fixture["valid_frames_mask"])[None],
        lddt_logits=plddt_logits,
        lddt_bin_centers=midpoint_bin_centers(0.0, 1.0, 50),
        pae_logits=pae_logits,
        pae_bin_centers=midpoint_bin_centers(0.0, 32.0, 64),
    )
    score_path = tmp_path / "scores.model_idx_0.npz"
    np.savez(score_path, **get_scores(ranking))
    with np.load(score_path, allow_pickle=False) as scores:
        assert set(scores.files) == _SCORE_FIELDS
        for name in sorted(_SCORE_FIELDS):
            expected = fixture[f"torch_score__{name}"]
            if expected.dtype.kind in "biu":
                np.testing.assert_array_equal(scores[name], expected)
            else:
                np.testing.assert_allclose(
                    scores[name], expected, rtol=2e-2, atol=2e-3, err_msg=name
                )

    np.testing.assert_array_equal(
        np.asarray(ranking.clash_scores.total_clashes),
        fixture["torch_total_clashes"],
    )
    np.testing.assert_array_equal(
        np.asarray(ranking.clash_scores.total_inter_chain_clashes),
        fixture["torch_total_inter_chain_clashes"],
    )
