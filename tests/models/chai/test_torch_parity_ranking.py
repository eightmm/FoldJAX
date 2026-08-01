"""Parity gates for pure JAX Chai confidence postprocessing and ranking."""

from __future__ import annotations

import subprocess
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.chai.ranking import (
    confidence_logits_to_scores,
    midpoint_bin_centers,
    rank,
)
from foldjax.models.chai.ranking.rank import get_scores

pytestmark = pytest.mark.official_parity


def _synthetic_inputs() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(20260714)
    n_token, n_atom = 6, 8
    coords = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [6.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [9.0, 0.0, 0.0],
            [12.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
            [23.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )[None]
    return {
        "pae_logits": rng.normal(size=(1, n_token, n_token, 64)).astype(np.float32),
        "pde_logits": rng.normal(size=(1, n_token, n_token, 64)).astype(np.float32),
        "plddt_logits": rng.normal(size=(1, n_atom, 50)).astype(np.float32),
        "atom_coords": coords,
        "atom_mask": np.ones((1, n_atom), dtype=np.bool_),
        "atom_token_index": np.asarray([[0, 0, 1, 2, 3, 3, 4, 4]], dtype=np.int64),
        "token_exists_mask": np.asarray(
            [[True, True, True, True, True, False]], dtype=np.bool_
        ),
        "token_asym_id": np.asarray([[1, 1, 2, 2, 3, 0]], dtype=np.int64),
        "token_entity_type": np.asarray([[0, 0, 0, 0, 3, 6]], dtype=np.int64),
        "token_valid_frames_mask": np.asarray(
            [[True, True, True, True, True, False]], dtype=np.bool_
        ),
    }


def _run_upstream_reference(
    tmp_path: Path,
    inputs: dict[str, np.ndarray],
    upstream_chai_dir: Path,
    upstream_chai_python: Path,
) -> dict[str, np.ndarray]:
    input_path = tmp_path / "ranking_inputs.npz"
    output_path = tmp_path / "ranking_reference.npz"
    np.savez(input_path, **inputs)
    script = r"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path.cwd()))
from chai_lab.ranking.rank import get_scores, rank

d = np.load(sys.argv[1])
t = lambda name: torch.from_numpy(d[name])
pair_centers = torch.linspace(0.0, 32.0, 2 * 64 + 1)[1::2]
plddt_centers = torch.linspace(
    0.0, 1.0, 2 * d["plddt_logits"].shape[-1] + 1
)[1::2]
ranking = rank(
    atom_coords=t("atom_coords"),
    atom_mask=t("atom_mask"),
    atom_token_index=t("atom_token_index"),
    token_exists_mask=t("token_exists_mask"),
    token_asym_id=t("token_asym_id"),
    token_entity_type=t("token_entity_type"),
    token_valid_frames_mask=t("token_valid_frames_mask"),
    lddt_logits=t("plddt_logits"),
    lddt_bin_centers=plddt_centers,
    pae_logits=t("pae_logits"),
    pae_bin_centers=pair_centers,
    max_clashes=1,
    max_clash_ratio=0.5,
)
scores = get_scores(ranking)

token_mask = t("token_exists_mask")[0]
atom_mask = t("atom_mask")[0]
atom_token_index = t("atom_token_index")[0]
pae = (torch.softmax(t("pae_logits"), -1) * pair_centers).sum(-1)
pde = (torch.softmax(t("pde_logits"), -1) * pair_centers).sum(-1)
pae = pae[:, token_mask][:, :, token_mask]
pde = pde[:, token_mask][:, :, token_mask]
atom_plddt = (torch.softmax(t("plddt_logits"), -1) * plddt_centers).sum(-1)
token_plddt = []
for sample in atom_plddt:
    numerator = torch.bincount(
        atom_token_index[atom_mask],
        weights=sample[atom_mask],
        minlength=token_mask.numel(),
    )
    denominator = torch.bincount(
        atom_token_index[atom_mask], minlength=token_mask.numel()
    ).clamp(min=1)
    token_plddt.append(numerator / denominator)
token_plddt = torch.stack(token_plddt)[:, token_mask]

np.savez(
    sys.argv[2],
    **scores,
    complex_plddt=ranking.plddt_scores.complex_plddt.numpy(),
    per_chain_plddt=ranking.plddt_scores.per_chain_plddt.numpy(),
    per_atom_plddt=ranking.plddt_scores.per_atom_plddt.numpy(),
    total_clashes=ranking.clash_scores.total_clashes.numpy(),
    total_inter_chain_clashes=ranking.clash_scores.total_inter_chain_clashes.numpy(),
    conf_pae=pae.numpy(),
    conf_pde=pde.numpy(),
    conf_plddt=token_plddt.numpy(),
    conf_atom_plddt=atom_plddt.numpy(),
)
"""
    subprocess.run(
        [upstream_chai_python, "-c", script, input_path, output_path],
        cwd=upstream_chai_dir,
        check=True,
        capture_output=True,
        text=True,
    )
    with np.load(output_path) as result:
        return {key: result[key] for key in result.files}


def test_midpoint_bin_contract() -> None:
    np.testing.assert_array_equal(
        np.asarray(midpoint_bin_centers(0.0, 32.0, 64)),
        np.linspace(0.0, 32.0, 129, dtype=np.float32)[1::2],
    )
    np.testing.assert_allclose(
        np.asarray(midpoint_bin_centers(0.0, 1.0, 50)),
        np.linspace(0.0, 1.0, 101, dtype=np.float32)[1::2],
        rtol=1e-7,
        atol=1e-7,
    )


def test_ranking_and_confidence_postprocessing_match_upstream(
    tmp_path: Path, upstream_chai_dir: Path, upstream_chai_python: Path
) -> None:
    inputs = _synthetic_inputs()
    reference = _run_upstream_reference(
        tmp_path, inputs, upstream_chai_dir, upstream_chai_python
    )
    array = {key: jnp.asarray(value) for key, value in inputs.items()}
    pair_centers = midpoint_bin_centers(0.0, 32.0, 64)
    plddt_centers = midpoint_bin_centers(0.0, 1.0, 50)
    ranking = rank(
        atom_coords=array["atom_coords"],
        atom_mask=array["atom_mask"],
        atom_token_index=array["atom_token_index"],
        token_exists_mask=array["token_exists_mask"],
        token_asym_id=array["token_asym_id"],
        token_entity_type=array["token_entity_type"],
        token_valid_frames_mask=array["token_valid_frames_mask"],
        lddt_logits=array["plddt_logits"],
        lddt_bin_centers=plddt_centers,
        pae_logits=array["pae_logits"],
        pae_bin_centers=pair_centers,
        max_clashes=1,
        max_clash_ratio=0.5,
    )
    scores = get_scores(ranking)
    for key in (
        "aggregate_score",
        "ptm",
        "iptm",
        "per_chain_ptm",
        "per_chain_pair_iptm",
    ):
        np.testing.assert_allclose(scores[key], reference[key], rtol=2e-6, atol=2e-6)
    for key in ("has_inter_chain_clashes", "chain_chain_clashes"):
        np.testing.assert_array_equal(scores[key], reference[key])

    np.testing.assert_allclose(
        np.asarray(ranking.plddt_scores.complex_plddt),
        reference["complex_plddt"],
        rtol=2e-6,
        atol=2e-6,
    )
    np.testing.assert_allclose(
        np.asarray(ranking.plddt_scores.per_chain_plddt),
        reference["per_chain_plddt"],
        rtol=2e-6,
        atol=2e-6,
    )
    np.testing.assert_allclose(
        np.asarray(ranking.plddt_scores.per_atom_plddt),
        reference["per_atom_plddt"],
        rtol=2e-6,
        atol=2e-6,
    )
    np.testing.assert_array_equal(
        np.asarray(ranking.clash_scores.total_clashes), reference["total_clashes"]
    )
    np.testing.assert_array_equal(
        np.asarray(ranking.clash_scores.total_inter_chain_clashes),
        reference["total_inter_chain_clashes"],
    )
    expected_aggregate = (
        0.2 * np.asarray(ranking.ptm_scores.complex_ptm)
        + 0.8 * np.asarray(ranking.ptm_scores.interface_ptm)
        - 100.0
        * np.asarray(ranking.clash_scores.has_inter_chain_clashes, dtype=np.float32)
    )
    np.testing.assert_allclose(
        np.asarray(ranking.aggregate_score), expected_aggregate, rtol=0, atol=0
    )

    confidence = confidence_logits_to_scores(
        array["pae_logits"],
        array["pde_logits"],
        array["plddt_logits"],
        token_mask=array["token_exists_mask"][0],
        atom_mask=array["atom_mask"][0],
        atom_token_index=array["atom_token_index"][0],
    )
    for actual, key in (
        (confidence.pae, "conf_pae"),
        (confidence.pde, "conf_pde"),
        (confidence.plddt, "conf_plddt"),
        (confidence.atom_plddt, "conf_atom_plddt"),
    ):
        np.testing.assert_allclose(
            np.asarray(actual), reference[key], rtol=2e-6, atol=2e-6
        )
