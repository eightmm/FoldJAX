"""MSA subsampling against upstream's ``_subsample_all_msa``.

The released config subsamples the MSA to 1024 rows at inference, with no training
guard, choosing rows with ``torch.randperm``. This port performs the selection
outside the embedder so a caller can match upstream when it knows the permutation,
and agrees exactly when there are fewer valid rows than the target.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.openfold3.models.input_embedders import subsample_msa

pytestmark = pytest.mark.torch_parity

N_MSA, N_TOKEN = 10, 5


def _features(torch, n_valid: int) -> dict:
    generator = torch.Generator().manual_seed(3)
    mask = torch.zeros(N_MSA, N_TOKEN)
    mask[:n_valid] = 1.0
    return {
        "msa": torch.randn((N_MSA, N_TOKEN, 4), generator=generator),
        "has_deletion": torch.randn((N_MSA, N_TOKEN), generator=generator),
        "deletion_value": torch.randn((N_MSA, N_TOKEN), generator=generator),
        "msa_mask": mask,
    }


def _as_jax(features: dict) -> dict:
    return {key: jnp.asarray(value.numpy()) for key, value in features.items()}


def _upstream(torch, features: dict, target: int, monkeypatch, permutation=None):
    from openfold3.core.model.feature_embedders.input_embedders import (
        MSAModuleEmbedder,
    )

    if permutation is not None:
        monkeypatch.setattr(
            torch, "randperm", lambda n, device=None: torch.as_tensor(permutation)
        )
    # msa_feat is the concatenation the embedder builds before subsampling; the
    # function only reorders rows, so any per-row payload exercises it.
    feat = torch.cat(
        [
            features["msa"],
            features["has_deletion"][..., None],
            features["deletion_value"][..., None],
        ],
        dim=-1,
    )
    return MSAModuleEmbedder._subsample_all_msa(
        msa_feat=feat, msa_mask=features["msa_mask"], no_subsampled_all_msa=target
    )


def test_fewer_valid_rows_than_target_matches_upstream(
    openfold3_source: Path, monkeypatch
) -> None:
    """With filler needed, the valid rows must keep their order and their values."""
    import torch

    target = 8
    features = _features(torch, n_valid=6)
    feat_ref, mask_ref = _upstream(torch, features, target, monkeypatch)

    out = subsample_msa(_as_jax(features), target)
    # The valid prefix is what carries signal; the filler rows are all-masked.
    valid = int(mask_ref.sum(dim=-1).gt(0).sum())
    assert valid == 6
    np.testing.assert_allclose(
        np.asarray(out["msa"])[:valid],
        feat_ref[:valid, :, :4].numpy(),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(out["msa_mask"])[:valid], mask_ref[:valid].numpy()
    )
    assert out["msa_mask"].shape == (target, N_TOKEN)
    assert float(jnp.sum(out["msa_mask"][valid:])) == 0.0


def test_more_valid_rows_than_target_matches_a_known_permutation(
    openfold3_source: Path, monkeypatch
) -> None:
    """Above the threshold the selection is random upstream; given the same
    permutation the two must agree exactly."""
    import torch

    target = 4
    features = _features(torch, n_valid=N_MSA)
    permutation = [7, 2, 9, 0, 5, 1, 8, 3, 6, 4]
    feat_ref, mask_ref = _upstream(
        torch, features, target, monkeypatch, permutation=permutation
    )

    out = subsample_msa(
        _as_jax(features),
        target,
        valid_order=jnp.asarray(permutation),
    )
    np.testing.assert_allclose(
        np.asarray(out["msa"]), feat_ref[..., :4].numpy(), rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(np.asarray(out["msa_mask"]), mask_ref.numpy())


def test_a_small_msa_is_returned_unchanged(openfold3_source: Path) -> None:
    import torch

    features = _as_jax(_features(torch, n_valid=N_MSA))
    out = subsample_msa(features, N_MSA + 5)
    for key in ("msa", "msa_mask"):
        np.testing.assert_array_equal(np.asarray(out[key]), np.asarray(features[key]))


def test_valid_rows_are_ranked_before_invalid_ones(openfold3_source: Path) -> None:
    """Interleaved masks: the kept rows must be exactly the valid ones."""
    import torch

    features = _features(torch, n_valid=0)
    mask = np.zeros((N_MSA, N_TOKEN), dtype=np.float32)
    mask[[1, 4, 7]] = 1.0
    features["msa_mask"] = torch.as_tensor(mask)
    out = subsample_msa(_as_jax(features), 3)
    np.testing.assert_allclose(
        np.asarray(out["msa"]),
        features["msa"].numpy()[[1, 4, 7]],
        rtol=1e-6,
        atol=1e-6,
    )
