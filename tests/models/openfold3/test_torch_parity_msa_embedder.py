"""Torch-vs-JAX parity for MSA input embedding (AF3 Algorithm 8, lines 1-4)."""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.openfold3.bridge.torch_mapping import map_msa_embedder
from foldjax.models.openfold3.models.input_embedders import msa_embedder

pytestmark = pytest.mark.torch_parity

RTOL = 1e-4
ATOL = 1e-4

C_MSA, C_M, C_S_INPUT = 32, 8, 6
N_MSA, N_TOKEN = 4, 5


def _torch():
    import torch

    torch.manual_seed(0)
    return torch


def _module():
    from openfold3.core.model.feature_embedders.input_embedders import (
        MSAModuleEmbedder,
    )

    return MSAModuleEmbedder(
        c_m_feats=C_MSA + 2,
        c_m=C_M,
        c_s_input=C_S_INPUT,
        subsample_main_msa=False,
        subsample_all_msa=False,
        min_subsampled_all_msa=1,
        max_subsampled_all_msa=N_MSA,
    )


def _batch(torch) -> dict:
    return {
        "msa": torch.rand(1, N_MSA, N_TOKEN, C_MSA),
        "has_deletion": torch.randint(0, 2, (1, N_MSA, N_TOKEN)).float(),
        "deletion_value": torch.rand(1, N_MSA, N_TOKEN),
        "msa_mask": torch.ones(1, N_MSA, N_TOKEN),
        "num_paired_seqs": torch.tensor(2),
        "asym_id": torch.zeros(1, N_TOKEN, dtype=torch.long),
    }


def test_msa_embedder_matches_torch(openfold3_source: Path, randomized) -> None:
    torch = _torch()
    module = randomized(_module())
    batch = _batch(torch)
    s_input = torch.randn(1, N_TOKEN, C_S_INPUT)

    with torch.no_grad():
        expected_m, expected_mask = module(batch=batch, s_input=s_input)

    actual_m, actual_mask = msa_embedder(
        {k: jnp.asarray(v.numpy()) for k, v in batch.items()},
        jnp.asarray(s_input.numpy()),
        map_msa_embedder(dict(module.state_dict())),
    )
    assert actual_m.shape == tuple(expected_m.shape)
    np.testing.assert_allclose(
        np.asarray(actual_m, dtype=np.float64),
        expected_m.detach().numpy().astype(np.float64),
        rtol=RTOL,
        atol=ATOL,
        err_msg="MSA embedding diverged from the OpenFold3 reference",
    )
    np.testing.assert_allclose(
        np.asarray(actual_mask), expected_mask.detach().numpy()
    )


def test_single_representation_is_shared_across_msa_rows(
    openfold3_source: Path, randomized
) -> None:
    """s_input is broadcast over sequences; a per-row add would differ."""
    torch = _torch()
    module = randomized(_module())
    batch = _batch(torch)
    # Identical MSA rows must give identical embeddings.
    batch["msa"] = batch["msa"][:, :1].repeat(1, N_MSA, 1, 1)
    batch["has_deletion"] = torch.zeros(1, N_MSA, N_TOKEN)
    batch["deletion_value"] = torch.zeros(1, N_MSA, N_TOKEN)
    s_input = torch.randn(1, N_TOKEN, C_S_INPUT)

    m, _mask = msa_embedder(
        {k: jnp.asarray(v.numpy()) for k, v in batch.items()},
        jnp.asarray(s_input.numpy()),
        map_msa_embedder(dict(module.state_dict())),
    )
    rows = np.asarray(m)
    for index in range(1, N_MSA):
        np.testing.assert_allclose(rows[0, 0], rows[0, index], rtol=1e-6)


def test_deletion_features_are_appended_in_order(
    openfold3_source: Path, randomized
) -> None:
    """has_deletion then deletion_value; swapping them changes the result."""
    torch = _torch()
    module = randomized(_module())
    batch = _batch(torch)
    s_input = torch.randn(1, N_TOKEN, C_S_INPUT)
    params = map_msa_embedder(dict(module.state_dict()))
    jax_batch = {k: jnp.asarray(v.numpy()) for k, v in batch.items()}

    normal, _ = msa_embedder(jax_batch, jnp.asarray(s_input.numpy()), params)
    swapped_batch = dict(jax_batch)
    swapped_batch["has_deletion"], swapped_batch["deletion_value"] = (
        jax_batch["deletion_value"],
        jax_batch["has_deletion"],
    )
    swapped, _ = msa_embedder(swapped_batch, jnp.asarray(s_input.numpy()), params)
    assert not np.allclose(np.asarray(normal), np.asarray(swapped), rtol=1e-4)


def test_state_dict_layout(openfold3_source: Path) -> None:
    _torch()
    assert set(_module().state_dict()) == {
        "linear_m.weight",
        "linear_s_input.weight",
    }
