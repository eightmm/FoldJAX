"""Torch-vs-JAX parity for pair-rep blocking and the noisy position embedder."""

from __future__ import annotations

import math
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.openfold3.bridge.torch_mapping import map_noisy_position_embedder
from foldjax.models.openfold3.models.atom_blocks import pair_rep_to_blocks
from foldjax.models.openfold3.models.atom_features import noisy_position_embedder

pytestmark = pytest.mark.torch_parity

RTOL = 1e-4
ATOL = 1e-4

C_S, C_Z, C_ATOM, C_ATOM_PAIR = 10, 6, 8, 4
N_TOKEN, N_QUERY, N_KEY = 4, 4, 8
COUNTS = [4, 2, 3, 3]
N_ATOM = sum(COUNTS)


def _torch():
    import torch

    torch.manual_seed(0)
    return torch


def _batch(torch, counts: list[int] | None = None) -> dict:
    counts = counts or COUNTS
    n_atom = sum(counts)
    # Upstream indexes the pair rep with this directly, so it must be an integer
    # tensor: convert_pair_rep_to_blocks casts q_indices with .long() but not the
    # gathered k_indices.
    atom_to_token = torch.cat(
        [
            torch.full((count,), index, dtype=torch.long)
            for index, count in enumerate(counts)
        ]
    )
    return {
        "token_mask": torch.ones(1, len(counts)),
        "num_atoms_per_token": torch.tensor([counts], dtype=torch.float32),
        "atom_to_token_index": atom_to_token.reshape(1, n_atom),
        "atom_mask": torch.ones(1, n_atom),
    }


def _as_jax(batch: dict) -> dict:
    return {key: jnp.asarray(value.numpy()) for key, value in batch.items()}


def _upstream_pair_blocks():
    from openfold3.core.utils.atom_attention_block_utils import (
        convert_pair_rep_to_blocks,
    )

    return convert_pair_rep_to_blocks


def _module():
    from openfold3.core.model.layers.sequence_local_atom_attention import (
        NoisyPositionEmbedder,
    )

    return NoisyPositionEmbedder(
        c_s=C_S, c_z=C_Z, c_atom=C_ATOM, c_atom_pair=C_ATOM_PAIR
    )


@pytest.mark.parametrize("counts", [COUNTS, [6, 1, 1, 4], [3, 3, 3, 3]])
def test_pair_rep_to_blocks_matches_torch(
    openfold3_source: Path, counts: list[int]
) -> None:
    torch = _torch()
    batch = _batch(torch, counts)
    zij = torch.randn(1, len(counts), len(counts), C_ATOM_PAIR)
    expected = _upstream_pair_blocks()(
        batch=batch, zij_trunk=zij, n_query=N_QUERY, n_key=N_KEY
    )
    actual = pair_rep_to_blocks(
        jnp.asarray(zij.numpy()),
        jnp.asarray(batch["atom_to_token_index"].numpy()),
        jnp.asarray(batch["atom_mask"].numpy()),
        n_query=N_QUERY,
        n_key=N_KEY,
    )
    assert actual.shape == tuple(expected.shape)
    np.testing.assert_allclose(
        np.asarray(actual, dtype=np.float64),
        expected.detach().numpy().astype(np.float64),
        rtol=RTOL,
        atol=ATOL,
    )


def test_noisy_position_embedder_matches_torch(
    openfold3_source: Path, randomized
) -> None:
    torch = _torch()
    module = randomized(_module())
    batch = _batch(torch)
    num_blocks = math.ceil(N_ATOM / N_QUERY)
    cl = torch.randn(1, N_ATOM, C_ATOM)
    plm = torch.randn(1, num_blocks, N_QUERY, N_KEY, C_ATOM_PAIR)
    si = torch.randn(1, N_TOKEN, C_S)
    zij = torch.randn(1, N_TOKEN, N_TOKEN, C_Z)
    rl = torch.randn(1, N_ATOM, 3)

    with torch.no_grad():
        expected = module(
            batch=batch,
            cl=cl,
            plm=plm,
            si_trunk=si,
            zij_trunk=zij,
            rl=rl,
            n_query=N_QUERY,
            n_key=N_KEY,
        )
    actual = noisy_position_embedder(
        _as_jax(batch),
        jnp.asarray(cl.numpy()),
        jnp.asarray(plm.numpy()),
        jnp.asarray(si.numpy()),
        jnp.asarray(zij.numpy()),
        jnp.asarray(rl.numpy()),
        map_noisy_position_embedder(dict(module.state_dict())),
        n_query=N_QUERY,
        n_key=N_KEY,
    )
    for got, want, name in zip(actual, expected, ("cl", "plm", "ql"), strict=True):
        assert got.shape == tuple(want.shape), name
        np.testing.assert_allclose(
            np.asarray(got, dtype=np.float64),
            want.detach().numpy().astype(np.float64),
            rtol=RTOL,
            atol=ATOL,
            err_msg=f"{name} diverged from the OpenFold3 reference",
        )


def test_ql_carries_the_coordinate_projection_but_cl_does_not(
    openfold3_source: Path, randomized
) -> None:
    """cl and ql differ by exactly linear_r(rl); mixing them up is easy."""
    torch = _torch()
    module = randomized(_module())
    batch = _batch(torch)
    num_blocks = math.ceil(N_ATOM / N_QUERY)
    args = (
        _as_jax(batch),
        jnp.asarray(torch.randn(1, N_ATOM, C_ATOM).numpy()),
        jnp.asarray(torch.randn(1, num_blocks, N_QUERY, N_KEY, C_ATOM_PAIR).numpy()),
        jnp.asarray(torch.randn(1, N_TOKEN, C_S).numpy()),
        jnp.asarray(torch.randn(1, N_TOKEN, N_TOKEN, C_Z).numpy()),
        jnp.asarray(torch.randn(1, N_ATOM, 3).numpy()),
    )
    params = map_noisy_position_embedder(dict(module.state_dict()))
    cl, _plm, ql = noisy_position_embedder(
        *args, params, n_query=N_QUERY, n_key=N_KEY
    )
    assert not np.allclose(np.asarray(cl), np.asarray(ql), rtol=1e-3)

    from foldjax.models.openfold3.models.primitives import linear

    expected_ql = cl + linear(args[5], params.linear_r)
    np.testing.assert_allclose(
        np.asarray(ql), np.asarray(expected_ql), rtol=1e-6, atol=1e-6
    )


def test_state_dict_layout(openfold3_source: Path) -> None:
    _torch()
    state = set(_module().state_dict())
    assert state == {
        "layer_norm_s.weight",
        "linear_s.weight",
        "layer_norm_z.weight",
        "linear_z.weight",
        "linear_r.weight",
    }
    # Both norms are scale-only, so neither has a bias entry.
    assert "layer_norm_s.bias" not in state
    assert "layer_norm_z.bias" not in state


def test_mapper_reports_a_missing_projection(openfold3_source: Path) -> None:
    _torch()
    state = dict(_module().state_dict())
    del state["linear_r.weight"]
    with pytest.raises(KeyError, match="linear_r.weight"):
        map_noisy_position_embedder(state)
