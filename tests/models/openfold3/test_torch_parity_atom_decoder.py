"""Torch-vs-JAX parity for AtomAttentionDecoder (AF3 Algorithm 6)."""

from __future__ import annotations

import math
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.openfold3.bridge.torch_mapping import map_atom_attention_decoder
from foldjax.models.openfold3.models.atom_features import atom_attention_decoder

pytestmark = pytest.mark.torch_parity

RTOL = 1e-4
ATOL = 1e-4

C_ATOM, C_ATOM_PAIR, C_TOKEN = 8, 6, 8
N_QUERY, N_KEY, HEADS = 4, 8, 2
COUNTS = [4, 3, 5]
N_ATOM, N_TOKEN = sum(COUNTS), len(COUNTS)


def _torch():
    import torch

    torch.manual_seed(0)
    return torch


def _decoder():
    from openfold3.core.model.layers.sequence_local_atom_attention import (
        AtomAttentionDecoder,
    )

    return AtomAttentionDecoder(
        c_token=C_TOKEN,
        c_atom=C_ATOM,
        c_atom_pair=C_ATOM_PAIR,
        c_hidden=4,
        no_heads=HEADS,
        no_blocks=2,
        n_transition=2,
        n_query=N_QUERY,
        n_key=N_KEY,
        use_ada_layer_norm=True,
    )


def _batch(torch) -> dict:
    return {
        "token_mask": torch.ones(1, N_TOKEN),
        "num_atoms_per_token": torch.tensor([COUNTS], dtype=torch.float32),
        "atom_mask": torch.ones(1, N_ATOM),
    }


def test_decoder_matches_torch(openfold3_source: Path, randomized) -> None:
    torch = _torch()
    module = randomized(_decoder())
    batch = _batch(torch)
    blocks = math.ceil(N_ATOM / N_QUERY)
    ai = torch.randn(1, N_TOKEN, C_TOKEN)
    ql = torch.randn(1, N_ATOM, C_ATOM)
    cl = torch.randn(1, N_ATOM, C_ATOM)
    plm = torch.randn(1, blocks, N_QUERY, N_KEY, C_ATOM_PAIR)

    with torch.no_grad():
        expected = module(batch=batch, ai=ai, ql=ql, cl=cl, plm=plm)

    actual = atom_attention_decoder(
        {key: jnp.asarray(v.numpy()) for key, v in batch.items()},
        jnp.asarray(ai.numpy()),
        jnp.asarray(ql.numpy()),
        jnp.asarray(cl.numpy()),
        jnp.asarray(plm.numpy()),
        map_atom_attention_decoder(dict(module.state_dict())),
        n_query=N_QUERY,
        n_key=N_KEY,
        no_heads=HEADS,
    )
    assert actual.shape == (1, N_ATOM, 3)
    np.testing.assert_allclose(
        np.asarray(actual, dtype=np.float64),
        expected.detach().numpy().astype(np.float64),
        rtol=RTOL,
        atol=ATOL,
        err_msg="AtomAttentionDecoder diverged from the OpenFold3 reference",
    )


def test_output_is_three_coordinate_components(openfold3_source: Path) -> None:
    _torch()
    weight = _decoder().state_dict()["linear_q_out.weight"]
    assert tuple(weight.shape) == (3, C_ATOM)


def test_decoder_layer_norm_is_scale_only(openfold3_source: Path) -> None:
    _torch()
    state = set(_decoder().state_dict())
    assert "layer_norm.weight" in state
    assert "layer_norm.bias" not in state


def test_mapper_reports_a_missing_projection(openfold3_source: Path) -> None:
    _torch()
    state = dict(_decoder().state_dict())
    del state["linear_q_out.weight"]
    with pytest.raises(KeyError, match="linear_q_out.weight"):
        map_atom_attention_decoder(state)
