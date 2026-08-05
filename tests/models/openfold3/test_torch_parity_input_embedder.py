"""Torch-vs-JAX parity for InputEmbedderAllAtom (AF3 Algorithm 2)."""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.openfold3.bridge.torch_mapping import map_input_embedder
from foldjax.models.openfold3.models.input_embedders import input_embedder

pytestmark = pytest.mark.torch_parity

RTOL = 1e-4
ATOL = 1e-4

C_ATOM, C_ATOM_PAIR, C_TOKEN = 8, 6, 8
C_S, C_Z = 8, 4
C_ELEMENT, C_CHARS = 12, 5
C_RESTYPE, C_PROFILE = 32, 32
N_QUERY, N_KEY, HEADS = 4, 8, 2
MAX_IDX, MAX_CHAIN = 4, 2
COUNTS = [4, 3, 5]
N_ATOM, N_TOKEN = sum(COUNTS), len(COUNTS)
C_S_INPUT = C_TOKEN + C_RESTYPE + C_PROFILE + 1


def _torch():
    import torch

    torch.manual_seed(0)
    return torch


def _module():
    from ml_collections import ConfigDict
    from openfold3.core.model.feature_embedders.input_embedders import (
        InputEmbedderAllAtom,
    )

    # The encoder is configured through a dict, not flattened kwargs.
    return InputEmbedderAllAtom(
        c_s_input=C_S_INPUT,
        c_s=C_S,
        c_z=C_Z,
        max_relative_idx=MAX_IDX,
        max_relative_chain=MAX_CHAIN,
        atom_attn_enc={
            "c_atom_ref": ConfigDict(
                {"element": C_ELEMENT, "name_chars": 4 * C_CHARS}
            ),
            "c_atom": C_ATOM,
            "c_atom_pair": C_ATOM_PAIR,
            "c_token": C_TOKEN,
            "c_hidden": 4,
            "no_heads": HEADS,
            "no_blocks": 1,
            "n_transition": 2,
            "n_query": N_QUERY,
            "n_key": N_KEY,
            "use_ada_layer_norm": False,
        },
    )


def _batch(torch) -> dict:
    atom_to_token = torch.cat(
        [torch.full((c,), i, dtype=torch.long) for i, c in enumerate(COUNTS)]
    )
    return {
        "ref_pos": torch.randn(1, N_ATOM, 3),
        "ref_charge": torch.randn(1, N_ATOM),
        "ref_mask": torch.ones(1, N_ATOM),
        "ref_element": torch.rand(1, N_ATOM, C_ELEMENT),
        "ref_atom_name_chars": torch.rand(1, N_ATOM, 4, C_CHARS),
        "ref_space_uid": (torch.arange(N_ATOM).float() // 4).reshape(1, N_ATOM),
        "atom_mask": torch.ones(1, N_ATOM),
        "atom_to_token_index": atom_to_token.reshape(1, N_ATOM),
        "token_mask": torch.ones(1, N_TOKEN),
        "num_atoms_per_token": torch.tensor([COUNTS], dtype=torch.float32),
        "restype": torch.rand(1, N_TOKEN, C_RESTYPE),
        "profile": torch.rand(1, N_TOKEN, C_PROFILE),
        "deletion_mean": torch.rand(1, N_TOKEN),
        "token_bonds": torch.randint(0, 2, (1, N_TOKEN, N_TOKEN)).float(),
        "residue_index": torch.arange(N_TOKEN).reshape(1, N_TOKEN),
        "asym_id": torch.tensor([[0, 0, 1]]),
        "entity_id": torch.tensor([[0, 0, 0]]),
        "token_index": torch.arange(N_TOKEN).reshape(1, N_TOKEN),
        "sym_id": torch.tensor([[0, 0, 1]]),
    }


def _kwargs() -> dict:
    return {
        "n_query": N_QUERY,
        "n_key": N_KEY,
        "atom_heads": HEADS,
        "n_token": N_TOKEN,
        "max_relative_idx": MAX_IDX,
        "max_relative_chain": MAX_CHAIN,
    }


def test_input_embedder_matches_torch(openfold3_source: Path, randomized) -> None:
    torch = _torch()
    module = randomized(_module())
    batch = _batch(torch)
    with torch.no_grad():
        expected = module(batch=batch)

    actual = input_embedder(
        {k: jnp.asarray(v.numpy()) for k, v in batch.items()},
        map_input_embedder(dict(module.state_dict())),
        **_kwargs(),
    )
    names = ("s_input", "s", "z")
    for got, want, name in zip(actual, expected, names, strict=True):
        assert got.shape == tuple(want.shape), name
        np.testing.assert_allclose(
            np.asarray(got, dtype=np.float64),
            want.detach().numpy().astype(np.float64),
            rtol=RTOL,
            atol=ATOL,
            err_msg=f"InputEmbedder.{name} diverged from the reference",
        )


def test_pair_representation_is_asymmetric(
    openfold3_source: Path, randomized
) -> None:
    """Row and column use different projections, so z must not be symmetric."""
    torch = _torch()
    module = randomized(_module())
    batch = _batch(torch)
    # Symmetric bonds and a single chain remove the other asymmetry sources.
    batch["token_bonds"] = torch.zeros(1, N_TOKEN, N_TOKEN)
    batch["asym_id"] = torch.zeros(1, N_TOKEN, dtype=torch.long)
    batch["sym_id"] = torch.zeros(1, N_TOKEN, dtype=torch.long)

    _s_input, _s, z = input_embedder(
        {k: jnp.asarray(v.numpy()) for k, v in batch.items()},
        map_input_embedder(dict(module.state_dict())),
        **_kwargs(),
    )
    array = np.asarray(z)
    assert not np.allclose(array, np.swapaxes(array, -2, -3), atol=1e-4)


def test_state_dict_prefixes(openfold3_source: Path) -> None:
    _torch()
    prefixes = {key.split(".")[0] for key in _module().state_dict()}
    assert prefixes == {
        "atom_attn_enc",
        "linear_s",
        "linear_z_i",
        "linear_z_j",
        "linear_relpos",
        "linear_token_bonds",
    }


def test_mapper_reports_a_missing_projection(openfold3_source: Path) -> None:
    _torch()
    state = dict(_module().state_dict())
    del state["linear_z_j.weight"]
    with pytest.raises(KeyError, match="linear_z_j.weight"):
        map_input_embedder(state)
