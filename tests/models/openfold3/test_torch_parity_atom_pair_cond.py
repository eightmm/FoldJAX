"""Torch-vs-JAX parity for the atom pair conditioning stage.

This is the part of ``AtomAttentionEncoder.get_atom_reps`` that folds the atom
single conditioning into the blocked pair conditioning. It is gated against the
real encoder rather than a hand-built stand-in.
"""

from __future__ import annotations

import math
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.openfold3.bridge.torch_mapping import map_atom_pair_conditioning
from foldjax.models.openfold3.models.atom_features import atom_pair_conditioning

pytestmark = pytest.mark.torch_parity

RTOL = 1e-4
ATOL = 1e-4

C_ATOM, C_ATOM_PAIR = 8, 6
N_ATOM, N_QUERY, N_KEY = 12, 4, 8


def _torch():
    import torch

    torch.manual_seed(0)
    return torch


def _encoder():
    from ml_collections import ConfigDict
    from openfold3.core.model.layers.sequence_local_atom_attention import (
        AtomAttentionEncoder,
    )

    return AtomAttentionEncoder(
        c_atom_ref=ConfigDict({"element": 12, "name_chars": 20}),
        c_atom=C_ATOM,
        c_atom_pair=C_ATOM_PAIR,
        c_token=8,
        add_noisy_pos=False,
        c_hidden=4,
        no_heads=2,
        no_blocks=1,
        n_transition=2,
        n_query=N_QUERY,
        n_key=N_KEY,
    )


def _reference(module, torch, cl, plm, atom_mask):
    """The pair-conditioning slice of get_atom_reps, transcribed."""
    from openfold3.core.utils.atom_attention_block_utils import (
        convert_single_rep_to_blocks,
    )

    cl_l, cl_m, block_mask = convert_single_rep_to_blocks(
        ql=cl, n_query=N_QUERY, n_key=N_KEY, atom_mask=atom_mask
    )
    cl_lm = (
        module.linear_l(module.relu(cl_l.unsqueeze(-2)))
        + module.linear_m(module.relu(cl_m.unsqueeze(-3)))
    ) * block_mask.unsqueeze(-1)
    out = plm + cl_lm
    out = out + module.pair_mlp(out)
    return out * block_mask.unsqueeze(-1)


@pytest.mark.parametrize("n_valid", [N_ATOM, 7])
def test_atom_pair_conditioning_matches_torch(
    openfold3_source: Path, randomized, n_valid: int
) -> None:
    torch = _torch()
    module = randomized(_encoder())
    blocks = math.ceil(N_ATOM / N_QUERY)
    cl = torch.randn(1, N_ATOM, C_ATOM)
    plm = torch.randn(1, blocks, N_QUERY, N_KEY, C_ATOM_PAIR)
    atom_mask = torch.zeros(1, N_ATOM)
    atom_mask[:, :n_valid] = 1.0

    with torch.no_grad():
        expected = _reference(module, torch, cl, plm, atom_mask)

    actual = atom_pair_conditioning(
        jnp.asarray(cl.numpy()),
        jnp.asarray(plm.numpy()),
        jnp.asarray(atom_mask.numpy()),
        map_atom_pair_conditioning(dict(module.state_dict())),
        n_query=N_QUERY,
        n_key=N_KEY,
    )
    assert actual.shape == tuple(expected.shape)
    np.testing.assert_allclose(
        np.asarray(actual, dtype=np.float64),
        expected.detach().numpy().astype(np.float64),
        rtol=RTOL,
        atol=ATOL,
        err_msg=f"atom pair conditioning diverged (n_valid={n_valid})",
    )


def test_pair_mlp_linears_live_at_odd_sequential_indices(
    openfold3_source: Path,
) -> None:
    """ReLUs occupy 0/2/4, so the Linears are at pair_mlp.1/.3/.5."""
    _torch()
    keys = {
        key for key in _encoder().state_dict() if key.startswith("pair_mlp.")
    }
    assert keys == {"pair_mlp.1.weight", "pair_mlp.3.weight", "pair_mlp.5.weight"}


def test_pair_mlp_applies_relu_before_the_first_projection(
    openfold3_source: Path, randomized
) -> None:
    """The Sequential starts with ReLU; skipping it changes the result."""
    torch = _torch()
    module = randomized(_encoder())
    params = map_atom_pair_conditioning(dict(module.state_dict()))
    blocks = math.ceil(N_ATOM / N_QUERY)
    # Strongly negative input: a leading ReLU zeroes it, no ReLU does not.
    plm = jnp.full((1, blocks, N_QUERY, N_KEY, C_ATOM_PAIR), -5.0)
    cl = jnp.zeros((1, N_ATOM, C_ATOM))
    mask = jnp.ones((1, N_ATOM))

    out = atom_pair_conditioning(
        cl, plm, mask, params, n_query=N_QUERY, n_key=N_KEY
    )
    with torch.no_grad():
        expected = _reference(
            module,
            torch,
            torch.zeros(1, N_ATOM, C_ATOM),
            torch.full((1, blocks, N_QUERY, N_KEY, C_ATOM_PAIR), -5.0),
            torch.ones(1, N_ATOM),
        )
    np.testing.assert_allclose(
        np.asarray(out, dtype=np.float64),
        expected.detach().numpy().astype(np.float64),
        rtol=RTOL,
        atol=ATOL,
    )


def test_mapper_reports_a_missing_pair_mlp_layer(openfold3_source: Path) -> None:
    _torch()
    state = dict(_encoder().state_dict())
    del state["pair_mlp.3.weight"]
    with pytest.raises(KeyError, match=r"pair_mlp\.3\.weight"):
        map_atom_pair_conditioning(state)
