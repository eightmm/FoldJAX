"""Torch-vs-JAX parity for the full AtomAttentionEncoder (AF3 Algorithm 5)."""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.openfold3.bridge.torch_mapping import map_atom_attention_encoder
from foldjax.models.openfold3.models.atom_features import atom_attention_encoder

pytestmark = pytest.mark.torch_parity

RTOL = 1e-4
ATOL = 1e-4

C_ATOM, C_ATOM_PAIR, C_TOKEN = 8, 6, 8
C_S, C_Z = 6, 4
C_ELEMENT, C_CHARS = 12, 5
N_QUERY, N_KEY, HEADS = 4, 8, 2
COUNTS = [4, 3, 5]
N_ATOM, N_TOKEN = sum(COUNTS), len(COUNTS)


def _torch():
    import torch

    torch.manual_seed(0)
    return torch


def _encoder(noisy: bool):
    from ml_collections import ConfigDict
    from openfold3.core.model.layers.sequence_local_atom_attention import (
        AtomAttentionEncoder,
    )

    return AtomAttentionEncoder(
        c_atom_ref=ConfigDict({"element": C_ELEMENT, "name_chars": 4 * C_CHARS}),
        c_atom=C_ATOM,
        c_atom_pair=C_ATOM_PAIR,
        c_token=C_TOKEN,
        add_noisy_pos=noisy,
        c_hidden=4,
        no_heads=HEADS,
        no_blocks=2,
        n_transition=2,
        n_query=N_QUERY,
        n_key=N_KEY,
        use_ada_layer_norm=noisy,
        c_s=C_S if noisy else None,
        c_z=C_Z if noisy else None,
    )


def _batch(torch) -> dict:
    atom_to_token = torch.cat(
        [torch.full((count,), i, dtype=torch.long) for i, count in enumerate(COUNTS)]
    )
    uid = torch.arange(N_ATOM).float() // 4
    return {
        "ref_pos": torch.randn(1, N_ATOM, 3),
        "ref_charge": torch.randn(1, N_ATOM),
        "ref_mask": torch.ones(1, N_ATOM),
        "ref_element": torch.rand(1, N_ATOM, C_ELEMENT),
        "ref_atom_name_chars": torch.rand(1, N_ATOM, 4, C_CHARS),
        "ref_space_uid": uid.reshape(1, N_ATOM),
        "atom_mask": torch.ones(1, N_ATOM),
        "atom_to_token_index": atom_to_token.reshape(1, N_ATOM),
        "token_mask": torch.ones(1, N_TOKEN),
        "num_atoms_per_token": torch.tensor([COUNTS], dtype=torch.float32),
    }


def _as_jax(batch: dict) -> dict:
    return {key: jnp.asarray(value.numpy()) for key, value in batch.items()}


def _close(actual, expected, name: str) -> None:
    np.testing.assert_allclose(
        np.asarray(actual, dtype=np.float64),
        expected.detach().numpy().astype(np.float64),
        rtol=RTOL,
        atol=ATOL,
        err_msg=f"{name} diverged from the OpenFold3 reference",
    )


def test_input_path_matches_torch(openfold3_source: Path, randomized) -> None:
    """The no-noise path: ql starts as the atom conditioning."""
    torch = _torch()
    module = randomized(_encoder(noisy=False))
    batch = _batch(torch)
    with torch.no_grad():
        expected = module(batch=batch)
    params = map_atom_attention_encoder(dict(module.state_dict()))
    assert params.noisy_position_embedder is None

    actual = atom_attention_encoder(
        _as_jax(batch),
        params,
        n_query=N_QUERY,
        n_key=N_KEY,
        no_heads=HEADS,
        n_token=N_TOKEN,
    )
    names = ("ai", "ql", "cl", "plm")
    for got, want, name in zip(actual, expected, names, strict=True):
        assert got.shape == tuple(want.shape), name
        _close(got, want, f"AtomAttentionEncoder.{name}")


def test_diffusion_path_matches_torch(openfold3_source: Path, randomized) -> None:
    """The noisy path adds trunk conditioning and the coordinate projection."""
    torch = _torch()
    module = randomized(_encoder(noisy=True))
    batch = _batch(torch)
    rl = torch.randn(1, N_ATOM, 3)
    si = torch.randn(1, N_TOKEN, C_S)
    zij = torch.randn(1, N_TOKEN, N_TOKEN, C_Z)
    with torch.no_grad():
        expected = module(batch=batch, rl=rl, si_trunk=si, zij_trunk=zij)
    params = map_atom_attention_encoder(dict(module.state_dict()))
    assert params.noisy_position_embedder is not None

    actual = atom_attention_encoder(
        _as_jax(batch),
        params,
        n_query=N_QUERY,
        n_key=N_KEY,
        no_heads=HEADS,
        n_token=N_TOKEN,
        rl=jnp.asarray(rl.numpy()),
        si_trunk=jnp.asarray(si.numpy()),
        zij_trunk=jnp.asarray(zij.numpy()),
    )
    names = ("ai", "ql", "cl", "plm")
    for got, want, name in zip(actual, expected, names, strict=True):
        assert got.shape == tuple(want.shape), name
        _close(got, want, f"AtomAttentionEncoder(noisy).{name}")


def test_noisy_inputs_require_a_noisy_embedder(
    openfold3_source: Path, randomized
) -> None:
    torch = _torch()
    module = randomized(_encoder(noisy=False))
    batch = _batch(torch)
    params = map_atom_attention_encoder(dict(module.state_dict()))
    with pytest.raises(ValueError, match="no noisy_position_embedder"):
        atom_attention_encoder(
            _as_jax(batch),
            params,
            n_query=N_QUERY,
            n_key=N_KEY,
            no_heads=HEADS,
            n_token=N_TOKEN,
            rl=jnp.zeros((1, N_ATOM, 3)),
        )


def test_noisy_path_requires_trunk_representations(
    openfold3_source: Path, randomized
) -> None:
    torch = _torch()
    module = randomized(_encoder(noisy=True))
    batch = _batch(torch)
    params = map_atom_attention_encoder(dict(module.state_dict()))
    with pytest.raises(ValueError, match="si_trunk and zij_trunk are required"):
        atom_attention_encoder(
            _as_jax(batch),
            params,
            n_query=N_QUERY,
            n_key=N_KEY,
            no_heads=HEADS,
            n_token=N_TOKEN,
            rl=jnp.zeros((1, N_ATOM, 3)),
        )


def test_linear_q_is_a_sequential_with_a_trailing_relu(
    openfold3_source: Path,
) -> None:
    """linear_q is Sequential(Linear, ReLU), so the weight is at linear_q.0."""
    _torch()
    keys = {
        key for key in _encoder(noisy=False).state_dict() if key.startswith("linear_q")
    }
    assert keys == {"linear_q.0.weight"}
