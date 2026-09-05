"""Torch-vs-JAX parity for sequence-local (blocked) attention with pair bias."""

from __future__ import annotations

import math
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.openfold3.bridge.torch_mapping import map_cross_attention_pair_bias
from foldjax.models.openfold3.models.attention_pair_bias import (
    cross_attention_pair_bias,
)

pytestmark = pytest.mark.torch_parity

RTOL = 1e-4
ATOL = 1e-4

C_Q, C_S, C_Z, C_HIDDEN, HEADS = 8, 6, 4, 4, 2
N_ATOM, N_QUERY, N_KEY = 12, 4, 8


def _torch():
    import torch

    torch.manual_seed(0)
    return torch


def _module():
    from openfold3.core.model.layers.attention_pair_bias import (
        CrossAttentionPairBias,
    )

    return CrossAttentionPairBias(
        c_q=C_Q,
        c_k=C_Q,
        c_v=C_Q,
        c_s=C_S,
        c_z=C_Z,
        c_hidden=C_HIDDEN,
        no_heads=HEADS,
        n_query=N_QUERY,
        n_key=N_KEY,
        inf=1e9,
    )


def _inputs(torch, n_valid: int = N_ATOM):
    blocks = math.ceil(N_ATOM / N_QUERY)
    a = torch.randn(1, N_ATOM, C_Q)
    z = torch.randn(1, blocks, N_QUERY, N_KEY, C_Z)
    s = torch.randn(1, N_ATOM, C_S)
    mask = torch.zeros(1, N_ATOM)
    mask[:, :n_valid] = 1.0
    return a, z, s, mask


def _close(actual: jnp.ndarray, expected, name: str) -> None:
    np.testing.assert_allclose(
        np.asarray(actual, dtype=np.float64),
        expected.detach().numpy().astype(np.float64),
        rtol=RTOL,
        atol=ATOL,
        err_msg=f"{name} diverged from the OpenFold3 reference",
    )


@pytest.mark.parametrize("n_valid", [N_ATOM, 7])
def test_matches_torch(
    openfold3_source: Path, randomized, n_valid: int
) -> None:
    torch = _torch()
    module = randomized(_module())
    a, z, s, mask = _inputs(torch, n_valid)
    with torch.no_grad():
        expected = module(a=a, z=z, s=s, mask=mask)
    params = map_cross_attention_pair_bias(dict(module.state_dict()))
    actual = cross_attention_pair_bias(
        jnp.asarray(a.numpy()),
        jnp.asarray(z.numpy()),
        params,
        no_heads=HEADS,
        n_query=N_QUERY,
        n_key=N_KEY,
        mask=jnp.asarray(mask.numpy()),
        s=jnp.asarray(s.numpy()),
    )
    assert actual.shape == (1, N_ATOM, C_Q)
    _close(actual, expected, f"CrossAttentionPairBias(n_valid={n_valid})")


def test_conditioned_matches_torch(openfold3_source: Path, randomized) -> None:
    torch = _torch()
    module = randomized(_module())
    a, z, s, mask = _inputs(torch)
    with torch.no_grad():
        expected = module(a=a, z=z, s=s, mask=mask)
    params = map_cross_attention_pair_bias(dict(module.state_dict()))
    actual = cross_attention_pair_bias(
        jnp.asarray(a.numpy()),
        jnp.asarray(z.numpy()),
        params,
        no_heads=HEADS,
        n_query=N_QUERY,
        n_key=N_KEY,
        mask=jnp.asarray(mask.numpy()),
        s=jnp.asarray(s.numpy()),
    )
    _close(actual, expected, "CrossAttentionPairBias(AdaLN)")


def test_conditioned_requires_the_single_representation(
    openfold3_source: Path, randomized
) -> None:
    torch = _torch()
    module = randomized(_module())
    a, z, _s, mask = _inputs(torch)
    params = map_cross_attention_pair_bias(dict(module.state_dict()))
    with pytest.raises(ValueError, match="s is required"):
        cross_attention_pair_bias(
            jnp.asarray(a.numpy()),
            jnp.asarray(z.numpy()),
            params,
            no_heads=HEADS,
            n_query=N_QUERY,
            n_key=N_KEY,
            mask=jnp.asarray(mask.numpy()),
        )


def test_query_and_key_norms_are_separate(openfold3_source: Path) -> None:
    """Two blockings of one tensor get two norms; sharing one would be wrong."""
    _torch()
    state = set(_module().state_dict())
    assert "layer_norm_a_q.layer_norm_s.weight" in state
    assert "layer_norm_a_k.layer_norm_s.weight" in state
    # No pair norm here, unlike the trunk and diffusion variants.
    assert not any(key.startswith("layer_norm_z") for key in state)


def test_output_is_unblocked_back_to_atoms(openfold3_source: Path, randomized) -> None:
    """The block axis is flattened and the block padding trimmed off."""
    torch = _torch()
    module = randomized(_module())
    a, z, s, mask = _inputs(torch)
    actual = cross_attention_pair_bias(
        jnp.asarray(a.numpy()),
        jnp.asarray(z.numpy()),
        map_cross_attention_pair_bias(dict(module.state_dict())),
        no_heads=HEADS,
        n_query=N_QUERY,
        n_key=N_KEY,
        mask=jnp.asarray(mask.numpy()),
        s=jnp.asarray(s.numpy()),
    )
    # N_ATOM=12 with n_query=4 pads to exactly 12, but the trim must still apply.
    assert actual.shape[-2] == N_ATOM


def test_mapper_reports_a_missing_projection(openfold3_source: Path) -> None:
    _torch()
    state = dict(_module().state_dict())
    del state["linear_z.weight"]
    with pytest.raises(KeyError, match="linear_z.weight"):
        map_cross_attention_pair_bias(state)
