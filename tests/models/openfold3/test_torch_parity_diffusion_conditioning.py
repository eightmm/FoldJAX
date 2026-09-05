"""Torch-vs-JAX parity for noise/single conditioning (AF3 Algorithm 21)."""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.openfold3.bridge.torch_mapping import map_single_conditioning
from foldjax.models.openfold3.models.diffusion_conditioning import (
    fourier_embedding,
    single_conditioning,
)

pytestmark = pytest.mark.torch_parity

RTOL = 1e-4
ATOL = 1e-4

C_S, C_S_INPUT, C_FOURIER, N_TOKEN = 8, 6, 16, 5
SIGMA_DATA = 16.0


def _torch():
    import torch

    torch.manual_seed(0)
    return torch


def _fourier():
    from openfold3.core.model.feature_embedders.input_embedders import (
        FourierEmbedding,
    )

    return FourierEmbedding(c=C_FOURIER, seed=42)


def test_fourier_embedding_matches_torch(openfold3_source: Path) -> None:
    torch = _torch()
    module = _fourier()
    x = torch.randn(3, 1)
    with torch.no_grad():
        expected = module(x)
    from foldjax.models.openfold3.models.diffusion_conditioning import (
        FourierEmbeddingParams,
    )

    state = dict(module.state_dict())
    params = FourierEmbeddingParams(
        w=jnp.asarray(state["w"].numpy()), b=jnp.asarray(state["b"].numpy())
    )
    actual = fourier_embedding(jnp.asarray(x.numpy()), params)
    assert actual.shape == tuple(expected.shape)
    np.testing.assert_allclose(
        np.asarray(actual, dtype=np.float64),
        expected.detach().numpy().astype(np.float64),
        rtol=RTOL,
        atol=ATOL,
    )


def test_fourier_buffers_are_not_learned_parameters(openfold3_source: Path) -> None:
    """w/b are buffers from a seeded generator, so they live in the state_dict."""
    _torch()
    module = _fourier()
    assert set(module.state_dict()) == {"w", "b"}
    assert list(module.parameters()) == []


def test_single_conditioning_matches_transcribed_reference(
    openfold3_source: Path, randomized
) -> None:
    """The concat order and the noise-embedding addition are what matter here."""
    torch = _torch()
    from openfold3.core.model.layers.transition import SwiGLUTransition
    from openfold3.core.model.primitives import LayerNorm, Linear

    ln_s = randomized(LayerNorm(C_S + C_S_INPUT, create_offset=False))
    lin_s = randomized(Linear(C_S + C_S_INPUT, C_S, bias=False))
    fourier = _fourier()
    ln_n = randomized(LayerNorm(C_FOURIER, create_offset=False))
    lin_n = randomized(Linear(C_FOURIER, C_S, bias=False))

    si_input = torch.randn(1, N_TOKEN, C_S_INPUT)
    si_trunk = torch.randn(1, N_TOKEN, C_S)
    t = torch.tensor([25.0])

    with torch.no_grad():
        si = torch.cat([si_trunk, si_input], dim=-1)
        si = lin_s(ln_s(si))
        n = 0.25 * torch.log(t / SIGMA_DATA)
        n = fourier(n.unsqueeze(-1))
        expected = si + lin_n(ln_n(n)).unsqueeze(-2)

    state = {}
    for name, module in (
        ("layer_norm_s", ln_s),
        ("linear_s", lin_s),
        ("fourier_emb", fourier),
        ("layer_norm_n", ln_n),
        ("linear_n", lin_n),
        ("layer_norm_z", LayerNorm(C_S, create_offset=False)),
        ("linear_z", Linear(C_S, C_S, bias=False)),
        ("transition_z.0", SwiGLUTransition(c_in=C_S, n=2)),
        ("transition_z.1", SwiGLUTransition(c_in=C_S, n=2)),
    ):
        for key, value in module.state_dict().items():
            state[f"{name}.{key}"] = value

    params = map_single_conditioning(state)
    assert params.transition_s == ()
    actual = single_conditioning(
        jnp.asarray(si_input.numpy()),
        jnp.asarray(si_trunk.numpy()),
        jnp.asarray(t.numpy()),
        params,
        sigma_data=SIGMA_DATA,
    )
    np.testing.assert_allclose(
        np.asarray(actual, dtype=np.float64),
        expected.detach().numpy().astype(np.float64),
        rtol=RTOL,
        atol=ATOL,
    )


def test_noise_level_changes_the_conditioning(openfold3_source: Path) -> None:
    """0.25*log(t/sigma_data) must actually reach the output."""
    _torch()
    from openfold3.core.model.layers.transition import SwiGLUTransition
    from openfold3.core.model.primitives import LayerNorm, Linear

    fourier = _fourier()
    state = {}
    for name, module in (
        ("layer_norm_s", LayerNorm(C_S + C_S_INPUT, create_offset=False)),
        ("linear_s", Linear(C_S + C_S_INPUT, C_S, bias=False)),
        ("fourier_emb", fourier),
        ("layer_norm_n", LayerNorm(C_FOURIER, create_offset=False)),
        ("linear_n", Linear(C_FOURIER, C_S, bias=False)),
        ("layer_norm_z", LayerNorm(C_S, create_offset=False)),
        ("linear_z", Linear(C_S, C_S, bias=False)),
        ("transition_z.0", SwiGLUTransition(c_in=C_S, n=2)),
        ("transition_z.1", SwiGLUTransition(c_in=C_S, n=2)),
    ):
        for key, value in module.state_dict().items():
            state[f"{name}.{key}"] = value
    params = map_single_conditioning(state)

    args = (
        jnp.zeros((1, N_TOKEN, C_S_INPUT)),
        jnp.zeros((1, N_TOKEN, C_S)),
    )
    low = single_conditioning(
        *args, jnp.asarray([1.0]), params, sigma_data=SIGMA_DATA
    )
    high = single_conditioning(
        *args, jnp.asarray([100.0]), params, sigma_data=SIGMA_DATA
    )
    assert not np.allclose(np.asarray(low), np.asarray(high), rtol=1e-4)


def test_mapper_requires_the_fourier_buffers(openfold3_source: Path) -> None:
    """The buffers cannot be re-sampled in JAX, so their absence must fail."""
    _torch()
    from openfold3.core.model.layers.transition import SwiGLUTransition
    from openfold3.core.model.primitives import LayerNorm, Linear

    state = {}
    for name, module in (
        ("layer_norm_s", LayerNorm(C_S + C_S_INPUT, create_offset=False)),
        ("linear_s", Linear(C_S + C_S_INPUT, C_S, bias=False)),
        ("layer_norm_n", LayerNorm(C_FOURIER, create_offset=False)),
        ("linear_n", Linear(C_FOURIER, C_S, bias=False)),
        ("layer_norm_z", LayerNorm(C_S, create_offset=False)),
        ("linear_z", Linear(C_S, C_S, bias=False)),
        ("transition_z.0", SwiGLUTransition(c_in=C_S, n=2)),
        ("transition_z.1", SwiGLUTransition(c_in=C_S, n=2)),
    ):
        for key, value in module.state_dict().items():
            state[f"{name}.{key}"] = value
    # Everything present except the Fourier buffers.
    with pytest.raises(KeyError, match="fourier_emb.w"):
        map_single_conditioning(state)
