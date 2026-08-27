"""Whole-diffusion-module parity against upstream's ``DiffusionModule.forward``.

Covers the composition the per-layer tests cannot: that conditioning runs both
paths and that the *conditioned* pair representation reaches both the atom
attention encoder and the diffusion transformer. Passing the raw trunk pair
representation to either has the right shape and the wrong value, so only a
composite gate catches it.

The sampler is excluded on purpose -- its PRNG stream cannot be matched across
frameworks -- so the noisy coordinates and the noise level are fixed inputs here.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.openfold3.bridge.torch_mapping import (
    map_denoiser,
    map_diffusion_conditioning,
)
from foldjax.models.openfold3.models.denoiser import denoise
from foldjax.models.openfold3.models.diffusion_conditioning import (
    diffusion_conditioning,
)

from .of3_features import make_batch
from .test_composite_parity_trunk import _reduced_config

pytestmark = pytest.mark.torch_parity

# A whole-model forward pass amplifies weight magnitude; see the randomized
# fixture for why the scale is lowered instead of the tolerance.
COMPOSITE_SCALE = 0.05

RTOL = 1e-4
ATOL = 1e-4

N_TOKEN, N_MSA, N_TEMPL = 12, 6, 2


@pytest.fixture(scope="module")
def reference(openfold3_source: Path, randomized):
    """Upstream's diffusion module, its output, and every input it was given.

    Randomized for the reason given in the trunk composite test: default init
    leaves output projections at zero.
    """
    import torch
    from openfold3.projects.of3_all_atom.model import OpenFold3

    torch.manual_seed(0)
    config = _reduced_config()
    model = randomized(OpenFold3(config), scale=COMPOSITE_SCALE)
    batch = make_batch(n_token=N_TOKEN, n_msa=N_MSA, n_templ=N_TEMPL)

    with torch.no_grad():
        s_input, s_trunk, z = model.run_trunk(batch=batch, num_recycles=1)

        n_atom = batch["atom_mask"].shape[-1]
        generator = torch.Generator().manual_seed(1)
        xl_noisy = torch.randn((1, n_atom, 3), generator=generator)
        t = torch.full((1,), 12.0)

        out = model.diffusion_module(
            batch=batch,
            xl_noisy=xl_noisy,
            token_mask=batch["token_mask"],
            atom_mask=batch["atom_mask"],
            t=t,
            si_input=s_input,
            si_trunk=s_trunk,
            zij_trunk=z,
            use_conditioning=True,
        )
    return model, batch, (s_input, s_trunk, z), (xl_noisy, t), out


def _as_jax(batch: dict) -> dict:
    return {
        key: jnp.asarray(value.numpy())
        for key, value in batch.items()
        if hasattr(value, "numpy")
    }


def test_conditioned_pair_representation_is_not_the_trunk_one(reference) -> None:
    """Guards the gate: if the two were interchangeable, nothing below is load-
    bearing."""
    model, batch, (s_input, s_trunk, z), (_xl, t), _out = reference
    config = _reduced_config().architecture

    _si, zij = diffusion_conditioning(
        _as_jax(batch),
        jnp.asarray(s_input.numpy()),
        jnp.asarray(s_trunk.numpy()),
        jnp.asarray(z.numpy()),
        jnp.asarray(t.numpy()),
        map_diffusion_conditioning(
            dict(model.diffusion_module.diffusion_conditioning.state_dict())
        ),
        sigma_data=config.diffusion_module.diffusion_module.sigma_data,
        max_relative_idx=config.diffusion_module.diffusion_conditioning.max_relative_idx,
        max_relative_chain=(
            config.diffusion_module.diffusion_conditioning.max_relative_chain
        ),
        token_mask=jnp.asarray(batch["token_mask"].numpy()),
    )
    assert zij.shape == tuple(z.shape)
    assert not np.allclose(
        np.asarray(zij), z.detach().numpy(), rtol=1e-2, atol=1e-2
    ), "conditioned zij equals the trunk zij, so this test proves nothing"


def test_denoiser_matches_upstream(reference) -> None:
    model, batch, (s_input, s_trunk, z), (xl_noisy, t), expected = reference
    config = _reduced_config().architecture
    conditioning = config.diffusion_module.diffusion_conditioning

    jax_batch = _as_jax(batch)
    si, zij = diffusion_conditioning(
        jax_batch,
        jnp.asarray(s_input.numpy()),
        jnp.asarray(s_trunk.numpy()),
        jnp.asarray(z.numpy()),
        jnp.asarray(t.numpy()),
        map_diffusion_conditioning(
            dict(model.diffusion_module.diffusion_conditioning.state_dict())
        ),
        sigma_data=config.diffusion_module.diffusion_module.sigma_data,
        max_relative_idx=conditioning.max_relative_idx,
        max_relative_chain=conditioning.max_relative_chain,
        token_mask=jax_batch["token_mask"],
    )
    actual = denoise(
        jax_batch,
        jnp.asarray(xl_noisy.numpy()),
        jnp.asarray(t.numpy()),
        si,
        jnp.asarray(s_trunk.numpy()),
        zij,
        map_denoiser(dict(model.diffusion_module.state_dict())),
        n_query=config.diffusion_module.atom_attn_enc.n_query,
        n_key=config.diffusion_module.atom_attn_enc.n_key,
        atom_heads=config.diffusion_module.atom_attn_enc.no_heads,
        token_heads=config.diffusion_module.diffusion_transformer.no_heads,
        n_token=N_TOKEN,
        sigma_data=config.diffusion_module.diffusion_module.sigma_data,
    )
    assert actual.shape == tuple(expected.shape)
    np.testing.assert_allclose(
        np.asarray(actual, dtype=np.float64),
        expected.detach().numpy().astype(np.float64),
        rtol=RTOL,
        atol=ATOL,
        err_msg="denoised coordinates diverged from upstream DiffusionModule",
    )
