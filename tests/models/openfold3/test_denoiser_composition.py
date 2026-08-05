"""The denoiser body's composition order.

Every piece it calls is already gated against upstream individually
(`test_torch_parity_atom_encoder`, `_atom_decoder`, `_diffusion_transformer`,
`_diffusion_schedule`). What this file checks is the wiring between them, which
is where a composition can go wrong while every part stays correct.

Upstream's `DiffusionModule` takes a full ConfigDict and builds `si`/`zij` with an
internal `DiffusionConditioning`, so it is not constructible at unit scale; the
reference here is its forward body transcribed against our own gated pieces.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.openfold3.models.atom_features import (
    atom_attention_decoder,
    atom_attention_encoder,
)
from foldjax.models.openfold3.models.denoiser import denoise
from foldjax.models.openfold3.models.diffusion_schedule import (
    combine_denoiser_output,
    scale_noisy_positions,
)
from foldjax.models.openfold3.models.diffusion_transformer import diffusion_transformer
from foldjax.models.openfold3.models.primitives import layer_norm, linear

pytestmark = pytest.mark.torch_parity

C_ATOM, C_ATOM_PAIR, C_TOKEN, C_S, C_Z = 8, 6, 8, 8, 4
C_ELEMENT, C_CHARS = 12, 5
N_QUERY, N_KEY, ATOM_HEADS, TOKEN_HEADS = 4, 8, 2, 2
COUNTS = [4, 3, 5]
N_ATOM, N_TOKEN = sum(COUNTS), len(COUNTS)
SIGMA_DATA = 16.0


def _torch():
    import torch

    torch.manual_seed(0)
    return torch


def _parts(torch, randomized):
    """Build and map each sub-module the denoiser composes."""
    from ml_collections import ConfigDict
    from openfold3.core.model.layers.diffusion_transformer import DiffusionTransformer
    from openfold3.core.model.layers.sequence_local_atom_attention import (
        AtomAttentionDecoder,
        AtomAttentionEncoder,
    )
    from openfold3.core.model.primitives import LayerNorm, Linear

    from foldjax.models.openfold3.bridge.torch_mapping import (
        map_atom_attention_decoder,
        map_atom_attention_encoder,
        map_diffusion_transformer,
        map_layer_norm,
        map_linear,
    )

    enc = randomized(
        AtomAttentionEncoder(
            c_atom_ref=ConfigDict(
                {"element": C_ELEMENT, "name_chars": 4 * C_CHARS}
            ),
            c_atom=C_ATOM,
            c_atom_pair=C_ATOM_PAIR,
            c_token=C_TOKEN,
            add_noisy_pos=True,
            c_hidden=4,
            no_heads=ATOM_HEADS,
            no_blocks=1,
            n_transition=2,
            n_query=N_QUERY,
            n_key=N_KEY,
            use_ada_layer_norm=True,
            c_s=C_S,
            c_z=C_Z,
        )
    )
    dec = randomized(
        AtomAttentionDecoder(
            c_token=C_TOKEN,
            c_atom=C_ATOM,
            c_atom_pair=C_ATOM_PAIR,
            c_hidden=4,
            no_heads=ATOM_HEADS,
            no_blocks=1,
            n_transition=2,
            n_query=N_QUERY,
            n_key=N_KEY,
            use_ada_layer_norm=True,
        )
    )
    dt = randomized(
        DiffusionTransformer(
            c_a=C_TOKEN,
            c_s=C_S,
            c_z=C_Z,
            c_hidden=4,
            no_heads=TOKEN_HEADS,
            no_blocks=1,
            n_transition=2,
            use_ada_layer_norm=True,
            n_query=None,
            n_key=None,
            inf=1e9,
        )
    )
    ln_s = randomized(LayerNorm(C_S))
    lin_s = randomized(Linear(C_S, C_TOKEN, bias=False))
    ln_a = randomized(LayerNorm(C_TOKEN))

    from foldjax.models.openfold3.models.denoiser import DenoiserParams

    params = DenoiserParams(
        atom_attn_enc=map_atom_attention_encoder(dict(enc.state_dict())),
        layer_norm_s=map_layer_norm(dict(ln_s.state_dict())),
        linear_s=map_linear(dict(lin_s.state_dict()), bias=False),
        diffusion_transformer=map_diffusion_transformer(dict(dt.state_dict())),
        layer_norm_a=map_layer_norm(dict(ln_a.state_dict())),
        atom_attn_dec=map_atom_attention_decoder(dict(dec.state_dict())),
    )
    return params


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
    }


def _inputs(torch):
    return (
        {k: jnp.asarray(v.numpy()) for k, v in _batch(torch).items()},
        jnp.asarray(torch.randn(1, N_ATOM, 3).numpy()),
        jnp.asarray(torch.tensor([10.0]).numpy()),
        jnp.asarray(torch.randn(1, N_TOKEN, C_S).numpy()),
        jnp.asarray(torch.randn(1, N_TOKEN, C_S).numpy()),
        jnp.asarray(torch.randn(1, N_TOKEN, N_TOKEN, C_Z).numpy()),
    )


def _kwargs() -> dict:
    return {
        "n_query": N_QUERY,
        "n_key": N_KEY,
        "atom_heads": ATOM_HEADS,
        "token_heads": TOKEN_HEADS,
        "n_token": N_TOKEN,
        "sigma_data": SIGMA_DATA,
    }


def test_denoise_matches_the_transcribed_forward(openfold3_source, randomized) -> None:
    torch = _torch()
    params = _parts(torch, randomized)
    batch, xl, t, si, si_trunk, zij = _inputs(torch)

    actual = denoise(batch, xl, t, si, si_trunk, zij, params, **_kwargs())

    # Transcription of DiffusionModule.forward lines 2-9.
    atom_mask = batch["atom_mask"]
    xl_masked = xl * atom_mask[..., None]
    rl_noisy = scale_noisy_positions(xl_masked, t, sigma_data=SIGMA_DATA)
    ai, ql, cl, plm = atom_attention_encoder(
        batch,
        params.atom_attn_enc,
        n_query=N_QUERY,
        n_key=N_KEY,
        no_heads=ATOM_HEADS,
        n_token=N_TOKEN,
        rl=rl_noisy,
        si_trunk=si_trunk,
        zij_trunk=zij,
    )
    ai = ai + linear(layer_norm(si, params.layer_norm_s), params.linear_s)
    ai = diffusion_transformer(
        ai,
        si,
        zij,
        params.diffusion_transformer,
        no_heads=TOKEN_HEADS,
        mask=batch["token_mask"],
    )
    ai = layer_norm(ai, params.layer_norm_a)
    rl_update = atom_attention_decoder(
        batch, ai, ql, cl, plm, params.atom_attn_dec,
        n_query=N_QUERY, n_key=N_KEY, no_heads=ATOM_HEADS,
    )
    expected = combine_denoiser_output(
        xl_masked, rl_update, t, sigma_data=SIGMA_DATA
    ) * atom_mask[..., None]

    np.testing.assert_allclose(
        np.asarray(actual), np.asarray(expected), rtol=1e-6, atol=1e-6
    )


def test_output_is_a_blend_not_the_raw_update(openfold3_source, randomized) -> None:
    """At low noise the denoised output must stay near the noisy input."""
    torch = _torch()
    params = _parts(torch, randomized)
    batch, xl, _t, si, si_trunk, zij = _inputs(torch)
    tiny = jnp.asarray([1e-4])
    out = denoise(batch, xl, tiny, si, si_trunk, zij, params, **_kwargs())
    np.testing.assert_allclose(
        np.asarray(out), np.asarray(xl * batch["atom_mask"][..., None]),
        rtol=1e-3, atol=1e-3,
    )


def test_masked_atoms_stay_zero(openfold3_source, randomized) -> None:
    torch = _torch()
    params = _parts(torch, randomized)
    batch, xl, t, si, si_trunk, zij = _inputs(torch)
    batch = dict(batch)
    batch["atom_mask"] = jnp.asarray(
        np.concatenate([np.ones(N_ATOM - 3), np.zeros(3)])[None, :], dtype=jnp.float32
    )
    out = np.asarray(denoise(batch, xl, t, si, si_trunk, zij, params, **_kwargs()))
    assert np.allclose(out[:, -3:], 0.0)


def test_output_shape_is_coordinates(openfold3_source, randomized) -> None:
    torch = _torch()
    params = _parts(torch, randomized)
    batch, xl, t, si, si_trunk, zij = _inputs(torch)
    out = denoise(batch, xl, t, si, si_trunk, zij, params, **_kwargs())
    assert out.shape == (1, N_ATOM, 3)
    assert np.isfinite(np.asarray(out)).all()
