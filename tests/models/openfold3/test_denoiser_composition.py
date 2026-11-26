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

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.openfold3 import inference
from foldjax.models.openfold3.dtype import narrow_floats
from foldjax.models.openfold3.models import atom_features, denoiser
from foldjax.models.openfold3.models.atom_blocks import pair_rep_to_blocks
from foldjax.models.openfold3.models.atom_features import (
    atom_attention_decoder,
    atom_attention_encoder,
)
from foldjax.models.openfold3.models.augmentation import centre_random_augmentation
from foldjax.models.openfold3.models.denoiser import denoise
from foldjax.models.openfold3.models.diffusion_conditioning import (
    pair_conditioning,
    single_conditioning,
)
from foldjax.models.openfold3.models.diffusion_schedule import (
    combine_denoiser_output,
    noise_schedule,
    scale_noisy_positions,
)
from foldjax.models.openfold3.models.diffusion_transformer import (
    atom_transformer,
    diffusion_transformer,
)
from foldjax.models.openfold3.models.primitives import layer_norm, linear
from foldjax.models.openfold3.models.sampler import sample_diffusion

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


@pytest.mark.torch_parity
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


@pytest.mark.torch_parity
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


@pytest.mark.torch_parity
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


@pytest.mark.torch_parity
def test_output_shape_is_coordinates(openfold3_source, randomized) -> None:
    torch = _torch()
    params = _parts(torch, randomized)
    batch, xl, t, si, si_trunk, zij = _inputs(torch)
    out = denoise(batch, xl, t, si, si_trunk, zij, params, **_kwargs())
    assert out.shape == (1, N_ATOM, 3)
    assert np.isfinite(np.asarray(out)).all()


def _shared_pair_inputs(width: int = 3, *, dtype=jnp.float32, real_atoms: int = 75):
    from tests.models.openfold3 import atom_cp_fixtures as fx

    case = fx.build_case(real_atoms=real_atoms)
    base_params = fx.build_params()
    params = base_params if dtype == jnp.float32 else narrow_floats(base_params, dtype)
    xl = jax.random.normal(jax.random.key(7), (width, fx.N_ATOM, 3))
    si = jnp.broadcast_to(case.si[:1], (width, *case.si.shape[1:])).astype(dtype)
    si_trunk = jnp.broadcast_to(
        case.si_trunk[:1], (width, *case.si_trunk.shape[1:])
    ).astype(dtype)
    zij_1 = case.zij[:1].astype(dtype)
    zij_w = jnp.broadcast_to(zij_1, (width, *zij_1.shape[1:]))
    batch_w = {
        k: jnp.broadcast_to(v[:1], (width, *v.shape[1:]))
        for k, v in case.batch.items()
    }
    batch_pair1 = {
        k: (v[:1] if k in ("atom_to_token_index", "atom_mask") else batch_w[k])
        for k, v in case.batch.items()
    }
    kw = dict(
        n_query=fx.N_QUERY,
        n_key=fx.N_KEY,
        atom_heads=fx.ATOM_HEADS,
        token_heads=fx.TOKEN_HEADS,
        n_token=fx.N_TOKEN,
        sigma_data=fx.SIGMA_DATA,
    )
    dims = (fx.N_TOKEN, fx.C_Z, fx.C_ATOM_PAIR)
    return (
        batch_w, batch_pair1, xl, case.t, si, si_trunk, zij_1, zij_w, params, kw, dims
    )


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("scan_blocks", [True, False])
def test_width_one_pair_matches_widened_pair_bitwise(
    dtype, scan_blocks: bool, monkeypatch
) -> None:
    """Keeping the sample-shared pair one sample wide preserves exact arithmetic."""
    batch_w, batch_pair1, xl, t, si, si_trunk, zij_1, zij_w, params, kw, _ = (
        _shared_pair_inputs(width=3, dtype=dtype)
    )
    if not scan_blocks:
        monkeypatch.setattr(
            denoiser,
            "diffusion_transformer",
            functools.partial(diffusion_transformer, scan_blocks=False),
        )
        monkeypatch.setattr(
            atom_features,
            "atom_transformer",
            functools.partial(atom_transformer, scan_blocks=False),
        )

    run = jax.jit(lambda b, x, z: denoise(b, x, t, si, si_trunk, z, params, **kw))
    out_wide = np.asarray(run(batch_w, xl, zij_w))
    out_pair1 = np.asarray(run(batch_pair1, xl, zij_1))
    out_zonly = np.asarray(run(batch_w, xl, zij_1))

    np.testing.assert_array_equal(out_pair1, out_wide)
    np.testing.assert_array_equal(out_zonly, out_wide)
    assert not np.allclose(out_pair1[0], out_pair1[1])
    np.testing.assert_array_equal(out_pair1[:, 75:], 0.0)


@pytest.mark.parametrize(
    ("dtype", "tag"), [(jnp.float32, "f32"), (jnp.bfloat16, "bf16")]
)
def test_denoiser_hlo_keeps_pair_ln_one_sample_wide(dtype, tag: str) -> None:
    """No [width, N, N, C_z] or [width, N, N, C_atom_pair] survives in denoiser HLO."""
    width = 3
    batch_w, batch_pair1, xl, t, si, si_trunk, zij_1, zij_w, params, kw, dims = (
        _shared_pair_inputs(width=width, dtype=dtype)
    )
    n_tok, c_z, c_ap = dims
    run = jax.jit(lambda b, x, z: denoise(b, x, t, si, si_trunk, z, params, **kw))
    narrow_hlo = (
        run.lower(batch_pair1, xl, zij_1).compiler_ir(dialect="hlo").as_hlo_text()
    )
    zonly_hlo = run.lower(batch_w, xl, zij_1).compiler_ir(dialect="hlo").as_hlo_text()
    wide_hlo = run.lower(batch_w, xl, zij_w).compiler_ir(dialect="hlo").as_hlo_text()

    assert f"{tag}[{width},{n_tok},{n_tok},{c_z}]" in wide_hlo
    for hlo in (narrow_hlo, zonly_hlo):
        for shape in (
            f"{tag}[{width},{n_tok},{n_tok},{c_z}]",
            f"f32[{width},{n_tok},{n_tok},{c_z}]",
            f"{tag}[{width},{n_tok},{n_tok},{c_ap}]",
            f"f32[{width},{n_tok},{n_tok},{c_ap}]",
        ):
            assert shape not in hlo
        assert f"{tag}[1,{n_tok},{n_tok},{c_z}]" in hlo
        assert f"{tag}[1,{n_tok},{n_tok},{c_ap}]" in hlo


def test_sampler_rollout_matches_explicitly_widened_pair(monkeypatch) -> None:
    """``_predict_from_trunk`` rollout at num_samples=3 matches widened-pair rollout."""
    from tests.models.openfold3.scripts import diffusion_chunk_cp_census as census

    num_samples = 3
    batch, trunk_output = census._batch(), census._trunk_output()
    cond_params = census._conditioning_params(census._relpos_dims(batch))
    denoiser_params = census.build_params()
    config = census._config(devices=1, layout="1d")._replace(
        num_samples=num_samples, num_steps=3, diffusion_chunk_size=None,
        cp_atom_windows=False, dtype="float32", confidence_dtype="float32",
    )
    n_tok, pae_bins = census.N_TOKEN, census.PAE_BINS
    stubs = {
        "token_representative_atoms": lambda b, c, m, t: (
            c[..., :n_tok, :], jnp.ones((num_samples, n_tok))
        ),
        "token_frame_atoms": lambda *a, **k: (
            (None, None, None), jnp.ones((num_samples, n_tok), dtype=jnp.bool_)
        ),
        "pairformer_embedding": lambda si_in, si, z, x, p, **k: (
            si, jnp.zeros((*z.shape[:-1], 3), z.dtype)
        ),
        "predicted_aligned_error_head": lambda z, p: jnp.zeros(
            (*z.shape[:-1], pae_bins), z.dtype
        ),
        "predicted_distance_error_head": lambda z, p: jnp.zeros(
            (*z.shape[:-1], pae_bins), z.dtype
        ),
        "distogram_head": lambda z, p: jnp.zeros(
            (1, n_tok, n_tok, pae_bins), z.dtype
        ),
        "atom_logit_head": lambda s, p, m, *, c_out, n_atom, **k: jnp.zeros(
            (*s.shape[:-2], n_atom, c_out), s.dtype
        ),
    }
    for name, stub in stubs.items():
        monkeypatch.setattr(inference, name, stub)
    params = inference.InferenceParams(
        None, cond_params, denoiser_params, None, None, None, None, None, None
    )
    key = jax.random.key(19)
    pred_fn = jax.jit(
        lambda b, tr: inference._predict_from_trunk(
            key, b, params, config, None, trunk_output=tr
        ).coordinates
    )
    rollout_hlo = (
        pred_fn.lower(batch, trunk_output).compiler_ir(dialect="hlo").as_hlo_text()
    )
    assert f"f32[{num_samples},{n_tok},{n_tok},{census.C_Z}]" not in rollout_hlo
    assert f"f32[1,{n_tok},{n_tok},{census.C_Z}]" in rollout_hlo
    actual = pred_fn(batch, trunk_output)

    schedule = noise_schedule(
        config.num_steps, sigma_data=config.sigma_data,
        s_max=config.s_max, s_min=config.s_min, p=config.p,
    )

    def run_widened(b, tr):
        s_input, s_trunk, z = tr
        zij_base = pair_conditioning(
            b, z, cond_params, max_relative_idx=config.max_relative_idx,
            max_relative_chain=config.max_relative_chain, token_mask=b["token_mask"],
        )

        def wide_denoise(xl_noisy, t):
            w = xl_noisy.shape[0]
            si = inference._expand_samples(
                single_conditioning(
                    s_input, s_trunk, t, cond_params,
                    sigma_data=config.sigma_data, token_mask=b["token_mask"],
                ), w,
            )
            return denoise(
                {k: inference._expand_samples(v, w) for k, v in b.items()},
                xl_noisy, t, si, inference._expand_samples(s_trunk, w),
                inference._expand_samples(zij_base, w), denoiser_params,
                n_query=config.n_query,
                n_key=config.n_key,
                atom_heads=config.atom_heads,
                token_heads=config.token_heads, n_token=config.n_token,
                sigma_data=config.sigma_data,
            )

        return sample_diffusion(
            key, schedule, (num_samples, config.n_atom, 3), wide_denoise,
            gamma_0=config.gamma_0, gamma_min=config.gamma_min,
            noise_scale=config.noise_scale, step_scale=config.step_scale,
            augment_fn=lambda k, xl: centre_random_augmentation(
                k, xl, inference._expand_samples(b["atom_mask"], xl.shape[0])
            ),
        )

    expected = jax.jit(run_widened)(batch, trunk_output)
    assert actual.shape == (num_samples, config.n_atom, 3)
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
    assert not np.allclose(np.asarray(actual[0]), np.asarray(actual[1]))


def test_pair_rep_to_blocks_rejects_incompatible_batch_widths() -> None:
    """Non-broadcastable pair/index batch widths fail loudly."""
    with pytest.raises(ValueError, match="Incompatible shapes"):
        pair_rep_to_blocks(
            jnp.zeros((2, 4, 4, 3)),
            jnp.zeros((3, 8), dtype=jnp.int32),
            jnp.ones((3, 8)),
            n_query=4,
            n_key=8,
        )
