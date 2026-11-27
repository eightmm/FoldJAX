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

from foldjax.models._stacking import prestack_layer_lists
from foldjax.models.openfold3 import inference
from foldjax.models.openfold3.dtype import narrow_floats
from foldjax.models.openfold3.models import atom_features, denoiser
from foldjax.models.openfold3.models.atom_blocks import pair_rep_to_blocks
from foldjax.models.openfold3.models.atom_features import (
    atom_attention_decoder,
    atom_attention_encoder,
)
from foldjax.models.openfold3.models.attention_pair_bias import (
    cross_attention_pair_bias,
)
from foldjax.models.openfold3.models.augmentation import centre_random_augmentation
from foldjax.models.openfold3.models.denoiser import (
    AtomDiffusionCache,
    denoise,
    prepare_atom_diffusion_cache,
)
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
    AtomTransformerParams,
    atom_transformer,
    atom_transformer_pair_biases,
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


def _assert_width_invariant(actual, expected, real_atoms: int = 75) -> None:
    """Width-1 against widened denoiser output, to a few float32 ULP.

    Bitwise on the development host, but a hosted CPU runner's XLA picks its
    GEMM kernel by the M extent, which moved results 2.9e-6 at |x| ~ 5 (push
    CI run 45916764746). Device bitwise identity is checked on GPU instead.
    Padded atoms stay exactly zero on both sides.
    """
    actual, expected = np.asarray(actual), np.asarray(expected)
    np.testing.assert_array_equal(actual[:, real_atoms:], 0.0)
    np.testing.assert_array_equal(expected[:, real_atoms:], 0.0)
    np.testing.assert_allclose(
        actual[:, :real_atoms], expected[:, :real_atoms], rtol=0, atol=1e-5
    )


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
def test_width_one_pair_matches_widened_pair(
    dtype, scan_blocks: bool, monkeypatch
) -> None:
    """Keeping the sample-shared pair one sample wide preserves the arithmetic."""
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

    _assert_width_invariant(out_pair1, out_wide)
    _assert_width_invariant(out_zonly, out_wide)
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
    # Hoisting width-1 atom conditioning changes GEMM M-shapes in the atom
    # embedder/MLP/bias projections relative to the width-3 reference (~1 ULP
    # float32 rounding per step, ~2.6e-5 rel over 3 EDM steps at ~1e3 scale).
    # Same-width cached vs uncached is checked bitwise below.
    np.testing.assert_allclose(
        np.asarray(actual), np.asarray(expected), rtol=1e-4, atol=1e-3
    )
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


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("scan_blocks", [True, False])
@pytest.mark.parametrize("width", [1, 3])
def test_atom_diffusion_cache_matches_uncached_bitwise_at_same_width(
    dtype, scan_blocks: bool, width: int, monkeypatch
) -> None:
    """At matched leading width, cached and uncached denoise are bitwise identical."""
    batch_w, _batch_pair1, xl, t, si, si_trunk, _zij_1, zij_w, params, kw, _ = (
        _shared_pair_inputs(width=width, dtype=dtype, real_atoms=75)
    )
    if not scan_blocks:
        monkeypatch.setattr(
            atom_features,
            "atom_transformer",
            functools.partial(atom_transformer, scan_blocks=False),
        )

    out_uncached = np.asarray(
        jax.jit(
            lambda b, x, st, z: denoise(b, x, t, si, st, z, params, **kw)
        )(batch_w, xl, si_trunk, zij_w)
    )

    def run_cached(b, x, st, z):
        cache = prepare_atom_diffusion_cache(
            b, st, z, params, n_query=kw["n_query"], n_key=kw["n_key"]
        )
        return denoise(b, x, t, si, st, z, params, **kw, atom_cache=cache)

    out_cached = np.asarray(jax.jit(run_cached)(batch_w, xl, si_trunk, zij_w))

    np.testing.assert_array_equal(out_cached, out_uncached)
    np.testing.assert_array_equal(out_cached[:, 75:], 0.0)
    if width > 1:
        assert not np.allclose(out_cached[0], out_cached[1])


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("prestack", [False, True])
def test_width_one_atom_diffusion_cache_matches_widened_uncached(
    dtype, prestack: bool
) -> None:
    """Width-1 cache matches multi-sample uncached denoise to GEMM rounding."""
    batch_w, batch_pair1, xl, t, si, si_trunk, zij_1, zij_w, params, kw, _ = (
        _shared_pair_inputs(width=3, dtype=dtype, real_atoms=75)
    )
    if prestack:
        params = prestack_layer_lists(params)
    batch_1 = {k: v[:1] for k, v in batch_w.items()}
    si_trunk_1 = si_trunk[:1]

    cache_1 = jax.jit(
        lambda b, st, z: prepare_atom_diffusion_cache(
            b, st, z, params, n_query=kw["n_query"], n_key=kw["n_key"]
        )
    )(batch_1, si_trunk_1, zij_1)
    assert cache_1.cl.shape[0] == 1
    assert cache_1.enc_pair_bias.shape[1] == 1
    assert cache_1.dec_pair_bias.shape[1] == 1

    out_uncached = np.asarray(
        jax.jit(lambda b, x, z: denoise(b, x, t, si, si_trunk, z, params, **kw))(
            batch_w, xl, zij_w
        )
    )
    run_cached = jax.jit(
        lambda b, x, z, c: denoise(
            b, x, t, si, si_trunk, z, params, **kw, atom_cache=c
        )
    )
    # Both width-1 atom_mask (the inference.py layout) and width-3 atom_mask
    # (direct callers passing widened batch) must match the uncached reference.
    out_cached_p1 = np.asarray(run_cached(batch_pair1, xl, zij_1, cache_1))
    out_cached_w = np.asarray(run_cached(batch_w, xl, zij_1, cache_1))

    # Width-1 vs width-3 flattens a 3x smaller GEMM M-dimension in the atom
    # projections (~1-2 float32 ULPs, measured <= 3e-6 abs at ~15 scale).
    for out in (out_cached_p1, out_cached_w):
        np.testing.assert_allclose(
            out.astype(np.float32),
            out_uncached.astype(np.float32),
            rtol=1e-5,
            atol=1e-5,
        )
        np.testing.assert_array_equal(out[:, 75:], 0.0)
        assert not np.allclose(out[0], out[1])


def test_rollout_hlo_hoists_atom_conditioning_and_supports_chunked_width1(
    monkeypatch,
) -> None:
    """``plm`` dies before the scan; chunked width-1 rollout uses the cache too."""
    from tests.models.openfold3 import atom_cp_fixtures as fx
    from tests.models.openfold3.scripts import diffusion_chunk_cp_census as census

    num_samples = 3
    batch, trunk_output = census._batch(), census._trunk_output()
    cond_params = census._conditioning_params(census._relpos_dims(batch))
    denoiser_params = prestack_layer_lists(census.build_params())
    config = census._config(devices=1, layout="1d")._replace(
        num_samples=num_samples,
        num_steps=3,
        diffusion_chunk_size=None,
        cp_atom_windows=False,
        dtype="float32",
        confidence_dtype="float32",
    )
    n_tok, pae_bins = census.N_TOKEN, census.PAE_BINS
    stubs = {
        "token_representative_atoms": lambda b, c, m, t: (
            c[..., :n_tok, :],
            jnp.ones((num_samples, n_tok)),
        ),
        "token_frame_atoms": lambda *a, **k: (
            (None, None, None),
            jnp.ones((num_samples, n_tok), dtype=jnp.bool_),
        ),
        "pairformer_embedding": lambda si_in, si, z, x, p, **k: (
            si,
            jnp.zeros((*z.shape[:-1], 3), z.dtype),
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
    key = jax.random.key(23)

    def run(cfg):
        fn = jax.jit(
            lambda b, tr: inference._predict_from_trunk(
                key, b, params, cfg, None, trunk_output=tr
            ).coordinates
        )
        hlo = fn.lower(batch, trunk_output).compiler_ir(dialect="hlo").as_hlo_text()
        return fn(batch, trunk_output), hlo

    whole_coords, whole_hlo = run(config)
    chunked_coords, chunked_hlo = run(config._replace(diffusion_chunk_size=1))

    plm_wide = (
        f"f32[{num_samples},{fx.N_BLOCKS},{fx.N_QUERY},{fx.N_KEY},{fx.C_ATOM_PAIR}]"
    )
    plm_one = f"f32[1,{fx.N_BLOCKS},{fx.N_QUERY},{fx.N_KEY},{fx.C_ATOM_PAIR}]"
    bias_stack = (
        f"f32[{fx.NO_BLOCKS},1,{fx.N_BLOCKS},{fx.ATOM_HEADS},{fx.N_QUERY},{fx.N_KEY}]"
    )
    for hlo, coord_width in ((whole_hlo, num_samples), (chunked_hlo, 1)):
        assert plm_wide not in hlo
        assert bias_stack in hlo
        coords = f"f32[{coord_width},{fx.N_ATOM},3]"
        # Sub-computations carrying both the noisy coordinates and the
        # precomputed bias stack belong to the diffusion rollout scan (not
        # ENTRY or `atom_transformer_pair_biases`'s own `lax.map`, which has no
        # coordinates). None of those rollout scan bodies may retain `plm`.
        scan_bodies = [
            block
            for block in hlo.split("\n\n")
            if bias_stack in block
            and coords in block
            and not block.lstrip().startswith("ENTRY ")
        ]
        assert scan_bodies
        for body in scan_bodies:
            assert plm_one not in body

    np.testing.assert_allclose(
        np.asarray(chunked_coords),
        np.asarray(whole_coords),
        rtol=1e-4,
        atol=1e-3,
    )


def test_use_atom_diffusion_cache_respects_cp_and_width1_atom_limit(
    monkeypatch,
) -> None:
    """Cache gating covers width-1 limits, width >= 2, and active CP meshes."""
    limit = inference._MAX_WIDTH1_ATOM_CACHE_ATOMS
    cfg_w1 = inference.released_config(
        n_token=8, n_atom=16, num_samples=1, diffusion_chunk_size=None
    )
    cfg_chunk1 = inference.released_config(
        n_token=8, n_atom=16, num_samples=5, diffusion_chunk_size=1
    )
    cfg_w5 = inference.released_config(
        n_token=8, n_atom=16, num_samples=5, diffusion_chunk_size=None
    )

    for cfg in (cfg_w1, cfg_chunk1):
        assert inference._use_atom_diffusion_cache(cfg, n_atom=0) is True
        assert inference._use_atom_diffusion_cache(cfg, n_atom=limit) is True
        assert inference._use_atom_diffusion_cache(cfg, n_atom=limit + 1) is False

    assert inference._use_atom_diffusion_cache(cfg_w5, n_atom=limit + 1) is True

    monkeypatch.setattr(inference, "cp_mesh", lambda: object())
    assert inference._use_atom_diffusion_cache(cfg_w5, n_atom=16) is False
    assert inference._use_atom_diffusion_cache(cfg_w1, n_atom=16) is False


def test_atom_cache_validation_and_edge_cases(monkeypatch) -> None:
    """Exercise error exits and zero/one-block edges across the cache path."""
    batch_w, _bp1, xl, t, si, si_trunk, _z1, zij_w, params, kw, _ = (
        _shared_pair_inputs(width=1, dtype=jnp.float32, real_atoms=75)
    )
    enc = params.atom_attn_enc
    dec = params.atom_attn_dec
    n_q, n_k, heads = kw["n_query"], kw["n_key"], kw["atom_heads"]

    cache = prepare_atom_diffusion_cache(
        batch_w, si_trunk, zij_w, params, n_query=n_q, n_key=n_k
    )
    assert isinstance(cache, AtomDiffusionCache)

    # 1-block (non-scannable) and 0-block pair-bias / transformer paths
    cl_0 = cache.cl
    plm_dummy = jnp.zeros((1, 24, n_q, n_k, 6), dtype=jnp.float32)
    one_block_params = AtomTransformerParams(
        blocks=(enc.atom_transformer.blocks[0],),
        layer_norm_z=enc.atom_transformer.layer_norm_z,
    )
    pb_1 = atom_transformer_pair_biases(plm_dummy, one_block_params)
    assert pb_1.shape == (1, 1, 24, heads, n_q, n_k)
    np.testing.assert_array_equal(
        np.asarray(
            atom_transformer(
                cl_0,
                cl_0,
                None,
                one_block_params,
                no_heads=heads,
                n_query=n_q,
                n_key=n_k,
                mask=batch_w["atom_mask"],
                pair_biases=pb_1,
            )
        ),
        np.asarray(
            atom_transformer(
                cl_0,
                cl_0,
                plm_dummy,
                one_block_params,
                no_heads=heads,
                n_query=n_q,
                n_key=n_k,
                mask=batch_w["atom_mask"],
            )
        ),
    )

    zero_block_params = AtomTransformerParams(
        blocks=(), layer_norm_z=enc.atom_transformer.layer_norm_z
    )
    pb_0 = atom_transformer_pair_biases(plm_dummy, zero_block_params)
    assert pb_0.shape == (0, 1, 24, 0, n_q, n_k)
    np.testing.assert_array_equal(
        np.asarray(
            atom_transformer(
                cl_0,
                cl_0,
                None,
                zero_block_params,
                no_heads=heads,
                n_query=n_q,
                n_key=n_k,
                mask=batch_w["atom_mask"],
                pair_biases=pb_0,
            )
        ),
        np.asarray(cl_0),
    )

    # cross_attention_pair_bias validation
    block_apb = enc.atom_transformer.blocks[0].attention_pair_bias
    with pytest.raises(ValueError, match="exactly one of z or pair_bias"):
        cross_attention_pair_bias(
            cl_0, None, block_apb, no_heads=heads, n_query=n_q, n_key=n_k, s=cl_0
        )
    with pytest.raises(ValueError, match="exactly one of z or pair_bias"):
        cross_attention_pair_bias(
            cl_0,
            plm_dummy,
            block_apb,
            no_heads=heads,
            n_query=n_q,
            n_key=n_k,
            s=cl_0,
            pair_bias=cache.enc_pair_bias[0],
        )

    # atom_transformer validation
    with pytest.raises(ValueError, match="either z or pair_biases is required"):
        atom_transformer(
            cl_0,
            cl_0,
            None,
            enc.atom_transformer,
            no_heads=heads,
            n_query=n_q,
            n_key=n_k,
        )
    with pytest.raises(ValueError, match="pass either z or pair_biases, not both"):
        atom_transformer(
            cl_0,
            cl_0,
            plm_dummy,
            enc.atom_transformer,
            no_heads=heads,
            n_query=n_q,
            n_key=n_k,
            pair_biases=cache.enc_pair_bias,
        )
    with pytest.raises(ValueError, match="leading block axis"):
        atom_transformer(
            cl_0,
            cl_0,
            None,
            enc.atom_transformer,
            no_heads=heads,
            n_query=n_q,
            n_key=n_k,
            pair_biases=pb_1,
        )

    # atom_attention_encoder validation
    with pytest.raises(ValueError, match="supplied together"):
        atom_attention_encoder(
            batch_w,
            enc,
            n_query=n_q,
            n_key=n_k,
            no_heads=heads,
            n_token=kw["n_token"],
            rl=xl,
            cached_cl=cache.cl,
        )
    with pytest.raises(ValueError, match="rl is required"):
        atom_attention_encoder(
            batch_w,
            enc,
            n_query=n_q,
            n_key=n_k,
            no_heads=heads,
            n_token=kw["n_token"],
            cached_cl=cache.cl,
            cached_pair_bias=cache.enc_pair_bias,
        )
    enc_no_noisy = enc._replace(noisy_position_embedder=None)
    with pytest.raises(ValueError, match="noisy_position_embedder"):
        atom_attention_encoder(
            batch_w,
            enc_no_noisy,
            n_query=n_q,
            n_key=n_k,
            no_heads=heads,
            n_token=kw["n_token"],
            rl=xl,
            cached_cl=cache.cl,
            cached_pair_bias=cache.enc_pair_bias,
        )
    with pytest.raises(ValueError, match="noisy_position_embedder"):
        prepare_atom_diffusion_cache(
            batch_w,
            si_trunk,
            zij_w,
            params._replace(atom_attn_enc=enc_no_noisy),
            n_query=n_q,
            n_key=n_k,
        )

    # atom_attention_decoder validation
    ai_dummy = jnp.zeros((1, kw["n_token"], 8), dtype=jnp.float32)
    with pytest.raises(ValueError, match="exactly one of plm or cached_pair_bias"):
        atom_attention_decoder(
            batch_w,
            ai_dummy,
            cl_0,
            cl_0,
            None,
            dec,
            n_query=n_q,
            n_key=n_k,
            no_heads=heads,
        )
    with pytest.raises(ValueError, match="exactly one of plm or cached_pair_bias"):
        atom_attention_decoder(
            batch_w,
            ai_dummy,
            cl_0,
            cl_0,
            plm_dummy,
            dec,
            n_query=n_q,
            n_key=n_k,
            no_heads=heads,
            cached_pair_bias=cache.dec_pair_bias,
        )

    # CP refusal under active mesh
    monkeypatch.setattr(denoiser, "cp_mesh", lambda: object())
    monkeypatch.setattr(atom_features, "cp_mesh", lambda: object())
    with pytest.raises(ValueError, match="context-parallel mesh"):
        prepare_atom_diffusion_cache(
            batch_w, si_trunk, zij_w, params, n_query=n_q, n_key=n_k
        )
    with pytest.raises(ValueError, match="distributed cp_atom_windows"):
        denoise(
            batch_w,
            xl,
            t,
            si,
            si_trunk,
            zij_w,
            params,
            **kw,
            cp_atom_windows=True,
            atom_cache=cache,
        )
    with pytest.raises(ValueError, match="distributed cp_atom_windows"):
        atom_attention_encoder(
            batch_w,
            enc,
            n_query=n_q,
            n_key=n_k,
            no_heads=heads,
            n_token=kw["n_token"],
            rl=xl,
            cp_atom_windows=True,
            cached_cl=cache.cl,
            cached_pair_bias=cache.enc_pair_bias,
        )
    with pytest.raises(ValueError, match="distributed cp_atom_windows"):
        atom_attention_decoder(
            batch_w,
            ai_dummy,
            cl_0,
            cl_0,
            None,
            dec,
            n_query=n_q,
            n_key=n_k,
            no_heads=heads,
            cp_atom_windows=True,
            cached_pair_bias=cache.dec_pair_bias,
        )


def _token_dot_counts(hlo: str, width: int, n_token: int) -> tuple[int, int]:
    """Count token-axis dots at ``[width, n_token, ...]`` and ``[1, n_token, ...]``."""
    dots = [line for line in hlo.splitlines() if "dot(" in line]
    return (
        sum(f"[{width},{n_token}," in line for line in dots),
        sum(f"[1,{n_token}," in line for line in dots),
    )


def test_denoiser_and_rollout_keep_single_one_sample_wide(monkeypatch) -> None:
    """Width-1 ``si``/``si_trunk`` matches width-5 in denoiser and rollout HLO."""
    from tests.models.openfold3 import atom_cp_fixtures as fx
    from tests.models.openfold3.scripts import diffusion_chunk_cp_census as census

    width = 5
    keep_t1 = ("atom_to_token_index", "atom_mask", "token_mask", "num_atoms_per_token")
    for dtype in (jnp.float32, jnp.bfloat16):
        batch_w, batch_p1, xl, t, si_w, st_w, zij_1, _zw, params, kw, _ = (
            _shared_pair_inputs(width=width, dtype=dtype, real_atoms=75)
        )
        batch_1 = {k: v[:1] for k, v in batch_w.items()}
        batch_t1 = {k: (v[:1] if k in keep_t1 else v) for k, v in batch_w.items()}
        si_1, st_1 = si_w[:1], st_w[:1]
        cache = prepare_atom_diffusion_cache(
            batch_1, st_1, zij_1, params, n_query=kw["n_query"], n_key=kw["n_key"]
        )
        for scan_blocks in (True, False):
            with monkeypatch.context() as mp:
                if not scan_blocks:
                    mp.setattr(
                        denoiser, "diffusion_transformer",
                        functools.partial(diffusion_transformer, scan_blocks=False),
                    )
                run_c = jax.jit(
                    lambda b, s, st: denoise(
                        b, xl, t, s, st, zij_1, params, **kw, atom_cache=cache
                    )
                )
                out_w = np.asarray(run_c(batch_p1, si_w, st_w))
                out_1 = np.asarray(run_c(batch_t1, si_1, st_1))
                _assert_width_invariant(out_1, out_w)
                assert not np.allclose(out_1[0], out_1[1])
                np.testing.assert_array_equal(out_1[:, 75:], 0.0)
                if dtype == jnp.float32 and scan_blocks:
                    hw = run_c.lower(batch_p1, si_w, st_w).compiler_ir("hlo")
                    h1 = run_c.lower(batch_t1, si_1, st_1).compiler_ir("hlo")
                    assert _token_dot_counts(hw.as_hlo_text(), width, fx.N_TOKEN) == (
                        16, 1
                    )
                    assert _token_dot_counts(h1.as_hlo_text(), width, fx.N_TOKEN) == (
                        9, 8
                    )
                if scan_blocks:
                    run_u = jax.jit(
                        lambda b, s, st: denoise(b, xl, t, s, st, zij_1, params, **kw)
                    )
                    _assert_width_invariant(
                        np.asarray(run_u(batch_t1, si_1, st_1)),
                        np.asarray(run_u(batch_p1, si_w, st_w)),
                    )

    batch, trunk_out = census._batch(), census._trunk_output()
    stub_names = (
        "token_representative_atoms", "token_frame_atoms", "pairformer_embedding",
        "predicted_aligned_error_head", "predicted_distance_error_head",
        "distogram_head", "atom_logit_head",
    )
    originals = {name: getattr(inference, name) for name in stub_names}
    holder: dict[str, object] = {}
    census._install_stubs(holder, census._relpos_dims(batch))
    for name, orig in originals.items():
        stub = getattr(inference, name)
        setattr(inference, name, orig)
        monkeypatch.setattr(inference, name, stub)
    params = holder["params"]
    params = params._replace(denoiser=prestack_layer_lists(params.denoiser))
    cond_params, d_params = params.diffusion_conditioning, params.denoiser
    cfg = census._config(devices=1, layout="1d")._replace(
        num_samples=width, num_steps=3, diffusion_chunk_size=None,
        cp_atom_windows=False, dtype="float32", confidence_dtype="float32",
    )
    si_ps = single_conditioning(
        trunk_out[0], trunk_out[1], jnp.asarray([10.0, 8.0, 6.0, 4.0, 2.0]),
        cond_params, sigma_data=cfg.sigma_data, token_mask=batch["token_mask"],
    )
    assert si_ps.shape == (width, census.N_TOKEN, census.C_S)
    assert not np.allclose(np.asarray(si_ps[0]), np.asarray(si_ps[1]))

    key = jax.random.key(29)
    pred_fn = jax.jit(
        lambda b, tr: inference._predict_from_trunk(
            key, b, params, cfg, None, trunk_output=tr
        ).coordinates
    )
    hlo = pred_fn.lower(batch, trunk_out).compiler_ir("hlo").as_hlo_text()
    wide_dots, narrow_dots = _token_dot_counts(hlo, width, census.N_TOKEN)
    assert wide_dots == 9 and narrow_dots >= 8
    actual = np.asarray(pred_fn(batch, trunk_out))

    sched = noise_schedule(
        cfg.num_steps, sigma_data=cfg.sigma_data, s_max=cfg.s_max,
        s_min=cfg.s_min, p=cfg.p,
    )
    exp = inference._expand_samples

    def widened_rollout(b, tr):
        si_in, si_tr, z = tr
        zij = pair_conditioning(
            b, z, cond_params, max_relative_idx=cfg.max_relative_idx,
            max_relative_chain=cfg.max_relative_chain, token_mask=b["token_mask"],
        )
        cache = prepare_atom_diffusion_cache(
            b, si_tr, zij, d_params, n_query=cfg.n_query, n_key=cfg.n_key
        )

        def wide_denoise(xl_noisy, t_step):
            w = xl_noisy.shape[0]
            si_w = exp(
                single_conditioning(
                    si_in, si_tr, t_step, cond_params,
                    sigma_data=cfg.sigma_data, token_mask=b["token_mask"],
                ), w,
            )
            keep = ("atom_to_token_index", "atom_mask")
            b_w = {k: (v if k in keep else exp(v, w)) for k, v in b.items()}
            return denoise(
                b_w, xl_noisy, t_step, si_w, exp(si_tr, w), zij, d_params,
                n_query=cfg.n_query, n_key=cfg.n_key, atom_heads=cfg.atom_heads,
                token_heads=cfg.token_heads, n_token=cfg.n_token,
                sigma_data=cfg.sigma_data, atom_cache=cache,
            )

        return sample_diffusion(
            key, sched, (width, cfg.n_atom, 3), wide_denoise,
            gamma_0=cfg.gamma_0, gamma_min=cfg.gamma_min,
            noise_scale=cfg.noise_scale, step_scale=cfg.step_scale,
            augment_fn=lambda k, xl: centre_random_augmentation(
                k, xl, exp(b["atom_mask"], xl.shape[0])
            ),
        )

    expected = np.asarray(jax.jit(widened_rollout)(batch, trunk_out))
    # Bitwise on the development host. A hosted CPU runner picks its GEMM
    # kernel by the M extent, as `_assert_width_invariant` notes, and over the
    # three-step rollout that moved 126 of 1,440 coordinates by up to 8.8e-6
    # relative (2.4e-4 at |x| ~ 600; push CI run 45995632806).
    np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=1e-5)
    assert not np.allclose(actual[0], actual[1])


def test_diffusion_attention_tokamax_switch_and_wiring(monkeypatch) -> None:
    """Env switch keeps unset HLO identical, validates, yields to CP, and matches."""
    import warnings
    from types import SimpleNamespace

    from foldjax.models.openfold3.models import attention as attn_mod
    from foldjax.models.openfold3.models import attention_pair_bias as apb_mod
    from foldjax.models.openfold3.models.diffusion_transformer import (
        DiffusionTransformerParams,
    )
    from foldjax.models.protenix.models.primitives import attention_tokamax

    bw, bp1, xl, t, si, si_t, z1, _, params, kw, _ = _shared_pair_inputs(
        3, real_atoms=75
    )
    from foldjax import _openfold3_compile as compile_mod
    from foldjax.models import _pallas_pair

    env, h = compile_mod.DIFFUSION_ATTENTION_BACKEND_ENV, kw["token_heads"]
    resolve = compile_mod.resolve_diffusion_attention_backend
    assert compile_mod.DIFFUSION_ATTENTION_BACKENDS == attn_mod.ATTENTION_BACKENDS
    monkeypatch.delenv(env, raising=False)
    # Omitted: tokamax on a serial GPU process only.
    assert resolve(1) == "xla"
    with monkeypatch.context() as mp:
        mp.setattr(_pallas_pair, "gpu_process", lambda: True)
        assert (resolve(1), resolve(2)) == ("tokamax", "xla")
    for raw, want in (("xla", "xla"), ("XLA", "xla"), ("TOKAMAX", "tokamax")):
        monkeypatch.setenv(env, raw)
        assert (resolve(1), resolve(2)) == (want, "xla")
    for bad in ("", "triton"):
        monkeypatch.setenv(env, bad)
        with pytest.raises(ValueError, match=env):
            resolve(1)
        # A mesh never reads the variable, on any entry point.
        assert resolve(2) == "xla"
    monkeypatch.delenv(env, raising=False)

    def lower(ex):
        fn = jax.jit(lambda b, x, z: denoise(b, x, t, si, si_t, z, params, **kw, **ex))
        return fn.lower(bp1, xl, z1).compiler_ir(dialect="hlo").as_hlo_text()

    assert lower({}) == lower({"attention_backend": "xla"})

    dt0 = DiffusionTransformerParams((), params.diffusion_transformer.layer_norm_z)
    apb_p = params.diffusion_transformer.blocks[0].attention_pair_bias
    for call, pat in (
        (lambda: denoise(bp1, xl, t, si, si_t, z1, params, **kw, attention_backend="x"),
         "unsupported"),
        (lambda: diffusion_transformer(si, si, z1, dt0, no_heads=h,
                                       attention_backend="x"), "unsupported"),
        (lambda: attn_mod.attention(si, si, apb_p.mha, no_heads=h,
                                    mask=jnp.ones((3, 1, 1, 12), bool)), "boolean"),
        (lambda: attn_mod.attention(si, si, apb_p.mha, no_heads=h, biases=(z1, z1),
                                    backend="tokamax"), "at most"),
        (lambda: attn_mod.attention(si, si, apb_p.mha, no_heads=h, high_precision=True,
                                    backend="tokamax"), "high_precision"),
    ):
        with pytest.raises(ValueError, match=pat):
            call()

    # Masked tokens (samples 0..1) + fully masked sample 2 + width-1 pair bias.
    tm = jnp.array([[1.0] * 9 + [0.0] * 3] * 2 + [[0.0] * 12], jnp.float32)
    bp_m = {**bp1, "token_mask": tm}
    calls: list[tuple[int, ...]] = []

    def ref_dpa(q, k, v, *, bias=None, mask=None, scale=1.0, implementation=None):
        assert q.dtype == k.dtype == v.dtype == jnp.float32 and scale == 1.0
        assert bias is not None and bias.shape[0] == 1
        assert mask is None or mask.dtype == jnp.bool_
        calls.append(bias.shape)
        logits = jnp.einsum("...qhd,...khd->...hqk", q, k) * scale + bias
        logits = logits if mask is None else jnp.where(mask, logits, -jnp.inf)
        return jnp.einsum("...hqk,...khd->...qhd", jax.nn.softmax(logits, axis=-1), v)

    monkeypatch.setattr(attention_tokamax, "_TOKAMAX_AVAILABLE", True)
    monkeypatch.setattr(attention_tokamax, "_WARNED_FLOAT32", False)
    monkeypatch.setattr(
        attention_tokamax, "tokamax", SimpleNamespace(dot_product_attention=ref_dpa)
    )
    out_xla = np.asarray(
        jax.jit(lambda b, x, z: denoise(b, x, t, si, si_t, z, params, **kw))(
            bp_m, xl, z1
        )
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out_tok = np.asarray(
            jax.jit(
                lambda b, x, z: denoise(
                    b, x, t, si, si_t, z, params, **kw, attention_backend="tokamax"
                )
            )(bp_m, xl, z1)
        )
    assert not any(issubclass(w.category, UserWarning) for w in caught)
    assert len(calls) == 1 and np.isfinite(out_tok).all()
    np.testing.assert_allclose(out_tok[:2], out_xla[:2], rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(out_tok[:, 75:], 0.0)
    apb = lambda bk: np.asarray(  # noqa: E731
        apb_mod.ada_attention_pair_bias(
            si, si, z1, apb_p, no_heads=h, attention_backend=bk
        )
    )
    np.testing.assert_allclose(apb("tokamax"), apb("xla"), rtol=1e-5, atol=1e-5)

    # Env var propagates from _predict_from_trunk; unavailable & CP behave as required.
    seen: list[str] = []
    monkeypatch.setenv(env, "tokamax")
    with monkeypatch.context() as mp:
        def spy_denoise(*a, attention_backend="xla", **k):
            seen.append(attention_backend)
            return a[1]

        mp.setattr(inference, "denoise", spy_denoise)
        for name, fn in (
            ("prepare_atom_diffusion_cache", lambda *a, **k: None),
            ("pair_conditioning", lambda *a, **k: z1),
            ("single_conditioning", lambda *a, **k: si[:1]),
            ("sample_diffusion", lambda k, s, sh, f, **kw: f(xl[:1], t)),
            ("distogram_head", lambda *a, **k: (_ for _ in ()).throw(StopIteration)),
        ):
            mp.setattr(inference, name, fn)
        cfg = inference.released_config(
            **{"n_" + "token": kw["n_token"]},
            n_atom=xl.shape[1],
            num_samples=1,
            num_steps=1,
        )
        ip = inference.InferenceParams(
            None, None, params, None, None, None, None, None, None
        )
        def trace() -> None:
            with pytest.raises(StopIteration):
                inference._predict_from_trunk(
                    jax.random.key(0), {k: v[:1] for k, v in bw.items()}, ip, cfg,
                    None, trunk_output=(si[:1], si_t[:1], z1),
                )

        trace()
        # A pooled or streamed trace uses the value its key was built from,
        # whatever the variable says by then.
        with inference.diffusion_attention_backend("xla"):
            trace()
        # A direct call under a mesh resolves as the pooled paths do.
        mp.setenv(env, "triton")
        mp.setattr(inference, "cp_mesh", lambda: object())
        trace()
    assert seen == ["tokamax", "xla", "xla"]
    with pytest.raises(ValueError, match="unsupported"):
        with inference.diffusion_attention_backend("triton"):
            pass

    monkeypatch.setattr(attention_tokamax, "_TOKAMAX_AVAILABLE", False)
    monkeypatch.setattr(attention_tokamax, "_IMPORT_ERROR", ImportError("missing"))
    with pytest.raises(RuntimeError, match="needs the tokamax package"):
        denoise(bp1, xl, t, si, si_t, z1, params, **kw, attention_backend="tokamax")

    for mod in (denoiser, apb_mod, inference):
        monkeypatch.setattr(mod, "cp_mesh", lambda: object())
    monkeypatch.setattr(
        attention_tokamax, "tokamax_attention", lambda *a, **k: pytest.fail("CP")
    )
    assert np.isfinite(
        np.asarray(
            denoise(bp1, xl, t, si, si_t, z1, params, **kw, attention_backend="tokamax")
        )
    ).all()
