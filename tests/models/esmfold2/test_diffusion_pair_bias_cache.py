"""Token-transformer pair biases computed once match the per-step path on CPU.

Every denoiser step in `diffusion.sample` previously re-ran each token
transformer block's `pair_norm` and `pair_bias_proj` on the same loop-invariant
`condition_pair` output. `with_token_pair_biases` runs those per-block
projections once per prediction on the `build_cache` result, places an
`optimization_barrier` on the `condition_pair` output to keep the LayerNorm
fusion boundary identical, and drops the full `[B, L, L, c_z]` pair tensor
before sampling starts. GPU runs may round differently at larger sizes.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.esmfold2.models import diffusion

D_TOKEN = 16
D_PAIR = 8
N_HEADS = 2
C_ATOM = 64
C_INPUTS = 12
N_TOKENS = 5
N_ATOMS = 15
FOURIER_DIM = 8


def _tiny_settings(*, token_n_blocks: int = 2, num_steps: int = 2):
    return diffusion.DiffusionSettings(
        c_atom=C_ATOM,
        c_token=D_TOKEN,
        c_z=D_PAIR,
        atom_n_blocks=1,
        atom_n_heads=2,
        token_n_blocks=token_n_blocks,
        token_n_heads=N_HEADS,
        half_window=4,
        num_steps=num_steps,
        noise_scale=1.003,
    )


def _tiny_params(
    seed: int = 0,
    *,
    token_n_blocks: int = 2,
    prefix: str = "",
) -> dict[str, jnp.ndarray]:
    rng = np.random.default_rng(seed)
    dot = f"{prefix}." if prefix else ""

    def arr(*shape: int, scale: float = 0.25) -> jnp.ndarray:
        return jnp.asarray(rng.normal(size=shape, scale=scale), jnp.float32)

    params: dict[str, jnp.ndarray] = {}

    def add_adaln(p: str, width: int) -> None:
        params[f"{p}.s_scale"] = arr(width, scale=0.1) + 1.0
        params[f"{p}.s_gate.weight"] = arr(width, width)
        params[f"{p}.s_gate.bias"] = arr(width, scale=0.1)
        params[f"{p}.s_shift.weight"] = arr(width, width)

    def add_transition(p: str, width: int) -> None:
        params[f"{p}.norm.weight"] = arr(width, scale=0.1) + 1.0
        params[f"{p}.norm.bias"] = arr(width, scale=0.1)
        params[f"{p}.a_proj.weight"] = arr(2 * width, width)
        params[f"{p}.b_proj.weight"] = arr(2 * width, width)
        params[f"{p}.out_proj.weight"] = arr(width, 2 * width)

    def add_atom_transformer(p: str) -> None:
        block = f"{p}.blocks.0"
        params[f"{block}.adaln_modulation.1.weight"] = arr(6 * C_ATOM, C_ATOM)
        params[f"{block}.adaln_modulation.1.bias"] = arr(6 * C_ATOM, scale=0.1)
        params[f"{block}.attn.Wqkv.weight"] = arr(3 * C_ATOM, C_ATOM)
        params[f"{block}.attn.gate_proj.weight"] = arr(C_ATOM, C_ATOM)
        params[f"{block}.attn.out_proj.weight"] = arr(C_ATOM, C_ATOM)
        params[f"{block}.ffn.w_up.weight"] = arr(4 * C_ATOM, C_ATOM)
        params[f"{block}.ffn.w_down.weight"] = arr(C_ATOM, 2 * C_ATOM)

    # Conditioning (pair and single).
    cond = f"{dot}conditioning"
    params[f"{cond}.z_input_norm.weight"] = arr(2 * D_PAIR, scale=0.1) + 1.0
    params[f"{cond}.z_input_norm.bias"] = arr(2 * D_PAIR, scale=0.1)
    params[f"{cond}.z_proj.weight"] = arr(D_PAIR, 2 * D_PAIR)
    for index in range(2):
        add_transition(f"{cond}.z_transitions.{index}", D_PAIR)

    params[f"{cond}.s_input_norm.weight"] = arr(C_INPUTS, scale=0.1) + 1.0
    params[f"{cond}.s_input_norm.bias"] = arr(C_INPUTS, scale=0.1)
    params[f"{cond}.s_proj.weight"] = arr(D_TOKEN, C_INPUTS)
    params[f"{cond}.fourier.w"] = arr(FOURIER_DIM)
    params[f"{cond}.fourier.b"] = arr(FOURIER_DIM)
    params[f"{cond}.noise_norm.weight"] = arr(FOURIER_DIM, scale=0.1) + 1.0
    params[f"{cond}.noise_norm.bias"] = arr(FOURIER_DIM, scale=0.1)
    params[f"{cond}.noise_proj.weight"] = arr(D_TOKEN, FOURIER_DIM)
    for index in range(2):
        add_transition(f"{cond}.s_transitions.{index}", D_TOKEN)

    # Atom encoder & decoder.
    enc = f"{dot}atom_encoder"
    params[f"{enc}.atom_linear.weight"] = arr(C_ATOM, 389)
    params[f"{enc}.atom_norm.weight"] = arr(C_ATOM, scale=0.1) + 1.0
    params[f"{enc}.atom_norm.bias"] = arr(C_ATOM, scale=0.1)
    params[f"{enc}.coords_linear.weight"] = arr(C_ATOM, 6)
    params[f"{enc}.atom_to_token_linear.weight"] = arr(D_TOKEN, C_ATOM)
    add_atom_transformer(f"{enc}.atom_transformer")

    dec = f"{dot}atom_decoder"
    params[f"{dec}.token_to_atom_linear.weight"] = arr(C_ATOM, D_TOKEN)
    add_atom_transformer(f"{dec}.atom_transformer")
    params[f"{dec}.norm.weight"] = arr(C_ATOM, scale=0.1) + 1.0
    params[f"{dec}.norm.bias"] = arr(C_ATOM, scale=0.1)
    params[f"{dec}.output_linear.weight"] = arr(3, C_ATOM)

    # Step & output token norms.
    params[f"{dot}s_step_norm.weight"] = arr(D_TOKEN, scale=0.1) + 1.0
    params[f"{dot}s_step_norm.bias"] = arr(D_TOKEN, scale=0.1)
    params[f"{dot}s_to_token.weight"] = arr(D_TOKEN, D_TOKEN)
    params[f"{dot}token_norm.weight"] = arr(D_TOKEN, scale=0.1) + 1.0
    params[f"{dot}token_norm.bias"] = arr(D_TOKEN, scale=0.1)

    # Token transformer blocks.
    tt = f"{dot}token_transformer"
    for index in range(token_n_blocks):
        attn = f"{tt}.attn_blocks.{index}"
        add_adaln(f"{attn}.adaln", D_TOKEN)
        params[f"{attn}.q_proj.weight"] = arr(D_TOKEN, D_TOKEN)
        params[f"{attn}.q_proj.bias"] = arr(D_TOKEN, scale=0.1)
        params[f"{attn}.kv_proj.weight"] = arr(2 * D_TOKEN, D_TOKEN)
        params[f"{attn}.pair_norm.weight"] = arr(D_PAIR, scale=0.1) + 1.0
        params[f"{attn}.pair_norm.bias"] = arr(D_PAIR, scale=0.1)
        params[f"{attn}.pair_bias_proj.weight"] = arr(N_HEADS, D_PAIR)
        params[f"{attn}.g_proj.weight"] = arr(D_TOKEN, D_TOKEN)
        params[f"{attn}.out_proj.weight"] = arr(D_TOKEN, D_TOKEN)
        params[f"{attn}.out_gate.weight"] = arr(D_TOKEN, D_TOKEN)
        params[f"{attn}.out_gate.bias"] = arr(D_TOKEN, scale=0.1)

        trans = f"{tt}.transition_blocks.{index}"
        add_adaln(f"{trans}.adaln", D_TOKEN)
        params[f"{trans}.lin_swish.weight"] = arr(4 * D_TOKEN, D_TOKEN)
        params[f"{trans}.lin_out.weight"] = arr(D_TOKEN, 2 * D_TOKEN)
        params[f"{trans}.output_gate.weight"] = arr(D_TOKEN, D_TOKEN)
        params[f"{trans}.output_gate.bias"] = arr(D_TOKEN, scale=0.1)

    return params


def _tiny_inputs(seed: int = 10) -> dict[str, jnp.ndarray]:
    rng = np.random.default_rng(seed)
    element = np.zeros((1, N_ATOMS, 128), dtype=np.float32)
    element[..., 6] = 1.0
    chars = np.zeros((1, N_ATOMS, 4, 64), dtype=np.float32)
    chars[..., 0] = 1.0
    tok_idx = (np.arange(N_ATOMS, dtype=np.int32) // 3)[None]
    return {
        "ref_pos": jnp.asarray(rng.normal(size=(1, N_ATOMS, 3)), jnp.float32),
        "ref_charge": jnp.zeros((1, N_ATOMS), jnp.float32),
        "ref_mask": jnp.ones((1, N_ATOMS), jnp.float32),
        "ref_element": jnp.asarray(element),
        "ref_atom_name_chars": jnp.asarray(chars),
        "ref_space_uid": jnp.asarray(tok_idx),
        "atom_to_token": jnp.asarray(tok_idx),
        "s_inputs": jnp.asarray(rng.normal(size=(1, N_TOKENS, C_INPUTS)), jnp.float32),
        "z_trunk": jnp.asarray(
            rng.normal(size=(1, N_TOKENS, N_TOKENS, D_PAIR)), jnp.float32
        ),
        "rel_pos": jnp.asarray(
            rng.normal(size=(1, N_TOKENS, N_TOKENS, D_PAIR)), jnp.float32
        ),
        "token_mask": jnp.asarray([[1.0, 1.0, 1.0, 1.0, 0.0]], jnp.float32),
    }


def _cache_with_biases(*args, settings, **kwargs):
    cache = diffusion.build_cache(*args, settings=settings, **kwargs)
    prefix = args[10] if len(args) > 10 else ""
    return diffusion.with_token_pair_biases(cache, args[9], prefix, settings=settings)


def _build_cached(inputs, params, settings, *, num_samples: int = 1, prefix: str = ""):
    return _cache_with_biases(
        inputs["ref_pos"],
        inputs["ref_charge"],
        inputs["ref_mask"],
        inputs["ref_element"],
        inputs["ref_atom_name_chars"],
        inputs["ref_space_uid"],
        inputs["atom_to_token"],
        inputs["z_trunk"],
        inputs["rel_pos"],
        params,
        prefix,
        settings=settings,
        num_samples=num_samples,
        n_tokens=N_TOKENS,
    )


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("use_mask", [False, True])
@pytest.mark.parametrize("num_samples", [1, 2])
def test_cached_pair_bias_matches_z_path_bitwise_on_cpu_diffusion_module(
    compiled: bool, use_mask: bool, num_samples: int
) -> None:
    """On CPU, cached `pair_bias` matches the `z` path bit-for-bit (GPU may round)."""
    settings = _tiny_settings(token_n_blocks=2)
    params = _tiny_params(seed=3, token_n_blocks=2, prefix="dm")
    inputs = _tiny_inputs(seed=4)
    rng = np.random.default_rng(5)
    x_noisy = jnp.asarray(
        rng.normal(size=(num_samples, N_ATOMS, 3), scale=4.0), jnp.float32
    )
    t_hat = jnp.asarray([12.0] * num_samples, jnp.float32)
    token_mask = inputs["token_mask"] if use_mask else None

    cached = _build_cached(
        inputs, params, settings, num_samples=num_samples, prefix="dm"
    )
    assert cached.pair is None
    assert cached.pair_bias is not None
    assert len(cached.pair_bias) == settings.token_n_blocks
    for entry in cached.pair_bias:
        assert entry.shape == (1, N_TOKENS, N_TOKENS, N_HEADS)

    # Strip pair_norm/pair_bias_proj from params on the cached denoiser arm to
    # prove the denoiser never recomputes them when `cache.pair_bias` is set.
    stripped_params = {
        k: v
        for k, v in params.items()
        if ".pair_norm." not in k and ".pair_bias_proj." not in k
    }

    def run_cached(x, t, z_trunk, rel_pos, mask):
        cache = _cache_with_biases(
            inputs["ref_pos"],
            inputs["ref_charge"],
            inputs["ref_mask"],
            inputs["ref_element"],
            inputs["ref_atom_name_chars"],
            inputs["ref_space_uid"],
            inputs["atom_to_token"],
            z_trunk,
            rel_pos,
            params,
            "dm",
            settings=settings,
            num_samples=num_samples,
            n_tokens=N_TOKENS,
        )
        return diffusion.diffusion_module(
            x,
            t,
            inputs["s_inputs"],
            cache,
            stripped_params,
            "dm",
            settings=settings,
            token_mask=mask,
            num_samples=num_samples,
        )

    def run_legacy(x, t, z_trunk, rel_pos, mask):
        cache = _cache_with_biases(
            inputs["ref_pos"],
            inputs["ref_charge"],
            inputs["ref_mask"],
            inputs["ref_element"],
            inputs["ref_atom_name_chars"],
            inputs["ref_space_uid"],
            inputs["atom_to_token"],
            z_trunk,
            rel_pos,
            params,
            "dm",
            settings=settings,
            num_samples=num_samples,
            n_tokens=N_TOKENS,
        )
        conditioned_pair = jax.lax.optimization_barrier(
            diffusion.condition_pair(z_trunk, rel_pos, params, "dm.conditioning")
        )
        legacy = dataclasses.replace(cache, pair=conditioned_pair, pair_bias=None)
        return diffusion.diffusion_module(
            x,
            t,
            inputs["s_inputs"],
            legacy,
            params,
            "dm",
            settings=settings,
            token_mask=mask,
            num_samples=num_samples,
        )

    def run_perturbed(x, t, z_trunk, rel_pos, mask):
        cache = _cache_with_biases(
            inputs["ref_pos"],
            inputs["ref_charge"],
            inputs["ref_mask"],
            inputs["ref_element"],
            inputs["ref_atom_name_chars"],
            inputs["ref_space_uid"],
            inputs["atom_to_token"],
            z_trunk,
            rel_pos,
            params,
            "dm",
            settings=settings,
            num_samples=num_samples,
            n_tokens=N_TOKENS,
        )
        perturbed = dataclasses.replace(
            cache, pair_bias=tuple(b * 2.0 + 1.0 for b in cache.pair_bias)
        )
        return diffusion.diffusion_module(
            x,
            t,
            inputs["s_inputs"],
            perturbed,
            stripped_params,
            "dm",
            settings=settings,
            token_mask=mask,
            num_samples=num_samples,
        )

    wrap = jax.jit if compiled else (lambda f: f)
    call_args = (x_noisy, t_hat, inputs["z_trunk"], inputs["rel_pos"], token_mask)
    got_x, got_tok = wrap(run_cached)(*call_args)
    ref_x, ref_tok = wrap(run_legacy)(*call_args)

    assert float(np.abs(np.asarray(ref_x)).max()) > 0.0
    assert float(np.abs(np.asarray(got_x) - np.asarray(ref_x)).max()) == 0.0
    assert float(np.abs(np.asarray(got_tok) - np.asarray(ref_tok)).max()) == 0.0
    np.testing.assert_array_equal(np.asarray(got_x), np.asarray(ref_x))
    np.testing.assert_array_equal(np.asarray(got_tok), np.asarray(ref_tok))

    # Tripwire: perturbing the cached bias changes the denoiser output.
    moved_x, _ = wrap(run_perturbed)(*call_args)
    assert not np.array_equal(np.asarray(got_x), np.asarray(moved_x))


@pytest.mark.parametrize("sample_sequential", [False, True])
def test_jitted_cpu_sampler_matches_z_path_bitwise_and_drops_full_pair(
    sample_sequential: bool,
) -> None:
    """On CPU inside one JIT the biased cache + `sample` match the z-path bitwise."""
    settings = _tiny_settings(token_n_blocks=2, num_steps=2)
    params = _tiny_params(seed=7, token_n_blocks=2)
    inputs = _tiny_inputs(seed=8)
    num_samples = 2
    cache_samples = 1 if sample_sequential else num_samples

    def run_cached(key, z_trunk, rel_pos):
        cache = _cache_with_biases(
            inputs["ref_pos"],
            inputs["ref_charge"],
            inputs["ref_mask"],
            inputs["ref_element"],
            inputs["ref_atom_name_chars"],
            inputs["ref_space_uid"],
            inputs["atom_to_token"],
            z_trunk,
            rel_pos,
            params,
            settings=settings,
            num_samples=cache_samples,
            n_tokens=N_TOKENS,
        )
        return diffusion.sample(
            key,
            inputs["s_inputs"],
            cache,
            params,
            settings=settings,
            token_mask=inputs["token_mask"],
            num_samples=num_samples,
            sample_sequential=sample_sequential,
        )

    def run_legacy(key, z_trunk, rel_pos):
        cache = _cache_with_biases(
            inputs["ref_pos"],
            inputs["ref_charge"],
            inputs["ref_mask"],
            inputs["ref_element"],
            inputs["ref_atom_name_chars"],
            inputs["ref_space_uid"],
            inputs["atom_to_token"],
            z_trunk,
            rel_pos,
            params,
            settings=settings,
            num_samples=cache_samples,
            n_tokens=N_TOKENS,
        )
        conditioned = diffusion.condition_pair(z_trunk, rel_pos, params, "conditioning")
        legacy_cache = dataclasses.replace(cache, pair=conditioned, pair_bias=None)
        return diffusion.sample(
            key,
            inputs["s_inputs"],
            legacy_cache,
            params,
            settings=settings,
            token_mask=inputs["token_mask"],
            num_samples=num_samples,
            sample_sequential=sample_sequential,
        )

    key = jax.random.key(99)
    jax.clear_caches()
    got_x, got_tok = jax.jit(run_cached)(key, inputs["z_trunk"], inputs["rel_pos"])
    jax.clear_caches()
    ref_x, ref_tok = jax.jit(run_legacy)(key, inputs["z_trunk"], inputs["rel_pos"])

    assert float(np.abs(np.asarray(ref_x)).max()) > 0.0
    np.testing.assert_array_equal(np.asarray(got_x), np.asarray(ref_x))
    np.testing.assert_array_equal(np.asarray(got_tok), np.asarray(ref_tok))

    # Lowered program checks: optimization_barrier precedes the block LNs, and
    # the sampler scan `while` carries only `[1, L, L, H]` biases, not `[1, L, L, c_z]`.
    lowered = (
        jax.jit(run_cached).lower(key, inputs["z_trunk"], inputs["rel_pos"]).as_text()
    )
    assert "optimization_barrier" in lowered
    full_pair_type = f"tensor<1x{N_TOKENS}x{N_TOKENS}x{D_PAIR}xf32>"
    bias_type = f"tensor<1x{N_TOKENS}x{N_TOKENS}x{N_HEADS}xf32>"
    while_lines = [line for line in lowered.splitlines() if "stablehlo.while" in line]
    assert while_lines
    for line in while_lines:
        assert full_pair_type not in line, line
    assert any(bias_type in line for line in while_lines), while_lines


def test_token_pair_biases_and_attention_edge_cases_and_errors() -> None:
    """Zero/empty, boundary, malformed inputs, and partial-param compatibility."""
    settings = _tiny_settings(token_n_blocks=2)
    params = _tiny_params(seed=11, token_n_blocks=2)
    inputs = _tiny_inputs(seed=12)
    pair = jnp.ones((1, N_TOKENS, N_TOKENS, D_PAIR), jnp.float32)

    # Zero blocks returns empty tuple and diffusion_transformer is an identity.
    assert diffusion.token_pair_biases(pair, {}, n_blocks=0) == ()
    a = jnp.ones((1, N_TOKENS, D_TOKEN), jnp.float32)
    np.testing.assert_array_equal(
        diffusion.diffusion_transformer(
            a, None, None, {}, n_blocks=0, n_heads=N_HEADS, pair_bias=()
        ),
        a,
    )

    # A negative block count keeps the historical identity; a malformed pair
    # rank fails explicitly.
    assert (
        diffusion.token_pair_biases(pair, params, "token_transformer", n_blocks=-1)
        == ()
    )
    np.testing.assert_array_equal(
        diffusion.diffusion_transformer(
            a, None, pair, params, "token_transformer", n_blocks=-1, n_heads=N_HEADS
        ),
        a,
    )
    with pytest.raises(ValueError, match="rank-4"):
        diffusion.token_pair_biases(
            jnp.ones((N_TOKENS, N_TOKENS, D_PAIR), jnp.float32),
            params,
            "token_transformer",
            n_blocks=1,
        )

    # Length mismatch on pair_bias (empty when n_blocks > 0, too short, too long).
    valid_biases = diffusion.token_pair_biases(
        pair, params, "token_transformer", n_blocks=2
    )
    for bad in ((), valid_biases[:1], valid_biases + valid_biases[:1]):
        with pytest.raises(ValueError, match="pair_bias has"):
            diffusion.diffusion_transformer(
                a,
                None,
                None,
                params,
                "token_transformer",
                n_blocks=2,
                n_heads=N_HEADS,
                pair_bias=bad,
            )

    # attention_pair_bias: missing both z and pair_bias.
    block_prefix = "token_transformer.attn_blocks.0"
    s = jnp.zeros((1, N_TOKENS, D_TOKEN), jnp.float32)
    with pytest.raises(ValueError, match="either z or pair_bias"):
        diffusion.attention_pair_bias(a, s, None, params, block_prefix, n_heads=N_HEADS)
    # A one-channel cached bias broadcasts over the heads, as the z path does.
    single_channel = jnp.full((1, N_TOKENS, N_TOKENS, 1), 0.5, jnp.float32)
    np.testing.assert_array_equal(
        diffusion.attention_pair_bias(
            a,
            s,
            None,
            params,
            block_prefix,
            n_heads=N_HEADS,
            pair_bias=single_channel,
        ),
        diffusion.attention_pair_bias(
            a,
            s,
            None,
            params,
            block_prefix,
            n_heads=N_HEADS,
            pair_bias=jnp.broadcast_to(
                single_channel, (1, N_TOKENS, N_TOKENS, N_HEADS)
            ),
        ),
    )

    # build_cache with partial token_transformer params (block 1 missing) keeps
    # the per-step path, so the missing key fails where it always did.
    partial_params = {
        k: v
        for k, v in params.items()
        if not k.startswith("token_transformer.attn_blocks.1.")
    }
    partial_cache = _build_cached(inputs, partial_params, settings)
    assert partial_cache.pair_bias is None
    assert partial_cache.pair is not None

    # build_cache alone keeps the reusable pair form, so a caller that reuses it
    # with other projection weights or fewer blocks reads them as before.
    plain = diffusion.build_cache(
        inputs["ref_pos"],
        inputs["ref_charge"],
        inputs["ref_mask"],
        inputs["ref_element"],
        inputs["ref_atom_name_chars"],
        inputs["ref_space_uid"],
        inputs["atom_to_token"],
        inputs["z_trunk"],
        inputs["rel_pos"],
        params,
        settings=settings,
        n_tokens=N_TOKENS,
    )
    assert plain.pair_bias is None and plain.pair is not None
    biased = diffusion.with_token_pair_biases(plain, params, settings=settings)
    assert biased.pair is None and len(biased.pair_bias) == 2
    assert diffusion.with_token_pair_biases(biased, params, settings=settings) is biased

    # Five blocks x two heads outnumber the eight pair channels: keep the pair.
    wide_settings = _tiny_settings(token_n_blocks=5)
    wide_params = _tiny_params(seed=13, token_n_blocks=5)
    wide = diffusion.with_token_pair_biases(plain, wide_params, settings=wide_settings)
    assert wide is plain

    # The guard counts bytes: float64 projections of a float32 pair retain
    # 2 x 2 x 8 bytes per pair entry against 8 x 4, so they still fit, while
    # four float64 blocks (64 bytes) would not.
    f64_params = {
        k: (v.astype(jnp.float64) if "pair_bias_proj" in k else v)
        for k, v in params.items()
    }
    if jnp.zeros((), jnp.float64).dtype == jnp.float64:
        wide64 = _tiny_params(seed=13, token_n_blocks=4)
        wide64 = {
            k: (v.astype(jnp.float64) if "pair_bias_proj" in k else v)
            for k, v in wide64.items()
        }
        settings4 = _tiny_settings(token_n_blocks=4)
        assert (
            diffusion.with_token_pair_biases(plain, wide64, settings=settings4)
            is plain
        )
        assert (
            diffusion.with_token_pair_biases(plain, f64_params, settings=settings).pair
            is None
        )

    # build_cache with zero token_transformer params preserves legacy fixture
    # compatibility (pair kept, pair_bias is None).
    no_tt_params = {
        k: v for k, v in params.items() if not k.startswith("token_transformer.")
    }
    legacy_cache = _build_cached(inputs, no_tt_params, settings)
    assert legacy_cache.pair_bias is None
    assert legacy_cache.pair is not None
    assert legacy_cache.pair.shape == (1, N_TOKENS, N_TOKENS, D_PAIR)
