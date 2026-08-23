from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

import foldjax.models.protenix.models.diffusion.diffusion as diffusion_module_impl
from foldjax.models.protenix.bridge.torch_mapping import (
    map_diffusion_conditioning_state_dict,
)
from foldjax.models.protenix.models.diffusion.atom import (
    AtomAttentionDecoderParams,
    AtomAttentionEncoderCacheParams,
    AtomAttentionEncoderParams,
    AtomPairMlpParams,
)
from foldjax.models.protenix.models.diffusion.diffusion import (
    DiffusionModuleParams,
    diffusion_conditioning,
    diffusion_conditioning_prepare_cache,
    diffusion_module_f_forward,
    diffusion_module_forward,
)
from foldjax.models.protenix.models.diffusion.transformer import (
    ConditionedTransitionParams,
    DiffusionTransformerBlockParams,
    DiffusionTransformerStackParams,
)
from foldjax.models.protenix.models.primitives.attention import (
    AttentionPairBiasParams,
    AttentionParams,
)
from foldjax.models.protenix.models.primitives.primitives import (
    AdaptiveLayerNormParams,
    LayerNormParams,
    LinearParams,
    layer_norm,
    linear,
)
from foldjax.models.protenix.models.trunk_blocks.embedders import (
    fourier_embedding,
    relative_position_encoding,
)


def test_diffusion_conditioning_matches_reference_formula() -> None:
    state = _diffusion_conditioning_state(c_z=2, c_s=3, c_s_inputs=2, c_noise=4)
    params = map_diffusion_conditioning_state_dict(state, "dc")
    relp = jnp.asarray(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.5, 0.5], [1.0, 1.0]],
        ],
        dtype=jnp.float32,
    )
    s_inputs = jnp.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32)
    s_trunk = jnp.asarray([[0.5, 1.5, 2.5], [3.5, 4.5, 5.5]], dtype=jnp.float32)
    z_trunk = jnp.ones((2, 2, 2), dtype=jnp.float32)
    noise = jnp.asarray([2.0, 8.0], dtype=jnp.float32)

    single_s, pair_z = diffusion_conditioning(
        noise,
        relp,
        s_inputs,
        s_trunk,
        z_trunk,
        params,
        sigma_data=4.0,
    )

    expected_pair = linear(
        layer_norm(
            jnp.concatenate(
                [z_trunk, relative_position_encoding(relp, params.relpe)],
                -1,
            ),
            params.layernorm_z,
        ),
        params.linear_z,
    )
    base_s = linear(
        layer_norm(jnp.concatenate([s_trunk, s_inputs], axis=-1), params.layernorm_s),
        params.linear_s,
    )
    noise_embedding = fourier_embedding(jnp.log(noise / 4.0) / 4.0, params.fourier)
    noise_embedding = linear(
        layer_norm(noise_embedding, params.layernorm_n),
        params.linear_n,
    )
    expected_s = base_s[None, :, :] + noise_embedding[:, None, :]

    np.testing.assert_allclose(np.asarray(pair_z), np.asarray(expected_pair), atol=1e-5)
    np.testing.assert_allclose(np.asarray(single_s), np.asarray(expected_s), atol=1e-5)


def test_diffusion_conditioning_prepare_cache_can_be_reused() -> None:
    state = _diffusion_conditioning_state(c_z=2, c_s=3, c_s_inputs=2, c_noise=4)
    params = map_diffusion_conditioning_state_dict(state, "dc")
    relp = jnp.ones((2, 2, 2), dtype=jnp.float32)
    z_trunk = jnp.ones((2, 2, 2), dtype=jnp.float32)
    cache = diffusion_conditioning_prepare_cache(relp, z_trunk, params)
    s_inputs = jnp.ones((2, 2), dtype=jnp.float32)
    s_trunk = jnp.ones((2, 3), dtype=jnp.float32)

    _, pair_z = diffusion_conditioning(
        jnp.asarray([4.0], dtype=jnp.float32),
        relp,
        s_inputs,
        s_trunk,
        z_trunk,
        params,
        pair_z=cache,
        sigma_data=4.0,
    )

    np.testing.assert_allclose(np.asarray(pair_z), np.asarray(cache), atol=1e-6)


def test_map_diffusion_conditioning_state_dict_uses_protenix_keys() -> None:
    state = _diffusion_conditioning_state(c_z=2, c_s=3, c_s_inputs=2, c_noise=4)

    params = map_diffusion_conditioning_state_dict(state, "dc")

    assert tuple(params.relpe.linear_no_bias.weight.shape) == (2, 2)
    assert tuple(params.layernorm_z.weight.shape) == (4,)
    assert params.layernorm_z.bias is None
    assert tuple(params.linear_z.weight.shape) == (2, 4)
    assert tuple(params.linear_s.weight.shape) == (3, 5)
    assert tuple(params.fourier.w.shape) == (4,)
    assert tuple(params.linear_n.weight.shape) == (3, 4)
    assert tuple(params.transition_s1.linear_out.weight.shape) == (3, 6)


def test_diffusion_module_f_forward_cache_path_smoke(monkeypatch) -> None:
    params = _zero_diffusion_module_params()
    atom_to_token_idx = jnp.asarray([0, 1, 1])
    ref_pos = jnp.zeros((3, 3), dtype=jnp.float32)
    ref_charge = jnp.zeros((3,), dtype=jnp.float32)
    ref_mask = jnp.ones((3,), dtype=jnp.float32)
    ref_atom_name_chars = jnp.zeros((3, 4, 64), dtype=jnp.float32)
    ref_element = jnp.zeros((3, 128), dtype=jnp.float32)
    d_lm = jnp.zeros((2, 2, 4, 3), dtype=jnp.float32)
    v_lm = jnp.ones((2, 2, 4, 1), dtype=jnp.float32)
    pad_info = {"mask_trunked": jnp.ones((2, 2, 4), dtype=bool)}
    r_noisy = jnp.ones((1, 3, 3), dtype=jnp.float32)
    noise = jnp.asarray([4.0], dtype=jnp.float32)
    relp = jnp.ones((2, 2, 2), dtype=jnp.float32)
    s_inputs = jnp.ones((2, 1), dtype=jnp.float32)
    s_trunk = jnp.ones((2, 2), dtype=jnp.float32)
    z_trunk = jnp.ones((2, 2, 2), dtype=jnp.float32)
    p_lm = jnp.zeros((1, 2, 2, 4, 2), dtype=jnp.float32)
    c_l = jnp.zeros((3, 2), dtype=jnp.float32)
    structural_bias = jnp.asarray([[0.0, 0.5], [-0.2, 0.0]], dtype=jnp.float32)
    captured_bias = []
    original_transformer = diffusion_module_impl.diffusion_transformer_stack

    def recording_transformer(*args, **kwargs):
        captured_bias.append(kwargs.get("extra_attn_bias"))
        return original_transformer(*args, **kwargs)

    monkeypatch.setattr(
        diffusion_module_impl,
        "diffusion_transformer_stack",
        recording_transformer,
    )

    r_update = diffusion_module_f_forward(
        atom_to_token_idx,
        ref_pos,
        ref_charge,
        ref_mask,
        ref_atom_name_chars,
        ref_element,
        d_lm,
        v_lm,
        pad_info,
        r_noisy,
        noise,
        relp,
        s_inputs,
        s_trunk,
        z_trunk,
        params,
        pair_z=None,
        p_lm=p_lm,
        c_l=c_l,
        extra_attn_bias=structural_bias,
        n_token=2,
        atom_encoder_heads=1,
        token_heads=1,
        atom_decoder_heads=1,
        n_queries=2,
        n_keys=4,
        sigma_data=4.0,
    )

    assert r_update.shape == (1, 3, 3)
    np.testing.assert_allclose(captured_bias[0], structural_bias)
    np.testing.assert_allclose(np.asarray(r_update), 0.0, atol=1e-6)


def test_diffusion_module_forward_applies_edm_rescale() -> None:
    params = _zero_diffusion_module_params()
    atom_to_token_idx = jnp.asarray([0, 1, 1])
    x_noisy = jnp.ones((1, 3, 3), dtype=jnp.float32) * 10.0
    noise = jnp.asarray([4.0], dtype=jnp.float32)
    zeros_atom = jnp.zeros((3, 3), dtype=jnp.float32)
    ref_charge = jnp.zeros((3,), dtype=jnp.float32)
    ref_mask = jnp.ones((3,), dtype=jnp.float32)
    ref_atom_name_chars = jnp.zeros((3, 4, 64), dtype=jnp.float32)
    ref_element = jnp.zeros((3, 128), dtype=jnp.float32)
    d_lm = jnp.zeros((2, 2, 4, 3), dtype=jnp.float32)
    v_lm = jnp.ones((2, 2, 4, 1), dtype=jnp.float32)
    pad_info = {"mask_trunked": jnp.ones((2, 2, 4), dtype=bool)}
    relp = jnp.ones((2, 2, 2), dtype=jnp.float32)
    s_inputs = jnp.ones((2, 1), dtype=jnp.float32)
    s_trunk = jnp.ones((2, 2), dtype=jnp.float32)
    z_trunk = jnp.ones((2, 2, 2), dtype=jnp.float32)

    x_denoised = diffusion_module_forward(
        atom_to_token_idx,
        zeros_atom,
        ref_charge,
        ref_mask,
        ref_atom_name_chars,
        ref_element,
        d_lm,
        v_lm,
        pad_info,
        x_noisy,
        noise,
        relp,
        s_inputs,
        s_trunk,
        z_trunk,
        params,
        pair_z=None,
        p_lm=jnp.zeros((1, 2, 2, 4, 2), dtype=jnp.float32),
        c_l=jnp.zeros((3, 2), dtype=jnp.float32),
        n_token=2,
        atom_encoder_heads=1,
        token_heads=1,
        atom_decoder_heads=1,
        n_queries=2,
        n_keys=4,
        sigma_data=4.0,
    )

    np.testing.assert_allclose(np.asarray(x_denoised), 5.0, atol=1e-5)


def _diffusion_conditioning_state(
    *,
    c_z: int,
    c_s: int,
    c_s_inputs: int,
    c_noise: int,
) -> dict[str, np.ndarray]:
    state = {
        "dc.relpe.linear_no_bias.weight": np.arange(c_z * 2, dtype=np.float32).reshape(
            c_z, 2
        )
        / 10.0,
        "dc.layernorm_z.weight": np.ones((2 * c_z,), dtype=np.float32),
        "dc.linear_no_bias_z.weight": (
            np.arange(c_z * 2 * c_z, dtype=np.float32).reshape(c_z, 2 * c_z)
            / 10.0
        ),
        "dc.layernorm_s.weight": np.ones((c_s + c_s_inputs,), dtype=np.float32),
        "dc.linear_no_bias_s.weight": np.arange(
            c_s * (c_s + c_s_inputs), dtype=np.float32
        ).reshape(c_s, c_s + c_s_inputs)
        / 10.0,
        "dc.fourier_embedding.w": np.linspace(0.1, 0.4, c_noise, dtype=np.float32),
        "dc.fourier_embedding.b": np.linspace(0.2, 0.5, c_noise, dtype=np.float32),
        "dc.layernorm_n.weight": np.ones((c_noise,), dtype=np.float32),
        "dc.linear_no_bias_n.weight": (
            np.arange(c_s * c_noise, dtype=np.float32).reshape(c_s, c_noise)
            / 10.0
        ),
    }
    for name, c_in in (
        ("transition_z1", c_z),
        ("transition_z2", c_z),
        ("transition_s1", c_s),
        ("transition_s2", c_s),
    ):
        _add_zero_transition_state(state, f"dc.{name}", c_in)
    return state


def _add_zero_transition_state(
    state: dict[str, np.ndarray],
    prefix: str,
    c_in: int,
) -> None:
    hidden = 2 * c_in
    state[f"{prefix}.layernorm1.weight"] = np.ones((c_in,), dtype=np.float32)
    state[f"{prefix}.layernorm1.bias"] = np.zeros((c_in,), dtype=np.float32)
    state[f"{prefix}.linear_no_bias_a.weight"] = np.zeros(
        (hidden, c_in), dtype=np.float32
    )
    state[f"{prefix}.linear_no_bias_b.weight"] = np.zeros(
        (hidden, c_in), dtype=np.float32
    )
    state[f"{prefix}.linear_no_bias.weight"] = np.zeros(
        (c_in, hidden), dtype=np.float32
    )


def _zero_diffusion_module_params() -> DiffusionModuleParams:
    conditioning = map_diffusion_conditioning_state_dict(
        _diffusion_conditioning_state(c_z=2, c_s=2, c_s_inputs=1, c_noise=4),
        "dc",
    )
    atom_block = _zero_effect_atom_block(c_atom=2, c_atompair=2)
    atom_encoder = AtomAttentionEncoderParams(
        cache=AtomAttentionEncoderCacheParams(
            linear_ref_pos=LinearParams(weight=jnp.zeros((2, 3)), bias=None),
            linear_ref_charge=LinearParams(weight=jnp.zeros((2, 1)), bias=None),
            linear_f=LinearParams(weight=jnp.zeros((2, 385)), bias=None),
            linear_d=LinearParams(weight=jnp.zeros((2, 3)), bias=None),
            linear_invd=LinearParams(weight=jnp.zeros((2, 1)), bias=None),
            linear_v=LinearParams(weight=jnp.zeros((2, 1)), bias=None),
        ),
        linear_cl=LinearParams(weight=jnp.zeros((2, 2)), bias=None),
        linear_cm=LinearParams(weight=jnp.zeros((2, 2)), bias=None),
        small_mlp=AtomPairMlpParams(
            linear_1=LinearParams(weight=jnp.zeros((2, 2)), bias=None),
            linear_2=LinearParams(weight=jnp.zeros((2, 2)), bias=None),
            linear_3=LinearParams(weight=jnp.zeros((2, 2)), bias=None),
        ),
        atom_transformer=DiffusionTransformerStackParams(blocks=(atom_block,)),
        linear_q=LinearParams(weight=jnp.zeros((2, 2)), bias=None),
        layernorm_s=LayerNormParams(weight=jnp.ones((2,)), bias=None),
        linear_s=LinearParams(weight=jnp.zeros((2, 2)), bias=None),
        layernorm_z=LayerNormParams(weight=jnp.ones((2,)), bias=None),
        linear_z=LinearParams(weight=jnp.zeros((2, 2)), bias=None),
        linear_r=LinearParams(weight=jnp.zeros((2, 3)), bias=None),
    )
    atom_decoder = AtomAttentionDecoderParams(
        linear_a=LinearParams(weight=jnp.zeros((2, 2)), bias=None),
        layernorm_q=LayerNormParams(weight=jnp.ones((2,)), bias=None),
        linear_out=LinearParams(weight=jnp.zeros((3, 2)), bias=None),
        atom_transformer=DiffusionTransformerStackParams(blocks=(atom_block,)),
    )
    return DiffusionModuleParams(
        conditioning=conditioning,
        atom_encoder=atom_encoder,
        layernorm_s=LayerNormParams(weight=jnp.ones((2,)), bias=None),
        linear_s=LinearParams(weight=jnp.zeros((2, 2)), bias=None),
        diffusion_transformer=DiffusionTransformerStackParams(blocks=(atom_block,)),
        layernorm_a=LayerNormParams(weight=jnp.ones((2,)), bias=None),
        atom_decoder=atom_decoder,
    )


def _zero_effect_atom_block(
    *,
    c_atom: int,
    c_atompair: int,
) -> DiffusionTransformerBlockParams:
    zero_atom_atom = LinearParams(weight=jnp.zeros((c_atom, c_atom)), bias=None)
    zero_atom_atom_bias = LinearParams(
        weight=jnp.zeros((c_atom, c_atom)),
        bias=jnp.zeros((c_atom,)),
    )
    adaln = AdaptiveLayerNormParams(
        layernorm_a=LayerNormParams(weight=None, bias=None),
        layernorm_s=LayerNormParams(weight=jnp.ones((c_atom,)), bias=None),
        linear_s=zero_atom_atom_bias,
        linear_no_bias_s=zero_atom_atom,
    )
    return DiffusionTransformerBlockParams(
        attention_pair_bias=AttentionPairBiasParams(
            layernorm_a=adaln,
            layernorm_kv=adaln,
            attention=AttentionParams(
                linear_q=zero_atom_atom_bias,
                linear_k=zero_atom_atom,
                linear_v=zero_atom_atom,
                linear_o=zero_atom_atom,
                linear_g=zero_atom_atom,
            ),
            layernorm_z=LayerNormParams(weight=jnp.ones((c_atompair,)), bias=None),
            linear_z=LinearParams(weight=jnp.zeros((1, c_atompair)), bias=None),
            linear_a_last=zero_atom_atom_bias,
            has_s=True,
            cross_attention_mode=True,
        ),
        conditioned_transition=ConditionedTransitionParams(
            adaln=adaln,
            linear_a1=LinearParams(weight=jnp.zeros((2 * c_atom, c_atom)), bias=None),
            linear_a2=LinearParams(weight=jnp.zeros((2 * c_atom, c_atom)), bias=None),
            linear_b=LinearParams(weight=jnp.zeros((c_atom, 2 * c_atom)), bias=None),
            linear_s=zero_atom_atom_bias,
        ),
    )


def test_the_pair_conditioning_cache_stays_float32_on_a_bfloat16_trunk() -> None:
    """Upstream's fp32 island around diffusion, and what actually anchors it here.

    Upstream builds this cache with autocast DISABLED and every floating input
    explicitly upcast: `protenix.py:538-542` wraps
    `diffusion_conditioning.prepare_cache` in
    `autocasting_disable_decorator(configs.skip_amp.sample_diffusion)`,
    `configs_base.py:137` sets that flag True, and `torch_utils.py:199-215`
    both disables autocast and runs `tensor.to(dtype=torch.float32)` on each
    floating argument. So float32 here is a decision, not a consequence.

    This port holds the same invariant by a different and *sufficient*
    mechanism: `cast_trunk_params` narrows only `input_embedder` and
    `pairformer_output`, deliberately "preserving FP32 output-head islands", so
    the diffusion conditioning weights stay float32 and `linear` promotes to
    float32 whatever dtype the activations arrive in. The island is anchored by
    the parameters, not by the storage dtype of any feature -- which is why
    this test passes with both inputs in bfloat16.

    It is a regression test, not a fix: it fails if someone extends
    `cast_trunk_params` to the diffusion parameters, which is the one edit that
    would actually narrow the island. Narrowing `relp` alone would not.

    OpenDDE reaches this same function and inherits the same guarantee.
    """
    state = _diffusion_conditioning_state(c_z=2, c_s=3, c_s_inputs=2, c_noise=4)
    params = map_diffusion_conditioning_state_dict(state, "dc")
    relp = jnp.ones((2, 2, 2), dtype=jnp.bfloat16)
    z_trunk = jnp.ones((2, 2, 2), dtype=jnp.bfloat16)

    cache = diffusion_conditioning_prepare_cache(relp, z_trunk, params)

    assert cache.dtype == jnp.float32


class _ParamsForCastTest(NamedTuple):
    """Minimal stand-in: `cast_trunk_params` rebuilds through `_replace`."""

    input_embedder: jnp.ndarray
    pairformer_output: jnp.ndarray
    diffusion: jnp.ndarray


def test_cast_trunk_params_leaves_the_diffusion_island_alone() -> None:
    """The other half of the invariant above, asserted where it is decided.

    `--trunk-dtype bf16` must not reach the diffusion conditioning weights: they
    are what keeps the pair conditioning cache in float32, matching upstream's
    explicit upcast. Extending this function to the diffusion parameters is the
    edit that would narrow the island, and it fails here.
    """
    from foldjax.models.protenix.models.model import cast_trunk_params

    narrowed = cast_trunk_params(
        _ParamsForCastTest(
            input_embedder=jnp.ones((2, 2), jnp.float32),
            pairformer_output=jnp.ones((2, 2), jnp.float32),
            diffusion=jnp.ones((2, 2), jnp.float32),
        ),
        jnp.bfloat16,
    )

    assert narrowed.input_embedder.dtype == jnp.bfloat16
    assert narrowed.pairformer_output.dtype == jnp.bfloat16
    assert narrowed.diffusion.dtype == jnp.float32, (
        "the diffusion island must survive --trunk-dtype bf16"
    )
