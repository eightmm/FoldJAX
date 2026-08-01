from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from foldjax.models.protenix.bridge.torch_mapping import (
    map_atom_attention_decoder_state_dict,
    map_atom_attention_encoder_cache_state_dict,
    map_atom_attention_encoder_state_dict,
)
from foldjax.models.protenix.models.diffusion.atom import (
    AtomAttentionDecoderParams,
    AtomAttentionEncoderCacheParams,
    AtomAttentionEncoderParams,
    AtomPairMlpParams,
    aggregate_atom_to_token,
    atom_attention_decoder,
    atom_attention_encoder,
    atom_attention_encoder_prepare_cache,
    atom_attention_encoder_prepare_diffusion_cache,
    broadcast_token_to_atom,
    broadcast_token_to_local_atom_pair,
    rearrange_qk_to_dense_trunk,
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
)


def test_rearrange_qk_to_dense_trunk_shapes_and_mask() -> None:
    q = jnp.arange(5, dtype=jnp.float32)
    k = jnp.arange(5, dtype=jnp.float32) + 10

    q_trunked, k_trunked, pad_info = rearrange_qk_to_dense_trunk(
        q,
        k,
        n_queries=2,
        n_keys=4,
        compute_mask=True,
    )

    np.testing.assert_array_equal(np.asarray(q_trunked), [[0, 1], [2, 3], [4, 0]])
    np.testing.assert_array_equal(
        np.asarray(k_trunked),
        [[0, 10, 11, 12], [11, 12, 13, 14], [13, 14, 0, 0]],
    )
    assert pad_info["q_pad"] == 1
    assert pad_info["k_pad_left"] == 1
    assert pad_info["k_pad_right"] == 2
    assert pad_info["mask_trunked"].shape == (3, 2, 4)
    assert not bool(pad_info["mask_trunked"][0, 0, 0])
    assert bool(pad_info["mask_trunked"][1, 0, 1])
    assert not bool(pad_info["mask_trunked"][2, 1, 0])


def test_broadcast_and_aggregate_atom_token_helpers() -> None:
    x_token = jnp.asarray([[10.0, 0.0], [20.0, 1.0], [30.0, 2.0]])
    atom_to_token_idx = jnp.asarray([0, 0, 1, 2, 2])

    atom = broadcast_token_to_atom(x_token, atom_to_token_idx)
    mean = aggregate_atom_to_token(atom, atom_to_token_idx, n_token=3, reduce="mean")
    summed = aggregate_atom_to_token(atom, atom_to_token_idx, n_token=3, reduce="sum")

    np.testing.assert_allclose(np.asarray(atom[2]), [20.0, 1.0])
    np.testing.assert_allclose(np.asarray(mean), np.asarray(x_token))
    np.testing.assert_allclose(
        np.asarray(summed),
        [[20.0, 0.0], [20.0, 1.0], [60.0, 4.0]],
    )


def test_broadcast_token_to_local_atom_pair_gathers_pair_embedding() -> None:
    z_token = jnp.arange(9, dtype=jnp.float32).reshape(3, 3, 1)
    atom_to_token_idx = jnp.asarray([0, 1, 1, 2])

    blocked, pad_info = broadcast_token_to_local_atom_pair(
        z_token,
        atom_to_token_idx,
        n_queries=2,
        n_keys=4,
        compute_mask=True,
    )

    assert blocked.shape == (2, 2, 4, 1)
    np.testing.assert_allclose(
        np.asarray(blocked[0, :, :, 0]),
        [[0, 0, 1, 1], [3, 3, 4, 4]],
    )
    assert pad_info["mask_trunked"].shape == (2, 2, 4)


def test_atom_attention_encoder_prepare_cache_base_path() -> None:
    params = AtomAttentionEncoderCacheParams(
        linear_ref_pos=LinearParams(weight=jnp.eye(3, dtype=jnp.float32), bias=None),
        linear_ref_charge=LinearParams(
            weight=jnp.asarray([[2.0], [3.0], [4.0]]),
            bias=None,
        ),
        linear_f=LinearParams(weight=jnp.ones((3, 385), dtype=jnp.float32), bias=None),
        linear_d=LinearParams(weight=jnp.ones((2, 3), dtype=jnp.float32), bias=None),
        linear_invd=LinearParams(weight=jnp.asarray([[5.0], [7.0]]), bias=None),
        linear_v=LinearParams(weight=jnp.asarray([[11.0], [13.0]]), bias=None),
    )
    ref_pos = jnp.asarray([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    ref_charge = jnp.asarray([0.0, 1.0])
    ref_mask = jnp.asarray([1.0, 0.0])
    ref_element = jnp.zeros((2, 128), dtype=jnp.float32)
    ref_atom_name_chars = jnp.zeros((2, 4, 64), dtype=jnp.float32)
    d_lm = jnp.asarray([[[[1.0, 2.0, 2.0]]]])
    v_lm = jnp.ones((1, 1, 1, 1), dtype=jnp.float32)
    pad_info = {"mask_trunked": jnp.ones((1, 1, 1), dtype=bool)}

    p_lm, c_l = atom_attention_encoder_prepare_cache(
        ref_pos,
        ref_charge,
        ref_mask,
        ref_element,
        ref_atom_name_chars,
        d_lm,
        v_lm,
        pad_info,
        params,
    )

    assert p_lm.shape == (1, 1, 1, 2)
    assert c_l.shape == (2, 3)
    np.testing.assert_allclose(np.asarray(c_l[1]), [0.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(np.asarray(p_lm[0, 0, 0]), [16.5, 18.7], atol=1e-6)


def test_map_atom_attention_encoder_cache_state_dict_shapes() -> None:
    state = {
        "input_embedder.atom_attention_encoder.linear_no_bias_ref_pos.weight": (
            np.ones((128, 3), dtype=np.float32)
        ),
        "input_embedder.atom_attention_encoder.linear_no_bias_ref_charge.weight": (
            np.ones((128, 1), dtype=np.float32)
        ),
        "input_embedder.atom_attention_encoder.linear_no_bias_f.weight": np.ones(
            (128, 385), dtype=np.float32
        ),
        "input_embedder.atom_attention_encoder.linear_no_bias_d.weight": np.ones(
            (16, 3), dtype=np.float32
        ),
        "input_embedder.atom_attention_encoder.linear_no_bias_invd.weight": np.ones(
            (16, 1), dtype=np.float32
        ),
        "input_embedder.atom_attention_encoder.linear_no_bias_v.weight": np.ones(
            (16, 1), dtype=np.float32
        ),
    }

    params = map_atom_attention_encoder_cache_state_dict(
        state,
        "input_embedder.atom_attention_encoder",
    )

    assert params.linear_ref_pos.weight.shape == (128, 3)
    assert params.linear_f.weight.shape == (128, 385)
    assert params.linear_d.weight.shape == (16, 3)


def test_map_atom_attention_encoder_has_coords_state_dict_shapes() -> None:
    prefix = "diffusion_module.atom_attention_encoder"
    state = {
        f"{prefix}.linear_no_bias_ref_pos.weight": np.ones(
            (128, 3), dtype=np.float32
        ),
        f"{prefix}.linear_no_bias_ref_charge.weight": np.ones(
            (128, 1), dtype=np.float32
        ),
        f"{prefix}.linear_no_bias_f.weight": np.ones((128, 385), dtype=np.float32),
        f"{prefix}.linear_no_bias_d.weight": np.ones((16, 3), dtype=np.float32),
        f"{prefix}.linear_no_bias_invd.weight": np.ones((16, 1), dtype=np.float32),
        f"{prefix}.linear_no_bias_v.weight": np.ones((16, 1), dtype=np.float32),
        f"{prefix}.layernorm_s.weight": np.ones((384,), dtype=np.float32),
        f"{prefix}.linear_no_bias_s.weight": np.ones((128, 384), dtype=np.float32),
        f"{prefix}.layernorm_z.weight": np.ones((256,), dtype=np.float32),
        f"{prefix}.linear_no_bias_z.weight": np.ones((16, 256), dtype=np.float32),
        f"{prefix}.linear_no_bias_r.weight": np.ones((128, 3), dtype=np.float32),
        f"{prefix}.linear_no_bias_cl.weight": np.ones((16, 128), dtype=np.float32),
        f"{prefix}.linear_no_bias_cm.weight": np.ones((16, 128), dtype=np.float32),
        f"{prefix}.small_mlp.1.weight": np.ones((16, 16), dtype=np.float32),
        f"{prefix}.small_mlp.3.weight": np.ones((16, 16), dtype=np.float32),
        f"{prefix}.small_mlp.5.weight": np.ones((16, 16), dtype=np.float32),
        f"{prefix}.linear_no_bias_q.weight": np.ones((768, 128), dtype=np.float32),
    }
    _add_transformer_block_state(
        state,
        f"{prefix}.atom_transformer.diffusion_transformer.blocks.0",
        c_a=128,
        c_s=128,
        c_z=16,
        n_heads=4,
    )

    params = map_atom_attention_encoder_state_dict(state, prefix, has_coords=True)

    assert params.layernorm_s is not None
    assert params.linear_s is not None
    assert params.layernorm_z is not None
    assert params.linear_z is not None
    assert params.linear_r is not None
    assert params.linear_s.weight.shape == (128, 384)
    assert params.linear_z.weight.shape == (16, 256)
    assert params.linear_r.weight.shape == (128, 3)
    assert params.linear_q.weight.shape == (768, 128)


def test_atom_attention_encoder_has_coords_false_smoke() -> None:
    params = AtomAttentionEncoderParams(
        cache=AtomAttentionEncoderCacheParams(
            linear_ref_pos=LinearParams(
                weight=jnp.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
                bias=None,
            ),
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
        atom_transformer=DiffusionTransformerStackParams(
            blocks=(_zero_effect_atom_block(c_atom=2, c_atompair=2),)
        ),
        linear_q=LinearParams(weight=jnp.eye(2), bias=None),
    )
    atom_to_token_idx = jnp.asarray([0, 0, 1, 1])
    ref_pos = jnp.asarray(
        [[1.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 4.0, 0.0]]
    )
    ref_charge = jnp.zeros((4,))
    ref_mask = jnp.ones((4,))
    ref_element = jnp.zeros((4, 128))
    ref_atom_name_chars = jnp.zeros((4, 4, 64))
    d_lm = jnp.zeros((2, 2, 4, 3))
    v_lm = jnp.ones((2, 2, 4, 1))
    pad_info = {"mask_trunked": jnp.ones((2, 2, 4), dtype=bool)}

    a, q_l, c_l, p_lm = atom_attention_encoder(
        atom_to_token_idx,
        ref_pos,
        ref_charge,
        ref_mask,
        ref_atom_name_chars,
        ref_element,
        d_lm,
        v_lm,
        pad_info,
        params,
        n_token=2,
        n_heads=1,
        n_queries=2,
        n_keys=4,
    )

    np.testing.assert_allclose(np.asarray(a), [[2.0, 0.0], [0.0, 3.0]], atol=1e-5)
    np.testing.assert_allclose(np.asarray(q_l), np.asarray(c_l), atol=1e-5)
    assert p_lm.shape == (2, 2, 4, 2)


def test_atom_attention_encoder_has_coords_true_smoke() -> None:
    params = AtomAttentionEncoderParams(
        cache=AtomAttentionEncoderCacheParams(
            linear_ref_pos=LinearParams(
                weight=jnp.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
                bias=None,
            ),
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
        atom_transformer=DiffusionTransformerStackParams(
            blocks=(_zero_effect_atom_block(c_atom=2, c_atompair=2),)
        ),
        linear_q=LinearParams(weight=jnp.eye(2), bias=None),
        layernorm_s=LayerNormParams(weight=jnp.ones((2,)), bias=None),
        linear_s=LinearParams(weight=jnp.eye(2), bias=None),
        layernorm_z=LayerNormParams(weight=jnp.ones((2,)), bias=None),
        linear_z=LinearParams(weight=jnp.eye(2), bias=None),
        linear_r=LinearParams(
            weight=jnp.asarray([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
            bias=None,
        ),
    )
    atom_to_token_idx = jnp.asarray([0, 0, 1, 1])
    ref_pos = jnp.asarray(
        [[1.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 4.0, 0.0]]
    )
    ref_charge = jnp.zeros((4,))
    ref_mask = jnp.ones((4,))
    ref_element = jnp.zeros((4, 128))
    ref_atom_name_chars = jnp.zeros((4, 4, 64))
    d_lm = jnp.zeros((2, 2, 4, 3))
    v_lm = jnp.ones((2, 2, 4, 1))
    pad_info = {"mask_trunked": jnp.ones((2, 2, 4), dtype=bool)}
    r_l = jnp.asarray(
        [[[0.5, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 0.0, 2.5], [0.0, 0.0, 3.5]]]
    )
    s = jnp.asarray([[[0.0, 1.0], [1.0, 0.0]]])
    z = jnp.zeros((1, 2, 2, 2))

    a, q_l, c_l, p_lm = atom_attention_encoder(
        atom_to_token_idx,
        ref_pos,
        ref_charge,
        ref_mask,
        ref_atom_name_chars,
        ref_element,
        d_lm,
        v_lm,
        pad_info,
        params,
        r_l=r_l,
        s=s,
        z=z,
        n_token=2,
        n_heads=1,
        n_queries=2,
        n_keys=4,
    )

    assert a.shape == (1, 2, 2)
    assert q_l.shape == (1, 4, 2)
    assert c_l.shape == (1, 4, 2)
    assert p_lm.shape == (1, 2, 2, 4, 2)
    np.testing.assert_allclose(
        np.asarray(a),
        [[[2.0, 1.0], [1.0, 5.0]]],
        atol=1e-4,
    )

    cached_p_lm, cached_c_l = atom_attention_encoder_prepare_diffusion_cache(
        atom_to_token_idx,
        ref_pos,
        ref_charge,
        ref_mask,
        ref_element,
        ref_atom_name_chars,
        d_lm,
        v_lm,
        pad_info,
        z,
        params,
        n_queries=2,
        n_keys=4,
    )
    cached = atom_attention_encoder(
        atom_to_token_idx,
        ref_pos,
        ref_charge,
        ref_mask,
        ref_atom_name_chars,
        ref_element,
        d_lm,
        v_lm,
        pad_info,
        params,
        r_l=r_l,
        s=s,
        z=z,
        p_lm=cached_p_lm,
        c_l=cached_c_l,
        n_token=2,
        n_heads=1,
        n_queries=2,
        n_keys=4,
    )
    for cached_value, uncached_value in zip(cached, (a, q_l, c_l, p_lm), strict=True):
        np.testing.assert_allclose(cached_value, uncached_value, rtol=1e-6, atol=1e-6)


def test_map_atom_attention_decoder_state_dict_shapes() -> None:
    prefix = "diffusion_module.atom_attention_decoder"
    state = {
        f"{prefix}.linear_no_bias_a.weight": np.ones((128, 768), dtype=np.float32),
        f"{prefix}.layernorm_q.weight": np.ones((128,), dtype=np.float32),
        f"{prefix}.linear_no_bias_out.weight": np.ones((3, 128), dtype=np.float32),
    }
    _add_transformer_block_state(
        state,
        f"{prefix}.atom_transformer.diffusion_transformer.blocks.0",
        c_a=128,
        c_s=128,
        c_z=16,
        n_heads=4,
    )

    params = map_atom_attention_decoder_state_dict(state, prefix)

    assert params.linear_a.weight.shape == (128, 768)
    assert params.layernorm_q.weight.shape == (128,)
    assert params.linear_out.weight.shape == (3, 128)
    assert len(params.atom_transformer.blocks) == 1


def test_atom_attention_decoder_smoke() -> None:
    params = AtomAttentionDecoderParams(
        linear_a=LinearParams(weight=jnp.eye(2), bias=None),
        layernorm_q=LayerNormParams(weight=jnp.ones((2,)), bias=None),
        linear_out=LinearParams(
            weight=jnp.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
            bias=None,
        ),
        atom_transformer=DiffusionTransformerStackParams(
            blocks=(_zero_effect_atom_block(c_atom=2, c_atompair=2),)
        ),
    )
    atom_to_token_idx = jnp.asarray([0, 1, 1])
    a = jnp.asarray([[[1.0, 3.0], [2.0, 4.0]]])
    q_skip = jnp.zeros((1, 3, 2), dtype=jnp.float32)
    c_skip = jnp.zeros((1, 3, 2), dtype=jnp.float32)
    p_skip = jnp.zeros((1, 2, 2, 4, 2), dtype=jnp.float32)

    out = atom_attention_decoder(
        atom_to_token_idx,
        a,
        q_skip,
        c_skip,
        p_skip,
        params,
        n_heads=1,
        n_queries=2,
        n_keys=4,
    )

    assert out.shape == (1, 3, 3)
    np.testing.assert_allclose(
        np.asarray(out),
        [[[-1.0, 1.0, 0.0], [-1.0, 1.0, 0.0], [-1.0, 1.0, 0.0]]],
        atol=1e-4,
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


def _add_transformer_block_state(
    state: dict[str, np.ndarray],
    prefix: str,
    *,
    c_a: int,
    c_s: int,
    c_z: int,
    n_heads: int,
) -> None:
    state.update(
        {
            f"{prefix}.attention_pair_bias.layernorm_a.layernorm_s.weight": np.ones(
                (c_s,), dtype=np.float32
            ),
            f"{prefix}.attention_pair_bias.layernorm_a.linear_s.weight": np.ones(
                (c_a, c_s), dtype=np.float32
            ),
            f"{prefix}.attention_pair_bias.layernorm_a.linear_s.bias": np.zeros(
                (c_a,), dtype=np.float32
            ),
            f"{prefix}.attention_pair_bias.layernorm_a.linear_nobias_s.weight": (
                np.ones((c_a, c_s), dtype=np.float32)
            ),
            f"{prefix}.attention_pair_bias.layernorm_kv.layernorm_s.weight": np.ones(
                (c_s,), dtype=np.float32
            ),
            f"{prefix}.attention_pair_bias.layernorm_kv.linear_s.weight": np.ones(
                (c_a, c_s), dtype=np.float32
            ),
            f"{prefix}.attention_pair_bias.layernorm_kv.linear_s.bias": np.zeros(
                (c_a,), dtype=np.float32
            ),
            f"{prefix}.attention_pair_bias.layernorm_kv.linear_nobias_s.weight": (
                np.ones((c_a, c_s), dtype=np.float32)
            ),
            f"{prefix}.attention_pair_bias.attention.linear_q.weight": np.ones(
                (c_a, c_a), dtype=np.float32
            ),
            f"{prefix}.attention_pair_bias.attention.linear_q.bias": np.zeros(
                (c_a,), dtype=np.float32
            ),
            f"{prefix}.attention_pair_bias.attention.linear_k.weight": np.ones(
                (c_a, c_a), dtype=np.float32
            ),
            f"{prefix}.attention_pair_bias.attention.linear_v.weight": np.ones(
                (c_a, c_a), dtype=np.float32
            ),
            f"{prefix}.attention_pair_bias.attention.linear_o.weight": np.ones(
                (c_a, c_a), dtype=np.float32
            ),
            f"{prefix}.attention_pair_bias.attention.linear_g.weight": np.ones(
                (c_a, c_a), dtype=np.float32
            ),
            f"{prefix}.attention_pair_bias.layernorm_z.weight": np.ones(
                (c_z,), dtype=np.float32
            ),
            f"{prefix}.attention_pair_bias.linear_nobias_z.weight": np.ones(
                (n_heads, c_z), dtype=np.float32
            ),
            f"{prefix}.attention_pair_bias.linear_a_last.weight": np.ones(
                (c_a, c_s), dtype=np.float32
            ),
            f"{prefix}.attention_pair_bias.linear_a_last.bias": np.zeros(
                (c_a,), dtype=np.float32
            ),
            f"{prefix}.conditioned_transition_block.adaln.layernorm_s.weight": (
                np.ones((c_s,), dtype=np.float32)
            ),
            f"{prefix}.conditioned_transition_block.adaln.linear_s.weight": np.ones(
                (c_a, c_s), dtype=np.float32
            ),
            f"{prefix}.conditioned_transition_block.adaln.linear_s.bias": np.zeros(
                (c_a,), dtype=np.float32
            ),
            f"{prefix}.conditioned_transition_block.adaln.linear_nobias_s.weight": (
                np.ones((c_a, c_s), dtype=np.float32)
            ),
            f"{prefix}.conditioned_transition_block.linear_nobias_a1.weight": (
                np.ones((2 * c_a, c_a), dtype=np.float32)
            ),
            f"{prefix}.conditioned_transition_block.linear_nobias_a2.weight": (
                np.ones((2 * c_a, c_a), dtype=np.float32)
            ),
            f"{prefix}.conditioned_transition_block.linear_nobias_b.weight": (
                np.ones((c_a, 2 * c_a), dtype=np.float32)
            ),
            f"{prefix}.conditioned_transition_block.linear_s.weight": np.ones(
                (c_a, c_s), dtype=np.float32
            ),
            f"{prefix}.conditioned_transition_block.linear_s.bias": np.zeros(
                (c_a,), dtype=np.float32
            ),
        }
    )
