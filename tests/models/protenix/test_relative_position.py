from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from foldjax.models.protenix.bridge.torch_mapping import (
    map_relative_position_state_dict,
)
from foldjax.models.protenix.models.diffusion.atom import AtomAttentionEncoderParams
from foldjax.models.protenix.models.primitives.primitives import LinearParams
from foldjax.models.protenix.models.trunk_blocks.embedders import (
    ConstraintEmbedderParams,
    InputFeatureEmbedderParams,
    SubstructureMlpParams,
    constraint_embedder,
    input_feature_embedder,
    relative_position_encoding,
    relative_position_features,
)


def test_relative_position_features_shape_and_cross_chain_bins() -> None:
    features = {
        "asym_id": jnp.asarray([1, 1, 2]),
        "residue_index": jnp.asarray([10, 12, 10]),
        "entity_id": jnp.asarray([5, 5, 6]),
        "sym_id": jnp.asarray([1, 2, 1]),
        "token_index": jnp.asarray([0, 1, 2]),
    }

    relp = np.asarray(relative_position_features(features, r_max=2, s_max=1))

    assert relp.shape == (3, 3, 17)
    expected_sum = np.array(
        [
            [4.0, 4.0, 3.0],
            [4.0, 4.0, 3.0],
            [3.0, 3.0, 4.0],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(relp.sum(axis=-1), expected_sum)
    assert relp[0, 2, 5] == 1.0  # cross-chain residue overflow bin
    assert relp[0, 2, 11] == 1.0  # cross-chain token overflow bin
    assert relp[0, 2, 12] == 0.0  # not same entity
    assert relp[0, 2, 16] == 1.0  # cross-entity chain overflow bin


def test_relative_position_encoding_applies_mapped_linear() -> None:
    relp = np.eye(5, dtype=np.float32).reshape(1, 5, 5)
    state = {
        "rel.linear_no_bias.weight": np.arange(10, dtype=np.float32).reshape(2, 5),
    }

    params = map_relative_position_state_dict(state, "rel")
    actual = np.asarray(relative_position_encoding(jnp.asarray(relp), params))
    expected = relp @ state["rel.linear_no_bias.weight"].T

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_input_feature_embedder_concatenates_token_features_and_esm() -> None:
    params = InputFeatureEmbedderParams(
        atom_encoder=_dummy_atom_encoder_params(),
        linear_esm=LinearParams(weight=jnp.ones((67, 3), dtype=jnp.float32), bias=None),
    )
    features = {
        "atom_to_token_idx": jnp.asarray([0, 0, 1, 1]),
        "ref_pos": jnp.asarray(
            [
                [1.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 4.0, 0.0],
            ]
        ),
        "ref_charge": jnp.zeros((4,)),
        "ref_mask": jnp.ones((4,)),
        "ref_atom_name_chars": jnp.zeros((4, 4, 64)),
        "ref_element": jnp.zeros((4, 128)),
        "d_lm": jnp.zeros((2, 2, 4, 3)),
        "v_lm": jnp.ones((2, 2, 4, 1)),
        "pad_info": {"mask_trunked": jnp.ones((2, 2, 4), dtype=bool)},
        "restype": jnp.arange(64, dtype=jnp.float32).reshape(2, 32),
        "profile": jnp.arange(100, 164, dtype=jnp.float32).reshape(2, 32),
        "deletion_mean": jnp.asarray([30.0, 31.0]),
        "esm_token_embedding": jnp.asarray([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]),
    }

    s_inputs = input_feature_embedder(
        features,
        params,
        n_token=2,
        n_heads=1,
        n_queries=2,
        n_keys=4,
    )

    expected_no_esm = np.concatenate(
        [
            np.asarray([[2.0, 0.0], [0.0, 3.0]], dtype=np.float32),
            np.arange(64, dtype=np.float32).reshape(2, 32),
            np.arange(100, 164, dtype=np.float32).reshape(2, 32),
            np.asarray([[30.0], [31.0]], dtype=np.float32),
        ],
        axis=-1,
    )
    expected = expected_no_esm + np.asarray([[1.0] * 67, [2.0] * 67])
    np.testing.assert_allclose(np.asarray(s_inputs), expected, atol=1e-5)


def test_constraint_embedder_sums_enabled_pair_features() -> None:
    params = ConstraintEmbedderParams(
        pocket_z=LinearParams(weight=jnp.asarray([[1.0], [2.0]]), bias=None),
        contact_z=LinearParams(
            weight=jnp.asarray([[1.0, 0.0], [0.0, 1.0]]),
            bias=None,
        ),
        contact_atom_z=LinearParams(weight=jnp.asarray([[3.0], [4.0]]), bias=None),
    )
    features = {
        "pocket": jnp.asarray([[[1.0], [2.0]], [[3.0], [4.0]]]),
        "contact": jnp.asarray(
            [
                [[10.0, 20.0], [30.0, 40.0]],
                [[50.0, 60.0], [70.0, 80.0]],
            ]
        ),
        "contact_atom": jnp.asarray([[[2.0], [3.0]], [[4.0], [5.0]]]),
    }

    actual = constraint_embedder(features, params)
    expected = np.asarray(
        [
            [[17.0, 30.0], [41.0, 56.0]],
            [[65.0, 82.0], [89.0, 108.0]],
        ],
        dtype=np.float32,
    )

    np.testing.assert_allclose(np.asarray(actual), expected, atol=1e-6)


def test_constraint_embedder_supports_substructure_mlp() -> None:
    params = ConstraintEmbedderParams(
        substructure_z=SubstructureMlpParams(
            layers=(
                LinearParams(
                    weight=jnp.asarray([[1.0, -1.0], [0.5, 0.5]]),
                    bias=None,
                ),
                LinearParams(weight=jnp.asarray([[2.0, 3.0]]), bias=None),
            )
        )
    )
    features = {"substructure": jnp.asarray([[[4.0, 1.0], [1.0, 5.0]]])}

    actual = constraint_embedder(features, params)
    expected = np.asarray([[[13.5], [9.0]]], dtype=np.float32)

    np.testing.assert_allclose(np.asarray(actual), expected, atol=1e-6)


def test_constraint_embedder_returns_none_when_all_paths_disabled() -> None:
    assert constraint_embedder({}, ConstraintEmbedderParams()) is None


def _dummy_atom_encoder_params() -> AtomAttentionEncoderParams:
    from foldjax.models.protenix.models.diffusion.atom import (
        AtomAttentionEncoderCacheParams,
        AtomPairMlpParams,
    )
    from foldjax.models.protenix.models.diffusion.transformer import (
        DiffusionTransformerStackParams,
    )

    return AtomAttentionEncoderParams(
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


def _zero_effect_atom_block(
    *,
    c_atom: int,
    c_atompair: int,
):
    from foldjax.models.protenix.models.diffusion.transformer import (
        ConditionedTransitionParams,
        DiffusionTransformerBlockParams,
    )
    from foldjax.models.protenix.models.primitives.attention import (
        AttentionPairBiasParams,
        AttentionParams,
    )
    from foldjax.models.protenix.models.primitives.primitives import (
        AdaptiveLayerNormParams,
        LayerNormParams,
    )

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
