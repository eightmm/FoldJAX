from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from foldjax.models.protenix.bridge.torch_mapping import (
    map_pairformer_output_state_dict,
    map_trunk_initialization_state_dict,
)
from foldjax.models.protenix.models.primitives.attention import (
    AttentionPairBiasParams,
    AttentionParams,
)
from foldjax.models.protenix.models.primitives.primitives import (
    LayerNormParams,
    LinearParams,
    TransitionParams,
)
from foldjax.models.protenix.models.triangle.triangle import (
    TriangleAttentionParams,
    TriangleMultiplicationParams,
)
from foldjax.models.protenix.models.trunk_blocks.embedders import (
    ConstraintEmbedderParams,
    RelativePositionParams,
    relative_position_encoding,
)
from foldjax.models.protenix.models.trunk_blocks.msa import MSAModuleParams
from foldjax.models.protenix.models.trunk_blocks.pairformer import (
    PairformerBlockParams,
    PairformerStackParams,
)
from foldjax.models.protenix.models.trunk_blocks.template import TemplateEmbedderParams
from foldjax.models.protenix.models.trunk_blocks.trunk import (
    PairformerOutputParams,
    RecyclingProjectionParams,
    TrunkInitializationParams,
    TrunkParams,
    pairformer_output_from_s_inputs,
    recycle_embeddings,
    trunk_initial_embeddings,
)


def test_trunk_initial_embeddings_match_protenix_formula() -> None:
    params = TrunkInitializationParams(
        linear_sinit=LinearParams(
            weight=jnp.asarray([[1.0, 2.0], [3.0, 4.0]]),
            bias=None,
        ),
        linear_zinit1=LinearParams(
            weight=jnp.asarray([[1.0, 0.0], [0.0, 10.0]]),
            bias=None,
        ),
        linear_zinit2=LinearParams(
            weight=jnp.asarray([[2.0, 0.0], [0.0, 20.0]]),
            bias=None,
        ),
        relative_position=RelativePositionParams(
            linear_no_bias=LinearParams(
                weight=jnp.asarray([[0.5, 1.0], [1.5, 2.0]]),
                bias=None,
            )
        ),
        linear_token_bond=LinearParams(weight=jnp.asarray([[3.0], [4.0]]), bias=None),
    )
    s_inputs = jnp.asarray([[1.0, 2.0], [3.0, 4.0]])
    relp = jnp.ones((2, 2, 2), dtype=jnp.float32)
    token_bonds = jnp.asarray([[0.0, 1.0], [2.0, 0.0]])

    s_init, z_init = trunk_initial_embeddings(s_inputs, relp, token_bonds, params)

    expected_s = np.asarray(s_inputs) @ np.asarray(params.linear_sinit.weight).T
    expected_z = (
        expected_s @ np.asarray(params.linear_zinit1.weight).T
    )[:, None, :] + (
        expected_s @ np.asarray(params.linear_zinit2.weight).T
    )[None, :, :]
    expected_z += np.asarray(relp) @ np.asarray(
        params.relative_position.linear_no_bias.weight
    ).T
    expected_z += np.asarray(token_bonds)[..., None] @ np.asarray(
        params.linear_token_bond.weight
    ).T

    np.testing.assert_allclose(np.asarray(s_init), expected_s, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.asarray(z_init), expected_z, rtol=1e-6, atol=1e-6)


def _compact_relative_positions() -> np.ndarray:
    rng = np.random.default_rng(7)
    relp = np.zeros((8, 8, 139), dtype=np.int8)
    for start, width in ((0, 66), (66, 66), (133, 6)):
        bins = rng.integers(0, width, size=(8, 8))
        np.put_along_axis(relp[..., start : start + width], bins[..., None], 1, -1)
    relp[..., 132] = rng.integers(0, 2, size=(8, 8))
    return relp


def test_compact_relative_positions_keep_the_bfloat16_trunk_projection() -> None:
    """Model entry used to narrow the float32 host feature to trunk BF16."""
    rng = np.random.default_rng(8)
    relp = _compact_relative_positions()
    weight = rng.normal(size=(32, 139)).astype(np.float32).astype(jnp.bfloat16)
    params = RelativePositionParams(
        linear_no_bias=LinearParams(weight=weight, bias=None)
    )

    actual = relative_position_encoding(jnp.asarray(relp), params)
    historical = relative_position_encoding(
        jnp.asarray(relp, dtype=jnp.bfloat16), params
    )

    assert actual.dtype == jnp.bfloat16
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(historical))


def test_compact_relative_positions_keep_the_float32_diffusion_projection() -> None:
    """Diffusion consumes the original feature with its float32 weights."""
    rng = np.random.default_rng(9)
    relp = _compact_relative_positions()
    weight = rng.normal(size=(32, 139)).astype(np.float32)
    params = RelativePositionParams(
        linear_no_bias=LinearParams(weight=jnp.asarray(weight), bias=None)
    )

    actual = relative_position_encoding(jnp.asarray(relp), params)
    historical = relative_position_encoding(
        jnp.asarray(relp, dtype=jnp.float32), params
    )

    assert actual.dtype == jnp.float32
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(historical))


def test_recycle_embeddings_match_protenix_projection_formula() -> None:
    params = RecyclingProjectionParams(
        layernorm_z=LayerNormParams(weight=jnp.ones((2,)), bias=jnp.zeros((2,))),
        linear_z=LinearParams(weight=jnp.asarray([[1.0, 0.0], [0.0, 2.0]]), bias=None),
        layernorm_s=LayerNormParams(weight=jnp.ones((2,)), bias=jnp.zeros((2,))),
        linear_s=LinearParams(weight=jnp.asarray([[2.0, 0.0], [0.0, 3.0]]), bias=None),
    )
    s_init = jnp.asarray([[1.0, 1.0], [2.0, 2.0]])
    z_init = jnp.ones((2, 2, 2), dtype=jnp.float32)
    s = jnp.asarray([[1.0, 3.0], [2.0, 6.0]])
    z = jnp.asarray(
        [
            [[1.0, 3.0], [2.0, 4.0]],
            [[5.0, 7.0], [8.0, 10.0]],
        ],
        dtype=jnp.float32,
    )

    s_out, z_out = recycle_embeddings(s_init, z_init, s, z, params)

    assert s_out.shape == s_init.shape
    assert z_out.shape == z_init.shape
    np.testing.assert_allclose(np.asarray(s_out[0]), [-1.0, 4.0], atol=1e-4)
    np.testing.assert_allclose(np.asarray(z_out[0, 0]), [0.0, 3.0], atol=1e-4)


def test_map_trunk_initialization_state_dict_shapes() -> None:
    state = {
        "linear_no_bias_sinit.weight": np.ones((384, 449), dtype=np.float32),
        "linear_no_bias_zinit1.weight": np.ones((128, 384), dtype=np.float32),
        "linear_no_bias_zinit2.weight": np.ones((128, 384), dtype=np.float32),
        "relative_position_encoding.linear_no_bias.weight": np.ones(
            (128, 139), dtype=np.float32
        ),
        "linear_no_bias_token_bond.weight": np.ones((128, 1), dtype=np.float32),
        "layernorm_z_cycle.weight": np.ones((128,), dtype=np.float32),
        "layernorm_z_cycle.bias": np.zeros((128,), dtype=np.float32),
        "linear_no_bias_z_cycle.weight": np.ones((128, 128), dtype=np.float32),
        "layernorm_s.weight": np.ones((384,), dtype=np.float32),
        "layernorm_s.bias": np.zeros((384,), dtype=np.float32),
        "linear_no_bias_s.weight": np.ones((384, 384), dtype=np.float32),
    }

    params = map_trunk_initialization_state_dict(state)

    assert params.initial.linear_sinit.weight.shape == (384, 449)
    assert params.initial.linear_zinit1.weight.shape == (128, 384)
    assert params.initial.linear_token_bond.weight.shape == (128, 1)
    assert params.recycling.linear_z.weight.shape == (128, 128)


def test_pairformer_output_from_s_inputs_matches_recycling_order() -> None:
    params = _pairformer_output_params()
    features = {
        "relp": jnp.zeros((2, 2, 2), dtype=jnp.float32),
        "token_bonds": jnp.zeros((2, 2), dtype=jnp.float32),
    }
    s_inputs = jnp.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32)

    s_out_inputs, s, z = pairformer_output_from_s_inputs(
        features,
        s_inputs,
        params,
        n_cycle=2,
    )

    s_init, z_init = trunk_initial_embeddings(
        s_inputs,
        features["relp"],
        features["token_bonds"],
        params.trunk.initial,
    )
    first_s, first_z = recycle_embeddings(
        s_init,
        z_init,
        jnp.zeros_like(s_init),
        jnp.zeros_like(z_init),
        params.trunk.recycling,
    )
    expected_s, expected_z = recycle_embeddings(
        s_init,
        z_init,
        first_s,
        first_z,
        params.trunk.recycling,
    )

    np.testing.assert_allclose(np.asarray(s_out_inputs), np.asarray(s_inputs))
    np.testing.assert_allclose(np.asarray(s), np.asarray(expected_s), atol=1e-5)
    np.testing.assert_allclose(np.asarray(z), np.asarray(expected_z), atol=1e-5)


def test_map_pairformer_output_state_dict_groups_trunk_params() -> None:
    state = _pairformer_output_state()

    params = map_pairformer_output_state_dict(state)

    assert params.trunk.initial.linear_sinit.weight.shape == (4, 2)
    assert params.template.linear_a.weight.shape == (2, 108)
    assert params.msa.linear_m.weight.shape == (3, 34)
    assert len(params.pairformer_stack.blocks) == 1


def _pairformer_output_params() -> PairformerOutputParams:
    trunk = TrunkInitializationParams(
        linear_sinit=LinearParams(weight=jnp.eye(2), bias=None),
        linear_zinit1=LinearParams(weight=jnp.eye(2), bias=None),
        linear_zinit2=LinearParams(weight=jnp.eye(2), bias=None),
        relative_position=RelativePositionParams(
            linear_no_bias=LinearParams(weight=jnp.zeros((2, 2)), bias=None)
        ),
        linear_token_bond=LinearParams(weight=jnp.zeros((2, 1)), bias=None),
    )
    recycling = RecyclingProjectionParams(
        layernorm_z=LayerNormParams(weight=jnp.ones((2,)), bias=jnp.zeros((2,))),
        linear_z=LinearParams(weight=jnp.eye(2), bias=None),
        layernorm_s=LayerNormParams(weight=jnp.ones((2,)), bias=jnp.zeros((2,))),
        linear_s=LinearParams(weight=jnp.eye(2), bias=None),
    )
    return PairformerOutputParams(
        trunk=TrunkParams(initial=trunk, recycling=recycling),
        constraint=ConstraintEmbedderParams(),
        template=_unused_template_params(c_z=2),
        msa=MSAModuleParams(
            linear_m=LinearParams(weight=jnp.zeros((2, 34)), bias=None),
            linear_s=LinearParams(weight=jnp.zeros((2, 2)), bias=None),
            blocks=(),
        ),
        pairformer_stack=PairformerStackParams(blocks=(_zero_pairformer_block(2, 2),)),
    )


def _unused_template_params(c_z: int) -> TemplateEmbedderParams:
    return TemplateEmbedderParams(
        linear_z=LinearParams(weight=jnp.zeros((2, c_z)), bias=None),
        layernorm_z=LayerNormParams(weight=jnp.ones((c_z,)), bias=jnp.zeros((c_z,))),
        linear_a=LinearParams(weight=jnp.zeros((2, 108)), bias=None),
        pairformer_stack=PairformerStackParams(blocks=()),
        layernorm_v=LayerNormParams(weight=jnp.ones((2,)), bias=jnp.zeros((2,))),
        linear_u=LinearParams(weight=jnp.zeros((c_z, 2)), bias=None),
    )


def _zero_pairformer_block(c_s: int, c_z: int) -> PairformerBlockParams:
    return PairformerBlockParams(
        tri_mul_out=_zero_tri_mul(c_z),
        tri_mul_in=_zero_tri_mul(c_z),
        tri_att_start=_zero_tri_att(c_z),
        tri_att_end=_zero_tri_att(c_z),
        pair_transition=_zero_transition(c_z, factor=4),
        attention_pair_bias=_zero_attention_pair_bias(c_s, c_z),
        single_transition=_zero_transition(c_s, factor=4),
    )


def _zero_tri_mul(c_z: int) -> TriangleMultiplicationParams:
    return TriangleMultiplicationParams(
        layer_norm_in=LayerNormParams(weight=jnp.ones((c_z,)), bias=jnp.zeros((c_z,))),
        layer_norm_out=LayerNormParams(weight=jnp.ones((c_z,)), bias=jnp.zeros((c_z,))),
        linear_a_p=LinearParams(weight=jnp.zeros((c_z, c_z)), bias=None),
        linear_a_g=LinearParams(weight=jnp.zeros((c_z, c_z)), bias=None),
        linear_b_p=LinearParams(weight=jnp.zeros((c_z, c_z)), bias=None),
        linear_b_g=LinearParams(weight=jnp.zeros((c_z, c_z)), bias=None),
        linear_z=LinearParams(weight=jnp.zeros((c_z, c_z)), bias=None),
        linear_g=LinearParams(weight=jnp.zeros((c_z, c_z)), bias=None),
    )


def _zero_tri_att(c_z: int) -> TriangleAttentionParams:
    attention = AttentionParams(
        linear_q=LinearParams(weight=jnp.zeros((c_z, c_z)), bias=None),
        linear_k=LinearParams(weight=jnp.zeros((c_z, c_z)), bias=None),
        linear_v=LinearParams(weight=jnp.zeros((c_z, c_z)), bias=None),
        linear_o=LinearParams(weight=jnp.zeros((c_z, c_z)), bias=None),
        linear_g=LinearParams(weight=jnp.zeros((c_z, c_z)), bias=None),
    )
    return TriangleAttentionParams(
        layer_norm=LayerNormParams(weight=jnp.ones((c_z,)), bias=jnp.zeros((c_z,))),
        linear=LinearParams(weight=jnp.zeros((1, c_z)), bias=None),
        attention=attention,
    )


def _zero_attention_pair_bias(c_s: int, c_z: int) -> AttentionPairBiasParams:
    attention = AttentionParams(
        linear_q=LinearParams(weight=jnp.zeros((c_s, c_s)), bias=jnp.zeros((c_s,))),
        linear_k=LinearParams(weight=jnp.zeros((c_s, c_s)), bias=None),
        linear_v=LinearParams(weight=jnp.zeros((c_s, c_s)), bias=None),
        linear_o=LinearParams(weight=jnp.zeros((c_s, c_s)), bias=None),
        linear_g=LinearParams(weight=jnp.zeros((c_s, c_s)), bias=None),
    )
    return AttentionPairBiasParams(
        layernorm_a=LayerNormParams(weight=jnp.ones((c_s,)), bias=jnp.zeros((c_s,))),
        layernorm_kv=None,
        attention=attention,
        layernorm_z=LayerNormParams(weight=jnp.ones((c_z,)), bias=jnp.zeros((c_z,))),
        linear_z=LinearParams(weight=jnp.zeros((1, c_z)), bias=None),
        linear_a_last=None,
        has_s=False,
        cross_attention_mode=False,
    )


def _zero_transition(c_in: int, *, factor: int) -> TransitionParams:
    hidden = c_in * factor
    return TransitionParams(
        layer_norm=LayerNormParams(weight=jnp.ones((c_in,)), bias=jnp.zeros((c_in,))),
        linear_a=LinearParams(weight=jnp.zeros((hidden, c_in)), bias=None),
        linear_b=LinearParams(weight=jnp.zeros((hidden, c_in)), bias=None),
        linear_out=LinearParams(weight=jnp.zeros((c_in, hidden)), bias=None),
    )


def _pairformer_output_state() -> dict[str, np.ndarray]:
    state = {
        "linear_no_bias_sinit.weight": np.ones((4, 2), dtype=np.float32),
        "linear_no_bias_zinit1.weight": np.ones((5, 4), dtype=np.float32),
        "linear_no_bias_zinit2.weight": np.ones((5, 4), dtype=np.float32),
        "relative_position_encoding.linear_no_bias.weight": np.ones(
            (5, 139), dtype=np.float32
        ),
        "linear_no_bias_token_bond.weight": np.ones((5, 1), dtype=np.float32),
        "layernorm_z_cycle.weight": np.ones((5,), dtype=np.float32),
        "layernorm_z_cycle.bias": np.zeros((5,), dtype=np.float32),
        "linear_no_bias_z_cycle.weight": np.ones((5, 5), dtype=np.float32),
        "layernorm_s.weight": np.ones((4,), dtype=np.float32),
        "layernorm_s.bias": np.zeros((4,), dtype=np.float32),
        "linear_no_bias_s.weight": np.ones((4, 4), dtype=np.float32),
        "template_embedder.linear_no_bias_z.weight": np.ones((2, 5), dtype=np.float32),
        "template_embedder.layernorm_z.weight": np.ones((5,), dtype=np.float32),
        "template_embedder.layernorm_z.bias": np.zeros((5,), dtype=np.float32),
        "template_embedder.linear_no_bias_a.weight": np.ones(
            (2, 108), dtype=np.float32
        ),
        "template_embedder.layernorm_v.weight": np.ones((2,), dtype=np.float32),
        "template_embedder.layernorm_v.bias": np.zeros((2,), dtype=np.float32),
        "template_embedder.linear_no_bias_u.weight": np.ones((5, 2), dtype=np.float32),
        "msa_module.linear_no_bias_m.weight": np.ones((3, 34), dtype=np.float32),
        "msa_module.linear_no_bias_s.weight": np.ones((3, 2), dtype=np.float32),
    }
    _add_min_pairformer_block_state(
        state,
        "template_embedder.pairformer_stack.blocks.0",
        2,
    )
    _add_min_msa_block_state(state, "msa_module.blocks.0", c_m=3, c_z=5)
    _add_min_pairformer_block_state(state, "pairformer_stack.blocks.0", 5, c_s=4)
    return state


def _add_min_msa_block_state(
    state: dict[str, np.ndarray],
    prefix: str,
    *,
    c_m: int,
    c_z: int,
) -> None:
    opm = f"{prefix}.outer_product_mean_msa"
    state[f"{opm}.layer_norm.weight"] = np.ones((c_m,), dtype=np.float32)
    state[f"{opm}.layer_norm.bias"] = np.zeros((c_m,), dtype=np.float32)
    state[f"{opm}.linear_1.weight"] = np.ones((2, c_m), dtype=np.float32)
    state[f"{opm}.linear_2.weight"] = np.ones((2, c_m), dtype=np.float32)
    state[f"{opm}.linear_out.weight"] = np.ones((c_z, 4), dtype=np.float32)
    state[f"{opm}.linear_out.bias"] = np.zeros((c_z,), dtype=np.float32)
    _add_min_pairformer_block_state(state, f"{prefix}.pair_stack", c_z)


def _add_min_pairformer_block_state(
    state: dict[str, np.ndarray],
    prefix: str,
    c_z: int,
    *,
    c_s: int | None = None,
) -> None:
    for block_name in ("tri_mul_out", "tri_mul_in"):
        tri_prefix = f"{prefix}.{block_name}"
        state[f"{tri_prefix}.layer_norm_in.weight"] = np.ones((c_z,), dtype=np.float32)
        state[f"{tri_prefix}.layer_norm_in.bias"] = np.zeros((c_z,), dtype=np.float32)
        state[f"{tri_prefix}.layer_norm_out.weight"] = np.ones((c_z,), dtype=np.float32)
        state[f"{tri_prefix}.layer_norm_out.bias"] = np.zeros((c_z,), dtype=np.float32)
        for name in ("linear_a_p", "linear_a_g", "linear_b_p", "linear_b_g"):
            state[f"{tri_prefix}.{name}.weight"] = np.ones((c_z, c_z), dtype=np.float32)
        state[f"{tri_prefix}.linear_z.weight"] = np.ones((c_z, c_z), dtype=np.float32)
        state[f"{tri_prefix}.linear_g.weight"] = np.ones((c_z, c_z), dtype=np.float32)
    for block_name in ("tri_att_start", "tri_att_end"):
        att_prefix = f"{prefix}.{block_name}"
        state[f"{att_prefix}.layer_norm.weight"] = np.ones((c_z,), dtype=np.float32)
        state[f"{att_prefix}.layer_norm.bias"] = np.zeros((c_z,), dtype=np.float32)
        state[f"{att_prefix}.linear.weight"] = np.ones((1, c_z), dtype=np.float32)
        for name in ("linear_q", "linear_k", "linear_v", "linear_o", "linear_g"):
            state[f"{att_prefix}.mha.{name}.weight"] = np.ones(
                (c_z, c_z), dtype=np.float32
            )
    trans_prefix = f"{prefix}.pair_transition"
    state[f"{trans_prefix}.layernorm1.weight"] = np.ones((c_z,), dtype=np.float32)
    state[f"{trans_prefix}.layernorm1.bias"] = np.zeros((c_z,), dtype=np.float32)
    state[f"{trans_prefix}.linear_no_bias_a.weight"] = np.ones(
        (4 * c_z, c_z), dtype=np.float32
    )
    state[f"{trans_prefix}.linear_no_bias_b.weight"] = np.ones(
        (4 * c_z, c_z), dtype=np.float32
    )
    state[f"{trans_prefix}.linear_no_bias.weight"] = np.ones(
        (c_z, 4 * c_z), dtype=np.float32
    )
    if c_s is not None:
        att_prefix = f"{prefix}.attention_pair_bias"
        state[f"{att_prefix}.layernorm_a.weight"] = np.ones((c_s,), dtype=np.float32)
        state[f"{att_prefix}.layernorm_a.bias"] = np.zeros((c_s,), dtype=np.float32)
        state[f"{att_prefix}.layernorm_z.weight"] = np.ones((c_z,), dtype=np.float32)
        state[f"{att_prefix}.layernorm_z.bias"] = np.zeros((c_z,), dtype=np.float32)
        state[f"{att_prefix}.linear_nobias_z.weight"] = np.ones(
            (1, c_z), dtype=np.float32
        )
        for name in ("linear_q", "linear_k", "linear_v", "linear_o", "linear_g"):
            state[f"{att_prefix}.attention.{name}.weight"] = np.ones(
                (c_s, c_s), dtype=np.float32
            )
        state[f"{att_prefix}.attention.linear_q.bias"] = np.zeros(
            (c_s,), dtype=np.float32
        )
        _add_transition_state_for_test(state, f"{prefix}.single_transition", c_s)


def _add_transition_state_for_test(
    state: dict[str, np.ndarray],
    prefix: str,
    c_in: int,
) -> None:
    state[f"{prefix}.layernorm1.weight"] = np.ones((c_in,), dtype=np.float32)
    state[f"{prefix}.layernorm1.bias"] = np.zeros((c_in,), dtype=np.float32)
    state[f"{prefix}.linear_no_bias_a.weight"] = np.ones(
        (4 * c_in, c_in), dtype=np.float32
    )
    state[f"{prefix}.linear_no_bias_b.weight"] = np.ones(
        (4 * c_in, c_in), dtype=np.float32
    )
    state[f"{prefix}.linear_no_bias.weight"] = np.ones(
        (c_in, 4 * c_in), dtype=np.float32
    )
