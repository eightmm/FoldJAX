from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.protenix.models.diffusion.atom import (
    AtomAttentionDecoderParams,
    AtomAttentionEncoderCacheParams,
    AtomAttentionEncoderParams,
    AtomPairMlpParams,
)
from foldjax.models.protenix.models.diffusion.diffusion import (
    DiffusionConditioningParams,
    DiffusionModuleParams,
)
from foldjax.models.protenix.models.diffusion.transformer import (
    ConditionedTransitionParams,
    DiffusionTransformerBlockParams,
    DiffusionTransformerStackParams,
)
from foldjax.models.protenix.models.heads.confidence import (
    ConfidenceDistanceEmbeddingParams,
    ConfidenceHeadParams,
    ConfidenceOutputParams,
)
from foldjax.models.protenix.models.heads.head import DistogramParams
from foldjax.models.protenix.models.model import (
    ProtenixInferenceParams,
    cast_trunk_params,
    protenix_infer_static,
)
from foldjax.models.protenix.models.primitives.attention import (
    AttentionPairBiasParams,
    AttentionParams,
)
from foldjax.models.protenix.models.primitives.primitives import (
    AdaptiveLayerNormParams,
    LayerNormParams,
    LinearParams,
    TransitionParams,
)
from foldjax.models.protenix.models.trunk_blocks.embedders import (
    ConstraintEmbedderParams,
    FourierParams,
    InputFeatureEmbedderParams,
    RelativePositionParams,
    constraint_embedder,
)
from foldjax.models.protenix.models.trunk_blocks.msa import MSAModuleParams
from foldjax.models.protenix.models.trunk_blocks.pairformer import PairformerStackParams
from foldjax.models.protenix.models.trunk_blocks.template import TemplateEmbedderParams
from foldjax.models.protenix.models.trunk_blocks.trunk import (
    PairformerOutputParams,
    RecyclingProjectionParams,
    TrunkInitializationParams,
    TrunkParams,
)


def test_protenix_infer_static_returns_core_outputs() -> None:
    params = _toy_params()
    features = _toy_features()
    noise_schedule = jnp.asarray([1.0, 0.0], dtype=jnp.float32)
    init_noise = jnp.ones((1, 3, 3), dtype=jnp.float32)
    step_noise = jnp.zeros_like(init_noise)

    out = protenix_infer_static(
        features,
        params,
        noise_schedule,
        key=None,
        n_sample=1,
        init_noise=init_noise,
        step_noises=(step_noise,),
        n_cycle=1,
        input_atom_heads=1,
        atom_encoder_heads=1,
        token_heads=1,
        atom_decoder_heads=1,
        n_queries=2,
        n_keys=4,
        sigma_data=4.0,
        centre_each_step=False,
    )

    assert out["s_inputs"].shape == (2, 67)
    assert out["s_trunk"].shape == (2, 2)
    assert out["z_trunk"].shape == (2, 2, 2)
    assert out["coordinate"].shape == (1, 3, 3)
    assert out["distogram_logits"].shape == (2, 2, 3)
    assert out["plddt"].shape == (1, 3, 2)
    assert out["pae"].shape == (1, 2, 2, 1)
    assert out["pde"].shape == (1, 2, 2, 1)
    assert out["resolved"].shape == (1, 3, 1)
    assert out["atom_plddt"].shape == (1, 3)
    assert out["token_pair_pae"].shape == (1, 2, 2)
    assert out["token_pair_pde"].shape == (1, 2, 2)
    assert out["contact_probs"].shape == (2, 2)
    assert out["summary_plddt"].shape == (1,)
    assert out["summary_gpde"].shape == (1,)
    assert out["summary_ptm"].shape == (1,)
    assert out["summary_iptm"].shape == (1,)
    assert out["summary_ranking_score"].shape == (1,)
    assert out["has_clash"].shape == (1,)
    assert out["has_vdw_clash"].shape == (1,)
    assert out["summary_ranking_score_vdw_penalized"].shape == (1,)
    assert out["chain_plddt"].shape == (1, 2)
    assert out["chain_pair_plddt"].shape == (1, 2, 2)
    assert out["chain_pair_pae_mean"].shape == (1, 2, 2)
    assert out["chain_pair_pae_min"].shape == (1, 2, 2)
    assert out["chain_gpde"].shape == (1, 2)
    assert out["chain_pair_gpde"].shape == (1, 2, 2)
    np.testing.assert_allclose(
        np.asarray(out["coordinate"]),
        0.91176474,
        atol=1e-5,
    )
    np.testing.assert_allclose(np.asarray(out["summary_plddt"]), 50.0, atol=1e-6)


def test_constraint_embedder_rejects_feature_without_matching_weights() -> None:
    params = ConstraintEmbedderParams(
        pocket_z=None,
        contact_z=None,
        contact_atom_z=None,
        substructure_z=None,
    )

    with pytest.raises(ValueError, match="contact.*no matching embedder weights"):
        constraint_embedder(
            {"contact": jnp.ones((2, 2, 1), dtype=jnp.float32)}, params
        )


def test_constraint_embedder_uses_only_present_weighted_channels() -> None:
    params = ConstraintEmbedderParams(
        pocket_z=LinearParams(weight=jnp.ones((2, 1)), bias=None),
        contact_z=LinearParams(weight=jnp.ones((2, 1)), bias=None),
        contact_atom_z=None,
        substructure_z=None,
    )

    embedded = constraint_embedder(
        {
            "contact": jnp.ones((2, 2, 1), dtype=jnp.float32),
            "contact_atom_index_pairs": jnp.zeros((0, 2), dtype=jnp.int32),
            "contact_atom_token_pairs": jnp.zeros((0, 2), dtype=jnp.int32),
        },
        params,
    )

    assert embedded.shape == (2, 2, 2)


def test_cast_trunk_params_preserves_fp32_diffusion_island() -> None:
    params = _toy_params()

    mixed = cast_trunk_params(params, jnp.bfloat16)

    trunk_float_leaves = [
        leaf
        for leaf in jax.tree.leaves((mixed.input_embedder, mixed.pairformer_output))
        if hasattr(leaf, "dtype") and jnp.issubdtype(leaf.dtype, jnp.floating)
    ]
    diffusion_float_leaves = [
        leaf
        for leaf in jax.tree.leaves(mixed.diffusion)
        if hasattr(leaf, "dtype") and jnp.issubdtype(leaf.dtype, jnp.floating)
    ]
    assert trunk_float_leaves
    assert all(leaf.dtype == jnp.bfloat16 for leaf in trunk_float_leaves)
    assert diffusion_float_leaves
    assert all(leaf.dtype == jnp.float32 for leaf in diffusion_float_leaves)


def test_protenix_infer_sampler_scan_matches_loop() -> None:
    params = _toy_params()
    features = _toy_features()
    noise_schedule = jnp.asarray([1.0, 0.5, 0.0], dtype=jnp.float32)
    init_noise = jnp.ones((1, 3, 3), dtype=jnp.float32)
    step_noises = (jnp.zeros_like(init_noise), jnp.ones_like(init_noise) * 0.1)
    kwargs = {
        "key": None,
        "n_sample": 1,
        "init_noise": init_noise,
        "step_noises": step_noises,
        "n_cycle": 1,
        "input_atom_heads": 1,
        "atom_encoder_heads": 1,
        "token_heads": 1,
        "atom_decoder_heads": 1,
        "n_queries": 2,
        "n_keys": 4,
        "sigma_data": 4.0,
        "centre_each_step": False,
        "run_confidence": False,
    }

    loop = protenix_infer_static(
        features,
        params,
        noise_schedule,
        use_sampler_scan=False,
        **kwargs,
    )
    scanned = protenix_infer_static(
        features,
        params,
        noise_schedule,
        use_sampler_scan=True,
        **kwargs,
    )

    np.testing.assert_allclose(
        scanned["coordinate"], loop["coordinate"], rtol=1e-5, atol=1e-5
    )


def test_protenix_infer_diffusion_efficient_fusion_matches_standard() -> None:
    params = _toy_params()
    features = _toy_features()
    noise_schedule = jnp.asarray([1.0, 0.5, 0.0], dtype=jnp.float32)
    init_noise = jnp.ones((1, 3, 3), dtype=jnp.float32)
    step_noises = (jnp.zeros_like(init_noise), jnp.ones_like(init_noise) * 0.1)
    kwargs = {
        "key": None,
        "n_sample": 1,
        "init_noise": init_noise,
        "step_noises": step_noises,
        "n_cycle": 1,
        "input_atom_heads": 1,
        "atom_encoder_heads": 1,
        "token_heads": 1,
        "atom_decoder_heads": 1,
        "n_queries": 2,
        "n_keys": 4,
        "sigma_data": 4.0,
        "centre_each_step": False,
        "run_confidence": False,
    }

    standard = protenix_infer_static(
        features,
        params,
        noise_schedule,
        use_diffusion_efficient_fusion=False,
        **kwargs,
    )
    fused = protenix_infer_static(
        features,
        params,
        noise_schedule,
        use_diffusion_efficient_fusion=True,
        **kwargs,
    )

    np.testing.assert_allclose(
        fused["coordinate"], standard["coordinate"], rtol=1e-5, atol=1e-5
    )


def test_protenix_infer_denoiser_jit_matches_eager() -> None:
    params = _toy_params()
    features = _toy_features()
    noise_schedule = jnp.asarray([1.0, 0.5, 0.0], dtype=jnp.float32)
    init_noise = jnp.ones((1, 3, 3), dtype=jnp.float32)
    step_noises = (jnp.zeros_like(init_noise), jnp.ones_like(init_noise) * 0.1)
    kwargs = {
        "key": None,
        "n_sample": 1,
        "init_noise": init_noise,
        "step_noises": step_noises,
        "n_cycle": 1,
        "input_atom_heads": 1,
        "atom_encoder_heads": 1,
        "token_heads": 1,
        "atom_decoder_heads": 1,
        "n_queries": 2,
        "n_keys": 4,
        "sigma_data": 4.0,
        "centre_each_step": False,
        "run_confidence": False,
    }

    eager = protenix_infer_static(
        features,
        params,
        noise_schedule,
        use_denoiser_jit=False,
        **kwargs,
    )
    compiled = protenix_infer_static(
        features,
        params,
        noise_schedule,
        use_denoiser_jit=True,
        **kwargs,
    )

    np.testing.assert_allclose(
        compiled["coordinate"], eager["coordinate"], rtol=1e-5, atol=1e-5
    )


def _toy_features() -> dict[str, jnp.ndarray | dict[str, jnp.ndarray]]:
    atom_to_token_idx = jnp.asarray([0, 1, 1])
    _, _, pad_info = _dense_atom_layout(atom_to_token_idx)
    return {
        "atom_to_token_idx": atom_to_token_idx,
        "ref_pos": jnp.zeros((3, 3), dtype=jnp.float32),
        "ref_charge": jnp.zeros((3,), dtype=jnp.float32),
        "ref_mask": jnp.ones((3,), dtype=jnp.float32),
        "ref_atom_name_chars": jnp.zeros((3, 4, 64), dtype=jnp.float32),
        "ref_element": jnp.zeros((3, 128), dtype=jnp.float32),
        "d_lm": jnp.zeros((2, 2, 4, 3), dtype=jnp.float32),
        "v_lm": jnp.ones((2, 2, 4, 1), dtype=jnp.float32),
        "pad_info": pad_info,
        "restype": jnp.zeros((2, 32), dtype=jnp.float32),
        "profile": jnp.zeros((2, 32), dtype=jnp.float32),
        "deletion_mean": jnp.zeros((2,), dtype=jnp.float32),
        "relp": jnp.zeros((2, 2, 2), dtype=jnp.float32),
        "token_bonds": jnp.zeros((2, 2), dtype=jnp.float32),
        "has_frame": jnp.asarray([1, 1]),
        "asym_id": jnp.asarray([0, 1]),
        "distogram_rep_atom_mask": jnp.asarray([1, 1, 0]),
        "atom_to_tokatom_idx": jnp.asarray([0, 0, 1]),
    }


def _toy_params() -> ProtenixInferenceParams:
    c_atom = 2
    c_atompair = 2
    c_token = 2
    c_s = 2
    c_z = 2
    c_s_inputs = 67
    atom_block = _zero_atom_block(c_atom=c_atom, c_atompair=c_atompair)
    atom_encoder_input = _atom_encoder_params(
        c_atom=c_atom,
        c_atompair=c_atompair,
        c_token=c_token,
        atom_block=atom_block,
        has_coords=False,
        c_s=c_s,
        c_z=c_z,
    )
    atom_encoder_diffusion = _atom_encoder_params(
        c_atom=c_atom,
        c_atompair=c_atompair,
        c_token=c_token,
        atom_block=atom_block,
        has_coords=True,
        c_s=c_s,
        c_z=c_z,
    )
    return ProtenixInferenceParams(
        input_embedder=InputFeatureEmbedderParams(atom_encoder=atom_encoder_input),
        pairformer_output=_pairformer_output_params(
            c_s_inputs=c_s_inputs,
            c_s=c_s,
            c_z=c_z,
        ),
        diffusion=DiffusionModuleParams(
            conditioning=_diffusion_conditioning_params(
                c_z=c_z,
                c_s=c_s,
                c_s_inputs=c_s_inputs,
            ),
            atom_encoder=atom_encoder_diffusion,
            layernorm_s=LayerNormParams(weight=jnp.ones((c_s,)), bias=None),
            linear_s=LinearParams(weight=jnp.zeros((c_token, c_s)), bias=None),
            diffusion_transformer=DiffusionTransformerStackParams(blocks=(atom_block,)),
            layernorm_a=LayerNormParams(weight=jnp.ones((c_token,)), bias=None),
            atom_decoder=AtomAttentionDecoderParams(
                linear_a=LinearParams(weight=jnp.zeros((c_atom, c_token)), bias=None),
                layernorm_q=LayerNormParams(weight=jnp.ones((c_atom,)), bias=None),
                linear_out=LinearParams(weight=jnp.zeros((3, c_atom)), bias=None),
                atom_transformer=DiffusionTransformerStackParams(blocks=(atom_block,)),
            ),
        ),
        distogram=DistogramParams(
            linear=LinearParams(weight=jnp.zeros((3, c_z)), bias=jnp.zeros((3,)))
        ),
        confidence=_confidence_params(
            c_s_inputs=c_s_inputs,
            c_s=c_s,
            c_z=c_z,
        ),
    )


def _atom_encoder_params(
    *,
    c_atom: int,
    c_atompair: int,
    c_token: int,
    atom_block: DiffusionTransformerBlockParams,
    has_coords: bool,
    c_s: int,
    c_z: int,
) -> AtomAttentionEncoderParams:
    return AtomAttentionEncoderParams(
        cache=AtomAttentionEncoderCacheParams(
            linear_ref_pos=LinearParams(weight=jnp.zeros((c_atom, 3)), bias=None),
            linear_ref_charge=LinearParams(weight=jnp.zeros((c_atom, 1)), bias=None),
            linear_f=LinearParams(weight=jnp.zeros((c_atom, 385)), bias=None),
            linear_d=LinearParams(weight=jnp.zeros((c_atompair, 3)), bias=None),
            linear_invd=LinearParams(weight=jnp.zeros((c_atompair, 1)), bias=None),
            linear_v=LinearParams(weight=jnp.zeros((c_atompair, 1)), bias=None),
        ),
        linear_cl=LinearParams(weight=jnp.zeros((c_atompair, c_atom)), bias=None),
        linear_cm=LinearParams(weight=jnp.zeros((c_atompair, c_atom)), bias=None),
        small_mlp=AtomPairMlpParams(
            linear_1=LinearParams(
                weight=jnp.zeros((c_atompair, c_atompair)),
                bias=None,
            ),
            linear_2=LinearParams(
                weight=jnp.zeros((c_atompair, c_atompair)),
                bias=None,
            ),
            linear_3=LinearParams(
                weight=jnp.zeros((c_atompair, c_atompair)),
                bias=None,
            ),
        ),
        atom_transformer=DiffusionTransformerStackParams(blocks=(atom_block,)),
        linear_q=LinearParams(weight=jnp.zeros((c_token, c_atom)), bias=None),
        layernorm_s=LayerNormParams(weight=jnp.ones((c_s,)), bias=None)
        if has_coords
        else None,
        linear_s=LinearParams(weight=jnp.zeros((c_atom, c_s)), bias=None)
        if has_coords
        else None,
        layernorm_z=LayerNormParams(weight=jnp.ones((c_z,)), bias=None)
        if has_coords
        else None,
        linear_z=LinearParams(weight=jnp.zeros((c_atompair, c_z)), bias=None)
        if has_coords
        else None,
        linear_r=LinearParams(weight=jnp.zeros((c_atom, 3)), bias=None)
        if has_coords
        else None,
    )


def _pairformer_output_params(
    *,
    c_s_inputs: int,
    c_s: int,
    c_z: int,
) -> PairformerOutputParams:
    return PairformerOutputParams(
        trunk=TrunkParams(
            initial=TrunkInitializationParams(
                linear_sinit=LinearParams(
                    weight=jnp.zeros((c_s, c_s_inputs)),
                    bias=None,
                ),
                linear_zinit1=LinearParams(weight=jnp.zeros((c_z, c_s)), bias=None),
                linear_zinit2=LinearParams(weight=jnp.zeros((c_z, c_s)), bias=None),
                relative_position=RelativePositionParams(
                    linear_no_bias=LinearParams(weight=jnp.zeros((c_z, 2)), bias=None)
                ),
                linear_token_bond=LinearParams(weight=jnp.zeros((c_z, 1)), bias=None),
            ),
            recycling=RecyclingProjectionParams(
                layernorm_z=LayerNormParams(weight=jnp.ones((c_z,)), bias=None),
                linear_z=LinearParams(weight=jnp.zeros((c_z, c_z)), bias=None),
                layernorm_s=LayerNormParams(weight=jnp.ones((c_s,)), bias=None),
                linear_s=LinearParams(weight=jnp.zeros((c_s, c_s)), bias=None),
            ),
        ),
        constraint=None,
        template=_template_params(c_z),
        msa=MSAModuleParams(
            linear_m=LinearParams(weight=jnp.zeros((c_s, 34)), bias=None),
            linear_s=LinearParams(weight=jnp.zeros((c_s, c_s_inputs)), bias=None),
            blocks=(),
        ),
        pairformer_stack=PairformerStackParams(blocks=()),
    )


def _template_params(c_z: int) -> TemplateEmbedderParams:
    return TemplateEmbedderParams(
        linear_z=LinearParams(weight=jnp.zeros((2, c_z)), bias=None),
        layernorm_z=LayerNormParams(weight=jnp.ones((c_z,)), bias=None),
        linear_a=LinearParams(weight=jnp.zeros((2, 108)), bias=None),
        pairformer_stack=PairformerStackParams(blocks=()),
        layernorm_v=LayerNormParams(weight=jnp.ones((2,)), bias=None),
        linear_u=LinearParams(weight=jnp.zeros((c_z, 2)), bias=None),
    )


def _diffusion_conditioning_params(
    *,
    c_z: int,
    c_s: int,
    c_s_inputs: int,
) -> DiffusionConditioningParams:
    return DiffusionConditioningParams(
        relpe=RelativePositionParams(
            linear_no_bias=LinearParams(weight=jnp.zeros((c_z, 2)), bias=None)
        ),
        layernorm_z=LayerNormParams(weight=jnp.ones((2 * c_z,)), bias=None),
        linear_z=LinearParams(weight=jnp.zeros((c_z, 2 * c_z)), bias=None),
        transition_z1=_zero_transition(c_z),
        transition_z2=_zero_transition(c_z),
        layernorm_s=LayerNormParams(weight=jnp.ones((c_s + c_s_inputs,)), bias=None),
        linear_s=LinearParams(weight=jnp.zeros((c_s, c_s + c_s_inputs)), bias=None),
        fourier=FourierParams(w=jnp.zeros((4,)), b=jnp.zeros((4,))),
        layernorm_n=LayerNormParams(weight=jnp.ones((4,)), bias=None),
        linear_n=LinearParams(weight=jnp.zeros((c_s, 4)), bias=None),
        transition_s1=_zero_transition(c_s),
        transition_s2=_zero_transition(c_s),
    )


def _confidence_params(
    *,
    c_s_inputs: int,
    c_s: int,
    c_z: int,
) -> ConfidenceHeadParams:
    return ConfidenceHeadParams(
        input_strunk_ln=LayerNormParams(
            weight=jnp.ones((c_s,)),
            bias=jnp.zeros((c_s,)),
        ),
        linear_s1=LinearParams(weight=jnp.zeros((c_z, c_s_inputs)), bias=None),
        linear_s2=LinearParams(weight=jnp.zeros((c_z, c_s_inputs)), bias=None),
        distance_embedding=ConfidenceDistanceEmbeddingParams(
            lower_bins=jnp.asarray([0.0]),
            upper_bins=jnp.asarray([10.0]),
            linear_d=LinearParams(weight=jnp.zeros((c_z, 1)), bias=None),
            linear_d_wo_onehot=LinearParams(weight=jnp.zeros((c_z, 1)), bias=None),
        ),
        pairformer_stack=PairformerStackParams(blocks=()),
        output=ConfidenceOutputParams(
            pae_ln=LayerNormParams(weight=jnp.ones((c_z,)), bias=jnp.zeros((c_z,))),
            pde_ln=LayerNormParams(weight=jnp.ones((c_z,)), bias=jnp.zeros((c_z,))),
            plddt_ln=LayerNormParams(weight=jnp.ones((c_s,)), bias=jnp.zeros((c_s,))),
            resolved_ln=LayerNormParams(
                weight=jnp.ones((c_s,)),
                bias=jnp.zeros((c_s,)),
            ),
            linear_pae=LinearParams(weight=jnp.zeros((1, c_z)), bias=None),
            linear_pde=LinearParams(weight=jnp.zeros((1, c_z)), bias=None),
            plddt_weight=jnp.zeros((2, c_s, 2)),
            resolved_weight=jnp.zeros((2, c_s, 1)),
        ),
    )


def _zero_atom_block(
    *,
    c_atom: int,
    c_atompair: int,
) -> DiffusionTransformerBlockParams:
    zero_atom = LinearParams(weight=jnp.zeros((c_atom, c_atom)), bias=None)
    zero_atom_bias = LinearParams(
        weight=jnp.zeros((c_atom, c_atom)),
        bias=jnp.zeros((c_atom,)),
    )
    adaln = AdaptiveLayerNormParams(
        layernorm_a=LayerNormParams(weight=None, bias=None),
        layernorm_s=LayerNormParams(weight=jnp.ones((c_atom,)), bias=None),
        linear_s=zero_atom_bias,
        linear_no_bias_s=zero_atom,
    )
    return DiffusionTransformerBlockParams(
        attention_pair_bias=AttentionPairBiasParams(
            layernorm_a=adaln,
            layernorm_kv=adaln,
            attention=AttentionParams(
                linear_q=zero_atom_bias,
                linear_k=zero_atom,
                linear_v=zero_atom,
                linear_o=zero_atom,
                linear_g=zero_atom,
            ),
            layernorm_z=LayerNormParams(weight=jnp.ones((c_atompair,)), bias=None),
            linear_z=LinearParams(weight=jnp.zeros((1, c_atompair)), bias=None),
            linear_a_last=zero_atom_bias,
            has_s=True,
            cross_attention_mode=True,
        ),
        conditioned_transition=ConditionedTransitionParams(
            adaln=adaln,
            linear_a1=LinearParams(weight=jnp.zeros((2 * c_atom, c_atom)), bias=None),
            linear_a2=LinearParams(weight=jnp.zeros((2 * c_atom, c_atom)), bias=None),
            linear_b=LinearParams(weight=jnp.zeros((c_atom, 2 * c_atom)), bias=None),
            linear_s=zero_atom_bias,
        ),
    )


def _zero_transition(c_in: int) -> TransitionParams:
    hidden = 2 * c_in
    return TransitionParams(
        layer_norm=LayerNormParams(weight=jnp.ones((c_in,)), bias=None),
        linear_a=LinearParams(weight=jnp.zeros((hidden, c_in)), bias=None),
        linear_b=LinearParams(weight=jnp.zeros((hidden, c_in)), bias=None),
        linear_out=LinearParams(weight=jnp.zeros((c_in, hidden)), bias=None),
    )


def _dense_atom_layout(
    atom_to_token_idx: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, dict[str, jnp.ndarray]]:
    del atom_to_token_idx
    d_lm = jnp.zeros((2, 2, 4, 3), dtype=jnp.float32)
    v_lm = jnp.ones((2, 2, 4, 1), dtype=jnp.float32)
    return d_lm, v_lm, {"mask_trunked": jnp.ones((2, 2, 4), dtype=bool)}
