"""The two policy tables, and the dtypes they actually produce.

There are two: upstream's token gate (``--amp-policy upstream``) and the
port's released default (``auto``), which narrows the confidence head at every
size and keeps upstream's 3,840 gate on the diffusion sampler. Both resolvers
are arithmetic and are pinned at their boundaries. Everything below that reads
the traced program or records the operands a stage was handed, never the
option string that produced them: a test asserting that ``--amp-policy bf16``
was requested proves nothing about whether a matmul narrowed, which is the
whole property this feature delivers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.backends import protenix as backend_impl
from foldjax.backends.protenix import ProtenixBackend
from foldjax.models.protenix.amp_policy import (
    AMP_POLICY_CHOICES,
    DEFAULT_AMP_POLICY,
    AmpPolicy,
    amp_policy_for_tokens,
    default_amp_policy_for_tokens,
    realise_amp_policy,
    requested_amp_policy,
)
from foldjax.models.protenix.cli import predict as predict_cli
from foldjax.models.protenix.models import model as model_module
from foldjax.models.protenix.models import predict as predict_impl
from foldjax.models.protenix.models.diffusion import diffusion as diffusion_module
from foldjax.models.protenix.models.heads import confidence as confidence_module
from foldjax.models.protenix.models.heads.confidence import (
    ConfidenceDistanceEmbeddingParams,
    can_compact_confidence_distance_embedding,
)
from foldjax.models.protenix.models.input_precision import (
    native_confidence_autocast_params,
    native_diffusion_autocast_params,
)
from foldjax.models.protenix.models.model import (
    cast_trunk_params,
    protenix_infer_static,
)
from foldjax.models.protenix.models.primitives.primitives import (
    AutocastLinearF32OutParams,
    AutocastLinearParams,
    Fp32PrecisionLinearParams,
    LinearParams,
    linear,
)
from foldjax.models.protenix.models.trunk_blocks.pairformer import PairformerStackParams
from foldjax.schema import PredictionRequest

from .test_model import _toy_features, _toy_params
from .test_trunk import _zero_pairformer_block

# ---------------------------------------------------------------- resolver --


@pytest.mark.parametrize(
    ("n_token", "expected"),
    [
        # Both bounds are `>` upstream, so the threshold value itself keeps the
        # lower stage's policy. Getting that backwards would put every job at
        # exactly 2560 tokens on a different program than upstream builds.
        (2560, AmpPolicy(False, False)),
        (2561, AmpPolicy(True, False)),
        (3840, AmpPolicy(True, False)),
        (3841, AmpPolicy(True, True)),
    ],
)
def test_the_gate_turns_at_the_upstream_thresholds(n_token, expected) -> None:
    """`amp_policy_for_tokens` is upstream's table and only upstream's table.

    It is no longer what `auto` resolves; it stays because `--amp-policy
    upstream` has to reproduce a native capture's configuration at any size.
    """
    assert amp_policy_for_tokens(n_token) == expected
    assert requested_amp_policy("upstream", n_token) == expected


@pytest.mark.parametrize(
    ("n_token", "expected"),
    [
        # The confidence half no longer turns: it is on from zero tokens up.
        (0, AmpPolicy(True, False)),
        (2560, AmpPolicy(True, False)),
        (2561, AmpPolicy(True, False)),
        # The diffusion half keeps upstream's threshold, and keeps it exclusive.
        (3840, AmpPolicy(True, False)),
        (3841, AmpPolicy(True, True)),
    ],
)
def test_the_released_default_moves_only_the_confidence_half(n_token, expected) -> None:
    """`auto` narrows the head everywhere and leaves 3,840 where it was.

    Below 2,561 tokens this is the whole difference from upstream, and the
    pair at 3840/3841 is the half that must not have moved: that stage owns
    the coordinates.
    """
    assert default_amp_policy_for_tokens(n_token) == expected
    assert requested_amp_policy("auto", n_token) == expected


def test_the_diffusion_threshold_has_exactly_one_owner() -> None:
    """Both tables read `DIFFUSION_AUTOCAST_ABOVE_TOKENS`, never two copies.

    Re-spelling the surviving half of the gate in the new resolver is how the
    two tables would drift apart on the stage they agree about.
    """
    for n_token in (0, 2560, 2561, 3840, 3841, 8192):
        assert (
            default_amp_policy_for_tokens(n_token).diffusion_autocast
            == amp_policy_for_tokens(n_token).diffusion_autocast
        )


def test_protenix_v2_runs_its_confidence_head_under_autocast_at_every_size() -> None:
    """Upstream's `else` branch, which only reaches sizes below the gate."""
    assert amp_policy_for_tokens(64, "protenix-v2") == AmpPolicy(True, False)
    assert amp_policy_for_tokens(2560, "protenix-v2") == AmpPolicy(True, False)
    # Above the threshold the model name stops mattering: every variant is
    # already on the same branch.
    assert amp_policy_for_tokens(2561, "protenix-v2") == amp_policy_for_tokens(2561)


def test_upstream_keeps_the_base_models_confidence_head_fp32_below_the_gate() -> None:
    for name in (None, "protenix_base_default_v1.0.0", "protenix_mini_esm_v0.5.0"):
        assert amp_policy_for_tokens(2560, name) == AmpPolicy(False, False)


def test_the_model_name_stops_deciding_the_confidence_half_under_auto() -> None:
    """`protenix-v2` was the one variant already narrowing below the gate.

    Under the released default every variant does, so the name can only still
    reach the diffusion half -- and there it never mattered.
    """
    for name in (
        None,
        "protenix-v2",
        "protenix_base_default_v1.0.0",
        "protenix_mini_esm_v0.5.0",
    ):
        assert default_amp_policy_for_tokens(2560, name) == AmpPolicy(True, False)
        assert default_amp_policy_for_tokens(4096, name) == AmpPolicy(True, True)


def test_the_resolver_does_not_own_the_protenix_v2_token_limit() -> None:
    """`runtime_policy.PROTENIX_V2_MAX_TOKENS` owns it; two owners drift apart."""
    assert amp_policy_for_tokens(4096, "protenix-v2") == AmpPolicy(True, True)


def test_pinned_policies_ignore_the_token_count() -> None:
    for n_token in (16, 2560, 2561, 4096):
        assert requested_amp_policy("fp32", n_token) == AmpPolicy(False, False)
        assert requested_amp_policy("bf16", n_token) == AmpPolicy(True, True)


def test_auto_is_the_default_and_upstreams_gate_keeps_a_spelling() -> None:
    """Four values, and the one a parity run needs is still reachable."""
    assert DEFAULT_AMP_POLICY == "auto"
    assert set(AMP_POLICY_CHOICES) == {"auto", "upstream", "fp32", "bf16"}
    for n_token in (16, 2561, 4096):
        assert requested_amp_policy("auto", n_token) == default_amp_policy_for_tokens(
            n_token
        )
        assert requested_amp_policy("upstream", n_token) == amp_policy_for_tokens(
            n_token
        )
    # Below the gate the two spellings are the change this default made.
    assert requested_amp_policy("auto", 2560) != requested_amp_policy("upstream", 2560)


def test_an_unknown_policy_names_the_ones_that_exist() -> None:
    with pytest.raises(ValueError, match="auto, upstream, fp32, bf16"):
        requested_amp_policy("bfloat16", 100)


def test_an_fp32_trunk_opens_no_autocast_for_a_stage_to_run_in() -> None:
    """Upstream's skip_amp only chooses whether to *leave* the ambient context.

    With `configs.dtype = fp32` there is no ambient context, so both stages run
    FP32 whatever the gate says, and `--amp-policy bf16` is not a way round an
    FP32 trunk.
    """
    for policy in (AmpPolicy(True, True), AmpPolicy(True, False)):
        assert realise_amp_policy(policy, trunk_is_bf16=False) == AmpPolicy(
            False, False
        )
        assert realise_amp_policy(policy, trunk_is_bf16=True) == policy


def test_the_label_is_readable_in_a_log_line() -> None:
    assert AmpPolicy(True, False).label() == "confidence=bf16 diffusion=fp32"
    assert AmpPolicy(False, True).label() == "confidence=fp32 diffusion=bf16"


# --------------------------------------------------------- projection nodes --


def _dot_operand_dtypes(jaxpr) -> list[tuple[str, ...]]:
    """Every `dot_general` in a closed jaxpr, as its two operand dtypes."""
    inner = jaxpr.jaxpr if hasattr(jaxpr, "jaxpr") else jaxpr
    return [
        tuple(str(var.aval.dtype) for var in eqn.invars[:2])
        for eqn in inner.eqns
        if eqn.primitive.name == "dot_general"
    ]


def test_an_fp32_precision_projection_computes_wide_and_returns_narrow() -> None:
    """The exempt projections are not "stay FP32": they narrow their result.

    Upstream's `Linear.forward` ends `.to(dtype=input_dtype)`, so a projection
    that opted out of autocast still hands BF16 to whatever follows it. Losing
    that cast would widen the whole denoiser downstream of the first geometry
    projection, which is how an autocast quietly stops being one.
    """
    weight = jnp.asarray(
        np.random.default_rng(0).normal(size=(4, 3)), dtype=jnp.float32
    )
    params = Fp32PrecisionLinearParams(weight)
    narrow = jnp.ones((2, 3), dtype=jnp.bfloat16)
    assert _dot_operand_dtypes(jax.make_jaxpr(lambda v: linear(v, params))(narrow)) == [
        ("float32", "float32")
    ]
    assert linear(narrow, params).dtype == jnp.bfloat16
    # FP32 in, FP32 out: under the FP32 policy this node does exactly the
    # arithmetic an ordinary LinearParams would have done, to the bit.
    wide = jnp.asarray(np.random.default_rng(2).normal(size=(2, 3)), jnp.float32)
    assert np.array_equal(
        np.asarray(linear(wide, params)),
        np.asarray(linear(wide, LinearParams(weight))),
    )


def test_an_autocast_projection_narrows_both_operands() -> None:
    params = AutocastLinearParams(
        jnp.asarray(np.random.default_rng(1).normal(size=(4, 3)), dtype=jnp.bfloat16)
    )
    assert _dot_operand_dtypes(
        jax.make_jaxpr(lambda v: linear(v, params))(jnp.ones((2, 3), jnp.float32))
    ) == [("bfloat16", "bfloat16")]


# ------------------------------------------------- realised parameter trees --


def _float_leaf_dtypes(tree) -> set[str]:
    return {
        str(leaf.dtype)
        for leaf in jax.tree.leaves(tree)
        if hasattr(leaf, "dtype") and jnp.issubdtype(leaf.dtype, jnp.floating)
    }


def test_the_confidence_output_stage_stays_fp32_under_every_policy() -> None:
    """Upstream runs it inside `autocast(enabled=False)` on upcast inputs.

    That block is FP32 at 76 tokens and at 3,012 alike, so rounding its weights
    would be a change the gate never asks for.
    """
    original = _toy_params().confidence
    realised = native_confidence_autocast_params(original)
    assert _float_leaf_dtypes(realised.output) == {"float32"}
    # The distance bins are compared against FP32 distances, not multiplied.
    assert realised.distance_embedding.lower_bins.dtype == jnp.float32
    assert realised.distance_embedding.upper_bins.dtype == jnp.float32


def test_the_confidence_distance_projections_become_autocast_projections() -> None:
    """They are reached with FP32: upstream computes cdist outside autocast."""
    realised = native_confidence_autocast_params(_toy_params().confidence)
    for name in ("linear_d", "linear_d_wo_onehot"):
        node = getattr(realised.distance_embedding, name)
        assert isinstance(node, AutocastLinearParams), name
        assert node.weight.dtype == jnp.bfloat16, name


def test_compact_binning_keeps_the_dense_paths_realised_dtype() -> None:
    """The compact bin projection must not widen what the dense one narrows.

    Both compiled wrappers resolve ``compact_distance_bins`` to True on any
    released checkpoint, so this is the path the policy actually runs. It
    promoted against the FP32 distances instead of following the weight, which
    handed the head an FP32 pair tensor under a fully realised BF16 policy and
    promoted every pairformer block after it -- invisible in the output and
    invisible to a test that traces only the dense path.
    """
    embedding = native_confidence_autocast_params(
        _toy_params().confidence
    ).distance_embedding
    coords = jnp.asarray(
        np.random.default_rng(3).normal(size=(4, 3)) * 8.0, jnp.float32
    )

    dense = confidence_module.confidence_distance_embedding(
        coords, embedding, compact_bins=False
    )
    compact = confidence_module.confidence_distance_embedding(
        coords, embedding, compact_bins=True
    )
    assert dense.dtype == jnp.bfloat16
    assert compact.dtype == dense.dtype
    np.testing.assert_array_equal(np.asarray(compact), np.asarray(dense))


def test_the_confidence_stack_takes_the_trunks_realisation() -> None:
    """BF16 activations reach it, so a BF16 weight is enough -- as in the trunk."""
    realised = native_confidence_autocast_params(_toy_params().confidence)
    assert realised.input_strunk_ln.weight.dtype == jnp.bfloat16


def test_the_outer_sum_initialiser_narrows_its_own_operands() -> None:
    """`s_inputs` reaches the head FP32, so a narrowed weight is not enough.

    The input embedder concatenates raw reference features that native
    autocast does not narrow, so its output is FP32 even under a BF16 trunk.
    A plain BF16 weight would promote this matmul back to FP32 and carry the
    pair tensor, and every block after it, along.
    """
    realised = native_confidence_autocast_params(_toy_params().confidence)
    for name in ("linear_s1", "linear_s2"):
        node = getattr(realised, name)
        assert isinstance(node, AutocastLinearParams), name
        assert node.weight.dtype == jnp.bfloat16, name


def test_realising_a_narrowed_tree_is_refused() -> None:
    """Widening rounded operands cannot recover what the rounding dropped."""
    already = jax.tree.map(lambda x: x.astype(jnp.bfloat16), _toy_params().confidence)
    with pytest.raises(ValueError, match="original FP32"):
        native_confidence_autocast_params(already)


def test_every_upstream_fp32_precision_projection_is_exempt_in_the_diffusion() -> None:
    """The ten `precision=torch.float32` projections, named individually.

    This is the list a `precision=` census over `protenix/` returns: six in
    `modules/transformer.py` (the atom encoder's geometry and conditioning
    projections plus the decoder's coordinate update) and four in
    `modules/diffusion.py`. Nothing else in the model constructs one.
    """
    realised = native_diffusion_autocast_params(_toy_params().diffusion)
    exempt = {
        "conditioning.linear_z": realised.conditioning.linear_z,
        "conditioning.linear_s": realised.conditioning.linear_s,
        "conditioning.linear_n": realised.conditioning.linear_n,
        "linear_s": realised.linear_s,
        "atom_encoder.cache.linear_ref_pos": realised.atom_encoder.cache.linear_ref_pos,
        "atom_encoder.cache.linear_d": realised.atom_encoder.cache.linear_d,
        "atom_encoder.linear_s": realised.atom_encoder.linear_s,
        "atom_encoder.linear_z": realised.atom_encoder.linear_z,
        "atom_encoder.linear_r": realised.atom_encoder.linear_r,
        "atom_decoder.linear_out": realised.atom_decoder.linear_out,
    }
    assert len(exempt) == 10
    for name, node in exempt.items():
        assert isinstance(node, Fp32PrecisionLinearParams), name
        assert node.weight.dtype == jnp.float32, name


def test_the_rest_of_the_diffusion_narrows_its_own_operands() -> None:
    """This subtree mixes FP32 geometry with BF16 conditioning.

    Unlike the confidence head it cannot inherit one dtype from its caller, so
    every non-exempt projection narrows its own operands rather than relying on
    whatever activation reaches it.
    """
    realised = native_diffusion_autocast_params(_toy_params().diffusion)
    for node in (
        realised.atom_encoder.linear_cl,
        realised.atom_encoder.cache.linear_ref_charge,
        realised.atom_encoder.cache.linear_f,
        realised.atom_decoder.linear_a,
    ):
        assert isinstance(node, AutocastLinearParams)
        assert node.weight.dtype == jnp.bfloat16


def test_every_denoiser_pair_bias_is_delivered_in_fp32() -> None:
    """The one result in the stage that is not rounded back down.

    All three denoiser stacks -- the token transformer and the two atom
    transformers -- project their per-head attention bias with a BF16 GEMM and
    keep the result FP32. Measured on 5DEI at 2,096 tokens: rounding it loses
    one chain of the homotetramer in every sample.
    """
    realised = native_diffusion_autocast_params(_toy_params().diffusion)
    stacks = {
        "diffusion_transformer": realised.diffusion_transformer,
        "atom_encoder.atom_transformer": realised.atom_encoder.atom_transformer,
        "atom_decoder.atom_transformer": realised.atom_decoder.atom_transformer,
    }
    seen = 0
    for name, stack in stacks.items():
        assert stack.blocks, name
        for index, block in enumerate(stack.blocks):
            node = block.attention_pair_bias.linear_z
            assert isinstance(node, AutocastLinearF32OutParams), f"{name}[{index}]"
            assert node.weight.dtype == jnp.bfloat16, f"{name}[{index}]"
            seen += 1
    assert seen == sum(len(stack.blocks) for stack in stacks.values())


def test_a_bf16_pair_bias_projection_returns_fp32() -> None:
    """The operands narrow, the result does not."""
    weight = jnp.arange(6, dtype=jnp.float32).reshape(2, 3) / 7.0
    x = jnp.arange(3, dtype=jnp.float32).reshape(1, 3) / 3.0
    narrowed = AutocastLinearParams(weight.astype(jnp.bfloat16))
    widened = AutocastLinearF32OutParams(weight.astype(jnp.bfloat16))
    assert linear(x, narrowed).dtype == jnp.bfloat16
    out = linear(x, widened)
    assert out.dtype == jnp.float32
    assert jnp.allclose(out.astype(jnp.bfloat16), linear(x, narrowed), atol=0)


def test_layer_norm_affine_values_are_left_alone_in_the_diffusion() -> None:
    """The port's layer norm quantizes them from the *input* dtype already.

    Upstream's `OpenFoldLayerNorm.forward` casts its affine operands to the
    input dtype and accumulates in FP32, which is what the port reproduces.
    Narrowing the stored copy would be the same arithmetic spelled twice.
    """
    realised = native_diffusion_autocast_params(_toy_params().diffusion)
    assert realised.layernorm_s.weight.dtype == jnp.float32
    assert realised.conditioning.layernorm_z.weight.dtype == jnp.float32


# ------------------------------------------------- realised stage operands --


def _bf16_trunk_params():
    return cast_trunk_params(_toy_params(), jnp.bfloat16)


def _infer(params, **kwargs):
    return protenix_infer_static(
        _toy_features(),
        params,
        jnp.asarray([1.0, 0.0], dtype=jnp.float32),
        key=None,
        num_samples=1,
        init_noise=jnp.ones((1, 3, 3), dtype=jnp.float32),
        step_noises=(jnp.zeros((1, 3, 3), dtype=jnp.float32),),
        num_recycles=1,
        input_atom_heads=1,
        atom_encoder_heads=1,
        token_heads=1,
        atom_decoder_heads=1,
        n_queries=2,
        n_keys=4,
        sigma_data=4.0,
        centre_each_step=False,
        **kwargs,
    )


@pytest.mark.parametrize("confidence_autocast", [False, True])
def test_the_confidence_stack_is_entered_in_the_policys_dtype(
    monkeypatch, confidence_autocast
) -> None:
    """Record what the pairformer stack was handed, not what was configured.

    The toy checkpoint has an empty stack, so a sentinel block turns the call
    on; the recorder replaces the stack itself, which is why the sentinel never
    has to be a real block. `seen` doubling as the tripwire is deliberate --
    a recorder that never fired would otherwise assert nothing at all.
    """
    seen: list[tuple[str, str]] = []

    def record(s_single, z_pair, *args, **kwargs):
        seen.append((str(s_single.dtype), str(z_pair.dtype)))
        return s_single, z_pair

    monkeypatch.setattr(confidence_module, "pairformer_stack", record)
    params = _bf16_trunk_params()
    confidence = params.confidence._replace(
        pairformer_stack=PairformerStackParams(blocks=(object(),))
    )
    if confidence_autocast:
        confidence = native_confidence_autocast_params(confidence)
    _infer(
        params._replace(confidence=confidence),
        trunk_dtype=jnp.bfloat16,
        confidence_autocast=confidence_autocast,
    )

    assert seen, "the confidence pairformer stack was never entered"
    expected = "bfloat16" if confidence_autocast else "float32"
    assert set(seen) == {(expected, expected)}


@pytest.mark.parametrize("diffusion_autocast", [False, True])
def test_the_denoiser_is_entered_in_the_policys_dtype(
    monkeypatch, diffusion_autocast
) -> None:
    """The network's own operands, recorded at its boundary.

    The conditioning narrows and the coordinates do not. ``x_noisy`` staying
    FP32 is the load-bearing half: its only matmul consumer is ``linear_r``,
    which upstream builds with ``precision=torch.float32`` and the comment
    "use high precision for ref_pos", so narrowing it here would be exactly
    the rounding that exemption exists to prevent.
    """
    seen: list[dict[str, str]] = []
    original = diffusion_module.diffusion_module_forward

    def record(*args, **kwargs):
        # Positional layout: the tenth argument is ``x_noisy`` and the
        # fifteenth is ``s_trunk``.
        seen.append(
            {
                "x_noisy": str(args[9].dtype),
                "s_trunk": str(args[14].dtype),
                "pair_z": str(kwargs["pair_z"].dtype),
            }
        )
        return original(*args, **kwargs)

    monkeypatch.setattr(diffusion_module, "diffusion_module_forward", record)
    params = _bf16_trunk_params()
    if diffusion_autocast:
        params = params._replace(
            diffusion=native_diffusion_autocast_params(params.diffusion)
        )
    out = _infer(
        params,
        trunk_dtype=jnp.bfloat16,
        diffusion_autocast=diffusion_autocast,
    )

    assert seen, "the denoising network was never entered"
    conditioning = "bfloat16" if diffusion_autocast else "float32"
    assert seen == [
        {"x_noisy": "float32", "s_trunk": conditioning, "pair_z": conditioning}
    ] * len(seen)
    # The sampler keeps its own FP32 state on both sides of the boundary.
    assert out["coordinate"].dtype == jnp.float32


def test_a_policy_the_parameters_were_not_rebuilt_for_is_refused() -> None:
    """BF16 activations against FP32 weights promote back to an FP32 matmul.

    Nothing about that failure is visible in the output, so the two halves of
    the realisation are checked against each other rather than trusted.
    """
    params = _bf16_trunk_params()
    with pytest.raises(ValueError, match="confidence_autocast=True"):
        _infer(params, trunk_dtype=jnp.bfloat16, confidence_autocast=True)
    with pytest.raises(ValueError, match="diffusion_autocast=True"):
        _infer(params, trunk_dtype=jnp.bfloat16, diffusion_autocast=True)
    prepared = params._replace(
        confidence=native_confidence_autocast_params(params.confidence)
    )
    with pytest.raises(ValueError, match="confidence_autocast=False"):
        _infer(prepared, trunk_dtype=jnp.bfloat16)


def test_the_resolved_fp32_policy_is_the_same_program_as_passing_nothing() -> None:
    """Below the gate `upstream` resolves to FP32, and FP32 adds no arithmetic.

    This used to be `auto`; the released default narrows the confidence head
    here now, and `upstream` is the spelling that still reproduces the wide
    program. Both arms run this branch's code, so it pins that the resolved
    policy adds nothing on top of the defaults. That the defaults themselves
    still match `main` is a two-snapshot check -- one process on each source
    tree -- which no single-tree test can make.
    """
    params = _bf16_trunk_params()
    policy = realise_amp_policy(
        requested_amp_policy("upstream", 2560), trunk_is_bf16=True
    )
    assert policy == AmpPolicy(False, False)

    before = _infer(params, trunk_dtype=jnp.bfloat16)
    after = _infer(
        params,
        trunk_dtype=jnp.bfloat16,
        confidence_autocast=policy.confidence_autocast,
        diffusion_autocast=policy.diffusion_autocast,
    )
    assert set(before) == set(after)
    for name, value in before.items():
        assert np.array_equal(np.asarray(value), np.asarray(after[name])), name


def _compactable_confidence(params):
    """Give the toy head a released-shaped bin table and a nonempty stack.

    The fixture ships one bin, which `can_compact_confidence_distance_embedding`
    rejects (`lower.size <= 1`), and an empty Pairformer stack. Forcing the
    compact path onto either would exercise code no released checkpoint
    reaches, which is the failure mode this helper exists to avoid.
    """
    bins = jnp.asarray([0.0, 5.0, 10.0, 15.0, 20.0], dtype=jnp.float32)
    return params._replace(
        confidence=params.confidence._replace(
            distance_embedding=ConfidenceDistanceEmbeddingParams(
                lower_bins=bins[:-1],
                upper_bins=bins[1:],
                linear_d=LinearParams(
                    weight=jnp.asarray(
                        np.random.default_rng(0).normal(size=(2, 4)), jnp.float32
                    ),
                    bias=None,
                ),
                linear_d_wo_onehot=LinearParams(
                    weight=jnp.zeros((2, 1), jnp.float32), bias=None
                ),
            ),
            pairformer_stack=PairformerStackParams(
                blocks=(_zero_pairformer_block(2, 2),)
            ),
        )
    )


def test_auto_realises_a_bf16_confidence_head_at_the_old_fp32_boundary(
    monkeypatch,
) -> None:
    """2,560 tokens: upstream's last FP32 size, and the port's first BF16 one.

    Everything here is recorded at an execution boundary rather than read off
    a parameter tree, because the defect this stage keeps producing is a
    silent promotion: BF16 storage whose matmul runs FP32 anyway looks exactly
    like a policy that was never applied.

    The compact bin projection is included deliberately. It once returned FP32
    from a fully realised BF16 policy -- `jnp.result_type(distance.dtype,
    weight.dtype)` promoted against the FP32 distances -- which widened
    `z_pair` and every confidence Pairformer block after it. That path was
    only ever reached above 2,560 tokens before this default; here it runs at
    a size it never ran at.
    """
    policy = realise_amp_policy(requested_amp_policy("auto", 2560), trunk_is_bf16=True)
    assert policy == AmpPolicy(True, False)

    params = cast_trunk_params(_compactable_confidence(_toy_params()), jnp.bfloat16)
    realised = params._replace(
        confidence=native_confidence_autocast_params(params.confidence)
    )
    assert can_compact_confidence_distance_embedding(
        realised.confidence.distance_embedding
    )

    seen: dict[str, list] = {"entry": [], "stack": [], "linear": [], "compact": []}
    original_head = model_module.confidence_head
    original_stack = confidence_module.pairformer_stack
    original_linear = confidence_module.linear
    original_compact = confidence_module._compact_confidence_bin_projection

    def head(
        features, s_inputs, s_trunk, z_trunk, pair_mask, coords, head_params, **kw
    ):
        seen["entry"].append(
            {
                "s_inputs": str(s_inputs.dtype),
                "s_trunk": str(s_trunk.dtype),
                "z_trunk": str(z_trunk.dtype),
                "coords": str(coords.dtype),
            }
        )
        return original_head(
            features, s_inputs, s_trunk, z_trunk, pair_mask, coords, head_params, **kw
        )

    def stack(s, z, pair_mask, stack_params, **kw):
        out = original_stack(s, z, pair_mask, stack_params, **kw)
        seen["stack"].append(
            (str(s.dtype), str(z.dtype), str(out[0].dtype), str(out[1].dtype))
        )
        return out

    def linear(x, linear_params):
        result = original_linear(x, linear_params)
        seen["linear"].append(
            (
                str(x.dtype),
                type(linear_params).__name__,
                str(linear_params.weight.dtype),
                str(result.dtype),
            )
        )
        return result

    def compact(distance, embedding_params):
        out = original_compact(distance, embedding_params)
        seen["compact"].append((str(distance.dtype), str(out.dtype)))
        return out

    monkeypatch.setattr(model_module, "confidence_head", head)
    monkeypatch.setattr(confidence_module, "pairformer_stack", stack)
    monkeypatch.setattr(confidence_module, "linear", linear)
    monkeypatch.setattr(
        confidence_module, "_compact_confidence_bin_projection", compact
    )
    out = _infer(
        realised,
        trunk_dtype=jnp.bfloat16,
        compact_confidence_distance_bins=True,
        confidence_autocast=policy.confidence_autocast,
        diffusion_autocast=policy.diffusion_autocast,
    )
    jax.block_until_ready(out)

    for boundary, records in seen.items():
        assert records, f"{boundary} was not exercised"

    # The trunk representations reach the head unwidened; `s_inputs` does not,
    # because it never was BF16 -- the input embedder concatenates raw
    # reference features, and `confidence_s_inputs = s_inputs` under autocast
    # only stops widening an array that is already FP32.
    assert seen["entry"] == [
        {
            "s_inputs": "float32",
            "s_trunk": "bfloat16",
            "z_trunk": "bfloat16",
            "coords": "float32",
        }
    ]
    assert out["s_inputs"].dtype == jnp.float32

    # The compact projection follows the weight, not the FP32 distances.
    assert seen["compact"] == [("float32", "bfloat16")]
    # ...so the pair tensor it built stays narrow through every block.
    assert seen["stack"] == [("bfloat16", "bfloat16", "bfloat16", "bfloat16")]

    # `linear_s1`/`linear_s2` are the FP32-fed pair: an autocast node narrows
    # its own operand, which is what keeps an FP32 `s_inputs` from promoting
    # the outer-sum initialiser and the stack behind it.
    narrowing = ("float32", "AutocastLinearParams", "bfloat16", "bfloat16")
    assert [row for row in seen["linear"] if row[1] == "AutocastLinearParams"] == [
        narrowing
    ] * 3
    # The output stage is FP32 under every policy, including this one.
    assert [row for row in seen["linear"] if row[1] == "LinearParams"] == [
        ("float32", "LinearParams", "float32", "float32")
    ] * 2
    for name in ("plddt", "pae", "pde", "resolved"):
        assert out[name].dtype == jnp.float32
        assert np.isfinite(np.asarray(out[name])).all()


def test_auto_leaves_the_diffusion_stage_untouched_below_the_gate() -> None:
    """The half of the gate that did not move, proved on the coordinates.

    Nothing the confidence head computes feeds the sampler, so a BF16 head and
    an FP32 head must produce the same structure to the bit. The two flags are
    traced arguments rather than an environment variable, so one process runs
    both arms without a compile cache deciding the answer.
    """
    policy = realise_amp_policy(requested_amp_policy("auto", 2560), trunk_is_bf16=True)
    assert policy.diffusion_autocast is False

    params = cast_trunk_params(_compactable_confidence(_toy_params()), jnp.bfloat16)
    realised = params._replace(
        confidence=native_confidence_autocast_params(params.confidence)
    )
    # The realisation rebuilt one subtree, and it was not this one.
    assert realised.diffusion is params.diffusion
    assert not isinstance(
        realised.diffusion.conditioning.linear_z, Fp32PrecisionLinearParams
    )
    assert _float_leaf_dtypes(realised.diffusion) == {"float32"}

    narrow_head = _infer(
        realised,
        trunk_dtype=jnp.bfloat16,
        compact_confidence_distance_bins=True,
        confidence_autocast=True,
        diffusion_autocast=False,
    )
    wide_head = _infer(
        params,
        trunk_dtype=jnp.bfloat16,
        compact_confidence_distance_bins=True,
        confidence_autocast=False,
        diffusion_autocast=False,
    )
    assert narrow_head["coordinate"].dtype == jnp.float32
    np.testing.assert_array_equal(
        np.asarray(narrow_head["coordinate"]), np.asarray(wide_head["coordinate"])
    )


def test_the_two_stages_are_part_of_the_compiled_programs_identity() -> None:
    """2k and 3k jobs must never share one executable across policies."""
    assert "confidence_autocast" in model_module.GRAPH_STATIC_ARGNAMES
    assert "diffusion_autocast" in model_module.GRAPH_STATIC_ARGNAMES


# ------------------------------------------------------------- option path --


def test_preparing_a_tree_without_the_stage_leaves_the_named_error_to_the_model() -> (
    None
):
    """Now that `auto` narrows at every size, every run reaches the preparer.

    A tree with no confidence stage is not a checkpoint any release ships,
    and the model already refuses it with a sentence that names the stage.
    Reaching through it here would replace that sentence with an
    `AttributeError` naming a tuple, and would do it on the default path.
    """
    assert predict_cli._amp_realised_params((), AmpPolicy(True, True), {}) == ()
    params = _bf16_trunk_params()
    stripped = params._replace(confidence=None)
    assert (
        predict_cli._amp_realised_params(stripped, AmpPolicy(True, False), {})
        is stripped
    )
    with pytest.raises(ValueError, match="carry no confidence stage"):
        _infer(stripped, trunk_dtype=jnp.bfloat16, confidence_autocast=True)


def test_the_cli_flag_reads_its_vocabulary_from_the_policy_module() -> None:
    """One list of values, so a new policy cannot exist in only one of them."""
    parser = _parser()
    action = next(a for a in parser._actions if a.dest == "amp_policy")
    assert tuple(action.choices) == AMP_POLICY_CHOICES
    assert action.default == DEFAULT_AMP_POLICY


def _parser() -> argparse.ArgumentParser:
    captured: list[argparse.ArgumentParser] = []

    def capture(parser, *_args, **_kwargs):
        captured.append(parser)
        raise _CapturedError

    with mock.patch.object(argparse.ArgumentParser, "parse_args", capture):
        with pytest.raises(_CapturedError):
            predict_cli.main([])
    return captured[0]


class _CapturedError(Exception):
    pass


def test_the_backend_carries_the_option_into_the_compile_identity() -> None:
    """A 2k and a 3k job must not be able to share one cached executable."""
    assert "amp_policy" in ProtenixBackend.compile_options
    assert "amp_policy" in backend_impl._CLI_OPTIONS
    assert backend_impl._RELEASED_COMPILE_DEFAULTS["amp_policy"] == "auto"


def test_asking_for_the_released_default_is_the_same_namespace_as_not_asking(
    tmp_path: Path,
) -> None:
    """`auto` is what an unset run does, so spelling it out cannot fork a cache."""
    backend = ProtenixBackend()
    job = tmp_path / "job.json"
    job.write_text("[]", encoding="utf-8")

    def request(**options):
        return PredictionRequest(
            model="protenix",
            input=job,
            output_dir=tmp_path / "out",
            options=options,
        )

    unset = backend.cache_profile(request())
    assert backend.cache_profile(request(amp_policy="auto")) == unset
    assert backend.cache_profile(request(amp_policy="bf16")) != unset
    # `upstream` is a different program below 2,560 tokens, so it must not
    # collapse into the default's namespace.
    assert backend.cache_profile(request(amp_policy="upstream")) != unset


@pytest.mark.parametrize(
    ("argv", "n_token", "expected"),
    (
        # The released default narrows the head at every size; only the
        # diffusion half is still driven by the job's own size.
        ((), 2, AmpPolicy(True, False)),
        ((), 3000, AmpPolicy(True, False)),
        ((), 4000, AmpPolicy(True, True)),
        # Upstream's gate, still reachable, still turning at 2560.
        (("--amp-policy", "upstream"), 2, AmpPolicy(False, False)),
        (("--amp-policy", "upstream"), 3000, AmpPolicy(True, False)),
        (("--amp-policy", "upstream"), 4000, AmpPolicy(True, True)),
        (("--amp-policy", "bf16"), 2, AmpPolicy(True, True)),
        (("--amp-policy", "fp32"), 4000, AmpPolicy(False, False)),
        # An FP32 trunk opens no autocast, so no table has anything to move.
        (("--trunk-dtype", "fp32"), 4000, AmpPolicy(False, False)),
        (("--trunk-dtype", "fp32"), 2, AmpPolicy(False, False)),
    ),
    ids=(
        "small",
        "gated-confidence",
        "gated-both",
        "upstream-small",
        "upstream-gated-confidence",
        "upstream-gated-both",
        "pinned-bf16",
        "pinned-fp32",
        "fp32-trunk",
        "fp32-trunk-small",
    ),
)
def test_the_cli_resolves_the_policy_per_job_and_threads_it(
    tmp_path: Path, monkeypatch, argv, n_token, expected
) -> None:
    """The flag is worth nothing if it stops at ``args``.

    ``restype`` is the only feature the token count is read from, so widening
    it alone drives the gate without building a 3,000-token fixture: the
    prediction itself is captured, not run.
    """
    job = tmp_path / "job.json"
    job.write_text(
        json.dumps([{"name": "tiny", "modelSeeds": [0], "sequences": []}]),
        encoding="utf-8",
    )
    weights = tmp_path / "protenix.jax"
    weights.write_bytes(b"native fixture")

    features = dict(_toy_features())
    if n_token != int(features["restype"].shape[-2]):
        features["restype"] = jnp.zeros(
            (n_token, features["restype"].shape[-1]), dtype=features["restype"].dtype
        )

    from foldjax.models.protenix.data import featurize_json

    monkeypatch.setattr(
        featurize_json, "featurize_protein_json", lambda *a, **k: dict(features)
    )
    monkeypatch.setattr(
        predict_cli, "_load_prepared_params", lambda *a, **k: _toy_params()
    )
    captured: list[AmpPolicy] = []
    prepared: list[bool] = []

    def capture_predict(params, *_args, **kwargs):
        captured.append(
            AmpPolicy(kwargs["confidence_autocast"], kwargs["diffusion_autocast"])
        )
        prepared.append(
            isinstance(
                params.confidence.distance_embedding.linear_d, AutocastLinearParams
            )
        )
        return {}

    monkeypatch.setattr(predict_impl, "protenix_predict_static", capture_predict)
    predict_cli.main(
        [
            "--input-json",
            str(job),
            "--weights",
            str(weights),
            "--out",
            str(tmp_path / "out"),
            "--model-name",
            "protenix_mini_default_v0.5.0",
            "--no-compile-cache",
            "--prewarm-only",
            *argv,
        ]
    )

    assert captured == [expected]
    # The parameter tree the CLI handed over was rebuilt to match, which is
    # what makes the flag a realisation rather than a label.
    assert prepared == [expected.confidence_autocast]


def test_the_realised_policy_is_reported_per_job(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """The realised pair is not derivable from the options a bench row records.

    ``--amp-policy auto`` is one string whichever side of the gate a job lands
    on, and the token count that decides it is only known inside this CLI. The
    line below is where a run says which policy it actually ran, so its shape
    is pinned rather than left as incidental logging.
    """
    job = tmp_path / "job.json"
    job.write_text(
        json.dumps([{"name": "tiny", "modelSeeds": [0], "sequences": []}]),
        encoding="utf-8",
    )
    weights = tmp_path / "protenix.jax"
    weights.write_bytes(b"native fixture")

    features = dict(_toy_features())
    features["restype"] = jnp.zeros(
        (3000, features["restype"].shape[-1]), dtype=features["restype"].dtype
    )
    from foldjax.models.protenix.data import featurize_json

    monkeypatch.setattr(
        featurize_json, "featurize_protein_json", lambda *a, **k: dict(features)
    )
    monkeypatch.setattr(
        predict_cli, "_load_prepared_params", lambda *a, **k: _toy_params()
    )
    monkeypatch.setattr(predict_impl, "protenix_predict_static", lambda *a, **k: {})
    predict_cli.main(
        [
            "--input-json",
            str(job),
            "--weights",
            str(weights),
            "--out",
            str(tmp_path / "out"),
            "--model-name",
            "protenix_mini_default_v0.5.0",
            "--no-compile-cache",
            "--prewarm-only",
        ]
    )

    assert (
        "tiny: amp policy confidence=bf16 diffusion=fp32 "
        "(--amp-policy auto, n_token=3000, trunk=bf16)"
    ) in capsys.readouterr().out
