"""The token gate, and the dtypes it actually produces.

The resolver is arithmetic and is pinned at its boundaries. Everything below
that reads the traced program or records the operands a stage was handed,
never the option string that produced them: a test asserting that
``--amp-policy bf16`` was requested proves nothing about whether a matmul
narrowed, which is the whole property this feature delivers.
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
    realise_amp_policy,
    requested_amp_policy,
)
from foldjax.models.protenix.cli import predict as predict_cli
from foldjax.models.protenix.models import model as model_module
from foldjax.models.protenix.models import predict as predict_impl
from foldjax.models.protenix.models.diffusion import diffusion as diffusion_module
from foldjax.models.protenix.models.heads import confidence as confidence_module
from foldjax.models.protenix.models.input_precision import (
    native_confidence_autocast_params,
    native_diffusion_autocast_params,
)
from foldjax.models.protenix.models.model import (
    cast_trunk_params,
    protenix_infer_static,
)
from foldjax.models.protenix.models.primitives.primitives import (
    AutocastLinearParams,
    Fp32PrecisionLinearParams,
    LinearParams,
    linear,
)
from foldjax.models.protenix.models.trunk_blocks.pairformer import PairformerStackParams
from foldjax.schema import PredictionRequest

from .test_model import _toy_features, _toy_params

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
    assert amp_policy_for_tokens(n_token) == expected


def test_protenix_v2_runs_its_confidence_head_under_autocast_at_every_size() -> None:
    """Upstream's `else` branch, which only reaches sizes below the gate."""
    assert amp_policy_for_tokens(64, "protenix-v2") == AmpPolicy(True, False)
    assert amp_policy_for_tokens(2560, "protenix-v2") == AmpPolicy(True, False)
    # Above the threshold the model name stops mattering: every variant is
    # already on the same branch.
    assert amp_policy_for_tokens(2561, "protenix-v2") == amp_policy_for_tokens(2561)


def test_the_base_model_keeps_an_fp32_confidence_head_below_the_gate() -> None:
    for name in (None, "protenix_base_default_v1.0.0", "protenix_mini_esm_v0.5.0"):
        assert amp_policy_for_tokens(2560, name) == AmpPolicy(False, False)


def test_the_resolver_does_not_own_the_protenix_v2_token_limit() -> None:
    """`runtime_policy.PROTENIX_V2_MAX_TOKENS` owns it; two owners drift apart."""
    assert amp_policy_for_tokens(4096, "protenix-v2") == AmpPolicy(True, True)


def test_pinned_policies_ignore_the_token_count() -> None:
    for n_token in (16, 2560, 2561, 4096):
        assert requested_amp_policy("fp32", n_token) == AmpPolicy(False, False)
        assert requested_amp_policy("bf16", n_token) == AmpPolicy(True, True)


def test_auto_is_the_default_and_reproduces_the_gate() -> None:
    assert DEFAULT_AMP_POLICY == "auto"
    assert set(AMP_POLICY_CHOICES) == {"auto", "fp32", "bf16"}
    for n_token in (16, 2561, 4096):
        assert requested_amp_policy("auto", n_token) == amp_policy_for_tokens(n_token)


def test_an_unknown_policy_names_the_ones_that_exist() -> None:
    with pytest.raises(ValueError, match="auto, fp32, bf16"):
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

    ``x_noisy`` is the interesting one: the sampler state stays FP32 under
    every policy, so a narrowed denoiser has to narrow it at the boundary or
    the first geometry projection would widen everything after it.
    """
    seen: list[dict[str, str]] = []
    original = diffusion_module.diffusion_module_forward

    def record(*args, **kwargs):
        # Positional layout: the ninth argument is ``r_noisy`` and the
        # thirteenth to fifteenth are s_inputs, s_trunk, z_trunk.
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
    expected = "bfloat16" if diffusion_autocast else "float32"
    assert all(record == dict.fromkeys(record, expected) for record in seen), seen
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


def test_a_small_case_is_bitwise_unchanged_under_the_resolved_fp32_policy() -> None:
    """Below the gate `auto` resolves to FP32, and FP32 must be the old program.

    The port's parity at these sizes is 0.04-0.1 A and was measured on the
    program that existed before this option. Passing the resolved policy has to
    leave that program alone, not merely close to it -- so this compares the
    arrays, not a tolerance.
    """
    params = _bf16_trunk_params()
    policy = realise_amp_policy(requested_amp_policy("auto", 2560), trunk_is_bf16=True)
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


def test_the_two_stages_are_part_of_the_compiled_programs_identity() -> None:
    """2k and 3k jobs must never share one executable across policies."""
    assert "confidence_autocast" in model_module.GRAPH_STATIC_ARGNAMES
    assert "diffusion_autocast" in model_module.GRAPH_STATIC_ARGNAMES


# ------------------------------------------------------------- option path --


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


@pytest.mark.parametrize(
    ("argv", "n_token", "expected"),
    (
        ((), 2, AmpPolicy(False, False)),
        # The gate, driven only by the job's own size.
        ((), 3000, AmpPolicy(True, False)),
        ((), 4000, AmpPolicy(True, True)),
        (("--amp-policy", "bf16"), 2, AmpPolicy(True, True)),
        (("--amp-policy", "fp32"), 4000, AmpPolicy(False, False)),
        # An FP32 trunk opens no autocast, so the gate has nothing to move.
        (("--trunk-dtype", "fp32"), 4000, AmpPolicy(False, False)),
    ),
    ids=(
        "small",
        "gated-confidence",
        "gated-both",
        "pinned-bf16",
        "pinned-fp32",
        "fp32-trunk",
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
