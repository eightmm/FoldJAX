"""The opt-in bfloat16 denoising network, asserted on realised array dtypes.

Every assertion below reads a dtype off an array or a jaxpr equation. None
reads a config spelling: a test that pins ``--diffusion-dtype bf16`` proves
only that argparse stored a string, and the failure this option can actually
have -- narrowed activations meeting FP32 weights, which promote back to an
FP32 matmul and look exactly like a policy that never fired -- is invisible
to it.
"""

from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import foldjax.models.opendde.models.diffusion_module as diffusion_impl
from foldjax.models.opendde.cli.predict import (
    DIFFUSION_DTYPE_CHOICES,
    _resolve_diffusion_autocast,
)
from foldjax.models.opendde.models.diffusion_conditioning import (
    DiffusionConditioningParams,
    diffusion_conditioning_prepare_cache,
)
from foldjax.models.opendde.models.diffusion_precision import (
    native_diffusion_autocast_params,
)
from foldjax.models.opendde.models.model import (
    GRAPH_STATIC_ARGNAMES,
    _require_realised_diffusion_params,
    opendde_infer_static,
)
from foldjax.models.protenix.models.input_precision import (
    native_diffusion_autocast_params as protenix_diffusion_autocast_params,
)
from foldjax.models.protenix.models.primitives.primitives import (
    AutocastLinearF32OutParams,
    AutocastLinearParams,
    Fp32PrecisionLinearParams,
    LayerNormParams,
    LinearParams,
)

from ..protenix.test_model import _toy_params
from .test_model import _static_validation_features

#: Every parameter node type the walk below treats as a leaf.
_PARAMETER_NODES = (
    LinearParams,
    AutocastLinearParams,
    AutocastLinearF32OutParams,
    Fp32PrecisionLinearParams,
    LayerNormParams,
)

#: The eleven projections upstream OpenDDE constructs
#: ``precision=torch.float32``. Ten are Protenix's; ``linear_z_trunk`` is the
#: pair-compression projection OpenDDE adds
#: (``opendde/model/modules/diffusion.py:170-174``).
_EXEMPT_PATHS = (
    ("conditioning", "linear_z_trunk"),
    ("conditioning", "linear_z"),
    ("conditioning", "linear_s"),
    ("conditioning", "linear_n"),
    ("atom_encoder", "cache", "linear_ref_pos"),
    ("atom_encoder", "cache", "linear_d"),
    ("atom_encoder", "linear_s"),
    ("atom_encoder", "linear_z"),
    ("atom_encoder", "linear_r"),
    ("linear_s",),
    ("atom_decoder", "linear_out"),
)

_C_Z = 2


def _opendde_diffusion_params():
    """Protenix's toy denoiser with OpenDDE's compressing conditioner."""
    base = _toy_params().diffusion
    shared = base.conditioning
    conditioning = DiffusionConditioningParams(
        relpe=shared.relpe,
        layernorm_z_trunk=LayerNormParams(weight=jnp.ones((_C_Z,)), bias=None),
        linear_z_trunk=LinearParams(
            weight=jnp.full((_C_Z, _C_Z), 0.25, dtype=jnp.float32), bias=None
        ),
        layernorm_z=shared.layernorm_z,
        linear_z=shared.linear_z,
        transition_z1=shared.transition_z1,
        transition_z2=shared.transition_z2,
        layernorm_s=shared.layernorm_s,
        linear_s=shared.linear_s,
        fourier=shared.fourier,
        layernorm_n=shared.layernorm_n,
        linear_n=shared.linear_n,
        transition_s1=shared.transition_s1,
        transition_s2=shared.transition_s2,
    )
    return base._replace(conditioning=conditioning)


def _resolve(tree, path):
    node = tree
    for name in path:
        node = getattr(node, name)
    return node


def _parameter_nodes(tree):
    flat = jax.tree_util.tree_flatten_with_path(
        tree, is_leaf=lambda x: isinstance(x, _PARAMETER_NODES)
    )[0]
    return [
        (jax.tree_util.keystr(k), v) for k, v in flat if isinstance(v, _PARAMETER_NODES)
    ]


def test_released_default_leaves_the_diffusion_tree_untouched():
    """FP32 is the default and the checkpoint's own storage reaches the graph."""
    assert DIFFUSION_DTYPE_CHOICES[0] == "fp32"
    assert _resolve_diffusion_autocast("fp32", "bf16") is False
    original = _opendde_diffusion_params()
    for name, node in _parameter_nodes(original):
        assert isinstance(node, (LinearParams, LayerNormParams)), name
        for leaf in jax.tree.leaves(node):
            assert leaf.dtype == jnp.float32, name
    # And a run with the policy off accepts exactly that tree.
    _require_realised_diffusion_params(type("P", (), {"diffusion": original})(), False)


def test_the_eleventh_exemption_is_opendde_s_and_protenix_narrows_it():
    """The census difference, asserted as a dtype rather than a claim."""
    original = _opendde_diffusion_params()
    protenix = protenix_diffusion_autocast_params(original)
    opendde = native_diffusion_autocast_params(original)

    narrowed = protenix.conditioning.linear_z_trunk
    assert isinstance(narrowed, AutocastLinearParams)
    assert narrowed.weight.dtype == jnp.bfloat16

    exempt = opendde.conditioning.linear_z_trunk
    assert isinstance(exempt, Fp32PrecisionLinearParams)
    assert exempt.weight.dtype == jnp.float32
    np.testing.assert_array_equal(
        exempt.weight, original.conditioning.linear_z_trunk.weight
    )


def test_every_projection_lands_in_exactly_one_of_the_three_classes():
    original = _opendde_diffusion_params()
    realised = native_diffusion_autocast_params(original)

    exempt = {id(_resolve(realised, path)) for path in _EXEMPT_PATHS}
    assert len(exempt) == len(_EXEMPT_PATHS)

    counts = {"exempt": 0, "narrowed": 0, "pair_bias": 0, "norm": 0}
    for name, node in _parameter_nodes(realised):
        if isinstance(node, Fp32PrecisionLinearParams):
            assert id(node) in exempt, name
            assert node.weight.dtype == jnp.float32, name
            counts["exempt"] += 1
        elif isinstance(node, AutocastLinearF32OutParams):
            assert name.endswith(".attention_pair_bias.linear_z"), name
            assert node.weight.dtype == jnp.bfloat16, name
            counts["pair_bias"] += 1
        elif isinstance(node, AutocastLinearParams):
            assert node.weight.dtype == jnp.bfloat16, name
            counts["narrowed"] += 1
        else:
            assert isinstance(node, LayerNormParams), name
            for leaf in jax.tree.leaves(node):
                assert leaf.dtype == jnp.float32, name
            counts["norm"] += 1

    assert counts["exempt"] == len(_EXEMPT_PATHS) == 11
    # One per transformer stack: the atom encoder, the token transformer and
    # the atom decoder. The toy tree carries a single block in each.
    assert counts["pair_bias"] == 3
    assert counts["narrowed"] > 0 and counts["norm"] > 0


def test_a_pair_bias_projection_multiplies_in_bf16_and_delivers_f32():
    realised = native_diffusion_autocast_params(_opendde_diffusion_params())
    bias = realised.diffusion_transformer.blocks[0].attention_pair_bias.linear_z
    from foldjax.models.protenix.models.primitives.primitives import linear

    x = jnp.ones((2, bias.weight.shape[-1]), dtype=jnp.float32)
    jaxpr = jax.make_jaxpr(lambda value: linear(value, bias))(x)
    dots = [eqn for eqn in jaxpr.jaxpr.eqns if eqn.primitive.name == "dot_general"]
    assert len(dots) == 1
    assert all(var.aval.dtype == jnp.bfloat16 for var in dots[0].invars)
    assert dots[0].outvars[0].aval.dtype == jnp.float32
    assert jax.eval_shape(lambda value: linear(value, bias), x).dtype == jnp.float32


def test_narrowed_weights_reach_a_bf16_matmul_and_exempt_ones_do_not():
    """Tripwire: the realised tree changes the matmuls, not just the storage.

    Traced through the real conditioning cache, which reaches both classes:
    ``linear_z_trunk`` and ``linear_z`` are exempt and must widen their
    operands, while the two transitions after them are ordinary projections
    and must narrow theirs.
    """
    realised = native_diffusion_autocast_params(_opendde_diffusion_params())
    conditioning = realised.conditioning
    z_trunk = jnp.ones((3, 3, _C_Z), dtype=jnp.bfloat16)
    relp = jnp.ones((3, 3, _C_Z), dtype=jnp.bfloat16)

    def run(value):
        return diffusion_conditioning_prepare_cache(
            None, value, conditioning, relp_encoding=relp
        )

    jaxpr = jax.make_jaxpr(run)(z_trunk)
    widths = [
        {var.aval.dtype for var in eqn.invars}
        for eqn in jaxpr.jaxpr.eqns
        if eqn.primitive.name == "dot_general"
    ]
    assert widths, "the conditioning cache contains no matmul to classify"
    assert widths.count({jnp.dtype("bfloat16")}) > 0
    assert widths.count({jnp.dtype("float32")}) == 2
    # The stage still leaves at the trunk's width, so the narrowing carries.
    assert jax.eval_shape(run, z_trunk).dtype == jnp.bfloat16

    original = _opendde_diffusion_params().conditioning
    control = jax.make_jaxpr(
        lambda value: diffusion_conditioning_prepare_cache(
            None, value, original, relp_encoding=relp.astype(jnp.float32)
        )
    )(z_trunk.astype(jnp.float32))
    assert not any(
        var.aval.dtype == jnp.bfloat16
        for eqn in control.jaxpr.eqns
        if eqn.primitive.name == "dot_general"
        for var in eqn.invars
    )


def _guard_inputs():
    return (
        jnp.zeros((2,), dtype=jnp.int32),  # atom_to_token_idx
        jnp.zeros((2, 3), dtype=jnp.float32),  # ref_pos
        jnp.zeros((2,), dtype=jnp.float32),  # ref_charge
        jnp.ones((2,), dtype=jnp.float32),  # ref_mask
        jnp.zeros((2, 4, 64), dtype=jnp.float32),  # ref_atom_name_chars
        jnp.zeros((2, 128), dtype=jnp.float32),  # ref_element
        jnp.zeros((1, 1, 3), dtype=jnp.float32),  # d_lm
        jnp.ones((1, 1), dtype=jnp.float32),  # v_lm
        {},  # pad_info
    )


@pytest.mark.parametrize("denoiser_autocast", [False, True])
def test_the_denoiser_output_guard_fires_and_restores_float32(
    monkeypatch, denoiser_autocast
):
    """Off, the EDM blend leaves at BF16; on, the sampler gets FP32 back.

    The stub stands in for a narrowed network, and ``calls`` is its tripwire:
    without it a guard test could pass against a network that never ran.
    """
    calls = []

    def fake_f_forward(*args, **kwargs):
        calls.append(True)
        return jnp.full((1, 2, 3), 0.5, dtype=jnp.bfloat16)

    monkeypatch.setattr(diffusion_impl, "diffusion_module_f_forward", fake_f_forward)

    x_noisy = jnp.full((1, 2, 3), 0.25, dtype=jnp.float32)
    result = diffusion_impl.diffusion_module_forward(
        *_guard_inputs(),
        x_noisy=x_noisy,
        t_hat_noise_level=jnp.full((1,), 2.0, dtype=jnp.float32),
        relp_feature=None,
        s_inputs=jnp.zeros((2, 2), dtype=jnp.bfloat16),
        s_trunk=jnp.zeros((2, 2), dtype=jnp.bfloat16),
        z_trunk=jnp.zeros((2, 2, 2), dtype=jnp.bfloat16),
        params=object(),
        n_token=2,
        atom_encoder_heads=1,
        token_heads=1,
        atom_decoder_heads=1,
        n_queries=2,
        n_keys=4,
        denoiser_autocast=denoiser_autocast,
    )

    assert calls == [True]
    expected = jnp.float32 if denoiser_autocast else jnp.bfloat16
    assert result.dtype == expected


def test_the_sampler_state_and_its_schedule_stay_float32():
    """The Euler state, the noise schedule and the augmentation are FP32.

    ``sampling.py:129`` casts the schedule to the sampler ``dtype``, which
    ``sampling.py:114`` defaults to FP32 and ``model.py`` never overrides;
    ``sampling.py:227,238,241`` pin the augmentation's rotations and
    translations to FP32 outright. The guard above keeps the denoiser's
    prediction at that width rather than handing the loop a BF16 array.
    """
    from foldjax.models.opendde.models.sampling import sample_diffusion
    from foldjax.models.protenix.models.diffusion.diffusion import (
        inference_noise_schedule,
    )

    schedule = inference_noise_schedule(num_steps=2)
    assert schedule.dtype == jnp.float32

    seen = []

    def denoise_fn(x_noisy, t_hat):
        seen.append((x_noisy.dtype, t_hat.dtype))
        # What the guard delivers: a narrowed network's prediction, widened.
        return jnp.zeros_like(x_noisy, dtype=jnp.bfloat16).astype(x_noisy.dtype)

    coordinates = sample_diffusion(
        denoise_fn,
        schedule,
        num_samples=1,
        n_atom=2,
        key=jax.random.PRNGKey(0),
    )
    assert seen and all(pair == (jnp.float32, jnp.float32) for pair in seen)
    assert coordinates.dtype == jnp.float32


def test_the_policy_and_the_parameter_tree_must_agree():
    original = _opendde_diffusion_params()
    realised = native_diffusion_autocast_params(original)

    def holder(diffusion):
        return type("P", (), {"diffusion": diffusion})()

    with pytest.raises(ValueError, match="left FP32"):
        _require_realised_diffusion_params(holder(original), True)
    with pytest.raises(ValueError, match="prepared for autocast"):
        _require_realised_diffusion_params(holder(realised), False)
    with pytest.raises(ValueError, match="no compression projection to exempt"):
        _require_realised_diffusion_params(
            holder(protenix_diffusion_autocast_params(original)), True
        )
    _require_realised_diffusion_params(holder(realised), True)


def test_already_narrowed_parameters_are_refused():
    narrowed = jax.tree.map(
        lambda x: x.astype(jnp.bfloat16) if hasattr(x, "dtype") else x,
        _opendde_diffusion_params(),
    )
    with pytest.raises(ValueError, match="original FP32"):
        native_diffusion_autocast_params(narrowed)


def test_a_protenix_shaped_conditioner_is_refused_by_name():
    with pytest.raises(ValueError, match="linear_z_trunk"):
        native_diffusion_autocast_params(_toy_params().diffusion)


def test_an_unknown_value_is_refused_naming_the_allowed_ones():
    with pytest.raises(ValueError, match="fp32, bf16"):
        _resolve_diffusion_autocast("bfloat16", "bf16")


def test_a_bf16_denoiser_needs_a_bf16_trunk():
    assert _resolve_diffusion_autocast("bf16", "bf16") is True
    with pytest.raises(ValueError, match="needs --trunk-dtype bf16"):
        _resolve_diffusion_autocast("bf16", "fp32")


def test_the_option_is_refused_under_a_context_parallel_mesh():
    """Deliberately deferred rather than designed for: single GPU comes first."""
    with pytest.raises(ValueError, match="not supported under context parallelism"):
        opendde_infer_static(
            {},
            None,
            None,
            key=None,
            num_samples=1,
            diffusion_autocast=True,
            cp_shards=2,
        )


def test_the_policy_joins_the_compilation_cache_identity():
    from foldjax.backends import opendde as backend_impl

    assert "diffusion_autocast" in GRAPH_STATIC_ARGNAMES
    assert "diffusion_dtype" in backend_impl._CLI_OPTIONS
    assert backend_impl._RELEASED_COMPILE_DEFAULTS["diffusion_dtype"] == "fp32"


def _stub_trunk(monkeypatch, model_impl, dtype, captured):
    """Replace every stage around the denoiser with a typed constant."""
    s_inputs = jnp.full((3, 5), 4.0, dtype=dtype)
    s_trunk = jnp.full((3, 4), 5.0, dtype=dtype)
    z_trunk = jnp.full((3, 3, 3), 6.0, dtype=dtype)
    pair_bias = jnp.zeros((3, 3), dtype=dtype)
    features = _static_validation_features()
    features["ref_charge"] = jnp.zeros((3,), dtype=jnp.float32)
    features["ref_mask"] = jnp.ones((3,), dtype=jnp.float32)
    features["ref_element"] = jnp.zeros((3, 128), dtype=jnp.float32)
    features["ref_atom_name_chars"] = jnp.zeros((3, 4, 64), dtype=jnp.float32)
    features["d_lm"] = jnp.zeros((1, 1, 3), dtype=jnp.float32)
    features["v_lm"] = jnp.ones((1, 1), dtype=jnp.float32)
    features["pad_info"] = {}
    structural_features = {**features, "structural_pair_attn_bias": pair_bias}

    monkeypatch.setattr(
        model_impl, "input_feature_embedder", lambda *a, **k: jnp.full((2, 5), 1.0)
    )
    monkeypatch.setattr(
        model_impl,
        "pairformer_output_from_s_inputs",
        lambda *a, **k: (
            jnp.full((2, 5), 1.0),
            jnp.full((2, 4), 2.0),
            jnp.full((2, 2, 3), 3.0),
        ),
    )
    monkeypatch.setattr(
        model_impl,
        "structural_token_expand",
        lambda *a, **k: (
            s_inputs,
            s_trunk,
            z_trunk,
            {"structural_pair_attn_bias": pair_bias},
        ),
    )
    monkeypatch.setattr(
        model_impl, "prepare_structural_features", lambda *a, **k: structural_features
    )
    monkeypatch.setattr(
        model_impl, "structural_refiner_stack", lambda s, z, *a, **k: (s, z)
    )
    monkeypatch.setattr(
        model_impl,
        "relative_position_encoding_from_features",
        lambda *a, **k: jnp.full((3, 3, 2), 7.0, dtype=dtype),
    )
    monkeypatch.setattr(
        model_impl,
        "diffusion_conditioning_prepare_cache",
        lambda *a, **k: jnp.full((3, 3, 2), 8.0, dtype=dtype),
    )
    monkeypatch.setattr(
        model_impl,
        "atom_attention_encoder_prepare_diffusion_cache",
        lambda *a, **k: (object(), object()),
    )

    def fake_denoiser(_idx, *args, s_inputs, s_trunk, z_trunk, **kwargs):
        captured["dtypes"] = (s_inputs.dtype, s_trunk.dtype, z_trunk.dtype)
        captured["denoiser_autocast"] = kwargs["denoiser_autocast"]
        return jnp.zeros((1, 3, 3), dtype=jnp.float32)

    monkeypatch.setattr(model_impl, "diffusion_module_forward", fake_denoiser)

    def fake_sampler(denoise_fn, *a, **k):
        denoise_fn(
            jnp.zeros((1, 3, 3), dtype=jnp.float32),
            jnp.ones((1,), dtype=jnp.float32),
        )
        return jnp.zeros((1, 3, 3), dtype=jnp.float32)

    monkeypatch.setattr(model_impl, "sample_diffusion", fake_sampler)
    monkeypatch.setattr(
        model_impl, "distogram_head", lambda *a, **k: jnp.zeros((2, 2, 2))
    )
    return features


@pytest.mark.parametrize("diffusion_autocast", [False, True])
def test_the_trunk_reaches_the_denoiser_at_the_policy_s_own_width(
    monkeypatch, diffusion_autocast
):
    """Off, the three trunk representations are widened; on, they are not.

    This is the activation half of the policy, and nothing else in this file
    would notice if it were reverted: the parameter tree would still be
    narrowed and every projection would still promote back to FP32 against
    widened inputs, which is the silent failure the whole option guards.
    """
    import foldjax.models.opendde.models.model as model_impl

    captured: dict[str, object] = {}
    features = _stub_trunk(monkeypatch, model_impl, jnp.bfloat16, captured)
    exempt = Fp32PrecisionLinearParams(jnp.zeros((2, 2), dtype=jnp.float32), None)
    conditioning = SimpleNamespace(
        relpe=object(),
        linear_z_trunk=exempt if diffusion_autocast else LinearParams(exempt.weight),
    )
    params = SimpleNamespace(
        input_embedder=object(),
        pairformer_output=object(),
        structural_expander=object(),
        structural_refiner=object(),
        diffusion=SimpleNamespace(conditioning=conditioning, atom_encoder=object()),
        distogram=object(),
        confidence=None,
    )

    model_impl.opendde_infer_static(
        features,
        params,
        jnp.asarray([1.0, 0.0], dtype=jnp.float32),
        key=jax.random.PRNGKey(0),
        num_samples=1,
        num_recycles=1,
        run_confidence=False,
        diffusion_autocast=diffusion_autocast,
    )

    expected = jnp.bfloat16 if diffusion_autocast else jnp.float32
    assert captured["dtypes"] == (expected, expected, expected)
    assert captured["denoiser_autocast"] is diffusion_autocast


def _run_cli(monkeypatch, tmp_path, argv_extra):
    """Drive the native CLI down to ``_predict`` with a real diffusion tree."""
    import json

    import numpy as np

    import foldjax.models.opendde.cli.predict as predict_impl
    from foldjax.models.opendde.models.model import OpenDDEInferenceParams

    input_path = tmp_path / "tiny.json"
    weights_path = tmp_path / "opendde.jax"
    job = {"name": "tiny", "modelSeeds": [1], "sequences": []}
    input_path.write_text(json.dumps([job]), encoding="utf-8")
    weights_path.write_bytes(b"native fixture")
    toy = _toy_params()
    loaded = OpenDDEInferenceParams(
        **{
            name: (
                _opendde_diffusion_params()
                if name == "diffusion"
                else toy.confidence
                if name == "confidence"
                else {}
            )
            for name in OpenDDEInferenceParams._fields
        }
    )
    calls = []

    monkeypatch.setattr(predict_impl, "_load_jobs", lambda path: [job])
    monkeypatch.setattr(
        predict_impl,
        "_featurize",
        lambda value, **kwargs: {"restype": np.zeros((2, 32), dtype=np.float32)},
    )
    monkeypatch.setattr(
        predict_impl, "_load_prepared_params", lambda path, trunk_dtype: loaded
    )

    def fake_predict(value, model_params, **kwargs):
        calls.append((model_params, kwargs))
        return {"coordinate": np.zeros((1, 3, 3), dtype=np.float32)}

    monkeypatch.setattr(predict_impl, "_predict", fake_predict)
    monkeypatch.setattr(predict_impl, "_score", lambda output, *a, **k: output)
    monkeypatch.setattr(
        predict_impl, "_write", lambda root, **kwargs: [tmp_path / "tiny.cif"]
    )

    predict_impl.main(
        [
            "--input-json",
            str(input_path),
            "--weights",
            str(weights_path),
            "--out",
            str(tmp_path / "out"),
            "--n-sample",
            "1",
            "--n-step",
            "2",
            *argv_extra,
        ]
    )
    return loaded, calls


def test_the_cli_default_hands_the_model_the_checkpoint_s_own_tree(
    monkeypatch, tmp_path
):
    loaded, calls = _run_cli(monkeypatch, tmp_path, ())
    params, kwargs = calls[0]
    assert kwargs["diffusion_autocast"] is False
    assert params.diffusion is loaded.diffusion


def test_the_cli_flag_rebuilds_the_tree_and_reaches_the_model(monkeypatch, tmp_path):
    loaded, calls = _run_cli(monkeypatch, tmp_path, ("--diffusion-dtype", "bf16"))
    params, kwargs = calls[0]
    assert kwargs["diffusion_autocast"] is True
    assert params.diffusion is not loaded.diffusion
    assert isinstance(
        params.diffusion.conditioning.linear_z_trunk, Fp32PrecisionLinearParams
    )
    assert params.diffusion.linear_s.weight.dtype == jnp.float32
    assert params.diffusion.atom_encoder.linear_q.weight.dtype == jnp.bfloat16
    # The session's own tree is never mutated: a second policy in the same
    # process must not be served this one's weights.
    assert isinstance(loaded.diffusion.conditioning.linear_z_trunk, LinearParams)


def test_the_cli_refuses_a_bf16_denoiser_on_a_pinned_float32_trunk(
    monkeypatch, tmp_path
):
    # The CLI turns a job failure into a SystemExit carrying the message.
    with pytest.raises(SystemExit, match="needs --trunk-dtype bf16"):
        _run_cli(
            monkeypatch,
            tmp_path,
            ("--diffusion-dtype", "bf16", "--trunk-dtype", "fp32"),
        )


def test_the_two_head_policies_compose_without_sharing_a_tree(monkeypatch, tmp_path):
    """Both options in one run, and neither is served the other's weights.

    The backend's weight session memoizes on `("trunk_dtype", trunk_dtype)`
    alone (`backends/_weight_session.py:177`), so what it caches is the
    pre-policy tree and both preparations have to happen outside it. They
    rebuild disjoint fields with `_replace`, so the session's own tree is
    untouched by either and a second policy in the same process cannot be
    handed the first one's weights.
    """
    from foldjax.models.protenix.models.heads.confidence import ConfidenceHeadParams

    loaded, calls = _run_cli(
        monkeypatch,
        tmp_path,
        ("--diffusion-dtype", "bf16", "--confidence-dtype", "bf16"),
    )
    params, kwargs = calls[0]
    assert kwargs["diffusion_autocast"] is True
    assert kwargs["confidence_dtype"] == jnp.bfloat16

    # Each field realised by its own preparer, read off the arrays.
    assert isinstance(
        params.diffusion.conditioning.linear_z_trunk, Fp32PrecisionLinearParams
    )
    assert params.diffusion.atom_encoder.linear_q.weight.dtype == jnp.bfloat16
    assert isinstance(
        params.confidence.distance_embedding.linear_d, AutocastLinearParams
    )
    assert params.confidence.distance_embedding.linear_d.weight.dtype == jnp.bfloat16

    # The session's tree is still the checkpoint's, in both fields.
    assert isinstance(loaded.diffusion.conditioning.linear_z_trunk, LinearParams)
    assert isinstance(loaded.confidence, ConfidenceHeadParams)
    assert isinstance(loaded.confidence.distance_embedding.linear_d, LinearParams)
    assert loaded.confidence.distance_embedding.linear_d.weight.dtype == jnp.float32

    # And both guards accept the realised tree while refusing a half-applied
    # one, so neither policy can ride on the other's preparation.
    from foldjax.models.opendde.models.model import (
        _require_realised_confidence_params,
    )

    _require_realised_diffusion_params(params, True)
    _require_realised_confidence_params(params, jnp.bfloat16)
    with pytest.raises(ValueError, match="left FP32"):
        _require_realised_diffusion_params(
            params._replace(diffusion=loaded.diffusion), True
        )
    with pytest.raises(ValueError, match="left FP32"):
        _require_realised_confidence_params(
            params._replace(confidence=loaded.confidence), jnp.bfloat16
        )


@pytest.mark.parametrize("prepared", [False, True])
def test_the_model_itself_refuses_a_half_applied_policy(prepared):
    """The guard fires from inside ``opendde_infer_static``, not only alone.

    It runs before any feature validation, so garbage inputs are fine; what
    is under test is that the call site still exists.
    """
    original = _opendde_diffusion_params()
    diffusion = native_diffusion_autocast_params(original) if prepared else original
    params = SimpleNamespace(diffusion=diffusion, confidence=None)
    expected = "prepared for autocast" if prepared else "left FP32"
    with pytest.raises(ValueError, match=expected):
        opendde_infer_static(
            {},
            params,
            None,
            key=None,
            num_samples=1,
            run_confidence=False,
            diffusion_autocast=not prepared,
        )
