"""OpenDDE's opt-in bfloat16 confidence head, asserted as realised dtypes.

Every assertion here reads an array's ``dtype`` or an operand of a traced
``dot_general``. None reads a configuration value: a run that asked for the
option and a run whose weights were never rebuilt for it look identical from
the outside, which is the whole reason the option carries a guard.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import foldjax.models.opendde.models.model as model_impl
import foldjax.models.protenix.models.heads.confidence as confidence_impl
from foldjax.models.opendde.models.model import (
    cast_confidence_params,
    cast_trunk_params,
)
from foldjax.models.protenix.models.heads.confidence import (
    ConfidenceDistanceEmbeddingParams,
    ConfidenceHeadParams,
    ConfidenceOutputParams,
)
from foldjax.models.protenix.models.primitives.primitives import (
    AutocastLinearParams,
    LayerNormParams,
    LinearParams,
)
from foldjax.models.protenix.models.trunk_blocks.pairformer import PairformerStackParams
from tests.models.opendde.toy_params import (
    C_S,
    C_S_INPUTS,
    C_Z,
    N_ATOM,
    N_OUT,
    N_TOKEN,
)
from tests.models.opendde.toy_params import array as _array
from tests.models.opendde.toy_params import inference_params as _params


def _features() -> dict[str, object]:
    return {
        "restype": jnp.zeros((N_TOKEN, 32), dtype=jnp.float32),
        "ref_pos": jnp.zeros((N_ATOM, 3), dtype=jnp.float32),
        "token_index": jnp.arange(N_TOKEN, dtype=jnp.int32),
        "asym_id": jnp.zeros((N_TOKEN,), dtype=jnp.int32),
        "residue_index": jnp.asarray([10, 11], dtype=jnp.int32),
        "entity_id": jnp.zeros((N_TOKEN,), dtype=jnp.int32),
        "sym_id": jnp.zeros((N_TOKEN,), dtype=jnp.int32),
        "atom_to_token_idx": jnp.asarray([0, 1, 1], dtype=jnp.int32),
        "atom_to_tokatom_idx": jnp.asarray([0, 0, 1], dtype=jnp.int32),
        "has_frame": jnp.asarray([True, True]),
        "frame_atom_index": jnp.zeros((N_TOKEN, 3), dtype=jnp.int32),
        "pae_rep_atom_mask": jnp.asarray([1, 1, 0], dtype=jnp.int32),
        "distogram_rep_atom_mask": jnp.asarray([1, 1, 0], dtype=jnp.int32),
        "parent_residue_idx": jnp.asarray([0, 0, 1], dtype=jnp.int32),
        "subtoken_role_id": jnp.asarray([1, 2, 1], dtype=jnp.int32),
        "structural_token_index": jnp.arange(N_ATOM, dtype=jnp.int32),
        "atom_to_structural_token_idx": jnp.asarray([0, 2, 2], dtype=jnp.int32),
        "atom_to_structural_tokatom_idx": jnp.asarray([0, 0, 1], dtype=jnp.int32),
        "structural_distogram_rep_atom_mask": jnp.asarray([1, 1, 0], dtype=jnp.int32),
        "structural_pae_rep_atom_mask": jnp.asarray([1, 0, 1], dtype=jnp.int32),
        "structural_has_frame": jnp.asarray([True, False, True]),
        "structural_frame_atom_index": jnp.zeros((N_ATOM, 3), dtype=jnp.int32),
        "ref_charge": jnp.zeros((N_ATOM,), dtype=jnp.float32),
        "ref_mask": jnp.ones((N_ATOM,), dtype=jnp.float32),
        "ref_element": jnp.zeros((N_ATOM, 128), dtype=jnp.float32),
        "ref_atom_name_chars": jnp.zeros((N_ATOM, 4, 64), dtype=jnp.float32),
        "d_lm": jnp.zeros((1, 1, N_ATOM), dtype=jnp.float32),
        "v_lm": jnp.ones((1, 1), dtype=jnp.float32),
        "pad_info": {},
    }


def _stub_everything_but_the_confidence_head(monkeypatch, trunk_dtype) -> None:
    """Replace every stage before the confidence head with fixed arrays.

    The trunk representations are produced in ``trunk_dtype`` so that the
    model's own ``as_float32`` widening -- the step that makes OpenDDE's head
    different from Protenix's -- is exercised rather than bypassed.
    """

    s_inputs = _array(N_TOKEN, C_S_INPUTS).astype(trunk_dtype)
    s_trunk = _array(N_TOKEN, C_S).astype(trunk_dtype)
    z_trunk = _array(N_TOKEN, N_TOKEN, C_Z).astype(trunk_dtype)
    structural = (
        _array(N_ATOM, C_S_INPUTS).astype(trunk_dtype),
        _array(N_ATOM, C_S).astype(trunk_dtype),
        _array(N_ATOM, N_ATOM, C_Z).astype(trunk_dtype),
    )
    coordinates = _array(1, N_ATOM, 3) * 6.0
    pair_bias = _array(N_ATOM, N_ATOM)

    monkeypatch.setattr(model_impl, "input_feature_embedder", lambda *a, **k: s_inputs)
    monkeypatch.setattr(
        model_impl,
        "pairformer_output_from_s_inputs",
        lambda *a, **k: (s_inputs, s_trunk, z_trunk),
    )
    monkeypatch.setattr(
        model_impl,
        "structural_token_expand",
        lambda *a, **k: (*structural, {"structural_pair_attn_bias": pair_bias}),
    )
    monkeypatch.setattr(
        model_impl,
        "prepare_structural_features",
        lambda features, extra, *a, **k: {
            **features,
            **extra,
            "atom_to_token_idx": features["atom_to_structural_token_idx"],
        },
    )
    monkeypatch.setattr(
        model_impl,
        "structural_refiner_stack",
        lambda s, z, *a, **k: (s, z),
    )
    monkeypatch.setattr(
        model_impl,
        "relative_position_encoding_from_features",
        lambda *a, **k: _array(N_ATOM, N_ATOM, 2),
    )
    monkeypatch.setattr(
        model_impl,
        "diffusion_conditioning_prepare_cache",
        lambda *a, **k: _array(N_ATOM, N_ATOM, 2),
    )
    monkeypatch.setattr(
        model_impl,
        "atom_attention_encoder_prepare_diffusion_cache",
        lambda *a, **k: (object(), object()),
    )
    monkeypatch.setattr(
        model_impl,
        "diffusion_module_forward",
        lambda *a, **k: jnp.zeros((1, N_ATOM, 3), dtype=jnp.float32),
    )
    monkeypatch.setattr(model_impl, "sample_diffusion", lambda *a, **k: coordinates)
    monkeypatch.setattr(
        model_impl,
        "distogram_head",
        lambda z, params: jnp.zeros((N_TOKEN, N_TOKEN, 2), dtype=z.dtype),
    )


def _setup(monkeypatch, trunk_dtype=jnp.bfloat16):
    """Install the stubs and build the concrete inputs, outside any trace.

    Arrays built while a trace is open become tracers of it, so the inputs
    cannot be created inside the function ``make_jaxpr`` traces.
    """

    _stub_everything_but_the_confidence_head(monkeypatch, trunk_dtype)
    return _features(), jnp.linspace(1.0, 0.0, 3)


def _run(params, features, schedule, **kwargs):
    return model_impl.opendde_infer_static(
        features,
        params,
        schedule,
        key=None,
        num_samples=1,
        # The value both compiled wrappers resolve to on a released
        # checkpoint. Tracing the dense path instead would measure a route the
        # CLI never takes.
        compact_confidence_distance_bins=True,
        # As `opendde_infer_compiled` traces it: the value checks read array
        # contents, so the wrapper runs them on the concrete features first.
        validate_feature_values=False,
        **kwargs,
    )


def _infer(params, monkeypatch, *, trunk_dtype=jnp.bfloat16, **kwargs):
    features, schedule = _setup(monkeypatch, trunk_dtype)
    return _run(params, features, schedule, **kwargs)


def _float_leaf_dtypes(tree) -> set[str]:
    return {
        str(leaf.dtype)
        for leaf in jax.tree.leaves(tree)
        if hasattr(leaf, "dtype") and jnp.issubdtype(leaf.dtype, jnp.floating)
    }


def _dot_operand_dtypes(jaxpr) -> list[tuple[str, ...]]:
    """Every ``dot_general`` in a closed jaxpr, as its two operand dtypes."""

    inner = jaxpr.jaxpr if hasattr(jaxpr, "jaxpr") else jaxpr
    return [
        tuple(str(var.aval.dtype) for var in eqn.invars[:2])
        for eqn in inner.eqns
        if eqn.primitive.name == "dot_general"
    ]


# ------------------------------------------------- realised parameter trees --


def test_the_narrowed_group_is_the_re_embedding_stack() -> None:
    """AF3 narrows the re-embedding Pairformer and what feeds it, nothing else.

    ``input_strunk_ln`` and the blocks take a plain narrow weight because the
    activation reaching them is already narrow; the four projections reached
    with FP32 -- the two distance projections and the outer-sum initialiser --
    narrow their own operands instead.
    """

    realised = cast_confidence_params(_params(), jnp.bfloat16).confidence

    assert realised.input_strunk_ln.weight.dtype == jnp.bfloat16
    assert realised.input_strunk_ln.bias.dtype == jnp.bfloat16
    for name in ("linear_s1", "linear_s2"):
        node = getattr(realised, name)
        assert isinstance(node, AutocastLinearParams), name
        assert node.weight.dtype == jnp.bfloat16, name
    for name in ("linear_d", "linear_d_wo_onehot"):
        node = getattr(realised.distance_embedding, name)
        assert isinstance(node, AutocastLinearParams), name
        assert node.weight.dtype == jnp.bfloat16, name


def test_the_output_logit_heads_stay_wide() -> None:
    """The four logits feed a softmax, so nothing in that group is rounded.

    ``confidence_scores_from_logits`` exponentiates all four -- pLDDT and
    resolved through their bin softmaxes, PAE and PDE through theirs -- which
    is the one rounding this port has measured as harmful elsewhere. AF3 casts
    ``pair_act`` back to FP32 at :163 and ``single_act`` at :244 for exactly
    this reason, and ``confidence_output_logits`` already reproduces that
    boundary on the activation side.
    """

    realised = cast_confidence_params(_params(), jnp.bfloat16).confidence

    assert _float_leaf_dtypes(realised.output) == {"float32"}
    # The bins are compared against FP32 distances, never multiplied by them.
    assert realised.distance_embedding.lower_bins.dtype == jnp.float32
    assert realised.distance_embedding.upper_bins.dtype == jnp.float32


def test_the_trunk_cast_leaves_the_confidence_tree_untouched() -> None:
    """`cast_trunk_params` narrows four subtrees; this is not one of them.

    Both casts are released defaults now, so what keeps them two decisions is
    that neither preparer reaches the other's fields.
    """

    params = _params()
    narrowed_trunk = cast_trunk_params(params, jnp.bfloat16)

    assert narrowed_trunk.confidence is params.confidence
    assert _float_leaf_dtypes(narrowed_trunk.confidence) == {"float32"}


def test_a_width_the_preparer_cannot_realise_is_refused() -> None:
    """The allowed set is named in the message rather than left to be guessed."""

    with pytest.raises(ValueError, match="float32, bfloat16"):
        cast_confidence_params(_params(), jnp.float16)


def test_realising_an_already_narrowed_tree_is_refused() -> None:
    """Widening rounded operands cannot recover what the rounding dropped."""

    once = cast_confidence_params(_params(), jnp.bfloat16)
    with pytest.raises(ValueError, match="original FP32"):
        cast_confidence_params(once, jnp.bfloat16)


# ------------------------------------------------------- the model boundary --


def test_a_dtype_the_parameters_were_not_rebuilt_for_is_refused(monkeypatch) -> None:
    """Out of step, BF16 activations meet FP32 weights and promote silently.

    Nothing about that failure is visible in the output -- it is a run that
    looks exactly like one that never asked for the option -- so the two
    halves are checked against each other rather than trusted.
    """

    with pytest.raises(ValueError, match="were left FP32"):
        _infer(_params(), monkeypatch, confidence_dtype=jnp.bfloat16)
    with pytest.raises(ValueError, match="were narrowed"):
        _infer(cast_confidence_params(_params(), jnp.bfloat16), monkeypatch)


def test_parameters_without_a_confidence_head_refuse_the_dtype(monkeypatch) -> None:
    """A ``stop_after`` tree carries no head; an absent stage cannot comply."""

    headless = _params()._replace(confidence=object())
    with pytest.raises(ValueError, match="no confidence head"):
        _infer(headless, monkeypatch, confidence_dtype=jnp.bfloat16)
    # And it is still accepted when nothing was asked of it.
    _infer(headless, monkeypatch, run_confidence=False)


@pytest.mark.parametrize("narrow", [False, True])
def test_the_re_embedding_stack_is_entered_in_the_requested_dtype(
    monkeypatch, narrow: bool
) -> None:
    """Record what the stack was handed, not what was configured.

    The toy head has an empty stack, so a sentinel block turns the call on and
    the recorder replaces the stack itself. ``seen`` doubling as the tripwire
    is deliberate: a recorder that never fired would assert nothing at all.
    """

    seen: list[tuple[str, str]] = []

    def record(s_single, z_pair, *args, **kwargs):
        seen.append((str(s_single.dtype), str(z_pair.dtype)))
        return s_single, z_pair

    monkeypatch.setattr(confidence_impl, "pairformer_stack", record)
    params = _params()
    confidence = params.confidence._replace(
        pairformer_stack=PairformerStackParams(blocks=(object(),))
    )
    if narrow:
        confidence = cast_confidence_params(
            params._replace(confidence=params.confidence), jnp.bfloat16
        ).confidence._replace(
            pairformer_stack=PairformerStackParams(blocks=(object(),))
        )
    _infer(
        params._replace(confidence=confidence),
        monkeypatch,
        confidence_dtype=jnp.bfloat16 if narrow else None,
    )

    assert seen, "the confidence pairformer stack was never entered"
    expected = "bfloat16" if narrow else "float32"
    assert set(seen) == {(expected, expected)}


def test_the_narrowed_weights_reach_a_bfloat16_matmul(monkeypatch) -> None:
    """The tripwire: narrowed weights are worthless if XLA promotes them back.

    A plain BF16 weight against an FP32 activation gives an FP32 ``dot_general``
    with an upcast operand -- all of the rounding and none of the saving. This
    traces the program the CLI compiles, with compact binning on, and reads the
    operand dtypes off the equations. The FP32 arm is the x2 control: the same
    projections, no BF16 dot anywhere.
    """

    def dots(params, confidence_dtype):
        # Traced as a closure: the parameter tree carries sentinel stages for
        # everything this test stubs out, and those are not abstract values.
        features, schedule = _setup(monkeypatch)
        traced = jax.make_jaxpr(
            lambda: _run(params, features, schedule, confidence_dtype=confidence_dtype)
        )()
        return _dot_operand_dtypes(traced)

    wide = dots(_params(), None)
    narrow = dots(cast_confidence_params(_params(), jnp.bfloat16), jnp.bfloat16)

    # Same program, same matmuls: only their operand widths move.
    assert len(narrow) == len(wide)
    assert wide.count(("float32", "float32")) == len(wide)
    # `linear_s1`, `linear_s2` and `linear_d_wo_onehot`: the three real matmuls
    # a compact-binned head runs before its stack. The bin projection is a
    # gather on this path, which is why it is not a fourth.
    assert narrow.count(("bfloat16", "bfloat16")) == 3
    # What is left wide is the output stage -- `linear_pae`, `linear_pde` and
    # the pLDDT and resolved einsums -- which is the group that feeds a
    # softmax and the group AF3 casts back to FP32 before.
    assert narrow.count(("float32", "float32")) == len(wide) - 3


def test_the_scores_come_back_in_float32(monkeypatch) -> None:
    """Whatever the stack ran in, the published logits are FP32."""

    output = _infer(
        cast_confidence_params(_params(), jnp.bfloat16),
        monkeypatch,
        confidence_dtype=jnp.bfloat16,
    )
    for name in ("plddt", "pae", "pde", "resolved"):
        assert output[name].dtype == jnp.float32, name
    assert output["coordinate"].dtype == jnp.float32


def test_the_option_is_independent_of_the_trunk_dtype(monkeypatch) -> None:
    """An FP32 trunk still gets a BF16 confidence head, unlike Protenix.

    Protenix's flag reproduces a torch autocast context, so an FP32 trunk
    leaves it nothing to narrow. OpenDDE widens its trunk outputs to FP32
    before every head, so this option casts the head's activations itself --
    AF3's own arrangement -- and the two decisions stay separable.
    """

    seen: list[str] = []

    def record(s_single, z_pair, *args, **kwargs):
        seen.append(str(z_pair.dtype))
        return s_single, z_pair

    monkeypatch.setattr(confidence_impl, "pairformer_stack", record)
    params = _params()
    confidence = cast_confidence_params(params, jnp.bfloat16).confidence._replace(
        pairformer_stack=PairformerStackParams(blocks=(object(),))
    )
    _infer(
        params._replace(confidence=confidence),
        monkeypatch,
        trunk_dtype=jnp.float32,
        confidence_dtype=jnp.bfloat16,
    )

    assert seen == ["bfloat16"]




# ----------------------------------------------- the head that is not narrowed --


def test_the_distogram_head_never_sees_the_confidence_dtype(monkeypatch) -> None:
    """It reads the FP32 trunk copy and runs before the cast, at :1175.

    Two things keep it out: ``cast_confidence_params`` rebuilds only
    ``params.confidence``, and the model projects the distogram from
    ``head_z_trunk`` before it narrows the three activations the head takes.
    """

    params = _params()
    narrowed = cast_confidence_params(params, jnp.bfloat16)
    assert narrowed.distogram is params.distogram

    output = _infer(narrowed, monkeypatch, confidence_dtype=jnp.bfloat16)
    assert output["distogram_logits"].dtype == jnp.float32


# ---------------------------------------------------- the shared Protenix head --


def test_the_shared_head_takes_its_width_from_the_parameters(monkeypatch) -> None:
    """OpenDDE's default cannot reach Protenix, and this is the mechanism.

    ``confidence_head`` is Protenix's module -- OpenDDE imports it rather than
    owning a copy -- so a default flipped in OpenDDE's CLI could only reach
    Protenix through the shared code. It cannot: nothing in the shared head
    names a dtype. The width arrives entirely through the parameter tree and
    the activations, and Protenix resolves those from its own token gate in
    ``foldjax.models.protenix.amp_policy``, never from
    ``cast_confidence_params``.

    So the same call with an unprepared tree -- what Protenix's fp32 policy
    passes -- still runs float32 end to end, under any OpenDDE default.
    """

    import inspect

    for function in (
        confidence_impl.confidence_head,
        confidence_impl.confidence_head_single_sample,
        confidence_impl.confidence_distance_embedding,
    ):
        named = sorted(inspect.signature(function).parameters)
        assert not [name for name in named if "dtype" in name], (
            function.__name__,
            named,
        )

    features, schedule = _setup(monkeypatch)
    wide = _dot_operand_dtypes(
        jax.make_jaxpr(lambda: _run(_params(), features, schedule))()
    )
    assert wide, "no matmul was traced"
    assert set(wide) == {("float32", "float32")}


# ------------------------------------------------------------------ wiring --


def _captured_parser():
    """The real ``predict`` parser, captured before it consumes any argv."""

    import argparse

    from foldjax.models.opendde.cli import predict as predict_cli

    captured: list[argparse.ArgumentParser] = []

    class _StopError(Exception):
        pass

    def capture(parser, *args, **kwargs):
        captured.append(parser)
        raise _StopError

    original = argparse.ArgumentParser.parse_args
    argparse.ArgumentParser.parse_args = capture
    try:
        predict_cli.main([])
    except _StopError:
        pass
    finally:
        argparse.ArgumentParser.parse_args = original
    return captured[0]


def _parser_default(name: str) -> str:
    parser = _captured_parser()
    return next(
        action.default for action in parser._actions if action.dest == name
    )


def _cache_request(tmp_path, **options):
    import json

    from foldjax.schema import PredictionRequest

    input_path = tmp_path / "job.json"
    input_path.write_text(
        json.dumps([{"name": "tiny", "modelSeeds": [0], "sequences": []}]),
        encoding="utf-8",
    )
    weights = tmp_path / "opendde.jax"
    weights.write_bytes(b"native fixture")
    return PredictionRequest(
        model="opendde",
        input=input_path,
        input_format="native",
        weights=weights,
        output_dir=tmp_path / "out",
        cache_dir=tmp_path / "cache",
        options=dict(options),
    )


def test_the_value_joins_the_compilation_cache_identity() -> None:
    """A run that narrows the head must not hit a cache entry that did not."""

    from foldjax.backends.opendde import _CLI_OPTIONS, OpenDDEBackend

    assert "confidence_dtype" in _CLI_OPTIONS
    assert "confidence_dtype" in OpenDDEBackend.compile_options
    assert "confidence_dtype" in model_impl.GRAPH_STATIC_ARGNAMES


def test_the_default_spelled_out_names_the_same_namespace_as_unset(tmp_path) -> None:
    """Whatever the default is, saying it must not select a second namespace.

    Written against the parser's own value rather than a literal so that the
    property survives the next flip: asking for what the CLI would have
    supplied is the same run, and the other width is not.
    """

    from foldjax.backends.opendde import OpenDDEBackend

    backend = OpenDDEBackend()
    default = _parser_default("confidence_dtype")
    other = next(value for value in ("fp32", "bf16") if value != default)

    omitted = backend.cache_profile(_cache_request(tmp_path))
    spelled = backend.cache_profile(
        _cache_request(tmp_path, confidence_dtype=default)
    )
    pinned = backend.cache_profile(_cache_request(tmp_path, confidence_dtype=other))

    assert "confidence_dtype" not in omitted
    assert spelled == omitted
    assert pinned["confidence_dtype"] == other
    assert pinned != omitted


def test_the_cli_refuses_an_unknown_width_naming_the_allowed_ones(capsys) -> None:
    """An unrecognised width must name the set, not fall back to a default."""

    parser = _captured_parser()
    required = ["--input-json", "in.json", "--out", "out", "--weights", "w.npz"]

    with pytest.raises(SystemExit):
        parser.parse_args([*required, "--confidence-dtype", "bfloat16"])
    assert "'fp32', 'bf16'" in capsys.readouterr().err


# --------------------------------------------------------- the released run --


def _run_cli(monkeypatch, tmp_path, argv_extra=()):
    """Drive the native CLI down to ``_predict`` with a real confidence tree."""

    import json

    import foldjax.models.opendde.cli.predict as predict_impl

    input_path = tmp_path / "tiny.json"
    weights_path = tmp_path / "opendde.jax"
    job = {"name": "tiny", "modelSeeds": [1], "sequences": []}
    input_path.write_text(json.dumps([job]), encoding="utf-8")
    weights_path.write_bytes(b"native fixture")
    loaded = _params()
    calls: list[tuple[object, dict[str, object]]] = []

    monkeypatch.setattr(predict_impl, "_load_jobs", lambda path: [job])
    monkeypatch.setattr(
        predict_impl,
        "_featurize",
        lambda value, **kwargs: {
            "restype": np.zeros((N_TOKEN, 32), dtype=np.float32)
        },
    )
    monkeypatch.setattr(
        predict_impl, "_load_prepared_params", lambda path, trunk_dtype: loaded
    )

    def fake_predict(value, model_params, **kwargs):
        calls.append((model_params, kwargs))
        return {"coordinate": np.zeros((1, N_ATOM, 3), dtype=np.float32)}

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


def test_a_run_that_asks_for_nothing_gets_a_realised_bfloat16_head(
    monkeypatch, tmp_path
) -> None:
    """No flag, and the tree reaching the model is already rebuilt.

    The parser's string is not the evidence: an option that is resolved but
    never applied is exactly the failure ``_require_realised_confidence_params``
    exists to catch, and it would look identical from the outside.
    """

    loaded, calls = _run_cli(monkeypatch, tmp_path)
    params, kwargs = calls[0]

    assert kwargs["confidence_dtype"] == jnp.bfloat16
    node = params.confidence.distance_embedding.linear_d
    assert isinstance(node, AutocastLinearParams)
    assert node.weight.dtype == jnp.bfloat16
    assert params.confidence.input_strunk_ln.weight.dtype == jnp.bfloat16
    # The weight session memoizes on `trunk_dtype` alone, so the cast has to
    # leave its tree alone or a second policy in one process is served this
    # one's weights.
    assert isinstance(loaded.confidence.distance_embedding.linear_d, LinearParams)
    assert _float_leaf_dtypes(loaded.confidence) == {"float32"}


def test_pinning_fp32_hands_the_model_the_checkpoints_own_head(
    monkeypatch, tmp_path
) -> None:
    """The escape hatch: no rebuild, no dtype, the loader's own subtree."""

    loaded, calls = _run_cli(monkeypatch, tmp_path, ("--confidence-dtype", "fp32"))
    params, kwargs = calls[0]

    assert kwargs["confidence_dtype"] is None
    assert params.confidence is loaded.confidence
    assert _float_leaf_dtypes(params.confidence) == {"float32"}


# ------------------------------------ the stack, with blocks that are not empty --


def _uniform_head(n_blocks: int = 2, *, channels: int = 8, heads: int = 2):
    """A confidence head whose Pairformer stack is real.

    The toy head above carries ``blocks=()``, which is enough to record what
    the stack is handed and says nothing about what the blocks realise. This
    one has the same channel count on both representations so that one set of
    shapes builds every projection in a block.
    """

    from foldjax.models.protenix.models.primitives.attention import (
        AttentionPairBiasParams,
        AttentionParams,
    )
    from foldjax.models.protenix.models.primitives.primitives import (
        TransitionParams,
    )
    from foldjax.models.protenix.models.triangle.triangle import (
        TriangleAttentionParams,
        TriangleMultiplicationParams,
    )
    from foldjax.models.protenix.models.trunk_blocks.pairformer import (
        PairformerBlockParams,
    )

    c = channels

    def lin(out_features, in_features):
        return LinearParams(_array(out_features, in_features), _array(out_features))

    def norm():
        return LayerNormParams(_array(c) * 0.1 + 1.0, _array(c) * 0.1)

    def attention():
        return AttentionParams(lin(c, c), lin(c, c), lin(c, c), lin(c, c), lin(c, c))

    def transition():
        return TransitionParams(
            layer_norm=norm(),
            linear_a=lin(2 * c, c),
            linear_b=lin(2 * c, c),
            linear_out=lin(c, 2 * c),
        )

    def block():
        return PairformerBlockParams(
            tri_mul_out=TriangleMultiplicationParams(
                layer_norm_in=norm(), layer_norm_out=norm(),
                linear_a_p=lin(c, c), linear_a_g=lin(c, c),
                linear_b_p=lin(c, c), linear_b_g=lin(c, c),
                linear_z=lin(c, c), linear_g=lin(c, c),
            ),
            tri_mul_in=TriangleMultiplicationParams(
                layer_norm_in=norm(), layer_norm_out=norm(),
                linear_a_p=lin(c, c), linear_a_g=lin(c, c),
                linear_b_p=lin(c, c), linear_b_g=lin(c, c),
                linear_z=lin(c, c), linear_g=lin(c, c),
            ),
            tri_att_start=TriangleAttentionParams(
                layer_norm=norm(),
                linear=LinearParams(_array(heads, c)),
                attention=attention(),
            ),
            tri_att_end=TriangleAttentionParams(
                layer_norm=norm(),
                linear=LinearParams(_array(heads, c)),
                attention=attention(),
            ),
            pair_transition=transition(),
            attention_pair_bias=AttentionPairBiasParams(
                layernorm_a=norm(), layernorm_kv=None, attention=attention(),
                layernorm_z=norm(),
                linear_z=LinearParams(_array(heads, c)),
                has_s=False, cross_attention_mode=False,
            ),
            single_transition=transition(),
        )

    n_bins = 8
    lower = jnp.asarray(np.linspace(2.0, 18.0, n_bins), dtype=jnp.float32)
    upper = jnp.concatenate([lower[1:], jnp.asarray([22.0], dtype=jnp.float32)])
    return ConfidenceHeadParams(
        input_strunk_ln=norm(),
        linear_s1=lin(c, c), linear_s2=lin(c, c),
        distance_embedding=ConfidenceDistanceEmbeddingParams(
            lower_bins=lower, upper_bins=upper,
            linear_d=lin(c, n_bins), linear_d_wo_onehot=lin(c, 1),
        ),
        pairformer_stack=PairformerStackParams(
            blocks=tuple(block() for _ in range(n_blocks))
        ),
        output=ConfidenceOutputParams(
            pae_ln=norm(), pde_ln=norm(), plddt_ln=norm(), resolved_ln=norm(),
            linear_pae=lin(N_OUT, c), linear_pde=lin(N_OUT, c),
            plddt_weight=_array(2, c, N_OUT), resolved_weight=_array(2, c, N_OUT),
        ),
    )


def _head_dots(head, activation_dtype):
    """Trace the shared head the way OpenDDE calls it, and read every matmul."""

    n_token, n_atom, channels = 6, 9, 8
    cast = (lambda x: x) if activation_dtype is None else (
        lambda x: x.astype(activation_dtype)
    )
    s_inputs, s_trunk = _array(n_token, channels), _array(n_token, channels)
    z_trunk = _array(n_token, n_token, channels)
    rep_coords = _array(n_token, 3) * 6.0
    atom_to_token = jnp.asarray(
        np.random.default_rng(1).integers(0, n_token, n_atom), jnp.int32
    )
    atom_to_tokatom = jnp.asarray(
        np.random.default_rng(2).integers(0, 2, n_atom), jnp.int32
    )

    entered: list[tuple[str, str]] = []
    real_stack = confidence_impl.pairformer_stack

    def recording_stack(s_single, z_pair, *args, **kwargs):
        entered.append((str(s_single.dtype), str(z_pair.dtype)))
        return real_stack(s_single, z_pair, *args, **kwargs)

    confidence_impl.pairformer_stack = recording_stack
    try:
        traced = jax.make_jaxpr(
            lambda: confidence_impl.confidence_head_single_sample(
                cast(s_inputs), cast(s_trunk), cast(z_trunk), None, rep_coords,
                atom_to_token, atom_to_tokatom, head,
                # What both compiled wrappers resolve to on a released
                # checkpoint, so this is the route the CLI takes.
                compact_distance_bins=True, use_scan=False,
            )
        )()
    finally:
        confidence_impl.pairformer_stack = real_stack
    return entered, _dot_operand_dtypes(traced)


def test_every_confidence_pairformer_block_realises_bfloat16() -> None:
    """Blocks, not just the entry: an FP32 operand anywhere is the old defect.

    The FP32 arm is the x2 control -- same program, same matmuls -- so a count
    that moved because the head changed shape cannot be read as a saving.

    What stays FP32 is pinned to the output stage structurally rather than by
    line number: a head with no blocks at all runs the same three
    re-embedding matmuls and the same output stage, so its FP32 count is the
    output stage's own, and the blocked head must not add to it.
    """

    head = _uniform_head()
    narrow = cast_confidence_params(_params()._replace(confidence=head), jnp.bfloat16)

    wide_entry, wide = _head_dots(head, None)
    narrow_entry, narrowed = _head_dots(narrow.confidence, jnp.bfloat16)

    assert wide_entry == [("float32", "float32")]
    assert narrow_entry == [("bfloat16", "bfloat16")]
    assert len(narrowed) == len(wide)
    assert set(wide) == {("float32", "float32")}

    stackless = _uniform_head(n_blocks=0)
    _, outside = _head_dots(
        cast_confidence_params(
            _params()._replace(confidence=stackless), jnp.bfloat16
        ).confidence,
        jnp.bfloat16,
    )
    output_stage = outside.count(("float32", "float32"))
    # `linear_s1`, `linear_s2` and `linear_d_wo_onehot`: the three real
    # matmuls a compact-binned re-embedding runs. The bin projection is a
    # gather on this path, which is why it is not a fourth.
    assert outside.count(("bfloat16", "bfloat16")) == 3

    assert narrowed.count(("float32", "float32")) == output_stage
    assert narrowed.count(("bfloat16", "bfloat16")) == len(wide) - output_stage


def test_the_compact_bin_projection_follows_the_weight_on_openddes_route() -> None:
    """The defect that returned FP32 from a fully realised BF16 policy.

    Protenix pins this for its own preparer; the head is the same module and
    OpenDDE reaches it through ``cast_confidence_params``, so the arm that
    matters here is that tree. Promoting against the FP32 distances instead
    of following the weight widened ``z_pair`` and every block after it.
    """

    narrow = cast_confidence_params(_params(), jnp.bfloat16).confidence
    coords = jnp.asarray(
        np.random.default_rng(3).normal(size=(N_TOKEN, 3)) * 8.0, jnp.float32
    )

    dense = confidence_impl.confidence_distance_embedding(
        coords, narrow.distance_embedding, compact_bins=False
    )
    compact = confidence_impl.confidence_distance_embedding(
        coords, narrow.distance_embedding, compact_bins=True
    )
    assert dense.dtype == jnp.bfloat16
    assert compact.dtype == jnp.bfloat16
    np.testing.assert_array_equal(np.asarray(compact), np.asarray(dense))
