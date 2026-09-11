"""OpenDDE's opt-in bfloat16 confidence head, asserted as realised dtypes.

Every assertion here reads an array's ``dtype`` or an operand of a traced
``dot_general``. None reads a configuration value: a run that asked for the
option and a run whose weights were never rebuilt for it look identical from
the outside, which is the whole reason the option carries a guard.
"""

from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import foldjax.models.opendde.models.model as model_impl
import foldjax.models.protenix.models.heads.confidence as confidence_impl
from foldjax.models.opendde.models.model import (
    OpenDDEInferenceParams,
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

N_TOKEN, N_ATOM, C_S_INPUTS, C_S, C_Z, N_BINS, N_OUT = 2, 3, 5, 4, 3, 8, 2


def _array(*shape: int) -> jnp.ndarray:
    rng = np.random.default_rng(sum(shape) * 7 + len(shape))
    return jnp.asarray(rng.normal(size=shape), dtype=jnp.float32)


def _confidence_params() -> ConfidenceHeadParams:
    """A released-shaped head: adjacent finite bins, so compact binning is exact."""

    lower = jnp.asarray(np.linspace(2.0, 18.0, N_BINS), dtype=jnp.float32)
    upper = jnp.concatenate([lower[1:], jnp.asarray([22.0], dtype=jnp.float32)])
    return ConfidenceHeadParams(
        input_strunk_ln=LayerNormParams(_array(C_S), _array(C_S)),
        linear_s1=LinearParams(_array(C_Z, C_S_INPUTS)),
        linear_s2=LinearParams(_array(C_Z, C_S_INPUTS)),
        distance_embedding=ConfidenceDistanceEmbeddingParams(
            lower_bins=lower,
            upper_bins=upper,
            linear_d=LinearParams(_array(C_Z, N_BINS)),
            linear_d_wo_onehot=LinearParams(_array(C_Z, 1)),
        ),
        pairformer_stack=PairformerStackParams(blocks=()),
        output=ConfidenceOutputParams(
            pae_ln=LayerNormParams(_array(C_Z), _array(C_Z)),
            pde_ln=LayerNormParams(_array(C_Z), _array(C_Z)),
            plddt_ln=LayerNormParams(_array(C_S), _array(C_S)),
            resolved_ln=LayerNormParams(_array(C_S), _array(C_S)),
            linear_pae=LinearParams(_array(N_OUT, C_Z)),
            linear_pde=LinearParams(_array(N_OUT, C_Z)),
            plddt_weight=_array(2, C_S, N_OUT),
            resolved_weight=_array(2, C_S, N_OUT),
        ),
    )


def _params() -> OpenDDEInferenceParams:
    return OpenDDEInferenceParams(
        input_embedder=object(),
        pairformer_output=object(),
        structural_expander=object(),
        structural_refiner=object(),
        diffusion=SimpleNamespace(
            conditioning=SimpleNamespace(relpe=object()),
            atom_encoder=object(),
        ),
        distogram=object(),
        confidence=_confidence_params(),
    )


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


def test_the_released_default_leaves_the_confidence_tree_untouched() -> None:
    """The trunk default narrows four subtrees; this is not one of them."""

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


# ------------------------------------------------------------------ wiring --


def test_the_value_joins_the_compilation_cache_identity() -> None:
    """A run that narrows the head must not hit a cache entry that did not."""

    from foldjax.backends.opendde import (
        _CLI_OPTIONS,
        _RELEASED_COMPILE_DEFAULTS,
        OpenDDEBackend,
    )

    assert "confidence_dtype" in _CLI_OPTIONS
    assert "confidence_dtype" in OpenDDEBackend.compile_options
    assert _RELEASED_COMPILE_DEFAULTS["confidence_dtype"] == "fp32"
    assert "confidence_dtype" in model_impl.GRAPH_STATIC_ARGNAMES


def test_the_cli_refuses_an_unknown_width_naming_the_allowed_ones(
    monkeypatch, capsys
) -> None:
    """An unrecognised width must name the set, not fall back to a default."""

    import argparse

    from foldjax.models.opendde.cli import predict as predict_cli

    captured: list[argparse.ArgumentParser] = []

    class _StopError(Exception):
        pass

    def capture(parser, *args, **kwargs):
        captured.append(parser)
        raise _StopError

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", capture)
    with pytest.raises(_StopError):
        predict_cli.main([])
    monkeypatch.undo()
    parser = captured[0]

    required = ["--input-json", "in.json", "--out", "out", "--weights", "w.npz"]

    assert parser.parse_args(required).confidence_dtype == "fp32"
    with pytest.raises(SystemExit):
        parser.parse_args([*required, "--confidence-dtype", "bfloat16"])
    assert "'fp32', 'bf16'" in capsys.readouterr().err
