"""OpenFold3's ``dtype`` must narrow the arrays, not merely be configured.

Asserting a setting is the failure mode this file exists to avoid. ESMFold2's
``ModelSettings.trunk_dtype`` said ``bfloat16``, its gate asserted that field,
and all forty-eight folding layers ran float32 for as long as anyone looked --
because one float32 operand entering a residual stream widens everything after
it, and a parameter cast says nothing about an activation. So every assertion
below reads ``.dtype`` off an array the model actually produced, or reads the
lowered program.

The parameters here are synthetic and the dimensions are tiny. That is
deliberate and it is what lets this run on CPU in every checkout: the
``torch_parity`` fixtures need an upstream checkout and a torch install, and a
gate that can only run where those exist is a gate that runs nowhere.
"""

from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.openfold3 import inference as inference_module
from foldjax.models.openfold3.dtype import (
    DEFAULT_DTYPE,
    DTYPES,
    narrow_dtype,
    narrow_floats,
)
from foldjax.models.openfold3.inference import cast_narrow_params, released_config
from foldjax.models.openfold3.models import trunk as trunk_module
from foldjax.models.openfold3.models.attention import AttentionParams
from foldjax.models.openfold3.models.attention_pair_bias import AttentionPairBiasParams
from foldjax.models.openfold3.models.heads import (
    PairformerEmbeddingParams,
    pairformer_embedding,
)
from foldjax.models.openfold3.models.input_embedders import (
    MSAEmbedderParams,
    msa_embedder,
)
from foldjax.models.openfold3.models.pair_block import PairBlockParams
from foldjax.models.openfold3.models.pairformer import (
    PairformerBlockParams,
    PairformerStackParams,
    pairformer_stack,
)
from foldjax.models.openfold3.models.primitives import (
    LayerNormParams,
    LinearParams,
    SwiGLUParams,
    SwiGLUTransitionParams,
)
from foldjax.models.openfold3.models.triangle import TriangleMultiplicationParams
from foldjax.models.openfold3.models.triangle_attention import TriangleAttentionParams

N_TOKEN = 5
C_S = 4
C_Z = 4
C_HIDDEN = 2
NO_HEADS = 2
NO_BINS = 3


def _rng():
    return np.random.default_rng(0)


def _linear(out_dim: int, in_dim: int, rng, *, bias: bool = True) -> LinearParams:
    return LinearParams(
        weight=jnp.asarray(rng.normal(size=(out_dim, in_dim)), jnp.float32),
        bias=(jnp.asarray(rng.normal(size=(out_dim,)), jnp.float32) if bias else None),
    )


def _layer_norm(dim: int, rng) -> LayerNormParams:
    return LayerNormParams(
        weight=jnp.asarray(rng.normal(size=(dim,)), jnp.float32),
        bias=jnp.asarray(rng.normal(size=(dim,)), jnp.float32),
    )


def _transition(dim: int, rng) -> SwiGLUTransitionParams:
    return SwiGLUTransitionParams(
        layer_norm=_layer_norm(dim, rng),
        swiglu=SwiGLUParams(
            linear_a=_linear(2 * dim, dim, rng, bias=False),
            linear_b=_linear(2 * dim, dim, rng, bias=False),
        ),
        linear_out=_linear(dim, 2 * dim, rng, bias=False),
    )


def _tri_mul(rng) -> TriangleMultiplicationParams:
    return TriangleMultiplicationParams(
        layer_norm_in=_layer_norm(C_Z, rng),
        layer_norm_out=_layer_norm(C_HIDDEN, rng),
        linear_a_p=_linear(C_HIDDEN, C_Z, rng),
        linear_a_g=_linear(C_HIDDEN, C_Z, rng),
        linear_b_p=_linear(C_HIDDEN, C_Z, rng),
        linear_b_g=_linear(C_HIDDEN, C_Z, rng),
        linear_g=_linear(C_Z, C_Z, rng),
        linear_z=_linear(C_Z, C_HIDDEN, rng),
    )


def _mha(c_in: int, rng, *, gated: bool = True) -> AttentionParams:
    width = NO_HEADS * C_HIDDEN
    return AttentionParams(
        linear_q=_linear(width, c_in, rng, bias=False),
        linear_k=_linear(width, c_in, rng, bias=False),
        linear_v=_linear(width, c_in, rng, bias=False),
        linear_o=_linear(c_in, width, rng),
        linear_g=_linear(width, c_in, rng, bias=False) if gated else None,
    )


def _tri_att(rng) -> TriangleAttentionParams:
    return TriangleAttentionParams(
        layer_norm=_layer_norm(C_Z, rng),
        linear_z=_linear(NO_HEADS, C_Z, rng, bias=False),
        mha=_mha(C_Z, rng),
    )


def _pair_block(rng) -> PairBlockParams:
    return PairBlockParams(
        tri_mul_out=_tri_mul(rng),
        tri_mul_in=_tri_mul(rng),
        tri_att_start=_tri_att(rng),
        tri_att_end=_tri_att(rng),
        pair_transition=_transition(C_Z, rng),
    )


def _pairformer(rng, *, blocks: int = 1) -> PairformerStackParams:
    return PairformerStackParams(
        blocks=tuple(
            PairformerBlockParams(
                pair_stack=_pair_block(rng),
                attn_pair_bias=AttentionPairBiasParams(
                    layer_norm_a=_layer_norm(C_S, rng),
                    layer_norm_z=_layer_norm(C_Z, rng),
                    linear_z=_linear(NO_HEADS, C_Z, rng, bias=False),
                    mha=_mha(C_S, rng),
                ),
                single_transition=_transition(C_S, rng),
            )
            for _ in range(blocks)
        )
    )


def _confidence(rng) -> PairformerEmbeddingParams:
    return PairformerEmbeddingParams(
        linear_i=_linear(C_Z, C_S, rng, bias=False),
        linear_j=_linear(C_Z, C_S, rng, bias=False),
        linear_distance=_linear(C_Z, NO_BINS, rng, bias=False),
        pairformer_stack=_pairformer(rng),
    )


def _stack_inputs():
    rng = _rng()
    s = jnp.asarray(rng.normal(size=(N_TOKEN, C_S)), jnp.float32)
    z = jnp.asarray(rng.normal(size=(N_TOKEN, N_TOKEN, C_Z)), jnp.float32)
    mask = jnp.ones((N_TOKEN,), jnp.float32)
    return s, z, mask


def _both(name: str):
    """The two narrowing dtypes for a profile that sets only ``dtype``."""
    from foldjax.models.openfold3.inference import resolve_dtypes

    return resolve_dtypes(released_config(n_token=8, n_atom=32, dtype=name))


def _leaf_dtypes(tree) -> set:
    return {
        leaf.dtype
        for leaf in jax.tree.leaves(tree)
        if jnp.issubdtype(getattr(leaf, "dtype", jnp.int32), jnp.floating)
    }


# ---------------------------------------------------------------- the default


def test_the_released_default_narrows_nothing() -> None:
    """The shipped profile is upstream's 32-true; bfloat16 is opt-in.

    models/openfold3/dtype.py records why: the narrow profile is 23-35%
    faster and 24-40% smaller at every size measured, and at 3,012 tokens it
    moves the structure 4.65-5.80 A against a 0.619 A control spread.

    Asserted as the resolved pair rather than as config strings, because the
    strings are what a dead branch would also still carry. Both entries,
    because ``confidence_dtype`` follows ``dtype``: a default that narrowed
    only the head would still satisfy a one-sided assertion.
    """
    from foldjax.models.openfold3.inference import resolve_dtypes

    assert resolve_dtypes(released_config(n_token=8, n_atom=32)) == (None, None)
    assert narrow_dtype(DEFAULT_DTYPE) is None
    assert narrow_dtype("bfloat16") is jnp.bfloat16
    assert resolve_dtypes(
        released_config(n_token=8, n_atom=32, dtype="bfloat16")
    ) == (jnp.bfloat16, jnp.bfloat16)


def test_an_unknown_dtype_is_refused_naming_the_allowed_values() -> None:
    with pytest.raises(ValueError, match=r"must be one of float32, bfloat16"):
        narrow_dtype("bf16")
    with pytest.raises(ValueError, match=r"must be one of float32, bfloat16"):
        released_config(n_token=8, n_atom=32, dtype="fp8")
    assert DTYPES == ("float32", "bfloat16")


def test_a_default_run_returns_the_parameter_tree_it_was_given() -> None:
    """Not "equal to": the same object, so a default run cannot pay for a copy.

    The shipped profile must emit no cast at all rather than a tree of
    float32-to-float32 converts that a reader would have to prove are free.
    """
    params = _confidence(_rng())
    assert cast_narrow_params(params, *_both("float32")) is params
    shipped = inference_module.resolve_dtypes(released_config(n_token=8, n_atom=32))
    assert cast_narrow_params(params, *shipped) is params


# ------------------------------------------------- which parameters are cast

#: Every field of `InferenceParams`, split by what `cast_narrow_params` does to
#: it. Spelled out rather than derived so that adding a subtree to the model and
#: forgetting to classify it fails here instead of silently defaulting to wide.
_NARROWED = (
    ("trunk", "msa_module_embedder"),
    ("trunk", "msa_module"),
    ("trunk", "pairformer_stack"),
    ("trunk", "layer_norm_z"),
    ("trunk", "linear_z"),
    ("trunk", "layer_norm_s"),
    ("trunk", "linear_s"),
    ("trunk", "template_embedder"),
    ("diffusion_conditioning", "layer_norm_z"),
    ("diffusion_conditioning", "linear_z"),
    ("diffusion_conditioning", "transition_z"),
    ("pairformer_embedding", "pairformer_stack"),
)

_LEFT_WIDE = (
    ("trunk", "input_embedder"),
    ("diffusion_conditioning", "layer_norm_s"),
    ("diffusion_conditioning", "linear_s"),
    ("diffusion_conditioning", "fourier_emb"),
    ("diffusion_conditioning", "layer_norm_n"),
    ("diffusion_conditioning", "linear_n"),
    ("diffusion_conditioning", "transition_s"),
    ("pairformer_embedding", "linear_i"),
    ("pairformer_embedding", "linear_j"),
    ("pairformer_embedding", "linear_distance"),
    ("denoiser",),
    ("plddt_head",),
    ("pae_head",),
    ("pde_head",),
    ("distogram_head",),
    ("experimentally_resolved_head",),
)


def _select(params, path):
    for name in path:
        params = getattr(params, name)
    return params


def _synthetic_inference_params():
    """An `InferenceParams` whose structure is real and whose shapes are not.

    `cast_narrow_params` selects subtrees by name and casts leaves; it never
    looks at a shape. Giving every leaf the same scalar keeps this test about
    the classification, which is the part that can be wrong.
    """
    import typing

    from foldjax.models.openfold3.inference import InferenceParams

    def build(annotation):
        annotation = _strip_optional(annotation)
        fields = getattr(annotation, "_fields", None)
        if fields is not None:
            hints = typing.get_type_hints(annotation)
            return annotation(*(build(hints[name]) for name in fields))
        origin = typing.get_origin(annotation)
        if origin is tuple:
            (element, _ellipsis) = typing.get_args(annotation)
            return (build(element),)
        return jnp.ones((2, 2), jnp.float32)

    def _strip_optional(annotation):
        args = [arg for arg in typing.get_args(annotation) if arg is not type(None)]
        if args and typing.get_origin(annotation) is not tuple:
            return args[0]
        return annotation

    hints = typing.get_type_hints(InferenceParams)
    return InferenceParams(*(build(hints[name]) for name in InferenceParams._fields))


def test_the_cast_narrows_exactly_the_named_subtrees() -> None:
    params = _synthetic_inference_params()
    narrowed = cast_narrow_params(params, *_both("bfloat16"))

    for path in _NARROWED:
        assert _leaf_dtypes(_select(narrowed, path)) == {jnp.dtype(jnp.bfloat16)}, path
    for path in _LEFT_WIDE:
        assert _leaf_dtypes(_select(narrowed, path)) == {jnp.dtype(jnp.float32)}, path


def test_the_opt_in_profile_narrows_the_classified_subtrees() -> None:
    """What ``--option dtype=bfloat16`` pays for, and what it leaves alone.

    Driven through :func:`released_config` rather than by handing
    ``cast_narrow_params`` two literals, so the option's own resolution is in
    the path. Identity on the float32 side, not "still float32", because a
    tree cast to float32 and back would also be float32.
    """
    params = _synthetic_inference_params()
    config = released_config(n_token=8, n_atom=32, dtype="bfloat16")
    narrowed = cast_narrow_params(params, *inference_module.resolve_dtypes(config))

    for path in _NARROWED:
        assert _leaf_dtypes(_select(narrowed, path)) == {jnp.dtype(jnp.bfloat16)}, path
    for path in _LEFT_WIDE:
        assert _select(narrowed, path) is _select(params, path), path
    assert narrowed.denoiser is params.denoiser
    assert narrowed.distogram_head is params.distogram_head


def test_the_input_embedder_is_untouched() -> None:
    """Upstream pins it even under `bf16-mixed`; this port measured it collapse.

    Identity, not dtype equality: the embedder subtree must be the very tree
    that was loaded, so no future edit can "preserve float32" by round-tripping
    it through a cast.
    """
    params = _synthetic_inference_params()
    narrowed = cast_narrow_params(params, *_both("bfloat16"))
    assert narrowed.trunk.input_embedder is params.trunk.input_embedder
    assert narrowed.denoiser is params.denoiser


def test_every_subtree_is_classified() -> None:
    """A new parameter group must be named above, not silently left wide.

    Without this, adding a field to `InferenceParams` or to one of the two
    subtrees that get a partial cast would pass every assertion above while
    going unconsidered.
    """
    from foldjax.models.openfold3.inference import InferenceParams

    params = _synthetic_inference_params()
    classified: set[tuple[str, ...]] = set()
    for path in _NARROWED + _LEFT_WIDE:
        classified.add(path)

    def walk(prefix: tuple[str, ...], value) -> None:
        if prefix in classified:
            return
        fields = getattr(value, "_fields", None)
        assert fields is not None, f"unclassified parameter group: {prefix}"
        for name in fields:
            walk((*prefix, name), getattr(value, name))

    for name in InferenceParams._fields:
        walk((name,), getattr(params, name))


# ----------------------------------------------- which activations are narrow


def test_the_msa_embedder_narrows_a_float32_single_representation() -> None:
    """`s_input` stays float32 by design, so the entry cast is load-bearing.

    Without it one float32 operand promotes `m`, and through the MSA module's
    pair carry the whole trunk with it.
    """
    rng = _rng()
    params = MSAEmbedderParams(
        linear_m=_linear(C_S, 34, rng, bias=False),
        linear_s_input=_linear(C_S, C_S, rng, bias=False),
    )
    batch = {
        "msa": jnp.zeros((1, 2, N_TOKEN, 32), jnp.int32),
        "has_deletion": jnp.zeros((1, 2, N_TOKEN), jnp.float32),
        "deletion_value": jnp.zeros((1, 2, N_TOKEN), jnp.float32),
        "msa_mask": jnp.ones((1, 2, N_TOKEN), jnp.float32),
    }
    s_input = jnp.zeros((1, N_TOKEN, C_S), jnp.float32)

    wide_m, wide_mask = msa_embedder(batch, s_input, params)
    assert wide_m.dtype == jnp.float32
    assert wide_mask.dtype == jnp.float32

    narrow = narrow_dtype("bfloat16")
    narrow_m, narrow_mask = msa_embedder(
        batch, s_input, narrow_floats(params, narrow), dtype=narrow
    )
    assert narrow_m.dtype == jnp.bfloat16
    assert narrow_mask.dtype == jnp.bfloat16


def test_the_pairformer_stack_runs_and_returns_bfloat16() -> None:
    """The stack is the largest narrowed subtree, and it carries `(s, z)`.

    Running it scanned is also the cheapest tripwire there is for a boundary
    mistake: `lax.scan` refuses to trace when a float32 leak widens the carry
    ("carry input and carry output must have equal types"), so a pass here is
    evidence about every operand inside the block, not only the two returned.
    """
    narrow = narrow_dtype("bfloat16")
    params = narrow_floats(_pairformer(_rng()), narrow)
    s, z, mask = _stack_inputs()
    s_out, z_out = pairformer_stack(
        s.astype(narrow),
        z.astype(narrow),
        params,
        single_mask=mask.astype(narrow),
        pair_mask=(mask[:, None] * mask[None, :]).astype(narrow),
        no_heads_pair=NO_HEADS,
        no_heads_pair_bias=NO_HEADS,
        scan_blocks=True,
    )
    assert s_out.dtype == jnp.bfloat16
    assert z_out.dtype == jnp.bfloat16


def _confidence_call(params, narrow):
    s, z, mask = _stack_inputs()
    x_pred = jnp.asarray(_rng().normal(size=(N_TOKEN, 3)), jnp.float32)
    return pairformer_embedding(
        s,
        s if narrow is None else s.astype(narrow),
        z if narrow is None else z.astype(narrow),
        x_pred,
        params,
        single_mask=mask,
        pair_mask=mask[:, None] * mask[None, :],
        no_heads_pair=NO_HEADS,
        no_heads_pair_bias=NO_HEADS,
        min_bin=0.0,
        max_bin=4.0,
        no_bin=NO_BINS,
        dtype=narrow,
    )


def _narrowed_confidence_params():
    return cast_narrow_params(
        _synthetic_inference_params()._replace(
            pairformer_embedding=_confidence(_rng())
        ),
        *_both("bfloat16"),
    ).pairformer_embedding


def test_the_confidence_head_runs_narrow_between_two_float32_boundaries() -> None:
    """Both edges of AlphaFold 3's confidence boundary, asserted separately.

    Entering, upstream's ``embed_zij`` runs before its autocast context
    opens (``heads/prediction_heads.py:193`` against ``:224``), and the port
    gets the same float32 entry for free from the parameters on
    ``linear_i``/``linear_j``/``linear_distance``.

    Leaving, AlphaFold 3 restores float32 at ``confidence_head.py:163`` and
    ``:244``, and it does so *before* the logit heads' layer norms. Relying on
    the heads' float32 weights instead would promote only at the matmul, one
    step too late, so three of the five heads would take layer-norm statistics
    in bfloat16. The returned dtype is therefore float32 in both profiles.
    """
    params = _narrowed_confidence_params()
    assert _leaf_dtypes(params.linear_i) == {jnp.dtype(jnp.float32)}
    assert _leaf_dtypes(params.linear_j) == {jnp.dtype(jnp.float32)}
    assert _leaf_dtypes(params.linear_distance) == {jnp.dtype(jnp.float32)}
    assert _leaf_dtypes(params.pairformer_stack) == {jnp.dtype(jnp.bfloat16)}

    s_conf, z_conf = _confidence_call(params, narrow_dtype("bfloat16"))
    assert s_conf.dtype == jnp.float32
    assert z_conf.dtype == jnp.float32


def test_the_confidence_stack_itself_still_runs_bfloat16() -> None:
    """The float32 restore above must not have undone the narrowing.

    A boundary cast on the way out makes the returned dtype float32 whether or
    not anything inside ran narrow, so the returned dtype cannot be the whole
    assertion. This reads the lowered program instead: the confidence path
    must contain bfloat16 dots under the option and none without it.
    """
    wide = jax.jit(_confidence_call, static_argnums=1).lower(_confidence(_rng()), None)
    narrow = jax.jit(_confidence_call, static_argnums=1).lower(
        _narrowed_confidence_params(), jnp.bfloat16
    )
    assert _dot_operand_dtypes(wide.as_text()) == {"f32"}
    assert "bf16" in _dot_operand_dtypes(narrow.as_text())


def test_initialize_trunk_narrows_the_state_and_not_the_input_representation(
    monkeypatch,
) -> None:
    """The embedder's outputs are cast; `s_input` is not, and the embedder runs.

    The stub is what makes this cheap, and a stub that never fires would make
    it vacuous, so the call is counted.
    """
    calls: list[int] = []

    def stub(batch, params, **kwargs):
        calls.append(1)
        return (
            jnp.zeros((N_TOKEN, C_S), jnp.float32),
            jnp.zeros((N_TOKEN, C_S), jnp.float32),
            jnp.zeros((N_TOKEN, N_TOKEN, C_Z), jnp.float32),
        )

    monkeypatch.setattr(trunk_module, "input_embedder", stub)
    kwargs = dict(
        n_query=2,
        n_key=4,
        atom_heads=1,
        n_token=N_TOKEN,
        max_relative_idx=2,
        max_relative_chain=1,
    )
    params = SimpleNamespace(input_embedder=None)
    s_input, s_init, z_init = trunk_module.initialize_trunk(
        {}, params, dtype=narrow_dtype("bfloat16"), **kwargs
    )
    assert calls == [1]
    assert s_input.dtype == jnp.float32
    assert s_init.dtype == jnp.bfloat16
    assert z_init.dtype == jnp.bfloat16

    wide = trunk_module.initialize_trunk({}, params, dtype=None, **kwargs)
    assert [leaf.dtype for leaf in wide] == [jnp.float32] * 3


def _ada_ln(rng):
    from foldjax.models.openfold3.models.primitives import AdaLNParams

    return AdaLNParams(
        layer_norm_a=LayerNormParams(weight=None, bias=None),
        layer_norm_s=_layer_norm(C_S, rng),
        linear_g=_linear(C_S, C_S, rng, bias=False),
        linear_s=_linear(C_S, C_S, rng),
    )


def _diffusion_transformer(rng):
    from foldjax.models.openfold3.models.attention_pair_bias import (
        AdaAttentionPairBiasParams,
    )
    from foldjax.models.openfold3.models.diffusion_transformer import (
        DiffusionTransformerBlockParams,
        DiffusionTransformerParams,
    )
    from foldjax.models.openfold3.models.primitives import (
        ConditionedTransitionBlockParams,
    )

    return DiffusionTransformerParams(
        blocks=(
            DiffusionTransformerBlockParams(
                attention_pair_bias=AdaAttentionPairBiasParams(
                    layer_norm_a=_ada_ln(rng),
                    linear_ada_out=_linear(C_S, C_S, rng),
                    linear_z=_linear(NO_HEADS, C_Z, rng, bias=False),
                    mha=_mha(C_S, rng, gated=False),
                ),
                conditioned_transition=ConditionedTransitionBlockParams(
                    layer_norm=_ada_ln(rng),
                    swiglu=SwiGLUParams(
                        linear_a=_linear(2 * C_S, C_S, rng, bias=False),
                        linear_b=_linear(2 * C_S, C_S, rng, bias=False),
                    ),
                    linear_g=_linear(C_S, C_S, rng),
                    linear_out=_linear(C_S, 2 * C_S, rng, bias=False),
                ),
            ),
        ),
        layer_norm_z=_layer_norm(C_Z, rng),
    )


def test_the_token_diffusion_transformer_stays_float32_on_a_narrow_pair_input() -> None:
    """AlphaFold 3's one load-bearing upcast, reproduced without a cast.

    ``diffusion_head.py:265-268`` casts four tensors to float32 before the
    token transformer, and only ``trunk_pair_cond`` actually changes dtype
    there -- the other three arrive float32 already. Here the transformer's
    parameters are float32, so a bfloat16 ``z`` promotes at the first
    projection and the same shape falls out with no cast to maintain. Storing
    ``z`` narrow is what AlphaFold 3 does too: it rounds ``trunk_pair_cond`` to
    bfloat16 in the conditioning and only widens it again at this boundary, so
    the rounding has already happened either way.
    """
    from foldjax.models.openfold3.models.diffusion_transformer import (
        diffusion_transformer,
    )

    rng = _rng()
    params = _diffusion_transformer(rng)
    a = jnp.asarray(rng.normal(size=(N_TOKEN, C_S)), jnp.float32)
    single = jnp.asarray(rng.normal(size=(N_TOKEN, C_S)), jnp.float32)
    z = jnp.asarray(rng.normal(size=(N_TOKEN, N_TOKEN, C_Z)), jnp.float32)
    mask = jnp.ones((N_TOKEN,), jnp.float32)

    wide = diffusion_transformer(a, single, z, params, no_heads=NO_HEADS, mask=mask)
    narrow_pair = diffusion_transformer(
        a, single, z.astype(jnp.bfloat16), params, no_heads=NO_HEADS, mask=mask
    )
    assert wide.dtype == jnp.float32
    assert narrow_pair.dtype == jnp.float32
    # Close, because only the pair bias was rounded; not equal, because it was.
    np.testing.assert_allclose(wide, narrow_pair, atol=2e-2, rtol=2e-2)


# ------------------------------------------------------------- the matmul


def _dot_operand_dtypes(text: str) -> set[str]:
    """Element types of every `dot` in a lowered StableHLO program."""
    found: set[str] = set()
    for line in text.splitlines():
        if "stablehlo.dot_general" not in line:
            continue
        for name in ("bf16", "f32", "f16", "f64"):
            if f"x{name}>" in line or f"<{name}>" in line:
                found.add(name)
    return found


def test_the_narrowed_weights_reach_a_matmul() -> None:
    """A cast that never reaches an operand is a cast that bought nothing.

    Reading the lowered program rather than an output dtype: an output can be
    bfloat16 because something converted it on the way out, while every dot
    inside ran float32.
    """
    narrow = narrow_dtype("bfloat16")
    s, z, mask = _stack_inputs()

    def run(params, state_dtype):
        return pairformer_stack(
            s.astype(state_dtype) if state_dtype else s,
            z.astype(state_dtype) if state_dtype else z,
            params,
            single_mask=mask.astype(state_dtype) if state_dtype else mask,
            pair_mask=(
                (mask[:, None] * mask[None, :]).astype(state_dtype)
                if state_dtype
                else mask[:, None] * mask[None, :]
            ),
            no_heads_pair=NO_HEADS,
            no_heads_pair_bias=NO_HEADS,
            scan_blocks=True,
        )

    wide_params = _pairformer(_rng())
    wide = jax.jit(run, static_argnums=1).lower(wide_params, None).as_text()
    assert "bf16" not in _dot_operand_dtypes(wide)
    assert "f32" in _dot_operand_dtypes(wide)

    narrow_text = (
        jax.jit(run, static_argnums=1)
        .lower(narrow_floats(wide_params, narrow), jnp.bfloat16)
        .as_text()
    )
    assert "bf16" in _dot_operand_dtypes(narrow_text)

    # The arm that proves the boundary casts are doing the work: narrow
    # parameters against float32 state promote back, so every dot is float32
    # again. This is the failure a weights-only implementation ships, and it
    # is invisible from the parameter tree.
    weights_only = (
        jax.jit(run, static_argnums=1)
        .lower(narrow_floats(wide_params, narrow), None)
        .as_text()
    )
    assert _dot_operand_dtypes(weights_only) == {"f32"}


def test_the_config_carries_the_dtype_into_the_graph_identity() -> None:
    """Two dtypes are two programs; sharing one cache entry would be wrong."""
    base = released_config(n_token=8, n_atom=32)
    narrow = released_config(n_token=8, n_atom=32, dtype="bfloat16")
    assert base != narrow
    identity = inference_module._PredictGraphIdentity
    assert "config" in identity.__dataclass_fields__


# ------------------------------------------------------------ cache identity


def _request(tmp_path, **options):
    from foldjax.schema import PredictionRequest

    job = tmp_path / "job.json"
    job.write_text("{}")
    weights = tmp_path / "weights"
    weights.mkdir(exist_ok=True)
    return PredictionRequest(
        model="openfold3",
        input=job,
        weights=weights,
        output_dir=tmp_path / "out",
        seed=5,
        cache_dir=tmp_path / "cache",
        options=options,
    )


def test_the_dtype_partitions_the_compilation_cache(tmp_path) -> None:
    """Two dtypes are two executables and must not share a namespace.

    The repeated-default case is the other half: an explicit ``float32``
    names the program an unasked run gets, so it must land in the same scope
    rather than forking a second one that also splits the in-process JIT
    pool.
    """
    from foldjax.registry import get_backend

    backend = get_backend("openfold3")
    unasked = backend.cache_profile(_request(tmp_path))
    repeated = backend.cache_profile(_request(tmp_path, dtype="float32"))
    narrow = backend.cache_profile(_request(tmp_path, dtype="bfloat16"))

    assert "dtype" not in unasked
    assert repeated == unasked
    assert narrow["dtype"] == "bfloat16"
    assert narrow != unasked


def test_a_native_dtype_spelling_is_refused_by_name(tmp_path) -> None:
    """The native option bypasses the neutral translation; it is still checked."""
    from foldjax.registry import get_backend

    with pytest.raises(ValueError, match=r"dtype must be one of float32, bfloat16"):
        get_backend("openfold3").validate_native_options({"dtype": "bf16"})


# ------------------------------------------- the composed program, end to end


def _released_checkpoint():
    from foldjax.paths import weights_dir

    store = weights_dir("openfold3")
    if not store.is_dir():
        return None
    return next(iter(sorted(store.glob("*.pt"))), None)


def test_the_whole_program_traces_under_bfloat16_on_real_weights() -> None:
    """Every unit test above watches one boundary; this watches the composition.

    The end-to-end gates in this directory are all ``torch_parity``, so they
    skip wherever an upstream checkout and a torch install are absent -- which
    is where a boundary mistake in the parts nothing else traces (the sampler
    rollout, the atom encoder reading a narrow ``zij``, the confidence map)
    would first show up. ``jax.eval_shape`` reaches all of it for the cost of a
    trace: no compilation, no arithmetic, real parameter shapes and dtypes.

    The two claims are that it traces at all -- a float32 leak into either
    ``lax.scan`` carry is a trace-time error, not a silent slowdown -- and that
    every returned array is float32 in both arms, which is what keeps a
    narrowed run's outputs comparable with a released one's.
    """
    path = _released_checkpoint()
    if path is None:
        pytest.skip("no OpenFold3 checkpoint in the weight store")

    from foldjax.models.openfold3.bridge import checkpoint, chemistry, torch_mapping
    from foldjax.models.openfold3.inference import predict
    from tests.models.openfold3.feature_fixture import minimal_features

    def as_shapes(tree):
        return jax.tree.map(
            lambda v: jax.ShapeDtypeStruct(np.shape(v), np.asarray(v).dtype), tree
        )

    state = checkpoint.load_checkpoint(path)
    prefix = torch_mapping.resolve_model_prefix(state, None)
    torch_mapping.prune_sample_diffusion_aliases(state, prefix=prefix)
    params = torch_mapping.map_inference_params(state, prefix)
    # Reduce both trees to shapes and drop the 2.2 GB of real weights before
    # tracing: `eval_shape` never reads a value, and this machine runs several
    # jobs at once.
    shaped = {
        name: as_shapes(cast_narrow_params(params, *_both(name))) for name in DTYPES
    }
    del state, params

    features = minimal_features(tokens=6, atoms=12, msa_rows=2, templates=1)
    table = chemistry.representative_atom_table()
    key = jax.random.key(0)

    returned = {}
    for name in DTYPES:
        config = released_config(
            n_token=features["token_mask"].shape[-1],
            n_atom=features["atom_mask"].shape[-1],
            num_recycles=2,
            num_steps=3,
            num_samples=2,
            msa_depth=None,
            dtype=name,
        )
        out = jax.eval_shape(
            lambda k, b, p, config=config: predict(k, b, p, config, table),
            key,
            as_shapes(features),
            shaped[name],
        )
        returned[name] = {
            field: (value.shape, value.dtype)
            for field, value in zip(out._fields, out, strict=True)
            if value is not None
        }

    assert returned["bfloat16"], "the narrowed program returned nothing"
    assert returned["bfloat16"] == returned["float32"]
    assert all(
        dtype == jnp.float32 for _shape, dtype in returned["bfloat16"].values()
    ), returned["bfloat16"]


# ------------------------------------------- the confidence head, separately


def test_the_confidence_dtype_follows_dtype_unless_it_is_set() -> None:
    """One default, stated once, so two files cannot disagree about it.

    The consequence worth naming: because the head follows ``dtype``, opting
    into a bfloat16 trunk already narrows it, and ``confidence_dtype`` is not
    a second thing to turn on. It exists to hold this one region *wide*
    against a narrowed trunk, or narrow it against a wide one.

    Every cell of the 2x2 is asserted, because "follows ``dtype``" and
    "ignores ``dtype``" agree on the diagonal and differ only off it.
    """
    from foldjax.models.openfold3.inference import resolve_dtypes

    def config(**options):
        return released_config(n_token=8, n_atom=32, **options)

    # Unset, the head is wide because the trunk is: not a separate default.
    assert resolve_dtypes(config()) == (None, None)
    assert config(confidence_dtype="float32") == config()
    # Opting into the narrow trunk narrows the head with it, which is why
    # there is no second value to set.
    assert resolve_dtypes(config(dtype="bfloat16")) == (jnp.bfloat16, jnp.bfloat16)
    assert config(dtype="bfloat16", confidence_dtype="bfloat16") == config(
        dtype="bfloat16"
    )

    # Off the diagonal, the knob wins in both directions.
    assert resolve_dtypes(config(confidence_dtype="bfloat16")) == (None, jnp.bfloat16)
    assert resolve_dtypes(config(dtype="bfloat16", confidence_dtype="float32")) == (
        jnp.bfloat16,
        None,
    )

    with pytest.raises(ValueError, match=r"must be one of float32, bfloat16"):
        config(confidence_dtype="bf16")


def test_the_confidence_option_is_realized_as_bfloat16_activations() -> None:
    """The option has to reach the arrays, not only the config.

    Everything here is driven through :func:`released_config`, so the
    option's own resolution is in the path rather than two literals handed
    to the cast. Three separate claims, because the first two are each true
    of a program that runs entirely float32 inside:

    * the head's parameters are bfloat16;
    * its two boundaries are float32, so the returned representations are
      float32 whatever ran inside;
    * the lowered program contains bfloat16 matmul operands, which only the
      narrowed stack can produce, and contains none when the knob is set
      back to float32.
    """
    from foldjax.models.openfold3.inference import resolve_dtypes

    config = released_config(n_token=8, n_atom=32, dtype="bfloat16")
    confidence = resolve_dtypes(config)[1]
    shipped = cast_narrow_params(
        _synthetic_inference_params()._replace(
            pairformer_embedding=_confidence(_rng())
        ),
        *resolve_dtypes(config),
    ).pairformer_embedding

    assert _leaf_dtypes(shipped.pairformer_stack) == {jnp.dtype(jnp.bfloat16)}
    for entry in ("linear_i", "linear_j", "linear_distance"):
        assert _leaf_dtypes(getattr(shipped, entry)) == {jnp.dtype(jnp.float32)}, entry

    s_conf, z_conf = _confidence_call(shipped, confidence)
    assert s_conf.dtype == jnp.float32
    assert z_conf.dtype == jnp.float32

    lowered = jax.jit(_confidence_call, static_argnums=1).lower(shipped, confidence)
    assert "bf16" in _dot_operand_dtypes(lowered.as_text())

    wide_config = released_config(n_token=8, n_atom=32)
    wide = cast_narrow_params(
        _synthetic_inference_params()._replace(
            pairformer_embedding=_confidence(_rng())
        ),
        *resolve_dtypes(wide_config),
    ).pairformer_embedding
    wide_lowered = jax.jit(_confidence_call, static_argnums=1).lower(
        wide, resolve_dtypes(wide_config)[1]
    )
    assert _dot_operand_dtypes(wide_lowered.as_text()) == {"f32"}


def test_the_entry_projections_run_before_the_narrowing_cast() -> None:
    """``_project_distance_bins`` is asked for ``zij``'s *pre-cast* dtype.

    Its ``jnp.result_type(dtype, weight.dtype)`` (models/heads.py:208) is the
    kind of site that has silently promoted a realised bfloat16 policy back
    to float32 on another port. It cannot here, and this pins why: the helper
    runs at models/heads.py:296-315, before the cast at :319, so the float32
    it returns is the intended entry island rather than a lost narrowing --
    which is why the separate lowered-program assertion above is the one that
    proves the stack narrowed.
    """
    from foldjax.models.openfold3.inference import resolve_dtypes
    from foldjax.models.openfold3.models import heads as heads_module

    seen: dict[str, object] = {}
    original = heads_module._project_distance_bins

    def spy(squared_distance, params, *, dtype, **kwargs):
        output = original(squared_distance, params, dtype=dtype, **kwargs)
        seen["asked"] = dtype
        seen["returned"] = output.dtype
        return output

    config = released_config(n_token=8, n_atom=32, dtype="bfloat16")
    shipped = cast_narrow_params(
        _synthetic_inference_params()._replace(
            pairformer_embedding=_confidence(_rng())
        ),
        *resolve_dtypes(config),
    ).pairformer_embedding

    heads_module._project_distance_bins = spy
    try:
        _confidence_call(shipped, resolve_dtypes(config)[1])
    finally:
        heads_module._project_distance_bins = original

    assert seen, "the distance-bin projection never ran"
    assert jnp.dtype(seen["asked"]) == jnp.float32
    assert seen["returned"] == jnp.float32


def test_each_group_narrows_without_the_other() -> None:
    """The point of two knobs: either region can be bisected out alone.

    Object identity on the untouched side, because "still float32" is also
    true of a tree that was cast to float32 and back.
    """
    params = _synthetic_inference_params()

    trunk_only = cast_narrow_params(params, jnp.bfloat16, None)
    assert _leaf_dtypes(trunk_only.trunk.pairformer_stack) == {jnp.dtype(jnp.bfloat16)}
    assert trunk_only.pairformer_embedding is params.pairformer_embedding, (
        "the confidence head must be untouched when only the trunk narrows"
    )

    confidence_only = cast_narrow_params(params, None, jnp.bfloat16)
    assert _leaf_dtypes(confidence_only.pairformer_embedding.pairformer_stack) == {
        jnp.dtype(jnp.bfloat16)
    }
    assert confidence_only.trunk is params.trunk
    assert confidence_only.diffusion_conditioning is params.diffusion_conditioning
    assert _leaf_dtypes(confidence_only.pairformer_embedding.linear_i) == {
        jnp.dtype(jnp.float32)
    }


def test_the_confidence_knob_partitions_the_cache_only_when_it_differs(
    tmp_path,
) -> None:
    """Spelling the resolved value names the namespace an unasked run names.

    Both halves of the invariant, on every trunk dtype: the value the request
    would have resolved to is the same program as saying nothing, and the
    other one is a different program that keeps its own scope.
    """
    from foldjax.registry import get_backend

    backend = get_backend("openfold3")
    # The third arm is the one a literal comparison would get wrong: under
    # `dtype=bfloat16` the alias to strip is `bfloat16`, not the shipped
    # `float32`.
    arms = (
        ({}, "float32", "bfloat16"),
        ({"dtype": "float32"}, "float32", "bfloat16"),
        ({"dtype": "bfloat16"}, "bfloat16", "float32"),
    )
    for trunk, alias, other in arms:
        base = backend.cache_profile(_request(tmp_path, **trunk))
        repeated = backend.cache_profile(
            _request(tmp_path, **trunk, confidence_dtype=alias)
        )
        split = backend.cache_profile(
            _request(tmp_path, **trunk, confidence_dtype=other)
        )

        assert "confidence_dtype" not in base, trunk
        assert repeated == base, trunk
        assert split["confidence_dtype"] == other, trunk
        assert split != base, trunk

    with pytest.raises(
        ValueError, match=r"confidence_dtype must be one of float32, bfloat16"
    ):
        backend.validate_native_options({"confidence_dtype": "bf16"})


def test_the_resolved_confidence_dtype_does_not_fork_the_jit_owner() -> None:
    """The config is a field of ``_PredictGraphIdentity``, so it counts too.

    ``cache_profile`` unifying the persistent directory is only half of it. A
    ``None`` sentinel left on the config -- which is what this field used to
    carry -- makes two unequal configs, and therefore two in-process JIT
    owners, out of one program. ``released_config`` resolves it instead, so
    the omitted and the spelled-out request build the same object.
    """
    omitted = released_config(n_token=8, n_atom=32)
    assert omitted.confidence_dtype == DEFAULT_DTYPE
    assert omitted == released_config(
        n_token=8, n_atom=32, confidence_dtype=DEFAULT_DTYPE
    )
    assert omitted != released_config(
        n_token=8, n_atom=32, confidence_dtype="bfloat16"
    )

    # The same must hold on the other trunk dtype, where the resolved value
    # is `bfloat16` and the literal default is the wrong thing to compare to.
    narrow = released_config(n_token=8, n_atom=32, dtype="bfloat16")
    assert narrow.confidence_dtype == "bfloat16"
    assert narrow == released_config(
        n_token=8, n_atom=32, dtype="bfloat16", confidence_dtype="bfloat16"
    )


# --------------------------------------- the streamed route takes both knobs


def test_the_streamed_route_forwards_both_keywords(monkeypatch) -> None:
    """The padded path builds its own trunk calls, so it can be missed alone.

    ``streaming.py`` does not go through :func:`predict`'s trunk call; it
    assembles ``initialize_trunk`` and ``trunk_cycle`` itself. A keyword
    threaded everywhere except here produces a padded run that silently takes
    the released path under the cache name of the narrowed one, which is the
    one failure a cache namespace exists to prevent.

    Both keywords are checked, not only ``dtype``. They were threaded by two
    changes that met in a rebase, and the resolution is exactly where one of
    them could have been dropped -- so a guard that watched only its author's
    own keyword would not have been a guard at all.
    """
    from foldjax.models.openfold3 import streaming

    seen: dict[str, dict] = {}

    def initialize(batch, params, **kwargs):
        seen["initialize"] = kwargs
        one = jnp.zeros((1, 4), jnp.float32)
        return one, one, one[..., None]

    def cycle(batch, msa, params, initial, carry, **kwargs):
        seen["cycle"] = kwargs
        return carry

    monkeypatch.setattr(streaming, "initialize_trunk", initialize)
    monkeypatch.setattr(streaming, "trunk_cycle", cycle)

    from tests.models.openfold3.test_stable_compile import _config

    config = _config(
        n_token=4,
        dtype="bfloat16",
        confidence_dtype="float32",
        glu_backend="tokamax",
    )
    graph = streaming._StreamedGraph(
        inference_module._PredictGraphIdentity(
            config=config,
            n_chain=None,
            augment=False,
            use_trunk_pair_embedding=True,
            rng_route="native",
            triangle_kernel="xla",
            cp_topology=(),
            cache_scope=None,
        ),
        compiled=False,
    )
    common = {"token_mask": jnp.ones((1, 4), jnp.float32)}
    initial, carry = graph.initialize(common, None)
    graph.cycle(common, {}, None, initial, carry)

    assert set(seen) == {"initialize", "cycle"}
    for stage, kwargs in seen.items():
        assert kwargs["glu_backend"] == "tokamax", stage
        assert kwargs["dtype"] == jnp.bfloat16, stage
