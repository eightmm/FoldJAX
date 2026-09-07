"""Protect the native Pairformer autocast-disabled parameter island."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models._stacking import StackedLayers
from foldjax.models.boltz2.models.trunk_blocks import trunk


def test_trunk_amp_cast_preserves_geometry_norms_embeddings_and_biases():
    weight = jnp.asarray([1.001, -0.1001], jnp.float32)
    params = {
        "input_embedder": {
            "atom_encoder": {"embed_atom_features": {"kernel": weight}},
            "atom_attention_encoder": {
                "atom_to_token_trans": {"kernel": weight},
                "atom_transformer": {"proj_q": {"kernel": weight, "bias": weight}},
            },
            "method_conditioning_init": weight,
        },
        "z_norm": {"scale": weight, "bias": weight},
        "token_bonds_type": weight,
        "template_module": {"a_proj": {"kernel": weight}},
        "contact_conditioning": {
            "encoding_unspecified": weight,
            "fourier_embedding": {"proj": {"kernel": weight, "bias": weight}},
        },
    }
    actual = trunk._cast_trunk_params(params, jnp.bfloat16)
    for path, expected in jax.tree_util.tree_flatten_with_path(params)[0]:
        keys = [entry.key for entry in path]
        value = actual
        for key in keys:
            value = value[key]
        amp = keys[-1] == "kernel" and (
            "atom_transformer" in keys or "fourier_embedding" in keys
        )
        assert value.dtype == (jnp.bfloat16 if amp else jnp.float32), keys
        np.testing.assert_array_equal(value, expected.astype(value.dtype))


@pytest.mark.parametrize("jit", [False, True])
def test_native_bfloat16_sigmoid_rounds_only_completed_operator(jit):
    from foldjax.models.boltz2.models.primitives._common import sigmoid

    x = jnp.asarray(np.linspace(-8, 8, 10001), jnp.bfloat16)
    xf = np.asarray(x, np.float64)
    expected = jnp.asarray(1 / (1 + np.exp(-xf)), jnp.bfloat16)
    function = jax.jit(sigmoid) if jit else sigmoid
    np.testing.assert_array_equal(function(x), expected)


@pytest.mark.parametrize("jit", [False, True])
def test_native_amp_layer_norm_affine_precedes_output_cast(jit):
    from foldjax.models.boltz2.models.primitives._common import layer_norm

    x = jnp.asarray([[0.13, 0.26, 0.51, 0.9]], jnp.bfloat16)
    scale = jnp.asarray([1.001, 0.73, 1.29, 0.8], jnp.float32)
    bias = jnp.asarray([0.003, -0.007, 0.01, 0.02], jnp.float32)
    xf = x.astype(jnp.float32)
    centered = xf - xf.mean(-1, keepdims=True)
    expected = centered * jax.lax.rsqrt((centered**2).mean(-1, keepdims=True) + 1e-5)
    expected = expected * scale + bias
    fn = jax.jit(layer_norm) if jit else layer_norm
    actual = fn(x, scale, bias, 1e-5)
    assert actual.dtype == jnp.float32
    np.testing.assert_allclose(actual, expected, atol=2e-7, rtol=2e-7)


@pytest.mark.parametrize("jit", [False, True])
def test_native_amp_linear_has_one_output_rounding_with_bias(jit):
    from foldjax.models.boltz2.models.primitives._common import linear

    # The dot and bias each add half a BF16 ULP; neither may round early.
    x = jnp.asarray([[1, 1]], jnp.float32)
    kernel = jnp.asarray([[1], [2**-8]], jnp.bfloat16)
    bias = jnp.asarray([2**-8], jnp.float32)
    fn = jax.jit(linear) if jit else linear
    actual = fn(x, kernel, bias)
    assert actual.dtype == jnp.bfloat16
    np.testing.assert_array_equal(actual.astype(jnp.float32), [[1 + 2**-7]])


@pytest.mark.parametrize("jit", [False, True])
def test_native_amp_linear_narrows_activations_at_the_kernel_boundary(jit):
    from foldjax.models.boltz2.models.primitives._common import linear

    x = jnp.asarray([[1.003, 0.1001]], jnp.float32)
    kernel = jnp.asarray([[1], [1]], jnp.bfloat16)
    fn = jax.jit(linear) if jit else linear
    actual = fn(x, kernel)
    assert actual.dtype == jnp.bfloat16
    np.testing.assert_array_equal(actual, x.astype(jnp.bfloat16) @ kernel)


@pytest.mark.parametrize("chunk", [None, 2])
def test_atom_attention_keeps_scores_and_value_contraction_in_fp32(monkeypatch, chunk):
    from foldjax.models.boltz2.models.diffusion import diffusion_transformer as module

    observed = []
    original = module._no_proj_qblock

    def observe(q, k, v, bias, *args):
        observed.append((q.dtype, k.dtype, v.dtype, bias.dtype))
        return original(q, k, v, bias, *args)

    monkeypatch.setattr(module, "_no_proj_qblock", observe)
    eye = jnp.eye(4, dtype=jnp.bfloat16)
    params = {
        name: {"kernel": eye} for name in ("proj_q", "proj_k", "proj_v", "proj_o")
    }
    params["proj_q"]["bias"] = jnp.zeros(4, jnp.float32)
    params["proj_g"] = {"kernel": jnp.zeros_like(eye)}
    result = module._attention_pair_bias_no_proj_z_forward(
        params,
        s=jnp.ones((1, 3, 4), jnp.float32),
        k_in=jnp.ones((1, 3, 4), jnp.float32),
        bias=jnp.zeros((1, 3, 3, 2), jnp.bfloat16),
        mask=jnp.ones((1, 3), jnp.float32),
        multiplicity=1,
        inf=1e6,
        chunk_size=chunk,
    )
    assert result.dtype == jnp.bfloat16
    assert observed and all(dtypes == (jnp.float32,) * 4 for dtypes in observed)


@pytest.mark.parametrize("jit", [False, True])
def test_amp_glu_activation_rounds_only_its_completed_value(jit):
    from foldjax.models.boltz2.models.primitives.glu_backend import gated_linear_unit

    x = jnp.linspace(-10, 10, 1001).reshape(-1, 1)
    weight = jnp.ones((1, 1), jnp.bfloat16)

    def run(x, weight):
        return gated_linear_unit(x, weight, weight, jax.nn.sigmoid)

    narrowed = x.astype(jnp.bfloat16)
    expected = jax.nn.sigmoid(narrowed.astype(jnp.float32)).astype(jnp.bfloat16)
    expected = expected * narrowed
    actual = (jax.jit(run) if jit else run)(x, weight)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("prestack", [False, True])
@pytest.mark.parametrize("stack_name", ["pairformer_module", "pairformer_stack"])
def test_trunk_cast_preserves_original_single_track_weights(
    dtype, prestack, stack_name
):
    # Deliberately not representable in bf16: widening rounded values must fail.
    weight = jnp.asarray([1.001, -0.1001, 0.3333], dtype=jnp.float32)
    layer = {
        "pre_norm_s": {"scale": weight, "bias": weight},
        "attention": {"proj_q": {"kernel": weight}},
        "transition_s": {"fc1": {"kernel": weight}},
        "transition_z": {"fc1": {"kernel": weight}},
    }
    layers = StackedLayers.from_layers([layer, layer]) if prestack else [layer, layer]
    params = {
        stack_name: {"layers": layers},
        "input_embedder": {"kernel": weight},
        "index": jnp.asarray([1, 2], dtype=jnp.int32),
    }
    cast = trunk._cast_trunk_params(params, dtype)
    for converted in cast[stack_name]["layers"]:
        for name in ("pre_norm_s", "attention", "transition_s"):
            for actual, expected in zip(
                jax.tree.leaves(converted[name]),
                jax.tree.leaves(layer[name]),
                strict=True,
            ):
                assert actual.dtype == jnp.float32
                np.testing.assert_array_equal(actual, expected)
        assert converted["transition_z"]["fc1"]["kernel"].dtype == dtype
    assert cast["input_embedder"]["kernel"].dtype == dtype
    assert cast["index"].dtype == jnp.int32
    np.testing.assert_array_equal(params["input_embedder"]["kernel"], weight)
    assert params["input_embedder"]["kernel"].dtype == jnp.float32


@pytest.mark.parametrize("jit", [False, True])
def test_mixed_projection_lazy_matches_eager_after_fp32_entry(jit):
    from foldjax.models.boltz2.models.diffusion.atom import _projection_layer_forward
    from foldjax.models.boltz2.models.diffusion.diffusion_conditioning import (
        _projection_input_norm,
        _projection_list_forward,
    )

    rng = np.random.default_rng(9)
    x = jnp.asarray(rng.normal(size=(1, 3, 3, 8)), dtype=jnp.bfloat16)
    params = [
        {
            "norm": {
                "scale": jnp.asarray(rng.normal(size=8), dtype=jnp.float32),
                "bias": jnp.asarray(rng.normal(size=8), dtype=jnp.float32),
            },
            "linear": {"kernel": jnp.asarray(rng.normal(size=(8, 2)), jnp.float32)},
        }
        for _ in range(2)
    ]

    def evaluate(params, x):
        eager = _projection_list_forward(params, x, 1e-5, compute_dtype=jnp.bfloat16)
        normalized = _projection_input_norm(
            x.astype(jnp.float32), 1e-5, native_amp=True
        )
        lazy = jnp.concatenate(
            [
                _projection_layer_forward(
                    layer,
                    None,
                    1e-5,
                    normed_input=normalized,
                    compute_dtype=jnp.bfloat16,
                )
                for layer in params
            ],
            axis=-1,
        )
        return eager, lazy

    eager, lazy = (jax.jit(evaluate) if jit else evaluate)(params, x)
    assert eager.dtype == jnp.bfloat16
    assert lazy.dtype == jnp.float32
    np.testing.assert_array_equal(eager.astype(jnp.float32), lazy)


@pytest.mark.parametrize("use_scan", [False, True])
def test_mixed_conditioning_eager_lazy_score_parity(use_scan):
    from foldjax.models.boltz2.models.diffusion.diffusion import (
        diffusion_score_model_forward,
    )
    from foldjax.models.boltz2.models.diffusion.diffusion_conditioning import (
        diffusion_conditioning_forward,
    )
    from tests.models.boltz2.test_atom_cp_model_integration import (
        _model_inputs,
        _native_params,
    )

    native, inputs = _native_params(), _model_inputs()

    def evaluate(lazy):
        conditioning = diffusion_conditioning_forward(
            native["diffusion_conditioning"],
            s_trunk=inputs["s_trunk"],
            z_trunk=inputs["z_trunk"],
            relative_position_encoding=inputs["relative_position_encoding"],
            feats=inputs["feats"],
            token_layers=1,
            lazy_token_trans_bias=lazy,
            compute_dtype=jnp.bfloat16,
        )
        assert conditioning["q"].dtype == jnp.float32
        assert conditioning["c"].dtype == jnp.float32
        assert conditioning["atom_enc_bias"].dtype == jnp.bfloat16
        return diffusion_score_model_forward(
            native["score_model"],
            s_inputs=inputs["s_inputs"],
            s_trunk=inputs["s_trunk"],
            r_noisy=inputs["r_noisy"],
            times=inputs["times"],
            feats=inputs["feats"],
            diffusion_conditioning=conditioning,
            use_scan=use_scan,
            token_layers=1,
        )

    eager, lazy = jax.jit(lambda: (evaluate(False), evaluate(True)))()
    assert eager.dtype == lazy.dtype == jnp.float32
    np.testing.assert_allclose(eager, lazy, atol=1e-6, rtol=1e-6)


def test_relative_position_has_one_bfloat16_output_rounding():
    from foldjax.models.boltz2.models.trunk_blocks.trunk import (
        relative_position_forward,
    )

    # Native concatenates the categories into one Linear. Two half-ULP
    # contributions must accumulate before the single BF16 output rounding.
    kernel = jnp.asarray([[1], [0], [2**-8], [0], [2**-8], [0], [0]], jnp.bfloat16)
    feats = {
        name: jnp.zeros((1, 1), jnp.int32)
        for name in (
            "asym_id",
            "residue_index",
            "entity_id",
            "token_index",
            "sym_id",
            "cyclic_period",
        )
    }
    params = {"linear_layer": {"kernel": kernel}}
    for forward in (
        relative_position_forward,
        jax.jit(relative_position_forward, static_argnames=("r_max", "s_max")),
    ):
        actual = forward(params, feats, r_max=0, s_max=0)
        assert actual.dtype == jnp.bfloat16
        np.testing.assert_array_equal(np.asarray(actual, np.float32), [[[[1 + 2**-7]]]])
