from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from foldjax.models.protenix.models.primitives.primitives import (
    AdaptiveLayerNormParams,
    LayerNormParams,
    LinearParams,
    TransitionParams,
    adaptive_layer_norm,
    layer_norm,
    transition,
)
from foldjax.models.protenix.models.trunk_blocks.embedders import (
    FourierParams,
    fourier_embedding,
)


def test_layer_norm_matches_reference_formula() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=(2, 3, 4)).astype(np.float32)
    weight = rng.normal(size=(4,)).astype(np.float32)
    bias = rng.normal(size=(4,)).astype(np.float32)
    params = LayerNormParams(weight=jnp.asarray(weight), bias=jnp.asarray(bias))

    actual = np.asarray(layer_norm(jnp.asarray(x), params, eps=1e-5))
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    expected = (x - mean) * np.reciprocal(np.sqrt(var + 1e-5))
    expected = expected * weight + bias

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_layer_norm_supports_missing_scale_and_offset() -> None:
    x = np.array([[1.0, 2.0, 4.0]], dtype=np.float32)
    params = LayerNormParams(weight=None, bias=None)

    actual = np.asarray(layer_norm(jnp.asarray(x), params, eps=1e-5))

    assert actual.shape == x.shape
    np.testing.assert_allclose(actual.mean(axis=-1), np.array([0.0]), atol=1e-6)


def test_transition_matches_protenix_training_formula() -> None:
    rng = np.random.default_rng(2)
    x = rng.normal(size=(2, 3, 4)).astype(np.float32)
    params = TransitionParams(
        layer_norm=LayerNormParams(
            weight=rng.normal(size=(4,)).astype(np.float32),
            bias=rng.normal(size=(4,)).astype(np.float32),
        ),
        linear_a=LinearParams(
            weight=rng.normal(size=(8, 4)).astype(np.float32),
            bias=None,
        ),
        linear_b=LinearParams(
            weight=rng.normal(size=(8, 4)).astype(np.float32),
            bias=None,
        ),
        linear_out=LinearParams(
            weight=rng.normal(size=(4, 8)).astype(np.float32),
            bias=None,
        ),
    )

    actual = np.asarray(transition(jnp.asarray(x), params))
    y = np.asarray(layer_norm(jnp.asarray(x), params.layer_norm))
    a = y @ np.asarray(params.linear_a.weight).T
    b = y @ np.asarray(params.linear_b.weight).T
    expected = (a / (1.0 + np.exp(-a)) * b) @ np.asarray(params.linear_out.weight).T

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_adaptive_layer_norm_matches_protenix_formula() -> None:
    rng = np.random.default_rng(3)
    a = rng.normal(size=(2, 3, 5)).astype(np.float32)
    s = rng.normal(size=(2, 3, 4)).astype(np.float32)
    params = AdaptiveLayerNormParams(
        layernorm_a=LayerNormParams(weight=None, bias=None),
        layernorm_s=LayerNormParams(
            weight=rng.normal(size=(4,)).astype(np.float32),
            bias=None,
        ),
        linear_s=LinearParams(
            weight=rng.normal(size=(5, 4)).astype(np.float32),
            bias=rng.normal(size=(5,)).astype(np.float32),
        ),
        linear_no_bias_s=LinearParams(
            weight=rng.normal(size=(5, 4)).astype(np.float32),
            bias=None,
        ),
    )

    actual = np.asarray(adaptive_layer_norm(jnp.asarray(a), jnp.asarray(s), params))
    a_norm = np.asarray(layer_norm(jnp.asarray(a), params.layernorm_a))
    s_norm = np.asarray(layer_norm(jnp.asarray(s), params.layernorm_s))
    gate = s_norm @ np.asarray(params.linear_s.weight).T
    gate = gate + np.asarray(params.linear_s.bias)
    shift = s_norm @ np.asarray(params.linear_no_bias_s.weight).T
    expected = (1.0 / (1.0 + np.exp(-gate))) * a_norm + shift

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_fourier_embedding_matches_protenix_formula() -> None:
    t = np.array([0.0, 0.5, 1.0], dtype=np.float32)
    params = FourierParams(
        w=jnp.asarray(np.array([0.25, -0.5], dtype=np.float32)),
        b=jnp.asarray(np.array([0.1, 0.2], dtype=np.float32)),
    )

    actual = np.asarray(fourier_embedding(jnp.asarray(t), params))
    expected = np.cos(
        2 * np.pi * (t[..., None] * np.asarray(params.w) + np.asarray(params.b))
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
