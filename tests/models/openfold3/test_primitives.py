"""Torch-free unit tests for the ported primitives.

These run everywhere; the numerical reference lives in
``test_torch_parity_primitives.py``.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.openfold3.bridge.torch_mapping import map_layer_norm, map_linear
from foldjax.models.openfold3.models.primitives import (
    LayerNormParams,
    LinearParams,
    SwiGLUParams,
    SwiGLUTransitionParams,
    layer_norm,
    linear,
    silu,
    swiglu,
    swiglu_transition,
)


def test_linear_uses_pytorch_weight_layout() -> None:
    # weight is [out, in]; a [.., in] input must produce [.., out].
    params = LinearParams(weight=jnp.arange(6, dtype=jnp.float32).reshape(2, 3))
    out = linear(jnp.ones((4, 3), dtype=jnp.float32), params)
    assert out.shape == (4, 2)
    np.testing.assert_allclose(np.asarray(out)[0], [3.0, 12.0])


def test_linear_adds_the_bias_when_present() -> None:
    params = LinearParams(
        weight=jnp.zeros((2, 3), dtype=jnp.float32),
        bias=jnp.asarray([1.0, -1.0], dtype=jnp.float32),
    )
    out = linear(jnp.ones((3,), dtype=jnp.float32), params)
    np.testing.assert_allclose(np.asarray(out), [1.0, -1.0])


def test_layer_norm_standardizes_the_final_axis() -> None:
    x = jnp.asarray([[1.0, 2.0, 3.0, 4.0]], dtype=jnp.float32)
    out = layer_norm(x, LayerNormParams())
    assert float(jnp.mean(out)) == pytest.approx(0.0, abs=1e-5)
    assert float(jnp.std(out)) == pytest.approx(1.0, abs=1e-3)


def test_layer_norm_scale_and_offset_are_independently_optional() -> None:
    x = jnp.asarray([[1.0, 2.0, 3.0, 4.0]], dtype=jnp.float32)
    scaled = layer_norm(x, LayerNormParams(weight=jnp.full((4,), 2.0)))
    plain = layer_norm(x, LayerNormParams())
    np.testing.assert_allclose(np.asarray(scaled), np.asarray(plain) * 2.0, rtol=1e-6)

    shifted = layer_norm(x, LayerNormParams(bias=jnp.full((4,), 5.0)))
    np.testing.assert_allclose(np.asarray(shifted), np.asarray(plain) + 5.0, rtol=1e-6)


def test_silu_is_stable_for_large_negative_inputs() -> None:
    out = silu(jnp.asarray([-100.0, 0.0, 100.0], dtype=jnp.float32))
    assert np.all(np.isfinite(np.asarray(out)))
    assert float(out[0]) == pytest.approx(0.0, abs=1e-6)
    assert float(out[1]) == pytest.approx(0.0, abs=1e-6)
    assert float(out[2]) == pytest.approx(100.0, rel=1e-4)


def _swiglu_params(c_in: int, c_hidden: int) -> SwiGLUParams:
    return SwiGLUParams(
        linear_a=LinearParams(weight=jnp.full((c_hidden, c_in), 0.1)),
        linear_b=LinearParams(weight=jnp.full((c_hidden, c_in), 0.2)),
    )


def test_swiglu_gates_the_first_projection_by_the_second() -> None:
    x = jnp.ones((2, 4), dtype=jnp.float32)
    out = swiglu(x, _swiglu_params(4, 3))
    a, b = 0.4, 0.8
    expected = (a / (1.0 + np.exp(-a))) * b
    np.testing.assert_allclose(np.asarray(out), np.full((2, 3), expected), rtol=1e-5)


def _random_transition_params(c_in: int, c_hidden: int, seed: int) -> tuple:
    """Random weights, because constant weights make the output degenerately 0.

    With every weight equal, the layer-normed input sums symmetrically through
    ``linear_out`` and the update collapses to ~1e-16, which would let a broken
    mask pass a "masked position is zero" assertion trivially.
    """
    rng = np.random.default_rng(seed)

    def weight(shape: tuple[int, int]) -> jnp.ndarray:
        return jnp.asarray(rng.normal(size=shape), dtype=jnp.float32)

    return SwiGLUTransitionParams(
        layer_norm=LayerNormParams(),
        swiglu=SwiGLUParams(
            linear_a=LinearParams(weight=weight((c_hidden, c_in))),
            linear_b=LinearParams(weight=weight((c_hidden, c_in))),
        ),
        linear_out=LinearParams(weight=weight((c_in, c_hidden))),
    )


def test_swiglu_transition_mask_zeroes_masked_positions() -> None:
    params = _random_transition_params(4, 8, seed=0)
    x = jnp.asarray(np.random.default_rng(0).normal(size=(1, 3, 4)), dtype=jnp.float32)
    mask = jnp.asarray([[1.0, 0.0, 1.0]], dtype=jnp.float32)
    out = swiglu_transition(x, params, mask=mask)
    assert np.allclose(np.asarray(out)[0, 1], 0.0)
    # The unmasked rows must survive, or the assertion above proves nothing.
    assert not np.allclose(np.asarray(out)[0, 0], 0.0)
    assert not np.allclose(np.asarray(out)[0, 2], 0.0)


def test_swiglu_transition_without_a_mask_matches_an_all_ones_mask() -> None:
    params = SwiGLUTransitionParams(
        layer_norm=LayerNormParams(),
        swiglu=_swiglu_params(4, 8),
        linear_out=LinearParams(weight=jnp.full((4, 8), 0.5)),
    )
    x = jnp.asarray(np.random.default_rng(1).normal(size=(2, 3, 4)), dtype=jnp.float32)
    np.testing.assert_allclose(
        np.asarray(swiglu_transition(x, params)),
        np.asarray(swiglu_transition(x, params, mask=jnp.ones((2, 3)))),
        rtol=1e-6,
    )


def test_mapper_reports_the_missing_key_by_name() -> None:
    with pytest.raises(KeyError, match="block.linear.weight"):
        map_linear({}, "block.linear")


def test_mapper_rejects_an_unexpected_bias() -> None:
    state = {"l.weight": np.zeros((2, 2)), "l.bias": np.zeros((2,))}
    with pytest.raises(KeyError, match="unexpected bias"):
        map_linear(state, "l", bias=False)


def test_mapper_requires_a_bias_when_one_is_declared() -> None:
    with pytest.raises(KeyError, match="l.bias"):
        map_linear({"l.weight": np.zeros((2, 2))}, "l", bias=True)


def test_layer_norm_mapper_tolerates_absent_scale_and_offset() -> None:
    params = map_layer_norm({}, "norm")
    assert params.weight is None
    assert params.bias is None
