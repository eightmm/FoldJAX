from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.protenix.bridge.torch_mapping import (
    map_triangle_attention_state_dict,
    map_triangle_multiplication_state_dict,
)
from foldjax.models.protenix.models.primitives.primitives import (
    layer_norm,
    linear,
    sigmoid,
)
from foldjax.models.protenix.models.triangle.triangle import (
    triangle_attention,
    triangle_multiplication,
)


def test_triangle_multiplication_outgoing_matches_reference_formula() -> None:
    rng = np.random.default_rng(7)
    z = rng.normal(size=(1, 3, 3, 4)).astype(np.float32)
    mask = rng.integers(0, 2, size=(1, 3, 3)).astype(np.float32)
    state = _triangle_state(rng, c_z=4, c_hidden=5)
    params = map_triangle_multiplication_state_dict(state, "tri")

    actual = np.asarray(
        triangle_multiplication(jnp.asarray(z), jnp.asarray(mask), params, "outgoing")
    )
    compiled = np.asarray(
        triangle_multiplication(
            jnp.asarray(z),
            jnp.asarray(mask),
            params,
            "outgoing",
            use_jit=True,
        )
    )

    z_norm = layer_norm(jnp.asarray(z), params.layer_norm_in)
    a = (
        jnp.asarray(mask)[..., None]
        * sigmoid(linear(z_norm, params.linear_a_g))
        * linear(z_norm, params.linear_a_p)
    )
    b = (
        jnp.asarray(mask)[..., None]
        * sigmoid(linear(z_norm, params.linear_b_g))
        * linear(z_norm, params.linear_b_p)
    )
    expected = jnp.einsum("...ikd,...jkd->...ijd", a, b)
    expected = layer_norm(expected, params.layer_norm_out)
    expected = linear(expected, params.linear_z)
    expected = expected * sigmoid(linear(z_norm, params.linear_g))

    np.testing.assert_allclose(actual, np.asarray(expected), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(compiled, actual, rtol=1e-5, atol=1e-5)


def test_triangle_multiplication_incoming_matches_reference_formula() -> None:
    rng = np.random.default_rng(8)
    z = rng.normal(size=(1, 3, 3, 4)).astype(np.float32)
    mask = rng.integers(0, 2, size=(1, 3, 3)).astype(np.float32)
    state = _triangle_state(rng, c_z=4, c_hidden=5)
    params = map_triangle_multiplication_state_dict(state, "tri")

    actual = np.asarray(
        triangle_multiplication(jnp.asarray(z), jnp.asarray(mask), params, "incoming")
    )

    z_norm = layer_norm(jnp.asarray(z), params.layer_norm_in)
    a = (
        jnp.asarray(mask)[..., None]
        * sigmoid(linear(z_norm, params.linear_a_g))
        * linear(z_norm, params.linear_a_p)
    )
    b = (
        jnp.asarray(mask)[..., None]
        * sigmoid(linear(z_norm, params.linear_b_g))
        * linear(z_norm, params.linear_b_p)
    )
    expected = jnp.einsum("...kid,...kjd->...ijd", a, b)
    expected = layer_norm(expected, params.layer_norm_out)
    expected = linear(expected, params.linear_z)
    expected = expected * sigmoid(linear(z_norm, params.linear_g))

    np.testing.assert_allclose(actual, np.asarray(expected), rtol=1e-5, atol=1e-5)


def test_map_triangle_multiplication_state_dict_uses_protenix_keys() -> None:
    state = {
        "tri.layer_norm_in.weight": np.ones((4,), dtype=np.float32),
        "tri.layer_norm_in.bias": np.zeros((4,), dtype=np.float32),
        "tri.layer_norm_out.weight": np.ones((5,), dtype=np.float32),
        "tri.layer_norm_out.bias": np.zeros((5,), dtype=np.float32),
        "tri.linear_a_p.weight": np.ones((5, 4), dtype=np.float32),
        "tri.linear_a_g.weight": np.ones((5, 4), dtype=np.float32),
        "tri.linear_b_p.weight": np.ones((5, 4), dtype=np.float32),
        "tri.linear_b_g.weight": np.ones((5, 4), dtype=np.float32),
        "tri.linear_z.weight": np.ones((4, 5), dtype=np.float32),
        "tri.linear_g.weight": np.ones((4, 4), dtype=np.float32),
    }

    params = map_triangle_multiplication_state_dict(state, "tri")

    assert tuple(params.layer_norm_in.weight.shape) == (4,)
    assert tuple(params.layer_norm_out.weight.shape) == (5,)
    assert tuple(params.linear_a_p.weight.shape) == (5, 4)
    assert tuple(params.linear_b_g.weight.shape) == (5, 4)
    assert tuple(params.linear_z.weight.shape) == (4, 5)
    assert tuple(params.linear_g.weight.shape) == (4, 4)


def test_triangle_attention_starting_matches_reference_formula() -> None:
    rng = np.random.default_rng(9)
    x = rng.normal(size=(1, 3, 3, 4)).astype(np.float32)
    mask = rng.integers(0, 2, size=(1, 3, 3)).astype(np.float32)
    state = _triangle_attention_state(rng, c_in=4, num_heads=2, head_dim=3)
    params = map_triangle_attention_state_dict(state, "tri_att")

    actual = np.asarray(
        triangle_attention(
            jnp.asarray(x),
            jnp.asarray(mask),
            params,
            num_heads=2,
            starting=True,
        )
    )
    expected = _triangle_attention_reference(
        jnp.asarray(x),
        jnp.asarray(mask),
        params,
        num_heads=2,
    )

    np.testing.assert_allclose(actual, np.asarray(expected), rtol=1e-5, atol=1e-5)


def test_triangle_attention_ending_matches_transposed_starting() -> None:
    rng = np.random.default_rng(10)
    x = rng.normal(size=(1, 3, 3, 4)).astype(np.float32)
    mask = rng.integers(0, 2, size=(1, 3, 3)).astype(np.float32)
    state = _triangle_attention_state(rng, c_in=4, num_heads=2, head_dim=3)
    params = map_triangle_attention_state_dict(state, "tri_att")

    actual = np.asarray(
        triangle_attention(
            jnp.asarray(x),
            jnp.asarray(mask),
            params,
            num_heads=2,
            starting=False,
        )
    )
    expected = _triangle_attention_reference(
        jnp.swapaxes(jnp.asarray(x), -2, -3),
        jnp.swapaxes(jnp.asarray(mask), -1, -2),
        params,
        num_heads=2,
    )
    expected = jnp.swapaxes(expected, -2, -3)

    np.testing.assert_allclose(actual, np.asarray(expected), rtol=1e-5, atol=1e-5)


def test_triangle_attention_query_chunks_match_dense() -> None:
    rng = np.random.default_rng(11)
    x = rng.normal(size=(1, 5, 5, 4)).astype(np.float32)
    mask = rng.integers(0, 2, size=(1, 5, 5)).astype(np.float32)
    state = _triangle_attention_state(rng, c_in=4, num_heads=2, head_dim=3)
    params = map_triangle_attention_state_dict(state, "tri_att")

    dense = triangle_attention(
        jnp.asarray(x),
        jnp.asarray(mask),
        params,
        num_heads=2,
    )
    chunked = triangle_attention(
        jnp.asarray(x),
        jnp.asarray(mask),
        params,
        num_heads=2,
        q_chunk_size=2,
    )
    compiled = triangle_attention(
        jnp.asarray(x),
        jnp.asarray(mask),
        params,
        num_heads=2,
        attention_backend="xla_jit",
    )

    np.testing.assert_allclose(
        np.asarray(chunked),
        np.asarray(dense),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(compiled),
        np.asarray(dense),
        rtol=1e-5,
        atol=1e-5,
    )


def test_triangle_attention_rejects_unknown_backend(monkeypatch) -> None:
    rng = np.random.default_rng(12)
    x = rng.normal(size=(1, 3, 3, 4)).astype(np.float32)
    state = _triangle_attention_state(rng, c_in=4, num_heads=2, head_dim=3)
    params = map_triangle_attention_state_dict(state, "tri_att")
    monkeypatch.setenv("PROTENIX_TRIANGLE_BACKEND", "typo")

    with pytest.raises(ValueError, match="unsupported triangle attention backend"):
        triangle_attention(jnp.asarray(x), None, params, num_heads=2)


def test_map_triangle_attention_state_dict_uses_protenix_keys() -> None:
    state = {
        "tri_att.layer_norm.weight": np.ones((4,), dtype=np.float32),
        "tri_att.layer_norm.bias": np.zeros((4,), dtype=np.float32),
        "tri_att.linear.weight": np.ones((2, 4), dtype=np.float32),
        "tri_att.mha.linear_q.weight": np.ones((6, 4), dtype=np.float32),
        "tri_att.mha.linear_k.weight": np.ones((6, 4), dtype=np.float32),
        "tri_att.mha.linear_v.weight": np.ones((6, 4), dtype=np.float32),
        "tri_att.mha.linear_o.weight": np.ones((4, 6), dtype=np.float32),
        "tri_att.mha.linear_g.weight": np.ones((6, 4), dtype=np.float32),
    }

    params = map_triangle_attention_state_dict(state, "tri_att")

    assert tuple(params.layer_norm.weight.shape) == (4,)
    assert tuple(params.linear.weight.shape) == (2, 4)
    assert tuple(params.attention.linear_q.weight.shape) == (6, 4)
    assert tuple(params.attention.linear_o.weight.shape) == (4, 6)


def _triangle_state(
    rng: np.random.Generator,
    *,
    c_z: int,
    c_hidden: int,
) -> dict[str, np.ndarray]:
    return {
        "tri.layer_norm_in.weight": rng.normal(size=(c_z,)).astype(np.float32),
        "tri.layer_norm_in.bias": rng.normal(size=(c_z,)).astype(np.float32),
        "tri.layer_norm_out.weight": rng.normal(size=(c_hidden,)).astype(np.float32),
        "tri.layer_norm_out.bias": rng.normal(size=(c_hidden,)).astype(np.float32),
        "tri.linear_a_p.weight": rng.normal(size=(c_hidden, c_z)).astype(np.float32),
        "tri.linear_a_g.weight": rng.normal(size=(c_hidden, c_z)).astype(np.float32),
        "tri.linear_b_p.weight": rng.normal(size=(c_hidden, c_z)).astype(np.float32),
        "tri.linear_b_g.weight": rng.normal(size=(c_hidden, c_z)).astype(np.float32),
        "tri.linear_z.weight": rng.normal(size=(c_z, c_hidden)).astype(np.float32),
        "tri.linear_g.weight": rng.normal(size=(c_z, c_z)).astype(np.float32),
    }


def _triangle_attention_state(
    rng: np.random.Generator,
    *,
    c_in: int,
    num_heads: int,
    head_dim: int,
) -> dict[str, np.ndarray]:
    total_hidden = num_heads * head_dim
    return {
        "tri_att.layer_norm.weight": rng.normal(size=(c_in,)).astype(np.float32),
        "tri_att.layer_norm.bias": rng.normal(size=(c_in,)).astype(np.float32),
        "tri_att.linear.weight": rng.normal(size=(num_heads, c_in)).astype(
            np.float32
        ),
        "tri_att.mha.linear_q.weight": rng.normal(
            size=(total_hidden, c_in)
        ).astype(np.float32),
        "tri_att.mha.linear_k.weight": rng.normal(
            size=(total_hidden, c_in)
        ).astype(np.float32),
        "tri_att.mha.linear_v.weight": rng.normal(
            size=(total_hidden, c_in)
        ).astype(np.float32),
        "tri_att.mha.linear_o.weight": rng.normal(
            size=(c_in, total_hidden)
        ).astype(np.float32),
        "tri_att.mha.linear_g.weight": rng.normal(
            size=(total_hidden, c_in)
        ).astype(np.float32),
    }


def _triangle_attention_reference(
    x: jnp.ndarray,
    mask: jnp.ndarray,
    params,
    *,
    num_heads: int,
    inf: float = 1e9,
) -> jnp.ndarray:
    x = layer_norm(x, params.layer_norm)
    q = _project_heads(x, params.attention.linear_q, num_heads)
    k = _project_heads(x, params.attention.linear_k, num_heads)
    v = _project_heads(x, params.attention.linear_v, num_heads)
    q = q / jnp.sqrt(jnp.asarray(q.shape[-1], dtype=q.dtype))

    logits = jnp.einsum("...hid,...hjd->...hij", q, k)
    mask_bias = inf * (mask.astype(jnp.float32) - 1.0)
    mask_bias = mask_bias[..., :, None, None, :]
    tri_bias = linear(x, params.linear)
    tri_bias = jnp.moveaxis(tri_bias, -1, -3)
    tri_bias = jnp.expand_dims(tri_bias, axis=-4)
    logits = logits + mask_bias + tri_bias

    probs = jnp.exp(logits - jnp.max(logits, axis=-1, keepdims=True))
    probs = probs / jnp.sum(probs, axis=-1, keepdims=True)
    out = jnp.einsum("...hij,...hjd->...hid", probs.astype(v.dtype), v)
    out = jnp.swapaxes(out, -2, -3)
    gate = sigmoid(linear(x, params.attention.linear_g))
    gate = gate.reshape(gate.shape[:-1] + (num_heads, -1))
    out = out * gate
    out = out.reshape(out.shape[:-2] + (-1,))
    return linear(out, params.attention.linear_o)


def _project_heads(x, params, num_heads: int) -> jnp.ndarray:
    y = linear(x, params)
    y = y.reshape(y.shape[:-1] + (num_heads, -1))
    return jnp.swapaxes(y, -2, -3)
