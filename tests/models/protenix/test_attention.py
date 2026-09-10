from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.protenix.bridge.torch_mapping import (
    map_attention_pair_bias_state_dict,
    map_attention_state_dict,
)
from foldjax.models.protenix.models.primitives.attention import (
    AttentionPairBiasParams,
    AttentionParams,
    attention,
    attention_pair_bias,
    local_attention,
    local_attention_pair_bias,
    prepare_qkv,
)
from foldjax.models.protenix.models.primitives.primitives import (
    AdaptiveLayerNormParams,
    LayerNormParams,
    LinearParams,
    adaptive_layer_norm,
    layer_norm,
    linear,
    sigmoid,
)


def test_prepare_qkv_uses_torch_layout_and_scaling() -> None:
    x = jnp.asarray([[[1.0, 2.0], [3.0, 4.0]]], dtype=jnp.float32)
    state = {
        "attn.linear_q.weight": np.array(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, -1.0]], dtype=np.float32
        ),
        "attn.linear_q.bias": np.array([0.5, 0.0, 0.0, -0.5], dtype=np.float32),
        "attn.linear_k.weight": np.ones((4, 2), dtype=np.float32),
        "attn.linear_v.weight": np.full((4, 2), 0.25, dtype=np.float32),
        "attn.linear_o.weight": np.ones((2, 4), dtype=np.float32),
        "attn.linear_g.weight": np.zeros((4, 2), dtype=np.float32),
    }
    params = map_attention_state_dict(state, "attn")

    q, k, v = prepare_qkv(x, x, params, num_heads=2, apply_scale=True)

    expected_q_flat = np.asarray(x) @ state["attn.linear_q.weight"].T
    expected_q_flat = expected_q_flat + state["attn.linear_q.bias"]
    expected_q = expected_q_flat.reshape(1, 2, 2, 2).transpose(0, 2, 1, 3)
    expected_q = expected_q / np.sqrt(2.0)
    np.testing.assert_allclose(np.asarray(q), expected_q, rtol=1e-6, atol=1e-6)
    assert k.shape == (1, 2, 2, 2)
    assert v.shape == (1, 2, 2, 2)


def test_attention_matches_reference_formula_with_bias_and_gating() -> None:
    rng = np.random.default_rng(4)
    q_x = rng.normal(size=(1, 3, 4)).astype(np.float32)
    kv_x = rng.normal(size=(1, 3, 4)).astype(np.float32)
    bias = rng.normal(size=(1, 2, 3, 3)).astype(np.float32)
    state = {
        "attn.linear_q.weight": rng.normal(size=(4, 4)).astype(np.float32),
        "attn.linear_q.bias": rng.normal(size=(4,)).astype(np.float32),
        "attn.linear_k.weight": rng.normal(size=(4, 4)).astype(np.float32),
        "attn.linear_v.weight": rng.normal(size=(4, 4)).astype(np.float32),
        "attn.linear_o.weight": rng.normal(size=(4, 4)).astype(np.float32),
        "attn.linear_g.weight": rng.normal(size=(4, 4)).astype(np.float32),
    }
    params = map_attention_state_dict(state, "attn")

    actual = np.asarray(
        attention(jnp.asarray(q_x), jnp.asarray(kv_x), params, 2, jnp.asarray(bias))
    )

    q, k, v = prepare_qkv(jnp.asarray(q_x), jnp.asarray(kv_x), params, 2)
    logits = np.einsum("bhid,bhjd->bhij", np.asarray(q), np.asarray(k))
    logits = logits + bias
    probs = np.asarray(jax.nn.softmax(jnp.asarray(logits), axis=-1))
    out = np.einsum("bhij,bhjd->bhid", probs, np.asarray(v))
    out = out.transpose(0, 2, 1, 3)
    gate = 1.0 / (1.0 + np.exp(-(q_x @ state["attn.linear_g.weight"].T)))
    out = (out * gate.reshape(1, 3, 2, 2)).reshape(1, 3, 4)
    expected = out @ state["attn.linear_o.weight"].T

    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


def test_attention_query_chunk_matches_unchunked() -> None:
    rng = np.random.default_rng(44)
    q_x = rng.normal(size=(2, 5, 4)).astype(np.float32)
    kv_x = rng.normal(size=(2, 5, 4)).astype(np.float32)
    bias = rng.normal(size=(2, 2, 5, 5)).astype(np.float32)
    state = {
        "attn.linear_q.weight": rng.normal(size=(4, 4)).astype(np.float32),
        "attn.linear_q.bias": rng.normal(size=(4,)).astype(np.float32),
        "attn.linear_k.weight": rng.normal(size=(4, 4)).astype(np.float32),
        "attn.linear_v.weight": rng.normal(size=(4, 4)).astype(np.float32),
        "attn.linear_o.weight": rng.normal(size=(4, 4)).astype(np.float32),
        "attn.linear_g.weight": rng.normal(size=(4, 4)).astype(np.float32),
    }
    params = map_attention_state_dict(state, "attn")

    unchunked = attention(
        jnp.asarray(q_x),
        jnp.asarray(kv_x),
        params,
        2,
        jnp.asarray(bias),
    )
    chunked = attention(
        jnp.asarray(q_x),
        jnp.asarray(kv_x),
        params,
        2,
        jnp.asarray(bias),
        q_chunk_size=2,
    )

    np.testing.assert_allclose(
        np.asarray(chunked),
        np.asarray(unchunked),
        rtol=1e-5,
        atol=1e-5,
    )


def test_attention_pair_bias_has_s_false_matches_reference_formula() -> None:
    rng = np.random.default_rng(5)
    a = rng.normal(size=(1, 3, 4)).astype(np.float32)
    z = rng.normal(size=(1, 3, 3, 5)).astype(np.float32)
    state = {
        "apb.layernorm_a.weight": rng.normal(size=(4,)).astype(np.float32),
        "apb.layernorm_a.bias": rng.normal(size=(4,)).astype(np.float32),
        "apb.layernorm_z.weight": rng.normal(size=(5,)).astype(np.float32),
        "apb.layernorm_z.bias": rng.normal(size=(5,)).astype(np.float32),
        "apb.linear_nobias_z.weight": rng.normal(size=(2, 5)).astype(np.float32),
        "apb.attention.linear_q.weight": rng.normal(size=(4, 4)).astype(np.float32),
        "apb.attention.linear_q.bias": rng.normal(size=(4,)).astype(np.float32),
        "apb.attention.linear_k.weight": rng.normal(size=(4, 4)).astype(np.float32),
        "apb.attention.linear_v.weight": rng.normal(size=(4, 4)).astype(np.float32),
        "apb.attention.linear_o.weight": rng.normal(size=(4, 4)).astype(np.float32),
        "apb.attention.linear_g.weight": rng.normal(size=(4, 4)).astype(np.float32),
    }
    params = map_attention_pair_bias_state_dict(state, "apb", has_s=False)

    actual = np.asarray(
        attention_pair_bias(jnp.asarray(a), None, jnp.asarray(z), params, num_heads=2)
    )

    a_norm = layer_norm(jnp.asarray(a), params.layernorm_a)
    z_norm = layer_norm(jnp.asarray(z), params.layernorm_z)
    bias = np.asarray(z_norm) @ state["apb.linear_nobias_z.weight"].T
    bias = bias.transpose(0, 3, 1, 2)
    expected = np.asarray(
        attention(a_norm, a_norm, params.attention, 2, jnp.asarray(bias))
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)

    extra_bias = jnp.asarray(
        [[0.0, 0.7, -0.2], [-0.4, 0.0, 0.5], [0.3, -0.6, 0.0]],
        dtype=jnp.float32,
    )
    actual_with_extra = attention_pair_bias(
        jnp.asarray(a),
        None,
        jnp.asarray(z),
        params,
        num_heads=2,
        extra_attn_bias=extra_bias,
    )
    expected_with_extra = attention(
        a_norm,
        a_norm,
        params.attention,
        2,
        jnp.asarray(bias) + extra_bias[None, None, :, :],
    )
    np.testing.assert_allclose(
        actual_with_extra,
        expected_with_extra,
        rtol=1e-5,
        atol=1e-5,
    )


def test_attention_pair_bias_accepts_shared_plain_normalized_z() -> None:
    rng = np.random.default_rng(51)
    a = jnp.asarray(rng.normal(size=(2, 3, 4)).astype(np.float32))
    z = jnp.asarray(rng.normal(size=(3, 3, 5)).astype(np.float32))
    params = AttentionPairBiasParams(
        layernorm_a=LayerNormParams(weight=jnp.ones((4,)), bias=None),
        layernorm_kv=None,
        attention=AttentionParams(
            linear_q=LinearParams(jnp.asarray(rng.normal(size=(4, 4)))),
            linear_k=LinearParams(jnp.asarray(rng.normal(size=(4, 4)))),
            linear_v=LinearParams(jnp.asarray(rng.normal(size=(4, 4)))),
            linear_o=LinearParams(jnp.asarray(rng.normal(size=(4, 4)))),
            linear_g=LinearParams(jnp.asarray(rng.normal(size=(4, 4)))),
        ),
        layernorm_z=LayerNormParams(
            weight=jnp.asarray(rng.normal(size=(5,))),
            bias=None,
        ),
        linear_z=LinearParams(jnp.asarray(rng.normal(size=(2, 5)))),
    )
    expected = attention_pair_bias(a, None, z[None, ...], params, num_heads=2)
    shared = attention_pair_bias(a, None, z, params, num_heads=2)
    z_plain_normalized = layer_norm(z, LayerNormParams())

    actual = attention_pair_bias(
        a,
        None,
        z_plain_normalized,
        params,
        num_heads=2,
        z_is_normalized=True,
    )

    np.testing.assert_allclose(shared, expected, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


def test_attention_pair_bias_query_chunk_matches_unchunked() -> None:
    rng = np.random.default_rng(55)
    a = rng.normal(size=(1, 5, 4)).astype(np.float32)
    z = rng.normal(size=(1, 5, 5, 5)).astype(np.float32)
    state = {
        "apb.layernorm_a.weight": rng.normal(size=(4,)).astype(np.float32),
        "apb.layernorm_a.bias": rng.normal(size=(4,)).astype(np.float32),
        "apb.layernorm_z.weight": rng.normal(size=(5,)).astype(np.float32),
        "apb.layernorm_z.bias": rng.normal(size=(5,)).astype(np.float32),
        "apb.linear_nobias_z.weight": rng.normal(size=(2, 5)).astype(np.float32),
        "apb.attention.linear_q.weight": rng.normal(size=(4, 4)).astype(np.float32),
        "apb.attention.linear_q.bias": rng.normal(size=(4,)).astype(np.float32),
        "apb.attention.linear_k.weight": rng.normal(size=(4, 4)).astype(np.float32),
        "apb.attention.linear_v.weight": rng.normal(size=(4, 4)).astype(np.float32),
        "apb.attention.linear_o.weight": rng.normal(size=(4, 4)).astype(np.float32),
        "apb.attention.linear_g.weight": rng.normal(size=(4, 4)).astype(np.float32),
    }
    params = map_attention_pair_bias_state_dict(state, "apb", has_s=False)

    unchunked = attention_pair_bias(
        jnp.asarray(a),
        None,
        jnp.asarray(z),
        params,
        num_heads=2,
    )
    chunked = attention_pair_bias(
        jnp.asarray(a),
        None,
        jnp.asarray(z),
        params,
        num_heads=2,
        q_chunk_size=2,
    )

    np.testing.assert_allclose(
        np.asarray(chunked),
        np.asarray(unchunked),
        rtol=1e-5,
        atol=1e-5,
    )


def test_local_cross_attention_normalizes_kv_from_normalized_query() -> None:
    a = jnp.asarray(
        [[1.0, 2.0], [2.0, -1.0], [-1.0, 3.0], [4.0, 0.5]],
        dtype=jnp.float32,
    )
    s = jnp.asarray(
        [[0.5, -1.0], [1.5, 0.25], [-0.5, 2.0], [1.0, -2.0]],
        dtype=jnp.float32,
    )
    z = jnp.zeros((2, 2, 4, 1), dtype=jnp.float32)
    layernorm = LayerNormParams(weight=jnp.ones((2,)), bias=None)
    adaln_q = AdaptiveLayerNormParams(
        layernorm_a=LayerNormParams(weight=None, bias=None),
        layernorm_s=layernorm,
        linear_s=LinearParams(weight=jnp.zeros((2, 2)), bias=jnp.zeros((2,))),
        linear_no_bias_s=LinearParams(weight=jnp.eye(2), bias=None),
    )
    adaln_kv = AdaptiveLayerNormParams(
        layernorm_a=LayerNormParams(weight=None, bias=None),
        layernorm_s=layernorm,
        linear_s=LinearParams(weight=jnp.zeros((2, 2)), bias=jnp.zeros((2,))),
        linear_no_bias_s=LinearParams(
            weight=jnp.asarray([[0.5, 0.25], [-0.25, 0.75]]),
            bias=None,
        ),
    )
    attention_params = AttentionParams(
        linear_q=LinearParams(weight=jnp.eye(2), bias=jnp.zeros((2,))),
        linear_k=LinearParams(weight=jnp.eye(2), bias=None),
        linear_v=LinearParams(weight=jnp.eye(2), bias=None),
        linear_o=LinearParams(weight=jnp.eye(2), bias=None),
        linear_g=None,
    )
    gate = LinearParams(weight=jnp.zeros((2, 2)), bias=jnp.zeros((2,)))
    params = AttentionPairBiasParams(
        layernorm_a=adaln_q,
        layernorm_kv=adaln_kv,
        attention=attention_params,
        layernorm_z=LayerNormParams(weight=jnp.ones((1,)), bias=None),
        linear_z=LinearParams(weight=jnp.zeros((1, 1)), bias=None),
        linear_a_last=gate,
        has_s=True,
        cross_attention_mode=True,
    )

    actual = local_attention_pair_bias(
        a,
        s,
        z,
        params,
        num_heads=1,
        n_queries=2,
        n_keys=4,
    )

    q = adaptive_layer_norm(a, s, adaln_q)
    kv = adaptive_layer_norm(q, s, adaln_kv)
    expected = local_attention(
        q,
        kv,
        attention_params,
        1,
        trunked_attn_bias=jnp.zeros((1, 2, 2, 4)),
        n_queries=2,
        n_keys=4,
    )
    expected = sigmoid(linear(s, gate)) * expected
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_builtin_sdpa_matches_manual_global_attention() -> None:
    x = jnp.arange(24, dtype=jnp.float32).reshape(1, 4, 6) / 10.0
    params = _identity_attention_params(width=6)
    bias = jnp.arange(32, dtype=jnp.float32).reshape(1, 2, 4, 4) / 100.0

    manual = attention(x, x, params, 2, bias, attention_backend="xla")
    builtin = attention(x, x, params, 2, bias, attention_backend="xla_sdpa")
    compiled = attention(x, x, params, 2, bias, attention_backend="xla_jit")

    np.testing.assert_allclose(builtin, manual, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(compiled, manual, rtol=1e-5, atol=1e-5)


def test_builtin_sdpa_matches_manual_local_attention() -> None:
    x = jnp.arange(36, dtype=jnp.float32).reshape(1, 6, 6) / 10.0
    params = _identity_attention_params(width=6)
    bias = jnp.arange(48, dtype=jnp.float32).reshape(1, 2, 3, 2, 4) / 100.0

    manual = local_attention(
        x,
        x,
        params,
        2,
        trunked_attn_bias=bias,
        n_queries=2,
        n_keys=4,
        attention_backend="xla",
    )
    builtin = local_attention(
        x,
        x,
        params,
        2,
        trunked_attn_bias=bias,
        n_queries=2,
        n_keys=4,
        attention_backend="xla_sdpa",
    )
    compiled = local_attention(
        x,
        x,
        params,
        2,
        trunked_attn_bias=bias,
        n_queries=2,
        n_keys=4,
        attention_backend="xla_jit",
    )

    np.testing.assert_allclose(builtin, manual, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(compiled, manual, rtol=1e-5, atol=1e-5)


def _identity_attention_params(*, width: int) -> AttentionParams:
    identity = LinearParams(weight=jnp.eye(width), bias=None)
    return AttentionParams(
        linear_q=identity,
        linear_k=identity,
        linear_v=identity,
        linear_o=identity,
        linear_g=None,
    )


def _tokamax_xla(monkeypatch) -> None:
    """Run the tokamax branch through tokamax's XLA implementation.

    The shipped value is Triton, which raises `NotImplementedError` off a GPU
    -- deliberately, so a card that cannot run the kernel says so instead of
    silently measuring XLA under the tokamax name. Moving the seam is how the
    same branch is exercised here: the layout, the bias, the mask handling and
    the dtype contract are the parts under test, and they are identical.
    """

    from foldjax.models.protenix.models.primitives import attention_tokamax

    monkeypatch.setattr(attention_tokamax, "TOKAMAX_IMPLEMENTATION", "xla")


def _bfloat16_attention_params(*, width: int) -> AttentionParams:
    identity = LinearParams(weight=jnp.eye(width, dtype=jnp.bfloat16), bias=None)
    return AttentionParams(
        linear_q=identity,
        linear_k=identity,
        linear_v=identity,
        linear_o=identity,
        linear_g=None,
    )


def test_tokamax_matches_manual_global_attention_with_additive_padding(
    monkeypatch,
) -> None:
    """The global site, with the padding mask left additive inside the bias.

    A fully padded query row therefore receives one constant on every key,
    which cancels in the softmax -- the same distribution the XLA path
    produces. Handing tokamax a boolean mask instead would soften those rows
    to a uniform average and quietly change what padding means.
    """

    _tokamax_xla(monkeypatch)
    rng = np.random.default_rng(11)
    x = jnp.asarray(rng.normal(size=(1, 7, 8)), dtype=jnp.bfloat16)
    params = _bfloat16_attention_params(width=8)
    pair_bias = rng.normal(size=(1, 2, 7, 7)) * 0.5
    padding = np.zeros((1, 1, 1, 7), dtype=np.float32)
    padding[..., 5:] = -1.0e10
    bias = jnp.asarray(pair_bias + padding, dtype=jnp.bfloat16)

    manual = attention(x, x, params, 2, bias, attention_backend="xla")
    fused = attention(x, x, params, 2, bias, attention_backend="tokamax")

    assert fused.dtype == manual.dtype == jnp.bfloat16
    np.testing.assert_allclose(
        np.asarray(fused, dtype=np.float32),
        np.asarray(manual, dtype=np.float32),
        rtol=2e-2,
        atol=2e-2,
    )


def test_tokamax_matches_manual_local_attention_with_window_mask(
    monkeypatch,
) -> None:
    """The windowed atom site, with a padded trunk and a masked atom.

    Seven atoms into trunks of two leaves one padded query row, and the false
    entry in `sequence_mask` masks a second. Both are rows the fused kernel
    could return NaN for and the trailing `* sequence_mask` could not rescue,
    so they are the point of the case rather than incidental.
    """

    _tokamax_xla(monkeypatch)
    rng = np.random.default_rng(12)
    x = jnp.asarray(rng.normal(size=(1, 7, 8)), dtype=jnp.bfloat16)
    params = _bfloat16_attention_params(width=8)
    bias = jnp.asarray(rng.normal(size=(1, 2, 4, 2, 4)) * 0.5, dtype=jnp.bfloat16)
    sequence_mask = jnp.asarray([True, True, False, True, True, True, True])
    kwargs = dict(
        trunked_attn_bias=bias,
        n_queries=2,
        n_keys=4,
        sequence_mask=sequence_mask,
    )

    manual = local_attention(x, x, params, 2, **kwargs, attention_backend="xla")
    fused = local_attention(x, x, params, 2, **kwargs, attention_backend="tokamax")

    assert not bool(jnp.isnan(fused).any())
    assert fused.dtype == manual.dtype == jnp.bfloat16
    np.testing.assert_allclose(
        np.asarray(fused, dtype=np.float32),
        np.asarray(manual, dtype=np.float32),
        rtol=2e-2,
        atol=2e-2,
    )


def test_tokamax_ignores_the_query_chunk_size_and_says_so(monkeypatch) -> None:
    """A chunk size cannot reach a kernel that takes the whole query axis."""

    import foldjax.models.protenix.models.primitives.attention as attention_module

    _tokamax_xla(monkeypatch)
    monkeypatch.setattr(attention_module, "_WARNED_UNCHUNKABLE", False)
    rng = np.random.default_rng(13)
    x = jnp.asarray(rng.normal(size=(1, 8, 8)), dtype=jnp.bfloat16)
    params = _bfloat16_attention_params(width=8)
    bias = jnp.asarray(rng.normal(size=(1, 2, 8, 8)) * 0.5, dtype=jnp.bfloat16)

    unchunked = attention(x, x, params, 2, bias, attention_backend="tokamax")
    with pytest.warns(UserWarning, match="q_chunk_size=2 is not used"):
        chunked = attention(
            x, x, params, 2, bias, q_chunk_size=2, attention_backend="tokamax"
        )

    np.testing.assert_array_equal(np.asarray(chunked), np.asarray(unchunked))


def test_tokamax_warns_once_on_float32_inputs(monkeypatch) -> None:
    """An fp32 x tokamax cell is measurable, but it must not look like the fast one."""

    from foldjax.models.protenix.models.primitives import attention_tokamax

    _tokamax_xla(monkeypatch)
    monkeypatch.setattr(attention_tokamax, "_WARNED_FLOAT32", False)
    x = jnp.arange(24, dtype=jnp.float32).reshape(1, 4, 6) / 10.0
    params = _identity_attention_params(width=6)
    bias = jnp.arange(32, dtype=jnp.float32).reshape(1, 2, 4, 4) / 100.0

    with pytest.warns(UserWarning, match="float32 path"):
        first = attention(x, x, params, 2, bias, attention_backend="tokamax")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        second = attention(x, x, params, 2, bias, attention_backend="tokamax")

    assert first.dtype == second.dtype == jnp.float32
    manual = attention(x, x, params, 2, bias, attention_backend="xla")
    np.testing.assert_allclose(first, manual, rtol=1e-5, atol=1e-5)


def test_bfloat16_inputs_do_not_warn(monkeypatch) -> None:
    """The intended cell is silent, so the float32 warning stays informative."""

    from foldjax.models.protenix.models.primitives import attention_tokamax

    _tokamax_xla(monkeypatch)
    monkeypatch.setattr(attention_tokamax, "_WARNED_FLOAT32", False)
    x = jnp.asarray(np.random.default_rng(14).normal(size=(1, 4, 8)), jnp.bfloat16)
    params = _bfloat16_attention_params(width=8)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        attention(x, x, params, 2, None, attention_backend="tokamax")


def test_tokamax_is_refused_under_context_parallelism(monkeypatch) -> None:
    """These two sites have no `shard_map` wrapper the way triangle attention does."""

    import foldjax.models.protenix.models.primitives.attention as attention_module

    monkeypatch.setattr(attention_module, "cp_mesh", lambda: object())
    x = jnp.arange(24, dtype=jnp.float32).reshape(1, 4, 6) / 10.0
    params = _identity_attention_params(width=6)

    with pytest.raises(ValueError, match="not supported under context parallelism"):
        attention(x, x, params, 2, None, attention_backend="tokamax")
    with pytest.raises(ValueError, match="not supported under context parallelism"):
        local_attention(
            x,
            x,
            params,
            2,
            trunked_attn_bias=None,
            n_queries=2,
            n_keys=4,
            attention_backend="tokamax",
        )

    # The backends that are wired for a mesh keep working under the same guard.
    attention(x, x, params, 2, None, attention_backend="xla")


def test_both_backend_options_reach_the_tokamax_sites(monkeypatch) -> None:
    """The two option strings thread from the model kwargs down to the kernel.

    The parity cases above call the attention functions directly, so nothing
    there would notice a site that stopped passing its backend along. This
    watches the module the branch imports: `diffusion_attention_backend`
    reaches the atom encoder/decoder window and the diffusion transformer's
    global attention, `single_attention_backend` reaches the Pairformer's
    single attention, and neither fires for the default.
    """

    import jax

    from foldjax.models.protenix.bridge.torch_mapping import (
        map_pairformer_stack_state_dict,
    )
    from foldjax.models.protenix.models.model import protenix_infer_static
    from foldjax.models.protenix.models.primitives import attention_tokamax
    from foldjax.models.protenix.models.trunk_blocks.pairformer import (
        pairformer_stack,
    )

    from .test_model import _toy_features, _toy_params
    from .test_pairformer import _pairformer_block_state

    _tokamax_xla(monkeypatch)
    # The toy checkpoint is float32, so the diffusion sites warn here. Set
    # through monkeypatch so the flag is restored and no later test in the
    # session finds the once-only warning already spent.
    monkeypatch.setattr(attention_tokamax, "_WARNED_FLOAT32", False)
    reached: list[tuple[int, ...]] = []
    wrapped = attention_tokamax.tokamax_attention

    def spy(q, k, v, **kwargs):
        reached.append(tuple(q.shape))
        return wrapped(q, k, v, **kwargs)

    monkeypatch.setattr(attention_tokamax, "tokamax_attention", spy)

    rng = np.random.default_rng(15)
    state = _pairformer_block_state(rng, c_s=4, c_z=4, heads=2, prefix="s.blocks.0")
    stack = map_pairformer_stack_state_dict(state, "s", has_s=True)
    stack = jax.tree.map(lambda leaf: jnp.asarray(leaf, jnp.bfloat16), stack)
    single = jnp.asarray(rng.normal(size=(1, 3, 4)), dtype=jnp.bfloat16)
    pair = jnp.asarray(rng.normal(size=(1, 3, 3, 4)), dtype=jnp.bfloat16)
    pair_mask = jnp.ones((1, 3, 3), dtype=jnp.bfloat16)

    pairformer_stack(
        single, pair, pair_mask, stack, use_scan=False, single_attention_backend="xla"
    )
    assert reached == []
    pairformer_stack(
        single,
        pair,
        pair_mask,
        stack,
        use_scan=False,
        single_attention_backend="tokamax",
    )
    assert len(reached) == 1

    infer = dict(
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
    )
    schedule = jnp.asarray([1.0, 0.0], dtype=jnp.float32)

    reached.clear()
    protenix_infer_static(_toy_features(), _toy_params(), schedule, **infer)
    assert reached == []

    coordinate = protenix_infer_static(
        _toy_features(),
        _toy_params(),
        schedule,
        diffusion_attention_backend="tokamax",
        **infer,
    )["coordinate"]
    assert reached, "diffusion_attention_backend=tokamax reached no fused site"
    assert bool(jnp.isfinite(coordinate).all())
