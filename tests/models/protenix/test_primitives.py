from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.protenix.models.primitives import primitives as primitives_module
from foldjax.models.protenix.models.primitives.primitives import (
    AdaptiveLayerNormParams,
    LayerNormParams,
    LinearParams,
    TransitionParams,
    _compiled_transition,
    _transition_block,
    _transition_for_runtime,
    adaptive_layer_norm,
    compiled_transition,
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


def _transition_params(rng, channels: int, hidden: int) -> TransitionParams:
    def linear(out_features: int, in_features: int) -> LinearParams:
        return LinearParams(
            weight=rng.normal(size=(out_features, in_features)).astype(np.float32),
            bias=None,
        )

    return TransitionParams(
        layer_norm=LayerNormParams(
            weight=rng.normal(size=(channels,)).astype(np.float32),
            bias=rng.normal(size=(channels,)).astype(np.float32),
        ),
        linear_a=linear(hidden, channels),
        linear_b=linear(hidden, channels),
        linear_out=linear(channels, hidden),
    )


_SERIAL_IDENTITY = ("serial", 1, (1, 1), ())


def _historical_transition(
    x: jnp.ndarray,
    params: TransitionParams,
    *,
    chunk_size: int,
) -> jnp.ndarray:
    return jnp.concatenate(
        [
            _transition_block(x[start : start + chunk_size], params)
            for start in range(0, x.shape[0], chunk_size)
        ],
        axis=0,
    )


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_sequenced_transition_preserves_chunks_and_short_tail(dtype) -> None:
    rng = np.random.default_rng(13)
    x = jnp.asarray(rng.normal(size=(18, 7, 6)), dtype=dtype)
    params = jax.tree.map(
        lambda value: jnp.asarray(value, dtype=dtype),
        _transition_params(rng, channels=6, hidden=16),
    )

    expected = jax.jit(
        lambda value: _historical_transition(value, params, chunk_size=4)
    )(x)
    actual = jax.jit(
        lambda value: _transition_for_runtime(
            value,
            params,
            chunk_size=4,
            runtime_identity=_SERIAL_IDENTITY,
        )
    )(x)
    expected = np.asarray(expected)
    actual = np.asarray(actual)

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
    # lax.map evaluates the non-divisible tail through a separate vmap with
    # the same short shape as the historical final slice.
    np.testing.assert_array_equal(actual[-2:], expected[-2:])


def test_sequenced_transition_preserves_nonfinite_classes() -> None:
    rng = np.random.default_rng(14)
    values = rng.normal(size=(18, 7, 6)).astype(np.float32)
    values[0, 0, 0] = np.nan
    values[5, 1, 1] = np.inf
    values[12, 2, 2] = -np.inf
    values[-1, 3, 3] = np.nan
    x = jnp.asarray(values)
    params = _transition_params(rng, channels=6, hidden=16)

    expected = np.asarray(
        jax.jit(lambda value: _historical_transition(value, params, chunk_size=4))(
            x
        )
    )
    actual = np.asarray(
        jax.jit(
            lambda value: _transition_for_runtime(
                value,
                params,
                chunk_size=4,
                runtime_identity=_SERIAL_IDENTITY,
            )
        )(x)
    )

    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    np.testing.assert_array_equal(np.isposinf(actual), np.isposinf(expected))
    np.testing.assert_array_equal(np.isneginf(actual), np.isneginf(expected))
    finite = np.isfinite(actual) & np.isfinite(expected)
    np.testing.assert_allclose(actual[finite], expected[finite], rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(
        actual[-2:].view(np.uint32),
        expected[-2:].view(np.uint32),
    )


def _compiled_transition_hlo(
    *,
    rows: int,
    chunk_size: int,
    dtype: jnp.dtype,
    runtime_identity: tuple,
) -> str:
    rng = np.random.default_rng(15)
    x = jnp.zeros((rows, 7, 6), dtype=dtype)
    params = jax.tree.map(
        lambda value: jnp.asarray(value, dtype=dtype),
        _transition_params(rng, channels=6, hidden=16),
    )
    return jax.export.export(
        _compiled_transition,
        platforms=("cpu",),
    )(
        x,
        params,
        chunk_size=chunk_size,
        runtime_identity=runtime_identity,
    ).mlir_module()


def test_large_serial_cpu_float32_transition_uses_one_loop() -> None:
    hlo = _compiled_transition_hlo(
        rows=16,
        chunk_size=4,
        dtype=jnp.float32,
        runtime_identity=_SERIAL_IDENTITY,
    )

    assert hlo.count("stablehlo.while") == 1
    assert "stablehlo.concatenate" not in hlo


@pytest.mark.parametrize(
    "function",
    [transition, compiled_transition],
    ids=("transition", "compiled-transition"),
)
@pytest.mark.parametrize(
    ("platform", "uses_loop"),
    [("cpu", True), ("cuda", False), ("tpu", False)],
)
def test_public_transition_route_follows_the_actual_lowering_target(
    function,
    platform: str,
    uses_loop: bool,
) -> None:
    """Portable lowering must not inherit the build process's CPU default."""

    rng = np.random.default_rng(17)
    x = jnp.zeros((16, 7, 6), dtype=jnp.float32)
    params = _transition_params(rng, channels=6, hidden=16)

    def run(value, parameters):
        return function(value, parameters, chunk_size=4)

    # Cross-platform export lowers StableHLO without requiring a target device,
    # so CUDA and TPU routing remain covered by CPU-only CI.
    exported = jax.export.export(
        jax.jit(run),
        platforms=(platform,),
    )(x, params)
    hlo = exported.mlir_module()

    if uses_loop:
        assert hlo.count("stablehlo.while") == 1
        assert "stablehlo.concatenate" not in hlo
    else:
        assert "stablehlo.while" not in hlo
        assert hlo.count("stablehlo.concatenate") == 1


@pytest.mark.parametrize(
    ("rows", "dtype", "runtime_identity"),
    [
        (12, jnp.float32, _SERIAL_IDENTITY),
        (16, jnp.bfloat16, _SERIAL_IDENTITY),
        (16, jnp.float32, ("1d", 4, (4, 1), ("cp",))),
        (
            16,
            jnp.float32,
            ("2d", 4, (2, 2), ("cp_row", "cp_col")),
        ),
    ],
)
def test_transition_keeps_historical_route_outside_cpu_map_gate(
    rows: int,
    dtype: jnp.dtype,
    runtime_identity: tuple,
) -> None:
    hlo = _compiled_transition_hlo(
        rows=rows,
        chunk_size=4,
        dtype=dtype,
        runtime_identity=runtime_identity,
    )

    assert "stablehlo.while" not in hlo
    assert hlo.count("stablehlo.concatenate") == 1


def test_cpu_mapped_transition_preserves_vmap_grad_semantics() -> None:
    rng = np.random.default_rng(18)
    cpu = jax.local_devices(backend="cpu")[0]
    x = jax.device_put(
        rng.normal(size=(3, 18, 7, 6)).astype(np.float32),
        cpu,
    )
    params = jax.tree.map(
        lambda value: jax.device_put(value, cpu),
        _transition_params(rng, channels=6, hidden=16),
    )

    def mapped_loss(value):
        result = transition(value, params, chunk_size=4)
        return jnp.sum(jnp.square(result))

    def historical_loss(value):
        result = _historical_transition(value, params, chunk_size=4)
        return jnp.sum(jnp.square(result))

    mapped_values, mapped_grads = jax.jit(
        jax.vmap(jax.value_and_grad(mapped_loss))
    )(x)
    historical_values, historical_grads = jax.jit(
        jax.vmap(jax.value_and_grad(historical_loss))
    )(x)

    np.testing.assert_allclose(mapped_values, historical_values, rtol=1e-6, atol=1e-6)
    gradient_delta = np.max(np.abs(mapped_grads - historical_grads))
    gradient_scale = np.max(np.abs(historical_grads))
    assert gradient_delta <= 1e-6 * gradient_scale


def test_unchunked_transition_stays_whole() -> None:
    hlo = _compiled_transition_hlo(
        rows=16,
        chunk_size=0,
        dtype=jnp.float32,
        runtime_identity=_SERIAL_IDENTITY,
    )

    assert "stablehlo.while" not in hlo
    assert "stablehlo.concatenate" not in hlo


def test_compiled_transition_keys_cache_by_public_topology_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(16)
    x = jnp.asarray(rng.normal(size=(16, 7, 6)), dtype=jnp.float32)
    params = _transition_params(rng, channels=6, hidden=16)
    serial = ("serial", 1, (1, 1), ())
    one_dimensional = ("1d", 4, (4, 1), ("cp",))

    _compiled_transition.clear_cache()
    try:
        monkeypatch.setattr(primitives_module, "cp_identity", lambda: serial)
        serial_result = compiled_transition(x, params, chunk_size=4)
        jax.block_until_ready(serial_result)
        assert _compiled_transition._cache_size() == 1

        monkeypatch.setattr(
            primitives_module,
            "cp_identity",
            lambda: one_dimensional,
        )
        cp_result = compiled_transition(x, params, chunk_size=4)
        jax.block_until_ready(cp_result)
        assert _compiled_transition._cache_size() == 2
        np.testing.assert_allclose(cp_result, serial_result, rtol=1e-6, atol=1e-6)
    finally:
        _compiled_transition.clear_cache()


def test_blocking_the_transition_does_not_change_what_it_computes() -> None:
    """The transition widens before it narrows, and holds three wide copies.

    On OpenDDE's structural pair representation those are three
    f32[946, 946, 768] buffers -- 7,866 MiB of a 10,914 MiB temp arena for an
    operation that reduces only over the channel axis. Blocking the leading
    axis is therefore mathematically exact; it is not bit-identical only
    because XLA tiles the smaller GEMM differently.
    """
    rng = np.random.default_rng(11)
    x = jnp.asarray(rng.normal(size=(24, 5, 6)).astype(np.float32))
    params = _transition_params(rng, channels=6, hidden=16)

    whole = np.asarray(transition(x, params, chunk_size=0))
    for chunk_size in (1, 5, 7, 23, 24, 100):
        blocked = np.asarray(transition(x, params, chunk_size=chunk_size))
        assert blocked.shape == whole.shape
        np.testing.assert_allclose(blocked, whole, rtol=1e-3, atol=1e-3)


def test_the_transition_blocks_itself_only_when_the_wide_form_is_large() -> None:
    """No knob: there is no accuracy to trade, only kernel launches.

    Blocking a tensor that already fits would pay launch overhead for nothing,
    so the budget has to bind on size rather than on rank or call site.
    """
    from foldjax.models.protenix.models.primitives.primitives import (
        _TRANSITION_WIDE_BUDGET_BYTES,
        _transition_chunk_rows,
    )

    rng = np.random.default_rng(12)
    params = _transition_params(rng, channels=6, hidden=16)

    # Small: left whole.
    assert _transition_chunk_rows(jnp.zeros((24, 5, 6), jnp.float32), params) is None
    # A single row cannot be split further.
    assert _transition_chunk_rows(jnp.zeros((1, 5, 6), jnp.float32), params) is None

    # Large enough to bind: the widened form is rows * 4096 * 16 * 4 bytes.
    rows = _transition_chunk_rows(jnp.zeros((4096, 4096, 6), jnp.float32), params)
    assert rows is not None and 1 <= rows < 4096
    assert rows * 4096 * 16 * 4 <= _TRANSITION_WIDE_BUDGET_BYTES
