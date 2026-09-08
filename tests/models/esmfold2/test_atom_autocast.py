from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.esmfold2.models import atom, model
from tests.models.esmfold2.test_distogram_output_flag import _cheap_features


@pytest.mark.parametrize("compiled", [False, True])
def test_native_rope_rounds_uid_and_phase_before_trigonometry(compiled):
    position = jnp.zeros((1, 2, 3), jnp.float32)
    uid = jnp.array([[257, 259]], jnp.int32)

    def run(p, u):
        return atom.build_3d_rope(p, u, head_dim=32, native_autocast=True)

    cos, sin = (jax.jit(run) if compiled else run)(position, uid)
    phase = uid.astype(jnp.bfloat16).astype(jnp.float32)
    np.testing.assert_array_equal(cos[..., 6], jnp.cos(phase).astype(jnp.bfloat16))
    np.testing.assert_array_equal(sin[..., 6], jnp.sin(phase).astype(jnp.bfloat16))
    ordinary_cos, _ = atom.build_3d_rope(position, uid, head_dim=32)
    assert not np.array_equal(cos[..., 6], ordinary_cos[..., 6])


@pytest.mark.parametrize("native", [False, True])
def test_swiglu_storage_and_legacy_dtype(native):
    rng = np.random.default_rng(1)
    x = jnp.asarray(rng.normal(size=(1, 4, 32)), jnp.float32)
    params = {
        "w_up.weight": jnp.asarray(rng.normal(scale=0.1, size=(64, 32)), jnp.float32),
        "w_down.weight": jnp.asarray(rng.normal(scale=0.1, size=(32, 32)), jnp.float32),
    }
    result = jax.jit(lambda x, p: atom.swiglu_ffn(x, p, native_autocast=native))(
        x, params
    )
    assert result.dtype == (jnp.bfloat16 if native else jnp.float32)
    assert np.isfinite(result).all()


def test_native_adaln_retains_bf16_scale_addition():
    x = jnp.array([[1.0, 2.0, 3.0, 4.0]], jnp.float32)
    scale = jnp.full_like(x, 0.003, dtype=jnp.bfloat16)
    shift = jnp.zeros_like(scale)
    expected = atom.rms_norm(x) * (1 + scale).astype(jnp.float32)
    result = jax.jit(lambda a, b, c: atom._adaln(a, b, c, atom.FLOAT32_EPS, True))(
        x, scale, shift
    )
    np.testing.assert_allclose(result, expected, rtol=1e-7, atol=1e-7)
    assert result.dtype == jnp.float32
    assert not np.allclose(result, atom.rms_norm(x) * (1 + scale.astype(jnp.float32)))


def test_native_window_accumulates_in_fp32_then_stores_bf16():
    rng = np.random.default_rng(4)
    q, k, v = (
        jnp.asarray(rng.normal(size=(1, 24, 2, 32)), jnp.bfloat16) for _ in range(3)
    )
    valid = jnp.arange(24)[None] < 21
    actual = atom._windowed_attention(
        q,
        k,
        v,
        valid,
        half_window=2,
        rows_per_block=8,
        scale=32**-0.5,
        native_autocast=True,
    )
    logits = (
        jnp.einsum("bihd,bjhd->bhij", q.astype(jnp.float32), k.astype(jnp.float32))
        * 32**-0.5
    )
    logits = jnp.where(
        atom.sliding_window_mask(valid, 2)[:, None], logits, jnp.finfo(jnp.float32).min
    )
    expected = jnp.einsum(
        "bhij,bjhd->bihd", jax.nn.softmax(logits, axis=-1), v.astype(jnp.float32)
    ).astype(jnp.bfloat16)
    assert actual.dtype == jnp.bfloat16
    np.testing.assert_array_equal(actual[:, :21], expected[:, :21])


@pytest.mark.parametrize("cp", [False, True])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_predict_routes_original_atom_weights_only_in_native_scope(
    monkeypatch, cp, dtype
):
    monkeypatch.setattr(model, "cp_mesh", lambda: object() if cp else None)
    captured = {}

    class StopError(Exception):
        pass

    def stop(*args, **kwargs):
        captured.update(params=args[10], kwargs=kwargs)
        raise StopError

    monkeypatch.setattr(model, "inputs_embedding", stop)
    params = {"inputs_embedder.probe.weight": jnp.array([1.0037], jnp.float32)}
    with pytest.raises(StopError):
        model.predict(
            jax.random.key(1),
            _cheap_features(),
            params,
            settings=replace(model.ModelSettings(), trunk_dtype=dtype),
            n_chains=1,
        )
    native = dtype == "bfloat16" and not cp
    assert captured["kwargs"]["native_autocast"] is native
    expected = params["inputs_embedder.probe.weight"].astype(
        jnp.float32 if native else dtype
    )
    np.testing.assert_array_equal(
        captured["params"]["inputs_embedder.probe.weight"], expected
    )
