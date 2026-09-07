"""FFI multiplication must not silently discard the active JAX policy."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import pytest

from foldjax.models._cueq import triangle_multiplication_precision


@pytest.mark.parametrize(
    "policy, expected",
    [
        (None, "DEFAULT"),
        ("default", "DEFAULT"),
        ("bfloat16", "DEFAULT"),
        ("high", "TF32"),
        ("tensorfloat32", "TF32"),
        ("highest", "IEEE"),
        ("float32", "IEEE"),
        ("TF32_TF32_F32", "TF32"),
        ("TF32_TF32_F32_X3", "TF32x3"),
        ("F32_F32_F32", "IEEE"),
    ],
)
def test_triangle_precision_respects_context(policy, expected):
    cuex = SimpleNamespace(
        TriMulPrecision=SimpleNamespace(
            DEFAULT="DEFAULT", TF32="TF32", TF32x3="TF32x3", IEEE="IEEE"
        )
    )
    previous = jax.config.jax_default_matmul_precision
    with jax.default_matmul_precision(policy):
        assert triangle_multiplication_precision(cuex, dtype=jnp.float32) == expected
    assert jax.config.jax_default_matmul_precision == previous


def test_unsupported_precision_is_not_silently_downgraded():
    with jax.default_matmul_precision("F64_F64_F64"):
        with pytest.raises(ValueError, match="select XLA multiplication"):
            triangle_multiplication_precision(SimpleNamespace(), dtype=jnp.float32)


@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float16])
@pytest.mark.parametrize("policy", [None, "default", "high", "highest",
                                   "TF32_TF32_F32", "TF32_TF32_F32_X3"])
def test_reduced_precision_never_enters_float32_only_trimul_mode(dtype, policy):
    cuex = SimpleNamespace(TriMulPrecision=SimpleNamespace(DEFAULT="DEFAULT"))
    with jax.default_matmul_precision(policy):
        assert triangle_multiplication_precision(cuex, dtype=dtype) == "DEFAULT"


@pytest.mark.parametrize(
    "policy, expected",
    [
        (None, jax.lax.Precision.DEFAULT),
        ("default", jax.lax.Precision.DEFAULT),
        ("bfloat16", jax.lax.Precision.DEFAULT),
        ("high", jax.lax.Precision.HIGH),
        ("tensorfloat32", jax.lax.Precision.HIGH),
        ("highest", jax.lax.Precision.HIGHEST),
        ("float32", jax.lax.Precision.HIGHEST),
        ("TF32_TF32_F32", jax.lax.Precision.HIGH),
        ("F32_F32_F32", jax.lax.Precision.HIGHEST),
    ],
)
def test_attention_ffi_respects_active_precision(monkeypatch, policy, expected):
    from foldjax.models import _cueq

    captured = {}

    def attention(**kwargs):
        captured.update(kwargs)
        return kwargs["q"], None, None

    monkeypatch.setattr(
        _cueq, "load_cueq", lambda: SimpleNamespace(triangle_attention=attention)
    )
    q = jnp.ones((1, 2, 1, 3, 4), dtype=jnp.float32)
    with jax.default_matmul_precision(policy):
        _cueq.cueq_attention_core(
            q,
            q,
            q,
            jnp.zeros((1, 1, 1, 3, 3)),
            jnp.zeros((1, 2, 1, 1, 3)),
            scale=0.5,
        )
    assert captured["precision"] is expected


@pytest.mark.parametrize("policy", ["TF32_TF32_F32_X3", "F64_F64_F64"])
def test_attention_rejects_unrepresentable_precision(monkeypatch, policy):
    from foldjax.models import _cueq

    monkeypatch.setattr(_cueq, "load_cueq", lambda: SimpleNamespace())
    q = jnp.ones((1, 2, 1, 3, 4), dtype=jnp.float32)
    with jax.default_matmul_precision(policy):
        with pytest.raises(ValueError, match="select XLA attention"):
            _cueq.cueq_attention_core(
                q,
                q,
                q,
                jnp.zeros((1, 1, 1, 3, 3)),
                jnp.zeros((1, 2, 1, 1, 3)),
                scale=0.5,
            )
