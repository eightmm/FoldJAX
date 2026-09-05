"""FFI multiplication must not silently discard the active JAX policy."""

from types import SimpleNamespace

import jax
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
        assert triangle_multiplication_precision(cuex) == expected
    assert jax.config.jax_default_matmul_precision == previous


def test_unsupported_precision_is_not_silently_downgraded():
    with jax.default_matmul_precision("F64_F64_F64"):
        with pytest.raises(ValueError, match="select XLA multiplication"):
            triangle_multiplication_precision(SimpleNamespace())
