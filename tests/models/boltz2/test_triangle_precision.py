from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from foldjax.models.boltz2.models.triangle.triangle import (
    triangle_multiplication_forward,
)


@pytest.fixture(autouse=True)
def _use_xla_triangle_multiplication(monkeypatch) -> None:
    monkeypatch.setenv("BOLTZ_JAX_TRIANGLE_MULTIPLICATION_BACKEND", "xla")


def _params(dtype: jnp.dtype) -> dict[str, dict[str, jax.Array]]:
    key = jax.random.key(0)
    keys = jax.random.split(key, 4)
    return {
        "norm_in": {"scale": jnp.ones(4, dtype), "bias": jnp.zeros(4, dtype)},
        "g_in": {"kernel": jax.random.normal(keys[0], (4, 8), dtype=dtype)},
        "p_in": {"kernel": jax.random.normal(keys[1], (4, 8), dtype=dtype)},
        "norm_out": {"scale": jnp.ones(4, dtype), "bias": jnp.zeros(4, dtype)},
        "p_out": {"kernel": jax.random.normal(keys[2], (4, 4), dtype=dtype)},
        "g_out": {"kernel": jax.random.normal(keys[3], (4, 4), dtype=dtype)},
    }


@pytest.mark.parametrize("precision", ["float32", "bf16"])
def test_triangle_contraction_precision_modes_preserve_output_contract(
    precision: str,
) -> None:
    dtype = jnp.bfloat16
    x = jnp.arange(1 * 3 * 3 * 4, dtype=jnp.float32).reshape(1, 3, 3, 4)
    x = (x / 16).astype(dtype)
    mask = jnp.ones((1, 3, 3), dtype=dtype)

    output = triangle_multiplication_forward(
        _params(dtype),
        x,
        mask,
        "outgoing",
        chunk_size=2,
        contraction_precision=precision,
    )

    assert output.shape == x.shape
    assert output.dtype == dtype
    assert bool(jnp.isfinite(output).all())


def test_triangle_contraction_precision_rejects_unknown_mode() -> None:
    dtype = jnp.bfloat16
    x = jnp.ones((1, 2, 2, 4), dtype=dtype)
    mask = jnp.ones((1, 2, 2), dtype=dtype)

    with pytest.raises(ValueError, match="Unsupported contraction_precision"):
        triangle_multiplication_forward(
            _params(dtype),
            x,
            mask,
            "outgoing",
            contraction_precision="auto",
        )
