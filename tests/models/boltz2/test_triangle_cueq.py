from __future__ import annotations

import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp

from foldjax.models.boltz2.models.triangle.triangle import (
    triangle_multiplication_forward,
)
from foldjax.models.boltz2.models.triangle.triangle_cueq import (
    cueq_attention_core,
    cueq_triangle_multiplication_forward,
)


def test_cueq_triangle_maps_boltz_kernel_layout(monkeypatch) -> None:
    captured = {}

    def fake_triangle_multiplicative_update(**kwargs):
        captured.update(kwargs)
        return kwargs["x"]

    monkeypatch.setitem(
        sys.modules,
        "cuequivariance_jax",
        SimpleNamespace(
            triangle_multiplicative_update=fake_triangle_multiplicative_update
        ),
    )
    params = {
        "norm_in": {"scale": jnp.ones(4), "bias": jnp.zeros(4)},
        "g_in": {"kernel": jnp.arange(32).reshape(4, 8)},
        "p_in": {"kernel": jnp.arange(32, 64).reshape(4, 8)},
        "norm_out": {"scale": jnp.ones(4), "bias": jnp.zeros(4)},
        "p_out": {"kernel": jnp.arange(16).reshape(4, 4)},
        "g_out": {"kernel": jnp.arange(16, 32).reshape(4, 4)},
    }
    x = jnp.ones((1, 3, 3, 4))
    mask = jnp.ones((1, 3, 3))

    output = cueq_triangle_multiplication_forward(
        params, x, mask, "incoming", eps=1e-4
    )

    assert output is x
    assert captured["direction"] == "incoming"
    assert captured["fallback"] is False
    assert captured["eps"] == 1e-4
    assert jnp.array_equal(captured["p_in_weight"], params["p_in"]["kernel"].T)
    assert jnp.array_equal(captured["g_in_weight"], params["g_in"]["kernel"].T)
    assert jnp.array_equal(captured["p_out_weight"], params["p_out"]["kernel"].T)
    assert jnp.array_equal(captured["g_out_weight"], params["g_out"]["kernel"].T)


def test_triangle_dispatches_to_cueq_by_default(monkeypatch) -> None:
    import foldjax.models.boltz2.models.triangle.triangle_cueq as cueq_module

    sentinel = jnp.full((1, 2, 2, 4), 7, dtype=jnp.bfloat16)
    monkeypatch.delenv("BOLTZ_JAX_TRIANGLE_MULTIPLICATION_BACKEND", raising=False)
    monkeypatch.setattr(
        cueq_module,
        "cueq_triangle_multiplication_forward",
        lambda *args, **kwargs: sentinel,
    )
    params = {
        "norm_in": {"scale": jnp.ones(4), "bias": jnp.zeros(4)},
        "g_in": {"kernel": jnp.ones((4, 8))},
        "p_in": {"kernel": jnp.ones((4, 8))},
        "norm_out": {"scale": jnp.ones(4), "bias": jnp.zeros(4)},
        "p_out": {"kernel": jnp.ones((4, 4))},
        "g_out": {"kernel": jnp.ones((4, 4))},
    }

    output = triangle_multiplication_forward(
        params,
        jnp.ones((1, 2, 2, 4), dtype=jnp.bfloat16),
        jnp.ones((1, 2, 2), dtype=jnp.bfloat16),
        "outgoing",
    )

    assert output is sentinel


def test_cuda13_extra_installs_cueq_runtime() -> None:
    # FoldJAX carries the pure-JAX half of cuEq in its base requirements, since
    # every vendored port imports it, and keeps only the CUDA 13 ops build
    # behind the extra. Installing `cuda13` still yields the whole runtime.
    project = Path(__file__).resolve().parents[3] / "pyproject.toml"
    parsed = tomllib.loads(project.read_text(encoding="utf-8"))["project"]
    base = parsed["dependencies"]
    cuda13 = parsed["optional-dependencies"]["cuda13"]

    assert "cuequivariance==0.11.1" in base + cuda13
    assert "cuequivariance-jax==0.11.1" in base + cuda13
    assert "cuequivariance-ops-jax-cu13==0.11.1" in cuda13


def test_cueq_attention_maps_mask_and_returns_primary_output(monkeypatch) -> None:
    captured = {}
    expected = jnp.ones((1, 2, 1, 3, 4), dtype=jnp.bfloat16)

    def fake_triangle_attention(**kwargs):
        captured.update(kwargs)
        return expected, jnp.zeros(1), jnp.zeros(1)

    monkeypatch.setitem(
        sys.modules,
        "cuequivariance_jax",
        SimpleNamespace(triangle_attention=fake_triangle_attention),
    )
    q = jnp.ones_like(expected)
    mask_bias = jnp.array([[[[[0.0, -1e9, 0.0]]], [[[0.0, 0.0, -1e9]]]]])

    output = cueq_attention_core(
        q,
        q,
        q,
        jnp.zeros((1, 1, 1, 3, 3)),
        mask_bias,
        scale=0.5,
        precision=None,
    )

    assert output is expected
    assert captured["scale"] == 0.5
    assert captured["mask"].dtype == jnp.bool_
    assert jnp.array_equal(captured["mask"], mask_bias == 0)
