from __future__ import annotations

import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
from jax import lax

from foldjax.models.protenix.models.primitives.primitives import (
    LayerNormParams,
    LinearParams,
)
from foldjax.models.protenix.models.triangle.triangle import (
    TriangleMultiplicationParams,
    _triangle_attention_backend,
    triangle_multiplication,
)
from foldjax.models.protenix.models.triangle.triangle_cueq import (
    cueq_attention_core,
    cueq_triangle_multiplication,
)


def _params(c_z: int = 4, c_hidden: int = 4) -> TriangleMultiplicationParams:
    return TriangleMultiplicationParams(
        layer_norm_in=LayerNormParams(jnp.ones(c_z), jnp.zeros(c_z)),
        layer_norm_out=LayerNormParams(jnp.ones(c_hidden), jnp.zeros(c_hidden)),
        linear_a_p=LinearParams(jnp.arange(c_hidden * c_z).reshape(c_hidden, c_z)),
        linear_a_g=LinearParams(jnp.ones((c_hidden, c_z))),
        linear_b_p=LinearParams(jnp.arange(c_hidden * c_z).reshape(c_hidden, c_z) + 10),
        linear_b_g=LinearParams(jnp.full((c_hidden, c_z), 2)),
        linear_z=LinearParams(jnp.ones((c_z, c_hidden))),
        linear_g=LinearParams(jnp.ones((c_z, c_z))),
    )


def test_cueq_triangle_maps_upstream_torch_weights(monkeypatch) -> None:
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
    params = _params()
    z = jnp.ones((3, 3, 4), dtype=jnp.bfloat16)
    mask = jnp.ones((3, 3), dtype=jnp.bfloat16)

    output = cueq_triangle_multiplication(z, mask, params, "incoming")

    assert jnp.array_equal(output, z)
    assert captured["x"].shape == (1, 3, 3, 4)
    assert captured["mask"].shape == (1, 3, 3)
    assert captured["direction"] == "incoming"
    assert captured["fallback"] is False
    assert jnp.array_equal(
        captured["p_in_weight"],
        jnp.concatenate((params.linear_a_p.weight, params.linear_b_p.weight)),
    )
    assert jnp.array_equal(
        captured["g_in_weight"],
        jnp.concatenate((params.linear_a_g.weight, params.linear_b_g.weight)),
    )
    assert captured["p_out_weight"] is params.linear_z.weight
    assert captured["g_out_weight"] is params.linear_g.weight


def test_triangle_multiplication_uses_cueq_by_default(monkeypatch) -> None:
    import foldjax.models.protenix.models.triangle.triangle_cueq as cueq_module

    sentinel = jnp.full((1, 2, 2, 32), 7, dtype=jnp.bfloat16)
    monkeypatch.delenv("PROTENIX_TRIANGLE_MULTIPLICATION_BACKEND", raising=False)
    monkeypatch.setattr(
        cueq_module,
        "cueq_triangle_multiplication",
        lambda *args, **kwargs: sentinel,
    )

    output = triangle_multiplication(
        jnp.ones((1, 2, 2, 32), dtype=jnp.bfloat16),
        jnp.ones((1, 2, 2), dtype=jnp.bfloat16),
        _params(32, 32),
        "outgoing",
    )

    assert output is sentinel


def test_cueq_attention_maps_torch_mask_and_scale(monkeypatch) -> None:
    captured = {}
    kernel_output = jnp.ones((1, 2, 1, 17, 4), dtype=jnp.bfloat16)
    q = kernel_output[0]

    def fake_triangle_attention(**kwargs):
        captured.update(kwargs)
        return kernel_output, jnp.zeros(1), jnp.zeros(1)

    monkeypatch.setitem(
        sys.modules,
        "cuequivariance_jax",
        SimpleNamespace(triangle_attention=fake_triangle_attention),
    )
    mask_bias = jnp.zeros((2, 1, 1, 17), dtype=jnp.float32)

    output = cueq_attention_core(
        q,
        q,
        q,
        jnp.zeros((1, 1, 17, 17)),
        mask_bias,
        scale=0.5,
    )

    assert output.shape == q.shape
    assert captured["q"].shape == kernel_output.shape
    assert captured["bias"].shape == (1, 1, 1, 17, 17)
    assert captured["scale"] == 0.5
    assert captured["precision"] == lax.Precision.DEFAULT
    assert jnp.array_equal(captured["mask"], (mask_bias == 0)[None])


def test_the_blocking_backend_is_the_default_triangle_attention_backend(
    monkeypatch,
) -> None:
    """cuEquivariance was the default until its cost was actually measured.

    It takes the whole tensor, so the row block that bounds the score never
    reaches it -- and it does not fuse the score away in exchange. At 490
    tokens cuEquivariance and the unblocked XLA path peak identically (6,048
    and 6,049 MiB) where the blocked XLA path peaks at 4,348, and at 976 the
    gap is 12,893 against 8,974. The default is now the one that blocks.
    """
    monkeypatch.delenv("PROTENIX_TRIANGLE_BACKEND", raising=False)
    assert _triangle_attention_backend() == "xla_jit"


def test_cueq_jit_is_valid_triangle_attention_backend(monkeypatch) -> None:
    monkeypatch.setenv("PROTENIX_TRIANGLE_BACKEND", "cueq_jit")
    assert _triangle_attention_backend() == "cueq_jit"


def test_cuda13_extra_installs_cueq_runtime() -> None:
    # FoldJAX carries the pure-JAX half of cuEq in its base requirements, since
    # every vendored port imports it, and keeps only the CUDA 13 ops build
    # behind the extra. Installing `cuda13` still yields the whole runtime.
    project = Path(__file__).resolve().parents[3] / "pyproject.toml"
    parsed = tomllib.loads(project.read_text(encoding="utf-8"))["project"]
    base = parsed["dependencies"]
    cuda13 = parsed["optional-dependencies"]["cuda13"]

    assert "cuequivariance-jax==0.10.0" in base + cuda13
    assert "cuequivariance-ops-jax-cu13==0.9.0" in cuda13
