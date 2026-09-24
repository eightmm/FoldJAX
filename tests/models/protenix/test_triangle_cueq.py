from __future__ import annotations

import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
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


@pytest.mark.parametrize(
    "policy, expected", [("default", "DEFAULT"), ("high", "DEFAULT"),
                         ("highest", "DEFAULT")]
)
def test_cueq_triangle_maps_upstream_torch_weights(
    monkeypatch, policy, expected
) -> None:
    captured = {}

    def fake_triangle_multiplicative_update(**kwargs):
        captured.update(kwargs)
        return kwargs["x"]

    monkeypatch.setitem(
        sys.modules,
        "cuequivariance_jax",
        SimpleNamespace(
            TriMulPrecision=SimpleNamespace(DEFAULT="DEFAULT", IEEE="IEEE"),
            triangle_multiplicative_update=fake_triangle_multiplicative_update
        ),
    )
    params = _params()
    z = jnp.ones((3, 3, 4), dtype=jnp.bfloat16)
    mask = jnp.ones((3, 3), dtype=jnp.bfloat16)

    with jax.default_matmul_precision(policy):
        output = cueq_triangle_multiplication(z, mask, params, "incoming")

    assert jnp.array_equal(output, z)
    assert captured["x"].shape == (1, 3, 3, 4)
    assert captured["mask"].shape == (1, 3, 3)
    assert captured["direction"] == "incoming"
    assert captured["fallback"] is False
    assert captured["precision"] == expected
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


def test_bf16_high_cueq_gpu_preserves_nonzero_triangle_update():
    """Catch the real fused-kernel failure, not only the requested enum."""
    if jax.default_backend() != "gpu":
        pytest.skip("requires a JAX GPU and the cuEq runtime")
    pytest.importorskip("cuequivariance_jax")
    rng = np.random.default_rng(173)
    params = jax.tree.map(
        lambda value: jnp.asarray(
            rng.normal(0, 0.1, value.shape), dtype=jnp.bfloat16
        ),
        _params(32, 32),
    )
    z = jnp.asarray(rng.normal(size=(7, 7, 32)), dtype=jnp.bfloat16)
    mask = jnp.ones((7, 7), dtype=z.dtype)

    def run(z, params):
        return cueq_triangle_multiplication(z, mask, params, "outgoing")

    with jax.default_matmul_precision("default"):
        expected = np.asarray(jax.jit(run)(z, params), dtype=np.float32)
    with jax.default_matmul_precision("high"):
        actual = np.asarray(jax.jit(run)(z, params), dtype=np.float32)
    assert np.isfinite(expected).all() and np.count_nonzero(expected) > 0
    np.testing.assert_array_equal(actual, expected)


def test_triangle_multiplication_uses_cueq_by_default(monkeypatch) -> None:
    """No `auto`, no probe: the default is the fused kernel, unconditionally.

    This briefly resolved `auto` from a runtime availability probe, so a machine
    without the wheel silently ran a different kernel under the same default --
    two installs, two numerics, one configuration. If the kernel cannot load the
    import raises; it does not quietly become XLA.
    """
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


def test_the_blocked_path_is_still_reachable_by_name(monkeypatch) -> None:
    """A card the fused arena does not fit needs a way out, and it is explicit."""
    import foldjax.models.protenix.models.triangle.triangle_cueq as cueq_module

    sentinel = jnp.full((1, 2, 2, 32), 7, dtype=jnp.bfloat16)
    monkeypatch.setenv("PROTENIX_TRIANGLE_MULTIPLICATION_BACKEND", "xla")
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

    assert output is not sentinel


def test_an_unsupported_width_falls_back_even_when_cueq_is_available(
    monkeypatch,
) -> None:
    """The kernel needs `c_hidden == c_z`; Protenix's template stack does not have it.

    Upstream carries the same guard (`triangular.py:491`) and takes the same
    fallback, so this is a shape fact both sides agree on rather than a policy
    this port chose.
    """
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
        _params(32, 64),  # c_z=32, c_hidden=64
        "outgoing",
    )

    assert output is not sentinel


@pytest.mark.parametrize("on_cuda", [False, True])
def test_cueq_attention_maps_torch_mask_and_scale(monkeypatch, on_cuda) -> None:
    import foldjax.models._cueq as cueq_module

    calls = []
    q = jnp.ones((2, 1, 17, 4), dtype=jnp.bfloat16)

    def fake_triangle_attention(**kwargs):
        calls.append(kwargs)
        return jnp.ones_like(kwargs["q"]), jnp.zeros(1), jnp.zeros(1)

    monkeypatch.setitem(
        sys.modules,
        "cuequivariance_jax",
        SimpleNamespace(triangle_attention=fake_triangle_attention),
    )
    monkeypatch.setattr(cueq_module, "pads_attention_extents", lambda: on_cuda)
    mask_bias = jnp.zeros((2, 1, 1, 17), dtype=jnp.float32)

    output = cueq_attention_core(
        q,
        q,
        q,
        jnp.zeros((1, 1, 17, 17)),
        mask_bias,
        scale=0.5,
    )

    # On CUDA the kernel gets bf16 extents padded from 17 to 24 with the
    # padded keys invalid; elsewhere the arrays as they are. Either way the
    # output comes back at 17.
    (call,) = calls
    extent = 24 if on_cuda else 17
    assert output.shape == q.shape
    assert call["q"].shape == call["k"].shape == call["v"].shape == (1, 2, 1, extent, 4)
    assert call["bias"].shape == (1, 1, 1, extent, extent)
    assert jnp.array_equal(call["mask"][..., :17], (mask_bias == 0)[None])
    assert not call["mask"][..., 17:].any()
    assert call["scale"] == 0.5
    assert call["precision"] == lax.Precision.DEFAULT


@pytest.mark.parametrize(
    "dtype, extent, padded",
    [(jnp.bfloat16, 17, 24), (jnp.bfloat16, 24, 24), (jnp.float16, 9, 16),
     (jnp.float32, 17, 17)],
)
def test_attention_extents_are_aligned_for_half_precision_only(
    dtype, extent, padded
) -> None:
    from foldjax.models._cueq import align_attention_arguments

    arguments = {
        "q": jnp.ones((1, 3, 2, extent, 4), dtype),
        "k": jnp.ones((1, 3, 2, extent, 4), dtype),
        "v": jnp.ones((1, 3, 2, extent, 4), dtype),
        "bias": jnp.ones((1, 1, 2, extent, extent), jnp.float32),
        "mask": jnp.ones((1, 3, 1, 1, extent), bool),
    }
    aligned = align_attention_arguments(arguments)
    assert aligned["q"].shape[-2] == aligned["k"].shape[-2] == padded
    assert aligned["bias"].shape[-2:] == (padded, padded)
    assert aligned["mask"].shape[-1] == padded
    assert bool(aligned["mask"][..., :extent].all())
    assert not bool(aligned["mask"][..., extent:].any())
    if padded == extent:
        assert all(aligned[name] is arguments[name] for name in arguments)


def test_aligned_attention_keeps_every_valid_row_on_the_reference_kernel() -> None:
    """On a host without the CUDA kernel the wheel runs its reference body.

    The padded arguments the CUDA branch builds, run through that body, give
    every row with a valid key bit for bit; the fully masked row is the
    kernel's own convention and is not compared (the GPU paths differ on it
    anyway).
    """

    import foldjax.models._cueq as cueq_module

    rng = np.random.default_rng(0)
    n = 21
    q, k, v = (
        jnp.asarray(rng.normal(size=(1, 5, 2, n, 8)), jnp.bfloat16) for _ in range(3)
    )
    bias = jnp.asarray(rng.normal(size=(1, 1, 2, n, n)), jnp.float32)
    valid = rng.random((1, 5, 1, 1, n)) > 0.2
    valid[0, 0] = False
    mask = jnp.where(jnp.asarray(valid), 0.0, -1e9).astype(jnp.float32)
    plain = jax.jit(lambda *a: _reference_core(cueq_module, *a, align=False))(
        q, k, v, bias, mask
    )
    aligned = jax.jit(lambda *a: _reference_core(cueq_module, *a, align=True))(
        q, k, v, bias, mask
    )
    rows = valid[0, :, 0, 0].any(-1)
    assert aligned.shape == plain.shape == q.shape
    assert np.array_equal(np.asarray(aligned)[0][rows], np.asarray(plain)[0][rows])


def _reference_core(module, q, k, v, bias, mask, *, align):
    cuex = module.load_cueq()
    lead, arguments = module.cueq_attention_arguments(q, k, v, bias, mask)
    queries = arguments["q"].shape[-2]
    if align:
        arguments = module.align_attention_arguments(arguments)
    output, _, _ = cuex.triangle_attention(
        **arguments, scale=0.35, precision=lax.Precision.HIGHEST
    )
    output = output[..., :queries, :]
    return output.reshape((*lead, *output.shape[-4:]))


def test_the_fused_kernel_is_the_default_triangle_attention_backend(
    monkeypatch,
) -> None:
    """cuEquivariance, which is what upstream Protenix runs.

    This asserted `xla_jit`, on a 490-token measurement where the blocked XLA
    path peaked at 4,348 MiB against cuEquivariance's 6,048. That reading does
    not survive a real length: at 1,531 tokens the fused kernel is both faster
    and smaller -- 167.1 s / 22,639 MiB against 254.5 s / 24,764 -- because what
    grows is the `[rows, heads, N, N]` score tensor the blocked path writes to
    HBM and the fused one never builds. Blocking bounds that tensor; it does not
    stop paying for it. At 970 the same switch gives 98 -> 74.2 s.

    There is deliberately no `auto`: a probe that fell back to XLA when the
    wheel was missing would put two machines on two kernels under one default.
    """
    monkeypatch.delenv("PROTENIX_TRIANGLE_BACKEND", raising=False)
    assert _triangle_attention_backend() == "cueq_jit"


def test_the_blocked_attention_path_is_still_reachable_by_name(monkeypatch) -> None:
    """The escape hatch for a card whose arena the fused kernel overflows."""
    monkeypatch.setenv("PROTENIX_TRIANGLE_BACKEND", "xla_jit")
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

    assert "cuequivariance==0.11.1" in base + cuda13
    assert "cuequivariance-jax==0.11.1" in base + cuda13
    assert "cuequivariance-ops-jax-cu13==0.11.1" in cuda13
