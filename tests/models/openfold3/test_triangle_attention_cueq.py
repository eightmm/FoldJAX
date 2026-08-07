"""The fused triangle-attention path, checked without a GPU.

The kernel itself needs CUDA, but the thing most likely to be wrong does not: the
layouts handed to it. ``cuequivariance_jax.triangle_attention`` wants
``bias`` as ``[B, 1, H, S, S]`` -- row axis 1, broadcast across rows -- and a
per-row bias of ``[B, N, H, S, S]`` would typecheck, run, and quietly cost more
memory than the XLA path it replaced, which is the entire reason for using it.

So these monkeypatch the kernel and assert what it was called with. The numerical
comparison against the XLA path belongs in a GPU run, not here.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.openfold3.models.attention import AttentionParams
from foldjax.models.openfold3.models.primitives import LayerNormParams, LinearParams
from foldjax.models.openfold3.models.triangle_attention import (
    TriangleAttentionParams,
    _default_backend,
    triangle_attention,
)

N, C, HEADS, C_HIDDEN = 11, 8, 2, 4


def _params(seed: int = 0) -> TriangleAttentionParams:
    generator = np.random.default_rng(seed)

    def array(*shape):
        return jnp.asarray(generator.normal(size=shape) * 0.3, dtype=jnp.float32)

    return TriangleAttentionParams(
        layer_norm=LayerNormParams(weight=array(C), bias=array(C)),
        linear_z=LinearParams(weight=array(HEADS, C), bias=None),
        mha=AttentionParams(
            linear_q=LinearParams(weight=array(HEADS * C_HIDDEN, C), bias=None),
            linear_k=LinearParams(weight=array(HEADS * C_HIDDEN, C), bias=None),
            linear_v=LinearParams(weight=array(HEADS * C_HIDDEN, C), bias=None),
            linear_o=LinearParams(weight=array(C, HEADS * C_HIDDEN), bias=None),
            linear_g=LinearParams(weight=array(HEADS * C_HIDDEN, C), bias=None),
        ),
    )


def _fake_cueq(captured: dict):
    class _Fake:
        @staticmethod
        def triangle_attention(*, q, k, v, bias, mask, scale, precision):
            captured.update(
                q=q, k=k, v=v, bias=bias, mask=mask, scale=scale
            )
            return jnp.zeros_like(q), None, None

    return _Fake()


@pytest.fixture
def captured(monkeypatch):
    seen: dict = {}
    monkeypatch.setattr(
        "foldjax.models._cueq.load_cueq", lambda: _fake_cueq(seen)
    )
    return seen


def test_the_default_backend_is_upstream_s(monkeypatch) -> None:
    """Upstream's ``use_cueq_triangle_kernels`` is ``False``, so this is ``xla``.

    A port that silently ran a different kernel than its upstream's default would
    be reporting numbers for a configuration upstream never runs.
    """
    monkeypatch.delenv("OPENFOLD3_TRIANGLE_BACKEND", raising=False)
    assert _default_backend() == "xla"
    monkeypatch.setenv("OPENFOLD3_TRIANGLE_BACKEND", "CUEQ")
    assert _default_backend() == "cueq", "the value is case-insensitive"


def test_an_unknown_backend_is_refused() -> None:
    x = jnp.zeros((1, N, N, C))
    with pytest.raises(ValueError, match="unsupported triangle attention backend"):
        triangle_attention(x, _params(), no_heads=HEADS, backend="flash")


def test_the_kernel_gets_the_layouts_it_documents(captured) -> None:
    generator = np.random.default_rng(1)
    x = jnp.asarray(generator.normal(size=(1, N, N, C)), dtype=jnp.float32)
    mask = jnp.asarray((generator.random((1, N, N)) > 0.2).astype(np.float32))

    triangle_attention(x, _params(), no_heads=HEADS, mask=mask, backend="cueq")

    # [B, N_row, H, N_col, D]
    assert captured["q"].shape == (1, N, HEADS, N, C_HIDDEN)
    assert captured["k"].shape == captured["q"].shape
    assert captured["v"].shape == captured["q"].shape
    # [B, 1, H, S_qo, S_kv]: the 1 is what makes this cheaper than the XLA path.
    assert captured["bias"].shape == (1, 1, HEADS, N, N)
    # [B, N_row, 1, 1, S_kv], boolean, True meaning valid.
    assert captured["mask"].shape == (1, N, 1, 1, N)
    assert captured["mask"].dtype == jnp.bool_


def test_the_mask_marks_valid_positions_not_masked_ones(captured) -> None:
    """The kernel's convention is inverted from the additive bias it comes from.

    ``mask_bias`` is ``inf * (mask - 1)``: zero where the pair is real and a large
    negative where it is not. The kernel wants True where the pair is real. Getting
    this backwards attends to exactly the wrong positions and raises no error.
    """
    generator = np.random.default_rng(2)
    x = jnp.asarray(generator.normal(size=(1, N, N, C)), dtype=jnp.float32)
    mask = np.ones((1, N, N), dtype=np.float32)
    mask[:, :, 3] = 0.0

    triangle_attention(
        x, _params(), no_heads=HEADS, mask=jnp.asarray(mask), backend="cueq"
    )

    kernel_mask = np.asarray(captured["mask"])[0, :, 0, 0, :]
    assert not kernel_mask[:, 3].any(), "column 3 was masked out and must be False"
    assert kernel_mask[:, 0].all(), "column 0 is valid and must be True"


def test_scaling_happens_once(captured) -> None:
    """The XLA path divides the queries; the kernel takes a ``scale``.

    Doing both would scale twice, which softmax does not forgive.
    """
    from foldjax.models.openfold3.models.attention import split_heads
    from foldjax.models.openfold3.models.primitives import layer_norm, linear

    generator = np.random.default_rng(3)
    x = jnp.asarray(generator.normal(size=(1, N, N, C)), dtype=jnp.float32)
    params = _params()
    triangle_attention(x, params, no_heads=HEADS, backend="cueq")

    assert captured["scale"] == pytest.approx(C_HIDDEN**-0.5)
    # The queries reach the kernel as the bare projection -- no sqrt(D) division,
    # which `attention` would have applied and which would then happen twice.
    normed = layer_norm(x, params.layer_norm, eps=1e-5)
    expected = jnp.swapaxes(
        split_heads(linear(normed, params.mha.linear_q), HEADS), -2, -3
    )
    np.testing.assert_allclose(
        np.asarray(captured["q"]), np.asarray(expected), rtol=1e-6, atol=1e-6
    )
