"""Torch-vs-JAX parity gate for ported primitives (fp32, rtol/atol 1e-4).

Each primitive is gated against torch's own op (the authoritative reference for
the math recovered from the TorchScript graph).
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import jax.numpy as jnp  # noqa: E402

from foldjax.models.chai.models import primitives as prim  # noqa: E402


def _diff(a, b) -> float:
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def test_linear_fp32() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2, 5, 7)).astype(np.float32)
    w = rng.standard_normal((11, 7)).astype(np.float32)
    b = rng.standard_normal((11,)).astype(np.float32)
    ref = torch.nn.functional.linear(
        torch.from_numpy(x), torch.from_numpy(w), torch.from_numpy(b)
    ).numpy()
    out = np.asarray(prim.linear(x, w, b))
    assert _diff(out, ref) < 1e-4


def test_layer_norm() -> None:
    rng = np.random.default_rng(1)
    x = rng.standard_normal((3, 8)).astype(np.float32)
    w = rng.standard_normal((8,)).astype(np.float32)
    b = rng.standard_normal((8,)).astype(np.float32)
    ref = torch.nn.functional.layer_norm(
        torch.from_numpy(x), (8,), torch.from_numpy(w), torch.from_numpy(b), eps=1e-5
    ).numpy()
    out = np.asarray(prim.layer_norm(x, w, b, eps=1e-5))
    assert _diff(out, ref) < 1e-4


def test_embedding() -> None:
    rng = np.random.default_rng(2)
    w = rng.standard_normal((33, 32)).astype(np.float32)
    idx = rng.integers(0, 33, size=(4, 6)).astype(np.int64)
    ref = torch.embedding(torch.from_numpy(w), torch.from_numpy(idx)).numpy()
    out = np.asarray(prim.embedding(w, idx))
    assert _diff(out, ref) < 1e-4


def test_one_hot() -> None:
    idx = np.array([[0, 2], [1, 4]], dtype=np.int64)
    ref = torch.nn.functional.one_hot(torch.from_numpy(idx), 5).float().numpy()
    out = np.asarray(prim.one_hot(idx, 5))
    assert _diff(out, ref) < 1e-4


def test_rbf_restraint_encoding() -> None:
    # Reproduce the exported graph op directly in torch as the reference.
    radii = np.array([6.0, 10.8, 15.6, 20.4, 25.2, 30.0], dtype=np.float32)
    width = 4.8
    raw = np.array([[6.0, 12.0, -1.0]], dtype=np.float32)  # (1, 3)

    rt = torch.from_numpy(raw).unsqueeze(-1)  # (...,1)
    rr = torch.from_numpy(radii).view(1, 1, -1)
    e = torch.clamp_max(((rr - rt) / width) ** 2, 16.0)
    enc = torch.exp(-e)
    enc[e == 16.0] = 0.0
    mask = (rt == -1.0).float()
    enc = enc * (1 - mask)
    ref = torch.cat([enc, mask], dim=-1).numpy()

    out = np.asarray(
        prim.rbf_restraint_encoding(raw, jnp.asarray(radii), width=width)
    )
    assert out.shape == ref.shape
    assert _diff(out, ref) < 1e-4
