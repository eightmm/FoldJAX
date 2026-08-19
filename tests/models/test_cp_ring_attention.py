"""Proof gates for the gather-free two-dimensional Fold-CP attention ring."""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

_PREAMBLE = textwrap.dedent(
    r"""
    import os

    import jax
    import jax.numpy as jnp
    import numpy as np

    from foldjax.models._cp import context_parallel, cp_layout
    from foldjax.models._cp_attention import ring_triangle_attention_2d

    DEVICES = int(os.environ["FOLDJAX_CP_PROBE_DEVICES"])
    SIDE = int(round(DEVICES ** 0.5))
    assert SIDE * SIDE == DEVICES
    assert jax.device_count() == DEVICES, jax.devices()

    rng = np.random.default_rng(20260819)
    BATCH, HEADS, DIM = 2, 3, 5
    N = int(os.environ["FOLDJAX_CP_PROBE_TOKENS"])

    def arr(*shape, scale=0.4):
        return jnp.asarray(rng.normal(size=shape, scale=scale), dtype=jnp.float32)

    q = arr(BATCH, N, HEADS, N, DIM)
    k = arr(BATCH, N, HEADS, N, DIM)
    v = arr(BATCH, N, HEADS, N, DIM)
    # Large positive/negative values exercise online-softmax rescaling rather
    # than merely comparing a benign near-uniform distribution.
    bias = arr(BATCH, 1, HEADS, N, N, scale=7.0)
    keep = rng.random((BATCH, N, N)) > 0.2
    # Every row retains at least one key; padded/all-masked-row behaviour is a
    # model-level finite-mask contract, not an excuse for a vacuous attention.
    keep[..., 0] = True
    mask = jnp.where(
        jnp.asarray(keep)[:, :, None, None, :],
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.asarray(-1.0e9, dtype=jnp.float32),
    )

    def dense(q_in, k_in, v_in, b_in, m_in):
        scores = jnp.einsum("...hqd,...hkd->...hqk", q_in, k_in)
        scores = scores + b_in + m_in
        probs = jax.nn.softmax(scores.astype(jnp.float32), axis=-1)
        return jnp.einsum("...hqk,...hkd->...hqd", probs, v_in)

    reference = jax.device_get(jax.jit(dense)(q, k, v, bias, mask))
    traced = []

    def ring(q_in, k_in, v_in, b_in, m_in):
        traced.append(cp_layout())
        return ring_triangle_attention_2d(q_in, k_in, v_in, b_in, m_in)

    with context_parallel(DEVICES, layout="2d"):
        compiled = jax.jit(ring)
        result = compiled(q, k, v, bias, mask)
        got = jax.device_get(result)
        lowered = compiled.lower(q, k, v, bias, mask)
        hlo = lowered.compiler_ir(dialect="hlo").as_hlo_text().lower()

    assert traced == ["2d"], traced
    np.testing.assert_allclose(reference, got, atol=3e-5, rtol=3e-5)
    collective_permute = "collective-permute" in hlo or "collective_permute" in hlo
    assert collective_permute, hlo
    assert "all-gather" not in hlo and "all_gather" not in hlo, hlo
    print("RING_PARITY_AND_HLO_OK")
    """
)

_DIVISIBILITY_PROBE = textwrap.dedent(
    r"""
    import jax
    import jax.numpy as jnp

    from foldjax.models._cp import context_parallel
    from foldjax.models._cp_attention import ring_triangle_attention_2d

    n = 13
    q = jnp.zeros((1, n, 2, n, 4), dtype=jnp.float32)
    bias = jnp.zeros((1, 1, 2, n, n), dtype=jnp.float32)
    mask = jnp.zeros((1, n, 1, 1, n), dtype=jnp.float32)
    with context_parallel(4, layout="2d"):
        try:
            ring_triangle_attention_2d(q, q, q, bias, mask)
        except ValueError as error:
            message = str(error)
            assert "Pad semantic" in message and "divisible" in message, message
        else:
            raise AssertionError("non-divisible 2-D attention was accepted")
    print("DIVISIBILITY_GUARD_OK")
    """
)


def _run(source: str, *, devices: int, tokens: int | None = None) -> str:
    env = {
        "JAX_PLATFORMS": "cpu",
        "XLA_FLAGS": f"--xla_force_host_platform_device_count={devices}",
        "FOLDJAX_CP_PROBE_DEVICES": str(devices),
        "PATH": "/usr/bin",
    }
    if tokens is not None:
        env["FOLDJAX_CP_PROBE_TOKENS"] = str(tokens)
    completed = subprocess.run(
        [sys.executable, "-c", source],
        capture_output=True,
        text=True,
        env=env,
        timeout=180,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return completed.stdout


@pytest.mark.parametrize(("devices", "tokens"), [(4, 8), (9, 9)])
def test_ring_triangle_attention_matches_dense_without_all_gather(
    devices: int,
    tokens: int,
) -> None:
    """2x2 covers the common topology; 3x3 pins every shift direction."""

    assert "RING_PARITY_AND_HLO_OK" in _run(
        _PREAMBLE,
        devices=devices,
        tokens=tokens,
    )


def test_ring_refuses_hidden_pad_gather_fallback() -> None:
    """Uneven shards must be fixed at the data boundary, never in the hot path."""

    assert "DIVISIBILITY_GUARD_OK" in _run(_DIVISIBILITY_PROBE, devices=4)
