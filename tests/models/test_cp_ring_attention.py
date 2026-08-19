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


@pytest.mark.parametrize("devices", [4, 9])
def test_ring_pads_an_indivisible_axis_without_gathering(devices: int) -> None:
    """Thirteen tokens split neither grid, and must still work.

    Refusing them would forfeit the contract the 1-D path already keeps, and
    real chains are the indivisible case far more often than not. The pad
    happens at the ``shard_map`` boundary on a global array, so the HLO
    assertion inside the probe is what shows it bought its evenness without a
    full-axis gather.
    """

    assert "RING_PARITY_AND_HLO_OK" in _run(_PREAMBLE, devices=devices, tokens=13)
# --- appended to tests/models/test_cp_ring_attention.py ---------------------
# Three properties the landed gates leave open.

_DISTINCT_AXES_PROBE = textwrap.dedent(
    r"""
    import os

    import jax
    import jax.numpy as jnp
    import numpy as np

    from foldjax.models._cp import context_parallel, cp_layout
    from foldjax.models._cp_attention import ring_triangle_attention_2d

    DEVICES = int(os.environ["FOLDJAX_CP_PROBE_DEVICES"])
    SIDE = int(round(DEVICES ** 0.5))
    assert jax.device_count() == DEVICES, jax.devices()

    rng = np.random.default_rng(20260819)
    BATCH, HEADS, DIM = 2, 3, 5
    # The landed probe gives the outer axis and the attended axis the same
    # extent, so a schedule that rotated one where it meant the other still
    # type-checks and still matches. Different extents separate them.
    OUTER, TOKENS = SIDE * 2, SIDE * 3
    assert OUTER != TOKENS

    def arr(*shape, scale=0.4):
        return jnp.asarray(rng.normal(size=shape, scale=scale), dtype=jnp.float32)

    q = arr(BATCH, OUTER, HEADS, TOKENS, DIM)
    k = arr(BATCH, OUTER, HEADS, TOKENS, DIM)
    v = arr(BATCH, OUTER, HEADS, TOKENS, DIM)
    bias = arr(BATCH, 1, HEADS, TOKENS, TOKENS, scale=7.0)
    keep = rng.random((BATCH, OUTER, TOKENS)) > 0.2
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
        got = jax.device_get(jax.jit(ring)(q, k, v, bias, mask))

    assert traced == ["2d"], traced
    np.testing.assert_allclose(reference, got, atol=3e-5, rtol=3e-5)
    print("DISTINCT_AXES_OK")
    """
)

_NO_MESH_PROBE = textwrap.dedent(
    r"""
    import jax.numpy as jnp

    from foldjax.models._cp import context_parallel
    from foldjax.models._cp_attention import ring_triangle_attention_2d

    q = jnp.zeros((1, 4, 2, 4, 3), dtype=jnp.float32)
    bias = jnp.zeros((1, 1, 2, 4, 4), dtype=jnp.float32)
    mask = jnp.zeros((1, 4, 1, 1, 4), dtype=jnp.float32)

    # No mesh at all.
    try:
        ring_triangle_attention_2d(q, q, q, bias, mask)
    except RuntimeError as error:
        assert "2-D" in str(error) or "2d" in str(error).lower(), error
    else:
        raise AssertionError("the ring ran with no mesh active")

    # A 1-D mesh has no column axis to rotate on. Falling back to dense
    # attention here would be silently correct and silently unsharded, which
    # is the failure the whole module exists to prevent.
    with context_parallel(4, layout="1d"):
        try:
            ring_triangle_attention_2d(q, q, q, bias, mask)
        except RuntimeError as error:
            assert "2-D" in str(error) or "2d" in str(error).lower(), error
        else:
            raise AssertionError("the ring ran under a 1-D layout")

    print("NO_MESH_GUARD_OK")
    """
)

_OUTPUT_SHARDING_PROBE = textwrap.dedent(
    r"""
    import os

    import jax
    import jax.numpy as jnp

    from foldjax.models._cp import context_parallel
    from foldjax.models._cp_attention import ring_triangle_attention_2d

    DEVICES = int(os.environ["FOLDJAX_CP_PROBE_DEVICES"])
    SIDE = int(round(DEVICES ** 0.5))
    OUTER = TOKENS = SIDE * 2

    q = jnp.zeros((1, OUTER, 2, TOKENS, 3), dtype=jnp.float32)
    bias = jnp.zeros((1, 1, 2, TOKENS, TOKENS), dtype=jnp.float32)
    mask = jnp.zeros((1, OUTER, 1, 1, TOKENS), dtype=jnp.float32)

    with context_parallel(DEVICES, layout="2d"):
        out = jax.jit(ring_triangle_attention_2d)(q, q, q, bias, mask)

    # Parity and a clean HLO still hold if every device ends up owning the
    # whole answer. The point of the schedule is that it does not, so the
    # result's own sharding is the property to assert.
    shards = out.sharding.shard_shape(out.shape)
    assert shards != out.shape, (shards, out.shape)
    per_device = 1
    for a, b in zip(out.shape, shards):
        per_device *= b
    whole = 1
    for a in out.shape:
        whole *= a
    assert per_device * DEVICES == whole, (shards, out.shape, DEVICES)
    print("OUTPUT_SHARDING_OK")
    """
)


def test_ring_distinguishes_the_outer_axis_from_the_attended_axis() -> None:
    """Equal extents let a swapped-axis schedule pass; unequal ones do not."""

    assert "DISTINCT_AXES_OK" in _run(_DISTINCT_AXES_PROBE, devices=9)


def test_ring_refuses_every_layout_it_cannot_rotate_on() -> None:
    """No mesh and a 1-D mesh must both raise, not fall back to dense."""

    assert "NO_MESH_GUARD_OK" in _run(_NO_MESH_PROBE, devices=4)


def test_ring_leaves_its_result_sharded() -> None:
    """A replicated answer would pass parity and the HLO gate alike."""

    assert "OUTPUT_SHARDING_OK" in _run(_OUTPUT_SHARDING_PROBE, devices=4)
