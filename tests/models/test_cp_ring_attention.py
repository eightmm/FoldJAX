"""Proof gates for the gather-free two-dimensional Fold-CP attention ring."""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

from tests.models.cp_probe_env import inherited_environment

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
        **inherited_environment(),
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


# --- the local tile is evaluated in query blocks ----------------------------
# One ring step used to build f32[..., rows, heads, N/s, N/s] whole, which is
# cubic in the local token extent: 17.15 GiB per device for Boltz-2's 2,096
# tokens on a 2x2 mesh, about three of them co-live. The blocks bound that
# without touching the rotations.

_QUERY_BLOCK_PROBE = textwrap.dedent(
    r"""
    import functools
    import os
    import re

    import jax
    import jax.numpy as jnp
    import numpy as np

    from foldjax.models._cp import context_parallel
    from foldjax.models._cp_attention import ring_triangle_attention_2d

    DEVICES = int(os.environ["FOLDJAX_CP_PROBE_DEVICES"])
    SIDE = int(round(DEVICES ** 0.5))
    assert SIDE * SIDE == DEVICES
    assert jax.device_count() == DEVICES, jax.devices()

    rng = np.random.default_rng(20260915)
    BATCH, HEADS, DIM = 2, 3, 5
    # Every extent differs: the outer axis is not the token axis, the channel
    # count is neither, and the four local query rows coincide with no block
    # size used below. A schedule that rotated one axis where it meant
    # another, or a dot that kept the full-width rows, cannot hide.
    OUTER, TOKENS = 2 * SIDE, 4 * SIDE
    LOCAL = TOKENS // SIDE
    assert LOCAL == 4 and DIM != LOCAL

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
    DOT = re.compile(r"\[([0-9,]+)\](?:\{[^}]*\})?\s*dot\(")

    def run(**kwargs):
        # A fresh closure per arm. Jitting one closure twice would reuse the
        # traced program and make every comparison below vacuous.
        fn = jax.jit(functools.partial(ring_triangle_attention_2d, **kwargs))
        with context_parallel(DEVICES, layout="2d"):
            out = jax.device_get(fn(q, k, v, bias, mask))
            hlo = (
                fn.lower(q, k, v, bias, mask)
                .compiler_ir(dialect="hlo")
                .as_hlo_text()
            )
        shapes = [tuple(int(e) for e in s.split(",")) for s in DOT.findall(hlo)]
        # Inside the `shard_map` body the dots carry local shapes: the score
        # dot ends in the local key extent, the value dot in the channels.
        assert shapes, hlo
        assert all(s[-1] in (LOCAL, DIM) for s in shapes), shapes
        return {
            "out": out,
            "rows": sorted({s[-2] for s in shapes}),
            "largest": max(int(np.prod(s)) for s in shapes),
            "permutes": hlo.lower().count("collective-permute"),
        }

    whole = run()
    off = run(q_block=0)
    even = run(q_block=2)
    ragged = run(q_block=3)

    # The default clamps to the local axis here and the explicit off covers
    # it, so both take the unblocked branch -- a trace-time Python `if`, hence
    # the same HLO and an exactly equal result.
    np.testing.assert_array_equal(whole["out"], off["out"])
    assert whole["rows"] == off["rows"] == [LOCAL], (whole["rows"], off["rows"])
    assert whole["largest"] == off["largest"]

    # Positive control for the two assertions after it: the unblocked arm
    # really does emit a full-width dot, so a blocked arm without one is a
    # change rather than a shape that was never there.
    assert even["rows"] == [2], even["rows"]
    assert ragged["rows"] == [1, 3], ragged["rows"]

    # The dot output that dominates per-device score memory shrinks exactly
    # with the block.
    assert even["largest"] * LOCAL == whole["largest"] * 2, (even, whole)
    assert ragged["largest"] * LOCAL == whole["largest"] * 3, (ragged, whole)

    # Same rotations, same collectives: the blocks live inside a ring step.
    assert whole["permutes"] > 0
    for arm in (off, even, ragged):
        assert arm["permutes"] == whole["permutes"], (arm, whole)

    for arm in (whole, off, even, ragged):
        np.testing.assert_allclose(reference, arm["out"], atol=3e-5, rtol=3e-5)
    # Blocking a free axis is exact arithmetic -- every query row still
    # reduces over the same whole local key axis -- but not bitwise: XLA picks
    # its contraction schedule from the operand extents. Measured on this
    # fixture: 0.0 for the even split, 0.0 for the ragged one; a 16-row local
    # axis split 3+3+3+3+3+1 moved by 1.2e-07. A closed tolerance is what the
    # backend actually promises.
    deviations = []
    for arm in (even, ragged):
        np.testing.assert_allclose(whole["out"], arm["out"], atol=1e-6, rtol=1e-6)
        deviations.append(float(np.max(np.abs(whole["out"] - arm["out"]))))
    print("QUERY_BLOCK_DEVIATION", deviations)
    print("QUERY_BLOCK_OK")
    """
)


@pytest.mark.parametrize("devices", [4, 9])
def test_ring_query_blocks_keep_the_tile_and_the_schedule(devices: int) -> None:
    """Blocked tiles, unchanged collectives, and no full-width score dot."""

    assert "QUERY_BLOCK_OK" in _run(_QUERY_BLOCK_PROBE, devices=devices)


def test_ring_query_block_default_fires_on_the_local_axis() -> None:
    """The default has to bite at the size the 2-D layout exists for.

    Boltz-2's 2,096 tokens leave 1,048 local query rows on a 2x2 mesh. The
    serial resolver's gate -- 512 rows once the sequence passes 2,048 tokens
    -- reads the global count, so reusing it verbatim on the local axis would
    never fire there and would leave the cubic tile in place.
    """

    from foldjax.models._cp_attention import (
        RING_QUERY_BLOCK,
        resolve_ring_query_block,
    )

    assert RING_QUERY_BLOCK == 512
    assert resolve_ring_query_block(1048) == 512
    assert resolve_ring_query_block(1048, None) == 512
    assert resolve_ring_query_block(1048, 128) == 128
    # Clamped to a short axis; non-positive or covering means one block.
    assert resolve_ring_query_block(4) == 4
    assert resolve_ring_query_block(4, 0) == 4
    assert resolve_ring_query_block(4, -1) == 4
    assert resolve_ring_query_block(4, 4) == 4
    assert resolve_ring_query_block(4, 9) == 4
    assert resolve_ring_query_block(4, 3) == 3


_PASS_THROUGH_PROBE = textwrap.dedent(
    r"""
    import functools
    import os
    import re

    import jax
    import jax.numpy as jnp
    import numpy as np

    from foldjax.models._cp import context_parallel

    DEVICES = int(os.environ["FOLDJAX_CP_PROBE_DEVICES"])
    SIDE = int(round(DEVICES ** 0.5))
    assert jax.device_count() == DEVICES, jax.devices()

    C, HEADS = 8, 2
    TOKENS = 4 * SIDE
    LOCAL = TOKENS // SIDE
    rng = np.random.default_rng(20260915)
    DOT = re.compile(r"\[([0-9,]+)\](?:\{[^}]*\})?\s*dot\(")

    def arr(*shape):
        return jnp.asarray(rng.normal(size=shape, scale=0.5), dtype=jnp.float32)

    def query_rows(fn, *args):
        # Query-axis extents of the attention dots in one lowering. The
        # projections are rank-3 dots; only the attention tiles carry a head
        # axis, so rank four is the filter, and their query axis is the extent
        # the block is supposed to set.
        with context_parallel(DEVICES, layout="2d"):
            hlo = jax.jit(fn).lower(*args).compiler_ir(dialect="hlo").as_hlo_text()
        shapes = [tuple(int(e) for e in s.split(",")) for s in DOT.findall(hlo)]
        wide = [s for s in shapes if len(s) >= 4]
        assert wide, shapes
        return sorted({s[-2] for s in wide})

    def norm(width):
        return {"scale": arr(width) * 0.1 + 1.0, "bias": arr(width) * 0.1}

    def weight(fan_in, fan_out):
        return arr(fan_in, fan_out) * (1.0 / (0.5 * np.sqrt(fan_in)))

    boltz_params = {
        "layer_norm": norm(C),
        "linear": {"kernel": weight(C, HEADS)},
        "mha": {
            "linear_q": {"kernel": weight(C, C)},
            "linear_k": {"kernel": weight(C, C)},
            "linear_v": {"kernel": weight(C, C)},
            "linear_g": {"kernel": weight(C, C)},
            "linear_o": {"kernel": weight(C, C)},
        },
    }
    from foldjax.models.boltz2.models.triangle import (
        triangle_attention as boltz_entry,
    )
    from foldjax.models.boltz2.models.triangle import (
        triangle_attention_cp as boltz_cp_entry,
    )

    from foldjax.models.protenix.models.primitives.primitives import (
        LayerNormParams,
        LinearParams,
    )
    from foldjax.models.protenix.models.triangle import triangle as protenix_entry
    from foldjax.models.protenix.models.triangle import (
        triangle_attention_cp as protenix_cp_entry,
    )
    from foldjax.models.protenix.models.triangle.triangle import (
        AttentionParams,
        TriangleAttentionParams,
    )

    def protenix_linear(out_channels, in_channels):
        return LinearParams(
            weight=arr(out_channels, in_channels),
            bias=arr(out_channels),
        )

    def protenix_norm(width):
        return LayerNormParams(weight=arr(width) * 0.1 + 1.0, bias=arr(width) * 0.1)

    protenix_params = TriangleAttentionParams(
        layer_norm=protenix_norm(C),
        linear=protenix_linear(HEADS, C),
        attention=AttentionParams(
            linear_q=protenix_linear(C, C),
            linear_k=protenix_linear(C, C),
            linear_v=protenix_linear(C, C),
            linear_o=protenix_linear(C, C),
            linear_g=protenix_linear(C, C),
        ),
    )

    boltz_pair = arr(1, TOKENS, TOKENS, C)
    boltz_mask = jnp.ones((1, TOKENS, TOKENS), dtype=jnp.float32)
    protenix_pair = arr(TOKENS, TOKENS, C)
    protenix_mask = jnp.ones((TOKENS, TOKENS), dtype=jnp.float32)

    # All four routes that reach the ring: each port's serial module (which
    # dispatches to its own CP branch under a mesh) and each port's CP
    # dispatch module. A dropped keyword anywhere is a silent return to the
    # full-width tile, so every one is checked rather than labelled.
    arms = []
    for entry in (boltz_entry, boltz_cp_entry):
        arms.append(
            lambda block, entry=entry: query_rows(
                functools.partial(
                    entry.triangle_attention_forward,
                    boltz_params,
                    q_chunk_size=block,
                ),
                boltz_pair,
                boltz_mask,
            )
        )
    for entry in (protenix_entry, protenix_cp_entry):
        arms.append(
            lambda block, entry=entry: query_rows(
                functools.partial(
                    entry.triangle_attention,
                    num_heads=HEADS,
                    q_chunk_size=block,
                ),
                protenix_pair,
                protenix_mask,
                protenix_params,
            )
        )

    for index, arm in enumerate(arms):
        off = arm(0)
        blocked = arm(2)
        assert LOCAL in off, (index, off, LOCAL)
        assert 2 in blocked, (index, blocked)
        assert LOCAL not in blocked, (index, blocked, LOCAL)
    print("PASS_THROUGH_OK")
    """
)


def test_port_query_chunks_reach_the_ring() -> None:
    """Each port's ``q_chunk_size`` must change the ring's lowering.

    The 2-D path used to document the chunk as deliberately unused, so a
    pass-through that silently stopped at the dispatch boundary would look
    exactly like the old, intended behaviour.
    """

    assert "PASS_THROUGH_OK" in _run(_PASS_THROUGH_PROBE, devices=4)
