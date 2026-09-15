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

_ROW_BLOCK_PROBE = textwrap.dedent(
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
    # count is neither, and the local pair rows coincide with no block size
    # used below. A schedule that rotated one axis where it meant another, or
    # a dot that kept the full-width rows, cannot hide.
    OUTER, TOKENS = 8 * SIDE, 4 * SIDE
    LOCAL = TOKENS // SIDE
    ROWS = OUTER // SIDE
    assert LOCAL == 4 and DIM != LOCAL and ROWS == 8

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
        # dot ends in the local key extent, the value dot in the channels, and
        # both carry the block's pair rows at -4 and the heads at -3.
        assert shapes, hlo
        assert all(s[-1] in (LOCAL, DIM) for s in shapes), shapes
        assert all(s[-3] == HEADS and s[-2] == LOCAL for s in shapes), shapes
        return {
            "out": out,
            "rows": sorted({s[-4] for s in shapes}),
            "largest": max(int(np.prod(s)) for s in shapes),
            "permutes": hlo.lower().count("collective-permute"),
        }

    def block_sizes(block):
        full, remainder = divmod(ROWS, block)
        return sorted({block} if not remainder else {block, remainder})

    whole = run()
    off = run(q_block=0)
    even = run(q_block=2)
    ragged = run(q_block=3)

    # The rule leaves eight local rows whole here and the explicit off covers
    # it, so both take the unblocked branch -- a trace-time Python `if`, hence
    # the same HLO and an exactly equal result.
    np.testing.assert_array_equal(whole["out"], off["out"])
    assert whole["rows"] == off["rows"] == [ROWS], (whole["rows"], off["rows"])
    assert whole["largest"] == off["largest"]

    # Positive control for the two assertions after it: the unblocked arm
    # really does emit a full-row dot, so a blocked arm without one is a
    # change rather than a shape that was never there.
    assert even["rows"] == block_sizes(2) == [2], even["rows"]
    assert ragged["rows"] == block_sizes(3) == [2, 3], ragged["rows"]
    assert ROWS not in even["rows"] and ROWS not in ragged["rows"]

    # The dot output that dominates per-device score memory shrinks exactly
    # with the block -- the largest block, where the tail is ragged.
    assert even["largest"] * ROWS == whole["largest"] * 2, (even, whole)
    assert ragged["largest"] * ROWS == whole["largest"] * 3, (ragged, whole)

    # The rotation schedule per block is the unblocked ring's, minus the two
    # initial bias skews, which are loop-invariant and hoisted above the row
    # loop; a blocked program emits that body twice, for the peeled block and
    # for the scan.
    assert whole["permutes"] > 0
    for arm in (even, ragged):
        assert arm["permutes"] == 2 + 2 * (whole["permutes"] - 2), (arm, whole)
    assert off["permutes"] == whole["permutes"], (off, whole)

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
    print("ROW_BLOCK_DEVIATION", deviations)
    print("ROW_BLOCK_OK")
    """
)


@pytest.mark.parametrize("devices", [4, 9])
def test_ring_row_blocks_keep_the_tile_and_the_schedule(devices: int) -> None:
    """Blocked rows, one rotation schedule per block, no full-row score dot."""

    assert "ROW_BLOCK_OK" in _run(_ROW_BLOCK_PROBE, devices=devices)


def test_ring_row_block_follows_the_serial_rule() -> None:
    """The ring's block is the serial path's rule on the local row axis.

    Two things this pins. The blocked axis is the pair row, so what bounds one
    block is ``rows * heads`` -- the serial sweep's invariant, 64 rows at 4
    heads and 24 at 12 -- and not a row count that would let the score tile
    grow with the width. And the byte ceiling still narrows it on a very wide
    local tile, which is the case a square mesh is chosen for.
    """

    from foldjax.models._cp_attention import (
        RING_MAX_ROWS_PER_BLOCK,
        RING_MIN_ROWS_PER_BLOCK,
        RING_SCORE_CEILING_BYTES,
        RING_SCORE_ROWS_TIMES_HEADS,
        resolve_ring_row_block,
    )

    assert RING_SCORE_ROWS_TIMES_HEADS == 288

    def block(rows, heads, keys, requested=None):
        return resolve_ring_row_block(
            rows, heads=heads, local_keys=keys, requested=requested
        )

    # `rows * heads`, capped, and independent of the width until the ceiling
    # bites: a 2x2 mesh at 2,096 tokens leaves 1,048 local rows.
    assert block(1048, 4, 1048) == RING_MAX_ROWS_PER_BLOCK == 64
    assert block(1048, 12, 1048) == RING_SCORE_ROWS_TIMES_HEADS // 12 == 24
    assert block(1048, 12, 512) == 24
    # 12 heads at 3,072 local keys cost 452 MiB per row, so the ceiling does.
    assert block(3072, 12, 3072) == RING_SCORE_CEILING_BYTES // (12 * 3072**2 * 4)
    assert block(3072, 12, 3072) < 24
    assert block(3072, 12, 1 << 20) == RING_MIN_ROWS_PER_BLOCK == 8
    # The caller's knob narrows, never widens; non-positive means one block.
    assert block(1048, 4, 1048, 16) == 16
    assert block(1048, 4, 1048, 4096) == 64
    assert block(1048, 4, 1048, 0) == 1048
    assert block(1048, 4, 1048, -1) == 1048
    # Clamped to a short axis, which is what leaves small fixtures unblocked.
    assert block(8, 3, 4) == 8
    assert block(8, 3, 4, 3) == 3
    assert block(1, 3, 4) == 1


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

    # The head count divides the channels and equals neither the local
    # extents nor the block, so a projection dot cannot be read as an
    # attention dot below.
    C, HEADS = 9, 3
    TOKENS = 4 * SIDE
    LOCAL = TOKENS // SIDE
    rng = np.random.default_rng(20260915)
    DOT = re.compile(r"\[([0-9,]+)\](?:\{[^}]*\})?\s*dot\(")

    def arr(*shape):
        return jnp.asarray(rng.normal(size=shape, scale=0.5), dtype=jnp.float32)

    def block_rows(fn, *args):
        # Pair-row extents of the attention dots in one lowering. Only the
        # attention tiles carry the head axis at -3 above the local key extent
        # at -2; the projections, which now happen inside the block, do not.
        # The row axis at -4 is the extent the block is supposed to set.
        with context_parallel(DEVICES, layout="2d"):
            hlo = jax.jit(fn).lower(*args).compiler_ir(dialect="hlo").as_hlo_text()
        shapes = [tuple(int(e) for e in s.split(",")) for s in DOT.findall(hlo)]
        wide = [
            s
            for s in shapes
            if len(s) >= 4 and s[-3] == HEADS and s[-2] == LOCAL
        ]
        assert wide, shapes
        return sorted({s[-4] for s in wide})

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

    from foldjax.models.openfold3.models import (
        triangle_attention as of3_entry,
    )
    from foldjax.models.openfold3.models import (
        triangle_attention_cp as of3_cp_entry,
    )
    from foldjax.models.openfold3.models.attention import (
        AttentionParams as Of3AttentionParams,
    )
    from foldjax.models.openfold3.models.primitives import (
        LayerNormParams as Of3LayerNormParams,
    )
    from foldjax.models.openfold3.models.primitives import (
        LinearParams as Of3LinearParams,
    )
    from foldjax.models.openfold3.models.triangle_attention import (
        TriangleAttentionParams as Of3TriangleAttentionParams,
    )

    def of3_linear(out_channels, in_channels):
        # Upstream's projections are bias-free; the weight is [out, in].
        return Of3LinearParams(weight=arr(out_channels, in_channels), bias=None)

    of3_params = Of3TriangleAttentionParams(
        layer_norm=Of3LayerNormParams(
            weight=arr(C) * 0.1 + 1.0,
            bias=arr(C) * 0.1,
        ),
        linear_z=of3_linear(HEADS, C),
        mha=Of3AttentionParams(
            linear_q=of3_linear(C, C),
            linear_k=of3_linear(C, C),
            linear_v=of3_linear(C, C),
            linear_o=of3_linear(C, C),
            linear_g=of3_linear(C, C),
        ),
    )

    boltz_pair = arr(1, TOKENS, TOKENS, C)
    boltz_mask = jnp.ones((1, TOKENS, TOKENS), dtype=jnp.float32)
    protenix_pair = arr(TOKENS, TOKENS, C)
    protenix_mask = jnp.ones((TOKENS, TOKENS), dtype=jnp.float32)
    of3_pair = arr(TOKENS, TOKENS, C)

    # All six routes that reach the ring: each port's serial module (which
    # dispatches to its own CP branch under a mesh) and each port's CP
    # dispatch module. A dropped keyword anywhere is a silent return to the
    # full-row tile, so every one is checked rather than labelled. The knob is
    # each port's existing one -- `q_chunk_size` for Boltz-2 and Protenix,
    # `chunk_size` for OpenFold3, whose serial path already blocks rows with
    # it.
    arms = []
    for entry in (boltz_entry, boltz_cp_entry):
        arms.append(
            lambda block, entry=entry: block_rows(
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
            lambda block, entry=entry: block_rows(
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

    for entry in (of3_entry, of3_cp_entry):
        arms.append(
            lambda block, entry=entry: block_rows(
                functools.partial(
                    entry.triangle_attention,
                    no_heads=HEADS,
                    chunk_size=block,
                ),
                of3_pair,
                of3_params,
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


def test_port_chunk_knobs_reach_the_ring_row_block() -> None:
    """Each port's chunk knob must change the ring's lowering.

    The 2-D path used to document the chunk as deliberately unused, so a
    pass-through that silently stopped at the dispatch boundary would look
    exactly like the old, intended behaviour. It is one knob per port and it
    is the one the serial path already blocks rows with.
    """

    assert "PASS_THROUGH_OK" in _run(_PASS_THROUGH_PROBE, devices=4)


# --- the blocks are sequential by construction ------------------------------
# An unrolled Python loop over the blocks was not enough: the blocks are
# independent, so XLA scheduled them together and kept every tile live. The
# 2,096-token 2x2 GPU run then asked for 68.18 GiB against the unblocked
# path's 56.78 GiB. The loop is a `lax.scan` now, and this is the gate that
# says so: one while loop, whose body holds one block-sized score tile, with
# nothing of that shape left outside it but the peeled block.

_SEQUENTIAL_BLOCK_PROBE = textwrap.dedent(
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
    OUTER, TOKENS = 8 * SIDE, 4 * SIDE
    LOCAL = TOKENS // SIDE
    OUTER_LOCAL = OUTER // SIDE
    # A score tile is a rank-5 buffer whose last two axes are both the local
    # key extent and whose head axis sits above them, so the rows it was
    # evaluated for can be read straight off the shape. The pair bias has that
    # same shape with a singleton row axis, which is why 1 is excluded and why
    # no block size below may be 1.
    assert (LOCAL, OUTER_LOCAL, DIM) == (4, 8, 5)

    SHAPE = re.compile(r"\[([0-9,]+)\](?:\{[^}]*\})?\s+([a-z-]+)\(")
    REFS = re.compile(r"(?:calls|body|condition|to_apply)=\{?%?([\w.\-$]+)")
    HEADER = re.compile(r"^(ENTRY\s+)?%?([\w.\-$]+)")

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

    def parse(text):
        # HLO text as computations, their callees, the entry, and the names
        # used as a `while` body.
        blocks, refs, entry, bodies = {}, {}, None, set()
        name = None
        for line in text.splitlines():
            if line.endswith("{") and not line.startswith(" "):
                match = HEADER.match(line)
                name = match.group(2)
                if match.group(1):
                    entry = name
                blocks[name] = []
            elif line.startswith("}"):
                name = None
            elif name is not None:
                stripped = line.strip()
                blocks[name].append(stripped)
                refs.setdefault(name, set()).update(REFS.findall(stripped))
                bodies.update(re.findall(r"body=%?([\w.\-$]+)", stripped))
        assert entry is not None, text[:400]
        return blocks, refs, entry, bodies

    def reachable(refs, roots):
        seen, stack = set(), list(roots)
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            stack.extend(refs.get(current, ()))
        return seen

    def score_extents(lines):
        found = []
        for line in lines:
            for dims, _op in SHAPE.findall(line):
                shape = tuple(int(d) for d in dims.split(","))
                if (
                    len(shape) == 5
                    and shape[-1] == LOCAL
                    and shape[-2] == LOCAL
                    and shape[-3] == HEADS
                    and shape[-4] != 1
                ):
                    found.append(shape[-4])
        return found

    def arm(block):
        fn = jax.jit(functools.partial(ring_triangle_attention_2d, q_block=block))
        with context_parallel(DEVICES, layout="2d"):
            out = jax.device_get(fn(q, k, v, bias, mask))
            lowered = fn.lower(q, k, v, bias, mask)
            low = lowered.compiler_ir(dialect="hlo").as_hlo_text()
            optimized = lowered.compile().as_text()
        result = {"out": out, "permutes": low.lower().count("collective-permute")}
        for label, text in (("lowered", low), ("optimized", optimized)):
            blocks, refs, entry, bodies = parse(text)
            inside = reachable(refs, bodies)
            result[label] = {
                "bodies": len(bodies),
                "in_loop": sorted(
                    {
                        extent
                        for name in inside
                        for extent in score_extents(blocks.get(name, ()))
                    }
                ),
                "entry": score_extents(blocks[entry]),
            }
        return result

    off = arm(0)
    even = arm(2)
    ragged = arm(3)

    # Positive control. Without a block there is no loop at all and the whole
    # local row extent really is one buffer in the entry computation, which is
    # what makes its absence below mean something. One score buffer per
    # max-pass step and two per accumulate step, per ring step.
    assert off["optimized"]["bodies"] == 0, off["optimized"]
    assert set(off["optimized"]["entry"]) == {OUTER_LOCAL}, off["optimized"]
    whole_tiles = len(off["optimized"]["entry"])
    assert whole_tiles == 3 * SIDE, off["optimized"]

    for label, result, block, peel in (
        ("even", even, 2, 2),
        ("ragged", ragged, 3, 2),
    ):
        low_view = result["lowered"]
        opt_view = result["optimized"]
        # One scan around the whole ring, and the block-sized score tile lives
        # inside its body.
        assert low_view["bodies"] == 1, (label, low_view)
        assert opt_view["bodies"] == 1, (label, opt_view)
        assert low_view["in_loop"] == [block], (label, low_view)
        assert opt_view["in_loop"] == [block], (label, opt_view)
        # Nothing keeps the full local row extent, anywhere.
        assert OUTER_LOCAL not in low_view["in_loop"], (label, low_view)
        assert OUTER_LOCAL not in low_view["entry"], (label, low_view)
        assert OUTER_LOCAL not in opt_view["entry"], (label, opt_view)
        # Nothing hoisted above the scan: the only score buffers left outside
        # a loop body are the peeled block's, and there are exactly as many of
        # them as the unblocked arm had whole tiles. A hoisted or re-unrolled
        # block would multiply that count by the number of blocks.
        assert set(opt_view["entry"]) == {peel}, (label, opt_view)
        assert len(opt_view["entry"]) == whole_tiles, (label, opt_view)
        # One rotation schedule per emitted block body -- the peeled one and
        # the scan's -- and the two loop-invariant bias skews hoisted above
        # the row loop.
        assert result["permutes"] == 2 + 2 * (off["permutes"] - 2), (
            label,
            result["permutes"],
            off["permutes"],
        )
        np.testing.assert_allclose(off["out"], result["out"], atol=1e-6, rtol=1e-6)

    print(
        "SEQUENTIAL_BLOCK_DEVIATION",
        [float(np.max(np.abs(off["out"] - one["out"]))) for one in (even, ragged)],
    )
    print("SEQUENTIAL_BLOCK_OK")
    """
)


@pytest.mark.parametrize("devices", [4, 9])
def test_ring_row_blocks_run_one_at_a_time(devices: int) -> None:
    """The blocks must be sequential in the program, not merely written so.

    The first landing wrote them as an unrolled Python loop, which reads as
    blocked and compiles as parallel: on GPU the 2,096-token 2x2 Boltz-2 run
    asked for 68.18 GiB where the unblocked path asked 56.78 GiB. Only the
    lowering can tell those apart, so this gate reads it.
    """

    assert "SEQUENTIAL_BLOCK_OK" in _run(_SEQUENTIAL_BLOCK_PROBE, devices=devices)


# --- what the loop keeps alive ---------------------------------------------
# The reason the row axis is the blocked one. Blocking the query axis bounded
# the score tile and left everything else whole: at 1,024 OpenDDE structural
# tokens on a 2x2 mesh the pass-2 loop tuple held five f32[512, 12, 512, 32]
# tensors -- Q, K, V, the accumulator and its Neumaier correction -- 1,969 MiB
# of carry, quadratic in the local width. Blocking the rows and projecting
# inside the block leaves one tensor of that shape: the output destination,
# at the value dtype, which is the floor for a function that returns it.

_CARRY_PROBE = textwrap.dedent(
    r"""
    import functools
    import os
    import re

    import jax
    import jax.numpy as jnp
    import numpy as np

    from foldjax.models._cp import context_parallel
    from foldjax.models._cp_attention import (
        ring_triangle_attention_2d,
        ring_triangle_attention_2d_from_pair,
    )

    DEVICES = int(os.environ["FOLDJAX_CP_PROBE_DEVICES"])
    SIDE = int(round(DEVICES ** 0.5))
    assert SIDE * SIDE == DEVICES
    assert jax.device_count() == DEVICES, jax.devices()

    rng = np.random.default_rng(20260915)
    BATCH, HEADS, DIM = 2, 3, 5
    OUTER, TOKENS = 8 * SIDE, 4 * SIDE
    LOCAL = TOKENS // SIDE
    ROWS = OUTER // SIDE
    CHANNELS = HEADS * DIM
    BLOCK = 3
    assert (ROWS, LOCAL) == (8, 4)

    # The trunk dtype is bf16 and the biases are fp32, as they are in a real
    # trunk. That is what lets the census below separate the destination --
    # the rounded result -- from an fp32 accumulator of the same shape.
    def arr(*shape, scale=0.4, dtype=jnp.float32):
        return jnp.asarray(rng.normal(size=shape, scale=scale), dtype=dtype)

    bias = arr(BATCH, 1, HEADS, TOKENS, TOKENS, scale=7.0)
    keep = rng.random((BATCH, OUTER, TOKENS)) > 0.2
    keep[..., 0] = True
    mask = jnp.where(
        jnp.asarray(keep)[:, :, None, None, :],
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.asarray(-1.0e9, dtype=jnp.float32),
    )
    pair = arr(BATCH, OUTER, TOKENS, CHANNELS, dtype=jnp.bfloat16)
    weights = {
        name: arr(CHANNELS, CHANNELS, scale=0.2, dtype=jnp.bfloat16)
        for name in ("q", "k", "v", "g")
    }

    def project(params, rows):
        def heads_of(name):
            y = jnp.matmul(rows, params[name])
            y = y.reshape(y.shape[:-1] + (HEADS, DIM))
            return jnp.swapaxes(y, -2, -3)

        gate = jax.nn.sigmoid(heads_of("g"))
        return heads_of("q"), heads_of("k"), heads_of("v"), gate

    WHILE = re.compile(r"=\s*\(([^)]*)\)\s*while\(")
    OPERAND = re.compile(r"(f32|bf16|f16|s32|u32|pred)\[([0-9,]*)\]")
    WIDTH = {"f32": 4, "s32": 4, "u32": 4, "bf16": 2, "f16": 2, "pred": 1}

    def carries(text):
        found = []
        for tuple_text in WHILE.findall(text):
            operands = [
                (dtype, tuple(int(d) for d in dims.split(",") if d))
                for dtype, dims in OPERAND.findall(tuple_text)
            ]
            total = sum(
                int(np.prod(shape)) * WIDTH[dtype] for dtype, shape in operands
            )
            found.append({"operands": operands, "bytes": total})
        return found

    def widest(text):
        found = carries(text)
        assert found, "no while loop in the lowering"
        return max(found, key=lambda one: one["bytes"])

    def full_rows(operands):
        # Operands carrying the ring's whole local output geometry: the local
        # rows, the heads and the local key extent, in the Q/K/V layout.
        return [
            (dtype, shape)
            for dtype, shape in operands
            if len(shape) >= 4
            and shape[-4] == ROWS
            and shape[-3] == HEADS
            and shape[-2] == LOCAL
        ]

    def lower(fn, *args):
        # Both texts: the lowering is what this module wrote, the compiled
        # module is what the allocator is handed. XLA rewrites the tuple --
        # it hoists the loop-invariant bias rotations into it, for one -- so
        # the census has to hold on the second one as well.
        with context_parallel(DEVICES, layout="2d"):
            compiled = jax.jit(fn)
            out = jax.device_get(compiled(*args))
            lowered = compiled.lower(*args)
            text = lowered.compiler_ir(dialect="hlo").as_hlo_text()
            optimized = lowered.compile().as_text()
        return out, text, optimized

    # Positive control: the same row block, with the projections left outside.
    # Q, K and V are then loop operands beside the destination, which is what
    # makes the count of one below a change rather than a shape that never
    # existed. The q/k/v entry point is that arm by construction.
    query, key, value, gate = project(weights, pair)
    control_out, control_text, control_optimized = lower(
        functools.partial(ring_triangle_attention_2d, q_block=BLOCK),
        query,
        key,
        value,
        bias,
        mask,
    )
    control = widest(control_text)
    control_rows = full_rows(control["operands"])
    assert len(control_rows) == 4, control_rows
    assert {shape[-1] for _dtype, shape in control_rows} == {DIM}, control_rows
    assert len(full_rows(widest(control_optimized)["operands"])) == 4, (
        control_optimized[:400]
    )

    blocked_out, blocked_text, blocked_optimized = lower(
        functools.partial(
            ring_triangle_attention_2d_from_pair, project=project, q_block=BLOCK
        ),
        pair,
        bias,
        mask,
        weights,
    )
    blocked = widest(blocked_text)
    kept = full_rows(blocked["operands"])
    # One operand of that geometry: the destination, at the value dtype. The
    # projections, the score tile, the running maximum, the normalizer and
    # both Neumaier corrections are a block wide now, so none of them is here
    # -- including the `[rows, heads, keys, 1]` statistics, which is the shape
    # a re-widened accumulator would come back as.
    assert len(kept) == 1, kept
    assert kept[0][0] == "bf16", kept
    assert kept[0][1][-1] == DIM, kept
    assert not [
        shape for _dtype, shape in full_rows(blocked["operands"]) if shape[-1] == 1
    ], blocked["operands"]
    assert not [
        (dtype, shape)
        for dtype, shape in full_rows(blocked["operands"])
        if dtype == "f32"
    ], blocked["operands"]
    assert blocked["bytes"] < control["bytes"], (blocked["bytes"], control["bytes"])

    # The same census on the compiled module, where the allocator reads it.
    scheduled = widest(blocked_optimized)
    scheduled_rows = full_rows(scheduled["operands"])
    assert len(scheduled_rows) == 1, scheduled_rows
    assert scheduled_rows[0][0] == "bf16", scheduled_rows
    assert not [
        (dtype, shape)
        for dtype, shape in scheduled_rows
        if dtype == "f32" or shape[-1] == 1
    ], scheduled_rows

    # An unblocked ring has no loop to carry anything.
    unblocked_out, unblocked_text, _unblocked_optimized = lower(
        functools.partial(
            ring_triangle_attention_2d_from_pair, project=project, q_block=0
        ),
        pair,
        bias,
        mask,
        weights,
    )
    assert not carries(unblocked_text), unblocked_text[:400]

    # Projecting inside the block is the same arithmetic as projecting the
    # whole tile: a linear contracts over the channel axis, and a pair row is
    # projected exactly once either way.
    np.testing.assert_array_equal(unblocked_out, control_out * gate)
    # And writing the destination at the value dtype is the rounding the ring
    # already did at the end: the fp32 accumulator is divided and rounded per
    # row either way, so a block's rows come back bit-identical.
    np.testing.assert_array_equal(blocked_out, unblocked_out)
    print(
        "CARRY_BYTES",
        control["bytes"],
        blocked["bytes"],
        len(control_rows),
        len(kept),
    )
    print(
        "CARRY_DEVIATION",
        float(
            np.max(
                np.abs(
                    blocked_out.astype(np.float32)
                    - unblocked_out.astype(np.float32)
                )
            )
        ),
    )
    print("CARRY_OK")
    """
)


@pytest.mark.parametrize("devices", [4, 9])
def test_ring_row_blocks_leave_only_the_destination_in_the_carry(
    devices: int,
) -> None:
    """The loop must not hold anything of the full local output geometry.

    This is the defect the row block exists for. Query blocking left Q, K, V
    and both fp32 accumulators whole in the pass-2 loop -- 1,969 MiB of carry
    at 1,024 OpenDDE structural tokens on a 2x2 mesh, 67.5 GiB at 6,144 --
    and only a gate that reads the loop tuple can tell that apart from a
    program that merely looks blocked.
    """

    output = _run(_CARRY_PROBE, devices=devices)
    assert "CARRY_OK" in output, output
