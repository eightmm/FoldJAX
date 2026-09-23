"""Gates for the 2-D grid's streamed-gather triangle attention.

`gather_triangle_attention_2d_from_pair` replaces the ring's rotation by a
per-row-block gather along `cp_col` and a transpose-partner redistribution of
the bias, then runs one normalising attention per block. The GPU body is
cuEquivariance's kernel, which this host cannot load, so the gates split the
program at the seams that can be executed here:

* ownership -- what every rank of a real 2x2 / 3x3 fake mesh holds right before
  attention, compared *bitwise* against slices of the global arrays, through
  the same helpers the production body calls;
* arithmetic -- the XLA reference body against serial XLA triangle attention,
  on pre-projected operands and through Boltz-2's projections, gate and output
  projection at both entries and both directions;
* the cuEq boundary -- the argument construction and the mask polarity,
  against NVIDIA's own reference lowering. `cuequivariance_jax` registers a
  `platform=None` lowering that runs its JAX reference; with
  `cuequivariance_ops_jax` hidden (its import dlopens `libcue_ops.so`, which
  needs CUDA), the frontend's `try/except ImportError` leaves exactly that
  lowering, so `cuex.triangle_attention` itself runs on the CPU. That is a
  test-only device: the probes set it in their own subprocess.

Every probe runs in a subprocess because a forced device count has to be set
before JAX initialises.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

from tests.models.cp_probe_env import inherited_environment


def _run(source: str, *, devices: int, **extra: str) -> str:
    env = {
        "JAX_PLATFORMS": "cpu",
        "XLA_FLAGS": f"--xla_force_host_platform_device_count={devices}",
        "FOLDJAX_CP_PROBE_DEVICES": str(devices),
        **{f"FOLDJAX_CP_PROBE_{key.upper()}": value for key, value in extra.items()},
        **inherited_environment(),
    }
    completed = subprocess.run(
        [sys.executable, "-c", source],
        capture_output=True,
        text=True,
        env=env,
        timeout=600,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return completed.stdout


# --- gate: ownership ----------------------------------------------------------

_OWNERSHIP = textwrap.dedent(
    r"""
    import os

    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import PartitionSpec

    from foldjax.models._cp import (
        CP_COL_AXIS,
        CP_ROW_AXIS,
        context_parallel,
        cp_mesh,
    )
    from foldjax.models._cp_attention import (
        _pad_ring_biases,
        _two_axis_spec,
        _widen,
        fold_cp_pad_width,
        gather_triangle_bias_rows,
        gather_triangle_block_operands,
    )

    DEVICES = int(os.environ["FOLDJAX_CP_PROBE_DEVICES"])
    SIDE = int(round(DEVICES ** 0.5))
    N = int(os.environ["FOLDJAX_CP_PROBE_TOKENS"])
    BLOCK = int(os.environ["FOLDJAX_CP_PROBE_BLOCK"])
    TRANSPOSED = os.environ["FOLDJAX_CP_PROBE_TRANSPOSED"] == "1"
    B, H, D = 2, 2, 3
    C = H * D
    assert jax.device_count() == DEVICES

    # Every element a distinct integer, exact in f32 (< 2**24), so a value in
    # the wrong place -- another token, head, batch or rank -- cannot compare
    # equal. The scale factors in `project` keep the roles distinct too.
    ids = np.arange(B * N * N * C, dtype=np.float32).reshape(B, N, N, C) + 1.0
    pair = ids if not TRANSPOSED else np.swapaxes(ids, 1, 2).copy()
    bias = -(np.arange(B * H * N * N, dtype=np.float32).reshape(B, 1, H, N, N) + 1.0)
    # A non-prefix mask: an interior hole pattern, different per batch and
    # row, carrying its own distinct ids so a misplaced row is visible.
    rng = np.random.default_rng(7)
    holes = rng.random((B, N, N)) > 0.6
    mask = np.where(holes, -1.0e9, 0.0).astype(np.float32)
    mask = mask + np.arange(B * N * N, dtype=np.float32).reshape(B, N, N) * 1e-3
    mask = mask[:, :, None, None, :]

    def split(a):
        return jnp.swapaxes(a.reshape(a.shape[:-1] + (H, D)), -2, -3)

    def project(params, rows):
        # Exact elementwise roles: q = x, k = 2x, v = 4x, gate = -x.
        return split(rows), split(rows * 2.0), split(rows * 4.0), split(-rows)

    pad = (-N) % SIDE
    P = N + pad
    L = P // SIDE
    pair_p = np.pad(pair, ((0, 0), (0, pad), (0, pad), (0, 0)))
    bias_p = np.pad(bias, ((0, 0), (0, 0), (0, 0), (0, pad), (0, pad)))
    mask_p = np.pad(mask, ((0, 0), (0, pad), (0, 0), (0, 0), (0, 0)))
    mask_p = np.concatenate(
        [mask_p, np.full(mask_p.shape[:-1] + (pad,), -np.inf, np.float32)], -1
    )

    starts = list(range(0, L, BLOCK))
    assert len(starts) >= 2, "the probe must exercise more than one block"

    def split_np(a):
        a = a.reshape(a.shape[:-1] + (H, D))
        return np.swapaxes(a, -2, -3)

    def rows_np(array, r, start, axis):
        # Local rows [start, start + BLOCK) of grid row r, zero past L.
        stop = min(start + BLOCK, L)
        take = np.take(array, range(r * L + start, r * L + stop), axis=axis)
        short = start + BLOCK - min(start + BLOCK, L)
        if short:
            widths = [(0, 0)] * array.ndim
            widths[axis] = (0, short)
            take = np.pad(take, widths)
        return take

    with context_parallel(DEVICES, layout="2d"):
        mesh = cp_mesh()
        assert fold_cp_pad_width(N) == pad
        # The production padding, so the probe holds what the path holds.
        pair_j = _widen(jnp.asarray(pair), ((-3, pad), (-2, pad)))
        bias_j, mask_j = _pad_ring_biases(
            jnp.asarray(bias), jnp.asarray(mask), pad_rows=pad, pad_tokens=pad
        )
        assert np.array_equal(np.asarray(mask_j), mask_p)
        stacked = PartitionSpec(CP_ROW_AXIS, CP_COL_AXIS)

        def probe(pair_l, bias_l, mask_l):
            held = {"bias": gather_triangle_bias_rows(bias_l, SIDE)}
            for start in starts:
                q, k, v, gate, m = gather_triangle_block_operands(
                    pair_l, mask_l, None, start,
                    size=BLOCK, rows=L, tokens=N, project=project,
                )
                held.update({f"{name}{start}": value for name, value in
                             zip("qkvgm", (q, k, v, gate, m))})
            return {name: value[None, None] for name, value in held.items()}

        got = jax.jit(jax.shard_map(
            probe,
            mesh=mesh,
            in_specs=(
                _two_axis_spec(4, -3, -2),
                _two_axis_spec(5, -2, -1),
                _two_axis_spec(5, -4, -1),
            ),
            out_specs=stacked,
            # A probe of what each rank holds: the bias rows are replicated
            # over cp_row by construction, which a checked map would refuse
            # to stack. The production map is checked (XLA body).
            check_vma=False,
        ))(pair_j, bias_j, mask_j)
        got = jax.device_get(got)

    checked = 0
    for r in range(SIDE):
        for c in range(SIDE):
            cols = slice(c * L, (c + 1) * L)
            want_bias = bias_p[:, :, :, cols, :]
            have = got["bias"][r, c]
            assert have.shape == want_bias.shape, (have.shape, want_bias.shape)
            assert np.array_equal(have, want_bias), (r, c, "bias")
            for start in starts:
                rows = rows_np(pair_p, r, start, 1)
                want = {
                    "q": split_np(rows[:, :, cols]),
                    "k": split_np(rows[:, :, :N] * 2.0),
                    "v": split_np(rows[:, :, :N] * 4.0),
                    "g": split_np(-rows[:, :, cols]),
                    "m": rows_np(mask_p, r, start, 1)[..., :N],
                }
                for name, value in want.items():
                    have = got[f"{name}{start}"][r, c]
                    assert have.shape == value.shape, (name, have.shape, value.shape)
                    assert np.array_equal(have, value), (r, c, start, name)
                    checked += 1
    tail = L % BLOCK
    print(f"OWNERSHIP_OK side={SIDE} N={N} P={P} L={L} block={BLOCK} "
          f"tail={tail} transposed={int(TRANSPOSED)} arrays={checked + SIDE * SIDE}")
    """
)


@pytest.mark.parametrize(
    ("devices", "tokens", "block"),
    [
        (4, 7, 3),  # P=8, L=4: a one-row tail, one padded token
        (4, 8, 2),  # divisible everywhere
        (9, 13, 2),  # P=15, L=5: every off-diagonal pair of a 3x3 grid, a tail
        (9, 11, 3),  # P=12, L=4: one padded token on the last rank, a tail
    ],
)
@pytest.mark.parametrize("transposed", [False, True])
def test_each_rank_holds_exactly_its_gathered_operands(
    devices: int, tokens: int, block: int, transposed: bool
) -> None:
    """Bitwise Q/K/V/gate/bias/mask per rank before attention.

    ``transposed`` feeds the ending node's input -- the pair with its two
    token axes swapped, which is what both Boltz-2 entries hand this path for
    ``starting=False`` -- so nothing in the ownership can lean on a symmetric
    input.
    """

    out = _run(
        _OWNERSHIP,
        devices=devices,
        tokens=str(tokens),
        block=str(block),
        transposed="1" if transposed else "0",
    )
    assert "OWNERSHIP_OK" in out, out


# --- gate: arithmetic, pre-projected operands ---------------------------------

_PREPROJECTED = textwrap.dedent(
    r"""
    import os

    import jax
    import jax.numpy as jnp
    import numpy as np

    from foldjax.models._cp import context_parallel
    from foldjax.models._cp_attention import gather_triangle_attention_2d_from_pair
    from foldjax.models.boltz2.models.triangle.triangle_attention import _attention_core

    DEVICES = int(os.environ["FOLDJAX_CP_PROBE_DEVICES"])
    N = int(os.environ["FOLDJAX_CP_PROBE_TOKENS"])
    Q_BLOCK = int(os.environ["FOLDJAX_CP_PROBE_BLOCK"])
    B, H, D = 2, 3, 4
    rng = np.random.default_rng(20260923 + N)

    def arr(*shape, scale=0.6):
        return jnp.asarray(rng.normal(size=shape, scale=scale), dtype=jnp.float32)

    # Pre-projected: the pair channels are the heads' q/k/v, so `project` is
    # a reshape and the comparison isolates the data movement and the body.
    qp, kp, vp = arr(B, N, N, H * D), arr(B, N, N, H * D), arr(B, N, N, H * D)
    bias = arr(B, 1, H, N, N, scale=5.0)
    keep = rng.random((B, N, N)) > 0.25
    keep[1, 2, :] = False  # one genuinely fully masked row
    mask = jnp.where(jnp.asarray(keep)[:, :, None, None, :], 0.0, -1.0e9).astype(
        jnp.float32
    )

    def split(a):
        return jnp.swapaxes(a.reshape(a.shape[:-1] + (H, D)), -2, -3)

    def project(params, rows):
        q_rows, k_rows, v_rows = rows
        return split(q_rows), split(k_rows), split(v_rows), None

    scale = D ** -0.5
    q, k, v = split(qp), split(kp), split(vp)
    serial = _attention_core(
        q / jnp.sqrt(jnp.asarray(D, jnp.float32)), k, v, bias, mask, chunk_size=0
    )
    with context_parallel(DEVICES, layout="2d"):
        got = jax.jit(
            lambda pair, b, m: gather_triangle_attention_2d_from_pair(
                pair, b, m, None, project=project, scale=scale,
                q_block=Q_BLOCK or None,
            )
        )((qp, kp, vp), bias, mask)
    serial, got = np.asarray(serial), np.asarray(got)
    np.testing.assert_allclose(got, serial, atol=1e-5, rtol=1e-5)
    abs_err = float(np.max(np.abs(got - serial)))
    # Relative error where the reference is not near zero; the absolute bound
    # covers the rest (assert_allclose above checks both together).
    big = np.abs(serial) >= 1e-3
    rel_err = float(np.max(np.abs(got - serial)[big] / np.abs(serial)[big]))
    print(f"PREPROJECTED_OK devices={DEVICES} N={N} max_abs={abs_err:.3e} "
          f"max_rel(|ref|>=1e-3)={rel_err:.3e}")
    """
)


@pytest.mark.parametrize(
    ("devices", "tokens", "block"),
    [(4, 8, 0), (4, 13, 2), (9, 9, 0), (9, 13, 2), (9, 16, 3)],
)
def test_gather_body_matches_serial_on_preprojected_operands(
    devices: int, tokens: int, block: int
) -> None:
    out = _run(_PREPROJECTED, devices=devices, tokens=str(tokens), block=str(block))
    assert "PREPROJECTED_OK" in out, out
    print(out.strip())


# --- gate: arithmetic, through Boltz-2's projections -------------------------

_BOLTZ = textwrap.dedent(
    r"""
    import os
    import sys

    CUEQ = os.environ["FOLDJAX_CP_PROBE_BODY"] == "cueq"
    if CUEQ:
        # Test-only: hide the CUDA ops package so the frontend keeps its
        # platform-independent reference lowering (see the module docstring).
        sys.modules["cuequivariance_ops_jax"] = None

    import jax
    import jax.numpy as jnp
    import numpy as np

    from foldjax.models._cp import context_parallel
    from foldjax.models._cp_attention import triangle_attention_grid_scope
    from foldjax.models.boltz2.models.triangle import (
        triangle_attention as serial_module,
    )
    from foldjax.models.boltz2.models.triangle import (
        triangle_attention_cp as trunk_module,
    )

    DEVICES = int(os.environ["FOLDJAX_CP_PROBE_DEVICES"])
    N = int(os.environ["FOLDJAX_CP_PROBE_TOKENS"])
    B, CZ, H = 2, 12, 3
    rng = np.random.default_rng(20260924 + N)

    def arr(*shape, scale=0.5):
        return jnp.asarray(rng.normal(size=shape, scale=scale), dtype=jnp.float32)

    def weight(fan_in, fan_out):
        return jnp.asarray(
            rng.normal(size=(fan_in, fan_out), scale=1.0 / np.sqrt(fan_in)),
            dtype=jnp.float32,
        )

    params = {
        "layer_norm": {"scale": arr(CZ) * 0.1 + 1.0, "bias": arr(CZ) * 0.1},
        "linear": {"kernel": weight(CZ, H) * 3.0},
        "mha": {
            name: {"kernel": weight(CZ, CZ)}
            for name in ("linear_q", "linear_k", "linear_v", "linear_g", "linear_o")
        },
    }
    x = arr(B, N, N, CZ, scale=1.0)
    # Token padding: the last token is absent in batch 1, so its pair row
    # and column are fully masked -- a genuine fully-masked row the serial
    # contract defines -- and interior pair holes on top.
    token = np.ones((B, N), dtype=bool)
    token[1, -1] = False
    pair_mask = token[:, :, None] & token[:, None, :]
    pair_mask &= rng.random((B, N, N)) > 0.1
    pair_mask[:, np.arange(N), np.arange(N)] &= token
    mask = jnp.asarray(pair_mask.astype(np.float32))

    if CUEQ:
        # The GPU body, on this host through NVIDIA's reference lowering.
        trunk_module.resolve_gather_attention_body = lambda: "cueq"
        serial_module.resolve_gather_attention_body = lambda: "cueq"
    backend = "cueq" if CUEQ else "xla"

    report = []
    for direction in (True, False):
        serial = jax.jit(
            lambda p, z, m: serial_module.triangle_attention_forward(
                p, z, m, starting=direction, triangle_backend=backend,
                chunk_size=0,
            )
        )(params, x, mask)
        with context_parallel(DEVICES, layout="2d"), triangle_attention_grid_scope(
            "gather"
        ):
            for name, entry in (
                ("trunk", trunk_module.triangle_attention_forward),
                ("msa", serial_module.triangle_attention_forward),
            ):
                got = jax.jit(
                    lambda p, z, m, entry=entry: entry(
                        p, z, m, starting=direction, q_chunk_size=2
                    )
                )(params, x, mask)
                got_np, serial_np = np.asarray(got), np.asarray(serial)
                np.testing.assert_allclose(got_np, serial_np, atol=1e-5, rtol=1e-5)
                abs_err = float(np.max(np.abs(got_np - serial_np)))
                big = np.abs(serial_np) >= 1e-3
                rel_err = float(np.max(
                    np.abs(got_np - serial_np)[big] / np.abs(serial_np)[big]
                ))
                report.append(
                    f"{name}_{'start' if direction else 'end'} "
                    f"max_abs={abs_err:.3e} max_rel(|ref|>=1e-3)={rel_err:.3e}"
                )
    head = f"BOLTZ_GATHER_OK body={backend} devices={DEVICES} N={N} "
    print(head + "; ".join(report))
    """
)


@pytest.mark.parametrize(("devices", "tokens"), [(4, 8), (4, 11), (9, 13)])
@pytest.mark.parametrize("body", ["xla", "cueq"])
def test_boltz2_gather_entries_match_serial(
    devices: int, tokens: int, body: str
) -> None:
    """Both entries, both directions, through projection, gate and output.

    ``xla`` is the reference body against the serial XLA path. ``cueq`` is
    the GPU body's argument construction run through NVIDIA's own reference
    lowering, against the serial ``triangle_backend='cueq'`` path through the
    same lowering: that is the gate for the kernel's rectangular shapes, the
    one polarity conversion, and a fully-masked row on a padded grid -- which
    only holds because the gathered key axis drops the grid padding (the
    kernel *replaces* a masked logit with ``-1e9``, so a padded key it could
    see would join the uniform average serial takes over ``N`` keys).
    """

    out = _run(_BOLTZ, devices=devices, tokens=str(tokens), body=body)
    assert "BOLTZ_GATHER_OK" in out, out
    print(out.strip())


# --- gate: arithmetic, through Protenix's and OpenFold3's projections ---------

_PORTS = textwrap.dedent(
    r"""
    import os
    import sys

    CUEQ = os.environ["FOLDJAX_CP_PROBE_BODY"] == "cueq"
    if CUEQ:
        # Test-only: hide the CUDA ops package so the frontend keeps its
        # platform-independent reference lowering (see the module docstring).
        sys.modules["cuequivariance_ops_jax"] = None

    import jax
    import jax.numpy as jnp
    import numpy as np

    from foldjax.models._cp import context_parallel
    from foldjax.models._cp_attention import triangle_attention_grid_scope

    PORT = os.environ["FOLDJAX_CP_PROBE_PORT"]
    DEVICES = int(os.environ["FOLDJAX_CP_PROBE_DEVICES"])
    N = int(os.environ["FOLDJAX_CP_PROBE_TOKENS"])
    CZ, H = 12, 3
    rng = np.random.default_rng(20260925 + N + DEVICES)

    def arr(*shape, scale=0.5):
        return jnp.asarray(rng.normal(size=shape, scale=scale), dtype=jnp.float32)

    # Every kernel call is recorded, so a `cueq` arm cannot quietly compare
    # the XLA path with itself (Protenix's serial path takes the kernel only
    # above 16 columns).
    calls = []
    if CUEQ:
        from foldjax.models._cueq import load_cueq

        cuex = load_cueq()
        real = cuex.triangle_attention

        def recording(**kwargs):
            calls.append(tuple(kwargs["q"].shape) + tuple(kwargs["k"].shape[-2:]))
            return real(**kwargs)

        cuex.triangle_attention = recording

    token = np.ones(N, dtype=bool)
    token[-1] = False  # an absent token: a genuinely fully masked pair row
    pair_mask = token[:, None] & token[None, :]
    pair_mask &= rng.random((N, N)) > 0.1  # interior holes
    pair_mask[np.arange(N), np.arange(N)] = token

    if PORT == "protenix":
        from foldjax.models.protenix.models.primitives.attention import (
            AttentionParams,
        )
        from foldjax.models.protenix.models.primitives.primitives import (
            LayerNormParams,
            LinearParams,
        )
        from foldjax.models.protenix.models.triangle import (
            triangle as serial_module,
        )
        from foldjax.models.protenix.models.triangle import (
            triangle_attention_cp as cp_module,
        )
        from foldjax.models.protenix.models.triangle.triangle import (
            TriangleAttentionParams,
        )

        def linear(o, i, scale=1.0):
            return LinearParams(weight=arr(o, i, scale=scale / np.sqrt(i)), bias=None)

        params = TriangleAttentionParams(
            layer_norm=LayerNormParams(weight=arr(CZ) * 0.1 + 1.0, bias=arr(CZ) * 0.1),
            linear=linear(H, CZ, 3.0),
            attention=AttentionParams(
                linear_q=linear(CZ, CZ), linear_k=linear(CZ, CZ),
                linear_v=linear(CZ, CZ), linear_o=linear(CZ, CZ),
                linear_g=linear(CZ, CZ),
            ),
        )
        # Unbatched, as this port's trunk hands it over.
        x = arr(N, N, CZ, scale=1.0)
        mask = jnp.asarray(pair_mask.astype(np.float32))
        arms = (("start", {"starting": True}), ("end", {"starting": False}))

        def serial(kw):
            return jax.jit(lambda z, m, p: serial_module.triangle_attention(
                z, m, p, num_heads=H, attention_backend="cueq" if CUEQ else "xla",
                **kw,
            ))(x, mask, params)

        def sharded(entry, kw):
            return jax.jit(lambda z, m, p: entry.triangle_attention(
                z, m, p, num_heads=H, q_chunk_size=2, **kw,
            ))(x, mask, params)
    else:
        from foldjax.models.openfold3.models import (
            triangle_attention as serial_module,
        )
        from foldjax.models.openfold3.models import (
            triangle_attention_cp as cp_module,
        )
        from foldjax.models.openfold3.models.attention import AttentionParams
        from foldjax.models.openfold3.models.primitives import (
            LayerNormParams,
            LinearParams,
        )
        from foldjax.models.openfold3.models.triangle_attention import (
            TriangleAttentionParams,
        )

        def linear(o, i, scale=1.0):
            return LinearParams(weight=arr(o, i, scale=scale / np.sqrt(i)), bias=None)

        params = TriangleAttentionParams(
            layer_norm=LayerNormParams(weight=arr(CZ) * 0.1 + 1.0, bias=arr(CZ) * 0.1),
            linear_z=linear(H, CZ, 3.0),
            mha=AttentionParams(
                linear_q=linear(CZ, CZ), linear_k=linear(CZ, CZ),
                linear_v=linear(CZ, CZ), linear_o=linear(CZ, CZ),
                linear_g=linear(CZ, CZ),
            ),
        )
        # Batched (B=2, the second member masked differently), as the
        # template stack and the confidence head hand it over.
        x = arr(2, N, N, CZ, scale=1.0)
        second = pair_mask & (rng.random((N, N)) > 0.2)
        mask = jnp.asarray(np.stack([pair_mask, second]).astype(np.float32))
        # PairBlock runs `starting=True` twice and transposes the pair itself,
        # with the corrected bias orientation on the second; `starting=False`
        # is the standalone module's ending node.
        arms = (
            ("start", {"starting": True}),
            ("end", {"starting": False}),
            ("start_transposed_bias", {"starting": True, "transpose_bias": True}),
        )

        def serial(kw):
            return jax.jit(lambda z, m, p: serial_module.triangle_attention(
                z, p, no_heads=H, mask=m, backend="cueq" if CUEQ else "xla", **kw,
            ))(x, mask, params)

        def sharded(entry, kw):
            return jax.jit(lambda z, m, p: entry.triangle_attention(
                z, p, no_heads=H, mask=m, chunk_size=2, **kw,
            ))(x, mask, params)

    if CUEQ:
        # The GPU body, on this host through NVIDIA's reference lowering.
        cp_module.resolve_gather_attention_body = lambda: "cueq"
        serial_module.resolve_gather_attention_body = lambda: "cueq"

    report = []
    for arm, kw in arms:
        calls.clear()
        reference = np.asarray(serial(kw))
        assert len(calls) == (1 if CUEQ else 0), (arm, calls)
        for name, entry in (("cp", cp_module), ("serial_module", serial_module)):
            calls.clear()
            with context_parallel(DEVICES, layout="2d"), (
                triangle_attention_grid_scope("gather")
            ):
                got = np.asarray(sharded(entry, kw))
            if CUEQ:
                # Rectangular: local query columns against all N keys.
                assert calls and all(c[-2] == N and c[-3] < N for c in calls), calls
            else:
                assert not calls, calls
            assert np.all(np.isfinite(got)), (name, arm)
            np.testing.assert_allclose(got, reference, atol=1e-5, rtol=1e-5)
            abs_err = float(np.max(np.abs(got - reference)))
            big = np.abs(reference) >= 1e-3
            rel_err = float(np.max(
                np.abs(got - reference)[big] / np.abs(reference)[big]
            ))
            report.append(
                f"{name}_{arm} max_abs={abs_err:.3e} "
                f"max_rel(|ref|>=1e-3)={rel_err:.3e}"
            )
    head = f"PORT_GATHER_OK port={PORT} body={'cueq' if CUEQ else 'xla'} "
    print(head + f"devices={DEVICES} N={N} " + "; ".join(report))
    """
)


@pytest.mark.parametrize(("devices", "tokens"), [(4, 17), (4, 20), (9, 19)])
@pytest.mark.parametrize("body", ["xla", "cueq"])
@pytest.mark.parametrize("port", ["protenix", "openfold3"])
def test_port_gather_entries_match_serial(
    port: str, body: str, devices: int, tokens: int
) -> None:
    """Protenix's and OpenFold3's two 2-D entries under ``gather``, vs serial.

    Through each port's own layer norm, bias projection, Q/K/V/gate
    projections and output projection, both directions (and OpenFold3's
    transposed-bias orientation), against the port's serial path with the
    same body: ``xla`` against serial XLA, ``cueq`` -- through NVIDIA's
    reference lowering -- against the port's serial cuEquivariance call. That
    pins each port's scale placement (in the kernel, on an unscaled query),
    its precision (the policy: ``precision=None`` on both sides) and its
    mask/bias conventions. Every token count exceeds 16, because Protenix's
    serial path takes the kernel only above that; the probe counts kernel
    calls so neither side can be the XLA path under a ``cueq`` label. 17 and
    19 pad the grid, 20 does not, and the two-row block leaves a tail.
    """

    out = _run(_PORTS, devices=devices, tokens=str(tokens), body=body, port=port)
    assert "PORT_GATHER_OK" in out, out
    print(out.strip())


# --- gate: padding invariance -------------------------------------------------

_PADDING = textwrap.dedent(
    r"""
    import os

    import jax
    import jax.numpy as jnp
    import numpy as np

    from foldjax.models._cp import context_parallel
    from foldjax.models._cp_attention import triangle_attention_grid_scope
    from foldjax.models.boltz2.models.triangle import (
        triangle_attention_cp as trunk_module,
    )

    DEVICES = int(os.environ["FOLDJAX_CP_PROBE_DEVICES"])
    VALID, EXTRA = 9, 4
    B, CZ, H = 1, 8, 2
    rng = np.random.default_rng(99)

    def weight(fan_in, fan_out):
        return jnp.asarray(
            rng.normal(size=(fan_in, fan_out), scale=1.0 / np.sqrt(fan_in)),
            dtype=jnp.float32,
        )

    params = {
        "layer_norm": {"scale": jnp.ones(CZ), "bias": jnp.zeros(CZ)},
        "linear": {"kernel": weight(CZ, H) * 3.0},
        "mha": {
            name: {"kernel": weight(CZ, CZ)}
            for name in ("linear_q", "linear_k", "linear_v", "linear_g", "linear_o")
        },
    }
    full = VALID + EXTRA
    x = jnp.asarray(rng.normal(size=(B, full, full, CZ)), dtype=jnp.float32)
    valid = np.zeros((B, full, full), dtype=np.float32)
    valid[:, :VALID, :VALID] = 1.0

    results = {}
    with context_parallel(DEVICES, layout="2d"), triangle_attention_grid_scope(
        "gather"
    ):
        for direction in (True, False):
            padded = jax.jit(
                lambda p, z, m: trunk_module.triangle_attention_forward(
                    p, z, m, starting=direction, q_chunk_size=2
                )
            )(params, x, jnp.asarray(valid))
            bare = jax.jit(
                lambda p, z, m: trunk_module.triangle_attention_forward(
                    p, z, m, starting=direction, q_chunk_size=2
                )
            )(params, x[:, :VALID, :VALID], jnp.ones((B, VALID, VALID)))
            padded = np.asarray(padded)[:, :VALID, :VALID]
            bare = np.asarray(bare)
            assert np.all(np.isfinite(padded))
            np.testing.assert_allclose(padded, bare, atol=1e-5, rtol=1e-5)
            results[direction] = float(np.max(np.abs(padded - bare)))
    print(f"PADDING_INVARIANT_OK devices={DEVICES} start={results[True]:.3e} "
          f"end={results[False]:.3e}")
    """
)


@pytest.mark.parametrize("devices", [4, 9])
def test_gather_result_is_invariant_to_masked_padding_tokens(devices: int) -> None:
    """Nine valid tokens padded to thirteen give the nine-token answer.

    Thirteen splits neither grid, so this also runs the grid padding on top
    of the masked serving padding; the padded rows are fully masked and must
    stay finite.
    """

    out = _run(_PADDING, devices=devices)
    assert "PADDING_INVARIANT_OK" in out, out
    print(out.strip())


# --- gate: dead projections are removed ---------------------------------------

_DOTS = textwrap.dedent(
    r"""
    import os

    import jax
    import jax.numpy as jnp
    import numpy as np

    from foldjax.models._cp import context_parallel
    from foldjax.models._cp_attention import triangle_attention_grid_scope
    from foldjax.models.boltz2.models.triangle import (
        triangle_attention_cp as trunk_module,
    )

    B, N, CZ, H = 1, 8, 8, 2
    rng = np.random.default_rng(3)
    w = lambda i, o: jnp.asarray(rng.normal(size=(i, o)), jnp.float32)  # noqa: E731
    params = {
        "layer_norm": {"scale": jnp.ones(CZ), "bias": jnp.zeros(CZ)},
        "linear": {"kernel": w(CZ, H)},
        "mha": {n: {"kernel": w(CZ, CZ)} for n in
                ("linear_q", "linear_k", "linear_v", "linear_g", "linear_o")},
    }
    x = jnp.asarray(rng.normal(size=(B, N, N, CZ)), jnp.float32)
    with context_parallel(4, layout="2d"), triangle_attention_grid_scope("gather"):
        # One block (q_chunk_size <= 0): one trace of the block body.
        lowered = jax.jit(
            lambda p, z: trunk_module.triangle_attention_forward(
                p, z, None, q_chunk_size=0
            )
        ).lower(params, x)
    text = lowered.compile().as_text()
    dots = [line for line in text.splitlines() if " dot(" in line]
    # bias linear, q|g (local), k|v (full width), scores, probabilities @ v,
    # output linear. A surviving full-width q|g or local k|v would be 7 or 8.
    assert len(dots) == 6, "\n".join(dots)
    # The collective census, before XLA's combiner merges the two cp_col
    # gathers: bias rows (cp_row), pair rows and mask (cp_col), one
    # transpose-partner permute -- and no ring hop, no softmax reduction.
    import collections
    import re

    hlo = lowered.compiler_ir(dialect="hlo").as_hlo_text()
    census = collections.Counter(re.findall(
        r"\b(all-gather|all-reduce|collective-permute|all-to-all|reduce-scatter)"
        r"(?:-start)?\(",
        hlo,
    ))
    assert census == {"all-gather": 3, "collective-permute": 1}, census
    print(f"DOTS_OK dots={len(dots)} census={dict(census)}")
    """
)


def test_unused_projection_halves_are_dead_code() -> None:
    """``project`` runs twice per block; its unused halves must not compile."""

    out = _run(_DOTS, devices=4)
    assert "DOTS_OK" in out, out


_PORT_DOTS = textwrap.dedent(
    r"""
    import collections
    import os
    import re

    import jax
    import jax.numpy as jnp
    import numpy as np

    from foldjax.models._cp import context_parallel
    from foldjax.models._cp_attention import triangle_attention_grid_scope

    PORT = os.environ["FOLDJAX_CP_PROBE_PORT"]
    N, CZ, H = 8, 8, 2
    rng = np.random.default_rng(3)
    w = lambda o, i: jnp.asarray(rng.normal(size=(o, i)), jnp.float32)  # noqa: E731
    if PORT == "protenix":
        from foldjax.models.protenix.models.primitives.attention import (
            AttentionParams,
        )
        from foldjax.models.protenix.models.primitives.primitives import (
            LayerNormParams,
            LinearParams,
        )
        from foldjax.models.protenix.models.triangle import (
            triangle_attention_cp as entry,
        )
        from foldjax.models.protenix.models.triangle.triangle import (
            TriangleAttentionParams,
        )

        lin = lambda o, i: LinearParams(weight=w(o, i), bias=None)  # noqa: E731
        params = TriangleAttentionParams(
            layer_norm=LayerNormParams(weight=jnp.ones(CZ), bias=jnp.zeros(CZ)),
            linear=lin(H, CZ),
            attention=AttentionParams(
                linear_q=lin(CZ, CZ), linear_k=lin(CZ, CZ), linear_v=lin(CZ, CZ),
                linear_o=lin(CZ, CZ), linear_g=lin(CZ, CZ),
            ),
        )
        x = jnp.asarray(rng.normal(size=(N, N, CZ)), jnp.float32)
        call = lambda p, z: entry.triangle_attention(  # noqa: E731
            z, None, p, num_heads=H, q_chunk_size=0
        )
    else:
        from foldjax.models.openfold3.models import triangle_attention_cp as entry
        from foldjax.models.openfold3.models.attention import AttentionParams
        from foldjax.models.openfold3.models.primitives import (
            LayerNormParams,
            LinearParams,
        )
        from foldjax.models.openfold3.models.triangle_attention import (
            TriangleAttentionParams,
        )

        lin = lambda o, i: LinearParams(weight=w(o, i), bias=None)  # noqa: E731
        params = TriangleAttentionParams(
            layer_norm=LayerNormParams(weight=jnp.ones(CZ), bias=jnp.zeros(CZ)),
            linear_z=lin(H, CZ),
            mha=AttentionParams(
                linear_q=lin(CZ, CZ), linear_k=lin(CZ, CZ), linear_v=lin(CZ, CZ),
                linear_o=lin(CZ, CZ), linear_g=lin(CZ, CZ),
            ),
        )
        x = jnp.asarray(rng.normal(size=(1, N, N, CZ)), jnp.float32)
        call = lambda p, z: entry.triangle_attention(  # noqa: E731
            z, p, no_heads=H, chunk_size=0
        )
    with context_parallel(4, layout="2d"), triangle_attention_grid_scope("gather"):
        lowered = jax.jit(call).lower(params, x)
    text = lowered.compile().as_text()
    dots = [line for line in text.splitlines() if " dot(" in line]
    # Neither port fuses q|g or k|v into one linear, so: bias, q, gate (local),
    # k, v (full width), scores, probabilities @ v, output = 8. A surviving
    # full-width q/gate or local k/v would add up to four more.
    assert len(dots) == 8, "\n".join(dots)
    hlo = lowered.compiler_ir(dialect="hlo").as_hlo_text()
    census = collections.Counter(re.findall(
        r"\b(all-gather|all-reduce|collective-permute|all-to-all|reduce-scatter)"
        r"(?:-start)?\(",
        hlo,
    ))
    assert census == {"all-gather": 3, "collective-permute": 1}, census
    print(f"PORT_DOTS_OK port={PORT} dots={len(dots)} census={dict(census)}")
    """
)


@pytest.mark.parametrize("port", ["protenix", "openfold3"])
def test_port_unused_projection_halves_are_dead_code(port: str) -> None:
    """The same dot and collective census through each port's ``project``."""

    out = _run(_PORT_DOTS, devices=4, port=port)
    assert "PORT_DOTS_OK" in out, out
    print(out.strip())


# --- gate: mask polarity at the cuEq boundary ---------------------------------


def test_polarity_sentinel_on_the_xla_reference_body() -> None:
    """Two keys, zero logits, values 1 and 2, only the second valid -> 2."""

    import jax.numpy as jnp
    import numpy as np

    from foldjax.models._cp_attention import gather_attention_xla

    q = jnp.zeros((1, 1, 1, 1, 4), jnp.float32)
    k = jnp.zeros((1, 1, 1, 2, 4), jnp.float32)
    v = jnp.asarray([1.0, 2.0], jnp.float32).reshape(1, 1, 1, 2, 1) * jnp.ones(
        (1, 1, 1, 2, 4), jnp.float32
    )
    bias = jnp.zeros((1, 1, 1, 1, 2), jnp.float32)
    mask = jnp.asarray([-1.0e9, 0.0], jnp.float32).reshape(1, 1, 1, 1, 2)
    out = gather_attention_xla(q, k, v, bias, mask, scale=0.5)
    np.testing.assert_array_equal(np.asarray(out), np.full((1, 1, 1, 1, 4), 2.0))


def test_polarity_sentinel_on_the_cueq_argument_construction() -> None:
    """The kernel's boolean is ``True`` = valid; the additive 0 is the valid key.

    Compile-free: ``jax.eval_shape`` of the argument construction for the
    gather body's rectangular call (``S_qo`` local query columns, ``S_kv``
    keys) and the concrete boolean it produces for the sentinel.
    """

    import jax
    import jax.numpy as jnp
    import numpy as np

    from foldjax.models._cueq import cueq_attention_arguments

    lead, arguments = cueq_attention_arguments(
        jnp.zeros((1, 1, 1, 1, 4)),
        jnp.zeros((1, 1, 1, 2, 4)),
        jnp.zeros((1, 1, 1, 2, 4)),
        jnp.zeros((1, 1, 1, 1, 2)),
        jnp.asarray([-1.0e9, 0.0], jnp.float32).reshape(1, 1, 1, 1, 2),
    )
    assert lead == (1,)
    np.testing.assert_array_equal(
        np.asarray(arguments["mask"]).reshape(-1), [False, True]
    )

    # A gather block: [B=2, R=3, H=4, L=5, D=8] queries against N=11 keys.
    shapes = jax.eval_shape(
        lambda q, k, v, b, m: cueq_attention_arguments(q, k, v, b, m)[1],
        jax.ShapeDtypeStruct((2, 3, 4, 5, 8), jnp.bfloat16),
        jax.ShapeDtypeStruct((2, 3, 4, 11, 8), jnp.bfloat16),
        jax.ShapeDtypeStruct((2, 3, 4, 11, 8), jnp.bfloat16),
        jax.ShapeDtypeStruct((2, 1, 4, 5, 11), jnp.float32),
        jax.ShapeDtypeStruct((2, 3, 1, 1, 11), jnp.float32),
    )
    assert shapes["q"].shape == (2, 3, 4, 5, 8)
    assert shapes["k"].shape == shapes["v"].shape == (2, 3, 4, 11, 8)
    assert shapes["bias"].shape == (2, 1, 4, 5, 11)
    assert shapes["mask"].shape == (2, 3, 1, 1, 11)
    assert shapes["mask"].dtype == jnp.bool_


_WHEEL_SENTINEL = textwrap.dedent(
    r"""
    import sys

    # Test-only: hide the CUDA ops package; see the module docstring.
    sys.modules["cuequivariance_ops_jax"] = None

    import jax
    import jax.numpy as jnp
    import numpy as np

    from foldjax.models._cp import context_parallel
    from foldjax.models._cp_attention import gather_attention_cueq
    from foldjax.models._cueq import load_cueq

    cuex = load_cueq()
    q = jnp.zeros((1, 1, 1, 1, 4), jnp.float32)
    k = jnp.zeros((1, 1, 1, 2, 4), jnp.float32)
    v = jnp.asarray([1.0, 2.0], jnp.float32).reshape(1, 1, 1, 2, 1) * jnp.ones(
        (1, 1, 1, 2, 4), jnp.float32
    )
    bias = jnp.zeros((1, 1, 1, 1, 2), jnp.float32)
    mask = jnp.asarray([-1.0e9, 0.0], jnp.float32).reshape(1, 1, 1, 1, 2)
    out = gather_attention_cueq(q, k, v, bias, mask, scale=0.5)
    np.testing.assert_array_equal(np.asarray(out), np.full((1, 1, 1, 1, 4), 2.0))

    # What the kernel receives from inside the gather body's unchecked
    # shard_map on a 2x2 grid: the arguments recorded at the wheel's entry.
    from foldjax.models._cp_attention import gather_triangle_attention_2d_from_pair

    seen = []
    real = cuex.triangle_attention

    def recording(**kwargs):
        seen.append({n: (a.shape, a.dtype) for n, a in kwargs.items()
                     if hasattr(a, "shape")} | {"precision": kwargs["precision"],
                                                "scale": kwargs["scale"]})
        return real(**kwargs)

    cuex.triangle_attention = recording
    B, N, H, D = 2, 11, 2, 8
    rng = np.random.default_rng(0)
    x = jnp.asarray(rng.normal(size=(B, N, N, H * D)), jnp.bfloat16)

    def split(a):
        return jnp.swapaxes(a.reshape(a.shape[:-1] + (H, D)), -2, -3)

    def project(params, rows):
        return split(rows), split(rows), split(rows), None

    with context_parallel(4, layout="2d"):
        jax.eval_shape(
            lambda z, b, m: gather_triangle_attention_2d_from_pair(
                z, b, m, None, project=project, scale=D ** -0.5,
                precision=jax.lax.Precision.HIGHEST, q_block=2, body="cueq",
            ),
            x,
            jnp.zeros((B, 1, H, N, N), jnp.float32),
            jnp.zeros((B, N, 1, 1, N), jnp.float32),
        )
    # P = 12, L = 6, R = 2: every block call is S_qo = 6 against S_kv = 11.
    assert seen, "the gather body never reached the kernel"
    for call in seen:
        assert call["q"] == ((B, 2, H, 6, D), jnp.bfloat16), call
        assert call["k"] == ((B, 2, H, N, D), jnp.bfloat16), call
        assert call["bias"] == ((B, 1, H, 6, N), jnp.float32), call
        assert call["mask"] == ((B, 2, 1, 1, N), jnp.bool_), call
        assert call["precision"] == jax.lax.Precision.HIGHEST, call
        assert abs(call["scale"] - D ** -0.5) < 1e-12, call
    print(f"WHEEL_SENTINEL_OK calls={len(seen)}")
    """
)


def test_polarity_sentinel_through_nvidias_reference_lowering() -> None:
    """The same sentinel through ``cuex.triangle_attention`` itself.

    Then the argument shapes the gather body hands the wheel from inside its
    unchecked ``shard_map``, recorded at the wheel's entry under
    ``jax.eval_shape``: rectangular, boolean mask, the caller's precision.
    """

    out = _run(_WHEEL_SENTINEL, devices=4)
    assert "WHEEL_SENTINEL_OK" in out, out


# --- the option's vocabulary and refusals -------------------------------------


def test_grid_scope_publishes_and_validates() -> None:
    from foldjax.models._cp_attention import (
        TRIANGLE_ATTENTION_GRIDS,
        triangle_attention_grid,
        triangle_attention_grid_scope,
    )

    assert TRIANGLE_ATTENTION_GRIDS == ("ring", "gather")
    assert triangle_attention_grid() == "ring"
    with triangle_attention_grid_scope("gather") as name:
        assert name == "gather" == triangle_attention_grid()
    with triangle_attention_grid_scope(None) as name:
        assert name == "ring" == triangle_attention_grid()
    assert triangle_attention_grid() == "ring"
    with pytest.raises(ValueError, match="triangle_attention_grid"):
        with triangle_attention_grid_scope("rotate"):
            pass


def test_gather_body_off_a_gpu_is_the_reference(monkeypatch) -> None:
    import jax

    from foldjax.models import _cp_attention

    monkeypatch.setattr(jax, "default_backend", lambda: "cpu")
    assert _cp_attention.resolve_gather_attention_body() == "xla"


def test_gather_on_a_gpu_without_cueq_is_refused(monkeypatch) -> None:
    """Refused, never downgraded to the reference body under the gather label."""

    import jax

    from foldjax.models import _cp_attention, _cueq

    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")

    def missing():
        raise RuntimeError("cuEquivariance JAX is required by this backend")

    monkeypatch.setattr(_cueq, "load_cueq", missing)
    with pytest.raises(RuntimeError, match="triangle_attention_grid='gather'"):
        _cp_attention.resolve_gather_attention_body()
