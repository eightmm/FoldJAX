"""Context parallelism must not change what the Boltz-2 stack computes.

Same contract as the OpenDDE test: the CP path reroutes triangle attention
through ``shard_map``, forces the triangle multiplication onto the unchunked
XLA einsum, and disables the row chunks that would slice the sharded axis --
so the property to hold is numerical parity against the unsharded program on
a mesh whose size does not divide the token count. Under the 2-D layout the
multiplication is rerouted again, onto Cannon's algorithm, while attention
stays row-sharded. The parity checks run in a subprocess with forced CPU
devices because the device count is fixed at process start; the grid probes
run at 2x2 and again at 3x3, since a side of two is its own inverse under
every shift in the schedule and so cannot falsify a sign.

``jax.jit`` caches its jaxpr on the callable, and the mesh lives in a module
global that the trace reads, so a second ``jit`` of the same function object
replays the unsharded program and compares it against itself. Written that way
these probes passed against a triangle contraction whose max absolute error
exceeded the result's own scale. Clearing the cache fixes that, but nothing
then checks it stayed fixed, so every probe here also jits a freshly built
closure and asserts, through the ``traced`` witness, which layout each trace
actually saw. The witness is the part that makes a green run evidence.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

_PREAMBLE = textwrap.dedent(
    """
    import os
    os.environ["BOLTZ_JAX_TRIANGLE_MULTIPLICATION_BACKEND"] = "xla"

    import jax
    import jax.numpy as jnp
    import numpy as np
    import pytest

    from foldjax.models._cp import context_parallel, cp_layout
    from foldjax.models.boltz2.models.trunk_blocks.pairformer_noseq import (
        pairformer_no_seq_module_forward,
    )
    from foldjax.models.boltz2.models.triangle.triangle import (
        triangle_multiplication_forward,
    )
    from foldjax.models.boltz2.models.triangle.triangle_attention import (
        triangle_attention_forward,
    )

    # A 2x2 grid cannot falsify a sign: (x + 1) % 2 == (x - 1) % 2, so every
    # skew and every ring hop in the schedule is its own inverse there. The
    # probes therefore take their grid size from the harness and are run at 3x3
    # as well, where the signs are pinned.
    DEVICES = int(os.environ["FOLDJAX_CP_PROBE_DEVICES"])
    assert jax.device_count() == DEVICES, jax.devices()

    C, HEADS = 8, 2
    rng = np.random.default_rng(0)

    def arr(*shape):
        return jnp.asarray(rng.normal(size=shape, scale=0.5), dtype=jnp.float32)

    def norm(c):
        return {"scale": arr(c) * 0.1 + 1.0, "bias": arr(c) * 0.1}

    def tri_mult():
        return {
            "norm_in": norm(C),
            "norm_out": norm(C),
            "g_in": {"kernel": arr(C, 2 * C)},
            "p_in": {"kernel": arr(C, 2 * C)},
            "p_out": {"kernel": arr(C, C)},
            "g_out": {"kernel": arr(C, C)},
        }

    def tri_att():
        return {
            "layer_norm": norm(C),
            "linear": {"kernel": arr(C, HEADS)},
            "mha": {
                "linear_q": {"kernel": arr(C, C)},
                "linear_k": {"kernel": arr(C, C)},
                "linear_v": {"kernel": arr(C, C)},
                "linear_g": {"kernel": arr(C, C)},
                "linear_o": {"kernel": arr(C, C)},
            },
        }

    def transition():
        return {
            "norm": norm(C),
            "fc1": {"kernel": arr(C, 4 * C)},
            "fc2": {"kernel": arr(C, 4 * C)},
            "fc3": {"kernel": arr(4 * C, C)},
        }

    def layer():
        return {
            "tri_mul_out": tri_mult(),
            "tri_mul_in": tri_mult(),
            "tri_att_start": tri_att(),
            "tri_att_end": tri_att(),
            "transition_z": transition(),
        }

    def pair_mask_for(n):
        keep = rng.random(n) > 0.15
        return jnp.asarray((keep[:, None] & keep[None, :])[None]).astype(jnp.float32)

    traced = []

    def compiled(fn):
        \"\"\"Jit a brand-new closure and record the layout its trace saw.

        Two independent guards against replaying the unsharded graph: the
        wrapper is a fresh function object every call, so `jax.jit` has nothing
        to hit, and `traced` records what `cp_layout()` actually returned while
        the body was being traced.
        \"\"\"

        def run(*args):
            traced.append(cp_layout())
            return fn(*args)

        return jax.jit(run)
    """
)

_PARITY_PROBE = _PREAMBLE + textwrap.dedent(
    """
    N = 13  # 13 rows over 4 shards: padding path exercised
    params = {"layers": [layer(), layer()]}
    z = arr(1, N, N, C)
    pair_mask = pair_mask_for(N)

    def run(z_in):
        return pairformer_no_seq_module_forward(
            params, z_in, pair_mask, triangle_backend="xla"
        )

    ref = jax.device_get(compiled(run)(z))
    with context_parallel(DEVICES):
        got = jax.device_get(compiled(run)(z))
    assert traced == [None, "1d"], traced
    np.testing.assert_allclose(ref, got, atol=3e-5, rtol=3e-5)

    # cueq attention runs per-shard inside `shard_map` and is accepted; the
    # pallas flash kernel has no per-shard story here and is refused.
    with context_parallel(DEVICES):
        with pytest.raises(ValueError, match="supports"):
            triangle_attention_forward(
                tri_att(), z, None, triangle_backend="pallas"
            )

    print("CP_PARITY_OK")
    """
)

_CANNON_PROBE = _PREAMBLE + textwrap.dedent(
    """
    # 12 divides the 2x2 grid on both pair axes; 13 forces the pad-and-slice
    # path, which is where a ring schedule fails separately from its algebra.
    for N in (12, 13):
        params = tri_mult()
        z = arr(1, N, N, C)
        pair_mask = pair_mask_for(N)
        for direction in ("outgoing", "incoming"):

            def run(z_in, d=direction):
                return triangle_multiplication_forward(
                    params, z_in, pair_mask, d, chunk_size=0
                )

            traced.clear()
            ref = jax.device_get(compiled(run)(z))
            with context_parallel(DEVICES, layout="2d"):
                got = jax.device_get(compiled(run)(z))
            assert traced == [None, "2d"], (N, direction, traced)
            np.testing.assert_allclose(
                ref, got, atol=3e-5, rtol=3e-5,
                err_msg=f"N={N} direction={direction}",
            )
    print("CANNON_PARITY_OK")
    """
)

_GRID_STACK_PROBE = _PREAMBLE + textwrap.dedent(
    """
    # The whole pair stack on the square grid. Triangle multiplication runs
    # Cannon; triangle attention stays row-sharded by design, because its
    # softmax spans a whole column axis, so its `shard_map` asks for whole rows
    # and the partitioner gathers the column axis around it.
    N = 13
    params = {"layers": [layer(), layer()]}
    z = arr(1, N, N, C)
    pair_mask = pair_mask_for(N)

    def run(z_in):
        return pairformer_no_seq_module_forward(
            params, z_in, pair_mask, triangle_backend="xla"
        )

    ref = jax.device_get(compiled(run)(z))
    with context_parallel(DEVICES, layout="2d"):
        got = jax.device_get(compiled(run)(z))
    assert traced == [None, "2d"], traced
    np.testing.assert_allclose(ref, got, atol=3e-5, rtol=3e-5)

    # The trunk pins the single stream and the pair rows itself, by axis
    # *name*. The 2-D grid calls its axes `cp_row`/`cp_col`, so a trunk still
    # asking for `cp` would build a constraint against a mesh axis that does
    # not exist -- a hard failure, and one no pair-block test would reach.
    from foldjax.models._cp import CP_ROW_AXIS
    from foldjax.models.boltz2.models.trunk_blocks.trunk import (
        _shard_pair, _shard_single,
    )

    with context_parallel(DEVICES, layout="2d") as mesh:
        def pin(s_in, z_in):
            return (
                _shard_single(s_in, mesh, CP_ROW_AXIS, True),
                _shard_pair(z_in, mesh, CP_ROW_AXIS, True),
            )

        pinned_s, pinned_z = jax.jit(pin)(arr(1, N, C), z)
        assert pinned_s.shape == (1, N, C), pinned_s.shape
        assert pinned_z.shape == z.shape, pinned_z.shape

    print("GRID_PARITY_OK")
    """
)


def _run_probe(source: str, devices: int = 4) -> str:
    completed = subprocess.run(
        [sys.executable, "-c", source],
        capture_output=True,
        text=True,
        env={
            "JAX_PLATFORMS": "cpu",
            "XLA_FLAGS": f"--xla_force_host_platform_device_count={devices}",
            "FOLDJAX_CP_PROBE_DEVICES": str(devices),
            "PATH": "/usr/bin",
        },
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return completed.stdout


def test_context_parallel_matches_the_unsharded_program() -> None:
    assert "CP_PARITY_OK" in _run_probe(_PARITY_PROBE)


def test_cannon_triangle_multiplication_matches_the_unsharded_form() -> None:
    """Both directions, on a mesh that splits rows *and* columns."""
    assert "CANNON_PARITY_OK" in _run_probe(_CANNON_PROBE)


def test_cannon_holds_on_a_three_by_three_grid() -> None:
    """The 2x2 grid is blind to every sign in the schedule.

    Each skew shifts by the device's own coordinate and each ring hop by one,
    so on a side of two ``(x + 1) % 2 == (x - 1) % 2`` and a schedule that
    skewed or hopped the wrong way is indistinguishable from the right one.
    A side of three separates them, which is what pins ``row_skew_perm`` /
    ``col_skew_perm`` / ``ring_perm`` as correct rather than as coincidence.
    Nine forced CPU devices; N=12 divides three, N=13 pads to fifteen.
    """
    assert "CANNON_PARITY_OK" in _run_probe(_CANNON_PROBE, devices=9)


def test_square_grid_pair_stack_matches_the_unsharded_program() -> None:
    """Cannon multiplication and row-sharded attention in one stack."""
    assert "GRID_PARITY_OK" in _run_probe(_GRID_STACK_PROBE)


def test_single_shard_predict_flag_is_validated() -> None:
    from foldjax.models.boltz2 import api

    try:
        api.predict(seq=["ACD"], weights="/nonexistent", mols="/nonexistent",
                    cp_devices=0)
    except ValueError as error:
        assert "cp_devices" in str(error)
    else:  # pragma: no cover - the guard must fire before any file access
        raise AssertionError("cp_devices=0 was accepted")


@pytest.mark.parametrize(
    ("layout", "devices", "expected"),
    [
        ("auto", 1, "1d"),
        ("auto", 2, "1d"),
        ("auto", 4, "1d"),
        ("auto", 9, "1d"),
        ("1d", 4, "1d"),
        ("2d", 4, "2d"),
    ],
)
def test_cp_layout_resolution(layout: str, devices: int, expected: str) -> None:
    """"auto" must not silently change the program anyone has measured.

    The grid is the better layout and is gated above, but every published
    number for this feature came from the 1-D layout, so "auto" stays there
    until the grid has its own GPU evidence. Flipping the default should
    require editing this test.
    """
    from foldjax.models.boltz2.api import _resolve_cp_layout

    assert _resolve_cp_layout(layout, devices) == expected


def test_square_layout_needs_a_square_device_count() -> None:
    """Refused at the entry, before featurization pays for an MSA search."""
    from foldjax.models.boltz2.api import _resolve_cp_layout

    with pytest.raises(ValueError, match="square cp_devices"):
        _resolve_cp_layout("2d", 3)
    with pytest.raises(ValueError, match="square cp_devices"):
        _resolve_cp_layout("2d", 1)
    with pytest.raises(ValueError, match="must be one of"):
        _resolve_cp_layout("grid", 4)
