"""Context parallelism must not change what the Boltz-2 stack computes.

The one-dimensional path reroutes triangle attention through ``shard_map`` and
runs triangle multiplication as an unchunked distributed contraction.  The
square-grid path keeps both pair axes tiled: triangle multiplication uses
Cannon's algorithm and triangle attention uses the Fold-CP bias/KV ring with an
online softmax, so neither operation materialises a full pair axis.

The parity checks run in subprocesses with forced CPU devices because the
device count is fixed at process start.  Grid probes run at 2x2 and 3x3: a
side of two is its own inverse under every unit shift and therefore cannot
falsify a wrong ring direction.

``jax.jit`` caches its jaxpr on the callable, while the mesh is ambient state
read at trace time.  A second ``jit`` of the same function object can replay an
unsharded executable and compare it against itself.  Every probe therefore
jits a fresh closure and records which layout the trace actually observed.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

from tests.models.cp_probe_env import inherited_environment

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

    def weight(fan_in, fan_out, gain=1.0):
        # Match ordinary fan-in scaling. The former fixed sigma=0.5 made
        # widened synthetic transitions an artificial error amplifier.
        return arr(fan_in, fan_out) * (gain / (0.5 * np.sqrt(fan_in)))

    def tri_mult():
        return {
            "norm_in": norm(C),
            "norm_out": norm(C),
            "g_in": {"kernel": weight(C, 2 * C)},
            "p_in": {"kernel": weight(C, 2 * C)},
            "p_out": {"kernel": weight(C, C)},
            "g_out": {"kernel": weight(C, C)},
        }

    def tri_att():
        return {
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

    def transition():
        return {
            "norm": norm(C),
            "fc1": {"kernel": weight(C, 4 * C)},
            "fc2": {"kernel": weight(C, 4 * C)},
            "fc3": {"kernel": weight(4 * C, C)},
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
        \"\"\"Jit a brand-new closure and record the layout its trace saw.\"\"\"

        def run(*args):
            traced.append(cp_layout())
            return fn(*args)

        return jax.jit(run)
    """
)

_PARITY_PROBE = _PREAMBLE + textwrap.dedent(
    """
    N = 13  # 13 rows over 4 shards: the 1-D padding path is exercised
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

    # cueq attention runs per-shard in the 1-D path and is accepted; pallas has
    # no distributed partitioning contract and is refused.
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
    # 12 divides both tested square grids; 13 exercises Cannon's own
    # pad-and-slice contraction path independently of ring attention.
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
    # The complete pair stack on the square grid. Both pair axes remain tiled:
    # multiplication uses Cannon and attention uses the Fold-CP ring. Thirteen
    # divides neither grid, which is the point: both schedules pad to their own
    # side and slice back, so parity here covers the padded path as well.
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
        compiled_ring = compiled(run)
        got_array = compiled_ring(z)
        got = jax.device_get(got_array)
        hlo = compiled_ring.lower(z).compiler_ir(dialect="hlo").as_hlo_text().lower()
    assert traced == [None, "2d"], traced
    np.testing.assert_allclose(ref, got, atol=3e-5, rtol=3e-5)
    assert "collective-permute" in hlo or "collective_permute" in hlo, hlo
    assert "all-gather" not in hlo and "all_gather" not in hlo, hlo

    # The trunk pins the single stream and pair state by axis name. The 2-D
    # grid calls its axes cp_row/cp_col, so a stale literal cp constraint would
    # fail before reaching any pair block.
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
            **inherited_environment(),
        },
        timeout=240,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return completed.stdout


def test_context_parallel_matches_the_unsharded_program() -> None:
    assert "CP_PARITY_OK" in _run_probe(_PARITY_PROBE)


def test_cannon_triangle_multiplication_matches_the_unsharded_form() -> None:
    """Both directions, on a mesh that splits rows and columns."""
    assert "CANNON_PARITY_OK" in _run_probe(_CANNON_PROBE)


def test_cannon_holds_on_a_three_by_three_grid() -> None:
    """The 2x2 grid is blind to every sign in the schedule."""
    assert "CANNON_PARITY_OK" in _run_probe(_CANNON_PROBE, devices=9)


@pytest.mark.parametrize("devices", [4, 9])
def test_square_grid_pair_stack_matches_without_full_axis_gather(devices: int) -> None:
    """Cannon multiplication and ring attention remain tiled in one stack."""
    assert "GRID_PARITY_OK" in _run_probe(_GRID_STACK_PROBE, devices=devices)


def test_single_shard_predict_flag_is_validated() -> None:
    from foldjax.models.boltz2 import api

    try:
        api.predict(
            seq=["ACD"], weights="/nonexistent", mols="/nonexistent", cp_devices=0
        )
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
    """``auto`` stays reproducible until square-grid GPU evidence is recorded."""
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


_PLACEMENT_PROBE = textwrap.dedent(
    """
    import numpy as np

    import jax
    import jax.numpy as jnp

    from foldjax.models._cp import (
        ATOM_FEATURE_AXES,
        PAIR_FEATURE_NAMES,
        context_parallel,
        replicate_tree,
    )
    from foldjax.models.boltz2.data.ownership import ATOM_TO_TOKEN_INDEX

    N, A = 8, 16
    feats = {}
    for name in sorted(PAIR_FEATURE_NAMES):
        if name in ("type_bonds", "pair_mask"):
            feats[name] = np.arange(N * N, dtype=np.int64).reshape(1, N, N)
        else:
            feats[name] = np.arange(N * N * 3, dtype=np.float32).reshape(1, N, N, 3)
    for name, axis in sorted(ATOM_FEATURE_AXES.items()):
        if name == ATOM_TO_TOKEN_INDEX:
            shape = [1, A]
            dtype = np.int32
        else:
            shape = [1, 1, A, 3] if axis == 2 else [1, A, 5]
            dtype = np.float32
        feats[name] = np.arange(int(np.prod(shape)), dtype=dtype).reshape(shape)
    feats["msa"] = np.zeros((1, 12, N), dtype=np.int64)
    feats["atom_to_token"] = np.zeros((1, A, N), dtype=np.float32)
    # A pair feature whose token axis does not divide the mesh: must stay
    # replicated, so the comparison below is not vacuously all-replicated.
    feats["contact_threshold"] = np.zeros((1, N + 1, N + 1), dtype=np.float32)

    def summarise(tree):
        return {
            name: (str(value.sharding), str(value.dtype), tuple(value.shape))
            for name, value in tree.items()
        }

    with context_parallel(4, layout="1d"):
        for pair, atom in ((True, True), (False, True)):
            placed = summarise(replicate_tree(
                {k: jnp.asarray(v) for k, v in feats.items()},
                shard_pair_features=pair, shard_atom_features=atom,
            ))
            host = summarise(replicate_tree(
                {k: np.asarray(v) for k, v in feats.items()},
                shard_pair_features=pair, shard_atom_features=atom,
            ))
            assert placed == host, (pair, atom)
            sharded = [k for k, v in placed.items() if "P()" not in v[0]]
            assert sharded, "nothing was sharded; the probe proves nothing"
            if pair:
                assert any(k in PAIR_FEATURE_NAMES for k in sharded)
                assert "P()" in placed["contact_threshold"][0]
    print("CP_PLACEMENT_OK")
    """
)

_FILTERED_PLACEMENT_PROBE = textwrap.dedent(
    """
    import numpy as np

    from foldjax.models._cp import context_parallel, replicate_tree
    from foldjax.models.boltz2.data.bucket import select_model_features

    N = 32
    feats = {
        "token_pad_mask": np.ones((1, N), dtype=np.float32),
        "atom_pad_mask": np.ones((1, 64), dtype=np.float32),
        "msa": np.zeros((1, 8, N), dtype=np.int32),
        "disto_target": np.zeros((1, N, N, 1, 64), dtype=np.float32),
        "writer_only": np.zeros((1, 17), dtype=np.float32),
    }
    filtered = select_model_features(feats)
    assert "disto_target" not in filtered
    assert "writer_only" not in filtered

    def physical_bytes(tree):
        return sum(
            shard.data.size * shard.data.dtype.itemsize
            for value in tree.values()
            for shard in value.addressable_shards
        )

    with context_parallel(4, layout="1d"):
        raw = replicate_tree(feats)
        placed = replicate_tree(filtered)
    assert physical_bytes(placed) < physical_bytes(raw)
    assert set(placed) == set(filtered)
    print("CP_DEAD_FEATURE_FILTER_OK")
    """
)

_PADDING_RNG_PROBE = textwrap.dedent(
    """
    import jax
    import jax.numpy as jnp
    import numpy as np

    from foldjax.models._cp import context_parallel
    from foldjax.models._cp_atom import shard_atoms
    from foldjax.models.boltz2.api import _prefix_stable_noise_tape
    from foldjax.models.boltz2.models.trunk_blocks.trunk import _prefix_atom_normal

    jax.config.update("jax_threefry_partitionable", True)
    key = jax.random.PRNGKey(17)
    multiplicity, storage_atoms, target_atoms = 3, 5, 32
    expected = _prefix_stable_noise_tape(
        key,
        multiplicity=multiplicity,
        storage_atoms=storage_atoms,
        target_atoms=target_atoms,
        steps=1,
    )[0]
    _, init_key = jax.random.split(key)

    for layout in ("1d", "2d"):
        with context_parallel(4, layout=layout):
            run = jax.jit(
                lambda draw_key, storage: shard_atoms(
                    _prefix_atom_normal(
                        draw_key,
                        storage,
                        multiplicity=multiplicity,
                        target_atoms=target_atoms,
                        dtype=jnp.float32,
                    ),
                    atom_axis=1,
                )
            )
            actual = run(init_key, jnp.asarray(storage_atoms, dtype=jnp.int32))
            executable = run.lower(
                init_key, jnp.asarray(storage_atoms, dtype=jnp.int32)
            ).compile()
            hlo = executable.runtime_executable().hlo_modules()[0].to_string().lower()
        np.testing.assert_array_equal(
            np.asarray(actual).view(np.uint8), np.asarray(expected).view(np.uint8)
        )
        assert "all-gather(" not in hlo
        assert "all-reduce(" not in hlo
        assert "collective-permute(" not in hlo

    print("CP_PADDING_RNG_OK")
    """
)


def test_cp_placement_is_the_same_from_numpy_and_from_placed_leaves() -> None:
    """`predict` hands the compiled path NumPy features; CP still gets placed ones.

    The gate is `place=steering_active or cp_devices > 1`, so context
    parallelism never sees a NumPy leaf today. But `replicate_tree` decides per
    leaf via `_is_movable_array`, a predicate leaf *type* has broken twice
    (bfloat16 reports dtype kind 'V'; typed PRNG keys report no kind at all).
    A silent flip from "placed on the mesh" to "replicated" would fail no test
    and would not move a single-device HLO hash, so pin that the walk is
    type-invariant rather than relying on the gate alone.
    """
    assert "CP_PLACEMENT_OK" in _run_probe(_PLACEMENT_PROBE)


def test_cp_never_places_features_dead_to_the_nonsteering_graph() -> None:
    assert "CP_DEAD_FEATURE_FILTER_OK" in _run_probe(_FILTERED_PLACEMENT_PROBE)


def test_cp_storage_prefix_rng_is_exact_without_new_collectives() -> None:
    assert "CP_PADDING_RNG_OK" in _run_probe(_PADDING_RNG_PROBE)
