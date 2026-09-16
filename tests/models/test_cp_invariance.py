"""Context parallelism must be invisible to a program that is not using it.

The active mesh lives in a ``ContextVar`` that any traced function may read, so
every serial run in a process that has ever entered a mesh is exposed to two
failures the numbers alone do not show: a program that quietly keeps a shard
annotation or a collective it never asked for, and a program that is served an
executable compiled for a topology that is no longer active. Both produce
finite, plausible coordinates.

These gates pin the invariance rather than the arithmetic. The fixture is one
small pair contraction, deliberately layout-aware in the same way the runtime's
own ring code is: ``with_sharding_constraint`` lowers no collective at all
before SPMD partitioning and its trace sees only global shapes, so the sharded
branch goes through ``shard_map`` and ``psum``. That is what puts a collective
in the unoptimised HLO and what lets the tripwire read the *local* shard shape
of each arm -- without it, "the sharded variant ran" would be an assertion
about a program that never shards anything.

A forced device count has to be set before JAX initialises, so the gates run in
a subprocess with four CPU devices: enough for the 1-D layout and for the only
small perfect square the 2-D layout accepts. Probe sources use ``#`` comments
rather than docstrings because they live inside a triple-quoted literal.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

from tests.models.cp_probe_env import inherited_environment

#: Four devices give a 4-shard 1-D mesh and a 2x2 grid. The probes spell out
#: the local shard shapes both layouts must produce, so they assert this count.
_DEVICES = 4

_PREAMBLE = textwrap.dedent(
    r"""
    import hashlib
    import os
    import re

    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import NamedSharding, PartitionSpec

    from foldjax.models._cp import (
        CP_AXIS,
        CP_COL_AXIS,
        CP_ROW_AXIS,
        context_parallel,
        cp_identity,
        cp_layout,
        cp_mesh,
        cp_shards,
        pair_spec,
        shard_pair_rows,
        shard_single,
        single_spec,
    )

    DEVICES = int(os.environ["FOLDJAX_CP_PROBE_DEVICES"])
    assert jax.device_count() == DEVICES, jax.devices()
    # The expected local shard shapes below are written out for this count.
    assert DEVICES == 4, DEVICES

    SERIAL_IDENTITY = ("serial", 1, (1, 1), ())
    N, C = 8, 4
    GLOBAL_WITNESS = (None, (N, N, C), (N, C))

    rng = np.random.default_rng(20260915)
    Z = jnp.asarray(rng.normal(size=(N, N, C), scale=0.5), dtype=jnp.float32)
    S = jnp.asarray(rng.normal(size=(N, C), scale=0.5), dtype=jnp.float32)

    COLLECTIVES = (
        "all-gather",
        "all_gather",
        "all-reduce",
        "all_reduce",
        "all-to-all",
        "all_to_all",
        "collective-permute",
        "collective_permute",
        "ppermute",
        "reduce-scatter",
        "reduce_scatter",
    )


    # One pair contraction, plus a tripwire the trace has to fire.
    # `out[j, c] = sum_i z[i, j, c] * s[i, c]` contracts the axis both layouts
    # shard, so the distributed form needs a real reduction across devices
    # rather than a relabelled local one.
    def build(witness):
        def program(z_in, s_in):
            z_in = shard_pair_rows(z_in)
            s_in = shard_single(s_in)
            mesh = cp_mesh()
            if mesh is None:
                witness.append(
                    (cp_layout(), tuple(z_in.shape), tuple(s_in.shape))
                )
                return jnp.einsum("ijc,ic->jc", z_in, s_in)
            layout = cp_layout()
            rows = CP_ROW_AXIS if layout == "2d" else CP_AXIS
            out_spec = (
                PartitionSpec(CP_COL_AXIS, None)
                if layout == "2d"
                else PartitionSpec()
            )

            def local(z_l, s_l):
                witness.append(
                    (cp_layout(), tuple(z_l.shape), tuple(s_l.shape))
                )
                return jax.lax.psum(
                    jnp.einsum("ijc,ic->jc", z_l, s_l), axis_name=rows
                )

            return jax.shard_map(
                local,
                mesh=mesh,
                in_specs=(pair_spec(z_in.ndim), single_spec(s_in.ndim)),
                out_specs=out_spec,
            )(z_in, s_in)

        return program


    # A fresh closure and its tripwire. `jax.jit` keys its cache on the
    # callable, and the mesh is a context variable the trace reads rather than
    # an argument, so reusing one callable across a topology change replays
    # whatever it compiled first. Every arm below therefore gets its own
    # closure; the tripwire is what proves the arm actually traced.
    def variant():
        witness = []
        return witness, jax.jit(build(witness))


    # Lowered text minus the names that carry no program content.
    def normalised(text):
        return re.sub(r"\s+", " ", re.sub(r"jit_[A-Za-z0-9_]+", "jit_fn", text))


    # Normalised (StableHLO, HLO) text of one lowering.
    def lowered(fn):
        low = fn.lower(Z, S)
        return (
            normalised(low.as_text()),
            normalised(low.compiler_ir(dialect="hlo").as_hlo_text()),
        )


    def digest(text):
        return hashlib.sha256(text.encode()).hexdigest()[:16]


    def collectives(text):
        lowered_text = text.lower()
        return sorted({name for name in COLLECTIVES if name in lowered_text})
    """
)

_SERIAL_IDENTITY_PROBE = _PREAMBLE + textwrap.dedent(
    r"""
    default_identity = cp_identity()
    assert default_identity == SERIAL_IDENTITY, default_identity
    serial_witness, serial_fn = variant()
    serial_stablehlo, serial_hlo = lowered(serial_fn)
    serial_out = np.asarray(serial_fn(Z, S))

    with context_parallel(1) as mesh:
        # A one-device request is a null context: it yields nothing and never
        # sets the runtime, so the getters must stay exactly at their defaults.
        assert mesh is None
        assert cp_mesh() is None
        assert cp_layout() is None
        assert cp_shards() == 1
        assert cp_identity() == default_identity, cp_identity()
        null_witness, null_fn = variant()
        null_stablehlo, null_hlo = lowered(null_fn)
        null_out = np.asarray(null_fn(Z, S))

    assert serial_witness == [GLOBAL_WITNESS], serial_witness
    assert null_witness == [GLOBAL_WITNESS], null_witness
    assert null_stablehlo == serial_stablehlo
    assert null_hlo == serial_hlo
    # Bitwise, not allclose: the same program on the same inputs.
    np.testing.assert_array_equal(serial_out, null_out)

    for text in (serial_stablehlo, serial_hlo):
        assert collectives(text) == [], collectives(text)
        assert "sharding" not in text.lower(), text
        assert "sdy" not in text.lower(), text
    assert "mhlo.num_partitions = 1 : i32" in serial_stablehlo, serial_stablehlo

    print("SERIAL_IDENTITY_OK")
    """
)

_ROUND_TRIP_PROBE = _PREAMBLE + textwrap.dedent(
    r"""
    before_witness, before_fn = variant()
    before_stablehlo, before_hlo = lowered(before_fn)
    before_out = np.asarray(before_fn(Z, S))
    assert before_witness == [GLOBAL_WITNESS], before_witness

    arms = {}
    for layout in ("1d", "2d"):
        with context_parallel(DEVICES, layout=layout) as mesh:
            assert mesh is not None
            assert cp_layout() == layout
            witness, fn = variant()
            stablehlo, hlo = lowered(fn)
            out = np.asarray(fn(Z, S))
            arms[layout] = (witness, stablehlo, hlo, out)
        # Leaving restores the defaults, whatever the layout was.
        assert cp_mesh() is None
        assert cp_layout() is None
        assert cp_shards() == 1
        assert cp_identity() == SERIAL_IDENTITY, cp_identity()

    after_witness, after_fn = variant()
    after_stablehlo, after_hlo = lowered(after_fn)
    after_out = np.asarray(after_fn(Z, S))

    # The serial program on the far side of two meshes is the one from before.
    assert after_witness == [GLOBAL_WITNESS], after_witness
    assert after_stablehlo == before_stablehlo
    assert after_hlo == before_hlo
    np.testing.assert_array_equal(before_out, after_out)
    for text in (after_stablehlo, after_hlo):
        assert collectives(text) == [], collectives(text)
        assert "sharding" not in text.lower(), text

    # ... and the round trip went somewhere: each arm traced its own layout,
    # lowered a collective, and computed the same contraction.
    for layout, (witness, _, hlo, out) in arms.items():
        assert [entry[0] for entry in witness] == [layout], witness
        assert collectives(hlo), hlo
        np.testing.assert_allclose(out, before_out, atol=1e-5, rtol=1e-5)

    print("ROUND_TRIP_OK")
    """
)

_REUSED_CALLABLE_PROBE = _PREAMBLE + textwrap.dedent(
    r"""
    # Why the gates above build a fresh closure per arm. JAX cannot see this
    # project's context variable, so one jitted callable carried across a
    # topology change is served the executable it compiled first -- in both
    # directions. Model entry points guard this by taking the topology as a
    # static argument and refusing a mismatch (`cp_shards` in
    # opendde/models/model.py, protenix/models/model.py and
    # openfold3/inference.py; `cp_identity()` as the separately compiled
    # transition's static identity in protenix/.../primitives.py). The runtime
    # by itself cannot, and a gate that reuses a callable measures the first
    # arm twice.
    witness, reused = variant()
    with context_parallel(DEVICES, layout="1d") as mesh:
        inside = reused(Z, S)
        inside_hlo = lowered(reused)[1]
        cp_axes = mesh.axis_names
    local = ("1d", (N // DEVICES, N, C), (N // DEVICES, C))
    assert witness == [local], witness
    assert cp_mesh() is None

    outside = reused(Z, S)
    outside_hlo = lowered(reused)[1]
    # No retrace, no new program: the mesh executable is still what runs.
    assert witness == [local], witness
    assert outside_hlo == inside_hlo
    assert collectives(outside_hlo), outside_hlo
    assert isinstance(outside.sharding, NamedSharding), outside.sharding
    assert outside.sharding.mesh.axis_names == cp_axes, outside.sharding
    np.testing.assert_array_equal(np.asarray(inside), np.asarray(outside))

    # The mirror image: a callable first traced serially stays unsharded under
    # an active mesh, which is the silently-unsharded run the CP gates exist
    # to prevent.
    serial_witness, serial_reused = variant()
    first = serial_reused(Z, S)
    assert serial_witness == [GLOBAL_WITNESS], serial_witness
    with context_parallel(DEVICES, layout="1d"):
        under_mesh = serial_reused(Z, S)
        assert serial_witness == [GLOBAL_WITNESS], serial_witness
        assert collectives(lowered(serial_reused)[1]) == []
        assert under_mesh.sharding == first.sharding, under_mesh.sharding
        # What the static topology argument keys on, and the callable does not.
        assert cp_identity() != SERIAL_IDENTITY, cp_identity()

    assert cp_identity() == SERIAL_IDENTITY, cp_identity()
    print("REUSED_CALLABLE_OK")
    """
)

_EXCEPTION_PROBE = _PREAMBLE + textwrap.dedent(
    r"""
    class Boom(RuntimeError):
        pass


    try:
        with context_parallel(DEVICES, layout="2d") as mesh:
            assert mesh is not None
            assert cp_identity() == (
                "2d",
                DEVICES,
                (2, 2),
                (CP_ROW_AXIS, CP_COL_AXIS),
            ), cp_identity()
            raise Boom("raised inside the context body")
    except Boom:
        pass
    else:
        raise AssertionError("the body's exception did not propagate")

    assert cp_mesh() is None
    assert cp_layout() is None
    assert cp_shards() == 1
    assert cp_identity() == SERIAL_IDENTITY, cp_identity()

    # A token that was not reset would make the next entry look like nesting,
    # which is the observable difference between restoring the variable and
    # merely leaving the mesh unreferenced.
    with context_parallel(DEVICES, layout="1d") as mesh:
        assert mesh is not None
        assert cp_identity() == ("1d", DEVICES, (DEVICES, 1), (CP_AXIS,))
    assert cp_mesh() is None

    witness, fn = variant()
    stablehlo, hlo = lowered(fn)
    assert witness == [GLOBAL_WITNESS], witness
    assert collectives(hlo) == [], collectives(hlo)
    assert "sharding" not in stablehlo.lower(), stablehlo

    print("EXCEPTION_RESTORE_OK")
    """
)

_NESTING_PROBE = _PREAMBLE + textwrap.dedent(
    r"""
    with context_parallel(DEVICES, layout="2d") as outer:
        outer_identity = cp_identity()

        # A nominal one-device context inside a distributed one is refused
        # rather than inherited: `context_parallel` yields the mesh it
        # activated, and a nested serial context would have to yield None
        # while `cp_mesh()` still reported the outer mesh. Refusing keeps the
        # yielded value and the getters one decision.
        try:
            with context_parallel(1):
                raise AssertionError("a nested serial context was accepted")
        except RuntimeError as error:
            assert "does not nest" in str(error), error

        # The refusal must leave the mesh already in force untouched.
        assert cp_mesh() is outer
        assert cp_layout() == "2d"
        assert cp_shards() == DEVICES
        assert cp_identity() == outer_identity, cp_identity()

        witness, fn = variant()
        fn(Z, S)
        assert witness == [("2d", (N // 2, N // 2, C), (N // 2, C))], witness

        try:
            with context_parallel(DEVICES, layout="1d"):
                raise AssertionError("a nested mesh was accepted")
        except RuntimeError as error:
            assert "does not nest" in str(error), error
        assert cp_mesh() is outer
        assert cp_identity() == outer_identity, cp_identity()

    assert cp_mesh() is None
    assert cp_identity() == SERIAL_IDENTITY, cp_identity()

    print("NESTING_OK")
    """
)

_NON_VACUOUS_PROBE = _PREAMBLE + textwrap.dedent(
    r"""
    serial_witness, serial_fn = variant()
    serial_stablehlo, serial_hlo = lowered(serial_fn)
    serial_out = np.asarray(serial_fn(Z, S))
    assert serial_witness == [GLOBAL_WITNESS], serial_witness
    assert "mhlo.num_partitions = 1 : i32" in serial_stablehlo, serial_stablehlo

    # Layout, the local shard shapes the tripwire must see inside the
    # `shard_map` body, the mesh the lowering has to name, and the shard shape
    # of the answer.
    expected = (
        (
            "1d",
            ((N // DEVICES, N, C), (N // DEVICES, C)),
            '<["cp"=4]>',
            (N, C),
        ),
        (
            "2d",
            ((N // 2, N // 2, C), (N // 2, C)),
            '<["cp_row"=2, "cp_col"=2]>',
            (N // 2, C),
        ),
    )

    arms = {}
    for layout, local_shapes, mesh_text, out_shards in expected:
        with context_parallel(DEVICES, layout=layout):
            witness, fn = variant()
            stablehlo, hlo = lowered(fn)
            out = fn(Z, S)
            assert witness == [
                (layout, local_shapes[0], local_shapes[1])
            ], witness
            assert "all-reduce" in hlo, hlo
            assert "sdy.sharding_constraint" in stablehlo, stablehlo
            assert mesh_text in stablehlo, stablehlo
            partitions = f"mhlo.num_partitions = {DEVICES} : i32"
            assert partitions in stablehlo, stablehlo
            shards = out.sharding.shard_shape(out.shape)
            assert shards == out_shards, shards
            np.testing.assert_allclose(
                np.asarray(out), serial_out, atol=1e-5, rtol=1e-5
            )
            arms[layout] = (stablehlo, hlo)

    # Three distinct programs, not one program measured three times.
    stablehlo_digests = {
        digest(serial_stablehlo),
        digest(arms["1d"][0]),
        digest(arms["2d"][0]),
    }
    hlo_digests = {
        digest(serial_hlo),
        digest(arms["1d"][1]),
        digest(arms["2d"][1]),
    }
    assert len(stablehlo_digests) == 3, stablehlo_digests
    assert len(hlo_digests) == 3, hlo_digests

    print("NON_VACUOUS_OK")
    """
)


_PORT_AUTO_LAYOUT_PROBE = _PREAMBLE + textwrap.dedent(
    r"""
    # What `cp_layout="auto"` actually builds, per port, read from each port's
    # own resolver rather than from a copy of the rule. OpenDDE, Boltz-2 and
    # OpenFold3 pick the square grid on a perfect-square device count, each on
    # its own four-card measurement (4 x 96 GiB, 2x2). A 2,096-token 5DEI
    # completes at 32.1 GiB per device on OpenDDE where serial, 1-D on two
    # cards and 1-D on four cards all run out of memory, and at 16.6 GiB on
    # Boltz-2 against 19.0 in the 1-D layout. OpenFold3's target is a
    # 6,568-token one: the grid completes it at 42,209 MiB per device where a
    # single card runs out of memory and 1-D on four cards runs out on every
    # rank, each asking for a 101 GiB arena. Protenix measured better on the
    # 1-D mesh there (10.7 against 11.6 GiB per device), so it keeps rows. The
    # grid is the slower program, chosen for the memory ceiling.
    from foldjax.models._cp import resolve_cp_layout as shared_resolver
    from foldjax.models.boltz2.api import _resolve_cp_layout as boltz2_resolver
    from foldjax.models.opendde.models.model import (
        _resolve_cp_layout as opendde_resolver,
    )
    from foldjax.models.openfold3.inference import InferenceConfig
    from foldjax.models.openfold3.inference import (
        resolve_cp_layout as openfold3_resolver,
    )
    from foldjax.padding import square_grid_auto_layout


    def openfold3(requested, devices):
        return openfold3_resolver(
            InferenceConfig(
                n_token=1, n_atom=1, n_query=1, n_key=1, atom_heads=1,
                token_heads=1, no_heads_msa=1, no_heads_pair=1,
                no_heads_pair_bias=1, max_relative_idx=1, max_relative_chain=1,
                num_recycles=1, num_samples=1, max_atoms_per_token=1,
                plddt_bins=1, pae_bins=1, pae_bin_max=1.0, num_steps=1,
                cp_shards=devices, cp_layout=requested,
            )
        )


    GRID_PORTS = {
        "opendde": opendde_resolver,
        "boltz2": boltz2_resolver,
        "openfold3": openfold3,
    }
    ROW_PORTS = {
        # Protenix hands `auto` to `context_parallel` unexpanded, so the
        # shared resolver's own default is this port's rule.
        "protenix": shared_resolver,
    }
    GRID_IDENTITY = ("2d", DEVICES, (2, 2), (CP_ROW_AXIS, CP_COL_AXIS))
    ROW_IDENTITY = ("1d", DEVICES, (DEVICES, 1), (CP_AXIS,))

    # The rule, including counts this four-device process cannot build.
    for devices, grid in ((1, "1d"), (2, "1d"), (3, "1d"), (4, "2d"), (9, "2d")):
        assert square_grid_auto_layout(devices) == grid, devices
        for name, resolver in GRID_PORTS.items():
            assert resolver("auto", devices) == grid, (name, devices)
            # An explicit spelling is never touched by the flip.
            assert resolver("1d", devices) == "1d", (name, devices)
        for name, resolver in ROW_PORTS.items():
            assert resolver("auto", devices) == "1d", (name, devices)

    # And the mesh each of those answers builds.
    for name, resolver in GRID_PORTS.items():
        with context_parallel(DEVICES, layout=resolver("auto", DEVICES)) as mesh:
            assert mesh is not None, name
            assert cp_identity() == GRID_IDENTITY, (name, cp_identity())
        with context_parallel(DEVICES, layout=resolver("1d", DEVICES)):
            assert cp_identity() == ROW_IDENTITY, (name, cp_identity())
    for name, resolver in ROW_PORTS.items():
        with context_parallel(DEVICES, layout=resolver("auto", DEVICES)) as mesh:
            assert mesh is not None, name
            assert cp_identity() == ROW_IDENTITY, (name, cp_identity())

    # Two devices are not a square, so even a grid port's `auto` is rows
    # there: a 2x1 mesh, and never a shape the ring schedules cannot accept.
    TWO_ROW_IDENTITY = ("1d", 2, (2, 1), (CP_AXIS,))
    for name, resolver in {**GRID_PORTS, **ROW_PORTS}.items():
        assert resolver("auto", 2) == "1d", name
        with context_parallel(2, layout=resolver("auto", 2)) as mesh:
            assert mesh is not None, name
            assert cp_identity() == TWO_ROW_IDENTITY, (name, cp_identity())

    # One device is the serial program on every port: the layout decides
    # nothing without a mesh, and `auto` must not turn a one-card run into a
    # request for a grid that cannot exist.
    for name, resolver in {**GRID_PORTS, **ROW_PORTS}.items():
        with context_parallel(1, layout=resolver("auto", 1)) as mesh:
            assert mesh is None, name
            assert cp_identity() == SERIAL_IDENTITY, (name, cp_identity())

    print("PORT_AUTO_LAYOUT_OK")
    """
)


def _run_probe(source: str) -> str:
    completed = subprocess.run(
        [sys.executable, "-c", source],
        capture_output=True,
        text=True,
        env={
            "JAX_PLATFORMS": "cpu",
            "XLA_FLAGS": f"--xla_force_host_platform_device_count={_DEVICES}",
            "FOLDJAX_CP_PROBE_DEVICES": str(_DEVICES),
            **inherited_environment(),
        },
        timeout=300,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return completed.stdout


def test_a_one_device_context_compiles_the_serial_program() -> None:
    """`context_parallel(1)` must be indistinguishable from no context.

    Identical outputs are not enough: a null context that annotated or
    gathered anything would still produce them on one device.
    """

    assert "SERIAL_IDENTITY_OK" in _run_probe(_SERIAL_IDENTITY_PROBE)


def test_a_mesh_leaves_no_trace_in_the_next_serial_program() -> None:
    """Serial -> 1-D -> 2-D -> serial, in one process.

    The mesh lives in a context variable, so the risk is residue: the serial
    lowering after two meshes must be the one from before them, with no
    sharding annotation and no collective.
    """

    assert "ROUND_TRIP_OK" in _run_probe(_ROUND_TRIP_PROBE)


def test_one_callable_carried_across_a_topology_change_is_stale() -> None:
    """The runtime on its own cannot invalidate a JAX cache.

    Pinned because it is the trap every gate in this file is written around,
    and because it is what the model entry points' static topology arguments
    are for.
    """

    assert "REUSED_CALLABLE_OK" in _run_probe(_REUSED_CALLABLE_PROBE)


def test_an_exception_inside_the_context_restores_the_runtime() -> None:
    """A mesh that survived a failed request would leak into the next one."""

    assert "EXCEPTION_RESTORE_OK" in _run_probe(_EXCEPTION_PROBE)


def test_a_nested_context_is_refused_and_changes_nothing() -> None:
    """The nominal one-device case included, outer mesh undisturbed."""

    assert "NESTING_OK" in _run_probe(_NESTING_PROBE)


def test_both_layouts_lower_to_their_own_sharded_program() -> None:
    """Serial, 1-D and 2-D must be three programs with three shard shapes.

    An invariance suite whose CP arms secretly lowered the serial program
    would pass every assertion in this file.
    """

    assert "NON_VACUOUS_OK" in _run_probe(_NON_VACUOUS_PROBE)


def test_each_port_resolves_auto_to_the_mesh_it_has_evidence_for() -> None:
    """`auto` is a per-port decision, and this is where it is pinned.

    OpenDDE, Boltz-2 and OpenFold3 resolve it to the square grid on a
    perfect-square device count, Protenix to the row mesh, every port to the
    row mesh on a non-square count, and a one-device request to the serial
    program everywhere. Read from the ports' own resolvers and then from the
    mesh those answers build, because a rule that agreed with itself while
    building the other topology would be invisible in a table of strings.
    """

    assert "PORT_AUTO_LAYOUT_OK" in _run_probe(_PORT_AUTO_LAYOUT_PROBE)
