"""The context-parallel pair transition must block its own local rows.

SwiGLU widens the pair representation to ``4 * C_z`` twice before the output
projection reads the product, and that widened form is the largest buffer in
the sharded program. Leaving the row chunk switched off under a mesh made it
one whole local tile: measured at 2,112 tokens on four devices,
``f32[1115136, 512]`` -- 2,178 MiB -- at 18 allocated sites, *identically*
under the 1-D layout (528 x 2112) and the 2-D one (1056 x 1056), because
``N/4 x N`` and ``N/2 x N/2`` are the same area.

The property is therefore about the compiled program, not about how the source
spells the loop: no value in the context-parallel program may carry a widened
tile wider than one row block. A test that asserted a ``shard_map`` were
present, or that ``map_row_chunks`` were called, would pass a rewrite that
reintroduced the full tile through some other route and would fail a rewrite
that kept the property by different means.

Three things have to hold together, and each of the first two is the other's
tripwire:

* the *unblocked* program must still show the full-tile shape, or the size
  bound below has no power and would pass a program that never sharded;
* the blocked program must show the block shape, or a regex that stopped
  matching would pass silently;
* the collective census must be identical between the two, because the whole
  point is that a local tile is the entire computation for its own rows and
  columns -- the block must buy its saving with no communication. The
  transition on its own emits no collective at this size either way, so that
  equality is read on the whole ``pair_block`` under the 1-D layout, where it
  is nine instructions rather than none. It cannot be read there under the 2-D
  layout, because ``chunk_size`` is also the ring's query block and the ring's
  own ``collective-permute`` count moves with it (64 against 44 here); the
  non-vacuous 2-D measurement is the full model's, 640 against 640 at 2,112
  tokens on four devices.

Values are checked against the *serial* blocked program rather than against a
loose tolerance on the sharded one: on this backend they agree to the bit in
both layouts and in both dtypes, while the unblocked sharded program does not
(3.8e-6 at float32, the blocking's own rounding).

A forced device count has to be set before JAX initialises, so the checks run
in a subprocess with four CPU devices: enough for a 4-shard 1-D mesh and for
the smallest perfect square the 2-D layout accepts. The probe uses ``#``
comments rather than docstrings because it lives inside a triple-quoted
literal.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

from tests.models.cp_probe_env import inherited_environment

#: Four devices give a 4-shard 1-D mesh and a 2x2 grid. The probe asserts the
#: local tile width each layout must produce, so it depends on this count.
_DEVICES = 4

_PROBE = textwrap.dedent(
    r"""
    import collections
    import re

    import jax
    import jax.numpy as jnp
    import numpy as np

    from foldjax.models._cp import context_parallel, shard_pair_rows
    from foldjax.models.openfold3.models import pair_block as pair_block_module
    from foldjax.models.openfold3.models.attention import AttentionParams
    from foldjax.models.openfold3.models.pair_block import (
        PairBlockParams,
        pair_block,
    )
    from foldjax.models.openfold3.models.primitives import (
        LayerNormParams,
        LinearParams,
        SwiGLUParams,
        SwiGLUTransitionParams,
    )
    from foldjax.models.openfold3.models.triangle import (
        TriangleMultiplicationParams,
    )
    from foldjax.models.openfold3.models.triangle_attention import (
        TriangleAttentionParams,
    )

    assert jax.device_count() == 4, jax.devices()

    # `N` divides both grids so no padding is involved, and `BLOCK` divides
    # neither local row count (4 under 1-D, 8 under 2-D): the ragged last
    # block is the case a row-block rewrite is most likely to get wrong.
    N, C, HEADS, BLOCK = 16, 8, 2, 3
    WIDE = 4 * C
    # Two dims at least, so the `[WIDE]` projection bias vectors -- which are
    # not activations -- cannot satisfy the bound by being small.
    WIDENED = re.compile(r"\b(f32|bf16)\[((?:[0-9]+,)+" + str(WIDE) + r")\]")
    INSTRUCTION = re.compile(r"^\s+(?:ROOT\s+)?(%?[\w.\-]+) = (.+?) ([\w-]+)\(")
    COLLECTIVES = (
        "all-gather",
        "all-gather-start",
        "all-reduce",
        "all-reduce-start",
        "all-to-all",
        "collective-permute",
        "collective-permute-start",
        "reduce-scatter",
    )

    def widened_rows(text, local_columns):
        # Token positions per widened value, i.e. its element count divided by
        # the channel width, expressed in rows of the local tile.
        found = set()
        for _, dims in WIDENED.findall(text):
            sizes = [int(size) for size in dims.split(",")]
            positions = 1
            for size in sizes[:-1]:
                positions *= size
            if positions % local_columns:
                # Not a whole number of local rows: a per-row or reshaped
                # form, which the row bound does not describe.
                continue
            found.add(positions // local_columns)
        return found

    def collective_census(text):
        census = collections.Counter()
        computation = "?"
        for line in text.splitlines():
            match = INSTRUCTION.match(line)
            if not match:
                continue
            _, shape, opcode = match.groups()
            if opcode in COLLECTIVES:
                census[(opcode, shape.split(" ")[0])] += 1
        return census

    rng = np.random.default_rng(0)

    def arr(*shape):
        return jnp.asarray(rng.normal(size=shape, scale=0.5), dtype=jnp.float32)

    def lin(o, i):
        return LinearParams(weight=arr(o, i), bias=arr(o))

    def ln(c):
        return LayerNormParams(weight=arr(c) * 0.1 + 1.0, bias=arr(c) * 0.1)

    def tri_mult():
        return TriangleMultiplicationParams(
            layer_norm_in=ln(C), layer_norm_out=ln(C),
            linear_a_p=lin(C, C), linear_a_g=lin(C, C),
            linear_b_p=lin(C, C), linear_b_g=lin(C, C),
            linear_g=lin(C, C), linear_z=lin(C, C),
        )

    def attn():
        return AttentionParams(lin(C, C), lin(C, C), lin(C, C), lin(C, C), lin(C, C))

    def tri_att():
        return TriangleAttentionParams(
            layer_norm=ln(C),
            linear_z=LinearParams(weight=arr(HEADS, C), bias=None),
            mha=attn(),
        )

    transition = SwiGLUTransitionParams(
        layer_norm=ln(C),
        swiglu=SwiGLUParams(linear_a=lin(WIDE, C), linear_b=lin(WIDE, C)),
        linear_out=lin(C, WIDE),
    )
    params = PairBlockParams(
        tri_mul_out=tri_mult(), tri_mul_in=tri_mult(),
        tri_att_start=tri_att(), tri_att_end=tri_att(),
        pair_transition=transition,
    )

    z = arr(1, N, N, C)
    keep = rng.random(N) > 0.15
    pair_mask = jnp.asarray(keep[:, None] & keep[None, :], dtype=jnp.float32)[None]

    def transition_only(chunk, masked):
        # A fresh closure per arm: `jax.jit` keys its cache on the callable and
        # the mesh is a context variable the trace reads, so a reused one would
        # replay the first arm's program.
        def program(z_in, mask_in):
            return pair_block_module._pair_transition(
                shard_pair_rows(z_in),
                transition,
                pair_mask=mask_in if masked else None,
                eps=1e-5,
                glu_backend="xla",
                chunk_size=chunk,
            )

        return program

    def whole_block(chunk):
        def program(z_in, mask_in):
            return pair_block(
                z_in, params, pair_mask=mask_in, no_heads_pair=HEADS,
                chunk_size=chunk, glu_backend="xla",
            )

        return program

    def compile_and_run(program, layout):
        jax.clear_caches()
        if layout is None:
            compiled = jax.jit(program).lower(z, pair_mask).compile()
            value = jax.jit(program)(z, pair_mask)
            local = None
        else:
            with context_parallel(4, layout=layout):
                compiled = jax.jit(program).lower(z, pair_mask).compile()
                value = jax.jit(program)(z, pair_mask)
                value.block_until_ready()
                local = tuple(
                    int(size)
                    for size in next(iter(value.addressable_shards)).data.shape
                )
        return compiled.as_text(), np.asarray(jax.device_get(value)), local

    for masked in (True, False):
        serial_text, serial_value, _ = compile_and_run(
            transition_only(BLOCK, masked), None
        )
        serial_widths = widened_rows(serial_text, N)
        assert serial_widths and max(serial_widths) <= BLOCK, (
            "serial", masked, serial_widths
        )

        for layout, (rows, columns) in (("1d", (4, 1)), ("2d", (2, 2))):
            local_rows, local_columns = N // rows, N // columns
            assert local_rows > BLOCK, (layout, local_rows)

            blocked_text, blocked, local = compile_and_run(
                transition_only(BLOCK, masked), layout
            )
            whole_text, whole, whole_local = compile_and_run(
                transition_only(None, masked), layout
            )

            # Tripwire: an arm that quietly ran unsharded would satisfy every
            # size bound below.
            assert local == (1, local_rows, local_columns, C), (layout, local)
            assert whole_local == local, (layout, whole_local)

            # The defect, still present without the block: the widened form is
            # the whole local tile. Without this the bound has no power.
            assert local_rows in widened_rows(whole_text, local_columns), (
                layout, masked, local_rows, widened_rows(whole_text, local_columns)
            )
            # The property. Non-empty, so a regex that stopped matching cannot
            # pass silently.
            blocked_widths = widened_rows(blocked_text, local_columns)
            assert blocked_widths, (layout, masked, blocked_text[:2000])
            assert max(blocked_widths) <= BLOCK, (layout, masked, blocked_widths)

            # The block buys its saving with no communication.
            assert collective_census(blocked_text) == collective_census(whole_text), (
                layout,
                masked,
                collective_census(blocked_text) - collective_census(whole_text),
                collective_census(whole_text) - collective_census(blocked_text),
            )

            # A local tile is the whole computation for its own rows and
            # columns, so the sharded blocked program is the serial blocked
            # program: it agrees to the bit here, and the tolerance is three
            # orders above that, well under anything a mis-sliced block or a
            # mis-specified shard could produce.
            np.testing.assert_allclose(blocked, serial_value, rtol=1e-5, atol=1e-5)
            print(
                f"{layout} masked={masked} local={local} "
                f"blocked_rows={sorted(blocked_widths)} "
                f"whole_rows={sorted(widened_rows(whole_text, local_columns))} "
                f"collectives={sum(collective_census(blocked_text).values())} "
                f"blocked-serial={float(np.abs(blocked - serial_value).max()):.3g} "
                f"whole-serial={float(np.abs(whole - serial_value).max()):.3g}"
            )

    # The shipped call path has to reach it: `pair_block` itself must not carry
    # a full-tile widened value under either layout.
    for layout, (rows, columns) in (("1d", (4, 1)), ("2d", (2, 2))):
        local_rows, local_columns = N // rows, N // columns
        text, _, _ = compile_and_run(whole_block(BLOCK), layout)
        widths = widened_rows(text, local_columns)
        assert widths, (layout, "no widened value found in pair_block")
        assert max(widths) <= BLOCK, (layout, sorted(widths))
        census = collective_census(text)
        note = ""
        if layout == "1d":
            # The non-vacuous form of "no new collectives": nine instructions
            # either way. Under 2-D `chunk_size` is also the ring's query
            # block, so the arms are not comparable there.
            inert, _, _ = compile_and_run(whole_block(None), layout)
            assert census, (layout, "no collective found in the CP pair block")
            assert census == collective_census(inert), (
                layout,
                census - collective_census(inert),
                collective_census(inert) - census,
            )
            note = f" collectives={sum(census.values())} (unchunked: same)"
        print(f"pair_block {layout} widened_rows={sorted(widths)}{note}")

    print("PAIR_TRANSITION_ROW_CHUNK_CP_OK")
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
            **inherited_environment(),
        },
        timeout=600,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return completed.stdout


def test_the_cp_pair_transition_never_widens_a_whole_local_tile() -> None:
    """One row block's widened form per device, on the 1-D mesh and the 2x2 grid."""

    assert "PAIR_TRANSITION_ROW_CHUNK_CP_OK" in _run_probe(_PROBE)
