"""The context-parallel MSA transition must keep its widened form one block wide.

The MSA transition is the one transition whose axis 1 is *not* a token axis:
``m`` is ``[B, M, N, C]`` with ``M`` the alignment depth. No layout shards that
axis -- the carry measures ``PartitionSpec(None, None, "cp")`` under the 1-D
layout and ``PartitionSpec(None, None, "cp_row")`` under the 2-D one -- so the
serial row block is shard-aligned as written and needs no ``shard_map``, unlike
the pair transition next door (``test_transition_cp_row_block``).

Dropping it anyway is what ``transition_forward`` used to do for every caller
that was not a declared pair tensor, and it made the MSA transition the largest
tenant of the sharded program: at 2,096 tokens over an 8,192-row alignment on
four devices, the eight unrolled hidden-chunk accumulators stood at full
``M x N_local`` width -- ``bf16[8585216, 64]`` x8, 8.4 GiB, half the 16,956 MiB
per-device peak-live set of the 2-D program.

The properties are asserted against the compiled program rather than against
the source spelling, and always *against the unblocked spelling in the same
process*, so none of them can pass by no longer measuring anything. Two
branches are covered, because the call site chooses between them on token
count and they fail differently:

* below 384 tokens the transition runs as one op, the widened pre-gate form is
  the tenant, and no value may carry more of it than one row block holds --
  where the unblocked spelling carries the whole local tile. The result there
  is bitwise equal to the unblocked one;
* above 384 tokens -- the released regime, and where the 2,096-token peak was
  measured -- the call site passes ``chunk_size=32`` and the tenant is instead
  the eight co-live output accumulators of the unrolled hidden-chunk loop.
  There the full-width count has to fall and block-sized values appear, and
  the result is *not* bitwise equal: the block lets XLA reassociate that
  eight-chunk accumulation, which moves the transition's output by a rounding
  of the compute dtype. The row axis is still a pure batch axis, so no
  reduction is split, and the serial program has always taken this same block.

A forced device count has to be set before JAX initialises, so the check runs
in a subprocess with four CPU devices: enough for the 1-D layout and for the
smallest perfect square the 2-D layout accepts. The probe uses ``#`` comments
rather than docstrings because it lives inside a triple-quoted literal.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

from tests.models.cp_probe_env import inherited_environment

#: Four devices give a 4-shard 1-D mesh and a 2x2 grid. The probe asserts the
#: local tile shape each layout must produce, so it depends on this count.
_DEVICES = 4

_PROBE = textwrap.dedent(
    r"""
    import collections
    import re

    import jax
    import jax.numpy as jnp
    import numpy as np

    from foldjax.models._cp import context_parallel, cp_layout
    from foldjax.models.boltz2.models.primitives import transition as transition_module
    from foldjax.models.boltz2.models.trunk_blocks import msa as msa_module

    # `N` divides both meshes (4, and 2 on each side of the grid). `S` is the
    # alignment depth, which no layout shards; seven rows against a three-row
    # block leave a short last block, and neither `S` nor the local token
    # width is a multiple of the block.
    N, S, CM, CZ, HEADS, BLOCK = 20, 7, 8, 12, 2, 3
    # The MSA transition's widened pre-gate width, `fc1 + fc2`. No other
    # dimension in this program carries it: `CM` is 8 and its hidden form 32,
    # `CZ` is 12 and the pair widened form 96, the triangle projections are 24
    # and the head count is 2.
    WIDE = 2 * 4 * CM
    # Set so the shipped `_auto_row_chunk` rule returns exactly `BLOCK` for a
    # float32 `[1, S, N, CM]`: `N * WIDE * 4` = 5,120 bytes per row, and
    # `16384 // 5120 == 3`. The rule itself is the code under test; only its
    # budget is moved, so that a probe-sized tensor reaches the same branch a
    # 2,096-token job reaches.
    BUDGET = 16384

    # `(?:[0-9]+,)*` rather than `[0-9,]*`: the latter also matches a shape
    # whose dimensions merely end in the right digits.
    PRE_GATE = re.compile(r"\b\w+\[((?:[0-9]+,)*" + str(WIDE) + r")\]")
    COLLECTIVES = (
        "all-gather",
        "all-reduce",
        "all-to-all",
        "collective-permute",
        "reduce-scatter",
    )

    def elements(dims):
        total = 1
        for size in dims.split(","):
            total *= int(size)
        return total

    rng = np.random.default_rng(20260917)

    def arr(*shape):
        return jnp.asarray(rng.normal(size=shape, scale=0.5), dtype=jnp.float32)

    def norm(c):
        return {"scale": arr(c) * 0.1 + 1.0, "bias": arr(c) * 0.1}

    def weight(fan_in, fan_out, dtype):
        return jnp.asarray(
            rng.normal(size=(fan_in, fan_out), scale=1.0 / np.sqrt(fan_in)),
            dtype=dtype,
        )

    def build(dtype, CM):
        def tri_mult():
            return {
                "norm_in": norm(CZ),
                "norm_out": norm(CZ),
                "g_in": {"kernel": weight(CZ, 2 * CZ, dtype)},
                "p_in": {"kernel": weight(CZ, 2 * CZ, dtype)},
                "p_out": {"kernel": weight(CZ, CZ, dtype)},
                "g_out": {"kernel": weight(CZ, CZ, dtype)},
            }

        def tri_att():
            return {
                "layer_norm": norm(CZ),
                "linear": {"kernel": weight(CZ, HEADS, dtype)},
                "mha": {
                    name: {"kernel": weight(CZ, CZ, dtype)}
                    for name in (
                        "linear_q",
                        "linear_k",
                        "linear_v",
                        "linear_g",
                        "linear_o",
                    )
                },
            }

        def transition(c):
            return {
                "norm": norm(c),
                "fc1": {"kernel": weight(c, 4 * c, dtype)},
                "fc2": {"kernel": weight(c, 4 * c, dtype)},
                "fc3": {"kernel": weight(4 * c, c, dtype)},
            }

        return {
            "pair_weighted_averaging": {
                "norm_m": norm(CM),
                "norm_z": norm(CZ),
                "proj_m": {"kernel": weight(CM, HEADS * 4, dtype)},
                "proj_z": {"kernel": weight(CZ, HEADS, dtype)},
                "proj_g": {"kernel": weight(CM, HEADS * 4, dtype)},
                "proj_o": {"kernel": weight(HEADS * 4, CM, dtype)},
            },
            "msa_transition": transition(CM),
            "outer_product_mean": {
                "norm": norm(CM),
                "proj_a": {"kernel": weight(CM, 4, dtype)},
                "proj_b": {"kernel": weight(CM, 4, dtype)},
                "proj_o": {"kernel": weight(16, CZ, dtype), "bias": arr(CZ)},
            },
            "pairformer_layer": {
                "tri_mul_out": tri_mult(),
                "tri_mul_in": tri_mult(),
                "tri_att_start": tri_att(),
                "tri_att_end": tri_att(),
                "transition_z": transition(CZ),
            },
        }

    z0 = arr(1, N, N, CZ)
    keep = rng.random(N) > 0.15
    token_mask = jnp.asarray(
        (keep[:, None] & keep[None, :])[None]
    ).astype(jnp.float32)
    msa_mask = jnp.asarray(
        ((rng.random((S, N)) > 0.1) & keep[None, :])[None]
    ).astype(jnp.float32)

    transition_module._WIDE_BUDGET_BYTES = BUDGET
    shipped = msa_module.transition_forward
    fired = {"unblocked": 0, "blocked": 0}

    def unblocked(params, x, **kwargs):
        # The program as it stood before the declaration existed: dropping it
        # sends the MSA tensor down the branch that zeroes the row chunk.
        if kwargs.pop("cp_msa", False):
            fired["unblocked"] += 1
        return shipped(params, x, **kwargs)

    def blocked(params, x, **kwargs):
        if kwargs.get("cp_msa", False):
            fired["blocked"] += 1
        return shipped(params, x, **kwargs)

    def one(params, m0, spelling, layout, marker):
        seen = []
        msa_module.transition_forward = spelling

        # A fresh closure per arm: `jax.jit` keys its cache on the callable and
        # the mesh is a context variable the trace reads, so a reused one would
        # replay the first arm's program.
        def run(z, m, p):
            seen.append(cp_layout())
            return msa_module.msa_layer_forward(p, z, m, token_mask, msa_mask)

        try:
            with context_parallel(_DEVICES, layout=layout):
                jitted = jax.jit(run)
                compiled = jitted.lower(z0, m0, params).compile()
                z, m = jitted(z0, m0, params)
                jax.block_until_ready((z, m))
                local = tuple(
                    int(size)
                    for size in next(iter(m.addressable_shards)).data.shape
                )
        finally:
            msa_module.transition_forward = shipped
        assert seen == [layout], (layout, seen)
        text = compiled.as_text()
        matches = marker.findall(text)
        return {
            "z": np.asarray(z, np.float64),
            "m": np.asarray(m, np.float64),
            "found": {dims: elements(dims) for dims in matches},
            # Occurrences, not distinct shapes: eight co-live accumulators of
            # the same width are eight values, and collapsing them to one key
            # is what would make the count below unable to move.
            "counts": collections.Counter(matches),
            "local": local,
            "collectives": sum(text.count(" " + name + "(") for name in COLLECTIVES),
        }

    m_single = arr(1, S, N, CM)
    for dtype_name, dtype in (("float32", jnp.float32), ("bfloat16", jnp.bfloat16)):
        params = build(dtype, CM)
        for layout, rows in (("1d", 4), ("2d", 2)):
            before = one(params, m_single, unblocked, layout, PRE_GATE)
            after = one(params, m_single, blocked, layout, PRE_GATE)

            # Both patched spellings have to have reached the MSA transition,
            # or one of the arms is the other one measured twice.
            assert fired["unblocked"] and fired["blocked"], fired
            # The tripwire: without it a program that quietly ran unsharded
            # would satisfy every bound below. The alignment depth stays whole
            # and the token axis is halved on the grid, quartered on the mesh.
            expected = (1, S, N // rows, CM)
            assert before["local"] == after["local"] == expected, (
                dtype_name, layout, before["local"], after["local"]
            )

            # Non-empty, so a regex that stopped matching cannot pass in
            # silence: the widened form has to appear in both programs.
            assert before["found"] and after["found"], (dtype_name, layout)
            bound = BLOCK * (N // rows) * WIDE
            oversized = sorted(
                (dims, size)
                for dims, size in after["found"].items()
                if size > bound
            )
            assert not oversized, (dtype_name, layout, bound, oversized)
            # And the bound discriminates: the unblocked spelling breaks it by
            # the full local tile, which is the defect this test exists for.
            assert max(before["found"].values()) > bound, (
                dtype_name, layout, bound, sorted(before["found"])
            )

            # The row axis is a pure batch axis of every op in the transition,
            # so the block splits no reduction and adds no communication.
            assert before["collectives"] == after["collectives"], (
                dtype_name, layout, before["collectives"], after["collectives"]
            )
            for name in ("z", "m"):
                np.testing.assert_array_equal(
                    after[name], before[name], err_msg=f"{dtype_name} {layout} {name}"
                )
            print(
                f"single-op {dtype_name} {layout} local={after['local']} "
                f"bound={bound} widest {max(before['found'].values())} -> "
                f"{max(after['found'].values())} "
                f"collectives {after['collectives']} bitwise-equal"
            )

    # The branch the released program actually takes. Above 384 tokens the
    # call site passes `chunk_size=32`, which trades the widened form for an
    # unrolled loop over the hidden axis -- and it is *that* loop's output
    # accumulators, eight of them co-live and `C_m` wide, that the 2,096-token
    # peak was made of, not the pre-gate. `CM_WIDE = 64` reproduces the
    # released hidden width (4 x 64 = 256, eight chunks of 32); the threshold
    # is moved so a 20-token probe reaches the branch, and both arms see the
    # same threshold, so the head grouping in pair-weighted averaging and the
    # native chunking in OuterProductMean flip identically in each.
    #
    # The shape bound cannot be asserted on this branch: the accumulator is
    # `C_m` wide, which is also the width of `m` itself and of the
    # transition's own output, so a full-width value is expected rather than
    # forbidden. What is asserted is that the full-width count falls and
    # block-sized values appear -- the same movement the 2,112-token census
    # reads as 16 -> 1.
    #
    # Nor is this branch bitwise equal, where the single-op one above is:
    # blocking the rows lets XLA reassociate the eight-chunk accumulation, and
    # the result moves by a rounding of the compute dtype -- measured
    # `m` 0.0 (float32 1-D), 9.5e-7 (float32 2-D) and 7.8e-3 (bfloat16, one
    # ulp of bfloat16 at this scale) in both layouts. The serial program has
    # always taken this same row block, and on three of the four arms the
    # blocked result is *closer* to the serial one than the unblocked result
    # was (bfloat16 1-D `m` 7.8e-3 -> 2.0e-3), which is the direction that
    # settles whether the block is the perturbation or the reference.
    CM_WIDE, BUDGET_WIDE = 64, 131072
    ACCUMULATOR = re.compile(r"\b\w+\[((?:[0-9]+,)*" + str(CM_WIDE) + r")\]")
    msa_module._NATIVE_CHUNK_THRESHOLD = 0
    transition_module._WIDE_BUDGET_BYTES = BUDGET_WIDE
    m_chunked = arr(1, S, N, CM_WIDE)
    for dtype_name, dtype in (("float32", jnp.float32), ("bfloat16", jnp.bfloat16)):
        params = build(dtype, CM_WIDE)
        for layout, rows in (("1d", 4), ("2d", 2)):
            before = one(params, m_chunked, unblocked, layout, ACCUMULATOR)
            after = one(params, m_chunked, blocked, layout, ACCUMULATOR)

            assert fired["unblocked"] and fired["blocked"], fired
            expected = (1, S, N // rows, CM_WIDE)
            assert before["local"] == after["local"] == expected, (
                dtype_name, layout, before["local"], after["local"]
            )
            full = S * (N // rows) * CM_WIDE
            block = BLOCK * (N // rows) * CM_WIDE

            def occurrences(arm, wanted):
                return sum(
                    count
                    for dims, count in arm["counts"].items()
                    if elements(dims) == wanted
                )

            full_before = occurrences(before, full)
            full_after = occurrences(after, full)
            block_after = occurrences(after, block)
            assert full_after < full_before, (
                dtype_name, layout, full_before, full_after
            )
            assert block_after, (dtype_name, layout, sorted(after["found"]))
            assert before["collectives"] == after["collectives"], (
                dtype_name, layout, before["collectives"], after["collectives"]
            )
            # A few roundings of the compute dtype on the transition's own
            # output, which is `m`. `z` is the same difference amplified by
            # the pair path downstream, so it is printed rather than bounded.
            ulp = 2.0**-8 if dtype == jnp.bfloat16 else 2.0**-23
            scale = max(float(np.abs(before["m"]).max()), 1.0)
            gap_m = float(np.abs(after["m"] - before["m"]).max())
            gap_z = float(np.abs(after["z"] - before["z"]).max())
            assert gap_m <= 8 * ulp * scale, (
                dtype_name, layout, gap_m, 8 * ulp * scale
            )
            print(
                f"hidden-chunk {dtype_name} {layout} local={after['local']} "
                f"full-width {full_before} -> {full_after}, "
                f"block-width {block_after} "
                f"collectives {after['collectives']} "
                f"maxabs m={gap_m:.3e} z={gap_z:.3e} "
                f"(<= {8 * ulp * scale:.3e})"
            )

    print("MSA_TRANSITION_CP_ROW_BLOCK_OK")
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


def test_the_cp_msa_transition_never_widens_more_than_one_row_block() -> None:
    """Both layouts, both dtypes, against the unblocked spelling."""

    assert "MSA_TRANSITION_CP_ROW_BLOCK_OK" in _run_probe(
        _PROBE.replace("_DEVICES", str(_DEVICES))
    )
