"""The other two pair transitions declare themselves to the CP row block too.

``transition_forward`` drops the row chunk for any rank-4 caller that has not
declared its tensor, because a block of a *sharded* row axis is not
shard-aligned. Two pair call sites were left undeclared when
``_cp_pair_transition`` landed, so their widened SwiGLU form stayed one whole
local tile under context parallelism:

* ``trunk_blocks/pairformer_noseq.py`` -- the template module and the affinity
  head, both reached from released inputs;
* ``trunk_blocks/conditioning.py`` -- ``pairwise_conditioning``, which every
  diffusion sample reads.

Same property and same evidence shape as ``test_transition_cp_row_block``: the
bound is read off the compiled program, and each arm is measured against the
undeclared spelling in the same process, so a block that stopped engaging would
fail rather than pass in silence. ``pairwise_conditioning`` takes no chunk
argument -- its block comes from ``_WIDE_BUDGET_BYTES`` -- so that arm asserts
the block halves the widened form rather than a particular width.

The values agree exactly rather than bitwise: at these probe sizes the block
moves float32 results by one last place (1.2e-7 measured), the same GEMM-tiling
effect ``transition_forward`` records for the serial row chunk. The MSA
transition next door *is* bitwise equal because its block needs no
``shard_map`` at all -- see ``test_msa_transition_cp_row_block``.

A forced device count has to be set before JAX initialises, so the check runs
in a subprocess with four CPU devices. The probe uses ``#`` comments rather
than docstrings because it lives inside a triple-quoted literal.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

from tests.models.cp_probe_env import inherited_environment

#: Four devices give a 4-shard 1-D mesh and a 2x2 grid.
_DEVICES = 4

_PROBE = textwrap.dedent(
    r"""
    import re

    import jax
    import jax.numpy as jnp
    import numpy as np

    from foldjax.models._cp import context_parallel, cp_layout
    from foldjax.models.boltz2.models.primitives import transition as transition_module
    from foldjax.models.boltz2.models.trunk_blocks import conditioning as cond_module
    from foldjax.models.boltz2.models.trunk_blocks import (
        pairformer_noseq as noseq_module,
    )

    # `N` leaves more than one row block on every device: 96 rows give 24 on
    # the 1-D mesh and 48 on the grid. The widened width `2 * 4 * C` = 64 is
    # carried by no other dimension here -- C is 8, the hidden form 32, the
    # triangle projections 16, the head count 2, and the conditioning inputs
    # are 4 and 5 wide.
    N, C, HEADS, BLOCK = 96, 8, 2, 8
    WIDE = 2 * 4 * C
    # Small enough that `_auto_row_chunk` blocks a probe-sized local tile:
    # under the grid one f32 row of the local tile is `48 * 64 * 4` = 12,288
    # bytes, so this is eight rows there and four on the 1-D mesh.
    BUDGET = 100_000

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

    def weight(fan_in, fan_out):
        return arr(fan_in, fan_out) * (1.0 / np.sqrt(fan_in))

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
                name: {"kernel": weight(C, C)}
                for name in (
                    "linear_q",
                    "linear_k",
                    "linear_v",
                    "linear_g",
                    "linear_o",
                )
            },
        }

    def transition():
        return {
            "norm": norm(C),
            "fc1": {"kernel": weight(C, 4 * C)},
            "fc2": {"kernel": weight(C, 4 * C)},
            "fc3": {"kernel": weight(4 * C, C)},
        }

    noseq_params = {
        "tri_mul_out": tri_mult(),
        "tri_mul_in": tri_mult(),
        "tri_att_start": tri_att(),
        "tri_att_end": tri_att(),
        "transition_z": transition(),
    }
    cond_params = {
        "dim_pairwise_init_proj": {
            "norm": norm(4 + 5),
            "linear": {"kernel": weight(4 + 5, C)},
        },
        "transitions": [transition(), transition()],
    }

    z = arr(1, N, N, C)
    z_trunk = arr(1, N, N, 4)
    rel_pos = arr(1, N, N, 5)
    keep = rng.random(N) > 0.15
    pair_mask = jnp.asarray(
        (keep[:, None] & keep[None, :])[None]
    ).astype(jnp.float32)

    shipped = transition_module.transition_forward
    fired = {"declared": 0, "undeclared": 0}

    def undeclared(params, x, **kwargs):
        # The program as it stood before the declaration: dropping `cp_pair`
        # sends the pair tensor down the branch that zeroes the row chunk.
        if kwargs.pop("cp_pair", False):
            fired["undeclared"] += 1
        return shipped(params, x, **kwargs)

    def declared(params, x, **kwargs):
        if kwargs.get("cp_pair", False):
            fired["declared"] += 1
        return shipped(params, x, **kwargs)

    def noseq(spelling):
        noseq_module.transition_forward = spelling

        def run():
            return noseq_module.pairformer_no_seq_layer_forward(
                noseq_params, z, pair_mask, chunk_size=BLOCK, triangle_backend="xla"
            )

        return run

    def conditioning(spelling):
        cond_module.transition_forward = spelling

        def run():
            return cond_module.pairwise_conditioning_forward(
                cond_params, z_trunk, rel_pos
            )

        return run

    def one(make, spelling, layout, module):
        seen = []
        program = make(spelling)

        # A fresh closure per arm: `jax.jit` keys its cache on the callable and
        # the mesh is a context variable the trace reads, so a reused one would
        # replay the first arm's program.
        def run():
            seen.append(cp_layout())
            return program()

        try:
            with context_parallel(_DEVICES, layout=layout):
                jitted = jax.jit(run)
                compiled = jitted.lower().compile()
                value = jitted()
                value.block_until_ready()
                local = tuple(
                    int(size)
                    for size in next(iter(value.addressable_shards)).data.shape
                )
        finally:
            module.transition_forward = shipped
        assert seen == [layout], (layout, seen)
        text = compiled.as_text()
        return {
            "value": np.asarray(value, np.float64),
            "found": {dims: elements(dims) for dims in PRE_GATE.findall(text)},
            "local": local,
            "collectives": sum(text.count(" " + name + "(") for name in COLLECTIVES),
        }

    transition_module._WIDE_BUDGET_BYTES = BUDGET

    for name, make, module in (
        ("pairformer_noseq", noseq, noseq_module),
        ("pairwise_conditioning", conditioning, cond_module),
    ):
        for layout, rows in (("1d", 4), ("2d", 2)):
            before = one(make, undeclared, layout, module)
            after = one(make, declared, layout, module)

            assert fired["declared"] and fired["undeclared"], fired
            # The tripwire: without it a program that quietly ran unsharded
            # would satisfy every bound below.
            columns = N // rows if layout == "2d" else N
            expected = (1, N // rows, columns, C)
            assert before["local"] == after["local"] == expected, (
                name, layout, before["local"], after["local"]
            )
            assert before["found"] and after["found"], (name, layout)

            widest_before = max(before["found"].values())
            widest_after = max(after["found"].values())
            if name == "pairformer_noseq":
                # `chunk_size` reaches this site as the row block, so the
                # width it must not exceed is exact.
                bound = BLOCK * columns * WIDE
                oversized = sorted(
                    (dims, size)
                    for dims, size in after["found"].items()
                    if size > bound
                )
                assert not oversized, (name, layout, bound, oversized)
                assert widest_before > bound, (name, layout, bound, widest_before)
            else:
                # The block comes from the byte budget rather than from an
                # argument, so assert that it blocks at all, and by at least
                # half -- vacuity is the failure mode this guards.
                assert widest_after * 2 <= widest_before, (
                    name, layout, widest_before, widest_after
                )

            # The row axis is a pure batch axis of every op in the transition,
            # so the block splits no reduction and adds no communication.
            assert before["collectives"] == after["collectives"], (
                name, layout, before["collectives"], after["collectives"]
            )
            # Exact, not bit-identical: the block splits no reduction, but XLA
            # tiles the smaller GEMMs differently, which at float32 is a
            # last-place difference (`transition_forward` records 1.2e-4
            # relative for the serial row chunk from the same cause). The
            # ceiling is five orders below anything a misplaced block could
            # do -- a wrong row or an unsliced pad moves values by O(1).
            difference = np.abs(after["value"] - before["value"])
            largest = float(difference.max())
            assert largest < 1e-5, (name, layout, largest)
            print(
                f"{name} {layout} local={after['local']} widest "
                f"{widest_before} -> {widest_after} "
                f"collectives {after['collectives']} maxabs {largest:.3e}"
            )

    print("PAIR_TRANSITION_CP_DECLARATIONS_OK")
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


def test_the_remaining_pair_transitions_block_local_rows_under_cp() -> None:
    """The template/affinity pair layer and the diffusion pair conditioning."""

    assert "PAIR_TRANSITION_CP_DECLARATIONS_OK" in _run_probe(
        _PROBE.replace("_DEVICES", str(_DEVICES))
    )
