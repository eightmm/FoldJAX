"""The context-parallel transition must keep its widened form one block wide.

The transition widens its input to ``fc1 + fc2`` channels before narrowing it
again, and that pre-gate buffer is the largest tenant of the sharded Boltz-2
program when it is built whole: at 2,112 tokens on four devices it measured
``[1, 1056, 1056, 1024]`` per device under the 2-D layout and
``[1, 528, 2112, 1024]`` under the 1-D one -- about half the per-device arena
-- against ``[1, 64, 2112, 1024]`` in the serial program, which blocks it.

The property is about the compiled program, not about how the source spells the
loop: no value in the sharded program may carry more of the widened form than
one row block holds. Asserting that a ``row_chunk_size`` reached the transition
instead would pass a rewrite that blocked the global row axis -- which the
partitioner answers with a gather, because a block of a sharded axis is not
shard-aligned, and which measured 0.2% off the arena for that reason.

What the bound asserts is the program the source asks for. A backend may still
merge adjacent block dots afterwards -- XLA's CPU backend does, two blocks at a
time, at 2,112 tokens -- so this is a floor on the blocking, not a promise
about every fusion decision. It holds exactly at the size probed here, and it
fails on the unblocked spelling by the full local row extent, which is the
defect it exists to catch.

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
    import re

    import jax
    import jax.numpy as jnp
    import numpy as np

    from foldjax.models._cp import context_parallel, cp_layout
    from foldjax.models.boltz2.models.trunk_blocks.msa import (
        pairformer_no_seq_layer_forward,
    )
    from foldjax.models.boltz2.models.trunk_blocks.pairformer import (
        pairformer_layer_forward,
    )

    # `N` must leave more than one row block on every device: 96 rows give 24
    # on the 1-D mesh and 48 on the grid against a block of 8. The widened
    # width is `2 * 4 * C` = 64, which no other dimension in this program
    # carries -- C is 8, the hidden form is 32, the triangle projections are
    # 16 and the head count is 2 -- so a value whose trailing dimension is 64
    # is the pre-gate form and nothing else.
    N, C, HEADS, BLOCK = 96, 8, 2, 8
    WIDE = 2 * 4 * C

    # `(?:[0-9]+,)*` rather than `[0-9,]*`: the latter also matches a shape
    # whose dimensions merely end in the right digits.
    PRE_GATE = re.compile(r"\b\w+\[((?:[0-9]+,)*" + str(WIDE) + r")\]")
    DEFINITION = re.compile(r"^\s+(?:ROOT\s+)?%?[\w.\-]+ = \S+ ")

    def elements(dims):
        total = 1
        for size in dims.split(","):
            total *= int(size)
        return total

    rng = np.random.default_rng(20260916)

    def arr(*shape):
        return jnp.asarray(rng.normal(size=shape, scale=0.5), dtype=jnp.float32)

    def norm(c):
        return {"scale": arr(c) * 0.1 + 1.0, "bias": arr(c) * 0.1}

    def weight(fan_in, fan_out):
        return arr(fan_in, fan_out) * (1.0 / (0.5 * np.sqrt(fan_in)))

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

    pair_params = {
        "tri_mul_out": tri_mult(),
        "tri_mul_in": tri_mult(),
        "tri_att_start": tri_att(),
        "tri_att_end": tri_att(),
        "transition_z": transition(),
    }
    seq_params = dict(pair_params)
    seq_params["pre_norm_s"] = norm(C)
    seq_params["attention"] = {
        "proj_z_norm": norm(C),
        "proj_q": {"kernel": weight(C, C), "bias": arr(C)},
        "proj_k": {"kernel": weight(C, C)},
        "proj_v": {"kernel": weight(C, C)},
        "proj_g": {"kernel": weight(C, C)},
        "proj_z": {"kernel": weight(C, HEADS)},
        "proj_o": {"kernel": weight(C, C)},
    }
    seq_params["transition_s"] = transition()

    z = arr(1, N, N, C)
    s = arr(1, N, C)
    keep = rng.random(N) > 0.15
    mask = jnp.asarray(keep[None]).astype(jnp.float32)
    pair_mask = jnp.asarray((keep[:, None] & keep[None, :])[None]).astype(jnp.float32)

    def pair_only(z_in, params_in):
        return pairformer_no_seq_layer_forward(
            params_in, z_in, pair_mask, chunk_size=BLOCK, triangle_backend="xla"
        )

    def with_single(z_in, params_in):
        return pairformer_layer_forward(
            params_in,
            s,
            z_in,
            mask,
            pair_mask,
            chunk_size=BLOCK,
            triangle_backend="xla",
        )[1]

    for name, program, params in (
        ("msa_pair_layer", pair_only, pair_params),
        ("pairformer_layer", with_single, seq_params),
    ):
        for layout, rows in (("1d", 4), ("2d", 2)):
            seen = []

            # A fresh closure per arm: `jax.jit` keys its cache on the
            # callable and the mesh is a context variable the trace reads, so
            # a reused one would replay the first arm's program.
            def run(z_in, params_in, program=program):
                seen.append(cp_layout())
                return program(z_in, params_in)

            with context_parallel(_DEVICES, layout=layout):
                jitted = jax.jit(run)
                compiled = jitted.lower(z, params).compile()
                value = jitted(z, params)
                value.block_until_ready()
                local = tuple(
                    int(size)
                    for size in next(iter(value.addressable_shards)).data.shape
                )

            assert seen == [layout], (name, seen)
            # The tripwire: without it a program that quietly ran unsharded
            # would satisfy every bound below.
            columns = N // rows if layout == "2d" else N
            assert local == (1, N // rows, columns, C), (name, layout, local)

            text = compiled.as_text()
            found = {
                dims: elements(dims) for dims in PRE_GATE.findall(text)
            }
            # Non-empty, so a regex that stopped matching cannot pass in
            # silence: the widened form has to appear somewhere.
            assert found, (name, layout, text[:2000])
            # One block of local rows, all local columns, the widened width.
            bound = BLOCK * columns * WIDE
            oversized = sorted(
                ((dims, size) for dims, size in found.items() if size > bound),
                key=lambda item: -item[1],
            )
            assert not oversized, (name, layout, bound, oversized)
            widest = max(found.values())
            print(
                f"{name} {layout} local={local} bound={bound} "
                f"widest={widest} shapes={sorted(found)[:4]}"
            )

    print("TRANSITION_CP_ROW_BLOCK_OK")
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


def test_the_cp_transition_never_widens_more_than_one_row_block() -> None:
    """Both pair call sites, on the 1-D mesh and on the 2x2 grid."""

    assert "TRANSITION_CP_ROW_BLOCK_OK" in _run_probe(
        _PROBE.replace("_DEVICES", str(_DEVICES))
    )
