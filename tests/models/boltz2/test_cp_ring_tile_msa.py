"""`triangle_attention_ring_kernel` reaches the MSA stack's ring, not only the trunk's.

Boltz-2 enters the gather-free 2-D ring through two different modules. The
trunk's Pairformer and pair-only Pairformer go through the CP dispatcher
(`triangle/triangle_attention_cp.py`), which has read the scope since the
option landed. The MSA stack does not: `trunk_blocks/msa.py` carries its own
`pairformer_no_seq_layer_forward`, which imports the *serial* module's
`triangle_attention_forward` and reaches the same ring through that module's
own context-parallel branch (`_attention_ring_2d`). That branch took the
default tile, so a prediction asking for the fused tile ran it in the trunk and
the shipped two-pass body in the MSA stack -- which is a fused label on a run
that was only partly fused, the failure `resolve_ring_tile_kernel` refuses
rather than downgrades for.

What a CPU can settle is here; which implementation a card selects and what it
costs is not:

* **the census** -- every ring call site the MSA stack reaches asks the scope
  for its tile kernel, and the answer reaches the tile the ring evaluates. The
  spy sits on the *consumer* module's binding of `resolve_ring_tile_kernel`,
  because `from ... import` binds the name locally, and its argument is the
  scope value the call site read;
* **the sharding contract** -- with the fused tile resolved to tokamax's own
  portable XLA implementation, which returns the same
  `normalize_output=False, return_residuals=True` triple the Triton kernel
  does, the option-on ring matches the option-off ring on deliberately
  asymmetric per-rank inputs, at 2x2 and at 3x3.

The one place the two bodies legitimately disagree is a query row whose every
key is masked: tokamax masks with `finfo.min` and the tile adapter forces such
a row to zeros, where the two-pass body reduces over the `-1e9`-biased keys as
the serial path does (`docs/context_parallel.md`, and
`tests/models/test_cp_fused_ring_tile.py` pins it). The contract arm therefore
reads the rows the model keeps; the masked arm asserts the documented zeros
instead of a tolerance that would hide them.

The option-off program is not compared here but in
`tests/models/boltz2/scripts/cp_ring_tile_fingerprints.py`, which fingerprints
both ring entries and the MSA stack across two source trees.

Each probe runs in a subprocess because a forced device count has to be set
before JAX initialises, and every arm builds its own `jax.jit` object: two arms
over one function object share one trace and the second never enters the scope
(`cp-parity-tests-can-be-vacuous`). Probe sources use `#` comments rather than
docstrings because they live inside a triple-quoted literal.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

from tests.models.cp_probe_env import inherited_environment


def _run(source: str, *, devices: int) -> str:
    env = {
        "JAX_PLATFORMS": "cpu",
        "XLA_FLAGS": f"--xla_force_host_platform_device_count={devices}",
        "FOLDJAX_CP_PROBE_DEVICES": str(devices),
        **inherited_environment(),
    }
    completed = subprocess.run(
        [sys.executable, "-c", source],
        capture_output=True,
        text=True,
        env=env,
        timeout=900,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return completed.stdout


#: Parameters and features for the MSA stack and for one triangle-attention
#: entry, at the smallest shapes that still divide both grid sides.
_PRELUDE = r"""
import math
import os

import jax
import jax.numpy as jnp
import numpy as np

import foldjax.models._cp_attention as cp_attention
from foldjax.models._cp import context_parallel
from foldjax.models._cp_attention import ring_tile_kernel_scope
from foldjax.models.boltz2.models.triangle import triangle_attention as serial_triangle
from foldjax.models.boltz2.models.trunk_blocks import msa as msa_module

DEVICES = int(os.environ["FOLDJAX_CP_PROBE_DEVICES"])
SIDE = math.isqrt(DEVICES)
assert jax.device_count() == DEVICES, jax.devices()
assert SIDE * SIDE == DEVICES and SIDE > 1, DEVICES

# Divisible by both grid sides on every split axis, so no arm measures a
# padded remainder.
CS, CM, CZ, HEADS, NUM_TOKENS, LAYERS = 5, 10, 12, 2, 7, 2
TOKENS, DEPTH = 12, 36

rng = np.random.default_rng(20260922)


def arr(*shape, scale=0.5):
    return jnp.asarray(rng.normal(size=shape, scale=scale), dtype=jnp.float32)


def norm(size):
    return {"scale": arr(size) * 0.1 + 1.0, "bias": arr(size) * 0.1}


def weight(fan_in, fan_out):
    return jnp.asarray(
        rng.normal(size=(fan_in, fan_out), scale=1.0 / np.sqrt(fan_in)),
        dtype=jnp.float32,
    )


def tri_att():
    return {
        "layer_norm": norm(CZ),
        "linear": {"kernel": weight(CZ, HEADS)},
        "mha": {
            name: {"kernel": weight(CZ, CZ)}
            for name in ("linear_q", "linear_k", "linear_v", "linear_g", "linear_o")
        },
    }


def transition(c):
    return {
        "norm": norm(c),
        "fc1": {"kernel": weight(c, 4 * c)},
        "fc2": {"kernel": weight(c, 4 * c)},
        "fc3": {"kernel": weight(4 * c, c)},
    }


def tri_mult():
    return {
        "norm_in": norm(CZ),
        "norm_out": norm(CZ),
        "g_in": {"kernel": weight(CZ, 2 * CZ)},
        "p_in": {"kernel": weight(CZ, 2 * CZ)},
        "p_out": {"kernel": weight(CZ, CZ)},
        "g_out": {"kernel": weight(CZ, CZ)},
    }


def msa_layer():
    return {
        "pair_weighted_averaging": {
            "norm_m": norm(CM),
            "norm_z": norm(CZ),
            "proj_m": {"kernel": weight(CM, HEADS * 3)},
            "proj_z": {"kernel": weight(CZ, HEADS)},
            "proj_g": {"kernel": weight(CM, HEADS * 3)},
            "proj_o": {"kernel": weight(HEADS * 3, CM)},
        },
        "msa_transition": transition(CM),
        "outer_product_mean": {
            "norm": norm(CM),
            "proj_a": {"kernel": weight(CM, 8)},
            "proj_b": {"kernel": weight(CM, 8)},
            "proj_o": {"kernel": weight(8 * 8, CZ), "bias": arr(CZ)},
        },
        "pairformer_layer": {
            "tri_mul_out": tri_mult(),
            "tri_mul_in": tri_mult(),
            "tri_att_start": tri_att(),
            "tri_att_end": tri_att(),
            "transition_z": transition(CZ),
        },
    }


def pair(tokens):
    # Rank-distinguishable: every entry carries its own row, column and
    # channel, so a tile that lands on the wrong device cannot cancel against
    # the tile that should have been there.
    rows = np.arange(tokens)[None, :, None, None]
    cols = np.arange(tokens)[None, None, :, None]
    chan = np.arange(CZ)[None, None, None, :]
    return jnp.asarray(
        0.05 * rows
        + 0.011 * cols
        + 0.003 * chan
        + rng.normal(size=(1, tokens, tokens, CZ)) * 0.1,
        dtype=jnp.float32,
    )


def keep_tokens(tokens):
    keep = rng.random(tokens) > 0.2
    keep[0] = True
    return keep


# --- the two spies ---------------------------------------------------------
#
# `requests` is what each ring call site asked the scope for, recorded on the
# *consumer* module's binding: `from ... import resolve_ring_tile_kernel`
# binds the name in that module, so patching `_cp_attention` would not be seen
# by the call site. The real resolver refuses `tokamax` off a GPU, which is the
# whole point of it, so the spy returns the request unchanged and the tile
# below is what makes the fused body runnable here.
requests = []
bodies = []
_real_resolve = serial_triangle.resolve_ring_tile_kernel
_real_tile = cp_attention._resolve_tile_attention


def spy_resolve(kernel):
    requests.append(kernel)
    if kernel == "tokamax":
        return kernel
    return _real_resolve(kernel)


def marker_tile(tile_kernel, precision):
    # The tile the ring evaluates, replaced by a marker that records the body
    # it was selected for. `TILE_IMPLEMENTATION` is the arithmetic: the
    # portable tile for a dispatch census, and tokamax's own XLA
    # implementation -- same `normalize_output=False, return_residuals=True`
    # triple as the Triton kernel -- for the contract arm.
    bodies.append(tile_kernel)
    if TILE_IMPLEMENTATION == "portable":
        return lambda *tile: cp_attention.tile_attention_xla(
            *tile, precision=precision
        )
    return lambda *tile: cp_attention.tile_attention_tokamax(
        *tile, precision=precision, implementation="xla"
    )


serial_triangle.resolve_ring_tile_kernel = spy_resolve
cp_attention._resolve_tile_attention = marker_tile
"""


_CENSUS_PROBE = _PRELUDE + textwrap.dedent(
    r"""
    TILE_IMPLEMENTATION = "portable"

    params = {
        "msa_proj": {"kernel": weight(NUM_TOKENS + 3, CM)},
        "s_proj": {"kernel": weight(CS, CM)},
        "layers": [msa_layer() for _ in range(LAYERS)],
    }
    z0 = arr(1, TOKENS, TOKENS, CZ)
    emb = arr(1, TOKENS, CS)
    keep = keep_tokens(TOKENS)
    feats = {
        "msa": jnp.asarray(
            rng.integers(0, NUM_TOKENS, size=(1, DEPTH, TOKENS)), dtype=jnp.int32
        ),
        "has_deletion": jnp.asarray(
            (rng.random((1, DEPTH, TOKENS)) > 0.7).astype(np.float32)
        ),
        "deletion_value": jnp.asarray(
            rng.random((1, DEPTH, TOKENS)), dtype=jnp.float32
        ),
        "msa_paired": jnp.asarray(
            (rng.random((1, DEPTH, TOKENS)) > 0.5).astype(np.float32)
        ),
        "msa_mask": jnp.asarray(
            ((rng.random((DEPTH, TOKENS)) > 0.15) & keep[None, :])[None].astype(
                np.float32
            )
        ),
        "token_pad_mask": jnp.asarray(keep[None].astype(np.float32)),
    }


    def stack(use_scan):
        def call():
            return msa_module.msa_module_forward(
                params,
                z0,
                emb,
                feats,
                num_tokens=NUM_TOKENS,
                use_scan=use_scan,
                chunk_size=3,
                pair_averaging_chunk=2,
            )

        return call


    # The unrolled stack traces one ring per layer and direction; the scan
    # traces one per direction and runs it once per layer. So this census is
    # over trace-time call sites, and the executed count of a prediction is
    # MSA blocks times two directions times the trunk's recycling passes.
    EXPECTED = {False: 2 * LAYERS, True: 2}

    with context_parallel(DEVICES, layout="2d"):
        for use_scan in (False, True):
            requests.clear()
            bodies.clear()
            jax.block_until_ready(jax.jit(stack(use_scan))())
            print(
                f"CENSUS_OFF scan={use_scan} sites={len(requests)} "
                f"requests={sorted(set(requests))} bodies={sorted(set(bodies))}"
            )
            # Off, every call site still reads the scope -- and reads the
            # shipped value, whose body never resolves a tile at all.
            assert len(requests) == EXPECTED[use_scan], (use_scan, requests)
            assert set(requests) == {"xla"}, requests
            assert not bodies, bodies

            requests.clear()
            bodies.clear()
            with ring_tile_kernel_scope("tokamax"):
                jax.block_until_ready(jax.jit(stack(use_scan))())
            print(
                f"CENSUS_ON scan={use_scan} sites={len(requests)} "
                f"requests={sorted(set(requests))} tiles={len(bodies)} "
                f"bodies={sorted(set(bodies))}"
            )
            # On, every call site asks for the fused tile and the answer
            # reaches the tile the ring evaluates. `bodies` counts block-body
            # traces (a peeled block and a `lax.scan` body), not call sites,
            # so it is bounded from below rather than equated.
            assert len(requests) == EXPECTED[use_scan], (use_scan, requests)
            assert set(requests) == {"tokamax"}, requests
            assert set(bodies) == {"tokamax"}, bodies
            assert len(bodies) >= len(requests), (bodies, requests)

    print("CENSUS_OK")
    """
)


_CONTRACT_PROBE = _PRELUDE + textwrap.dedent(
    r"""
    TILE_IMPLEMENTATION = "tokamax_xla"

    attention = tri_att()
    x = pair(TOKENS)
    keep = keep_tokens(TOKENS)
    dense_mask = jnp.ones((1, TOKENS, TOKENS), dtype=jnp.float32)
    sparse_mask = jnp.asarray((keep[:, None] & keep[None, :])[None]).astype(
        jnp.float32
    )


    def entry(mask, starting):
        # A fresh closure per arm: `jax.jit` keys its cache on the callable
        # and the scope is a context variable the trace reads, so a reused one
        # would replay the first arm's program.
        def call():
            return serial_triangle.triangle_attention_forward(
                attention, x, mask, starting=starting
            )

        return call


    def run(mask, starting, kernel):
        if kernel is None:
            return np.asarray(
                jax.device_get(jax.jit(entry(mask, starting))()), np.float64
            )
        with ring_tile_kernel_scope(kernel):
            return np.asarray(
                jax.device_get(jax.jit(entry(mask, starting))()), np.float64
            )


    with context_parallel(DEVICES, layout="2d"):
        for starting in (True, False):
            bodies.clear()
            off = run(dense_mask, starting, None)
            assert not bodies, bodies
            on = run(dense_mask, starting, "tokamax")
            assert set(bodies) == {"tokamax"}, bodies
            gap = float(np.abs(off - on).max())
            scale = max(float(np.abs(off).max()), 1.0)
            print(
                f"CONTRACT side={SIDE} starting={starting} dense "
                f"scale={scale:.3e} gap={gap:.3e}"
            )
            assert gap <= 1e-5 * scale, (starting, gap, scale)

            # The masked arm. A pair position the token mask keeps still has
            # the query token's own key, so the two bodies must agree there; a
            # query row with no valid key anywhere in the ring is the one
            # documented divergence -- zeros from the fused tile against the
            # two-pass body's reduction over `-1e9`-biased keys.
            #
            # Which *output* entries those are is not the same axis in the two
            # directions, which is the part worth writing down: the ending
            # node transposes its operands, attends, and transposes back, so
            # an all-masked internal query row `i` leaves the divergence in
            # output column `i` rather than output row `i`. The kept-by-kept
            # block is the agreement region either way.
            off = run(sparse_mask, starting, None)
            on = run(sparse_mask, starting, "tokamax")
            rows = np.asarray(keep)
            # The seed has to drop at least one token, or the excluded region
            # this arm reads is empty and says nothing.
            assert not rows.all(), rows
            block = np.ix_([0], np.flatnonzero(rows), np.flatnonzero(rows))
            gap = float(np.abs(off[block] - on[block]).max())
            scale = max(float(np.abs(off[block]).max()), 1.0)
            dropped = (
                on[:, ~rows] if starting else np.swapaxes(on, 1, 2)[:, ~rows]
            )
            dropped_off = (
                off[:, ~rows] if starting else np.swapaxes(off, 1, 2)[:, ~rows]
            )
            print(
                f"CONTRACT side={SIDE} starting={starting} sparse "
                f"kept={int(rows.sum())}/{TOKENS} scale={scale:.3e} "
                f"gap={gap:.3e} dropped_on={float(np.abs(dropped).max()):.3e} "
                f"dropped_off={float(np.abs(dropped_off).max()):.3e}"
            )
            assert gap <= 1e-5 * scale, (starting, gap, scale)
            # The divergence is the documented one and not an arbitrary
            # mismatch: the fused arm's keyless rows are zeros (these
            # parameters give `linear_o` no bias to add to them), and the
            # shipped arm's are not, so the region excluded above is excluded
            # for the reason given.
            assert np.abs(dropped).max() == 0.0, float(np.abs(dropped).max())
            assert np.abs(dropped_off).max() > 1e-3, float(
                np.abs(dropped_off).max()
            )

    print("CONTRACT_OK")
    """
)


def test_the_msa_stack_asks_the_scope_at_every_ring_call_site() -> None:
    """The defect, as a count: every MSA ring entry reads the option.

    Before this gate the MSA stack's ring took the default tile whatever the
    scope said, so `triangle_attention_ring_kernel=tokamax` described a
    prediction whose MSA passes ran the shipped two-pass body. The count is
    over trace-time call sites -- two triangle directions per MSA layer
    unrolled, two for the whole stack under `lax.scan` -- and the prediction
    level follows by arithmetic: four MSA blocks times two directions times
    the trunk's eleven recycling passes is 88 ring calls.
    """

    output = _run(_CENSUS_PROBE, devices=4)
    assert "CENSUS_OK" in output, output
    assert output.count("CENSUS_ON") == 2, output


@pytest.mark.parametrize("devices", [4, 9])
def test_the_msa_ring_is_the_same_attention_with_the_fused_tile(
    devices: int,
) -> None:
    """The option-on ring against the option-off ring, on asymmetric ranks.

    3x3 as well as 2x2: a 2x2 grid cannot tell a hop from its inverse
    (`cannon-sign-errors-need-side-3`), and this entry point builds its own
    `shard_map` -- the geometry checks, the row block and the bias skews are
    reached through this module's arguments, not the dispatcher's.
    """

    output = _run(_CONTRACT_PROBE, devices=devices)
    assert "CONTRACT_OK" in output, output
    assert output.count("CONTRACT ") == 4, output
