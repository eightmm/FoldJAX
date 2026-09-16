"""The MSA stack splits its alignment depth on the grid's column axis.

Under the two-dimensional layout the MSA tensor ``m`` [B, M, N, C] is split on
BOTH grid axes -- alignment depth on the column axis, tokens on the row axis --
so rank ``(r, s)`` owns alignment rows ``M_s`` against token positions ``N_r``,
while the pair tensor at that same rank owns token rows ``N_r`` against token
columns ``N_s``.  The column axis therefore names two different things in the
two stacks, and the two operators that read one and write the other bridge them
with explicit collectives: PairWeightedAveraging gathers its small projected
head logits along the column axis and streams the widened values around a ring
on the row axis; OuterProductMean streams its key operand and mask around that
same ring and reduces numerator and mask count along the column axis.

Neither bridge can be bitwise equal to the serial program -- summing a shard at
a time reassociates both reductions -- so what is pinned here is equality
within tolerance *plus* the three things reassociation is not allowed to
become:

* the mean must be taken after the column reduction.  An implementation that
  averaged locally normalised means would be wrong by about a factor of the
  number of shards that contribute, and the arm whose mask zeroes every
  alignment row of one depth shard is the one that would show it: the correct
  answer there is the serial answer, and a per-shard mean is not.
* the output bias must be added once.  The arm whose mask is empty everywhere
  has a zero numerator, so the whole output is the bias and nothing else; a
  bias added per shard is off by a factor of the grid side.  That arm runs on
  both sides of ``_NATIVE_CHUNK_THRESHOLD``, because the released regime above
  it is where this module adds that bias itself, outside its hidden-chunk
  loop, rather than leaving it to ``linear``.
* the mask must count the same rows.  Both arms above compare denominators
  against the serial ones, and the sparse arms compare against a serial
  program that has every row.

The tolerances are calibrated rather than chosen: the operator arms and the
single-layer arm are bounded at eight roundings of the compute dtype, and the
whole-module arm against the residual the *one-dimensional* layout already has
against serial in the same process, because that is the reassociation that was
there before this change and the question is whether splitting the depth adds a
different kind of error rather than more of the same one.

One arm enters at ``msa_layer_forward`` with a depth and a token count that
divide neither grid.  That is the only way to reach the transition's own
pad-and-slice -- the module pads the depth once above it, for the whole stack
-- and it is also where both bridges compose with the pair stack rather than
being measured one at a time.  That arm's 1-D residual is printed and not used:
at a token count the 1-D mesh does not divide either, it is of order the values
themselves, which is a defect somewhere in the row-sharded pair stack and not
in anything here, and calibrating against it would hide this file's own
arithmetic behind that one's size.

The census is a before/after pair in one process: ``_on_msa_grid`` patched off
and ``shard_msa`` patched to identity is exactly the program that ran before the
depth axis was split, so the per-device shape, the widest value carrying the
residual stream's channel width, and the all-gathers all have something to move
against.  A forced device count has to be set before JAX initialises, so the
gates run in a subprocess, on four devices for the 2x2 grid and on nine for the
3x3 one -- a 2x2 grid cannot catch a ring that hops the wrong way, because one
hop forward and one hop back are the same schedule on two devices.  Probe
sources use ``#`` comments rather than docstrings because they live inside a
triple-quoted literal.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

from tests.models.cp_probe_env import inherited_environment

_PROBE = textwrap.dedent(
    r"""
    import math
    import os
    import re

    import jax
    import jax.numpy as jnp
    import numpy as np

    from jax.sharding import NamedSharding, PartitionSpec

    from foldjax.models._cp import CP_ROW_AXIS, context_parallel, cp_mesh
    from foldjax.models.boltz2.models.trunk_blocks import msa as msa_module

    DEVICES = int(os.environ["FOLDJAX_CP_PROBE_DEVICES"])
    SIDE = math.isqrt(DEVICES)
    assert jax.device_count() == DEVICES, jax.devices()
    assert SIDE * SIDE == DEVICES and SIDE > 1, DEVICES

    # `CM` is the residual stream's channel width and the census marker: no
    # other *activation* in this program carries it, and the parameters that do
    # (`fc3`, `proj_o`, `msa_proj`, `s_proj`, the norms) are all smaller than
    # one device's share of the stream at the census shape below, which is what
    # lets an element-count bound separate the two.
    CS, CM, CZ, HEADS, CH, NUM_TOKENS, LAYERS = 5, 10, 12, 2, 3, 7, 2
    # Divisible by 2 and by 3, so the census arm's per-device shape is exact
    # rather than a padded remainder, and large enough that one device's share
    # of the stream outweighs the largest parameter that ends in `CM`.
    CENSUS_DEPTH, CENSUS_TOKENS = 36, 12

    # `(?:[0-9]+,)*` rather than `[0-9,]*`: the latter also matches a shape
    # whose dimensions merely end in the right digits.
    CM_TAIL = re.compile(r"\b\w+\[((?:[0-9]+,)*" + str(CM) + r")\]")
    RESULT_SHAPE = re.compile(r"=\s*\w+\[((?:[0-9]+,)*[0-9]+)\]")

    rng = np.random.default_rng(20260917)


    def elements(dims):
        total = 1
        for size in dims.split(","):
            total *= int(size)
        return total


    def widest_stream(text):
        # The largest value in the program whose last dimension is the
        # residual stream's channel width, counted in elements so a value XLA
        # flattened still counts.
        return max((elements(dims) for dims in CM_TAIL.findall(text)), default=0)


    def gathered(text):
        found = []
        for line in text.splitlines():
            if " all-gather(" not in line and " all-gather-start(" not in line:
                continue
            match = RESULT_SHAPE.search(line)
            if match:
                found.append(match.group(1))
        return sorted(set(found))


    def arr(*shape, scale=0.5):
        return jnp.asarray(rng.normal(size=shape, scale=scale), dtype=jnp.float32)


    def norm(c):
        return {"scale": arr(c) * 0.1 + 1.0, "bias": arr(c) * 0.1}


    def weight(fan_in, fan_out, dtype):
        return jnp.asarray(
            rng.normal(size=(fan_in, fan_out), scale=1.0 / np.sqrt(fan_in)),
            dtype=dtype,
        )


    def pwa_params(dtype):
        return {
            "norm_m": norm(CM),
            "norm_z": norm(CZ),
            "proj_m": {"kernel": weight(CM, HEADS * CH, dtype)},
            "proj_z": {"kernel": weight(CZ, HEADS, dtype)},
            "proj_g": {"kernel": weight(CM, HEADS * CH, dtype)},
            "proj_o": {"kernel": weight(HEADS * CH, CM, dtype)},
        }


    # Eight projected channels rather than four, so the AMP path's four-wide
    # hidden chunking is two chunks and not one: the bias that branch adds by
    # hand is added outside that loop, and one chunk could not tell the two
    # placements apart.
    OPM_HIDDEN = 8


    def opm_params(dtype):
        return {
            "norm": norm(CM),
            "proj_a": {"kernel": weight(CM, OPM_HIDDEN, dtype)},
            "proj_b": {"kernel": weight(CM, OPM_HIDDEN, dtype)},
            "proj_o": {
                "kernel": weight(OPM_HIDDEN * OPM_HIDDEN, CZ, dtype),
                "bias": arr(CZ),
            },
        }


    def transition(c, dtype):
        return {
            "norm": norm(c),
            "fc1": {"kernel": weight(c, 4 * c, dtype)},
            "fc2": {"kernel": weight(c, 4 * c, dtype)},
            "fc3": {"kernel": weight(4 * c, c, dtype)},
        }


    def tri_mult(dtype):
        return {
            "norm_in": norm(CZ),
            "norm_out": norm(CZ),
            "g_in": {"kernel": weight(CZ, 2 * CZ, dtype)},
            "p_in": {"kernel": weight(CZ, 2 * CZ, dtype)},
            "p_out": {"kernel": weight(CZ, CZ, dtype)},
            "g_out": {"kernel": weight(CZ, CZ, dtype)},
        }


    def tri_att(dtype):
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


    def layer_params(dtype):
        return {
            "pair_weighted_averaging": pwa_params(dtype),
            "msa_transition": transition(CM, dtype),
            "outer_product_mean": opm_params(dtype),
            "pairformer_layer": {
                "tri_mul_out": tri_mult(dtype),
                "tri_mul_in": tri_mult(dtype),
                "tri_att_start": tri_att(dtype),
                "tri_att_end": tri_att(dtype),
                "transition_z": transition(CZ, dtype),
            },
        }


    def module_params(dtype):
        return {
            "msa_proj": {"kernel": weight(NUM_TOKENS + 3, CM, dtype)},
            "s_proj": {"kernel": weight(CS, CM, dtype)},
            "layers": [layer_params(dtype) for _ in range(LAYERS)],
        }


    def stream(depth, tokens):
        # Rank-distinguishable: every entry carries its own alignment row,
        # token position and channel, so a tile that lands on the wrong device
        # cannot cancel against the tile that should have been there.
        rows = np.arange(depth)[None, :, None, None]
        toks = np.arange(tokens)[None, None, :, None]
        chan = np.arange(CM)[None, None, None, :]
        return jnp.asarray(
            0.05 * rows
            + 0.011 * toks
            + 0.003 * chan
            + rng.normal(size=(1, depth, tokens, CM)) * 0.1,
            dtype=jnp.float32,
        )


    def masks(depth, tokens, mode):
        keep = rng.random(tokens) > 0.2
        keep[0] = True
        token_mask = jnp.asarray(
            (keep[:, None] & keep[None, :])[None]
        ).astype(jnp.float32)
        rows = (rng.random((depth, tokens)) > 0.15) & keep[None, :]
        if mode == "empty_shard":
            # Every alignment row of the *second* depth shard, which is the
            # arm that separates a mean taken after the column reduction from
            # one averaged over locally normalised shards.
            per = -(-depth // SIDE)
            rows[per : 2 * per] = False
        elif mode == "empty":
            rows[:] = False
        elif mode != "sparse":
            raise AssertionError(mode)
        return token_mask, jnp.asarray(rows[None]).astype(jnp.float32)


    def run(fn, layout, *, want_local=False):
        # A fresh closure per arm: `jax.jit` keys its cache on the callable and
        # the mesh is a context variable the trace reads, so a reused one would
        # replay the first arm's program.
        def call():
            return fn()

        if layout == "serial":
            out = jax.jit(call)()
            jax.block_until_ready(out)
            return np.asarray(out, np.float64), None, None
        with context_parallel(DEVICES, layout=layout):
            jitted = jax.jit(call)
            text = jitted.lower().compile().as_text()
            out = jitted()
            jax.block_until_ready(out)
            local = (
                tuple(
                    int(size)
                    for size in next(iter(out.addressable_shards)).data.shape
                )
                if want_local
                else None
            )
        return np.asarray(out, np.float64), text, local


    def ulps(dtype):
        return 2.0**-8 if dtype == jnp.bfloat16 else 2.0**-23


    # --- the two bridges, against serial ----------------------------------
    #
    # `sparse` covers uneven padding on both axes: 7 alignment rows and 13
    # tokens divide neither grid, so both bridges pad and slice, and 12/12
    # divides both so they do not.
    CONFIGS = (
        (12, 12, "sparse"),
        (7, 13, "sparse"),
        (8, 12, "empty_shard"),
        (9, 12, "empty"),
    )

    # A probe-sized token count sits below `_NATIVE_CHUNK_THRESHOLD`, where
    # the AMP path runs every head in one group and OuterProductMean neither
    # chunks its hidden axis nor adds its bias by hand. The released regime is
    # the other side of it: one head group at a time, four-wide hidden chunks
    # with a reduction per chunk, and the FP32 bias added outside that loop --
    # which is the branch where "the bias is added once" is this module's own
    # arithmetic rather than `linear`'s. The threshold is therefore moved, as
    # the transition gate moves it, and both arms see the same value. Only the
    # AMP path reads it, so float32 keeps the one setting.
    SHIPPED_THRESHOLD = msa_module._NATIVE_CHUNK_THRESHOLD
    ARMS = (
        ("float32", jnp.float32, (SHIPPED_THRESHOLD,)),
        ("bfloat16", jnp.bfloat16, (SHIPPED_THRESHOLD, 0)),
    )

    for dtype_name, dtype, thresholds in ARMS:
        for threshold in thresholds:
            msa_module._NATIVE_CHUNK_THRESHOLD = threshold
            native = "native" if threshold == 0 else "plain"
            for depth, tokens, mode in CONFIGS:
                pwa, opm = pwa_params(dtype), opm_params(dtype)
                m0 = stream(depth, tokens)
                z0 = arr(1, tokens, tokens, CZ)
                token_mask, msa_mask = masks(depth, tokens, mode)
                for row_chunk in (None, 2):
                    def pwa_forward(
                        pwa=pwa,
                        m0=m0,
                        z0=z0,
                        token_mask=token_mask,
                        row_chunk=row_chunk,
                    ):
                        return msa_module.pair_weighted_averaging_forward(
                            pwa, m0, z0, token_mask, row_chunk_size=row_chunk
                        )

                    ref, _, _ = run(pwa_forward, "serial")
                    got, _, _ = run(pwa_forward, "2d")
                    assert got.shape == ref.shape == (1, depth, tokens, CM), got.shape
                    scale = max(float(np.abs(ref).max()), 1.0)
                    gap = float(np.abs(got - ref).max())
                    bound = 8 * ulps(dtype) * scale
                    assert gap <= bound, (
                        dtype_name, native, depth, tokens, mode, row_chunk, gap, bound
                    )
                    print(
                        f"PWA {dtype_name} {native} side={SIDE} M={depth} "
                        f"N={tokens} {mode} rows={row_chunk} maxabs={gap:.3e} "
                        f"(<= {bound:.3e})",
                        flush=True,
                    )

                def opm_forward(opm=opm, m0=m0, msa_mask=msa_mask):
                    return msa_module.outer_product_mean_forward(
                        opm,
                        m0,
                        msa_mask,
                        1e-5,
                        chunk_size=3,
                        preserve_native_amp_shape=True,
                    )

                ref, _, _ = run(opm_forward, "serial")
                got, _, _ = run(opm_forward, "2d")
                assert got.shape == ref.shape == (1, tokens, tokens, CZ), got.shape
                scale = max(float(np.abs(ref).max()), 1.0)
                gap = float(np.abs(got - ref).max())
                bound = 8 * ulps(dtype) * scale
                assert gap <= bound, (
                    dtype_name, native, depth, tokens, mode, gap, bound
                )
                extra = ""
                if mode == "empty":
                    # A zero numerator over a clamped denominator leaves the
                    # output bias and nothing else: bitwise what the serial
                    # program produces, constant over every (i, j), and one
                    # rounding from the bias itself. `side` times the bias is
                    # what a bias added per shard would give, and that is not
                    # a rounding away -- which is the reading that holds on
                    # both branches, where the bias is added in different
                    # places.
                    np.testing.assert_array_equal(got, ref)
                    np.testing.assert_array_equal(
                        got, np.broadcast_to(got[:, :1, :1], got.shape)
                    )
                    bias = np.asarray(opm["proj_o"]["bias"], np.float64)
                    away = float(np.abs(got[0, 0, 0] - bias).max())
                    allowed = 8 * ulps(dtype) * max(float(np.abs(bias).max()), 1.0)
                    assert away <= allowed, (dtype_name, native, away, allowed)
                    extra = f" == bias ({away:.3e} <= {allowed:.3e})"
                print(
                    f"OPM {dtype_name} {native} side={SIDE} M={depth} "
                    f"N={tokens} {mode} maxabs={gap:.3e} (<= {bound:.3e}){extra}",
                    flush=True,
                )
    msa_module._NATIVE_CHUNK_THRESHOLD = SHIPPED_THRESHOLD

    # --- one layer, at a depth and a token count that divide neither grid ---
    #
    # `msa_module_forward` pads the alignment depth once for the whole stack,
    # so nothing below it ever sees an indivisible one -- except a caller that
    # enters at the layer, which is what this arm is. It is the only arm that
    # reaches `_msa_transition_grid`'s pad-and-slice, and it composes both
    # bridges with the pair stack rather than testing them one at a time.
    for dtype_name, dtype in (("float32", jnp.float32), ("bfloat16", jnp.bfloat16)):
        layer = layer_params(dtype)
        depth, tokens = 7, 13
        m0 = stream(depth, tokens)
        z0 = arr(1, tokens, tokens, CZ)
        token_mask, msa_mask = masks(depth, tokens, "sparse")

        def one_layer(layer=layer, z0=z0, m0=m0, token=token_mask, rows=msa_mask):
            z, m = msa_module.msa_layer_forward(
                layer, z0, m0, token, rows, chunk_size=3, pair_averaging_chunk=2
            )
            # One value, so the arms compare in one program: the pair carry
            # scaled beside the MSA carry, both of which the layer updates.
            return jnp.concatenate(
                (
                    jnp.reshape(z.astype(jnp.float32), (1, -1)),
                    jnp.reshape(m.astype(jnp.float32), (1, -1)),
                ),
                axis=1,
            )

        ref, _, _ = run(one_layer, "serial")
        one_d, _, _ = run(one_layer, "1d")
        two_d, _, _ = run(one_layer, "2d")
        scale = max(float(np.abs(ref).max()), 1.0)
        gap_1d = float(np.abs(one_d - ref).max())
        gap_2d = float(np.abs(two_d - ref).max())
        # Roundings, not a multiple of the 1-D residual, and deliberately so
        # here: at a token count that does not divide the 1-D mesh either,
        # that arm's own residual against serial is of order the values
        # themselves (measured 3.7 in float32 on four devices, against 4.6e-6
        # for the grid) -- something in the row-sharded pair stack, not in
        # anything this file changes, and a bound calibrated against it would
        # be the size of that defect rather than of this one. It is printed
        # rather than used. Sixty-four roundings rather than the operators'
        # eight, because a layer is six reassociated reductions composed with
        # the pair stack; measured 4.6e-6 in float32 and 9.9e-2 in bfloat16
        # against bounds of 3.3e-5 and 1.1, where the 1-D arm above misses by
        # six orders of magnitude.
        bound = 64 * ulps(dtype) * scale
        assert gap_2d <= bound, (dtype_name, scale, gap_1d, gap_2d, bound)
        print(
            f"layer {dtype_name} side={SIDE} M={depth} N={tokens} "
            f"scale={scale:.3e} 1d={gap_1d:.3e} 2d={gap_2d:.3e} "
            f"(<= {bound:.3e})",
            flush=True,
        )

    # --- the whole module, against serial and against the 1-D layout -------
    for dtype_name, dtype in (("float32", jnp.float32), ("bfloat16", jnp.bfloat16)):
        params = module_params(dtype)
        depth, tokens = 35, 12
        z0 = arr(1, tokens, tokens, CZ)
        emb = arr(1, tokens, CS)
        keep = rng.random(tokens) > 0.2
        keep[0] = True
        feats = {
            "msa": jnp.asarray(
                rng.integers(0, NUM_TOKENS, size=(1, depth, tokens)), dtype=jnp.int32
            ),
            "has_deletion": jnp.asarray(
                (rng.random((1, depth, tokens)) > 0.7).astype(np.float32)
            ),
            "deletion_value": jnp.asarray(
                rng.random((1, depth, tokens)), dtype=jnp.float32
            ),
            "msa_paired": jnp.asarray(
                (rng.random((1, depth, tokens)) > 0.5).astype(np.float32)
            ),
            "msa_mask": jnp.asarray(
                ((rng.random((depth, tokens)) > 0.15) & keep[None, :])[None].astype(
                    np.float32
                )
            ),
            "token_pad_mask": jnp.asarray(keep[None].astype(np.float32)),
        }

        for use_scan in (True, False):
            def module(params=params, z0=z0, emb=emb, feats=feats, use_scan=use_scan):
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

            ref, _, _ = run(module, "serial")
            scale = max(float(np.abs(ref).max()), 1.0)
            one_d, _, _ = run(module, "1d")
            two_d, _, _ = run(module, "2d")
            gap_1d = float(np.abs(one_d - ref).max())
            gap_2d = float(np.abs(two_d - ref).max())
            # The 1-D residual is the reassociation context parallelism had
            # before the depth axis was split. Splitting it may not introduce
            # a different *kind* of error, which is what a multiple of that
            # residual bounds and an absolute constant would not.
            bound = max(4 * gap_1d, 8 * ulps(dtype) * scale)
            assert gap_2d <= bound, (dtype_name, use_scan, gap_1d, gap_2d, bound)
            print(
                f"module {dtype_name} scan={use_scan} side={SIDE} M={depth} "
                f"N={tokens} 1d={gap_1d:.3e} 2d={gap_2d:.3e} (<= {bound:.3e})",
                flush=True,
            )

    # --- the census, against the program that ran before ------------------
    #
    # `_on_msa_grid` off and `shard_msa` replaced by the spec the MSA tensor
    # used to carry -- `PartitionSpec(None, None, "cp_row", None)`, tokens on
    # the grid rows and the alignment whole -- is exactly the 2-D program as
    # it stood before the depth split: both bridges fall back to their
    # partitioner-served spellings against a replicated alignment axis.
    shipped_grid = msa_module._on_msa_grid
    shipped_shard = msa_module.shard_msa
    fired = {"grid": 0, "shard": 0, "identity": 0}


    def grid_off():
        fired["grid"] += 1
        return False


    def shard_tokens_only(x, **kwargs):
        fired["shard"] += 1
        mesh = cp_mesh()
        if mesh is None:
            return x
        entries = [None] * x.ndim
        entries[2] = CP_ROW_AXIS
        return jax.lax.with_sharding_constraint(
            x, NamedSharding(mesh, PartitionSpec(*entries))
        )


    def shard_identity(x, **kwargs):
        fired["identity"] += 1
        return x


    def arm(fn, *, split, want_local=False):
        if not split:
            msa_module._on_msa_grid = grid_off
            msa_module.shard_msa = shard_tokens_only
        try:
            return run(fn, "2d", want_local=want_local)
        finally:
            msa_module._on_msa_grid = shipped_grid
            msa_module.shard_msa = shipped_shard


    dtype = jnp.bfloat16
    params = module_params(dtype)
    depth, tokens = CENSUS_DEPTH, CENSUS_TOKENS
    z0 = arr(1, tokens, tokens, CZ)
    emb = arr(1, tokens, CS)
    keep = np.ones(tokens, dtype=bool)
    feats = {
        "msa": jnp.asarray(
            rng.integers(0, NUM_TOKENS, size=(1, depth, tokens)), dtype=jnp.int32
        ),
        "has_deletion": jnp.asarray(
            (rng.random((1, depth, tokens)) > 0.7).astype(np.float32)
        ),
        "deletion_value": jnp.asarray(
            rng.random((1, depth, tokens)), dtype=jnp.float32
        ),
        "msa_paired": jnp.asarray(
            (rng.random((1, depth, tokens)) > 0.5).astype(np.float32)
        ),
        "msa_mask": jnp.asarray(
            (rng.random((depth, tokens)) > 0.15)[None].astype(np.float32)
        ),
        "token_pad_mask": jnp.asarray(keep[None].astype(np.float32)),
    }
    token_mask = jnp.asarray(
        (keep[:, None] & keep[None, :])[None]
    ).astype(jnp.float32)


    def census_module(use_scan=False):
        # Unrolled, so the widest value carrying `CM` is an activation: under
        # a scan the stacked `fc3` parameter is wider than one device's share
        # of the stream at this size, and the bound would be measuring it.
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


    def census_layer():
        return msa_module.msa_layer_forward(
            params["layers"][0],
            z0,
            stream(depth, tokens),
            token_mask,
            feats["msa_mask"],
            chunk_size=3,
            pair_averaging_chunk=2,
        )[1]


    _, before_text, _ = arm(census_module, split=False)
    _, after_text, _ = arm(census_module, split=True)
    assert fired["grid"] and fired["shard"], fired

    # The tripwire the rest of the census rests on: one device's share of the
    # residual stream. Without it "the depth was split" would be an assertion
    # about a program that never split anything.
    _, _, before_local = arm(census_layer, split=False, want_local=True)
    _, _, after_local = arm(census_layer, split=True, want_local=True)
    assert before_local == (1, depth, tokens // SIDE, CM), before_local
    assert after_local == (1, depth // SIDE, tokens // SIDE, CM), after_local

    share = (depth // SIDE) * (tokens // SIDE) * CM
    before_widest, after_widest = widest_stream(before_text), widest_stream(after_text)
    assert after_widest <= share, (after_widest, share)
    # And the bound discriminates: the program that did not split the depth
    # breaks it, by the whole alignment.
    assert before_widest > share, (before_widest, share)

    # The only value that may be gathered is the projected head logits, whose
    # key axis is the one the softmax has to normalise over:
    # `[B, heads, N/side, N]`. Nothing carrying the alignment may appear --
    # which is what the unsplit program does gather.
    logits = HEADS * (tokens // SIDE) * tokens
    after_gathers = gathered(after_text)
    before_gathers = gathered(before_text)
    assert after_gathers, after_gathers
    assert all(elements(dims) <= 2 * logits for dims in after_gathers), (
        after_gathers, logits
    )
    assert any(elements(dims) > 2 * logits for dims in before_gathers), (
        before_gathers, logits
    )

    print(
        f"census side={SIDE} local {before_local} -> {after_local}, widest "
        f"{before_widest} -> {after_widest} (<= {share}), gathers "
        f"{before_gathers} -> {after_gathers}",
        flush=True,
    )

    # --- and neither layout below the grid may have moved -----------------
    #
    # `shard_msa` is an identity off the 2-D layout and the bridges are not
    # dispatched there, so the serial and 1-D programs have to be the ones the
    # patched arm lowers -- the same text, not merely the same numbers.
    for layout in ("serial", "1d"):
        def lowered(layout=layout):
            if layout == "serial":
                return jax.jit(census_module).lower().as_text()
            with context_parallel(DEVICES, layout="1d"):
                return jax.jit(census_module).lower().as_text()

        msa_module._on_msa_grid = grid_off
        msa_module.shard_msa = shard_identity
        try:
            patched = lowered()
        finally:
            msa_module._on_msa_grid = shipped_grid
            msa_module.shard_msa = shipped_shard
        shipped = lowered()
        assert fired["identity"], fired
        normalise = lambda text: re.sub(
            r"\s+", " ", re.sub(r"jit_[A-Za-z0-9_]+", "jit_fn", text)
        )
        assert normalise(patched) == normalise(shipped), layout
        print(f"{layout} lowering unchanged by the grid dispatch", flush=True)

    print("MSA_GRID_BRIDGES_OK")
    """
)


def _run_probe(devices: int) -> str:
    completed = subprocess.run(
        [sys.executable, "-c", _PROBE],
        capture_output=True,
        text=True,
        env={
            "JAX_PLATFORMS": "cpu",
            "XLA_FLAGS": f"--xla_force_host_platform_device_count={devices}",
            "FOLDJAX_CP_PROBE_DEVICES": str(devices),
            **inherited_environment(),
        },
        timeout=1800,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return completed.stdout


#: Four devices are the smallest grid the 2-D layout accepts; nine are the
#: smallest that can tell a ring hop from its reverse.
@pytest.mark.parametrize("devices", [4, 9])
def test_the_msa_stack_splits_its_depth_over_the_grid_columns(devices: int) -> None:
    """Both bridges, both dtypes, uneven padding, empty masks, the census."""

    assert "MSA_GRID_BRIDGES_OK" in _run_probe(devices)
