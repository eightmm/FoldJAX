"""The float32 OuterProductMean's output has to survive the row-shard boundary.

The plain (float32) ``outer_product_mean_forward`` used to assemble its output
with a chain of ``out = out.at[:, start:end].set(...)`` over token-row blocks.
That is arithmetically the same assembly as a concatenation -- the blocks are
disjoint and their offsets are constants -- and in the serial program the two
are bitwise equal. Under the one-dimensional context-parallel layout they are
not: when a downstream ``with_sharding_constraint`` pins the result on the row
axis, which ``msa_layer_forward`` does one statement later by handing ``z +
OuterProductMean(...)`` to ``pairformer_no_seq_layer_forward``'s
``shard_pair_rows``, XLA's SPMD partitioner (jax 0.11.1, Shardy and legacy
GSPMD alike) miscompiled the chain: the last row of every device's shard except
the last one -- row ``k * ceil(N / devices) - 1`` for ``k = 1 .. devices - 1``
-- came out wrong, by half the output scale, with no error anywhere.

So what is pinned here is the operator's own output against a float64 NumPy
reference of the same arithmetic, under the constraint that used to trigger it,
at two geometries:

* 848 tokens at chunk 77 -- the production geometry. 848 is a token count
  ``cp_aligned_padding`` produces and 77 is the chunk
  ``_auto_outer_product_chunk`` picks there from the shipped ``c_hidden`` of
  32, so nothing about this case is contrived: measured here on four devices,
  rows 211, 423 and 635 were 33.0 wrong on a scale of 58.3 -- 56% of the
  output -- while the same token count at chunk 128 was clean, which is why no
  chunk rule replaces removing the pattern. Two devices gave the same 33.0 at
  row 423 and eight gave 41.3.
* 16 tokens at chunk 3 -- a token count two, four and eight devices all divide
  exactly, to keep the record that this was never about token alignment. Its
  boundary rows were wrong by the whole output scale, 21.6 of 21.6 on four and
  eight devices and 13.2 on two.

Both arms enter at the operator rather than at the layer, because the layer
composes six more reassociated reductions and the question here is only whether
this operator's output is the value it computed. The constraint is applied to
the operator's result and, separately, to the residual add that the layer
actually pins, since the partitioner's choice depends on the surrounding
program: it fired for a one-layer unrolled MSA stack and not for the released
four layers under ``lax.scan``, and the released bfloat16 ``compute_dtype``
never met it at all because ``_outer_product_mean_amp`` has always assembled by
``jnp.concatenate``. A test that only covered the configuration that shipped
would therefore have passed throughout.

The bound is roundings of float32 against the NumPy reference, not a multiple
of any other arm's residual: the serial arm is measured against the same
reference in the same process, so neither arm can pass by the other one being
wrong. Sixty-four roundings, where the fixed one-dimensional arm measures 2.3
of them at 848 tokens and 1.7 at 16, against 2.5 and 1.4 for the serial arm.
The defect it replaces measured 4.7 million and 8.4 million.

A forced device count has to be set before JAX initialises, so the check runs
in a subprocess with four CPU devices. The probe uses ``#`` comments rather
than docstrings because it lives inside a triple-quoted literal.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

from tests.models.cp_probe_env import inherited_environment

#: Four shards, so the 848-token case splits at rows 211/423/635 and the
#: 16-token one at 3/7/11. The probe reads the count from the environment and
#: derives the boundaries, so the same body runs on any count the 1-D mesh
#: accepts -- two and eight were measured by hand, and failed by 33.0 and 41.3
#: at the first case before the fix.
_DEVICES = 4

_PROBE = textwrap.dedent(
    r"""
    import os

    import jax
    import jax.numpy as jnp
    import numpy as np

    from foldjax.models._cp import context_parallel, shard_pair_rows
    from foldjax.models.boltz2.models.trunk_blocks import msa as msa_module

    # `foldjax` is installed editable, so a child that lost `PYTHONPATH`
    # measures the install rather than the tree under test. Printed, because a
    # tripwire run against an older checkout has to be able to show which one
    # it loaded.
    print("module", msa_module.__file__, flush=True)

    DEVICES = int(os.environ["FOLDJAX_CP_PROBE_DEVICES"])
    assert jax.device_count() == DEVICES, jax.devices()

    # `HIDDEN` is the projected width `c_hidden`; the shipped checkpoint's is
    # 32 and only the chunk the budget derives from it matters here, which the
    # cases pass explicitly.
    CM, CZ, HIDDEN, DEPTH = 10, 12, 8, 4
    EPS = 1e-5
    ULP = 2.0**-23
    rng = np.random.default_rng(20260917)


    def f32(*shape, scale=0.5):
        return jnp.asarray(rng.normal(size=shape, scale=scale), dtype=jnp.float32)


    params = {
        "norm": {"scale": f32(CM) * 0.1 + 1.0, "bias": f32(CM) * 0.1},
        "proj_a": {"kernel": f32(CM, HIDDEN)},
        "proj_b": {"kernel": f32(CM, HIDDEN)},
        "proj_o": {"kernel": f32(HIDDEN * HIDDEN, CZ), "bias": f32(CZ)},
    }


    def reference(m, mask, rows=64):
        # The operator in float64 NumPy, block by block over output rows so
        # the [i, j, c, d] product never stands whole. Independent of the
        # assembly under test and of the partitioner: the blocks are written
        # into a preallocated result here, which is the spelling the defect
        # lives in, and NumPy has no partitioner.
        x = np.asarray(m, np.float64)
        valid = np.asarray(mask, np.float64)
        mean = x.mean(axis=-1, keepdims=True)
        variance = np.square(x - mean).mean(axis=-1, keepdims=True)
        normed = (x - mean) / np.sqrt(variance + EPS)
        normed = normed * np.asarray(
            params["norm"]["scale"], np.float64
        ) + np.asarray(params["norm"]["bias"], np.float64)
        a = normed @ np.asarray(params["proj_a"]["kernel"], np.float64)
        b = normed @ np.asarray(params["proj_b"]["kernel"], np.float64)
        a = a * valid[..., None]
        b = b * valid[..., None]
        count = np.maximum(np.einsum("bsi,bsj->bij", valid, valid), 1.0)[..., None]
        kernel = np.asarray(params["proj_o"]["kernel"], np.float64)
        bias = np.asarray(params["proj_o"]["bias"], np.float64)
        tokens = a.shape[2]
        out = np.empty((a.shape[0], tokens, tokens, CZ), np.float64)
        for start in range(0, tokens, rows):
            end = min(start + rows, tokens)
            z = np.einsum("bsic,bsjd->bijcd", a[:, :, start:end], b)
            z = z.reshape(*z.shape[:3], -1) / count[:, start:end]
            out[:, start:end] = z @ kernel + bias
        return out


    def run(fn, layout):
        # A fresh closure per arm: `jax.jit` keys its cache on the callable
        # and the mesh is a context variable the trace reads, so a reused one
        # would replay the first arm's program.
        def call():
            return fn()

        if layout == "serial":
            out = jax.block_until_ready(jax.jit(call)())
        else:
            with context_parallel(DEVICES, layout=layout):
                out = jax.block_until_ready(jax.jit(call)())
        return np.asarray(out, np.float64)


    # Sixty-four roundings of float32. The fixed arms measure a few; the
    # miscompile they replace measured 33.0 at the first case, five orders of
    # magnitude above this bound.
    ROUNDINGS = 64

    # (tokens, chunk): the production geometry that failed, and a token count
    # every device count here divides exactly, at a chunk that failed too --
    # so a reader cannot take token divisibility for the trigger.
    for tokens, chunk in ((848, 77), (16, 3)):
        m0 = f32(1, DEPTH, tokens, CM)
        mask = jnp.asarray((rng.random((1, DEPTH, tokens)) > 0.15).astype(np.float32))
        z0 = f32(1, tokens, tokens, CZ)

        def bare(chunk=chunk, m0=m0, mask=mask):
            return msa_module.outer_product_mean_forward(
                params,
                m0,
                mask,
                EPS,
                chunk_size=chunk,
                preserve_native_amp_shape=False,
            )

        # `pinned` is the operator's own output under the row constraint;
        # `residual` is the value `msa_layer_forward` actually hands the pair
        # stack, `z + OuterProductMean(...)`, whose constraint lives inside
        # `pairformer_no_seq_layer_forward`. The partitioner's choice depends
        # on the surrounding program, so both spellings are measured.
        def pinned(chunk=chunk):
            return shard_pair_rows(bare(chunk))

        def residual(chunk=chunk, z0=z0):
            return shard_pair_rows(z0 + bare(chunk))

        ref = reference(m0, mask)
        scale = max(float(np.abs(ref).max()), 1.0)
        bound = ROUNDINGS * ULP * scale
        shard = -(-tokens // DEVICES)
        boundaries = [step * shard - 1 for step in range(1, DEVICES)]

        serial = run(bare, "serial")
        serial_gap = float(np.abs(serial - ref).max())
        # The reference is the arm everything else is read against, so it is
        # checked first: eight roundings, which is what one float32 assembly
        # of this arithmetic costs against float64.
        assert serial_gap <= 8 * ULP * scale, (tokens, chunk, serial_gap, scale)

        for name, fn in (("pinned", pinned), ("residual", residual)):
            got = run(fn, "1d")
            if name == "residual":
                # Reading the operator's own value back out of the sum costs
                # one float32 rounding of that sum, which the bound below
                # covers twenty times over: measured 1.6e-5 against 4.5e-4.
                got = got - np.asarray(z0, np.float64)
            gap = float(np.abs(got - ref).max())
            per_row = np.abs(got - ref).max(axis=(0, 2, 3))
            bad = np.flatnonzero(per_row > bound).tolist()
            # The row indices are reported because the defect this replaces
            # had a signature: the last row of every shard but the last.
            assert gap <= bound, (
                tokens, chunk, name, gap, bound, scale, bad, boundaries
            )
            print(
                f"OPM N={tokens} chunk={chunk} {name} devices={DEVICES} "
                f"shard={shard} boundaries={boundaries} scale={scale:.3e} "
                f"serial={serial_gap:.3e} 1d={gap:.3e} (<= {bound:.3e})",
                flush=True,
            )

    print("OPM_ROW_SHARD_ASSEMBLY_OK")
    """
)


def test_the_float32_outer_product_mean_survives_the_row_shard_boundary() -> None:
    """The operator's output, row-constrained, against a float64 reference."""

    completed = subprocess.run(
        [sys.executable, "-c", _PROBE],
        capture_output=True,
        text=True,
        env={
            "JAX_PLATFORMS": "cpu",
            "XLA_FLAGS": f"--xla_force_host_platform_device_count={_DEVICES}",
            "FOLDJAX_CP_PROBE_DEVICES": str(_DEVICES),
            **inherited_environment(),
        },
        timeout=1800,
    )
    output = completed.stdout + completed.stderr
    assert completed.returncode == 0, output
    assert "OPM_ROW_SHARD_ASSEMBLY_OK" in output, output
