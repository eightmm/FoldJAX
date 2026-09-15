"""Compare OpenFold3's distributed diffusion atom graph against the serial one.

Run under a forced device count with ``FOLDJAX_CP_PROBE_DEVICES`` and
``FOLDJAX_CP_PROBE_LAYOUT`` set; prints per-stage residuals, the collectives
each program contains and the lowering fingerprints, then asserts the things
that make the comparison mean something:

* the mesh variant traced a *different* program (fingerprints differ) and saw a
  live mesh and a live atom-block plan (positive tripwires);
* the sharded lowering carries the collectives the mechanism needs and no
  ``all-gather``;
* no full-width atom activation, atom-pair block cache, projected token-pair
  tensor or token attention bias survives per device;
* every stage agrees with serial to FP32 reduction-order tolerance.

A CP parity check is vacuous if the same closure is re-jitted -- ``jit`` keys
its cache on the callable, not on the mesh ``ContextVar`` -- so each variant
gets a freshly built closure, and the fingerprint inequality is what proves the
sharded program was traced rather than a cached serial one replayed.

There is no unrolled arm. ``diffusion_transformer`` and ``atom_transformer``
both default to ``scan_blocks=True``, so both stacks already run as one
``lax.scan`` body *inside* the ``shard_map``: the ring gathers' ``ppermute``
lives in the scan body and the plan's window tables are scan constants. That
composition is what every arm here traces, rather than something a separate
arm would have to opt into.

Also usable on a GPU with real device counts: it takes its shapes from
``tests.models.openfold3.atom_cp_fixtures`` and its arms from the environment.
"""

from __future__ import annotations

import hashlib
import os
import re
import sys

import jax
import numpy as np

import foldjax.models.openfold3.models.atom_blocks as atom_blocks
from foldjax.models._cp import context_parallel, cp_layout
from foldjax.models.openfold3.models.atom_cp import atom_block_plan
from foldjax.models.openfold3.models.atom_features import (
    atom_attention_decoder,
    atom_attention_encoder,
)
from foldjax.models.openfold3.models.denoiser import denoise
from foldjax.models.openfold3.models.diffusion_schedule import noise_schedule
from foldjax.models.openfold3.models.sampler import sample_diffusion
from tests.models.openfold3.atom_cp_fixtures import (
    ATOM_HEADS,
    C_ATOM,
    C_ATOM_PAIR,
    N_ATOM,
    N_BLOCKS,
    N_KEY,
    N_QUERY,
    N_SAMPLE,
    N_TOKEN,
    REAL_ATOMS,
    SIGMA_DATA,
    TOKEN_HEADS,
    build_case,
    build_params,
)

#: Reduction order changes; nothing else does. The atom->token mean is a
#: reduce-scatter instead of one local contraction, the key blocks are a masked
#: sum over rotated atom shards, and the sparse token-pair lookup is a masked
#: sum over rotated pair tiles. Agreement is to FP32 accumulation noise rather
#: than bitwise.
RELATIVE_TOLERANCE = 1e-5

#: Steps in the short sampler arm. Three is enough for the rollout's own scan
#: to carry a sharded denoiser body and for an error to compound visibly.
SAMPLER_STEPS = 3


def _fingerprint(text: str) -> str:
    text = re.sub(r",?\s*metadata=\{[^}]*\}", "", text)
    text = re.sub(r",?\s*stack_frame_id=\d+", "", text)
    return hashlib.sha256(re.sub(r"\s+", " ", text).strip().encode()).hexdigest()


def _report(name: str, reference: np.ndarray, got: np.ndarray) -> None:
    delta = np.abs(reference - got)
    scale = max(float(np.abs(reference).max()), 1e-12)
    print(
        f"  {name:<14} shape {tuple(reference.shape)} "
        f"max {delta.max():.3e} p99 {np.percentile(delta, 99):.3e} "
        f"rms {np.sqrt((delta**2).mean()):.3e} scale {scale:.3f}"
    )
    np.testing.assert_allclose(
        reference,
        got,
        rtol=RELATIVE_TOLERANCE,
        # Scaled by the stage's own magnitude rather than an absolute floor:
        # the sampler's early steps carry coordinates tens of units wide and
        # the decoder update fractions of one, and a single constant would be
        # vacuous for the first and impossible for the second.
        atol=RELATIVE_TOLERANCE * scale,
    )


def main() -> int:
    devices = int(os.environ["FOLDJAX_CP_PROBE_DEVICES"])
    layout = os.environ["FOLDJAX_CP_PROBE_LAYOUT"]
    assert jax.device_count() == devices, jax.devices()

    case = build_case()
    params = build_params()
    schedule = noise_schedule(
        SAMPLER_STEPS, sigma_data=SIGMA_DATA, s_max=160.0, s_min=4e-4, p=7.0
    )
    tripwires: list[tuple[str | None, bool, int, bool]] = []

    def build(distributed: bool):
        """One fresh closure per variant; a reused one would never retrace."""

        def run(batch, xl_noisy, t, si, si_trunk, zij):
            ai, ql, cl, plm = atom_attention_encoder(
                batch,
                params.atom_attn_enc,
                n_query=N_QUERY,
                n_key=N_KEY,
                no_heads=ATOM_HEADS,
                n_token=N_TOKEN,
                rl=xl_noisy,
                si_trunk=si_trunk,
                zij_trunk=zij,
                cp_atom_windows=distributed,
            )
            update = atom_attention_decoder(
                batch,
                ai,
                ql,
                cl,
                plm,
                params.atom_attn_dec,
                n_query=N_QUERY,
                n_key=N_KEY,
                no_heads=ATOM_HEADS,
                cp_atom_windows=distributed,
            )
            settings = dict(
                n_query=N_QUERY,
                n_key=N_KEY,
                atom_heads=ATOM_HEADS,
                token_heads=TOKEN_HEADS,
                n_token=N_TOKEN,
                sigma_data=SIGMA_DATA,
                cp_atom_windows=distributed,
            )
            step = denoise(batch, xl_noisy, t, si, si_trunk, zij, params, **settings)
            sampled = sample_diffusion(
                jax.random.key(3),
                schedule,
                (N_SAMPLE, N_ATOM, 3),
                lambda x, level: denoise(
                    batch, x, level, si, si_trunk, zij, params, **settings
                ),
                gamma_0=0.8,
                gamma_min=1.0,
                noise_scale=1.003,
                step_scale=1.5,
            )
            return {
                "encoder_token": ai,
                "encoder_atom": ql,
                "encoder_cond": cl,
                "encoder_pair": plm,
                "decoder": update,
                "step": step,
                "sampler": sampled,
            }

        return run

    def plan_probe(distributed: bool):
        """A one-stage closure whose body reports what it could see.

        Patched at ``atom_blocks.single_rep_to_blocks``, which
        ``cross_attention_pair_bias`` imports *inside* its body and therefore
        looks up per call -- so both arms reach the patched object. Patching
        the plan's own method instead would fire on the sharded arm only, and
        an arm that never fires reports "no plan" in exactly the same way as
        an arm that saw none.
        """

        def run(batch, xl_noisy, si_trunk, zij):
            seen: list[bool] = []
            original = atom_blocks.single_rep_to_blocks

            def spy(*args, **kwargs):
                seen.append(atom_block_plan() is not None)
                return original(*args, **kwargs)

            atom_blocks.single_rep_to_blocks = spy
            try:
                out = atom_attention_encoder(
                    batch,
                    params.atom_attn_enc,
                    n_query=N_QUERY,
                    n_key=N_KEY,
                    no_heads=ATOM_HEADS,
                    n_token=N_TOKEN,
                    rl=xl_noisy,
                    si_trunk=si_trunk,
                    zij_trunk=zij,
                    cp_atom_windows=distributed,
                )[0]
            finally:
                atom_blocks.single_rep_to_blocks = original
            tripwires.append((cp_layout(), distributed, len(seen), all(seen)))
            return out

        return run

    # The batch is a traced argument rather than a closed-over literal, and
    # that is load-bearing for the per-device shape assertions below. With the
    # features as constants XLA folds `atom_to_token_index` through the
    # aggregate's one-hot and the block tables, so a full-width tensor can
    # appear as a `constant` -- or vanish -- for reasons the real program, whose
    # features arrive as arguments, does not share. Measured: the literal-batch
    # module carries a folded `f32[S, N_atom, N_token]` constant that the traced
    # one does not.
    args = (case.batch, case.xl_noisy, case.t, case.si, case.si_trunk, case.zij)
    plan_args = (case.batch, case.xl_noisy, case.si_trunk, case.zij)

    serial_compiled = jax.jit(build(False))
    reference = jax.device_get(serial_compiled(*args))
    serial_hash = _fingerprint(
        serial_compiled.lower(*args).compiler_ir(dialect="hlo").as_hlo_text()
    )
    jax.jit(plan_probe(False))(*plan_args)
    jax.clear_caches()

    with context_parallel(devices, layout=layout):
        compiled = jax.jit(build(True))
        got = jax.device_get(compiled(*args))
        lowered = compiled.lower(*args)
        hlo = lowered.compiler_ir(dialect="hlo").as_hlo_text()
        spmd = lowered.compile().as_text()
        distributed_hash = _fingerprint(hlo)
        jax.jit(plan_probe(True))(*plan_args)

    print(
        f"devices={devices} layout={layout} "
        f"real_atoms={REAL_ATOMS}/{N_ATOM} tokens={N_TOKEN} blocks={N_BLOCKS}"
    )
    for name in (
        "encoder_token",
        "encoder_atom",
        "encoder_cond",
        "encoder_pair",
        "decoder",
        "step",
        "sampler",
    ):
        _report(
            name,
            np.asarray(reference[name], dtype=np.float64),
            np.asarray(got[name], dtype=np.float64),
        )

    low = hlo.lower()
    counts = {
        op: low.count(op)
        for op in (
            "collective-permute",
            "all-gather",
            "all-reduce",
            "reduce-scatter",
            "all-to-all",
        )
    }
    print("  collectives", counts)
    print("  tripwires", tripwires)
    print("  serial_hash", serial_hash[:16], "cp_hash", distributed_hash[:16])

    # Both arms must have reached the patched site the same number of times: a
    # plan reported absent by an arm that never ran the site would be a
    # tripwire that certifies nothing.
    serial_wire, distributed_wire = tripwires
    assert serial_wire[2] > 0 and serial_wire[2] == distributed_wire[2], tripwires
    assert serial_wire[:2] == (None, False) and not serial_wire[3], tripwires
    assert distributed_wire[:2] == (layout, True) and distributed_wire[3], tripwires
    assert serial_hash != distributed_hash, "the mesh variant reused a program"
    # The three ring rotations -- atom shards for the key blocks, token shards
    # for the token->atom broadcast, pair row tiles for the atom-pair gather --
    # are collective permutes; the atom->token mean is a reduce-scatter. A
    # full-key gather would be an `all-gather` over the atom axis, which is the
    # thing being removed.
    assert counts["collective-permute"] > 0, counts
    assert counts["reduce-scatter"] > 0, counts
    assert counts["all-gather"] == 0, counts
    # The column `psum` of the sparse token-pair lookup exists only when there
    # is a column axis to reduce over.
    if layout == "2d":
        assert counts["all-reduce"] > 0, counts

    full_width = {
        "atom activation": f"f32[{N_SAMPLE},{N_ATOM},{C_ATOM}]",
        "atom-pair block cache": (
            f"f32[{N_SAMPLE},{N_BLOCKS},{N_QUERY},{N_KEY},{C_ATOM_PAIR}]"
        ),
        "projected token pair": (f"f32[{N_SAMPLE},{N_TOKEN},{N_TOKEN},{C_ATOM_PAIR}]"),
        # Four dimensions with two full token axes: no atom-path shape can
        # spell that, which is what makes this unambiguous. The fixture keeps
        # N_TOKEN != N_BLOCKS so the three above are unambiguous too.
        "token pair bias": f"f32[{N_SAMPLE},{TOKEN_HEADS},{N_TOKEN},{N_TOKEN}]",
        # The atom->token aggregate's one-hot membership matrix, the one
        # coupled atom-by-token tensor in the graph. `N_TOKEN + 1` is the
        # overflow bin masked atoms are routed to, and the reason the bin is
        # dropped before the reduce-scatter rather than after.
        "atom-token one-hot": f"f32[{N_SAMPLE},{N_ATOM},{N_TOKEN + 1}]",
    }
    for label, shape in full_width.items():
        assert spmd.count(shape) == 0, (label, shape, spmd.count(shape))
    print("  per-device full-width intermediates", {k: 0 for k in full_width})
    print("OPENFOLD3_ATOM_CP_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
