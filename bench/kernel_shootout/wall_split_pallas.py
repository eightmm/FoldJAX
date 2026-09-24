# wall_split with a `pallas` arm, for the x43 Boltz-2 1k peak question.
#
# Copied from `foldjax-bench/wall-split/wall_split.py` with four changes:
#
# * arm `pallas`: the released arm with BOLTZ_JAX_TRIANGLE_MULTIPLICATION_BACKEND
#   =pallas and glu_backend=pallas (cuEq triangle attention kept);
# * its tripwire: cuEq multiplication forbidden, the Pallas kernels counted;
# * 1,003 tokens mapped to L1000_3og2 with its alignment depth (8,808 rows);
# * WALL_SPLIT_COMPILE_ONLY=1: records each family's `memory_analysis`, skips timing;
# * arm `trimul`: the released arm with only the Pallas triangle multiplication
#   (glu_backend stays tokamax), the multiplication's share of a pallas row;
# * 3,012 tokens mapped to L3000_6ztx with its alignment depth (8,192 rows), x50.
#
# Each family's argument, output and temp bytes are compared across the arms.
# The family that grows is the family that moved the peak x43 measured
# (11,165 vs 8,454 MiB at L1000_3og2).
# ruff: noqa
"""Where does Boltz-2's wall go, family by family, serial and on the 2x2 grid?

Why this exists
---------------
Boltz-2 is the slowest port under context parallelism and nothing says which
module pays for it.  Measured on four RTX PRO 6000 at 2,096 tokens (5DEI, the
released schedule: 5 samples, 200 diffusion steps, 10 recycles):

    serial, released fused kernels      231 s
    serial, every kernel on XLA         555 s
    2-D grid, XLA kernels               943 s
    2-D grid + tokamax ring tile        618 s

The one per-layer number in hand is one triangle-attention layer's ring at
2,096 tokens on the 2x2 grid: 281 ms on XLA against 147 ms with the tokamax
tile (`../fused-ring/experiment.py`, phase 1).  This harness generalises that
measurement to every repeated module family, so the four walls above can be
decomposed instead of guessed at.

What it does
------------
For one token count and one layout it times each repeated module family
**once** with frozen inputs at the real shapes and the released weights, then
multiplies by the number of times one released prediction calls that family.
The call counts are read from the checkpoint's own layer stacks and from the
schedule, never guessed (`--print-counts` shows them and exits).

    predicted_seconds = median_per_call_latency * calls_in_one_prediction

The sum of the predicted seconds is printed against the measured wall, which
is the only honest check on a decomposition assembled this way.  It will not
be exact and is not meant to be: see "What this is not".

Families
--------
Trunk Pairformer layer, one piece each (64 layers x 11 passes):
    trunk_tri_mul_out / trunk_tri_mul_in       triangle multiplication
    trunk_tri_att_start / trunk_tri_att_end    triangle attention (the ring, 2-D)
    trunk_pair_transition                      the pair GLU transition
    trunk_single_attention                     attention with pair bias
    trunk_single_transition                    the single GLU transition
MSA module layer (4 layers x 11 passes):
    msa_pwa                                    pair-weighted averaging
    msa_transition                             the MSA GLU transition
    msa_opm                                    outer product mean
    msa_noseq_layer                            the pair-only Pairformer layer
                                               each MSA layer also runs
Confidence (8 layers x 5 samples):
    conf_pairformer_layer
Diffusion, per denoising step at multiplicity 5 (200 steps):
    diff_token_transformer                     the whole 24-layer token stack
    diff_atom_encoder / diff_atom_decoder      the windowed atom transformers

Two families are timed as whole layers and reported as cross-checks rather than
summed: `trunk_pairformer_layer` and `msa_layer`.  Their number against the sum
of their pieces is what says whether the per-piece jit boundaries, or the
resharding between pieces, are inflating the decomposition.

What this is not
----------------
It is not a prediction.  Every family here is timed as its own jit program, so
each one pays an entry and an exit that a fused trunk pays once per layer, and
none of them sees the fusion a neighbour would have given it.  Stages that run
**once** per prediction are not timed at all and are listed in the report's
`not_timed` field: the input embedder, the diffusion conditioning's O(N^2)
projection stack, the distogram and B-factor heads, the confidence heads below
the Pairformer stack, and compile time.  So the sum is a lower bound with a
per-family error of unknown sign, and its job is to rank the families, not to
reproduce 943 s.

Under the 2-D layout every operand is placed on the grid with the partition
spec its consumer constrains it to, *before* the warm call.  Without that each
timed call would pay a replicated-to-2-D redistribution that a real run pays
once per pass, which would land entirely on whichever family was timed.

Arms
----
`--layout serial` takes `--kernels released,xla`:

    released   triangle attention cueq, triangle multiplication cueq (its
               backend is the environment variable
               BOLTZ_JAX_TRIANGLE_MULTIPLICATION_BACKEND, default cueq),
               GLU tokamax, diffusion attention tokamax
    xla        every one of those on its XLA path

`--layout 2d` takes `--ring-kernels xla,tokamax`, which is the tile kernel one
ring step evaluates its local attention with.  Everything else under a mesh is
already XLA and cannot be otherwise: `api.predict` resolves `triangle_backend`,
`glu_backend` and `diffusion_attention_backend` to their XLA paths when
`cp_devices > 1`, because none of those kernels can be partitioned.

`--grid gather` (2-D only) replaces the ring itself: every family is called
under `triangle_attention_grid_scope("gather")`, so both Boltz-2 2-D entries
run `gather_triangle_attention_2d_from_pair` -- per row block, the full-width
pair rows gathered along `cp_col` and one normalising attention on them.  Its
local body is the platform's decision, not a flag: cuEquivariance's triangle
attention on a GPU, the XLA reference body anywhere else.  The arm is named
`gather` and takes no ring tile kernel, because it runs no ring.

Every arm carries a dispatch tripwire.  A fused arm that silently ran the XLA
body, or an XLA arm that silently ran a fused one, raises instead of emitting a
labelled row -- the failure the fused-ring experiment's two-sided tile counter
was written for.

CPU smoke
---------
    JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4 \\
      FOLDJAX_HOME=<repo>/.foldjax PYTHONPATH=<repo>/src \\
      <repo>/.venv/bin/python wall_split.py --layout 2d --tokens 64 \\
        --msa-depth 64 --atoms 128 --repeats 1 --ring-kernels xla

    # the gather arm, on the XLA reference body
    ... wall_split.py --layout 2d --tokens 64 --msa-depth 64 --atoms 128 \\
        --repeats 1 --grid gather --stem-suffix gather

The fused serial arm cannot run on the CPU: cueq is an FFI call into a CUDA
library and tokamax is a Triton kernel, and both refuse rather than falling
back.  So the smoke validates the harness, the shapes, the placement and the
call counts -- not any kernel.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable

HERE = Path(__file__).resolve().parent
REPO = Path(os.environ.get("WALL_SPLIT_REPO", "/home/jaemin/non-project/optimizing/foldjax"))

#: The bench case each default describes, and the two sizes the parent runs.
#: `bench/spec.py` resolves a case's length from `FOLDJAX_BENCH_DATA`; both
#: numbers below are that length, which is also the token count Boltz-2's
#: tokenizer produces for these protein-only jobs.
CASES = {
    1003: "L1000_3og2",
    2096: "L2000_5dei",
    3012: "L3000_6ztx",
    6568: "L6500_6nyf8",
}

#: MSA alignment depth per case, as the featurizer hands it to the MSA module.
#: 5DEI's alignment is 8,192 rows; the 6,568-token case has 377.  Neither is
#: capped: `boltz predict` does not subsample (`models/predict.py`'s
#: `subsample_msa=False`) and `const.max_msa_seqs` is 16,384.  6ZTX's a3m has
#: 17,542 rows; the port's featurizer hands the MSA module 8,192 of them
#: (`featurize_yaml` on the L3000_6ztx job, msa (1, 8192, 3012)).
CASE_MSA_DEPTH = {1003: 8808, 2096: 8192, 3012: 8192, 6568: 377}

#: The released schedule every wall above was measured under
#: (`bench/spec.py:SCHEDULE`).
SCHEDULE = {"num_samples": 5, "num_steps": 200, "num_recycles": 10}

#: Both of Boltz-2's precision surfaces, at the values a released prediction
#: sets them to.  `api.MATMUL_PRECISION = "high"` is the ambient scope, which
#: governs every matmul carrying no explicit `precision` and from which
#: tokamax selects its dot-algorithm preset.  The op-level string stays at its
#: signature default `highest`, because `api.predict` never sets it and
#: `resolve_matmul_precision` deliberately refuses `"high"`.  Getting this
#: pair wrong is what turned an earlier tokamax measurement into a 75%
#: "regression" that was really a precision pin.
MATMUL_SCOPE = "high"
MATMUL_PRECISION = "highest"

#: The released trunk width, and the pair residual's storage width under it.
#: `compute_dtype="bfloat16"` with `pair_residual_dtype="auto"` resolves the
#: residual to bfloat16 too, which is what makes `native_amp=True` reach the
#: triangle ops.
COMPUTE_DTYPE = "bfloat16"
DIFFUSION_COMPUTE_DTYPE = "float32"

#: Atom-window geometry, from `atom_attention_encoder_forward`'s signature
#: defaults.  Under the 2-D grid an atom shard must still be query-window
#: aligned, so the atom count has to divide `ATOM_WINDOW_QUERIES * cp_rows`.
ATOM_WINDOW_QUERIES = 32
ATOM_WINDOW_KEYS = 128

DEVICES_2D = 4
SIDE_2D = 2

#: Substrings that identify a Pallas/Triton custom call in compiled HLO.  More
#: than one because the target name has moved between JAX releases, and a
#: census that silently matched nothing would certify a fallback as a fused
#: run.
TRITON_MARKERS = ("triton_kernel_call", "__gpu$xla.gpu.triton", "triton", "mosaic")

#: cuEquivariance's triangle-attention forward, as the FFI registers it
#: (`cuequivariance_ops_jax/_common.py`) and as the serial released arm's
#: census on the card already shows it.  The gather arm's GPU body must put it
#: in the compiled program; the CPU body must not.
CUEQ_ATTENTION_TARGET = "triangle_attention_cuda_fwd"

#: The 2-D triangle-attention algorithms `--grid` selects, as the port names
#: them (`_cp_attention.TRIANGLE_ATTENTION_GRIDS`).
GRIDS = ("ring", "gather")


# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #


def install_flags() -> str:
    """Bound XLA's collective rendezvous; never overwrite XLA_FLAGS.

    Appended rather than assigned: the CPU smoke passes
    `--xla_force_host_platform_device_count=4`, and an overwrite would leave
    the mesh one device and a 1x1 grid, which still runs and would report a
    serial number under a 2-D label.
    """

    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    flags = os.environ.get("XLA_FLAGS", "")
    from foldjax import oom

    if oom.gpu_is_possible():
        composed = oom.set_rendezvous_timeout()
        if composed:
            flags = composed
    return flags


# --------------------------------------------------------------------------- #
# Call counts
# --------------------------------------------------------------------------- #


def stack_lengths(params: Any) -> dict[str, int]:
    """Layer counts read from the checkpoint's own stacks.

    `load_params(prestack=True)` collapses each homogeneous layer list onto a
    leading layer axis behind a `StackedLayers` that still reports `len()`, so
    these are the counts the released graph iterates -- not a config constant
    that could have drifted from the weights.
    """

    def count(*path: str) -> int:
        node = params
        for key in path:
            node = node[key]
        return len(node)

    return {
        "trunk_pairformer_layers": count("trunk", "pairformer_module", "layers"),
        "trunk_msa_layers": count("trunk", "msa_module", "layers"),
        "confidence_pairformer_layers": count(
            "confidence", "pairformer_stack", "layers"
        ),
        "diffusion_token_layers": count(
            "conditioned_diffusion", "score_model", "token_transformer", "layers"
        ),
        "diffusion_atom_encoder_layers": count(
            "conditioned_diffusion",
            "score_model",
            "atom_attention_encoder",
            "atom_encoder",
            "diffusion_transformer",
            "layers",
        ),
        "diffusion_atom_decoder_layers": count(
            "conditioned_diffusion",
            "score_model",
            "atom_attention_decoder",
            "atom_decoder",
            "diffusion_transformer",
            "layers",
        ),
        "diffusion_token_bias_projections": count(
            "conditioned_diffusion", "diffusion_conditioning", "token_trans_proj_z"
        ),
    }


def call_counts(stacks: dict[str, int], schedule: dict[str, int]) -> dict[str, int]:
    """How many times one released prediction calls each family.

    Trunk passes are `recycling_steps + 1`: `boltz2_trunk_forward` scans its
    recycle body over that many steps and each step runs the MSA module and
    then the Pairformer module once.  The confidence stack runs once per
    diffusion sample, because `confidence_sequentially` is `num_samples > 1`
    and `predict.py` then maps the head over the samples one at a time.  The
    diffusion score model is evaluated once per sampling step, at multiplicity
    `num_samples`.
    """

    passes = int(schedule["num_recycles"]) + 1
    samples = int(schedule["num_samples"])
    steps = int(schedule["num_steps"])
    per_pf = passes * stacks["trunk_pairformer_layers"]
    per_msa = passes * stacks["trunk_msa_layers"]
    return {
        "trunk_passes": passes,
        "trunk_tri_mul_out": per_pf,
        "trunk_tri_mul_in": per_pf,
        "trunk_tri_att_start": per_pf,
        "trunk_tri_att_end": per_pf,
        "trunk_pair_transition": per_pf,
        "trunk_single_attention": per_pf,
        "trunk_single_transition": per_pf,
        "trunk_pairformer_layer": per_pf,
        "msa_pwa": per_msa,
        "msa_transition": per_msa,
        "msa_opm": per_msa,
        "msa_noseq_layer": per_msa,
        "msa_layer": per_msa,
        "conf_pairformer_layer": samples * stacks["confidence_pairformer_layers"],
        "diff_token_transformer": steps,
        "diff_atom_encoder": steps,
        "diff_atom_decoder": steps,
    }


#: Families timed as whole layers.  Reported, never summed: each one contains
#: other families in this table, so adding it would double-count.
CROSS_CHECK = ("trunk_pairformer_layer", "msa_layer")

#: Stages one released prediction runs but this harness does not time, named
#: so the unaccounted remainder has somewhere to point.
NOT_TIMED = (
    "input embedder, including its atom-window transformer (once per run)",
    "diffusion conditioning, including the pairwise conditioner and the 24 "
    "token-bias projections' shared normalisation (once per run)",
    "MSA input embedding and the recycle projections (once per pass)",
    "distogram and B-factor heads (once per run)",
    "confidence heads below the Pairformer stack (once per sample)",
    "the sampler's own per-step arithmetic: preconditioning, the Euler step, "
    "augmentation and any steering (once per step)",
    "compile time, which is reported per family but never summed into the wall",
)


# --------------------------------------------------------------------------- #
# Operands
# --------------------------------------------------------------------------- #


def atom_count(tokens: int, alignment: int) -> tuple[int, str]:
    """A heavy-atom count for the case at `tokens`, and where it came from.

    Counted from the bench job's own sequence against the port's CCD atom
    table (`data/const.ref_atoms`) when the bench data is reachable, which is
    exact for these protein-only jobs up to the featurizer's terminal-atom
    handling; the fallback is a documented ratio.  Either way it is an
    estimate of the shape, not a measurement of one, so it is reported with
    its source and `--atoms` overrides it.
    """

    case = CASES.get(tokens)
    if case is not None:
        try:
            from foldjax.models.boltz2.data import const

            data = Path(
                os.environ.get(
                    "FOLDJAX_BENCH_DATA", str(REPO.parent / "foldjax-bench")
                )
            )
            document = json.loads((data / "sequences.json").read_text())
            job = json.loads((data / "jobs" / f"{case}.json").read_text())
            sequence = document[case]["sequence"]
            chains = sum(
                len(entity.get("id", ["?"]))
                for entity in job["entities"]
                if entity.get("type") == "protein"
            )
            per_chain = 0
            for letter in sequence:
                residue = const.prot_letter_to_token.get(letter)
                if residue in const.ref_atoms:
                    per_chain += len(const.ref_atoms[residue])
            if per_chain and chains:
                total = per_chain * chains
                rounded = -(-total // alignment) * alignment
                return rounded, f"bench sequence {case}: {chains} x {per_chain}"
        except Exception:  # pragma: no cover - the mirror may carry no bench data
            pass
    total = tokens * 8
    rounded = -(-total // alignment) * alignment
    return rounded, "fallback ratio of 8 heavy atoms per token"


def widths(params: Any) -> dict[str, int]:
    """Every channel width and head count, read off the released weights."""

    pf = params["trunk"]["pairformer_module"]["layers"][0]
    msa = params["trunk"]["msa_module"]["layers"][0]
    token = params["conditioned_diffusion"]["score_model"]["token_transformer"][
        "layers"
    ][0]
    atom_enc = params["conditioned_diffusion"]["score_model"][
        "atom_attention_encoder"
    ]["atom_encoder"]["diffusion_transformer"]["layers"][0]
    cond = params["conditioned_diffusion"]["diffusion_conditioning"]
    return {
        "c_z": int(pf["tri_att_start"]["layer_norm"]["scale"].shape[0]),
        "c_s": int(pf["pre_norm_s"]["scale"].shape[0]),
        "c_m": int(msa["msa_transition"]["norm"]["scale"].shape[0]),
        "triangle_heads": int(pf["tri_att_start"]["linear"]["kernel"].shape[-1]),
        "single_heads": int(pf["attention"]["proj_z"]["kernel"].shape[-1]),
        "msa_pwa_heads": int(
            msa["pair_weighted_averaging"]["proj_z"]["kernel"].shape[-1]
        ),
        "c_token": int(token["adaln"]["s_norm"]["scale"].shape[0]),
        "token_bias_heads": int(
            cond["token_trans_proj_z"][0]["linear"]["kernel"].shape[-1]
        ),
        "c_atom": int(atom_enc["adaln"]["s_norm"]["scale"].shape[0]),
        "atom_bias_channels": int(
            cond["atom_enc_proj_z"][0]["linear"]["kernel"].shape[-1]
        ),
        "c_atompair": int(cond["atom_enc_proj_z"][0]["norm"]["scale"].shape[0]),
    }


class Operands:
    """Random arrays at the real shapes, dtypes and grid placement.

    Random values, real shapes: latency at these widths is a property of the
    shapes and the kernels, not of what the numbers are.  The weights are the
    released ones, so the widths, the head counts and the AMP arm are the
    deployed ones rather than invented.
    """

    def __init__(self, *, tokens: int, depth: int, atoms: int, sizes: dict[str, int],
                 samples: int, layout: str, mesh: Any, seed: int = 20260922) -> None:
        import jax.numpy as jnp
        import numpy as np

        self.tokens = tokens
        self.depth = depth
        self.atoms = atoms
        self.sizes = sizes
        self.samples = samples
        self.layout = layout
        self.mesh = mesh
        self.rng = np.random.default_rng(seed)
        self.jnp = jnp
        self.np = np
        self.pair_dtype = jnp.dtype(COMPUTE_DTYPE)
        self.score_dtype = jnp.dtype(DIFFUSION_COMPUTE_DTYPE)
        # A real run's token pad mask is not all ones at a padded shape, and
        # the masked keys are the path the ring's empty-tile handling sits on.
        # Every token keeps at least itself.
        keep = self.rng.random((1, tokens)) > 0.05
        keep[:, 0] = True
        self.keep = keep

    # -- construction ------------------------------------------------------ #

    def normal(self, shape: tuple[int, ...], dtype: Any, scale: float = 0.5) -> Any:
        return self.jnp.asarray(
            self.rng.normal(size=shape, scale=scale), dtype=dtype
        )

    def token_mask(self) -> Any:
        return self.jnp.asarray(self.keep.astype(self.np.float32))

    def pair_mask(self) -> Any:
        mask = self.keep.astype(self.np.float32)
        return self.jnp.asarray(mask[:, :, None] * mask[:, None, :])

    # -- placement --------------------------------------------------------- #

    def place(self, array: Any, spec: Any) -> Any:
        """Put `array` on the grid under `spec`, or replicate it.

        A timed family must not pay for a layout change its consumer would
        have been handed for free: in a real run the pair stream reaches a
        layer already row-and-column sharded, and a replicated entry operand
        would make every call redistribute it.
        """

        import jax
        from jax.sharding import NamedSharding, PartitionSpec

        if self.mesh is None:
            return jax.device_put(array)
        return jax.device_put(
            array, NamedSharding(self.mesh, spec if spec is not None else PartitionSpec())
        )

    # Every spec below answers `None` off the grid.  The port's own helpers
    # are not all total functions there -- `atom_axis_name` raises without a
    # mesh, by design, because an atom spec off the grid has no axis to name
    # -- so the guard belongs here rather than in a caller that would have to
    # remember it per family.
    def pair_spec(self, ndim: int = 4, row_axis: int = -3,
                  col_axis: int | None = None) -> Any:
        from foldjax.models._cp import pair_spec

        if self.mesh is None:
            return None
        return pair_spec(ndim, row_axis=row_axis, col_axis=col_axis)

    def single_spec(self, ndim: int = 3, token_axis: int = -2) -> Any:
        from foldjax.models._cp import single_spec

        if self.mesh is None:
            return None
        return single_spec(ndim, token_axis=token_axis)

    def msa_spec(self, ndim: int = 4, depth_axis: int = 1,
                 token_axis: int = 2) -> Any:
        from foldjax.models._cp import msa_spec

        if self.mesh is None:
            return None
        return msa_spec(ndim, depth_axis=depth_axis, token_axis=token_axis)

    def atom_spec(self, ndim: int = 3, atom_axis: int = 1) -> Any:
        from foldjax.models._cp_atom import atom_spec

        if self.mesh is None:
            return None
        return atom_spec(ndim, atom_axis=atom_axis)

    def window_spec(self, ndim: int = 5, window_axis: int = 1) -> Any:
        from foldjax.models._cp_atom import window_spec

        if self.mesh is None:
            return None
        return window_spec(ndim, window_axis=window_axis)


# --------------------------------------------------------------------------- #
# Dispatch census and tripwires
# --------------------------------------------------------------------------- #


def custom_calls(text: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for match in re.finditer(r'custom_call_target="([^"]+)"', text):
        counts[match.group(1)] = counts.get(match.group(1), 0) + 1
    return counts


def fused_call_count(counts: dict[str, int]) -> int:
    return sum(
        count
        for target, count in counts.items()
        if any(marker in target.lower() for marker in TRITON_MARKERS)
    )


class Tripwire:
    """Counts entries into every fused and every XLA dispatch point.

    A census of the compiled HLO cannot do this alone: on the CPU there is no
    fused target to find, and a census that matched nothing would certify a
    fallback as a fused run.  So the dispatch *functions* are counted too, on
    both sides, which is what makes the check two-sided -- a control arm that
    silently ran a fused body and a fused arm that silently ran the XLA one
    both show as a nonzero count on the wrong side rather than as a plausible
    number.

    Patching is confined to this harness's own process and undone in
    `__exit__`; nothing under `src/` is modified.
    """

    #: (module path, attribute, label).  A name is patched in *every* module
    #: that binds it, not only where it is defined: `tokamax_dot_product_
    #: attention` and `pair_bias_attention_2d` are imported by name at module
    #: import, so the consumer holds the original object and patching the
    #: defining module alone would count nothing.  The cueq entries are the
    #: opposite case -- imported inside their own function bodies -- and there
    #: the defining module is the only place to patch.  Several sites share a
    #: label on purpose; the counts add up under it.
    SITES = (
        # Fused kernels, i.e. what the serial released arm is.
        ("foldjax.models.boltz2.models.triangle.triangle_cueq",
         "cueq_triangle_multiplication_forward", "fused:tri_mul_cueq"),
        ("foldjax.models.boltz2.models.triangle.triangle_cueq",
         "cueq_attention_core", "fused:tri_att_cueq"),
        ("foldjax.models._glu", "_fused", "fused:glu_tokamax"),
        ("foldjax.models._pallas_pair", "triangle_multiplication", "fused:tri_mul_pallas"),
        ("foldjax.models._pallas_pair", "transition", "fused:transition_pallas"),
        ("foldjax.models.boltz2.models.primitives.attention_backend",
         "tokamax_dot_product_attention", "fused:attention_tokamax"),
        ("foldjax.models.boltz2.models.primitives.attention",
         "tokamax_dot_product_attention", "fused:attention_tokamax"),
        ("foldjax.models.boltz2.models.diffusion.diffusion_transformer",
         "tokamax_dot_product_attention", "fused:attention_tokamax"),
        # The 2-D ring's two tile bodies.  Exactly one may fire.
        ("foldjax.models._cp_attention", "_resolve_tile_attention",
         "ring:merge_tile"),
        ("foldjax.models._cp_attention", "_tile_terms", "ring:two_pass_tile"),
        # The structural 2-D adapters: one per module that has one.  These are
        # what says a family under `--layout 2d` ran the partitioned program
        # and not a replicated one that merely had a mesh in scope.
        # There are *two* 2-D ring entries, and which one a layer takes is
        # decided by which `triangle_attention_forward` it imported: the
        # Pairformer modules take `triangle_attention_cp`'s dispatcher, which
        # projects Q/K/V inside the row loop; the MSA module imports the
        # serial module's name, whose own CP branch reaches the ring through
        # `_attention_ring_2d`.  Both count here, under one label, because the
        # question this harness asks is whether the ring ran.
        ("foldjax.models.boltz2.models.triangle.triangle_attention_cp",
         "ring_triangle_attention_2d_from_pair", "cp:ring_2d"),
        ("foldjax.models.boltz2.models.triangle.triangle_attention",
         "_attention_ring_2d", "cp:ring_2d"),
        # The gather path's adapter, at both of the same two consumers, which
        # import it by name.  `_attention_ring_2d` above is the MSA entry's
        # *dispatcher*, not the ring: it is entered under `gather` too and
        # picks the gather branch inside, so `cp:ring_2d` fires once per MSA
        # triangle attention in the gather arm and says nothing there.  Which
        # algorithm ran is read from these and from the tile bodies.
        ("foldjax.models.boltz2.models.triangle.triangle_attention_cp",
         "gather_triangle_attention_2d_from_pair", "cp:gather_2d"),
        ("foldjax.models.boltz2.models.triangle.triangle_attention",
         "gather_triangle_attention_2d_from_pair", "cp:gather_2d"),
        # The gather path's two local bodies.  `_resolve_gather_attention`
        # reads these module globals at call time, so the defining module is
        # the place to patch.  The cuEq body reaches `_cueq.cueq_attention_
        # core` through an in-function import, not `triangle_cueq`'s binding,
        # so `fused:tri_att_cueq` stays silent under the gather arm.
        ("foldjax.models._cp_attention", "gather_attention_xla",
         "gather:body_xla"),
        ("foldjax.models._cp_attention", "gather_attention_cueq",
         "gather:body_cueq"),
        ("foldjax.models.boltz2.models.triangle.triangle", "_cannon_contract",
         "cp:cannon"),
        ("foldjax.models.boltz2.models.primitives.transition",
         "_cp_pair_transition", "cp:pair_transition_grid"),
        ("foldjax.models.boltz2.models.trunk_blocks.msa",
         "_pair_weighted_averaging_grid", "cp:msa_pwa_grid"),
        ("foldjax.models.boltz2.models.trunk_blocks.msa",
         "_outer_product_mean_grid", "cp:msa_opm_grid"),
        ("foldjax.models.boltz2.models.trunk_blocks.msa",
         "_msa_transition_grid", "cp:msa_transition_grid"),
        ("foldjax.models.boltz2.models.diffusion.diffusion_transformer",
         "pair_bias_attention_2d", "cp:token_attention_2d"),
        ("foldjax.models.boltz2.models.diffusion.atom",
         "_atom_transformer_forward_cp", "cp:atom_windows_2d"),
    )

    def __init__(self) -> None:
        self.counts: dict[str, int] = {}
        self._saved: list[tuple[Any, str, Any]] = []

    def __enter__(self) -> "Tripwire":
        import importlib

        for path, attribute, label in self.SITES:
            try:
                module = importlib.import_module(path)
            except Exception:
                continue
            original = getattr(module, attribute, None)
            if original is None:
                continue

            def wrapper(*args, _original=original, _label=label, **keywords):
                self.counts[_label] = self.counts.get(_label, 0) + 1
                return _original(*args, **keywords)

            setattr(module, attribute, wrapper)
            self._saved.append((module, attribute, original))
        return self

    def __exit__(self, *_exception: object) -> None:
        for module, attribute, original in self._saved:
            setattr(module, attribute, original)
        self._saved.clear()

    def reset(self) -> None:
        self.counts.clear()

    def snapshot(self) -> dict[str, int]:
        return dict(self.counts)


# --------------------------------------------------------------------------- #
# Timing
# --------------------------------------------------------------------------- #


def time_program(
    function: Callable[..., Any],
    args: tuple[Any, ...],
    *,
    repeats: int,
    dump_hlo: bool,
) -> dict[str, Any]:
    """Compile, warm, then time `repeats` synchronised calls and take the median.

    Every call is blocked on before the next is timed, so the number is a
    whole-program latency including the collectives a 2-D family runs -- not
    an async dispatch.
    """

    import jax
    import numpy as np

    before = {
        str(device.id): int(device.memory_stats().get("peak_bytes_in_use", -1))
        for device in jax.local_devices()
        if hasattr(device, "memory_stats") and device.memory_stats()
    }
    jitted = jax.jit(function)
    started = time.perf_counter()
    compiled = jitted.lower(*args).compile()
    row: dict[str, Any] = {"compile_seconds": time.perf_counter() - started}
    text = compiled.as_text() if dump_hlo else ""
    counts = custom_calls(text) if dump_hlo else {}
    row["custom_calls"] = counts
    row["fused_custom_calls"] = fused_call_count(counts)
    try:
        analysis = compiled.memory_analysis()
        row["memory_analysis"] = {
            name: int(getattr(analysis, name))
            for name in (
                "argument_size_in_bytes",
                "output_size_in_bytes",
                "temp_size_in_bytes",
                "alias_size_in_bytes",
            )
            if getattr(analysis, name, None) is not None
        }
    except Exception as error:  # pragma: no cover
        row["memory_analysis"] = f"unavailable: {error}"

    if os.environ.get("WALL_SPLIT_COMPILE_ONLY") == "1":
        row["latency_seconds"] = []
        row["latency_median"] = float("nan")
        row["latency_min"] = float("nan")
        return row
    jax.block_until_ready(compiled(*args))
    timings = []
    for _ in range(repeats):
        tick = time.perf_counter()
        jax.block_until_ready(compiled(*args))
        timings.append(time.perf_counter() - tick)
    row["latency_seconds"] = timings
    row["latency_median"] = float(np.median(timings))
    row["latency_min"] = float(np.min(timings))
    # A high-water mark for the *process*, not this family: every row after
    # the first large family reports that family's peak.  Recorded with its
    # before value and a flag saying whether this family actually raised it,
    # so it cannot be read as attribution.
    after = {
        str(device.id): int(device.memory_stats().get("peak_bytes_in_use", -1))
        for device in jax.local_devices()
        if hasattr(device, "memory_stats") and device.memory_stats()
    }
    row["process_peak_bytes_high_water"] = after
    row["process_peak_bytes_before"] = before
    row["raised_process_peak"] = any(
        after.get(key, -1) > before.get(key, -1) for key in after
    )
    return row


# --------------------------------------------------------------------------- #
# Family builders
# --------------------------------------------------------------------------- #


def build_families(
    params: Any,
    operands: Operands,
    *,
    chunks: dict[str, int | None],
    backends: dict[str, str],
    sizes: dict[str, int],
) -> dict[str, Callable[[], tuple[Callable[..., Any], tuple[Any, ...]]]]:
    """One builder per family: returns `(callable, args)` ready for `jax.jit`.

    Each builder hands the weights in as jit *arguments* rather than closing
    over them, so the program's argument footprint is visible in
    `memory_analysis` and the placement is explicit.

    The keyword arguments each family is called with are the ones its released
    caller passes, read from `pairformer_layer_forward`, `msa_layer_forward`,
    `pairformer_no_seq_layer_forward` and `diffusion_score_model_forward`.
    Anything this harness had to choose for itself -- the chunk policy, the
    backends -- arrives through `chunks` and `backends` and is reported.
    """

    import jax
    import jax.numpy as jnp

    from foldjax.models.boltz2.models.diffusion.atom import (
        atom_transformer_forward,
        diffusion_transformer_forward,
        diffusion_transformer_s_terms,
        get_indexing_matrix,
        single_to_keys,
    )
    from foldjax.models.boltz2.models.primitives.attention import (
        attention_pair_bias_forward,
    )
    from foldjax.models.boltz2.models.primitives.transition import transition_forward
    from foldjax.models.boltz2.models.triangle.triangle import (
        triangle_multiplication_forward,
    )
    from foldjax.models.boltz2.models.triangle.triangle_attention_cp import (
        triangle_attention_forward,
    )
    # `msa.py` carries its *own* `pairformer_no_seq_layer_forward`, and that is
    # the one `msa_layer_forward` calls.  It is not the same function as
    # `pairformer_noseq.py`'s: this one takes `matmul_precision`,
    # `glu_backend`, `pair_residual_dtype` and answers `native_amp`
    # explicitly, and it imports the *serial* module's
    # `triangle_attention_forward` rather than the 2-D dispatcher -- so it
    # reaches the ring by the other of the two entries.  Timing the
    # `pairformer_noseq.py` copy instead would measure a function the MSA
    # stack never runs.
    from foldjax.models.boltz2.models.trunk_blocks.msa import (
        _msa_transition_grid,
        _NATIVE_CHUNK_THRESHOLD,
        _on_msa_grid,
        msa_layer_forward,
        outer_product_mean_forward,
        pair_weighted_averaging_forward,
        pairformer_no_seq_layer_forward,
    )
    from foldjax.models.boltz2.models.trunk_blocks.pairformer import (
        pairformer_layer_forward,
    )

    tokens = operands.tokens
    depth = operands.depth
    atoms = operands.atoms
    samples = operands.samples
    pair_dtype = operands.pair_dtype
    score_dtype = operands.score_dtype
    chunk_size = int(chunks["chunk_size"])
    tri_chunk = chunks["triangle_attention_chunk"]
    tri_q_chunk = chunks["triangle_attention_q_chunk"]
    token_chunk = chunks["token_attention_chunk"]
    transition_hidden_chunk = chunks["transition_hidden_chunk"]

    pf_layer = params["trunk"]["pairformer_module"]["layers"][0]
    msa_layer = params["trunk"]["msa_module"]["layers"][0]
    conf_layer = params["confidence"]["pairformer_stack"]["layers"][0]
    score = params["conditioned_diffusion"]["score_model"]
    cond = params["conditioned_diffusion"]["diffusion_conditioning"]

    def replicate(tree: Any) -> Any:
        return jax.tree.map(lambda leaf: operands.place(leaf, None), tree)

    # -- shared placed operands ------------------------------------------- #

    # The token pad mask reaches a real run's graph replicated: it is not a
    # pair feature, so `_cp.feature_spec` returns no spec for it.
    mask = operands.place(operands.token_mask(), None)
    # The pair-shaped mask is loop-invariant -- the trunk builds it once per
    # pass and every layer reads it -- and each grid `shard_map` names it
    # `pair_spec(3, row_axis=1, col_axis=2)`.  So it is an already-placed
    # entry operand here, not an outer product recomputed inside every timed
    # call, which would charge one layer for work the stack does once.
    pair_mask = operands.place(
        operands.pair_mask(), operands.pair_spec(3, row_axis=1, col_axis=2)
    )
    pair = operands.place(
        operands.normal((1, tokens, tokens, sizes["c_z"]), pair_dtype),
        operands.pair_spec(4),
    )
    # `s` is float32 in every released caller: upstream's single branch runs
    # inside `torch.autocast(enabled=False)`, and the port keeps the scan
    # carry there.
    single = operands.place(
        operands.normal((1, tokens, sizes["c_s"]), jnp.float32),
        operands.single_spec(3),
    )
    msa = operands.place(
        operands.normal((1, depth, tokens, sizes["c_m"]), jnp.float32),
        operands.msa_spec(4),
    )
    msa_mask = operands.place(
        jnp.asarray(
            (operands.rng.random((1, depth, tokens)) > 0.05).astype("float32")
        ),
        operands.msa_spec(3),
    )

    # `native_amp` answers "is this the CUDA-autocast configuration".  The
    # released trunk stores its pair residual narrow, so every caller below
    # answers it explicitly rather than letting the op infer it from the
    # activation width and drop to the FP32-model program.
    native_amp = True
    msa_transition_dtype = msa_layer["msa_transition"]["fc1"]["kernel"].dtype
    msa_transition_amp = msa_transition_dtype in (jnp.bfloat16, jnp.float16)

    builders: dict[str, Callable[[], tuple[Callable[..., Any], tuple[Any, ...]]]] = {}

    # -- trunk Pairformer pieces ------------------------------------------ #

    def tri_mul(direction: str, key: str):
        def build():
            def run(weights, z, z_mask):
                return triangle_multiplication_forward(
                    weights,
                    z,
                    z_mask,
                    direction,
                    chunk_size=chunk_size,
                    glu_backend=backends["glu_backend"],
                    native_amp=native_amp,
                )

            return run, (replicate(pf_layer[key]), pair, pair_mask)

        return build

    builders["trunk_tri_mul_out"] = tri_mul("outgoing", "tri_mul_out")
    builders["trunk_tri_mul_in"] = tri_mul("incoming", "tri_mul_in")

    def tri_att(starting: bool, key: str):
        def build():
            def run(weights, z, z_mask):
                return triangle_attention_forward(
                    weights,
                    z,
                    z_mask,
                    starting=starting,
                    chunk_size=tri_chunk if tri_chunk is not None else chunk_size,
                    q_chunk_size=tri_q_chunk,
                    matmul_precision=MATMUL_PRECISION,
                    triangle_backend=backends["triangle_backend"],
                    native_amp=native_amp,
                )

            return run, (replicate(pf_layer[key]), pair, pair_mask)

        return build

    builders["trunk_tri_att_start"] = tri_att(True, "tri_att_start")
    builders["trunk_tri_att_end"] = tri_att(False, "tri_att_end")

    def build_pair_transition():
        def run(weights, z):
            return transition_forward(
                weights,
                z,
                chunk_size=transition_hidden_chunk,
                row_chunk_size=chunk_size,
                glu_backend=backends["glu_backend"],
                native_amp_norm=True,
                cp_pair=True,
            )

        return run, (replicate(pf_layer["transition_z"]), pair)

    builders["trunk_pair_transition"] = build_pair_transition

    def build_single_attention():
        from foldjax.models.boltz2.models.primitives._common import (
            layer_norm as _layer_norm,
        )

        def run(norm_weights, attention_weights, s, z, token_mask):
            s_normed = _layer_norm(
                s, norm_weights["scale"], norm_weights["bias"], 1e-5
            )
            return attention_pair_bias_forward(
                attention_weights,
                s=s_normed,
                # Native's single branch disables autocast, including its
                # pair-bias projection.
                z=z.astype(jnp.float32),
                mask=token_mask.astype(jnp.float32),
                k_in=s_normed,
                chunk_size=chunk_size,
                attention_backend=backends["attention_backend"],
            )

        return run, (
            replicate(pf_layer["pre_norm_s"]),
            replicate(pf_layer["attention"]),
            single,
            pair,
            mask,
        )

    builders["trunk_single_attention"] = build_single_attention

    def build_single_transition():
        def run(weights, s):
            return transition_forward(
                weights, s, glu_backend=backends["glu_backend"]
            )

        return run, (replicate(pf_layer["transition_s"]), single)

    builders["trunk_single_transition"] = build_single_transition

    def build_pairformer_layer():
        def run(weights, s, z, token_mask, z_mask):
            return pairformer_layer_forward(
                weights,
                s,
                z,
                token_mask,
                z_mask,
                chunk_size=chunk_size,
                triangle_attention_chunk=tri_chunk,
                triangle_attention_q_chunk=tri_q_chunk,
                transition_hidden_chunk=transition_hidden_chunk,
                matmul_precision=MATMUL_PRECISION,
                attention_backend=backends["attention_backend"],
                triangle_backend=backends["triangle_backend"],
                glu_backend=backends["glu_backend"],
                pair_residual_dtype=pair_dtype,
            )

        return run, (replicate(pf_layer), single, pair, mask, pair_mask)

    builders["trunk_pairformer_layer"] = build_pairformer_layer

    # -- confidence ------------------------------------------------------- #

    # The confidence head assembles its own pair stream, and it comes out
    # float32 even under the bfloat16 trunk: the relative-position encoding,
    # the float32 `token_bonds` linear and the two embedding lookups
    # (`token_bonds_type`, `dist_bin_pairwise_embed`) are not Linear kernels,
    # so `_cast_trunk_params` leaves them wide and the sum promotes.  With a
    # float32 pair against bfloat16 kernels `resolve_native_amp` answers True
    # by inference, which is why the stack takes no pin -- and why timing it
    # on the trunk's narrow pair would measure a different program.
    conf_pair = operands.place(
        operands.normal((1, tokens, tokens, sizes["c_z"]), jnp.float32),
        operands.pair_spec(4),
    )

    def build_conf_layer():
        # The confidence stack takes no pair-residual pin -- `predict.py`
        # passes none -- so its carry keeps whatever width it is handed, and
        # its GLU backend is the sampler's.  It runs at multiplicity 1 per
        # sample under the sequential map.
        def run(weights, s, z, token_mask, z_mask):
            return pairformer_layer_forward(
                weights,
                s,
                z,
                token_mask,
                z_mask,
                chunk_size=chunk_size,
                triangle_attention_chunk=tri_chunk,
                triangle_attention_q_chunk=tri_q_chunk,
                transition_hidden_chunk=transition_hidden_chunk,
                matmul_precision=MATMUL_PRECISION,
                attention_backend=backends["attention_backend"],
                triangle_backend=backends["triangle_backend"],
                glu_backend=backends["glu_backend"],
            )

        return run, (replicate(conf_layer), single, conf_pair, mask, pair_mask)

    builders["conf_pairformer_layer"] = build_conf_layer

    # -- MSA module pieces ------------------------------------------------ #

    def build_msa_pwa():
        def run(weights, m, z, z_mask):
            return pair_weighted_averaging_forward(weights, m, z, z_mask)

        return run, (
            replicate(msa_layer["pair_weighted_averaging"]),
            msa,
            pair,
            pair_mask,
        )

    builders["msa_pwa"] = build_msa_pwa

    def build_msa_transition():
        # `msa_layer_forward` selects `_msa_transition_grid` on the grid and
        # `transition_forward` off it, and passes the row block only above the
        # publisher's 384-token threshold.  Both decisions are reproduced here
        # rather than approximated, because the grid one is a `shard_map` and
        # the other is not.
        callable_used = _msa_transition_grid if _on_msa_grid() else transition_forward
        row_chunk = 32 if tokens > _NATIVE_CHUNK_THRESHOLD else None
        keywords: dict[str, Any] = {
            "glu_backend": backends["glu_backend"],
            "cp_msa": True,
            "chunk_size": row_chunk,
        }
        if msa_transition_amp:
            keywords["compute_dtype"] = msa_transition_dtype
            keywords["native_amp_norm"] = msa_transition_dtype == jnp.bfloat16
        else:
            keywords.pop("chunk_size")

        def run(weights, m):
            return callable_used(weights, m, eps=1e-5, **keywords)

        return run, (replicate(msa_layer["msa_transition"]), msa)

    builders["msa_transition"] = build_msa_transition

    def build_msa_opm():
        def run(weights, m, alignment_mask):
            return outer_product_mean_forward(
                weights,
                m,
                alignment_mask,
                1e-5,
                chunk_size=chunk_size,
                preserve_native_amp_shape=True,
            )

        return run, (replicate(msa_layer["outer_product_mean"]), msa, msa_mask)

    builders["msa_opm"] = build_msa_opm

    def build_msa_noseq():
        def run(weights, z, z_mask):
            return pairformer_no_seq_layer_forward(
                weights,
                z,
                z_mask,
                chunk_size=chunk_size,
                triangle_attention_chunk=tri_chunk,
                triangle_attention_q_chunk=tri_q_chunk,
                transition_hidden_chunk=transition_hidden_chunk,
                matmul_precision=MATMUL_PRECISION,
                triangle_backend=backends["triangle_backend"],
                glu_backend=backends["glu_backend"],
                pair_residual_dtype=pair_dtype,
            )

        return run, (replicate(msa_layer["pairformer_layer"]), pair, pair_mask)

    builders["msa_noseq_layer"] = build_msa_noseq

    def build_msa_layer():
        def run(weights, z, m, z_mask, alignment_mask):
            return msa_layer_forward(
                weights,
                z,
                m,
                z_mask,
                alignment_mask,
                chunk_size=chunk_size,
                triangle_attention_chunk=tri_chunk,
                triangle_attention_q_chunk=tri_q_chunk,
                transition_hidden_chunk=transition_hidden_chunk,
                matmul_precision=MATMUL_PRECISION,
                triangle_backend=backends["triangle_backend"],
                glu_backend=backends["glu_backend"],
                pair_residual_dtype=pair_dtype,
            )

        return run, (replicate(msa_layer), pair, msa, pair_mask, msa_mask)

    builders["msa_layer"] = build_msa_layer

    # -- diffusion -------------------------------------------------------- #

    def build_token_transformer():
        # The released sampler runs the token stack with a *lazy* pair bias:
        # `boltz2_sample_forward` passes `lazy_token_trans_bias=True`, so the
        # conditioning hands over the 24 projections plus one shared normed
        # pair input and each layer projects its own [.., heads] block inside
        # the scan.  Handing a dense [N, N, 24 * heads] bias instead would be
        # a different program and, at 6,568 tokens, tens of gigabytes.
        from foldjax.models.boltz2.models.diffusion.diffusion_conditioning import (
            _projection_input_norm,
        )

        # What the released conditioning emits for this call, traced through
        # `boltz2_sample_forward`: `low_precision` is true for the bfloat16
        # trunk, so `diffusion_conditioning_forward` gets
        # `compute_dtype=bfloat16` and publishes
        # `token_trans_bias_precision` at that width -- which reaches the
        # stack as `bias_compute_dtype`, so the 24 per-layer
        # [N, N, c_z] -> [N, N, heads] projections are bfloat16 matmuls, not
        # float32 ones.  `bias_dtype` stays None because the score model is
        # float32, so `bias_out_dtype` is None and each projection returns at
        # its input width.  The shared normalisation is the float32 array the
        # conditioning builds from a trunk pair it has already widened, under
        # the native-AMP LayerNorm the bfloat16 arm selects.
        bias_compute_dtype = jnp.dtype(COMPUTE_DTYPE)
        activation = operands.place(
            operands.normal((samples, tokens, sizes["c_token"]), score_dtype),
            operands.single_spec(3, 1),
        )
        conditioning = operands.place(
            operands.normal((samples, tokens, sizes["c_token"]), score_dtype),
            operands.single_spec(3, 1),
        )
        normed = operands.place(
            _projection_input_norm(
                operands.normal((1, tokens, tokens, sizes["c_z"]), jnp.float32),
                1e-5,
                native_amp=bias_compute_dtype == jnp.bfloat16,
            ),
            operands.pair_spec(4),
        )
        step_mask = operands.place(
            jnp.repeat(operands.token_mask(), samples, axis=0),
            operands.single_spec(2, -1),
        )

        def run(weights, bias_weights, a, s, bias_normed_input, token_mask):
            return diffusion_transformer_forward(
                weights,
                a=a,
                s=s,
                bias=None,
                mask=token_mask.astype(jnp.float32),
                multiplicity=samples,
                attention_backend=backends["diffusion_attention_backend"],
                chunk_size=token_chunk,
                bias_params=bias_weights,
                bias_normed_input=bias_normed_input,
                bias_compute_dtype=bias_compute_dtype,
            )

        return run, (
            replicate(score["token_transformer"]),
            replicate(cond["token_trans_proj_z"]),
            activation,
            conditioning,
            normed,
            step_mask,
        )

    builders["diff_token_transformer"] = build_token_transformer

    def atom_transformer(stack_key: str, inner_key: str):
        def build():
            windows = atoms // ATOM_WINDOW_QUERIES
            # `atom_attention_encoder_forward` hands `atom_transformer_forward`
            # the `atom_encoder` subtree, which looks up its own
            # `diffusion_transformer` inside; so the weights operand is that
            # subtree, not the transformer under it.
            stack = score[stack_key][inner_key]
            layers = len(stack["diffusion_transformer"]["layers"])
            q = operands.place(
                operands.normal((samples, atoms, sizes["c_atom"]), score_dtype),
                operands.atom_spec(3, 1),
            )
            c = operands.place(
                operands.normal((samples, atoms, sizes["c_atom"]), score_dtype),
                operands.atom_spec(3, 1),
            )
            # The bias is sample-invariant and stays compact over samples: the
            # released atom transformer broadcasts it per layer rather than
            # carrying `multiplicity` copies through the scan.  Its channel
            # extent is one projection's head count per layer.
            bias = operands.place(
                operands.normal(
                    (
                        1,
                        windows,
                        ATOM_WINDOW_QUERIES,
                        ATOM_WINDOW_KEYS,
                        layers * sizes["atom_bias_channels"],
                    ),
                    score_dtype,
                ),
                operands.window_spec(5, 1),
            )
            atom_mask = operands.place(
                jnp.asarray(
                    (operands.rng.random((samples, atoms)) > 0.02).astype("float32")
                ),
                operands.atom_spec(2, 1),
            )
            indexing = get_indexing_matrix(
                windows, ATOM_WINDOW_QUERIES, ATOM_WINDOW_KEYS
            )
            atom_cp = operands.mesh is not None
            # The two arms run two different programs here, and the
            # difference is `boltz2_sample_forward`'s own condition: it
            # hoists these AdaLN scale/bias and output-gate terms out of the
            # denoising loop -- once per prediction, off the *compact*
            # [windows, 32, c] conditioning -- and passes them in as
            # `precomputed_s_terms`, but only when atom context parallelism
            # is not active under a mesh.  So the serial step pays zero AdaLN
            # projections and the 2-D step recomputes them on its local
            # shard every step.  Timing the serial arm without the hoist
            # would inflate it and understate the 2-D penalty; passing them
            # under the mesh is refused by `atom_transformer_forward` anyway.
            s_terms = None
            if not atom_cp:
                compact = operands.normal(
                    (windows, ATOM_WINDOW_QUERIES, sizes["c_atom"]), score_dtype
                )
                s_terms = jax.tree.map(
                    lambda leaf: operands.place(leaf, None),
                    diffusion_transformer_s_terms(
                        stack["diffusion_transformer"], compact, eps=1e-5
                    ),
                )

            def to_keys(x):
                return single_to_keys(
                    x, indexing, w=ATOM_WINDOW_QUERIES, h_keys=ATOM_WINDOW_KEYS
                )

            def run(weights, q_in, c_in, bias_in, mask_in, terms):
                return atom_transformer_forward(
                    weights,
                    q_in,
                    c_in,
                    bias_in,
                    to_keys,
                    mask_in,
                    attn_window_queries=ATOM_WINDOW_QUERIES,
                    attn_window_keys=ATOM_WINDOW_KEYS,
                    multiplicity=samples,
                    attention_backend=backends["diffusion_attention_backend"],
                    atom_context_parallel=atom_cp,
                    precomputed_s_terms=terms,
                )

            return run, (replicate(stack), q, c, bias, atom_mask, s_terms)

        return build

    builders["diff_atom_encoder"] = atom_transformer(
        "atom_attention_encoder", "atom_encoder"
    )
    builders["diff_atom_decoder"] = atom_transformer(
        "atom_attention_decoder", "atom_decoder"
    )
    return builders


# --------------------------------------------------------------------------- #
# Arms
# --------------------------------------------------------------------------- #

#: The four backend knobs, at the values `api.predict` resolves for each arm.
#:
#: `serial/released`: `triangle_backend="cueq"`, `glu_backend="tokamax"` and
#: `diffusion_attention_backend="tokamax"` are the signature defaults, and
#: triangle *multiplication* takes its backend from the environment variable
#: `BOLTZ_JAX_TRIANGLE_MULTIPLICATION_BACKEND`, whose default is also `cueq`
#: -- so the fused tri-mul arm is selected by leaving that unset and the XLA
#: arm by spelling it.  `attention_backend` is `xla` in every arm, which is
#: its released value.
#:
#: `2d`: `api.predict` resolves all three fused defaults to their XLA paths
#: when `cp_devices > 1`, because a kernel that consumes the whole token axis
#: cannot be partitioned.  What remains selectable is the *ring tile* kernel,
#: which is a different knob travelling in a scope.
ARMS = {
    "released": {
        "triangle_backend": "cueq",
        "glu_backend": "tokamax",
        "attention_backend": "xla",
        "diffusion_attention_backend": "tokamax",
        "tri_mul_env": "cueq",
    },
    "xla": {
        "triangle_backend": "xla",
        "glu_backend": "xla",
        "attention_backend": "xla",
        "diffusion_attention_backend": "xla",
        "tri_mul_env": "xla",
    },
    "pallas": {
        "triangle_backend": "cueq",
        "glu_backend": "pallas",
        "attention_backend": "xla",
        "diffusion_attention_backend": "tokamax",
        "tri_mul_env": "pallas",
    },
    "trimul": {
        "triangle_backend": "cueq",
        "glu_backend": "tokamax",
        "attention_backend": "xla",
        "diffusion_attention_backend": "tokamax",
        "tri_mul_env": "pallas",
    },
}

#: Which tripwire labels must have fired, and which must not, for each arm.
#: A family that reaches none of the six dispatch points -- a transition on
#: the XLA path, say -- constrains nothing and is checked only on the
#: "must not" side.
EXPECTED = {
    "released": {
        "forbidden": (),
        "fused_labels": (
            "fused:tri_mul_cueq",
            "fused:tri_att_cueq",
            "fused:glu_tokamax",
            "fused:attention_tokamax",
        ),
    },
    "xla": {
        "forbidden": (
            "fused:tri_mul_cueq",
            "fused:tri_att_cueq",
            "fused:glu_tokamax",
            "fused:attention_tokamax",
        ),
        "fused_labels": (),
    },
    "pallas": {
        "forbidden": ("fused:tri_mul_cueq",),
        "fused_labels": (
            "fused:tri_mul_pallas",
            "fused:transition_pallas",
            "fused:tri_att_cueq",
            "fused:glu_tokamax",
            "fused:attention_tokamax",
        ),
    },
    "trimul": {
        "forbidden": ("fused:tri_mul_cueq", "fused:transition_pallas"),
        "fused_labels": (
            "fused:tri_mul_pallas",
            "fused:tri_att_cueq",
            "fused:glu_tokamax",
            "fused:attention_tokamax",
        ),
    },
}


#: The 2-D adapter each family must reach, by family.  An empty tuple is a
#: claim, not an omission: the trunk's single attention and its single
#: transition have no `shard_map` of their own under this layout -- the
#: partitioner handles them from the sharding constraints on `s` and `z` --
#: so there is nothing there to fire and a row that fired one would be wrong.
CP_EXPECTED = {
    "trunk_tri_mul_out": ("cp:cannon",),
    "trunk_tri_mul_in": ("cp:cannon",),
    "trunk_tri_att_start": ("cp:ring_2d",),
    "trunk_tri_att_end": ("cp:ring_2d",),
    "trunk_pair_transition": ("cp:pair_transition_grid",),
    "trunk_single_attention": (),
    "trunk_single_transition": (),
    "trunk_pairformer_layer": (
        "cp:cannon",
        "cp:ring_2d",
        "cp:pair_transition_grid",
    ),
    "conf_pairformer_layer": (
        "cp:cannon",
        "cp:ring_2d",
        "cp:pair_transition_grid",
    ),
    "msa_pwa": ("cp:msa_pwa_grid",),
    "msa_transition": ("cp:msa_transition_grid",),
    "msa_opm": ("cp:msa_opm_grid",),
    "msa_noseq_layer": ("cp:cannon", "cp:ring_2d", "cp:pair_transition_grid"),
    "msa_layer": (
        "cp:msa_pwa_grid",
        "cp:msa_transition_grid",
        "cp:msa_opm_grid",
        "cp:cannon",
        "cp:ring_2d",
    ),
    "diff_token_transformer": ("cp:token_attention_2d",),
    "diff_atom_encoder": ("cp:atom_windows_2d",),
    "diff_atom_decoder": ("cp:atom_windows_2d",),
}

#: Which families run a triangle attention, and therefore a ring whose tile
#: kernel the `--ring-kernels` arm selects.  Every other family is timed
#: identically in both ring arms, so its two rows are one measurement repeated
#: and the report says so rather than implying a comparison.
RING_FAMILIES = tuple(
    name for name, sites in CP_EXPECTED.items() if "cp:ring_2d" in sites
)


def check_tripwire(
    family: str,
    arm: str,
    layout: str,
    ring_kernel: str | None,
    fired: dict[str, int],
    *,
    grid: str = "ring",
    platform: str = "cpu",
    census: dict[str, int] | None = None,
) -> str:
    """Refuse to label a row whose program is not the arm it claims.

    Returns a short verdict for the report.  Raises when the evidence says the
    timed program was the *other* arm: a wrong label is worse than a missing
    row, because it is the one failure that survives into a conclusion.

    `census` is the compiled program's custom-call count, or None when the
    HLO census was skipped; only the gather arm on a GPU requires it.
    """

    if layout == "2d":
        for label in EXPECTED["xla"]["forbidden"]:
            if fired.get(label, 0):
                raise RuntimeError(
                    f"{family}/2d: {label} fired {fired[label]} times, but no "
                    "fused kernel over the whole token axis is partitionable "
                    "under a mesh -- this row did not run the 2-D program"
                )
        # First, that the family ran its partitioned program at all.  Without
        # this a family that silently ran replicated under a mesh would report
        # a plausible latency, and a replicated program is exactly what the
        # 2-D wall question is not about.  Under `gather` the triangle
        # attention's adapter is the gather's, wherever the ring's was.
        wanted = CP_EXPECTED.get(family)
        if wanted is None:
            raise RuntimeError(f"no 2-D adapter expectation recorded for {family!r}")
        if grid == "gather":
            wanted = tuple(
                "cp:gather_2d" if label == "cp:ring_2d" else label
                for label in wanted
            )
        missing = [label for label in wanted if not fired.get(label, 0)]
        if "cp:gather_2d" in missing:
            raise RuntimeError(
                f"{family}/2d gather: the gather adapter never fired ({fired}), "
                "so this row did not time the gather path"
            )
        if missing:
            raise RuntimeError(
                f"{family}/2d: {missing} never fired ({fired}), so this row "
                "timed a program that is not partitioned across the grid"
            )
        # Every `cp:` label that fired, not only the ones this family had to
        # reach: a layer that contains another family's adapter should say so
        # in its row rather than look like it skipped it.
        structural = ", ".join(
            f"{label} {count}x"
            for label, count in sorted(fired.items())
            if label.startswith("cp:") and count
        ) or "no 2-D adapter at this site"
        # Then the ring's tile body, which is two-sided: the merge body
        # resolves a callable once per row block, the two-pass body evaluates
        # its terms once per key tile per block, so exactly one may be nonzero.
        merge = fired.get("ring:merge_tile", 0)
        two_pass = fired.get("ring:two_pass_tile", 0)
        gathered = fired.get("cp:gather_2d", 0)
        body_xla = fired.get("gather:body_xla", 0)
        body_cueq = fired.get("gather:body_cueq", 0)
        if family not in RING_FAMILIES:
            if merge or two_pass:
                raise RuntimeError(
                    f"{family}/2d: a ring tile body fired ({fired}) in a "
                    "family that has no triangle attention"
                )
            if gathered or body_xla or body_cueq:
                raise RuntimeError(
                    f"{family}/2d: the gather path fired ({fired}) in a "
                    "family that has no triangle attention"
                )
            return f"{structural}; no ring"
        if grid == "gather":
            return _check_gather(
                family, fired, merge, two_pass, body_xla, body_cueq,
                structural=structural, platform=platform, census=census,
            )
        if gathered or body_xla or body_cueq:
            raise RuntimeError(
                f"{family}/2d ring: the gather path fired ({fired}), so this "
                "row did not measure the ring"
            )
        if ring_kernel == "tokamax":
            if merge < 1:
                raise RuntimeError(
                    f"{family}/2d ring=tokamax: the merge tile body never "
                    f"fired ({fired}), so this row measured the XLA ring"
                )
            if two_pass:
                raise RuntimeError(
                    f"{family}/2d ring=tokamax: the two-pass tile body fired "
                    f"{two_pass} times ({fired}), so this row ran both bodies"
                )
            return f"{structural}; merge tile {merge}x, two-pass 0x"
        if two_pass < 1:
            raise RuntimeError(
                f"{family}/2d ring=xla: the two-pass tile body never fired "
                f"({fired}), so this row did not measure the released ring"
            )
        if merge:
            raise RuntimeError(
                f"{family}/2d ring=xla: the merge tile body fired {merge} "
                f"times ({fired}), so the control arm is not a control"
            )
        return f"{structural}; two-pass tile {two_pass}x, merge 0x"

    stray = {
        label: count
        for label, count in fired.items()
        if label.startswith(("cp:", "ring:", "gather:")) and count
    }
    if stray:
        raise RuntimeError(
            f"{family}/serial: {stray} fired without a mesh, so this row did "
            "not time the serial program"
        )
    spec = EXPECTED[arm]
    for label in spec["forbidden"]:
        if fired.get(label, 0):
            raise RuntimeError(
                f"{family}/serial arm {arm}: {label} fired {fired[label]} "
                f"times ({fired}), so the control arm ran a fused kernel"
            )
    hit = {
        label: fired[label]
        for label in spec["fused_labels"]
        if fired.get(label, 0)
    }
    if arm in ("released", "pallas", "trimul") and not hit:
        return "no fused dispatch point reached (family has no fused path)"
    if arm == "xla":
        return "no fused dispatch point reached, as required"
    return ", ".join(f"{label} {count}x" for label, count in sorted(hit.items()))


def _check_gather(
    family: str,
    fired: dict[str, int],
    merge: int,
    two_pass: int,
    body_xla: int,
    body_cueq: int,
    *,
    structural: str,
    platform: str,
    census: dict[str, int] | None,
) -> str:
    """The gather arm's verdict for a family that runs a triangle attention.

    Two-sided on both axes the arm could be wrong on.  The algorithm: the
    gather adapter must have fired (checked by the caller against
    `CP_EXPECTED`) and neither ring tile body may have.  The body: on a GPU
    the cuEq body must have been entered *and* its FFI target must be in the
    compiled program -- entering the Python wrapper is not proof the kernel
    was lowered -- with the XLA reference silent; anywhere else the reference
    must have run and neither the cuEq body nor its target may appear.
    """

    if merge or two_pass:
        raise RuntimeError(
            f"{family}/2d gather: a ring tile body fired (merge {merge}x, "
            f"two-pass {two_pass}x; {fired}), so this row ran the ring"
        )
    kernels = None if census is None else census.get(CUEQ_ATTENTION_TARGET, 0)
    if platform == "gpu":
        if body_cueq < 1:
            raise RuntimeError(
                f"{family}/2d gather: the cuEq body never fired on a GPU "
                f"({fired})"
            )
        if body_xla:
            raise RuntimeError(
                f"{family}/2d gather: the XLA reference body fired "
                f"{body_xla} times on a GPU ({fired}), so this row is not "
                "the gather arm's kernel"
            )
        if kernels is None:
            raise RuntimeError(
                f"{family}/2d gather: no HLO census, and on a GPU the "
                f"{CUEQ_ATTENTION_TARGET} custom call is half the check"
            )
        if kernels < 1:
            raise RuntimeError(
                f"{family}/2d gather: the cuEq body was entered but "
                f"{CUEQ_ATTENTION_TARGET} is not in the compiled program "
                f"({census})"
            )
        return (
            f"{structural}; gather body cueq {body_cueq}x, "
            f"{CUEQ_ATTENTION_TARGET} {kernels}x in HLO, ring tiles 0x"
        )
    if body_xla < 1:
        raise RuntimeError(
            f"{family}/2d gather: the XLA reference body never fired off a "
            f"GPU ({fired})"
        )
    if body_cueq or kernels:
        raise RuntimeError(
            f"{family}/2d gather: the cuEq body fired {body_cueq}x and its "
            f"target appears {kernels}x off a GPU ({fired}, {census})"
        )
    return f"{structural}; gather body xla {body_xla}x, ring tiles 0x"


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #


def markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    header = (
        f"# Boltz-2 wall split -- {report['tokens']} tokens, "
        f"{report['layout']} layout"
    )
    lines += [header, ""]
    lines += [
        f"- case `{report.get('case') or 'synthetic'}`, MSA depth "
        f"{report['msa_depth']}, atoms {report['atoms']} "
        f"({report['atoms_source']})",
        f"- schedule {report['schedule']['num_samples']} samples / "
        f"{report['schedule']['num_recycles']} recycles / "
        f"{report['schedule']['num_steps']} steps, "
        f"{report['call_counts']['trunk_passes']} trunk passes",
        f"- chunk policy {report['chunks']}"
        + (
            f", ring row block {report['ring_row_block']}"
            if report.get("ring_row_block") is not None
            else ""
        ),
        f"- {report['devices']} device(s), {report['platform']}, jax "
        f"{report['jax']}, {report['repeats']} timed repeats",
    ]
    # Only named when it is not the released ring, so a ring report reads as
    # it always has.
    if report.get("triangle_attention_grid", "ring") != "ring":
        lines.append(
            f"- triangle attention grid `{report['triangle_attention_grid']}` "
            "(no ring; the ring row block above is the gather's row block)"
        )
    lines.append("")
    for arm_name, arm in report["arms"].items():
        lines += [f"## arm `{arm_name}`", ""]
        lines += [f"- backends {arm['backends']}", ""]
        lines += [
            "| family | per call (ms) | calls | predicted (s) | share | verdict |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        total = arm["predicted_total_seconds"]
        for name, row in arm["families"].items():
            if "error" in row:
                lines.append(
                    f"| {name} | -- | {row['calls']} | -- | -- | "
                    f"FAILED: {row['error'].splitlines()[0][:80]} |"
                )
                continue
            role = " (cross-check)" if name in CROSS_CHECK else ""
            share = (
                "--"
                if name in CROSS_CHECK or not total
                else f"{100.0 * row['predicted_seconds'] / total:.1f}%"
            )
            lines.append(
                f"| {name}{role} | {row['latency_median'] * 1e3:.2f} | "
                f"{row['calls']} | {row['predicted_seconds']:.1f} | {share} | "
                f"{row['verdict']} |"
            )
        lines += ["", f"- predicted sum: **{total:.1f} s**"]
        measured = arm.get("measured_wall_seconds")
        if measured:
            lines.append(
                f"- measured wall: {measured:.1f} s; predicted sum is "
                f"{100.0 * total / measured:.1f}% of it, leaving "
                f"{measured - total:.1f} s unaccounted"
            )
        else:
            lines.append(
                "- measured wall: not supplied (`--measured-wall`), so there "
                "is no sanity line for this arm"
            )
        for name in CROSS_CHECK:
            whole = arm["families"].get(name, {})
            pieces = arm.get("cross_check", {}).get(name)
            if pieces and "latency_median" in whole:
                lines.append(
                    f"- `{name}`: whole layer {whole['latency_median'] * 1e3:.2f} "
                    f"ms against {pieces['sum_ms']:.2f} ms of summed pieces "
                    f"({pieces['ratio']:.2f}x)"
                )
        lines.append("")
    lines += ["## not timed", ""]
    lines += [f"- {entry}" for entry in report["not_timed"]]
    lines.append("")
    return "\n".join(lines)


#: Which timed families make up each cross-check family, so the whole-layer
#: number has something to be compared against.
CROSS_CHECK_PIECES = {
    "trunk_pairformer_layer": (
        "trunk_tri_mul_out",
        "trunk_tri_mul_in",
        "trunk_tri_att_start",
        "trunk_tri_att_end",
        "trunk_pair_transition",
        "trunk_single_attention",
        "trunk_single_transition",
    ),
    "msa_layer": ("msa_pwa", "msa_transition", "msa_opm", "msa_noseq_layer"),
}


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tokens", type=int, default=2096)
    parser.add_argument(
        "--msa-depth",
        type=int,
        default=None,
        help="alignment rows the MSA module sees; defaults to the case's own "
        "depth (5DEI 8,192; the 6,568-token case 377)",
    )
    parser.add_argument(
        "--atoms",
        type=int,
        default=None,
        help="padded heavy-atom count; defaults to a count from the bench "
        "job's sequence, rounded up to the window alignment",
    )
    parser.add_argument("--layout", choices=("serial", "2d"), default="serial")
    parser.add_argument(
        "--kernels",
        default="released,xla",
        help="serial arms, comma-separated: released, xla, pallas, trimul",
    )
    parser.add_argument(
        "--ring-kernels",
        default=None,
        help="2-D ring arms: the ring tile kernel(s), xla and/or tokamax "
        "(default xla); refused beyond xla under --grid gather",
    )
    parser.add_argument(
        "--grid",
        choices=GRIDS,
        default="ring",
        help="2-D triangle-attention algorithm: the released ring, or the "
        "streamed gather (one arm, named `gather`)",
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--samples", type=int, default=SCHEDULE["num_samples"])
    parser.add_argument("--steps", type=int, default=SCHEDULE["num_steps"])
    parser.add_argument("--recycles", type=int, default=SCHEDULE["num_recycles"])
    parser.add_argument(
        "--families",
        default="",
        help="comma-separated subset to time; default is every family",
    )
    parser.add_argument(
        "--measured-wall",
        default=None,
        help="measured prediction wall in seconds for the sanity line: one "
        "number for every arm, or per arm as `xla=943,tokamax=618`",
    )
    parser.add_argument(
        "--transition-hidden-chunk",
        type=int,
        default=None,
        help="the released value is unset; spell it only to match a row that did",
    )
    parser.add_argument("--out", default=str(HERE / "out"))
    parser.add_argument(
        "--stem-suffix",
        default="",
        help="appended to the report filename, so two single-arm runs of the "
        "same size and layout do not overwrite each other",
    )
    parser.add_argument(
        "--print-counts",
        action="store_true",
        help="print the call counts read from the checkpoint and exit",
    )
    parser.add_argument("--no-hlo", action="store_true", help="skip the HLO census")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    # The code under test: WALL_SPLIT_SRC when set (a snapshot), else the
    # checkout's src. This insert outranks PYTHONPATH, so a caller that only
    # set PYTHONPATH measured the checkout (job 2368 timed main on both arms).
    sys.path.insert(0, os.environ.get("WALL_SPLIT_SRC", str(REPO / "src")))
    install_flags()

    import jax
    import jax.numpy as jnp

    from foldjax.execution import matmul_precision_scope
    from foldjax.models._cp import context_parallel, cp_grid
    from foldjax.models._cp_attention import (
        ring_tile_kernel_scope,
        triangle_attention_grid_scope,
    )
    from foldjax.models.boltz2.bridge.native import load_params
    from foldjax.models.boltz2.models.trunk_blocks.trunk import (
        _cast_trunk_params,
        _resolve_pair_residual_dtype,
        resolve_long_sequence_chunks,
    )

    home = Path(os.environ["FOLDJAX_HOME"])
    weights = home / "weights/boltz2/boltz2_conf"
    print(f"loading {weights}", flush=True)
    raw = load_params(weights)
    stacks = stack_lengths(raw)
    schedule = {
        "num_samples": args.samples,
        "num_steps": args.steps,
        "num_recycles": args.recycles,
    }
    counts = call_counts(stacks, schedule)
    if args.print_counts:
        print(json.dumps({"stacks": stacks, "call_counts": counts}, indent=2))
        return 0

    sizes = widths(raw)
    depth = (
        args.msa_depth
        if args.msa_depth is not None
        else CASE_MSA_DEPTH.get(args.tokens, max(1, args.tokens // 4))
    )
    rows = SIDE_2D if args.layout == "2d" else 1
    alignment = ATOM_WINDOW_QUERIES * rows
    if args.atoms is not None:
        atoms, atoms_source = args.atoms, "given on the command line"
    else:
        atoms, atoms_source = atom_count(args.tokens, alignment)
    if atoms % alignment:
        raise SystemExit(
            f"--atoms must divide {alignment} (query window {ATOM_WINDOW_QUERIES} "
            f"x {rows} CP row(s)); got {atoms}"
        )
    if args.layout == "2d" and depth % SIDE_2D:
        # `msa_module_forward` pads the alignment depth once for the whole
        # stack before any layer runs, so a harness that handed an unpadded
        # depth to one layer would be timing a shape the model never sees.
        depth += (-depth) % SIDE_2D
        print(f"padded MSA depth to {depth} for the grid", flush=True)

    # `compute_dtype="bfloat16"` with `pair_residual_dtype="auto"`: the port's
    # own cast, not a hand-written one, because `_cast_trunk_params` exempts
    # the input embedder's atom encoder, the template projection and the
    # Pairformer single branch by path.
    compute_dtype = jnp.dtype(COMPUTE_DTYPE)
    params = {
        "trunk": _cast_trunk_params(raw["trunk"], compute_dtype),
        "confidence": _cast_trunk_params(raw["confidence"], compute_dtype),
        # The diffusion score model is the one module upstream runs outside
        # autocast, so the released port keeps it float32 whatever the trunk
        # does: it takes no cast here.
        "conditioned_diffusion": raw["conditioned_diffusion"],
    }
    pair_residual = _resolve_pair_residual_dtype("auto", compute_dtype)
    assert pair_residual == compute_dtype, pair_residual

    chunks = resolve_long_sequence_chunks(
        args.tokens,
        chunk_size=128,
        triangle_attention_chunk=None,
        triangle_attention_q_chunk=None,
        token_attention_chunk=None,
    )
    chunks["transition_hidden_chunk"] = args.transition_hidden_chunk

    measured: dict[str, float] = {}
    default_measured: float | None = None
    if args.measured_wall:
        for entry in str(args.measured_wall).split(","):
            if "=" in entry:
                name, value = entry.split("=", 1)
                measured[name.strip()] = float(value)
            else:
                default_measured = float(entry)

    # The ring's local row block, which is what one ring step's cost is set
    # by, resolved the way the released 2-D program resolves it: the caller's
    # `triangle_attention_q_chunk` narrowed by the ring's own rule.  Reported
    # because it is not the caller's number -- at 2,096 tokens the policy asks
    # for 256 and the rule answers 64.
    ring_row_block = None
    if args.layout == "2d":
        from foldjax.models._cp_attention import resolve_ring_row_block

        local = args.tokens // SIDE_2D
        ring_row_block = int(
            resolve_ring_row_block(
                local,
                heads=sizes["triangle_heads"],
                local_keys=local,
                requested=chunks["triangle_attention_q_chunk"],
            )
        )

    report: dict[str, Any] = {
        "tokens": args.tokens,
        "case": CASES.get(args.tokens),
        "msa_depth": depth,
        "atoms": atoms,
        "atoms_source": atoms_source,
        "layout": args.layout,
        "schedule": schedule,
        "stacks": stacks,
        "call_counts": counts,
        "widths": sizes,
        "chunks": chunks,
        "ring_row_block": ring_row_block,
        "compute_dtype": COMPUTE_DTYPE,
        "diffusion_compute_dtype": DIFFUSION_COMPUTE_DTYPE,
        "pair_residual_dtype": str(pair_residual),
        "matmul_precision_scope": MATMUL_SCOPE,
        "matmul_precision_explicit": MATMUL_PRECISION,
        "repeats": args.repeats,
        "triangle_attention_grid": args.grid,
        "platform": jax.default_backend(),
        "jax": jax.__version__,
        "devices": jax.device_count(),
        "not_timed": list(NOT_TIMED),
        "arms": {},
    }
    try:
        import tokamax

        report["tokamax"] = getattr(tokamax, "__version__", "unknown")
    except Exception as error:  # pragma: no cover
        report["tokamax"] = f"unavailable: {error}"

    wanted = [name for name in args.families.split(",") if name]
    if args.layout == "2d":
        if jax.device_count() < DEVICES_2D:
            raise SystemExit(
                f"the 2-D layout needs {DEVICES_2D} devices, saw "
                f"{jax.device_count()}"
            )
        if args.grid == "gather":
            # `backends/boltz2.py` refuses the same pair: a ring body beside
            # an algorithm that runs no ring would label a knob that did
            # nothing.
            if args.ring_kernels not in (None, "xla"):
                raise SystemExit(
                    f"--grid gather runs no ring, so --ring-kernels "
                    f"{args.ring_kernels!r} would name a tile kernel nothing "
                    "calls"
                )
            if args.no_hlo and report["platform"] == "gpu":
                raise SystemExit(
                    "--grid gather on a GPU needs the HLO census: the "
                    f"{CUEQ_ATTENTION_TARGET} custom call is half its tripwire"
                )
            arm_names = ["gather"]
        else:
            arm_names = [
                name for name in (args.ring_kernels or "xla").split(",") if name
            ]
    else:
        if args.grid != "ring":
            raise SystemExit("--grid gather is a 2-D algorithm; pass --layout 2d")
        arm_names = [name for name in args.kernels.split(",") if name]
        for name in arm_names:
            if name not in ARMS:
                raise SystemExit(f"unknown serial arm {name!r}; pick from {list(ARMS)}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"wall-split-{args.tokens}-{args.layout}"
    # A gather report written under the ring's unsuffixed stem would overwrite
    # a ring report of the same size.
    suffix = args.stem_suffix or ("gather" if args.grid == "gather" else "")
    if suffix:
        stem = f"{stem}-{suffix}"

    for arm_name in arm_names:
        gather = arm_name == "gather"
        ring_kernel = arm_name if args.layout == "2d" and not gather else None
        backend_arm = "xla" if args.layout == "2d" else arm_name
        backends = dict(ARMS[backend_arm])
        # Triangle multiplication reads its backend from the environment, and
        # `triangle_multiplication_forward` reads it per call -- so this is a
        # real arm switch, not a label.  Set explicitly in both arms: leaving
        # it unset would make "not touched" mean cueq in one arm and inherit
        # whatever the shell had in the other.
        os.environ["BOLTZ_JAX_TRIANGLE_MULTIPLICATION_BACKEND"] = backends.pop(
            "tri_mul_env"
        )
        arm: dict[str, Any] = {
            "backends": dict(backends),
            "tri_mul_backend_env": os.environ[
                "BOLTZ_JAX_TRIANGLE_MULTIPLICATION_BACKEND"
            ],
            "ring_tile_kernel": ring_kernel,
            "families": {},
        }
        if gather:
            arm["triangle_attention_grid"] = "gather"
        print(f"== arm {arm_name} ({args.layout}) ==", flush=True)

        with contextlib.ExitStack() as stack:
            stack.enter_context(matmul_precision_scope(MATMUL_SCOPE))
            mesh = None
            if args.layout == "2d":
                mesh = stack.enter_context(
                    context_parallel(DEVICES_2D, layout="2d")
                )
                assert cp_grid() == (SIDE_2D, SIDE_2D), cp_grid()
                if gather:
                    # Every family traces inside `time_program`, within this
                    # stack, so the scope covers all of them.  No tile scope:
                    # the gather reads none.
                    stack.enter_context(triangle_attention_grid_scope("gather"))
                else:
                    stack.enter_context(ring_tile_kernel_scope(ring_kernel))
                arm["cp_grid"] = list(cp_grid())
            tripwire = stack.enter_context(Tripwire())

            operands = Operands(
                tokens=args.tokens,
                depth=depth,
                atoms=atoms,
                sizes=sizes,
                samples=args.samples,
                layout=args.layout,
                mesh=mesh,
            )
            builders = build_families(
                params,
                operands,
                chunks=chunks,
                backends=backends,
                sizes=sizes,
            )
            names = wanted or list(builders)
            for name in names:
                if name not in builders:
                    raise SystemExit(
                        f"unknown family {name!r}; pick from {list(builders)}"
                    )
                calls = counts[name]
                row: dict[str, Any] = {"calls": calls}
                tripwire.reset()
                try:
                    function, arguments = builders[name]()
                    row.update(
                        time_program(
                            function,
                            arguments,
                            repeats=args.repeats,
                            dump_hlo=not args.no_hlo,
                        )
                    )
                    fired = tripwire.snapshot()
                    row["tripwire"] = fired
                    row["verdict"] = check_tripwire(
                        name,
                        backend_arm,
                        args.layout,
                        ring_kernel,
                        fired,
                        grid="gather" if gather else "ring",
                        platform=report["platform"],
                        census=None if args.no_hlo else row["custom_calls"],
                    )
                    row["predicted_seconds"] = row["latency_median"] * calls
                    print(
                        f"  {name}: {row['latency_median'] * 1e3:.2f} ms x "
                        f"{calls} = {row['predicted_seconds']:.1f} s  "
                        f"[{row['verdict']}]",
                        flush=True,
                    )
                except Exception as error:
                    # One family that OOMs or refuses must not take the table
                    # with it: the remaining rows are still the answer to the
                    # question, and the failure is itself a finding.
                    row["error"] = "".join(
                        traceback.format_exception_only(type(error), error)
                    ).strip()
                    row["traceback"] = traceback.format_exc()[-4000:]
                    print(f"  {name}: FAILED -- {row['error']}", flush=True)
                arm["families"][name] = row

        total = sum(
            row["predicted_seconds"]
            for name, row in arm["families"].items()
            if name not in CROSS_CHECK and "predicted_seconds" in row
        )
        arm["predicted_total_seconds"] = total
        arm["measured_wall_seconds"] = measured.get(arm_name, default_measured)
        arm["cross_check"] = {}
        for whole, pieces in CROSS_CHECK_PIECES.items():
            if whole not in arm["families"] or "latency_median" not in arm[
                "families"
            ][whole]:
                continue
            available = [
                arm["families"][piece]["latency_median"]
                for piece in pieces
                if piece in arm["families"]
                and "latency_median" in arm["families"][piece]
            ]
            if len(available) != len(pieces):
                continue
            summed = sum(available)
            arm["cross_check"][whole] = {
                "sum_ms": summed * 1e3,
                "whole_ms": arm["families"][whole]["latency_median"] * 1e3,
                "ratio": (
                    arm["families"][whole]["latency_median"] / summed
                    if summed
                    else float("nan")
                ),
            }
        report["arms"][arm_name] = arm
        print(f"  predicted sum {total:.1f} s", flush=True)

    (out / f"{stem}.json").write_text(json.dumps(report, indent=2, default=str))
    (out / f"{stem}.md").write_text(markdown(report))
    print(f"wrote {out / (stem + '.json')} and {out / (stem + '.md')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
