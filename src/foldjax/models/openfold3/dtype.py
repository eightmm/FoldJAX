"""Which parts of OpenFold3 a narrow ``dtype`` reaches, and which it does not.

The one-sentence shape: **the token/pair representation track narrows; anything
atom- or coordinate-shaped stays float32.** That is AlphaFold 3's released
inference shape, read out of its source rather than out of its config name, and
it is close to the regime this checkpoint was trained under.

**The default is ``bfloat16``**, and ``dtype="float32"`` is the opt-out back to
upstream's own inference precision. The measurement behind that choice is the
28-row panel near the end of this docstring. What it settles -- and what the
two earlier accounts of the same target got wrong -- is that the one failure
anybody found at 3,012 tokens belongs to the target and not to the dtype.

Reading AlphaFold 3's shape correctly takes two facts, not one.
``diffusion_head.py:221`` opens ``utils.bfloat16_context()`` around the whole
diffusion head, which looks like "the diffusion head is bfloat16". It is not.
``components/haiku_modules.py:175`` requests every parameter with
``hk.get_parameter(..., inputs.dtype, ...)`` and ``components/utils.py:48``'s
getter narrows only when the requested dtype is *already* bfloat16, so the
context narrows nothing on its own: a module's dtype is decided by the dtype of
the activation that enters it. Following that rule through the diffusion head:

* ``atom_cross_attention.py:49-53`` builds the per-atom conditioning from
  ``ref_structure.positions``, which is float32, so ``queries_single_cond``,
  ``keys_single_cond`` and ``pair_act`` are float32; the noisy positions that
  become ``queries_act`` are float32 too. **The atom cross-attention encoder and
  decoder run in float32**, inside the bfloat16 context. ``model.py:176``'s
  explicit ``.astype(dtype)`` on the concatenated encoder output is the tell --
  it would be redundant if the encoder returned bfloat16.
* ``diffusion_head.py:265-268`` then casts ``act``, ``trunk_single_cond``,
  ``trunk_pair_cond`` and ``sequence_mask`` to float32 before the token
  transformer. Only two of those four change anything: ``act`` arrived float32
  from the atom encoder, and ``single_cond`` was promoted to float32 back at
  ``diffusion_head.py:192-196`` where the float32 noise embedding is added.
  The load-bearing cast is ``trunk_pair_cond``.
* ``confidence_head.py:129-131`` casts ``pair``, ``single`` and ``target_feat``
  *down* to bfloat16, so the confidence head's re-embedding Pairformer does
  narrow. It comes back up the moment that stack returns -- ``:163`` for the
  pair track, ``:244`` for the single track -- and both casts land *before*
  the logit heads' own layer norms, which is what makes them load-bearing
  rather than cosmetic. Only the pTM reduction is pulled out of the context
  entirely (``:230``).

Upstream OpenFold3 is the second source, and it is about this checkpoint rather
than a sibling's. Inference runs float32 (``entry_points/validator.py:127``
``precision: "32-true"``), but all four released training configs run
``bf16-mixed`` -- ``examples/training_yamls/initial_training.yml:23`` and
``finetune_1/2/3.yml`` -- so these weights were *trained* with bfloat16
activations. Under that regime upstream pins exactly two islands to float32
inside the model, and both agree with AlphaFold 3's shape:

* ``feature_embedders/input_embedders.py:129-131`` wraps the input embedder's
  ``atom_attn_enc`` in ``autocast(dtype=torch.float32)``;
* ``heads/prediction_heads.py:223-224`` opens
  ``autocast(device_type="cuda", dtype=pairformer_dtype)`` around the
  confidence head's *Pairformer stack* and restores the incoming dtype at
  ``:241``; ``pairformer_dtype`` defaults to ``torch.float32``
  (``:131``, ``:192``, ``:261`` and ``heads/head_modules.py:106``) and no
  released config overrides it. Read against ``openfold3-v050``: the island is
  the stack, not ``embed_zij``, which runs before the context opens
  (``:193``).

The port measured the first island independently: a whole-trunk bfloat16 cast
takes pLDDT from 0.858 to 0.466 and the damage starts in the input embedder,
while keeping the embedder float32 recovers it (pLDDT -0.001, CA RMSD 0.040 A
against a 0.005 A rerun floor, at 1,003 tokens on 2026-08-10).

The confidence head is therefore its own narrowing group, with its own knob
(``confidence_dtype``). That knob *follows* ``dtype`` when unset, so opting
into a bfloat16 trunk narrows the head with it and there is no second value
to set; what the knob is for is holding this one region wide against a
narrowed trunk (``confidence_dtype="float32"``), or narrowing it against a
wide one. Narrowing it alone buys nothing measurable -- +0.4% wall and -0.1%
peak at 1,003 tokens, -1.1% and -0.0% at 2,096 -- and under a narrowed trunk
it is already subsumed: ``dtype=bfloat16`` and ``dtype=bfloat16
confidence_dtype=bfloat16`` reported byte-identical peaks (5,553 MiB at 1k,
19,107 MiB at 2k) and walls within 0.3%.

It is separable for a reason: it consumes predicted coordinates and emits
scores, never coordinates, so narrowing it cannot move a structure at all --
which is why no coordinate result below, including the 3,012-token basin
miss, can be laid at this knob's door. Protenix measured the same
change at 3,012 tokens -- coordinates bitwise unchanged, atom pLDDT at most
0.0099, chain pTM/ipTM at most 1.9e-4, PAE means at most 0.005 -- which is
evidence from a sibling, not from here.

What the option narrows in *this* port, which is the same shape stated as a
parameter split, is :func:`~foldjax.models.openfold3.inference.cast_narrow_params`
and the boundary casts it needs are in ``models/trunk.py``,
``models/input_embedders.py`` and ``models/heads.py``. Two deliberate
departures from the readings above:

* the input embedder's *five projections* stay float32 here, where upstream's
  autocast would run them at bfloat16. Only its atom encoder has to be wide,
  but keeping the whole subtree wide costs two ``[N, C]`` and one ``[N, N, C]``
  projection and buys an embedder that no future edit can narrow by accident.
  The cast lands on ``s`` and ``z`` immediately after.
* the template tower narrows its *body*, which upstream's autocast does too,
  but it gets there by being handed a narrowed feature mapping rather than by
  a cast inside the tower.

A narrowed region's attention softmax runs bfloat16, pair bias included.
Upstream does that on purpose: ``core/model/primitives/attention.py:111-127``
(``softmax_no_cast``, whose docstring is "Softmax, but without automatic
casting to fp32 when the input is of type bfloat16") disables autocast so a
bfloat16 softmax stays bfloat16, overriding torch's float32 default, and
AlphaFold 3's evoformer is the same shape.

**Layer norm is the opposite.** Upstream's
``primitives/normalization.py:54-70`` disables autocast for the reverse
reason -- to *force* float32: it takes ``x.float()`` with ``weight.float()``
and ``bias.float()`` (``:61-64``), normalises in float32, and rounds once on
the way out (``:70``), with the comment "LayerNorm should be upcasted to fp32
anyway in torch / This enforces it if not running with autocast context"
(``:57-58``). This port's ``models/primitives.py`` ``layer_norm`` used not to:
``jnp.mean`` accumulates in float32 but rounds the mean and the variance back
to bfloat16 before ``x - mean`` and the rsqrt multiply, and the scale and bias
were applied at bfloat16 too -- four extra roundings per layer norm against
upstream's one, at every layer norm in a narrowed region. It now upcasts, as
Protenix's ``layer_norm`` already did. Against a float64 reference on the
exact bfloat16 input values, per layer norm, eager on CPU:

====================  ================  ================  =====================
[4096, 128] input     narrow (before)   wide (now)        upstream, fp32 affine
====================  ================  ================  =====================
N(0, 1)               0.00461 / 0.0617  0.00244 / 0.0298  0.00165 / 0.0155
N(5, 1)               0.01001 / 0.0546  0.00245 / 0.0253  0.00165 / 0.0155
N(50, 1)              0.06767 / 0.1966  0.00245 / 0.0274  0.00165 / 0.0154
====================  ================  ================  =====================

RMS / peak absolute error; widths 64 and 384 track 128 to the third digit.
The regime matters more than the width: a zero-mean slab understates this by
an order of magnitude, and a pair residual after 48 blocks and ten cycles is
not zero-mean. The new arrangement is **bit-identical** to upstream's spelled
directly when both are handed the same affine; the residual 1.5x against the
last column is the bfloat16 *weight*, which ``cast_narrow_params`` narrows
with the rest of the trunk and which upstream's ``weight.float()`` never
rounds because its module parameters stay float32. With an identity affine,
exactly representable in bfloat16, the two columns are bit-identical.
Excluding layer-norm affine parameters from the narrowing would close that
1.5x at negligible memory cost; it is a parameter-split decision, not a
``layer_norm`` one, it was **tried and reverted**, and the reason is the
closing note below.

**What the upcast changes on GPU was read wrongly twice, and the way it was
read wrongly is the reusable part.** The first rows were two seeds at
6ZTX/3,012, each bfloat16 arm scored as a same-index residual against a
float32 run at the same seed: 1.121 A at seed 101 and 25.09 A at seed 202.
The float32 arm's own within-set spread looked healthy at both seeds, 0.619
and 0.569, and that was read as "the control is fine, so it is the arm".
Neither half of that inference holds. A residual between two arms cannot say
which of them moved. And a within-arm spread cannot see a basin the whole arm
is sitting in -- five samples agree with each other just as tightly in the
wrong place as in the right one. Both mistakes point the same way, and both
were made here.

What settles it is scoring against the deposited structure rather than
against the other arm, which the panel at the end of this docstring does.

The upcast is kept on its own terms and never rested on that reading. It is
bit-identical to upstream's arrangement where the old code was four roundings
wider, it is free in time (655.17 s wide against 664.61 narrow and 950.84
float32 at 3,012 tokens), byte-identical in peak at 34,893 MiB, and it emits
not one instruction under ``dtype="float32"``.

**The layer-norm affine is excluded from the narrowing, and that is where
the memory is.** :func:`narrow_floats` leaves every ``LayerNormParams``
float32 -- 0.381 MiB across 1,163 affine tensors -- which closes the last
1.5x above, twelve of twelve float64 rows bit-identical to upstream. The
0.381 MiB is not the point. Excluding the affine forces the guard in
:func:`~foldjax.models.openfold3.models.primitives.layer_norm` from the
promoted dtype to ``x.dtype``, because a float32 affine everywhere would
otherwise widen every trunk norm and the whole Pairformer with it. Reading
``x.dtype`` narrows the two stack-level pair norms *inside* the float32
denoiser, which the promoted guard was widening, and those are large:

=========================================  ==========  ============
3,012 tokens, 6ZTX, seed 101               wall        peak
=========================================  ==========  ============
promoted guard, affine narrowed (before)   656.7 s     34,893.0 MiB
``x.dtype`` guard, affine excluded         617.0 s     24,676.7 MiB
 .. only ``atom_features``'s norm wide     616.6 s     31,134.5 MiB
 .. only ``diffusion_transformer``'s wide  655.9 s     35,867.2 MiB
=========================================  ==========  ============

**-29.3% peak and -6.3% wall**, and -28.1% / -7.8% at 2,096 tokens (19,107.3
-> 13,742.5 MiB, 265.0 -> 243.7 s). The two attribution rows say the bulk
belongs to ``diffusion_transformer.py``'s stack-level ``layer_norm_z``:
widening it alone gives the whole saving back and then some, because the
explicit cast materialises a copy the promoted guard did not. So there is no
arrangement that keeps that norm wide and keeps the memory, and the question
is only whether narrowing it holds the structure.

**It does, at six seeds.** Scored against deposited 6ZTX on a
permutation-aware chain assignment, fitting the catalase core (122-753) and
reading the N-terminal arm (27-121) separately, five samples per seed:

* the core is 0.58-1.02 A in all six, against 0.56-0.93 A for the arrangement
  that keeps the norms wide;
* the arm is 0.38-0.48 A in five of six and 56.67-56.75 A at seed 404 -- and
  the wide arrangement misses the same arm at **the same seed** by the same
  amount, 56.70-56.79 A. That is the target's own bimodality reproduced, not
  a regression introduced here.

**This was reverted once, and how it was reverted is the reusable part.** The
revert rested on a five-sample within-set spread going 0.747 to 1.823 at one
seed. That statistic cannot rank two arrangements: its ordering reverses
between seeds, and the second seed did not reproduce it. A float64 error
table per operation had also pointed the other way earlier. Per-operation
error is not per-model error, an arrangement bit-identical to upstream is not
automatically the one to ship on a port whose other operations are not, and
neither of those instruments can settle a coordinate question -- what settles
it is scoring against a structure that exists.

Two facts about the exclusion a future reader should not have to rediscover:

* **No trunk site's output width moves.** Censused across ``trunk_cycle`` at
  the released widths, 62 calls at 13 distinct sites --
  ``msa.py:83``/``:139``/``:143``, ``triangle.py:119``/``:148``,
  ``triangle_attention.py:189``, ``attention_pair_bias.py:63``/``:72``,
  ``primitives.py`` (AdaLN and the SwiGLU transition),
  ``template_module.py:91``/``:150``, ``trunk.py:175``/``:208`` -- every one
  takes bfloat16 and returns bfloat16 under both guards, and the confidence
  stack runs the same code. In the trunk the whole difference is that each
  norm multiplies by an unrounded scale; the width change is in the denoiser.
* **It changes what ``triangle_kernel=cueq-full`` hands cuEquivariance.**
  ``norm_in_weight``/``norm_out_weight`` become float32 against a bfloat16
  ``x``. That is upstream's own operand signature -- its module parameters
  stay float32 under autocast -- so this should make that arm *more* faithful,
  and it is a candidate explanation for the stable 0.216 A it sits from native
  cuEq where the XLA-multiplication arm sits at 0.133. The kernel composes
  ``layer_norm_transpose`` as its own primitive and validates shapes rather
  than dtypes, and Triton specialises on pointer dtypes, so the mix compiles
  rather than misreads; whether its internal accumulation matches the pure-JAX
  reference is still unread, and that is what a ``cueq-full`` promotion owes.

**Why 3,012 tokens and not 2,096, as far as source can say.** Nothing in the
port keys on token count between the two. ``auto_pair_chunk_size`` is smooth
rather than stepped (502 rows at 1,003 tokens, 117 at 2,096, 58 at 3,012),
``per_sample_token_cutoff`` defaults to 0 and its own docstring records the
per-sample branch as bitwise identical on coordinates, cuEquivariance's
``CUEQ_TRIMUL_FALLBACK_THRESHOLD`` is 100 and below both sizes -- and
irrelevant under the default ``cueq``, which keeps that multiplication in XLA
-- and no bin edge, clamp or mask is conditioned on ``n_token``. The MSA row
cap is not a discriminator either: both targets are far above it, 17,542 rows
for 6ZTX and 13,280 for 5DEI against ``inference.RELEASED_MSA_DEPTH``'s 1,024.
The two sizes run the same program shape with different constants, so what
differs is the case and not the size. The panel below is the measurement that
says the same thing from the other end.

One more thing this does not reach: the two
triangle-multiplication norms under ``triangle_kernel=cueq-full``, where
``layer_norm_in``/``layer_norm_out`` are passed into cuEquivariance's fused
kernel (``models/triangle.py:180-220``) rather than computed here; the
default serial kernel is ``cueq``, which keeps that multiplication in XLA,
and the 3,012-token rows below were measured on it.

One caution for whoever measures that row. Under ``jax.jit`` on CPU the two
arrangements were bit-identical at ``[20000, 128]`` while differing at
``[64, 128]``, and the new arrangement's jitted result differs from its own
eager result at ``[20000, 128]`` as well -- XLA is not honouring the declared
widths there in either direction, and every number above is therefore eager,
cross-checked bit for bit against a numpy/``ml_dtypes`` emulation of the
jaxpr. Whether the GPU compiler does the same at trunk shapes was not
established, and it is the one reading under which this change is a no-op on
GPU. A jit-against-eager byte comparison of one norm at a trunk-shaped slab
settles it before the 3,012-token row is spent.

What is *not* narrowed anywhere is an output logit: the heads keep
float32 parameters, so distogram, PAE, PDE, pLDDT and experimentally-resolved
logits are float32 in both profiles. The one place a bfloat16 pair bias into a
softmax is known to be fatal on a port is the diffusion token transformer,
which is float32 here.

**The measurement the default rests on.** 28 rows, one input, one checkpoint,
10 recycles, 200 diffusion steps and five samples per seed, both dtypes built
from one source tree, sha256
``85a1ee17a35adcea26fb064f87a8e51eea57c2f57251ee804afcbacca505b7ee``:
seeds 101/202/303/404 at 1,003 tokens (3OG2) and 2,096 (5DEI), and those four
plus 505/606 at 3,012 (6ZTX). Every row finished with finite outputs.

======  ===========================  ==========================  ======================
tokens  warm wall float32 -> bf16    peak float32 -> bf16        CA RMSD vs deposited
======  ===========================  ==========================  ======================
1,003   98.5-98.8 -> 75.5-76.9 s     9,273 -> 5,552 MiB          0.72-0.92 / 0.70-0.92
2,096   402-405 -> 262-263 s         25,066 -> 19,107 MiB        0.45-0.58 on both arms
3,012   949-958 -> 658-665 s         50,412 -> 34,893 MiB        0.55-1.03 on both arms
======  ===========================  ==========================  ======================

The accuracy column is per chain against the deposited coordinates under a
permutation-aware chain assignment, which a homotetramer requires: 6ZTX is
catalase HPII, one sequence in four chains, and scoring it chain-for-chain by
label reads a relabelling as a large uniform displacement.

The peak column predates the affine exclusion above and is what the arm this
panel measured actually ran. The shipped profile is lower: 24,676.7 MiB at
3,012 tokens and 13,742.5 at 2,096, with the wall 617.0 s and 243.7 s.

**The 3,012-token failure is the target's, not the dtype's.** Fitting on the
catalase core (residues 122-753) and reading the N-terminal arm (27-121)
separately, over twelve arms of five samples each:

* the core is 0.55-1.03 A from the deposited structure in **all twelve** --
  both dtypes, every seed;
* the arm is 0.39-0.50 A in ten of them and about 56 A in two: **float32 at
  seed 202 (56.41-56.66 A) and bfloat16 at seed 404 (56.67-56.79 A)**, each
  time in all five samples of that seed.

So each dtype misses the arm's basin at one seed in six, and neither misses it
at a seed the other one makes. That is what retires the earlier "bfloat16
drifts above 2,000 tokens" conclusion: the 25 A at seed 202 was a float32 run
in the alternative basin and a bfloat16 run in the deposited one, measured as
the distance between them.

Six seeds per arm cannot estimate a rate, and nothing here claims the two
dtypes fail at equal rates -- only that the failure is not a property of
either. The mechanism is not measured. It is consistent with the per-recycle
MSA resubsampling this port performs (``backends/openfold3.py:636`` into
``data/featurize.py:1183-1190``, 1,024 of 17,542 rows redrawn on each of ten
recycles), which would make the arm's basin a draw the seed decides. The
arm's own pLDDT reports the miss locally -- 83.5 against 90.8-92.8 elsewhere,
no overlap -- where the whole-complex mean barely moves, 92.0 to 89.2.

Two upstream facts, for the record, both read out of ``openfold3-v050``:
inference runs 32-true (``openfold3/entry_points/validator.py:127``,
``examples/reference_full_config/full_config.yml:29``), and the confidence
Pairformer additionally carries its own ``pairformer_dtype`` defaulting to
``torch.float32`` (``heads/prediction_heads.py:131``, ``:192``, ``:261``,
``heads/head_modules.py:106``), so upstream pins that region wide rather than
merely leaving it wide. This port's default departs from both, under the
policy that an execution default is judged by whether the structure holds
rather than by whether it matches upstream's arithmetic -- and
``dtype="float32"`` is the one spelling that restores that arithmetic.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from foldjax.models.openfold3.models.primitives import LayerNormParams

#: Accepted spellings, in `foldjax.execution`'s neutral vocabulary. The default
#: is named here rather than repeated at each caller so the released profile has
#: one definition.
DTYPES: tuple[str, ...] = ("float32", "bfloat16")

#: What a request that says nothing gets. One value rather than two --
#: ``confidence_dtype`` follows it when unset, so this single definition sets
#: both regions. Every other layer reads it: the config fields,
#: ``released_config``'s signature and the backend's cache-namespace strip.
DEFAULT_DTYPE = "bfloat16"


def narrow_dtype(name: str) -> Any:
    """Return the dtype ``name`` narrows to, or ``None`` when it narrows nothing.

    ``None`` rather than ``jnp.float32`` on purpose: every narrowing site below
    is guarded on it, so the released profile emits no casts at all rather than
    a tree of float32-to-float32 converts that a reader would have to prove are
    free.
    """

    if name not in DTYPES:
        raise ValueError(
            f"OpenFold3 dtype must be one of {', '.join(DTYPES)}; got {name!r}"
        )
    return jnp.bfloat16 if name == "bfloat16" else None


def narrow_floats[T](tree: T, dtype: Any) -> T:
    """Cast every floating leaf of ``tree`` to ``dtype``, leaving the rest alone.

    Integer and boolean leaves stay as they are: JAX's promotion lattice puts
    every integer below every float, so an integer one-hot multiplied by a
    bfloat16 weight is already bfloat16, and casting a mask's index arithmetic
    to a float would be a different program.

    A :class:`~foldjax.models.openfold3.models.primitives.LayerNormParams` is
    the one floating node this does not cast, wherever it appears -- including
    the two nested inside an ``AdaLNParams``. Upstream's ``LayerNorm.forward``
    reads ``self.weight.float()`` off a module autocast never casts, so it
    normalises by the unrounded scale; rounding it here would add a rounding
    the port's reference does not have, and it is the only error left in
    ``layer_norm`` once the accumulation is wide. The price is exact rather
    than estimated: 1,163 affine tensors across the narrowed subtrees of the
    released checkpoint, 798,252 bytes at float32 against 399,126 at
    bfloat16, so 0.381 MiB out of a 623 MiB narrowed parameter set. No buffer
    changes shape -- the affine multiplies a ``[C]`` vector into an
    already-upcast activation, one broadcast and no copy.

    What it *does* change, and why it is worth 10 GiB at 3,012 tokens, is the
    guard in :func:`~foldjax.models.openfold3.models.primitives.layer_norm`:
    with a float32 affine everywhere, a guard that read the promoted dtype
    would return float32 at every trunk norm and widen the whole Pairformer,
    so the guard reads ``x.dtype`` instead. That narrows the two stack-level
    pair norms inside the float32 denoiser, which is where the memory is.
    """

    if dtype is None:
        return tree

    def cast(value):
        if isinstance(value, LayerNormParams):
            return value
        value_dtype = getattr(value, "dtype", None)
        if value_dtype is not None and jnp.issubdtype(value_dtype, jnp.floating):
            return value.astype(dtype)
        return value

    return jax.tree.map(
        cast, tree, is_leaf=lambda node: isinstance(node, LayerNormParams)
    )
