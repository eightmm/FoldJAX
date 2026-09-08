# Where Boltz-2's 5SAK error is born: the MSA module, on identical inputs

Status: a localization from existing captures. No fix, no admission. It narrows
the one cell that fails the cross-model gate.

## The cell this is about

The [cross-model gate](common-error-standard-2026-09-09.md) is crossed exactly
once, by Boltz-2 on 5SAK at 1.1761 Å. That cell is already decomposed:
substituting upstream trunk tensors collapses all five sampler trajectories from
`[1.18, 0.17, 0.06, 1.04, 0.48]` to below 0.001 Å, so the sampler and the scan
lowering are excluded. What remains is a trunk `s_rmse` of 0.0065 and `z_rmse`
of 0.0135 that this target's long diffusion trajectory turns into an angstrom.

This narrows *where in the trunk*.

## The stage table

Native capture `boltz-amp-closure-20260907/native-5sak` against a JAX capture
built from it (`--upstream-capture`), so the two are paired by construction.
437 tokens, four recycles.

| Stage | RMSE | Relative |
| --- | ---: | ---: |
| `rel_pos` (pair initialisation) | **0.000000** | 0.000e+00 |
| `input_embedder` (single) | 5.901610e-06 | 2.460e-05 |
| cycle-00 MSA `input_z` | 6.158335e-04 | 9.765e-05 |
| **cycle-00 MSA `delta_z`** | **3.238075e-02** | 2.073e-03 |
| cycle-00 pairformer `output_z` | 7.860285e-02 | 3.295e-03 |
| cycle-03 MSA `input_z` | 1.369353e-02 | 1.877e-03 |
| cycle-03 MSA `delta_z` | 3.427136e-02 | 2.035e-03 |
| cycle-03 pairformer `output_z` | 7.577320e-02 | 3.050e-03 |

The pair representation starts **bitwise identical**. The single embedding is
2.5e-5 relative. The pair entering the first MSA module is 6.2e-4. The update
that module returns is 3.2e-2 -- **a factor of 53 across one module**, and the
largest single step anywhere in the trunk. Everything after it is that error
being carried and roughly doubled by the pairformer, then recycled.

## The control that makes it arithmetic

Every MSA input the JAX side consumed was compared against the native feature
file:

| Feature | |
| --- | --- |
| `msa`, `msa_mask`, `msa_paired` | bitwise equal |
| `deletion_value`, `has_deletion` | bitwise equal |
| `token_pad_mask` | bitwise equal |

Six of six identical. Combined with a pair input already matched to 6.2e-4 and
a bitwise-identical `rel_pos`, the 53x is not a different input being fed to the
same arithmetic. It is the same input through different arithmetic.

## What this does and does not say

It says the search for Boltz-2's remaining trunk residual belongs inside the MSA
module at the first recycle, not in the pair initialisation, the input embedder,
the pairformer, the sampler or the scan -- all of which are now either exact or
downstream carriers.

It does not name the operation inside that module. The native capture stores
only `input_z` and `delta_z` for this stage, so no finer boundary exists in the
artifacts to compare; the JAX side's extra keys are that module's inputs, which
is what made the control above possible but adds no internal boundary. Isolating
further needs a native capture with sub-module boundaries, which is a capture
change rather than an analysis one.

It also does not establish that closing this closes the cell. The decomposition
says the trunk residual is what the trajectory amplifies; it does not say this
stage is the whole trunk residual.

## Teacher-forcing the module: the input contributes nothing

Feeding the JAX MSA module native's own `input_z`, with the bitwise-identical
features, and comparing its `delta_z` against native's:

| Arm | RMSE against native `delta_z` |
| --- | ---: |
| in-run captured `delta_z` | 3.238075e-02 |
| JAX `input_z` -> JAX module | 3.375386e-02 |
| **native `input_z` -> JAX module** | **3.375255e-02** |

The two teacher-forced arms agree to five significant figures. The 6.2e-4
difference in the pair entering the module contributes essentially nothing to
the 3.2e-2 leaving it. The residual is the module's arithmetic, on identical
inputs.

(The standalone arms sit 4% above the in-run capture, because a module run
outside the whole graph does not get the same chunking and autotuning context.
Read the arms against each other, not against the in-run absolute.)

## Every exposed knob is excluded, and two make it worse

Same teacher-forced setup, one knob at a time:

| Arm | RMSE | |
| --- | ---: | --- |
| shipped (cueq triangle, xla glu, `highest`) | 3.375651e-02 | |
| `triangle_backend=xla` | 3.375084e-02 | unchanged |
| `use_scan=False` | 3.374830e-02 | unchanged |
| `matmul_precision=tensorfloat32` | 4.059741e-02 | **20% worse** |
| parameters cast to bfloat16 | 4.308405e-02 | **28% worse** |
| parameters cast to float32 | 3.375281e-02 | unchanged; they already are |

The fused cuEquivariance triangle kernel was the obvious suspect and moves the
result by 0.02%. The scan lowering moves it by 0.02%. The two knobs that do
move it both move it the wrong way, which is itself a result: `highest` is the
right precision for this module, and a blanket bfloat16 is not what native
does.

The parameters load as 58 float32 leaves **through this standalone loader**,
which is not what the run does -- see the correction below.

## Correction: the shipped configuration was not the baseline

`_amp_dtype(kernel)` (`msa.py:27`) returns the kernel dtype when it is bfloat16,
and that is what switches the module's autocast emulation on. Placing the
boundary by leaf name:

| Boundary | RMSE against native `delta_z` |
| --- | ---: |
| fp32 throughout | 3.375415e-02 |
| **`kernel` only -> bf16** | **3.233621e-02** |
| `kernel` + `bias` -> bf16 | 3.693807e-02 |
| everything -> bf16 | 4.308405e-02 |
| `scale` only -> bf16 (control) | 3.789293e-02 |

The direction is exactly autocast's: matmul kernels low, bias and normalisation
high. The control moves the wrong way, so this is not "any bfloat16 helps", and
the spread between repeated baseline evaluations is 6e-6, so 4% is two hundred
times the noise.

**But the in-run captured `delta_z` is 3.238075e-02, within 0.14% of the
`kernel`-only arm.** The run already casts parameters to the trunk dtype, so the
shipped configuration *is* that arm. The fp32 row above is the artificial one,
introduced by loading parameters through a standalone loader that does not.

So this is not a reduction of the shipped error. It establishes something else,
and something more useful: **parameter dtype placement is already optimal in the
shipped port**, in the same direction native uses, and every alternative
placement is worse. Together with the excluded knobs, the whole
parameter-and-configuration search space around this residual is now closed.

## What that leaves

Not a knob. The port is at its best available configuration and still 3.375e-2
from native, so closing this is an implementation difference rather than a
setting.

The port already places the parameter-side boundary where native does, and no
other placement is closer. What is left is the boundary on the *activation*
path: under autocast a linear returns bfloat16, so the next operation sees a
rounded input, whereas casting only the kernel leaves JAX promoting the product
back to fp32. Reproducing that is a source change per operation inside the
module, not a dtype choice, and it is what the native AMP boundary
investigation already underway is about.

## The activation boundary is excluded too, and the scan pins it

Same teacher-forced setup, `use_scan=False` in every arm because a bfloat16
pair carry does not type-check against the scan (`scan body function carry
input and carry output must have equal types`) -- the module's layer scan
holds `z` in fp32 by construction.

| Arm | RMSE | |
| --- | ---: | --- |
| shipped equivalent (kernels bf16, fp32 `z`) | 3.240702e-02 | |
| kernels bf16, `z` -> bf16 | 3.427809e-02 | **6% worse** |
| kernels bf16, `z` + `emb` -> bf16 | 3.427809e-02 | same as above |
| kernels bf16, `emb` -> bf16 only | 3.240702e-02 | no effect at all |

Rounding the pair activation the way autocast would makes it worse, so the
port's fp32 pair carry is closer to native than a bfloat16 one. The single
embedding's dtype changes nothing in either direction.

## The dtype and configuration axis is now closed

Fourteen arms, all teacher-forced on native's own `input_z` with bitwise
identical features:

| Axis | Arms | Result |
| --- | ---: | --- |
| Knobs (triangle backend, scan, matmul precision, chunking) | 5 | shipped best; two alternatives worse |
| Parameter dtype placement | 5 | shipped best; four alternatives worse |
| Activation dtype placement | 4 | shipped best; one worse, two inert |

The shipped configuration is optimal in every one, and the repeated-baseline
spread is 6e-6, so none of this is noise.

The residual of 3.23e-2 is therefore not a dtype, not a boundary placement and
not a setting. It is in how the operations themselves are carried out --
accumulation order and algorithm detail inside the module's own kernels --
which needs a comparison against upstream's implementation rather than another
arm of this harness.

That is the honest end of this axis. It is not a reduction of the cell; it is
the reason the next attempt should not spend itself here.

## The chunk policy diverges from upstream and it does not matter

Upstream's `trunkv2.py:594-600` switches to fixed per-operation chunk widths
once the token count passes `const.chunk_size_threshold = 384`. At 5SAK's 437
tokens that is: MSA transition 32, PWA head chunking on, triangle attention 128,
transition-z 64, **outer product mean 4**.

The port reproduces two of those exactly -- `msa.py:263` uses 32 for the MSA
transition and `heads_per_group = 1` turns on PWA head chunking above the same
384 -- but resolves the outer product mean's width from a memory budget
(`_auto_outer_product_chunk`), capped by the caller's `chunk_size`. At this size
the budget does not bind, so the run used 128 where upstream uses 4.

That is a real divergence from the released configuration, and it changes
nothing:

| Arm | RMSE |
| --- | ---: |
| shipped `chunk_size=128` | 3.233621e-02 |
| `chunk_size=4` (upstream's OPM width) | 3.233621e-02 |
| `chunk_size=32` | 3.233621e-02 |
| `chunk_size=64` | 3.233621e-02 |
| `chunk_size=128` + triangle attention 128 | 3.233621e-02 |
| `chunk_size=4` + triangle attention 128 | 3.233621e-02 |

Six arms, identical to seven significant figures. Both sides chunk the *token*
axis, so each chunk writes a disjoint set of output rows and nothing accumulates
across chunks. The width is free.

Worth fixing as a configuration-fidelity matter, and worth knowing that it is
not this residual.

## Closing position

Twenty arms across four axes -- knobs, parameter dtype, activation dtype, chunk
width -- and the shipped configuration is the best or equal in every one, with a
repeated-baseline spread of 6e-6.

The residual is not reachable from configuration. It is in how the operations
are computed, and separating that needs the port's kernels compared against
upstream's implementation op by op, with native sub-module boundaries to compare
against. The upstream source is readable here; the native intermediate captures
are not part of this branch.

## The outer product mean is faithfully ported, including the branch upstream takes

Read against `boltz/src/boltz/model/layers/outer_product_mean.py`. Above 384
tokens upstream takes its *chunked* branch, which differs from its unchunked one
in more than width: it splits the **hidden channel** axis rather than the token
axis, accumulates the projected result across those splits, keeps the outer
product in the input dtype instead of upcasting to float32, and adds the output
bias once at the end.

The port reproduces all of that in `_outer_product_mean_amp`: the same 384
threshold, `hidden_chunk = 4` matching upstream's `chunk_size_outer_product`, the
hidden-axis accumulation, the AMP-dtype einsum, and the deferred bias. The token
tiling around it is an extra the port adds for memory and, as the width sweep
above shows, is numerically free.

One difference survived that reading. The port computes the validity count in
float32 throughout, and its comment says this "counts the same entries as the
native FP32 sum" -- but upstream has no FP32 sum here. Both of its branches cast
the mask to the MSA tensor's dtype first, which under autocast is bfloat16, and
this target has 4,436 alignment rows while bfloat16 cannot represent integers
above 256 exactly. Every output element is divided by that count.

It is not the cause. Round-tripping the count through the AMP dtype leaves the
result bitwise identical, so the counts that actually occur here are inside
bfloat16's exact range -- the padded rows are masked out and the surviving
per-pair counts are small. The comment is still wrong about upstream, and the
divergence is still real; it just does not reach the output on this input.

## Retraction: the count exclusion above was measured on dead code

The section above reports that round-tripping the outer product mean's validity
count through the autocast dtype is bitwise identical, and concludes the count
is not the cause. **That conclusion does not stand.**

A later probe patched the pair-weighted-averaging head loop in
`_pair_weighted_averaging_amp` and found it inert. Suspecting the harness rather
than the arithmetic, the same branch was then made to *double* its logits -- a
change nothing could survive numerically. The result was still bitwise
identical. The branch never executes.

`msa.py` carries more than one implementation of that head loop --
`_pair_weighted_averaging_amp` and `_pair_weighted_averaging_chunked` at least,
selected by a row-chunk decision -- and the patched one is not the one this
input takes. The count patch sat in `_outer_product_mean_amp`, a different
function, and was never given the same tripwire, so it inherits the same doubt.

Both are withdrawn. What remains standing from the source reading is the
comparison itself: upstream masks and softmaxes the pair bias in the autocast
dtype while the port upcasts to float32, and upstream's count is bfloat16-valued
while the port's is float32. Those are real divergences in the source. Whether
they reach the output is unmeasured, because the arms that claimed to measure
them were pointed at code that does not run.

**Every dtype arm in this document that was applied by patching a named function
needs a tripwire before it can be read.** The arms that changed *arguments* --
knobs, chunk widths, parameter and activation dtypes -- are unaffected: those
reach the module through its signature and their effects were visible.

## The probe harness stopped being trustworthy, and that ends this line

The retraction above blamed a dead branch. That reason was itself unreliable.
A later arm that patched the same head loop *unconditionally* did move the
result -- by a hundredfold -- so the branch is live. Two arms after that,
intended to differ only in whether JAX's compilation cache was enabled,
returned different numbers *and* reported different environment labels, which
means they did not differ only in the cache.

So the harness was leaking environment between arms, its label printed a
different variable than the one under test, and the job identifiers used to
read results back were computed by arithmetic on a shared queue. Any one of
those is enough to void an arm.

What stands and what does not:

| Arms | Status |
| --- | --- |
| Knobs, chunk widths, parameter dtype, activation dtype -- passed as **arguments** | stand; several moved the result, and each was read from its own labelled run |
| PWA logit dtype, OPM count dtype -- applied by **patching a function** | void; unmeasured |
| "The AMP head loop never executes" | withdrawn; it does execute |

The two source divergences found by reading upstream remain real and remain
open: upstream masks and softmaxes the pair bias in the autocast dtype where
the port upcasts to float32, and upstream's validity count is bfloat16-valued
where the port's is float32.

Measuring them needs a harness that runs one arm per process with an explicit
environment, prints the variable actually under test, reads results by job id
rather than by arithmetic, and proves each patched branch fires. Building that
is the first task of whoever continues, before any further exclusion is
recorded.

## Remeasured on a harness that cannot leak: the PWA logit dtype is excluded

The previous arms were voided for good reason, so they were rebuilt without the
mechanisms that voided them. No environment variable and no branch: two complete
source snapshots, differing only in the three lines under test, each run in its
own job with the compilation cache off and its identifier captured from the
queue rather than computed.

**Positive control first.** Under the same conditions, doubling the logits in
this exact loop moves the result from 3.233621e-02 to 3.126505e+00 -- a hundred
fold. The loop executes and source changes to it reach the output. Without this,
an identical pair of arms would mean nothing.

| Arm | RMSE against native `delta_z` |
| --- | ---: |
| shipped: logits upcast to float32, masked and softmaxed there | 3.233621e-02 |
| upstream's placement: logits masked in the autocast dtype | 3.233621e-02 |

Identical to seven significant figures. The divergence is real in the source --
upstream keeps the pair bias in bfloat16 through the mask and the softmax where
the port upcasts first -- and it does not reach the output. The softmax weights
are cast to the autocast dtype either way, and on this input they round to the
same bfloat16 values whether the exponential saw a rounded logit or not.

This exclusion stands where the earlier ones did not, because the control proves
the arm could have moved.

## Where this leaves the cell

Still 1.1761 Å. Nothing in this document reduces it.

What it now rules out, with the arms that survive scrutiny: every configuration
knob, every parameter and activation dtype placement, every chunk width, and the
pair-bias softmax dtype. What it rules in: the residual is born in the MSA
module, on bitwise-identical inputs, with the pair input contributing nothing.

The one divergence still unmeasured is the validity count's dtype, which needs
the same two-snapshot treatment and the same positive control.

## The validity count dtype is excluded, control in the same run

Three snapshots this time, the control queued alongside rather than cited from
an earlier job:

| Arm | RMSE against native `delta_z` |
| --- | ---: |
| shipped: count kept in float32 | 3.227007e-02 |
| upstream's placement: count round-tripped through the autocast dtype | 3.227007e-02 |
| **control: count doubled** | **1.371935e+00** |

The control moves the result forty-two fold, so the code executes and changes to
it reach the output. Against that, the two real arms are identical.

Upstream's count really is bfloat16-valued where the port's is float32, and on
this input it does not matter: the surviving per-pair counts sit inside
bfloat16's exact range once the padded alignment rows are masked out.

## Final position on this cell

**Still 1.1761 Å. This document does not reduce it.**

Excluded, each on an arm whose effect was demonstrable -- either because it was
passed as an argument and moved the result, or because a control in the same
design moved it:

- every configuration knob: triangle backend, scan, matmul precision
- every parameter dtype placement: kernels, bias, norms, and combinations
- every activation dtype placement: pair carry, single embedding
- every chunk width, at four values, and the port's divergence from upstream's
  fixed widths
- the pair-bias softmax dtype
- the validity count dtype

Ruled in: the residual is born inside the MSA module, on bitwise-identical MSA
features, with the pair entering it contributing essentially nothing, and it is
not reachable from dtype, boundary placement or configuration.

What is left is the arithmetic of the triangle operations themselves, against
upstream's implementation. The upstream source is readable in this workspace and
the harness that can measure it now exists: independent snapshots, one job each,
no shared environment, compilation cache off, and a positive control queued with
every claim.
