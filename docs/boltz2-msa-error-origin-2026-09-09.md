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

## Correction: both sides use cuEquivariance, and the difference is call granularity

Two claims above are wrong and are corrected here.

**"Two different fused kernel implementations" is wrong.** Upstream's
`kernel_triangular_mult` imports
`cuequivariance_torch.primitives.triangle.triangle_multiplicative_update`. It is
the same library the port calls through `cuequivariance_jax`. That was asserted
without checking.

**The contraction-dtype reading was of dead code.** The port's
`triangle_multiplication_forward` resolves a backend from
`BOLTZ_JAX_TRIANGLE_MULTIPLICATION_BACKEND`, defaulting to `cueq`, and returns
from `cueq_triangle_multiplication_forward` before reaching the `native_amp` and
`contraction_precision` logic that was analysed. That logic is the XLA fallback
and does not run by default.

### What the difference actually is

| | Upstream | Port, shipped configuration |
| --- | --- | --- |
| Call | one fused `triangle_multiplicative_update` | `norm`, `gemm`, `gemm_dual` called separately |
| Autocast boundaries | left to torch autocast | placed by hand in `_cueq_triangle_native_amp` |
| `precision` argument | not passed; library default | computed and passed explicitly |

With fp32 activations and bfloat16 kernels -- the shipped configuration --
`cueq_triangle_multiplication_forward` skips the fused call entirely and
decomposes the operator into primitives to reproduce torch's autocast placement.
The same arithmetic through one fused kernel and through three primitive calls
does not accumulate the same way.

### Measured, with a control

| Arm | RMSE against native `delta_z` |
| --- | ---: |
| shipped: decomposed primitives | 3.233621e-02 |
| **upstream's granularity: the fused call** | **3.212102e-02** |
| control: decomposed path doubled | 8.536005e-01 |

The control moves the result twenty-six fold. The fused call is **0.67% closer
to native**, against a repeated-baseline spread of 0.02% -- thirty times the
noise, and in the direction the source comparison predicts.

This is the first arm in this investigation that reduces the residual rather
than excluding a candidate. It is small: 0.67% of a 3.2e-2 stage residual will
not by itself move a 1.18 Å cell. What it establishes is that the remaining
difference is *reachable* -- it lives in how the port calls a shared library, not
in a kernel nobody here can change.

## The stage improvement does not survive to the coordinates

The 0.67% is real at the stage it was measured. It reverses at the output.

Two end-to-end runs of the same capture harness against the same native 5SAK
reference, same snapshot, same weights, differing only in
`BOLTZ_JAX_TRIANGLE_AMP_GRANULARITY`. Per-sample RMSD after one whole-system
Kabsch fit, no rematching:

| Arm | Per sample | Max | Mean |
| --- | --- | ---: | ---: |
| `decomposed` (shipped) | 1.897, 0.082, 0.049, 1.477, 0.636 | **1.897** | 0.828 |
| `fused` (upstream's granularity) | 1.983, 0.143, 0.040, 2.832, 0.627 | **2.832** | 1.125 |

The maximum is 49% worse and the mean 36% worse. Sample 3 moves from 1.48 Å to
2.83 Å.

So the switch stays off, and now for a measured reason rather than a cautious
one. Matching upstream's call granularity brings the first-cycle `delta_z`
closer and the final structure further away, on the same input, in the same run.

This is the third time this month that a stage-level improvement failed to
predict a coordinate one on this family of models -- after OpenFold3's TF32
operand rounding and OpenDDE's precision pin -- and the first where the
direction actually inverted rather than merely being unresolvable.

A caveat that does not change the decision: this port's rerun floor at 5SAK is
unmeasured, so 0.93 Å on one sample of five cannot be separated from trajectory
chaos with n=1. What can be said is that there is no evidence the fused
granularity improves the output, which is all the default needs.

## The reference was wrong: native's own module reproduces its stored output no better than the port

The paradox above -- source matches, output differs -- had a third reading that
this investigation never tested: that the comparison itself was mismeasured.

Upstream's `MSAModule` was built standalone in the upstream torch environment,
loaded with the checkpoint's 226 `msa_module.*` tensors (0 missing, 0
unexpected), and run under bfloat16 autocast with `use_kernels=True` on the same
inputs every arm above used -- native's own `input_z`, the captured embedding,
and the six bitwise-identical features.

| Run | RMSE against the *stored* native `delta_z` |
| --- | ---: |
| **Native module, standalone** | **3.217599e-02** |
| Port's module, standalone (shipped configuration) | 3.233621e-02 |

Native's own module, given native's own inputs, misses native's own stored
output by the same margin the port does. **The 3.2e-2 is not a JAX-versus-Torch
difference.** It is the difference between running this module standalone and
running it inside the full model, and both implementations show it.

### What this invalidates

Every teacher-forced arm in this document measured against a reference that
carries that offset. The comparisons *between* arms remain valid -- they shared
the reference and the controls moved -- so the exclusions of dtype, boundary
placement, chunk width and kernel choice still stand as statements about what
changes the port's output. What does not stand is the framing: those arms were
never measuring the port's distance from native.

The 0.67% "improvement" from upstream's call granularity, and its 49% coordinate
degradation, are unaffected -- that pair was measured end to end.

### What it opens

The port's MSA module may be correct. Its standalone output sits 0.5% *closer*
to the stored native value than native's own standalone run does, which is not
evidence of superiority but is evidence that the residual localised here was an
artefact of the harness rather than a port defect.

The next attempt should compare the two standalone runs **against each other**,
not against the stored in-run value. That comparison has never been made and is
now one line: both arrays exist.

## Closing measurement, 2026-09-09: the residual is one bf16 rounding

The comparison this investigation had never made -- the two *standalone* runs
against each other rather than each against the stored in-run value -- now
exists. All three pairwise distances on the MSA module's `delta_z` (shape
`(1, 437, 437, 128)`, rms `1.561856e+01`):

| Pair | rmse | relative |
| --- | ---: | ---: |
| native standalone vs stored in-run | 3.217599e-02 | 2.060e-03 |
| port standalone vs stored in-run | 3.233621e-02 | 2.070e-03 |
| **port standalone vs native standalone** | **3.157311e-02** | **2.022e-03** |

Max elementwise difference between the two standalone runs: `2.035156e+00`.

The three points are mutually equidistant. That is not the signature of one
side being wrong; it is the signature of a common floor. The floor is
identified by rounding the native output through bf16 once and back:

```
one bf16 round-trip of the native output: rel 1.788e-03
bf16 half-ulp                           : 1.953e-03
```

Every one of the three distances is the size of a single bf16 rounding of the
tensor being compared. Two differently-ordered bf16 evaluations of this module
-- different framework, different kernel, different chunking, or the same code
inside versus outside the full graph -- cannot land closer together than this.

MSA subsampling is off in this configuration (`"subsample_msa": false`, and the
port's only RNG use is gated behind that flag), so run-to-run randomness is
excluded as the cause.

### What this closes

The framing that opened this investigation -- "the port's MSA module is
3.2e-2 from native, find the arithmetic that explains it" -- was ill-posed.
There is no such arithmetic to find: 3.2e-2 *is* the bf16 resolution of the
comparison. This explains, in one mechanism, why thirty controlled arms all
returned 3.21e-2 to 3.23e-2 and only the deliberate x2 tripwire and the
`8.536005e-01` positive control ever moved. Those arms were not
underpowered relative to each other -- the tripwire and control prove the
harness discriminates -- but they were all measuring inside the floor.

### What it does not close

The shipped Boltz-2 / 5SAK cell is still `1.1761` A, and it is still the only
failed cell in the six-model panel. What changes is where the remaining cause
can live. It is not a defect in the MSA module's arithmetic. It is that the
trunk runs in bf16 at all: a module-level difference at the bf16 floor, fed
through this target's diffusion trajectory, arrives as an angstrom. The
already-measured arms are consistent with that -- raising trunk precision was
measured and was worse (tf32 20% worse) or bitwise inert, because those arms
changed matmul accumulation, not the bf16 activations that set this floor.

The next honest experiment is therefore not another module-level arm. It is
whether an fp32 *activation* trunk (not fp32 accumulation) closes the 1.1761 A
cell, measured end-to-end against native with the 0.0032 A rerun floor as the
control. That is a large arm, upstream does not run it, and it would not ship;
it would establish the cause, not the fix.

## The fp32-activation arm, measured: 8.5x worse

The previous section named an experiment and did not run it. It has now been
run. Source snapshot `boltz-e2e-fp32act-20260909`, identical to the shipped
`boltz-e2e-source-20260909` except that `_amp_dtype` in
`models/boltz2/models/trunk_blocks/msa.py` returns `None` unconditionally, so
the MSA module's activations stay fp32 while the weights are untouched. Same
harness, same weights, same reference, same queue.

Whole-system Kabsch RMSD against the native capture, five samples:

| Arm | max | median | per sample |
| --- | ---: | ---: | --- |
| shipped (bf16 activations) | 1.8971 | 0.6362 | 1.8971, 0.0815, 0.0491, 1.4773, 0.6362 |
| shipped, rerun control | 1.8970 | 0.6377 | 1.8970, 0.0815, 0.0491, 1.4767, 0.6377 |
| **fp32 MSA activations** | **16.1595** | **1.3322** | 16.1595, 0.2690, 0.0952, 5.9294, 1.3322 |

The rerun control reproduces the shipped arm to 0.0001-0.0006 A, so the
comparison resolves far below the effect. The arm fired -- an 8.5x move is not
a dead patch.

Raising the port's activation precision above native's makes agreement with
native **worse by 8.5x**. That is the correct sign, and it is worth stating as
a rule rather than a surprise: under a matched-tape contract the target is
native's arithmetic, not more accurate arithmetic. Native runs this module in
bf16 autocast. A port that rounds where native rounds tracks it; a port that
declines to round diverges, and on this target the divergence compounds through
the diffusion trajectory into sixteen angstroms.

### Status of the Boltz-2 / 5SAK cell

Closed as far as this investigation can take it. The residual is the bf16
floor established above, the port already sits on that floor, and the one
remaining named lever has now been measured and moves the wrong way by an
order of magnitude. Reducing this cell further would require native itself to
run in higher precision, which is a change to upstream, not to the port.


## The whole trunk in bf16 units, 2026-09-09

The closure above was argued from one boundary, the MSA module's `delta_z`.
The capture holds a second boundary that this investigation never read, and
putting both in units of one bf16 rounding settles where the residual lives.

| Cycle | Boundary | port-native rel | one bf16 rounding | ratio |
| --- | --- | ---: | ---: | ---: |
| 00 | `msa_module.delta_z` | 2.073e-03 | 1.788e-03 | 1.16x |
| 00 | `pairformer_module.output_z` | 3.295e-03 | 1.664e-03 | 1.98x |
| 00 | `pairformer_module.output_s` | 8.608e-04 | 1.624e-03 | **0.53x** |
| 03 | `msa_module.delta_z` | 2.035e-03 | 1.684e-03 | 1.21x |
| 03 | `pairformer_module.output_z` | 3.050e-03 | 1.656e-03 | 1.84x |
| 03 | `pairformer_module.output_s` | 8.840e-04 | 1.600e-03 | **0.55x** |

Every boundary sits between half a rounding and two roundings. The single-token
pathway is *below* one bf16 rounding -- the port tracks native there more
closely than bf16 can represent a difference.

### This corrects the localization

Earlier sections here localize the residual to the MSA module. In bf16 units
that is not what the numbers say: the MSA module is at 1.2 roundings and the
pairformer at 2. The pairformer carries slightly more, and neither is anomalous.

The correction does not change the verdict, it strengthens it. There is no
module to blame. The trunk accumulates between one and two bf16 roundings across
its two large blocks, which is what a bf16 trunk does, and 5SAK's diffusion
trajectory turns that into 1.18 A.

The pairformer amplifies rather than originates: it takes `input_z` at
1.516e-03 to `output_z` at 3.295e-03, roughly doubling, and takes `input_s` at
2.153e-04 to `output_s` at 8.608e-04. Both outputs land where a deep bf16
residual stack puts them.

## Per-entity check: the 5SAK cell is not a placement artifact

The Protenix work on 2026-09-09 found that a whole-system Kabsch fit on a
multi-chain target can be dominated by inter-chain placement rather than fold
accuracy -- there, chains folded identically while the complex assembled
differently, turning 3.4 A into 26 A. That is a confound this cell had never
been checked for.

5SAK has two entities: a 3086-atom protein (`asym_id` 0, `mol_type` 0) and an
18-atom ligand (`asym_id` 1, `mol_type` 3). Fitting each separately, maximum
across five samples:

| Arm | whole-system | protein (3086) | ligand (18) |
| --- | ---: | ---: | ---: |
| shipped | 1.8971 | **1.9025** | 0.0156 |
| rerun control | 1.8970 | **1.9025** | 0.0155 |

The answer is the opposite of Protenix's. The protein chain carries the entire
error at 1.90 A, and the ligand is placed to 0.016 A -- two orders of magnitude
better. There is no placement term hiding in this number.

That removes the confound and strengthens the closure rather than weakening it.
Accumulated bf16 rounding driven through a long diffusion trajectory should
express itself as a diffuse difference across the fold, which is what 1.90 A
spread over 3086 protein atoms with a correctly placed ligand looks like. A
placement artifact would have looked like the reverse.

It also scopes the Protenix caution correctly: read multi-entity targets per
entity before attributing, and expect the answer to differ by target.

---

# Scope correction: this investigation measured the bf16 path, the failing cell is FP32

Everything above concludes that the 5SAK residual is the bf16 rounding floor,
and applies that conclusion to the standard's failing cell of 1.1761 A. Reading
the source artifact's contract rather than the inherited table, those are two
different measurements.

`COMBINED_DIAGNOSTICS.md` states the matched-tape contract:

> Matched-tape runs also use n=5 and each model's released 200-step and 3- or
> 10-recycle schedule. **Both frameworks are forced to FP32** with identical
> feature tensors, captured upstream noise, and identity augmentation, so these
> isolate model-core parity rather than released mixed-precision timing.

So the 1.1761 A cell is an FP32-versus-FP32 comparison. Every arm in this
document ran the shipped bf16 configuration -- the kernels were cast to bfloat16
before the standalone calls, and the end-to-end arms scored 1.8971 A, not
1.1761 A. The bf16 floor is a true statement about the shipped path and says
nothing about the cell that fails.

## What the artifact already says about that cell

| replay variant | max raw RMSD Å | max Kabsch RMSD Å |
| --- | ---: | ---: |
| full FoldJAX core (scan) | 1.239340 | 1.176058 |
| full FoldJAX core (no scan) | 1.240488 | 1.175220 |
| upstream trunk + FoldJAX sampler | 0.000820 | 0.000766 |

with trunk correlations s = 0.999999994236, z = 0.999999872654 and RMSE 0.006470
and 0.013546.

## Why this reopens the cell

Under FP32 those RMSEs are not rounding. Against a z whose RMS is about 15.6,
0.013546 is roughly `8.7e-4` relative -- four orders of magnitude above FP32's
`~1e-7`. Whatever separates the two trunks at FP32, it is an arithmetic or
ordering difference large enough to see, not a precision floor.

The artifact's own wording, "small cross-framework trunk rounding differences
are amplified by this target's long diffusion trajectories", is right that the
amplification is real and that the sampler and scan are excluded. It does not
establish that the trunk difference is *rounding*, and at FP32 the magnitude
argues it is not.

## What this changes for the goal

The one failing cell in the panel is not closed. This document closed the wrong
thing: it closed the bf16 path, thoroughly and with controls, while the cell
that fails runs FP32. The correct next step is the boundary decomposition
already done here -- MSA module and pairformer deltas in units of the local
rounding -- repeated on an **FP32** matched-tape replay, where a `8.7e-4`
relative difference should localise sharply rather than sit at a floor.

That is a real, unblocked experiment, and it is the first one in this
investigation aimed at the configuration the failing cell actually uses.

## The FP32 residual, sized against the right tensors

The section above estimated the reported z RMSE at "roughly 8.7e-4 relative"
using an RMS of 15.6 carried over from the bf16 boundary captures. That is the
wrong tensor. The FP32 matched arm's own trunk is on disk --
`entity-parity-20260905/fresh-n5-20260905/fp32/protein_ligand_5sak/boltz2/torch/trunk.npz`,
recorded with `precision: '32'`, `kernels: False`, 200 steps, 5 samples, seed
101 -- and it holds `s`, `z` and `s_inputs` as float32.

Sizing the artifact's reported RMSEs against it:

| quantity | RMS | reported RMSE | relative | vs one bf16 rounding |
| --- | ---: | ---: | ---: | ---: |
| `s` | 60.2373 | 0.006470 | 1.074e-04 | **0.07x** |
| `z` | 24.8386 | 0.013546 | 5.454e-04 | **0.33x** |

An fp32 round-trip of these tensors is exactly zero, as it must be.

## What that settles

The FP32 trunk residual is **below one bf16 rounding** -- 7% of it on the single
representation and 33% on the pair. So the two arms are not separated by
anything bf16-scale, and the bf16 floor argument genuinely cannot reach this
cell, in either direction.

It is also three to four orders of magnitude above FP32's own resolution of
about `1e-7`. There is a real, well-conditioned difference here, and unlike the
bf16 path there is no floor for it to hide under. A boundary decomposition at
FP32 should localise it rather than return the same number at every boundary,
which is exactly what the bf16 decomposition did.

## What is still missing to run it

The FP32 native trunk exists. The FP32 port arm on disk
(`jax-patched-cueq/`) stores coordinates and metrics only, no trunk tensors, and
`bench/boltz_foldjax_capture.py` has no precision flag -- the port's
`compute_dtype` is `bfloat16` in every capture checked, including the one whose
directory is named `fp32-ffi` (that fp32 refers to the FFI linear kernel).

So the experiment needs a port source snapshot at `compute_dtype=float32`
replaying this tape with `--trunk-only` boundary capture. That is a bounded
change against an existing native reference, not a research problem.

## Both sides at FP32 without kernels: the cell essentially closes

`entity-parity-20260905/fresh-n5-20260905` holds three boltz2 arms on 5SAK that
this investigation never opened. All use the same harness, the same case, and
the scope "independent preprocessing, actual reference and sampler draws; five
paired samples" -- so the sampler draws are shared between the arms, and this is
not a free-running comparison.

| arm | precision | kernels | global max RMSD | protein | ligand |
| --- | --- | --- | ---: | ---: | ---: |
| `fp32` | 32 | False | 0.029911 | 0.029999 | 0.001267 |
| `fp32-repeat` | 32 | False | **0.002405** | 0.002412 | 0.000497 |
| `native-default` | bf16-mixed | True | **16.193149** | 16.209293 | 13.169447 |

`native-default`'s `status.json` records `replay: 1`; the log says why, and it is
a verdict rather than a crash:

```
PARITY FAILED:
  all-atom RMSD 18.9089 A > 0.5 A
```

The two FP32 arms differ from each other by an order of magnitude (0.0024 vs
0.0299), which is this comparison's own rerun spread, so FP32 agreement is
"somewhere between 0.002 and 0.03 A" rather than a single number.

## What this says

With both sides at FP32 and kernels off, the port and native track each other to
about a hundredth of an angstrom on the target that fails. With both sides in
the shipped configuration -- bf16 mixed precision with fused kernels -- they
diverge by 16 A. That is a factor of roughly 500 to 5000, and it is not ensemble
width, because the sampler draws are shared and the FP32 arms prove the harness
can resolve a hundredth of an angstrom.

So 5SAK's divergence lives in the shipped **bf16-plus-kernels** configuration,
not in the model core. That is consistent with the matched-tape row, which forces
FP32 on both sides and reports 1.1761 A rather than 16 A, and with the trunk
substitution collapsing to 0.0008 A.

## How this squares with the fp32-activation arm

Earlier here, raising the MSA module's activations to FP32 made agreement 8.5x
*worse* (1.8971 -> 16.1595 A). That is not in tension with this. That arm raised
precision on **one** side while native stayed in bf16 autocast; these arms move
**both** sides together. Matching native's arithmetic is what helps; diverging
from it hurts, in whichever direction.

## What is now worth separating

`fp32` and `native-default` differ in two variables at once, precision and
kernels. The harness ran `kernels=False` with FP32 and `kernels=True` with
bf16, so nothing here says which one carries the 16 A. Two more arms --
FP32-with-kernels and bf16-without-kernels -- would separate them, and they are
the same shape of run as the three that already exist.

## The 2x2: precision carries the failing cell, kernels do not

The previous section noted that the existing arms move precision and fused
kernels together, so neither could be attributed. Four arms with the two
separated, same driver, same case, shared sampler draws, entity RMSD maximum
over five samples:

| native precision | native kernels | global max | protein | ligand |
| --- | --- | ---: | ---: | ---: |
| 32 | off | **0.001305** | 0.001308 | 0.000142 |
| 32 | **on** | **0.002837** | 0.002846 | 0.000171 |
| **bf16-mixed** | off | **3.596264** | 3.606615 | 0.088017 |
| bf16-mixed | on | **16.043514** | 16.085801 | |

Turning the fused kernels **on** at FP32 moves the disagreement from 0.0013 to
0.0028 A. That is the entire kernel term, and it is nothing. Switching to bf16
with kernels **off** moves it to 3.60 A -- a factor of 2,750.

So the failing cell is a **precision** effect. The fused kernels contribute
essentially zero at FP32; what they do is amplify what bf16 has already
introduced, which is how bf16-off-kernels at 3.60 A becomes bf16-with-kernels at
16.19 A in the pre-existing arm.

### Why the FP32 arms are so much tighter than anything measured before

0.0013 A is three orders below the matched-tape cell's 1.1761 A and below this
document's own bf16 arms. With both sides at FP32 the two implementations agree
to about a thousandth of an angstrom on the target that fails every other way.
There is no residual to explain at FP32 -- the trunk RMSEs the artifact reports
for the matched cell (s 1.074e-04, z 5.454e-04 relative) evidently do not
survive into coordinates at this configuration.

### What this settles and what it leaves

Settled: the shipped bf16 path is where 5SAK's divergence comes from, the fused
kernels are not the cause, and the port is not carrying an algorithmic error --
at matched precision it tracks native to 1e-3 A.

Left open: the matched-tape row reports 1.1761 A under a contract that also
forces FP32 on both sides, and these arms give 0.0013 A at FP32. Those two
numbers disagree by three orders and the difference is not precision. The
remaining variables between them are the tape (captured upstream noise and
identity augmentation versus this driver's actual reference draws) and the
cueq triangle patch this driver applies. Naming them is not measuring them.

### The fourth cell landed, and it is the control

`bf16-mixed` with kernels on gives **16.043514 A** against the pre-existing
artifact's 16.193149 A for the same configuration. The two agree to 1%, so this
driver reproduces the arm it was derived from and the three new cells can be
read.

The factorial, complete:

| | kernels off | kernels on | kernels cost |
| --- | ---: | ---: | ---: |
| **FP32** | 0.001305 | 0.002837 | 2.2x |
| **bf16-mixed** | 3.596264 | 16.043514 | 4.5x |
| **precision cost** | **2,750x** | **5,650x** | |

Precision dominates by three orders in both columns. The kernels cost a factor
of two at FP32 and four and a half at bf16 -- they amplify what bf16 introduces
rather than introducing anything themselves.

## The cueq patch is excluded too

The previous section named two remaining variables between this driver's FP32
arms and the matched-tape row: the tape, and the cueq triangle-multiplication
patch this driver applies. The second is now measured.

| arm | trimul backend | global max | protein |
| --- | --- | ---: | ---: |
| FP32, cueq patch | cueq | 0.001305 | 0.001308 |
| FP32, no patch | xla | **0.005547** | 0.005563 |

Dropping the patch costs a factor of four and leaves the result at 0.0055 A --
still three orders below the matched-tape row's 1.1761 A. The patch is not the
explanation.

## Where the failing cell now stands

Excluded, each by measurement rather than argument:

| candidate | evidence |
| --- | --- |
| the sampler | upstream trunk + FoldJAX sampler collapses to 0.000766 |
| the scan lowering | no-scan control reproduces the outlier pattern |
| the fused kernels | 2.2x at FP32; they amplify, they do not originate |
| the cueq patch | 0.0013 -> 0.0055, three orders short |
| an algorithmic port error | at matched FP32 the port tracks native to 0.0013 A |

What remains named and unmeasured is the tape itself: the matched-tape contract
injects captured upstream noise with identity augmentation, while these arms use
each side's actual reference draws. Those two constructions differ by three
orders on this target, and nothing else in the list survives.

That is a statement about the harness, not the port, and it should be treated as
a hypothesis until someone runs the matched-tape construction and this driver's
construction against the *same* native capture. This document has been wrong
often enough today that naming the last suspect is not the same as convicting
it.

## The last difference, named to a line

Two candidates were left. Both are now resolved.

**Code version is excluded.** This driver's arm records
`foldjax_commit 3312e1da3df8`; the pre-existing FP32 arm from 2026-09-05 records
`83eb4eb19adc`, with two commits touching `models/boltz2` between them. Both give
the same order: 0.001305 A today, 0.002405-0.029911 A then. The port has not
changed materially, and neither reproduces 1.1761 A.

**Augmentation is the remaining difference, and it is not identity here.** The
`COMBINED_DIAGNOSTICS` contract states matched-tape runs use "captured upstream
noise, and identity augmentation". This driver does something else:

```python
def augment(*args, **kwargs):
    with patch.object(torch, "randn", ...), patch.object(torch, "randn_like", ...):
        return original(*args, **kwargs)
```

It calls upstream's real `center_random_augmentation` and records the draws, then
replays them. So the two harnesses differ in exactly this: one runs the actual
augmentation with its draws matched across both sides, the other substitutes
identity.

## The state of the failing cell

Two independent harnesses, four days apart, on different commits, both with both
sides at FP32, agree that the port tracks native on 5SAK to between 0.001 and
0.03 A. The panel's `matched RMSD` column reports 1.1761 A for the same target
and the same nominal precision.

Everything that could explain a 400-900x gap between them has been measured and
excluded except the augmentation handling. That is a difference between two
measurement constructions, not a property of the port.

I am not going to call it the cause. The mechanism by which identity
augmentation would cost three orders is not obvious, this document has been
wrong six times today, and the experiment that would settle it -- the same
native capture scored both ways -- has not been run. What is established is
narrower and still worth having: **on this target, at matched precision, the
Boltz-2 port is not carrying an angstrom-scale error, and the panel's one
failing cell is not reproduced by the harness that replays the actual
augmentation draws.**

## The identity-augmentation arm cannot be made by flipping one flag

`center_random_augmentation` takes `augmentation=True|False`, so forcing it False
looked like a one-line way to reproduce the contract's "identity augmentation"
and settle whether it explains the 400-900x gap. It is not.

The native capture succeeds; the port's replay dies:

```
ValueError: preprocessing replay exhausted
RuntimeError: featurizing boltz2_input failed
```

The replay consumes the preprocessing draws the capture recorded. Skipping the
augmentation on the native side removes draws the port still expects, so the two
sides desynchronise. The tape contract couples them: an identity-augmentation
arm needs the consumption side changed to match, not just the production side.

That is itself informative about the two harnesses. The panel's matched-tape
construction is not this driver with a flag flipped -- it handles identity
augmentation consistently on both sides, which is a different tape contract, not
a different setting.

## Final state of this investigation

Established by measurement:

- The Boltz-2 port carries no angstrom-scale error on 5SAK. With both sides at
  FP32 it tracks native to 0.0013 A (0.0055 A without the cueq patch), confirmed
  independently by a 2026-09-05 arm on a different commit at 0.0024-0.0299 A.
- The shipped divergence is a precision effect: 2,750x from bf16 with kernels
  off, 5,650x with them on. The fused kernels cost 2.2x at FP32 and amplify
  rather than originate.
- The sampler, the scan lowering, the cueq patch, the code version, and an
  algorithmic port error are each excluded by their own arm.

Not established, and now known to need more than a flag: whether the panel's
1.1761 A comes from its identity-augmentation tape contract. That experiment
requires matching changes on the replay side.

This document opened by attributing the cell to the MSA module, then to a bf16
floor, and both were measurements of the shipped path rather than the failing
one. What survives is narrower and better supported: at matched precision there
is nothing wrong with this port on this target.

## The reducible error, located

One reading of the factorial is "bf16 is lossy, nothing to do". That reading is
wrong, and the table says why.

In the bf16 arms **both sides are bf16**. Upstream runs `bf16-mixed` autocast and
the port runs `compute_dtype=bfloat16`. They still disagree by 3.596264 A with
kernels off. Two bf16 implementations of the same model, given the same tape,
diverging by three and a half angstroms is not the precision's fault -- it is the
two implementations rounding in different places.

The FP32 arms prove there is nothing else wrong: strip the differing rounding and
the same two implementations agree to 0.001305 A.

So the reducible error is **where the port places its bf16 casts relative to
torch's autocast**, and it is worth the whole 3.6 A on this target. Torch's
autocast keeps matmuls in bf16 and promotes reductions, normalisations and
softmax to FP32 by op-level policy; a port that casts activations at module
boundaries instead will round in a different set of places even with the same
nominal dtype.

This is the shipped configuration, so unlike the FP32 arms it is a defect a user
actually meets. It is also consistent with the earlier note recorded for this
target, that the residual survived every knob and what remained was autocast
boundary placement.

### The next arm, stated concretely

Enumerate torch's autocast policy for the ops this trunk uses -- which stay bf16,
which promote -- and compare against the port's cast sites in
`models/boltz2/models/trunk_blocks/`. The boundary decomposition already in this
document gives the per-module residual in units of one bf16 rounding, so a
corrected cast placement should show up there before it shows up in coordinates.

That is a source-reading task with a measurement attached, not another sweep.

## The pairformer attention core is not the cast-placement site

The section above located a concrete divergence and predicted it carried the
3.6 A. Upstream's `attentionv2.py` runs the whole attention block with autocast
explicitly disabled:

```python
with torch.autocast("cuda", enabled=False):
    attn = torch.einsum("bihd,bjhd->bhij", q.float(), k.float())
    attn = attn / (self.head_dim**0.5) + bias.float()
    attn = attn + (1 - mask[:, None, None].float()) * -self.inf
    attn = attn.softmax(dim=-1)
    o = torch.einsum("bhij,bjhd->bihd", attn, v.float()).to(v.dtype)
```

while the port's `_jax_attention` (`primitives/micro_modules.py`), reached from
`_jax_pairformer_layer`, ran all of it at the incoming dtype. Under
`compute_dtype=bfloat16` that is a bf16 softmax where upstream has an FP32 one.

Promoting the port's block to match, one variable, `bf16-nokernels`:

| arm | global max | protein |
| --- | ---: | ---: |
| bf16 baseline | 3.596264 | 3.606615 |
| bf16 + FP32 attention core | **3.596316** | 3.606667 |

A change of 5e-05 A. The divergence is real and the fix is correct on its own
terms -- the port now rounds where upstream rounds at this site -- but it is not
where the 3.6 A lives. Note also that this driver's own rerun spread is an order
of magnitude larger than the move (0.0024 vs 0.0299 on the two FP32 arms), so
5e-05 is not even resolvable here.

## Standing

The cast-placement hypothesis survives; this particular site does not carry it.
The trunk has other bf16 boundaries, and the per-module decomposition recorded
earlier in this document is the tool for finding which one -- but it was taken on
the bf16 path against a bf16 native, so it already contains whatever this
mismatch is, distributed across the modules at roughly one bf16 rounding each.

What that means practically: the next attempt should not guess a site. It should
diff the port's realised dtypes against a torch autocast trace of the same trunk,
op by op, and only then patch. That is a different kind of work from the arms in
this document and is where I would start.
