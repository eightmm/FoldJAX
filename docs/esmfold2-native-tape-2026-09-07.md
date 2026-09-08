# ESMFold2 native tape implementation progress

This is a partial implementation record, not full-model or performance admission.
The pinned model source is `ef32577f55da19a4989cd7b22e004dc43a4998cb`.
Native CUDA input/trunk execution uses BF16 autocast; FP32 checkpoint storage
does not mean the trunk executes in FP32. The comparison harness now separates
weight dtype, observed trunk input dtype and autocast policy. The tape replay's
old FP32 trunk override was removed; remaining missing routes still fail closed.

## LM dropout

The core now accepts `lm_dropout_masks` as boolean keep decisions with shape
`[loops, batch, tokens, tokens, pair_width]`. Shape/dtype errors and inactive
LM/dropout branches are rejected. The ordinary random draws and key splits are
preserved. This tape alone does not control MSA or diffusion randomness.

Inspection and numerical controls found that BF16/FP16 dropout scaling must
use native CUDA FP32 opmath before final storage. Dividing the low-precision
input by a low-precision keep probability gave different results. Both ordinary
and taped low-precision dropout now use FP32 scale multiplication, while the
existing FP32 path is unchanged.

The installed native Torch build is `cf30153c4c131c8164ee7798e5022d810682e2cb`
(2.13.0+cu130). Its CUDA Dropout implementation uses float accumulation for
BF16/FP16. Job 419 compared actual Torch CUDA dropout with JIT-compiled JAX CUDA
using identical 4,096-element BF16 operands and captured native keep decisions.
All four rates, 0.15/0.2/0.25/0.3, matched exactly, and native repeated draws
reproduced exactly. `JAX_PLATFORMS=cuda` prevented CPU fallback in this run.
The frozen execution snapshot is private task `esmfold2-dropout-20260907-bIBVUd`.

Parent focused CPU checks passed 22 tests with four opt-in CUDA tests skipped;
those four subsequently passed in the queued CUDA execution. A native-backed
tiny trunk/distogram test separately passed key independence with fixed tape
inside each eager/JIT mode. Eager versus JIT BF16 trunk output had a previously
observed maximum difference 0.00141144 and failed the 1e-4 diagnostic; it is not
claimed equivalent or hidden by the key-independence test.

## Remaining work

The four diffusion draw routes are now connected through `predict -> sample ->
step` as an all-or-none FP32 tape: initial noise, raw quaternions, translations
and churn noise. They retain native current/previous coordinate transformation,
`x @ R` orientation and non-zeroing of masked atom coordinates. The existing
schedule is regenerated, not replaced with a captured sigma tape; validation
uses its actual post-clipping step count. Dynamic JIT callers must perform
explicit concrete value preflight before tracing.

Parent review reproduced a validation hole: quaternion components of 1e-19
passed host norm validation but produced nonfinite JIT rotations because their
squares flushed to zero. Preflight now requires a normal-square component and
overflow margin without changing native forward arithmetic. A regression failed
before the fix. Parent combined diffusion/dropout/tape/resume checks passed
167 tests with four opt-in CUDA skips; a further near-overflow guard test was
added afterward. These are toy/CPU consumer checks, not full native replay.

The two MSA tape routes are now connected: native column keep decisions and
per-loop query-preserving sorted row selections. Shapes/dtypes, range,
uniqueness, inactive consumers and conflicting preselected loop features are
validated. The ordinary RNG split order is retained. Parent combined MMA,
ESMFold2 MSA/diffusion/dropout and resume checks passed **236 tests, 4 skipped**.
The worker's real tiny native-backed MSA/trunk checks passed within eager and
JIT separately; neither these nor the earlier dropout CUDA panel constitutes
full native structure/confidence replay. Parent review found a benchmark
contract inconsistency: classification dropped inactive empty MSA arrays while
replay required all eight fields. That harness repair is in progress before
the first complete replay. That harness repair is now implemented and tested;
confidence archives and source/checkpoint/input/tape identity are bound too.

## First full core replay, jobs 426/427

5SAK native capture completed with 437 tokens, 3,104 atoms, full MSA depth
4,449, five samples, four trunk loops, and the released clipped ten-step
diffusion schedule. All eight actual random-consumer arrays were captured,
including four row selections of size 1,024. Native raw confidence fields are
retained. This is the pinned runtime's pure-Torch ESMC fallback environment
(Transformer Engine/xformers/flash-attn absent), not proof for those alternate
fused backends. The instrumented run is not performance evidence.

The JAX replay completed on the same features/checkpoint/tape. One global
proper Kabsch fit per sample, followed by entity measurement without refit:

| Entity | Sample 0 | Sample 1 | Sample 2 | Sample 3 | Sample 4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Protein RMSD (Angstrom) | 1.716864 | 0.117173 | 0.134715 | 0.183399 | 0.174210 |
| Ligand RMSD (Angstrom) | 0.558091 | 0.029551 | 0.030242 | 0.051662 | 0.027713 |

Structure and strict confidence fail. Maximum differences include complex
pLDDT 0.00058746, pTM 0.001288 and ipTM 0.00321788. The initial JAX harness
also inherited `return_confidence_logits=False`, so six native fields are not
retained in job 427. Missing fields remain an incomplete raw-confidence gate,
not a pass. The harness now explicitly retains these outputs without changing
sampling/dtype settings; a fresh replay remains necessary. The report reuses
`bench.entity_parity` and records shared-feature equality only, not independent
preprocessing or arbitrary atom-order correctness. Private evidence:
`esmfold2-native-full-tape-20260907-5sak-a`,
`esmfold2-jax-full-tape-20260907-5sak-a-report-v1.json`.

The next counterfactual separates independently computed ESMC states from
downstream structure inference by exchanging the exact native LM output.
No full-model admission or main push is claimed.

## Independent implementation review follow-up

A separate Claude-family scoped review found a real intermediate contract gap
when native tapes were combined with prefix-preserving padding, and missing
Torch-free public wiring coverage. Native tapes now reject that unvalidated
combination instead of bypassing padding semantics. Initial pair-state
overrides also require exact floating shape and finite concrete values; the
existing floating dtype override is preserved. The public docstring now
requires concrete initial/MSA/diffusion preflight before dynamic JIT and
distinguishes partial diagnostic overrides from complete native replay.

New Torch-free tests retain actual predict, loop/dropout and sampler consumers
while stubbing expensive network boundaries. They check all eight inputs,
MSA query and row/deletion semantics, batch-times-samples and schedule length,
eager/JIT forwarding, conflicts and malformed initial states. Parent combined
consumer/preflight/resume verification passed **205 tests, 4 CUDA skipped**.
These validation repairs postdate the immutable 426/427/430–432 sources and
are not claimed as fresh GPU model evidence.

A toy eager-versus-scan full-denoiser comparison differed by 4.20e-6 in one
coordinate, exceeding the trial 3e-6 absolute tolerance. That difference was
not waived as model parity. The driver regression instead observes actual
step/tape consumption with a deterministic arithmetic stub and checks exact
indexing; full numerical comparisons remain separate.
Independent input identity, full
n=5 structures with global alignment/entity measurement, raw/public confidence,
ordinary RNG and uninstrumented performance remain separate requirements.
No full ESMFold2 closure or precision-lowering admission is claimed.

### Native repeat and autocast boundary controls, jobs 430–440

With the same native LM observer and tape, ordinary native repeats differed
by up to 0.0204816 A for protein and 0.00762027 A for ligand after the fixed
global fit. Several confidence leaves still failed the strict gate. An earlier
observer-changing comparison is confounded and does not establish a larger
native allowance.

Jobs 436/440 enabled deterministic algorithms with a pre-CUDA cuBLAS workspace
configuration. The two runs have identical tape and LM archive hashes, and all
five coordinate arrays and all 14 retained confidence leaves are byte-equal.
The fitted RMSD is about 1e-14 A (numerical Kabsch roundoff on equal coordinates).
This establishes repeatability for this finite native deterministic control,
not the identity of the individual nondeterministic operator or universal
repeatability. Ordinary native runs remain a separate arm.

Same-native-LM downstream JAX replay still differed (protein maximum
0.434752476 A; ligand 0.200597571 A). Independent JAX LM replay against the
same reference differed more (protein 1.133402377 A; ligand 0.767714012 A).
Both retained all raw heads and failed strict confidence. Exact native LM
interchange is a downstream-only diagnostic, not full model admission.

CUDA job 435 directly confirmed that native BF16 autocast returns FP32 from
LayerNorm even for BF16 inputs. Job 439 captured the actual real-weight LM
shim boundaries: FP32 hidden states -> FP32 LayerNorm -> BF16 Linear;
BF16 SingleToPair -> FP32 final LayerNorm. The combine parameter and its
softmax also remain FP32 before the autocast matmul. Premature hidden-state
and affine-parameter rounding in the port therefore needs correction, not
an increased acceptance tolerance. ESMC and shim corrections are being
validated separately from remaining trunk arithmetic.

Verification: native deterministic repeat artifacts and all saved confidence
leaves checked; current resume regression suite 135 passed, including source
dependency invalidation for ESMC and inference autocast repairs. No current
full-model closure, performance admission, or push is claimed.

### Native-autocast shim implementation and reduction control, jobs 441–443

The actual runtime now keeps original FP32 normalization/softmax parameters,
narrows at BF16 Linear boundaries, and returns the native FP32 final shim
pair. Direct helper and compact-cache routes share this policy; existing
direct FP32 behavior remains covered. Parent affected checks passed 188 tests.

Same-native-LM full-shim comparison still fails: 48,888,064 pair values have
maximum absolute error 0.13532209396362305 and RMSE 0.01030694825594233.
These are internal features, not coordinates or angstroms. A separate prefix
probe shows LayerNorm passing the strict tolerance (maximum 9.536743e-7),
but Linear differing on 632/2304 saved entries, maximum 0.125. The normalized
inputs are bitwise identical after BF16 casting on all 23,040 sampled values.

Job 443 changes only native
`allow_bf16_reduced_precision_reduction=False`, retaining full operand shape,
weights and BF16 autocast. All 2304 saved Linear entries then match the JAX
candidate bitwise. This controlled intervention identifies reduced-precision
GEMM accumulation as a contributor at this boundary. It does not authorize
changing native defaults for admission. The final pair still differs in that
control (maximum 0.11392736434936523, RMSE 0.009447586796224441), so this is
not a sole-cause or full-shim closure claim. Further pair-stage and downstream
encoder boundaries remain under investigation.

Private evidence families: `esmfold2-shim-fixed-20260908-2heJ5g`,
`esmfold2-shim-prefix-20260908-wp4JwQ`, and
`esmfold2-shim-reduction-20260908-ETuM0R`. Original failing arms are retained.
The prefix probe is a separate JIT of the same helpers with small output
slices, not a full intermediate trace. No performance result is inferred.

The following runtime repair restores four explicit native loop boundaries:
LM encoder entry, LM addition without an encoder, MSA output injection, and
refined LM injection. FP32 LM dropout still occurs before narrowing to the
receiving pair dtype. This is independently required by pinned native source
and is not a workaround for GEMM reduction differences. The expanded parent
CPU gate passed 206 tests, including actual eager/JIT loop wiring, the compact
LM cache, ESMC autocast, preflight and resume invalidation. A separate tape/
report gate passed 75 tests with four opt-in CUDA tests skipped.

Jobs 444/445 use the corrected immutable runtime against the deterministic
native capture, separately for exact native-LM interchange and independent
JAX LM computation. These are full n=5 core diagnostics on shared native
features, not independent preprocessing or uninstrumented performance tests.
Parent also ran the pinned native CPU model/ESMC parity suites after the loop
repair: 10 passed, with the expected warning that CUDA autocast is disabled
on CPU. These checks preserve the existing FP32 contract and are not CUDA
mixed-precision evidence.

Jobs 444/445 completed with full raw heads retained. Against the same native
deterministic reference, per-sample whole-system-fit entity RMSDs are:

| Arm | Protein (A), samples 0–4 | Ligand (A), samples 0–4 |
| --- | --- | --- |
| Exact native LM | 0.296008, 0.101544, 0.096086, 0.266999, 0.224175 | 0.081990, 0.049838, 0.032003, 0.070172, 0.085173 |
| Independent JAX LM | 1.479398, 0.154503, 0.305932, 0.242708, 0.135136 | 0.419853, 0.040798, 0.052194, 0.058532, 0.042217 |

Both fail overall strict confidence. Independent LM now has the correct FP32
output dtype but still differs: maximum 128, RMSE 0.534269325 over 90,616,320
internal values. This is not optional precision-lowering admission.

Job 451 replays the pre-autocast runtime against the same reference. Its
recorded source, runner, checkpoint, input, tape, LM, config and versions match
earlier job 431 (only the reference manifest differs), yet the resulting
coordinates differ. Compiler/environment identity was not fully recorded in
those runs. Consequently a single before/after RMSD reduction is not causal
proof of the runtime repair's aggregate effect.

Jobs 454/455 instead compile once per arm and execute three forwards using
identical device operands. All coordinates and all 14 confidence leaves are
byte-equal across all three forwards in both the ordinary arm and the
scatter-determinism-expander arm. The latter is an explicit non-shipped compiler
control, not a production default. HLO identity and relevant JAX/XLA policy are
recorded. This does not establish cross-process stability or implicate scatter
as the cause; fresh-compilation/autotuning remains to be separated. Parent
repeat/provenance and projection harness tests passed 43 checks.

### Native LM encoder autocast boundaries

The actual LM encoder now retains original FP32 norm parameters and uses
FP32 norm outputs, explicit BF16 Linear operands/results, native FP32-mask
promotion, BF16 contraction storage and output-i chunk64/tails. Sigmoid and
SiLU use FP32 opmath followed by BF16 storage. This path is scoped to the
native BF16 LM encoder; other trunk and direct FP32 paths remain unchanged.

Teacher-forced first-block job 460 uses exactly the native block input and
weights from job 446. All observed producer dtypes now match, including
FP32 norm outputs and BF16 contraction outputs. Full-block error remains:
RMSE `0.5753073036059803`, maximum `32`, and 15,540,524 failing values out
of 48,888,064 at the unchanged strict gate. Instrumented and uninstrumented
candidate block outputs are bitwise equal. These are representation values,
not Angstroms. The earlier job 450 had RMSE `0.8781653038`; neither is a pass.
Observed outgoing norm-output max error is `4.76837158203125e-7`; its sampled
contraction now differs in only one of 2,304 entries, max `1.375`. Incoming
sampled contraction still differs in 60 entries, max `64`. Full-shape
contraction and downstream reduction causes remain open.

Artifact: `runtime-integration-20260908-8gdWU0/esmfold2-block`. Its producer
dtype fields, rather than FP32 NPZ storage dtype, are authoritative. Parent
focused CPU verification passed 143 tests, followed by 145 encoder/resume
checks after repairing the empty-prefix helper contract. GPU job 460 uses a
nonempty prefix and is unaffected by that later empty-prefix repair.

The subsequent fresh-process control records an autotune producer and two
strict consumers, each executing the same compiled model three times with
the same native LM and tape. Each process has a separate executable cache;
per-fusion persistent XLA caches are disabled. Consumers require complete
autotune coverage and bind the loaded artifact before/after; cache misses do
not authorize retuning. Dump hashes and output finiteness are recorded
separately from byte equality. This is diagnostic-only CLI behavior, not a
FoldJAX compiler default. The helper rejects setup after JAX import, because
environment changes alone would not update an already-read cache config.
CPU repeat/tape checks passed 32 tests; no cross-process result is claimed
until the GPU producer and both strict consumers finish.

Jobs 461/462 completed: the producer and first strict fresh-process consumer
use the identical autotune SHA256
`48eecea26b9581215fd4b251abcc9bc7442f2096ad8689b03bf329871b4db4bc`;
the consumer's dump retains that identity. Their compiled HLO hashes match,
and all coordinate and confidence arrays are finite and bitwise equal across
processes. Both also retain three equal within-process forwards. This is HLO
and output evidence, not a compiled-binary identity claim, and it does not
alone prove the cause of older runs with incomplete compiler provenance.
Job 463, the second strict fresh-process consumer, also retains the identical
loaded/dumped autotune and compiled HLO hashes. Its coordinate/confidence
arrays are finite and bitwise equal to the producer and throughout its three
within-process forwards. Thus all nine measured forwards across these three
processes agree under this frozen-kernel control; independent autotuning and
cross-hardware reproducibility remain outside this result.

Against deterministic native, the producer's full n5 downstream-core result
still fails: protein RMSDs `1.275619, 0.139349, 0.080293, 0.313935, 0.230950`
Angstrom; ligand `0.321841, 0.046332, 0.033050, 0.059359, 0.050639` Angstrom.
One global valid-system Kabsch fit per sample is used, with entity measurement
only. Native LM is byte-identical, all raw confidence heads are retained, and
overall strict confidence fails. Reproducibility is not upstream parity.
Artifact root: `esmfold2-frozen-autotune-20260908-TAAo9a`.

### Full first-contraction operands refute a GEMM-only explanation

Jobs 465–467 capture the actual first outgoing chunk's full operands and
native strides, then replay the unchanged `[1,64,437,256]` output shape.
All 7,159,808 output values match the original native block **bitwise** in
all three arms: native default BF16 reduction, native reduction disabled,
and JAX BF16 operands/FP32 preferred accumulation/BF16 output. Each arm's
second execution is also bitwise identical. Thus reduced-precision GEMM
does not explain the remaining first-chunk block discrepancy on identical
operands, unlike the separately measured wide LM-shim projection.

Artifact: `esmfold2-contraction-20260908-tPNTcK`. The next discriminating
boundary is the block's complete effective contraction inputs: earlier
projection slices can agree while unobserved reduction-axis entries differ.
No contraction-kernel rewrite or native-default change follows from this
passing same-operand result. Parent probe contract checks passed 15 tests.

Job 470 confirms that the actual candidate block does not supply those same
effective operands: full first-chunk lhs has 1,118 unequal entries (776 strict
failures, max `0.25`), and full rhs has 6,391 unequal entries (4,986 strict
failures, max `0.5`). Native pre-autocast FP32 operands are explicitly mapped
to BF16 for this comparison; producer dtype and effective dtype are separately
recorded. Instrumentation still leaves the complete candidate block output
bitwise unchanged. The next capture is the full first outgoing norm before
projection; small earlier slices were insufficient to exclude that boundary.
Artifact: `esmfold2-full-operands-20260908-8Y6WUd/candidate`. Parent focused
operand/probe checks passed 16 tests.

### Full norm and reuse of the existing CUDA implementation

Jobs 475/476 capture the first outgoing norm in the actual block. All
48,888,064 FP32 outputs pass strict tolerance (max `1.9073486328125e-6`),
but narrowing produces 866 different BF16 values, including 421 strict
failures and max `0.03125`. Passing the FP32 leaf gate alone therefore did
not establish equality of the next GEMM's effective input.

Job 480 runs the same full operands through the existing vector-4 CUDA
Welford/FMA implementation already used by Boltz. Both raw FP32 and effective
BF16 outputs are **bitwise identical to native over all 48,888,064 values**,
with a bitwise repeated execution. Its generic standalone control retains
BF16 differences; it is a separate compilation from the full-block control.
Artifact: `esmfold2-welford-control-20260908-6UIBAI/candidate`.

The ESM native-autocast norm now reuses that CUDA implementation at width256.
Other widths, CPU/TPU, context parallelism and direct non-autocast paths retain
the ESM formula. Original norm parameters remain FP32. Resume identity and
the block probe bind the shared helper's source, even though it lives under
Boltz. Parent route, resume and loop checks passed 163 tests. Actual runtime
full-block job 482 is pending; isolated norm equality is not full-model closure.

Job 482 completes: the actual runtime's first full norm matches native
bitwise in both FP32 and effective BF16. Instrumented and uninstrumented
block outputs remain bitwise equal. The full block still fails, with RMSE
`0.5564370511546407`, max `32`, and 13,509,613 strict failures among
48,888,064 values. First-norm equality therefore moves the next investigation
to the actual projection and subsequent boundaries, not model admission.
Artifact: `esmfold2-runtime-welford-20260908-SBgKmD/candidate`.
## Full first projection capture (2026-09-08 continuation)

After job 482 made the actual first norm bitwise exact, the next diagnostic
captures the first `tri_mul_out._engine.proj_bundle` input and output in full,
not only three-entry boundary slices. Native job 488 records BF16 output shape
`[1,437,437,1024]` (195,552,256 values) and a separate uninstrumented block
baseline. The instrumented and uninstrumented native block outputs are bitwise
identical. This is still a dropout-disabled, teacher-forced first LM block,
not independent LM preprocessing or full-model admission.

The candidate capture validates that the full FP32 projection input is the
captured norm output, preserves the original producer dtypes, and requires
lossless BF16 storage for the projection output. Current validation also
rejects a native capture whose instrumentation changed the block; 30 focused
capture/schema tests pass. Job 489 compares the actual FoldJAX path; its
source snapshot predates only that additional fail-closed validation guard,
and the consumed native capture independently satisfies the guard.
Artifact: `esmfold2-projection-result-20260908-g0WH3I`, source snapshot
`esmfold2-projection-source-20260908-e03M46`.

Job 489 completed successfully. Both actual projection input (48,888,064
FP32 values) and output (195,552,256 BF16 values) are bitwise equal to native.
Candidate instrumentation also leaves the full block output bitwise unchanged.
The full block still fails: RMSE `0.5564370511546407`, max `32`, and
13,509,613 strict failures. Thus neither the first norm nor the first bundled
projection explains the remaining first-block discrepancy in this run. The
next unclosed boundaries are the complete outgoing result/incoming input and
the actual incoming contraction operands. The already captured three-entry
slices of every outgoing norm/projection are exact; incoming norm-start and
bundle slices are also exact, but incoming norm-mix input (the contraction
result) has maximum difference `32`. Slices cannot exclude earlier differences
outside their coverage, so the full preceding boundary must be checked before
assigning the cause to incoming GEMM arithmetic. These representation metrics are not
Angstroms, and the separate same-input first contraction proof must not be
extended to changed operands or the full block.

## Independent review and parent verification

A bounded independent review of the shared norm route identified missing ESM
resume dependencies. Parent regression tests reproduced both failures:
changing `models/esmfold2/models/primitives.py` or `models/_cp.py` incorrectly
reused a completed result. Both sources are now included in the manifest's
input dependencies. The first focused post-fix suite passed 182 tests.
Comments now limit the native norm equality claim to the captured first norm,
and the shared helper documents its ESM consumer and incomplete block parity.

The review's separate suggestion that BF16 affine storage causes low-precision
affine arithmetic was not confirmed. The actual private Pallas JAXPR converts
both affine operands to FP32 immediately before the final `fma.rn.f32`; the
non-CUDA ESM helper also explicitly promotes them. New IR regression coverage
checks FP32, FP16 and BF16 affine storage without changing the numerical
implementation. All three IR cases plus the affected norm/resume suite pass
185 tests after the repair. This does not admit a differently stored checkpoint against
the original FP32 checkpoint or constitute new GPU evidence.

CPU/CUDA result-resume identity and CP performance were raised as separate
pre-existing/broader risks; neither is resolved by these changes. Compiler
cache identity and result-resume semantics are distinct. No cross-platform,
multi-device or full-model admission follows from this bounded review.

## Full incoming boundary: earlier differences propagate

Jobs 490/491 capture the entire incoming triangle input, both effective
contraction operands and complete contraction output. Native capture preserves
the full pre-autocast FP32 operands and their interleaved strides/offsets;
comparison explicitly narrows them to the actual BF16 consumer dtype. The
candidate assembles the original 64-row chunks in native output-i order and
checks that the full RHS is unchanged between chunks. Native and candidate
instrumentation each preserve their uninstrumented block output bitwise.

Each row below compares 48,888,064 values. These are internal representation
errors, not coordinate distances or Angstroms.

| Boundary | RMSE | Max absolute | Strict failing entries |
| --- | ---: | ---: | ---: |
| Incoming block input | 0.00010857929892220854 | 0.0625 | 5,768 |
| Effective incoming left operand | 0.0011047735263161646 | 0.5 | 44,372 |
| Effective incoming right operand | 0.0012723287480862816 | 1 | 38,082 |
| Incoming contraction output | 8.714693894658438 | 8192 | 389,362 |

This refutes treating the incoming contraction as the first established cause:
its input already differs before the incoming norm/projection/routing. The
earlier exact three-entry slices did not cover these differences. Next capture
the complete outgoing operands/contraction/norm/output boundaries before
changing incoming GEMM arithmetic. Artifact:
`esmfold2-incoming-result-20260908-L8S8jo/{native,candidate}`; immutable source
`esmfold2-incoming-source-20260908-0WYzVV`.

The existing same-input contraction replay also now supports the first incoming
output-i chunk, restoring original strides and the right-half storage offset.
That new replay mode is prepared, not GPU-verified or launched: the observed
earlier mismatch takes diagnostic priority. Capture/layout tests pass 54 checks;
Ruff and diff checks pass. Future candidate captures additionally save the
optimized executable HLO separately from the previous pre-optimization HLO;
job 491 predates this metadata addition. No full-model admission follows.

## Complete outgoing boundary and compiler controls

Jobs 492/493 compare all 48,888,064 values at each outgoing boundary, preserving
both native and candidate uninstrumented block outputs bitwise. First norm,
first projection, effective BF16 left/right operands, and outgoing output gate
are bitwise equal. The outgoing contraction is the first established mismatch:
RMSE 2.6308347303996267, max 8192, 6,056 strict failing entries (6,084 bitwise
unequal). Differences occur in every 64-row chunk and the 53-row tail, not only
in the tail. Following norm and output projection have respectively 3,076 and
55,034 strict failing entries. These are representation values, not Angstroms.

The first chunk's native operands and output are byte-identical to the earlier
standalone replay reference. That standalone JAX contraction was bitwise equal
and lowered to cuBLASLt. The actual default block instead uses nested Triton
GEMMs and merges chunks into larger contractions. This is evidence of differing
lowered execution, not proof that one compiler switch is the complete fix.
Artifacts: `esmfold2-outgoing-result-20260908-cWFU6A/{native,candidate}`;
immutable source `esmfold2-outgoing-source-20260908-pL7ikC`.

| First-block uninstrumented control | RMSE | Max absolute | Instrumentation preserves output bytes |
| --- | ---: | ---: | --- |
| Default, job 493 | 0.5564370511546407 | 32 | Yes |
| Disable Triton GEMM, job 495 | 0.6508338696285835 | 32 | No |
| Also disable cuBLAS padding, job 497 | 0.6508071709052594 | 32 | No |

Job 495's actual HLO pads contraction K437 to 440. Job 497's actual HLO
preserves K437, yet does not repair the block. Padding alone is therefore not
a sufficient explanation. Both diagnostic compiler controls change the output
when intermediate tensors are returned: their captured interior differences
must not be assigned to the uninstrumented executable. Neither flag becomes a
runtime default. The unpadded block and successful standalone replay still
have different RHS storage/contracting axes in the cuBLAS calls, motivating a
same-value, storage-layout-only operator control rather than another full-model
run. Artifacts: `esmfold2-no-triton-result-20260908-9HHMBK/candidate` and
`esmfold2-unpadded-result-20260908-hwO1FI/candidate`.

Job 498 tests both channel-major RHS storage orders on exactly the same first
chunk values. The original replay and both layout controls are bitwise equal
to all 7,159,808 native output values, twice each. Actual optimized HLO confirms
distinct RHS contracting axes, so storage transposition alone is insufficient
to reproduce the block mismatch. Both standalone calls carry an explicitly
selected cuBLAS algorithm and nonzero autotune workspace; the block's GEMM
inside dynamic-slice fusion does not. This motivates a separately scoped
fusion control, not a claim that autotuning is already proven responsible.
Artifact: `esmfold2-layout-result-20260908-HfHAMO/candidate`; source
`esmfold2-layout-source-20260908-ZTyrVt`. New algebra/layout and affected probe
checks pass 113 CPU tests; Ruff and diff checks pass. No model runtime compiler
default, performance claim or full-model admission changes.

Job 499 removes dynamic-slice fusion: optimized HLO confirms its removal and
an explicitly selected cuBLAS algorithm. Block RMSE and captured arrays remain
unchanged from job 497; instrumentation still changes the final block output.
Thus this fusion is not a sufficient cause either. Importantly, the disabled
Triton-GEMM flag did **not** turn every contraction into cuBLAS: merged 128/117
chunks still contain dot operations in other fusions. Only the 64-row outgoing
chunk at offset 320 is bitwise equal; the other six chunks differ. Disabling
`dot-merger` in the same diagnostic profile is the next discriminating control.
Artifact: `esmfold2-unfused-result-20260908-jSGf4M/candidate`; source
`esmfold2-unfused-source-20260908-0ABqhU`. An additional affected runtime suite
passes 120 CPU tests. These compiler experiments do not change production.

Job 500 additionally disables `dot-merger`. Optimized uninstrumented HLO now
contains six 64-row plus one 53-row cuBLAS contractions per triangle, rather
than the larger merged shapes. In the **instrumented** graph every captured
outgoing boundary (including contraction, norm and output projection) and
incoming input/effective operands/contraction matches native bitwise. This is
a controlled improvement over job 499, not uninstrumented-model admission:
instrumentation still changes the final block output (RMSE 0.43108652886598087),
and the uninstrumented block vs native remains RMSE 0.6126431812163932, max 32,
18,915,516 strict failing values. First differing recorded slices in the
instrumented graph are the transition's final `ffn.w3` projection outputs;
their input slices match, but complete inputs still require verification.
Artifact: `esmfold2-native-chunks-result-20260908-4EmMdG/candidate`; source
`esmfold2-native-chunks-source-20260908-yFwZrr`. A full first-transition
projection input/output capture is added with unchanged-native-output, shape,
finite and lossless-BF16 checks. No production compiler options are changed.

Jobs 501/502 capture the complete first transition `ffn.w3` input/output.
Native instrumentation preserves the native block output bitwise. Under the
native-chunk candidate control, all 28,639,232 input BF16 values match native
bitwise; output RMSE is 0.5050302390562514, max 16, with 2,071,835 strict
failures among 7,159,808 values. The candidate's instrumentation effect remains
nonzero, so this isolates an operator difference in the captured execution,
not the first cause of the uninstrumented block. Artifact:
`esmfold2-transition-result-20260908-qhMCLw/{native,candidate}`; source
`esmfold2-transition-source-20260908-BAnmX1`. Capture and affected replay tests
pass 75 CPU checks; Ruff and diff checks pass. A same-input native reduction
policy replay is the next control; upstream default remains unchanged.

Job 503 isolates that reduction-policy difference on the exact complete native
input, original checkpoint and original input strides. Native default `True`
reproduces every original w3 output byte twice. Setting only
`allow_bf16_reduced_precision_reduction=False` reproduces the candidate w3
output **bitwise**, twice: all 7,159,808 values, zero unequal. Its difference
from native default is precisely the job 502 RMSE/max/2,071,835 differing
entries. This is a controlled causal result for this projection, not merely
similar aggregate metrics: parent compared the saved arrays byte-for-byte.
Artifact: `esmfold2-transition-replay-result-20260908-QkOJEl/native`; source
`esmfold2-transition-replay-source-20260908-TKT4nu`. The upstream default is
not changed to manufacture parity; reproducing its actual reduced-precision
accumulation remains implementation work. Full-block instrumentation effects,
full-model structures/confidence and performance remain open.

Job 504 profiles those two native policies with a byte-unchanged output gate
on each. Default uses CUTLASS
`cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_256x128_32x3_tn_align8` plus a
device memset; reduced accumulation disabled uses
`cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x128_32x4_tn_align8` without
that memset. Both gates pass. Kernel names establish a dispatch difference,
but do not by themselves specify partial-sum rounding order or authorize an
ad hoc BF16 split-K emulation. Launch dimensions and the actual native kernel
algorithm must be established before that implementation. Artifact:
`esmfold2-transition-profile-result-20260908-Izl4gw/native`; source
`esmfold2-transition-profile-source-20260908-1p5SFT`. This is operator
provenance, not a speed/memory benchmark.

Job 505 additionally captures launch geometry. Default GEMM uses grid
`[219,1,3]`, block `[256,1,1]`; reduction-disabled uses grid `[438,1,1]`, block
`[128,1,1]`. Both profiling output gates pass. This establishes a three-part
launch, not the exact split boundaries. Artifact:
`esmfold2-transition-launch-result-20260908-Jkjrx2/native`.

The first serial split-K hypothesis used CTA-K32 aligned ranges
`[0,352),[352,704),[704,1024)`. Job 506 did not preserve its explicit partial
BF16 rounding: optimized HLO removed the intermediate narrowing and fused
the additions into FP32 beta=1 cuBLAS calls. Job 507 sets
`xla_allow_excess_precision=False`; HLO then retains both intermediate BF16
conversions, but native parity still fails (RMSE 0.5852671179645258, max 16,
2,613,021 strict failing values). Thus neither an unverified split boundary
nor adding casts alone is an implementation fix. Artifacts:
`esmfold2-transition-splitk-result-20260908-vM4ifX/candidate` and
`esmfold2-rounding-result-20260908-aq1bvz/candidate`.

The public CUTLASS [Gemm template](https://github.com/NVIDIA/cutlass/blob/v2.11.0/include/cutlass/gemm/kernel/gemm.h)
and [UniversalParamsBase](https://github.com/NVIDIA/cutlass/blob/v2.11.0/include/cutlass/gemm/kernel/params_universal_base.h)
use different split alignment rules. The universal BF16 rule predicts
8-element aligned ranges `[0,344),[344,688),[688,1024)`. This is an explicit
alternative hypothesis, not proof of the cuBLAS binary's exact source version.
It is tested separately while retaining intermediate rounding. CPU regression
coverage checks that partial rounding is observable in eager and compiled
execution with excess precision disabled. No production path changes yet.

Job 508 retains BF16 rounding with the alternative 8-element aligned ranges.
It also fails: RMSE 0.5828931539672261, max 16, 2,603,187 strict failing values;
same-executable repeat is bitwise. Public template rules and launch dimensions
are therefore insufficient to reproduce this closed-library dispatch. Neither
split hypothesis is admitted to the runtime. Artifact:
`esmfold2-align8-result-20260908-LCzDKE/candidate`. The next read-only control
uses the library's documented dispatch logging rather than an unbounded sweep
of guessed partition boundaries.

Job 509's documented cuBLASLt logger confirms native default dispatch:
BF16 A/B/C/D, FP32 compute, algorithm 21, tile 256x128, stages 32x3,
`REDUCTION_SCHEME_INPLACE`, `numSplitsK=3`, 876-byte workspace. The disabled
policy requests compute-type reduction and chooses a non-split 128x128 kernel.
Logging preserves the recorded outputs. Artifact:
`esmfold2-cublas-log-result-20260908-b8H50e/native`; the private sibling log
contains raw process details and is not a publication artifact.

Job 510 adds only `xla_allow_excess_precision=False` to the native-chunk block
control. Its instrumented and uninstrumented block outputs are **bitwise
identical** (previously RMSE 0.43108652886598087). Full native block parity still
fails: RMSE 0.5434714484737314, 12,046,041 strict failing values; the complete
first w3 input remains exact and its output retains the previously isolated
reduction-policy difference. Artifact:
`esmfold2-strict-rounding-result-20260908-6hofnm/candidate`; source
`esmfold2-strict-rounding-source-20260908-PTNV7H`.

The actual native GEMM destination is BF16, while the existing JAX helper
explicitly requests a FP32 GEMM destination followed by narrowing. The next
same-input control requests BF16 GEMM output directly, retaining FP32 compute
verification from the lowered executable. This is a more direct native
interface match than adopting either failed manual split-K hypothesis.

Job 511 verifies that direct BF16 destination with HIGHEST operand precision
does lower to a BF16-output cuBLASLt call. It nevertheless reproduces the
reduction-disabled result, not native default: RMSE 0.5050302390562514, max 16,
2,071,835 strict failing values, repeat bitwise. Destination dtype alone is
not sufficient. Artifact: `esmfold2-bf16-output-result-20260908-jGJ11C/candidate`.
The next diagnostic varies the explicit BF16 operand-precision policy and
records library dispatch; no FP32 operand is silently reduced and no runtime
default is changed.

Job 512 explicitly uses DEFAULT rather than HIGHEST for BF16 operands. Its
complete output is unchanged. Library logs show that XLA actually benchmarks
the same native three-way in-place kernel, but finally dispatches a non-split
128x128 kernel. Thus operand precision alone does not establish native kernel
selection. Job 513 sets `xla_gpu_autotune_max_solutions=1`; logs show this does
**not** limit the cuBLAS candidate list in this build, and results remain
unchanged. The option's name is not evidence that the intended restriction
took effect. Artifacts: `esmfold2-default-precision-result-20260908-G1Pntu` and
`esmfold2-first-heuristic-result-20260908-sqaa4K`. A separate level-zero
autotuning control must verify its actual runtime dispatch before any claim.

Job 514 disables autotuning with level zero. The runtime log then selects a
non-split 128x128/64x3 fallback, not the native first heuristic. Complete output
again matches the prior reduction-disabled result and repeats bitwise.
Artifact: `esmfold2-no-autotune-result-20260908-j2Ngtz`. Neither limiting
solutions nor disabling tuning implements native dispatch in this build.
Further selection work should inspect the existing bound autotune-record
mechanism or a direct library interface; these failed controls do not justify
a new production default. The independently established rounding-preservation
fix for instrumentation remains valid. Latest affected probe checks pass
63 CPU tests, Ruff and diff checks; no full-model/performance or release gate
was run in this operator investigation.

Jobs 515/516 use the existing bound autotune-record mechanism. Preserve the
original measured record (algorithm index 1); a separate, explicitly labelled
counterfactual changes only the GEMM algorithm index to 0. Strict complete
loading and before/after hashes bind that control. Runtime logs then show
exactly the native 256x128/32x3 three-way in-place dispatch. **All 7,159,808
output values match native bitwise in both repeats.** No input, checkpoint or
expected output is changed. This proves the kernel-selection explanation for
the captured projection; the edited record is an experiment configuration,
not a claim that autotuning selected that algorithm by itself.
Artifacts: `esmfold2-gemm-record-result-20260908-rntI2k/candidate`,
`esmfold2-native-algorithm-control-20260908-V8oqvM`, and
`esmfold2-native-algorithm-result-20260908-6K3YL2/candidate`.

A private CUDA-only primitive now spells this native cuBLASLt call directly
in the installed source, with explicit BF16 operands/destination, layouts,
workspace and algorithm index; no Torch import or manual partial-sum code is
used. It rejects all unmeasured shapes/dtypes and is not wired into prediction
defaults. Job 517 initially failed before execution on a JAX lowering API
signature mismatch. A CPU-side real CUDA-lowering regression reproduced that
failure, then passed after supplying the required module context. Seven
focused checks pass. The corrected GPU primitive still requires its own
same-input validation before integration or model admission.

Job 518 exposed a deeper compiler constraint: emitting the internal
`__cublas$lt$matmul` call before GPU layout assignment aborts compilation with
the assertion that GEMM rewriting must occur after layout assignment. The
private primitive and its test were therefore removed from the installable
tree; the unsupported benchmark switch was removed too. Their exact source
is retained in `esmfold2-carried-gemm-fixed-source-20260908-rQazEZ` for diagnosis,
and the failed run is `esmfold2-carried-gemm-fixed-result-20260908-BlUihZ`.
Do not disable layout assignment to force this unsupported route through.
The successful algorithm-record control (job 516) remains valid; choosing a
supported integration route requires separate verification. No production
native-GEMM implementation or full-model closure is claimed by this attempt.

Jobs 519/520 extend the supported autotune-record control to the full block.
The benchmark-only native-output helper emits BF16 linear destinations with
DEFAULT BF16 operand precision; it rejects unverified bias-bearing linears.
Without an output boundary, job 519 fuses linear output and residual addition
into GEMM beta=1. Instrumentation changes the answer (RMSE
0.6457700750961907), so that run is not a valid full-boundary parity claim.

Job 520 adds an optimization barrier at each native linear output, retaining
the native-chunks/strict-rounding compiler control. Optimized HLO now keeps
the BF16 linears separate from residual addition (beta=0). All 48,888,064
full-block values are byte-identical with and without instrumentation.
Native parity remains open: RMSE 0.5434714484737314, max 32, and 12,046,041
strict failures. The full-block autotune record still selects index 1 for
the 27,968-row w3 projection, the kernel choice isolated by job 516.
Artifacts: `esmfold2-block-gemm-record-result-20260908-d3MEF4/candidate` and
`esmfold2-linear-barrier-result-20260908-Vx5HA4/candidate`.

Verification: 68 focused candidate/repeat-forward CPU tests, repository Ruff
and diff checks pass after the barrier change. These are benchmark controls,
not prediction defaults or full-model/performance/release admission.

Job 521 reuses exactly job 520's source and complete autotune record, changing
only the 27,968-row w3 GEMM algorithm index from 1 to 0 in a separately labelled
configuration copy. `diff -u` confirms a one-line change; the original measured
record is preserved. The loaded control SHA256 is
`3f3b9b0e37be9bd1187d0cbdaedfff4a3b76e5da6c1c3d37d6c0474ddef556a6`.
**All 48,888,064 full-block values now match native bitwise: RMSE 0, max 0,
strict failures 0.** Instrumented versus uninstrumented output also matches
bitwise, and the full first w3 input/output remain exact. This closes this
teacher-forced first-block numerical discrepancy for the bound profile; it
does not close the full LM encoder, diffusion, confidence or performance.
Artifact: `esmfold2-block-w3-result-20260908-E9f5Le/candidate`.
The control is not an autotuning performance result or a portable kernel ID;
supported production integration and fresh-process repeat remain separate gates.

Job 522 repeats job 521 in a fresh process with a separate executable cache.
Full native parity and the instrumentation gate again pass bitwise. The parent
independently reads the native and both candidate NPZ outputs and verifies all
are finite with identical complete array-byte SHA256
`2cb5378e47fb8c0563291964a53ad51f71154c7753f3ac33c7ec79c22ea9e572`.
Artifact: `esmfold2-block-w3-repeat-20260908-V3VirW/candidate`.
Verification: the ESM model tests, LM encoder/capture/projection/repeat probes
and resume-manifest checks finish with **541 passed, 21 skipped**; repository
Ruff and diff checks pass. Skips are retained, not counted as validation. This
is not the full repository CPU or release gate. The supported production
dispatch integration and full-model structure/confidence/performance remain open.

### Carried BF16 output boundary and default-policy regression control

The measured BF16 linear output/barrier now lives in `trunk.py`, not only the
benchmark monkeypatch. Bias-free single-GPU CUDA native-autocast linears take
this route; CPU/TPU, CP and bias-bearing linears retain the existing fallback.
Job 523 invokes the actual carried helper without either benchmark override
flag, retaining the explicit native-chunks/strict-rounding profile and bound
w3 algorithm control. The full native block and instrumentation gate remain
bitwise exact. The parent independently checks the complete output-array SHA256
against jobs 521/522 and native: `2cb5378e47fb8c0563291964a53ad51f71154c7753f3ac33c7ec79c22ea9e572`.
Artifact: `esmfold2-carried-boundary-result-20260908-2zMwld/candidate`.

Jobs 524/525 separately compare actual pre/post-change source with **default**
compiler policy and no autotune-record load or helper override. Complete
outputs match one another byte-for-byte, SHA256
`eaa4e900dffa364806553cd2631a001cbb6fb67f8669ddf0bc99f2045184dfc7`.
Both retain native RMSE 0.5564370511546407, max 32 and 13,509,613 strict failures;
both instrumentation gates pass. Thus the carried boundary does not regress
this default-policy block, but does not by itself close its native mismatch.
Artifacts: `esmfold2-default-linear-before-20260908-rOFf41/candidate` and
`esmfold2-default-linear-after-20260908-hrwD9s/candidate`.

Independent read-only helper review identified CPU-named tests that did not
select CPU explicitly. They now place inputs on CPU and assert result placement.
Job 526 runs the affected tests with CUDA as the default backend: **18 passed**,
including the explicitly CPU-pinned fallback checks. The expanded ESM/resume
CPU suite before that test-placement-only follow-up passes **546 tests with
21 skips**; the final affected CPU rerun passes **18 tests**. Ruff/diff pass.
The private review artifact ends `claude-new-bounded-evidence-and-one-narrowly-scoped-rea-20260908T042625Z-2625407.md`.

Review disposition: full-stack coverage and small/degenerate GEMM accumulation
remain open, not admitted by first-block evidence. A blanket large-K equality
test against FP32 accumulation would be the wrong native contract: jobs
503/516 already show native reduced-precision split-K differs from that control.
Bias-name handling is unchanged from the original helper, so the hypothetical
naming-drift concern is not evidence of a newly introduced bias drop. The
trace-time CP guard is not a new CP support claim; cross-context executable
reuse remains outside this measured single-GPU profile.

Do not implement production dispatch by repeatedly switching loaded autotune
files in one process. The [XLA revision pinned by JAX 0.11.1](https://github.com/jax-ml/jax/blob/jax-v0.11.1/third_party/xla/revision.bzl)
uses a [process-wide call-once in the file-loading path](https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/service/gpu/gpu_compiler.cc#L3370).
A supported FFI/cuBLASLt integration is a candidate requiring its own
build/stream/workspace/layout and native-shape validation; no such runtime
dependency or general dispatch implementation is introduced by this change.

### Vector-shaped linear product and input-rounding correction

Jobs 527/528 add eight synthetic same-host-operand shape controls using the
pinned native Torch BF16 autocast policy and the actual carried JAX helper.
The parent verifies all input/weight arrays, shared benchmark/reference hashes,
and bitwise within-process repeats for both engines. These are operator tests,
not real-weight structure or confidence evidence.

The vector-shaped dots expose a real lowering discrepancy: optimized HLO uses
BF16 elementwise products followed by FP32 reduction, rounding every product
before summation. Native output differs at 4/7 values for M=1,N=7,K=33 and
118/256 values for M=1,N=256,K=1024. The carried helper now requests an FP32
dot result for vector-shaped linears only, then stores the BF16 result at the
existing barrier; non-vector GEMM destinations remain BF16.

Job 529 shows that product correction alone is insufficient. Excess-precision
simplification removes some original input BF16 casts after rewriting the dot
to FP32 products; for M=N=1,K=1024 both inputs are multiplied directly as their
original FP32 values. The final vector route therefore preserves **both input
BF16 casts** with an optimization barrier before the FP32 dot. Job 532 confirms
separate BF16 input conversions and FP32 product/reduction in optimized HLO.

| M, N, K | Original unequal values (528) | Final unequal values (532) |
| --- | ---: | ---: |
| 1, 1, 1024 | 0 | 0 |
| 1, 7, 33 | 4 | 0 |
| 3, 1, 257 | 1 | 0 |
| 2, 3, 1024 | 0 | 0 |
| 17, 16, 31 | 0 | 0 |
| 65, 256, 1024 | 18 | 18 |
| 1, 256, 1024 | 118 | 0 |
| 64, 256, 1024 | 12 | 12 |

All four vector cases now match native bitwise in both repeats. Six of eight
cases are exact; the two remaining ordinary GEMMs have max absolute difference
0.5 and are **not admitted**. Their kernel/accumulation differences require
separate diagnosis; these intermediate activation units are not angstroms.
Artifacts: `esmfold2-linear-shapes-result-20260908-m3uoZ7`,
`esmfold2-vector-fix-result-20260908-0GziTI`, and
`esmfold2-vector-boundary-result-20260908-0VvDKB`.

Regression tests independently cover product cancellation (native result
2^-14 versus prematurely rounded product result 0) and explicit input rounding
(both inputs 1.0035 must first round to BF16 1). Final affected CPU tests pass
21 cases; job 533 passes the same 21 tests on a CUDA-default host. Job 531
preserves full-block bitwise native/instrumentation equality after the initial
vector-product change; the final input-barrier revision has its own separate
full-block regression job. No all-shape or full-model admission follows.

Job 534 validates the **final input-barrier revision** on the full first LM
block using the same explicit native-dispatch record and strict-rounding/chunk
profile: all 48,888,064 values match native bitwise, RMSE/max/failures all zero,
and the instrumentation gate remains bitwise. Artifact:
`esmfold2-vector-boundary-block-20260908-m02Uzf/candidate`.
Verification: final affected ESM/probe/resume CPU suite **549 passed, 21
skipped**; final GPU affected tests **21 passed** (job 533); Ruff/diff pass.
No full repository CPU, optional dependency, wheel, full-model structure/
confidence, performance, release or push gate is claimed by these checks.

### Supported native GEMM dispatch without a carried autotune record

Jobs 535–537 distinguish the two remaining small ordinary GEMMs. Default HLO
splits K=1024 into 16 Triton partial GEMMs. Either the benchmark-only
`xla_gpu_experimental_force_split_k=1` control or disabling Triton closes all
eight synthetic shapes bitwise in both repeats. Native library logs select a
non-split 16x16/128x2 kernel for M=64/65,N=256,K=1024. This does **not** justify
disabling split-K universally: the actual large w3 native dispatch uses
three-way in-place reduction. Artifact:
`esmfold2-small-gemm-policy-result-20260908-4g5xZX`.

`bench/native_cublaslt_ffi.cc` and its Python wrapper prototype the supported
typed JAX FFI interface. They are development-only, outside `src`, with an
explicit prebuilt library argument and no implicit build or dependency install.
The prototype queries the first current cuBLASLt heuristic with the measured
native BF16 operands/destination, FP32 compute, row-major matrix interpretation,
32 MiB workspace preference and 16-byte alignment preference. It does not ship
a device-specific algorithm index or edited autotune record. Per-call handles
are deliberately unoptimized and provide no performance claim.

The first build needed an existing external CCCL header include beyond the
JAX environment's CUDA headers; clean/portable packaging is unverified. The
successful compiler invocation still emits warnings from the upstream FFI
headers. Job 538 compiles the FFI call but fails execution because the runtime
does not provide the requested ScratchAllocator memory. The corrected interface
uses an explicit XLA-owned byte workspace result; Python never exposes that
scratch as model data. No hidden allocator or global compiler mutation is used.

Job 539 passes all eight synthetic native shapes and both repeats bitwise.
The parent additionally compares every input, weight and output array with both
the original native run and its logging-enabled control; all are unchanged.
Artifacts: `esmfold2-ffi-shapes-result-20260908-wHzXsm` (failed scratch allocator),
`esmfold2-ffi-workspace-result-20260908-AHf6Wc` (explicit workspace), and
`native-linear-ffi-workspace-build-20260908-GqN5go` (private compiled artifact).

Job 540 runs the complete actual first w3 input and original checkpoint through
this FFI with **empty compiler options and no autotune load**. Both full
7,159,808-value outputs match native bitwise. Library logs verify the native
256x128/32x3 kernel with in-place split-K=3 and 876-byte used workspace.
Artifact: `esmfold2-ffi-native-w3-result-20260908-nHGOCW/candidate`.

Job 541 replaces only the benchmark's linear helper with FFI for the full first
LM block. With the separately stated native-chunks/strict-rounding triangle
compiler profile, but **no autotune record**, all 48,888,064 output values match
native bitwise and the instrumentation gate is exact. The parent independently
checks both complete output arrays against the native SHA256
`2cb5378e47fb8c0563291964a53ad51f71154c7753f3ac33c7ec79c22ea9e572`.
Artifact: `esmfold2-ffi-block-result-20260908-Fer4vP/candidate`.

Verification: **239 affected CPU checks passed**, including typed FFI buffer/
workspace shape validation, rejection of invalid shapes and unverified bias,
existing block/vector/repeat/resume checks; Ruff/diff pass. The independent
FFI stream/workspace/descriptor review timed out after five minutes without an
answer; it is **not a passed review**. Its private artifact ends
`claude-read-only-bounded-correctness-review-of-new-deve-20260908T045442Z-2659948.md`.
Production build and
resource-lifecycle integration, full LM stack, full-model structures/confidence,
ordinary RNG and performance remain unverified. The prototype is not wired into
prediction defaults and does not close ESMFold2 or the six-model release goal.

### Complete four-block LM encoder

Jobs 543/544 extend the teacher-forced check to the complete native-configured
four-block LM encoder. The native engine instantiates the pinned `FoldingTrunk`,
strictly loads all `lm_encoder.*` parameters, and retains its default non-fused
64-row chunks and BF16 autocast. The candidate uses the actual carried
`folding_trunk` with the development-only FFI linear override and explicit
native-chunks/strict-rounding profile, without a loaded autotune record.
The full native BF16 input and pair mask are identical. Both engines execute
an uninstrumented stack and a separately observed stack capturing every block.

All four complete block outputs (48,888,064 values **per block**) match native
bitwise; the final output and observed final output do too. Both native and JAX
instrumentation gates pass. This is accumulated four-block execution, not
independent teacher forcing of each subsequent block. The input is still the
captured native pre-dropout pair: no per-loop dropout or diffusion RNG is
exercised here, and no structure/confidence or whole-model admission follows.
Artifact: `esmfold2-lm-stack-result-20260908-uHEsEG`.

The full-tape replay harness now accepts an explicit development FFI library
and compiler profile, binds the library/wrapper/compiler-policy sources, and
restores the ordinary linear helper after tracing even if compilation fails.
Its defaults remain unchanged. Job 545 starts a separate native-LM shared-feature
5SAK core comparison with all eight actual RNG tapes, n=5 and two forwards.
This deliberately isolates the downstream core; independent preprocessing and
ESMC parity remain separate gates. The running job is not a completed result.

Verification for the stack/replay-harness extension: **246 affected CPU tests
passed**, including the new finite/lossless-input and native-configuration
preflight tests plus existing tape/report/repeat/FFI/block/resume tests; Ruff
and diff checks pass. These checks do not replace job 545's pending full-core
structure/confidence comparison or the unresolved FFI resource/build review.

Job 545 completed, and the full core is **not admitted**. Its two complete
forwards reproduce all coordinates and all 14 retained confidence leaves
bitwise, with finite outputs. Native-LM/input/tape identities are preserved.
However, native parity still fails after one proper whole-system Kabsch fit
per original-order sample, measured separately per entity without refitting:

| Sample | Protein RMSD (angstrom) | Ligand RMSD (angstrom) |
| --- | ---: | ---: |
| 0 | 1.463228226 | 0.823427514 |
| 1 | 0.153964993 | 0.044868457 |
| 2 | 0.186440146 | 0.051419770 |
| 3 | 0.294445580 | 0.054918529 |
| 4 | 0.195770559 | 0.065605241 |

All 14 strict confidence-leaf comparisons fail. Examples: maximum absolute
complex-pLDDT difference 0.000634133816, pTM 0.001870453358, ipTM
0.013090670109 and per-atom pLDDT 0.064117193222. Raw heads are fully retained.
Artifact: `esmfold2-ffi-full-replay-result-20260908-spQVsX/comparison.json`.
These results do not contradict the exact captured-input four-block stack:
per-loop dropout/injection, MSA/main trunk, conditioning/sampling and confidence
still require their own native boundary checks.

Do not attribute the historical-to-current RMSD change to FFI alone: this run
also changes the whole-core compiler profile. Job 546 is the matched ablation
using exactly job 545's source snapshot, input, native LM, all eight tapes,
five samples, two forwards and compiler profile, with only FFI disabled.
Job 546 completed successfully. Both forwards have finite, bitwise-identical
coordinates and all retained confidence leaves, but native parity still fails:

| Sample | Protein RMSD (angstrom), FFI off | Ligand RMSD (angstrom), FFI off |
| --- | ---: | ---: |
| 0 | 0.627176968 | 0.415587716 |
| 1 | 0.112428793 | 0.040342654 |
| 2 | 0.121783802 | 0.019972731 |
| 3 | 0.136793999 | 0.036721693 |
| 4 | 0.166546317 | 0.075993874 |

Strict confidence also fails. Artifact:
`esmfold2-policy-only-full-replay-20260908-uocnvA/comparison.json`.
This matched contrast isolates the FFI-dispatch intervention, but does not
show a defective FFI operator: exact captured-input LM-stack arithmetic can
still produce a worse final structure when other stages differ. Neither arm
is admitted, and the lower coordinate error is not a reason to select a less
faithful isolated operator or relax the frozen gates.

The captured-input stack starts after the LM shim; the actual full replay
still computes that shim independently, even with `--native-lm` hidden states.
The previously measured shim mismatch is therefore not eliminated by jobs
543–544. Job 547 holds the runtime source, compiler profile, FFI, native LM,
eight tapes and n=5 settings of job 545 fixed and substitutes only the validated
native pre-dropout shim pair through a dynamic argument. This is an explicit
development-only `--native-shim` interchange, not a runtime fallback or a
full-port validation. Provenance, finite FP32 shape, exact single consumption,
helper restoration and report scope are checked. Job 547 completed with finite,
bitwise-identical coordinates and all 14 confidence leaves across its two
forwards. Native parity remains failing:

| Sample | Protein RMSD (angstrom), native shim | Ligand RMSD (angstrom), native shim |
| --- | ---: | ---: |
| 0 | 0.876219357 | 0.652099338 |
| 1 | 0.153106267 | 0.067783932 |
| 2 | 0.106412910 | 0.037382916 |
| 3 | 0.208171022 | 0.040125631 |
| 4 | 0.181043623 | 0.081496266 |

Strict confidence fails. Artifact:
`esmfold2-shim-interchange-result-20260908-W69GUY/comparison.json`.
The native pair substitution changes the downstream trajectory but does not
close it; the independently computed shim is not the sole remaining issue.

Verification: job 546 reference/report identities and both repeat archives
checked; shim-interchange/tape/report/repeat CPU checks **60 passed**; Ruff and
diff checks passed. No production default, acceptance tolerance or model
admission was changed; no commit/push or performance claim is made.

### Main pair trunk and coda parameter/autocast routing

Source comparison identified a separate semantic gap: pinned native `forward`
keeps its FP32 parameters while running `folding_trunk` and `parcae_coda` inside
CUDA BF16 autocast. They instantiate the same `FoldingTrunk` class as the LM
encoder. The port still gave these two consumers blanket-BF16 parameters and
the ordinary arithmetic path. Computing statistics in FP32 does **not** restore
already-rounded LayerNorm affine parameters; the old comment asserting that
equivalence was removed.

Actual runtime `predict` now selects original FP32 main-trunk/coda parameters
and the existing per-operation native-autocast path for single-device BF16.
The real recurrence forwards that parameter dictionary to every main trunk
call. FP32 and context-parallel routing remain unchanged. This does not yet
correct the separately implemented MSA/input embedding, recurrent dynamics,
readout, conditioning or confidence paths, and does not enable the development
cuBLASLt FFI in the installed runtime.

Routing/real-recurrence tests cover original FP32 values that cannot survive a
BF16 storage cast, both eager and JIT recurrence, and the FP32/CP fallback
branches. The focused combined checks pass **56 tests**. Job 548 evaluates the
changed actual model code against job 547 under the same native-shim,
native-LM/tape, n=5, FFI and compiler-profile diagnostic conditions. Job 548
completed with both forwards finite and bitwise-identical for coordinates and
all 14 confidence leaves:

| Sample | Protein RMSD (angstrom), corrected pair trunks | Ligand RMSD (angstrom), corrected pair trunks |
| --- | ---: | ---: |
| 0 | 0.758416366 | 0.303030042 |
| 1 | 0.106473080 | 0.039507410 |
| 2 | 0.167695480 | 0.041958203 |
| 3 | 0.204125613 | 0.056309697 |
| 4 | 0.093288610 | 0.075686354 |

Strict confidence still fails. Artifact:
`esmfold2-pair-trunk-native-result-20260908-Y1mzip/comparison.json`.
The largest protein/ligand residuals decrease, but not all sample values do;
this is not full-core admission or a generally monotonic improvement claim.
Expanded current ESMFold2/tape/report/repeat/resume CPU checks pass **541 tests,
21 skipped**. These skips and the still-open independent bounded routing
review remain distinct from GPU jobs 547–548. No performance admission or push
is inferred from this source-level correction.

A next source-level discrepancy remains in recurrence coefficients. Native
`_discretized_dynamics` evaluates FP32 `softplus(log_delta)`,
`exp(-delta * exp(log_a))` and `delta[:, None] * b_cont`, and narrows the final
coefficient tensors to the pair-state dtype. The current port receives already
BF16-rounded recurrence parameters and evaluates that path in BF16. A same-real-
checkpoint JAX CPU arithmetic contrast (not a native CUDA gate) found final
BF16 storage differences on **197/256 decay coefficients** (max 0.00244140625)
and **26,273/65,536 matrix entries** (max 0.0625). These are coefficients, not
angstroms. The structural contribution remains unmeasured; it needs a separate
original-parameter/native-precision intervention before attributing the
remaining RMSD to it.

### Original-parameter recurrence correction, jobs 549–550

The real `predict` now forwards original recurrence parameters to `run_loops`
on the same single-device BF16 arm. `delta`, decay and the continuous-to-
discrete matrix are computed in the original FP32 dtype, then decay/matrix
are cast to the pair-state dtype exactly where native does so. The ordinary
helper default and the FP32/CP routing remain unchanged. Eager/JIT tests cover
both coefficient consumers and public forwarding with non-BF16-representable
weights. Four new tests initially fail because the original-parameter route
is absent; after implementation the combined focused recurrence/tape checks
pass **55 tests, 4 skipped**.

Job 549 changes this recurrence calculation on top of job 548, retaining the
same native-shim/LM/features, eight actual tapes, n=5, FFI binary and compiler
profile. Both full forwards produce finite, bitwise-identical coordinates and
all 14 confidence leaves. Whole-system Kabsch, entity-only measurement gives:

| Sample | Protein RMSD (angstrom), FP32 dynamics | Ligand RMSD (angstrom), FP32 dynamics |
| --- | ---: | ---: |
| 0 | 0.456583117 | 0.213105707 |
| 1 | 0.097416605 | 0.029677315 |
| 2 | 0.157703796 | 0.051293215 |
| 3 | 0.151301867 | 0.029638010 |
| 4 | 0.123791262 | 0.067522719 |

Strict confidence still fails; no full-core or model admission. Artifact:
`esmfold2-native-dynamics-result-20260908-T4gKT2/comparison.json`.

Job 550 invokes pinned native `ESMFold2Model._discretized_dynamics` itself on
CUDA, using the actual three checkpoint parameters under BF16 autocast. It
confirms both returned tensors are FP32 and finite. After the native forward's
BF16 storage cast, the complete arrays match the JAX CPU original-parameter
arithmetic control by SHA256 (FP32 storage of BF16 values):

- decay, 256 values: `71436bb8c40fe78c1b302036411f018ab0a919b9a35a75fd6101dc4c1c81095b`
- matrix, 65,536 values: `563afa59e2df558f5209335106d9273d9126c5ea2c90fc5d837cc772373e3fa2`

Their pre-cast FP32 hashes differ, so this is a final-BF16-coefficient control,
not FP32 bitwise parity or a capture of internal coefficients in the full JAX
GPU executable. Native source SHA matches the original bound capture; Torch
is 2.13.0+cu130 at cf30153c4c131c8164ee7798e5022d810682e2cb.

The independent bounded pair-routing review completed. It confirmed the new
main/coda routing and raised remaining concerns: pre-existing LM CP gate
asymmetry, original-parameter dtype assumptions for nonstandard callers,
mixed-policy consumers elsewhere in the same region, and unmeasured scaling/
performance of native kernels/barriers. It did not run a native GPU gate or
approve release. The proposed comparison against the old arithmetic path is
not a native rounding oracle; the completed real native controls remain the
relevant evidence. The later recurrence change has not received an independent
review. CP behavior is deliberately not changed by these two fixes and remains
separate work; no general CP parity is inferred.

Verification: current expanded ESMFold2/tape/report/repeat/resume suite **546
passed, 21 skipped**; focused artifact-integrity/routing suite **69 passed**;
Ruff and diff checks passed. Jobs 547–550 finished successfully, while full
native coordinate/confidence gates remain failing. The development FFI is
still not installed in the runtime. Full repository/release/performance gates,
independent preprocessing/ESMC/shim, and commit/push remain unverified.

### Recycle input normalization and materialized recurrence terms, jobs 551–555

Job 551 runs actual Torch CUDA LayerNorm under BF16 autocast with the released
`parcae_input_norm` affine parameters and 4,096 deterministic BF16 input values.
The output is FP32 and exactly equals the explicit-FP32-input invocation.
Rounding the affine parameters first changes 1,177 final BF16 values. This
isolates a dtype/parameter issue, not full-model accuracy. Actual runtime
`predict` now forwards original input-norm parameters on the single-device
BF16 arm, and `run_loops` uses the existing native-autocast norm helper before
casting its result for the recurrence matrix product. FP32/CP/default helper
routes remain unchanged.

Job 552 adds only this input-normalization correction to the fixed native-
shim/tape n=5 diagnostic of job 549. The full comparison **still fails**, and
the largest residuals worsen; faithful local policy is not a guarantee of
monotonic structure improvement while other inputs/operators differ:

| Sample | Protein RMSD (angstrom), input norm | Ligand RMSD (angstrom), input norm |
| --- | ---: | ---: |
| 0 | 1.057028505 | 0.736777834 |
| 1 | 0.080315684 | 0.018389356 |
| 2 | 0.091922801 | 0.020947472 |
| 3 | 0.130650242 | 0.032114060 |
| 4 | 0.113007493 | 0.083586676 |

Artifact: `esmfold2-injection-norm-result-20260908-rhYGUY/comparison.json`.
Confidence remains a strict failure. No tolerance or success label is changed.

The full compiled HLO also confirms an unintended rounding change in the
recurrence: a BF16 `[190969,256]` product with a `[256,256]` matrix incorporates
the already-computed decayed state through GEMM `beta=1`. Native instead
materializes the BF16 linear result before addition. The new runtime
`_recurrence_update` preserves both BF16 terms with an optimization barrier
on the native pair-trunk arm; other routes retain their original expression.
This does not select a new cuBLAS algorithm or install the development FFI.

A cancellation regression distinguishes the policies: `1.0078125**2` is
stored as BF16 `1.015625` before adding `-1.015625`, giving zero rather than a
nonzero fused FP32 intermediate. Eager/JIT and scoped-JAXPR tests pass. Job 553
runs all four checks in the CUDA environment; job 555 additionally explicitly
asserts the CUDA backend and repeats those same four checks successfully.

Job 554 adds only the materialization boundary to job 552's source/policy arm.
The compiled matching BF16 matrix calls no longer contain `beta=1`. Both full
forwards are finite and bitwise-identical for coordinates and all 14 confidence
leaves, but native structure/confidence remain failing:

| Sample | Protein RMSD (angstrom), recurrence boundary | Ligand RMSD (angstrom), recurrence boundary |
| --- | ---: | ---: |
| 0 | 0.656704475 | 0.286674960 |
| 1 | 0.088254606 | 0.041217457 |
| 2 | 0.164814349 | 0.023892319 |
| 3 | 0.228876776 | 0.031513805 |
| 4 | 0.130268021 | 0.071083929 |

Artifact: `esmfold2-recurrence-barrier-result-20260908-rQbAhB/comparison.json`.
The source, native shim/LM, eight tapes, five samples, development FFI and
compiler profile remain explicitly bound. This is a downstream diagnostic,
not independent preprocessing/ESMC/shim or production-default admission.
The run reports a CUDA timer delay warning; no compile/runtime performance
conclusion is drawn from it, nor is algorithm timing assumed reliable.

Verification: affected CPU checks **211 passed, 1 skipped**; CUDA-backend-
asserted recurrence checks **4 passed**; Ruff and diff checks passed. Jobs
551–555 completed. The earlier 546/21 expanded suite predates these last two
runtime changes. Independent review of these changes, remaining embedding/MSA/
readout/conditioning boundaries, ordinary-RNG and production-default controls,
all-model release/performance gates and push remain open.

### Input-embedding boundary and FP32 sequence tail, jobs 556–562

`bench/esmfold2_input_boundary.py` executes each actual `predict` prefix and
stops immediately before `InputsEmbedder`. The native model is constructed on
the meta device, while real shared features and their preprocessing execute
on CUDA; a pre-hook stops before any parameter is consumed. The JAX prefix
is JIT-compiled and similarly stops before the learned embedding. Thus this
probe does not claim weight identity, learned-operator parity, independent
preprocessing or full-model admission. It binds configuration, feature archive,
reference manifest, runner and model-source files before/after execution,
records original dtypes separately from lossless storage and restores the
intercepted helper on all exits.

Native job 556 and candidate job 557 reveal differences before the network:

| Input | Native dtype | Candidate dtype before fix | Unequal / total | Max absolute difference |
| --- | --- | --- | ---: | ---: |
| residue one-hot | FP32 | BF16 | 0 / 14,421 | 0 |
| MSA profile | FP32 | BF16 | 7,901 / 14,421 | 0.001947402954 |
| deletion mean | FP32 | BF16 | 391 / 437 | 0.000432416797 |
| reference coordinates | FP32 | BF16 | 9,219 / 9,312 | 0.011977195740 |

The coordinate value is a maximum **input coordinate component difference**,
not final structural RMSD or crystal accuracy. Atom mask values/dtype match.
Element/name one-hots and charges retain values but differ in storage dtype;
native int64 atom indices/UIDs become int32 with equal values in this case.
This is not a claim that arbitrary int64 input values can be narrowed safely.
Artifacts: `esmfold2-input-boundary-result-20260908-pYLJGQ/{native,jax}`.

The first three sequence fields bypass the atom encoder and are only
concatenated onto its result. Actual runtime `predict` now retains them in
FP32 on the single-device BF16 arm, without changing atom inputs, FP32 or CP
routes. Job 558 shows residue one-hot and deletion mean exactly matching
native dtype and bytes. Profile loses the large BF16 error but retains 121
FP32 differences, maximum 5.960464477539063e-8. Both artifacts were compared
against count ratios evaluated in FP64 then rounded to FP32: native has zero
differences and JAX has those same 121, isolating division rounding in this
masked-MSA case.

Job 559 evaluates the sequence-tail change alone under the existing fixed
native-shim/LM/tape n=5 diagnostic. Both forwards are finite and bitwise equal
for coordinates/all 14 confidence leaves. Strict confidence and coordinates
still fail:

| Sample | Protein RMSD (angstrom), FP32 tail | Ligand RMSD (angstrom), FP32 tail |
| --- | ---: | ---: |
| 0 | 0.635684139 | 0.281876350 |
| 1 | 0.089510583 | 0.041190769 |
| 2 | 0.164602483 | 0.024276213 |
| 3 | 0.228074105 | 0.032064038 |
| 4 | 0.128427871 | 0.074353015 |

Artifact: `esmfold2-sequence-tail-result-20260908-kKFcBH/comparison.json`.

For masked-MSA count division in that native BF16/single-device arm, the real
runtime now uses CUDA `div.rn.f32` through the already-used Pallas/Triton
facility. Operands are FP32; padded tail lanes use 0/1 and are cropped. CPU,
FP32, CP and the separate no-mask mean path retain their previous behavior;
no global compiler policy or dependency is changed. Job 560 passes all eight
shape/dtype/CUDA rounding checks, including non-multiple-of-256 outputs.
Job 561 confirms **all three complete sequence fields now match native dtype
and every stored byte**. Remaining reference-coordinate/atom dtype differences
are not silently declared equal.

Artifact: `esmfold2-profile-division-result-20260908-jfZUlA/boundary`.
The subsequent n=5 full-core job 562 completed. Both forwards retain finite,
bitwise-identical coordinates and all 14 confidence leaves; native parity
still fails:

| Sample | Protein RMSD (angstrom), exact profile | Ligand RMSD (angstrom), exact profile |
| --- | ---: | ---: |
| 0 | 0.660605520 | 0.286238697 |
| 1 | 0.088118076 | 0.040284618 |
| 2 | 0.164443036 | 0.023821672 |
| 3 | 0.229086020 | 0.031926451 |
| 4 | 0.129172015 | 0.073569435 |

Strict confidence remains failing. Artifact:
`esmfold2-profile-division-result-20260908-jfZUlA/comparison.json`.
Exact sequence-input identity is not downstream model equivalence; these
figures retain native shim/LM/features and the development FFI/compiler profile.

The expanded CPU run after
the FP32-tail change passed **545 tests, 21 skipped**, but predates the final
division helper; focused updated checks are recorded separately. A mistakenly
named `test_primitives.py` command collected no tests and was corrected to
the existing `test_primitives_parity.py`, not counted as verification.
Final affected CPU checks pass **181 tests, 5 skipped**, and the queued CUDA
division suite passes **8 tests**. Ruff and diff checks pass. Independent
review of these input-path changes, all-model release/performance gates and
commit/push remain unverified.

The next material boundary is the atom path: original reference positions
feed both atom features and RoPE. Simply removing the outer coordinate cast
without matching the internal Linear/autocast path is not sufficient. Native
atom conditioning/attention/output and MSA learned operations remain open.

### Native atom encoder audit and first-stage compiler controls (jobs 563–571)

These are learned-input-embedding/operator diagnostics, not a new full-core
prediction or model admission. The n=5 coordinates above remain the latest
whole-core result. The new native capture uses the original checkpoint's
`inputs_embedder.*` FP32 parameters, strict state loading, the actual captured
native input boundary, pinned native source, and CUDA BF16 autocast. Archive,
source, input, configuration and checkpoint hashes are checked before/after
capture. No RNG tape is consumed in this deterministic inference submodule;
that does not establish numerical repeatability.

Job 563 directly observed Torch 2.13.0+cu130 CUDA `F.rms_norm` returning FP32
for BF16 input inside autocast, equal to explicitly normalizing FP32 input.
The old atom-module comment suggesting BF16 input always requires BF16 epsilon
was corrected; this is a documentation correction, not a completed RMS kernel
port. Outside-autocast epsilon/compute policy needs a separate native check.

Job 564 captures all three native input atom blocks. All Linear outputs are
BF16; initial LayerNorm and block residual outputs are FP32. Q/K normalization
and rotary implementation still need per-operator numerical checks. External
FlashAttention is unavailable in this native environment, so upstream selects
its Torch SDPA fallback; the selected internal SDPA kernel is not identified by
that availability check.

The native full input embedding has 197,087 values (`[1,437,451]`). Repeating
the uninstrumented native forward changed one value, max absolute 0.015625,
RMSE 3.519581584e-5. The instrumented forward matched the first output exactly.
Rounding only reference positions to BF16 and restoring FP32 before the same
native encoder changed three output values, max 0.03125, RMSE 8.621178988e-5.
This is not evidence that reference rounding alone explains large final
coordinate drift: native repeat variation is already nonzero, and this is one
counterfactual pair. Scatter nondeterminism is a hypothesis, not localized by
these observations.

Native capture: `esmfold2-atom-native-result-20260908-pLSpVe/native`;
source: `esmfold2-atom-native-source-20260908-udp8mT`.

`bench/esmfold2_atom_conditioning_control.py` compares the first learned stage
on shared native inputs. The proposed policy keeps FP32 features and original
norm affine parameters, uses the existing native BF16 Linear helper, then
normalizes its stored BF16 output in FP32. It is **not wired into prediction**.

| Control | Native Linear unequal / 397,312 | Linear RMSE | LayerNorm RMSE |
| --- | ---: | ---: | ---: |
| Default compiler, job 568 | 93,545 | 0.004011244978 | 0.000612565401 |
| Only disable excess precision, job 569 | 93,545 | 0.004011244978 | 0.000612565401 |
| Only disable Triton GEMM, job 570 | 93,545 | 0.004011244978 | 0.000612565401 |
| Disable Triton GEMM and cuBLAS padding, job 571 | 0 | 0 | 2.335725468e-8 |
| Existing full native-chunk/strict-rounding profile, job 566 | 0 | 0 | 2.335725468e-8 |

Job 570 versus 571 changes only the `cublas-pad-for-gemms` pass under the
no-Triton control. This localizes a padding-sensitive execution discrepancy
for this atom projection; actual lowered shapes/algorithm and a portable
production kernel remain to be established. It does not prove which internal
cuBLAS algorithm changed, nor authorize globally disabling a compiler pass.
Feeding the captured native Linear output into the candidate FP32 LayerNorm
also yields RMSE 2.335725468e-8, max absolute 4.768371582e-7. The norm is not
bitwise equal, but most first-stage error here originates before normalization.

Under strict rounding, the legacy stage has 9,219 changed feature entries from
reference rounding, Linear RMSE 0.003737903066 and LayerNorm RMSE 0.000984240167.
Under default excess-precision settings its intermediate round-trip feature
cast is elided in this instrumented control, so the returned feature array is
exact native instead. Do not mislabel that as the public BF16 input boundary
being equal: job 557 measured the actual boundary's dtype and values directly.

An attempted unconditional BF16 operand barrier in `_native_bf16_linear`
(job 567) did not change the default-compiler projection error. The tentative
runtime edit and its new test were withdrawn; the existing vector-only barrier
is preserved. No new runtime numerical correction is claimed from this audit.
The negative source snapshot remains
`esmfold2-atom-linear-barrier-source-20260908-jPCZD6`, with output
`esmfold2-atom-linear-barrier-result-20260908-z7c16g/jax`.

Remaining artifacts (all relative to the external benchmark root):

- Jobs 566/568: `esmfold2-atom-conditioning-result-20260908-EyuXwC/strict`
  and `default-frozen`, source `esmfold2-atom-conditioning-source-20260908-eMRiBX`.
- Job 565 is superseded by 568: its runner file was updated after completion
  when compiler-profile selection was added. Do not use 565 as an immutable
  source/artifact claim. The replacement run has its own intact bindings.
- Job 569: `esmfold2-atom-rounding-only-result-20260908-UpWen5/jax`,
  source `esmfold2-atom-rounding-only-source-20260908-G220Ew`.
- Jobs 570/571: `esmfold2-atom-gemm-control-result-20260908-sEAf1G/jax`
  and `no-padding`, source `esmfold2-atom-gemm-control-source-20260908-Sa2zWx`.

Verification: native and candidate queued jobs completed without nonfinite
outputs; 21 focused capture/input-boundary CPU tests passed, and changed-file
Ruff/diff checks passed. The expanded retained-code run
`JAX_PLATFORMS=cpu .venv-ci/bin/python -m pytest -q tests/models/esmfold2
tests/test_esmfold2_atom_encoder_probe.py tests/test_esmfold2_input_boundary.py`
passed **377 tests, 24 skipped**. Parent-side archive and every bound-file hash
were rechecked for jobs 564 and 566–571; all were intact. The temporary
barrier's 40-test run is not counted as
verification of a retained numerical fix. Full current release, independent
review, native atom-stack parity, end-to-end performance and main push remain
unverified. Next: preserve the native atom Linear/norm/RMS/rotary/residual
sequence coherently, test the remaining captured block boundaries, then rerun
the fixed-tape n=5 full-core control without converting local operator success
into model admission.

### Native input atom autocast and RoPE phase correction (jobs 572–576)

The real single-device BF16 prediction path now passes original reference
features and original FP32 input-encoder weights to an explicit atom
`native_autocast` route. Linear outputs, Q/K post-normalization storage,
rotary products, SwiGLU products and gated updates use BF16 storage boundaries;
normalization and residual state stay FP32. Mathematical attention uses FP32
scores/softmax/context accumulation followed by BF16 context storage. This is
not a claim of Torch fused-SDPA algorithm identity. Existing non-autocast
diffusion, FP32 and CP routing remain on the old path.

The first full atom capture (job 572) exposed a separate, large RoPE error:
cos/sin differed by up to 0.958984375 before attention. Native source explicitly
uses FP32 inputs for its two outer-product einsums, but that does not disable
autocast. Job 573 observed the actual CUDA `torch.einsum` results:

| Equation | Entering operands | Result |
| --- | --- | --- |
| `bna,k->bnak` | FP32, FP32 | BF16 `[1,3104,3,2]` |
| `bn,k->bnk` | FP32, FP32 | BF16 `[1,3104,10]` |

Recomputed native cos/sin matched job 564's saved tables exactly. The JAX
native route now narrows positions, UIDs and frequencies for those products,
stores BF16 phases, then performs trig opmath and stores BF16 tables. The
FP32 non-autocast route is unchanged. UID 257 rounding to 256 makes clear why
this is not equivalent to merely narrowing a final cosine. New eager/JIT
tests cover that distinction.

Job 574 matches all **99,328** native sin/cos table entries exactly. First
atom Linear output is also exact in this captured strict-compiler control.
Full uninstrumented input-embedding RMSE falls from 0.005426282162 (job 572)
to 0.000500657140 (574), max absolute 0.03125. Repeated candidate forwards
are bitwise equal, but instrumenting 574 changes 2,287 final embedding entries
(RMSE 0.000197568795). Thus intermediate captures are diagnostics, not proof
of the exact uninstrumented executable's boundaries. The candidate native
LayerNorm helper receives FP32 after explicit promotion, while the native
module pre-hook sees BF16 before autocast; those captured values match, but
`dtype_equal` is correctly false rather than hiding the different hook level.
Attention/scatter and residual numerical parity remain open.

Job 575 rechecks the actual `predict` input boundary. All ten fields match
native values; eight match dtype and every byte. `ref_space_uid` and
`atom_to_token` retain native int64 versus JAX int32 storage, with identical
values for this case. This is not a general integer-overflow proof or new
independent-preprocessor validation.

Job 576 reruns the same full shared-native-shim/LM/features diagnostic as 562:
eight fixed tapes, seed 101, five paired samples, the development FFI and
native-chunk strict-rounding compiler policy, two identical forwards. Both
forwards have finite, bitwise-equal coordinates and all 14 confidence leaves.
One proper whole-system Kabsch fit is used per sample, measuring each entity
without refitting:

| Sample | Protein RMSD before (Å) | Protein after (Å) | Ligand before (Å) | Ligand after (Å) |
| --- | ---: | ---: | ---: | ---: |
| 0 | 0.660605520 | 0.314695448 | 0.286238697 | 0.125691162 |
| 1 | 0.088118076 | 0.065889942 | 0.040284618 | 0.021143367 |
| 2 | 0.164443036 | 0.111259314 | 0.023821672 | 0.027854450 |
| 3 | 0.229086020 | 0.112468198 | 0.031926451 | 0.013478798 |
| 4 | 0.129172015 | 0.131377802 | 0.073569435 | 0.013732821 |

Maxima improve, but two individual entity/sample values worsen. Coordinate
0.05 Å and strict confidence gates **still fail**. No whole-model, ordinary
RNG, production-default compiler, independent LM/shim or performance admission
is made.

Artifacts:

- Job 572: `esmfold2-atom-autocast-result-20260908-xwEMk2/jax`,
  source `esmfold2-atom-autocast-source-20260908-4P3QK1`.
- Jobs 574–576: `esmfold2-atom-rope-autocast-result-20260908-vcQJtu/`
  (`jax`, `boundary`, `core`, `comparison.json`), source
  `esmfold2-atom-rope-autocast-source-20260908-bzowCV`.

Verification: expanded pre-RoPE affected CPU run **385 passed, 24 skipped**;
post-RoPE focused atom/attention/input/capture run **48 passed, 2 skipped**;
final expanded post-RoPE run **390 passed, 24 skipped**. Ruff, syntax and diff
checks pass. Native/candidate input archives were checked byte-for-byte for
the eight same-dtype fields, and every bound-file/archive hash in the candidate
atom and input-boundary reports was rechecked intact. Two initial
input-boundary assertions still expected
premature BF16 coordinates and were updated to the observed native FP32
contract; the corrected eager/JIT tests pass. A bounded read-only peer review
was requested for the new input atom route, not for release approval. It
terminated after the 600-second limit with exit 124 and no answer (thread
`th-20260908T070236Z-2790501`); there is no independent review approval for
these changes. Do not treat the timeout as review completion or restart it
solely because time elapsed. No new commit/push or all-model release gate has
run. The next discriminating core control is native input-embedding interchange
to separate remaining atom error from downstream MSA/conditioning/trunk error;
that control has not yet run.

### Native input-embedding interchange does not close the core (job 577)

The proposed diagnostic has now run. `bench/esmfold2_tape.py` accepts the
explicit `--native-input-embedding` control only with native LM and shim
controls. It verifies original reference/checkpoint/source/input bindings,
loads job 564's uninstrumented native baseline, and substitutes the embedding
exactly once as a dynamic compiled operand. Helpers are restored even on
failure. Saved embedding bytes and their schema/hash are revalidated at
completion and in `esmfold2_tape_report`; the report explicitly marks the
additional substitution and never admits the model.

Important limitation: this embedding was produced by a **separate native
InputsEmbedder execution**, not extracted from the original full native tape
capture. Job 564 observed native repeat variability. Substituting the input
also changes the compiled graph relative to job 576; identical flags and tapes
do not make these identical executables or freeze all GEMM algorithm choices.
Therefore this is a negative diagnostic, not a clean attribution of all
remaining error to a particular downstream operation.

With the same eight tapes, seed 101, five paired samples, development FFI,
native-chunk strict-rounding profile and native LM/shim:

| Sample | Protein: actual path 576 (Å) | Protein: native input 577 (Å) | Ligand: actual path 576 (Å) | Ligand: native input 577 (Å) |
| --- | ---: | ---: | ---: | ---: |
| 0 | 0.314695448 | 1.518148070 | 0.125691162 | 0.506009038 |
| 1 | 0.065889942 | 0.104707828 | 0.021143367 | 0.031287537 |
| 2 | 0.111259314 | 0.147883442 | 0.027854450 | 0.043119888 |
| 3 | 0.112468198 | 0.151347112 | 0.013478798 | 0.028645332 |
| 4 | 0.131377802 | 0.086235833 | 0.013732821 | 0.023438512 |

Coordinate and strict confidence gates still fail. Both forwards are finite
and bitwise equal for coordinates and all 14 confidence leaves. The run emitted
a CUDA timer warning about suboptimal measurement accuracy; no timing or
performance conclusion is drawn. This diagnostic does **not** replace the
latest actual-path result (job 576, maxima 0.314695/0.125691 Å), and no runtime
model implementation was changed in this interchange turn.

Artifact: `esmfold2-input-interchange-result-20260908-5bQO85/`
(`core`, `comparison.json`); source:
`esmfold2-input-interchange-source-20260908-YNTrHX`.

Verification: **78 affected replay/report/control tests passed**, including
dynamic nested LM-shim/input controls, exact consumption, restoration on
errors, schema/hash/provenance rejection and saved-artifact validation.
Parent-side native embedding byte equality and every source/reference binding
were checked; Ruff, syntax and diff checks pass. Independent review, release
and push remain unverified. The next useful comparison starts from an identical
embedding and audits `z_init_1`, `z_init_2`, relative-position/bond projections
and their BF16 additions before MSA/recurrence. A same-executable paired
embedding experiment would also be needed for a stronger causal estimate;
neither follow-up has run yet.

### Pair initialization agrees on a shared embedding (jobs 579–580)

`bench/esmfold2_pair_init_probe.py` now isolates the next stage. Both engines
consume the same bound native standalone embedding and original checkpoint.
Native uses the pinned model's four actual projection modules and reconstructs
the published eager addition expression in CUDA BF16 autocast. Candidate
traces the actual `predict` prefix, replaces only the input embedding, and
stops at the second `shard_pair_rows` call after `z_init` construction, before
LM/MSA/recurrence. Missing later weights are intentionally not needed.
The candidate retains the existing native-chunk strict-rounding compiler
diagnostic; no FFI override or new runtime change is introduced here.

| Boundary | Full output shape | Numerical unequal | Bitwise result |
| --- | --- | ---: | --- |
| `z_init_1` | `[1,437,256]` | 0 | exact |
| `z_init_2` | `[1,437,256]` | 0 | exact |
| relative position | `[1,437,437,256]` | 0 | exact |
| token bonds | `[1,437,437,256]` | 0 | signed-zero difference only |
| final `z_init` | `[1,437,437,256]` | 0 | exact |

All five native/candidate boundaries have BF16 dtype, with zero numerical
RMSE and max absolute difference. The bond output has **23,866,125** differing
zero sign bits: native has positive zero where candidate has negative zero.
These are verified zero-versus-zero differences, not a generic tolerance
pass. The final pair's **48,888,064** entries are bitwise equal, so the zero
sign differences do not survive this case's initialization sum. This does
not establish their irrelevance for every possible downstream input.

Each engine's two repeated captures agree. The original source/weights/input
bindings and saved archive hashes were rechecked intact. Capturing the JAX
prefix materializes intermediates and changes its compilation scope; this
result is not proof that an uninstrumented whole-core executable uses exactly
the same kernel choices. It provides no reason to change the pair-initializer
implementation on this case, and does not replace job 576's actual-path RMSD.

Artifacts: `esmfold2-pair-init-result-20260908-BMsc6N/{native,jax}`;
source: `esmfold2-pair-init-source-20260908-P9ooIb`.

Verification: **82 affected probe/replay/report tests passed**. New tests run
the actual eager/JIT prefix with only these projection parameters, check the
five outputs and BF16 addition, and verify restoration on missing-boundary
and execution-error exits. Ruff, syntax and diff checks pass. There was no new
full n=5 run, runtime fix, release gate, independent review or push in this
pair-initialization audit.

The next narrowed target is the released `modeling_esmfold2.py::MSAEncoder`,
not the similarly named experimental module. Source inspection shows that the
candidate still passes early-BF16 MSA features/affine parameters into legacy
norm/OPM/PWA/transition paths. Native starts with FP32 one-hot/deletion inputs
and uses per-operation autocast. The fixed tape has column mask `[1,437]` and
four row-index arrays `[4,1024]`, selecting from `[1,4449,437]` MSA features.
Native source applies the column mask once, then resamples rows per loop.
The first MSA encoder receives `z_init` (the LM encoder output is incorporated
later), so its first-loop input can be isolated without replaying LM inference.
This source-level lead still needs native first-loop boundary capture before
attributing any measured coordinate drift or changing its numerical policy.

### First MSA OPM autocast boundary (jobs 581–583)

Fixed native embedding, pair initialization, features and first-loop tape
selection isolate the released MSA encoder prefix. Jobs 581/582 have exactly
equal embedding outputs and all 57,278,720 values entering the first OPM norm.
Native norm returns FP32; the previous candidate returned BF16. Its first
projection differed in 22,301,311 values (RMSE 0.000316455865922).

The runtime now routes original FP32 OPM parameters separately, preserving
native norm/Linear/mask autocast boundaries. Other MSA operations retain their
existing policy; this is not a claim of full MSA fidelity. Candidate job 583
reduces norm-output max difference to 1.8067657947540283e-7 (RMSE
8.072879289877525e-9). First projection differs in 5,416 values, maximum
0.0009765625, RMSE 1.7521064498069652e-6. Thus the dtype correction improves
this boundary substantially, but neither norm nor projection is bitwise exact.
Early embedding input casts still differ, while their projection outputs agree.

Artifacts: `esmfold2-msa-prefix-result-20260908-A1F4Lu/{native,jax}` and
`esmfold2-msa-opm-result-20260908-1waLJd/jax`;
candidate source: `esmfold2-msa-opm-source-20260908-alZTsF`.

Verification: 20 affected CPU tests passed; scoped Ruff passed; queued CUDA
job 583 completed successfully and all eight saved prefix arrays were compared.
Full OPM output, updated n=5 core RMSD/confidence, broader modalities, production
performance, independent review and release/push remain unverified for this
change. Job 576 remains the latest actual-path coordinate result.

Full-core attempt 584 failed during tracing, before coordinate generation:
the diagnostic FFI wrapper rejects biased linears, and the newly routed OPM
`Wout` has bias. This is a diagnostic dispatch incompatibility, not a measured
structural regression. The replay wrapper now leaves biased projections on
the original runtime implementation and records `ffi_bias_policy`; the FFI's
bias-free rejection remains intact. The frozen 583/584 source was not edited.
Verification: replay/probe tests 28 passed, scoped Ruff passed; additional
model/tape wiring tests 38 passed with one upstream-dependency module skipped.
CUDA retry with the corrected wrapper and updated full-core metrics is pending.

Retry 585 was submitted using immutable source
`esmfold2-opm-bias-source-20260908-52fdSu` and output
`esmfold2-opm-bias-result-20260908-So7k9R/core`. It retains job 584's
native-LM/native-shim, fixed tape, n=5, two-forward diagnostic conditions,
with the documented biased-Linear runtime fallback. At this recording the
queue wait handle is live; no new coordinate or confidence result is admitted.
The broader ESMFold2 CPU test suite is also running, not yet a completed gate.

Job 585 completed; `esmfold2-opm-bias-result-20260908-So7k9R/comparison.json`
contains the fixed-order, global-Kabsch entity results (angstrom):

| Entity | Sample 1 | Sample 2 | Sample 3 | Sample 4 | Sample 5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Protein | 0.923487234 | 0.132686558 | 0.130584572 | 0.158367305 | 0.108667721 |
| Ligand | 0.305537003 | 0.017568578 | 0.040136167 | 0.020434040 | 0.015537181 |

Both forwards have finite, bitwise-identical coordinates and confidence.
Coordinate and strict confidence gates fail. Compared with job 576, maxima
worsen from protein 0.314695448 / ligand 0.125691162. The improved first OPM
projection does not establish full OPM or full-model improvement. Do not
admit this change as closure: next isolate outer contraction and biased Wout
against native, before attributing the end-to-end drift to any one operation.

Verification: expanded CPU run yielded 411 passed, 24 skipped, one failure in
a test double missing the new `native_opm_params` keyword. The double now
accepts it and asserts None for the legacy path; all 25 padding tests pass on
rerun. Skips cover missing Torch/Transformers and explicit CUDA-only probes.
Scoped Ruff passes. The full suite was not rerun after the test-double edit;
no release gate, independent review or push was performed.

The prefix probe now supports `--full-opm`: it retains the original eight
boundaries and adds Wout input (flattened outer product), Wout output, and
the final post-division OPM output. Native uses the released OPM module;
candidate uses the runtime OPM implementation. Both stop before subsequent
MSA operations. This instrumented scope does not establish uninstrumented
whole-core kernel equivalence or isolate Wout from upstream input differences.
Jobs 586 (native) and 587 (candidate native-autocast) were queued serially;
source `esmfold2-full-opm-source-20260908-OO46Zt`, output
`esmfold2-full-opm-result-20260908-VbeaTu/{native,jax}`. Results are pending.
Verification: 14 probe/dispatch tests passed, including hook restoration in
both capture modes on failures; scoped Ruff and diff checks passed.

Native job 586 completed successfully: captured Wout input, Wout output and
final OPM output are all BF16; the norm output is FP32. Candidate 587 is
still running, so no numerical full-OPM parity conclusion is available yet.
An additional small actual-runtime JIT test exercises the full 11-boundary
capture without providing subsequent triangle parameters and checks hook
restoration. All 15 probe/dispatch tests pass; scoped Ruff passes. This test
was added after the frozen GPU snapshot and is not claimed to be inside it.

Jobs 586/587 completed; all saved archive hashes and bound source/input/weight
files were revalidated intact before comparing all 11 arrays. Outputs:

| Boundary | Entries | Unequal | Maximum absolute error | RMSE |
| --- | ---: | ---: | ---: | ---: |
| First W output | 28,639,232 | 5,416 | 0.0009765625 | 1.752106449807e-6 |
| Wout input / outer | 195,552,256 | 231,972 | 0.0625 | 6.008333731996e-5 |
| Wout output | 48,888,064 | 154,225 | 0.5 | 0.001758088350514 |
| OPM output after division | 48,888,064 | 154,225 | 0.015625 | 1.259251662305e-5 |

These four outputs are BF16 on both sides. Embedding outputs are numerically
identical. The norm input has 57,278,464 entries (correcting the earlier prose
count of 57,278,720); all values match, but the candidate helper hook observes
the FP32 promotion whereas the native module hook observes incoming BF16.
Norm output remains FP32 on both sides, max error 1.8067657947540283e-7.
The candidate embedding input casts still differ from native FP32 inputs.

This experiment does not isolate Wout arithmetic: its input already differs.
The next discriminating control is the same saved native Wout input passed
through both implementations with the original checkpoint bias/weights.
Neither this small OPM output RMSE nor its matched dtype establishes that OPM
caused (or explains all of) job 585's end-to-end coordinate regression.
Verification: both queued captures exited successfully and all 11 arrays were
compared; latest runtime core remains job 585, not admitted. No new release,
performance, ordinary-RNG, independent-review or push gate was completed.

Queued job 588 adds a diagnostic `--native-wout-input` interchange: the saved
native outer product is passed as a dynamic JIT operand and replaces only
Wout's input. It validates native full-OPM provenance, matching features,
checkpoint, reference and pair capture, archive hash, and lossless BF16 storage.
It remains separate from runtime behavior; materialized capture can change
kernel choices. Source `esmfold2-wout-control-source-20260908-yjSr3O`, output
`esmfold2-wout-control-result-20260908-Gm4kjm/jax`. Results pending.
Verification: 16 probe tests passed, including actual JIT input replacement;
scoped Ruff passed. No numerical Wout conclusion is yet established.

Additional actual-JIT negative tests reject mismatched interchange shape and
interchange outside full-OPM mode while retaining hook restoration. All 18
probe tests and scoped Ruff pass. Job 588 remains live and is writing its
capture archive; no partial archive was used for a numerical conclusion.

Job 588 completed successfully. All original bindings and both archive hashes
were revalidated. The native Wout input's 195,552,256 stored entries match
exactly after interchange. Wout output and post-division OPM output each have
48,888,064 entries with zero unequal values, zero max error and zero RMSE.
Thus the biased projection plus division reproduces native numerically on
this fixed native input in the captured diagnostic. The earlier 587 output
difference is not reproduced when its incoming outer-product difference is
removed; investigate earlier norm/projection/outer arithmetic next. This is
not proof that every shape, uninstrumented kernel choice, or downstream model
is equivalent, and job 585's coordinate regression remains unresolved.
Verification: queued 588 exit 0; archive and source/input/weight hashes intact;
three full arrays compared. No further runtime change or full-core rerun here.

CPU analysis of 586/587 norm outputs shows 6,076 differing BF16-rounded
elements in 6,076 rows (out of 57,278,464 elements). All 789 rows with any
first-W output difference lie within these rounding-different rows; there
are zero differing W outputs elsewhere. This localizes a strong lead but
does not independently prove the contraction's behavior on identical inputs.
Job 589 tests the existing CUDA Welford helper at width128 in the probe only,
without expanding runtime dispatch. The helper source is included in capture
bindings. Source `esmfold2-opm-norm128-source-20260908-pU4WCP`, output
`esmfold2-opm-norm128-result-20260908-hqC7rC/jax`. CUDA result pending.
Verification: 18 existing probe tests passed and scoped Ruff passed; the new
CUDA diagnostic branch is not covered by these CPU runs.

Job 589 completed successfully. Direct full-array comparison gives zero
unequal values, max error 0 and RMSE 0 for both the FP32 norm output and BF16
first-W output. This supports the width128 Welford control at this captured
site. Runtime dispatch is still unchanged; full OPM, uninstrumented core and
other width128 sites remain to be verified before broadening implementation.

Runtime `_autocast_norm` now selects the existing CUDA Welford implementation
for width128 `msa_encoder.blocks.*.outer_product_mean.norm`, in addition to
its existing width256 route. CPU/TPU/CP and unrelated width128 sites retain
their prior implementation. Probe hooks now capture the actual autocast norm
helper so the runtime CUDA branch is observable without replacing it.
Job 590 runs full first OPM with this runtime route, without norm or Wout
input substitution. Source `esmfold2-opm-runtime-source-20260908-V2kutR`,
output `esmfold2-opm-runtime-result-20260908-3eWvIr/jax`; result pending.
Verification: 46 affected tests passed and scoped Ruff passed. Full-core n=5,
other OPM sites and model admission remain unverified for this runtime change.

Queued full-core job 591 uses the same immutable runtime source as 590, fixed
tape, n=5, two forwards, native LM/shim and the previously documented FFI and
compiler profile. Output: `esmfold2-opm-runtime-result-20260908-3eWvIr/core`.
590 remains live; 591 waits behind it. Neither result is admitted yet.
The CUDA helper in both 589 and 590 snapshots has SHA256
`d246546fff666aa559cab9a227bd09687efd717ea0ced7b2a7878165609d79ab`.
590's prefix manifest omitted this cross-model helper; it is explicitly
checked here, and later probe captures bind it whenever native OPM is enabled.
Snapshots are not edited retroactively. Verification: 48 focused tests pass,
including width128 OPM CP fallback selection; scoped Ruff passes.

Job 590 completed. All manifest-bound files and archive hashes were
revalidated, with the separately recorded helper identity above. Comparing
all 11 arrays against native 586: embedding outputs, norm input values, FP32
norm output, W input/output, flattened outer product, Wout output and final
OPM output are numerically exact (zero unequal values, max error and RMSE).
The outer has 195,552,256 entries and final OPM output 48,888,064 entries.
Early embedding inputs retain their previously documented BF16-vs-FP32
differences; this is not an all-input dtype identity claim. Norm hook levels
also differ as documented. No norm or Wout input substitution was used.
This closes this first captured OPM boundary numerically, not the whole MSA,
all shapes or uninstrumented core. Job 591 remains live; no new structural
acceptance is inferred from this operator result.

The remaining released MSA PWA source uses norm_single, compute_bias norm
and Linear, masked softmax, Wv/Wgate, three-operand einsum and Wout. Candidate
still uses legacy early-BF16 affine/normalization there. The equation matches
at source level, but native per-operation dtype and contraction order still
need measured boundaries; do not silently broaden the OPM result to PWA.

Full-core job 591 completed. Its `comparison.json` has protein entity RMSD
`[0.738681433, 0.062482652, 0.063138823, 0.148277485, 0.085770883]` and ligand
`[0.286840194, 0.022815883, 0.076416589, 0.036495971, 0.037826646]` angstrom,
in original sample order after one whole-system Kabsch each. Maxima improve
versus 585 (0.923487/0.305537), but remain worse than 576's 0.314695/0.125691.
The operator-exact first OPM therefore does not close end-to-end fidelity;
there is no monotonic improvement claim for individual samples. Both repeated
coordinate and confidence archives are finite and storage-byte-identical.
Coordinate gate still fails. Verification: queued run and report generation
exit 0; no full-model, performance, ordinary-RNG or release/push admission.
Strict confidence also fails at unchanged atol/rtol 1e-4; original output
dtypes agree and raw-head retention is complete.

## Resumed PWA source audit after bounded Boltz investigation

Current embedders.msa_block passes native_opm_params into outer_product_mean,
but the following msa_pair_weighted_averaging still receives converted params.
The PWA helper has no native-autocast selector: both norms use layer_norm,
which returns its input dtype, and the affine weights are those supplied by
the converted parameter mapping. In contrast, the pinned publisher class in
modeling_esmfold2_common.py uses nn.LayerNorm then separate Linear, softmax,
Wv/Wgate and the three-operand einsum inside the caller's execution context.
Do not infer the CUDA autocast leaf dtypes from this source alone: capture
the actual PWA boundaries next, retaining original FP32 parameters as a
separate candidate arm before changing runtime defaults.

Existing test_trunk_parity.test_msa_pair_weighted_averaging_matches uses small
FP32 inputs and does not cover this native mixed-precision route. Existing
pair_trunk_autocast_wiring tests prove original mappings are available upstream
but do not establish that PWA consumes them. No PWA runtime fix or numerical
closure is claimed by this audit; Boltz remains unresolved, not admitted.

735 extends the existing native first-MSA prefix observer through first PWA.
--full-pwa requires native --full-opm, loads six PWA norm/linear modules from
the checkpoint, observes their inputs/outputs and stops after PWA before later
unmaterialized modules. Hook tensors are cloned because masked_fill_ would
otherwise mutate captured bias projection values. Expected capture count27
includes the previous11 OPM arrays plus12 PWA leaf arrays and4 PWA boundaries.
This remains a controlled native prefix, not an uninstrumented full prediction.
Existing18 prefix tests and Ruff pass; new native branch execution and dtype
results remain pending. No model runtime default has changed.

735 completes exit0. Report/archive hashes and every input/source binding
revalidate; all11 previous OPM arrays equal586 numerically. Actual native PWA
norm_single and compute_bias.0 take BF16 inputs and return FP32. Bias Linear,
Wv and Wgate take FP32 and return BF16. Wout input/output and final PWA output
are BF16; pair mask is bool. These measured boundaries confirm the current
candidate's early narrowing is not the native LayerNorm policy. Artifact:
esmfold2-pwa-native-7KhRbt/native/{prefix.npz,report.json}.

Next candidate must retain original affine weights and FP32 norm outputs,
then narrow only at native Linear boundaries. Softmax/einsum internal dtypes
are not separately captured by735 and must not be inferred from Wout's BF16
input alone. Preserve the FP32 and CP fallback contracts and measure both
same-operand PWA and propagated first-MSA/full-core outcomes before admission.

The full-pwa observer now also records softmax input/output and the three
einsum operands/output, scoped to the first PWA forward. It calls the original
functions with unchanged arguments, clones observations and restores patched
functions on exit. Capture count increases from27 to33. A new CPU regression
checks forwarding, output identity, six observations and restoration;19 prefix
tests and Ruff pass. This expanded observer has not yet run on the GPU;735
remains the prior27-array evidence and is not reinterpreted as an internal-op
capture. No runtime numerical policy was changed.

736 runs the33-array native observer from esmfold2-pwa-ops-eNRy93, retaining
735 inputs and native source. Queue inspection confirms it running after a
GPU availability check. The observer restoration regression now also raises
inside the wrapped native call and verifies both functions are restored;
20 prefix tests, scoped Ruff and diff checks pass. This test-only addition
postdates the execution snapshot. Native internal-op results remain pending.

736 completed exit0. All source/input bindings and archive digest reverify;
all27 arrays from735 remain numerically equal after internal observation.
Native softmax takes BF16 and returns FP32. Einsum receives attention FP32,
value BF16 and gate BF16, returning BF16. This adds a second confirmed dtype
boundary absent from the legacy candidate: its BF16 bias softmax is not
explicitly promoted to FP32. Einsum output dtype alone does not determine its
internal pairwise contraction order or intermediate rounding, so compare the
captured same-operand contraction before selecting that implementation.
Artifact: esmfold2-pwa-ops-eNRy93/native. No default change or model admission.

737 replays actual736 einsum operands with three JAX contraction arms.
Narrow attention to BF16, contract attention/value, round the intermediate to
BF16, then multiply BF16 gate: native output and same-JIT repeat are bitwise
equal. Keeping FP32 attention through the sum then rounding gives RMSE
0.00011149388652584065 (6,169,308 unequal values); mixed three-operand JAX
einsum then final BF16 gives RMSE0.00014667710598682213 (16,890,916 unequal).
Both have maximum0.0078125, all repeats exact. Thus the measured native
contraction requires intermediate narrowing; indiscriminate FP32 promotion
does not reproduce it. These are representation units on one captured first
PWA, not a full-model result. Report: esmfold2-pwa-contraction-EhvMUP/report.json.
Verification:737 exit0; replay validates all bindings before/after execution,
Ruff passes. Next implementation can now encode the measured norm/softmax
FP32 islands and BF16 contraction boundaries, retaining separate regression
and propagated-output gates.

Runtime candidate now implements native_autocast PWA: original affine mapping
from msa_block, FP32 norm outputs, native-autocast Linear, FP32 softmax, then
BF16 attention/value contraction with intermediate narrowing before gate.
The selector follows existing native_opm_params availability, retaining the
previous non-native FP32/CP route. No external Torch/FFI dependency is added.
New eager/JIT CPU tests assert original mapping identity, FP32 projection
inputs, BF16 Wout input/output and finite fully-masked rows.30 focused tests
pass, scoped Ruff and diff checks pass. This is an implemented candidate,
not admitted: real captured PWA output, complete MSA, full n5 structure and
confidence, wider CPU regressions and performance remain unverified.

738 evaluates the actual runtime PWA on native736 msa/pair/mask with verified
original checkpoint weights, comparing legacy BF16 parameters with the new
native-autocast path. Legacy output RMSE0.0017341445793144243,
max0.03125, 37,779,494 unequal values; candidate RMSE0.0007502226970332566,
max0.015625, 17,247,246 unequal values. Both repeats are bitwise equal.
The measured dtype repair reduces RMSE by about57%, but does not close this
PWA. Remaining norm/Linear/softmax arithmetic boundaries need isolation before
claiming full-MSA or coordinate improvement. Report:
esmfold2-pwa-runtime-JL4ZdY/report.json. Verification:738 exit0, replay checks
native/source/weight bindings before and after, scoped Ruff passes. This is
same-operand runtime evidence only, not independent preprocessing or n5 parity.

739 isolates actual736 PWA leaf inputs with current runtime helpers. Wv,
Wgate, Wout, compute_bias.1 and compute_bias.0 normalization all equal native
numerically (zero unequal values). norm_single retains RMSE7.92805056558966e-8,
max2.86102294921875e-6, 24,454,781 unequal values; softmax retains
RMSE6.558018584204104e-10, max1.1920928955078125e-7,633,327 unequal.
Every same-JIT repeat is bitwise equal. Artifact:
esmfold2-pwa-leaves-LVzLWe/report.json. Verification:739 exit0; source/input
bindings are checked before/after by the probe, scoped Ruff passes. This
rules out a same-operand discrepancy in those five leaves on this input, not
in propagated inputs or every shape. Next discriminating intervention is
the MSA norm_single reduction, separately from softmax; do not conflate their
small standalone errors with proven full-PWA or coordinate causation.

740 isolates native width128 Welford norm_single in the same-operand PWA
probe. The norm leaf becomes bitwise equal to native, but PWA output remains
RMSE0.0007502226961579614, max0.015625,17,247,235 unequal values, essentially
unchanged versus738/739 RMSE0.0007502226970332566. Both repeats are exact.
Therefore ordinary norm reduction is not the dominant remaining error in this
PWA control; do not promote the additional norm kernel for an unmeasured
full-model benefit. Softmax/rounding/fusion across leaves remain to isolate.
Artifact: esmfold2-pwa-norm-5EUv0j/report.json. Verification:740 exit0, probe
checks all bindings before/after and scoped Ruff passes. Runtime norm routing
is unchanged; only the development process used the override.

741 isolates gate sigmoid with native Wgate output. Direct BF16 sigmoid
differs in9,790,127 values, RMSE1.4815318198515759e-5, max0.0009765625.
FP32 sigmoid then BF16 matches native gate bitwise, as do same-JIT repeats.
Artifact: esmfold2-pwa-gate-xyhz5x/report.json;741 exit0 and before/after
binding checks pass. Runtime native PWA now promotes gate logits for sigmoid
and narrows the result back to the projection dtype; non-native behavior is
retained. Eager/JIT tests assert FP32 sigmoid input, with30 focused tests,
scoped Ruff and diff checks passing. Whole-PWA improvement from this latest
change, full-MSA and n5 coordinates/confidence still require measurement.

742 evaluates the actual runtime FP32-sigmoid correction with no norm override.
Whole same-operand PWA RMSE falls to2.7090343749546737e-6, max0.00390625,
1,919 unequal values, versus738 RMSE0.0007502226970332566 and17,247,246
unequal values. The gate boundary is therefore a dominant contributor in this
controlled PWA, not merely a coincident isolated difference. Same-JIT repeat
is bitwise equal. Report: esmfold2-pwa-gate-runtime-B5VHAE/report.json.

743 reuses the exact742 snapshot and adds only the native width128 MSA norm
override: PWA RMSE6.82190474624215e-7, max0.001953125,120 unequal values;
repeat exact. Report: same directory/norm-report.json. Norm contribution is
visible after repairing gate precision;740's near-null aggregate change did
not establish that norm never mattered. Both runs exit0 and validate source/
input bindings before/after. No captured tensor is substituted into the PWA
body. Native norm remains diagnostic-only; full propagated MSA, n5 outputs,
confidence and performance remain unverified for the current runtime repair.

744 runs current runtime repair through full n5/two-forward replay using591's
features, tape, native LM/shim and diagnostic FFI/compiler profile, fresh
snapshot esmfold2-pwa-core-YzdRUX and output core. No extra MSA norm override
is used. This remains downstream/core diagnostic, not standalone admission.
A concurrent full tests/models/esmfold2 CPU run has reported failures during
progress; final failure traces and totals are pending. Do not describe the
expanded regression gate as passing on the basis of30 focused tests.

744 failed exit1 with missing PWA norm_single.weight. The expanded CPU gate
finished6 failed,376 passed,24 skipped; all six listed failures hit the same
missing PWA weight at native normalization. Root cause: model.predict still
filtered msa_opm_params to OPM keys only, although the revised block now
consumes that mapping for PWA too. The mapping filter now includes PWA keys,
and the top-level original-weight wiring test asserts both families and PWA
weight object identity. Ten focused wiring/PWA tests and scoped Ruff/diff
checks pass. A new full CPU run is in progress; no repaired full GPU outcome
exists yet. The failed744 snapshot/output is preserved, not overwritten.

745 reruns full n5/two-forward native-tape replay from a fresh snapshot after
the top-level PWA parameter-filter fix. It retains744/591 native LM/shim,
FFI and compiler controls and a fresh output/cache. Queue inspection confirms
745 running. The full CPU rerun remains active beyond its earlier failing
checkpoint tests; no passing final gate or coordinate result is claimed yet.

The repaired full CPU model suite completes382 passed,24 skipped in134.67s;
all six prior missing-PWA-key failures are resolved. Scoped Ruff also passes.
745 completes exit0 and the artifact-verifying report succeeds. Protein RMSDs
are [0.2233530154088353,0.09042856059041036,0.07921760991202118,
0.1795420487110124,0.09242410194764374]; ligand RMSDs are
[0.08468402148947013,0.012307336058549055,0.054218301341220104,
0.02899323969598975,0.018451656624990768] angstrom, one whole-system Kabsch
per sample without entity refit. Strict confidence still fails. Protein/ligand
maxima improve versus591 (0.738681/0.286840), but protein samples0/3 exceed
0.1 A and this remains not admitted. Report:
esmfold2-pwa-core-fixed-SgiPCo/comparison.json. Native LM/shim and FFI controls
remain, so this is not standalone production or performance evidence.

The native norm route now includes width128
msa_encoder.blocks.*.msa_pair_weighted_averaging.norm_single, using the
existing Welford helper verified by740/743. Other width128 sites and CP/CPU
fallbacks retain their prior routes. Expanded route-selection tests plus PWA
and wiring checks pass40 tests; scoped Ruff/diff checks pass.746 runs full
n5/two-forward comparison with the same745 diagnostic controls, now selecting
the norm through runtime code rather than a monkeypatch. Its final coordinate
and confidence result is pending; the earlier382-test CPU gate predates this
small routing change and is not represented as a final-tree gate.

746 completes exit0 and artifact-verifying report succeeds. Protein RMSDs:
[0.6217764754006075,0.06993555709798557,0.14118268676165954,
0.1774906781449936,0.12293282240908084]; ligand:
[0.30225414179207316,0.020570406077286885,0.07509702285544499,
0.03127379898498135,0.026517191800572534] angstrom. Strict confidence fails.
This is worse than745's maxima0.223353/0.084684 despite improved isolated
PWA equality. Current norm-routing candidate is NOT admitted. Source snapshots
and separate fresh caches are retained; do not call this monotonic fidelity
progress or infer that exact normalization itself is incorrect. Need matched
propagated-boundary/repeat controls to distinguish remaining-path amplification
from executable-selection changes. Report:
esmfold2-pwa-norm-core-2mZVe3/comparison.json. No push or performance claim.

745/746 repeat audit directly rereads both coordinate and confidence archives:
each run's repeat-1 is finite and storage-byte-identical to its own first
forward, for every retained confidence leaf. Config, input/tape hashes,
samples, seed, runtime versions, precision, native policy and LM arm match.
JAX policy differs only in snapshot path keys of ffi_bindings; corresponding
source/library digest values match. Runtime source-tree diff shows only
esmfold2/models/trunk.py changed (apart from generated bytecode), containing
the recorded MSA norm routing change. HLO hashes differ as expected. This
rules out observed same-executable random variability in these two repeats,
not between-compilation variability or an unsupported universal noise floor.
Next causal probe should capture propagated MSA/pair boundaries under both
routes before attributing the final coordinate change to a single exact leaf.

747 extends candidate_prefix through actual embedding/OPM/PWA, returning27
boundaries without substituting captured PWA operands. Shared native initial
embedding/pair remain explicitly part of this prefix diagnostic. Full-PWA JAX
requires native OPM params and no manual OPM norm override. Norm hook names
now follow actual prefixes; prior norm input observations still represent the
FP32 helper boundary, not necessarily the native module's incoming dtype.
Existing20 prefix tests, scoped Ruff/diff checks pass. New propagated GPU
branch is submitted via tsp; successful execution and native comparison remain
pending. This probe does not yet capture the complete MSA or full recurrence.

747 completed exit0. All report bindings and archive digest reverify. Actual
embedding/project_inputs outputs, full first OPM output and its norm/W/Wout
outputs equal native736 numerically. PWA msa_input and pair_input also equal;
both PWA norm outputs and Wv/Wgate/bias projection outputs equal. PWA final
Wout/output retains exactly the same120 differing values as743, RMSE
6.82190474624215e-7 and max0.001953125. This bridges the isolated PWA result
into the actual first-MSA prefix without injecting PWA operands. Artifact:
esmfold2-pwa-propagated-3yIhah/jax. No first-PWA input-generation discrepancy
is observed here; subsequent MSA-side transition, pair updates, later blocks
and recurrence remain unobserved. Do not generalize first-prefix equality to
the full run, whose746 structure/confidence gates remain failed.
# First MSA transition capture follow-up

### Native first-loop injection observer, job 786

Job 787 completed exit 0 (242.65 seconds). Bound bridge/core/boundary report
`esmfold2-jax-injection-54RDUF/injection-comparison.json` passed. Its coordinate
array has identical dtype/shape/storage bytes to unobserved diagnostic 785,
so this observer retains the existing full-output failure.

| First-loop boundary | Native/JAX dtype | RMSE | Max absolute |
| --- | --- | ---: | ---: |
| Refined LM output | BF16/BF16 | 0 | 0 |
| MSA output | BF16/BF16 | 5.96168148336 | 256 |
| Injection norm input | BF16/BF16 | 6.59518301523 | 256 |
| Injection norm output | FP32/FP32 | 0.00552543440228 | 0.148807287217 |
| Folding-trunk input after recurrence | BF16/BF16 | 0.0861549074129 | 5 |

These are activation units. The shared-native-input standalone MSA stack 784
was exact, but the MSA path in the actual full execution is not. Do not
generalize that earlier finite closure to this path. The earliest currently
observed difference is MSA output, not LM output or injection norm itself.
Next discriminate actual MSA entry pair/input embedding against the native
controls before changing downstream normalization or recurrence arithmetic.

Added `bench.esmfold2_injection_report`: first validates the existing full
candidate/reference report, then explicitly bridges the new native capture
via matching config/input/tape/checkpoint/native-source and exact tape/feature/
LM archive hashes. Five boundary arrays require matching nonempty rank-4
shapes, FP32 storage and finite values; original dtypes are retained separately.
Missing policy records remain explicit, and model admission is always false.
Boundary/observer/core-report tests: 22 passed; Ruff/diff passed. Job 787 is
confirmed running with injection archive saving; final comparison is pending.

Native 786 completed exit 0 in 204.56 seconds. `injection.npz` SHA256
`9d9bc6fb0e1993e12d98b3c1f5c9c8fa0f749f06d0ab2a2ebf00c86b42da609a`
verified; all four hook counts equal four. MSA/LM/injection input/trunk input
are BF16; injection output FP32. Tape, LM and feature archive hashes exactly
match the older `esmfold2-lm-control-20260907-lPw24m/native` reference, and
checkpoint/native-source identities match. A direct policy-key comparison
raised KeyError because the older manifest lacks `native_policy`; policy
identity is not claimed from the absent field. JAX job 787 is submitted using
the existing reference-bound LM/shim/FFI controls and new injection observer.
Matched JAX boundary results remain pending. Observer/repeat/report tests now
32 passed, including partial native registration and JAX exception cleanup;
Ruff/diff checks passed.

Matching JAX `--capture-injection` is now wired into the actual replay:
it wraps the existing `run_loops` scan body, preserves its carry/output, and
returns first-loop boundary arrays separately. It does not reimplement
recurrence arithmetic. The diagnostic observes MSA/LM outputs, injection norm
input/output and folding-trunk input; missing routes or multiple target scans
fail explicitly. Eager/JIT toy tests confirm original result preservation and
first-loop values. Combined observer/repeat tests: 21 passed; Ruff passed after
correcting a test-only lambda style violation. Real JAX capture is not yet run.
Native 786 remains active, with injection/output archives present but completion
metadata pending. Compare tape/LM identity before selecting the paired replay;
do not assume old shim controls bind a new native manifest automatically.

Added optional development capture `--capture-injection` to the native tape
runner. It clones the first MSA and refined-LM outputs, injection norm input/
output, and folding-trunk input (post recurrence), preserving hook returns.
All four modules must be active and observed once per configured loop; their
counts and actual dtypes accompany a hashed `injection.npz` in metadata.
Hooks are removed on success or failure, and the flag defaults off. Tests for
first-only retention, clone isolation, disabled behavior and repeat-forward
regression: 19 passed; scoped Ruff/diff passed. Job 786 is submitted to capture
these boundaries alongside native full outputs and actual RNG tape. Numerical
results and matching JAX injection instrumentation are not yet complete.

### Embedding boundary fix, jobs 783/784

Job 785 repeat inspection: main versus `repeat-1` coordinate archive has the
same one-array schema and identical storage bytes; all 14 raw confidence
arrays also have identical shapes/dtypes/storage bytes. This establishes
same-executable repeatability for this diagnostic, not fresh compilation or
uninstrumented native repeatability. Read-only source comparison of native
`modeling_esmfold2.py:827-846` and FoldJAX `model.py:864-909` confirms the
subsequent routing: MSA result/overwrite, refined LM addition, injection norm,
recurrence, folding trunk. No numerical identity is inferred from that source
ordering. The next localization target is first-loop injection boundaries,
keeping MSA/LM outputs separately observable; those captures are not yet run.

Full diagnostic job 785 completed exit 0 in 146.71 seconds. Bound report:
`esmfold2-runtime-embedding-boundaries-oRs0hx/core-comparison.json`.
Global-system Kabsch followed by entity measurement, in angstroms:

| Entity | Sample 0 | Sample 1 | Sample 2 | Sample 3 | Sample 4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Protein | 0.671718931 | 0.070241084 | 0.081766407 | 0.163999945 | 0.088531148 |
| Ligand | 0.278609212 | 0.027282631 | 0.027031093 | 0.032740189 | 0.031822792 |

Compared with pre-boundary job 770, maxima improve from protein 1.752230028
and ligand 0.521932936, but structure and strict confidence still fail.
For example pTM-family ipTM maximum absolute difference is 0.00612282753
and complex pLDDT 0.000472307205. This does not admit the model or optional
precision lowering. Remaining differences require downstream trunk/recycle/
diffusion localization; exact first-loop MSA outputs do not establish those
boundaries in the full compiled graph. No commit/push or performance claim.

The bound `stack-comparison.json` for 784 completed: **all eight outputs
(MSA and pair for blocks 0, 1, 2, 3) have zero RMSE, zero maximum difference
and zero unequal values versus native 771**. The actual runtime fix thus
closes this first-loop shared-input MSA-stack discrepancy without benchmark
barrier override. This finite diagnostic does not establish later-loop/full
model structure, confidence, independent preprocessing, or performance gates.

Actual-runtime full stack job 784 completed exit 0 in 176.38 seconds.
ESMFold2 CPU model suite after the runtime change: **402 passed, 24 skipped
in 131.55 seconds**. Job 785 is running full fixed-tape n=5, repeat-forwards=2
from the same `esmfold2-runtime-embedding-boundaries-oRs0hx` snapshot, retaining
the previous native LM/shim/linear-FFI diagnostic conditions. It is not an
independent standalone-JAX admission run. Structure/confidence result pending.

Job 783 changed only two embedding output optimization barriers relative to
782, keeping the entry/exit observer. It completed exit 0 (63.93 seconds),
artifact `esmfold2-embedding-barriers-LBFGZG/jax`. Archive/all binding checks
passed. Entry MSA and both first-block outputs now equal native exactly
(RMSE/max zero); entry pair remains equal to the no-barrier control.
This controlled intervention supports preserving the two native BF16 Linear
output boundaries, without attributing the mechanism solely to GEMM algorithm
indices or a specific fusion pass.

Applied these barriers in actual `embedders.msa_encoder`, only when the
native-autocast parameter path is active. Default non-native behavior retains
the original arithmetic. Regression checks cover both policies and boundary
ordering: 18 affected model tests passed; benchmark probe suite 47 passed;
scoped Ruff/diff pass. Full-stack job 784 is submitted from actual updated
runtime with no benchmark barrier override. Its numerical result and the
full-model n=5 structure/confidence retest remain pending. No admission/push.

### First-block entry capture, job 782

Benchmark-only `--stack-inputs` records first-block MSA/pair entry values
without internal leaf outputs. Tests cover both traversal modes and restoration:
46 passed; scoped Ruff/diff pass. Job 782 completed exit 0 in 63.94 seconds,
artifact `esmfold2-block-inputs-VPEW6P/jax`.

Revalidated archives and all bindings for 782, native leaf 757, native stack
771 and first-only 778. Entry MSA versus native OPM norm input already differs:
RMSE **0.00040279752532022264**, maximum **0.0625**. Both block exit arrays
are exactly equal numerically to 778, while retaining native output RMSE
MSA 0.112671750775 and pair 0.137248004073. This observer therefore preserves
the previously observed first-block failure while exposing an upstream
discrepancy. Investigate MSA embedding/project-input addition before blaming
OPM internal arithmetic. Entry pair has been saved but not yet independently
compared in this check. These are activation diagnostics, not model admission.

### Same-operand biased OPM output projection, jobs 780/781

Extended the development replay with `--opm-wout`, using captured native
Wout input/output and the bound checkpoint bias. Job 780 failed before arm
execution because the selected older FFI library lacked the FP32 symbol.
Verified both symbols with `nm -D` in the existing `native-fp32-ffi-fs5gCu`
build; job 781 then completed exit 0 in 11.53 seconds with that library.
Report: `esmfold2-opm-wout-replay-SxR4U0/report.json`.

| Same-operand arm | Native RMSE | Max absolute | Unequal values |
| --- | ---: | ---: | ---: |
| Current runtime biased linear | 0 | 0 | 0 |
| BF16 matmul materialized before bias | 0.0390952072124 | 4 | 11,192,980 |
| FP32 FFI matmul on BF16-rounded operands, then bias | 0.00113772426381 | 4 | 21,310 |

All same-executable repeats are bitwise equal. Units are activations, not
angstroms. The current standalone runtime matches native; replacing it with
either counterfactual would worsen this test. Neither this nor embedding 779
recreates the full-graph GEMM selection/operand context. No runtime replacement
is justified yet. Reference/weight/source/library hashes checked by replay;
Ruff, CLI import smoke and diff checks passed. Dedicated replay unit coverage
remains pending, and no full-model or performance admission is claimed.

### Same-operand MSA embedding replay, job 779

New development-only `bench.esmfold2_embedding_replay` validates native capture
bindings and checkpoint identity, then compares BF16-destination matmul,
FP32 accumulation followed by BF16 storage, and native cuBLAS FFI, each twice
on the same executable. No installable runtime code changed.
Job 779 completed exit 0 in 9.53 seconds; report
`esmfold2-embedding-replay-5VJBTH/report.json`. All three arms are bitwise
identical to captured native embedding output, and all repeats are bitwise
equal. This standalone replay does not recreate the differing full-graph
algorithm choices and therefore cannot rule out an in-context embedding
difference. It does establish that none of these isolated accumulation paths
reproduces the failure. The biased OPM output projection remains a separate
target: current bias-free FFI deliberately bypasses it. CLI/import smoke,
Ruff and diff checks pass; broader benchmark regression coverage is pending.

### First-block-only graph control, job 778

Completed exit 0 in 40.42 seconds. Revalidated archive and every report binding
for 778, native 771 and HLO stack 776. Both 778 arrays are numerically identical
to block 0 of 776 (zero RMSE/max), while native differences remain MSA
0.112671750775 and pair 0.137248004073. Removing downstream blocks therefore
does not remove this discrepancy.

Independent read-only OMS consultation completed and recommended comparing
existing GEMM configs before more GPU runs. Parent HLO inspection found:
MSA embedding result BF16[447488,128], strides 15662080/4480, has
`selected_algorithm=1` in leaf 777 and `0` in stack 776; the first OPM output
projection result FP32[190969,256], strides 195552256/262144, likewise has
1 versus 0. Algorithm indices are not assumed globally comparable identities;
this is evidence to prioritize same-operand contraction replay, not proof of
causality. Both calls retain the same listed output layouts and 32 MiB
autotune workspace. The review did not inspect external artifacts or pinned
native code and is not release approval. Its assertion that a changed config
alone confirms cause is not adopted. No runtime fix or admission yet.

Added benchmark-only `--stack-first-only`, restricted to JAX stack capture.
It captures the two outputs of the actual first block then stops tracing;
the configured layer count and first block's `is_final=False` remain intact.
This does not make block 0 a final block or add any internal leaf output.
Full-stack coverage checks must not admit this two-output diagnostic as a
complete stack. Tests cover first-only/full traversal, unchanged final-block
semantics and hook restoration on errors: 42 passed, scoped Ruff/diff passed.
Job 778 is submitted with unchanged runtime, strict compiler profile, FFI and
HLO capture. Its outcome is pending. This separates downstream-graph presence
from the extra internal outputs of the leaf observer.

### Executed-HLO capture, job 776

Paired leaf job 777 completed (exit 0, 374.38 seconds). The bound report
`esmfold2-stack-hlo-DW1qUj/leaf-comparison.json` verifies all 17 captured
`.output` arrays bitwise equal to native 757, preserving the earlier leaf
result under explicit compiled-executable execution. Compiled HLO SHA256:
`c96cc696981ed207e7eb84c0ecc1779bfea56596473be526c1cb8b5660a490a6`;
lowered: `c49b6915f58cd226e476d960b338e227b77d51a554dd0ec6fd97fcc2763141b7`.
An independent read-only Claude consultation on harness/FFI versus compiler
hypotheses has been requested through OMS; its response remains pending.
No new runtime change or model admission follows from this capture.

While job 777 is still saving outputs, its already-written compiled HLO
provides a counterexample to the simple "only the stack fuses OPM residual"
hypothesis: leaf `fused_add_convert` (lines 1281-1296) also contains projection
FP32-to-BF16 conversion, FP32 division, BF16 conversion, residual addition and
FP32 conversion. Stack `fused_add_convert.4` has that same arithmetic sequence.
The leaf additionally returns projected BF16 and divided BF16 values, whereas
the stack returns only the residual and its FP32 view. Presence of this fusion
therefore does not identify the cause. Extra output materialization and earlier
operand differences remain distinguishable hypotheses; emitted rounding has
not been inspected. Job 777's final numeric report remains pending.

Completed with exit 0 in 175.09 seconds. Bound comparison report
`esmfold2-stack-hlo-DW1qUj/comparison.json` validates all eight boundaries and
HLO hashes. All eight native-comparison metrics reproduce job 772 exactly
(not a direct JAX-array byte comparison). Compiled HLO SHA256:
`223757b456a10def8fbbfe569f0f45881dd925cd9535359da7e1e216e4234c0d`;
lowered HLO: `612d2346277d4237dfe7e687384431018c6bd0cf0e4bfcb7c11a6ca0115cfd11`.
Job 777 now runs the full first-block leaf capture with HLO on the same
snapshot/strict settings/FFI, output `esmfold2-stack-hlo-DW1qUj/leaf`.
That paired graph comparison remains pending; no causal/runtime claim yet.

Added benchmark-only `--capture-hlo`: saves lowered and compiled text and
executes that compiled executable, binding both files by SHA256 in the final
report. Default execution remains direct JIT invocation. Mocked executable
identity/default-path tests bring the focused probe suite to 40 passed; Ruff
and diff checks pass. Runtime remains the unchanged 772 snapshot.
Job 776 is confirmed running under `esmfold2-stack-hlo-DW1qUj`; compiled HLO
is already present, but final output/hash comparison is pending.

Initial HLO inspection shows an OPM division, BF16 conversion, residual add
and FP32 conversion grouped in `fused_add_convert.4` (lines 1249-1264 of the
compiled text). The BF16 conversions are present in HLO; this alone neither
proves their emitted-machine-code rounding nor identifies a faulty operation.
The next comparison must use the finished report and localize against the
matching leaf graph rather than treating this fusion's presence as a cause.

### Fusion-pass counterfactual submitted, job 775

Completed with exit 0 in 175.24 seconds. The bound stack report
`esmfold2-stack-no-fusion-ZX3Pzm/comparison.json` passed hash and full 8-boundary
coverage checks. Every boundary's native-comparison metrics are identical to
job 772: block-0 MSA RMSE 0.112671750775 and pair RMSE 0.137248004073;
later block metrics are unchanged too. This compares reported metrics, not
direct byte equality between the two JAX arrays. The report confirms the
disabled-pass string ends in `,fusion`, with excess precision false and Triton
GEMM false. This intervention provides no improvement. Without optimized-HLO
inspection it cannot exclude other fusion passes or prove that the named pass
was relevant. Next localization needs concrete intermediate boundaries/HLO,
not another speculative global compiler flag. Added pre-I/O rejection tests
for native and non-stack misuse: 38 probe tests pass; Ruff/diff checks pass.

Added a benchmark-only `--disable-fusion` option, restricted to JAX stack
capture. It appends `fusion` to the strict profile's disabled HLO passes;
all existing strict options remain unchanged and are now recorded in the
capture report. This requests a compiler control, not a claim that every
backend fusion has been eliminated. Runtime source is copied unchanged from
job 772. Job 775 is confirmed running; external snapshot/output root is
`esmfold2-stack-no-fusion-ZX3Pzm`. No numerical outcome is available yet.
Focused probe tests: 36 passed. Scoped Ruff and diff checks passed. This is
not a production performance policy or model-admission change.

### Same-source leaf versus stack control, job 774

Job 773 failed argument preflight before computation because hierarchical
capture flags were missing. Job 774 supplied all required flags and completed
with exit 0 in 376 seconds, using job 772's immutable source, native-autocast
policy, inputs and native linear FFI. Artifact:
`esmfold2-same-source-leaves-Okir5Q/jax`.

Archive hashes and every binding for 774, 772 and native 771 were checked;
all 60 shared bindings between the two JAX captures are identical. CPU BF16
final-residual reconstruction from 774 yields:

| Output | Target | RMSE | Max absolute | Unequal values |
| --- | --- | ---: | ---: | ---: |
| MSA | Native stack 771 | 0 | 0 | 0 |
| Pair | Native stack 771 | 0 | 0 | 0 |
| MSA | JAX stack 772 | 0.112671750775 | 16 | 1,101,178 |
| Pair | JAX stack 772 | 0.137248004073 | 32 | 1,087,843 |

This removes the older softmax source difference as an explanation for this
leaf-versus-stack discrepancy. It does not yet distinguish output-driven
compiler choices, intermediate materialization/rounding, or other capture
context effects. Both executions already use the strict-rounding compiler
profile (excess precision disabled); that option alone is not sufficient.
The leaf reconstruction is still not an uninstrumented model result or a
runtime fix. No structure/confidence admission or push is claimed.

The bound full-prefix report also completed against native 757:
`esmfold2-same-source-leaves-Okir5Q/comparison.json`. All 17 captured keys
ending in `.output` are bitwise equal, including both embedding projections,
OPM, PWA, MSA transition, both triangle updates and pair transition. This
statement covers captured outputs only, not all input dtypes or full-model
execution. Verification: report binding checks and `git diff --check` passed.

### CPU residual reconstruction control, 2026-09-09

Revalidated archive SHA256 and every report binding before reading the saved
arrays. Reconstructed each first-block final residual as FP32 input plus FP32
stored update, rounded to BF16 and returned to FP32 storage. This is a CPU
boundary diagnostic, not a replacement model execution.

Native job 757 (`esmfold2-msa-block-native-pEfpO3/native`) reconstructed MSA
and pair outputs both match job 771 (`esmfold2-msa-stack-h9NHsN/native`)
exactly: zero unequal values, RMSE and maximum absolute difference zero.
Thus these native leaf-versus-stack captures have consistent final residuals.

Reconstruction from JAX job 766 (`esmfold2-msa-propagated-fixed-3kqWEF/jax`):

| Output | Target | RMSE | Max absolute | Unequal values |
| --- | --- | ---: | ---: | ---: |
| MSA | Native stack 771 | 0.000711136851305 | 4 | 115 |
| Pair | Native stack 771 | 0 | 0 | 0 |
| MSA | JAX stack 772 | 0.112669900331 | 16 | 1,101,196 |
| Pair | JAX stack 772 | 0.137248004073 | 32 | 1,087,843 |

These are activation units, not angstroms. Job 766 predates the native-softmax
change while 772 includes it, and the capture graphs differ. Consequently this
does not isolate residual fusion as the cause. It narrows the next control to
same-source JAX first-block leaf versus stack execution, with residual
materialization examined only after accounting for that source difference.
No runtime policy change, GPU rerun, model admission or push follows from this
CPU diagnostic.

Jobs 771/772 completed with exit 0 and the bound stack report passed all
coverage/hash checks: external `esmfold2-msa-stack-h9NHsN/comparison.json`.
Post-residual MSA/pair activation RMSE by block (zero-indexed):
0 = 0.112671750775 / 0.137248004073;
1 = 0.857195514157 / 0.370964734405;
2 = 205.689159369 / 1.089342721066;
3 = 205.689159369 / 3.329650724855.
The final block omits the MSA update, consistent with unchanged MSA values.
All numbers are internal activation units, not coordinates or acceptance gates.

This stack capture already differs at block 0; previously exact submodule
updates cannot be extended to its post-residual result or to this different
instrumented graph. The next discriminating probe must separate residual
boundaries and capture/compiler context at block 0 before blaming later blocks.
No cause is established and no additional runtime edit is made. First-loop
shared native inputs/FFI remain diagnostic controls, not complete input or
standalone parity. No model admission or push.

Prefix reporting now supports `--stack`, requiring stack-mode reports and
matching every native block output key in the candidate rather than silently
comparing an incomplete subset. It retains hash/schema/shape checks and an
explicit instrumented/shared-native-start scope. New complete/missing-block
coverage tests bring report tests to four passing cases; Ruff passes. Jobs
771/772 remain running/queued at this check; no stack numbers are reported yet.

Started stack-level localization: job 771 captures native and job 772 captures
JAX post-residual MSA/pair outputs from all four first-loop blocks. The new
`--stack` mode is mutually exclusive with prefix intervention controls; JAX
requires native autocast and retains the diagnostic FFI path. Initial embedding,
pair and taped MSA features remain the same bound native controls. This tests
propagation across blocks, not whole-model/recycle parity, and returned arrays
still instrument the compiled graph. Runtime model code is unchanged this step.
Added stack propagation and exception-restoration tests; prefix suite now
passes 35 cases, Ruff passes. Native capture is running and candidate is queued;
no stack numerical result or admission is claimed yet.

Job 770 completed (exit 0), but full parity fails. Protein RMSDs:
[1.7522300281, 0.0869483313, 0.1011047728, 0.1774731182, 0.1527351566];
ligand RMSDs: [0.5219329364, 0.0224343519, 0.0420131560, 0.0224530068,
0.0218163543] angstroms. Strict confidence is false. Evidence: external
`esmfold2-native-softmax-core-F1i4XM/comparison.json`. Maxima worsen from job
764's 1.245786/0.368332 despite exact same-input first PWA. No full-model
admission follows; subsequent propagated blocks/recycles remain unverified.
Do not reinterpret this as proof the exact first PWA arithmetic is incorrect.

Latest ESMFold2 CPU suite: **400 passed, 24 skipped in 133.77 seconds**.
An additional real CPU JIT invocation of the (1,437,437,8) fully masked-profile
softmax fallback returns the expected shape with all finite values. GPU model
parity, standalone w3, independent full-panel coverage and release remain open.

The measured spatial softmax is now in the installable ESMFold2 runtime,
`models/native_softmax.py`, selected only for FP32 (1,437,437,8) on CUDA without
CP. Other shapes, dtypes and devices retain ordinary JAX softmax. Eight focused
PWA/dispatch/shape tests and Ruff pass. Job 769 completed (exit 0): the actual
runtime PWA output matches native bitwise, including repeat, without attention
injection. Evidence: external `esmfold2-pwa-runtime-softmax-ffY3Q9/report.json`,
`pwa_native`. The attention-injection arm no longer isolates an intervention on
this profile because runtime bypasses `jax.nn.softmax`; use job 767 for that
causal control. Full fixed-tape n=5 repeat2 diagnostic job 770 is submitted with
unchanged native-LM/shim/FFI controls; results and broader admission are pending.

Job 768 completed (exit 0): the diagnostic spatial-softmax control matches
native softmax **bitwise**, including repeat, for FP32 operands ending in
(437,8). It follows 128 dim lanes, four sequential lane contributions,
halving tree reduction, and rounded FP32 division. Evidence: external
`esmfold2-pwa-spatial-softmax-mLBAwK/report.json`, `softmax_spatial_control`.
Reference algorithm source is the exact recorded Torch commit:
https://raw.githubusercontent.com/pytorch/pytorch/cf30153c4c131c8164ee7798e5022d810682e2cb/aten/src/ATen/native/cuda/SoftMax.cu
(SpatialSoftMax_getBlockSize and cunn_SpatialSoftMaxForward). Source-based
dispatch inference has not been independently confirmed with a kernel trace.
The control is bench-only; runtime wiring, other shapes/devices/CP, composed
PWA and full-model structure/confidence remain to be verified. Ruff passes.

Job 767 completed (exit 0). Under matched strict compiler controls, native
softmax output injection makes the entire same-input PWA output bitwise exact,
including repeat. The ordinary runtime arm retains 120 unequal elements,
RMSE 6.8219047462e-7. Isolated softmax differs in 633,327 FP32 elements,
RMSE 6.5580185842e-10, maximum 1.1920928955e-7. Evidence: external
`esmfold2-pwa-attention-injection-baTurh/report.json`. This isolates the PWA
residual at the softmax-output boundary for this captured input; it does not
prove that the full-model coordinate discrepancy is entirely caused there.
The injection uses a scoped patch during tracing and restores `jax.nn.softmax`;
attention is a dynamic JIT argument. No runtime softmax modification or model
admission has been made. Scoped Ruff passes; GPU execution/repeat is verified.

Job 766 completed (exit 0); the new bound prefix report completed successfully
at external `esmfold2-msa-propagated-fixed-3kqWEF/comparison.json`. Both triangle
updates and pair transition have bitwise-exact propagated inputs and outputs
under the diagnostic FFI path. OPM and embedding projection outputs are exact.
The first residual arithmetic differences are on the MSA branch: PWA Wout input
has 226 unequal elements (RMSE 3.6304856976e-8); PWA output has 120 unequal
elements (RMSE 6.8219047462e-7). After residual addition the MSA transition input
has 13 unequal elements (RMSE 8.6933555676e-7); its update has 120 unequal
elements (RMSE 0.0007106319351, maximum 4). These are activation units.

Do not call all inputs identical: embed/project_inputs arguments are narrowed
to BF16 in the candidate while native module arguments are FP32, though their
linear outputs match. Norm-input observation is post-FP32-cast in the candidate
versus module entry in native; mask values match but candidate stores BF16
versus native bool. The report records these dtype differences rather than
hiding them. First pair-branch exactness does not establish subsequent blocks,
recycles, standalone linear behavior, or the uninstrumented full executable.
The next first-block numerical target is the existing PWA residual, not another
pair-update runtime edit. No new production change or model admission.

Added `bench/esmfold2_msa_prefix_report.py` for the pending propagated comparison.
It verifies capture bindings before/after, checks candidate schema and each
paired shape, and records dtype differences separately from array comparisons.
The explicit mapping is candidate `blocks.0.msa_transition.input` to native
`blocks.0.msa_transition.norm.input`; no independent alignment is applied to
activation tensors. Mapping and shape-rejection tests pass two cases; Ruff
passes. Job 766 is still saving its archive (observed growth to 2.0 GB), so no
propagated numerical report has yet been produced.

After the OPM-only routing assertion was updated, the complete ESMFold2 CPU
suite was rerun: **394 passed, 24 skipped in 131.13 seconds**. The prior
one-failure full-suite result is retained historically but the current CPU
regression gate is green within these skips. Added full-block exception and
missing-boundary cleanup coverage: prefix observer tests now pass 33 cases,
including restoration of PWA, transition and triangle callables. Ruff passes.
Job 766 remains running at archive saving; propagated numerical comparison is
not yet available and the CPU gate is not scientific parity admission.

Job 765 failed before propagation because `ffi_output_linear` was imported
from the tape module instead of its defining LM encoder candidate module.
Corrected the observer import and retried as 766; no model source change.
The failure is a harness import error, not a numerical parity measurement.

Job 765 extends the JAX prefix observer through the first MSA block's pair
transition update. It propagates actual runtime OPM/PWA/MSA-transition/triangle
outputs rather than independently injecting each native submodule input.
Starting embedding/pair remain shared native controls. It uses original block
parameters, matched strict compiler settings and the full diagnostic's native
cuBLAS FFI control; it captures 35 selected boundaries and stops before the
final pair residual. Returning boundaries changes the compiled graph, so this
is instrumented first-block evidence, not an uninstrumented full-model claim.
The JAX observer bypasses leaf logging after PWA to avoid treating native
chunk calls as duplicate projection boundaries, retaining module input/updates.
Existing prefix tests pass 29 cases; Ruff passes. The new propagated-block GPU
branch is running, not yet verified. Runtime model code is unchanged this step.

Job 764 finished (exit 0), but is not admitted. Protein RMSDs are
[1.2457860165, 0.0905023442, 0.1419310422, 0.1111550778, 0.0986955441];
ligand RMSDs are [0.3683318010, 0.0376788059, 0.0407767949, 0.0292556424,
0.0202854534] angstroms. Strict confidence fails. Maxima worsen relative to
754 (0.4048258953/0.1938571023). Evidence: external
`esmfold2-msa-pair-native-core-ODmggh/comparison.json`. Same-operand module
results cannot be promoted to propagated block/full-model equivalence. Next
probe should capture the propagated first MSA block, not widen runtime edits.

CPU full suite initially reported 1 failed, 393 passed, 24 skipped (136.83s).
The failure was the old test asserting original weights reached only OPM;
updated it to verify native selection for the now-connected pair updates and
unchanged non-native selection. The affected OPM/wiring tests now pass 16
cases; Ruff/diff checks pass. The complete suite has not been rerun after this
test-only correction, so no all-green full-suite claim is made.

Job 763 completed (exit 0): native cuBLAS FFI reproduces pair-transition w3
bitwise on native inputs, including repeat. Evidence: external
`esmfold2-pair-w3-ffi-fixed-TzwVkh/report.json`. The pure-JAX w3 residual remains
open; FFI is a diagnostic control, not standalone runtime admission.

The MSA block's outgoing/incoming triangles and pair transition are now wired
to original block parameters and native autocast when the native parameter map
is enabled. The map includes all MSA block weights; non-native/FP32 selection
is unchanged. This follows the measured same-operand improvement but does not
admit the incoming residual or standalone w3. The full fixed-tape n=5 repeat2
diagnostic job 764 retains native-LM/shim/FFI and tests the connected path.
Results are pending. Focused norm/wiring suite passes 46; the strengthened
parameter/triangle-direction/final-block wiring file separately passes 12.
Scoped Ruff and diff checks pass. No new full-suite or release/push claim.

Jobs 760/761 completed (exit 0). The isolated native pair transition exactly
reproduces its full-block captured update before saving seven leaf arrays.
Same-operand JAX norm, w12 and SiLU/product are bitwise exact; w3 alone has
RMSE 0.5798174629351378, maximum absolute difference 32, and 12,990,853 unequal
elements. Injecting native norm output leaves that error unchanged; all repeats
are exact. Thus this pair-transition residual is isolated at w3, unlike the
earlier MSA transition norm residual. Evidence: external
`esmfold2-pair-transition-leaves-Dpa1tz/report.json`.

This is pure-JAX linear evidence, whereas the full diagnostic uses a native
cuBLAS FFI control, so it cannot directly explain full diagnostic RMSD.
Added an FFI w3 arm to distinguish those paths. Job 762 failed because its
FFI target variable was shadowed by the comparison loop's target key; renamed
the captured identifier to `ffi_target`. Job 763 retries the corrected tool.
No runtime model change. Four replay guard tests and Ruff pass; the FFI branch
still requires the pending GPU result and is not established by those tests.

Job 761 is queued after native leaf capture 760, using `--pair-transition`
with matched strict compiler controls. It compares norm, w12, SiLU/product,
w3 and compositions (including injected native norm output). Capture 760 has
reached archive saving, which follows its full-block update equality check,
but its final binding verification/report is not yet complete. Its archive
continues growing; no restart or duplicate native run was launched. Neither
pending job is counted as final pair-transition evidence.

Pair-transition decomposition now uses a native isolated replay tool,
`bench/esmfold2_pair_transition_capture.py`: it loads the previously captured
pair input and original module weights, captures norm/w12/w3 chunk leaves, and
requires its final update to reproduce the full-block native output bytewise
before publishing a capture. The JAX replay accepts `--pair-transition` for
these leaves; it is mutually exclusive with aggregate `--pair-updates`.

Initial job 759 failed before model execution because the new tool imported
installed Transformers, whose Hugging Face Hub import is incompatible, instead
of the pinned source. The tool now selects the sole already-hashed native
modeling source from reference bindings and prepends its source root, as the
existing native observer does. Job 760 retries this harness fix; no dependency
upgrade or runtime model change was made. Guard tests: six pass; scoped Ruff
passes. Actual native output reproduction and leaf results remain pending.

Jobs 757 and 758 completed with exit 0. Actual transition chunk geometry is
recorded for all six leaf input/output streams: [64,64,64,64,64,64,53]. All three
pair-update module boundaries are BF16 input/output. Same-operand activation
RMSE (legacy -> native autocast): outgoing triangle 0.0064877466593 -> 0
(bitwise exact); incoming triangle 0.0028074471351 -> 0.000057414650435
(9,685 unequal elements); pair transition 0.851491144989 -> 0.579817462935
(12,990,853 unequal elements, maximum difference 32). Every arm's same-compiled
repeat matches bitwise. These are internal activation units, not angstroms.
Evidence: external `esmfold2-msa-pair-replay-dTQ1Wu/report.json`, with verified
capture bindings to `esmfold2-msa-block-native-pEfpO3/native`.

The outgoing native-autocast path is supported for this captured input; the
incoming residual and especially the pair-transition residual remain open.
Do not connect all three and declare closure from the outgoing result. Next
pair-transition investigation needs its norm/w12/SiLU/w3 boundaries to separate
normalization from projection/composition differences. No additional runtime
pair-update edit or full-model admission is made by this result.

Latest ESMFold2 CPU suite after transition native-norm routing completed:
**394 passed, 24 skipped in 130.11 seconds**. This updates the prior 386-case
result and does not replace missing native/GPU checks. At this verification,
native capture 757 remains running with its archive growing; job 758 is queued.
Neither pair-update numerical results nor full-model admission is claimed.

Job 758 is queued behind 757. `esmfold2_msa_transition_replay --pair-updates`
requires a full native MSA-block capture and compares each triangle direction
and pair transition against its own captured input/update. It uses the captured
pair mask and original weights for native autocast, BF16-converted weights for
the existing legacy path, and the matched strict compiler policy. These six
arms are same-operand submodule comparisons, not propagated block equivalence.
Bindings are verified before/after as in the transition replay. An incomplete
capture is rejected before GPU initialization. Observer/replay guard tests pass
33 cases; scoped Ruff passes. Native capture 757 is still running; pair replay
GPU results and runtime wiring changes are pending.

The job-756 native comparison report completed: all five entity RMSDs exactly
reproduce job 750 and strict confidence remains false.

Job 757 starts native first-MSA-block pair-update capture using the extended
prefix observer (`--full-msa-block` requires native full transition capture).
It preserves native module execution, loads the two triangle modules and pair
transition with strict original state dictionaries, and adds their input/update
outputs to the existing prefix boundaries. It stops at the pair-transition
update, before its caller's final residual addition; this is not a captured
post-residual block output. Expected boundaries: 46. Live MSA-transition chunk
geometry is checked and saved by this run. CPU prefix tests pass 29 cases and
Ruff passes, but the new native block branch is pending GPU validation. No
pair-update runtime change or admission has been made.

Job 756 completed (exit 0), reusing job 750's immutable source with fresh
process/output/autotune caches. Direct NPZ comparison finds both coordinate
and confidence archives identical to job 750 in keys, shapes, dtypes and every
stored byte. Thus the 750 outcome reproduces in this fresh-compilation control;
it is not evidence of a variable sample-0 result across these two compilations.
Compiled HLO text hashes differ (750: b538986e6d2479a0284f312036f823559a0bd2e0419121466e70c08826d0fe83;
756: dcb06214b086d95fe59841d724f23764ef1f1520893e358801e763a97aabbe05),
so no textual/executable identity is claimed. The separate bound native
comparison report is being generated in `esmfold2-750-fresh-QUDKRN`.

Chunk-geometry validation now has nine additional CPU cases covering native
partial chunks, unchunked mode, wrong order, missing/extra calls and invalid
sizes. Prefix suite: 29 passed; scoped Ruff passes. This is not a substitute
for the pending live native chunk-geometry capture.

Job 755 completed (exit 0) with the full strict compiler options recorded.
The current runtime native transition, norm, both linears, SiLU/product,
native-norm injection and CUDA-norm composition are exact against the captured
native operands. Legacy RMSE remains 0.6141288715. Evidence: external
`esmfold2-msa-strict-replay-YKkC19/report.json`. This resolves the compiler-profile
mismatch for this new replay only, not full-model parity.

Job 756 repeats the unchanged external job-750 source in a new process/output
with fresh compilation/autotune caches and `--capture-autotune`; results are
pending. It preserves the old runtime instead of retesting the latest source.
The native prefix observer now checks observed axis-1 chunk sizes against the
module's actual `_chunk_size` and saves each leaf's chunk geometry. Existing
reports lack this field and are not retroactively certified. CPU observer/replay
tests pass 23 cases and scoped Ruff passes; the new geometry branch has not
yet been exercised on GPU.

Job 754 completed (exit 0). Its bound comparison reports protein RMSDs
[0.4048258953, 0.0467784430, 0.0655244339, 0.1298368366, 0.0794268424]
and ligand RMSDs [0.1938571023, 0.0071126237, 0.0249542393, 0.0246560568,
0.0152372160] angstroms. Strict confidence remains false. Norm routing reduces
the maxima relative to job 750 but does not admit this model. Separately,
job 750's coordinates and confidence match their repeat-1 arrays in keys,
shapes, dtypes and storage bytes; this excludes only same-executable variability.

Cross-provider review identified a real compiler-policy mismatch: same-operand
replays through job 753 set only excess-precision=false, whereas full inference
also disables Triton GEMM and three HLO passes. Their exactness remains valid
for those diagnostic executables, not a matched compiler-policy assertion.
The replay now uses and records `compiler_control("native-chunks-strict-rounding")`
and binds that helper's source. Job 755 repeats under this matched policy.
Review also notes missing saved chunk geometry; upstream source confirms
PairTransition's 64-axis1 loop, but actual per-call geometry is not yet captured.
Fresh-compilation reproducibility remains an open control. Reviewer wording
that only sample 0 changes is not accepted literally: other samples change too.

Job 753 completed (exit 0): the CUDA width128 norm control and its composed
transition both match native **bitwise**, with exact repeats. Evidence:
external `esmfold2-msa-cuda-norm-186urO/report.json` (`cuda_norm` and
`cuda_norm_composed`). The existing runtime width128 native-norm dispatch now
also includes `msa_encoder.blocks.*.msa_transition.norm`; CPU/CP fallback
behavior is unchanged. Focused norm-routing and parameter-wiring tests pass
46 cases; Ruff and diff checks pass. Full diagnostic fixed-tape n=5, repeat2
job 754 has been submitted with the same native-LM/shim/FFI controls. Its
structure/confidence outcome is pending; first-input exactness is not admission
for all blocks, inputs or full-model inference. Cross-provider consultation
remains pending (no review approval claimed).

Job 752 completed (exit 0): injecting captured native norm output into one
compiled, native-chunked w12/SiLU/product/w3 composition produces the captured
transition output **bitwise exactly**, with an exact repeated call. Evidence:
external `esmfold2-msa-norm-injection-uOa92l/report.json`, arm
`native_norm_output_injected`. Unlike independent leaf tests, this checks the
composed downstream path. For this captured first transition input it isolates
normalization as the remaining intervention boundary, but does not explain the
full model's 1.4629 angstrom maximum or establish later-block parity. Next
discriminating control: replay an alternative native-width128 norm reduction
before changing production normalization. Scoped Ruff and three replay guard
tests pass; no new runtime change or admission.

Job 751 completed (exit 0) and closes the previously missing same-operand
SiLU/product check: native w12 output replayed through FP32 SiLU, BF16 narrowing,
and BF16 multiplication matches native w3 input bitwise, including a repeated
compiled call. Evidence: external `esmfold2-msa-hidden-replay-eqJboh/report.json`.
Together with job 749, isolated w12, SiLU/product and w3 are exact; norm retains
the measured small mismatch. This does not by itself prove the full-model
regression is caused by norm: propagated/fused composition and downstream
paths remain unisolated. No additional runtime edit was made. Scoped Ruff and
the replay's three CPU guard tests pass. A read-only cross-provider consultation
was requested to audit the observer/replay/wiring and next discriminating probe;
its response is pending, not an independent approval.

Full diagnostic job 750 finished (exit 0), but the transition change is **not
admitted** by end-to-end parity. The bound comparison report is external
`esmfold2-msa-transition-core-c2WZlv/comparison.json`. Protein entity RMSDs for
the five paired samples are [1.4629229261, 0.0656128046, 0.0687509326,
0.1260019922, 0.0988994532] angstroms; ligand values are [0.4856656232,
0.0227116239, 0.0305681299, 0.0268589918, 0.0170129885]. Strict confidence
fails. Relative to job 746, worst-case protein and ligand RMSD worsen from
0.6217764754/0.3022541418 to 1.4629229261/0.4856656232. Thus the isolated
transition improvement does not establish an end-to-end improvement or its
cause. The unchanged MSA pair triangle/transition paths still use legacy
parameters and require separate boundary evidence; do not infer that they
explain the regression without a controlled probe.

Verification: ESMFold2 CPU suite completed with 386 passed and 24 skipped
(133.29 seconds); the separately rerun wiring file passed 12 tests, including
four new native/non-native/final-block routing cases. Scoped Ruff and diff
checks pass. No release, standalone parity, performance, or push claim.

Jobs 748 and 749 completed with exit 0. Native transition norm accepts BF16
and returns FP32; w12 accepts FP32 and returns BF16, and w3 is BF16/BF16.
Same-operand legacy transition output RMSE is 0.6141288715191372 (maximum
absolute difference 16); original-parameter native-autocast RMSE is
0.00779944504947735 (maximum 8). These are internal activation units, not
coordinate angstroms. Both isolated w12 and w3 match native bitwise; the norm
RMSE is 3.1254958837859525e-8. All same-executable repeats match bitwise.
Evidence: external `esmfold2-msa-transition-replay-uYKPoB/report.json`.

The actual MSA encoder now routes its transition through native autocast when
the native MSA parameter map is enabled; that map includes original transition
parameters. FP32/non-native behavior is retained. The focused wiring/PWA/replay
suite passes 13 tests; scoped Ruff and `git diff --check` pass. Full ESMFold2
CPU tests and fixed-tape n=5 full diagnostic job 750 are running. The latter
retains the previously declared native-LM/shim/FFI controls and is not a
standalone production or performance gate. Full structure/confidence admission
remains pending, including the native transition's nonzero activation residual.

The follow-up same-operand replay is `bench/esmfold2_msa_transition_replay.py`,
queued as job 749 behind native capture 748. It compares the existing legacy
BF16-parameter transition and original-parameter native-autocast transition,
plus isolated norm/w12/w3 outputs. Projection replay preserves native axis-1
chunks of 64; each arm repeats the same compiled call. Captured native inputs
are used, so this is not a propagated/full-model parity claim. Reference,
archive, weight, and runtime source bindings are checked before/after replay.
Three new CPU guard tests cover wrong capture type and unbound weights; with
the existing prefix suite, 23 tests pass. Scoped Ruff passes. GPU results are
pending; no runtime default is changed by this follow-up.

The first native `PairTransition.forward` chunks axis 1 at its native chunk
size. The extended prefix observer now collects each transition leaf's input
and output per chunk, concatenates them along that axis, and checks coverage
against the transition input before saving. It does not change the upstream
chunk policy. Other prefix boundaries retain duplicate-call rejection.

Job 748 (`esmfold2-native-msa-transition`) captures the first MSA transition
after the previously observed OPM/PWA prefix, using an external source snapshot
and the existing bound native inputs/tape. Submission and running state were
verified; results and candidate parity are pending. This observer correction
is not evidence that native chunking caused the final coordinate mismatch.

Verification: `tests/test_esmfold2_msa_prefix_probe.py`: 20 passed; Ruff passed
for `bench/esmfold2_msa_prefix_probe.py`. These existing tests do not establish
the new native GPU transition branch's correctness. No model admission or
release claim is made.
