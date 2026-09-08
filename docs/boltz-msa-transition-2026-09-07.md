# Boltz MSA transition: matched-native-input normalization control

Status: local cause isolated; production selection implemented; full-model
replay still fails. Performance validation is pending. This is not model closure.

The first native MSA transition receives the captured full-width FP32 residual
`input_m + pwa`, without slicing MSA rows. Its native module capture had already
reproduced the original run. This probe verifies the input/output array and
tree hashes and uses the original mapped checkpoint leaves.

| Arm | Native-output RMSE | Max absolute difference | Unequal elements |
| --- | ---: | ---: | ---: |
| Existing generic norm | 0.0002676144431362421 | 0.5 | 20,519 |
| Manual same-order implementation | same | same | same |
| Barrier after each hidden-chunk projection | same | same | same |
| Only normalization replaced by native vector-4 Welford/FMA | 0 | 0 | 0 |

GPU queue job 395 completed successfully. The manual implementation is bitwise
identical to the existing production primitive before interpreting the norm
counterfactual. These are internal representation differences, not Angstroms.
The control isolates normalization for this one transition; it does not prove
that all MSA differences or the final 5SAK coordinate gap have the same cause.

The production change enables the existing native CUDA normalization at width
64 and selects it for BF16 MSA transitions only. FP32/FP16 MSA selection and
CPU/context-parallel fallback remain unchanged. No default dtype changes.
The extra non-fusible normalization's runtime and memory costs are unmeasured.

Verification: 56 focused CPU tests passed before adding the explicit MSA route
test; the updated native-normalization test file subsequently passed all 26
tests, including FP32, FP16 and BF16 route selection. Ruff passed. The dedicated
immutable-source GPU production check, job 397, also gives RMSE 0, max absolute
difference 0 and zero unequal elements against native. Job 396 was rejected
before execution because the current directory shadowed the selected snapshot;
job 397 used Python safe-path with unchanged source bytes.
The rejection is a harness invocation error, not a model numerical result.

## Historical comparison guard

The BF16 adoption commit `ebdb5cf` compares FoldJAX FP32 versus FoldJAX BF16
at 1,003 tokens (paired CA RMSD 0.070 Angstrom, plus crystal TM scores). It is
not native-BF16 versus FoldJAX-BF16 same-tape entity parity. The recoverable
5SAK FP32 baseline at `83eb4eb` instead uses native kernels disabled and
FoldJAX cuEq multiplication/XLA attention. Current BF16 captures enable native
kernels. Hold kernel policy as well as dtype constant in any regression claim.

The current full capture runner rejects FP32 by design and requires APIs absent
from old commits. Historical replay therefore needs a source-verified compatible
runner; silently applying current production fixes to an old arm is invalid.
Old/current same-condition coordinate replay remains pending.

## Full 5SAK follow-up

GPU job 398 completed the own-trunk, fixed-native-tape n=5/200-step/3-recycle
full-MSA replay from the immutable production snapshot. The full-system Kabsch
fit followed by per-entity measurement gives protein maximum
**2.0082260788169917 Angstrom** and ligand maximum **0.16802798338207467 Angstrom**.
Both raw and public strict confidence gates fail. The preceding pair-norm-only
snapshot gave 1.9120499147484837 and 0.14972648065157357 Angstrom respectively.

Thus the local transition correction does **not** improve this whole-model
comparison. Preserve that negative result: a bitwise teacher-forced operator
does not imply the remaining coupled trunk/sampler trajectory is corrected.
Do not revert the proven native arithmetic solely to optimize this one output,
or admit the full model. Other MSA operations and earlier input errors remain
to be isolated, along with the matched historical-code regression control.

## Next OPM control

Job 399 holds the first native OPM input, mask and mapped weights fixed. Native
standalone reproduction is an exact prerequisite. Normalization replacement
and contraction shape are separate controls:

| OPM arm | Native-output RMSE | Unequal elements |
| --- | ---: | ---: |
| Generic norm, token chunks of 128 | 0.001526486829031123 | 9,339,485 |
| Native norm, token chunks of 128 | 0.0015257042603323426 | 9,324,246 |
| Native norm, full 437-token contraction | 0 | 0 |

The last arm matches bitwise, but combines native norm with native contraction
shape. The production MSA caller already requests native contraction shape
within its memory budget. Confirm that actual production selection independently
before interpreting this as a production repair; the diagnostic changes no
production OPM normalization. Five focused probe CPU tests and Ruff passed.

Historical FP32 old/current replay was queued as jobs 400/401 using the same
frozen current runner, native artifacts, checkpoint and kernel settings. The
old candidate is clean `83eb4eb`; the current candidate is a source snapshot
including the MSA transition correction. Coordinates must still be evaluated
per entity. Legacy runner exit status is not scientific acceptance, and its
regenerated sigma schedule prevents claiming complete native tape consumption.

Job 400 completed successfully. Re-evaluating its original-index coordinates
with the current valid-atom mask and one global Kabsch gives old-source FP32
protein maximum **0.006745037141201129 Angstrom**, ligand maximum
**0.0008096437670903163 Angstrom**. The archived coordinate arrays contain 3,104
slots; masked padding is excluded by the entity evaluator. The historical
runner's loose global-RMSD/correlation exit is not used as this criterion.
The current-source counterpart remains pending, so this alone does not decide
whether FP32 regressed. It confirms the old source can still produce a small
native-relative discrepancy in the selected current runtime and frozen control.

Job 401 completed the current-source counterpart. Native artifacts, runner,
harness, checkpoint, effective settings and recorded runtime are equal across
arms, verified before reporting. Current FP32 protein maximum is
**0.004567448242420427 Angstrom**, ligand maximum
**0.0004993116514379307 Angstrom**. Both old and current arms pass the 0.05
Angstrom entity diagnostic in this one control. No FP32 coordinate regression
is observed here; this does not establish mixed-precision parity, confidence
closure, repeatability, or universal absence of regressions.

[Portable paired report](../bench/experiments/boltz-historical-fp32-2026-09-07.json)
retains every sample/entity value and binds raw coordinates and provenance.
The current source snapshot predates the OPM normalization change; that later
change is BF16-only but is not silently included in this result's source claim.

## Production OPM confirmation

Jobs 402/403 close the missing shape-controlled and production-selection probes.
On the same full 437-token contraction, generic normalization gives RMSE
0.0001519367744322505, max absolute difference 0.03125 and 153,993 unequal
elements; replacing normalization alone makes all outputs bitwise equal.
The unchanged pre-repair production shape selector produces exactly the generic
full-token comparison. After the BF16-only normalization change, the actual
`chunk_size=128, preserve_native_amp_shape=True` production call also has zero
unequal elements, without diagnostic monkeypatching.

FP32 and FP16 selection, mask-cast order, hidden-axis chunking, FP32 output bias
and the 256 MiB shape budget remain unchanged. Parent CPU regression coverage
for MSA, norm selection and replay/probe helpers passed 74 tests. An earlier
attempt named a nonexistent test path and collected nothing; it is not a pass.

The full native-policy own-trunk n=5 panel is queued from the post-OPM snapshot:
5SAK (job 404), 1URN (405), 3GCA (406). Full-model structure, confidence and
performance improvement remain unproven until their respective checks finish.

Job 404's full 5SAK result remains a failure: protein maximum
**2.837167096967245 Angstrom**, ligand **0.36632251170910995 Angstrom**;
raw/public strict confidence both fail. Final trunk RMSE is s=0.0559386702,
z=0.0788247444. These are worse than the transition-only full run despite the
teacher-forced OPM becoming exact. No end-to-end improvement is claimed.
[Bound full report](../bench/experiments/boltz-amp-5sak-v36-2026-09-07.json).

The existing native same-tape repeat was bitwise stable on this case; it does
not justify absorbing this discrepancy into a native-variability allowance.
Next controls use all 4,436 native MSA rows for PWA, avoiding the prior sliced
reference's GEMM-shape confound (jobs 407–409). Upstream-first operator repair
must continue to separate corrected arithmetic from final trajectory behavior.

## Separate public confidence output repair

The wrapper now adds native `predict_step`'s missing `confidence_score` after
all confidence samples are collected: `(4 * complex_plddt + tm) / 5`, selecting
pTM only when the **entire** ipTM batch is close to zero with Torch's default
`rtol=1e-5, atol=1e-8`. A per-sample fallback would be incorrect for mixed
zero/nonzero batches. Raw heads and sample order are unchanged.

The missing field was reproduced before the change. Focused wrapper tests
cover eager/JIT, sequential/batched execution and zero-boundary cases, with
33 passing and one slow test deselected. This implementation follows after
the v36 source snapshot; v36's missing public field and strict failures remain
historical facts. Fresh GPU public-output verification is still required.

Job 405 completed 1URN on the same v36 snapshot: protein maximum
0.0449577029330575 Angstrom, RNA maximum 0.3121218830412846 Angstrom; both
raw and public strict confidence fail. The RNA outlier therefore persists;
do not claim the OPM repair closes protein/RNA parity.

Job 406 completed 3GCA: RNA maximum 0.005124026892930201 Angstrom and ligand
maximum 0.003981299213028154 Angstrom. Structure passes the fixed diagnostic,
while both raw/public strict confidence fail. Bound full reports for
[1URN](../bench/experiments/boltz-amp-1urn-v36-2026-09-07.json) and
[3GCA](../bench/experiments/boltz-amp-3gca-v36-2026-09-07.json) preserve the
individual values, source identity and separate gates.

PWA native prerequisite job 407 passed on all 4,436 MSA rows: the explicit
decomposition and the original full-capture output are both bitwise equal.
The corresponding FoldJAX baseline/native-norm controls remain pending.

Jobs 408/409 completed those full-row controls. Generic-norm PWA output RMSE is
0.00016025566156648287 (220,241 unequal values); native-norm PWA is
0.00010594088450566563 (127,583 unequal). Both norms and all eight head value/
gate projections become bitwise exact in the decomposed trace. Nevertheless,
25 logits across all eight heads still differ; even heads with exact logits
and BF16-consumer attention weights retain averaging-contraction differences.

The native-norm decomposed path is not equal to the production-shaped path
(17,241 unequal values), whereas the generic baseline matched. Therefore do
not infer that changing PWA normalization alone closes the operator or blindly
admit that diagnostic decomposition as production. Targeted same-operand
native projection and averaging probes are the next discriminators.

The newly added prediction score is classified under public confidence only:
native forward does not contain that derived field. Report tests prove that a
wrong score fails the public gate while leaving an otherwise matching raw-head
gate unchanged; no raw native head is waived. All 20 report tests passed.

### Same-operand PWA averaging, jobs 410/411

The head-2 native averaging replay uses the complete 4,436-row input, with
captured BF16 consumer weights and values. Both original Torch einsum and
explicit wide BMM reproduce the native capture bitwise. JAX einsum and wide
BMM produce identical outputs to each other, but each has 5,004 unequal values
against native, RMSE 0.000049223767731439904 and maximum absolute error 0.015625.
These are intermediate activation errors, not coordinate RMSDs.

Optimized HLOs are not identical: einsum uses a transposed
`[141952,437]` dot with backend size 64; wide BMM uses `[437,141952]` with
backend size 32. Neither layout resolves the difference under `highest`
precision and `xla_allow_excess_precision=false`. Thus a same-operand
contraction mismatch is reproduced independently of normalization/projection;
its accumulation/backend cause remains unproven. Production PWA is unchanged.

Artifacts are the source-bound `pwa-average-native-v38` and
`pwa-average-jax-v38` reports, input/output archives and optimized HLOs in the
private Boltz closure task root. Verification: both GPU jobs exited zero;
51 focused averaging/adapter/report CPU tests and affected Ruff checks passed.
This is an operator diagnostic, not full-model or performance admission.

### Accumulation/backend controls, jobs 412/413

Native BF16 reduced-precision reduction enabled/disabled produced identical
outputs. JAX preferred FP32 dot output followed by BF16 conversion retained
the same 5,004 mismatches. Disabling Triton GEMM emitted an actual
`__cublas$lt$matmul` custom call, but left 5,158 mismatches (RMSE
0.00005024877171715147), with or without preferred FP32 output. None of these
counterfactuals closes the contraction. A generic `__triton` substring in HLO
can also describe a non-GEMM fusion and is not sufficient backend attribution.

A read-only sparse CPU diagnostic evaluated all 5,004 baseline mismatching
inner products in FP64, then converted FP64 -> FP32 -> BF16. That result matched
JAX at 4,752 positions and native at 195; 57 matched neither. No evaluated FP64
sum was exactly the midpoint of the two observed BF16 outputs. Median distance
to that midpoint was 3.749992174562067e-7 (maximum 1.1990079656243324e-5).
This is a numerical diagnostic, not an exact-arithmetic oracle or permission
to replace the native reference with the more accurate-looking candidate.
Native accumulation fidelity remains unresolved. Verification: both GPU jobs
exited zero, 23 focused probe CPU tests and affected Ruff passed.

### Real prediction score verification, job 416

A fresh n=5 3GCA full GPU replay now emits `confidence_score` with shape `[5]`.
Recomputing the native whole-batch formula from the candidate's saved ipTM,
pTM and complex pLDDT reproduces all five output values exactly. Thus the
previously missing public field is present and correctly derived in this run.
It does not repair inherited head differences: against native the score has
maximum absolute error 0.000606834888458252, with four strict failures. Both
raw and public confidence gates still fail. Entity RMSD maxima are RNA
0.005104914515503814 and ligand 0.004102210082385756 Angstrom, passing structure.
The [source-bound report](../bench/experiments/boltz-amp-3gca-v40-2026-09-07.json)
preserves all samples and separate gates; v36 remains an older snapshot.

### Native raw FP32 output bridge, job 418

With unchanged BF16 operands and native wide layout, the profiled native BMM
baseline again reproduces the captured output bitwise. Requesting FP32 output
then rounding to BF16 also reproduces the entire original output bitwise.
The two profiled kernels are distinct BF16-output/FP32-output variants of
`cutlass_75_tensorop_*s1688gemm_bf16_128x128_nn_align1`; this bridge does not
prove identity of the original hidden accumulator.

At the 5,004 earlier JAX/native mismatching positions, the captured raw FP32
control differs from FP64 dot evaluation by maximum 1.3083335943520069e-5 and
RMSE 2.262098346776882e-6; 4,595 values are nearer zero than the FP64 result.
Only 16 match FP64 rounded to FP32. This is consistent with a contraction
arithmetic difference before final BF16 storage, but instruction-level cause
remains unresolved. Kernel names alone do not prove a rounding mechanism.
The private report binds profiler evidence and mapped cuBLAS library identities;
mapped-file presence is not proof of a specific symbol dispatch. Verification:
GPU job exited zero and 35 focused CPU probe tests passed. No production
matmul policy was changed.

### Explicit one-warp MMA grouping control, job 424

Four fixed 16-by-8 tiles retain full native K=437 operands. Both explicit
`m16n8k8` and `m16n8k16` BF16/FP32 instructions passed the on-GPU basis,
tail, nonzero carried-accumulator and physical lane-mapping tests. The k16
result matches earlier JAX BF16 values on all 512 entries, but differs from
native on four. k8 reduces native differences to one (maximum BF16 error
0.0009765625, versus k16 0.015625). Raw FP32 native maximum errors are
9.5367431640625e-7 (k8) and 2.6226043701171875e-6 (k16).

This supports instruction grouping as one arithmetic contributor but does not
identify the native SASS, close even these four tiles, or justify changing
production PWA. Native tail/group ordering remains to be determined. Actual
compiler IR and comparison artifacts are bound in private `pwa-mma-v44`.

Earlier probe jobs 422/423 failed before numerical execution: JAX 0.11.1's
multi-result inline-assembly lowering returned a nested IR result list, then
the lane-ID observer requested an unsigned IR result type rejected by Triton.
A benchmark-only scoped result-list unwrap and signed lane observer corrected
those issues without editing dependencies or arithmetic. Failed artifacts
remain in `pwa-mma-v42/v43`; 31 parent CPU probe tests pass. The corrected
compiler experiment is diagnostic only, not a supported runtime workaround.

### CUTLASS residue-first control, job 429

Public CUTLASS v3.8.0's two-stage predicated iterator starts with the K residue,
then advances by that residue before processing full tiles. For K437 and
CTA_K32, this gives `[0..20, zero x11, 21..436]`, not end padding. The paired
control retains exactly the same 437 operands once and in order, inserts only
zeros, and uses 56 k8 MMAs in both arms.

The residue-first arm matches **all 512 raw FP32 and BF16 values bitwise**
against the original full-native output. Appending zeros to K448 does not
improve the earlier k8 result (413 raw FP32 and one BF16 mismatch). Each arm's
GPU basis/tail/carried-C gate passed. The private source-bound `pwa-mma-v45`
report retains original k8/k16 controls, layout maps and compiler evidence.
Parent combined MMA/averaging/norm/prediction/report tests passed **158 tests**.

This establishes a matching arithmetic construction for four selected tiles,
not the native private kernel's exact implementation or full contraction
parity. Full-shape validation is the next boundary; production PWA remains
unchanged and its full-model failures are not relabelled.

### Full same-operand contraction, job 438

The residue-first k8 construction now matches all **62,033,024** entries of
the full `[4436, 437, 32]` native contraction in both raw FP32 and BF16,
bitwise: zero unequal entries, zero maximum error, no nonfinite values.
On-device lossless operand packing avoids the expanded host register tape;
basis, tail and carried-accumulator checks pass, and bound inputs remain
unchanged. Evidence is retained in private `pwa-full-mma-v46`.

The full probe source SHA256 is
`18dd2dbc090750926e2c3a7b18d23f2407dfc7149581725b14585f419df09be9`.
Parent full-probe/LM-shim harness checks passed 46 tests. This is a full
contraction result at the declared shape, not a full-model result, proof of
native private kernel identity, or a performance benchmark. The benchmark-only
compiler workaround must not be silently installed into production; a scoped
runtime lowering and subsequent model replay remain required.

### Actual runtime primitive, job 447

The new `native_pwa_mma.py` owns its four-result primitive and Triton lowering;
no generic compiler rule is patched. Job 447 verifies this runtime directly:
all 62,033,024 raw FP32, actual device BF16, and device-rounding bridge values
match the same native full contraction bitwise. Actual Triton IR contains the
carried k8 instruction; physical-lane and basis gates pass and bindings remain
unchanged. Evidence is retained in `pwa-runtime-full-mma-v47`.

The verified runtime SHA256 is
`30fad8283dffd9cb2df97d13fbdd5b16e94fc2013216dae293d3672a12774036`.
The existing full-row AMP caller now reaches this bounded implementation.
Default auto-row chunks (1199/1199/1199/839 on this full MSA), CP, other shapes
and non-AMP still use the previous path. The next gate uses the exact native
full output's row slices; it must not be confused with separately invoking
native on a smaller MSA, which can select different arithmetic. No full-PWA,
full-model, arbitrary-shape or performance admission is claimed.

### Lane guard and default row-slice control, jobs 456/457

An independent scoped review identified that the physical-lane invariant was
only enforced by the benchmark. The runtime now checks every tile locally:
any physical/logical lane mismatch poisons all four output registers with
NaN, a nonfinite failure signal rather than a Python exception. There is no
host callback. Parent tests cover all 32 single-lane violations, invalid
inputs and actual unpatched CPU/Triton lowering: 81 passed. The review's
mid-edit lint finding is clear in the frozen file; it was not release approval.

Job 456 retains all 62,033,024 raw FP32 and BF16 values bitwise with the guard.
Job 457 then tests the four actual default row ranges independently, reusing
the compiled 1199-row executable with fresh operands for each range. All
62,033,024 combined raw FP32, BF16 and device-rounding bridge values are again
bitwise equal to the original full native output slices. Both jobs retain
unchanged bindings and reject nonfinite results. Private evidence is
`pwa-runtime-guarded-mma-v49` and `pwa-runtime-row-mma-v48`.

The guarded runtime SHA256 is
`39c82afbcd05595d6f858d16230a3cdc30b085cc6938e99179efef4f1e920ad9`.
Default caller integration is the next boundary and must preserve original
S4436 context; these results do not establish the behavior of native invoked
on a different original MSA or untested custom chunk profiles. All-head PWA,
subsequent MSA layers, full-model coordinates/confidence and performance remain
separate gates.

### Default caller integration

The AMP caller now passes the original MSA row count for automatic/default
chunking. The private selector requires original S4436 and actual
S1199/S839/S4436 together; explicit smaller custom chunks retain the original
einsum. Other original MSA sizes, CP and non-AMP are unchanged. CPU tracing
checks all 32 calls (four row chunks times eight heads) in the real default
caller, without claiming numerical GPU closure from a trace.

Frozen runtime SHA256:
`63effa15742a1d66733a4bd7c2cc38dea207a9136be82f1af391b38d0ac4596c`;
MSA caller SHA256:
`f7a7c05d81cafc5f08d718a3f701d4afec97d7a536dce3b1ae1ef6483af1d302`.
Parent focused verification includes all 131 kernel tests. Full MSA job 459
uses native-captured operands and exact mapped weights, with no optional
norm, dense-embedding or full-row overrides. Its completion and numerical
result must be recorded separately before any full-MSA claim.

Job 459 completed with all 226 mapped original weight leaves exact and all
44 expected stage captures present. Full-MSA parity still fails: final pair
RMSE `0.031884564463700686`, maximum absolute error `2.0995941162109375`.
The first MSA embedding already has 6,239 unequal values (RMSE
`0.00014411428600741148`, max `0.03125`), and the first complete PWA output
has 1,185,081 unequal values (RMSE `0.0003701947719562199`, max `0.1875`).
These are internal representation errors, not Angstroms. This propagated-input
test cannot attribute the remaining PWA output difference to the contraction;
the next control must hold the whole PWA input fixed. Artifact label:
`runtime-integration-20260908-8gdWU0/boltz-msa`. No full-model or performance
admission follows from this successful execution.

### Full PWA with fixed native input

Job 469 runs the actual default PWA, including all eight heads, gates and
output projections, using the original native BF16 input values losslessly
promoted to the FoldJAX scan carry's FP32 dtype. Eleven source/artifact
bindings link the original MSA capture, eight parameter leaves and native
full-PWA target; no norm or decomposition patch is used. Parent probe tests
passed 25 checks. Job 468 was rejected before GPU execution because the
requested output was inside the immutable source root; job 469 changes only
that output location, not source or operands.

Full-output parity still fails: 175,807 unequal values out of 124,066,048,
RMSE `0.00011014140457570776`, maximum `0.125`, no nonfinite values. This
isolates residual PWA differences from the earlier embedding mismatch; it
does not invalidate the separate bitwise same-operand contraction proof.
Artifact: `boltz-full-pwa-result-20260908-euOUUz/candidate`, source snapshot
`boltz-full-pwa-20260908-oZq80w`. Normalization, projection and gating remain
candidate boundaries; no full-model or performance claim is made.

### Initial embedding projection: actual native capture

Jobs 471–473 isolate native `s_proj` at full `[1,437,384] x [384,64]`
shape. The native outputs reproduce the three scalar values inferred from
the earlier CPU counterfactual. Native BF16-reduction enabled/disabled arms
agree at those values and both reconstruct all 124,066,048 native embedding
values exactly. The candidate projection differs at four of 27,968 outputs
(max `0.015625`); three differences survive the subsequent embedding addition,
reproducing exactly the same 6,239-entry initial mismatch. Native and candidate
are each bitwise repeatable. Native runtime defaults are not changed.

| Token/channel | Captured native BF16 value | Candidate BF16 value |
|---|---:|---:|
| 18/53 | `-0.0556640625` | `-0.055908203125` |
| 258/54 | `-3.140625` | `-3.15625` |
| 261/3 | `1.203125` | `1.2109375` |

Native profiler records one CUTLASS WMMA GEMM. Candidate HLO instead splits
K384 into four K96 contractions, reduces their FP32 outputs, then narrows to
BF16. This is a directly observed accumulation-graph difference. A split-K
control must test causality before changing runtime compiler policy; copying
the inferred scalar values is not an implementation fix. Sparse and earlier
dense-embedding controls give identical mismatch bytes, so dense replacement
alone is not supported by this evidence. Artifact:
`boltz-embedding-20260908-Bxcb6b`; parent probe tests: 42 passed.

Job 477 forces only `xla_gpu_experimental_force_split_k=1`. The emitted HLO
retains Triton GEMM but removes split-K and its final reduction. All 27,968
projection outputs and the reconstructed 124,066,048 embedding values become
bitwise identical to native. Artifact: `boltz-split-control-20260908-VafOzc`.

Full-MSA job 479 applies the same compiler control without changing runtime
defaults. First `input_m` is now bitwise exact; first PWA retains the same
175,807 unequal values and RMSE `0.00011014140457570776` as the fixed-native-input
PWA control. Final MSA pair RMSE remains `0.031742389998573074`, max
`2.2256317138671875`. Thus this control closes the initial embedding boundary
but not PWA or the full model. Artifact:
`runtime-affine-split-20260908-VLMhj2/boltz-msa`. Compiler policy adoption and
uninstrumented performance remain separate from this diagnostic.

The actual BF16 PWA now selects the existing native CUDA Welford norm for
both `m` and `z`; FP16/FP32 and the helper's CPU/CP fallback policies are
preserved. Earlier full-row controls had already demonstrated exact native
norm and value/gate projections with this arithmetic, separately from the
remaining contraction/logit differences. Parent route/kernel/probe checks
pass 188 tests. A new route test initially reused a cached JAX trace and
missed its instrumentation; a fresh wrapper trace fixes the test, without
changing runtime code. Job 483 measures the actual default full PWA with
unchanged native inputs and default compiler policy. It completed successfully:
99,059 unequal values out of 124,066,048, RMSE
`0.00007974575943452667`, maximum `0.125`, and no nonfinite values.
Compared with job 469's 175,807 unequal values, this improves the actual
production-shaped operator but does not close it. Artifact:
`boltz-pwa-norm-result-20260908-PZMJQe/candidate`, source snapshot
`boltz-pwa-native-norm-20260908-lMsIyy`. These are representation errors,
not Angstroms; no updated full-model structure or performance claim follows.

### Full logits same-input replay: native GEMV versus XLA reduction

Jobs 484–486 replay all eight one-column projections on the original full
normalized FP32 `z` and original FP32 weights, with native BF16 autocast.
Native job 484 reproduces every captured logits value bitwise and repeats
bitwise. Its profiler records cuBLAS `internal::gemvx::kernel` with BF16
operands/output and FP32 accumulation, not a tensor-core GEMM.

The production Linear candidate differs at 25 of 1,527,752 logits values,
maximum `0.0625`, RMSE `0.00011093877552469425`; its repeated execution is
bitwise identical. Its optimized HLO uses FP32 elementwise multiplication
and reduction after BF16 operand conversion. Disabling Triton GEMM produces
the exact same HLO SHA-256 and the same output differences. This control
does not select a different backend for a dot already rewritten to reduction,
so it supplies no evidence for changing that global compiler option.

The accumulation graphs are observably different; which native GEMV
accumulation steps produce the 25 rounded differences remains unisolated.
This is not evidence that all remaining full-PWA error comes from logits.
Artifact: `boltz-logits-result-20260908-69hWq4/{native,baseline,no-triton}`;
source snapshot `boltz-logits-source-20260908-Y73dZH`. The new probe plus
reused embedding-probe checks pass 67 tests. No runtime compiler default is
changed and no new full-model admission follows.

Job 487 additionally checks propagation through the actual complete MSA module
at default compiler policy, with no optional norm/dense/row-shape overrides.
The initial embedding still differs at 6,239 values. First PWA differs at
1,113,065 values (RMSE `0.0003627554297298411`, max `0.1875`), while final pair
RMSE is `0.03226996220674999` and max `2.04150390625`. The separate exact-input
PWA improvement therefore does not establish improved complete-MSA parity:
earlier embedding differences still propagate, and the complete MSA remains
open. Artifact: `boltz-logits-result-20260908-69hWq4/full-msa-default`.

### Current full-MSA combined control (job 618)

Snapshot `boltz-current-split-source-20260908-9jni0H` ran with
`XLA_FLAGS=--xla_gpu_experimental_force_split_k=1`, without optional norm,
dense-embedding or row-shape overrides. Result:
`boltz-current-split-result-20260908-HOrWdo/full-msa`. All 226 original weight
leaves match. First `input_m` is exact; first PWA has 99,059 unequal values,
RMSE 0.00007974575943452667 and maximum 0.125. This reproduces the earlier
exact-input PWA result inside the complete MSA path once the initial embedding
discrepancy is removed. Final pair RMSE remains 0.03130259639850445, maximum
2.0645599365234375, with 24,420,259 unequal values. The next boundary is the
remaining PWA discrepancy, not initial embedding in this controlled run.
These are representation errors, not structural RMSD or full-model admission.
Verification: job 618 exits 0; completed stage/report artifacts inspected.

### Native logits counterfactual (job 619)

The benchmark-only `--native-logits` intervention replaces exactly eight
head-wise logits projections using the hash-bound native stages. Production
model code is unchanged. Source `boltz-logits-intervention-source-20260908-H0fegA`,
result `boltz-logits-intervention-result-20260908-cTXpm0/candidate`.
The complete PWA still differs at 19,789 of 124,066,048 output values, RMSE
0.00004928337766576196, maximum 0.0625, with no nonfinite values. The baseline
had 99,059 unequal outputs and RMSE 0.00007974575943452667. This intervention
substantially reduces the discrepancy but does not eliminate it; it cannot
justify attributing all error to logits or shipping reference substitution.
Verification: job 619 exits 0, capture completes, bindings remain unchanged,
and exact output verdict remains false. A same-snapshot ordinary baseline is
still needed to isolate source/autotuning variability from this counterfactual.

Job 620 now supplies the same-snapshot ordinary baseline at the sibling
`baseline` directory. Its 99,059 unequal values, RMSE 0.00007974575943452667
and maximum 0.125 reproduce the historical production metrics. Direct report
checks establish equal source/reference bindings, original inputs, weights and
target between 619 and 620; the only additional input is native logits. Both
retain unchanged bindings. The 19,789 residual values after substitution
therefore are not explained by comparing different source snapshots. Distinct
compiled graphs and unrecorded autotuning remain possible confounders; this is
not a unique kernel-level causal proof or model admission.
Verification: job 620 exits 0; both reports and matched bindings checked.

Byte-bound output-position analysis of 619/620 against the same native PWA
target finds 95,441 formerly unequal values become equal, 3,618 remain unequal,
and 16,171 previously equal values become unequal. The latter are also all
positions where absolute error worsens. Thus subtracting mismatch counts must
not be described as isolating an independent error component: rounding and
downstream interactions change the mismatch support. Residual 19,789 is the
sum of persistent and newly unequal positions, not a subset of baseline errors.
Verification: native target and every output chunk SHA256 checked before the
CPU position-wise comparison; no extra inference or runtime change.

Softmax diagnostic job 621 is not valid same-dtype evidence: its candidate
weights were narrowed to BF16 while saved native weights are FP32 before that
cast. The reported near-all-entry mismatch is therefore not attributable to
softmax implementation differences. The live probe is corrected to compare
FP32 softmax outputs before conversion; the original job/source artifacts are
retained and are not admission evidence. Corrected GPU execution is pending.

Corrected job 622 compares FP32 softmax weights from the same native logits
and mask before casting. Maximum FP32 discrepancy across heads is
5.960464477539063e-8. Casting both outputs to BF16 leaves 18 unequal weights:
head0 1, head5 1, head7 16, other heads 0. Thus tiny softmax rounding differences
can cross BF16 boundaries even with native logits; their effect on the full
PWA residual has not yet been measured. Source
`boltz-softmax-fp32-source-20260908-i5ljkD`, output
`boltz-softmax-fp32-result-20260908-91hMSV/candidate`. Verification: job 622
exits 0; headwise report inspected. Mask/stage hashes are checked by the probe;
live post-snapshot change is line wrapping only. No runtime default changed.

Native-softmax full-PWA counterfactual job 623 completes with **all 124,066,048
output values bitwise equal** (zero RMSE/max error/nonfinite values). Source
`boltz-softmax-intervention-source-20260908-liwyIU`; result
`boltz-softmax-intervention-result-20260908-fPDDaQ/candidate`.
Exactly eight native FP32 softmax outputs replace the corresponding calls;
runtime BF16 conversion, value/gate projection, weighted contraction and output
accumulation otherwise follow the current PWA code. This supports prioritizing
the logits/softmax boundary: under these supplied weights the remaining path
can reproduce this finite full-size native PWA output. It does not prove every
intermediate or other input agrees, nor admit a runtime using native arrays.
The report's `passed=true` describes this counterfactual only and retains
`not_model_parity_admission=true`. Same-snapshot baseline/control and compiler
graph effects remain relevant to causal attribution.
Verification: job 623 exits 0, capture complete, source/reference bindings
unchanged and exact output comparison inspected. Production code unchanged.

Same-snapshot baseline job 624 reproduces 99,059 unequal PWA values, RMSE
0.00007974575943452667, maximum 0.125. Source/reference, original input,
weight and target bindings match job 623; only the supplied native softmax
array is additional in the counterfactual. Therefore the exact counterfactual
is not explained by a changed source snapshot. It prioritizes reproducing
native logits/softmax weights without substitution, while retaining the
finite-input and changed-compiled-graph limitations above.
Verification: job 624 exits 0 and paired bindings/comparison reports checked.

### Bound softmax baseline (job 625)

The standalone diagnostic now records its source digest, reference-report
digest, JAX/device identity, per-head compiled HLO digest and same-executable
bitwise repeat check. Job 625 reproduces job 622: FP32 maximum absolute error
5.960464477539063e-8 and BF16 unequal counts `[1, 0, 0, 0, 0, 1, 0, 16]`.
All eight repeats are bitwise equal and the bound files remain unchanged.
This is an operator diagnostic, not coordinate RMSD or model admission.

The local Boltz publisher environment identifies Torch 2.12.0+cu130, matching
the capture, with git version 7661cd9c6b841b62b7f411aa52ec51f05457263b.
Its PersistentSoftmax.cuh contains lane-strided sequential accumulation,
descending XOR warp reduction and elementwise division. The actual dispatched
kernel still needs a native profiler capture before attributing the discrepancy
to that implementation; other installed Torch versions are not its authority.

Verification: queued GPU job 625 exits 0; report inspected; production-probe
regression tests 31 passed; diagnostic Ruff, syntax and diff checks pass.
No production softmax change, full-model rerun, release gate or push in this step.

### Native kernel confirmation and mask correction (jobs 626–627)

The diagnostic had used a 1e9 mask penalty whereas both native PWA and the
FoldJAX production default use 1e6. Corrected the diagnostic, not production.
Job 626 profiles Torch 2.12.0+cu130 on the captured operands: all eight heads
and their repeats are bitwise equal to the stored FP32 softmax outputs.
Every head captures `softmax_warp_forward<float, float, float, 9, false,
false, 32>`, establishing the previously tentative kernel dispatch.
Job 627 uses the corrected 1e6 mask penalty in JAX and reproduces all prior
head-wise error counts and maxima, including the 18 BF16 differences. The
diagnostic mask mismatch therefore does not explain this finite panel.

A bench-only warp-reduction control now follows the 16 lane-strided terms
and descending XOR reduction, with JAX exp/div unchanged. It is not selected
by production. CPU coverage checks a sequential NumPy reference with numerical
tolerance (not a bitwise exp oracle) and rejects other widths/dtypes.
Verification: jobs 626 and 627 exit 0 and reports inspected; new CPU tests
3 passed, existing production-probe tests 31 passed; Ruff and diff checks pass.

Job 628 completes the warp-reduction control: BF16 unequal counts become
`[0, 0, 0, 0, 0, 0, 0, 0]`, compared with 18 total in job 627. FP32 outputs
are not exact (maximum absolute error 2.9802322387695312e-8); all eight
same-executable repeats are bitwise equal and source/reference bindings remain
unchanged. This supports matching reduction order at the consumed BF16
boundary on this native-logits panel. It does not yet establish parity using
FoldJAX-generated logits, full PWA, other shapes, or end-to-end structures.
Verification: job 628 exits 0 and per-head report inspected. The control remains
bench-only pending production-boundary and broader validation.

### Computed warp softmax inside full PWA (jobs 629–630)

Job 629 computes FoldJAX logits and the warp-reduction softmax inside the full
first-layer PWA, without reference logits/weights substitution. It retains
94,390 unequal values out of 124,066,048, RMSE 7.890265571313988e-5, maximum
0.125, and no nonfinite outputs. Same-source default job 630 reproduces
99,059 unequal values, RMSE 7.974575943452667e-5 and maximum 0.125.
The modest improvement does not close PWA or prove that standalone softmax
rounding carries through the combined compiled graph. Remaining native/FoldJAX
logits differences and compiler integration must be separated next.

The bench option can now combine computed warp softmax with native-logits
substitution for that diagnostic; reference-softmax substitution remains
mutually exclusive with computed warp softmax. This combined option has not
yet been GPU-run. Neither option changes production defaults.
Verification: jobs 629–630 exit 0 and comparison reports inspected; related
tests 37 passed, Ruff and diff checks pass. No full-model admission or push.

### Combined native-logits/computed-softmax control (job 631)

With native logits substituted but softmax computed by the warp-reduction
candidate, full first-layer PWA is bitwise equal: all 124,066,048 output
values match, zero RMSE, maximum error and nonfinite values. This establishes
that the candidate can preserve the consumed softmax boundary inside the
combined graph on this finite panel. Together with job 629 it prioritizes the
remaining logits projection discrepancy. It is still a counterfactual, not
an independently computed FoldJAX PWA pass or a unique kernel-causality proof.

A CPU diagnostic using BF16-rounded original operands, FP64 matrix products,
then FP32 and BF16 rounding differs from native logits at 22 values (maximum
0.0625). Thus merely increasing accumulation precision does not reproduce
native rounding. This diagnostic does not prove correctly rounded BF16 dots
(the FP32 intermediate can double-round) or identify cuBLAS accumulation order.
The prior native logits profiler identifies the cuBLAS internal gemvx kernel.

Verification: job 631 exits 0 and exact full-output report inspected; CPU
FP64 control exits 0; combined/default wrapper tests plus softmax tests
40 passed; Ruff and diff checks pass. No production default change or push.

### Logits sum-order hypothesis and GPU replay (job 632)

A source/reference-bound CPU control evaluates contiguous/strided accumulation
with 1, 2, 4, 8, 16 and 32 lanes, sequential FP32 per-lane sums and descending
XOR reduction. Strided four-lane accumulation alone matches all captured
native BF16 logits; other controls differ at 12–26 values. This is a measured
finite-input hypothesis, not source-level identification of cuBLAS gemvx.

The JAX implementation of that arithmetic is tested in job 632 on the original
full normalized input and head-wise weights: all 1,527,752 logits are bitwise
equal, repeated execution is bitwise equal, and all values are finite. No
reference logits are substituted in this projection control. The full PWA
integration is a separate experiment; no runtime default is changed here.
Verification: CPU diagnostic and GPU job 632 exit 0; exact reports inspected.

Job 633 integrates both computed four-lane logits and computed warp softmax
inside first-layer full PWA. No native logits or softmax weights are injected
(both intervention flags are false). All 124,066,048 output values are bitwise
equal, zero RMSE/max error/nonfinite values, capture complete, source/reference
and saved-output bindings unchanged. The original captured PWA input and
weights are still the operator-test operands: this is not independently
preprocessed full-model inference. The implementation remains bench-only;
production integration, broader shapes/operands, full MSA/trunk, and n=5
entity-wise structure/confidence checks remain required.
Verification: job 633 exits 0 and report checked; affected diagnostics and
boundary tests 73 passed; Ruff and diff checks pass. No commit or push.

### Runtime integration (job 634)

The measured logits and softmax arithmetic is now in FoldJAX's private
`native_pwa_weights` primitive and the AMP PWA route. Selection is restricted
to the observed single-batch 437-token, 128-channel, single-head BF16 CUDA
profile without context parallelism; other profiles retain existing arithmetic.
This restriction records current validation scope, not universal shape support.

Job 634 runs the actual production PWA without any bench interventions or
computed-control flags: all 124,066,048 outputs are bitwise equal, maximum and
RMSE zero, all finite, and source/reference/output bindings unchanged. The
operator operands remain captured, so independent preprocessing/full-model
closure is not implied. CPU arithmetic/fallback-scope and affected diagnostic
tests: 109 passed; Ruff and diff checks pass. Full MSA job 635 is queued after
634 using the same source and the existing split-k=1 embedding control.
Verification: job 634 exits 0 and report inspected; runtime integration is
verified at PWA scope only. Independent review, full model and push remain open.

Job 635 completes full MSA with all 226 original weight leaves exact and no
optional native-norm/dense/row-shape overrides (split-k=1 remains explicit).
First-layer input_m, PWA, MSA transition and OPM are now exactly equal.
The first differing recorded operation is triangle multiplication outgoing:
3,353 unequal values, maximum 0.00390625, RMSE 1.6013058298167042e-6.
Incoming multiplication then differs at 2,972 values, starting attention at
7,598 and ending attention at 52,960. Final MSA pair output still differs:
24,408,429 unequal values, RMSE 0.02914297502347775, maximum 1.97705078125.
These are representation values, not coordinate angstroms. This moves the
first observed boundary downstream; it does not establish full MSA parity.
Verification: job 635 exits 0, capture complete and stage/output report inspected.
The complete Boltz model CPU suite after integration reports 654 passed and
6 skipped in 123.40 seconds. Skipped coverage is not counted as verification;
this run did not print individual skip reasons. Syntax, affected Ruff and
diff checks pass. Independent review, remaining triangle discrepancy, full
model structural/confidence gates and publication are still open.

### First outgoing triangle boundary (jobs 636–637)

Added a bound standalone triangle diagnostic using the native first-layer
input_z plus OPM output and original eight parameter leaves. Native job 636
reproduces the stored triangle output bitwise while capturing both norms,
gated projections and contraction. Candidate job 637 decomposes the existing
cuEq AMP route and also runs its un-decomposed production function. Captured
and production output are bitwise equal; production reproduces the MSA
observation's 3,353 unequal values, max 0.00390625, RMSE
1.6013058298167042e-6. No model or triangle production change in this step.

The first observed difference is input cuEq LayerNorm: 10,953,689 FP32 values
unequal, max 2.86102294921875e-6, RMSE 7.877645631019198e-8. BF16-rounding both
captures still leaves 945 unequal values, max 0.0625. Gated input GEMM differs
at 6,712 values; contraction at 20,881; output norm at 11,654. This prioritizes
norm arithmetic/implementation investigation, not a claim that norm is the
sole cause of all downstream differences. No reference intermediates were
substituted in this candidate capture.

Verification: jobs 636–637 exit 0; native reproduction, candidate production
and capture bridge reports inspected; CPU BF16-boundary comparison exits 0;
new diagnostic Ruff, syntax and diff checks pass. These are operator captures,
not full-model gates or performance measurements. Candidate uses default JIT
compiler options; numerical production bridge is observed, not a claim of
identical compiled kernels to the full MSA executable.

### cuEq norm compiler-policy control (job 638)

Torch and JAX environments carry byte-identical fused_layer_norm_triton.py
(SHA256 15d0bbc3a7f88b9b569acb6a8cc49d2de53e1daf2a7a4b18fbe8a4fd19d512bc)
and use 64x64 tiles, eight warps and two stages. The JAX cuEq triton_call
defaults enable_fp_fusion=False; the native Torch launch does not override
Triton's enable_fp_fusion=True default. Triton versions differ: native 3.7.0,
JAX 3.7.1. These observations do not establish compiler identity.

Job 638 changes only the JAX norm launch's fusion flag through a temporary
diagnostic patch. Full triangle output differences decrease from 3,353 to
2,076 (max 0.00390625, RMSE 1.485374794206504e-6); captured and un-decomposed
outputs remain bitwise equal. Input norm still differs at 7,479,505 FP32
values, max 2.86102294921875e-6. Input gated GEMM differs at 4,521 values,
contraction at 13,864, and output norm at 1,658. Thus fusion mismatch is a
supported contributor, not a sufficient explanation or completed fix.
No dependency or production setting changed. Further compiler/specialization
and mean/rstd evidence is needed before claiming native-faithful norm parity.
Verification: queued GPU job 638 exits 0 and boundary/production reports
inspected; diagnostic syntax, Ruff and diff checks pass.

### Norm statistics discrimination (jobs 639–640)

The diagnostic now optionally captures first input norm mean/rstd directly
from native cuEq and the JAX primitive. Native job 639 reproduces both the
original triangle output and input norm output exactly. Candidate job 640
keeps the fusion control enabled: mean differs at 152,737 of 190,969 rows
(max 1.9371509552001953e-7), rstd at 54,752 rows (max
2.9802322387695312e-8), and norm output at 7,479,505 entries. The triangle
capture-to-production output bridge remains exact. Thus the remaining issue
is already present in mean reduction, not only variance/affine fusion.

A CPU check of pairwise first/last-64 accumulation followed by either a
halves tree or adjacent tree does not reproduce native mean (146,488 and
156,647 unequal rows respectively). Do not adopt either guessed order as
native. Compiled reduction layout/specialization remains to be examined.
Verification: jobs 639–640 exit 0, native reproduction and statistics reports
inspected, CPU controls exit 0, syntax/Ruff/diff checks pass. No production
norm policy or dependency changes; full model closure remains unproven.

### Compiled layout and alignment control (jobs 641–644)

Fresh isolated Triton caches from jobs 641–642 identify the actual norm
compilations. Native FP32 pointers carry tt.divisibility=16 and the blocked
layout has sizePerThread=[1,4], threadsPerWarp=[2,16], warpsPerCTA=[8,1].
JAX pointers lack this attribute and use sizePerThread=[1,1],
threadsPerWarp=[1,32], warpsPerCTA=[4,2]. Thus equal kernel source and tile
sizes did not imply equal compiled reduction layouts.

Job 643's diagnostic function replacement of ASTSource fails to compile and
is excluded. Job 644 retains ASTSource's class contract via a subclass and
adds the native pointer-divisibility attributes to norm compilation only,
with the existing fusion control. Input norm mean, rstd and output now match
bitwise, as does the gated input GEMM. First remaining boundary is contraction
(966 unequal entries); output norm differs at 264 entries. Final triangle
output differs at only 52 entries, maximum 3.0517578125e-5, RMSE
7.189783248690989e-9; capture-to-production bridge remains exact.

This is a private diagnostic compiler patch, not a production change.
Production implementation must establish the relevant input/output buffer
alignment guarantees and avoid process-global compiler mutation. Remaining
contraction differences and broader model/profile gates remain separate.
Verification: jobs 641–642 and 644 exit 0; 643 exits 1 at compilation; compiled
IR layouts and 644 per-boundary report inspected. Live wrapper class-name lint
fixed after the immutable 644 snapshot; syntax/Ruff/diff checks pass.

### Explicit vector-four mean without alignment assumptions

Inspection of native FP32 PTX shows elementwise addition of the two 64-channel
tiles, sequential sum of each four-element group, then xor8/4/2/1 reduction
over 16 lanes. The earlier adjacent-tree hypothesis paired the four terms
differently. An explicit NumPy implementation of this PTX order reproduces
all 190,969 native means exactly. Added the bounded arithmetic diagnostic and
cancellation-sensitive tests. This offers a route to explicit arithmetic
without global ASTSource mutation or unproved pointer-alignment hints.
It does not yet implement or verify variance, rsqrt, affine output, or GPU
production normalization. Existing OpenBind norm cannot be substituted:
its variance/epsilon contract is different.
Verification: captured native means compared with the new helper, zero unequal;
3 focused tests passed, Ruff and diff checks pass. No production norm change.

### Explicit input CUDA norm candidate (jobs 645–646)

Added an unselected private native_cueq_norm candidate using explicit FP32
PTX arithmetic: vector-four mean reduction, centered-square FMA accumulation,
separate variance division/epsilon addition, approximate FTZ rsqrt, normalized
multiply and affine FMA. It does not assert pointer alignment or modify the
process-global compiler. Job 645 fails during Pallas lowering because ordinary
array slice is unsupported; replacing those slices with lax.split resolves
the compilation failure. No numerical evidence is taken from job 645.

Job 646 reproduces native mean, rstd, input norm output and input gated GEMM
bitwise. Contraction still differs at 966 entries. Output norm is deliberately
unchanged in this control and differs at 9,892 entries, final triangle at 352
(maximum 0.001953125); captured and un-decomposed output agree exactly.
This does not contradict job 644's 52 entries: that control changed both
input and output norms' fusion/alignment, while 646 replaces input norm only.
The explicit implementation is in src but not selected by inference yet;
broader boundary tests, output norm and full MSA integration remain required.
Verification: 645 exits 1 at lowering, 646 exits 0 with exact input-boundary
reports inspected; 3 arithmetic tests pass, Ruff and diff checks pass.

### Input norm runtime selection (jobs 647–648)

Added pre-launch rejection for scalar/empty/non-FP32/non-128-wide inputs and
invalid affine shapes. Connected the explicit input norm to the existing
native AMP cuEq route for the observed (1,437,437,128) shape; other shapes
keep the prior cuEq input norm. No global compiler patch is used.

Job 647's production result improves but its hand-decomposed diagnostic still
uses the old input norm, so its capture bridge is not admitted. Updated the
diagnostic to follow runtime selection and ran job 648: input norm and gated
input GEMM are exact, capture-to-production bridge is exact, final output
retains 352 unequal entries, max 0.001953125 and RMSE
3.9676712647116004e-7. Output norm and contraction remain unresolved.
Verification: jobs 647–648 exit 0 (647 stale decomposition excluded); 648
boundary/production reports inspected; affected triangle/boundary tests
18 passed and arithmetic tests 3 passed; Ruff/diff checks pass. Full MSA,
broader-profile regression, independent review and publication remain open.

### Same-operand contraction controls (jobs 649–651)

Added a standalone outgoing contraction diagnostic using the native captured
gated-GEMM output. Native job 649 reproduces the stored contraction bitwise
and repeats exactly. Its profiler identifies
cutlass_75_tensorop_bf16_s1688gemm_bf16_128x128_tn_align1.
JAX job 650 reproduces the 966 unequal contraction values on the same operands,
RMSE 0.4143155556367285, max 2048, and repeats exactly. These unnormalized
internal representation magnitudes are not coordinate angstroms.

Disabling Triton GEMM in job 651 does not resolve the difference: 999 unequal
values, RMSE 0.4142621182519704, max 2048, exact repeat. Its compiled HLO pads
437 to 440 and calls __cublas$lt$matmul, so it is not the same native kernel
or original-shape execution. Do not promote this compiler flag to production.
The remaining arithmetic/kernel difference is reproducible without upstream
normalization drift; output norm still requires its own repair/verification.
Verification: jobs 649–651 exit 0; native profiler, comparison/repeat reports
and JAX HLO inspected. New diagnostic syntax/Ruff/diff checks pass. No runtime
contraction change or full-model admission in this step.

### Output norm statistics and distinct reduction profile (job 652)

Extended native statistics capture to BF16 dbij->bijd output normalization.
Job 652 reproduces both original norm outputs and the full triangle exactly.
Native PTX's output profile accumulates 16 channel terms strided across four
groups before xor2/1 reduction, unlike the input profile's contiguous four-term
groups and xor8/4/2/1 reduction. On captured native contraction operands,
the strided profile reproduces all 190,969 output means exactly; the initially
tested contiguous-16 interpretation differs at 131,409 rows and is rejected.

Added an unselected output-norm arithmetic candidate with a separate reduction
profile and BF16 output conversion. It has not yet been GPU-tested; mean
agreement alone does not validate variance/rsqrt/affine or the final norm.
Existing input-norm default behavior remains unchanged.
Verification: job 652 exits 0 and native norm reproduction assertions pass;
CPU output-mean comparison has zero unequal; existing boundary/arithmetic
tests 9 passed, Ruff and diff checks pass. Output candidate GPU validation,
contraction and full-model closure remain open.

### Output norm GPU validation and runtime integration (jobs 653–655)

Job 653 directly evaluates the explicit output norm on hash-checked native
contraction operands and original affine weights: mean, rstd and BF16 output
are all bitwise equal, and two same-executable runs are bitwise equal. The
short diagnostic is operator evidence, not full provenance/release admission.
Connected output norm to the observed (128,1,437,437) native AMP route; other
shapes remain on the previous cuEq path. Boundary/triangle tests: 21 passed.

Job 654's actual un-decomposed production triangle is now bitwise equal to
native. Its hand-decomposed intermediate-returning function still differs at
52 output values (max 3.0517578125e-5), so the capture bridge fails. Do not use
those intermediates as an exact explanation of the production executable:
returning stages changes the compiled graph. Actual output equality is the
validated observation, not proof that every internal contraction is equal.
Full MSA job 655 uses the same snapshot and explicit split-k=1 control.
Verification: jobs 653–654 exit 0 and numeric reports inspected; 21 affected
tests pass; Ruff/diff checks pass. Job 655 pending at this record. No full-model
admission, independent review or publication yet.

### Full MSA after both norm integrations (job 655)

The same-source full MSA control completed. Every captured stage in layers 0
and 1, including both layer outputs, is numerically exact against native.
Layer 2 starts with exact input_m and first differs at PWA: 6,349 unequal
values, RMSE 9.770757765801931e-5, maximum 0.125. Final MSA pair output has
RMSE 0.022976575405325 and maximum 1.98486328125. These are internal features,
not coordinate angstroms. The explicit split-k=1 setting remains a diagnostic
control, not a production default. Exact early layers do not establish model
closure or generalize the shape-specific norm implementation.

Verification: job 655 exited 0; its stage and output comparisons were inspected.
The full Boltz CPU suite completed with 663 passed and 6 skipped in 123.01 s.
All skips require unavailable Torch: three featurizer cases and one each in
import, micro-module and template parity tests. No skipped comparison is
counted as passed. The next whole-model 5SAK replay (job 656) retains the
historical capture settings without the split-k control, to measure final
structure/confidence effects separately from this internal diagnostic.

### Whole-model current-source 5SAK replay (job 656)

Completed n=5, 200 steps, 3 recycles with native BF16-mixed tape and features,
without the split-k control. Whole-system proper Kabsch then entity-only
measurement gives protein RMSDs [16.129029666, 2.294899474, 0.080331749,
5.568700800, 1.687943564] angstroms and ligand RMSDs [4.787218459,
0.248918784, 0.010373779, 0.557804118, 0.261104275]. Raw and public confidence
both fail the retained strict tolerances. This is not deferred gray-zone
evidence: multiple samples exceed 0.1 angstrom substantially. Earlier v36
maxima were 2.837167097 protein and 0.366322512 ligand; this observation is
worse but does not isolate a single code change or compiler cause.

Verification: job 656 exited 0; capture completed and the artifact-bound
boltz_amp_report completed successfully. A successful report command is not
a parity pass. Source snapshot suffix VMPbYS binds this replay; the report is
comparison-5sak.json alongside that snapshot. Early MSA exactness was measured
with split-k=1, unlike this whole-model replay. A same-source split-k-only
counterfactual is therefore next; no default setting is changed. Shared native
features and entry tape hashes still do not prove independent preprocessing
or per-step device-consumer identity. No model closure or publication.

### Same-source split-k-only counterfactual (job 657)

The full replay with split-k=1 also fails. Protein per-sample RMSDs are
[16.129929140, 2.310915048, 0.077977813, 5.569800208, 1.673339401] angstroms;
ligand RMSDs are [4.785021630, 0.258373487, 0.010183037, 0.552198595,
0.261178743]. Raw/public confidence both fail strict checks. First-cycle MSA
delta_z RMSE remains 0.032341485909008746; final pair output RMSE remains
0.07724685086720408. Thus adding this flag does not recover the full-model
failure. This rejects the flag's absence as a sufficient explanation, not all
compiler sensitivity or sampler differences. No production flag change.

Verification: job 657 exited 0 and artifact-bound comparison generation exited
0; numeric fields inspected in comparison-5sak-splitk.json alongside the same
VMPbYS source snapshot. An independent read-only diagnostic consultation is
pending; it is not a source review or release approval.

### Direct candidate comparison and downstream isolation

The v36 and job 656 effective-options dictionaries are equal, including
feature/tape identities. Hash-verified prediction archives differ directly:
single_inputs RMSE 1.7551137558613502e-9, single 0.06497726668815952, pair
0.09755741201481452 (pair max 10.131065368652344). Similar native-relative
aggregate errors therefore did not mean similar candidate trunk arrays.
Candidate-to-candidate entity maxima are protein 15.781291962881518 and
ligand 4.459245135197369 angstroms.

A 5-by-5 whole-system Kabsch diagnostic finds nearest native indices
[2,1,2,3,4] for candidate indices [0,1,2,3,4]. Candidate 0 remains over 15
angstroms from every native sample. This is not a simple sample permutation
and does not establish incorrect tape consumption. No rematching is used for
acceptance. Per-step tape consumption remains unverified.

Independent read-only consultation completed and suggested direct-array and
native-trunk downstream controls. These suggestions are hypotheses, not an
admission review. Job 658 repeats the preserved v36 executable; a separate
current-source native-conditioning downstream control is queued. No production
change is made pending these discriminating experiments.
Verification: capture hashes checked before direct-array comparisons; CPU
comparisons completed and effective-option equality inspected.

### Preserved v36 reproduction (job 658)

The preserved v36 source/harness completes again on the current runtime.
Protein RMSDs against native are [2.017117177, 0.188800821, 0.103545534,
2.835027474, 1.929296045]; ligand [0.158371045, 0.016405285, 0.019703939,
0.365654695, 0.165471438]. Both confidence strict gates remain false.
This reproduces the earlier approximate 2.837/0.366 maxima, not bitwise output:
the pair representation is exactly equal to original v36, but single RMSE is
0.010226279721996122 and unaligned coordinate-array RMSE 0.0370091604285398.
The latter is not a Kabsch RMSD or an acceptance allowance. Thus the old
baseline remains approximately reproducible while the current-source 16 A
failure is not explained by that observed old-source variation alone.

Verification: job 658 exited 0; artifact-bound comparison generation exited 0
and outputs inspected. An initial read before report completion found no file;
the same live report process was awaited, not restarted. Job 659 is running.

### Current sampler with native trunk and conditioning (job 659)

The substituted n=5 downstream diagnostic completes with maximum entity RMSD
0.00386380786170464 protein and 0.00040469468498928387 ligand angstroms.
This supports near-native coordinates for this isolated sampler configuration,
not the complete model, confidence, independent preprocessing or per-step tape
consumption. Native conditioning bypasses FoldJAX conditioning as well as its
trunk; the result cannot distinguish those two sources of full-model drift.
The complementary native-trunk/FoldJAX-conditioning arm is queued next.
Verification: job 659 exited 0; comparison.json inspected and its coordinate
diagnostic passes 0.05 angstrom. No complete-model admission is made.

### Native trunk with current FoldJAX conditioning (job 660)

The complementary downstream control passes the coordinate diagnostic with
protein maximum 0.007799017835426813 and ligand maximum
0.0006016225629647625 angstroms. This is better than the recorded historical
FoldJAX-conditioning control (0.2534417826/0.0162096412); it is not proof of
unchanged downstream numerics. Together with 659, it motivates investigating
the changed trunk, while not excluding whole-graph compilation interactions.
Confidence and independent tape-consumer proofs are outside these controls.

Job 661 isolates the PWA change group: a separate current-source copy replaces
only msa.py with its v36 arithmetic (norm, logits, softmax and contraction),
leaving current triangle norms and all other source unchanged. The resulting
msa.py is byte-equal to preserved v36 by cmp. This is a diagnostic ablation,
not a production rollback or an upstream-faithfulness recommendation.
Verification: job 660 exited 0; comparison inspected. Job 661 submitted;
numeric result pending. The actual worktree runtime remains unchanged.

### Old-PWA/current-triangle ablation result (job 661)

With only msa.py restored to v36, protein RMSDs are [0.9825421410039729,
0.07711559036978133, 0.06921617940717024, 2.773688758062704,
1.4016367991810417]; ligand [0.04849629833523935, 0.007578076195042999,
0.011180822079137887, 0.443931182746228, 0.23109846131357187]. Both strict
confidence checks fail. This intervention markedly reduces job 656's 16.129 A
protein maximum, but is not a passing fix or proof that old arithmetic is more
faithful. It localizes a strong full-graph effect to the PWA change group
(norms, logits/softmax, contraction and the graph changes they induce).
Those changes must next be separated, retaining the original native-boundary
evidence rather than optimizing only the final coordinate score.
Verification: job 661 and artifact-bound comparison generation exited 0;
numeric fields inspected. Report/downstream regression tests: 36 passed;
git diff --check passed before this record. No production rollback or push.

### Single-change PWA norm ablation queued (job 662)

A fresh copy of job 656 source changes only the PWA local norm selector from
amp_layer_norm to _layer_norm. Both norm_m and norm_z use this selector;
logits, softmax, contraction, triangle norms, tape and full n=5 settings remain
current. The one-line source diff was inspected before submission. This tests
the norm group, not each norm independently and not a proposed runtime fix.
Verification: job 662 submitted through the serial GPU queue after confirming
no compute process occupied the device. Numeric result pending.

The complementary job 663 starts from the same current snapshot and restores
only ordinary PWA projection/softmax, retaining current PWA norms, contraction
and triangle norms. It is queued after 662, not run concurrently. These two
single-group ablations test different interventions; neither changes the
production tree or implies upstream equivalence from a smaller RMSD.

### PWA norm-only ablation result (job 662)

Restoring only ordinary PWA norm gives protein RMSDs [1.9293051615541676,
0.16354887630187773, 0.015167713671226735, 1.6310896092819755,
1.6462895651182037]; ligand [0.15474770427213144, 0.017276828905309242,
0.010181540856558431, 0.08736459295550757, 0.16307789535624637]. Raw/public
confidence both fail strict checks. The norm-group intervention alone removes
the observed 16 A maximum, but still fails model parity. Exact first-layer
native norm evidence is not overturned by this output-score improvement:
later-layer operand rounding and full-graph interactions need investigation.
Do not select the old norm as a fix merely because these RMSDs are smaller.
Verification: job 662 exited 0; artifact-bound comparison generation completed
and per-sample fields were inspected. Job 663 is the complementary weights
control; no production code changed here.

### PWA logits/softmax-only ablation result (job 663)

Restoring only ordinary logits/softmax gives protein RMSDs
[2.796083717676901, 0.25691007836041296, 0.09480930366154726,
2.88384285264388, 2.1473401885040806]; ligand [0.21086829907008633,
0.02740657957654455, 0.0143397896779638, 0.39284927197302466,
0.15874520001309025]. Raw/public confidence still fail strict checks.
Either norm-only or weights-only restoration reduces the 16 A observation;
therefore the ablations do not isolate a unique faulty primitive. They reveal
sensitivity to interacting arithmetic/compiled-graph changes. Stop selecting
by final RMSD and return to the first remaining same-input native boundary
(MSA layer 2 PWA) for arithmetic evidence. Existing early-layer exactness
remains limited to its observed operands and diagnostic graph.
Verification: job 663 and artifact-bound comparison completed with exit 0;
per-sample outputs inspected. No production rollback, gate change or push.

### Layer-aware PWA diagnostic and pinned-source preflight (jobs 664–665)

Extended the first-layer diagnostic to select layers 0..3, load the preceding
layer's hash-bound pair output, and select matching weights. Layer 2 input_m
metadata is native FP32; the old first-layer-only runner always cast m to
BF16. The new native/candidate loader preserves recorded dtype instead of
introducing a lossy input change. This is a diagnostic correction, not evidence
that production made this cast. Seven tests cover selection, FP32 preservation,
invalid layers and prior-stage tampering; Ruff/diff checks pass.

Job 664 stopped at native source identity validation: the general Boltz work
checkout differs from the pinned MSA source. No numerical evidence is admitted
from that failure. The original pinned upstream-root checkout was located and
its complete source hash mapping verified equal before submitting job 665 with
that checkout and its original native environment. The guard is not weakened.
Verification: 664 exits 1 before model execution; pinned source hash equality
confirmed; 665 submitted for layer 2, all 4436 MSA rows.

Job 665 completed: layer 2 native decomposition equals its actual PWA output
and the original full-MSA capture exactly (zero unequal values, zero maximum
and RMSE). Recorded input_m is torch.float32. This supplies the validated
later-layer operands/stages for the next candidate boundary comparison.
Verification: 665 exited 0 and both exact-reproduction reports inspected;
candidate layer-2 boundary validation remains pending.

Updated candidate decomposition to call current runtime AMP norm,
PWA logits/softmax and contraction helpers and preserve FP32 sigmoid before
BF16 conversion. The actual PWA output remains separately compared with the
decomposition: returning intermediates can still change the compiled graph.
Job 666 uses the validated layer-2 native FP32 input and full-row capture;
no native intermediate is substituted into its normal computation.
Verification: seven loader/CLI tests, Ruff and diff checks pass; job 666
submitted from a fresh source copy. GPU boundary results remain pending.

### Layer 2 first mismatch isolated to softmax weights (job 666)

The current candidate decomposition equals actual PWA exactly and reproduces
the full-MSA discrepancy: 6,349 unequal outputs, RMSE
9.770757765801931e-5, maximum 0.125. Both norm_m and norm_z, all head value
projections, logits and gates are exact against native. Softmax FP32 weights
are the first unequal boundary (maximum at most 2.9802322387695312e-8).
After BF16 conversion, unequal counts by head are [0,4,0,0,1,0,3,2]. These
ten weights straddle BF16 rounding midpoints. For example native head1 weight
0.0010261536808684468 becomes 0.00102996826171875, whereas candidate
0.001026153564453125 becomes 0.0010223388671875.

This same-input boundary refutes treating the norm-only coordinate ablation
as proof of an incorrect layer-2 norm. Next isolate softmax exponential and
division arithmetic without changing operands or choosing a final-score win.
Pinned Torch PersistentSoftmax.cuh uses sequential lane accumulation, warp
reduction and elements/sum division; source syntax alone does not establish
the generated CUDA/XLA arithmetic identity.
Verification: job 666 exited 0; exact decomposition bridge and stage report
inspected; BF16 conversion comparison on stored weight arrays completed.
This is internal feature evidence, not coordinate angstroms or model closure.

### Explicit FP32 softmax division control (job 667)

Added a bench-only Pallas div.rn.f32 boundary, keeping the existing exponential
and warp reduction implementation. On the same layer-2 native logits/mask,
all eight heads' FP32 softmax outputs now match native exactly, including all
BF16 conversions. Same-executable repeats are bitwise equal. This intervention
eliminates the ten BF16 boundary mismatches observed in 666. It supports the
final division/lowering boundary as a repair target for these operands, not a
universal claim about exp or all softmax profiles. Production still uses the
previous division pending integration and regression verification.
Verification: job 667 exited 0; all eight reports have zero FP32/BF16 unequal
and repeat_bitwise_equal=true; reference/source bindings unchanged. Five CPU
tests pass (including division guard tests); import-order lint fixed and diff
check passes. Full PWA, MSA and model reruns remain required.

### Runtime division integration (jobs 668–669)

Carried the explicit div.rn.f32 operation into native_pwa_weights.py and
selected it in the existing CUDA warp-softmax route. CPU/other-platform
division and the existing shape/context-parallel fallback guards remain
unchanged. No reference outputs or bench imports enter production. Pre-launch
guards reject non-FP32 operands and empty/incomplete 32-element blocks.

Job 668 validates the full layer-2 PWA, while 669 repeats the same-source
full MSA split-k diagnostic, preserving its original comparison policy.
Verification: 19 affected CPU tests pass, Ruff and git diff --check pass.
Both jobs are submitted serially from snapshot 7UNq7B; numeric validation,
whole-model rerun, performance measurement and independent source review remain
pending. Integration is not yet model or release admission.

Job 668 completed: actual layer-2 PWA, decomposed PWA, both norms and every
head intermediate now equal native exactly. The decomposition-to-production
bridge is exact. All 6,349 output discrepancies from 666 are eliminated by
the integrated explicit division. This establishes the observed layer boundary,
not full-model parity. The full-MSA control 669 is running and same-source
full n=5 5SAK replay is queued without the split-k diagnostic flag.
Verification: 668 exited 0; every stage comparison has zero unequal and bridge
maximum/RMSE are zero. Full Boltz CPU regression suite has been started.

Job 669 completed: all captured stages across all four MSA layers and the
final output are numerically exact against native (zero unequal, RMSE and
maximum). This closes the observed first-recycle, fixed-native-input MSA
diagnostic under its explicit split-k=1 setting. It does not establish other
recycles, shape profiles, full-model outputs or performance. Full 5SAK job 670
uses the same source without that diagnostic compiler flag and remains the
required downstream check.
Verification: 669 exited 0; all stage comparisons and final output inspected.

Post-integration full Boltz CPU regression: 666 passed, 6 skipped in 144.73 s.
The six skips are unchanged Torch-unavailable checks (three featurizer cases,
one import case, one micro-module case, one template case); no new pass claim
is made for those upstream comparisons.

### Whole-model after explicit PWA division (job 670)

Same-source 5SAK n=5 without the diagnostic split-k flag completes. Protein
RMSDs are [1.9397402998783282, 0.04421908212665783, 0.03528746645886707,
0.8673928442644442, 1.6668086428359057]; ligand [0.14862407236860173,
0.012001329715387253, 0.007730401608617936, 0.14758745548647167,
0.1309394758478917]. Both confidence strict gates remain false. This is an
improvement over 656 (16.129/4.787 maxima), not model closure.

The full-model first MSA input_z already differs from native by RMSE
0.0006116185148676643. Its delta_z RMSE is 0.03246507077702084 despite the
same-native-input MSA control being exact. Therefore the isolated MSA result
cannot be extrapolated to the real upstream-input path; investigate the earlier
input_embedder/initial pair boundary and retain possible graph effects.
Final pair RMSE is 0.0704263172740131, not coordinate angstroms.
Verification: job 670 and artifact-bound comparison generation exited 0;
per-sample coordinates, confidence and stage metrics inspected. No tolerance
change, model admission or publication.

### Input embedder addition-order control queued (job 671)

Stored native/job-670 input_embedder outputs differ at 302 entries after BF16
conversion. Native InputEmbedder and the port express the same sequential
residue/profile/method/modified/cyclic/molecule additions. This does not prove
compiled arithmetic identity. A separate source copy inserts optimization
barriers after intermediate additions and runs trunk-only with the same native
inputs to test sensitivity to reassociation/fusion; all model operands and
other source stay unchanged. A barrier can change the graph beyond arithmetic,
so improvement alone will not prove a compiler root cause.
Verification: CPU BF16-boundary count inspected; job 671 submitted through tsp.
The production embedder remains unchanged and GPU result is pending.

Job 671 completed but does not repair the boundary. Input embedder RMSE is
8.879957826115508e-6 versus job 670's 5.812567678630158e-6; first MSA input_z
RMSE is 0.0014086638115393105 versus 0.0006116185148676643. Barriers are
therefore not promoted. This is also a trunk-only/full-graph comparison, so
it cannot isolate a unique compiler cause; native-relative boundary errors
remain clearly nonzero. The next useful evidence is matched internal native
input-embedder stages, not another blind final-output ablation.
Verification: job 671 and artifact-bound comparison generation exited 0;
boundary fields inspected. Production remains unchanged.

### Native input-embedder detail capture (job 672)

Added opt-in --input-details to the native observer. It captures atom encoder
tensor outputs, projected atom bias, atom-attention encoder tensor outputs and
each residue/profile/conditioning embedding output through forward hooks.
Callable tuple members are explicitly outside tensor capture; the hook returns
None and does not replace the native output or callback. Default observation
scope is unchanged. Missing requested submodules fail rather than disappear.

Job 672 runs the pinned upstream copy with original native n=5/200/3 settings
and input YAML. Before using its added boundaries, compare its feature/tape
identity and top-level output with the original native capture; a fresh
observation is not automatically an identical reference executable.
Verification: 12 observer tests pass including non-replacing tensor detail hook;
Ruff and diff checks pass after fixing a test lambda lint error. Job 672 queued;
no numerical evidence or source review admission yet.

Job 672 completed and its capture hashes validate. Features, sampler tape,
preprocessing tape, effective native settings and complete input_embedder output
archives are hash-identical to the original native reference. Added internal
stages can therefore be compared at this input-embedder boundary without
assuming observation neutrality from intent alone.

Against job 670, atom_encoder q/c each have 98,474 unequal FP32 entries, RMSE
2.1445574334454545e-8 and maximum 4.76837158203125e-7. Pair p has 3,833,958
unequal entries, RMSE 1.1472596377253706e-7, maximum 2.1457672119140625e-6.
Atom-attention token output has RMSE 5.8125479986175875e-6, maximum
0.0007848106324672699; its atom output RMSE is 0.002414091290201778.
These are internal representation values, not angstroms. Initial q/c/p are
the earlier observed mismatch; embedding-output addition barriers did not
isolate it. Next inspect atom_encoder's same-feature arithmetic boundaries.
Verification: 672 exited 0; capture-bound artifacts and listed hash identities
checked; direct array comparison completed. Whole-model parity remains open.

### Atom input projection arithmetic inspection

Pinned native AtomEncoder.forward disables CUDA autocast before constructing
atom features and calling embed_atom_features. The port's corresponding kernel
and bias are FP32, so this boundary is not explained by selecting BF16 for
the first projection. For the observed (1,3104,388) features and (388,128)
kernel, FP64 dot followed by FP32 bias addition differs from native c in
168,407 entries (RMSE 2.531860457022868e-8); FP64 dot plus bias with one final
FP32 cast differs in 220,645 entries (RMSE 2.7312027358709998e-8). Neither is
native arithmetic. NumPy concatenation promotes mixed integer/float features
to FP64 in this diagnostic; exact native-valued coordinates and one-hot entries
are retained, not promoted as a production input policy.

These CPU high-precision controls reject blindly using a more accurate dot as
an upstream-fidelity fix. Native FP32 GEMM accumulation and bias handling remain
the next same-operand GPU boundary to distinguish. No runtime edit here.
Verification: pinned source read, managed parameter shapes/dtypes inspected,
two CPU arithmetic comparisons completed. Native parameter identity relies on
the earlier checkpoint audit, not this CPU experiment alone.

### Same-operand atom projection GPU controls (jobs 673–676)

Job 673 failed before computation because the native environment lacks the
probe's safetensors dependency. Job 675 reruns the identical source with
safetensors 0.6.2 in a separate temporary tool directory; the pinned native
environment itself is unchanged. Both native functional.linear and explicit
matmul followed by bias reproduce the captured atom c bitwise, including
same-call repeats. Both profiles contain the same CUTLASS SIMT SGEMM
64x128_8x5 kernel and a cuBLASLt split-K reduction. The explicit expression
also has a separate elementwise bias kernel.

JAX job 674 differs at 87,102 entries (RMSE 2.1614854279944467e-8,
maximum 4.76837158203125e-7) and repeats exactly. Job 676 adds a diagnostic
optimization barrier between GEMM and bias, retaining the actual production
linear as its control. Both arms have the same mismatch count, RMSE and maximum
as 674; both repeat bitwise. No production change is made from this probe.

Thus neither explicit native bias separation nor the tested JAX barrier
removes the discrepancy. GEMM arithmetic/lowering remains the next boundary;
this is not proof of the precise accumulation order or of the cause of the
whole-model structural drift. Isolated JAX mismatch count also differs from
the full-graph atom c capture, so an isolated match must still be bridged into
the actual model. These are captured-input/managed-weight controls, not new
independent preprocessing or checkpoint conversion evidence.

Artifacts: boltz-atom-projection-20260908-WMf5x5/{native,foldjax}.json and
boltz-projection-barrier-yHLpFW/foldjax.json under the external benchmark root.
Verification: jobs 675 and 676 exited 0; reports inspected, bound inputs
unchanged, native capture reproduction exact. The atom-feature ordering CPU
test passes (1 passed); affected Ruff and git diff --check pass. Job 673's
failure is retained. Full-model parity, confidence, performance and release
remain open.

### Atom projection backend and storage-layout controls (jobs 677–678)

Job 677 compares the production linear and barrier-separated expression with
baseline, split-K=1, no-Triton and no-Triton/no-cuBLASLt requested compiler
options. Baseline and split-K=1 retain 87,102 unequal entries and RMSE
2.1614854279944467e-8. Both no-Triton requests yield 98,474 unequal entries,
RMSE 2.1445574334454545e-8 and maximum 4.76837158203125e-7. All repeats are
bitwise stable. Actual emitted HLO contains __cublas$lt$matmul for BOTH of
the latter requests: no-cuBLASLt is not an effective non-Lt control here.
Bias-fused and barrier-separated HLO have BIAS and DEFAULT epilogues respectively
but the same aggregate errors. These summaries do not prove cross-arm array
identity. Artifact: boltz-projection-backends-vaU4eG/foldjax.json.

Job 678 changes only native weight storage stride from (388,1) to (1,128),
checking that weight values remain exactly equal. Original storage again
reproduces the captured native output exactly. Transposed storage instead
produces 98,474 unequal entries, RMSE 2.1445574334454545e-8 and maximum
4.76837158203125e-7, with exact same-call repeat. The profiled native GEMM
changes from cutlass_80_simt_sgemm_64x128_8x5_tn_align1 to
cutlass_80_simt_sgemm_128x64_8x5_nn_align1; both use split-K reduction.
Artifact: boltz-projection-layout-DugT4U/native.json.

This controlled native-only intervention establishes that memory layout can
cause a first-projection discrepancy with identical values, dtype and logical
operator. Its aggregate errors match the full-graph JAX atom c observation,
but direct cross-output equality and a production layout-controlled rerun are
still required. This does not yet attribute the final coordinate drift to this
boundary. No model runtime or precision defaults were changed.

Verification: 677 and 678 exited 0, numerical summaries, actual lowering,
native kernel profiles and unchanged bindings inspected. Two focused CPU tests,
affected Ruff and git diff --check pass. These controls do not admit Boltz or
establish performance benefits.

### JAX layout constraint and compile-selection sensitivity (jobs 679–680)

Added a diagnostic Layout((1,0)) constraint on the [input,output] managed
kernel, corresponding to contiguous native [output,input] storage. Job 679
does not improve either baseline or no-Triton discrepancies; its no-Triton
custom call records selected_algorithm 1. Job 680 adds a transposed-GEMM
expression and more complete HLO operand/copy/transpose evidence. The
transposed expression alone still fails. However native_storage/no_triton in
680 is bitwise exact against the native target, with selected_algorithm 0.
HLO confirms a [388,128]{0,1} operand reaches cuBLASLt. This corrects the
provisional hypothesis that the requested storage constraint necessarily
disappears before GEMM: a different selected algorithm can also explain the
earlier failed layout control.

The sources differ in the added expression/reporting, so 679 versus 680 is
not a same-source compile-repeat experiment. A fresh-process rerun of the
identical 680 snapshot is the next discriminating check. No production layout
or global compiler policy has been changed; one exact isolated compile is
not repeatability, full-graph parity, or model admission.

Artifacts: boltz-projection-native-layout-svGdNF/foldjax.json (679) and
boltz-projection-transpose-YXtFLB/foldjax.json (680).
Verification: 679 and 680 exited 0; actual HLO, target comparisons and
algorithm selections inspected. Two focused CPU tests, Ruff and diff checks
pass. Compile-selection stability remains unverified at this point.

Job 681 repeats the identical 680 snapshot in a fresh process. Source/input
bindings are identical. Native-storage/no-Triton again selects algorithm 0
and equals the captured target bitwise; baseline and split-K=1 still fail.
The report is boltz-projection-transpose-YXtFLB/foldjax-repeat.json.
Verification: 681 exited 0; identical bindings and exact target comparisons
inspected. Two fresh-process successes establish this limited repeat check,
not a guarantee across autotuning decisions or full-model compilation. The
679 failed compile remains part of the evidence.

### Actual-trunk layout bridge rejected (jobs 682–683)

The installed JAX staging rule rejects compiler_options on nested jit, so the
isolated no-Triton option cannot simply be attached to this one projection
inside the trunk. A matched diagnostic uses global no-Triton in both arms:
682 uses snapshot 7UNq7B (its src is unchanged versus the current repository),
683 uses a separate snapshot with only an observed-shape, non-CP layout
constraint at embed_atom_features. This is an external ablation, not a
production edit or a default compiler-policy recommendation. Both are
trunk-only and use the native feature/tape capture.

Both actual atom c captures differ from native in 98,474 entries, RMSE
2.1445574334454545e-8 and maximum 4.76837158203125e-7. Direct cross-arm
comparisons of atom q/c/p and final pair/single are numerically exact.
single_inputs differs at 52 entries (RMSE 1.4495694618059649e-9, maximum
2.384185791015625e-7). Thus the local layout edit has not reproduced the
successful isolated boundary in the actual graph; it is rejected as a fix.
No coordinate, confidence or performance claim follows from trunk-only runs.

Artifacts: boltz-trunk-layout-Nk2jsT/{control,layout} and the corresponding
control-report.json/layout-report.json under the benchmark root. Actual-trunk
GEMM algorithm selection remains unobserved in these captures. An explicit
kernel boundary or controlled algorithm selection must be evaluated before
claiming this arithmetic path is reproducibly implemented.

Verification: 682 and 683 exited 0; artifact-bound reports generated, native
atom c and direct cross-arm arrays inspected. Twelve affected CPU tests,
Ruff and diff check pass. A CUDA timer warning in 682 rules out using these
runs as reliable timing evidence. Production source remains unchanged by
this layout experiment; six-model closure and release remain incomplete.

### First-projection value substitution, not a fix (job 684)

Before implementing an explicit GEMM boundary, an external source ablation
replaces only non-structure-prediction atom c with its hash-bound native
capture. The sampler, native-policy settings, five samples and other runtime
code remain unchanged. The substitution is restricted to observed FP32,
non-CP shape (1,3104,128). This is a causal control, never model admission.
The native value is a constant, so graph/constant-folding effects remain a
confound; a companion using the original candidate c constant is queued as 685.

Job 684 completes. Actual q/c equal native numerically; atom p still differs
at 18,003 entries, RMSE 9.213901293788114e-9 and maximum
1.430511474609375e-6. Whole-system Kabsch then per-entity protein RMSDs are
[1.9173089432910133, 0.16532355439774277, 0.049220380139962654,
2.834526646528475, 1.6626691979190047]; ligand
[0.14947409503110937, 0.01473295829016785, 0.009918881809133027,
0.37179139305579695, 0.15591445670073154]. Thus an exact first projection is
not sufficient to close this execution. Do not infer that this boundary has
no effect, or that the worsening versus 670 is exclusively numerical rather
than graph-related, before the companion control.

Artifact: boltz-native-c-control-yB0pUZ/{result,comparison.json}.
Verification: 684 exited 0; artifact-bound report and actual substituted q/c
inspected. Thirty capture/report CPU tests pass; Ruff and diff checks pass.
No production edit, admission or timing claim.

Companion 685 completes with candidate c reproduced bitwise at the actual
boundary. Source differs from 684 only in reference archive path and expected
hash. Protein RMSDs are [13.191308621023394, 0.1916778252808667,
0.04512653586532514, 6.078366349579394, 1.6417974596522102]; ligand
[10.803521989468397, 0.018445332888533818, 0.011061873231554479,
0.5077666114214101, 0.1597992768783138]. Both 684 and 685 retain failed
raw/public strict confidence. Artifact:
boltz-candidate-c-control-X21R6e/{result,comparison.json}.

The native constant arm is closer than the candidate constant arm in this
comparison, but the candidate constant arm itself is substantially worse
than the original computed-c job 670 despite identical c values. This exposes
a major graph/compile confound in value-substitution inference. It rules out
treating 684 minus 670 as a clean causal effect of the first projection.
No sole-cause attribution follows; neither constant arm closes the model.
Future arithmetic work must preserve the runtime dependency and validate
the actual compiled boundary, rather than importing captured values as a fix.
Verification: 685 exited 0; actual candidate c equality, both artifact-bound
reports, per-sample metrics and failed strict confidence inspected. The
substitution code exists only in external diagnostic snapshots.

### Dependency-preserving FP32 library boundary (job 686)

Extended the existing development-only typed cuBLASLt FFI with a separate
FP32 entrypoint. Its original BF16 entrypoint and default Python behavior are
retained. FP32 requires FP32 operands without implicit narrowing; both paths
retain XLA-owned 32 MiB workspace and first-current-heuristic dispatch. This
is not an admitted runtime implementation or a pinned universal algorithm.
The probe binds the binary and wrapper/C++ source hashes and computes from
the actual captured input and managed weights rather than a reference output.

Job 686 FP32 FFI plus separate bias equals native c bitwise, including a
second invocation. Emitted HLO contains the typed FFI call with native
contiguous [128,388] weight layout. Baseline JAX still differs. Artifact:
boltz-fp32-ffi-probe-MjA7mq/foldjax.json; binary/source:
native-fp32-ffi-fs5gCu. The binary SHA256 is
270cb6e044442f095defaf206bd53db9584c7bae28d3c4ee0b5e7e9218485683.

Build uses g++ C++17 shared/PIC/O2, installed JAX FFI and CUDA 13 headers/
libraries plus an existing external CCCL include directory. Upstream FFI
header warnings remain; portable build/packaging is not verified. No new
runtime package dependency or implicit compilation is introduced.

Verification: both BF16 and FP32 exported symbols inspected; 686 exited 0,
exact target/repeat comparisons and unchanged bindings inspected. Thirty-eight
focused CPU tests pass; affected Ruff and diff checks pass. Whole-model bridge
687 and old BF16 eight-shape regression 688 are queued separately; full-model
parity and backward GPU behavior are not yet established here.

Job 687 completes the dependency-preserving full-model FFI control using an
external snapshot, unchanged default compiler flags and real input/weights.
Actual atom q/c are numerically exact against native. Atom p still has
18,003 unequal entries, RMSE 9.213901293788114e-9 and maximum
1.430511474609375e-6. Full-model protein RMSDs are
[12.446419462824439, 0.14520901038042025, 0.07040286599135717,
5.968639579212108, 1.0554300636908416]; ligand
[7.378978120179993, 0.014232918862642451, 0.010063989732855874,
0.5620868375520909, 0.19674945625694423]. Both raw/public strict confidence
fail. The exact first boundary therefore remains insufficient, even without
substituting captured c. This is not justification to ship the prototype:
its whole-model result is worse than 670 and other compiled regions remain
uncontrolled. Artifact: boltz-full-fp32-ffi-sVvdP5/{result,comparison.json}.

Job 688 checks the retained BF16 FFI on the original eight synthetic shapes,
two calls each. All 32 input/weight/output archive members equal the former
539 result bitwise. Both archive hashes validate against their reports.
Artifact: native-fp32-ffi-fs5gCu/bf16-regression. This establishes the bounded
BF16 library regression, not an ESMFold2 model rerun.

Verification: 687 and 688 exited 0; actual q/c and artifact-bound full-model
metrics inspected, BF16 archive identities and 32 direct comparisons pass.
Thirty-eight focused CPU tests and affected Ruff/diff checks pass. Production
FoldJAX has not adopted this external FFI intervention. Full-model compiler
selection/remaining upstream boundaries, portable build, independent review,
performance and six-model release still require work.

### Common autotune-cache control (jobs 689–690)

Audit of persisted per-fusion records: 670 has 633 keys. Against 670, 684
shares 626 with 71 differing records, 685 shares 626 with 74 differing,
and 687 shares 630 with 95 differing. Differences include Triton block/
warp choices and BLOCK_LEVEL_EMITTER versus NATIVE_EMITTER. These records
demonstrate compiler-selection differences, not their coordinate contribution.

Prepared an unedited baseline-preferred union: all 633 records from 670 plus
four keys unique to 687. Both runs use this explicit external directory with
JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=none to prevent JAX replacing its path.
Separate fresh executable caches are retained. After both runs, all original
637 entries remain byte-identical; six additional NATIVE_EMITTER entries exist.
This is therefore a common-record replay, not a completely closed autotune
cache or a guarantee that every compiled choice is fixed.

Baseline 689 reproduces final pair/single exactly versus 670; single_inputs,
coordinates and confidence are not bitwise reproduced. Native-relative protein
RMSDs are [1.9397289564807312, 0.04425743335370852, 0.03526518095856763,
0.8665958341451531, 1.666748020397251]; ligand
[0.14863029448094156, 0.011993795238864381, 0.007718117833850052,
0.1476655021453175, 0.13093684060549438].

FFI 690 keeps actual c bitwise equal to native but protein RMSDs remain
[12.404003886708592, 0.147533099838703, 0.07035454306182173,
6.050200819118584, 1.0439577006815668]; ligand
[7.360865348475616, 0.01438726137915034, 0.00998386541185632,
0.5683638383740216, 0.1931046915074718]. Both runs fail raw/public strict
confidence. The previously observed large FFI drift survives this common-key
control; differing autotune records alone have not explained it away.

Artifacts: boltz-autotune-common-DlDb8d/{baseline,ffi,cache}, with
baseline-report.json and ffi-report.json. The capture harness now records
XLA_FLAGS and JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES in provenance through an
explicit whitelist, rather than treating code compiler_options as the whole
policy. No entire environment is captured. This additive record applies to
future snapshots; 689/690 retain their original harness and queue commands.

Verification: 689/690 exited 0; artifact-bound reports, direct baseline
arrays, actual FFI c, and all common cache bytes inspected. Thirty-one
capture/report CPU tests, affected Ruff and diff checks pass. No production
FFI adoption, complete compiler freeze, performance or release claim.

### Atom attention-bias normalization boundary (jobs 691–693)

With actual FFI q/c exact, 690 still has atom p RMSE 9.213901293788114e-9,
atom-attention output RMSE 0.002011550881366193, and token output RMSE
5.312693540423345e-6. Investigated the intervening atom_enc_proj_z
LayerNorm -> BF16 Linear using the exact native p as a dynamic input.

Job 691 ordinary normalization yields 135 unequal bias elements, RMSE
7.435286085872164e-5 and maximum 0.0625. Reusing native AMP normalization
reduces this to 7 unequal, RMSE 7.389667600707052e-6 and maximum 0.015625.
Job 692 confirms those results and native AMP normalization plus the BF16
FFI GEMM is bitwise exact, including repeat invocation. These are same-input
boundary results, not full-model admission. Artifacts:
boltz-atom-bias-pvdVR7/report.json and boltz-atom-bias-ffi-FTVz9a/report.json.

The actual input_embedder.py now uses existing amp_layer_norm only when the
following projection kernel is BF16; FP32 retains its existing route. No FFI
is installed into production. A route regression checks both dtype branches.
This is a work-in-progress fidelity change, not a passed end-to-end fix.

Full current-source 693 with the common-cache policy completes, but protein
RMSDs are [13.282219902190759, 2.21381099628801, 0.14945419584741754,
6.160049234500259, 2.066905084380415]; ligand
[8.130073667895353, 0.11084285633711079, 0.02673950220305799,
0.4991887777193455, 1.1633256408298982]. Raw/public strict confidence fail.
Thus better same-input bias arithmetic does not establish a better full-model
result. This change is not approved for release/push; remaining operand/GEMM
and downstream numerical differences still need closure. Artifact:
boltz-atom-norm-runtime-RH4eKO/{result,comparison.json}. Runtime source matches
the repository; the snapshot precedes a test-only line-wrapping lint fix.

Verification: 691–693 exited 0, bound reports and full-model metrics inspected.
Full Boltz CPU suite: 668 passed, 6 skipped in 120.42 s. The skipped upstream
checks are not admitted by this result. Focused dtype/norm tests pass, affected
Ruff and diff checks pass after fixing test line lengths. No performance,
independent release approval, or six-model completion claim.

### Atom-pair input arithmetic isolated (jobs 694–697)

Extended opt-in native input details to capture input/output pairs for the
three geometric projections, c-to-p query/key modules and p_mlp. Hooks return
None and leave native outputs unchanged. Job 694 capture validates; features,
sampler tape, preprocessing tape, atom encoder output and final input embedder
archives are hash-identical to the former detailed native capture 672.
Artifact: boltz-native-atom-pair-AtjDaf/native.

Job 695 replays all six modules with their actual native inputs, managed
FP32 weights and native ReLU order. Both ordinary JAX and the FP32 FFI are
bitwise exact for all six, including repeats. Thus these standalone modules
do not require FFI on this observed input. Artifact:
boltz-pair-probe-zVtCKz/report.json.

Instead, actual input to the distance projection differs. On captured native
displacements, ordinary 1/(1+sum(d*d)) differs in 109,061 entries, RMSE
7.835260312319759e-9, maximum 1.1920928955078125e-7. Barriers with sequential
(x*x+y*y)+z*z retain that error. Explicit div.rn.f32 reduces discrepancies
to 68,089 but is insufficient (696).

Job 697 separates the three possible three-term reduction trees and division.
Using (x*x+z*z)+y*y with separate rounded products/sums, then 1+sum and
explicit div.rn.f32, is bitwise exact against the captured native distance
input, including repeat. The same tree with ordinary division still differs
in 52,181 entries; the other tested trees remain nonexact. All twelve module
arms remain bitwise exact. This pins a matching observed arithmetic sequence,
not yet its full production integration or universal reduction policy.
Artifacts: boltz-atom-distance-EFUQnf/report.json (696),
boltz-distance-trees-Rl8ACz/report.json (697).

Verification: 694–697 exited 0; native capture/hash continuity, all module
and distance comparisons inspected. Twenty capture/division CPU tests pass,
affected Ruff and diff checks pass. The distance-tree control remains bench
only at this point; the previous production norm change remains unadmitted.

### Native distance arithmetic integrated, observed-profile only (job 698)

Added native_atom_geometry.inverse_squared_distance and routed the real atom
encoder distance input through it. Only CUDA FP32 shape (1,97,32,128,3),
without context parallelism, selects the observed 0/2 then 1 reduction and
existing explicit FP32 divide. Other shapes, dtypes and CPU/TPU retain the
ordinary formula. No external FFI or captured reference values are used.

Job 698 includes the actual production helper in the same-native-displacement
probe: its output and repeat are bitwise exact. This bridges the former
bench expression into the source function, not yet the whole-model outcome.
Artifact: boltz-geometry-runtime-7G6yQj/probe.json. The full same-source n=5
run is 699, using the common-cache policy and the current AMP norm change.

Verification: 698 exited 0, production target/repeat equality and unchanged
bindings inspected. Eleven geometry/division CPU tests pass; affected Ruff
and diff checks pass after import ordering correction. Whole Boltz CPU
regression and whole-model 699 are in progress at this point.

Job 699 completes. Protein RMSDs are [13.283296913188364,
2.214358433937676, 0.14943503966553376, 6.1619001784597085,
2.0672384715301138]; ligand [8.059160960484808, 0.11104887674093697,
0.026731040986273984, 0.49919037200179206, 1.163362583739519]. Both
strict confidence gates fail. First-projection q/c still have 98,474 unequal
entries (no production FFI); atom p has 3,830,468 unequal, RMSE
1.1450418363169239e-7 and maximum 2.1457672119140625e-6. Correcting the
distance subexpression alone therefore does not remove the earlier propagated
projection error or close the model. This remains an unadmitted work-in-progress
runtime change, not a claimed end-to-end fix.

Verification: 699 exited 0; artifact-bound comparison.json and actual atom
outputs inspected. Full Boltz CPU suite: 672 passed, 6 skipped in 120.75 s;
affected Ruff and diff checks pass. Skips and strict failures remain open.

### Joint atom-boundary actual-trunk bridge (job 700)

Added a development-only wrapper selecting the FP32 FFI first projection
only while input_embedder.atom_encoder executes, and BF16 FFI only for the
observed attention-bias projection. It uses the current production distance
and AMP normalization changes. Diffusion's atom encoder is not intercepted;
no captured values are substituted. Binary identity and wrapper trace counts
are recorded separately, explicitly not runtime invocation evidence.

Job 700 runs trunk-only with the common-cache policy. Actual atom encoder
q/c/p now all equal the native capture bitwise. This closes these observed
outputs in a real trunk graph with the diagnostic library boundary, not in
the default production graph and not for all profiles. Attention-bias output
is not separately observed here; its isolated same-native-input evidence
remains 692.

The subsequent atom attention output still differs: atom output has 46,875
unequal elements, RMSE 0.0019463563568876978 and maximum 0.1484375;
token output has 4,559 unequal, RMSE 5.277661608424275e-6 and maximum
0.0010468512773513794. Final input embedder has 4,486 unequal, RMSE
5.277712241524731e-6. Investigation now moves to the atom attention
transformer, without claiming that q/c/p closure closes the model.

Artifact: boltz-joint-atom-boundaries-OBq32F/trunk, including
ffi-intervention.json. Snapshot precedes a wrapper-only line-wrapping lint
fix; executed source identity is retained in its provenance.
Verification: 700 exited 0; capture hashes, intervention metadata and actual
q/c/p direct comparisons inspected. Twenty-one affected CPU tests, Ruff and
diff checks pass. No whole-model coordinates/confidence were evaluated in
this trunk-only run; no production FFI adoption or release approval.

### Actual attention-bias verification and sequential control (701–702)

The joint wrapper now optionally records the computed attention-bias tensor
with an ordered callback and binds its archive hash in ffi-intervention.json.
Job 701 confirms that actual bias is bitwise identical to native. This closes
the missing input-boundary evidence from 700, not the attention transformer.
Artifact: boltz-joint-bias-bridge-xLPqdT/trunk. Its snapshot precedes a
wrapper-only line-wrapping lint fix.

Source inspection finds native per-layer calls versus candidate batched
conditioning projections and scan. An external observed-shape control 702
computes each layer's conditioning terms separately and unrolls the three
input-atom layers, retaining the joint q/c/p/bias corrections. Actual q/c/p
and attention bias remain exact. Atom-attention output still differs at
46,887 elements, RMSE 0.0019465877813568352, maximum 0.1484375; token output
has 4,567 unequal, RMSE 5.2820930255106484e-6, maximum
0.0010468512773513794. This control does not resolve the observed attention
discrepancy, and is not adopted into production. It tests the combined
unbatched/unrolled path, not separate causal effects for vmap and scan.
Artifact: boltz-atom-sequential-PlMV6t/trunk.

Verification: 701/702 exited 0; actual input boundaries and native-relative
attention outputs inspected, 702 capture hashes validate. Seventeen affected
CPU tests and Ruff/diff checks pass. No full-model coordinate/confidence
result is claimed from these trunk-only controls. Next boundary is the first
atom attention layer's adaptive normalization, Q/K/V and attention arithmetic.

### First atom-attention leaf isolation (703–704)

Native capture 703 adds first-layer AdaLN and attention projection input/output
observers. Its features, diffusion tape, preprocessing tape, atom encoder and
final input embedder archives have identical SHA256 hashes to capture 694.
Thus these observed archives show no change from adding the observers.

Probe 704 uses each leaf's captured native input independently. All seven
ordinary JAX projections (AdaLN scale/bias and attention Q/K/V/gate/output)
are bitwise native-equal and repeat-equal. Ordinary AdaLN a_norm differs in
169,429 entries (RMSE 6.154454540076799e-8, max 1.9073486328125e-6);
s_norm differs in 159,779 (RMSE 6.339466647699648e-9,
max 2.384185791015625e-7). Existing native AMP normalization makes both
bitwise native-equal and repeat-equal on these inputs.

This identifies a local normalization arithmetic discrepancy, not its causal
contribution to final coordinate drift. Projection equality is conditional on
native inputs, not a claim about propagated candidate inputs. Next is a
controlled actual atom-attention graph normalization intervention, retaining
the already verified q/c/p/bias boundary controls. No production normalization
change or whole-model admission follows from this isolated probe alone.

Artifacts: boltz-native-atom-attention-GzYi0w/native and
boltz-attention-leaves-P2ONYc/report.json.
Verification: queue jobs 703/704 exited 0; all 11 report-bound source, weight
and capture hashes rechecked, plus the five native archive continuity hashes
above. All 11 probe arms repeat bitwise. Whole-model structure/confidence and
other layers/profiles remain unverified by this experiment.

### Actual input atom-stack native AdaLN control (705)

The diagnostic joint wrapper gains an opt-in native-atom-adaln flag. It scopes
native AMP normalization to input atom attention and transition AdaLN, leaving
diffusion atom and token stacks outside the intervention. Default production
behavior is unchanged. Actual q/c/p and attention bias remain bitwise native
equal. Atom attention output now has 9,491 unequal entries, RMSE
0.0007368603590254403, maximum 0.125. Token output has 1,888 unequal,
RMSE 1.2363030499643304e-6, maximum 0.00018817931413650513.
Both improve against 701 but remain nonzero; these are representation units,
not coordinate angstroms. This supports a contribution from normalization
arithmetic, not a sole-cause explanation or whole-model acceptance. Common
autotune cache remains update-mode, not a fully frozen compiler control.

Artifact: boltz-adaln-bridge-YV3Bde/trunk. Verification: 705 exited 0;
verify_capture checked 19 candidate and 98 native artifact hashes, and actual
boundary arrays were compared. Intervention metadata records nonzero trace
counts for both norm functions (not runtime counts). Wrapper syntax/Ruff and
diff checks pass. Full n=5 coordinates/confidence have not been measured for
this intervention; production adoption and release review remain open.

Full-model follow-up 706/707 uses the same immutable 705 source snapshot,
native capture, weights and diagnostic projection/bias controls. The only
requested arm option difference is native-atom-adaln (off/on). Outputs are
boltz-adaln-bridge-YV3Bde/full-control and full-native. Both retain n=5 native
sampling settings and fixed capture tape; the shared autotune cache is still
update-mode, so this does not assert identical compiled kernels. At submission,
706 is running and 707 queued; no final coordinate/confidence claim yet.
Twenty capture/FFI CPU regression tests pass. These are diagnostic full runs,
not uninstrumented performance measurements or production admission.

706/707 completed successfully. Artifact-bound control-report.json gives
protein RMSDs [2.0405133554641357, 0.15417303983179234,
0.07603592256937772, 2.8994823997276296, 1.613851581262373] and ligand
[0.15409917390818031, 0.012233468540938479, 0.009880095420228932,
0.3975544449102637, 0.18622739307004132]. Native-report.json (AdaLN on)
gives protein [5.947692443690196, 0.2036705131860199,
0.06596475125682741, 4.696780807686491, 0.37297460025086426] and ligand
[1.2205356080943173, 0.018336496873148728, 0.010615700622454692,
1.0156850017742582, 0.052876251897724165]. Both raw and public strict
confidence gates fail in both arms. Local attention representation improvement
therefore does not establish final structure improvement; the worst entity
RMSDs increase in this comparison. Do not promote this diagnostic AdaLN option
to a default based on 704/705. Remaining attention/trunk propagation and
compiler choices still need separation; this does not refute the observed
same-native-input normalization equality.

Verification: 706/707 exit 0, both artifact-verifying report builds exit 0,
per-sample entity RMSD and raw/public strict results inspected. Report success
is not scientific acceptance. No production default change, commit or push.

### Combined K/V projection hypothesis (708)

Source inspection finds separate native K/V Linear calls versus candidate
concatenated kernels and one Linear. Probe 704 only checked separate leaves.
Extended the probe with the actual concatenate/linear expression using the
same native K/V input (explicit equality guard), and compared its concatenated
output with both native outputs. Job 708 is bitwise native-equal and repeat
equal. Thus this isolated first-layer profile does not support blaming K/V
combination for the residual. This does not establish whole-graph fusion
equivalence. No runtime rewrite is justified by this result; subsequent
attention score/softmax/value arithmetic still requires direct probing.

Artifact: boltz-kv-probe-eP9NIQ/report.json. Verification: 708 exits 0,
bindings remain unchanged, combined output/repeat comparison inspected;
affected Ruff passes. Full-model admission remains open.

### Attention mask semantic audit

Pinned attentionv2 adds (1-mask)*-inf (finite configured penalty) before
softmax. Candidate _no_proj_qblock instead calls masked_softmax, using
negative infinity for excluded keys and exact zero for empty rows; its caller
does not forward the inf parameter. These differ for empty rows, nonbinary
masks and potentially extreme logits. The zero-empty-row behavior also serves
the package padding contract, so do not remove it without scoped evidence.

A CPU check of the actual 703 atom_pad_mask through production
get_indexing_matrix/single_to_keys (32 query, 128 key atoms) yields
shape (1,97,128), zero empty windows, zero nonbinary keys, and 49–128 valid
keys per window. Thus empty-window handling is not supported as the cause for
this profile. Extreme masked logits and FP32 attention arithmetic remain
unmeasured; this is not a universal mask-equivalence claim.

Verification: production key-window reconstruction completed successfully on
CPU from the captured feature archive. No runtime change made by this audit.

### Captured QKV core replay (709/710)

Added a bench-only explicit native/JAX core replay using first-layer captured
Q/K/V outputs, first-layer bias and reconstructed binary key mask. CPU check
confirms reconstruction equals production key-window mapping. Native replay
uses the pinned Torch environment and highest FP32 matmul policy; JAX uses
highest and disables excess precision. Both expose all four stages, so this
is not the unobserved production graph or an independent native core capture.

QK score and masked logits are bitwise equal. Softmax probability differs in
871,785 entries (RMSE 4.167869007244573e-9, max 3.5762786865234375e-7).
FP32 value output differs in 214,667 entries (RMSE 6.963034273793572e-8,
max 2.384185791015625e-6). Thus the first observed difference in this replay
is softmax, not the QK contraction. This is a diagnostic direction, not proof
that softmax explains final coordinate drift; actual production masked-softmax
and post-BF16 output/remaining layers need their own bridge checks.

Artifacts: boltz-core-replay-tcHixF/native and jax. Verification: 709/710
exit 0, both output archive hashes and all bound operand/source hashes
rechecked, four stage arrays compared. No runtime adoption or model admission.

Follow-up 711 separates value contraction from probability generation. Given
the native replay probability and identical V, JAX P@V is bitwise native
equal, including BF16 narrowing. Ordinary replay produces 12 BF16-unequal
output values. The actual production _no_proj_qblock on captured operands
produces 11 BF16-unequal values (FP32 RMSE 7.061055697025968e-8,
max 2.384185791015625e-6). This strengthens the softmax explanation for this
isolated replay, without establishing its full attention/model contribution.
Artifact: boltz-core-controls-1ZSQS1/jax. A captured native probability is
used only for this diagnostic; it is not substituted into production.

712 reuses the existing warp_softmax without a production edit: pad the 128
native logits to its supported 437 columns with negative infinity, then crop
the returned probabilities to 128. The extra exponentials are zero. The
result is bitwise identical to native replay probability, unlike ordinary
JAX softmax. This is evidence that existing warp reduction/division machinery
can match this observed input, not a performance recommendation for padding
or proof of fullgraph parity. A direct 128-column implementation and actual
attention bridge still require tests before runtime adoption.
Artifact: boltz-softmax-reuse-gOJEaB/jax. Verification: 712 exits 0;
probability arrays compared bitwise, affected Ruff passes.

713 directly supports 128 columns in the existing runtime warp_softmax helper
(four per-lane iterations; no 437-column extension). Prediction dispatch is
unchanged. Actual native-logit probability is bitwise equal; input/source and
output archive hashes rechecked. Ten CPU tests include retained 437-column
arithmetic, direct-versus-extended 128 equality and invalid profile rejection.
Artifact: boltz-softmax-128-sgNMsv/jax; 713 exits 0, Ruff/diff pass.

714 is submitted as a trunk-only diagnostic bridge enabling input atom AdaLN
and warp softmax together. It keeps invalid keys excluded and empty rows zero,
and retains the joint projection controls. Output target is
boltz-softmax-bridge-fddLWa/trunk. No completed result is claimed at submission.

714 completes: actual atom encoder q/c/p and attention bias remain bitwise
native equal. Input atom attention atom output is now bitwise native equal,
as is the returned conditioning output. Token output retains 917 unequal
entries, RMSE 3.860261479781141e-9 and max 1.1920928955078125e-7.
This closes the observed atom output in a real trunk with joint diagnostic
FFI projections, native AdaLN and warp softmax; it is not default production
closure, universal profile acceptance or final coordinate/confidence parity.
Remaining input-token discrepancy belongs after this equal atom boundary.
Verification: 714 exits 0, 19 candidate artifact hashes verified and actual
q/c/p, attention outputs and bias compared with native captures.

### Native-order token pooling bridge (717)

Native encodersv2 normalizes the FP32 atom-to-token mapping before BMM;
candidate scatter mean sums values before division. A scoped diagnostic
reconstructs the mapping from ownership and uses normalized-mapping matmul
inside input atom attention only, retaining 714's controls. Atom and
conditioning outputs remain bitwise native-equal. Token output unequal entries
drop from 917 to 4; RMSE 1.7061810829995417e-10, maximum
5.960464477539063e-8. This supports the pooling-order explanation for most
of the observed remainder, not complete token or whole-model equality.

715 failed before compute due to a mistyped checkpoint basename. 716 also
failed before compute because 715 had created the output directory. Failed
artifacts were preserved; 717 uses the corrected existing checkpoint and a
fresh output target. Artifact: boltz-pooling-bridge-yokIeW/trunk-retry.
Verification: 717 exits 0, 19 candidate artifact hashes checked and all three
atom-attention outputs compared. No default prediction routing change or
full n=5 coordinate/confidence result from this trunk-only experiment.

718 runs the same snapshot through full n=5 prediction. Protein RMSDs are
[1.799081882081487, 0.10881142621765172, 0.03401713696630487,
2.579807902386572, 1.6188393057887762]; ligand
[0.1257629685752705, 0.012208865589540522, 0.010285656520844353,
0.3091936617241005, 0.15141709063615783]. Both strict confidence gates fail.
Despite observed input-atom closure, final output remains far outside the
gray zone. This joint diagnostic does not close downstream trunk propagation
or establish a production-ready default. Source snapshot precedes only the
wrapper scope-description correction (now explicitly mentions softmax/pooling).
Artifact: boltz-pooling-bridge-yokIeW/full-report.json.
Verification: 718 and artifact-verifying report build exit 0; per-entity
five-sample arrays and strict gates inspected. Thirty affected CPU tests,
Ruff and diff checks pass. No commit/push or full-model admission.

### Full 718 stored-boundary localization

Direct inspection of existing arrays (no new GPU run) shows final input
embedder has only two unequal values, max 5.960464477539063e-8 and RMSE
1.4998168993664361e-10; relative position is bitwise equal. At cycle 00,
MSA input_z is bitwise equal but delta_z RMSE is 0.03196139647126932,
max 2.005828857421875. Pairformer input_s is bitwise equal; input_z carries
that MSA difference. Pairformer output_s/output_z RMSE is
0.052570079948517805 / 0.07836336212152968. Cycle 03 MSA delta_z RMSE is
0.033680333345465725 and pairformer output_s/output_z RMSE is
0.051078694527475654 / 0.06969527807015935.

The next prioritized boundary is therefore actual fullgraph first-recycle MSA,
not the two residual input-embedder values. This does not establish equality
of every MSA operand: the current MSA capture stores input_z and delta_z,
not all single/MSA features or internal layers. Earlier isolated all-layer
MSA equality must not be substituted for this contradictory fullgraph result.
Verification: all six paired stored boundaries and matching named leaves
inspected; values are representation units, not coordinate angstroms.

719/720 expand both MSA call observers with input_emb and six consumed
features. First-recycle input_z is equal. All six feature shapes/values are
equal, including 4,436 MSA rows; msa and msa_mask use candidate int32 versus
native int64 storage, so literal dtype identity is not claimed. Other captured
feature dtypes agree. Both embeddings are FP32 and differ in only two values
(max 5.960464477539063e-8). Delta_z retains the same RMSE
0.03196139647126932 as 718. Thus the large MSA output discrepancy is not
explained by differing captured feature values or pair input; embedding
rounding, internal dtype/dispatch/compiler behavior remain candidates.

Native 719 features/tapes/forward-output/predict-step-output archive hashes
equal 703, despite expanded observation. verify_capture checked 98 native
and 19 candidate artifacts. Comparison initially assumed common metadata dtype
keys and FP32-only leaves; corrected inspection uses native_dtype versus dtype
and direct integer/bool equality. Twenty-five observer tests pass, now asserting
that the additional operands are saved. Artifacts: boltz-msa-inputs-UEQ7gm
native and trunk. No model admission or prediction change.

The two unequal first-cycle input_emb values become exactly equal after BF16
conversion. Source _common.linear casts its input to a low-precision kernel's
dtype before matmul; _msa_input_embedding uses this function for s_proj.
This down-ranks the two FP32 embedding values as a direct explanation under
the BF16 projection policy, but does not prove the compiled fullgraph obeys
every intended rounding boundary. Next compare the actual initial MSA m
and layer boundaries against the earlier isolated MSA controls, which expose
input_m and used a different compilation boundary/split-k setting.
Verification: direct captured-embedding FP32 comparison has two unequal values,
BF16 comparison has zero. This CPU conversion check is not a GPU replay.

721/722 are queued serially from boltz-msa-initial-JgPk2d. Native input-details
now captures first-layer input_m/input_z/token_mask/msa_mask at first and last
recycle. The candidate's optional capture-msa-embedding observes each actual
embedding output before the layer stack, records runtime callback counts and
binds its archives in diagnostic metadata. No reference input is substituted.
The new native hook test verifies input identities and unchanged return behavior;
15 native observer tests pass. GPU results and observer continuity are pending
at submission, not admitted. The source snapshot precedes this test-only addition.

721/722 completed. Native 102 artifact hashes verify; features, tape and both
final-output archives match 719 exactly. Candidate 19 standard artifact hashes
and four extra embedding archives verify, with four runtime embedding callbacks.
Actual initial m differs from native first-layer input_m in 6,239 values,
RMSE 0.00014411428600741148, max 0.03125, identically at cycle 00 and 03.
The first-cycle MSA delta_z is unchanged from the previous 720 capture despite
the added embedding observation. Thus initial MSA projection/addition is the
next boundary to isolate before attributing divergence to later MSA layers.
This is actual generated m, not a substituted native tensor; all reported
differences are representation units. No final parity admission follows.

Historical reconciliation: current native cycle-00 input_m is array-equal to
boltz-amp-closure-20260907/msa-native/layers/00/input_m.npz. The 6,239-value
pattern is the already diagnosed s_proj split-K difference from 471–479,
not a new sparse-MSA defect. Prior split-K=1 closed that projection; 669
subsequently closed isolated MSA under that control. Job 723 now bridges that
compiler condition into the current joint atom-corrected trunk using the
same 721/722 snapshot and captured initial-m observation. Global split-K
affects other contractions too; this is not an isolated s_proj intervention.
Output: boltz-msa-initial-JgPk2d/trunk-splitk1. Result pending at submission.

723 completes successfully. Initial m and first-cycle MSA input_z/delta_z are
now exactly equal to native. First Pairformer input_s/input_z and output_z
are also equal; output_s retains RMSE 7.77198491202209e-5 and maximum
0.0023193359375 (164,303 unequal values). This bridges isolated MSA equality
into the observed full trunk and moves the first stored residual discrepancy
to Pairformer single output. It does not close later recycles, final structure,
confidence or default production execution. Verification: 19 candidate artifact
hashes plus four embedding archive hashes checked, four callbacks confirmed,
and named first-cycle boundaries directly compared; 723 exits 0.

723 last-cycle inspection prevents overgeneralizing first-cycle equality:
MSA input_z RMSE 0.012503763272840413, delta_z 0.033567909529391135;
Pairformer input_s 0.004668946912621924, input_z 0.03945822041684134,
output_s 0.0496004289293792, output_z 0.06959410889572237. Later recycle
propagation remains open. Job 724 records the current split-K1 joint full n=5
outcome, disabling only the large embedding observer; other diagnostic controls
remain. Output full-splitk1 under the same snapshot. The observation difference
must be retained in interpretation. Final coordinates/confidence pending.

724 full n=5 protein RMSDs: [1.8496473050290787, 0.15338052833086727,
0.049714247910335026, 0.762332797972459, 0.09279731883105626]; ligand:
[0.13699522444165682, 0.0151345257209326, 0.011529266245916048,
0.10911786245830263, 0.018933773670560494]. Raw/public strict confidence
both fail. Artifact: boltz-msa-initial-JgPk2d/full-splitk1-report.json.

725 adds a diagnostic-only native AMP norm at trunk s_norm/z_norm, preserving
all joint atom controls and split-K1. First-cycle MSA and Pairformer pair output
remain exact. Crucially, last-cycle MSA delta_z and Pairformer output_z now
also equal native exactly. Last single output still differs: RMSE
0.008207142547947905, max 0.207122802734375 (167,773 unequal values).
First single output retains RMSE 7.77198491202209e-5. This supports recycle
normalization as a contributor to pair-path divergence, not complete single
or coordinate closure. Artifact: boltz-recycle-norm-Lb5JcC/trunk.
Verification: 724/725 exit 0, 724 artifact-verifying report succeeds; 725's
19 bound artifacts checked and first/last boundary arrays compared. No default
prediction change or release admission.

726 full n=5 uses the same 725 snapshot/controls. Protein RMSDs are
[0.8491425695560969, 0.02355712083291684, 0.0021623450982683062,
0.5480897209757035, 0.05675326741554442]; ligand
[0.06545560120132589, 0.0013973325541501872, 0.00021617124761204065,
0.0498250954228395, 0.003335451110907709]. Both raw/public strict confidence
gates fail. Pair-path closure reduces the worst deviation versus 724, but
protein samples 0 and 3 remain above 0.1 A. Single-path residuals therefore
remain priority, not gray-zone admission. Artifact:
boltz-recycle-norm-Lb5JcC/full-report.json. Verification: 726 and the
artifact-verifying report exit 0; all five entity values/strict gates inspected.
Diagnostic FFI/split-K policy remains, no production release claim.

727 single-prenorm diagnostic returns the same first/last outputs as 725.
Source audit explains why: amp_layer_norm supports widths 16/64/128/256,
whereas single width is 384, so it falls back to ordinary normalization.
This is a null intervention, not evidence against single normalization as a
cause. Likewise 725 changes effective pair normalization at width128 while
single recycle normalization at width384 remains ordinary. Correct the earlier
joint s_norm/z_norm interpretation accordingly. The wrapper now rejects an
unsupported single-prenorm width rather than silently presenting a fallback
as a native intervention. Artifact: boltz-single-prenorm-2NAAFD/trunk.
Verification: 727 exits 0, 19 artifact hashes verified and output arrays
compared; fallback confirmed directly in native_amp_norm.py. No production
single normalization correction has been implemented or admitted.

728 captures actual native first Pairformer attention.norm_s input/output
(mapped candidate pre_norm_s), at first and last recycle, under input-details.
Artifact target: boltz-single-norm-native-8Ofn74/native. Width384 remains
unsupported by the custom norm helper; no support-list expansion is made
without a valid reduction implementation. The new hook preserves input/output
objects and returns None in its regression test. Sixteen native observer tests,
Ruff and diff checks pass. GPU capture result pending at submission; source
snapshot precedes the additional test only.

728 failed before inference while installing hooks (exit 1): AttentionPairBias
has no norm_s attribute. The actual Boltz2 model imports PairformerModule from
layers/pairformer.py, whose layer owns pre_norm_s and calls it on s.float().
The earlier assumption based on modules/trunk.py was the wrong implementation
path. The observer now targets pairformer_module.layers.0.pre_norm_s in hook
dispatch, installation and call-count validation; its regression checks the
recorded path and input/output identity. No width384 numerical result exists
from 728, and its failed output is preserved. Native/candidate observer tests:
27 passed. No runtime model change or parity admission follows from this fix.

729 reruns the corrected observer from snapshot
boltz-single-norm-corrected-5t5ttF, output native, with the pinned publisher
runtime and unchanged 5SAK settings (seed101, n5, 200 steps, 3 recycles).
GPU availability was checked before serial tsp submission; queue inspection
confirmed 729 running. The actual layers/pairformer.py sequence stack disables
CUDA autocast and calls pre_norm_s(s.float()), followed by attention and
transition_s in FP32. Thus this investigation is FP32 normalization arithmetic,
not a proposal to lower the single path to BF16. Capture continuity and numerical
comparison remain pending; no native-norm implementation change is admitted.

729 subsequently produced the cycle-00 pre_norm_s archive, confirming the
corrected hook is reached, but remains running at this observation. Job 730
is queued behind it using snapshot boltz-single-norm-replay-aBM91T and new
bench.boltz_single_norm_probe. The replay verifies native capture hashes,
requires two finite FP32 width384 boundary captures, binds converted weights
and the actual _common source, and compares the production layer_norm with
native output plus a repeated same-JIT call. It passes eps=1e-5 explicitly.
This is a same-operand leaf diagnostic, not independent parameter identity or
full-graph fidelity evidence. CLI import/help, Ruff and diff checks pass;
numerical results and native output continuity remain pending.

729/730 completed with exit0. Native capture verification succeeds; features,
preprocessing-tape, forward-output and predict-step-output NPZ hashes match
721 exactly. Replay bindings also verify. Ordinary FP32 pre_norm_s versus
native has cycle00 RMSE 1.986206526097223e-8, max 9.5367431640625e-7,
90,901 unequal values; cycle03 RMSE 2.0212239642752826e-8, same max,
92,641 unequal values. Both same-JIT repeats are bitwise equal. These are
representation units, not angstroms. Report: boltz-single-norm-replay-aBM91T/
report.json. This establishes small deterministic normalization differences,
not that they cause the 0.849 A coordinate deviation.

Next discriminating target is actual single attention projections: pinned
layers/attentionv2.py runs proj_q, proj_k, proj_v, proj_g separately, while
FoldJAX primitives/attention.py concatenates FP32 Q/G and K/V kernels.
This is a verified source-level execution-shape difference, not yet a proven
numerical cause. Same-operand native projection capture/replay should precede
any production de-fusion or additional custom width384 normalization kernel.

Verification detail: 106 native bound files verified. Diffusion tape.npz hash
also matches 721; tape.json differs only in its out directory field, with all
other parsed fields equal. Do not describe the metadata file itself as identical.

731 adds actual first/last-cycle single attention proj_q/proj_k/proj_v/proj_g
input/output hooks alongside pre_norm_s. SINGLE_LEAVES is shared by dispatch,
installation and count validation, avoiding inconsistent hook name lists.
Parameterized observer tests preserve the input/output object identities for
all five leaves; combined native/candidate observer suite passes 31 tests,
Ruff and diff checks pass. Native n5 capture is submitted through the serial
tsp queue with unchanged pinned settings. It remains an observation-only run;
projection fusion has not been changed in production and results are pending.

732 queues bench.boltz_single_projection_probe behind native capture731.
The probe requires complete verified native artifacts and equal native QKVG
inputs, then compares FP32 separate projections with the current concatenated
Q/G and K/V algebra, using highest matmul precision and excess precision off.
Both arms include same-JIT repeats; captured operands, weights and actual
linear helper source are hash-bound. CLI import/help and Ruff pass. This tests
leaf execution shapes, not the full attention graph or production closure.
GPU results remain pending; no fusion policy is changed based on source alone.

731/732 completed exit0. Native 122 bound files verify, and features, diffusion
tape, preprocessing tape, forward output and predict-step output NPZ hashes
match729. Replay bindings verify; all 16 same-JIT repeats are bitwise equal.
Separate and combined projections have identical reported RMSE/max/count for
each of Q/K/V/G at both cycles: cycle00 RMSE Q=2.2966047872104555e-7,
K=3.546266933833508e-7, V=7.318060342669954e-7,
G=8.693215416458894e-7. Cycle03 RMSE Q=2.3080281263731155e-7,
K=3.5525428236162936e-7, V=7.229158047983834e-7,
G=8.705733384292955e-7. Largest absolute error is 1.049041748046875e-5.
Report: boltz-single-projection-replay-Yalald/report.json. These are FP32
representation errors, not coordinate RMSD. De-fusion did not improve this
same-operand probe. Equal summary metrics do not prove the two arrays are
bitwise identical; that direct comparison was not saved. No production
de-fusion is justified by this evidence. FP32 GEMM arithmetic and subsequent
attention propagation remain distinct hypotheses for the full-model residual.

733 adds the existing development-only FP32 native GEMM FFI to the same-operand
projection replay and saves direct separated-versus-combined comparisons.
All eight native-GEMM projection outputs (Q/K/V/G, cycles00/03) now match native
bitwise, including Q after separate bias addition; same-JIT repeats also match.
All eight separated-versus-combined JAX comparisons are bitwise equal. This
isolates the measured projection residual to GEMM execution arithmetic under
these operands, not concatenation or checkpoint identity at these leaves.
It does not prove the downstream coordinate residual is fixed, nor does it
admit the external FFI into the JAX-only package. Report:
boltz-single-projection-ffi-zjg2IW/report.json. Verification: tsp733 exit0,
report bindings rehashed successfully, Ruff and CLI import/help pass.

734 bridges native FP32 projection GEMMs into the actual Pairformer trunk,
retaining725 joint atom/recycle-pair-norm and split-K1 conditions. New optional
--native-single-projections scopes the override to Pairformer attention calls
and FP32 384x768 Q/G or K/V kernels, splits them into two native GEMMs and
retains existing external Q bias addition. Pair-bias and output projections
are unchanged. A required nonzero trace counter prevents silent no-op admission.
The external FFI remains development-only. Six existing wrapper tests pass
(embedding callback validation only, not this GPU intervention); Ruff passes.
Actual trunk boundary results and intervention execution remain pending.

734 completed exit0; 19 bound artifacts verify. Required single_projections
trace count is2 (tracing, not runtime call count). First-cycle single output
RMSE is 7.412831237783762e-5, max0.002197265625, compared with725 RMSE
7.77198491202209e-5. Last-cycle single RMSE is 0.008912000342394352,
max0.202392578125, versus725 RMSE0.008207142547947905. First/last pair
outputs remain bitwise equal to native. Thus same-operand exact QKVG GEMMs
do not yield sustained single-path improvement in the actual trunk. Do not
promote this diagnostic override into production or claim coordinate closure.
Artifact: boltz-single-gemm-bridge-SOmJnw/trunk. No final n5 coordinate arm
was run for734; representation errors are not angstrom values.

Bounded triage conclusion: normalization and projection subprobes have isolated
small arithmetic differences, but the tested projection bridge does not close
single recurrence. Preserve Boltz as unresolved (>0.1 A on the last measured
full diagnostic), not passed or gray. Further custom-kernel work should be
deferred behind the next model's actionable fidelity gaps rather than treating
individual exact leaves as a reason for unlimited Boltz-only investigation.
