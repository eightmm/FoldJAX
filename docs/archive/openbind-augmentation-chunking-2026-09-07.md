# OpenBind sample chunking and augmentation identity

Status: reproduced sampler defect repaired; real-weight native parity and
performance remain unverified.

The previous implementation ran a separate diffusion scan for every sample
chunk. Initial and churn noise were drawn at full width and then sliced, but
the augmentation callback received each chunk's local width with the same key.
Consequently chunks reused augmentation transforms instead of retaining the
full-width sample-to-random-draw mapping.

The sampler now retains all sample coordinates in one scan and serially chunks
only the denoiser through `lax.map`. Augmentation still precedes churn and
denoising. The existing `(key, coordinates)` callback signature is unchanged;
it now receives the full sample axis regardless of denoiser chunk size. No new
public tape API, dtype default or checkpoint change is introduced.

Regression coverage includes augmentation enabled with native draws, explicit
noise tape and masked noise; chunk sizes 1, 2 and 3 for five samples; unequal
last chunks; full-width augmentation and bounded denoiser callback shapes; and
rejection of nonpositive chunk sizes. Samples are required to differ, avoiding
a vacuous identical-sample comparison.

Verification: the implementation worker's focused sampler/inference/compile/
padding CPU suite passed 73 tests with 8 optional-runtime skips. The parent
independently inspected the complete patch and ran sampler tests together with
Boltz normalization/probe tests: 54 passed. These are CPU unit/regression
checks, not real-weight entity RMSD or native confidence evidence.

Remaining work: native quaternion/translation runtime tape replay, real-weight
full versus chunked n=5 comparisons, and uninstrumented memory/runtime
measurements for the new serial-map graph. Historical chunk performance cannot
be attributed to this graph.

The parent separately reproduced NaNs for an all-masked sample in both eager
and JIT execution. Native augmentation clamps the mask-count denominator to
at least one; FoldJAX now applies the same clamp. Both failing regressions pass
after the change, with the valid sample row bitwise unchanged against its
same-shape control. This boundary repair is separate from the chunk RNG defect.

## Native augmentation tape runtime

An optional `AugmentationTape(quaternions, translations)` now reaches eager
prediction, compiled calls and already-lowered executables. Raw draws use
`[steps,samples,4]` and `[steps,samples,3]`; native batch-one layouts are accepted
without reordering. Draw values remain runtime arguments, while tape presence
selects a separate graph identity. Noise tape controls initial/churn separately.

The public concrete entry validates shape, dtype and values before launch.
Value-validated replay admits FP32/FP64 only, with finite nonzero quaternion
norms in a conservative normal-arithmetic range; low-precision and extreme
subnormal/overflow tapes fail explicitly. The native forward remains `q/norm(q)`
without an epsilon or clamp. Direct traced low-level callers must run the
documented concrete validation first; no device callbacks are inserted.

The worker's final affected CPU checks passed 154 tests after value-validation
hardening. A broader pre-hardening run passed 590 with 353 optional-runtime
skips. The parent independently ran 67 sampler/augmentation/compiled-tape tests
before the last hardening. Native pinned-source CPU augmentation with all 200
5SAK quaternion/translation batches and synthetic coordinates had max absolute
difference 1.1920928955078125e-6, RMSE 1.1118236642460033e-7. This is an operator
check, not GPU full-model validation. Native MSA draw mapping, complete tape
consumption, resolved configuration and real-weight parity remain separate work.

Parent verification after the final value-validation changes:
`JAX_PLATFORMS=cpu .venv-ci/bin/python -m pytest -q tests/models/openfold3 --tb=short`
passed **594 tests**, with **353 optional-runtime/data-dependent skips** and
25 warnings in 91.64 seconds. These skips are not native GPU parity evidence.
The new native tape adapter and Boltz report checks separately passed 33 tests.

The adapter validates the actual 609-draw capture (four MSA selection pairs,
one initial-noise draw, then 200 quaternion/translation/churn triples). It
preserves native arrays and records shared-feature core-only scope because
capture begins inside model forward, not independent preprocessing. The
resolved native full model/experiment configs were recovered from that run's
prediction artifacts; runtime tuned chunks must still be bound to their own
recorded execution rather than borrowed from another repeat.

### Native kernel policy remains a separate implementation gap

The repeat-a reference records Triton triangles enabled and cuEq disabled.
The current candidate rejects `triangle_kernel="triton"`; substituting `xla`
or `cueq` is not evidence for that native policy. Actual checkpoint mapping
covered all 4,890 tensors, and the repeat-a adapter separately binds its own
609 draws, full configuration and tuning trace. Adapter tests now pass 15 cases.

Source inspection of pinned native `triangular_multiplicative_update.py`
identifies a substantive multiplication-only normalization difference:
`var = E[x*x] - mean*mean`, followed by `1/sqrt(max(var, eps))`, rather than
the candidate's centered variance with additive epsilon. The native in-place
inference branch also fuses projection, gates and residual arithmetic. Native
triangle-attention input normalization is ordinary LayerNorm and must not be
changed along with multiplication. These findings require operator GPU
verification before attributing any full-model coordinate drift to them.

Read-only review of the current tape/chunk repairs found no actionable runtime
defect. It does not replace native full-network replay or GPU memory testing;
compiled tape tests use a sampler-backed network mock. Whole-project CPU
coverage verification is in progress, not yet a recorded pass.

An isolated `models/openfold3/models/native_triton_norm.py` Pallas candidate
now preserves the native second-moment formula, variance floor, FP32 arithmetic
and eight-row program grouping. It is deliberately not selected by inference.
The parent ran 47 focused CPU interpretation tests successfully, including
width/block/row tails, variance near epsilon, dtype islands and cancellation.
These tests compare formulas and contracts, not native CUDA arithmetic: Pallas
sqrt lowering differs from the publisher's fast sqrt and requires an actual
native kernel comparison before this path can be admitted.

Actual native/candidate GPU jobs 414/415 completed the 38-case synthetic panel
at FP32. Native Torch 2.12.1+cu130 / Triton 3.7.1 repeats are bitwise identical
in every case. Candidate outputs are bitwise identical in 24 cases and pass
the separately recorded `atol=rtol=1e-4` diagnostic in 32. All six large-mean,
small-variance cases fail, with maximum absolute error up to 11.801747376099229;
normal, near-epsilon, constant and synthetic 437-by-437 pair cases pass strict.
This cancellation-sensitive gap prevents admission. Native PTX/TTIR/TTGIR and
candidate HLO are saved with their source-bound manifests under the private
`openbind-norm-20260907-X40pYH` task root. These are not real trunk tensors or
coordinate RMSDs, and the new operator runtime is distinct from older captures.

### Parent integration verification

The complete CPU coverage command
`JAX_PLATFORMS=cpu .venv-ci/bin/python -m pytest -q --cov=foldjax --cov-report=term-missing --cov-fail-under=80`
finished with **4,430 passed, 400 skipped, 67 warnings**, orchestration coverage
**87.44%**, in 630.67 seconds. Collection preceded the standalone norm and
latest probe additions; these have separate 47-test norm and 38-test probe
passes. No skipped optional native runtime test is counted as GPU evidence.

A fresh wheel built and installed offline in a separate frozen-lock environment.
Its 438 Python files passed a forbidden-import AST scan; all six backend
capability routes loaded with forbidden tensor-runtime imports hard-blocked,
and the installed CLI `models --json` completed outside the checkout.
Wheel SHA256: `00d110ee9eaff7d78e6c1f29659db19861f4f1044fa4b7b68aeb53d34b951eed`.
This wheel includes the isolated norm candidate, not a newly admitted backend.
Repository Ruff and diff checks pass. Full-model replay, uninstrumented
performance, final release review and main push remain outstanding.

### FMA correction, job 417

Saved native PTX and same-input counterfactual arithmetic identify two missing
FMA boundaries: the first sum-of-squares butterfly and subtraction of squared
mean from the second moment. The GPU width64/128 candidate now explicitly
preserves these and the observed warp reduction tree; sqrt and final affine
were left unchanged for the control. CPU interpretation remains formula-only.

On the same 38 native captured cases, the corrected GPU candidate passes all
38 strict diagnostics, with 24 bitwise matches and maximum absolute error
9.5367431640625e-7. All six earlier cancellation failures are removed without
changing inputs or tolerances. The parent separately ran 54 focused CPU tests.
The source-bound candidate report is under private task
`openbind-fma-20260907-WM6EuH`. This admits only the measured synthetic operator
panel, not other widths, full triangle multiplication, model output or runtime
performance. Production backend selection is unchanged.

### Resume invalidation

Parent regression tests reproduced that changing OpenBind inference,
augmentation or sampler source did not invalidate a completed prediction under
`resume=True`. ESMFold2's model/dropout source had the same gap. These four
source dependencies are now part of resume identity, following the existing
Boltz repair policy. Tests first failed all four changed-source cases and now
the complete resume suite passes **132 tests**; unchanged-source reuse remains
covered. This prevents stale pre-repair outputs from being presented as new
predictions. It is a targeted dependency fix, not a claim that every source file
of every backend is fingerprinted. Ruff passes for both affected files.

### Independent scoped review follow-up

An OMS Claude-family read-only review found no reproducible blockers in the
six reviewed Boltz/OpenBind production files; this was not a whole-tree or
model-release approval. Its requested missing coverage is now added: real
`predict` reaches the real sampler with stubbed network boundaries, testing
ordinary callback versus tape selection and expanded mask forwarding. A separate
eager/JIT check observes initial/churn draws themselves and confirms that the
tape route leaves their bytes unchanged. All four parent tests pass.

The review's centroid-clamp question was resolved against pinned native
`core/model/structure/augmentation.py:69`, which explicitly clamps mask count
to one. A source comment records that authority; quaternion normalization has
its own unclamped native contract. The sampler's existing rank-2 mask contract
is explicitly documented and was not broadened by this follow-up. On-device
width64 norm evidence exists in jobs 397/403 outside the review's restricted
scope; performance remains unmeasured. New ESMFold, standalone kernel and
manifest changes were outside this review and are not covered by its verdict.

### Native linear operator screen, jobs 420/421

The standalone candidate remains disconnected from production. On 52 fixed
synthetic cases (width64/128, row tails, native plain and fused wrappers, one
full 437-by-437 pair shape), both native and candidate repeats are bitwise
stable. Native-versus-candidate passes only **2/52** strict diagnostics, both
clamp controls. The width128, nine-row plain-with-bias maximum difference is
0.0029070377349853516. Failed cases are retained in source-bound manifests
under private task `openbind-linear-20260907-x6G8ZJ`.

Saved-array CPU counterfactuals point to different TF32 input rounding, rather
than just a multi-tile accumulator issue: width64 single-tile controls show the
same discrepancy. A GPU same-panel RTZ operand control is pending. This is a
causal hypothesis supported by operand calculations, not a backend admission.
Parent immutable-snapshot harness tests passed **35 tests** before GPU capture.

Job 425 then applied only host-side RTZ truncation of candidate dot operands
before the unchanged JAX kernel. All **52/52 cases became bitwise identical**
to the same native archives, including every fused epilogue. The explicit
counterfactual report records effective operand hashes in private task
`openbind-linear-rtz-20260907-v1`; the original 2/52 failure is preserved.
This isolates TF32 operand rounding as the cause on this finite panel. Moving
the correction into the actual kernel and rerunning with unmodified archived
inputs remains necessary; the host control is not the implementation fix.

Job 428 completes that implementation check: FP32 operand RTZ is now inside
the actual standalone Pallas kernel, immediately before each dot. With the
original native input archives and host control disabled, **52/52 cases match
native bitwise again**. BF16/FP16 and CPU interpretation are unchanged. Parent
runtime/probe tests passed **139 tests**. Runtime SHA256 is
`3cbb2d1cb7ec410f7427d3882fe6d5a0af094e7258f386ef8fd065314b8fb606`;
the private source-bound report is `openbind-linear-fixed-20260907-Vfif18`.
This closes the declared synthetic FP32 linear operator panel only. Native
triangle attention, real triangle-multiplication integration, model replay and
performance remain open; no production backend switch has been made.

### Standalone native triangle attention, jobs 434/437

Actual native-wrapper captures and the candidate private carried-dot lowering
pass **17/17 finite cases** at the existing strict leaf tolerance (two bitwise).
Four additional true-infinity mask semantic cases preserve the native NaN
positions bitwise. At the full synthetic repeated-row shape
`[1,437,437,4,32]`, maximum absolute error is
`2.2351741790771484e-8` and RMSE is approximately `1.12318e-9`.
Parent attention/norm/linear CPU checks passed 185 tests.

The candidate source SHA256 is
`cc764c1c06728d33ee19107f50abd8e53712fff1b2d9f96cdc67cd72e3dc64be`.
It retains the live accumulator as the third operand of the private Triton
dot lowering, without a global compiler patch. Native N<=16 uses stock
attention and is outside this standalone Triton-core route.

This closes only the declared core operator panel. Actual triangle attention
uses ordinary Torch LayerNorm/Linear projections, not the custom triangle-
multiplication norm/linear kernels. Full real-weight module integration and
model replay remain open, with no public backend/default change. Chunk policy
must also stay distinct: multiplication uses 256; the captured 5SAK attention
tuning uses 1024, thus all 437 rows in one chunk.

### Real-weight full triangle module panel, jobs 448/449

The private full-module implementation preserves native multiplication chunk
boundaries and exposes update-only and fused-residual APIs separately. Its
attention path retains ordinary normalization/projections, plus the native
stock core for N<=16. No public backend is wired to it.

All 16 selected checkpoint module subsets strict-load. On the finite 52-case
panel, native repeats are 52/52 bitwise equal with zero execution errors.
Candidate versus native passes only **1/52** strict diagnostics. Maximum
absolute output errors by operator are: outgoing multiplication
0.053714752197265625; incoming multiplication 0.04686546325683594;
starting attention 0.61297607421875; ending attention 0.651421070098877.
These are operator features, not coordinates. The failing cases remain
recorded in private `runtime-operators-20260908-suRDO0`.

Parent combined new runtime/probe tests passed 166 checks. The verified private
wrapper SHA256 is
`44c84553dbaa32d62f59b76e8e0cabe2697c9961f5d44e7cf4084b4f95e019b6`.
Stock N16 attention also fails, so the newly implemented Triton attention
core cannot alone explain this discrepancy. Candidate HLO combines several
ordinary projections into larger GEMMs; native performs separate projections.
This is a hypothesis to test with identical captured normalized operands,
not yet proof of the cause or grounds to change tolerance/backend defaults.
