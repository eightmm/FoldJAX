# Boltz-2 native AMP repair — 2026-09-07

Status: **in progress, not admitted**. This record concerns the pinned Boltz-2
2.2.1 source `b1ebfc46ecf57f5414e0d1a6f9027bbb122c53bc`, not every upstream release
or every input modality. No crystal comparison is used.

Latest conditioning-specific evidence:
[native CUDA Welford/FMA repair](boltz-conditioning-fma-2026-09-07.md).
Its native-trunk controls are not full-model admission results.

## Fixed comparison contract

- Native BF16 AMP, kernels enabled, FP32 matmul precision `highest`.
- Five sample-index-paired outputs, 200 diffusion steps, 3 recycles (4 passes),
  seed 101; full MSA, no subsampling.
- Actual preprocessing and sampler draws are captured. FoldJAX core diagnostics
  consume captured native features and the actual noise/augmentation/sigma tape.
  This is **not independent preprocessing proof** or a device-consumer tape audit.
- One proper, unweighted whole-system Kabsch fit per sample; RMSD is then measured
  per entity/chain instance without another fit. No confidence-rank rematching.
- The 0.05 Å structural diagnostic and historical confidence
  `atol=rtol=1e-4` remain separate. No threshold was relaxed.
- Source snapshots are immutable and GPU work is serialized with `oms tsp-queue`.
  Hook/callback runs are instrumented diagnostics, not performance benchmarks.

## Confirmed implementation differences and repairs

| Area | Native policy now represented in FoldJAX |
| --- | --- |
| Parameter/features | Only AMP Linear kernels are narrowed; raw features, norm affine, embeddings and original biases survive. Bias casts happen at Linear calls. |
| FP32 islands | AtomEncoder geometry/projections, atom-to-token projection/pooling, template feature projection, and Pairformer single branches retain original weights. |
| Linear/norm/activation | Ordinary LayerNorm retains FP32 affine/output; AMP Linear bias has one output rounding; low-precision SiLU/sigmoid avoid extra intermediate rounding. |
| Recycling | Pair residual remains FP32; single initialization/recycle sum is BF16 before the FP32 Pairformer single branch. |
| Atom attention | QK, softmax and probability–value contraction remain FP32, with narrowing only at the completed attention output. |
| MSA | Sparse input projection rounds once. Above 384 physical tokens, native head-wise PWA and hidden-4 OPM accumulation, external FP32 OPM bias, and hidden-32 MSA transition are reproduced. |
| cuEq multiplication | FP32 input normalization precedes BF16 gated GEMMs/contraction; fused output normalization retains its native dtype boundary. |
| Heads | Native AMP confidence/distogram/B-factor projection boundaries and FP32 confidence aggregation are represented; raw coordinates remain FP32. |
| Compilation | Low-precision outer JIT must use `compiler_options={"xla_allow_excess_precision": False}` to preserve deliberate narrow/widen rounding; no global environment flag is changed. |
| Scalar-output AMP dot | A one-column projection explicitly requests an FP32 accumulator before BF16 output; other unbiased matrix projections and the FP32 path are unchanged. |
| MSA residual | Native eval dropout's FP32 identity mask promotes PWA updates before the MSA residual addition. The scan carries FP32 residuals with BF16 projections. |
| OPM tiling | The model preserves native full-token AMP GEMM shape when its product fits the existing 256 MiB product budget; larger products retain bounded token chunks. |

The AMP operator policy was also measured on the actual GPU in job 309:
LayerNorm with FP32 affine outputs FP32, Linear outputs BF16, and adding a BF16
update to an FP32 residual outputs FP32.

## Verified results so far

Native captures completed for protein–ligand 5SAK, protein–RNA 1URN, and
RNA–ligand 3GCA. The second 5SAK native capture has identical feature bytes,
838 preprocessing draws, sampler tape, and all five coordinate arrays.
All 21 raw and 16 public confidence leaves pass the unchanged strict diagnostic.
Kabsch residuals of these identical coordinates are numerical roundoff
(protein `7.02e-14 Å`, ligand `4.73e-14 Å`). This is a two-run repeatability
control, **not** a calibrated native-variability allowance.

The v1 repaired FoldJAX trunk completed without native tensor substitution:

| 5SAK boundary | RMSE | Relative RMSE |
| --- | ---: | ---: |
| Input embedding | 0.00011733 | 0.0004890 |
| Relative-position encoding | 0.00303289 | 0.0018924 |
| Final single representation | 0.05702525 | 0.0009466 |
| Final pair representation | 0.07196543 | 0.0028969 |

These tensor errors are not coordinate RMSDs. Remaining relative-position
rounding differences were isolated below. The v1 trunk panel has no real
templates; subsequent template/optional-XLA-GLU repairs are outside that snapshot.

### Confirmed GPU compilation cause

On the exact 5SAK encoder features and original matching checkpoint weights,
host BF16 cast followed by the sparse projection matches native **exactly**.
Moving that same cast inside JIT reproduces the v1 full-trunk encoder output
bit-for-bit: RMSE `0.0030328943`, maximum `0.0625`, 8,132,335 unequal values.
Native BF16 reduction enabled/disabled, dense FP32 accumulation of BF16 operands,
and exact FP64 summation followed by BF16 rounding all agree with native.

Three independent JAX controls restore exact native output: an optimization
barrier after the cast, explicit `reduce_precision(8, 7)` before the cast, or the
per-compilation `xla_allow_excess_precision=False` option. The latter preserves
all deliberate conversion boundaries, including biases and activations, without
adding barriers throughout the model or changing other models' global options.
This is not a change from BF16 to FP32 inference.

The installed JAXlib revision pins XLA
`dcf304bc5dca1932b99f740b911dbd73631a1a69`. Its GPU pipeline invokes
[`SimplifyFPConversions`](https://github.com/openxla/xla/blob/dcf304bc5dca1932b99f740b911dbd73631a1a69/xla/hlo/transforms/simplifiers/simplify_fp_conversions.cc)
under the excess-precision option; the pass removes consecutive floating-point
conversion chains, including FP32 → BF16 → FP32. The GPU counterfactual above
establishes the observed encoder effect; it does not by itself quantify final
structure/confidence improvement. CPU-only tests did not reproduce this pass.

The production high-level primary/affinity JIT factories now apply the policy
for BF16 and record it in cache identities and returned execution metadata.
FP32 factories remain unchanged. Eager active steering has no protected outer
JIT and is explicitly outside this policy; low-level graph composers must pass
`foldjax.models.boltz2.compile_policy.compiler_options(dtype)` to their outer JIT.

Portable artifacts:

- `bench/experiments/boltz-native-repeat-2026-09-07.json`
- `bench/experiments/boltz-amp-trunk-v1-2026-09-07.json`
- `bench/experiments/boltz-amp-rounding-2026-09-07.json`
- `bench/experiments/boltz-amp-5sak-v4-2026-09-07.json`
- `bench/experiments/boltz-amp-1urn-v4-2026-09-07.json`
- `bench/experiments/boltz-amp-3gca-v4-2026-09-07.json`

### Full v4 matched-tape result

All three full captures completed with five samples and all native raw confidence
fields. Values below are the worst sample for each entity after a single global
fit; no entity is fitted separately.

| Input | Entity 1 maximum RMSD (Å) | Entity 2 maximum RMSD (Å) | 0.05 Å structural diagnostic |
| --- | ---: | ---: | --- |
| 5SAK protein–ligand | Protein 15.977573 | Ligand 4.647920 | Fail |
| 1URN protein–RNA | Protein 0.045146 | RNA 0.309854 | Fail (RNA sample 1) |
| 3GCA RNA–ligand | RNA 0.009351 | Ligand 0.007716 | Pass |

The exact-cast policy closes the observed relative-position discrepancy, but
**does not close Boltz structural parity**. For 5SAK, input-embedding RMSE drops
to `5.8126e-6` and relative-position RMSE is zero; first-cycle MSA delta RMSE is
still `0.0369497`, and final single/pair RMSE is `0.0618775` / `0.0910037`.
Matched native intermediate-input probes are needed to separate remaining MSA
operator error from upstream-input error and downstream amplification.

All three cases fail the unchanged strict confidence diagnostic. Token pLDDT
maximum absolute differences (native 0–1 scale) are `0.1293703`, `0.0137460`,
and `0.0017723` respectively. The core output has no publisher predict-step
`confidence_score` field; the report explicitly marks it missing instead of
inventing a public-output match. These are not confidence admission results.

### Subsequent sigmoid repair (v5)

An intermediate CPU gate exposed an outdated XLA-triangle test reference after
the GLU activation repair. Reviewing every remaining triangle gate then found
that the post-cuEq attention sigmoid still evaluated exp/add/div in BF16.
Native sigmoid instead has one FP32 operator and one BF16 result rounding.
The shared sigmoid helper now also covers triangle output and CP gate calls;
a 10,001-value independent NumPy oracle and isolated post-cuEq gate tests cover
eager/JIT behavior. The old versus corrected sigmoid differs at 3,378 values,
maximum `0.00390625`, on that finite BF16 input grid. This is an operator fix,
not a tolerance change.

5SAK full n=5 v5 reduced maximum protein RMSD from `15.977573` to `2.843108 Å`
and ligand RMSD from `4.647920` to `0.364296 Å`, but **still fails**. All raw
confidence fields remain recorded and the strict confidence diagnostic still
fails. The other two v5 regression captures also completed:

| Input | Entity 1 maximum RMSD (Å) | Entity 2 maximum RMSD (Å) | Structure |
| --- | ---: | ---: | --- |
| 1URN protein–RNA | Protein 0.039153 | RNA 0.310312 | Fail |
| 3GCA RNA–ligand | RNA 0.037002 | Ligand 0.034609 | Pass |

Both still fail the unchanged strict confidence diagnostic. Improvement on
5SAK is not monotonic improvement on every sample or modality.
Artifact: `bench/experiments/boltz-amp-5sak-v5-2026-09-07.json`.
The corresponding `boltz-amp-1urn-v5-2026-09-07.json` and
`boltz-amp-3gca-v5-2026-09-07.json` retain all sample-index results.

### Same-input localization controls

Native standalone MSA reproduces the original full-run first-cycle output
bitwise (job 332). FoldJAX receives exactly these native operands and all 226
mapped original weights are exact. Its initial MSA embedding RMSE is
`0.0001441143`, first PWA output `0.0040508556`, first transition output
`0.0247845951`, and final fourth-layer pair output `0.0369443172` (job 336).
These are tensor errors, not coordinate RMSDs. Later stage differences also
contain propagation from earlier layers; they do not independently indict
each operator.

The sampler counterfactuals deliberately substitute native intermediate tensors:

| 5SAK substitution | Protein maximum RMSD (Å) | Ligand maximum RMSD (Å) |
| --- | ---: | ---: |
| Native trunk; FoldJAX conditioning | 0.253441783 | 0.016209641 |
| Native trunk and full native conditioning | 0.003540887 | 0.000355678 |

Both use all five original samples and 200 captured sampler steps. They execute
neither the FoldJAX trunk nor confidence heads. This localizes the large remaining
error to the trunk and conditioning for this capture; it is **not model parity
admission**, nor universal sampler correctness. Conditioning alone can push
protein sample 3 above 0.1 Å despite identical trunk inputs. All completion-bound
artifact hashes were rechecked before making the portable record
`bench/experiments/boltz-downstream-2026-09-07.json`.

### Additional confirmed causes (v6 / v8)

The PWA scalar-output projection has a distinct GPU lowering issue. With the
**native normalized operands** fixed, eight independent one-column BF16 dots
have RMSE `0.0152177130`, 392,096 unequal values against native. Explicit FP32
accumulation followed by one BF16 output cast reduces this to RMSE
`0.0001109388`, 25 unequal values. The FP64 sum/BF16-round oracle differs from
native at 22 values; exact arithmetic is not asserted to be bitwise identical to
native floating-point accumulation. The portable counterfactual is
`bench/experiments/boltz-gemv-2026-09-07.json`.

Native PWA decomposition was first required to reproduce its actual module
bitwise. Slicing it to 32 MSA rows is **not** numerically inert: native sliced
versus full output has RMSE `0.0019178133`. The repaired FoldJAX 32-row output
is closer to the original full native output (RMSE `0.0001055511`) than to the
sliced native replay; no general row-chunk equivalence is claimed.

In the full same-input MSA replay, v6 reduces first-PWA RMSE from `0.0040508556`
to `0.0003823899`, but first transition RMSE remains `0.0234729061`.
Full v6 5SAK still fails: protein max `2.055336 Å`, ligand max `0.631340 Å`.
The ligand result worsens versus v5, so this is not presented as monotonic
end-to-end improvement. Its full report is
`bench/experiments/boltz-amp-5sak-v6-2026-09-07.json`.

Inspection of the native stage dtype metadata then exposed the larger MSA
residual error. Native `get_dropout_mask` always returns FP32, including eval's
all-ones mask. Thus native layer 0 starts with BF16 MSA embeddings but produces
FP32 residuals, and layers 1–3 receive FP32. FoldJAX had skipped the identity
dropout multiplication and inadvertently kept these residuals BF16. v8 restores
promotion before the residual addition. Its scan starts with the exactly
BF16-rounded embedding stored as FP32 because carry dtypes must remain stable;
the first PWA already normalizes in FP32. No embedding precision is recovered
or invented. Five new regression cases failed before this fix and pass after it.

A separate matched-native-trunk conditioning control found that splitting the
two transition input projections instead of concatenating them gives bitwise
identical outputs on 5SAK. This source-level control does not prove that the
compiler emitted different kernels. The diagnostic alternative was not adopted
as a production change.

The v8 full panel remains unadmitted:

| Input | Entity 1 maximum RMSD (Å) | Entity 2 maximum RMSD (Å) |
| --- | ---: | ---: |
| 5SAK protein–ligand | Protein 3.030521 | Ligand 0.339955 |
| 1URN protein–RNA | Protein 0.044381 | RNA 0.302351 |
| 3GCA RNA–ligand | RNA 0.009785 | Ligand 0.011572 |

Only 3GCA passes the structural diagnostic; all three fail strict confidence.
Restoring the native residual policy reduces first-transition same-input MSA
RMSE from `0.0234729061` to `0.0041257524`, but does not close the model.
Full v8 reports are retained as `bench/experiments/boltz-amp-*-v8-2026-09-07.json`.

### OPM reduction and native shape (v14)

With the exact native first-transition input, production, a manually expanded
transition, and an optimization-barrier-before-partial-sum control are bitwise
identical. Their native-output RMSE is `0.0002676144`; the barrier hypothesis
does not explain that operator discrepancy on this capture. No barrier was
added to production.

The first OPM has a distinct shape/reduction effect. Its default native replay
first reproduces the original output bitwise. Disabling native BF16 reduced
precision reduction changes that native output by RMSE `0.0016583715` despite
unchanged BF16 input/output and FP32 `highest` setting. PyTorch exposes this
through its [BF16 cuBLAS reduction policy](https://github.com/pytorch/pytorch/blob/v2.12.0/aten/src/ATen/cuda/CUDABlas.cpp);
`highest` alone is not a complete mixed-precision execution specification.

| Matched native OPM input | Native-default output RMSE |
| --- | ---: |
| FoldJAX, 128-token chunks | 0.0015264868 |
| FoldJAX, native 437-token GEMM | 0.0001519368 |
| Native, BF16 reduction disabled | 0.0016583715 |

FoldJAX `highest` versus `default` is bitwise identical for this operator.
Disabling Triton, requesting cuBLAS instead of cuBLASLt, and disabling autotuning
did not improve it. The cuBLASLt-disable request produced identical optimized
HLO to the no-Triton arm, so it is **not evidence of a distinct cuBLAS backend**.
These experimental compiler flags were not adopted.

The selected v14 change preserves native token shape only when the complete
AMP product fits the existing 256 MiB product budget. The 5SAK product fits;
larger or batched products retain bounded chunks. The direct low-level operator
still honors its explicit chunk parameter unless native-shape policy is selected;
the model selects it. FP32 graph equivalence is tested and unchanged.
This is a bounded native-fidelity repair, not a guarantee for all sizes or a
measured total-memory/speed claim. Portable hash-verified operator controls are
in `bench/experiments/boltz-opm-and-conditioning-2026-09-07.json`.

### Latest full panel and handoff

All v14 captures completed; the reference remains the original native defaults,
not a precision-control arm chosen after seeing a failure.

| Input | Entity 1 maximum RMSD (Å) | Entity 2 maximum RMSD (Å) | Structure / strict confidence |
| --- | ---: | ---: | --- |
| 5SAK protein–ligand | Protein 2.862567 | Ligand 0.380979 | Fail / Fail |
| 1URN protein–RNA | Protein 0.042364 | RNA 0.311269 | Fail / Fail |
| 3GCA RNA–ligand | RNA 0.009712 | Ligand 0.011555 | Pass / Fail |

Each maximum covers five original sample indices after one whole-system fit.
No entity refit or best-sample selection is used. The v14 native-shape repair
improves the isolated OPM but does not close full 5SAK parity: first-cycle MSA
RMSE remains `0.0322924736`, and final single/pair RMSE is `0.0557326486` /
`0.0801112169`. The remaining complete-model discrepancy is not explained in
full by the repaired operators; neither implementation correctness for every
path nor a native-variability tolerance exemption is asserted.

The native-trunk/native-conditioning 1URN control further localizes the gap:
FoldJAX's five-sample sampler reaches maximum protein `0.0000785781 Å` and RNA
`0.0000710182 Å`. These substituted intermediates are diagnostic only and do
not replace either failed full-model comparison. Together with the analogous
5SAK control, they prioritize the trunk/conditioning boundaries over sampler
changes for follow-up.

All reports and completion-bound control artifacts were verified before the
portable summary `bench/experiments/boltz-amp-progress-2026-09-07.json` was
written. No baseline was overwritten, no upstream source was edited, and no
commit, push, global compiler setting, precision-default switch, or tolerance
relaxation was made.

## Execution and verification log

- 310: observer source-root preflight rejected the queue working directory;
  no native inference ran. Corrected in 311.
- 311/312/313/314: native 5SAK, repeat, 1URN and 3GCA captures completed.
- 315: JAX compiled, but host observation failed with `JAX_PLATFORMS=cuda`
  excluding a CPU callback device. Corrected to `cuda,cpu` in 316.
- 316: repaired v1 5SAK trunk capture completed.
- 317–319: full v2 traces exposed native-default contact-guidance entering an
  eager-only branch despite empty constraints; no final coordinate result.
  A statically proved zero-potential path is now compiled without disabling real
  constraints or modifying the requested native flags. Native CPU gradients are
  exactly zero; the empty-guidance affected suite passed 60 tests.
- 320–322: full v3 traces exposed dynamic chain-ID discovery for nested confidence
  dictionaries. A host-resolved static chain-ID tuple now preserves every output
  label and metric inside JIT/lax.map; 64 affected CPU tests passed.
- 323–327: matched relative-position operator probes completed; isolated the
  GPU conversion-chain issue and verified the per-JIT compiler control.
- 328–330: full v4 n=5/200/3 captures completed with exact-cast compiler policy;
  immutable captures and verified reports produced, results above.
- 331: full v5 5SAK capture completed after the triangle sigmoid repair.
- 332: native same-input MSA replay completed, original output bitwise exact.
- 334–335: native-trunk/conditioning substitution diagnostics completed.
  Substitution results are not model parity.
- Unstarted 333 was replaced by 336 using stored (uncompressed) NPZ to remove
  diagnostic serialization overhead; numerical inputs and equations unchanged.
- 336: FoldJAX same-input MSA replay completed with all 44 stage captures.
- 337–338: v5 1URN/3GCA full regression captures completed and reports verified.
- 339: PWA diagnostic reporting attempted NumPy conversion of a CUDA tensor;
  corrected to use the recorded host array. No inference result was admitted.
- 340: dependent PWA diagnostic could not read the incomplete 339 capture.
- 341–344: native PWA decomposition, FoldJAX stages, normalized-operand projection
  controls, and repaired projection replay completed.
- 345: full v6 5SAK completed; both entity maxima still fail.
- 346: native-input conditioning projection counterfactual completed; separate
  transition projections made no difference.
- 347: full same-input MSA v6 diagnostic completed.
- 348–350: v8 full three-case panel completed after native MSA residual promotion.
- 351: same-input MSA v8 completed with all stage observations.
- 352: transition diagnostic failed on a missing mapping-helper prefix; corrected
  and covered by two CPU mapping tests. No operator output was produced.
- 353: full native-input transition/barrier comparison completed, all arms identical.
- 354–355: native/FoldJAX OPM reduction controls completed.
- 356–357: OPM compiler-backend and native-shape controls completed; optimized
  HLO and all output hashes saved.
- 358–360: v14 native-shaped OPM full panel completed and reports verified.
- 361: native-trunk/native-conditioning 1URN sampler counterfactual completed.

Verification: focused Boltz default suite **368 passed, 6 skipped** during this
iteration; new capture/head/MSA/precision checks **76 passed**; lock check passed.
An intermediate full CPU suite passed **4,068 tests, 412 skipped** in 630.54 s;
this predates the final chain-ID/compiler changes and is not the final gate.
The next intermediate full suite had **32 failed, 4,156 passed, 412 skipped**.
Four triangle references have been corrected with the native operator contract;
the other 28 failures were in AF3 tests and did not reproduce in an isolated
fresh-store affected run (**80 passed**). No AF3 code was changed for this.

Verification: the post-sigmoid full CPU suite completed with **4,226 passed,
412 skipped**, 66 warnings in 456.65 s, using the validated AF3 build cache.
The earlier AF3 failure cause remains unconfirmed, but did not recur in this
full-order rerun. The Boltz affected suite passed **570 tests, 6 skipped**;
the stored/deflate probe writer tests passed separately (**19 tests**).
Ruff, `git diff --check`, and `uv lock --check` passed. CPU test success does
not override the failed structural/confidence diagnostics above.

Verification: the subsequent pre-MSA-residual full CPU run passed **4,236 tests,
412 skipped** (604.51 s); later source editing overlapped its remaining tests,
so it is not a frozen final residual-policy gate. The post-MSA
full CPU run passed **4,244 tests, 412 skipped**, 66 warnings (486.83 s). This
precedes native-shaped OPM selection, which has a separate affected-model gate.
The three OPM shape/budget/FP32-equivalence cases and affected MSA/predict checks
passed with **77 tests**. The diagnostic OPM CLI/profile tests passed separately.

Verification: after the final native-shaped OPM change, the complete affected
Boltz suite passed **597 tests, 6 skipped** in 177.69 s. Ruff,
`git diff --check`, and `uv lock --check` passed. An earlier mistyped
`uv run lock --check` did not execute the lock gate; the correct command was
then run successfully. The full cross-model suite was not repeated after this
last Boltz-only change; its prior 4,244-pass result is scoped above. Remote CI,
independent preprocessing, performance, affinity/real-guidance/template and
all-input acceptance remain unverified. The task queue has no pending/running
jobs at handoff.

## Remaining uncertainty

No final mixed-policy structural/confidence admission, independent input closure,
real-template/affinity/active-guidance closure, padding-across-384 validation, or
uninstrumented speed/VRAM claim is made by this record. Original FP32 behavior is
protected by the existing tests and selected direct equivalence controls.
