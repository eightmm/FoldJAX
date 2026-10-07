# Four-way precision follow-up

## Correction: replay compute dtype forwarding

The initial OpenDDE BF16 replay narrowed parameters but omitted the
`trunk_dtype` argument to `opendde_infer_static`. The production CLI already
forwards this argument. Therefore the initial BF16 results do **not** describe
the ordinary CLI's compute-dtype path. The replay now forwards it too, with
FP32/BF16 regression tests checking the actual model invocation.

With the existing LayerNorm correction held fixed, forwarding this one argument
reduced 3GCA's native-BF16-relative entity maxima from RNA 0.556799 / ligand
0.316092 Å to RNA **0.020462 / ligand 0.031812 Å**. Both pass 0.05 Å.
Preserving only the two native FP32 geometry projections did not reproduce
this improvement (RNA 0.557767 / ligand 0.316518 Å).

This identifies the dominant tested cause of the large 3GCA discrepancy, not
complete implementation equivalence. LayerNorm required a real implementation
correction; geometry FP32 exceptions and native AMP boundaries still differ.
The harness also retains its own explicit kernel choices, so matching the
compute-dtype argument is not an ordinary-CLI end-to-end acceptance claim.

Corrected frozen-source BF16 reruns for 1URN and 5SAK were submitted as jobs
184 and 185. Job 186 collects their completed results with the earlier 3GCA
control, preserving each route's provenance, into
`bench/experiments/opendde-corrected-replay-2026-09-06.json`. Pending jobs are
not successes. Fixed harness, gate and entity tests: **47 passed**; affected
Ruff and diff checks passed. Full CPU/release gates and push were not run.

## Original experiment record

The requested comparison is upstream FP32, upstream BF16, FoldJAX FP32 and
FoldJAX BF16. Native precision drift and port drift are separate measurements;
their RMSDs must not be subtracted as an additive causal decomposition.
The earlier 0.05 Å threshold remains a diagnostic column. This experiment does
not change production defaults or claim universal BF16 equivalence.

## OpenDDE

Submitted six FoldJAX runs: FP32 and the existing BF16 embedder/trunk candidate
for 5SAK (protein/ligand), 1URN (protein/RNA), and 3GCA (RNA/ligand).
Each uses five sample-index-paired predictions, 200 steps and 10 cycles.
The already captured native FP32/BF16 arms have exactly equal realized input,
MSA and sampler tapes. Each port independently featurizes the raw input and
must pass the existing input audit before replay. The BF16 port candidate does
not implement the same operator policy as native AMP; do not label it as such.

Native TF32 remains on. The replay harness now accepts an explicit JAX
`--matmul-precision high` setting, and records it with trunk dtype. `high`
permits TF32 but does not assert identical native kernels. Existing callers
retain the previous `highest` default. Production inference is unchanged.

Compare all six unordered pairs of the four arms. Fit one global proper
Kabsch per sample and measure each entity instance without another fit.
Use raw sampler coordinates, including predicted terminal OXT, not writer
repairs, confidence ranking or experimental coordinates.

Queue jobs 165–170 use a frozen source snapshot and record input, weight,
source and tape identity. Submission is not completion. The result collector
requires all six coordinate artifacts and successful input/tape audits; it
must fail rather than emit a completed partial matrix. Its output target is
`bench/experiments/opendde-four-way-2026-09-06.json`.

## OpenFold3 repeat diagnosis and port gap

The prior native FP32 repeat differed by up to 0.059586 Å for protein and
0.322996 Å for ligand despite exact captured input/tape. Pre-forward settings
did not record actual chosen chunks. There is no evidence yet attributing the
difference to chunk selection, TF32 or Triton kernels.

Two further native FP32 runs (queue 171–172) preserve model settings and record
actual selected chunks and trunk boundary hashes. Trunk capture adds CPU
synchronization, so these are instrumented diagnostic controls, not proof of
unmodified runtime determinism. Compare their actual tapes before coordinates.

FoldJAX OpenFold3 currently accepts initial/churn noise tapes but has no native
quaternion/translation tape adapter. The historical comparison disabled
augmentation. It is not eligible for this study. A native-policy BF16 port is
also not implemented; blanket weight conversion is not Lightning AMP parity.
Consequently OpenFold3's four-way port matrix remains unrun, separately from
its native precision measurements and the submitted repeat diagnostics.

Verification: focused replay, coordinate-gate and entity tests: 45 passed;
affected Ruff and `git diff --check` passed. GPU jobs were submitted, not yet
accepted as completed at the time of this record. Full CPU suite, release
gate, commit and push were not run.
