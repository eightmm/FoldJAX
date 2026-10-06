# OpenBind native-default panel preflight

## Seven-case private comparison complete, not admitted

See [compact results and conditions](openbind-private-panel-summary-2026-09-09.md).
Final job 1082 completed exit 0. 3GCA entity maxima are RNA 0.01106679 and
ligand 0.00548555 A; all n5 entity checks are below 0.05 A. Public maxima are
pLDDT 0.12575992 points, pTM 0.00020979242 and ipTM 0.00016882534.
New native job 1081's thirteen artifact hashes/byte counts were verified;
its 57 input and 605 tape fields equal the historical capture exactly.
The successful reporter binds public outputs to the new native trace.

All seven core comparisons now exist for one private-source snapshot. Only
1URN/3GCA meet the 0.05 A coordinate criterion for all entities and samples;
this is neither full confidence admission nor independent input proof.
No default, release or push claim is made. Detailed earlier checkpoints remain
historical and are superseded by the compact summary where marked pending.

## Fresh 1URN native capture verified

Update: candidate 1080 completed exit 0; `1urn-comparison.json` succeeds with
capture-time public digest binding. All ten entity/sample coordinate checks
are below 0.05 A: protein maximum 0.01259760, RNA maximum 0.01361143 A.
Public confidence maxima remain pLDDT 1.64165861 points, pTM 0.00067307658,
ipTM 0.00106954800. Structure alone passes the diagnostic threshold; confidence,
independent preprocessing and missing candidate raw-pLDDT still preclude model
admission. Six of seven private-route output comparisons are now complete.

Job 1079 completed exit 0. Its new capture under the private-panel snapshot
contains raw confidence and a pinned trace. All thirteen recorded public/raw/
trunk artifact hashes and byte counts were independently recomputed and match.
The current tape adapter and checkpoint/source provenance checks succeed:
noise shape (201,5,1225,3), MSA selections (4,1024).

Against historical `precision-policy-20260906/openfold3-v2/protein_rna_1urn/
32-true`, all 57 input fields and 609 tape fields have identical keys, dtypes
and array values. This does not establish output equality or independent JAX
preprocessing, but separates capture-completeness repair from changed inputs
or random draws. Candidate 1080 is running against this new capture; no output
comparison is claimed until its finished artifact is verified.

## 7ST3 private result: chain B remains above 0.1 A

Job 1078 completed exit 0; `7st3-comparison.json` generated successfully.
Chain A maximum is 0.05130189 A; B maximum is 0.11900713 A. B exceeds 0.1 A
in samples 1 (0.11900713) and 4 (0.10810645). System-wide maximum 0.06702241 A
would hide these entity failures and is not the acceptance metric. Public
maximum differences are pLDDT 2.50398968 points, pTM 0.00012588226 and ipTM
0.00024383967. No thresholds or public defaults changed.

This brings the new private-route output comparisons to five of seven cases;
none closes all required gates. At completion, 1079 (fresh native 1URN) is
running, with 1080-1082 retaining their serial queue positions. Full input
identity, raw-pLDDT coverage and performance remain distinct incomplete gates.

Verification: terminal job success, reporter completion and all n5 entity and
public/raw confidence values inspected. No model admission, review or push.

## Completed private artifacts rehashed

The first and both repeat prediction archives for 5SAK, 1UBQ, 7R6R and 3V7E
were independently rehashed against each `finished.json`; all twelve hashes
match. Every stored per-array same-callable repeat flag is true for these four
cases. This verifies artifact integrity and the saved repeat evidence, not
cross-process reproducibility or native numerical equivalence. The unresolved
native differences above 0.1 A remain unchanged.

At this checkpoint 1078 remains live, has saved its first 7ST3 prediction and
first repeat, and is not restarted. Native 1URN/3GCA producers and their
candidate consumers retain their queue positions. No release or push claim.

## 3V7E private result: RNA and ligand remain above 0.1 A

Job 1077 completed exit 0 and `3v7e-comparison.json` was generated against its
bound native capture. Maxima are protein P 0.09875127, RNA R 0.25379033 and
ligand L 0.14097612 A. Protein is gray; RNA (sample 1) and ligand (sample 4)
remain priority failures. Public maxima: pLDDT 2.23600003 points, pTM
0.00407008668, ipTM 0.00363260675. Private triangles do not close this case.

The missing raw pLDDT leaf is source-localized: `inference.py` computes
`plddt_logits` but returns only `compute_plddt(plddt_logits)` in Prediction.
No output contract was changed in this checkpoint; a future capture fix needs
an observer/output-return bridge, not retroactive completion of missing data.

Verification: terminal job and successful report inspected. Three additional
CLI regressions reject private controls combined with cuEq or either trunk
injection before any artifact read; combined core/report/block tests 26 passed,
Ruff and diff checks passed. These test-only additions do not alter the running
snapshot. Remaining queued cases, raw-pLDDT capture, review and push stay open.

## 7R6R private result and remaining native producers

Job 1076 completed exit 0; `7r6r-comparison.json` was generated successfully
under the shared private snapshot. Entity maxima are A 0.12425528, B 0.08653529,
D 0.08961738 A. Samples 1-4 are below 0.05 A for every entity; sample 5 gives
all three maxima. A remains above 0.1 A. Public maximum differences are pLDDT
0.80012105 points, pTM 0.00048290622 and ipTM 0.00089717601. No model admission.

Historical 1URN/3GCA inputs have no raw-output archive or trace. Fresh native
captures and dependent fail-closed candidate replays were therefore queued:
1079/1080 for 1URN, 1081/1082 for 3GCA. These reuse the same immutable native
observer/candidate snapshot and pinned weights; no existing artifacts are
overwritten. They follow 1077 (3V7E, running) and 1078 (7ST3, queued) in the
single-slot queue. A failed producer cannot be replaced silently by historical
captures: the consumer uses only its explicitly selected new directory.

Verification: 1076 terminal success and entity/public/raw report inspected;
1079-1082 accepted by tsp. Missing-case numerical results and fresh capture
identity still require verification after completion. Shared-input and missing
candidate raw-pLDDT limitations remain unchanged.

## Same-source private panel extension submitted

Update: 1075 completed exit 0 and its comparison report generated successfully.
1UBQ per-sample RMSDs are 0.12287701, 0.01107699, 0.07671923, 0.00895210,
0.01085691 A. Maximum 0.12287701 exceeds 0.1 A. Public pLDDT max difference
is 1.44345304 points and pTM 0.001509803. Thus the private route is not a
universal fix even after the 5SAK protein improvement. At this update 1076 is
running and 1077/1078 remain queued; no default is adopted.

Following 5SAK completion, jobs 1075/1076/1077/1078 use the unchanged
`openbind-private-batched-20260909-QK9YSM` snapshot for 1UBQ, 7R6R, 3V7E and
7ST3 respectively. Each requests n5, three calls, captured trunk outputs and
private sample-mapped triangles, with no intermediate injection. Existing
native captures are under `af3-final-warm-20260909-BjN099/openbind-native`.
Each replay must pass its capture/config/checkpoint validation before inference.

These new cases use independent empty compilation-cache directories and fresh
autotuning, unlike the loaded historical map in the 5SAK control. Cross-process
repeatability or identical algorithm selection is therefore not established.
GPU availability and the absence of running/queued jobs were checked before
submission; all jobs use the shared serial tsp queue. Other work is untouched.

Verification at submission checkpoint: 1075 is live with preflight saved;
1076-1078 were accepted by the queue. Numerical results remain pending and
submission is not model acceptance. Historical 1URN/3GCA captures were located
but need provenance checks before completing the seven-case extension.

## Full-core private control completed: protein gray, ligand unresolved

Job 1074 completed successfully. The current reporter produced
`openbind-private-batched-20260909-QK9YSM/comparison.json` against the unchanged
native capture. There is no native or candidate trunk injection. n5 coordinates
are globally aligned over the system before measuring each entity separately.

| Sample | Protein A RMSD (A) | Ligand L RMSD (A) |
| --- | ---: | ---: |
| 1 | 0.06223911 | 0.04487572 |
| 2 | 0.06056202 | 0.02578737 |
| 3 | 0.03217174 | 0.01548112 |
| 4 | 0.04877945 | 0.19397439 |
| 5 | 0.05188813 | 0.03829728 |

Compared with recorded baseline 1060 maxima (protein 0.67320428, ligand
0.27237043), protein improves approximately 10.8x; ligand remains >0.1 A in
sample 4. This is not a complete isolated rounding ablation: private kernels
and confidence sample scheduling differ, and new operations can autotune.
Public confidence maximum differences are pLDDT 2.52583510 points (0-100),
pTM 0.00051840023 and ipTM 0.00422997088. Do not infer confidence preservation.

All eleven stored outputs, including coordinates, raw confidence and trunk
representations, are exactly equal across three calls (all prediction archive
hashes also equal). Input/tape/checkpoint and prediction bindings pass the
reporter's checks. Independent preprocessing and device tape-consumption proof
are absent; the old native coordinate/public confidence artifacts lack
capture-time digests. Candidate raw pLDDT logits remain missing. These prevent
model admission even apart from the unresolved numerical differences.

Verification: job terminal exit 0; comparison generated successfully; all n5
entity values, public/raw confidence and repeat records inspected. This is a
shared-input core diagnostic, not performance, release or push. The earlier
running checkpoint below is superseded by this completed result.

## Sample-mapped private control running

Source inspection of `inference.confidence_pair` and its caller establishes
that the n5 confidence path below cutoff 750 expands pair inputs to five
samples. The private kernel accepts B=1 only. The diagnostic adapter now maps
independent samples through that kernel, preserving the existing PairBlock
residual and transpose flow. This explicitly differs from native batched
kernel scheduling; numerical equivalence remains to be measured, not assumed.
Production defaults are unchanged.

Job 1074 uses snapshot `openbind-private-batched-20260909-QK9YSM`, the same
5SAK native capture and prior tuning map as failed job 1073, without any trunk
injection. At this checkpoint it is live on the GPU, with preflight saved and
no finished prediction. Do not interpret submission as numerical acceptance.

Verification: 23 core replay/report/block tests passed, including JIT sample
and mask preservation at B=1/5, and diff check passed. Queue wait handle remains
live; full-model outputs, review, warm performance and push are pending.

## Full-core private control stops at confidence shape guard

Job 1073 in `openbind-private-full-20260909-bJYfQS` exits 1 during tracing.
The stack reaches `heads.pairformer_embedding` and then the private
multiplication validator, which requires a square FP32 `[1,N,N,C]` tensor with
C=64/128. No completed prediction or structure/confidence result exists.
The exception does not print the actual rejected shape, so its precise axes
must be established before implementing batching. This is an integration
limitation, not a measured numerical regression.

The new `--private-pair-operators` control requires XLA and rejects any trunk
injection. It records the private route and its helper digest in preflight;
the reporter also exposes this deviation. Inputs remain shared native features,
n5/200 steps/4 trunk passes, and the prior 5SAK autotune map is loaded (new
operations may still tune). No public default or production source changed.

Verification: job 1073 terminal exit 1 and traceback inspected; ten replay and
block-control tests passed before launch. No warm, full-model parity, review or
push claim. Next: validate the actual confidence pair shape and support its
independent batch axes without changing native kernel arithmetic.

## Propagated private PairBlock: real-input RMSE reduced 29.8x

Job 1072 completed in `openbind-private-block-20260909-O9X7Oh/candidate`.
Only the first block input is native; subsequent values propagate through the
candidate without intermediate teacher replacement. The existing PairBlock
residual, transpose and transition code is reused. During tracing only,
multiplication uses the native-private residual wrapper and attention uses the
native-private update wrapper. Both monkeypatches are restored on exceptions.

| Whole-block metric | Ordinary XLA (1070) | Private triangles (1072) |
| --- | ---: | ---: |
| RMSE | 0.00706784424 | 0.00023698661 |
| Maximum absolute error | 0.163155556 | 0.033325195 |

The approximately 29.8x RMSE reduction survives candidate propagation within
this real block. It does not prove accumulated trunk, diffusion or confidence
parity. Errors are feature units, not angstroms. All three outputs agree
exactly within this executable. Public defaults and production source are
unchanged; `--block-backend native-private` is diagnostic-only.

Verification: job exit 0 and both block reports inspected; four focused tests
pass, including sequential residual dispatch and restoration on trace failure;
scoped Ruff and diff checks pass. Next is a full-model fixed-tape control using
this candidate flow, without native trunk injection. Review and push pending.

## Existing private kernels on real 5SAK boundaries

Job 1071 completed using snapshot `openbind-real-private-20260909-BkSTD3`.
The native captures are unchanged from job 1069. The replay now offers an
explicit diagnostic-only `--boundary-backend native-private`; its full-block
output remains the ordinary XLA control, not an integrated private block.

| Operator | Ordinary XLA RMSE (1070) | Private RMSE (1071) |
| --- | ---: | ---: |
| tri_mul_out | 0.002257919 | 0 |
| tri_mul_in | 0.004156722 | 0 |
| tri_att_start | 0.002922137 | 0.00001264377 |
| tri_att_end | 0.003416166 | 0.00001369281 |
| pair_transition | 0.000128406 | 0.000128406 |

Both multiplication residuals exactly match the captured native values.
Attention RMSE decreases approximately 231x/249x, but maximum errors remain
0.00375175/0.00387764 in feature units. These are not angstroms. Each operator's
three same-callable outputs agree exactly. Saved output hashes and the final
source-stability check are now recorded by the replay.

This uses the existing native-faithful private implementation, including its
kernel-specific rounding and norms; it does not isolate one constituent fix.
It extends earlier synthetic tests to these actual first-block inputs only.
The next discriminating integration is an entire candidate PairBlock using
these operators, before a full-model panel. Public defaults remain unchanged.

Verification: queue job exit 0; all five operator reports and repeat flags read;
capture-validator tests 3 passed, scoped Ruff and diff checks passed. Full-model
structure/confidence, independent review and push remain incomplete.

## Real boundary replay: triangle paths dominate relative update error

Job 1070 completed in openbind-boundary-replay-20260909-JgMYYJ/candidate.
Each operator consumes its own captured native input, so errors below are local,
not propagation from earlier candidate operators. Native multiplication outputs
include the residual; relative update L2 uses native output minus captured input
as denominator. Attention and transition already return updates.

| Operator | Max absolute error | RMSE | Relative update L2 |
| --- | ---: | ---: | ---: |
| tri_mul_out | 0.04565430 | 0.002257919 | 0.00104147 |
| tri_mul_in | 0.07284546 | 0.004156722 | 0.00106376 |
| tri_att_start | 0.12168503 | 0.002922137 | 0.00106455 |
| tri_att_end | 0.16451073 | 0.003416166 | 0.00123200 |
| pair_transition | 0.01559067 | 0.000128406 | 0.00001758 |

The similar approximately 0.1-percent relative errors across all four triangle
operators prioritize shared projection/normalization/precision behavior over an
unproven single missing residual. They do not by themselves distinguish those
causes or justify changing a default. Native Triton versus candidate XLA remains
an explicit kernel deviation. All figures are feature errors, not angstroms.

Verification: job exit 0, five captured-input identity checks, saved operator
outputs and update-relative norms inspected. This is a teacher-forced first
block diagnostic; full-model and all-case acceptance remain incomplete.

## Real-block chunk control is unchanged

Native trace records Pairformer chunk 1024 on every recycle; effective settings
also specify 1024. Job 1068 reran the real-activation candidate block with
--chunk-size 1024 in openbind-pair-chunk-20260909-HWVoQk/candidate. Its entire
outputs.npz SHA256 equals the unchunked job 1067 archive, with all repeats equal.
Max error remains 0.16315556 and RMSE 0.007067844. Thus this numerical chunk
setting does not explain the measured first-block discrepancy in this pair.
Equal chunk numbers do not prove identical native/JAX internal partitioning.

Verification: native tuner/settings read, replay/capture tests 9 passed,
Ruff/diff passed, job 1068 exit 0 and output identity/report inspected. Kernel
arithmetic differences still require internal-boundary localization.

## Real-activation first PairBlock replay completes

Job 1067 used openbind-real-pair-replay-20260909-4NYNLG/candidate, captured
native first-block inputs and released checkpoint weights. Candidate XLA with
matmul high and no row chunking yields max absolute error 0.16315556 and
RMSE 0.007067844 against the native Triton block output. These are feature
units, not structural RMSD. All three candidate outputs are byte-identical.

This localizes a measurable discrepancy to a single real-activation block,
without attributing it yet to multiplication, attention, normalization or
transition. Native/Candidate kernel and chunk choices are not identical;
the replay is teacher-forced and cannot establish whole-model parity.

Verification: job exit 0, saved output digest and repeat bytes rechecked.
Three new capture-validator regressions pass for valid data, changed archive,
and invalid/nonfinite masks; Ruff/diff checks pass. Numerical tolerances and
production source are unchanged. Internal-boundary probes remain required.

## Real first PairBlock capture available

Job 1066 completed in openbind-real-pair-20260909-VX3M2t/native with the
optional first-pair observer. Input z and output z are finite FP32 arrays of
shape (1,437,437,128); pair_mask is finite FP32 (1,437,437). Input/output
archive sizes and SHA256 match the trace records. The captured module is
pairformer_stack.blocks.0.pair_stack, first invocation only, not all recycles.

Compared with the prior native capture openbind-native-outputs-20260909-yqosQT,
all input/tape arrays agree exactly in dtype and values. Final-coordinate
observer bridge maxima are protein 0.05146042 A and ligand 0.05389550 A.
The added observation is not established neutral, and these two runs do not
separate observation effects from native repeat variation. Use the capture as
a teacher-forced single-module reference, not a new ordinary execution claim.

Verification: job exit 0, recorded hashes, shapes, finite arrays, input/tape
identity and global-fit per-entity bridge inspected. Existing synthetic probes
do not substitute for replaying these real activations; that GPU replay remains
the next required numerical check.

## Pair-path dependency order and CPU reference check

Pinned native run_trunk and the port agree on dependent updates: pair recycling
projection, template update, MSA module, single recycling and Pairformer. The
port evaluates the independent MSA embedding earlier; no data-dependency error
was identified from that scheduling difference. PairBlock agrees on outgoing/
incoming multiplication, start/end attention with transposition, then masked
transition. This source inspection is not GPU arithmetic equivalence.

The existing randomized pinned-upstream CPU comparisons for pair_block,
pairformer and triangle were rerun in the parity environment: 29 passed in
5.84 seconds. They cover nonzero mapped weights and the native CPU reference,
not real-weight 5SAK GPU kernels, TF32 lowering or whole-model tolerance.
No order rewrite is justified by this check; the next localization must examine
real activations and GPU arithmetic rather than assume a missing residual.

## Component injection prioritizes pair, without excluding single

Jobs 1064/1065 completed in openbind-trunk-components-20260909-FWXWyU.
Both maps exactly match the 489-entry candidate-injection parent. Returned
representations exactly match the intended native/candidate component sources.
The single intervention replaces both single_inputs and single; pair replaces
only pair. All rows below are injection diagnostics, not ordinary execution.

| Native trunk components | Protein maximum RMSD A | Ligand maximum RMSD A |
| --- | ---: | ---: |
| None, candidate reinjection | 0.58102674 | 0.26655692 |
| single_inputs + single | 0.43882895 | 0.18589311 |
| pair only | 0.25608254 | 0.10682324 |
| All three | 0.10975947 | 0.08644099 |

Pair replacement reduces structural residuals more in this case, so pair-path
arithmetic is the next priority; single also contributes and effects need not
be additive. Maximum pLDDT errors are 3.64491827 points for single replacement
and 2.33439094 for pair; pTM 0.00065241380/0.00074704365 and ipTM
0.00155936386/0.00440348481 respectively. Confidence does not improve uniformly.
Verification: successful jobs, exact selected arrays, equal canonical tuning
maps and current hash-bound entity/confidence reports checked. No tolerance or
production-default change, and no full-model admission.

## Candidate reinjection bounds the graph confound

Job 1063 completed from openbind-candidate-trunk-control-20260909-6w9jHc/5sak.
The three returned representations equal the original candidate trunk exactly.
Both native- and candidate-injection tuning maps contain the same 489 records;
no additional entries were required. Reinjecting candidate trunk changes the
original baseline coordinates by maximum protein 0.09407861 A and ligand
0.05535519 A. Injection is therefore not a neutral transformation.

Relative to native, candidate-trunk reinjection has protein 0.58102674 A and
ligand 0.26655692 A, versus native-trunk injection 0.10975947/0.08644099 A.
Public errors with candidate trunk: pLDDT 3.67960817 points, pTM 0.00035713239,
ipTM 0.00247197065. This supports a substantial trunk-value contribution even
after examining the graph confound, not an exact additive decomposition or
proof that the diffusion tail is correct. Both injection arms remain diagnostic.

Verification: job exit 0, candidate provenance gate, returned trunk equality,
identical canonical tuning maps and system-fit entity comparisons checked.
Ordinary model admission and production changes are still withheld.

## Native trunk diagnostic substantially reduces 5SAK residuals

Job 1062 completed. Returned single_inputs/single/pair are exactly equal to
the three native trunk arrays. All 477 baseline tuning entries are retained
in the 489-entry injection map; the extra 12 entries reflect the changed graph.
Same-callable repeat flags are all true; injection is explicit in the report.

Native-trunk tail entity maxima are protein 0.10975947 A and ligand
0.08644099 A, versus full baseline 0.67320428/0.27237043 A. Public maximum
errors are pLDDT 2.23427898 points, pTM 0.00043980404, ipTM 0.00223510179.
This supports a substantial upstream-of-tail contribution, not a sole-cause
claim: the protein still exceeds 0.1 A, ligand is gray, confidence differs,
and injection changes compilation. Candidate-trunk injection under a shared
extended map is needed to separate graph effects before causal attribution.

Verification: successful execution, native returned representation equality,
parent-map inclusion, current hash-bound report and repeat flags checked.
This is explicitly native-intermediate injection, never ordinary model parity,
independent input evidence, or a warm-performance result. No production change.

## Native trunk injection diagnostic queued

Job 1062 uses openbind-native-trunk-control-20260909-2bZBNF/5sak, ordinary
production aggregation precision and the same 5SAK native capture as 1060.
It replaces the trunk call during JIT tracing with the three hash-checked
native arrays. This deliberately introduces native intermediate constants and
changes the compiled graph; no ordinary model or performance admission follows.
The baseline 5SAK autotune map is loaded with explicit extension allowed.

The bench-only --inject-native-trunk option validates captured array count,
identity, shape, FP32 dtype and finite values. Preflight and report expose the
injection identity. After completion, compare returned representations with
native arrays to check the intended replacement, then interpret the tail's
residuals separately from full-model error. Matched candidate-trunk injection
and complete-map repeat remain controls needed before causal attribution.
Verification so far: 39 focused tests and Ruff passed; GPU output pending.

## 5SAK aggregation expansion: input improves, structure remains unresolved

Jobs 1060/1061 completed. All 477 baseline tuning records are preserved in
the 479-record candidate map. Both processes report all same-callable repeats
equal. Current hash-bound reports give:

| Metric | Baseline | Aggregation highest |
| --- | ---: | ---: |
| Protein maximum RMSD A | 0.67320428 | 0.62814144 |
| Ligand maximum RMSD A | 0.27237043 | 0.27710985 |
| pLDDT maximum error, points | 3.93060929 | 3.96788603 |
| pTM maximum error | 0.000604354 | 0.001118999 |
| ipTM maximum error | 0.002104163 | 0.003222930 |
| Input embedding max error | 0.003640175 | 0.000124931 |
| Input embedding RMSE | 0.0000475138 | 0.000000477669 |

Input embedding RMSE improves about 99-fold, confirming the effect on a second
chemistry case. Final structure and confidence do not improve uniformly;
both entities still exceed 0.1 A. Aggregation precision is therefore not a
sufficient explanation or fix for 5SAK. Do not adopt the change based solely
on the favorable 1UBQ maximum or call the model closed. A frozen extended-map
cross-process repeat for this case is still absent.

Verification: successful jobs, baseline map inclusion, actual native-bound
coordinate/confidence reports, input representation differences and repeat flags
checked. The native/candidate backend deviation and shared-input scope remain.
Production source and acceptance thresholds are unchanged.

## 5SAK matched aggregation expansion queued

Jobs 1060/1061 compare baseline and aggregation-highest XLA paths against the
same native capture openbind-native-outputs-20260909-yqosQT/native. Native
uses Triton in both comparisons; this is not matched native/JAX kernel policy.
Both use n5, native confidence cutoff, observed trunk and three same-callable
executions. Baseline writes its own 5sak-autotune.textproto; highest loads and
extends that map. Verify parent-map inclusion before interpreting results.

Baseline lives under openbind-trunk-fixed-20260909-PvqlhI/5sak, candidate under
openbind-aggregate-highest-20260909-PvZTzA/5sak. Whole src hash comparisons
identify only models/openfold3/models/atomize.py as different. The two explicit
highest-precision aggregation contractions are the isolated source change.
Verification so far: source comparison and queue running/queued status only;
no 5SAK outcome or production adoption claimed before completed artifacts.

## Aggregation probe repeats exactly under the extended map

Job 1059 reloads the 496-entry map with complete coverage required and a
separate executable cache. Its map matches the producer's, and all 11 output
arrays (including three trunk representations) are bitwise identical across
processes. Thus the 0.07814142 A maximum and sample-3 regression are reproduced,
not discarded as tuning noise. This remains one input, not model admission.

Verification coverage correction: the earlier five test_atomize_indexed tests
exercise token-to-atom broadcasting, not the changed aggregation. The actual
test_torch_parity_aggregate module was now run against the isolated candidate
and pinned upstream in the existing parity environment with JAX_PLATFORMS=cpu:
9 passed, covering sum/mean, masking, empty tokens and invalid aggregation.
The earlier five-test result is not evidence for atom-to-token parity.
Production atomize source remains unchanged pending wider-case validation.

## Isolated aggregation precision probe reduces the first mismatch

Job 1058 ran openbind-aggregate-highest-20260909-PvZTzA/trunk. Only the two
atomize aggregation einsums in this external snapshot specify precision=highest;
production source is unchanged. All 494 parent autotune records are retained
and two new records are added. This is a controlled map extension, not identical
whole-graph compiler choices. Five scoped atomize CPU tests passed first.

Input embedding maximum error falls from 0.001691818 to 0.0000300407;
RMSE from 0.0000387522 to 0.000000168827 (about 230-fold). Differing values
fall from 305 to 75. Final single RMSE is 0.40300477, pair 0.03681163.
This supports aggregation contraction precision as a major contributor to the
observed input embedding discrepancy, without proving the cause of every
downstream structural difference.

Entity A RMSDs are 0.01421068, 0.00976420, 0.07814142, 0.01036108 and
0.00972850 A. Maximum falls from 0.37531818 to 0.07814142 A, but sample 3
worsens from 0.00874731 and remains gray. Maximum pLDDT difference is
0.41558447 points, pTM 0.00046649302, ipTM zero. This one-case result does
not authorize universal adoption or relaxed confidence gates. Freeze the
extended map and repeat before further interpretation or production changes.

Verification: job 1058 exit 0, parent-map inclusion, exact representation
shapes and current hash-bound coordinate/confidence report checked. No native
intermediate injection, production default change, commit or push.

## Input embedding difference localized by channels

For the fixed-map trunk pair, 305 of 34,124 input-embedding values differ,
confined to atom-encoder channels 21, 62, 201, 249 and 337. The appended
restype [384:416], profile [416:448] and deletion_mean [448:449] are exactly
equal. Source concatenation order was inspected; this is not a preprocessing
feature mismatch in those appended fields. Atom-encoder channels have maximum
error 0.00169181824 and RMSE 0.00004190386.

The port aggregates projected/ReLU atom features with a one-hot einsum, while
native uses its scatter-based aggregation. The contraction's precision policy
is therefore a candidate first arithmetic difference, not yet a demonstrated
cause. A scoped aggregation-only precision probe is preferable to changing
whole-model dtype or accepting a favorable autotune run. No model change was
made from this channel-level observation.

## Fixed-map trunk observer bridge succeeds

Job 1057 completed from openbind-trunk-fixed-20260909-PvqlhI/trunk with
--capture-trunk and three calls. The frozen producer map is unchanged, and
all eight original output leaves (coordinates plus confidence) equal the
unobserved producer output array-for-array. Same-callable repeat flags are
all true. Thus this particular observer bridge preserves final outputs.

Native nonfused versus candidate trunk statistics (feature units):

| Representation | Max absolute | RMSE | Relative L2 |
| --- | ---: | ---: | ---: |
| single_inputs | 0.001691818 | 0.0000387522 | 0.0000968892 |
| single | 7.96875 | 0.50275263 | 0.0000361118 |
| pair | 3.421875 | 0.03758798 | 0.0002077336 |

Single/pair relative differences are comparable in scale to the earlier native
Triton/nonfused control, not proof of harmlessness: the sampler can amplify
small representation changes. Input embedding already differs despite identical
captured model inputs. Locate that first arithmetic difference before inferring
a diffusion-only defect. No native intermediates were injected.

Verification: job exit 0, canonical tuning maps equal, matching representation
shapes and all original output arrays compared. This run also completed the
new harness/compiler-provenance guards. The remaining first-sample structure
error is unchanged; no model admission or performance claim follows.

## Native trunk backend control

The existing 1UBQ native Triton and nonfused trunk-00 archives were compared
after checking their trace-recorded SHA256 and byte counts. Shared input
embedding (1,76,449) is exactly equal. Final single (1,76,384) has maximum
absolute difference 10.0625, RMSE 0.55145277 and relative L2 3.9610221e-5.
Final pair (1,76,76,128) has maximum 2.85131836, RMSE 0.05271440 and relative
L2 2.9132102e-4. These are feature units, not angstroms, and a two-run backend
control, not a calibrated tolerance. Both captures have identical input/tape
values as verified above. Candidate trunk interpretation must distinguish this
native path variation from cross-framework error.

The next snapshot openbind-trunk-fixed-20260909-PvqlhI contains the updated
replay provenance checks. GPU submission is still deferred while the specific
collaborator Boltz process was confirmed live; no collaborator job was altered.

## Replay provenance completion

Future replay preflights now record the replay/adapter script digests, requested
call count and an explicit allowlist of compiler/cache environment controls.
The scripts and those controls are checked again before writing finished.json.
Previously source_hashes covered only src Python files, not the bench runner;
preflight equality therefore did not establish compiler environment equality.
Historical artifacts are unchanged and retain queue-command-only evidence for
the autotune flags. No arbitrary environment variables are collected.

Verification: 38 focused tests pass, including the compiler-control allowlist
and absent-value preservation; scoped Ruff and diff checks pass. No new GPU
result uses these added provenance fields yet. The intended frozen-map trunk
observer bridge was not launched while a collaborator's Boltz process was
observed using the GPU outside the tsp queue; that process was left untouched.

## Autotuning reuse stabilizes this cross-process pair

Jobs 1054/1055 completed successfully. Both tuning maps contain 494 canonical
records and are byte-identical. Across producer and consumer, coordinates and
all seven emitted confidence arrays are bitwise identical. Each process also
reports both same-callable repeats equal to its first output; repeat archive
hashes were checked. These are separate executable-cache directories with
complete-AOT coverage required on the consumer, not a reused executable.

Both processes have native entity A RMSDs 0.37531818, 0.00993069,
0.00874731, 0.01125812 and 0.01221259 A. Maximum public pLDDT difference is
2.39805830 points; pTM 0.00230375047; ipTM zero. Stabilizing the selected
compiler decisions therefore does not establish upstream equivalence: the
first sample remains outside tolerance, and its worse result must not be
replaced with a favorable independently tuned run.

This controlled pair supports autotuning-associated cross-process variability
as a leading explanation, but does not isolate the responsible contraction or
prove all earlier variability had that cause. The next operator-level comparison
should retain this map so independently retuned executions do not confound it.
Verification: queue completion, both 494-entry maps and bytes, all eight output
leaves and current hash-bound native reports checked. No default change, model
admission or speed/VRAM conclusion follows.

## Cross-process autotune control queued

Jobs 1054/1055 use the unchanged same-callable snapshot, n5 and three calls
per process. Producer writes autotune-produce.textproto; consumer loads it with
xla_gpu_require_complete_aot_autotune_results=true and writes
autotune-reuse.textproto. Separate JAX_COMPILATION_CACHE_DIR paths and
JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=none prevent executable/per-fusion-cache
reuse from replacing this control. The producer and consumer output directories
are autotune-produce and autotune-reuse. Queue command records carry these
environment controls; the historical replay preflight does not record them.
Do not infer identical environments from that preflight alone.

Results are pending. A missing producer map or incomplete consumer coverage
must fail, not trigger an unrecorded fallback. Compare both saved maps and all
coordinate/confidence arrays after successful completion before drawing any
conclusion about cross-process compilation variability.

## Same-callable real-weight repeat is bitwise stable

Job 1053 completed from openbind-same-callable-20260909-b4kU6y, xla-r3,
using --repeats 3 and the same native-unfused capture. The two additional
archives match the first output bitwise for coordinates and all seven emitted
confidence fields. Shape, dtype, storage bytes and recorded file SHA256 were
independently checked, not merely the producer's equality flags.

Native entity A RMSDs for this process are 0.06175566, 0.01082675,
0.00909259, 0.01213017 and 0.00990245 A; maximum pLDDT error is
0.70047542 points, pTM 0.00055935706, ipTM zero. Sample 1 is gray under the
user's triage policy, not a strict pass. This finite result supports prioritizing
cross-process compilation/runtime choices over within-executable variability;
it does not identify autotuning as the cause or prove universal determinism.
The sampler source uses provided arrays for initial/step noise and augmentation
when tapes are present; device-consumption proof remains distinct.

Verification: queue exit 0, hash-bound native report, both repeat archives and
all eight array leaves checked. This is an instrumented tape diagnostic, not
ordinary-RNG warm speed/VRAM evidence. No model default or tolerance changed.

## Next discriminating probe and advisor resolution

Current atom-to-token aggregation already uses a one-hot einsum, not floating
scatter-add (models/atomize.py:141-157). Consequently a scatter determinism
flag is not yet a supported first intervention. The replay now accepts
--repeats to reuse the same compiled callable and tape within one process;
additional outputs and per-leaf equality flags are saved separately. This is
a determinism diagnostic, not a warm-time or VRAM benchmark. Its real-weight
execution remains pending; the four focused modules pass 37 CPU tests.

The bounded advisor's remaining admission question was the actual upstream
ModelUpdate.custom type. Pinned project_entry.py defines it as dict; the
corrected configured_backend helper was additionally executed with real
OF3ProjectEntry and ModelUpdate(presets=['predict']) on CPU. It returned
xla=(false,false), triton=(false,true), cueq=(true,false), preserving the
original dict update. Thus this concern does not require a code change.
Backend evidence also relies on the existing native writer/effective settings
equality check, not merely hook invocation. Trunk/chunk observations remain
recorded diagnostic data, not independently verified admission evidence.

## Same-snapshot XLA repeat changes the interpretation

Job 1051 repeated job 1048 with the same source snapshot, native capture,
checkpoint and backend, writing candidate-xla-repeat under
openbind-native-cueq-20260909-ZTyuG0. It exited 0. Entire preflight JSONs agree.
Native versus repeated candidate entity A RMSDs are 0.02868619, 0.01073615,
0.00931086, 0.01087678 and 0.00850804 A. All are below 0.05 A, unlike the
first execution's sample 1 at 0.23731441 A. Candidate versus repeated candidate
sample 1 is 0.21389704 A; the other samples range 0.00921-0.01112 A.
Maximum native/repeat public pLDDT error is 0.30550670 points, pTM
0.00024665298, ipTM zero. Coordinates and six confidence leaves are not
array-equal across the candidate executions.

This is evidence of same-recorded-policy execution variability, not proof
of its cause or permission to admit the better repeat. In particular the
single-run backend differences below cannot be assigned exclusively to
backend choice. Scatter ordering, autotuning/lowering choices and other
unrecorded device execution state remain competing explanations. Actual tape
identity alone does not make every arithmetic operation deterministic.
Verification: job 1051 exit 0, complete preflight equality, current hash-bound
reporter and per-entity repeat comparison checked. No thresholds were relaxed;
the earlier failed sample and both raw runs remain preserved.

## Reporting review corrections

The native backend override now checks the returned effective flags and rejects
a requested override if the configuration hook never ran. The reporter rejects
disagreement between a replay-bound trace's backend request and the effective
backend. Raw-confidence reports explicitly distinguish replay-bound trace
verification from unbound historical data, and an undeclared raw archive raises
a descriptive ValueError. Neither numerical thresholds nor model defaults change.

The review's proposed chunk-floor relabeling was not applied: the pinned
OpenBind core/utils/chunk_utils.py:403-425 explicitly accepts max_chunk_size
and forwards it by that name to its tuning routine. The review cited a different
OpenFold-derived implementation, not this pin.

Verification: 36 adapter/replay/report/native-capture tests pass, including
effective-backend mismatch, unchanged default update, unbound/bound raw trace
labels and undeclared archive regressions. Scoped Ruff and diff checks pass.
The completed nonfused GPU pair was re-read with the corrected reporter:
both backends xla, replay-bound raw trace verified, RMSD unchanged at
0.23731440977128643 A. No new GPU run, final review admission or push is claimed.

## Completed nonfused 1UBQ control

Four-arm sample-resolved follow-up (all values A, one system fit):

| Pair | Sample 1 | Sample 2 | Sample 3 | Sample 4 | Sample 5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Native Triton / native nonfused | 0.050967 | 0.012435 | 0.011853 | 0.010819 | 0.011987 |
| Candidate cuEq / candidate XLA | 0.478594 | 0.011722 | 0.077845 | 0.010510 | 0.010228 |
| Native Triton / candidate cuEq | 0.282012 | 0.009785 | 0.078580 | 0.010563 | 0.009639 |
| Native nonfused / candidate XLA | 0.237314 | 0.012644 | 0.009255 | 0.011180 | 0.010673 |

Candidate preflight records differ only in native/candidate backend and capture
provenance; recorded input/tape digests, source identity, checkpoint and model
configuration agree. The candidate backend-path comparison is particularly
sensitive in sample 1, while native path variation is smaller in this pair.
These are single executions per arm, not a repeat-calibrated numerical floor
or isolation of one kernel. Sample 3 improves under the nonfused pair but the
sample 1 residual survives. Verification: all four coordinate archives compared
through compare_entity_parity, and candidate preflight differences inspected.

Jobs 1047/1048 both exited 0. The saved `nonfused-comparison.json` under
`openbind-native-cueq-20260909-ZTyuG0` verifies the replay-bound native trace
and capture-time public artifact digests. Native effective configuration has
both triangle flags false, matching the request; the original native capture
has cuEq false and Triton true. All 57 input fields and 609 tape arrays agree
exactly in shape, dtype and values between these native captures.

Native nonfused Torch versus candidate XLA entity A RMSDs for samples 1-5 are
0.23731441, 0.01264422, 0.00925527, 0.01117983 and 0.01067324 A. Maximum
public pLDDT difference is 0.92918769 points (0-100); pTM 0.00024124707;
ipTM zero. The original Triton/cuEq pair had maximum A RMSD 0.28201222 A.
Removing fused triangle paths on both arms therefore does not close this case;
four samples are below 0.05 A but sample 1 remains above 0.1 A. This does not
isolate a single operator because both framework backend paths changed.

Verification: queue exit statuses, effective flags, all input/tape fields and
hash-bound report checked. This remains shared-input core evidence, not an
independent preprocessing or warm-performance result. Candidate pLDDT logits
are missing; raw-confidence/model admission remain false. Final independent
review also identified missing automatic backend-request enforcement and raw
trace-binding disclosure in the reporting harness. The effective flags for
this particular run were checked manually; those general harness issues remain
open and no new backend capture or admission is justified by this record.

Public pLDDT unit check on the four new native captures: recomputing the
50-bin [0,1] expectation from saved FP32 logits with float64 NumPy softmax,
then multiplying by 100, matches each sample's public JSON within maximum
2.8441e-5 points (1UBQ), 2.6185e-5 (7R6R), 2.0064e-5 (3V7E), 2.7664e-5
(7ST3). Without scaling the maximum discrepancies are 93-97 points. This
directly supports the reporter's 0-1 to 0-100 conversion for these artifacts;
the reported 1-7 point candidate differences are not a factor-of-100 mistake.
Float64 reconstruction is a unit check, not bitwise native softmax replication.

Pinned writer sample-order check: core/runners/writer.py:233-272 takes
atom_positions_predicted[b], loops s in range(sample_size), writes
sample_{s+1}, and passes both predicted_coords_batch[s] and the confidence
sample selected with that same s. _take_sample_dim (line67) retains singleton
shared fields but indexes multi-sample arrays directly. No ranking reorder is
performed on this writer path. The observer captures the same model output
tensor before writing. This supports positional correspondence for this pin;
it is not a generic guarantee for other writers or unbound historical files.

## Nonfused backend control

Jobs 1047/1048 enqueue native nonfused 1UBQ and a FoldJAX XLA replay of its
actual tape. Both use openbind-native-cueq-20260909-ZTyuG0; outputs are
native-unfused and candidate-xla. Native `--triangle-backend xla` is the
observer's label for both fused-kernel flags disabled: it is ordinary Torch,
not execution through XLA. Candidate xla means JAX/XLA. Thus this is a
nonfused backend comparison, not identical low-level GEMM implementation.
No dependency installation or model default change is involved. Native input,
actual tape and effective config will be compared with the original Triton
capture before numerical interpretation. Results remain pending.

## Completed four-case expansion (shared-input core only)

Native jobs 1018-1021 and candidate jobs 1026-1029 completed. The candidate
uses cuEq, native uses Triton; FP32, n5/200, four trunk passes and native
confidence cutoff. No native intermediate is injected. One system fit is used
per sample; the table contains per-chain maxima over five samples.

| Case | Entity maximum RMSD A | pLDDT max error /100 | pTM max error | ipTM max error |
| --- | --- | ---: | ---: | ---: |
| 1UBQ | A 0.282012 | 1.992030 | 0.000350 | 0 |
| 7R6R | A 0.774537; B 0.584621; D 0.580117 | 1.435344 | 0.000681 | 0.001046 |
| 3V7E | P 0.084586; R 0.219048; L 0.059910 | 1.407033 | 0.002650 | 0.003544 |
| 7ST3 | A 0.186709; B 0.897094 | 7.300489 | 0.000338 | 0.003540 |

All four have at least one entity above 0.1 A. None is admitted. Reports live
under af3-final-warm-20260909-BjN099/openbind-candidate/CASE as
core-raw-reviewed-comparison.json; native captures are under openbind-native.
They retain historical public-artifact digest limits, shared atom-order trust,
and missing candidate pLDDT logits. Earlier 1URN/3GCA used confidence cutoff0,
so they must not be silently combined into a uniformly configured seven-case
panel. Compile-and-infer durations are not warm performance measurements.

Native cuEq 1UBQ control job1038 failed before producing coordinates from
openbind-native-cueq-20260909-ZTyuG0. CPU construction with the pinned upstream
ModelUpdate confirmed only the two triangle flags differ; configuration
override preservation tests pass. Native reports its cuEq package is not
installed; the wrapper correctly rejects the runner's zero-success summary.
This is an environment failure, not a numerical result. No dependency was
installed or shared environment modified. Actual tape identity, backend numerical
effects and comparison with FoldJAX remain pending. New captures record public
artifact digests; no historical artifact was retroactively changed.

3V7E candidate job1028 completed and its report was saved: system-fit maxima
P 0.08458581 A, R 0.21904764 A, L 0.05991000 A. Protein/ligand are deferred
gray; RNA exceeds 0.1 A. Public confidence max errors are pLDDT 1.40703269
points/100, pTM 0.002650244 and ipTM 0.003543915. It is not admitted.
Native cuEq control is the next backend-discriminating experiment: the upstream
core supports the cuEq flag and disables inplace_safe for that route. Therefore
switching native Triton to cuEq also changes in-place scheduling, and must be
recorded as a native backend control, not merely relabeled as the same kernel.

New shared-input candidate results (jobs 1026/1027): 1UBQ A maximum RMSD
0.28201222 A; 7R6R A/B/D maxima 0.77453745/0.58462057/0.58011668 A.
Both exceed the 0.1 A priority threshold. Public pLDDT max errors are
1.99203035/1.43534351 points on 0-100, pTM 0.000350135/0.000680913,
ipTM 0/0.001045923 respectively. Reports are saved in each candidate directory
as core-raw-reviewed-comparison.json. These retain non-admission, positional
sample/atom-order assumptions and historical public-artifact binding limits.
They use native Triton versus candidate cuEq, not a matched-kernel comparison.
The large residual is therefore not confined to the 5SAK ligand case; its cause
is not established by this panel expansion. 3V7E/7ST3 candidates remain pending.

Native jobs 1018-1020 exited 0 and pass the forward-tape/config adapter,
pinned-source/recorded-checkpoint validation and raw-output SHA256/size check.
Coordinates are finite with five samples: 1UBQ 601 atoms, 7R6R 2526 atoms,
3V7E 3315 atoms. Noise shapes are (201,5,A,3), covering initial plus 200 steps;
four-cycle MSA index shapes are (4,1024), (4,364), (4,1024) respectively.
7R6R's 364 rows reflect available MSA depth, not a 364-row configured cap.
All three record native Triton triangle kernels. 7ST3 native is still running,
and paired candidates remain queued behind collaborator Boltz jobs.

Candidate core replay jobs 1026-1029 are queued after native jobs 1018-1021
for 1UBQ/7R6R/3V7E/7ST3 respectively. They consume the captured actual forward
tapes, use native confidence cutoff, n5/200/four trunk passes and the existing
cuEq backend without native intermediate injection or trunk-return observation.
The same immutable snapshot supplies source and runner. Replay validates
capture/config/checkpoint identity before loading model weights; a failed or
missing producer capture fails rather than silently substituting an old one.
This remains a shared-input core control with explicit native Triton versus
candidate cuEq deviation, not independent preprocessing or warm proof.

Raw-confidence reporting now validates the native raw archive SHA256 and byte
count from trace.json and requires exactly one native outer batch axis; it
never broadcasts mismatched shapes. Job997/job1001 report saved separately as
openbind-trunk-replay-20260909-kCVt0M/5sak/core-raw-comparison.json. Raw RMSE:
PAE 0.05901061, PDE 0.09730045, distogram 0.30710779, experimentally resolved
0.03215205. Candidate pLDDT logits remain unavailable and explicitly missing.
Different predicted coordinates preclude attributing these to the confidence
head alone. Adapter/replay/report CPU suite: 21 passed. An initial digest-schema
assumption was corrected to the actual {sha256, bytes} record before reporting.

Precision source check: the pinned native entry point requests Torch "high";
FoldJAX inference's scoped default also requests JAX "high". No call sites for
the native temporary matmul_precision context were found beyond its definition.
This rejects the simple source-level hypothesis that the replay accidentally
uses the old global "highest" default. It does not prove equal realized GEMM
algorithms: native triangle Triton and candidate cuEq/XLA remain distinct, and
runtime precision overrides or per-operator pins need separate observation.
No precision default was changed on this source-only evidence.

Missing-case native captures queued after the AF3 warm panel: 1UBQ, 7R6R,
3V7E and 7ST3, using the snapshotted raw-output observer under
af3-final-warm-20260909-BjN099/openbind-native. Input existence, unused output
paths and pinned upstream commit were checked before submission. FP32 n5,
native predict preset, MSA server/templates disabled; actual forward RNG,
coordinates, raw confidence and trunk outputs are captured. These are observed
native diagnostics, not independent preprocessing parity or warm performance.
Candidate replay and successful capture validation are still pending.

## Native raw-output/trunk capture submitted

Observer bridge recheck (994 versus 1001): both saved prediction digests
validate. Candidate source, checkpoint, backend and tape hashes agree;
the only config difference is returned_representations. Capture provenance
and input archive hashes differ, so this is not a byte-identical preflight.
Whole-system-fit entity maxima between candidates are A 0.04267761 A and
L 0.12032871 A. Public confidence max absolute differences are pLDDT
0.007689923 on the 0-1 scale, pTM 0.000480652 and ipTM 0.005419433.
Thus returning trunk tensors is not output-neutral in these runs; observed
trunk differences cannot alone identify the ordinary execution's root cause.
This variation is nevertheless much smaller than the roughly 6.89 A native
versus candidate ligand residual. No new GPU run or parity admission.

Job 1001 exited 0 and its prediction digest validates. Native/candidate trunk
arrays have identical shapes. Differences (max absolute / RMSE / relative L2):
single_inputs .00364017 / .0000475138 / .000117679;
single 85.546875 / 3.9271262 / .000278057;
pair 22.7709961 / .32518750 / .003581127.
These are feature-space measurements, not angstroms, and the large absolute
single values must not be interpreted without their scale. Input embedding
already differs; final pair differs more relatively. Representation-return
graph changes and observer effects remain unseparated, so no sole-cause claim.
The report/adapter/runner CPU set now passes 20 tests, including rejection of
changed tape hashes and malformed confidence shapes.

Candidate trunk-output replay 1001 is queued behind collaborator ESMFold2
998-1000, from `openbind-trunk-replay-20260909-kCVt0M`. It consumes new capture
997 and enables existing single_inputs/single/pair representation outputs,
without injecting native representations. Native cutoff750 is retained.
New capture provenance/checkpoint verification and tape parsing pass on CPU;
adapter/runner suite passes 17 tests. Returning extra arrays changes the graph
and requires comparison with the unobserved candidate before interpreting
trunk differences as ordinary-runtime behavior. No result is available yet.

Job 997 subsequently exited 0. Input keys/values and all parsed MSA/noise/
quaternion/translation tapes exactly match the historical 5SAK capture. Raw
output now includes trunk single/pair, coordinates, distogram, pLDDT, PAE, PDE
and experimentally-resolved logits, all FP32. One full run_trunk return is saved
as s_input/s/z (not per-recycle snapshots). Every observed tuner chose 1024.
Old/new native system-fit maxima are A .05988149 and L .06574545 A; this small
but nonzero change does not establish observer neutrality.

Against candidate native-cutoff job994, raw logit max/RMSE differences are
PAE 2.6611791/.05854767, PDE 4.4735050/.09767199, experimentally-resolved
.32292032/.03327365. Confidence consumes differing predicted coordinates,
so these are whole-model differences, not isolated confidence-head errors.
Candidate prediction currently does not expose pLDDT logits, leaving that raw
field comparison incomplete. Captured batch/sample singleton axes are kept
explicit when matching fields. No raw-confidence parity is claimed.

Job 997 uses `openbind-native-outputs-20260909-yqosQT` to extend the existing
native FP32/n5 RNG observer with full returned output tensors, per-run_trunk
arrays and actual chunk tuner choices. Import preflight confirms the legacy
wrapper and imported OpenFold3 class resolve to the intended pinned checkout,
and the checkpoint exists. The job is queued behind collaborator ESMFold2
995/996, with no shared source edits. Raw confidence was missing from historical
captures; this capture is intended to fill that evidence gap. CPU synchronization
at observation points remains a potential observer effect, not ordinary-runtime
parity proof. Completion and captured field coverage are not established yet.

## Completed 5SAK full-core replay: unresolved large residual

Native confidence-cutoff control 994 exited 0. Compared with 989, preflight
differs only at config.per_sample_token_cutoff (0 -> 750); model-source hashes,
checkpoint, input, tape and requested backend are unchanged. Native/candidate
RMSDs per sample: A [1.18688318,.13823512,.07411887,.33170435,.30061050],
L [6.88604352,.07974439,.14839805,.36551447,.09883056] A.
Public confidence maximum differences: pLDDT 13.5424959 points/100,
pTM .0014823403, ipTM .0215583518. Matching this schedule does not remove the
large ligand residual. Output digest verified. This comparison is not proof
that the confidence head feeds coordinates: graph recompilation/reduction
effects remain possible. The matched native-cutoff runner is retained for
future upstream-policy diagnostics, without changing package defaults.

XLA control job 993 exited 0. Preflight differs from cuEq 989 only in
`candidate_backend`; output digest validates. Entity RMSD per sample:
A [.70728221,.04602568,.04923302,.21742019,.10860308],
L [.31051135,.02116529,.01959444,.13851561,.07017583] A.
Public confidence max differences: pLDDT 3.5405653 points/100,
pTM .0007669780, ipTM .0031645812. Ligand maximum decreases markedly from
6.885 A under cuEq, but protein remains above .1 A. This supports backend
sensitivity, not native-Triton equivalence, a sole-cause conclusion or a default
backend change. Native-cutoff750 cuEq control 994 is separately submitted from
`openbind-nativecut-20260909-ZeHud5`; prior controls retain cutoff0.

Configuration follow-up found a concrete replay deviation beyond kernels:
historical native effective `per_sample_token_cutoff` is 750, but the initial
runner inherited FoldJAX's memory-optimized cutoff 0 (serial confidence at n5).
The bench runner now reads the native cutoff explicitly; public defaults are
unchanged. Prior 3GCA/1URN/5SAK and repeat/XLA jobs retain cutoff 0 and must not
be relabeled as matched native confidence scheduling. Native noise schedule
and sampler constants match released_config (sigma16,160/.0004,p7,
gamma.8/min1, noise1.003, step1.5). No causal claim links confidence scheduling
to the coordinate residual; whole-program recompilation can change arithmetic.
Adapter/runner checks still pass 17 tests; the changed native-cutoff runner has
not yet been run on GPU. Kernel/chunk and raw-output gaps remain.

Candidate repeat job 992 exited 0. Both preflight manifests are exactly equal
and both output digests validate. First/repeat system-fit entity maxima are
A .14544811 and L .10423238 A; no returned array is exactly equal. Native/repeat
maxima remain A 1.12845516 and L 6.88201564 A. The large ligand residual thus
persists beyond observed candidate rerun spread. A same-source/tape XLA backend
control is submitted next; it is not a claim that XLA matches native Triton.

Job 989 exited 0; input/output hashes match manifests. Same shared-input,
native-Triton/candidate-cuEq policy as 3GCA and 1URN; system-fit entity RMSD:

| Entity | Sample 1 | Sample 2 | Sample 3 | Sample 4 | Sample 5 | Max RMSD A |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Protein A | .98841236 | .13164502 | .07343670 | .18377872 | .23817251 | .98841236 |
| Ligand L | 6.88522450 | .07395405 | .05250368 | .26256120 | .07820241 | 6.88522450 |

Public confidence max absolute/RMSE differences: pLDDT 12.1830141/.49033215
points on 0-100; pTM .0015352615/.0010934178; ipTM .0239218846/.0119017826.
This case is not structurally closed. Historical native repeat-v2 has identical
feature keys/values and all four parsed tape arrays exactly equal (archive key
ordering differs). Its native/native maxima are A .05958615, L .32299608 A.
Thus this pair's observed repeat spread is much smaller than the candidate
sample-1 residual; it is not a calibrated tolerance or a sole-cause attribution.
Kernel/chunk differences and candidate repeatability still need separation.

## Completed 1URN full-core replay

Job 988 exited 0 from the same batchfix snapshot as 3GCA. Input and prediction
hashes agree with preflight/finished records. Shared-input ordering comparison,
one whole-system fit per sample, 1225 valid atoms, no entity refit:

| Entity | Sample 1 | Sample 2 | Sample 3 | Sample 4 | Sample 5 | Max RMSD A |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Protein P | .01175127 | .01075005 | .04836122 | .01227087 | .01261881 | .04836122 |
| RNA R | .01145337 | .01140260 | .01289186 | .01321816 | .01160088 | .01321816 |

Public confidence max absolute/RMSE differences: pLDDT .60197043/.07667055
points on 0-100; pTM .0006574929/.0003809359; ipTM .0018126809/.0010509451.
Both entity coordinate maxima fall below .05 A; this does not imply confidence
acceptance. The same shared-input, backend, native-chunk, raw-logit and warm
performance exclusions as 3GCA apply. 5SAK job 989 was still running when this
comparison was recorded; no result is inferred for it.

## Completed 3GCA full-core replay

Expansion jobs 988 (1URN) and 989 (5SAK) use the identical batchfix source
snapshot and replay invocation policy, queued after collaborator job 987.
Both passed model-batch abstract tracing and augmentation-tape validation on
CPU. For all five native samples, captured atom keys match the corresponding
CIF keys exactly and forward coordinates match CIF coordinates exactly:
1URN (5,1225,3), 5SAK (5,3073,3). No candidate result from these jobs is claimed
until completion and output comparison.

Job 985 failed before JIT because native atom-array string annotations were
forwarded as graph inputs. The runner now selects the existing declared model
feature ABI and retains metadata in the original capture. Real-input abstract
tracing and 17 adapter/runner regression tests passed. Retry 986 exited 0 from
`openbind-core-batchfix-20260909-sHNQ6N`, with finite coordinates/confidence.
The input and prediction archive hashes match preflight/finished manifests.

One system Kabsch per sample, shared-input atom-order diagnostic (no entity
refit, 718 atoms each sample):

| Entity | Sample 1 | Sample 2 | Sample 3 | Sample 4 | Sample 5 | Max RMSD A |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RNA R | .01001868 | .00995650 | .00972927 | .01010704 | .00948421 | .01010704 |
| Ligand L | .00545549 | .00628567 | .00429074 | .00480334 | .00584612 | .00628567 |

Public-score differences: pLDDT max absolute .10725436 points, RMSE .01209684
after explicitly expressing the candidate 0-1 score on the native file's 0-100
scale; pTM max .0003222611, RMSE .0001489862; ipTM max .0001931143,
RMSE .0001108658. These are measured deltas, not a waived confidence gate.
Native raw logits are unavailable. Native Triton/candidate cuEq and unmatched
tuned chunk policy remain explicit deviations. Independent preprocessing,
device-consumption proof, warm performance and model admission remain open.

## Full core replay submitted

3GCA comparison preflight: all five native CIFs contain the same 718 unique
atom keys in the same order (RNA R:705, ligand L:13). Each sample's coordinates
are exactly equal to the correspondingly numbered `coordinate.npz` slice;
there is no sample permutation in this capture. The capture wrapper saves
`result[1]["atom_positions_predicted"]` after forward, not coordinates rebuilt
from CIF. Public per-atom confidence files contain pLDDT/PDE/PAE; raw confidence
logits were not saved by this wrapper. A public-score comparison must not be
presented as raw-logit parity. These are native-side identity checks only;
candidate atom ordering still requires a separate check after replay.

Job 985 runs 3GCA from `openbind-core-replay-20260909-VmneX0`, queued behind
collaborator Protenix jobs 983/984. It calls the ordinary compiled full model
with native feature rows and actual MSA/noise/augmentation tapes, without
injecting native intermediate representations. Native Triton versus candidate
cuEq is an explicit diagnostic backend deviation. The runner stores raw model
outputs; it does not claim independent preprocessing, writer parity, matched
native tuned chunks or warm performance. CPU config serialization and chemistry
table construction passed before submission. Native coordinate capture is
(5,718,3), with corresponding token/atom masks and five written CIFs available
for subsequent identity and per-entity comparison. No result is claimed yet.

## Existing forward captures recovered for replay

Core input conversion is now implemented in the bench adapter using the
existing `_max_atom_per_token_mask` builder, without weakening archive loading.
On the three real captures, its FP32 mask bytes equal outputs of pinned native
`broadcast_token_feat_to_atoms` executed on CPU (23 slots/token): 5SAK shape
(1,10051), sum 3073; 1URN (1,2714), sum 1225; 3GCA (1,1058), sum 718.
After tape preparation, each available MSA row field was gathered through the
candidate's compact union/remap and compared exactly against original captured
rows for all four cycles. All checks passed. Adapter regressions: 16 passed;
Ruff and diff checks passed. This closes these derived-input mappings only,
not independent featurization, chemistry-table identity or model replay.

Current public-route preflight found two unresolved integration boundaries:
native captures require Triton triangle kernels, while public FoldJAX triangle
attention explicitly accepts only XLA/cuEq. A CPU call with `backend="triton"`
reproduces the explicit unsupported-backend ValueError; no silent substitution
is permitted. A cuEq replay must record its backend deviation separately.
Also, `load_feature_archive` rejects all three native `input.npz` archives for
missing `max_atom_per_token_mask`. Native forward archives are not the public
FoldJAX archive ABI. Use the existing featurizer mapping with verified atom
identity and representative-atom metadata before replay; do not weaken the
public loader or fabricate the missing mask. These checks ran without GPU work.

Full adapter invocation subsequently completed for all three cases, creating
`adapted/<case>/{adapted-tape.npz,manifest.json}` under the complete snapshot.
The pinned checkout was clean at c4771653c5d0a3ebb0b3af71b05efd64bc44ee86;
each original capture's recorded checkpoint hash agrees with the rehashed
publisher checkpoint (`bd43301c011d5f87580d3e8b548658869433e4488399feb03035ba248f8e29e4`).
The manifests bind candidate source, adapter, raw capture files and adapted
draws; before/after capture digests agree. All three require the native Triton
triangle policy. These are now prepared replay artifacts, not candidate model
results. Historical provenance records are checked, not newly observed native
execution or independent preprocessing proof.

CPU `openbind_tape_adapter.load_capture` successfully validated the historical
native FP32 captures for 5SAK, 1URN and 3GCA under the precision-policy panel.
Each passed the saved/effective model configuration agreement and n5/200/four
trunk-pass checks, plus ordered draw shape/dtype/value checks. Parsed arrays:

| Case | MSA selection shape | Noise shape | Quaternion shape | Translation shape |
| --- | --- | --- | --- | --- |
| 5SAK | 4,1024 | 201,5,3073,3 | 200,5,4 | 200,5,3 |
| 1URN | 4,1024 | 201,5,1225,3 | 200,5,4 | 200,5,3 |
| 3GCA | 4,2 | 201,5,718,3 | 200,5,4 | 200,5,3 |

This recovers existing draws without new GPU capture. The 3GCA selection uses
its available shallow MSA; do not pad or relabel it as 1024 captured rows.
This check does not authenticate the complete historical capture provenance,
prove independent preprocessing, or execute a candidate replay. Those remain
required before using these archives for a model-level comparison.

## Completed native 1UBQ baseline

Job 982 exited 0. Complete snapshot `openbind-native-complete-20260909-LZea51`
contains five nonempty model CIFs, five per-sample confidence JSONs and five
aggregated confidence JSONs; all numeric confidence values were checked finite.
Saved model config records 3 recycles, 5 full-rollout samples and 200 steps.
Experiment config records trainer precision `32-true`; full confidence output
serialization is float16. These are saved settings, not tensor-level dtype
observation. Precomputed MSA and disabled templates/MSA server remain explicit
invocation choices.

| Sample | Native avg pLDDT (0-100) | pTM | ipTM |
| --- | ---: | ---: | ---: |
| 1 | 90.013420 | 0.887799 | 0 |
| 2 | 89.807442 | 0.884393 | 0 |
| 3 | 90.140915 | 0.886823 | 0 |
| 4 | 90.387543 | 0.891780 | 0 |
| 5 | 90.188057 | 0.888603 | 0 |

Native timing.json reports 5.035935 s; subprocess wall time is 17.69 s and
reported allocator peak 1746.5 MiB. None is a repeated warm benchmark. This
ordinary native baseline has no observed RNG tape and no paired FoldJAX result.
It therefore establishes execution only, not structural/confidence parity or
independent preprocessing identity. Verification: output counts, nonempty CIFs,
finite raw/aggregated confidence and saved schedule inspected after exit 0.

## Current execution recovery

Complete replacement snapshot `openbind-native-complete-20260909-LZea51`
includes src, bench, tests, pyproject and lock. CPU preflight passed case lookup,
native command construction, pinned checkout checks, runtime metadata, all
checkpoint/input/implicit-asset fingerprints and complete source fingerprints.
Job 982 submits that snapshot at n5/200/3 after the shared GPU queue was idle.
This is a native baseline, not fixed-tape parity or a warm performance claim.

Native baseline job 978 failed before inference because the external snapshot
derived the wrong benchmark data root. Job 979 set an existing data root but
selected the length-ladder manifest, which lacks `protein_1ubq`; it also failed
before inference. Neither failure is a model or numerical result. Job 981 uses
the verified seven-case manifest via `FOLDJAX_BENCH_DATA` and preserves the
same explicit native job, n5/200/3 settings and publisher runner. Failed outputs
remain intact; the retry writes a separate `native-paneldata` output directory.
Job 981 subsequently failed before inference: the bench-only snapshot lacks
`src/foldjax`, which the source-identity gate requires. No GPU inference result
exists. A complete source snapshot and CPU preflight are required before another
submission; the provenance gate must not be bypassed.

The existing `bench/openfold3_runner.yml` is an equal-workload benchmark
configuration: it sets ten recycles and disables confidence-head offload.
It must not be described as an unchanged native-default execution.

Pinned OpenFold3 v0.5.0 model configuration sets three recycles, five samples
and 200 rollout steps. The forward-tape adapter explicitly requires this
four-trunk-pass n5/200 profile. New `bench/openfold3_native_runner.yml` selects
the publisher `predict` preset and seed 101 only, without overriding model
settings. The existing `--openfold3-runner-yaml` option can select it; the
historical runner and its results remain unchanged.

This is invocation preparation, not a GPU result. Effective precision,
dynamic chunk choices, actual forward tape, independent preprocessing,
coordinates/confidence and warm memory/time still require execution and
verification. The existing forward adapter explicitly lacks preprocessing RNG
proof; its successful adaptation is not full model admission.

The caller must also record `--num-recycles 3` when using this runner;
`bench.run_upstream` otherwise inherits the equal-workload schedule of ten
and correctly rejects the mismatch. Its command builder checks the preset,
seed, recycle count and rollout steps, then passes the selected runner and
five diffusion samples to upstream without injecting a model recycle override.

Verification: eight selected OpenFold3 harness tests pass (47 unrelated tests
deselected), including the new real-file runner test, correct command wiring
and rejection of a falsely recorded ten-recycle schedule. Scoped Ruff passes.
This proves command construction, not execution-time effective settings.
