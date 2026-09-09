# cuEq-first validation decision

## Completed standard-output seven-case core panel

Jobs 1109–1115 completed. These use the same source snapshot and native tapes
as the raw-output panel, but optional raw pLDDT return is disabled. They retain
cuEq, FP32, n=5, 200 steps, four passes, native cutoff and no intermediate
injection. Values are entity maxima after one system Kabsch per sample.

| Case | Entity maximum RMSD (angstrom) | pLDDT max points | pTM max | ipTM max |
| --- | --- | ---: | ---: | ---: |
| 1UBQ | A 0.356949 | 1.997278 | 0.001632733 | 0 |
| 5SAK | A 1.109477; L 6.885722 | 11.674876 | 0.001222605 | 0.024573862 |
| 1URN | P 0.017832; R 0.014088 | 0.567263 | 0.000631307 | 0.001425865 |
| 3GCA | R 0.010177; L 0.004891 | 0.092770 | 0.000205330 | 0.000199842 |
| 7R6R | A 0.816620; B 0.612111; D 0.609748 | 1.550744 | 0.000489284 | 0.001132231 |
| 3V7E | P 0.080734; R 0.243014; L 0.095874 | 2.525440 | 0.005091651 | 0.005130889 |
| 7ST3 | A 0.210774; B 0.908877 | 6.585799 | 0.000192550 | 0.003731360 |

Verification: all seven reporter identity checks passed; all fourteen repeat
archive hashes were checked and all eight common arrays directly compared
bitwise equal across three calls per case. The last two jobs, 1114/1115,
completed for 1URN/3GCA with the same checks. Full per-sample results are the
seven `CASE-raw-off-comparison.json` files in the snapshot.

Two cases have every entity below 0.05 angstrom; five still have an entity
above 0.1. No confidence or structural gate is waived. Raw pLDDT logits are
intentionally absent here and measured separately in the diagnostic panel;
the two output policies are not bitwise interchangeable. These remain
shared-input tape controls, not independent-input end-to-end or ordinary-RNG
parity admission. Historical native public artifact bindings, device tape
consumption and matched-work performance evidence remain incomplete.

## Completed current-source seven-case raw-output core panel

This panel enables optional raw pLDDT return. It is not a measurement of the
default-output executable: the completed 3V7E output-off control below produced
different common outputs. Retain both arms rather than substituting their best
numbers into one table.

Jobs 1101–1107 all exited 0. Same source snapshot, explicit cuEq, native
FP32 policy, n=5, 200 steps, four passes, cutoff 750, actual captured tapes,
raw confidence enabled, no intermediate injection. Each row uses one proper
system alignment per sample and reports entity maxima across five samples.

| Case | Entity maximum RMSD (angstrom) | pLDDT max points | pTM max | ipTM max |
| --- | --- | ---: | ---: | ---: |
| 1UBQ | A 0.316980 | 1.494355 | 0.000156941 | 0 |
| 5SAK | A 1.080716; L 6.878858 | 10.534622 | 0.001362709 | 0.025933087 |
| 1URN | P 0.064314; R 0.014048 | 0.696871 | 0.000700058 | 0.001317001 |
| 3GCA | R 0.011771; L 0.003985 | 0.109860 | 0.000627109 | 0.000160578 |
| 7R6R | A 0.809469; B 0.607568; D 0.601382 | 2.080219 | 0.000387777 | 0.000573914 |
| 3V7E | P 0.137629; R 0.289124; L 0.318927 | 9.675431 | 0.017596968 | 0.011945508 |
| 7ST3 | A 0.190463; B 0.878583 | 8.047281 | 0.000307765 | 0.003507984 |

Verification: all seven reports passed their identity checks, all five raw
confidence fields are present per case, both repeated archive hashes per case
were checked independently, and all nine arrays were directly compared bitwise
equal across three calls. Full per-sample and raw metrics are in the snapshot's
seven `CASE-comparison.json` files. The last two jobs, 1106/1107, completed for
1URN/3GCA and received the same checks as the earlier cases.

Structural triage: one case below 0.05 angstrom, one gray, five with entities
above 0.1 angstrom. This is **completed measurement, not model closure**.
Confidence is not waived. Shared inputs, positional sample pairing, device
tape-consumption proof, native tuned chunks and historical native-public digest
limitations remain explicit. Ordinary warm job 1108 is separate and pending.

The user's latest direction is to use available cuEq implementations rather
than recreate publisher kernels merely to reduce floating-point differences.
This changes investigation priority, not the recorded acceptance gates.

## Execution policy

- Preserve upstream dtype, FP32 exceptions, inputs, sampling settings and actual
  random draws. Equal seeds alone are not a paired comparison.
- Prefer the existing cuEq path where supported; retain existing XLA paths
  elsewhere. Do not promote experimental `native-private` dispatch to default.
- Diagnose settings, masks, layout, operation semantics and RNG consumption
  before introducing custom rounding or kernel implementations.
- Keep >0.05 through 0.1 angstrom structural differences deferred, as already
  authorized. Larger differences remain investigation priorities, not automatic
  evidence that a fused kernel must be rewritten. Confidence stays separate.
- Measure ordinary warm inference separately from tape/instrumented runs.
  Backend-specific performance numbers must not be transferred to another backend.

## Current source and historical evidence

`resolve_triangle_kernel` in `src/foldjax/_openfold3_compile.py` already chooses
cuEq for serial OpenBind when neither an explicit nor environment override is
present; context parallelism defaults to XLA. No default change is needed.

The historical four-case cuEq panel in
[the OpenBind ledger](openbind-current-panel-2026-09-09.md#completed-four-case-expansion-shared-input-core-only)
used native Triton versus candidate cuEq, FP32, five samples, 200 diffusion
steps and four trunk passes with native confidence cutoff:

| Case | Entity maximum system-fit RMSD (angstrom) |
| --- | --- |
| 1UBQ | A 0.282012 |
| 7R6R | A 0.774537; B 0.584621; D 0.580117 |
| 3V7E | P 0.084586; R 0.219048; L 0.059910 |
| 7ST3 | A 0.186709; B 0.897094 |

These are historical report values, not new artifact revalidation or results
for today's source. Earlier 1URN/3GCA runs used a different confidence cutoff;
they cannot fill a uniformly configured current seven-case panel. The later
`native-private` seven-case results and warm measurements are a separate arm.

## Next comparison and exclusions

Use the existing core replay runner with explicit `--backend cueq`, no native
intermediate injection, native confidence cutoff and the captured native tape.
Bind a fresh immutable source snapshot; retain raw/public confidence and all
five entity-wise measurements after one system alignment. Prioritize 5SAK and
7ST3, then fill the remaining cases under the same policy. Independent input
identity is a separate gate; a shared feature archive does not satisfy it.

The failed native-cuEq control had a missing upstream dependency, not a
numerical result. It does not justify modifying the shared native environment
or claiming matched native/cuEq kernel evidence. Existing native Triton remains
the pinned default reference, with candidate cuEq recorded as a backend change.

Verification: current default resolver and historical ledger sections inspected;
no production implementation or default changed for this decision.
Not verified: a current-source cuEq seven-case panel, matched ordinary warm
performance, full model admission, independent review or remote publication.

## Submitted current-source controls

Snapshot `openbind-cueq-current-20260909-d2vPaq` contains copied current source
and benchmark harnesses. Queue job 1101 runs 5SAK; job 1102 runs 7ST3 serially.
Both explicitly select cuEq, n=5, 200 steps, four trunk passes, native confidence
cutoff, captured native draws, raw pLDDT logits, and three identical-tape calls.
Neither enables trunk returns, private pair operators or intermediate injection.
Preallocation is disabled. These are core parity runs, not warm benchmarks.

The existing runner records source, harness, checkpoint, input and tape hashes;
finished artifacts and comparisons must be checked before quoting results.
At submission, 1101 was running and 1102 queued; no result is inferred.
GPU process listing was empty before submission, and both known collaborator
CLI processes remained alive. Their files and processes were not modified.

Verification: replay/report CPU regression tests passed (21 tests).

Live preflight recheck of job 1101 verified all 449 recorded Python source
hashes against the isolated snapshot. Effective settings are 437 tokens,
3073 atoms, MSA depth 1024, confidence cutoff 750, n=5, 200 steps and four
passes; raw pLDDT logits are enabled and returned trunk representations empty.
Native backend is Triton and candidate backend cuEq. The preflight explicitly
retains unresolved device tape-consumption and native tuned-chunk evidence.
This verifies configuration and snapshot identity, not completed inference.

## Completed 5SAK cuEq result

Job 1101 exited 0. The existing reporter verified input/tape/prediction hashes
and replay-bound native raw-output trace. Both repeat archive hashes were
independently rechecked and all nine fields compared directly: all three calls
are bitwise identical. This excludes within-callable repeat variability for
this execution, not fresh-process or native variability.

| Entity | Sample 1 | Sample 2 | Sample 3 | Sample 4 | Sample 5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Protein A RMSD (angstrom) | 1.080716 | 0.126158 | 0.092934 | 0.351115 | 0.273013 |
| Ligand L RMSD (angstrom) | 6.878858 | 0.038070 | 0.084006 | 0.265137 | 0.085657 |

One system Kabsch fit per sample, 3073 atoms, no entity refit. Maximum public
confidence errors: pLDDT 10.534622 points (0–100), pTM 0.001362709,
ipTM 0.025933087. All five raw confidence fields are present; raw pLDDT-logit
max error is 1.469360 and RMSE 0.052364. Full raw and per-sample measurements
are in snapshot `protein_ligand_5sak-comparison.json`.

This is an unresolved large difference under the actual cuEq path. It does not
identify cuEq as the sole cause and is not a reason to waive structure or
confidence requirements. Native public coordinate/JSON files lack capture-time
digest binding; shared atom order and positional sample pairing remain explicit
limitations. Raw-output trace binding does not repair those historical gaps.

The other five native captures passed the existing tape/provenance loader on
CPU, all with cutoff 750. Jobs 1103–1107 cover 1UBQ, 7R6R, 3V7E, 1URN and
3GCA respectively, behind 7ST3 job 1102. All use the same snapshot and output
policy as 1101. Completion and numerical results remain pending for those jobs.

## Ordinary warm arm

Job 1108 queues 3GCA ordinary-RNG cuEq execution from the same snapshot after
the structural panel. The existing warm runner performs one initial call and
three synchronized warm calls, excludes loading/preprocessing/transfers/writing
from timing, and uses no tape or internal observers. It records allocator
process-lifetime peak including setup/compilation, not a reset warm-only peak.
Preallocation remains disabled. The warm harness CPU tests pass (9 tests).

This replaces no historical measurement. In particular, `native-private`
timings cannot describe cuEq performance. The earlier native 3GCA warm reference
includes full confidence/ranking work, whereas this FoldJAX runner times standard
prediction outputs; any eventual ratio must retain that scope mismatch and
cannot establish a matched-work speedup. New cuEq timings are pending.

## Completed 7ST3 cuEq result

Job 1102 exited 0. The existing reporter verified its bound inputs, tape,
prediction and raw-output trace. Both repeated prediction hashes were checked
independently, and all nine output arrays are bitwise equal across three calls.

| Entity | Sample 1 | Sample 2 | Sample 3 | Sample 4 | Sample 5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| A RMSD (angstrom) | 0.084274 | 0.027752 | 0.071582 | 0.190463 | 0.052541 |
| B RMSD (angstrom) | 0.327424 | 0.116966 | 0.572242 | 0.878583 | 0.074466 |

One system fit uses 4279 atoms per sample. Public confidence maximum errors:
pLDDT 8.047281 points (0–100), pTM 0.000307765, ipTM 0.003507984.
All five raw confidence fields are present. Full values are retained in
snapshot `protein_protein_7st3-comparison.json`.

Both entities exceed the 0.1 angstrom priority boundary. Together with 5SAK,
this shows that the current cuEq path has unresolved differences on more than
one case. It does not isolate the causal operator or authorize a default kernel
rewrite. This remains shared-input core evidence, not model admission.

## Completed 1UBQ cuEq result

Job 1103 exited 0. System-fit A RMSD for samples 1–5 is
0.316980, 0.009703, 0.077903, 0.009585, 0.009842 angstrom (601 atoms).
Public maximum errors: pLDDT 1.494355 points, pTM 0.000156941, ipTM zero.
All five raw confidence leaves are present; full report:
`protein_1ubq-comparison.json` in the same snapshot.

Verification: reporter identity checks passed; both repeat archive hashes were
checked and all nine arrays compared directly, bitwise equal across three calls.
This confirms a greater-than-0.1 angstrom sample in the single-protein case too;
it does not isolate the cause or establish independent-input parity.

## Completed 7R6R cuEq result

Job 1104 exited 0. One system alignment uses 2526 atoms per sample.

| Entity | Sample 1 | Sample 2 | Sample 3 | Sample 4 | Sample 5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| A RMSD (angstrom) | 0.232525 | 0.051473 | 0.048858 | 0.313418 | 0.809469 |
| B RMSD (angstrom) | 0.226051 | 0.037267 | 0.044772 | 0.232767 | 0.607568 |
| D RMSD (angstrom) | 0.217161 | 0.032673 | 0.041374 | 0.227082 | 0.601382 |

Public maximum errors: pLDDT 2.080219 points, pTM 0.000387777,
ipTM 0.000573914. All five raw confidence leaves are present. Full report is
`protein_dna_7r6r-comparison.json` in the snapshot.

Verification: reporter identity checks passed, both repeat hashes verified,
and all nine arrays directly compared bitwise equal across the three calls.
All three entity maxima exceed 0.1 angstrom; no gate is waived. Shared-input
and historical native-public provenance limitations remain unchanged.

## Completed 3V7E cuEq result

Job 1105 exited 0. One system fit uses 3315 atoms per sample.

| Entity | Sample 1 | Sample 2 | Sample 3 | Sample 4 | Sample 5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| P RMSD (angstrom) | 0.085414 | 0.094876 | 0.052935 | 0.137629 | 0.099230 |
| R RMSD (angstrom) | 0.230572 | 0.235132 | 0.075459 | 0.289124 | 0.151511 |
| L RMSD (angstrom) | 0.061566 | 0.114551 | 0.018174 | 0.318927 | 0.073070 |

Public maximum errors: pLDDT 9.675431 points (RMSE 1.711751),
pTM 0.017596968, ipTM 0.011945508. All five raw confidence leaves are present.
Full report: `protein_rna_ligand_3v7e-comparison.json` in the snapshot.

Verification: reporter identity checks passed; both repeat hashes verified,
all nine arrays directly compared bitwise equal across three calls. Historical
cuEq numbers differ, but source/output policy/compiler choices were not frozen
across those experiments. This is not an isolated regression attribution.
Structure and confidence remain unresolved; no kernel or tolerance changed.

## Completed cuEq ordinary warm measurement

Job 1108 exited 0 on 3GCA, using the same isolated source and explicit cuEq.
First call including compilation: 81.983673 s. Synchronized ordinary-RNG warm
calls: 1.116917, 1.116228, 1.116391 s; median **1.116391 s**.
Allocator process-lifetime peak bytes in use: **1,776,049,664 bytes** (about
1.654 GiB), including setup/compile, not reset warm-only memory. Preallocation
was disabled. Inputs and weights were resident before timing; preprocessing,
loading, host transfer and serialization were excluded. No tape or internal
observers were enabled, and standard output policy was used.

Verification: all four public output archive hashes checked and identical.
The runner's source/input/checkpoint/environment postflight guard completed.
This is one case, not a seven-case performance panel. Historical native warm
timing has a different confidence/ranking scope, so no matched-work speedup or
memory reduction is admitted. Output identity in this ordinary JAX run is not
cross-framework RNG identity or structural parity evidence.

## Next discriminating control: 3V7E optional output

Observation: current cuEq 3V7E pLDDT max error is 9.675431 points, versus
1.407033 in the historical four-case report. Those executions do not isolate
a source regression. Candidate explanations include output-policy/compiler
effects, fresh-process autotuning differences, and intervening implementation
changes. Same-callable repeats already exclude within-callable variability
for the current run only.

Source inspection shows the pLDDT logits are computed for public pLDDT either
way; `return_plddt_logits` selects only whether they are returned. This is not
evidence that the compiled graphs must produce identical values.

Job 1109 reuses the exact current snapshot, native capture, cuEq and tape,
but disables the optional raw pLDDT return. It changes no model arithmetic.
Three calls retain repeat checks. Fresh-process compiler autotuning remains
uncontrolled, so an observed difference would identify a confounded
output-policy/compiler control, not prove the flag alone caused it. If the
large discrepancy persists, optional raw return alone is insufficient to
explain the current problem. Default-output parity deserves a direct measure
regardless of this diagnostic outcome. No result is claimed before completion.

### Completed output-off control

Job 1109 exited 0. Both preflight documents differ only in
`config.return_plddt_logits` (true versus false). Fresh compiler autotuning
was not frozen. The default-output arm has entity maxima P **0.080734**,
R **0.243014**, L **0.095874** angstrom, versus raw-on
0.137629 / 0.289124 / 0.318927. Public maximum errors decrease to pLDDT
**2.525440** points, pTM **0.005091651**, ipTM **0.005130889**.

All eight common arrays differ between the two arms; each arm individually
repeats bitwise. Thus same-callable repeatability does not establish agreement
between separately compiled output policies. The seven-case raw-on panel must
not be presented as the standard-output executable's exact numerical result.
The RNA discrepancy persists above 0.1 angstrom with raw return disabled, so
that option alone does not remove the upstream difference. No sole-cause
attribution to the option or compiler algorithm is established.

Verification: reporter identity checks, both repeat hashes, all eight repeat
arrays, and direct on/off common-field comparisons completed. Full default-arm
report is `protein_rna_ligand_3v7e-raw-off-comparison.json`. Raw pLDDT logits
are intentionally absent from that arm; no missing-field gate was waived.

## Standard-output panel completion queued

Jobs 1110–1115 cover the remaining default-output cases in this order: 5SAK,
7ST3, 1UBQ, 7R6R, 1URN, 3GCA. They reuse snapshot
`openbind-cueq-current-20260909-d2vPaq`, the same native captures and actual
tapes, cuEq, n=5, 200 steps, four passes, native cutoff and three calls.
Only the optional raw pLDDT return is disabled relative to the raw-output panel;
no model arithmetic, dtype, intermediate injection or truncation is introduced.
The completed 3V7E default arm is job 1109, not rerun.

No GPU compute process was present before submission and both known
collaborator CLI processes remained alive. Queue serialization is retained.
Fresh compiler/autotuning is still a limitation for causal on/off claims.
Successful completion and per-case comparison are pending for 1110–1115.

### 5SAK standard output completed

Job 1110 exited 0. Default-output entity maxima: A **1.109477** angstrom,
L **6.885722** angstrom. Per-sample A values are 1.109477, 0.124292,
0.096963, 0.305751, 0.251854; L values are 6.885722, 0.067536,
0.022804, 0.273083, 0.079488. Public maximum errors: pLDDT **11.674876**
points, pTM **0.001222605**, ipTM **0.024573862**.

The large sample-1 ligand discrepancy persists with optional raw pLDDT return
disabled; the raw-on panel was not its sole explanation. This does not identify
the causal operator. Verification: reporter binding checks passed, both repeat
hashes verified, all eight outputs directly compared bitwise equal over three
calls. Full report: `protein_ligand_5sak-raw-off-comparison.json`.

### 7ST3 standard output completed

Job 1111 exited 0. Default-output entity maxima are A **0.210774** and
B **0.908877** angstrom. Per-sample A: 0.075453, 0.028302, 0.088979,
0.210774, 0.042639; B: 0.286846, 0.121312, 0.524312, 0.908877, 0.071425.
Public maximum errors: pLDDT **6.585799** points, pTM **0.000192550**,
ipTM **0.003731360**. The large B discrepancy persists at default output.

Verification: reporter identity checks, both repeated archive hashes and all
eight repeated output arrays passed (three calls bitwise equal). Comparing
preflight documents for both 5SAK and 7ST3 against their raw-on arms shows only
`return_plddt_logits` differs; fresh autotuning remains uncontrolled. Full report:
`protein_protein_7st3-raw-off-comparison.json` in the same snapshot.

### 1UBQ standard output completed

Job 1112 exited 0. A RMSD for samples 1–5 is 0.356949, 0.010661,
0.078313, 0.009625, 0.011448 angstrom. Public maximum errors:
pLDDT **1.997278** points, pTM **0.001632733**, ipTM zero.
The sample-1 discrepancy persists under standard output policy.

Verification: reporter identity checks passed, both repeat hashes verified,
all eight output arrays directly compared bitwise equal across three calls.
Full report: `protein_1ubq-raw-off-comparison.json`. No acceptance criterion
changed; standard-output 7R6R, 1URN and 3GCA results remain pending.

### 7R6R standard output completed

Job 1113 exited 0. Entity maxima: A **0.816620**, B **0.612111**,
D **0.609748** angstrom. Per-sample A: 0.238006, 0.048299, 0.040269,
0.350743, 0.816620; B: 0.235110, 0.031749, 0.030761, 0.257946, 0.612111;
D: 0.223890, 0.028276, 0.029927, 0.252704, 0.609748.
Public maximum errors: pLDDT **1.550744** points, pTM **0.000489284**,
ipTM **0.001132231**. Large coordinate differences persist at standard output.

Verification: reporter identity checks, both repeat hashes and direct comparison
of all eight repeated arrays passed; three calls bitwise equal. Preflight
differs from the raw-on arm only in optional pLDDT return, with the previously
stated fresh-autotuning limitation. Full report:
`protein_dna_7r6r-raw-off-comparison.json`. 1URN and 3GCA remain pending.

## Native cuEq environment recheck

Read-only distribution metadata confirms the pinned native interpreter has
Torch 2.12.1+cu130 but no cuequivariance, cuequivariance-torch, or Torch cu12/cu13
operator distribution. The existing parity interpreter has JAX cuEq packages
only. `python -m pip show` was unavailable because native pip is absent;
`importlib.metadata` provided the package check without changing environments.

The upstream pyproject declares a cuEq extra with cu12 Torch operators, while
the current native Torch is cu130. Installing that extra blindly is not an
established compatible environment. No packages were installed. A separate
development environment is required for a controlled native-cuEq comparison;
the shared native environment and runtime package must remain untouched.

Source also has explicit native cuEq attention fallbacks by token length,
hidden width and dtype. Therefore a cuEq flag alone cannot prove that the fused
kernel executed; any future capture must record effective dispatch. These
findings explain why the previous native-cuEq attempt produced no result, not
the current numerical residual. Native Triton remains the default reference.

## Independent preprocessing recheck after both panels

The current worktree's complete
`tests/models/openfold3/test_torch_parity_preprocessing.py` was rerun on CPU
in the existing parity environment with all seven panel inputs explicitly
configured and the native source explicitly pinned to
`c4771653c5d0a3ebb0b3af71b05efd64bc44ee86`.

Verification: **11 passed, 7 warnings in 31.88 s**. All seven parametrized panel
cases ran; none skipped. Warnings were the expected query-only dummy-MSA notices.
The native checkout remained clean. No dependencies or model code changed.
The interrupted conversation occurred after pytest had exited successfully;
the test was not restarted.

These checks separately rebuild native and candidate inputs and compare
features, atom metadata, bonds, masks and replayed augmentation draws within
the scope described in `openbind-independent-input-identity-2026-09-09.md`.
They do not connect those fresh preprocessing draws to historical model tapes,
prove every possible input, or establish that a particular fused operator
caused the residual. Existing end-to-end and device-consumption gaps remain.
