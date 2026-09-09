# OpenDDE current-source panel expansion

## Highest precision artifacts: independent output recheck

Completed collaborator jobs 933/938/939 were read-only rechecked with the
existing closure reporter against original native captures. These use the
`highest` directories under `opendde-current-panel-20260909-Nf5tvz`.

| Case | Entity maximum RMSD (angstrom) | Strict confidence |
| --- | --- | --- |
| 1URN | P 0.05396252; R 0.00254606 | fail |
| 7R6R | A 0.03115811; B 0.02318989; D 0.02212658 | fail |
| 7ST3 | A 0.02342866; B 0.01387774 | fail |

All three load successfully through the independent input gate; sampler and
MSA tape archive comparisons are exact. 7R6R/7ST3 are below 0.05 A for all
entities; 1URN protein remains gray. In particular, 7ST3's earlier high-path
B residual around 0.118 A is not present in this highest artifact. This is not
a universal precision recommendation: native OpenDDE's observed forward TF32
policy differs from Protenix, and each model retains its own policy evidence.
No collaborator source or jobs were changed. Full policy provenance, repeated
highest variability, device consumption and performance admission are not
established by this read-only output recheck. Strict confidence failures remain.

Verification: reporter completed without errors on all three pairs; n5 entity
RMSD and exact tape comparisons inspected. No default or release claim changed.

7ST3 provenance was additionally compared between unobserved/high and highest:
the only changed recorded field is `jax_matmul_precision`. Source, checkpoint,
runtime versions, native capture, input assets and observer policy match.
Highest maximum confidence errors are atom pLDDT 0.00052344799 on 0–1 scale
(0.052344799 on 0–100), pTM 0.00001400709 and ipTM 0.00000464916.
pTM/ipTM pass their frozen leaf gates; atom pLDDT does not. This distinguishes
the remaining strict confidence failure from a large score discrepancy without
waiving that gate. The single-variable recorded comparison supports the
precision-policy explanation for these artifacts, not repeatability proof.

Status: seven unobserved cases completed; strict confidence remains failed.
The dated progression below retains earlier results and pending-state history.

## 7ST3 repeat: residual reproduced

Job 916 exited 0. Both comparison reports have no schema error.

| Comparison | A maximum RMSD (angstrom) | B maximum RMSD (angstrom) |
| --- | ---: | ---: |
| Native vs repeated candidate | 0.05418689036 | 0.11823763007 |
| First vs repeated candidate | 0.00554700324 | 0.00508927670 |

The native/B residual again occurs in sample 2. Candidate repeat variability
in these two runs is much smaller than the native/candidate residual, so it
does not explain that residual alone. Native execution variability and a
reproducible arithmetic/path difference remain unseparated; native repeat
control is still needed before assigning a port defect. No thresholds changed.

Strict confidence fails in both comparisons. Native/repeat maximum atom pLDDT
delta is 0.00224280357 (0–1), pTM 0.00001937151, ipTM 0.00012922287.
Candidate/repeat atom pLDDT delta is 0.00033330917, pTM 0.00000458956,
ipTM 0.00000232458. Within-route confidence failure is retained explicitly.

Verification: completed `repeat-native-report.json` and
`unobserved-repeat-comparison.json` inspected; structure fails native/repeat
and passes candidate/repeat. No actual consumer or warm-performance claim.

Native repeat control 921 was subsequently submitted with the same raw 7ST3
input, seed101/n5/200-step/10-cycle profile, native FP32 TF32-on policy and
legacy capture driver, into `protein_protein_7st3/native-repeat` under the
current panel snapshot. Its captured sampler and MSA arrays must match the
original native arrays exactly before the output difference can be treated as
a fixed-tape native repeat. Seed equality alone is insufficient. At the time
of this entry the process is live; no native repeat result is claimed.

### Native repeat 921 completed

Native input arrays, sampler tape and MSA tape are all bitwise equal to the
original native capture. Checkpoint and legacy-driver hashes match, as do the
recorded forward-entry policies: deterministic algorithms false, cuDNN TF32
true, **matmul TF32 false**. Earlier "TF32-on" shorthand describes the requested
native flag, not all executed operators; the observed entry policy takes
precedence. No policy override was introduced for this repeat.

Native/native maximum system-fit entity RMSDs are A 0.00512907082 and
B 0.00552693211 angstrom. Strict confidence fails even within native:
atom pLDDT maximum delta 0.00038939714 (0–1), pTM 0.00000315905 and
ipTM 0.00000530481. The original/repeated FoldJAX B residuals of 0.118108/
0.118238 angstrom are much larger than the observed approximately 0.0055
native repeat and 0.0051 candidate repeat variations. This supports a
reproducible cross-route difference, not attribution to ordinary repeat
variation alone. It does not yet locate the responsible operator or rule out
rare variation; two runs are not a frozen repeat calibration.

Verification: 921 exited 0; `native-repeat-comparison.json` has no schema
error, exact input/tape checks pass and the structural gate passes. Strict
confidence fails and is retained; no relaxed tolerance or full admission.

## Completed unobserved panel summary

All seven independent input gates pass. These are native FP32, n5 tape
replays with whole-system alignment and entity-only measurement, not crystal
comparisons or warm benchmarks. Maxima are over five samples in angstrom.

| Case | Entity maximum RMSD | Structural triage |
| --- | --- | --- |
| 1UBQ | A 0.019998 | below 0.05 |
| 5SAK | A 0.054387; L 0.020908 | gray |
| 1URN | P 0.050100; R 0.004782 | gray; original run retained |
| 3GCA | R 0.003103; L 0.002747 | below 0.05 |
| 7R6R | A 0.063369; B 0.044255; D 0.042753 | gray |
| 3V7E | P 0.012840; R 0.010848; L 0.007256 | below 0.05 |
| 7ST3 | A 0.053511; B 0.118108 | above 0.1; investigate |

Every case fails strict whole-confidence comparison. Actual consumer
observation covers only 1URN/3GCA, whose strict observer bridges still fail;
copied tapes in the other cases do not establish actual consumption equality.

7ST3 job 913 exited 0. Chain A per-sample RMSDs are
`[0.00971206,0.00749004,0.00875781,0.05351113,0.01905790]`; B is
`[0.00827061,0.11810792,0.00986160,0.02764890,0.01363757]`.
Maximum atom pLDDT delta is 0.00224793 (0–1), full PAE 0.25246763
angstrom, pTM 0.00001811981 and ipTM 0.00012689829.

The above-0.1 residual could be reproducible port arithmetic or execution
variation; a single capture cannot distinguish them. Job 916 repeats 7ST3
without observers, using the identical immutable source and tape reference,
into a separate `unobserved-repeat` directory. No model math changed.
It must be compared both against the first candidate and native before a
cause is assigned. This is not a threshold relaxation or full admission.

Verification: all seven completed `unobserved-report.json` files and input
gates were inspected. 913 has no report schema error; strict parity fails.
916 was submitted through the single-slot GPU queue; its result is pending.

The completed AF3 finite panel does not establish any OpenDDE result. This
next batch uses a fresh source snapshot and the existing pinned native FP32
captures for 1URN and 3GCA. Each candidate independently featurizes the raw
input, audits native input reconciliation, indexes its own MSA using the
native row tape, and replays five samples, 200 steps and ten recycles.

| Case | Consumed-tape observer | Unobserved replay |
| --- | ---: | ---: |
| Protein-RNA 1URN | 903 | 904 |
| RNA-ligand 3GCA | 905 | 906 |

The observer checks actual FP32 sampler values and consumer-normalized MSA
values bytewise at execution. Its scope is unpadded, unbatched, scanned n5
inference. Unsupported/cached paths must fail through missing events. Compare
the unobserved replay separately to native and to the observed candidate;
callback instrumentation changes the compiled graph. These are not warm
performance measurements or a claim that earlier confidence failures pass.

Opus's existing Boltz jobs 901/902 retain their queue order. No Boltz runtime
or collaborator scratch files were modified. Four new jobs use the same
single-slot queue and separate outputs; native captures are read-only.

## CPU verification while the GPU queue is occupied

The current working tree passed 345 tests in 97.61 seconds with
`JAX_PLATFORMS=cpu .venv-ci/bin/python -m pytest -q tests/models/opendde
tests/test_opendde_closure_capture.py tests/test_opendde_consumed_tape.py
tests/test_opendde_closure_report.py tests/test_opendde_repeat_forward.py`.
Two warnings report requested int64 truncation to int32 with JAX x64 disabled,
in compact-category and trunk tests; there were no failures or skips.
This covers model/runtime and diagnostic contracts, not the queued real-weight
GPU parity results or warm time/VRAM. The earlier focused capture/consumer/
report run passed 50 tests.

Source inspection confirms the output report remains an instrumented output
diagnostic: copied tape archive equality is not actual consumption proof.
The separate `consumed-tape-audit.json` and observed/unobserved comparison must
be inspected in addition to that report; its exclusions are not waived merely
because the optional observer was requested.

## 1URN observed replay completed

Job 903 exited 0. Result root:
`opendde-current-panel-20260909-Nf5tvz/protein_rna_1urn/observed`.
`consumed-tape-audit.json` reports 211 observed events, zero missing events,
zero errors and finite bitwise equality. This verifies the explicitly mapped
sampler and MSA consumer values, not structural parity.

`observed-report.json` has no schema/profile error but fails the 0.05 angstrom
coordinate diagnostic and strict confidence checks:

| Entity | Per-sample system-fit RMSD (angstrom) | Maximum |
| --- | --- | ---: |
| Protein P | 0.00295991, 0.00311552, 0.14834686, 0.00503147, 0.05067952 | 0.14834686 |
| RNA R | 0.00398926, 0.00413173, 0.00482780, 0.00351247, 0.00472836 | 0.00482780 |

Maximum full atom pLDDT difference is 0.00024086237 on the 0–1 scale and
fails its strict leaf test. Summary pTM and ipTM maxima are 0.00000375509
and 0.00003826618 and pass their leaf tests. The protein maximum exceeds the
user's deferred 0.05–0.1 angstrom gray zone and requires investigation.

The consumed-tape result argues against missing/misrouted recorded draws at
the observed boundaries. Remaining hypotheses are observer/compiler effects,
execution repeat variability and a port arithmetic difference. No cause is
isolated yet. The cheapest next discriminator is already-running job 904:
compare its unobserved output both to native and to 903 before model edits.
If the discrepancy remains, native/candidate repeat controls are needed to
separate variability from a reproducible port difference. No tolerance change
or general OpenDDE admission is made.

Verification: GPU 903 exited 0, consumer audit passed, output report was read
through completion and its failures retained. Job 904 was confirmed running.

## 1URN unobserved comparison

904 exited 0. `unobserved-report.json` compares native with this unobserved
candidate; `observer-bridge.json` compares 903 with 904. Neither strict
coordinate/confidence diagnostic passes, and neither has a schema error.

| Comparison | Protein maximum RMSD (angstrom) | RNA maximum RMSD (angstrom) |
| --- | ---: | ---: |
| Native vs unobserved | 0.05009963604 | 0.00478151606 |
| Observed vs unobserved | 0.14530592200 | 0.00343132297 |

Native/unobserved protein per-sample values are `[0.00300536550,
0.00349051104,0.00771164283,0.00330603704,0.05009963604]`. The large third
sample seen in 903 is absent. Native/unobserved atom pLDDT maximum delta is
0.00021618605 (0–1); pTM 0.00000381470 and ipTM 0.00002545118.
Strict confidence remains false. The unobserved structural maximum is in the
user's deferred gray zone, not a coordinate pass at the original 0.05 gate.

This refutes treating the observed 0.1483 angstrom as an established invariant
port error, but does not isolate callbacks from execution-repeat variability.
A second unobserved run was submitted with the identical snapshot, settings
and tape reference to distinguish those hypotheses. No model math changed.
The consumed-tape observer still needs a satisfactory unobserved bridge before
admission; no failed bridge is waived.

Verification: 904 exited 0 and both completed reports were inspected.

## 3GCA observed replay completed

905 exited 0. Root:
`opendde-current-panel-20260909-Nf5tvz/rna_ligand_3gca/observed`.
Actual consumed-tape audit passes all 211 events with no missing events or
errors. `observed-report.json` has no schema error and passes the original
0.05 angstrom coordinate gate, but fails strict confidence.

| Entity | Per-sample system-fit RMSD (angstrom) | Maximum |
| --- | --- | ---: |
| RNA R | 0.00174513, 0.00177497, 0.00165957, 0.00219326, 0.00178756 | 0.00219326 |
| Ligand L | 0.00197786, 0.00149441, 0.00077015, 0.00164944, 0.00139472 | 0.00197786 |

Maximum absolute differences: atom pLDDT 0.00004953146 (0–1), summary pTM
0.00001180172 and ipTM 0.00000774860. Failed confidence leaves are full
token-pair PAE/PDE, raw contact probabilities and raw PAE/PDE/pLDDT/resolved.
Small public summary differences do not hide raw-head failures.

Verification: GPU 905 exited 0, consumer audit and completed output report
were inspected. Unobserved job 906 is running; observer bridge and full model
admission remain unverified. No warm-performance claim follows from this run.

## 3GCA unobserved replay completed

906 exited 0. Both completed comparison reports have no schema error and
pass the original structural diagnostic, but fail strict confidence:

| Comparison | RNA maximum RMSD (angstrom) | Ligand maximum RMSD (angstrom) |
| --- | ---: | ---: |
| Native vs unobserved | 0.00310302373 | 0.00274746513 |
| Observed vs unobserved | 0.00249096403 | 0.00125669696 |

Native/unobserved maximum absolute public deltas: atom pLDDT 0.00006312132
(0–1), pTM 0.00001144409 and ipTM 0.00001251698. Observed/unobserved deltas
are 0.00005781651, 0.00000423193 and 0.00000476837 respectively. Neither
small public differences nor passing coordinate gates override the failed
raw/full confidence checks. The consumed-tape and structure evidence is
positive, but a strict observer bridge and full closure are not established.

Verification: 906 exited 0; `unobserved-report.json` and
`observer-bridge.json` were read through completion. The 1URN repeat job 908
remains queued behind the collaborator's running Boltz repeat job 907.

## Unobserved confidence residual inventory

Maximum absolute differences from the completed unobserved native comparisons:

| Confidence field | 1URN | 3GCA |
| --- | ---: | ---: |
| Full atom pLDDT (0–1) | 0.00021618605 | 0.00006312132 |
| Full token-pair PAE (angstrom) | 0.01879310608 | 0.01154899597 |
| Full token-pair PDE (angstrom) | 0.00511550903 | 0.00183379650 |
| Raw contact probability | 0.00167763233 | 0.00111889839 |
| Raw PAE output | 0.04170799255 | 0.02005386353 |
| Raw PDE output | 0.01567363739 | 0.00598502159 |
| Raw pLDDT output | 0.00347280502 | 0.00354814529 |
| Raw resolved output | 0.00278687477 | 0.00296998024 |

Raw-head output differences are not public angstrom errors or pLDDT points;
retain the separate output contracts. All rows except 3GCA full atom pLDDT
fail their strict leaf diagnostic. This is a magnitude inventory, not a new
acceptance threshold, causal diagnosis or native-repeat calibration. Existing
summary pTM/ipTM passes do not establish whole-confidence parity.

Verification: values were read directly from both completed
`unobserved-report.json` confidence leaf maps. The pending 1URN repeat is
still required to distinguish reproducible residuals from execution variation.

## Remaining unobserved panel submitted

Native completion files and raw candidate input files were checked for 1UBQ,
5SAK, 7R6R, 3V7E and 7ST3. These five unobserved replays were submitted using
the same immutable source root `opendde-current-panel-20260909-Nf5tvz` and
FP32/n5/200-step/10-cycle native tape settings as 903–906. Outputs are separate
per-case `unobserved` directories. Earlier measurements are not overwritten.

This expands structural/confidence coverage, not actual consumer-observer
coverage or performance admission. Each completed case still needs its input
audit, native output comparison and provenance checked. Failed cases remain
failures; queue submission is not scientific success. Existing collaborator
jobs and the pending 1URN repeat retain their place ahead of this batch.

## 1URN unobserved repeat completed

908 exited 0 using the same source/settings/tape as 904. Reports:
`unobserved-repeat-comparison.json` (904 vs 908) and
`repeat-native-report.json` (native vs 908).

| Comparison | Protein max RMSD (angstrom) | RNA max RMSD (angstrom) | Strict confidence |
| --- | ---: | ---: | --- |
| Unobserved vs unobserved repeat | 0.00791440683 | 0.00347503086 | fail |
| Native vs unobserved repeat | 0.04691825559 | 0.00470491907 | fail |

Both comparisons pass the original structural gate. Native/repeat protein
per-sample RMSDs are `[0.00281581976,0.00360143797,0.00476977587,
0.00840670237,0.04691825559]`. Maximum atom pLDDT differences (0–1) are
0.00026392937 between unobserved repeats and 0.00028389692 native/repeat.
Thus the strict confidence test also fails within the same FoldJAX route,
not only between frameworks. It cannot by itself identify a port defect.

The observed run's 0.1483 angstrom deviation did not recur in either
unobserved run; ordinary unobserved repeat variation is much smaller here.
This strengthens, but does not conclusively isolate, the observer/compiler
hypothesis. Rare execution variability remains possible. Two repeats do not
constitute native-repeat calibration or justify changing thresholds. Preserve
the original 904 gray-zone result as well as 908's coordinate pass; do not
select the better run as a replacement. No model arithmetic was changed.

Verification: 908 exited 0, both completed reports inspected, no schema errors.
The outstanding observer bridge and strict confidence failures remain.

## 1UBQ current-source unobserved result

909 exited 0. Root:
`opendde-current-panel-20260909-Nf5tvz/protein_1ubq/unobserved`.
Independent input gate passes. Native comparison has no schema error and
passes the 0.05 angstrom structural diagnostic: protein A per-sample RMSDs
are `[0.00219691228,0.01999775624,0.00212992739,0.00217622429,
0.00237044639]`, maximum 0.01999775624 angstrom.

Strict confidence fails. Maximum public deltas are atom pLDDT 0.00029563904
(0–1), token-pair PAE 0.01656055450 angstrom and pTM 0.00000256300;
ipTM delta is zero. Passing or tiny summary scores do not override the full
confidence gate. Actual tape consumption was not observed in this run.

Verification: 909 exited 0; independent input audit and completed
`unobserved-report.json` inspected. This adds a third current unobserved case
to 1URN and 3GCA; remaining jobs 910–913 are running/queued.

## 5SAK current-source unobserved result

910 exited 0. Root:
`opendde-current-panel-20260909-Nf5tvz/protein_ligand_5sak/unobserved`.
Independent input gate passes; the report has no schema error but fails
the strict coordinate and confidence diagnostics.

| Entity | Per-sample system-fit RMSD (angstrom) | Maximum |
| --- | --- | ---: |
| Protein A | 0.02967468, 0.05438664, 0.04948935, 0.02249928, 0.03482242 | 0.05438664 |
| Ligand L | 0.00614845, 0.00863408, 0.02090792, 0.00594415, 0.01121110 | 0.02090792 |

The protein maximum is in the user's deferred structural gray zone, not a
pass at the frozen 0.05 threshold. Maximum public confidence deltas are
atom pLDDT 0.00201267004 (0–1), token-pair PAE 0.29221630096 angstrom,
pTM 0.00026673079 and ipTM 0.00071674585. Confidence and structure remain
separate gates. This observation does not explain or resolve Boltz 5SAK.

Verification: job 910 exited 0, independent input audit and completed output
report inspected. No actual consumer observer or warm-performance evidence
is claimed for this arm; earlier 5SAK measurements remain preserved.

## 7R6R current-source unobserved result

911 exited 0. Root:
`opendde-current-panel-20260909-Nf5tvz/protein_dna_7r6r/unobserved`.
Independent input gate passes; native comparison has no schema error.

| Entity | Per-sample system-fit RMSD (angstrom) | Maximum |
| --- | --- | ---: |
| Protein A | 0.02231075, 0.01744387, 0.01081217, 0.05173694, 0.06336865 | 0.06336865 |
| DNA B | 0.01816206, 0.01294553, 0.01002201, 0.03481784, 0.04425548 | 0.04425548 |
| DNA D | 0.01438456, 0.01121942, 0.00818963, 0.03238572, 0.04275253 | 0.04275253 |

Protein A lies in the user's deferred structural gray zone; the original
0.05 angstrom gate fails. Strict confidence also fails. Maximum public deltas:
atom pLDDT 0.00104165077 (0–1), PAE 0.08931350708 angstrom, pTM
0.00019663572 and ipTM 0.00006717443. Entity chemistry was checked against
the executed raw input, not inferred from chain letters.

Verification: 911 exited 0, input audit and completed comparison report
inspected. Five current unobserved cases are measured; 3V7E and 7ST3 remain.
Actual consumer observation and warm performance remain separate gaps.

## 3V7E current-source unobserved result

912 exited 0. Root:
`opendde-current-panel-20260909-Nf5tvz/protein_rna_ligand_3v7e/unobserved`.
Independent input gate passes. Native comparison has no schema error and
passes the original coordinate diagnostic, but strict confidence fails.

| Entity | Per-sample system-fit RMSD (angstrom) | Maximum |
| --- | --- | ---: |
| Protein P | 0.01137438, 0.00982121, 0.01284016, 0.01054897, 0.01003654 | 0.01284016 |
| RNA R | 0.00928042, 0.00930597, 0.01084820, 0.00979656, 0.00817935 | 0.01084820 |
| Ligand L | 0.00581525, 0.00630852, 0.00626292, 0.00725579, 0.00420434 | 0.00725579 |

Maximum public confidence differences: atom pLDDT 0.00055587292 (0–1),
token-pair PAE 0.06482267380 angstrom, pTM 0.00008696318 and ipTM
0.00003874302. These are reported independently of the structural pass.

Verification: GPU 912 exited 0; independent input audit and completed native
comparison inspected. Current unobserved panel coverage is 6/7, with 7ST3
remaining. No actual consumer observation or warm measurement is claimed here.
