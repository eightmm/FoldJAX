# OpenDDE current-source panel expansion

Status: submitted, no new GPU result claimed.

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
