# Native-first precision screening results

Status: native precision screening completed on a three-complex panel for
OpenDDE released and OpenFold3 v0.5/OpenBind. This is **not port acceptance**.
The prospective [protocol](precision-selection-protocol-2026-09-06.md) uses
0.05 Å for every entity-instance/sample after a single global proper Kabsch,
with five sample-index-paired predictions. No experimental coordinates.

Both native runs construct their own inputs. Captured input values/dtypes,
actual sampler draws and MSA choices must agree exactly before coordinates
are compared. Equal seed 101 alone is not sufficient; all six paired controls
passed this realized-input/tape prerequisite. Native TF32 remains enabled in
both arms. Checkpoints, schedules and kernel configuration are unchanged.
Native automatic kernel/chunk choices are retained, not presumed bit-identical.

| Model / complex | Protein max Å | RNA max Å | Ligand max Å | All entities <=0.05 Å |
|---|---:|---:|---:|---|
| OpenDDE / 5SAK | 0.579884 | — | 7.077839 | No |
| OpenDDE / 1URN | 0.201868 | 0.049336 | — | No |
| OpenDDE / 3GCA | — | 0.018522 | 0.045440 | Yes |
| OpenFold3 / 5SAK | 2.689832 | — | 11.030340 | No |
| OpenFold3 / 1URN | 0.102832 | 0.032917 | — | No |
| OpenFold3 / 3GCA | — | 0.016868 | 0.016579 | Yes |

OpenDDE uses native `fp32` versus `bf16`, n=5, 200 steps, 10 cycles.
OpenFold3 uses publisher Lightning `32-true` versus `bf16-mixed`, n=5,
200 steps and 3 configured recycles (4 passes). The actual predict preset,
augmentation and MSA sampling are retained; unlike the historical harness,
augmentation is not replaced by identity. Prepared local MSAs and disabled
remote search/templates keep network retrieval outside this comparison.

## Same-mode repeat floor

5SAK native FP32 repeated under the same exact realized input/tape:

| Model | Protein max Å | Ligand max Å | Coordinate gate |
|---|---:|---:|---|
| OpenDDE | 0.008524 | 0.002829 | Pass |
| OpenFold3 | 0.059586 | 0.322996 | Fail |

OpenDDE's precision change is much larger than this measured repeat residual.
OpenFold3's BF16 change is also larger, but its **native FP32 repeat already
exceeds the gate**. Preserve this failure rather than loosening the threshold.
Its default runtime reproducibility, including kernel/chunk autotuning, needs
separate isolation. These maxima are not additive causal decompositions.

## Precision decisions and boundaries

- OpenDDE released: do not admit optional BF16 on this panel; the comparison
  target remains native FP32/TF32-on. ABAG is a distinct checkpoint and is not
  covered by these measurements.
- OpenFold3 v0.5/OpenBind: do not admit optional BF16; retain native `32-true`
  as the target, with the failed repeat floor explicitly unresolved.
- ESMFold2: correction to earlier classification. Its checkpoint has FP32
  weights, but pinned CUDA forward explicitly enables BF16 autocast for trunk,
  selected conditioning and confidence operations. It belongs with native
  mixed-precision targets, not the optional-FP32-to-BF16 screening group.
- AlphaFold3, Boltz-2, ESMFold2 and Protenix must reproduce their native mixed
  operator policies. This run does not close their FoldJAX full-tape output
  gates. ESMFold2's core still lacks full tape injection.

No shipped precision defaults change in this testing turn. In particular,
FoldJAX's current OpenDDE BF16 default is **not approved by these results**;
runtime default alignment and a fresh FP32/TF32-on port gate remain follow-up
work. A native precision screen is never substituted for that port gate.

## Evidence and verification

[Portable per-sample records and hashes](../bench/experiments/native-precision-selection-2026-09-06.json)
contain six paired controls and two repeats (14 successful native predictions,
five samples each). Native source identity remained unchanged. Artifact hashes
were rechecked when collecting the report. Failed instrumentation attempts are
retained: initial AtomArray serialization and repeat-wrapper forkserver entry;
neither is counted as a model precision failure or successful measurement.

The ESMFold2 metadata now distinguishes FP32 checkpoint dtype from native
CUDA autocast. Fixed coordinate-gate tests reject missing samples, nonfinite
values, negative values, hidden entity-label collisions and any failing entity.
Focused gate/capture/entity/kernel tests: **48 passed**. Ruff, lock and diff
checks pass. Full-package CPU suite, all-six-model port GPU gates, hosted CI,
commit and push were not run for this follow-up.
