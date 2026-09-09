# OpenDDE's elevated row is the native rerun floor, not port error

Artifact: `foldjax-bench/opendde-current-panel-20260909-Nf5tvz`, seven cases,
run with this branch's precision pin in place (`precision_policies` records the
foldjax arm at `jax_matmul_precision: "high"` against native's `native_tf32:
true`).

## The gate

Per-entity max coordinate RMSD, threshold 0.05 A. Four of seven cases fail it:

| Case | entity max RMSD | gate |
| --- | --- | --- |
| protein_1ubq | A 0.02000 | pass |
| rna_ligand_3gca | L 0.00275, R 0.00310 | pass |
| protein_rna_ligand_3v7e | L 0.00726, P 0.01284, R 0.01085 | pass |
| protein_rna_1urn | P 0.05010, R 0.00478 | **fail** |
| protein_ligand_5sak | A 0.05439, L 0.02091 | **fail** |
| protein_dna_7r6r | A 0.06337, B 0.04426, D 0.04275 | **fail** |
| protein_protein_7st3 | A 0.05351, B 0.11811 | **fail** |

## The control that settles it

The panel also contains `repeat-native-report.json`: native compared against a
second native run of the same case, through the same gate.

| Case | native vs native (rerun) | foldjax vs native | native's own gate |
| --- | --- | --- | --- |
| protein_protein_7st3 | A 0.05419, B 0.11824 | A 0.05351, B 0.11811 | **fail** |
| protein_rna_1urn | P 0.04692, R 0.00470 | P 0.05010, R 0.00478 | pass |

On 7st3 native cannot reproduce itself: rerunning it moves entity B by 0.11824 A
and fails the same 0.05 A gate. The port lands at 0.11811 -- marginally closer to
native run 1 than native run 2 is. On 1urn native's own rerun moves P by 0.04692
against the port's 0.05010, so the port sits 0.003 A outside a floor of 0.047.

## What this means

The 0.05 A threshold is below OpenDDE's run-to-run reproducibility on these
targets. Four cells "fail" a gate set inside the noise, and on the worst of them
the port is indistinguishable from a native rerun.

This bounds error reduction for OpenDDE the way the bf16 floor bounds it for
Boltz-2, and for the same reason: you cannot drive agreement below the
reference's own reproducibility. Further precision work aimed at this row would
be measuring noise. The honest way to close the row is to state the floor
alongside the gate, or to raise the threshold above the measured rerun spread
for the cases where it has been measured.

## What is not established

The rerun control exists for two of the seven cases (7st3 and 1urn). 7r6r and
5sak fail the gate at 0.063 and 0.054 with no rerun control captured, so their
failures are not yet attributed. Capturing `repeat-native` for those two is the
one cheap measurement that would finish this row.

Note also that every case reports `passed: false` including the three that pass
the coordinate gate, so the report's overall verdict is gated on more than
coordinates (`confidence`, `tape`, and the recorded `exclusions`). This document
addresses the coordinate gate only.
