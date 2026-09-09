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

---

# RETRACTED, same day: the control I used was not a rerun control

Everything above rests on reading `repeat-native-report.json` as "native
compared against a second native run". It is not. The panel holds four arm
directories per case, and the actual rerun controls are the two
`*-repeat-comparison.json` files, which I did not open before committing:

| Comparison | entity A | entity B | what it is |
| --- | ---: | ---: | --- |
| `native-repeat-comparison.json` | 0.00513 | 0.00553 | native run 2 vs native run 1 |
| `unobserved-repeat-comparison.json` | 0.00555 | 0.00509 | port run 2 vs port run 1 |
| `unobserved-report.json` | 0.05351 | 0.11811 | port run 1 vs native |
| `repeat-native-report.json` | 0.05419 | 0.11824 | port run 2 vs native |

`native-repeat/provenance.json` records `"arm": "native"`, confirming the first
row is native against itself.

## What the numbers actually say

Both sides reproduce themselves to about **0.005 A**. The port-versus-native gap
on entity B is **0.118 A**, and it reproduces across two independent port runs
(0.11811 and 0.11824, differing by 0.0001).

So the conclusion inverts. This is not noise and the 0.05 A gate is not set
inside the floor -- the floor is ten times below the gate. The gap is a real,
reproducible port-versus-native difference, roughly **21x the rerun floor** on
the entity that fails worst.

OpenDDE's elevated row is therefore a legitimate error-reduction target, not a
measurement artefact, and the opposite of what the section above claims. The
retracted section's own stated remaining measurement -- capture rerun controls
for 7r6r and 5sak -- was also unnecessary for 7st3 and 1urn, because the
controls already existed in the artifact.

## The error I made

I attributed a gap to a control without checking which two arms the control
file compares, when a sibling file named for exactly that comparison was in the
same directory. This is the second time in this session that an attribution was
committed before the evidence beside it was read; the first was the AF3 config
gate, where a guard test caught it. Here nothing caught it but re-reading.
