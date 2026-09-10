# Boltz-2 upstream zeroes every MSA deletion feature (v2.2.0+), and the port reproduces it

Found 2026-09-10 while inventorying device arguments at 4,100 tokens: the
`has_deletion`, `deletion_value` and `deletion_mean` leaves handed to the
FoldJAX Boltz-2 program were all zero although the processed MSA held 16,880
deletion records per chain. The same is true of upstream.

## Mechanism (verified on both sides, CPU)

Upstream commit `04d27c71` (PR #441, 2025-07-04, "improve deletions dict
creation and population") rewrote, inside `construct_paired_msa` of
`boltz/data/feature/featurizerv2.py`,

```
chain_deletions = chain_msa.deletions[del_start:del_end]      # before
chain_deletions = chain_deletions[del_start:del_end]          # after (:427-437)
```

so each sequence slices the previous sequence's slice. The first MSA row of
every chain is the query (`del_start = del_end = 0`), the slice is empty, every
later sequence slices an empty array, and the numba deletions dict stays
empty. Releases v2.0.0-v2.1.1 (the Boltz-2 weights line, `afe4aa65`,
2025-06-06) compute deletions correctly; v2.2.0, v2.2.1, the pinned `b1ebfc4`
and GitHub main today carry the regression. The port vendored that loop
verbatim (`src/foldjax/models/boltz2/data/feature/featurizerv2.py:478-487`).

Probe (`scratchpad/wf/boltz-deletion/probe_deletions.py`, builds the
`PredictionDataset` from a case's processed directory exactly as
`featurize.py` does):

| case, mode | deletions dict | `has_deletion` nonzero | `deletion_value` max | `deletion_mean` nonzero |
| --- | ---: | ---: | ---: | ---: |
| L1000_3og2 as-is, port and upstream | 0 | 0 | 0 | 0 |
| L1000_3og2 pre-regression loop, port and upstream | 30,519 | 30,519 | 2.4515 (= π/2·atan(296/3)) | 897 / 1,003 |

All seven MSA arrays are bitwise equal port == upstream in both modes except
`deletion_mean` in the fixed mode (7.5e-9, float32 reduction order).

## What it means

- Every Boltz-2 row in this repository's tables (scale rows, panel cases,
  mixed set) ran with zeroed deletion features on both the port and upstream,
  so port-vs-native parity residuals and the tape-pinned results are
  unaffected. Absolute accuracy against experimental structures was measured
  with a featurizer that differs from the one the weights were trained with
  (inference from release dates; the training pipeline was not inspected).
- Trigger: a chain whose first MSA row carries no deletions, i.e. every real
  a3m/csv MSA. Jobs without an MSA are unaffected.
- The size of the effect on coordinates is unknown; it needs a native
  upstream run with the one-line patch against the unpatched native on a
  panel case.

## Decision

The port keeps upstream's released behaviour by default (the parity policy:
run what the released upstream runs). An opt-in native option
`msa_deletions=restored` reinstates the pre-`04d27c71` loop for users who
want the training-time features; its effect is measured on GPU before any
recommendation. The regression is filed upstream against PR #441.

## Measurement (2026-09-11)

Baseline, released deletions (the scale row `L1000_3og2-boltz2-scale`, 1003
residues, single chain): against the deposited 3OG2 chain A (986 CA matched by
sequence alignment), CA RMSD 0.93-1.41 Å and TM-score 0.990-0.995 over the
five samples. The `msa_deletions=restored` arm (job 1045) and a released
control from the same snapshot (job 1046) are compared the same way below.
