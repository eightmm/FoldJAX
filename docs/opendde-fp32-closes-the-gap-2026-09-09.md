# fp32 matmul cuts OpenDDE's coordinate error 2.5x on 7st3

The retraction in [the rerun-floor note](opendde-row-is-the-rerun-floor-2026-09-09.md)
established that OpenDDE's gap against native is real and reproducible, roughly
21x its rerun floor. This is the first arm that moves it.

## The measurement

Whole-system Kabsch RMSD per sample against the native reference
(`opendde-closure-20260906/protein_protein_7st3/audit/native`), five samples,
`raw.npz` coordinates. The only difference between the arms is
`--jax-matmul-precision`.

| Arm | max | median | per sample |
| --- | ---: | ---: | --- |
| `high` (shipped, TF32) | 0.05396 | 0.01808 | 0.0094, 0.0540, 0.0090, 0.0493, 0.0181 |
| `high`, rerun control | 0.05405 | 0.01638 | 0.0086, 0.0540, 0.0089, 0.0501, 0.0164 |
| **`highest` (fp32)** | **0.02181** | **0.00398** | 0.0040, 0.0031, 0.0034, 0.0218, 0.0043 |

The rerun control reproduces the shipped arm to 0.0001-0.002, so the comparison
resolves far below the effect. fp32 cuts the max by 2.5x and the median by 4.5x.

The port's own precision sensitivity is the right order for this to be the
mechanism: `highest` versus `high` on the port alone moves coordinates by up to
0.0539 A, against a port rerun floor of 0.00546.

## Why this is the opposite sign from Boltz-2

Boltz-2's fp32-activation arm was 8.5x *worse*, because upstream Boltz-2 runs
that module in bf16 and the target is native's arithmetic, not more accurate
arithmetic. OpenDDE differs in a way that matters: native runs **torch** TF32,
and the port at `high` runs **JAX** TF32. Those are two different TF32s -- the
same operand truncation with different tie handling -- so `high` is not
"matching native", it is a second approximation that happens to be nearby.
Plain fp32 turns out to sit closer to torch's TF32 than JAX's TF32 does.

That reading is consistent with the earlier finding recorded for this port, that
its trunk is exact and the residual traced to upstream's TF32.

## Status and what is not yet established

This is one target. Two further cases (`protein_rna_1urn`, `protein_dna_7r6r`)
are queued at `highest` to test whether the improvement generalizes; a default
change is not justified on a single case, and this branch does not make one.
Note also that the panel's own gate uses per-entity RMSD under one global
superposition, not the whole-system number used here, so the gate verdicts are
not directly comparable to this table -- the direction and the ratio are what
this arm establishes.

## It generalizes: three of three

Both queued cases returned. Same metric, same references, same arms.

| Case | `high` max | `highest` max | `high` median | `highest` median |
| --- | ---: | ---: | ---: | ---: |
| protein_protein_7st3 | 0.05396 | **0.02181** | 0.01808 | **0.00398** |
| protein_dna_7r6r | 0.05737 | **0.02855** | 0.02049 | **0.01139** |
| protein_rna_1urn | 0.04022 | 0.04324 | 0.00400 | **0.00229** |

**The median improves on all three** -- by 4.5x, 1.8x and 1.7x. The max improves
on two of three; on 1urn it rises by 0.0030 A, which is below the 0.0055 A rerun
floor measured for this port and so is not a real regression.

Two of the three cases are ones the panel's 0.05 A gate failed (7st3 and 7r6r);
both land under it at `highest`.

## What this justifies, and what it costs

The evidence now supports changing OpenDDE's default matmul precision from
`high` to `highest`: three cases, consistent direction on the metric that is not
dominated by a single outlier sample, effects between 1.7x and 4.5x against a
rerun floor an order of magnitude below.

The cost is real and unmeasured here: `highest` is fp32 matmul, so it gives up
the TF32 tensor-core path. Time and memory for these arms were not recorded, and
a default change should not land without them -- this port's users did not ask
to trade throughput for a hundredth of an angstrom without being told the price.

That is the next measurement, and it is a benchmark run rather than a numerical
one. This branch stops at the numerics it set out to establish: OpenDDE's gap
against native was real, reproducible, and is substantially closable.
