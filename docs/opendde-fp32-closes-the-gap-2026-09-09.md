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
