# OpenFold3 TF32 operand rounding: 2026-09-09

Status: opt-in policy landed with unit evidence only. No accuracy improvement
is claimed. The default path is unchanged and asserted to be unchanged.

## What was wrong

Upstream calls `torch.set_float32_matmul_precision("high")`. The port pins
JAX's `"high"` (`inference.py:378`). Both then round FP32 operands onto the
10-bit TF32 mantissa grid before multiplying. They round them differently.

The [2026-09-08 tie-rounding diagnosis](openbind-tf32-tie-rounding-2026-09-08.md)
established this on saved native LayerNorm output and unchanged projection
weights, quantizing in NumPy FP64 and casting back:

| Operand rounding | Maximum error against |
| --- | --- |
| nearest, ties to even | native, `1.34e-5` |
| nearest, ties away | JAX `high`, `1.34e-5` |
| ties away, against native | `~1.2e-3` |
| toward zero, against native | `2.4e-2` |

All 15 case/projection combinations agreed at three token counts. A full-shape
GPU control then prequantized every projection operand to the round-nearest-even
grid and ran the *unchanged* `high` GEMMs: 5/5 projections passed at each shape,
and the largest matched bitwise.

## What this change does

`OPENFOLD3_TF32_ROUNDING=rne` rounds both operands of the five projections that
control measured -- query, key, value, gate and pair bias -- onto the
round-nearest-even grid before the matmul.

The correction works because that grid is a **fixed point** of the ties-away
rounding JAX then applies: an operand already on the grid is unchanged by the
second rounding, so the port's rounding is the only one that decides the
result. This is a unit-tested property, not an assumption, and it is what
separates this from stacking a second approximation on the first.

## What it deliberately does not cover

The output projection is absent: it was not in the measured control, and a
measured contract is not extended by analogy.

The triangle *multiplication* projections are absent for a stronger reason.
Upstream dispatches those through a Triton kernel whose same-operand controls
established truncation, not round-to-nearest. Applying this rounding there
would replace one mismatch with another. Kernel contracts stay distinct.

The default is `none`, on every backend. The correction is only correct on
hardware that actually rounds to TF32; on CPU, where `high` is plain FP32,
applying it would introduce the error it exists to remove. A test asserts the
default projection is bit-identical to the projection before this existed.

## Verification

`tests/models/openfold3/test_tf32_rounding.py`: 10 passed. It pins the tie
direction in both signs, signed-zero and NaN-payload survival through the bit
arithmetic, the half-step bound, idempotence, the fixed-point property, and
that the policy is inert by default and reaches the projection when asked.
The OpenFold3 suite is 546 passed, 354 skipped; Ruff passes.

## Measured effect size, and why it argues for keeping the default off

Two GPU arms on `t0128` (132 tokens, 1,013 atoms, seed 101, five samples),
identical but for the policy. Per sample, after the same whole-system Kabsch
fit, with no rematching:

| Sample | Unfitted maximum | Fitted maximum | Fitted RMSD |
| ---: | ---: | ---: | ---: |
| 0 | 0.161982 | 0.148851 | 0.032545 |
| 1 | 0.192180 | 0.168351 | 0.033722 |
| 2 | 0.188900 | 0.191692 | 0.031023 |
| 3 | 3.641557 | **3.683244** | 0.365414 |
| 4 | 0.117269 | 0.101328 | 0.025786 |

Peak allocator memory is 1,902.0 against 1,902.6 MiB, so the extra elementwise
pass costs nothing measurable. Wall time is not comparable between these two
arms: the second reused the first's compile cache.

Fitting barely moves any of these, so the differences are internal divergence
rather than rigid-body drift. Four samples sit at 0.10–0.19 Å, already an order
above this port's roughly 0.02 Å rerun floor. One sample moves 3.68 Å: a tie in
the last mantissa bit of a projection operand is enough to send that diffusion
trajectory somewhere else.

That is the argument for the default staying `none` until there is a native
reference. The change is emphatically not numerically neutral, and **these arms
cannot say which of the two is closer to upstream** -- they differ from each
other, not from native. A lever this large chosen on a plausible mechanism
rather than a measurement would be a coin flip on one sample in five.

## What is still missing

Direction. Establishing that this moves the port *toward* native needs an
upstream Torch run on the same input with the same tape, and that capture path
is not part of this branch. Until then this is a matched operand contract with
unit evidence and a measured effect size, not a measured accuracy result.
