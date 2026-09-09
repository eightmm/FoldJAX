# Two different precision corrections both collapse Protenix's ensemble

## The attempt

OpenDDE's residual shrank 1.7-4.5x when the port stopped using JAX's TF32 and
used FP32, because upstream runs torch's TF32 and the two differ in tie
handling. The targeted version of that correction is to keep the TF32 grid and
match the *rounding rule*: prequantize FP32 operands with ties-to-even, which
the OpenFold3 work established is idempotent under the ties-away rounding JAX
then applies, making it a correction rather than a second approximation.

Applied to Protenix's single GEMM site,
`models/protenix/models/primitives/primitives.py::linear`, both operands
prequantized, precision left at `high`.

## Result

`protein_protein_7st3`, five samples, whole-system Kabsch against the panel's
native capture. Both arms built from the same `main` snapshot, differing only in
this patch:

| Arm | max | median | per sample |
| --- | ---: | ---: | --- |
| baseline (`high`) | 3.37652 | 0.47499 | 0.6827, 3.3765, 0.3973, 0.4421, 0.4750 |
| **RNE prequantized** | **24.33777** | **23.71819** | 23.7182, 24.3378, 18.2477, 3.1032, 23.8390 |

Tripwire: the two arms differ by 24.58 A, so the patch fired.

## What it means

This is the same failure as the earlier global-FP32 arm, which gave 26.4 A with
samples 15-17 A apart. Two precision interventions of very different shape --
one changing every matmul's accumulation, one changing only the operand grid of
the ordinary GEMMs -- produce the same collapse.

That rules out the hypothesis this arm was built on. If the residual were a
tie-breaking mismatch, matching the tie rule would move the port toward native;
instead it moves it 7x further away, exactly as switching accumulation did.
Protenix has something structurally sensitive to perturbation in the recycling
or diffusion path, and the sensitivity is what needs explaining before any
precision arm is meaningful here.

The OpenDDE lever does not transfer to Protenix in either of its forms.

## An observation that is not mine to conclude

The baseline here, built from `main`, scores max 3.37652 / median 0.47499. An
identically-shaped run earlier in this session, differing only in pointing
`PYTHONPATH` at the in-flight panel snapshot rather than `main`, scored max
0.26194 / median 0.13271 -- the same input, reference, weights and command.

That is a 13x difference attributable to source tree alone, which would mean the
uncommitted Protenix work in the shared checkout is a large accuracy improvement
over `main`. It is recorded as an observation rather than a result: those changes
are another session's, they are uncommitted and may have moved between the two
runs, and confirming it is that session's call, not mine.
