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

---

# Correction: it is not a collapse, it is inter-chain placement

Both this document and the earlier global-FP32 record call the 24-26 A results a
collapsed or incoherent ensemble. Fitting each chain separately says otherwise.

`protein_protein_7st3` has two chains (`asym_id` over 4281 atoms). Maximum
across five samples, native as reference:

| Arm | whole-system | chain 0 | chain 1 |
| --- | ---: | ---: | ---: |
| baseline (`high`, main) | 3.3765 | 3.7689 | 0.6191 |
| RNE prequantized | 24.3378 | **3.6937** | 1.6838 |
| in-flight panel snapshot | 26.4223 | **3.8138** | 2.4890 |

**Chain 0 does not move.** It sits at 3.69-3.81 A in every arm, including the
untouched baseline. Chain 1 goes from 0.62 to 1.68 and 2.49 -- worse, but by a
factor of three, not eighty.

The whole-system number explodes from 3.38 to 24-26 A because the two chains are
placed differently *relative to each other*. Each chain is folded the same way it
was; the complex is assembled differently. Nothing collapsed and nothing became
incoherent.

## What this retracts

Two conclusions of mine, both stated with more confidence than the metric
supported:

- "Protenix-v2's port does not run correctly with the global matmul default
  forced to `highest`." It runs. It docks differently.
- "Two precision interventions of different shape produce the same collapse,
  which rules out tie-breaking." The premise is wrong, so the inference is void.
  What the two interventions actually do is perturb inter-chain placement while
  leaving folds alone, and the RNE arm's effect on chain 1 (0.62 -> 1.68) is not
  obviously worse than a docking coin-flip.

The earlier "15-17 A sample spread" reading of the global-FP32 arm is the same
artifact: samples that dock differently look maximally spread under a
whole-system fit.

## What it means for the standard

The cross-model table measures "one whole-system Kabsch fit per sample, no
rematching". On a multi-chain target that metric mixes fold accuracy with
inter-chain placement, and this case shows the second term can dominate by an
order of magnitude. Protenix-v2's 0.1611 A on 5SAK and 0.0335 A on 7st3 are
small enough that placement cannot be dominating them -- but the sensitivity is
there, and any arm that moves a multi-chain target's whole-system number by more
than a few angstroms should be read per chain before it is called a regression.

That is a general caution, not a Protenix one. It applies to every multi-chain
row in the table.
