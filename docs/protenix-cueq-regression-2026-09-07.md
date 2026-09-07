# Protenix: BF16/cuEq regression and deferred structural gray zone

Status: **catastrophic regression reproduced and fixed; released 3GCA pilot
is structurally gray/deferred, not full model closure**. The user explicitly
assigned 0.05–0.1 Å structural residuals lower priority after the repair.
Frozen native-calibrated and strict failures remain recorded separately.

## Why this differed from earlier comparisons

Commit `d0eb469bbbf9ddd86ec7d6ef92c08f48ce95290f` (2026-09-05) began
forwarding JAX `high` to cuEq triangle multiplication as `TF32`, without
checking operand dtype. Earlier code omitted the explicit precision argument;
the historical Protenix matched-noise script pins `highest`. Neither executes
the broken BF16 + explicit TF32 branch used by the current production wrapper.
The historical capture additionally used FP32, TF32 off, cache/fusion off and
center-only augmentation; it was not a proof of the native-default mixed path.

The first new native-policy pilot had RNA/ligand maxima 24.810499/107.855538 Å.
Intermediate comparisons located a substantial difference before sampling:
`z_trunk` RMS error 49.649705, while the input embedding RMS error was 0.000364.
Identical-native-input module replay then found that all four inspected
FoldJAX triangle-multiplication calls left the residual **exactly unchanged**.

The installed native cuEq implementation forces `DEFAULT` for BF16/FP16 GEMM
operands. The JAX call did not. Its explicit TF32 fused branch reaches
FP32-only conversion instructions with BF16 operands. This is a dispatch bug,
not evidence that BF16 itself is unusable.

## Controlled GPU evidence

`tsp` 304 captured the actual publisher's first trunk module inputs/outputs;
305 replayed the individual modules in JAX. Job 306 varied only fused/fallback
selection, precision enum and norm-affine width on one actual native input
with the same released weights. Representative fused-kernel results:

| BF16 operand policy | Nonzero triangle-update entries | Maximum update magnitude | Residual RMS error vs native |
| --- | ---: | ---: | ---: |
| Explicit TF32, BF16 norm affine | 0 | 0 | 0.928755 |
| DEFAULT, BF16 norm affine | 270,848 | 12.5 | 0.021862 |
| IEEE, BF16 norm affine | 270,848 | 12.5 | 0.021862 |
| DEFAULT, FP32 norm affine | 270,848 | 12.4375 | 0.017770 |

Widening the norm affine did not rescue explicit TF32: its update was still
all zero. The JAX fallback did not show this zero-update defect. These controls
isolate the bad fused precision dispatch; they do not prove remaining fused
and native fallback paths equivalent.

The fix makes both Protenix/shared OpenDDE and Boltz adapters pass the actual
operand dtype. BF16/FP16 select `DEFAULT`; FP32 keeps the existing TF32/IEEE
mapping. No default trunk dtype or scientific tolerance was changed to obtain
the repaired result. Boltz resume now also fingerprints the shared adapter,
so an old affected result cannot silently survive an otherwise identical request.

## Full-schedule replay

Pinned Protenix `4c355be4553512f72453ecbfb65e69f4c35d1413`, released
`protenix_base_default_v1.0.0`, seed 101, five samples, 200 steps, 10 recycles,
46 tokens/719 atoms. Native outer/trunk BF16, diffusion/confidence FP32,
distogram BF16 and TF32 enabled. Native source has a recorded local sm120
build-architecture patch; it is not described as a clean upstream checkout.

Independent input comparison passes its finite schema/representation mappings.
The full checkpoint conversion audit matches all 4,174 arrays byte-for-byte.
Three native runs have identical actual input, MSA selections, initial/churn
noise, rotations, translations and FP32 schedules. Native-repeat entity maxima
are RNA 0.00227223 Å and ligand 0.00232014 Å; the pre-candidate calibrated
structure limit is 0.05 Å for both. The sealed calibration is not rewritten.

The math difference between pilot 303 and repaired job 307 is the dtype-aware
triangle precision dispatch; additional observers only strengthen metadata.
Both independently preprocess the same input and replay the same actual tape.
One whole-system proper Kabsch is fitted per paired sample, followed by
entity-wise RMSD without refitting or ranking/rematching:

| Entity | Before | After | Current structural triage |
| --- | ---: | ---: | --- |
| RNA | 24.810499 Å | 0.070145 Å | Gray zone, deferred; original 0.05 Å gate fails |
| Ligand | 107.855538 Å | 0.020094 Å | Baseline structural diagnostic passes |

RNA per-sample values are 0.070145, 0.020679, 0.008750, 0.018656, 0.021071 Å.
The repaired `z_trunk` RMS error is 1.071136, down from 49.649705. These
before/after controls confirm that the regression caused the huge discrepancy.

Confidence is **not closed**: 13 canonical leaves still exceed the frozen
native-repeat numeric allowance. Maximum atom pLDDT difference is 0.160974
points (0.00160974 on the stored 0–1 scale), pair PAE 0.154345 Å and pair
PDE 0.032825 Å. Raw contact probabilities differ by up to 0.154751; raw
distogram logits by 12.0. These are not waived by the structural gray zone.

## Evidence and verification

[Portable results, per-sample metrics, confidence failures and artifact hashes](../bench/experiments/protenix-cueq-regression-2026-09-07.json)
retain both candidates and all 12 operator controls. New captures distinguish
requested wrapper options from the actual inference-entry flags and tape
digests. Inference-entry observations are not device sampler-consumer proof.

- Before this dispatch repair, full CPU gate: 3,927 passed, 399 skipped,
  8 deselected; orchestration coverage 87.43%.
- After repair, affected precision/adapters/wrapper/resume tests: 198 passed,
  1 GPU-only test skipped on CPU. Ruff and diff checks pass.
- Real fused GPU regression, `tsp` 308: 1 passed. BF16 `high` and `default`
  produce identical nonzero updates on a deterministic synthetic input.
- Capture/report/checkpoint/calibration/augmentation CPU checks: 145 passed.
- Full n5 before/after captures, `tsp` 303/307: both exit 0; scientific
  acceptance is separate from successful execution.

The first native capture attempt (299) failed before model inference because
an observer rejected a text-valued object annotation. The serializer now stores
text as pickle-free Unicode and still rejects arbitrary objects; retry 300 and
native repeats 301/302 completed. A preliminary weight-audit attempt rejected
its own read-induced atime change; the corrected guard retains inode, size,
mtime and ctime checks, and the complete subsequent audit passes. These
observer/audit failures are retained, not represented as model failures or runs.

Remaining work: confidence, ordinary RNG, more chemistry and all other
Protenix profiles, device-consumer evidence, and uninstrumented speed/memory.
Native small-input fallback versus JAX fused kernels and the input embedder's
scalar precision are recorded residual hypotheses, not proven sole causes.
No crystal evaluation, dependency installation, commit, push or release occurs
as part of this diagnosis. The full CPU suite was not rerun after the final
dispatch edit; affected tests and a real GPU prediction were run instead.
