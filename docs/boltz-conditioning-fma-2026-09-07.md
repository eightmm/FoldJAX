# Boltz conditioning: native CUDA normalization arithmetic

Status: **conditioning cause reproduced and repaired; full model not admitted**.
This extends [the native AMP investigation](boltz-native-amp-2026-09-07.md).
The subsequent [trunk pair-normalization investigation](boltz-trunk-pair-norm-2026-09-07.md)
extends production correction to mixed-precision pair blocks; the v24 results
below remain an immutable historical checkpoint, not the latest source identity.
Portable metrics and artifact hashes are in
[`boltz-conditioning-fma-2026-09-07.json`](../bench/experiments/boltz-conditioning-fma-2026-09-07.json).

## What was wrong

The confirmed conditioning defect is numerical, not a different biological
condition or a checkpoint mismatch. On identical native operands, ordinary JAX
LayerNorm differed from the pinned native CUDA vector-4 Welford reduction and
FP32 fused multiply-add (FMA) arithmetic. These small FP32 differences sometimes
cross a BF16 rounding boundary before the next Linear projection.

At the first pair-conditioning norm on 5SAK, ordinary JAX output RMSE was
`7.3693e-8`, yet subsequent BF16 rounding changed **2,372 values**, with a maximum
difference of `0.0625`. Equal native mean/rstd alone did not restore the affine
output: the final scale/bias operation also needs native FMA rounding.

The arithmetic investigation follows the pinned
[PyTorch 2.12 CUDA LayerNorm source](https://github.com/pytorch/pytorch/blob/v2.12.0/aten/src/ATen/native/cuda/layer_norm_kernel.cu).
The causal evidence is the matched-operand replay and counterfactual below,
not source inspection alone. No FP64 or Torch dependency was added to inference.

## Controlled evidence

All coordinate comparisons use five original sample-index pairs, native BF16
AMP with FP32 islands, 200 diffusion steps, seed 101, and the native trunk from
3 recycles (4 passes) with full MSA. Features and sampler tape are captured from
native. Each sample gets one proper whole-system Kabsch fit; entity RMSDs are
measured without another fit. There is no crystal comparison.

**Every row below substitutes the native trunk.** Native conditioning is
substituted only in the explicitly named control; the repaired rows compute
conditioning and sampling in FoldJAX.

| 5SAK arm | Protein max RMSD (Å) | Ligand max RMSD (Å) |
| --- | ---: | ---: |
| Original conditioning, lazy (v15) | 0.240968 | 0.015838 |
| Original conditioning, materialized (v15) | 0.249654 | 0.016189 |
| Native conditioning, materialized control (v15) | 0.001183 | 0.000182 |
| Diagnostic corrected conditioning, materialized (v23) | 0.010290 | 0.000888 |
| Production corrected conditioning, lazy (v24) | 0.010367 | 0.000946 |
| Production corrected conditioning, materialized (v24) | 0.008367 | 0.000731 |

Thus changing lazy versus materialized execution alone did not explain the
original error. Correcting normalization materially reduces it in both routes.
The corrected production routes meet the unchanged `0.05 Å` structural
diagnostic on all five 5SAK samples. They are **not bitwise coordinate matches**.

The original native conditioning module reproduced all five captured outputs
exactly. All **119 mapped conditioning parameter leaves** matched the converted
FoldJAX weights exactly. Eighteen leaf operations were replayed on their native
inputs. Six captured norms now reproduce native mean, reciprocal standard
deviation and output exactly in the diagnostic kernel. Production direct norm
and shared-normalization-plus-affine also reproduce all six native outputs
bitwise. Two small atom-bias Linear discrepancies remain (1 and 35 values);
the tested pair-conditioning Linear and SiLU operators were exact.

This demonstrates a real conditioning cause. It does not establish that every
remaining 5SAK error comes from conditioning or explain input sensitivity by
size alone. In particular, the previous full FoldJAX v14 protein error was
`2.862567 Å`, from a different boundary than the native-trunk rows above.

The same production lazy correction also passes the other native-trunk n=5
controls (GPU jobs 378–379):

| Input | Entity | Maximum RMSD (Å) |
| --- | --- | ---: |
| 1URN | Protein | 0.00004854 |
| 1URN | RNA | 0.00005407 |
| 3GCA | RNA | 0.00003509 |
| 3GCA | Ligand | 0.00001300 |

## Production change and limits

In the historical v24 snapshot measured below, only BF16 pair conditioning and
diffusion bias projection select the CUDA Welford/FMA path. Ordinary FP32
inference and trunk transitions did not select it in that snapshot. Current
v30 additionally selects it for BF16 pair transitions in MSA/main Pairformer
and for mixed-precision triangle attention, including callers in template and
confidence stacks. See the [v30 scope and evidence](boltz-trunk-pair-norm-2026-09-07.md).
The lazy route still shares normalized input and defers layer projections;
there is no mandatory full token-bias materialization. Row-chunk recursion keeps
the conditioning opt-in. CPU/TPU/ROCm, context parallelism and unsupported widths
retain the generic JAX fallback. Exact CUDA evidence covers captured finite
operands at widths 16, 128 and 256, not every possible exceptional value/device.

Lazy affine is a separate Pallas call; its repeated diffusion-loop cost needs
measurement. No speedup, lower-memory result, confidence admission, independent
preprocessing equivalence or device-consumer tape audit is claimed here.
The current trunk-wide Pallas norms are non-fusible calls that materialize
FP32 pair activations. Their uninstrumented latency and peak-memory cost is
unmeasured and may regress throughput or memory relative to generic fused
LayerNorm, especially on large systems. This checkpoint is a numerical repair,
not admission of a faster or memory-optimized production profile.

## Verification

- Corrected focused CPU suite: **67 passed**. Existing Torch CPU parity checks
  plus new fallback checks: **16 passed** in the separate Torch 2.13 environment;
  this is not the pinned native CUDA 2.12 comparison.
- GPU job 375: all six production direct/shared norm outputs exact.
- GPU jobs 376–377: corrected 5SAK n=5 lazy/materialized coordinate diagnostics
  pass; completion-bound artifact hashes checked.
- Development failures were retained: GPU jobs 371–372 failed Pallas compiler
  compatibility checks (slice lowering, scalar literals), then job 373 passed.
  CPU initially exposed a dynamic-epsilon closure error (3 failed/64 passed);
  explicit kernel input fixed it. A Torch-only test collection failed in the
  standalone environment and passed in the existing parity environment.
- Wider affected Boltz CPU suite: **618 passed, 6 skipped**. New fallback tests
  also pass in the separate CPU-only JAX wheel environment: **10 passed**.
- At the v24 verification checkpoint, Ruff and `git diff --check` passed and
  all **489 src/bench Python files** matched that immutable snapshot. These
  identities do not describe the subsequent v30 worktree. No remote CI,
  commit or push was performed at that v24 checkpoint.
- Full v24 model capture and bound-artifact comparison completed (GPU job 380).
  Structure and confidence still fail, as detailed below.

The historical strict confidence tolerance remains `atol=rtol=1e-4`, separate
from the structure diagnostic. No acceptance threshold was relaxed.

## Full FoldJAX 5SAK result: the larger trunk issue remains

The same v24 snapshot also ran its own trunk, conditioning, sampler and
confidence heads with the captured native features/tape, five samples and the
full native settings. No native intermediate tensor was substituted in this
run. The [full portable report](../bench/experiments/boltz-amp-5sak-v24-2026-09-07.json)
verifies feature/tape entry identities and bound artifacts.

| Full FoldJAX result | Protein max RMSD (Å) | Ligand max RMSD (Å) |
| --- | ---: | ---: |
| Previous v14 | 2.862567 | 0.380979 |
| Conditioning repair v24 | 2.864511 | 0.380902 |

The structural diagnostic and both strict confidence reports **still fail**.
The public `confidence_score` field remains missing. The corrected conditioning
counterfactual therefore must not be presented as the full FoldJAX result.

Before conditioning begins, native versus FoldJAX final trunk RMSE is
`0.055363` for single representation and `0.080111` for pair representation.
First-recycle MSA input-pair RMSE is `0.000612`, while its pair-update RMSE is
`0.032292`. These are representation errors, not Angstrom coordinate errors;
matched-input MSA/Pairformer operator controls are the next localization step.

Across v14 and v24, final Pairformer input/output pair tensors are bitwise
identical. Single tensors are not (maximum output difference `0.703735`). There
was no intended trunk algorithm change, but the reason for this single-track
change has not been isolated; do not attribute the small full-run RMSD change
solely to the conditioning repair or call it calibrated native variability.

Conclusion: conditioning had a confirmed arithmetic defect and the repaired
downstream stage passes the three native-trunk structure controls. **Boltz as a
whole remains in progress and unadmitted**, primarily requiring upstream-of-
conditioning localization plus the separate confidence and performance gates.
