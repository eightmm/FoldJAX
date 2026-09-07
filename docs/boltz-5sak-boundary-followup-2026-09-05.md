# 5SAK boundary isolation and other-model status

This follows the [mixed-precision audit](boltz-mixed-precision-audit-2026-09-05.md).
It does not establish all-model upstream equivalence or crystal accuracy.

## 2026-09-07 priority follow-up

The user now defers structural residuals >0.05 and ≤0.1 Å as a gray zone;
5SAK's historical >0.1 Å errors remain a priority. The shared cuEq fix
described in the [Protenix regression report](protenix-cueq-regression-2026-09-07.md)
also makes Boltz BF16/FP16 use `DEFAULT`, while preserving FP32 policies and
invalidating affected legacy resumes. It does **not** establish a Boltz fix:
the historical 16.038990/6.988836 Å arm used `highest`, not the broken `high`
TF32 branch. Current Boltz model source otherwise matches that frozen arm.

The existing five-sample tape retains initialization, 200 churn/R/T draws
and 201 sigmas. It is a coordinate-only replay; FoldJAX confidence was not
executed. Native writer suffix `model_0` denotes confidence **rank**, not
sampler index. A fresh full-output capture must observe `Boltz2.forward`,
`predict_step`, and the sample-to-rank mapping; joining numbered confidence
files directly to sampler-index coordinates would corrupt the comparison.
Native raw confidence logits are not recovered from those writer files.

Next bounded control: reuse the independent native preprocessing/tape driver,
freeze its imported capture script with the model source, and capture the
complete native output before writing/ranking. Then isolate native-input
trunk and conditioning boundaries on 5SAK, retaining the original failed
coordinates and the separate confidence gate. This inventory is read-only;
no fresh Boltz GPU result is claimed here.

## Confirmed defects and fixes

The upstream-trunk diagnostic previously substituted captured `s`, `z`, and
`s_inputs`, but recomputed the learned relative-position encoding in FoldJAX.
Integer categorical inputs do not make its learned projection exact under
different arithmetic. The capture now retains that encoding; upstream-trunk
replays require all four captured tensors and reject incomplete old captures
before loading weights. Old three-tensor controls must not be described as
isolating the sampler completely.

Boltz's sparse relative-position projection also rounded each categorical
lookup addition in BF16, unlike the native single Linear accumulation. The
port now widens the already-narrowed selected weights to FP32, accumulates,
and narrows the output once. FP32 weights keep the existing path. No public
precision default, other model, or acceptance tolerance changes.

A half-ULP regression reproduces the old output `1.0` against the required
`1.0078125`; the corrected eager and JIT paths pass. On the actual captured
3GCA native relative-position output, a CPU JAX comparison gives:

| Policy | Different elements / total | Maximum absolute difference |
|---|---:|---:|
| Previous BF16 lookup sum | 71,262 / 270,848 | 0.0625 |
| FP32 accumulation, one BF16 output rounding | 0 / 270,848 | 0 |

This is an exact measured operator repair against native GPU output, not a
GPU end-to-end parity claim. Remaining trunk policies and conditioning
rounding differences are not repaired by this change.

The corresponding **same-GPU 5SAK operator control** also passes exactly:
7,790,185 / 24,444,032 values differed before (maximum 0.0625); zero differ
after. The CPU 5SAK control gives the same counts. This identifies a real
rounding defect, not merely a plausible source-level discrepancy.

## Full-schedule boundary controls

Fresh native mixed-precision capture: pinned Boltz commit
`b1ebfc46ecf57f5414e0d1a6f9027bbb122c53bc`, five samples, 200 steps, three
configured recycles, seed 101, native fused kernels. Paired FoldJAX arms use
the same native capture and realized sampler tape, independently constructed
inputs audited against that capture, cueq multiplication, and XLA attention.
One global proper Kabsch precedes entity-instance measurement without refits.

The baseline and exact-trunk arms use the preceding frozen model source;
the rounding-repair arm uses a separate frozen snapshot. This prevents a live
edit from changing the code during a queued replay. Exact-trunk substitution
is an internal-boundary control, not independent end-to-end port equivalence.
Per-arm source, harness, weight, feature and tape hashes accompany the arrays.

All four five-sample replays completed. Maxima over paired entity RMSDs (Å):

| Boundary supplied by native | Protein | Ligand |
|---|---:|---:|
| None, prior model source | 7.755733 | 1.544316 |
| Complete trunk, including learned relative position | 0.358655 | 0.030325 |
| Complete trunk plus conditioning q/c and all attention biases | 0.001720 | 0.000257 |
| None, with relative-position rounding repair | 16.038990 | 6.988836 |

Thus the major discrepancy enters through trunk and conditioning in this
capture. The remaining score/sampler residual is much smaller, not zero or
bitwise accepted. Conditioning substitution retains FoldJAX's deterministic
indexing/mask plumbing; all substituted tensor shapes are checked.

The local operator repair **worsens** end-to-end RMSD here. Correcting one
rounding site is not a monotonic structural improvement while other precision
policies differ. Error cancellation is a hypothesis, not an isolated cause.
Do not subtract these maxima to assign additive error contributions: they can
occur in different samples and the propagation is nonlinear. The 7.755733 Å
baseline is this fresh capture/replay, not the earlier 5.689754 Å measurement.

[All paired values and provenance](../bench/experiments/boltz-5sak-boundary-2026-09-05.json)
remain a failed/partial mixed-precision validation, not acceptance. The legacy
0.5 Å diagnostic exit is not the scientific gate, including for substituted
controls. Independent raw features retain the explicitly recorded <=1e-4
reference-coordinate arithmetic residual; they are not called bit-identical.

Next unresolved boundaries are raw-feature/weight narrowing before atom input
embedding, FP32 pair residual retention, normalization and MSA/triangle
accumulation, followed by conditioning quantization. These require further
native-input operator captures rather than an overall switch to FP32 or a
relaxed tolerance.

## Other models: evidence is not universal acceptance

| Model | Existing evidence | Open requirement |
|---|---|---|
| AlphaFold 3 | 35/35 compared preprocessing-panel dictionaries exact; vendored source with declared runtime patches | Fresh multi-complex, n=5 actual-RNG, entity-level output controls on the modified runtime |
| OpenDDE | Three-complex n=5 FP32/TF32-off controls: 5SAK protein/ligand maxima 0.000212/0.0000864 Å; 1URN protein/RNA 0.0000214/0.00000776 Å; 3GCA RNA/ligand 0.00000491/0.00000755 Å | Native TF32-enabled default and confidence parity; input acceptance exceptions remain classified |
| OpenFold3/OpenBind | Preprocessing and shared-tape DNA control; historical 1BNA global RMSD 5.78e-6 Å | Three composite atom-logit checks around 2.1e-3 remain unresolved at fixed 1e-4 tolerance; fresh multimodal n=5 entity controls |
| Protenix | Compared common preprocessing values exact in 33 panel cells; component and historical shared-noise controls | Old shared-feature/centre-only harness is insufficient for native-default full-tape parity; profile-specific output and confidence validation |
| ESMFold2 | Compared common preprocessing values exact in 34 panel cells; component and ordinary-RNG protein runs | Full sampler tape cannot yet be supplied through the carried core; matched-coordinate parity and active optional conditioning remain unverified |

Panel counts exclude unsupported/rejected cases and do not mean every possible
input is supported. Common-value agreement is not raw dictionary/dtype identity.
See [input classification](preprocessing-contract-audit.md),
[panel results](../bench/experiments/independent-input-entity-parity-2026-09-05.json),
and [fresh OpenDDE results](benchmark-followup-2026-09-05.md).

## Verification

- Before-fix half-ULP regression fails by 0.0078125; after-fix eager/JIT pass.
- Focused Boltz CPU suite: 285 passed, 3 skipped, 9 slow tests deselected.
- Complete-trunk boundary and mixed-policy selection: 14 passed.
- Four n=5 GPU controls completed; strict upstream equivalence remains open.
- Native updated-capture smoke: one sample, two steps, zero recycles on 3GCA;
  it writes all four trunk tensors. This is hook validation, not an n=5 result.
- No all-six-model GPU rerun, full-package CPU rerun, hosted CI, commit or push
  is claimed for this follow-up.
