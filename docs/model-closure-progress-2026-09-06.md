# Sequential native-precision closure

User direction: finish the current model's bounded native-precision proof,
then continue through every model, choosing smaller corrections first. The
[upstream-first contract](precision-selection-protocol-2026-09-06.md) governs
all admission. Optional dtype lowering is separate and never rescues a failed
native comparison. No crystal scoring or silent skipping. The user's
[2026-09-07 amendment](precision-selection-protocol-2026-09-06.md#2026-09-07-amendment-practical-native-repeat-allowance)
adds prospective native-repeat-calibrated practical gates while preserving
historical strict failures.

## Current order and evidence

| Model | Status | Why / next discriminating boundary |
| --- | --- | --- |
| AlphaFold 3 | Finite common-runtime, fixed-kernel panel complete | [54/54 gates, 9 cases × 5 samples](af3-closure-2026-09-06.md); publisher dependency lock and independent-autotuning exactness remain excluded |
| OpenDDE | Known cuEq residuals accepted by user for continuation, 2026-09-07 | Historical strict native-default failures remain; this is not full release or BF16 admission |
| Protenix | Released 3GCA structural gray zone deferred; not closed | BF16/cuEq TF32 regression fixed; n5 RNA/ligand 0.070145/0.020094 Å; confidence and expanded profiles remain open |
| OpenFold3 / OpenBind | Pending | Quaternion/translation tape adapter missing; native FP32 same-tape repeat already fails on 5SAK |
| Boltz-2 | Active priority; not admitted | Conditioning and [pair-block normalization](boltz-trunk-pair-norm-2026-09-07.md) repaired on native operands; full 5SAK structure/confidence remains a separate gate |
| ESMFold2 | Pending, largest identified tape gap | Seven missing sampler/MSA/LM-dropout tape routes; native CUDA BF16 policy and confidence adapters must also match |

The remaining order may change after a cheaper diagnostic demonstrates a
smaller fix. Protenix profiles are separate targets: released, base-20250630,
v2, mini-esm-v0.5.0 and mini-ism-v0.5.0. Passing one cannot close all five.
Other separately managed checkpoints, such as OpenDDE ABAG, also need their
own recorded boundary. Missing local weights do not authorize downloads.

The latest user direction prioritizes Boltz trunk; OpenBind work is deferred
until that priority is handled. A local operator repair is not model closure.

## OpenDDE work underway

### 2026-09-07 continuation decision

The user accepts cuEq for OpenDDE and requests moving to the next model with
native-repeat-informed practical tolerance. Shipping cuEq/high on 5SAK had
protein/ligand maxima **0.05585073 / 0.01985762 Å**; maximum atom pLDDT, PAE
and PDE differences were **0.1966357 points / 0.2830896 Å / 0.2268696 Å**.
The single recorded native repeat was **0.00658463 / 0.00322312 Å** and
**0.0873804 points / 0.1334667 Å / 0.0488148 Å**, respectively. These are
observations from the [existing immutable report](../bench/experiments/opendde-native-policy-panel-2026-09-06.json),
not proof that every cuEq residual is bounded by native variability.

Status is `accepted_known_residuals_for_continuation`; historical strict
failures and missing uninstrumented performance/ordinary-RNG coverage remain.
The new prospective calibration rule must not retroactively relabel this
exception as a calibrated pass. FP32 remains the native-policy target.

Protenix starts with released `protenix_base_default_v1.0.0`, five samples,
200 diffusion steps and 10 recycles. Pinned runner `update_inference_configs`
sets diffusion and confidence to FP32 for non-v2 inputs up to 2560 tokens,
despite the outer BF16 autocast. The earlier config-only confidence audit was
incorrect for this panel. v2 and larger inputs have different policies and
must be evaluated separately.

The [Protenix regression record](protenix-cueq-regression-2026-09-07.md)
explains why the new pilot initially diverged catastrophically. On the same
actual input and tape, the dtype-aware cuEq correction changed RNA/ligand
maxima from 24.810499/107.855538 Å to 0.070145/0.020094 Å. The user explicitly
deferred the 0.05–0.1 Å structural gray zone; this does not close the remaining
confidence, independent-consumer tape, multimodal/profile or performance gates.

### Historical investigation (strict criteria retained)

Publisher `ddfa1df8aff1babf1fddac4247b7d2351bd0ce9f` defaults to FP32 with
TF32 enabled, n=5, 200 steps and 10 cycles. The CLI and backend cache-default
table now use FP32; explicit BF16 remains available but is not admitted by the
native precision screen. This may increase memory relative to the previous
BF16 default. No automatic BF16 fallback is added.

The original three-case FP32 coordinate controls had entity maxima of
5SAK A=0.041316/L=0.020835 Å, 1URN P=0.025111/R=0.022576 Å and
3GCA R=0.017798/L=0.006295 Å. Those are historical **confidence-off** runs,
not current full port acceptance. Their source/artifacts remain in the
[four-way record](../bench/experiments/opendde-four-way-2026-09-06.json).

The new `bench.opendde_closure_capture` executes the native forward and the
public FoldJAX prediction route with captured sampler/MSA replay. It records
independently featurized inputs, raw heads, scored outputs, source/assets/runtime
identity and lifecycle completion. Instrumented times are not speed benchmarks.
Successful capture does not itself constitute parity.

Initial native 3GCA capture (tsp 225) completed. The first FoldJAX attempt
(226) failed its input gate before inference: the new wrapper omitted the
public request seed, so it used 0 while native used 101. A CPU probe reproduced
both sets of reference coordinates exactly at those respective seeds. The
wrapper now pins seed 101 and rejects unexpected seed/sampling settings; the
model featurizer was not changed to hide this harness error. Retry 227 completed:
independent inputs and MSA selection pass; RNA/ligand entity RMSD maxima are
0.0183138/0.00791017 Å. Raw confidence still fails the fixed 1e-4 absolute/relative
diagnostic (PAE logits max 0.0186329; extracted PAE max 0.0340767 Å).

The fresh native same-tape repeat (229 versus 225) also fails that confidence
diagnostic: PAE logits max 0.0198202, extracted PAE max 0.0164633 Å, despite
RNA/ligand RMSD maxima only 0.00190632/0.000934342 Å. TF32-off native (228)
versus JAX highest (230) reduces coordinate maxima to 0.000479120/0.000733347 Å,
but does not clear raw confidence. These controls do not replace native defaults.

Source and actual checkpoint inspection independently found a genuine shared
OpenDDE/Protenix confidence defect: native initializes directed pairs as
`s1[j] + s2[i]`, while FoldJAX used `s1[i] + s2[j]`. Both same-name checkpoint
leaves match exactly; no conversion-time swap compensates. The shared head now
uses native axes, with a nonzero asymmetric regression covering batched and
unbatched inputs. This does not imply that all confidence drift has been fixed.

The default-policy and confidence corrections also bind relevant source-file
identities into prediction-resume dependencies, preventing reuse of old BF16 or
pre-axis-fix results under an otherwise identical request. This is a targeted
resume safeguard, not an assertion that every model source is fingerprinted.

Additional jobs separate publisher-supported deterministic execution and
confidence-only identical intermediate/coordinate inputs. Native head captures
232/233 failed because the observer tried to serialize a lazy, trunk-only
relative-position object. The observer now captures the finite features actually
consumed by confidence; model features and native source remain unchanged.
Retries 236/237 and confidence-only 238 completed. Failed evidence is retained.
Native deterministic 3GCA repeat is bitwise identical in all coordinate and
confidence leaves. Given **identical native intermediate tensors and coordinates**,
the corrected JAX confidence head at `highest` passes all four raw heads; the
legacy-axis counterfactual fails. At `high`, even the corrected head fails.
This isolates a genuine implementation correction but is not end-to-end proof.

All 248 native confidence weight leaves match the managed checkpoint bitwise:
31,067,214 scalar values, with every native confidence key consumed exactly once.
The [portable weight record](../bench/experiments/opendde-confidence-weights-2026-09-06.json)
records both checkpoint identities and canonical content digest. Therefore the
remaining confidence discrepancy is not explained by confidence weight conversion.

### Fresh full-confidence panel (before sampler/kernel follow-up)

One proper unweighted whole-system Kabsch per paired sample; entries are maxima
over all five samples for each entity. No entity refit or side-chain rematching.
The rows labelled native TF32-off / JAX highest below are **requested-policy
controls, not fully matched IEEE executions**: a subsequently discovered FFI
precision bug retained TF32 inside JAX triangle attention. They remain evidence
of the old implementation, not evidence that true matched FP32 necessarily drifts.

| Input | Native policy / JAX control | Entity RMSD maxima (Å) | Structure <=0.05 Å | Strict confidence |
| --- | --- | --- | --- | --- |
| 3GCA RNA + ligand | Native TF32 / JAX high, axis corrected | R 0.0182184; L 0.00732784 | Pass | Fail |
| 1URN protein + RNA | Native TF32 / JAX high | P 0.150739; R 0.0206010 | Fail | Fail |
| 1URN protein + RNA | Deterministic native TF32-off / JAX highest | P 0.00566584; R 0.00393613 | Pass | Fail |
| 5SAK protein + ligand | Native TF32 / JAX high | A 0.0631446; L 0.0237821 | Fail | Fail |
| 5SAK protein + ligand | Deterministic native TF32-off / JAX highest | A 0.0607570; L 0.0188790 | Fail | Fail |

Independent input and per-cycle MSA gates pass for all these runs. Random tape
archives match bitwise. Public JAX GPU noise schedule also matches all 201 native
levels bitwise (a CPU-only schedule probe did not, so CPU results cannot substitute
for the deployed GPU path). A separate native 1URN same-tape repeat still fails
strict confidence (raw PAE max 0.0408974), although entity RMSDs are only
P 0.0161098 / R 0.00259554 Å. Thresholds have not been changed to absorb this.

Identical-operand Linear controls exclude a blanket CUDA-version explanation:
the original Torch runtime reproduces all 53 captured TF32 and FP32 controls
bitwise; an existing CUDA 13 Torch runtime reproduces 51/53 TF32 observations,
including the three selected operators that differ in JAX. Explicit RNE TF32
rounding before JAX high removes the selected operators' rounding discrepancies.
This is an operator diagnostic, not permission to replace the production policy
globally. The compiled HLO identifies a Triton nested GEMM for the discrepant
distogram operator; a no-Triton control selects cuBLASLt and removes its difference.
However that global switch **worsens** the corrected 1URN protein entity RMSD
from 0.0137915 to 0.149272 Å. It is not adopted as the default. No new dependency
installation was needed for these controls.

### Sampler precision correction

Native `rot_vec_mul` deliberately performs the three-dimensional rotation with
scalar FP32 arithmetic, avoiding AMP/TF32 matrix contraction. FoldJAX instead
used `einsum`, which inherited TF32 at `high`. On the actual 5SAK random tape,
with only the denoiser replaced by zero, the **first denoiser input** differed by
max 6.521484 Å (RMSE 0.685893). Restoring scalar rotation reduces this to
max 0.0009765625 Å (RMSE 0.000026255); high and highest now agree in this probe.
These noisy initial coordinates are not final-structure RMSDs.

In fresh complete 1URN inference, the same correction reduces protein/RNA
entity maxima from 0.150739/0.0206010 to **0.0137915/0.00497038 Å**. Both pass
the unchanged structural screen. Raw and extracted confidence still fail.
The old 5SAK requested-FP32 control still fails after rotation correction: A 0.0608106,
L 0.0188802 Å. Its independently produced input embedding agrees to max
3.12924e-7, but differences already exist at the trunk output (single max
0.252621, pair max 1.870310), before diffusion. A head-only probe distinguishes
this inherited difference from a further confidence implementation issue.

The scalar correction initially exposed two inner-scan padded-RNG regressions.
A barrier immediately after a lazy step-noise draw restores its FP32 tape
boundary; the existing bitwise tests pass again without tolerance changes or
materializing the entire step tape. Separately, eager versus outer-JIT rotation
generation already differed before this correction (max 5.96e-7); that existing
ordinary-RNG issue is not claimed solved by the matched-tape evidence.

A distinct sample-chunk bug was reproduced and fixed: per-step translation
tapes have their sample axis at `-2`, unlike coordinate/rotation tapes at `-3`.
The old rank-based slicer incorrectly sliced the step axis. Tests exercise
nonzero chunk starts, packed/tuple noise, and the final shorter chunk. This is
not the cause of the unchunked five-sample discrepancy.

Numerical records and arm identities are collected in the
[progress artifact](../bench/experiments/opendde-closure-progress-2026-09-06.json).
This is explicitly an in-progress diagnostic record, not model acceptance.

### Shared attention FFI correction

`models/_cueq.py` hard-coded `Precision.DEFAULT` for triangle attention while
already translating the active policy for triangle multiplication. Installed
cuEquivariance 0.11.1 interprets DEFAULT on FP32 operands as `use_tf32=True`.
Thus OpenDDE's public `highest` context reached the trunk correctly but was lost
at this FFI boundary, including attention inside MSA pair stacks. CPU calls of
the wrapper and the installed vendor policy function reproduce the mismatch.

The shared adapter now translates high/highest and their supported aliases into
explicit Precision enums. Unsupported algorithms fail rather than silently
lowering. Passing `None` would not be sufficient for this installed vendor
version, whose policy helper matches enums rather than the context's strings.
OpenDDE, Protenix and OpenFold3 resume dependencies bind this source change.
New completed GPU controls record actual FFI trace attributes
`float32 / HIGHEST / use_tf32=False`, not merely requested settings:

| Input | Native TF32-off / corrected JAX highest entity maxima (Å) | Raw PAE max absolute error | Strict confidence |
| --- | --- | --- | --- |
| 5SAK | A 0.000205347; L 0.0000821236 | 0.00128019 | Fail |
| 1URN | P 0.0000294132; R 0.00000761822 | 0.000532150 | Fail |
| 3GCA | R 0.00000519648; L 0.00000886281 | 0.000210762 | Pass |

All fifteen structures pass the unchanged 0.05 Å entity screen. The 5SAK trunk
single/pair maxima fall from 0.252621/1.870310 to 0.000366211/0.002075195.
On identical native confidence inputs, corrected highest PAE-logit error falls
from 0.0292313 to 0.000582695; pLDDT/resolved logits pass the strict numerical
test, but PAE/PDE still narrowly fail. Native-default TF32 is not replaced by
these IEEE controls, and raw-logit tolerances have not been relaxed.

The current default-policy 3GCA rerun passes the structural screen at
R 0.00203750 / L 0.00193613 Å but still fails strict confidence. Default-policy
5SAK after scalar rotation remains just outside the screen at A 0.0527655 /
L 0.0183603 Å. A same-QKV diagnostic finds that native-query pre-scaling alone
does not fix cuEq TF32 attention: at 437 tokens its max error is 0.0234821,
versus 0.000631690 for the existing XLA attention core on the same saved queries.
This motivates a full-model XLA-attention control, not a global policy change.
The installed cuEq FP32 kernel rejects head width 64; that synthetic probe is
explicitly unsupported, while the width-32 controls complete.
Per-sample values, strict confidence leaves and observed FFI policies are retained
in the [precision-boundary record](../bench/experiments/opendde-precision-boundaries-2026-09-06.json).

The completed native-default TF32 / JAX high **XLA-attention controls** pass
the structural screen for all seven finite-panel inputs (35 samples):

| Input | Entity RMSD maxima (Å) | Max atom pLDDT difference (points) | Max PAE difference (Å) | Strict confidence |
| --- | --- | --- | --- | --- |
| 5SAK | A 0.00882531; L 0.00551412 | 0.0737488 | 0.0985518 | Fail |
| 1URN | P 0.00740915; R 0.00369934 | 0.0297248 | 0.0158501 | Fail |
| 3GCA | R 0.00285510; L 0.00204522 | 0.00606775 | 0.00976467 | Fail |
| 7R6R protein + two DNA entities | A 0.0109785; B 0.00695048; D 0.00658548 | 0.0789762 | 0.0924911 | Fail |
| 3V7E protein + RNA + ligand | P 0.00768493; R 0.00470847; L 0.00307556 | 0.0510395 | 0.0749917 | Fail |
| 7ST3 two proteins | A 0.0361796; B 0.0107708 | 0.0346303 | 0.0413127 | Fail |
| 1UBQ protein | A 0.00972035 | 0.0195861 | 0.0193405 | Fail |

This is an explicit `PROTENIX_TRIANGLE_BACKEND=xla_jit` diagnostic, recorded in
each arm's provenance, not a newly adopted default. For example, 5SAK full PAE
still differs by max 0.0985518 Å, even though pTM/ipTM/ranking and mean pLDDT
pass the strict numerical screen. Raw confidence and pair matrices cannot be
replaced by their means. These are native-precision **non-default XLA-backend**
controls. All seven candidates preserve the descending order of the five
native ranking scores; this is recorded without rematching any samples and
does not replace the full confidence checks. The post-FFI-fix shipped cuEq/high rerun (290) still fails 5SAK's
structure gate: A 0.0558507 / L 0.0198576 Å. The comparable native-default
same-tape repeat (291) passes structure at A 0.00658463 / L 0.00322312 Å but
fails strict confidence: raw PAE max 0.0722647, extracted PAE max 0.133467 Å,
PDE max 0.0488148 Å and atom pLDDT max 0.0873804 points. This is one repeat,
not a statistical error floor or an approved tolerance. In particular, the
previously proposed 0.05 Å PAE maximum would reject this native repeat too.
No default switch or confidence-threshold change has been approved. The
[seven-case native-policy record](../bench/experiments/opendde-native-policy-panel-2026-09-06.json)
retains every sample/entity value, all confidence leaves, source/report hashes,
input gates, observer bridges and native-repeat controls.

A second same-QKV native observer (292) compares each actual QK/PV matmul
against a TF32-off side call on the identical operand objects. At 437 tokens
and head width 32, the TF32 toggle has no byte-level effect on QK, but changes
PV by max 0.000529170. At 32 tokens, both calls change. This is evidence that
the global TF32 flag does not specify the observed numerical effect of every
operator; it does not identify the actual hardware instruction of QK. The
observer's seven saved input/output arrays match the earlier unobserved probe
bitwise at all three shapes, so those finite observations have a direct bridge.

The benchmark-only consumer observer attaches execution callbacks to the
actual sampler step and MSA cycle scans with explicit indices, and observes
the initial FP32 draw/schedule boundary. It requires all 211 expected events,
exact shapes/dtypes/finite values and native-to-storage mappings, rejecting
missing/duplicate/wrong-index events. Twenty-one CPU tests cover the real sampler
and real trunk scan/cast routes. On GPU, 3GCA has all 211 events value-exact in
both default-TF32/XLA and common-IEEE/cuEq controls. The IEEE non-observer bridge
passes strict structure/confidence (R 0.00000440654 / L 0.00000646316 Å); it is
not bitwise output identity. The default-TF32 bridge fails strict confidence
despite R 0.00247399 / L 0.00178844 Å. Requiring the non-observer autotuning
cache fails closed: callback instrumentation creates 10 uncached fusion kernels
among 422 instructions. Extension control 288 retains all 460 producer kernel
records unchanged and adds ten observer-specific records. Nevertheless, its
bridge still fails strict confidence (R 0.00220290 / L 0.00171976 Å).
Callback observations are never performance evidence.

A signed-zero regression subsequently exposed that the initial collector used
value equality, not byte equality. It now requires finite byte-identical values,
and records `value_comparison=finite_bitwise_bytes`; older 211-event captures
are accurately labelled value-exact only. The strengthened GPU rerun (289)
passes all 211 byte-exact consumption events. It strictly reloads the extended
470-record kernel cache without permitting autotuning fallback, yet differs
from the previous observer run: R 0.00316154 / L 0.00158711 Å, strict confidence
fail. This excludes a differing recorded kernel choice as the sole explanation;
unrecorded runtime behavior is not thereby identified.

The same-process repeat wrapper (294) executes three forwards with unchanged
explicit sampler/MSA tape bytes and the same argument-value objects. The actual
JIT owner remains the same and the two warm dispatches retain one cache entry;
the 470-record autotune file is byte-identical to the loaded file. All three
pairs pass structure, with maximum entity RMSD 0.00259286 Å, but fail strict
confidence. The warm pair has identical distogram/contact outputs, while its
PAE differs by max 0.00939560 Å. This is host owner/cache evidence, not direct
runtime-executable identity: the public `_predict` route still rebuilds
normalized inputs and schedule before dispatch. The first wrapper
attempt (293) failed before inference because its autotune dump parent did not
yet exist; 294 used an existing parent without changing model code.

The next repeat (295) calls the final compiled pool directly with identical
normalized argument objects and byte-identical features, schedule, key and
explicit tapes before and after each dispatch. Weight arrays are the same
objects, but their bytes are not recopied or rehashed for this control. The
470-record autotune file remains byte-identical. Nevertheless, all three pairs
fail strict confidence, with maximum entity RMSD 0.00271546 Å. Repeated public
input preparation therefore does not explain this observed variation.

### Runtime scatter determinism control

The denoiser's atom-to-token aggregation uses floating-point scatter-add with
repeated indices. JAX documents that conflicting update order may be
[nondeterministic](https://docs.jax.dev/en/latest/_autosummary/jax.Array.at.html),
separately from [XLA compilation-time autotuning](https://openxla.org/xla/determinism).
This source evidence identifies a candidate, not the measured cause by itself.

A scatter-only compiler control sets
`--xla_gpu_enable_scatter_determinism_expander=true`. Its strict old-cache attempt
(296) fails closed on 29 new configurations among 441 instructions. Extension
297 retains every one of the original 470 records unchanged and adds those 29.
All three same-normalized-input forwards now produce **27/27 raw leaves
bitwise identical**, including coordinates and every returned raw confidence
head. A fresh process (298) strictly reloads all 499 records, also returns
bitwise-identical repeats, and agrees across processes in all 27 raw leaves and
35 canonical confidence leaves. Kabsch arithmetic alone contributes residuals
below 6e-15 Å to the otherwise byte-identical coordinate comparison.

This controlled intervention strongly implicates nondeterministic scatter
lowering in the observed 3GCA repeat variation. It does not isolate a single
GPU instruction, prove determinism on other cases or freeze cuEq's separate
internal autotuning. The unchanged XLA attention candidate and all precision
settings are retained; no production default is changed. Its speed/memory
tradeoff has not been measured without instrumentation.

Repeat stability is not native parity: the deterministic JAX result still
differs from default native 3GCA at R 0.00244247 / L 0.00244997 Å and max PAE
0.00879860 Å; strict confidence fails. Against the separately deterministic
native TF32 reference, R 0.00204414 / L 0.00216710 Å pass structure but strict
confidence still fails. Both comparisons check exact sampler/MSA archives.
Thus input/tape corruption and repeat instability are distinguished from the
remaining cross-implementation numerical discrepancy, not used to waive it.

The expanded observer/capture/preflight/packing/precision gate passes 84 tests.
The final repeat-wrapper, capture, consumer and attention-observer gate passes
66 tests. These diagnostic observers are not performance benchmarks.

A separate OMS advisor review confirms that XLA candidate success must not be
reported as shipped-default success, and identified the now-completed 5SAK
native repeat as a necessary calibration control. Its suggestion that independently
compiled FP32 could never meet the strict gate is **not adopted**: three sizes
are insufficient for a universal impossibility or rounding-floor claim.

The new attention regression failed eight cases before correction and passes
afterwards; 42 focused shared-attention tests pass. Before this FFI correction,
the expanded OpenDDE/head/manifest/benchmark gate passed 540 tests, including
the repaired padding and sample-chunk routes. A fresh complete CPU coverage
gate completed with 3,730 passes and one stale HLO assertion failure (399 skips,
8 network deselections; coverage 87.43%). The assertion banned every concatenate,
including the corrected scalar rotation's harmless xyz assembly. The test now
targets only an axis-zero stack with the full noise-tape shape. Its bitwise and
memory requirements remain: tuple/packed temporary bytes are 386,264/2,264,
still saving 384,000 bytes. All 61 focused sampler/geometry/precision/capture
tests pass. The fresh full gate passes **3,731 tests**, with 399 skips and
8 network deselections, at 87.43% coverage. A later independent review found
one stale BF16-default preflight warning; the message now states native FP32
default and unvalidated BF16 opt-in. The warning and new benchmark observer
integration have 43 passing focused tests after the full-suite collection.

Verification: the latest full CPU gate passes 3,731 tests, with 399 skipped,
8 network tests deselected and 87.43% coverage (gate 80%). It precedes the final
preflight-warning and benchmark-observer additions, covered by subsequent
affected gates above (84 and 66 tests). Default-dtype, directed confidence axes,
translation chunking and attention-precision regressions failed before their
corrections and passed afterward; the earlier 3,687-pass full run is historical.
Ruff, lock and whitespace checks pass. An extra repository-wide formatter check
reports 266 files needing reformatting, including unrelated existing source;
no bulk formatting was performed, and formatting is not the required Ruff gate.
No all-model completion, new performance advantage,
commit, push or release is claimed by this progress record.

## Decisions before admission and advancing the sequence

The project's allowed confidence loss is still undecided. The 1e-4 raw/extracted
checks remain diagnostic failures, not silently waived acceptance gates. A
single native repeat cannot choose a larger tolerance after observing this
panel. An explicit confidence criterion must be fixed before testing a fresh
validation panel. A user question is outstanding; no answer is inferred.

The XLA-attention candidate also requires an explicit default-policy decision
and uninstrumented performance/memory evidence. Current cuEq/default5SAK remains
outside the structure screen; optional BF16 does not rescue that failure.
Ordinary-RNG, writer, full chemistry/profile and extra-checkpoint admission
remain open. Consequently OpenDDE is not closed and the next model has not
been promoted past it. The remaining order is recorded at the top of this file.
