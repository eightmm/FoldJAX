# Upstream-first correctness and optimization contract

## Current contract: clarified project objective

### 2026-09-07 follow-up: structural gray-zone triage

After the Protenix cuEq regression repair, the user explicitly requested that
0.05–0.1 Å residuals receive lower investigation priority. Classify the maximum
sample-index-paired RMSD for **each entity**, using the same global fit:

| Entity RMSD | Structural triage |
| --- | --- |
| ≤0.05 Å | Baseline structural diagnostic passes |
| >0.05 and ≤0.1 Å | Gray zone; retain and defer, not a pass or model closure |
| >0.1 Å | Prioritize causal investigation |

This is a work-priority amendment, not a rewrite of frozen calibration or
historical pass/fail results. Continue reporting native-calibrated and strict
gates separately. Inputs, actual RNG tape, confidence, optional dtype lowering
and performance retain their independent requirements. The observed Protenix
3GCA RNA maximum 0.070145 Å is explicitly deferred under this amendment;
other profiles and chemistry are still untested by that pilot.

### 2026-09-07 amendment: practical native-repeat allowance

The user accepts the measured OpenDDE cuEq residuals for continuing to the
next model, Protenix. This is a known-residual acceptance, **not** a claim
that the historical strict gates passed or that one native repeat explains
all cross-framework differences. The old reports remain unchanged.

For fresh native-policy panels, freeze this engineering rule before observing
candidate results:

- Execute at least three native runs with the same complete realized tape,
  input, checkpoint and effective operator policy. Reject calibration if any
  of these identities differ. Retain every native pairwise comparison.
- Per entity, the practical RMSD limit is `max(0.05 Å, 3 * R_native)`, where
  `R_native` is the maximum across all native pairs and paired sample indices,
  measured after the same whole-system Kabsch used for port comparisons.
- For each floating confidence leaf, retain the original pointwise
  `1e-4 + 1e-4 * abs(reference)` allowance and add `3 * C_native`, where
  `C_native` is that leaf's maximum absolute native-pair difference across
  all entries and paired samples. Raw heads and extracted arrays are separate
  leaves. Report the resulting numeric limits before evaluating FoldJAX.
- Feature/tape identity, integer/boolean fields, atom/chain order and matching
  undefined-value positions remain exact prerequisites. Report sample ranking
  separately; never rematch samples to improve either gate.
- Continue reporting the original strict gates and every observed maximum
  alongside practical gates. Never silently recalibrate after a failed
  candidate. Three repeats and the fixed factor of three are an engineering
  margin, not a statistical confidence bound or universal biological limit.

This amendment supersedes the prohibition on explicitly approved practical
tolerance changes for native-policy port comparisons below. It does not set an
extra-loss budget for optional BF16, admit untested chemistry/profiles, prove
memory/speed benefits, or waive the release gates.

### Unchanged precision and evidence requirements

This section supersedes the historical coordinate-only screening rule below.
Port publisher inference to JAX and obtain faster post-compilation execution
and lower measured memory use while preserving **structure and confidence**.
AlphaFold 3 is already JAX upstream: its target is runtime integration and
optimization, not a Torch-to-JAX rewrite.

1. Pin upstream source, weights, assets, runtime and effective settings.
2. Match native dtype **and operator policy** first, including FP32 exceptions
   within mixed precision. A weight cast or dtype label alone is insufficient.
3. Validate independent preprocessing, actual RNG tape, structures and
   confidence before evaluating additional optimizations.
4. For optional precision lowering, compare native default versus native
   reduced precision where supported, FoldJAX versus native at matched
   precision, and the candidate directly versus native default.
5. Select lower precision only after predeclared structure/confidence criteria
   pass and measured memory/runtime tradeoffs justify it. The allowed extra
   loss is currently **undecided**, not automatically 0.05 Å. Do not loosen
   criteria after seeing results or change defaults with missing evidence.

A native BF16 default is the target as-is; it need not first resemble a
hypothetical FP32 model. Checkpoint storage dtype is not compute policy.

Each model closure requires:

- Independent feature keys, shapes, dtypes, values, metadata, atom identities
  and masks; record any finite representation exceptions.
- Actual draws for preprocessing, MSA, dropout, initial/churn noise and rigid
  augmentation as applicable. Equal integer seeds are insufficient. For shared
  JAX code, independently recording and proving exact realized draws is valid;
  merely assuming PRNG equivalence is not. Reaudit changed buckets/batching.
- Five sample-index-paired outputs at the native schedule. One proper global
  unweighted Kabsch, then every entity measured without refitting. No ranking,
  sample rematching or crystal comparison in this stage.
- Raw confidence heads and extracted per-sample confidence, including arrays,
  masks, chain order, booleans and matching native undefined/NaN positions.
- Cold/compile-inclusive latency separately from synchronized warm execution;
  explicit cache, device allocator and host peak-memory definitions.
  Instrumented tape captures are not performance benchmarks.
- Immutable evidence identity, tested chemistry/profile panel, failed and
  unsupported cases, and remaining uncertainty. Closure is panel-bounded,
  never universal chemistry or all-hardware correctness.

For identical JAX graphs, dtype and RNG identity still do not fix compiler
kernel choices. The AF3 closure additionally records Tokamax and XLA autotuning
decisions and matches them with **separate compiled-executable caches**. A
consumer fails on missing kernel coverage. The uninstrumented output bridge
may extend the audited compiler cache only after proving all old records remain
unchanged; coordinates and confidence still face the same gates. Fixed-kernel
parity and independent default-autotuning variability are separate claims.

The existing 0.05 Å coordinate diagnostic and `atol=rtol=1e-4` numerical
checks remain baseline investigation gates, not sufficient approval for
optional precision lowering. Native repeats separate runtime variability from
port drift; do not silently relax a threshold to absorb failed repeats.

Work proceeds model by model, smallest change first: **AlphaFold 3 v3.0.4** at
native mixed precision, n=5, with independent inputs, actual tape, raw and
postprocessed confidence. Include multimodal and monomer/multimer cases;
record shared binary/CCD assets and kernel/bucket deviations. Optional dtype
lowering and other-model optimizations are separate follow-ups.

## Historical initial screening rule (superseded for future selection)

The first [native screening results](native-precision-selection-results-2026-09-06.md)
reject optional BF16 for the tested OpenDDE/OpenBind profiles and retain a
separate failed native FP32 repeat for OpenBind.

Prospective user-selected coordinate threshold: **0.05 Å**, on every entity
instance in every one of five sample-index-paired predictions. Apply one proper
unweighted whole-system Kabsch transform and no entity refit. Experimental
coordinates, sample rematching and confidence reranking are excluded.

1. If pinned upstream defaults to BF16/mixed precision, reproduce its actual
   operator policy. A native FP32/BF16 difference does not disqualify its
   publisher-default BF16 target: port parity is against that target.
2. If upstream defaults to FP32, compare upstream FP32 against upstream's BF16
   candidate first. Keep checkpoint, features, actual sampler/MSA draws,
   schedule, TF32 and kernels fixed. Record unavoidable deviations separately.
3. Permit BF16 as a FoldJAX candidate only if all measured upstream precision
   controls pass. A failure keeps FP32 for the tested model/profile; missing
   evidence is not a pass. This is panel-bounded, not universal chemistry proof.
4. Then compare independent-input FoldJAX against the chosen upstream mode.
   For an optional BF16 candidate, also check FoldJAX directly against native
   default FP32: two individually small errors must not evade the total limit.
5. Input and realized RNG identity are prerequisites, not coordinate
   tolerances. Fixed leaf `atol=rtol=1e-4` remains separate. Record repeated
   same-mode runs when needed to distinguish precision effects from runtime
   variability; do not loosen thresholds to absorb a failed repeat.

Native-first controls may share/capture native input for isolating precision;
that is not independent FoldJAX preprocessing evidence. Equal integer seeds
alone never count as a fixed tape. Independently captured runs are admissible
only after the full realized sampler and MSA tapes compare exactly; otherwise
implement explicit native replay before drawing a precision conclusion.

Initial FP32-native panel: OpenDDE released and OpenFold3 0.5/OpenBind,
on 5SAK, 1URN and 3GCA where supported. BF16-native port targets: AlphaFold3,
Boltz-2, ESMFold2 and Protenix (profile-specific). ESMFold2's FP32 checkpoint
does not imply FP32 execution: pinned CUDA forward enables BF16 autocast in
the trunk, conditioning transitions and confidence trunk. Earlier FP32-native
classification is withdrawn; loader dtype and operator policy stay separate.
No shipped dtype
defaults are changed by this protocol or by an incomplete experiment.
