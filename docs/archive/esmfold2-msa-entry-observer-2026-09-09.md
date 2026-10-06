# ESMFold2 actual MSA-entry observer

The next discriminating control compares the actual first-loop MSA pair and
input embedding, not separately prepared standalone inputs. The optional
`--capture-msa-inputs` requires `--capture-injection` in `bench.esmfold2_tape`.
Native capture clones the first keyword inputs in a forward pre-hook; JAX
capture exposes the existing scan body's arguments without duplicating its
arithmetic. The report requires symmetric complete captures and validates
the embedding as rank 3 and pair tensors as rank 4.

Verification: focused injection capture, injection report, and tape report
tests: 26 passed on CPU. Tests cover eager/JIT observation preserving the toy
scan result, first-call capture, cloned native inputs, hook restoration, and
rejection of asymmetric or incomplete input captures.

Not verified: real-weight native/JAX equality of these newly captured entry
inputs, observer neutrality in the real compiled model, full-model admission,
release gates, commit, and push. These tests do not establish numerical model
parity. Do not change tolerances or infer closure from this instrumentation.

## Real-weight follow-up

Jobs 788 (native) and 789 (JAX) completed. Artifacts are under the external
`esmfold2-msa-entry-kEwW4I` run directory; `injection-comparison.json` passed
the native/reference input, tape, config, checkpoint, source and archive
bridge checks and candidate full-output binding checks.

The actual first-loop MSA entry pair already differs: RMSE
0.005227149506430628, maximum absolute difference 0.25, 11,189,184 unequal
values. Both entry pairs are BF16. The native entry embedding is FP32 and
the JAX entry embedding is BF16; after explicitly rounding native to BF16,
1,684 values still differ (RMSE 0.0004913949449131686, maximum 0.03125).
These are representation differences, not coordinate RMSDs in angstroms.

Refined LM output is bitwise equal. MSA output RMSE is 5.9628583749348305,
maximum 256. The entry mismatch rules out treating this as a controlled
same-input full-model MSA comparison. Next localization belongs to input
embedding and pair initialization; it does not yet prove their sole causality
for the final structural error.

Verification: the JAX coordinates with entry observation are bitwise equal
to job 787 coordinates. The focused MSA/injection/repeat/tape regression set
passed 91 CPU tests. Native observer neutrality and real-model intermediate
observer neutrality remain unproven; no full-model admission or release claim.

## Historical standalone reference cross-check

The recorded archive hashes for `esmfold2-pair-init-result-20260908-BMsc6N`
and `esmfold2-atom-native-result-20260908-pLSpVe` still match their files.
Against job 788's actual native MSA entry, the historical native standalone
embedding baseline differs at two values (maximum 0.015625, RMSE
0.00004977440009715625). Its historical standalone pair differs at 124,716
values (maximum 0.125, RMSE 0.0005722697743756867).

This is an artifact comparison, not a newly controlled causal experiment:
the historical execution context and current full execution differ. It shows
why standalone exactness cannot be promoted to full-execution exactness.
The full native/JAX pair discrepancy remains larger, but subtracting these
RMSEs would not estimate a port-specific residual. Use the existing
`--native-input-embedding` control to separate embedding propagation from
pair arithmetic under the full candidate graph before changing runtime code.

Job 790 launches that control from the same source snapshot as job 789,
retaining fixed tape, native LM/shim, strict compiler profile, FFI and two
forwards. Its output is `esmfold2-msa-entry-kEwW4I/core-native-input`.
The substituted embedding is the validated historical native `baseline`,
not job 788's observed entry; the two-value native discrepancy above therefore
remains a limitation of this control. At submission this job is running;
no numerical outcome is recorded yet.

### Job 790 outcome

Completed; `native-input-comparison.json` passed the reference bridge and
full-output provenance checks. Substituting the historical native embedding
reduces actual pair-entry RMSE from 0.005227149506430628 to
0.0005722697743756867, with maximum 0.125. The latter matches the historical
native/current-native pair discrepancy above. MSA-output RMSE decreases from
5.9628583749348305 to 3.3375914816603665, but does not vanish.

Final coordinate parity does not improve monotonically: maximum entity RMSDs
are protein 1.1797614066491913 angstrom and ligand 0.7738824760022808 angstrom.
Strict confidence remains false. This is a diagnostic substitution, not an
accepted runtime change. Smaller early representation errors do not establish
smaller final coordinate errors, and this result does not justify changing
pair arithmetic or relaxing gates. The remaining native entry discrepancy
must be removed in a truly matched-input control before attributing the
remaining MSA divergence to its implementation.

## Current observed native embedding control (job 791)

The new diagnostic loader accepts the actual full native capture, verifies its
reference/config/checkpoint/source bridge and archive identities, and records
the exact embedding consumed. Job 791 completed under
`esmfold2-current-native-input-oJKmup`; `comparison.json` passes the full
reference bridge and output checks.

With job 788's observed embedding substituted, first-loop pair entry, MSA
output, refined LM, injection input, injection norm output and folding-trunk
entry all match exactly (RMSE/max zero). Raw embedding comparison still sees
native FP32 versus candidate BF16 storage boundaries; effective BF16 input is
checked separately and is bitwise equal after native FP32-to-BF16 conversion.
This closes only the observed first-loop MSA/injection
control, not independently computed embeddings or later computation.

Final per-sample entity RMSD in angstroms:

- Protein: 0.8211128861698757, 0.09836247922925857,
  0.11909471566895022, 0.14399274390105396, 0.10011803129105722.
- Ligand: 0.5876828023027267, 0.031530477729081546,
  0.0278508687004071, 0.028131361822760537, 0.026378765002246996.

Strict confidence remains false. Therefore at least one downstream comparison
is still required despite exact first folding-trunk entry. Do not modify MSA
arithmetic on the strength of the previously mismatched-input control.
Verification: job 791 completed, report provenance checks passed, focused
input/injection/report/repeat CPU suite 64 passed. No model admission or push.

## First folding-trunk output localization

The extended entry-capture option now also saves the first folding-trunk
output. The comparison retains compatibility with historical captures lacking
that output, but refuses an asymmetric pair of captures. CPU tests verify
first-value cloning, hook cleanup, eager/JIT toy-scan output preservation and
symmetric report coverage: 29 passed.

Jobs 792 (native) and 793 (candidate) use one source snapshot under
`esmfold2-trunk-output-9K76Fd`. The candidate consumes the embedding from job
792 directly, retaining the original reference tape/LM/shim bridge, n=5 and
two forwards. At this record update native is running and candidate queued;
neither a trunk-output result nor full-model admission is claimed.

Remaining independent requirements include native-free embedding fidelity,
later-loop/trunk/diffusion/confidence fidelity, the finite modality panel,
ordinary-RNG and uninstrumented performance evidence, independent review and
release gates. Passing the teacher-forced first-loop control does not satisfy
those requirements.

Native job 792 completed and its raw-output schema and injection archive hash
were verified. Compared with job 788's native capture, embedding differs in
two values (maximum 0.03125), pair entry RMSE is 0.0007062647522059277 and
trunk entry RMSE is 0.059989156992173845. This cross-run comparison does not
separate observer effects from native variation. Job 793 consumes job 792's
embedding specifically; do not mix job 788 boundaries into that comparison.

### Jobs 792/793 completed

`esmfold2-trunk-output-9K76Fd/comparison.json` passed the reference bridge and
full output binding checks. First-loop pair entry, MSA output, refined LM,
injection input/output and folding-trunk input/output all have zero RMSE and
zero maximum difference. Thus the first folding-trunk output is also exact in
this matched native-embedding diagnostic. This does not close later recycles
or the independent embedding path.

Final maximum entity RMSDs remain protein 1.2670990942934344 angstrom and
ligand 0.37519853902042066 angstrom; strict confidence fails. These final
metrics use the original full-output reference bridge, not a newly claimed
native-repeat calibration. Next distinguish later-loop divergence from
downstream diffusion/confidence divergence by comparing the final trunk
representation. Do not change first-loop trunk arithmetic based on this run.

### Paired final-output reference correction

`comparison-paired-native.json` now retains two explicit final-output arms:
the original reference (`core`) and the actual native capture supplying the
embedding/boundaries (`paired_native_outputs`). Against job 792 directly,
job 793 maximum entity RMSD is protein 0.15711457400594878 angstrom and ligand
0.13023029929463095 angstrom. The earlier 1.267/0.375 maxima include the
difference from the original native reference and must not be presented as
the matched-native port residual. Both reports remain available; no historical
number was silently replaced. Full-model admission remains false.

Verification: regenerated paired report passed provenance checks; affected
report/coordinate tests 17 passed. This fixes comparison interpretation, not
runtime arithmetic or the remaining independent-input requirement.

## Final-recycle trunk control pending

The capture now retains `trunk_output_last` in addition to the first output.
Native updates a cloned final-call value; JAX selects the final existing scan
output without reimplementing recurrence. First/last selection and symmetric
comparison are CPU-tested. Jobs 794/795 are serialized under
`esmfold2-last-trunk-X3uWok`; the candidate consumes the same run's native
embedding. Native is still running at this update, candidate queued.

The paired-output report routing test additionally verifies that coordinates
and confidence are read from the observed-native capture, not implicitly from
the older tape reference. The focused report tests pass (19 tests). No final
trunk result or admission is asserted while these jobs remain incomplete.

### Jobs 794/795 completed

`esmfold2-last-trunk-X3uWok/comparison.json` passed the reference bridge and
output validation. Both first and final folding-trunk outputs match exactly
(RMSE/max zero) under the same native-embedding diagnostic. Intermediate
recycles are not separately observed, so do not claim every intermediate
value is proven equal. The final trunk endpoint does exclude an observed
endpoint discrepancy as the explanation for downstream differences here.

Against this run's paired native final outputs, maximum entity RMSD remains
protein 0.18490449074348525 angstrom and ligand 0.1331933850058061 angstrom;
strict confidence is false. Next compare `parcae_readout` and `parcae_coda`
before attributing residuals to diffusion. Both source implementations place
these operations between final recycle output and the FP32 diffusion input.
No runtime arithmetic change or full-model admission follows from this control.

The pinned checkpoint confirms that `parcae_readout` cannot be skipped as an
identity: its FP32 256-by-256 weight differs from identity by maximum
0.9027393460273743, despite identity initialization in native source. The
configured coda has two layers (36 checkpoint parameter tensors). Existing
ESMFold2 benchmark scripts do not yet expose this post-recycle boundary.
Consequently the next useful observation is the coda input (readout output)
and coda output, not a premature diffusion-only diagnosis.

## Coda boundary capture pending

`--capture-coda` requires injection capture and records the actual coda input
(readout output) and output, with a native single-call count and JAX
missing/duplicate-capture checks. No model arithmetic is duplicated. Jobs
796/797 use the same source snapshot in `esmfold2-coda-pr0Lle`, with the
candidate consuming that run's native embedding. At submission native is
running and candidate queued. No coda numerical result is asserted yet.

Native job 796 completed with output-schema/archive validation and exactly one
coda call. Native final trunk, coda input and coda output are finite BF16
tensors of shape `(1, 437, 437, 256)`. Job 797 is running and emitted a CUDA
delay-kernel timer accuracy warning; its timing must not be used as performance
evidence. Numerical comparison still awaits its finalized manifest.

### Jobs 796/797 completed

`esmfold2-coda-pr0Lle/comparison.json` passed provenance and output checks.
Last folding-trunk output, coda input (readout output), and coda output are
all bitwise equal. The paired-native final maximum entity RMSDs nevertheless
remain protein 0.2040475823304603 angstrom and ligand 0.1533300085215465
angstrom; strict confidence fails. This bounds the remaining observed
coordinate divergence downstream of the matched coda endpoint, without
proving that every other diffusion cache/input is identical. Compare those
inputs before changing sampler arithmetic. Independent embedding fidelity is
still an open, separate requirement. Timing remains excluded for this run.

## Pair-conditioning precision boundary candidate

Source inspection identifies a concrete boundary mismatch in
`diffusion.condition_pair`: FoldJAX casts the FP32 pair residual and all
`z_transitions` parameters, including LayerNorm affine parameters, to
`trunk_dtype` before `transition_layer`. Pinned native
`DiffusionConditioning.forward` instead passes its FP32 residual into each
`TransitionLayer` under BF16 autocast; that layer normalizes before its
Linear projections. Thus blanket BF16 casting does not preserve the native
LayerNorm boundary. This is source evidence of a precision-policy mismatch,
not yet a measured causal attribution for final RMSD. Next use a matched
transition input to verify native norm/output dtypes and compare corrected
FP32-normalization/BF16-linear arithmetic before admission.

### Runtime precision-boundary correction under test

`transition_layer` now optionally casts only post-normalization operands and
projection parameters through `linear_dtype`. `condition_pair` passes its
unrounded FP32 residual and original normalization parameters into that path.
The ordinary transition path remains unchanged. Three focused CPU tests cover
FP32 norm inputs/affines, narrowed Linear operands, original parameter
preservation and both pair-transition calls.

The primitive/diffusion native parity test modules skipped in the CPU-only
environment because their optional Torch/upstream imports were unavailable;
they are not counted as passing native comparisons. Full ESMFold2 CPU tests
are running. Job 798 (`esmfold2-pair-norm-fix-KtNOwT/core`) compares the actual
runtime correction using job 796's native embedding, the same tape and n=5
with two forwards. No structural improvement is claimed before its result.

Full ESMFold2 CPU regression completed after the precision-boundary change:
405 passed, 24 skipped in 133.97 seconds. The skipped checks are not native
parity evidence. Job 798 remains running at this update; runtime admission
still requires its measured output and further native comparison/review.

### Job 798 measured result

Completed with the same native 796 embedding/reference and passed the report
provenance checks. Coda output remains bitwise equal. Against paired native
outputs, maximum protein RMSD decreases from 0.2040475823304603 to
0.11370652225559758 angstrom; maximum ligand RMSD decreases from
0.1533300085215465 to 0.08966550914712709 angstrom. Per-sample protein RMSDs
are `[0.11370652225559758, 0.01952732839147694, 0.0561211618523254,
0.01432339243050241, 0.01668289230226987]`; ligand RMSDs are
`[0.08966550914712709, 0.0034389643480691695, 0.025841828710214353,
0.004186555522640327, 0.0050682143232015095]`.

Strict confidence still fails. This supports the precision correction's
effect on this diagnostic, not sole causality or full-model admission. Protein
maximum remains above 0.1 angstrom, ligand maximum is deferred gray-zone, and
the independent embedding path is not covered by this teacher-forced control.

Additional source audit: single diffusion conditioning does not use the same
blanket early BF16 cast. Confidence does: `heads.confidence_head` narrows the
pair and all folding-trunk parameters and calls the non-native-autocast trunk.
Pinned native passes the FP32 pair directly under a BF16 autocast context and
then adds its output back to the FP32 pair. This is a separate confidence
precision boundary candidate; it cannot explain already generated coordinate
differences. A read-only independent consultation on the correction and these
remaining policy boundaries is running; no review verdict is claimed yet.

Qualification from deeper native-source tracing: `FoldingTrunk.forward`
itself casts CUDA pair inputs to BF16 when its first block uses BACKEND_FUSED,
then restores the original output dtype. Other backends retain input dtype.
Thus the confidence entry cast alone is not yet a confirmed defect; effective
backend and original norm-affine handling must be checked before changing it.
Current saved native-policy metadata records matmul/SDPA settings but does not
identify this per-block backend, and config has no explicit backend field.

Pinned-config native meta construction (CUDA hidden, no inference/weights
allocation) reports all four confidence block backends as `None`, and all main
trunk backends as `None`. The capture code loads the model and ESMC, moves to
CUDA/eval, and never calls `set_kernel_backend`. Combined with the native
constructor/source trace, this supports the non-fused confidence path for
this baseline; the fused-only internal BF16 cast does not explain the port's
early cast here. This is a constructor/path check, not a retrospective
instrumented runtime backend capture for historical runs.

### Confidence residual precision correction (pending GPU validation)

The BF16 single-device confidence path now passes the original FP32 pair and
parameters to `folding_trunk(native_autocast=True)`. This follows the pinned
non-fused native trunk: only the fused backend explicitly narrows the residual
stream at entry. The existing FP32 and context-parallel paths are unchanged.
The original extra outer residual addition is preserved.

A new boundary test covers BF16/FP32 crossed with CP/non-CP, asserting original
residual values, norm parameter dtype/value and autocast selection. Together
with pair-trunk wiring and condition-pair precision coverage, 19 tests passed.
This is wiring evidence, not measured confidence parity. No new real-weight
GPU result, full-suite result, model admission or push is claimed for this edit.

The real-weight follow-up is tsp job 799, snapshot
`esmfold2-confidence-norm-fix-F433Av`, using observed native 796's embedding,
the existing bound tape, five samples and two repeated forwards. The strict
rounding/FFI/native-LM/native-shim controls are unchanged from job 798.
Submission and running state were verified; results are pending. The ESMFold2
CPU suite is also running. Neither launch is a passing numerical gate.

CPU completion: `JAX_PLATFORMS=cpu .venv-ci/bin/python -m pytest -q
tests/models/esmfold2` passed 409 tests, with 24 skipped, in 135.39 seconds.
Skipped upstream/Torch-dependent cases are not GPU parity evidence.

The completed independent read-only consultation agrees with the non-fused
confidence residual/affine correction at source level. It identifies two
unmeasured pair-conditioning hypotheses: BF16 SiLU intermediate rounding and
missing materialized BF16 projection boundaries. These have not been changed
in job 799. Its proposed discriminating check is a fixed-input native
TransitionLayer comparison against current, FP32-opmath-SiLU, and
SiLU-plus-native-linear arms. Source inspection also confirms that native
pair-conditioning autocast is not controlled by FoldJAX's `trunk_dtype`;
FoldJAX FP32 remains a distinct intervention, not native-policy equivalence.
The obsolete single-autocast claim in the settings comment was corrected and
the historical engineering account explicitly marked superseded for this pin.

Job 799 emitted CUDA delay-kernel timer warnings; do not use its timings as
uninstrumented runtime or memory performance evidence.

### Job 799 completed: mixed confidence outcome, not closure

Job 799 exited 0. The hash-bound `comparison.json` in
`esmfold2-confidence-norm-fix-F433Av` was generated successfully against native
796, retaining the older tape reference comparison separately. Coda output is
still bitwise equal. Same-executable repeated coordinates and confidence are
finite and storage-byte identical (two forwards).

Paired whole-system-aligned entity RMSDs in angstrom:

- Protein: `[0.1211142359843387, 0.01867801618990248,
  0.054066087932952776, 0.015123279352824929, 0.016412202641177234]`.
- Ligand: `[0.09723885639765831, 0.002649267343458968,
  0.024990893160133313, 0.004829023485248919, 0.006139874881291937]`.

Confidence RMSE versus the same native capture, job 798 -> 799:

| Leaf | Before | After |
| --- | ---: | ---: |
| plddt (native 0–1 scale) | 0.002162277983206589 | 0.0018166330557643946 |
| complex_plddt | 0.00017409155634744305 | 0.00011431158662900854 |
| ptm | 0.0002504333124503301 | 0.00010123831982945165 |
| iptm | 0.0009858635442683567 | 0.0010287799875192947 |
| pae_logits | 0.04227774947073696 | 0.04250668336954514 |
| pde_logits | 0.03132274910923311 | 0.03141741256467573 |

Strict confidence still fails. Coordinates also changed despite the
confidence-only source edit, so this whole-program recompilation does not
isolate the head's numerical effect: confidence consumes predicted distances.
Compiler/reduction effects are a hypothesis, not a measured cause. A fixed
coordinate and fixed head-input replay is needed before attributing remaining
confidence differences to this correction. Protein maximum remains above
0.1 angstrom; ligand maximum is deferred gray, not passed. No model admission,
independent-input closure, performance claim or push follows from this run.
