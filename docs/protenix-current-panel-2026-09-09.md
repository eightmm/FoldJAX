# Protenix panel and MC-dropout investigation

## Current decision checkpoint

- Latest complete CPU rerun after dtype-spelling and input-encoder coverage
  changes: `tests/models/protenix` passed 798 tests, skipped 4, in 192.47 s.
  Separate OpenBind adapter plus Protenix capture/replay/dropout/control set:
  71 passed. These supersede earlier CPU counts, not GPU parity findings.
  GPU-only and unavailable pinned Kalign skip limitations remain; no release
  admission or commit/push is established by these checks.
- Review coverage follow-up: the input precision test module now passes 10 CPU
  tests. It checks every prepared projection/LayerNorm parameter subtree and
  executes the real toy input encoder under JIT with scan enabled and disabled.
  Atom-module geometry projections realize FP32 input/output, ordinary
  projections realize BF16 output, and concatenated inputs remain FP32. This
  does not yet observe every internal transformer LayerNorm or prove GPU parity.
- Independent read-only review completed. Reproduced its dtype-spelling defect:
  `cast_trunk_params(..., "bfloat16")` bypassed the mixed encoder route and
  narrowed original geometry weights. Normalizing with `jnp.dtype` fixes this;
  regression failed before the fix (1 failed, 2 passed), then the input-precision
  and explicit-linear tests passed (9 passed). Scoped Ruff and diff checks pass.
  Other review coverage requests remain open; this is not review admission or
  new GPU parity evidence. The conversion's original-FP32 precondition and
  non-idempotence are now explicit in its docstring.
- Implemented, not yet admitted: native input-encoder mixed precision, including
  the prepared CLI loader. Original FP32 geometry/projection/LayerNorm values
  survive; ordinary encoder projections use explicit BF16 autocast.
- Corrected-loader real-weight 7R6R run 961: A/B/D maxima
  1.049986/0.873503/0.841028 A; strict confidence fails. This improves the
  historical baseline below but does not close the model. 3V7E run 972 has
  P/R/L maxima 0.843136/1.399272/1.354729 A and does not improve the baseline;
  no uniform updated seven-case result exists yet.
- Rejected as default: FP32 atom-aggregation override (3V7E not consistently
  improved), and highest precision as a substitute for native high policy.
- Partial implementation: actual MC-dropout mask capture/replay is implemented;
  ordinary MC decision/default generation and device-consumption proof remain.
- Verification on the loader-corrected tree: Protenix CPU suite 791 passed,
  4 skipped; capture/replay/dropout/control tests 55 passed. Changed-file Ruff
  passes. Skips retain GPU-only and unavailable pinned Kalign coverage limits.
- No recent patch admission, independent review, commit/push or model closure.
  Earlier historical measurements below must not be relabeled as results from
  the corrected implementation.

## Measured panel summary (not admission)

These are n5 native-policy fixed-tape diagnostics for base_default_v1.0.0.
All independent input comparisons pass; all strict confidence comparisons fail.
3GCA retains its earlier explicitly identified source/capture; other rows use
the newer snapshots documented below. This is not one uniform new-source
seven-case release run, nor general native-default or warm-performance proof.

| Case | Maximum entity RMSD (angstrom) | Structural triage |
| --- | --- | --- |
| 1UBQ | A 0.110220 | investigate |
| 5SAK | A 0.065820; L 0.034383 | gray; MC dropout tape |
| 1URN | P 0.031236; R 0.011259 | below 0.05 |
| 3GCA | R 0.064100; L 0.030735 | gray; earlier source |
| 7R6R | A 3.403698; B 2.739171; D 2.617449 | investigate; MC dropout tape |
| 3V7E | P 0.835259; R 1.329732; L 1.193529 | investigate |
| 7ST3 | A 3.321247; B 0.313937 | investigate |

Gray means deferred under the user's 0.05–0.1 policy, not a strict pass. Input
agreement and explicit host-entry tape do not prove device consumption or
remove observer/compiler effects. The current optional dropout implementation
does not yet complete ordinary MC generation/default-policy integration.

## Original 3GCA aggregation

CPU-only aggregation of completed tsp 802. No GPU job or model source edit
was made during this aggregation; the concurrent Claude Boltz MSA job 831
and its source snapshot were left untouched.

Candidate: external `public-other-models-20260909-4S8RAo/protenix-3gca`.
Reference: `protenix-closure-20260907/3GCA/native-r0b`.
Checkpoint: `protenix_base_default_v1.0.0.jax`; the input directory's legacy
`protenix-v2` name is not the checkpoint selection.

The existing `native_record`, `check_inputs`, `canonical_foldjax`,
`compare_entity_parity` and `compare_arrays` adapters were used directly.
Independent input comparison passes. Five samples retain their original order;
one whole-system alignment is used per sample with no entity refit.

| Entity | Per-sample RMSD (angstrom) | Maximum |
| --- | --- | ---: |
| RNA R | 0.06410044, 0.02767160, 0.01152783, 0.01629140, 0.02169107 | 0.06410044 |
| Ligand L | 0.01533241, 0.01963973, 0.00721899, 0.03073523, 0.01011131 | 0.03073523 |

Strict confidence fails overall. Maximum absolute differences in selected
canonical fields are atom pLDDT 0.001530766487121582 (native 0–1 scale),
summary pTM 0.00008690357208251953 and summary ipTM 0.000024974346160888672.
Summary agreement does not substitute for all raw confidence leaves.

RNA remains deferred gray, not admitted. This aggregation does not establish
performance, a full current-panel pass, candidate artifact rehash completion,
or independently observed device tape consumption. No commit/push or release
is claimed.

## Native panel expansion started

The reviewed closure reference inventory currently covers 3GCA only. A fresh
source copy, `protenix-panel-20260909-KzEzSW`, isolates the next native/candidate
captures from concurrent Boltz/OpenBind edits. Job 918 captures native 1UBQ
into `protein_1ubq/native` using the existing local publisher checkpoint and
CCD assets. The capture validates native n5/200-step/10-cycle/seed101 defaults
and records native stochastic draws; it makes no precision override.

This is a reference acquisition, not a completed comparison, actual candidate
consumer proof, repeat calibration, or warm benchmark. Candidate execution must
follow successful native completion and independent input validation. The
single-slot queue preserves collaborator job 917 ahead of this job and does
not interrupt the running OpenDDE repeat 916.

Verification: 53 current-source Protenix closure capture/replay/report tests
passed in 0.74 seconds under `JAX_PLATFORMS=cpu`. Native checkpoint and CCD
paths were checked locally; GPU 918 is queued and its result remains pending.

### 1UBQ native capture completed

918 exited 0. `capture-complete.json` passes: five samples, 200 steps,
ten MSA cycles, one initial draw, 200 churn/rotation/translation draws each,
and no applied MC dropout. BF16 captured values are numerically widened to
FP32 with original tree dtype metadata retained; this is storage mapping,
not a precision override. These are instrumented results, not warm timings.

Candidate job 919 uses this reference, the same source snapshot and audited
managed checkpoint. The replay independently featurizes inputs, requires input
equality before inference, and supplies native sampler/MSA tape. It explicitly
matches native diffusion fusion and uses materialized MSA cycles; that route
must not be silently equated with ordinary CLI index-tape performance. No
3GCA-specific repeat calibration is reused for 1UBQ. Candidate comparison
and actual device-consumer equality remain unverified until further evidence.

Verification: native completion counters and tsp exit status were inspected;
919 was submitted with a separate output directory through the GPU queue.

### 1UBQ candidate completed: not admitted

919 exited 0. Re-running `check_inputs` against independently generated
`foldjax-input.npz` passes. `native_record` and `canonical_foldjax` validate
returned output schemas before the existing entity/confidence comparisons.
Five system-fit chain A RMSDs are `[0.01994682,0.01221164,0.01308016,
0.11021986,0.03797622]` angstrom (602 atoms). Sample 4 exceeds the user's
0.1 angstrom gray-zone upper bound; it is not admitted.

Canonical confidence has no missing/extra leaves and matching canonical
dtypes but fails strict atol=rtol=0.0001. Maximum atom pLDDT delta is
0.00260323286 (0–1), pTM 0.00019496679 and ipTM 0. The first two fail;
zero ipTM for a single chain is not whole-confidence evidence. These values
are not repeat-calibrated and no causal port diagnosis follows from one pair.
The capture supplies native schedule and materialized MSA tape at the host
wrapper; actual candidate device consumption and an unobserved bridge remain
outstanding. No warm performance result follows from the job elapsed time.

Verification: GPU exit 0, input gate and canonical adapters passed; completed
coordinate/confidence arrays compared directly. Structural and confidence
failures retained, with no model arithmetic or threshold changes.

### Remaining native panel queued

After the successful 1UBQ capture, five additional single-job input JSONs
were checked and native captures submitted with the identical isolated source,
checkpoint, assets and default-policy capture command:

| Case | Native tsp job |
| --- | ---: |
| 5SAK | 922 |
| 1URN | 923 |
| 7R6R | 924 |
| 3V7E | 925 |
| 7ST3 | 926 |

Each writes to its own case directory under `protenix-panel-20260909-KzEzSW`.
The running OpenDDE native repeat 921 retains queue priority. These are pending
reference acquisitions, not completed candidate comparisons; each completion
record and tape must be checked before candidate replay. The 1UBQ residual
remains unresolved. Historical 3GCA is retained separately with its own source
provenance rather than silently relabeled as part of this fresh snapshot.

Verification: five input files parse as one-job native inputs; queue submission
returned jobs 922–926. No new scientific pass is claimed from submission.

### 5SAK exposes a missing native inference branch

922 exited 1: `native selected MC dropout; mask tape is not captured`.
The later incomplete-prediction exception is a consequence. Native base config
sets both MC-dropout application probability and dropout rate to 0.4. When
selected, native Protenix applies functional dropout to the projected recycled
pair state before adding z_init on each cycle. Current FoldJAX
`models/trunk_blocks/trunk.py::recycle_embeddings` explicitly omits dropout.
This is a capture-coverage gap and a missing native inference branch, not a
GPU allocation failure. Earlier successful non-dropout runs do not establish
universal native-default fidelity.

Required follow-up: capture and replay the native decision and actual cycle
masks, preserving dtype, scaling and operation order through public/scan paths.
Preserve OpenDDE's no-dropout default in the shared trunk. Do not switch seed
or disable dropout merely to obtain a passing capture. 923 (1URN) completed
with dropout not applied; its completed n5 tape is available for replay.

Verification: primary traceback, pinned native base config and recycling
forward, and FoldJAX recycling code inspected. No model edit or workaround.

### Fixed-mask recycling primitive started

Current `recycle_embeddings` now accepts an optional boolean pair keep mask
and static dropout rate. It scales the projected pair update before the z_init
residual, leaves the single update unchanged, and preserves the exact original
no-mask branch. Shape, boolean dtype and rate validation reject invalid tapes.
This is only the low-level implementation step: cycle-scan/public plumbing,
native decision/mask capture, ordinary RNG generation and real-weight BF16
operator parity remain outstanding. In particular, FP32 scaling before the
BF16 cast still needs comparison to the native CUDA dropout kernel. No default
policy change or full MC-dropout support is claimed.

Verification: 15 trunk/recycling CPU tests pass, including JIT fixed-mask
scaling, dropped residual preservation, unchanged single output and malformed
mask/rate rejection. Ruff check and diff check pass. Job 924 (7R6R) failed on
the same unsupported native MC-dropout branch; it is not silently retried with
a different seed.

### Cycle tape plumbing implemented at trunk entry

`pairformer_output_from_s_inputs` now accepts boolean keep masks with exact
shape `[num_recycles, *pair_shape]`. Scanned execution consumes mask and MSA
cycle entries together; unrolled execution indexes the corresponding mask.
The existing no-mask scan branches are preserved. Invalid cycle counts, shapes,
dtypes and rates fail before execution. This remains opt-in plumbing, not a
change to ordinary prediction defaults or completed native MC-dropout support.

Verification: 21 trunk/recycling/dtype CPU tests pass in 14.77 seconds,
including one-cycle and three-cycle JIT versus unrolled mask replay. Ruff and
diff checks pass. Real native mask capture, public inference/cache signatures,
ordinary RNG generation, MSA-plus-mask route coverage and GPU BF16 parity are
still required. Native panel jobs 925/926 exited 0; their candidate comparisons
are not yet performed.

### Prediction wrapper and compiled graph connected

The optional cycle keep-mask tensor now travels from `protenix_predict_static`
through eager/compiled inference to the trunk. Masks remain dynamic inputs;
the dropout rate is an explicit graph-static argument so changing its value
cannot reuse a graph with a stale scale. This is the model prediction wrapper,
not yet the top-level model-neutral request or ordinary CLI stochastic policy.

Verification: 15 prediction-wrapper/recycling tests pass in 21.82 seconds,
including compiled mask replay alongside both compact-index and materialized
MSA cycles and both trunk scan settings. The first test run failed twice
because the new test constructed masks using the input rather than output
weight axis; the fixture was corrected and the full affected pair rerun.
Ruff and diff checks pass. Native mask capture, native CUDA BF16 rounding
comparison and ordinary RNG/default-policy integration remain unfinished.

### Native CUDA mask observation validated on a small tensor

New benchmark-only `bench/protenix_dropout_tape.py` observes the actual
`aten.native_dropout` return mask without replacing the operator. It requires
CUDA boolean masks, matching shapes, the expected probability/training mode
and an exact event count. Missing/extra events fail closed. Torch imports stay
lazy and outside the installable package.

GPU probes 928/929 exited 0. For FP32 and BF16, a 1024-element p=0.4 dropout
had identical outputs and post-call CUDA RNG state with and without observation.
The captured mask reproduced native values exactly using FP32 reciprocal
scaling then casting to the original dtype (maximum error 0 in this probe).
This supports the implemented scaling order but is not a JAX CUDA comparison,
full-model observer bridge, or evidence for every shape/layout/nonfinite value.
Integration with native full-model capture and replay remains outstanding.

Verification: live CUDA probe assertions passed for both dtypes; Ruff and
diff checks pass. CPU functional dropout uses a different decomposed operator
path and is not treated as a supported mask-capture route.

### Full native capture integration

Native capture now observes the actual dropout masks only around a selected
MC-dropout trunk call. Completion requires exactly one mask per recycle, a
valid rate, consistent nonempty square pair dimensions and boolean storage;
the non-dropout branch rejects unexpected masks. The separate
`dropout-tape.npz` stores actual keep masks, not native trunk activations.
Legacy no-dropout captures remain unchanged in semantics.

Verification: 24 capture tests pass, including complete mask persistence and
count/dtype/shape/rate/branch rejection; Ruff and diff checks pass. A fresh
snapshot `protenix-dropout-20260909-sOjlTQ` was created and native 5SAK capture
submitted with the original seed101 and native defaults. Earlier failed output
is preserved. Full GPU completion, FoldJAX replay/report acceptance and
whole-model observer bridge are pending; no native-default closure is claimed.

### Candidate dropout tape loading connected

The replay adapter now validates native dropout rate/count against the effective
config, rejects unexpected tape on the non-dropout branch, and requires boolean
`[10, token, token, channel]` masks before forwarding them to the prediction
wrapper. The concrete inference observer hashes this additional dynamic tape
argument instead of trying to serialize it as an option. No raw native trunk
state is supplied. The strict report/calibration adapter still needs an explicit
dropout provenance extension; its historical non-dropout scope is not waived.

Verification: 18 loader/replay CPU tests pass in 0.66 seconds, including legacy
non-dropout compatibility and invalid rate/count/dtype/branch rejection. Ruff
and diff checks pass. Native full-model job 930 was confirmed live; current
replay changes are not in its older immutable source snapshot and require a
fresh candidate snapshot when native capture completes.

930 subsequently completed with exit 0 and a passing completion record:
MC dropout applied, rate 0.4, actual decision draw 0.06915902659, ten mask
calls and boolean mask shape `(10, 437, 437, 128)`. The n5 sampler completed
200 steps with all expected MSA/initial/churn/rotation/translation events.
This closes the native mask-acquisition failure for this case, not candidate
parity or the whole-model observed/unobserved bridge. No seed was changed.

### Fixed-mask candidate replay submitted and native record extended

931 is the first 5SAK FoldJAX replay with actual native dropout masks, using
fresh source `protenix-dropout-replay-20260909-K563n0` and native capture 930.
Independent input reconciliation still gates inference. No intermediate native
representations or sample permutation are introduced.

The native-record adapter now accepts a fully validated dropout tape and binds
its content hash and rate into the comparison target, including the tape file
in the artifact hashes. The historical no-dropout target encoding is unchanged.
This prevents runs with different masks from sharing a fixed-tape calibration
target. It does not automatically calibrate or admit the candidate.

Verification: 29 report/loader tests pass; the adapter successfully generated
`native-record.json` from the complete 5SAK real-weight capture. Ruff and diff
checks pass. 931 is live; its output and parity remain unverified.

The dropout loader additionally checks the actual decision draw against the
configured application probability, rejecting out-of-range/nonfinite draws
and contradictions before loading masks. 46 loader/replay/report tests pass
in 1.28 seconds; Ruff and diff checks pass. This validator refinement is newer
than the immutable 931 snapshot and must also be applied to its artifacts
before admitting a comparison.

Native job 932 was submitted for 7R6R with the successful actual-mask capture
snapshot `protenix-dropout-20260909-sOjlTQ`, original seed101 and native defaults.
It writes separately from failed 924 and queues behind live 931. No 7R6R
completion, candidate result or performance claim follows from submission.

### 5SAK actual MC-dropout replay completed

931 exited 0. The latest decision/config/mask validator accepts reference 930;
independent input reconciliation passes. The native boolean mask content hash
matches the concrete candidate inference-entry hash exactly. That is host-entry
evidence, not separately observed device-consumer or whole-model bridge proof.

System-fit entity RMSDs over five samples (angstrom):

| Entity | Per-sample RMSD | Maximum |
| --- | --- | ---: |
| Protein A | 0.04437450, 0.06582005, 0.03866408, 0.04118846, 0.02044300 | 0.06582005 |
| Ligand L | 0.01900326, 0.03438339, 0.02726535, 0.02227742, 0.01564430 | 0.03438339 |

Protein is deferred gray, not a strict 0.05 pass. Strict confidence fails:
maximum atom pLDDT delta 0.00368797779 (0–1), pTM 0.00002384186 and
ipTM 0.00041317940. No sample reorder, entity refit or native intermediate
injection was used. This establishes a working explicit dropout-tape path,
not full native-default support: ordinary MC decision/mask generation and the
model-neutral request path remain unfinished. The result is not a warm timing.

Verification: completion exit, latest native-record adapter, candidate input
gate, canonical structure/confidence outputs and host-entry mask hash checked.
The full strict gate remains failed; no closure, commit or push is claimed.

### Remaining candidate panel submitted

Latest dropout validation accepts completed native 7R6R mask shape
`(10, 245, 245, 128)` with MC dropout applied. Native 1URN, 3V7E and 7ST3
completion records pass n5/200 steps and select the no-dropout branch.
Using the same candidate source as 931, new replays are:

| Case | Native source result | Candidate tsp |
| --- | --- | ---: |
| 7R6R | dropout snapshot, native 932 | 934 |
| 1URN | panel snapshot, native 923 | 935 |
| 3V7E | panel snapshot, native 925 | 936 |
| 7ST3 | panel snapshot, native 926 | 937 |

All write separate case outputs under `protenix-dropout-replay-20260909-K563n0`.
The collaborator's live OpenDDE highest-precision job 933 retains queue order.
No collaborator source or job was changed. Latest loader validation was applied
before submission even though the immutable candidate snapshot predates that
validation refinement; actual reports still require post-run revalidation.
These submissions are not passes or warm performance measurements.

Shared-path regression verification: 361 CPU tests passed in 135.83 seconds
across all `tests/models/opendde`, Protenix trunk/recycling/prediction-wrapper,
and Protenix capture/dropout-loader/replay/report tests. Two known warnings
report requested int64 truncation with JAX x64 disabled; no failures or skips.
This validates the affected CPU contracts, not GPU closure or ordinary dropout
generation. Scoped Ruff and diff checks also pass.

### Compatibility and ordinary-RNG ordering audit

The updated native-record adapter was executed against all three historical
3GCA native calibration captures (`native-r0b`, `native-r1`, `native-r2`).
Every target and artifact hash map equals its saved adapter report exactly.
This verifies preservation of the existing no-dropout calibration identities;
it does not extend those allowances to dropout or another case.

Native `runner/inference.py` seeds before iterating/featurizing the dataloader;
the model later calls Python `random.random()` for the MC decision. A fresh
`Random(seed).random()` at the FoldJAX model boundary therefore cannot be
asserted to reproduce that state. Ordinary MC generation needs its own explicit
RNG contract, and paired comparison must continue to use actual captured draws.
The current optional mask path does not implement this ordinary policy yet.

Verification: three real historical adapter target/artifact comparisons pass;
native seed placement and model MC-decision call inspected. No change to
ordinary RNG/defaults was made based on an unverified seed-equivalence claim.

Loader review found one remaining contradictory-metadata case: a disabled
dropout branch with a stale non-null dropout rate but no tape file. This is
now rejected, matching recorder behavior. The added regression and related
loader/replay/report suite pass (47 tests, 0.73 seconds); Ruff/diff checks pass.
This validation change does not alter inference arithmetic or pending jobs.

### Per-cycle key generation path

Trunk entry additionally accepts explicit uint32 `[cycle, 2]` RNG keys,
mutually exclusive with captured masks. It generates the current cycle mask
inside the recycling body rather than requiring the full cycle-mask tensor as
an input. This prepares ordinary generation without claiming a measured VRAM
benefit or changing current prediction defaults. Public propagation and the
MC decision policy are still unfinished; padded RNG equivalence is unverified.

Verification: 19 trunk/recycling tests pass in 25.50 seconds. For one and three
cycles, generated-key execution agrees with explicit masks generated from those
same JAX keys in both scan and unrolled paths; simultaneous key/mask input is
rejected. Ruff and diff checks pass. This is JAX-internal RNG equivalence, not
Torch/JAX seed equivalence. Pending GPU jobs retain their immutable code.

Per-cycle keys now propagate through the prediction wrapper and eager/compiled
inference, remaining dynamic graph inputs. Fifteen wrapper/recycling tests pass
in 22.57 seconds, including compiled key-generated masks versus explicit masks
alongside compact MSA tape. Ruff/diff checks pass. Ordinary branch selection
and model-neutral request/default integration remain unfinished.

### 7R6R dropout replay fails substantially

934 exited 0 and independent inputs pass. Latest native-record validation
accepts the native reference. Native keep-mask bytes match the concrete
candidate inference-entry hash, with rate 0.4; device consumption is not yet
separately observed.

| Entity | Per-sample system-fit RMSD (angstrom) | Maximum |
| --- | --- | ---: |
| Protein A | 3.40369838, 0.26657083, 0.17057233, 0.24851367, 0.12828646 | 3.40369838 |
| DNA B | 2.73917138, 0.22250350, 0.07603522, 0.21275487, 0.10645579 | 2.73917138 |
| DNA D | 2.61744913, 0.18361113, 0.16458186, 0.21069838, 0.07681514 | 2.61744913 |

Strict confidence fails: atom pLDDT maximum delta 0.01040044427 (0–1),
pTM 0.00031524897, ipTM 0.00015693903. This exceeds the gray zone and
prevents dropout-path closure; 5SAK's smaller result is not general proof.
Remaining hypotheses include dropout/operator rounding, trunk differences,
and downstream trajectory amplification. Host tape equality alone cannot
locate the cause; staged comparison and repeat controls remain needed.

Verification: completion, independent-input and canonical output adapters
checked; actual arrays compared without sample permutation or entity refit.
No passing admission, ordinary-policy completion or warm-performance claim.

### Stored-boundary inventory and 1URN completion

Using existing captures only, native/candidate trunk boundary differences are:

| Case | s_inputs RMS error | trunk s RMS error | trunk z RMS error |
| --- | ---: | ---: | ---: |
| 5SAK | 0.00015406658 | 0.91448116 | 0.67649200 |
| 7R6R | 0.00017633753 | 1.00060542 | 0.72608386 |

These are activation units, not structural angstrom. Both s_inputs tensors
have exact final 65 raw-feature channels (restype/profile/deletion_mean), while
the first 384 atom-embedding channels already differ before recycling dropout:
878 entries for 5SAK and 697 for 7R6R. The similar activation error magnitudes
do not identify the cause of the very different final trajectory errors.
They refute treating all residual error as first arising at dropout; they do
not prove dropout arithmetic correct or isolate trajectory amplification.

1URN candidate 935 exited 0 and independent inputs pass. Five protein RMSDs
are `[0.01081445,0.03123564,0.02118634,0.03092142,0.01084787]`; RNA is
`[0.00900383,0.01125924,0.00976772,0.01007327,0.00875993]` angstrom.
Both structural maxima are below 0.05. Strict confidence still fails:
atom pLDDT delta 0.00205230713 (0–1), pTM 0.00001758337 and ipTM
0.00017273426. This native capture selected no MC dropout.

Verification: existing native/candidate NPZ boundaries inspected and compared;
1URN native-record, input and canonical output adapters passed. No new GPU
experiment was required for these measurements; no universal closure claim.

### 3V7E no-dropout replay also exceeds tolerance

936 exited 0 and independent inputs pass. Native selected no MC dropout.
Per-sample whole-system-fit entity RMSDs (angstrom):

| Entity | RMSDs | Maximum |
| --- | --- | ---: |
| Protein P | 0.12808374, 0.29041950, 0.21198050, 0.83525889, 0.21291147 | 0.83525889 |
| RNA R | 0.13285704, 0.34472400, 0.29225124, 1.32973188, 0.30827248 | 1.32973188 |
| Ligand L | 0.07049659, 0.10391157, 0.09862250, 1.19352912, 0.17366798 | 1.19352912 |

Strict confidence fails: maximum atom pLDDT delta 0.04504024982 (0–1),
pTM 0.00346517563 and ipTM 0.00129640102. This shows that above-tolerance
residuals are not confined to the newly supported MC-dropout branch. It does
not establish one common cause: native repeat controls and staged BF16
trunk/sampler comparison remain necessary. The CUDA timer warning in this job
does not invalidate its exit/completed outputs, but no timing is admitted as
warm performance.

Verification: native-record, independent inputs and canonical arrays checked;
strict failures retained. Native 7R6R repeat 940 was queued with unchanged
capture snapshot and seed; its mask/sampler/MSA equality must be checked before
interpreting repeat variance. 7ST3 candidate 937 was confirmed running.

Native 3V7E repeat 941 was also submitted using its original
`protenix-panel-20260909-KzEzSW` source snapshot, native settings and seed101.
It writes `protein_rna_ligand_3v7e/native-repeat` without modifying the first
capture. Both native repeats remain controls awaiting output: equality of the
full comparison target, including actual tape, must precede any repeat-floor
claim. At the latest check 937 is live and 940/941 are queued, not failed.

937 subsequently exited 0. 7ST3 independent inputs and native-record/canonical
output validation pass. Native did not apply MC dropout. Chain A RMSDs are
`[0.07700662,0.22998388,0.12757842,3.32124703,0.26301978]`, B is
`[0.01484436,0.01935117,0.01714529,0.31393710,0.01467787]` angstrom.
Strict confidence fails: atom pLDDT delta 0.03080987930 (0–1), pTM
0.00059103966 and ipTM 0.00135850906. Large no-dropout residuals are therefore
not confined to 3V7E. No specific shared root cause is established yet.

Verification: completed 937 arrays compared with existing canonical adapters;
sample order and whole-system fit retained. Panel table aggregates the recorded
measurements with source-scope caveats; it does not override any failed gate.

### BF16 reduction discriminator queued

Source inspection confirms both atom encoders project/ReLU atom activations
then aggregate them to tokens. Native `scatter_utils.scatter_mean` calls
dtype-preserving `scatter_add_` for sums/counts; FoldJAX uses JAX indexed add.
Matching source-level formulas do not prove CUDA accumulation/rounding equality.

`bench/protenix_scatter_probe.py` compares sum and mean of the same exactly
representable fixed values in FP32/BF16, three repetitions, without model
weights. Native job 942 and JAX job 943 were queued behind the existing work.
The initial inline-command submissions were rejected by the queue's sensitive
text detector and did not launch; the explicit benchmark script contains no
credentials and was submitted normally. Ruff initially found import ordering,
which was corrected; syntax, Ruff and diff checks pass. GPU results are pending.
This synthetic reduction test can isolate one kernel discrepancy but cannot
by itself assign the full model's structural error to that operator.

Precision cross-check: actual native operator-policy records for 1UBQ, 3V7E
and 7ST3 all report BF16 autocast, FP32 diffusion/confidence exceptions,
`float32_matmul_precision=high`, `allow_tf32=true` and non-deterministic
algorithms. The candidate wrapper pins JAX matmul precision `high`. Therefore
the OpenDDE observation of native matmul TF32=false cannot simply be reused as
the explanation here. Matching policy labels still do not prove identical
Torch/JAX kernels or accumulation. No precision policy was changed in response
to this check; repeats and reduction probes remain queued behind live 939.

### Native-repeat and BF16 reduction results

940/941 completed successfully. The native-record comparison targets are
identical in every field between each original/repeat pair, including actual
sampler/MSA tape and, for 7R6R, dropout mask bytes and rate.

| Native/native repeat | Entity maximum RMSD (angstrom) |
| --- | --- |
| 7R6R | A 3.00320127; B 2.41126653; D 2.30973170 |
| 3V7E | P 0.26733114; R 0.51785672; L 0.25126883 |

Both native repeats fail strict confidence. Maximum atom pLDDT/ pTM/ ipTM
deltas are 0.00909460 / 0.00019783 / 0.00020391 for 7R6R and
0.00560302 / 0.00106001 / 0.00048602 for 3V7E (pLDDT on 0–1 scale).
The large 7R6R sample-1 trajectory change also exists within native itself.
This contradicts attributing the whole cross-framework residual to a port
defect, but two runs do not establish a frozen allowance or justify passing.
3V7E cross-framework maxima still exceed this observed native repeat.

Synthetic reduction jobs 942/943 also exited 0. All three FP32 sums were 2
and means 0.0077821011655 in both frameworks. For BF16, native sums were
`[1.125,1.0,1.125]`, JAX `[1.0,1.015625,1.546875]`; native means were
`[0.00439453125,0.00439453125,0.00390625]`, JAX
`[0.00604248046875,0.00494384765625,0.00390625]`. Sum and mean are separate
executions, not paired intermediate values. This proves variable BF16
accumulation on this deliberately sensitive input, not its contribution to a
specific full-model error. Next discriminator is a matched native/JAX stable
reduction control, with baseline results retained and no default change yet.

Verification: native records/targets, actual output arrays and both GPU probe
logs inspected. Full Protenix CPU suite passed 786 tests with 4 skips in
141.14 seconds after key-path changes; GPU/operator/full-model admission is
separate and remains incomplete.

### Matched FP32 atom-reduction control prepared

Synthetic controls 947/948 repeat the fixed-input test with both frameworks
performing sum/count/division in FP32 then restoring output dtype. This is an
intentional native-policy change for diagnosis, not baseline admission. They
queue behind the collaborator's existing 7ST3 precision runs 945/946.

`bench/protenix_atom_reduction_control.py` provides a scoped override of the
native transformer aggregation reference or JAX atom aggregation reference,
widening only BF16 inputs and restoring the function on exit. It is not yet
wired into a full-model capture. Three CPU tests pass for JIT sum/mean, restored
output dtype/function and invalid arm rejection. Initial Ruff lambda-assignment
errors were corrected; Ruff/diff checks now pass. GPU/full-model effect and
repeat stability remain unverified; package/default arithmetic was not changed.

### Full-model reduction-control wiring prepared

Both native capture and candidate replay now accept
`--fp32-atom-aggregation`. Native records the override in operator policy;
candidate checks that its requested policy matches the reference before
inference. Baseline native records without the field retain their old policy
encoding. The override is scoped around execution, not installed into the
model implementation. It is never an unchanged-native-default comparison.

Verification: 43 capture/replay/control tests pass in 1.91 seconds, including
matching enabled/disabled policies and rejection of mismatched policies through
both inference routes. Diff checks pass. Fresh source snapshot
`protenix-fp32-aggregate-20260909-WOQ5It` is prepared; full-model control GPU
execution has not started. Small GPU controls 947/948 remain queued behind
the collaborator's 946 at this checkpoint.

### FP32 reduction GPU control completed; full-model controls launched

Jobs 947/948 exited 0. Both native and JAX produced identical values on all
three executions: FP32 sum `2.0`, mean `0.0077821011655032635`; BF16 output
after FP32 accumulation had sum `2.0`, mean `0.007781982421875`.
Unlike baseline jobs 942/943, no repeat variation was observed in this small
control. This establishes the synthetic control only, not a full-model cause.

The isolated `protenix-fp32-aggregate-20260909-WOQ5It` snapshot now has four
serialized jobs: native 7R6R/3V7E (949/950), then matched FoldJAX replays
(951/952). All request `--fp32-atom-aggregation`; the candidate preflight
requires a completed native reference and matching aggregation policy.
Native 949 is running with 10 recycles, 200 steps, BF16 and the real 7R6R
input (245 tokens, 2529 atoms, 364 MSA rows). Other jobs are queued at this
checkpoint. Existing native-default outputs and package defaults are retained.

Verification: inspected queue state, native startup log and both completed
synthetic GPU logs. Full-model structure/confidence results and stabilized
native repeat variability remain unverified; these diagnostic overrides do
not establish native-default admission or warm performance.

### Native full-model aggregation controls completed

949/950 exited 0 and both native records validate the complete n5 capture.
Compared with their respective baseline records, the input, source/checkpoint
and actual stochastic-tape target fields remain equal; only the runtime and
effective-policy hashes differ. For 3V7E, direct JSON inspection identifies
the added `fp32_atom_aggregation: true` operator field and the changed capture
wrapper/source-snapshot provenance.

Native baseline versus FP32-aggregation native control entity maxima (angstrom):

| Case | Entity maxima |
| --- | --- |
| 7R6R | A 2.59384134; B 2.00433917; D 1.94215125 |
| 3V7E | P 0.85192064; R 1.52377846; L 1.23403001 |

These are policy-change effects plus native execution variability, not a
cross-framework improvement result. In particular, widening accumulation is
not automatically structure-preserving. JAX job 951 is running and 952 is
queued; matched cross-framework and native-control repeat evidence is pending.

Verification: both complete native records and per-sample system-aligned
entity measurements were inspected. No default-policy change or pass issued.

### Read-only cross-check of collaborator 7ST3 precision runs

Completed collaborator jobs 945/946 were independently remeasured from
`protenix-highest-20260909/protein_protein_7st3/{foldjax-highest,foldjax-high}`
against `protenix-panel-20260909-KzEzSW/protein_protein_7st3/native`.
Both independent input checks pass. Entity maxima after whole-system fitting:

| Artifact arm | A RMSD (angstrom) | B RMSD (angstrom) | Strict confidence |
| --- | ---: | ---: | --- |
| foldjax-high | 0.29362866 | 0.10706015 | fail |
| foldjax-highest | 22.60694432 | 39.00724475 | fail |

This is an artifact-output cross-check, not a verified single-variable causal
comparison: source/runtime/tape equivalence between these arms has not yet
been audited here. Do not adopt highest precision or infer its sole causality
from the arm labels. No collaborator files or jobs were changed.

Verification: existing input validator, canonical output conversion, system
Kabsch/entity RMSD and strict confidence comparator ran successfully on both
artifacts. Our atom-aggregation replay 951 remains running, 952 queued.

### Matched 7R6R FP32 aggregation result

951 exited 0; independent input validation passes against control native 949.
With both arms using FP32 atom aggregation and the captured native tape, entity
RMSD per sample (angstrom) is:

| Entity | Samples 1–5 | Maximum |
| --- | --- | ---: |
| A | 0.69655364, 0.14059760, 0.13228719, 0.07194971, 0.03145266 | 0.69655364 |
| B | 0.54558265, 0.06611826, 0.04897874, 0.05139340, 0.03781973 | 0.54558265 |
| D | 0.51821846, 0.06179985, 0.05321816, 0.05153132, 0.03093913 | 0.51821846 |

These maxima are smaller than the baseline pair (A 3.40369838, B 2.73917138,
D 2.61744913), but still above the priority threshold. Strict confidence
overall fails: maximum atom pLDDT difference is 0.00223720 on 0–1 scale,
PAE 0.38766479 angstrom, pTM 0.00007689 and ipTM 0.00012058. The latter two
leaves individually pass their existing atol/rtol comparator; this does not
pass the full confidence gate. No calibrated native-control repeat floor has
been established, so the reduced residual is not closure or sole-cause proof.

The collaborator's 7ST3 source trees were also compared read-only: excluding
bytecode, the sole `src` difference is the prediction wrapper's default
`matmul_precision="high"` versus `"highest"`. Full runtime/tape provenance
between those artifacts is still unaudited here.

Verification: completed 951 artifacts, independent inputs, per-sample entity
RMSD and full confidence comparator inspected. 952 is running. Defaults remain
unchanged; full-model control repeat stability and warm performance are pending.

### 7ST3 precision-pair provenance audit

The 945/946 stored provenance was compared field by field. The complete
Python-source hash maps differ only in `src/foldjax/models/protenix/models/predict.py`;
the source diff changes only the default matmul precision from high to highest.
Requested wrapper options and schedule audits are equal. The concrete inference
boundary records differ only in `jax_default_matmul_precision`; their options
and tape-input records are equal. Remaining provenance fields are equal except
the corresponding source map and boundary hash. Capture completion differs
only in instrumented elapsed seconds.

Thus the recorded artifacts support a controlled high/highest comparison at
the captured host boundary, substantially strengthening the earlier label-only
comparison. This does not prove device-consumer tape identity or repeatability,
nor does highest represent the pinned native high/TF32 policy. Preserve high
as the native-matching baseline; the large highest residual is not evidence
that default FoldJAX has that residual. No execution or collaborator source was
modified during this audit.

Verification: JSON field comparison and full source-tree diff inspected.
Device-consumption and repeated high/highest effects remain unverified.

### Matched 3V7E control: no consistent improvement

952 exited 0 and independent inputs pass. Entity RMSDs after one system fit
per sample, with the same FP32 aggregation override in both arms:

| Entity | Samples 1–5 (angstrom) | Maximum |
| --- | --- | ---: |
| P | 0.92643490, 0.34388128, 0.26744498, 0.19413223, 0.38514062 | 0.92643490 |
| R | 1.57371271, 0.43074815, 0.35227806, 0.29199569, 0.59597457 | 1.57371271 |
| L | 0.98197121, 0.17227808, 0.12816783, 0.12215958, 0.33634218 | 0.98197121 |

Baseline maxima were P 0.83525889, R 1.32973188, L 1.19352912. The control
does not improve all entities. Strict confidence fails; maximum atom pLDDT
difference is 0.05609459 (0–1), pTM 0.00483340, ipTM 0.00143796, versus
baseline 0.04504025, 0.00346518, 0.00129640 respectively. This two-case
experiment does not support adopting FP32 aggregation as a parity fix.

7R6R control intermediate arrays also remain unequal: s_inputs RMSE
0.00007787741 (734 unequal entries, maximum 0.00390625), final single trunk
RMSE 0.98339836 and pair trunk RMSE 0.70498899. Widening aggregation does
not eliminate the discrepancy before recycling; these outputs locate remaining
differences but do not isolate a particular upstream operator as their cause.

Verification: both full-model control pairs are complete; independent input
checks, per-sample structure and confidence comparisons inspected. No default
change adopted. Native-control repeat variability, device tape observation,
warm performance and full model closure remain unverified.

### Next discriminator: native atom-geometry FP32 island

Source inspection identifies a concrete policy mismatch candidate rather than
another global matmul setting. Pinned native `AtomAttentionEncoder` constructs
`linear_no_bias_d` with `precision=torch.float32`. Its Linear implementation
disables autocast, converts input and weight to FP32, computes the projection,
then restores input dtype. Native inverse-distance arithmetic also operates on
the supplied geometry before the ordinary autocast projection.

FoldJAX `cast_trunk_params` currently casts the entire input embedder parameter
tree, including `cache.linear_d`, to trunk dtype. `protenix_infer_static` also
narrows floating input features before embedding (except restype/profile/
deletion_mean). The atom-cache helper uses the generic linear implementation,
which has no per-projection FP32 override. These source policies are not the
same. Actual intermediate dtypes/values must now be captured to quantify this
path before changing default behavior; input-array equality before the wrapper
does not establish equal internal dtype handling.

Verification: native transformer/primitives and JAX model/cast/cache/linear
sources inspected. The full-model residual attributable to this mismatch is
not yet measured; no model patch or causal closure claimed from source alone.

The native InputFeatureEmbedder forwards geometry directly to the atom encoder;
there is no local blanket feature cast in that path. A quantization probe on
the actual 7R6R native archives confirms `ref_pos` and derived `d_lm` are FP32.
The current JAX pre-embedding BF16 cast changes all 7587 ref_pos components
(maximum absolute change 0.015597343) and 968046 d_lm components (maximum
0.031207085). Charge is int64 and validity is boolean, so the floating-feature
cast does not narrow them. These are geometry quantization measurements, not
final structure errors or an observed native projection output.

Verification: corrected probe completes on all four fields; the initial probe
attempted boolean subtraction for v_lm and failed after printing the floating
results. The corrected probe branches on dtype and does not reinterpret bool
values as floating-precision evidence. Actual projection-output comparison and
native-faithful mixed-precision implementation remain pending.

### Actual-weight distance projection quantization probe (CPU)

Loaded the pinned checkpoint with CPU mmap and `weights_only=True`, selecting
`module.input_embedder.atom_attention_encoder.linear_no_bias_d.weight` (16x3).
The native LinearNoBias with precision FP32 was evaluated on actual 7R6R d_lm;
input, weight and result are FP32, output shape (80,32,128,16). Independent
input/weight quantization probes restore BF16-rounded operands to FP32 before
the same FP32 projection, isolating lost operand information:

| Rounded operands | Projection RMSE | Maximum absolute difference |
| --- | ---: | ---: |
| Geometry only | 0.00001918831 | 0.00027035922 |
| Weight only | 0.00001841032 | 0.00018316880 |
| Both | 0.00002658597 | 0.00036444142 |

These measurements establish a real-weight operand-quantization difference,
not Torch/JAX GPU parity or a full atom-encoder replay. Output BF16 rounding,
GPU accumulation and downstream propagation are deliberately not included.
Recovering FP32 operands after blanket casting cannot restore the lost values;
the original geometry and original projection weight must remain available to
implement the native island faithfully.

Verification: native publisher Linear on real checkpoint/geometry ran on CPU
with GPU hidden. No checkpoint or package file was modified. GPU mixed-policy
projection and full-model confirmation remain pending.

### Native GPU encoder dtype trace (953)

Added `bench/protenix_input_dtype_probe.py`: loads the real native input-embedder
weights strictly and the recorded 7R6R features, then traces leaf input/output
dtypes under CUDA BF16 autocast. It does not alter outputs or package code.
953 exited 0 in 2.25 seconds (diagnostic elapsed time, not warm benchmark).

Observed ref_pos and distance projections both return FP32; charge, feature,
inverse-distance, validity and ordinary attention projections return BF16.
Attention/transition LayerNorm inputs and outputs remain FP32 through all three
atom-transformer blocks. The final linear_q accepts FP32 and emits BF16, and
the concatenated encoder result is FP32 [245,449]. Thus the native input
encoder is a mixed FP32-residual/BF16-projection path, not an all-BF16 stack.
Preserving only the distance projection is insufficient to recreate this
observed dtype flow. Original FP32 geometry, selected projection weights and
normalization/residual handling must be considered together.

Verification: actual-weight isolated CUDA encoder trace completed; Ruff passes
for the new probe. This uses native default module construction and a recorded
feature archive, not independent end-to-end preprocessing, full-run observer
bridge, JAX encoder parity or release proof.

### Native input AMP implementation connected

`AutocastLinearParams` now explicitly selects projection operand dtype without
changing ordinary shared `LinearParams` behavior. The input-encoder conversion
retains original FP32 coordinate/distance projection weights and LayerNorm
parameters, and marks ordinary projections for BF16 autocast. It rejects
already-narrowed floating parameters instead of silently widening rounded data.
Protenix BF16 inference now feeds original input features to this typed encoder;
the separate trunk feature narrowing and diffusion/confidence paths remain.

The modified package is frozen in `protenix-input-amp-20260909-wPc6q4`.
Job 956 replays original native 7R6R capture from the dropout snapshot with n5,
without the discarded FP32 atom-aggregation override. No native intermediate
representation is supplied as an encoder replacement. GPU output validation
is pending; source integration is not yet scientific admission.

Verification: 11 focused dtype/confidence/input tests passed; a broader run
had 61 passes and one obsolete all-input-weights-BF16 expectation. That test
now separately checks BF16 trunk, original FP32 geometry projections and FP32
diffusion, and passes. Explicit autocast/conversion tests passed 4 tests.
Ruff and diff checks passed. A full Protenix CPU suite is now running; no
commit/push or release claim is made at this checkpoint.

### Loader bypass found before admission

The prepared CLI loader directly narrowed input_embedder and pairformer_output
while reading weights, bypassing `cast_trunk_params`. Therefore snapshot
`protenix-input-amp-20260909-wPc6q4` and job 956 do NOT establish execution of
the new native input-AMP route. Treat 956 as an old-loader control, not a fixed
candidate, regardless of its eventual numerical result. Pending jobs 959/960
were cancelled before execution; collaborator jobs were not modified.

The working-tree prepared loader now narrows only pairformer_output while
loading, retaining original input weights, then applies explicit native input
autocast preparation. A new snapshot and actual-weight replay are required.
The full CPU suite started before this loader edit and has shown one failure;
its final detail and loader-specific regressions must be handled before reuse.

### Prepared-loader correction and replacement run

The full Protenix CPU run completed with 789 passed, 4 skipped and one failure:
the input-only stage test substitutes a None encoder, which the new type check
dereferenced. The route now checks InputFeatureEmbedderParams before accessing
its encoder. Relevant loader/input-stage/static-CLI tests, including a real
archive save/load regression for the retained FP32 geometry weights, pass
32 tests. Ruff and diff checks pass after correcting a line-length violation.
The full run preceded the prepared-loader correction; it is not a full-suite
pass for the corrected tree.

Replacement snapshot `protenix-input-amp-loader-20260909-C81JNg` includes both
the prepared-loader correction and guarded route selection. GPU job 961 is
the original-native-tape 7R6R candidate; it is running. 956 finished but remains
classified as the old-loader control. No native aggregation override is used.

The replacement snapshot's prepared loader was also executed on CPU against
the real managed checkpoint. It returns ref_pos LinearParams FP32 [128,3],
distance LinearParams FP32 [16,3], final projection AutocastLinearParams BF16
[384,128], and seven LayerNorm parameter nodes whose affine arrays are FP32.
This confirms the actual checkpoint loader applies the intended typed policy,
not just the toy test. It is a separate CPU load, not observation of the
parameters consumed inside GPU job 961 or GPU numerical parity.

### Corrected-loader 7R6R result (961)

Independent input validation passes. Whole-system-aligned entity RMSDs for
the five samples are:

| Entity | Samples 1–5 (angstrom) |
| --- | --- |
| A | 1.04998646, 0.17949819, 0.11096229, 0.07793667, 0.10780952 |
| B | 0.87350321, 0.09566855, 0.03446204, 0.05616415, 0.03344358 |
| D | 0.84102759, 0.08524511, 0.03681306, 0.05545159, 0.03327234 |

Maxima are smaller than the original baseline pair but still exceed 0.1 A.
Strict confidence fails: atom pLDDT maximum difference 0.00336912 (0–1), pTM
0.00016481, ipTM 0.00033450. s_inputs RMSE is 0.00016199195, maximum
0.01328128576, with 685 unequal entries. This modest embedding improvement
does not close input-encoder numerical parity or establish sole causality.
The separate FP32 aggregation override is absent. Native repeat variability
remains relevant to trajectory interpretation; no tolerance was relaxed.

Verification: corrected-loader prediction artifacts, independent input check,
per-sample entity RMSD and confidence metrics inspected. Current full CPU
rerun is still in progress, and 3V7E job 972 is queued after collaborator AF3
work. No full-panel, warm-performance or release pass is claimed.

### Residual embedding representation discriminator

For 961, the last 65 raw-feature channels of s_inputs are exactly equal to
native. All 685 unequal entries are within the first 384 atom-embedding
channels (RMSE 0.00017516648). Native atom-embedding values are all exactly
representable in BF16; 652 candidate atom-embedding entries are not, tested by
BF16 roundtrip on the stored FP32 arrays. This is an output representation
observation, not proof of the responsible operator or compiler transformation.
Next probe must distinguish projection/aggregation dtype handling from omitted
rounding during output conversion. Do not attribute it to XLA fusion without
an isolated reproduction. Highest token-row errors occur at rows 205,189,132,
236,240; no chemistry interpretation is assigned from row indices alone.

For the 652 off-grid candidate embedding values, multiplying by the atom count
of the corresponding token returns exactly BF16-representable values for 540
entries and within 1e-6 for 647 (maximum BF16-grid distance 1.9073486e-6).
This supports a post-mean division/output-rounding hypothesis, not yet proof
of a compiler cause. Counts come from the validated candidate atom_to_token_idx.
The small CPU eager/JIT aggregate-then-FP32-export test preserves the BF16 grid,
so the symptom is not reproduced there. GPU discriminator 973 tests four atom
counts in eager and JIT modes and is queued after existing work; no full-model
job or default workaround was added for this hypothesis.

### Completed 3V7E correction and isolated rounding control

972 exited 0; independent inputs pass. Corrected-loader entity RMSDs:

| Entity | Samples 1–5 (angstrom) |
| --- | --- |
| P | 0.20905765, 0.38379881, 0.35619067, 0.84313582, 0.23082476 |
| R | 0.20832802, 0.40669055, 1.19809885, 1.39927185, 0.52389527 |
| L | 0.11301179, 0.11248509, 1.35472857, 1.25314422, 0.23846099 |

Strict confidence fails: maximum atom pLDDT difference 0.04510456 (0–1), pTM
0.00347626, ipTM 0.00137854. This does not improve the baseline consistently;
the source-policy correction is not a demonstrated full-model parity fix.

GPU rounding control 973 reports zero off-BF16-grid outputs for all eight arms
(3/7/11/23 atoms, eager/JIT). The isolated aggregate-then-widen expression does
not reproduce the full-model representation symptom. Do not add a generic
aggregation rounding workaround based on the previous hypothesis. A larger
encoder-boundary or fusion-context probe would be needed to isolate it.

Verification: actual 972 arrays and 973 output logs inspected; no thresholds
changed. Independent dtype-code review remains pending, and no code admission,
full-panel parity or performance conclusion is issued from these experiments.

Abstract execution of the corrected snapshot's real-weight encoder on actual
7R6R features reports aggregation input BF16, aggregation output BF16, and
concatenated encoder output FP32 [245,449]. This rules out an obvious FP32
aggregation dtype in the traced encoder as the explanation for the off-grid
stored values. It does not inspect optimized GPU execution; a fusion-context
or actual consumer/output observation remains necessary. The isolated GPU
control still does not reproduce the symptom. No rounding workaround adopted.

Isolated real-weight encoder GPU job 975 exited 0: output [245,449], zero
off-BF16-grid embedding values, native s_inputs RMSE 0.00019376223 and maximum
0.015625. It uses recorded native features and the corrected prepared loader,
with high matmul precision and scan enabled. Unlike full-model 961, its JIT
closure captures encoder inputs/parameters and does not include downstream
trunk/sampler consumers. Therefore the off-grid symptom depends on some
difference in the whole execution context; this experiment alone cannot select
fusion, argument specialization, extra feature fields or consumer conversion
as the cause. Remaining embedding disagreement exists even without off-grid
outputs. Do not equate restoring the BF16 output grid with native parity.

Verification: 975 completed successfully and its scalar diagnostics were
inspected. Shared-feature encoder control is not independent preprocessing,
full-model observer-bridge evidence or a warm-performance measurement.
