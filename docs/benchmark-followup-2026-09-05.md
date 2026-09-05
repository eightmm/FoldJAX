# Fresh multimodal upstream comparisons — 2026-09-05

The CI asset-isolation repair is verified; universal model parity is **not**.
These are measured port-to-pinned-upstream residuals, not comparisons with
experimental structures and not an accuracy leaderboard.

[Per-sample metrics, input audits and provenance](../bench/experiments/fresh-multimodal-parity-2026-09-05.json)
retain unsuccessful attempts as well as measured comparisons.
This follow-up completed nine fresh comparison cells plus one fixed-tape repeat,
each with five paired samples, across Boltz-2 and OpenDDE. One earlier
instrumentation-gate failure is retained separately.

## CI repair

Commit `83eb4eb19adc9786b0e67772c0a39c17bde5b7fc` changes tests and documentation
only. The portable SEP identity test now supplies synthetic CCD test data and
blocks external-asset fallback; the separate real-CCD integration remains.
The synthetic molecule does not exercise every optional attribute of the real
publisher pickle; that attribute-carrying path remains integration coverage.

- Local CPU gate: 3,576 passed, 399 skipped, 8 network tests deselected; line
  coverage 87.42%, Ruff and frozen dependency-lock check passed.
- [Hosted CI](https://github.com/eightmm/FoldJAX/actions/runs/33953700008): both
  jobs passed; CPU tests 3,540 passed, 435 skipped, 8 deselected; coverage 87.35%.
- No CI job, model calculation, or scientific tolerance was removed or relaxed.

Hosted skips increased from 434 to 435 because the newly separated real-CCD
integration is unavailable there. The formerly failing portable SEP regression
now executes and passes; it was not replaced by a skip. The real-CCD integration
executes locally. Coverage percentages are not test pass rates.

## Protocol

Five paired samples per cell, paired by sampler index rather than confidence
ranking. Match chemical atom identity and validity masks;
fit one proper, unweighted global Kabsch transform on all valid system atoms
per sample, then measure each entity instance without another fit. For each
entity, RMSD is the square root of the mean squared 3D residual over that
entity's valid predicted atoms (not just backbone/Cα); the table takes the
maximum of those five sample RMSDs, not a maximum over individual atoms.
Report all five values in the companion JSON and their maxima below, in angstroms.
There is no assigned universal entity-RMSD acceptance threshold. The old harness
0.5 Å scalar exit is diagnostic only. Fixed numerical leaf tolerances remain
`atol=rtol=1e-4`; this experiment does not claim every internal leaf was tested.

Boltz inputs are independently constructed from the original job with recorded
reference-augmentation draws, then sampled using actual captured sampler draws.
All common shapes/dtypes match. All common values match exactly except
`ref_pos` (maximum absolute difference 9.536743e-7 Å). Native-only `disto_target`
is the separately classified training target, not an inference input copied
into FoldJAX. No upstream feature archive replaces FoldJAX preprocessing.

## Boltz-2: native-default comparison remains divergent

Pinned native commit: `b1ebfc46ecf57f5414e0d1a6f9027bbb122c53bc`.
Both arms use seed 101, 200 steps, 3 configured recycles (4 trunk passes),
no MSA subsampling and 5 samples (native CLI sample default is 1).
The FP32 control disables native fused kernels. The native-default reference
retains upstream `bf16-mixed` and fused kernels; FoldJAX uses bf16 compute,
`highest` matmul precision, XLA attention and the patched cuEquivariance
multiplication path. Torch autocast and JAX bf16 are not assumed to have
identical operator-level arithmetic. FoldJAX replay is coordinate-only, so this
is neither confidence parity nor an end-to-end throughput measurement.

| Complex | FP32 control: maximum over 5 sample entity RMSDs (Å) | Native mixed-precision reference: maximum over 5 sample entity RMSDs (Å) |
|---|---|---|
| Protein–ligand, 5SAK | fresh: protein 0.0300; ligand 0.00127; same-tape repeat: protein 0.00241; ligand 0.000497 | protein **16.21**; ligand **13.17** |
| Protein–RNA, 1URN | protein 0.000044; RNA 0.000057 | protein 0.070164; RNA 0.313566 |
| RNA–ligand, 3GCA | RNA 0.000054; ligand 0.000019 | RNA 0.026244; ligand 0.028175 |

Small FP32 residuals therefore do not establish native-default equivalence.
The 5SAK native-default result is a substantial unresolved divergence; it is
retained, not discarded or converted into a pass by changing thresholds.
These settings confound precision and native kernel use, and execution variation
is an additional unresolved factor; the experiment does not isolate one cause.
All three native-mixed-precision replay cells also returned a failed legacy
diagnostic exit; all three FP32 cells and the fixed-tape FP32 repeat returned
zero. Those exits use the old raw-coordinate/trunk criteria, not an entity-level
scientific acceptance threshold. Individual exits remain in the JSON.

### Fixed-tape execution variability

The previous published 5SAK patched-cuEquivariance result and the fresh FP32
capture have exactly identical native reference draws (838), native features,
sampler tape, trunk, coordinates and independently rebuilt FoldJAX features.
Nevertheless, maximum globally aligned system RMSD changed from 0.003859 to
0.029911 Å. A further FoldJAX-only replay against the same frozen native
artifacts measured 0.002405 Å (entity maxima: protein 0.002412, ligand
0.000497 Å), again with exactly equal FoldJAX features.

This rules out changed realized inputs or native random draws as the explanation
for this difference. Cross-process execution variability remains, and a unique
compiler/kernel cause has not been isolated. Do not select the smallest result
as the sole representative number or claim bitwise reproducibility.
These few observed repetitions are not a confidence interval or an error bound.

## OpenDDE independent-input numerical control

Pinned native commit: `ddfa1df8aff1babf1fddac4247b7d2351bd0ce9f`.
Five samples, 200 steps, 10 cycles; FP32 and **TF32 disabled**, the latter an
explicit deviation from upstream defaults. Capture the input actually entering
native `forward` and compare it with independently featurized FoldJAX input
before inference. Common values must match exactly, with documented string,
template and geometry-mask dtype representations. Derived relative-position
and padding representations are checked separately. Actual sampled MSA row
indices are applied to FoldJAX MSA arrays and every selected array is rechecked;
no upstream MSA tensor is substituted.

Native preprocessing reference draws do not have a separate tape in this
wrapper. Instead, the gate requires exact equality of realized common inputs,
including reference coordinates. Sampler noise/rotations/translations and MSA
draws are recorded and replayed. This controls realized model inputs by
construction; it does not validate equality of the unmodified RNG generators or
their untaped distributions. The first 3GCA attempt stopped at a previously
unclassified native `inference_seed` field. Source tracing showed it seeds the
rollout RNG after preprocessing; the revised gate checks its scalar int64 value
against the tape's seed. A changed-seed negative control is rejected. The failed
attempt remains in the record and is not counted as a model failure or pass.

Entity RMSDs use raw sampler coordinates before terminal-OXT writer repair,
all unpadded predicted atoms, and the same whole-system fit protocol. Reference
conformer availability (`ref_mask`) is not misused as predicted-atom validity.
Confidence computation is not compared.

| Complex | Maximum over 5 sample entity RMSDs (Å), FP32/TF32-off |
|---|---|
| Protein–ligand, 5SAK | protein 0.000212; ligand 0.0000864 |
| Protein–RNA, 1URN | protein 0.0000214; RNA 0.00000776 |
| RNA–ligand, 3GCA | RNA 0.00000491; ligand 0.00000755 |

All three revised input gates and legacy coordinate diagnostics completed
successfully. This establishes these measured residuals under this control,
not OpenDDE's native TF32-enabled default equivalence or confidence parity.

## Scope still not established

- Native mixed-precision operator equivalence and stable repeated execution.
- Confidence-head/score equivalence for these coordinate-only replays.
- Fresh independent-input full-RNG end-to-end results for all six models.
  Existing input-panel results remain distinct from coordinate parity.
- Protenix's old shared-feature/centre-only harness is not a native-default
  control; ESMFold2's full sampler tape is still unavailable in the public path.
- Uncontrolled-RNG statistical comparisons, steady-state runtime/memory scaling,
  and experimental-structure accuracy are not measured by this follow-up.

Raw arrays/tapes remain in the local `fresh-n5-20260905` experiment archive;
portable records include pinned commits, input/MSA/weight hashes, per-sample
metrics, explicit configuration deviations and failed attempts. No parameters,
experimental coordinates, machine-specific paths or raw logs are published.
