# Model version targets

FoldJAX never uses “latest” as a model identity. A reproducible port target has
three parts: the upstream source revision whose graph was implemented, the
checkpoint or bundle loaded by that graph, and the execution boundary that has
actually been verified. This page records all three for every public model.

The `0.1.0` values still present in a few model-package `__init__.py` files
identify the JAX port code, not the upstream model or checkpoint. They must not
be used as the model identity of an experiment.

## Latest validation checkpoint (2026-09-05)

The [independent-input and entity-level audit](preprocessing-contract-audit.md)
supersedes the earlier unresolved Boltz 5SAK diagnosis below. Explicitly passing
the active JAX precision to fused triangle multiplication reduces five-sample
global-fit entity maxima to protein **0.003870 Å** and ligand **0.000689 Å**.
The original failed runs remain historical evidence, not current fixed-code
results. OpenFold3 cyclic preprocessing/relative positions and Protenix
ligand-aware confidence identity were also corrected.

The finite preprocessing panel compares 202 of 216 model/input cells; remaining
cells are unsupported or rejected and raw key/dtype exceptions remain explicit.
This is a partial validation checkpoint, **not** a release or all-model
coordinate-equivalence claim. See the
[machine-readable results](../bench/experiments/independent-input-entity-parity-2026-09-05.json)
for paired values, controls, and remaining gaps. Earlier crystal/reference
statistics below are historical only and are excluded from this parity gate.

| FoldJAX model | upstream source implemented | released parameter target | advertised boundary |
|---|---|---|---|
| `alphafold3` | AlphaFold 3 `3.0.4`, commit `85c4d20505fd5cef05eac22b534d4e793971ae69` | caller-supplied `af3.bin` | vendored upstream implementation with an audited eight-file FoldJAX runtime patch set; not a separate model port |
| `boltz2` | package metadata `2.2.1`, commit `b1ebfc46ecf57f5414e0d1a6f9027bbb122c53bc` | `boltz2_conf.ckpt`, `boltz2_aff.ckpt`, and `mols.tar` at Hugging Face revision `6fdef46d763fee7fbb83ca5501ccceff43b85607` | full JAX inference and preprocessing |
| `esmfold2` | untagged Biohub Transformers snapshot `ef32577f55da19a4989cd7b22e004dc43a4998cb` | ESMFold2 revision `8fc3ff471022fdce52c77030685eb775de0c00a3` plus ESMC-6B revision `45b0fa5d7fb06faefbd5e3b89bdcef35d564e79a` | JAX structure, ESMC inference, and all-biomolecule common preprocessing |
| `opendde` | OpenDDE `1.1.1`, commit `ddfa1df8aff1babf1fddac4247b7d2351bd0ce9f` | `released` (`opendde.pt`) and opt-in `abag` (`opendde_abag.pt`) at Hugging Face revision `eddd563ce96571f784012edd8f045181c8f8627d` | full JAX inference and preprocessing, including the 1.1.1 OXT/ion behavior and opt-in RNA-MSA/template paths |
| `openfold3` | OpenFold3 `v0.5.0`, commit `c4771653c5d0a3ebb0b3af71b05efd64bc44ee86` | managed OpenBind `of3-ob-2025-06-30-174k.pt` | v0.5/OpenBind-only JAX inference and portable preprocessing; legacy p1/p2 rejected |
| `protenix` | package metadata `2.0.0`, commit `4c355be4553512f72453ecbfb65e69f4c35d1413` | `released`, `base-20250630`, `v2`, `mini-esm-v0.5.0`, and `mini-ism-v0.5.0` profiles | full JAX inference and preprocessing |

The commit is authoritative even when an upstream also publishes a semantic
version. ESMFold2 is the one untagged source target; its source and both model
repositories are therefore identified only by immutable revisions.

## Latest-upstream audit (2026-09-04)

“Latest” below means the newest official release when the project publishes
releases, and the current canonical source/checkpoint revision otherwise. It is
not inferred from a moving package name.

| model | newest official upstream at audit date | FoldJAX status |
|---|---|---|
| AlphaFold 3 | [`v3.0.4`](https://github.com/google-deepmind/alphafold3/releases/tag/v3.0.4) | **current**: exact release commit, apart from the audited FoldJAX runtime patch set |
| Boltz-2 | [`v2.2.1`](https://github.com/jwohlwend/boltz/releases/tag/v2.2.1) | **current release family**: the port commit follows that tag and the pinned Hugging Face bundle was still canonical |
| ESMFold2 | untagged [Biohub Transformers](https://github.com/Biohub/transformers) source plus [ESMFold2](https://huggingface.co/biohub/ESMFold2) and [ESMC-6B](https://huggingface.co/biohub/ESMC-6B) repositories | **current snapshots**: all three audited heads equal the revisions above |
| OpenDDE | [`v1.1.1`](https://github.com/aurekaresearch/OpenDDE/releases/tag/v1.1.1) | **current**: exact release commit; both the base and ABAG checkpoints are managed profiles |
| OpenFold3 | [`v0.5.0`](https://github.com/aqlaboratory/openfold-3/releases/tag/v0.5.0) | **current**: exact release commit and its default OpenBind checkpoint |
| Protenix | [`v2.0.0`](https://github.com/bytedance/Protenix/releases/tag/v2.0.0) | **current model available**: the port commit follows that tag and the `v2` profile is supported; `released` intentionally remains the v1 baseline profile |

All six source targets were current under that definition on the audit date.
That is a dated audit result, not permission to record a moving word such as
`latest`: experiments must still name the source commit and checkpoint profile.

## Validation boundary (2026-09-04)

The closure matrix used real released weights, seed 101, one sample, two
diffusion steps, one recycle, MSA depth 16, and padding. Every model completed
protein, DNA, RNA, CCD ligand, SMILES ligand, ion, and one input containing all
six entity forms: 42/42 standard runs. The OpenDDE ABAG profile completed the
same all-entity input, for 43/43 manifests in total. All 12,188 written atoms
and 297 scalar scores were finite, every structure parsed with Gemmi, and every
recorded structure SHA-256 matched its file.

That short matrix is an execution, chemistry-intake, padding, output, and
serialization gate. It is not an accuracy benchmark and it is not a claim that
two different model families should predict the same coordinates. Port parity
is a separate per-model claim:

| model target | real-weight coverage beyond the 7/7 matrix | upstream/port parity evidence | remaining boundary |
|---|---|---|---|
| AlphaFold 3 3.0.4 | mapped 9FM7 template completed | complete vendored-tree comparison against `85c4d205`: every carried upstream file is exact except the eight declared runtime patches | upstream code is vendored rather than independently reimplemented; the user-supplied weight file remains experiment-specific |
| Boltz-2 2.2.1 | unmapped 9FM7 template and the real affinity head completed | 77 clean-upstream checkpoint/component/feature gates passed; on the all-entity input, a shared 20-step noise tape gave `0.194365 Å` raw all-atom RMSD and trunk correlations `>=0.99999992` | the 20-step parity run is a numerical port gate, not the released 200-step accuracy schedule |
| ESMFold2 | no template or affinity head exists | creator-source component suite passed; rotary tables are bit-identical; the separately versioned `transformers` 5.16.1 atom encoder passed on its pinned HF re-export; FoldJAX/upstream completed 6/6 real-weight protein cells at 128, 488, and 976 residues | the six end-to-end cells are independent stochastic samples, not matched-coordinate parity |
| OpenDDE 1.1.1 | released 7/7, ABAG all-entity, and mapped 9FM7 template completed | across ten protein/DNA/RNA/CCD/SMILES/ion/RNA-MSA/template cases, all 60 shared upstream feature arrays per case were bit-identical; ABAG with the same MSA and random tape gave `0.081714 Å` raw all-atom RMSD | template parity requires the recorded native Kalign 3.3.5 executable; base and ABAG are distinct checkpoints |
| OpenFold3 0.5.0/OpenBind | native-pipeline 9FM7 template completed | seven exact-v0.5 real-target/preprocessing gates covered DNA, protein, RNA/metal, protein/ligand, and direct-template geometry; the current-tree 1BNA shared-tape run gave `5.78e-6 Å` all-atom Kabsch RMSD | per-job common-schema templates remain unsupported because v0.5 constructs them in its native pipeline |
| Protenix 2.0.0 / v2 | mapped 9FM7 template completed | 75 local modality/search/feature gates passed; the all-entity shared-noise run gave `0.016047 Å` raw and `0.015961 Å` Kabsch all-atom RMSD with trunk correlations `>=0.99999961` | two optional tests need publisher example/database files absent from the repository; the equivalent 9FM7 template path was exercised with pinned assets |

Ion is represented by its CCD identity (`MG`) as a non-polymer ligand, not by
a seventh entity type. Template contracts intentionally differ: mapped input
for AlphaFold 3, OpenDDE, and Protenix; self-aligned/unmapped input for Boltz;
native query-pipeline input for OpenFold3; unsupported for ESMFold2. Binding
affinity reaches Boltz-2 alone. The machine-readable record, including queue
exit codes and aggregate digests, is
[`latest-model-validation-2026-09-04.json`](../bench/experiments/latest-model-validation-2026-09-04.json).

## Upstream-default n=5 comparison (2026-09-04)

A separate real-weight experiment kept each pinned upstream's released
inference schedule and set only the requested sample count to five. All arms
used the same 76-residue 1UBQ sequence, real MSA, experimental reference, and
integer seed 101. Boltz-2 changed from its released one sample to five and
ESMFold2 from 32 to five; the other four targets already default to five.

For every row, the analysis compared all 25 FoldJAX/upstream sample pairs with
the pooled 20 within-arm pairs. The integer seed is controlled but the random
tape is not: JAX and PyTorch do not generally consume an interchangeable RNG
stream.

| model/arm | released schedule used | cross CA RMSD (Å) | pooled within CA RMSD (Å) | ratio | reference CA RMSD (FoldJAX / upstream, Å) |
|---|---|---:|---:|---:|---:|
| AlphaFold 3 3.0.4 | 200 steps, 10 recycles | 0.262 | 0.352 | 0.74 | 0.767 / 0.767 |
| Boltz-2 2.2.1 | 200 steps, 3 recycles | 1.113 | 0.875 | 1.27 | 1.402 / 0.799 |
| OpenDDE 1.1.1 | 200 steps, 10 recycles | 0.334 | 0.356 | 0.94 | 0.935 / 0.924 |
| OpenFold3 0.5.0/OpenBind | 200 steps, 3 configured recycles | 1.198 | 1.413 | 0.85 | 0.980 / 1.113 |
| Protenix 2.0.0/v2 | 200 steps, 10 recycles | 1.050 | 0.932 | 1.13 | 1.182 / 1.806 |
| ESMFold2, FP32 parity control | 14 steps, 3 loops, MSA depth 1024 | 0.923 | 0.864 | 1.07 | 1.676 / 1.590 |
| ESMFold2, shipped FoldJAX BF16 | same creator schedule | 0.922 | 0.854 | 1.08 | 1.683 / 1.590 |

All 13 execution arms completed with five finite samples. Across these rows,
the CA cross/within ratio was 0.74–1.27 and the all-atom ratio was 0.89–1.10;
median pTM differences were at most 0.0094. This is evidence that the
framework difference is on the scale of the model's own five-sample spread for
this target. It is not a statistical equivalence result: the opposite 1UBQ
reference shifts for Boltz-2 and Protenix illustrate why five samples from one
protein cannot establish a general accuracy claim. Wall times are retained
only for audit because retries and shared compiler caches make their state cold
or unspecified.

The compact settings, results, runtime controls, queue jobs, and hashes of the
full local evidence are recorded in
[`upstream-default-n5-2026-09-04.json`](../bench/experiments/upstream-default-n5-2026-09-04.json).

## Multimodal upstream-default n=5 follow-up (2026-09-05)

The released-default comparison was repeated on protein, protein–RNA,
RNA–ligand, protein–RNA–ligand, protein–DNA, protein–ligand, and
protein–protein complexes. All 91 independent-RNG execution arms completed.
Across 49 model/target rows, neither the complete-output nor the
reference-resolved family contained a Holm-significant FoldJAX/upstream
separation.

A second control replayed five captured upstream random tapes at each
independent port's released 200-step and 3- or 10-recycle schedule. It passed
27/28 rows at a maximum per-sample coordinate RMSD of 0.5 Å and a recorded
trunk-correlation floor of 0.999. Both cores used shared features, FP32/highest
precision and captured noise. OpenFold3 and Protenix disabled random rigid
augmentation; Boltz-2 and OpenDDE replayed captured augmentation. Internal MSA
selection was also controlled, not uniformly left at native defaults. This is
a numerical core control rather than a complete native-default or independent
preprocessing parity gate. The single
preregistered failure was Boltz-2 on protein–ligand 5SAK (`1.176058 Å` Kabsch
RMSD). Feeding the captured upstream trunk to the FoldJAX sampler reduced that
maximum to `0.000766 Å`, while disabling JAX scans reproduced it at
`1.175220 Å`. These controls localize the large discrepancy upstream of the
sampler in this case and argue against scan lowering as its cause. Numerical
amplification is a hypothesis; they do not prove that trunk rounding, rather
than a port defect, is the sole cause.
The threshold remains failed rather than being relaxed after observing the
result.

The complete version identities, panel, schedules, result matrix, diagnostic
controls, artifact hashes, and claim boundary are recorded in
[`upstream-default-multimodal-n5-2026-09-05.json`](../bench/experiments/upstream-default-multimodal-n5-2026-09-05.json).

### Stricter entity-level audit in progress

The current validation target is **not yet satisfied**. Experimental/crystal
coordinates are excluded from the port-parity gate. For each paired sample,
fit one unweighted, proper Kabsch transform using every valid atom in the
system, then measure each chain/entity instance under that same transform.
There is no entity-wise refit, sample rematching, or averaging away a failing
ligand. Report every sample plus each entity's maximum over the five samples.

Reanalysis of the retained captures gives the following worst entity/sample
RMSDs (Å). These are core-control diagnostics, not claims that independently
generated inputs or all native stochastic operations have passed verification.

| case | Boltz-2 | OpenFold3 | Protenix v2 | OpenDDE, mirrored writer |
|---|---:|---:|---:|---:|
| 1UBQ protein | 0.001531 | 0.000012 | 0.002357 | 0.003419 |
| 7R6R protein–DNA | 0.051528 | 0.000184 | 0.010404 | 0.022980 |
| 5SAK protein–ligand | 14.233074 | 0.001017 | 0.161472 | 0.008667 |
| 7ST3 protein–protein | 0.007256 | 0.000111 | 0.037569 | 0.005255 |
| 1URN protein–RNA | 0.003917 | 0.000025 | 0.011679 | 0.083800 |
| 3V7E protein–RNA–ligand | 0.000918 | 0.000767 | 0.020603 | 0.001659 |
| 3GCA RNA–ligand | 0.000990 | 0.000005 | 0.000527 | 0.000760 |

For 5SAK, Boltz-2's ligand reaches 14.233074 Å while its protein reaches
1.045609 Å under the same global fit. OpenDDE's 1URN raw-network protein
maximum is 0.864494 Å before the documented, symmetric terminal-OXT writer
repair; the table retains the post-repair 0.083800 Å result, not a zero claim.

Independent native preprocessing exposed Protenix's missing seeded
translate-then-rotate reference augmentation, int8 deletion-count saturation
shared with OpenDDE, and invalid ligand/ion frame handling. After correction,
all common feature values match exactly for both models on all seven cases.
Schema-only keys and dtype differences remain explicitly separate from value
agreement; a full input-contract pass has not been established. A fresh
OpenDDE 5SAK replay gives protein 0.008767 Å and ligand 0.003719 Å after global
alignment, before writer repair. AlphaFold 3 and ESMFold2 still need complete
paired stochastic-output evidence, and all six models require completed
independent input-contract audits before a universal parity claim.

The independent OpenFold3 v0.5.0 audit subsequently found and corrected four
preprocessing differences: column indexing in MSA profiles, main-MSA
deduplication, the `2/pi` deletion transform (previously `8/pi`), and retention
of CCD ligand stereochemistry before conformer regeneration. With the native
reference-augmentation draws and RDKit 2025.09.6, all common feature values on
the seven-case panel are exact except `ref_pos`, whose maximum absolute error
is at most 1.90735e-6 Å. This is not bitwise reference-position agreement.
Using RDKit 2026.03.5 instead produces different conformers even with the same
integer seed; chemistry-library versions must therefore be part of the input
provenance. Existing OpenFold3 core RMSDs above used the earlier shared
feature archive and have not yet been rerun on these corrected native inputs.

AlphaFold 3's independent Python preprocessing agrees exactly on all seven
cases: all 59 array features and five object-metadata fields, including atom
layouts. These checks explicitly share the generated C++ extension and CCD
assets; they are not independent binary-build or model-output verification.

ESMFold2's single-protein/query-only builder matches all 23 feature keys,
shapes, dtypes, and values on 1UBQ. The separate full-panel audit against
Biohub/esm preprocessing at `bf343ba264b650dff7a073643725f9aaa1fdbe8d`
identified and corrected entity IDs, deletion scaling, paired/unpaired MSA
assembly, and routing of protein MSAs/multichain jobs through the common
all-atom builder. All common array shapes and dtypes now match on seven
cases; only float32 `deletion_mean` reductions differ, by at most 2.011657e-6.
Two further controls (SEP-modified 1UBQ with real MSA; protein/DNA/RNA/CCD,
SMILES and ion together with real MSA) have exact common-array values.
Native `pocket_feature`, `gt_coords`, `is_resolved`, and `frames_idx` are
explicitly ignored by the published model. `disto_cond` and `disto_cond_mask`
are inactive conditioning fields, zero in these cases; active conditioning
is not covered. FoldJAX's two string metadata arrays are writer-only.
These classifications do not make the raw key sets identical. The creator
preprocessing revision remains distinct from the Transformers core reference.

Boltz-2 has now been independently inspected on all seven cases. Matching
native RDKit 2026.03.2 eliminates the ligand distance-bound differences seen
with 2026.03.5 on 3V7E and 3GCA. Reference-position differences remain below
1e-6 Å. Repeating native feature construction with NumPy 2.1.3 also eliminates
the 5SAK/3V7E ligand frame-index differences: ordering depended on the NumPy
version. Version-controlled preprocessing is not a new output-parity result.

Additional input audits on 2026-09-05 found the following:

- Protenix and OpenDDE's derived `d_lm`, `v_lm`, trunk mask and relative-position
  values agree on all seven cases each. Some host integer/bool storage dtypes
  differ, with explicit consumer casts; BF16 projection parity is not implied.
- Protenix mapped JSON templates address observed residues of the first auth
  chain, not SEQRES positions or `chainId`. The parser now uses that native
  convention for JSON, merging disjoint protein/ligand/water chain segments.
  Search-hit A3M/HHR templates retain their separate SEQRES convention.
  On populated 9FM7, all seven template arrays have exact values after this
  correction (previous coordinate error 64.712116 Å); `template_aatype` is
  still native int64 versus FoldJAX int32. This is not full dtype parity.
- AlphaFold 3's populated, single-chain-filtered 9FM7 template control matches
  all 59 arrays and five metadata fields exactly. An older unfiltered
  two-polymer-chain artifact is rejected by both implementations.
- OpenDDE's native template loader rejects mapped JSON and accepts A3M/HHR.
  FoldJAX's mapped-JSON extension must not be claimed as native template
  equivalence. A populated A3M search-hit control using 5OCM now has exact
  values for all seven template arrays in both OpenDDE and Protenix.
  Native Protenix requires its parser cache to be populated when that cache
  directory is configured; a cold cache silently produced zero templates.
  FoldJAX independently parsed the same CIF. Template dtype differences and
  populated HHR coverage remain separate limitations.
- OpenFold3 output atom identities and its derived atom-slot mask match all
  seven cases. The per-recycle MSA correction has passed patch admission and
  is applied to the working tree: the trunk now gathers each cycle's rows,
  rather than fixing one subset. Captured selected rows reproduce all four
  MSA arrays exactly on all seven cases; 30 focused tests cover validation,
  compact storage and scan/eager cycle execution. This is not full-model RNG
  replay or a new five-sample output result.

### Expanded input-kind panel (2026-09-05)

The follow-up [consumer-contract audit](preprocessing-contract-audit.md)
records the subsequently corrected OpenFold3 cyclic input and Protenix ligand
confidence mask, as well as the fresh, still-failing independent-input Boltz
5SAK five-sample result. The table below describes the initial bounded panel,
not closure of those later model-output gates.

The independent CPU preprocessing survey covers 36 fixtures across all six
models (216 cells). It includes each of protein, DNA, RNA, CCD ligand, SMILES
ligand and ion; every two-kind combination; all six together; protein/DNA/RNA;
polymer copies; modified residues; disulfide and protein–glycan bonds; affinity;
and three MSA configurations. This is a finite input-kind panel, not every
possible sequence, CCD, stereoisomer or model-specific option.

| model | independently compared cells | common shape/value result | remaining raw-contract differences |
|---|---:|---|---|
| AlphaFold 3 | 35 | all exact, including all 59 arrays and five metadata fields | none in these cells; both use native bucket 256 and share CPP/CCD |
| Boltz-2 | 34 | only `ref_pos` differs, maximum 9.536743e-7 Å | raw key sets differ; common dtypes match |
| ESMFold2 | 34 | all common shapes, dtypes and values exact | native-only conditioning/ignored fields and writer metadata |
| OpenDDE | 33 | all common shapes and values exact | key sets, template dtypes and string storage widths |
| OpenFold3 | 33 | only `ref_pos` differs, maximum 1.430511e-6 Å | key sets and output string storage widths |
| Protenix | 33 | all common shapes and values exact | key sets, numerical storage dtypes and string storage widths |

Boltz-2 uses matching NumPy 2.1.3/RDKit 2026.03.2 and recorded native reference
draws; OpenFold3 uses RDKit 2025.09.6 and recorded native reference draws.
These controls do not replace independently generated features with a shared
archive. Protenix/OpenDDE/ESMFold2 captures use seed 101; this alone does not
establish a shared model-sampler RNG stream. Small differences are reported,
not rounded into bitwise equality. Different raw key sets are not declared
equivalent merely because their intersection matches.

The remaining 14 cells are explicitly classified: nine common-schema
unsupported cases (affinity outside Boltz, explicit paired MSA in Boltz/ESM,
and common covalent bonds in OpenFold3); native Protenix/OpenDDE failures on
ion-only input; FoldJAX rejection of RNA CCD 5MC placed in DNA in those two
models; and both Boltz implementations rejecting different MSA paths for
identical sequences. Correct DNA CCD 5CM and a shared-path homomer MSA are
positive controls. A failed ESM native capture was an adapter path/text error
and passed after correcting the adapter, not the model.

This panel exposed two further corrected defects: AlphaFold 3's tiny-input
extension changed native attention gathers even when ordinary padded storage
was sufficient; Protenix/OpenDDE's molecule IDs merged protein–glycan chains
that native inference labels independently. The fixes preserve native gathers
outside the tiny-storage extension and retain chemical bonds while correcting
the temporary molecule-label convention. Both glycan recaptures now have
exact common feature values.

Separately, the populated Boltz template control replays 1,166 native draws:
common shapes/dtypes match, and only `ref_pos` differs (4.768372e-7 Å).
Its native `template_force_threshold=+inf` sentinel was checked separately
for exact shape/dtype/value agreement; the generic nonfinite rejection was
not disabled.

Raw captures, failures, rechecks and SHA-256 evidence are retained in the
`entity-parity-20260905/expanded-inputs/consolidated-audit.json` benchmark
artifact. Cached unchanged cells and corrected recaptures span working-tree
states, rather than one immutable release commit. Updated model-output n=5
entity RMSDs, active conditioning, full key/consumer reconciliation and broader
native-format input coverage remain unfinished. No new crystal comparison
or GPU inference is part of this expanded panel.

These are bounded preprocessing controls, not exhaustive chemistry coverage,
new five-sample structure results, or a completed release gate.

## AlphaFold 3

FoldJAX carries AlphaFold 3 `v3.0.4` at upstream commit
[`85c4d20505fd5cef05eac22b534d4e793971ae69`](https://github.com/google-deepmind/alphafold3/commit/85c4d20505fd5cef05eac22b534d4e793971ae69).
The upstream package fallback still says `3.0.2` at that tag because the real
version is generated from Git; FoldJAX's explicit provenance constant is the
release identity. Every carried Python/C++ source file is byte-identical to the
commit except the eight paths listed in
[`provenance.py`](../src/foldjax/models/alphafold3/provenance.py), and the suite
compares the whole vendored tree against a clean upstream checkout.
FoldJAX cannot distribute, fetch, or choose a parameter release for the user.
Consequently `af3.bin` has no repository-wide FoldJAX hash: benchmark manifests
must record the hash of the particular user-supplied file they used.

## Boltz-2

The JAX graph tracks commit
[`b1ebfc46ecf57f5414e0d1a6f9027bbb122c53bc`](https://github.com/jwohlwend/boltz/commit/b1ebfc46ecf57f5414e0d1a6f9027bbb122c53bc),
whose package metadata is `2.2.1`. The complete managed bundle is pinned to
`boltz-community/boltz-2` revision
`6fdef46d763fee7fbb83ca5501ccceff43b85607`; it includes the confidence and
affinity checkpoints and the molecule archive. All three file hashes are
enforced by the [asset registry](../src/foldjax/assets.py).

## ESMFold2

Biohub did not assign a semantic release to the source snapshot used for this
port. Its identity is therefore commit
[`ef32577f55da19a4989cd7b22e004dc43a4998cb`](https://github.com/Biohub/transformers/commit/ef32577f55da19a4989cd7b22e004dc43a4998cb).
The default `released` profile combines the ESMFold2 structure checkpoint,
configuration, and CCD at revision
`8fc3ff471022fdce52c77030685eb775de0c00a3` with ESMC-6B at revision
`45b0fa5d7fb06faefbd5e3b89bdcef35d564e79a`. The `structure-only` profile
deliberately omits ESMC-6B and is a different model, not a smaller execution of
the released profile.

The later Hugging Face library integration is tracked as a secondary parity
reference, not silently substituted for the creator implementation above.
Atom-encoder parity uses `transformers` 5.16.1 and the merged
`biohub/ESMFold2-hf` re-export at revision
`bce015efb23b5dc604842d0ab5c2bbb02c7bd3ee`; its API and parameter names differ
from the creator snapshot, while the compared trained tensors are identical.

## OpenDDE

The port tracks OpenDDE `v1.1.1` at commit
[`ddfa1df8aff1babf1fddac4247b7d2351bd0ce9f`](https://github.com/aurekaresearch/OpenDDE/commit/ddfa1df8aff1babf1fddac4247b7d2351bd0ce9f).
Both managed checkpoints come from immutable Hugging Face revision
`eddd563ce96571f784012edd8f045181c8f8627d`:

| profile | published checkpoint | size (bytes) | SHA-256 |
|---|---|---:|---|
| `released` | `opendde.pt` | 2,625,249,069 | `7b826620390afad877ee2babc6a4d0df81b94d3a0be030959853d6a7da0807cc` |
| `abag` | `opendde_abag.pt` | 2,625,271,509 | `5cf37441ddef2a2f148b81dd4a218ad274f996fecaf17dec901ab6cf1351713d` |

`abag` is an antibody-antigen-optimized parameter target on the same
655,791,538-parameter graph. It has its own download/conversion root and is
selected explicitly with `weight_profile="abag"`; it never replaces the
general checkpoint implicitly.

## OpenFold3

FoldJAX implements OpenFold3 `v0.5.0` at commit
[`c4771653c5d0a3ebb0b3af71b05efd64bc44ee86`](https://github.com/aqlaboratory/openfold-3/commit/c4771653c5d0a3ebb0b3af71b05efd64bc44ee86)
and only its default OpenBind checkpoint. `foldjax setup` fetches
`of3-ob-2025-06-30-174k.pt` from the publisher's `openfold3-data` bucket,
requiring size `2,287,872,989` bytes and SHA-256
`bd43301c011d5f87580d3e8b548658869433e4488399feb03035ba248f8e29e4`.

The mapper requires OpenBind's stack-level diffusion pair norm and model version
tensor `[2, 0, 0]`; it does not carry the p1 block-local norm or old
ending-triangle-bias behavior. Legacy p1 and p2 checkpoints are rejected rather
than interpreted through a compatible-looking subset of keys.

The complete common/raw input route is available. Against the exact v0.5.0
publisher checkout, the preprocessing/direct-template suite passes on a real
target ladder containing protein, DNA, RNA, and ligand, and a captured upstream
random tape on 1BNA gives `5.78e-6 Å` all-atom Kabsch RMSD. These are separate
feature and model-core claims; neither means independently sampled predictions
from different random-number generators must be coordinate-identical.

## Protenix

The shared implementation tracks commit
[`4c355be4553512f72453ecbfb65e69f4c35d1413`](https://github.com/bytedance/Protenix/commit/4c355be4553512f72453ecbfb65e69f4c35d1413),
where `protenix/version.py` reports `2.0.0`. Checkpoint labels have their own
versions and do not change that source identity:

| profile | published checkpoint | SHA-256 |
|---|---|---|
| `released` | `protenix_base_default_v1.0.0.pt` | `2b7d5a8b30494514fc47fd2271a16260528cdba170ba09cc112fdecd8f85ec04` |
| `base-20250630` | `protenix_base_20250630_v1.0.0.pt` | `e850a6b16b3d254ba9a7f67cefd9b59e111fb2036ae036d2703d050aff6ab611` |
| `v2` | `protenix-v2.pt` | `8f931f9774a396b67033d0e58628e1834f4a1448165e04254b40a780b0c0d599` |
| `mini-esm-v0.5.0` | `protenix_mini_esm_v0.5.0.pt` | `1301bba9ad322518eace60fd244ded7904439f55317e90d402cc7a0c06026664` |
| `mini-ism-v0.5.0` | `protenix_mini_ism_v0.5.0.pt` | `a90d3040cdb2c84430878ea724ad817698ee68cc8b12f802a91dac3b081521a2` |

The mini profiles additionally pin their matching ESM2-3B or ISM encoder in
the asset registry. The `v2` profile is the original checkpoint identity fetched
from a hash-pinned mirror because the publisher's CDN object is unavailable.
Protenix v2 is a profile of the same FoldJAX backend, not a separate public
model name.

## Experimental records

This ledger identifies what FoldJAX implements. An upstream executable used in
a benchmark is a second identity and can move independently; every benchmark
result must therefore record its actual upstream Git revision, runtime package
versions, input hashes, checkpoint hashes, schedule, and seed. The benchmark
manifests are authoritative when they differ from a short model label.

When a port target changes, update this page and the README table in the same
change. Record an immutable source commit and checkpoint revision or SHA-256;
never replace either with a moving branch or `latest` URL.
