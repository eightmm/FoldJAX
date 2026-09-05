# Independent preprocessing contract audit

This is a partial validation checkpoint dated 2026-09-05, not a release-equivalence claim.
The finite input-kind panel and its exceptions are recorded in
[model-versions.md](model-versions.md#expanded-input-kind-panel-2026-09-05).
Raw dictionaries are compared before classifying differences; matching their
intersection does not establish a complete inference contract.

The [portable result record](../bench/experiments/independent-input-entity-parity-2026-09-05.json)
retains the 202-cell comparison summary and all five Boltz sample pairs for the
failed baseline, XLA control, and corrected fused path. Raw tapes and runtime
logs remain outside the package; publication includes no weights or raw datasets.

## Consumer reconciliation

| Model / fields | Actual boundary | Verification or remaining gap |
|---|---|---|
| Boltz `disto_target` | Native distogram training target, not a learned inference input | Native `model/loss/distogramv2.py` consumes it; common runtime fields are independently compared |
| OpenFold3 `seed`, `repeated_sample`, `valid_sample`, `num_paired_seqs` | Dataset bookkeeping | Explicit `_NON_FEATURES` filtering in `data/featurize.py` |
| OpenFold3 `output.res_id/res_name` | Writer metadata aliases | Map to `output.residue_id/residue_name`; do not compare different spellings as missing chemistry |
| OpenFold3 `max_atom_per_token_mask` | Derived atom-slot validity | Native-derived seven-case audit; portable validation checks atom counts, starts and ownership |
| OpenFold3 `cyclic_mask` | **Runtime relative-position input** | Previously missing; native-query parsing, per-token mask, archive/padding validation and all three relative-position blocks now preserve it |
| ESMFold2 `pocket_feature`, `gt_coords`, `is_resolved`, `frames_idx` | Explicitly ignored by the creator model | Not model-input equivalence for a different checkpoint or training mode |
| ESMFold2 `disto_cond`, `disto_cond_mask` | Optional conditioning | Zero/inactive in the tested panel; active conditioning remains unverified |
| ESMFold2 string arrays | FoldJAX writer metadata | Not extra learned input channels |
| Protenix/OpenDDE compact relative-position fields, `d_lm`, `v_lm` | Runtime values constructed at different stages | Seven-case derived-input comparisons reconstruct the categorical representation and compare geometry |
| Protenix `is_ligand` | **Confidence/ranking input** | Previously omitted at the caller; now generated from polymer identity, padded, retained by the compiled-input filter and supplied as `atom_is_polymer` |
| Protenix/OpenDDE `entity_mol_id`, `mol_atom_index` | Label/chain permutation metadata | Native unlabeled inference does not invoke the label-permutation path; labeled/training use is outside this audit |
| Protenix/OpenDDE `bond_mask`, `resolution` | Loss bookkeeping | Distinct from runtime `token_bonds`, which is compared |
| Protenix `pae_rep_atom_mask`, `plddt_m_rep_atom_mask` | Confidence loss/label permutation | No standard unlabeled coordinate-network consumer identified; this is not a training parity claim |
| Protenix/OpenDDE `deletion_matrix`, alignment counters | MSA construction intermediates/statistics | Compare resulting MSA, deletion transforms, profile and cycle selection; post-padding/sampling mask coverage remains separate |
| Protenix/OpenDDE atom/token provenance and chemical bond arrays | Output, constraint and OpenDDE structural expansion | Not blanket-ignored: each downstream path needs its own gate |

Native Protenix passes `1 - is_ligand` to confidence computation in
`protenix/model/protenix.py`. Using `argmax(restype) == 20` alone can classify a
modified polymer as ligand. The current raw builder emits both atom-level
`is_ligand` and token-level `token_is_ligand`; all 66 independent Protenix and
OpenDDE recaptures reproduce native atom ligand masks exactly. Synthetic
five-sample clash controls match native for protein/protein, protein/ligand and
ligand/ligand. This does not replace real-weight confidence-output parity.
Legacy archives with `token_polymer_type` retain polymer identity through a
fallback; native atom-only `is_ligand` also reconstructs token identity. Archives
without either source still use the historical restype fallback and should be
regenerated before claiming modified-residue confidence parity.

## Cyclic input correction

OpenFold3 v0.5.0 has a native `chains[].cyclic` field. The carried schema lacked
it, and the JAX relative-position calculation lacked cyclic offsets. Both now
carry the field through the native-input route. The implementation reproduces
native ordered-token wrapping per sample and chain, including its ties-to-even
rounding for odd-length rings; it does not substitute a generic shortest-path
distance. Six independent native/JAX encoding controls match exactly. A native
20-residue cyclic query has identical masks and common feature values except
reference coordinates (maximum 7.152557e-7 Å with recorded reference draws).

The cyclic field is optional for old linear archives. When present it must be a
boolean token-axis array and cannot mark padded tokens. Tests cover native
query ingestion, archive round-trip, padding, mixed chains and sample-specific
subsets. The old implementation fails five of the six encoding regressions.
General common-schema covalent bonds and native pocket constraints are still
separate unsupported paths, not implied by cyclic relative-position support.

## Fresh Boltz 5SAK: failed baseline and precision controls

The fresh experiment independently constructs native and FoldJAX inputs using
matched NumPy/RDKit and captured reference draws. Common shapes and dtypes agree;
the only value difference is `ref_pos`, at most 4.768372e-7 Å. The native sampler
tape is replayed for five paired samples at 200 steps and three configured
recycles (four trunk passes), without MSA subsampling. Both sides use FP32 and
highest JAX matmul precision. However, the initial run disabled fused attention
but inadvertently retained the separate default cuEquivariance multiplication
backend; that wrapper does not explicitly forward JAX precision. Thus this run
must **not** be described as all-XLA or as verified IEEE multiplication. It is
not the native mixed-precision throughput arm.

One global proper Kabsch is fitted per sample on all 3,073 valid atoms. There
is no entity refit or sample reassignment.

| Sample | Protein RMSD (Å) | Ligand RMSD (Å) |
|---|---:|---:|
| 1 | 0.440135 | 14.233100 |
| 2 | 0.166934 | 0.021855 |
| 3 | 0.057874 | 0.002340 |
| 4 | 1.018202 | 0.153306 |
| 5 | 0.481395 | 0.050531 |

Substituting the captured native trunk while retaining the independent FoldJAX
inputs and same sampler tape reduces entity maxima to protein 0.001340 Å and
ligand 0.000264 Å. This localizes the large discrepancy to trunk outputs on
this actual target/schedule, but does not establish whether its origin is a
port defect or numerical amplification. It is not a successful port gate.
The fresh trunk differences are single RMSE 0.006424/max 0.069565 and pair
RMSE 0.013462/max 0.395302; correlations alone do not resolve their cause.

On identical captured first-cycle inputs, switching multiplication alone to XLA
reduces layer 63 pair-output RMSE from 0.002781 to 0.000002669. All 24,444,032
pair elements then satisfy `atol=rtol=1e-4` (one single-output element does not).
This is a stage diagnostic, not a five-sample output pass. The first IEEE fused
control failed before inference because CUDA libraries were not initialized;
the explicit-runtime retry succeeds and gives layer 63 pair RMSE 0.000002673,
with zero pair elements outside the same tolerance. Selecting IEEE on the same
fused kernel therefore recovers the XLA-level stage agreement.

The complete XLA-multiplication control, with the independently constructed
inputs and the same five sampler trajectories, gives:

| Sample | Protein RMSD (Å) | Ligand RMSD (Å) |
|---|---:|---:|
| 1 | 0.002365 | 0.000533 |
| 2 | 0.000783 | 0.000094 |
| 3 | 0.000719 | 0.000027 |
| 4 | 0.000393 | 0.000064 |
| 5 | 0.003544 | 0.000267 |

The legacy harness's 0.5 Å scalar diagnostic passes, but this is **not** bitwise
equality or an approved universal entity tolerance. Maximum unaligned coordinate
component difference is 0.082363 Å; small average RMSD does not erase this tail.
All five paired coordinates and original failed outputs are retained separately.

The shared multiplication wrapper now translates the active JAX precision policy
to cuEquivariance's explicit enum: `highest`/`float32` selects IEEE, while default
retains DEFAULT. This applies to Boltz and the shared Protenix/OpenDDE path;
unsupported explicit algorithms fail rather than silently downgrade. It does not
change the shipped default to IEEE or establish full Protenix/OpenDDE output
parity. The Boltz replay CLI now selects multiplication separately from attention
and records the selected backend; its numerical-control default is XLA.

The actual patched fused path (no kernel monkeypatch) also completes all five
paired samples. Protein RMSDs are 0.003870, 0.000505, 0.001053, 0.000754 and
0.001685 Å; ligand RMSDs are 0.000689, 0.000069, 0.000037, 0.000105 and
0.000184 Å. Global maximum RMSD is 0.003859 Å. The maximum unaligned coordinate
component difference remains 0.146011 Å, so neither strict coordinate equality
nor an all-atom maximum-error gate is claimed. This verifies the large-error
fix for this target, not every chemistry/model combination.

Verification of this update: 1,499 orchestration tests passed (20 skipped),
132 cyclic/preprocessing/confidence tests passed, and 29 precision-policy,
kernel-forwarding and replay-configuration tests passed. Another 45 adjacent
attention/precision/confidence regressions also pass. Of the
20 orchestration skips, 18 are intentionally non-universal execution-knob
combinations. The other two upstream-source checks now accept
`ALPHAFOLD3_SOURCE` and pass against the pinned checkout; the tree check uses
Git-tracked native source rather than ignored editable-install license copies.
Native cyclic controls
and 66 ligand-identity recaptures are described above. The initial fused IEEE
diagnostic's runtime-import failure was resolved by initializing JAX's CUDA
runtime before standalone cuEquivariance import; the successful rerun is kept
separately from the failure.

Evidence is retained under the benchmark run
`entity-parity-20260905/independent-n5/protein_ligand_5sak/boltz2`, including
input audit, both tapes, paired coordinates and entity reports. No experimental
coordinates were used. The user accepted publication of this partial checkpoint;
the full scientific release gate remains incomplete and is not waived by a main
branch commit. No release tag is created.

## Publication review corrections

The replay harness now rejects an OpenFold3 runner YAML whose effective
seed/recycle/step values differ from the recorded request. OpenDDE checks every
post-writer sample RMSD for nonfinite values; Python's scalar `max` could
otherwise hide a NaN in a later sample. ESMFold2 capture/classification is
available, but its unfinished full-tape replay fails before loading weights
because the carried core does not yet accept every stochastic input.

Three OpenFold3 composite atom-logit checks had been relaxed to `3e-3` after
observing approximately `2.1e-3` deviations. That relaxation is withdrawn:
the declared `1e-4` tolerance is retained. These native-runtime composite gates
remain unresolved and are not counted as strict parity successes. The portable
CPU suite skips publisher-runtime comparisons when PyTorch is absent; its pass
does not close those gates. The review-correction regression selection passes
68 tests, including the mismatched-runner and later-sample-NaN controls.

The first full CPU publication check found 15 stale test-double failures after
the cycle-MSA API change (3,560 other tests passed). Four fixture modules now
provide the current planner/config/mask contract; all 15 focused regressions
pass without weakening runtime validation or changing model calculations.

## CI asset isolation

The published checkpoint's hosted CI exposed one test-only dependency: the SEP
confidence-identity regression loaded a full local CCD database without declaring
it. The regular regression now supplies a small synthetic phosphoserine CIF and
RDKit molecule, disables managed-asset fallback, and exercises the real parser,
modification expansion and ligand-identity builder. Synthetic coordinates are
test data, not scientific reference coordinates. A separate `official_parity`
test retains the full-database integration check and skips explicitly if those
publisher assets are unavailable. No model computation or CI job is removed.

Run the portable regression with
`pytest tests/models/protenix/test_featurize_complete.py -k modified_polymer_is_not`;
run the asset-backed check with
`pytest tests/models/protenix/test_featurize_complete.py -k modified_polymer_identity_with_official_ccd`.
Scientific reruns must still use pinned real assets and the five-sample protocol
above; passing the synthetic test is not a new model-parity result.
