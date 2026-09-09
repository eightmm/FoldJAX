# OpenBind private-kernel seven-case comparison

## Decision

Do not adopt the private route as a verified default yet. The 5SAK protein
improvement is real evidence, but several cases retain entity RMSD above 0.1 A.
These are shared-input core diagnostics, not independent preprocessing parity,
ordinary-RNG evaluation, warm performance or release approval.

## Fixed scope

- Upstream OpenFold3/OpenBind revision `c4771653c5d0a3ebb0b3af71b05efd64bc44ee86`.
- Checkpoint SHA256 `bd43301c011d5f87580d3e8b548658869433e4488399feb03035ba248f8e29e4`.
- FP32, five paired samples, 200 steps, four trunk passes (native three recycles),
  captured forward noise/MSA/augmentation, no intermediate trunk injection.
- One proper whole-system Kabsch fit per sample; entity measurements never refit.
- Candidate snapshot `openbind-private-batched-20260909-QK9YSM` uses existing
  private triangle kernels through PairBlock in MSA/template/Pairformer paths.
  Confidence triangles map independent samples through the B=1 private kernel;
  this differs from native batched scheduling. Public defaults are unchanged.
- 5SAK loads a historical autotune map; other cases tune in separate fresh caches.
  Same-callable repetition is not cross-process reproducibility evidence.

## Results

RMSDs are maxima over five paired samples, in angstroms. Confidence columns are
maximum absolute differences, not average score changes. pLDDT uses 0-100;
pTM/ipTM use 0-1. A coordinate threshold does not admit confidence.

| Case | Entity maximum RMSD | pLDDT | pTM | ipTM |
| --- | --- | ---: | ---: | ---: |
| 1UBQ | A 0.122877 | 1.443453 | 0.001509803 | 0 |
| 5SAK | A 0.062239; L 0.193974 | 2.525835 | 0.000518400 | 0.004229971 |
| 1URN | P 0.012598; R 0.013611 | 1.641659 | 0.000673077 | 0.001069548 |
| 3GCA | R 0.011067; L 0.005486 | 0.125760 | 0.000209792 | 0.000168825 |
| 7R6R | A 0.124255; B 0.086535; D 0.089617 | 0.800121 | 0.000482906 | 0.000897176 |
| 3V7E | P 0.098751; R 0.253790; L 0.140976 | 2.236000 | 0.004070087 | 0.003632607 |
| 7ST3 | A 0.051302; B 0.119007 | 2.503990 | 0.000125882 | 0.000243840 |

## Evidence and limitations

### Independent preprocessing recheck started

Seven-case extension completed: the original native-format JSONs for 1UBQ,
5SAK, 1URN, 3GCA, 7R6R, 3V7E and 7ST3 all pass the same independent-conformer
and draw-replay assertions. Each case uses an isolated temporary native fixture
and restores monkeypatches afterward. No native feature array is copied into
the candidate. Shared draws are only quaternion/translation standard normals;
the generated positions/masks are compared exactly before augmentation.

This extension used helper seed0 (not the core panel's seed101); native Torch
augmentation draws are captured from that run, not reconstructed from a seed.
All 33 common declared model feature shapes/dtypes match and values meet the
existing rtol=atol=1e-6 threshold including ref_pos. The additional head-derived
mask and output atom annotations still need their own explicit comparisons.
Verification: seven CPU invocations completed with PASS, no failures; scoped
Ruff/diff pass. Seed101/capture identity bridge and independent review remain.

Resolved follow-up: native InferenceDataset records `seeds=[0]` as metadata,
but does not seed Python RNG before conformer construction. The native runner
does so outside the dataset. The test fixture now scopes/restores Python seed0,
matching FoldJAX's existing conformer seed scope. Independently built conformer
positions and masks then match exactly; no native coordinates are substituted.
Actual native quaternion/translation normal draws are captured and replayed
through the unchanged NumPy augmentation. Every event must be consumed in order.

The new independent-input test passes for a short protein and actual 3GCA
RNA/PQ0 ligand input, including ref_pos (rtol=atol=1e-6), feature shapes/dtypes,
and exact pre-augmentation positions/masks. Only the candidate-derived
max_atom_per_token_mask is excluded from the native input dictionary comparison.
Verification: 2 passed, 2 deselected; two expected query-only MSA warnings;
scoped Ruff passes. This resolves the failing fixture below, not all seven input
cases, annotation identity or native-reference linkage to the older core panel.
No production conformer or augmentation arithmetic changed.

New draw-only regression currently fails before augmentation replay: independently
generated first-molecule reference coordinates differ in all 15 entries (max
absolute 2.062913 in the observed run). The test deliberately compares, rather
than replaces, those coordinates. Thus controlling rotation/translation alone
is insufficient in this fixture; conformer generation settings/randomness must
be aligned first. This failure is not a final-model RMSD measurement.

The native reference fixture also used `dummy.a3m`, silently excluded by native
`max_seq_counts`; an initial attempted basename without `_hits` was also invalid.
The fixture now uses supported `uniref90_hits.a3m`, exposing the genuine
pre-augmentation mismatch. The new test remains uncommitted and failing; no
preprocessing gate is claimed. Next: inspect native/NumPy RDKit generation seed
and conformer selection before changing production code or weakening assertions.

Current FoldJAX `featurize_query` independently processed the original 3GCA
native-format JSON at seed101, without native feature injection. Of 34 declared
MODEL_FEATURES, 32 have exactly matching shape/dtype/values against fresh native
capture 1081. `ref_pos` differs (max unaligned coordinate difference 12.89022017);
`max_atom_per_token_mask` is present only in the candidate, where it is a derived
head feature. Atom annotation identity was not included in this first probe.

This ref_pos difference is not output structure RMSD or evidence of wrong
chemistry: the NumPy path independently augments each reference molecule using
its own generator. Actual preprocessing draws were not controlled. The existing
`test_torch_parity_preprocessing` explicitly skips ref_pos, so that test cannot
close this requirement. Next is shared augmentation-draw capture/replay while
retaining independent feature construction, not copying native ref_pos.

Verification: CPU probe completed successfully; all 34 declared fields were
enumerated. This does not change the seven-case core results or admit inputs.

### Raw-pLDDT real-weight follow-up completed

Independent review completed with no blocking finding in its response, but the
typed tool result is `no-verdict`, not a formal gate pass. It identified writer
budget/documentation and direct two-arm test follow-ups. The claim that all
non-pair arrays are small and the planner always conservative was corrected:
optional atom logits count toward actual bytes and can cause more pair-logit
omissions. A regression now verifies this coupling while preserving requested
atom logits. Existing writer checks pass five tests and the new coupling test
passes separately. No arithmetic/default changes were needed. The broader
OpenFold3 CPU suite remains running; the direct synthetic two-arm assertion
and final review admission are not claimed complete.

The controlled-on native report is now saved as
`controlled-native-comparison.json`: RNA max RMSD 0.01081484 A, ligand
0.00600384 A; pLDDT-logit max error 0.10439920 and RMSE 0.00611784. Public
pLDDT max difference is 0.07114752 points, pTM 0.00025135492, ipTM
0.00028346756. Raw missing-fields is empty. Coverage is complete for the five
reported raw confidence leaves in this case; numerical equivalence is not.
Independent review's typed verdict is still incomplete, not pass.

Controlled jobs 1085/1086 completed exit 0. Their autotune maps are byte-equal
and contain 402 canonical records; on required complete pre-recorded choices.
All eleven common outputs (coordinates, public/raw confidence, trunk arrays)
are exactly equal between off/on. The sole added output is `plddt_logits`.
Both first-prediction and repeat archive hashes validate and every repeat flag
is true. `controlled-bridge.json` records the map identity and per-output checks.
This establishes output-return neutrality on this finite 3GCA/compiler control,
not all shapes, kernels or independent preprocessing. The prior independently
tuned off/on difference is not proof of a defect in the return option.
Independent review is still running; no commit, push or model admission yet.

Expanded pinned-source config/full-inference validation first yielded 8 passed,
1 failed: the field audit omitted pre-existing `stop_after_inputs` from its
FoldJAX-only declarations. HEAD source confirms this field and early-stop route
predate the raw-output change. The declaration is now included; all four config
tests pass on rerun. The five full-inference tests passed in the expanded run.
This repairs audit bookkeeping, not numerical tolerances or model defaults.
Independent review remains live and no duplicate review was started.

Controlled follow-up submitted as 1085/1086: off dumps its XLA autotune map;
on loads that exact map with `require_complete_aot_autotune_results=true` and
dumps the consumed map. An absent decision must fail, not silently retune.
Both use new caches and the same raw-output snapshot. Pending results must
include map identity and output comparisons; scheduling alone proves nothing.
A read-only independent Claude review was requested for the seven raw-output
production/test files only, excluding other models and experimental artifacts.
Review and controlled GPU results remain pending; no commit/push admission.

1084 completed exit 0. On-output contains `plddt_logits` of shape (5,718,50);
its report has no missing raw-confidence fields. Native pLDDT-logit max error
is 0.11297226, RMSE 0.00669946 (logit units). Public pLDDT max difference is
0.12047895 points. Native structural maxima are RNA 0.01058542 A and ligand
0.00463661 A. This measures coverage and residuals, not full confidence parity.

The off/on bridge validates identical source/input/tape/checkpoint/private route;
the only config difference is `return_plddt_logits`. Both completion digests and
all same-callable repeat flags validate. Nonetheless common outputs differ;
whole-system-fit off/on maxima are RNA 0.00993561 A and ligand 0.00512906 A.
Saved `output-return-bridge.json` records this. Independent autotuning remains
a confound, so output neutrality is not admitted and a compiler-controlled
bridge is still required. These results supersede the pending statuses below.

Follow-up implementation (not part of the seven-case snapshot): current
InferenceConfig/released_config now offer `return_plddt_logits=False`, with an
optional trailing Prediction field. Enabled runs return the already-computed
atom logits; default runs leave it None. Atom padding crop includes this field.
The core replay exposes `--capture-plddt-logits`. Configuration/padding checks
passed 21 tests; synthetic full-inference on/off cases passed 2 tests, including
recomputing public pLDDT from returned logits. Scoped Ruff/diff pass. Real-weight
output-return neutrality, writer integration, review and final-tree GPU parity
remain unverified. Historical missing logits are not retroactively filled.

Writer follow-up: five `write_arrays` tests pass, including opt-in atom logits
preserved under the pair-logit budget and absent from default archives. Real
3GCA output-return controls 1083 (off) and 1084 (on) were submitted serially from
`openbind-raw-plddt-20260909-0YkshM`, using identical native input/tape and three
calls each. Each has a fresh cache, so any difference is not uniquely attributable
to the output-return change without further compiler controls. At submission
checkpoint 1083 is running and 1084 queued; no neutrality result yet.

1083 subsequently completed and its native comparison succeeds: RNA maximum
0.01081484 A, ligand 0.00600384 A. Its output keys equal historical job 1082;
input embedding, pair trunk and distogram are exactly equal, while single trunk
and other outputs differ. Fresh autotuning and changed output schema/source
prevent attributing these differences to one cause; raw unaligned coordinate
differences are not structural RMSD. On job 1084 is still running at this
checkpoint. Broader affected CPU checks: 42 passed, 12 GPU-only skipped, two
expected query-only MSA warnings. Full output-return neutrality remains open.

Detailed per-sample values and raw confidence errors are in the snapshot's
`comparison.json` (5SAK) and `<case>-comparison.json` files. Execution history:
[current panel](openbind-current-panel-2026-09-09.md).

Fresh 1URN/3GCA native captures have 13 rehashed public/raw/trunk artifacts each.
Their input/tape arrays equal historical captures: 57/609 fields for 1URN,
57/605 for 3GCA. This checks capture continuity, not independently produced
FoldJAX inputs. Older native captures lack capture-time public-output hashes.

Candidate raw pLDDT logits are missing: inference computes them but Prediction
does not return them. Existing raw PAE/PDE/distogram/experimentally-resolved
comparisons cover stored entries, including padding. No claim of complete raw
confidence parity follows. Returning missing logits requires a separate output
bridge so compilation changes are not mistaken for arithmetic changes.

Verification: all seven paired reports inspected; 1082 completed exit 0 and
3GCA passes all entity/sample 0.05 A coordinate diagnostics. Only 1URN and 3GCA
meet that criterion across every entity/sample; none is a full model admission.
26 focused replay/report/block
tests passed. Four earlier cases' twelve output archive hashes were rechecked
against completed records. No independent review admission or push for these
diagnostic changes. Remaining tasks include raw output/input
coverage, resolving residuals, and ordinary warm/VRAM measurements.
