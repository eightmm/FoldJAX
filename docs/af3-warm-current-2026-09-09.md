# AF3 current-source warm comparison

Status: partial measurements completed; later sections supersede submission
status below. Seven-case completion and actual-tape admission remain separate.

External immutable source snapshot: `af3-warm-current-20260909-vnStaz`.
tsp 832 runs pinned native AF3; 833 runs current FoldJAX. Input is 1UBQ,
five samples, native ten recycles, native BF16 policy. Both arms reuse the
same existing validated Tokamax manifest and complete XLA autotune decisions.
This is a controlled fixed-kernel comparison, not independently retuned defaults.

Both commands use `--mode performance --warm-repeats 3
--no-preprocessing-observers`. The timed region is ModelRunner.run_inference,
including feature transfer and result transfer to host; pinned native code
materializes outputs with `jax.tree.map(np.asarray, result)` before returning.
Compilation/first inference is separate from the three warm repeats. The
wrapper verifies output repeatability against the first output outside timing.
No inference RNG or preprocessing observer is installed in this arm. It is
not evidence of observed tape identity; the audit arm remains separate.

Device memory stats are process-lifetime allocator statistics, including
initialization and compilation. Report that scope identically for both JAX
arms; do not label the value a reset warm-only peak. AF3 upstream is JAX,
not Torch. No result or speedup is claimed until both artifacts and timing
paths are checked.

Concurrency: prior Claude-owned Boltz MSA job 831 was observed finished.
No Boltz/OpenBind runtime files or Claude scratch files were edited. New GPU
work uses the shared single-slot tsp queue and a separate snapshot/output root.

## First completion and invocation correction

Native job 832 exited 0. Warm repeats were 4.27844872, 4.30588506 and
4.31671911 seconds (median 4.30588506). All three raw outputs were byte-equal
to the first inference. Lifetime peak device allocation was 1,825,833,216
bytes. These are native-only results, not a speedup comparison.

FoldJAX job 833 exited 1 before model inference because the invocation omitted
the required `--reference` directory. The error explicitly names that missing
argument; no model defect or parity result follows. Retry uses the same source,
input and compiler settings, the completed `native/` as reference, and a new
`foldjax-retry/` directory. Failed artifacts remain preserved.

## Completed paired measurement

Retry job 835 exited 0. `comparison.json` was generated with
`compare_arms(..., require_tape=False)`; this is an uninstrumented output bridge,
not a new observed-tape audit. Inputs, identities, weights, shared runtime and
compiler-policy checks pass. Coordinates, raw confidence and extracted
confidence match. Both arms pass `compare_first_warm` and all three warm
outputs match their own first output byte-for-byte.

| Measurement | Native AF3 | FoldJAX |
| --- | ---: | ---: |
| Warm repeat 1 (s) | 4.27844872 | 4.32726722 |
| Warm repeat 2 (s) | 4.30588506 | 4.35391848 |
| Warm repeat 3 (s) | 4.31671911 | 4.38100056 |
| Warm median (s) | 4.30588506 | 4.35391848 |
| Lifetime peak allocated bytes | 1,825,833,216 | 1,825,836,288 |

No speed or memory improvement is demonstrated: FoldJAX's observed median is
about 1.1 percent higher and allocation differs by only 3,072 bytes. Three
repeats in one process per arm do not establish statistical performance
equivalence. The result applies only to 1UBQ and the recorded fixed kernels.

The strict overall comparator remains false solely at config.json and
effective-config.json. Shared fields match; the candidate has the two extra
default fields `foldjax_return_representations=[]` and
`foldjax_stop_after="full"`. Do not silently waive that gate or promote this
one-case performance bridge to full current-panel or RNG admission.

## Protein-ligand expansion queued

tsp 837/838 run native/FoldJAX 5SAK with five samples and three warm repeats,
using the same immutable source as the completed 1UBQ pair. Both retain the
shared fixed Tokamax manifest and the existing 5SAK native XLA decisions;
uncovered decisions fail closed. The FoldJAX command explicitly references
the new native arm. No preprocessing or inference RNG observers are installed.
837 running and 838 queued were verified; no 5SAK warm result is claimed yet.
Claude-owned Boltz MSA activation job 836 was finished before submission;
its source and artifacts were not modified.

The 837/838 attempts both failed before output because the reused 3GCA
Tokamax manifest lacks the 5SAK FP32 GLU shape `[384,32,128]`. The failure is
`No config found`, not numerical drift. The previous successful 5SAK job 594
used `af3-closure-20260906-v4/kernels/protein_ligand_5sak/manifest.json`.
Both warm arms were resubmitted using that case-specific frozen manifest,
the same source and the same 5SAK XLA cache. Failed attempts are retained;
no measured result is claimed from them.

839/840 passed Tokamax selection but failed XLA's complete-cache gate:
19 of 459 unobserved instructions were missing from the old audit cache.
These runs produced no model output. The next paired invocation extends that
cache only in native (`--xla-autotune-extend`), then requires FoldJAX to reuse
the newly emitted native cache without extension. This is an explicit compiler
control bridge, not a numerical-gate relaxation. Preserve old failures and
record extension provenance separately from strict cache-replay evidence.

## Completed 5SAK warm bridge

Jobs 842/843 both exited 0. Artifacts:
`af3-warm-5sak-bridge-20260909-3kvNjd/{native,foldjax,comparison.json}`.
Native extended the missing XLA decisions once and FoldJAX reused the emitted
cache strictly. Comparator compiler/cache checks pass. The only failed checks
are the two exact config checks already explained above; coordinates and all
raw/extracted confidence checks pass. All five coordinate arrays are equal;
entity Kabsch maxima A 1.9883543175048644e-14 and L
1.1096994001486553e-14 angstrom are alignment roundoff. Both first/warm
comparisons pass, with three repeated raw outputs byte-equal in each arm.

| Measurement | Native AF3 | FoldJAX |
| --- | ---: | ---: |
| Warm repeat 1 (s) | 11.71090482 | 11.82578701 |
| Warm repeat 2 (s) | 11.80141255 | 11.91505263 |
| Warm repeat 3 (s) | 11.89026334 | 11.99079251 |
| Warm median (s) | 11.80141255 | 11.91505263 |
| Lifetime peak allocated bytes | 2,622,001,920 | 2,617,004,544 |

The observed FoldJAX median is about 1 percent higher, not a demonstrated
speedup. Lifetime allocation is about 5 MB lower, not a broad memory-saving
claim. This remains a one-process-per-arm controlled performance/output bridge;
it does not observe actual RNG tape or close the seven-case current panel.

## Protein-RNA expansion

The next pair uses 1URN, n=5 and three warm repeats, with the same source
snapshot. The Tokamax manifest is the one used by successful current-tree
1URN audit job 595, not inferred from target size. Native extends its prior
1URN audit XLA cache for the unobserved graph; FoldJAX strictly consumes the
new native cache. No new output or performance result is claimed at submission.

## Completed 1URN warm bridge

846/847 exited 0; artifacts are in `af3-warm-1urn-20260909-rY3ttJ`.
The completed output bridge passes all checks except the same two exact
configuration checks. All five coordinate arrays are equal; protein and RNA
Kabsch maxima are 7.813083125640918e-15 and 9.576138671688784e-15 angstrom.
Raw/extracted confidence checks and both first/warm comparisons pass.

| Measurement | Native AF3 | FoldJAX |
| --- | ---: | ---: |
| Warm repeat 1 (s) | 4.32195017 | 4.32634542 |
| Warm repeat 2 (s) | 4.35004137 | 4.36962698 |
| Warm repeat 3 (s) | 4.35245621 | 4.38296152 |
| Warm median (s) | 4.35004137 | 4.36962698 |
| Lifetime peak allocated bytes | 1,821,116,928 | 1,821,127,936 |

No speedup or memory improvement is demonstrated. Current measured warm
coverage is three cases (1UBQ, 5SAK, 1URN), not the complete seven-case panel.
Unobserved performance bridges do not replace observed actual RNG-tape audits.

## Remaining four cases submitted

Jobs 874/875 cover RNA-ligand 3GCA. Native 874 exited 0 with warm seconds
`[4.293022925034165, 4.326547090953682, 4.325514209980611]` and lifetime
allocator peak 1,820,881,152 bytes; all three outputs equal its first output.
Candidate 875 was still running when this submission record was written.

Jobs 876/877 cover protein-DNA 7R6R, 878/879 protein-RNA-ligand 3V7E,
and 880/881 protein-protein 7ST3. These use the same immutable source snapshot,
five samples and three warm repeats, with no preprocessing or RNG observers.
The first two use the established 3GCA Tokamax manifest; 7ST3 uses its
case-specific v4 manifest. Each native arm extends its prior v4 XLA cache,
and the candidate strictly loads that new native cache without extension.
No outputs from queued jobs are claimed. Opus jobs remain serialized through
the shared queue; no collaborator runtime files were edited.

## Completed 3GCA warm bridge

874/875 exited 0. Artifacts: `af3-warm-3gca-20260909-hR4yI6`, including
`comparison.json`. Five coordinate arrays are exactly equal. RNA R maximum
Kabsch RMSD is 1.0430821714579495e-14 angstrom; ligand L maximum is
2.973102245438931e-15 (alignment roundoff). Input, identity, compiler policy,
raw confidence and extracted confidence checks pass. The overall comparator
exits 1 only for the same two exact configuration checks: no changed shared
fields, just `foldjax_return_representations=[]` and `foldjax_stop_after="full"`
extensions. That strict failure is retained.

| Measurement | Native AF3 | FoldJAX |
| --- | ---: | ---: |
| Warm repeat 1 (s) | 4.29302293 | 4.30420222 |
| Warm repeat 2 (s) | 4.32654709 | 4.34370326 |
| Warm repeat 3 (s) | 4.32551421 | 4.36294239 |
| Warm median (s) | 4.32551421 | 4.34370326 |
| Lifetime peak allocated bytes | 1,820,881,152 | 1,825,838,336 |

All three warm raw outputs match each arm's own first output bytewise. No
speedup or memory improvement is demonstrated. This completes four measured
warm cases, not the seven-case actual-RNG-tape audit. Verification: both GPU
jobs exited 0; output-only comparison produced the recorded config-only strict
failure, with every other check true. Diff check passed.

## Completed 7R6R warm bridge

876/877 exited 0. Result root:
`af3-warm-remaining-20260909-L7Nxep/protein_dna_7r6r`; the output-only
`comparison.json` retains the config-only overall failure. All other checks,
including input/identity, coordinates and raw/extracted confidence, pass.
All five coordinate arrays (2,529 atoms each) are equal. System-fit entity
maxima are A 1.5720217060201747e-14, B 1.1688264056357194e-14 and
D 1.0982263432883787e-14 angstrom, consistent with alignment roundoff.

| Measurement | Native AF3 | FoldJAX |
| --- | ---: | ---: |
| Warm repeat 1 (s) | 4.43363550 | 4.43509066 |
| Warm repeat 2 (s) | 4.46950582 | 4.46863634 |
| Warm repeat 3 (s) | 4.48136740 | 4.49050551 |
| Warm median (s) | 4.46950582 | 4.46863634 |
| Lifetime peak allocated bytes | 1,825,478,656 | 1,820,822,016 |

Three warm outputs in each arm equal their own first output bytewise. No
meaningful speedup is established by the sub-millisecond median difference.
The allocator difference is not a broad memory improvement claim.

Verification: 876/877 exited 0; all comparator checks other than config.json
and effective-config.json are true, with only the two documented default
extension fields and no changed shared configuration fields. A recursive
content comparison of the snapshot's `src/foldjax/models/alphafold3` against
the current working tree (excluding `__pycache__`) found no differences.
That check covers AF3 model source, not every shared runtime file. Current
warm coverage is five cases; 3V7E/7ST3 and actual-tape admission remain pending.

## Completed 3V7E warm bridge

878/879 exited 0. Artifacts are under
`af3-warm-remaining-20260909-L7Nxep/protein_rna_ligand_3v7e`, including
`comparison.json`. Five coordinate arrays (3,317 atoms each) are exactly
equal. System-aligned entity maxima are protein P 2.4442717076183486e-14,
RNA R 1.7561066006793405e-14 and ligand L 4.180125496481331e-15 angstrom.
All input, identity, compiler and raw/extracted confidence checks pass.
The overall comparison remains false only at the two exact config checks:
the same default extension fields, with no changed shared fields.

| Measurement | Native AF3 | FoldJAX |
| --- | ---: | ---: |
| Warm repeat 1 (s) | 4.44476549 | 4.44557052 |
| Warm repeat 2 (s) | 4.46973414 | 4.48303535 |
| Warm repeat 3 (s) | 4.49673387 | 4.50150815 |
| Warm median (s) | 4.46973414 | 4.48303535 |
| Lifetime peak allocated bytes | 1,820,660,480 | 1,821,108,736 |

All three warm outputs equal their own first output in both arms. No speed
or memory improvement is demonstrated. Six warm cases have completed; 7ST3
and actual observed-tape admission remain outstanding. Verification: GPU
878/879 exited 0, output-only comparator has the documented config-only
failure; `tests/test_af3_closure.py` and `tests/test_af3_closure_capture.py`
passed (70 tests). These CPU tests do not establish real-weight tape identity.

## Current-source actual-draw audit follow-up submitted

Jobs 885/886 use the same immutable AF3 source for a 3GCA native/FoldJAX
`--mode audit` pair with preprocessing observers enabled. Native may extend
the completed 3GCA warm XLA cache for the observed graph; FoldJAX must consume
the new native audit cache without extension. This is a distinct actual-draw
and independent-input arm, not a relabeling of the warm run. No audit output
or tape identity is claimed at submission. Compare the completed audit arms
with each other and bridge each to its corresponding warm arm before drawing
an observer-independence conclusion. Opus jobs queued between our batches
remain in their existing order; no jobs were cancelled or restarted.

## Config-extension source check and 7ST3 native completion

The carried model uses `foldjax_return_representations` only to collect
requested intermediate arrays, and `foldjax_stop_after` to return early at
inputs or trunk. With the measured empty tuple and `full`, neither early
return is taken and no representation arrays are added. The seven existing
representation/control-flow tests pass. This source-backed explanation does
not override exact configuration equality or establish arbitrary settings
parity; the historical strict config failures remain.

7ST3 native job 880 exited 0. Warm repeats are
`[24.68483158503659,24.9759571000468,25.125159420014825]` seconds; lifetime
allocator peak is 3,871,702,016 bytes. All three warm outputs match its own
first output. Candidate 881 is running, so no paired 7ST3 result is claimed.

Concurrent commits advanced HEAD to `0ee02f5` during this work. A fresh
recursive AF3 model-source comparison against the execution snapshot still
finds no content differences (excluding bytecode caches). Existing commits
and collaborators' changes were preserved; this check is not remote-push or
release verification.

## Completed 7ST3 and seven-case warm summary

880/881 exited 0. Artifacts:
`af3-warm-remaining-20260909-L7Nxep/protein_protein_7st3/comparison.json`.
Five coordinate arrays of 4,281 atoms are exactly equal. Entity maxima after
the system fit are A 7.601760654448837e-14 and B 3.7028500115005564e-14
angstrom (alignment roundoff). All input/identity, compiler and raw/extracted
confidence checks pass. Only config.json and effective-config.json fail,
with the same two default extensions and no changed shared fields.

FoldJAX warm repeats are `[24.823123686015606,25.066856154997367,
25.202678318019025]` seconds, median 25.066856154997367. Its lifetime
allocator peak is 3,871,668,224 bytes. All three warm outputs equal its own
first output. Native measurements are recorded immediately above.

| Case | Native warm median (s) | FoldJAX warm median (s) | Native peak (GB) | FoldJAX peak (GB) |
| --- | ---: | ---: | ---: | ---: |
| 1UBQ | 4.3059 | 4.3539 | 1.8258 | 1.8258 |
| 5SAK | 11.8014 | 11.9151 | 2.6220 | 2.6170 |
| 1URN | 4.3500 | 4.3696 | 1.8211 | 1.8211 |
| 3GCA | 4.3255 | 4.3437 | 1.8209 | 1.8258 |
| 7R6R | 4.4695 | 4.4686 | 1.8255 | 1.8208 |
| 3V7E | 4.4697 | 4.4830 | 1.8207 | 1.8211 |
| 7ST3 | 24.9760 | 25.0669 | 3.8717 | 3.8717 |

GB is decimal, framework allocator process-lifetime peak, not warm-only or
total GPU VRAM. AF3 native is already JAX. No material speed or memory benefit
is established. All seven output bridges have exactly equal five-sample
coordinates and passing raw/public confidence checks under the recorded
fixed-kernel policy. All retain the exact config failures. These results do
not constitute actual-RNG-tape admission or six-model closure.

Verification: final GPU pair exited 0; comparator completed with config-only
failure. An initial report read ran before the comparator finished and found
no file; after the original comparator handle completed, the emitted report
was read successfully. No inference or comparison was restarted.

## Completed current-source 3GCA actual-draw audit

885/886 exited 0. Root: `af3-audit-current-3gca-20260909-YUmBCq`.
The strict `comparison.json` passes actual tape, tape coverage, preprocessing
tape, independent inputs/identity, parameters, coordinates and raw/extracted
confidence. Only the two previously documented exact config checks fail.
Five coordinate arrays are equal. Observed events are initial 1, churn 1,000,
rotation 1,000, translation 1,000 and padding_gumbel 11: 3,012 total.
RDKit internal PRNG is not observed; seed/conformer boundary equality is the
explicit scope. Candidate preprocessing creates its own features and asserts
equality against native, rather than replacing them with native features.

`native-warm-bridge.json` and `foldjax-warm-bridge.json` both pass every
output-only comparator check against their corresponding completed warm arm.
Thus adding these observers did not change the saved coordinates/confidence
in this case and kernel policy. This is current 3GCA evidence only; the other
six current-source observed-tape pairs remain to be established. No strict
config waiver, general observer-independence claim or model closure follows.

Verification: both GPU audit exits are 0; actual-draw comparison has only the
documented config failures; both audit/warm bridges exit 0. Preserve these
reports separately from the seven-case unobserved warm summary.

## Remaining six actual-draw pairs submitted

The queue and GPU were idle at preflight. Jobs 889–900 use the same fixed
source snapshot, `--mode audit`, native n=5 and preprocessing observers:

| Case | Native job | FoldJAX job |
| --- | ---: | ---: |
| 1UBQ | 889 | 890 |
| 5SAK | 891 | 892 |
| 1URN | 893 | 894 |
| 7R6R | 895 | 896 |
| 3V7E | 897 | 898 |
| 7ST3 | 899 | 900 |

Each native arm extends its own completed warm cache if the observed graph
requires additional decisions; its candidate strictly reuses the resulting
native audit cache. Tokamax manifests match each case's completed warm pair.
No successful tape, output or observer-bridge result is inferred from
submission. Compare every completed pair with tape required, then compare
each arm to its corresponding warm result with output-only scope. 1UBQ's
successful candidate warm directory is `foldjax-retry`, not the failed
original `foldjax` attempt. Existing artifacts are preserved.

Verification at submission: all referenced native warm completion files and
kernel manifests exist; job 889 is running and later jobs are queued.

## Completed current-source 1UBQ audit

889/890 exited 0. Root:
`af3-audit-panel-20260909-IIR5zw/protein_1ubq`. The strict actual-tape
comparison fails only the same two configuration equality checks. All tape,
coverage, preprocessing, independent input/identity, parameter, coordinate and
confidence checks pass. Entity A maximum RMSD is 6.415250900883943e-15
angstrom. Both `native-warm-bridge.json` and `foldjax-warm-bridge.json` pass
every output-only comparator check against the completed warm pair.

Verification: both GPU jobs exited 0; the three comparison reports completed
successfully, with the strict config-only failure preserved. This adds 1UBQ
to 3GCA's current-source actual-draw and observer-bridge evidence; the five
other queued cases remain pending, as does exact configuration admission.

## Completed current-source 5SAK audit

891/892 exited 0. Root:
`af3-audit-panel-20260909-IIR5zw/protein_ligand_5sak`. The actual-tape
comparison passes every check except the two documented exact config checks.
This includes tape coverage/equality, preprocessing tape, independent inputs,
identity, coordinates and raw/extracted confidence. Entity maxima are protein
A 1.9883543175048644e-14 and ligand L 1.1096994001486553e-14 angstrom.
Both native and FoldJAX audit/warm output bridges pass every check.

Verification: both GPU jobs exited 0; three completed comparison reports were
read, retaining the strict config-only failure. Current-source audit and
observer-bridge evidence now covers 1UBQ, 3GCA and 5SAK (3/7); no acceptance
threshold was changed. This AF3 5SAK result does not resolve Boltz's separate
5SAK discrepancy.

## Completed current-source 1URN audit

893/894 exited 0. Root:
`af3-audit-panel-20260909-IIR5zw/protein_rna_1urn`. Strict actual-tape
comparison passes all checks except the documented two exact config checks.
This includes draw identity/coverage, preprocessing tape, independent inputs,
parameters, identities, coordinates and raw/extracted confidence. Protein P
maximum system-fit RMSD is 7.813083125640918e-15 angstrom; RNA R maximum
is 9.576138671688784e-15. Both native and FoldJAX audit/warm output bridges
pass every check. These are alignment-roundoff residuals, not a relaxed gate.

Verification: 893/894 exited 0; three completed reports were inspected.
Current-source audit and observer-bridge coverage is now 4/7 (1UBQ, 5SAK,
1URN, 3GCA). The remaining three cases are still in the serial queue; strict
config admission and broader six-model work are not complete.

## Completed current-source 7R6R audit

895/896 exited 0. Root:
`af3-audit-panel-20260909-IIR5zw/protein_dna_7r6r`. The actual-tape
comparison passes all checks except the same two exact config checks.
Draw coverage/equality, preprocessing, independent inputs/identity,
parameters, coordinates and raw/extracted confidence pass. System-fit entity
maxima (angstrom) are A 1.5720217060201747e-14, B 1.1688264056357194e-14,
and D 1.0982263432883787e-14. Both audit/warm output bridges pass every check.

Verification: GPU 895/896 exited 0, all three reports completed and were
inspected. Current-source audit plus observer-bridge coverage is 5/7;
3V7E and 7ST3 remain queued/running. The config-only strict failure and RDKit
internal-PRNG observation limitation remain unchanged.

## Completed current-source 3V7E audit

897/898 exited 0. Root:
`af3-audit-panel-20260909-IIR5zw/protein_rna_ligand_3v7e`. The strict
actual-tape comparison passes every check except the two documented exact
configuration checks. Draw identity/coverage, preprocessing tape, independent
inputs, parameters, atom identities, coordinates and raw/extracted confidence
pass. System-fit entity maxima (angstrom) are protein P
2.4442717076183486e-14, RNA R 1.7561066006793405e-14 and ligand L
4.180125496481331e-15. Both audit/warm bridges pass all output-only checks.

Verification: both GPU jobs exited 0 and three reports were inspected.
Current-source actual-draw plus observer-bridge evidence covers 6/7 cases;
7ST3 remains pending. No config waiver or broader release admission is made.

## Completed seven-case current-source actual-draw panel

899/900 exited 0. Last-case root:
`af3-audit-panel-20260909-IIR5zw/protein_protein_7st3`. Its actual-tape
comparison fails only the two exact configuration checks. Coordinates are
equal; system-fit maxima are A 7.601760654448837e-14 and
B 3.7028500115005564e-14 angstrom. Both audit/warm bridges pass every check.

All seven actual-tape reports and all fourteen audit/warm bridge reports were
then re-read together. Every audit has exactly the two documented config
failures and every other check true, including actual draw coverage/identity,
independent inputs, preprocessing observations, atom identities, parameters,
coordinates and raw/extracted confidence. All five coordinate arrays in every
case are equal. Every bridge has `passed=true` with all checks true.

This closes collection and comparison of this fixed-kernel AF3 seven-case
panel, not strict configuration admission, arbitrary-input support, independent
autotuning equivalence, release verification or the six-model goal. RDKit
internal PRNG remains unobserved; its captured boundary is explicit. AF3
speed/memory improvement is not established. No numerical tolerance was
relaxed and no model-intermediate native output was injected.

Verification: 899/900 exited 0; seven audit comparisons and fourteen bridges
passed the aggregate assertions above except the preserved strict config
checks. Scope remains pinned native policy, fixed Tokamax/XLA decisions and
the recorded source snapshot. Commit/push and independent release review are
separate outstanding work.
