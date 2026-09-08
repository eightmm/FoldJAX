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
