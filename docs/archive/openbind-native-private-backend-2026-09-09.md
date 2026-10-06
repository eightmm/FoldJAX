# Experimental OpenBind native-private dispatch

A subsequent transition trial was rejected and its routing reverted; see
`openbind-transition-control-2026-09-09.md`. The active PairBlock bytes again
match the seven-case snapshot. Evidence remains scoped to the recorded
snapshots, not automatic validation of unrelated newer code.

The existing native-faithful triangle operators can now be selected inside the
production PairBlock with `triangle_kernel="native-private"` on
`compile_predict`, or `OPENFOLD3_TRIANGLE_BACKEND=native-private` for an ambient
choice. This is experimental dispatch, not default or full-model admission.
The default remains cuEq for serial execution and XLA for context parallelism.

Unlike the earlier benchmark context manager, the new route does not replace
module functions. It uses the existing kernel-selection scope and compile
identity. A regression checks that cuEq, XLA and native-private produce three
distinct cached graphs when selected through one compiled predictor factory.

Multiplication returns the native fused residual directly, without adding the
residual a second time. Attention returns an update, with the original block
retaining its residual and ending-node transposes. Multiple samples are mapped
through B=1 kernels, as in the measured diagnostic. This preserves that
arithmetic but does not claim native batched scheduling or its memory bound.
The operator restrictions remain FP32, released widths 64/128, four attention
heads, and no context parallelism. Unsupported cases fail; no silent fallback
is added. Ordinary transition arithmetic is unchanged.

Verification: affected CPU tests: 74 passed, 10 GPU skips. The additional
three-backend compile-identity regression passed separately. Ruff and diff
checks passed for the initial dispatch patch.

## Real-weight dispatch bridge

Queue job 1087 exited zero: 3GCA, n=5, 200 steps, four trunk passes, FP32,
fixed native forward tape, raw confidence and trunk return, three calls. The
new `native-private` source route used no operator monkeypatch and no trunk
injection. All 12 output arrays were bitwise equal to the earlier controlled
monkeypatch arm. The loaded/dumped autotune map was byte-identical; prediction
and repeat archive SHA256s matched their finished records and all repeats were
bitwise equal. This closes the dispatch bridge on this case, not full parity.

Versus native, global-system alignment then per-entity maxima: RNA
0.010814844 Å; ligand 0.006003839 Å. Public maximum differences: pLDDT
0.071147516 points, pTM 0.000251355, ipTM 0.000283468. Raw confidence has no
missing fields; it remains measured, not admitted under a relaxed threshold.
Artifact bundle: `openbind-source-backend-20260909-A3bJ4y`; `comparison.json`
holds per-sample RMSD and raw/public confidence. The reference bridge arm is
`openbind-raw-plddt-20260909-0YkshM/controlled-on`.

Not verified: uninstrumented warm performance through this new dispatch,
full seven-case acceptance, default selection, independent review of this
dispatch patch, commit or push. Other historical monkeypatch panel cases need
their own source-route verification.

### 5SAK source-route bridge

Job 1088 exited zero with the same native tape and historical loaded autotune
map, n=5 and three calls. All 11 fields present in the historical monkeypatch
arm were bitwise identical, including coordinates, all common confidence
arrays, and trunk representations. The new pLDDT logits field was also
recorded; no raw confidence fields are missing. Prediction/repeat hashes
matched finished metadata and each repeat was bitwise equal to the first.

Entity maxima versus native remain protein 0.062239110 Å and ligand
0.193974387 Å. Maximum public differences: pLDDT 2.525835099 points, pTM
0.000518400, ipTM 0.004229971. Thus the integration preserves the improvement,
but does not fix the remaining ligand discrepancy. See `5sak-comparison.json`
in the same bundle. The other five source-route cases were queued as jobs
1089–1093; queue submission alone is not result evidence.

### Remaining source panel and verification

Jobs 1089–1092 exited zero. Each completed case's prediction and repeat hashes
were checked, and all three calls were bitwise equal within that case. These
four cases used fresh autotuning, so differences from historical monkeypatch
runs cannot be attributed solely to dispatch. Native-relative entity maxima:

| Case | Entity RMSD maxima (Å) | Maximum pLDDT difference (points) |
| --- | --- | --- |
| 1UBQ | A 0.134589711 | 1.689036097 |
| 1URN | P 0.069200007; R 0.013396830 | 0.466991771 |
| 7R6R | A 0.086024222; B 0.056761449; D 0.057492357 | 0.754534941 |
| 3V7E | P 0.037483039; R 0.105073297; L 0.025329125 | 0.945306077 |

Whole OpenBind CPU suite: 887 passed, 361 skipped, 25 warnings. Skips are not
native/GPU proof; the independent preprocessing environment was tested
separately as recorded in the input-identity note.

7ST3 job 1093 failed with CUDA OOM (9.35 GiB allocation) before producing a
finished prediction. A subsequent live check found the requested read-only
reviewer executing GPU pytest despite its no-GPU instruction. Only that
reviewer's processes were terminated; unrelated existing collaborators were
left running. The review returned no verdict and is not an independent pass.
The observed pytest started after the recorded OOM, so that observation alone
does not prove the OOM's cause or exclude earlier reviewer activity.

Job 1094 retries the same 7ST3 source, dtype, samples, tape and requested outputs
on a verified idle GPU, loading the saved complete autotune map. Both reviewer
interference and autotune memory pressure remain hypotheses, not established
causes. Dispatch admission remains subject to independent review.

### Completed seven-case source-route panel

Job 1094 exited zero with all requested outputs retained. The loaded/dumped
autotune map was byte-identical. This demonstrates successful execution under
the controlled conditions, not a unique diagnosis of the first OOM: reviewer
activity and autotune execution both changed between attempts.

All seven successful source-route records were rechecked together: prediction
and two repeat archive hashes, 12 output fields bitwise equal on each repeat,
five RMSD samples per entity, native-private selection, no operator monkeypatch
or intermediate trunk injection, and complete raw-confidence field coverage.

| Case | Entity RMSD maxima (Å) | pLDDT max (points) | pTM max | ipTM max |
| --- | --- | ---: | ---: | ---: |
| 1UBQ | A 0.134590 | 1.689036 | 0.001654 | 0 |
| 5SAK | A 0.062239; L 0.193974 | 2.525835 | 0.000518 | 0.004230 |
| 1URN | P 0.069200; R 0.013397 | 0.466992 | 0.000401 | 0.000827 |
| 3GCA | R 0.010815; L 0.006004 | 0.071148 | 0.000251 | 0.000283 |
| 7R6R | A 0.086024; B 0.056761; D 0.057492 | 0.754535 | 0.000340 | 0.000537 |
| 3V7E | P 0.037483; R 0.105073; L 0.025329 | 0.945306 | 0.003269 | 0.003088 |
| 7ST3 | A 0.046807; B 0.109559 | 1.757912 | 0.000654 | 0.004396 |

These are native-relative differences, not confidence scores. Global-system
proper Kabsch precedes entity measurements. All atoms/samples must meet the
coordinate criterion for case-level triage: 3GCA is below 0.05 Å, 1URN/7R6R
are in the deferred 0.05–0.1 Å zone, and four cases exceed 0.1 Å. Confidence
gates remain separate and are not admitted by this structural triage. The
panel uses shared native features and a fixed forward tape, not ordinary RNG
or uninstrumented warm inference. Independent input-path evidence is recorded
separately; no universal port equivalence is claimed.

The second, CPU-constrained read-only review attempt timed out after 180 s
without a verdict. Neither review attempt is an independent approval. The
dispatch patch remains uncommitted pending review; the earlier preprocessing
commit is separate. No remote push or full-model closure is claimed.

An alternate-family advisor was attempted after the two failed review paths.
It returned an explicit provider quota error and no verdict. This is not
independent approval; dispatch admission/main publication remains pending.
