# Recycling defaults: paper inference settings

Policy selected on 2026-09-08: prefer each selected model's published inference
or benchmark schedule over training tables and generic algorithm signatures.
For AF3, the subsequent explicit decision selects Algorithm 1 instead: four
total trunk passes. This change concerns recycling only. Diffusion steps, samples, seeds, MSA data,
precision and checkpoints must also match to reproduce a paper's results.

## Effective common API defaults

| Managed model | Common `num_recycles` | Executed main trunk passes | Evidence and status |
| --- | --- | --- | --- |
| AlphaFold 3 | **3** | **4** | Explicitly select SI Algorithm 1, `N_cycle=4`, over the timing schedule. |
| Boltz2 | **5** | 6 | Paper Appendix D.1 PDB evaluation uses five recycling rounds; changed from native CLI default 3. |
| Protenix base v1.0.0 | 10 | 10 | Protenix-v1 section 3.1 evaluation fixes inference recycles at 10. |
| OpenDDE | 10 | 10 | **Publisher inference fallback**, not a verified paper benchmark count. |
| OpenFold3 / OpenBind-0 | 3 | 4 | **Publisher checkpoint default fallback**; the older preview2 report is not evidence for OpenBind-0's evaluation schedule. |
| ESMFold2 | **9** | **10** | Paper Appendix A.2.11 specifies ten loops; Algorithm 1 iterates T times. Current port adds one to `num_recycles`. |

AF3, Boltz2 and ESMFold2 defaults are supplied by their common backend adapters for
predict and cache warm, whether padding is on or off. Explicit common or native
recycle options take precedence. Their effective counts remain in compile
profiles: omitted and explicitly equal requests share a namespace, while former
release defaults remain separate. Native low-level model APIs and checkpoint
files retain their publisher defaults. Other Protenix variants retain their
variant-specific policy (mini/tiny: 4).

## Primary evidence and differences in counting

- [AlphaFold 3 supplementary information](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41586-024-07487-w/MediaObjects/41586_2024_7487_MOESM1_ESM.pdf):
  Algorithm 1, printed p. 9, explicitly has `N_cycle=4`, iterating 1 through
  `N_cycle`. Section 5.10, printed p. 33, reports timings using 10 trunk recycles.
  Thus the user's reference to 4 was valid; it is not solely another model's
  setting. The released [runner](https://github.com/google-deepmind/alphafold3/blob/main/run_alphafold.py)
  defaults to 10, while the released model runs `num_recycles + 1`. The common
  backend now supplies 3 to execute exactly four passes, matching Algorithm 1.
  Explicit `num_recycles=10` still selects the publisher's 11-pass route.
- [Boltz-2 paper, author-hosted PDF](https://jeremywohlwend.com/assets/boltz2.pdf):
  Appendix D.1, printed p. 35, specifies 5 recycling rounds, 5 samples and one
  seed for PDB structure evaluation. Appendix B.5.1 uses 5 for affinity too.
  The 3 recycles in Appendix A.1.3 belong to **Boltz-1 distillation data**, not
  Boltz-2 PDB evaluation. The common structure sample default is still 1;
  this is a recycle-policy change, not full benchmark reproduction.
- [Protenix-v1 report](https://github.com/bytedance/Protenix/blob/main/docs/PTX_V1_Technical_Report_202602042356.pdf):
  section 3.1, printed p. 3, fixes inference recycles at 10, with five seeds
  and five diffusion samples per seed. Base v1.0.0 is the managed checkpoint.
- [OpenDDE report](https://arxiv.org/html/2607.03787v1):
  section 2, Table 1(b) lists `model.N_cycle=4` in training/sampling parameters.
  We did not find a separate numerical benchmark recycle count in this report.
  The publisher's [inference demo](https://github.com/aurekaresearch/OpenDDE/blob/main/inference_demo.sh)
  sets `N_cycle=10`. This supports a fallback of 10, not a claim that the paper
  explicitly benchmarked at 10.
- [OpenFold3-preview2 report](https://portal.openfold.omsf.io/reports/of3p2_technical_report.pdf):
  Results section specifies 10 trunk recycles and five seeds times five samples.
  However, FoldJAX's managed OpenFold3 checkpoint is OpenBind-0, not preview2.
  The [OpenBind release](https://openbind.uk/news/blog-openbind-0-advancing-open-molecular-structure-prediction/)
  and its linked [benchmark results](https://github.com/OpenBind-Consortium/OpenBind-0-model-release-info/blob/main/benchmarking_results/README.md)
  did not establish that same recycle schedule. Keep the checkpoint default 3
  additional recycles (4 passes) rather than label a different model's protocol
  as verified for OpenBind-0.
- [ESMFold2 paper](https://biohub.ai/papers/esm_protein.pdf?__clerk_synced=true):
  Appendix A.2.11 (printed p. 49) identifies ten loops and 68 diffusion steps
  as default inference settings; Algorithm 1 (printed p. 37) loops from 0 to
  T-1. The current carried native implementation interprets its `num_loops`
  option as additional iterations and performs `max(1, num_loops+1)`; FoldJAX
  preserves that explicit-option contract as `num_recycles`. The managed
  default is therefore 9 to execute the paper's ten total loops. Passing 10
  explicitly still executes 11. The released checkpoint's 14 diffusion steps
  and 32 samples are unchanged by this recycling-only request; it is not the
  complete paper inference protocol.

## Verification

Source-level loop bounds and primary paper/report passages were checked.
Focused CPU tests: **577 passed** across `test_cache.py`, `test_padding.py`,
`test_backends.py`, `test_esmfold2_backend.py`, `test_cli.py` and
`test_resume_manifest.py` (`JAX_PLATFORMS=cpu uv run pytest -q ...`).
They cover request translation, padded and unpadded paths,
explicit override precedence, cache separation and backend dispatch. No GPU
inference, numerical parity, memory or timing measurement was performed.
OpenDDE and OpenBind benchmark recycle counts remain unverified. AF3 now
selects the unambiguous four-pass Algorithm 1 rather than the timing setting.

AF3 Algorithm 1 follow-up: **519 CPU tests passed** across cache, AF3 session,
padding, backend and resume-manifest tests. The session test observes the
actual runner receiving `num_recycles=3`; the carried loop executes `3+1`
passes. Omitted/explicit defaults reuse a runner and cache namespace, while
an explicit 10 remains a distinct profile. Ruff and whitespace checks passed.
