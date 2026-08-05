# Chai-JAX completion gates

Chai-JAX must not be described as complete or release-ready until every
applicable gate below is green on the final relevant tree. Passing leaf-module
tests is not a substitute for end-to-end inference. GitHub publication and
artwork are intentionally deferred; their absence does not weaken the local
correctness contract.

## Functional parity

- [x] Protein, DNA, RNA, ligand, glycan, modified-residue and covalent-bond
  inputs are parsed, tokenized and featurized with upstream parity fixtures.
- [x] Empty, local aligned-Parquet and remote ColabFold MSA paths work with
  content-addressed cache provenance.
- [x] Empty, native-NPZ, local m8/CIF and MSA-server template artifacts are
  supported, including Kalign alignment and four-template collation.
- [x] Native ESM2 generation/cache, precomputed ESM embeddings, contact/pocket/
  covalent restraints, cropping and padding are wired into public inference.
- [x] Native Torch-free bundles cover feature embedding, token embedding, trunk,
  diffusion, confidence and bond-loss projection with strict manifests.
- [x] The corrected trunk passes official `forward_256` parity. Non-trivial
  MSA, template, restraint and recycle branches have independent feature,
  trunk-effect and matched-sampler evidence.
- [x] The diffusion denoiser and Chai midpoint EDM-Heun sampler pass component
  parity and identical-random-tape 200-step comparisons on the tested fixtures.
- [x] Confidence logits, PAE, PDE, pLDDT, pTM/iPTM, ranking and clash fields pass
  the real official 256-crop protein-ligand fixture through score-NPZ output.
- [x] The Python API and CLI execute the native pipeline and write ranked
  `pred.model_idx_N.cif`, `scores.model_idx_N.npz`, and MSA coverage output.
- [x] Fresh-process XLA inference succeeds through 1536 tokens with exact
  same-process repeated fingerprints.
- [x] CUDA 13/cuEquivariance inference succeeds through 1536 from a cold cache
  and at 2048 in a fresh process with a populated persistent compilation cache.
  This does not claim cold 2048 cache construction or XLA 2048 support.

## Scientific parity

- [x] The scientific harness records FASTA and source-weight hashes, prepared
  feature hashes, branch state, settings, seeds and output arrays for both
  backends.
- [x] Identical-random-tape 200-step comparisons report coordinate-component
  RMSE and correlation for protein, RNA+DNA, protein-ligand, deep-MSA, template,
  restraint, recycle-2 and modified/glycan/covalent fixtures.
- [x] MSA, template, restraint, covalent and recycle branch-effect tests show
  that accepted inputs are active rather than silently ignored.
- [x] A real official confidence crop reports logit and confidence-score drift,
  pTM/iPTM/ranking fields, and exact clash counts.
- [ ] Multi-seed and larger-complex evaluation establishes release-quality
  scientific bounds beyond the committed single-seed fixtures.

The green scientific gates are scoped to the fixtures and metrics recorded in
[`SCIENTIFIC_PARITY.md`](SCIENTIFIC_PARITY.md). They do not imply exact
coordinate equality between framework-native PRNG streams.

## Performance and release quality

- [x] With compiled executables cached, the recorded 256-token short smoke is
  faster than matching Torch and uses less peak accelerator memory.
- [x] Cold compile time, cached warm latency and peak memory are separated and
  recorded with configuration and hardware provenance through 1536, plus the
  cached-serving-only 2048 cuEquivariance result.
- [x] The final relevant tree passes the portable CPU and official parity
  suites, builds a wheel, installs it in isolation, and passes installed
  `chai-jax`, `fold`, and `assets` help checks.
- [ ] The installed wheel completes a native-asset fold and writes its public
  CIF and score outputs.
- [ ] A representative production-default Torch/JAX comparison demonstrates
  cached warm latency and peak-memory improvement at matching settings.

## Deferred publication gates

GitHub CI, license/notice and attribution review, repository history, artwork,
and publication metadata are deferred until publication is authorized. They
remain mandatory before a public release, but do not block the local technical
completion verdict.

Until the unchecked local scientific, installed-fold, and performance items
are resolved, the correct status is **active release hardening**, not complete.
