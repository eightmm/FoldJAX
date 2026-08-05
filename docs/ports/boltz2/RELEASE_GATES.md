# Boltz-JAX inference release gates

## Supported scope

- Boltz-2 inference; Boltz-1 and training are intentionally out of scope.
- Protein monomer/multimer, DNA, RNA, CCD and SMILES ligands.
- Empty, precomputed A3M/CSV, and MSA-server inputs.
- CIF/PDB templates and contact/pocket constraints.
- Trunk, template/MSA Pairformer, diffusion, distogram, bfactor, confidence,
  steering potentials, and the complete two-stage affinity ensemble.
- Native Torch-free runtime weights with XLA, Tokamax, Pallas, and CUDA 13 cuEq
  backend policies where each kernel is supported.

## Automated gates

`JAX_PLATFORMS=cpu uv run pytest -q` collects 132 tests and must have no skips.
The gates include checkpoint parity for primitive, atom, triangle, MSA,
Pairformer, trunk, diffusion, confidence, bfactor, distogram, and both affinity
modules. Raw-input tests exercise all molecule types, template features,
precomputed MSA/cache determinism, mocked MMseqs2 submit/poll/download contracts,
unpaired and paired multimer feature integration, authentication, transient HTTP
failures, corrupt/unsafe archives, affinity cropping, and the 256-token /
2048-atom affinity limits. `ruff check .`, `compileall`, and `uv lock --check`
must also pass.

## GPU smokes verified on 2026-07-14

- CUDA 13 cuEq affinity: raw YAML → initial structure → best-ipTM selection →
  predicted-coordinate crop → separate affinity checkpoint → both ensemble
  members. A short 3-step, 2-sample-per-stage smoke produced finite
  `2 × 2976 × 3` initial coordinates, affinity value `1.6804434`, member values
  `2.0695601` / `1.2913268`, and binary probability `0.3404060`.
- Steering: raw 1UBQ input with FK resampling plus physical/contact guidance,
  two diffusion steps, and CIF writing produced finite `1 × 608 × 3`
  coordinates.
- Public MSA server: the vendored client completed an actual
  `api.colabfold.com` submit/poll/download/A3M parse, then the high-level
  `boltz_jax.featurize` path generated a processed structure and model MSA
  tensor from a raw sequence with no `import boltz` dependency.

The short smokes validate graph connectivity, not scientific-quality sampling.
Production predictions retain the upstream 200-step defaults. Cross-framework
correctness remains governed by fixed-feature, matched-noise FP32 parity tests;
cached BF16 serving benchmarks are performance evidence, not coordinate-parity
evidence.
