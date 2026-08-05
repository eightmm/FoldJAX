# Torch/JAX Scientific Inference Parity

This benchmark compares the official `chai_lab` Torch inference path with the
standalone Chai-JAX path. It is deliberately separate from performance
benchmarking: a faster result is not scientifically equivalent unless the
input, source weights, branch configuration, and randomness contract match.

## Contract

Every artifact records:

- FASTA SHA256 and fixture ID;
- all six official source-weight SHA256 values (the native bundle manifest
  retains these hashes);
- recycle, diffusion timestep, sample, and seed settings;
- MSA, template, and restraint branch state;
- all 32 prepared model-feature shapes and semantic hashes;
- coordinates, PAE, PDE, pLDDT, ranking, pTM/ipTM, and clash arrays.

Scalar Torch features retain a final singleton channel that the JAX bridge
removes. The hash normalizes that layout. Masked `ResidueType` padding is also
normalized because Torch uses zero while JAX uses the dedicated category 32.
Float feature hashes are exact. The one declared numerical exception,
`InverseSquaredBlockedAtomPairDistances`, is stored in full and compared with
`atol=2e-7, rtol=1e-6`; its actual drift is included in the report. A mismatch
outside that bound, or in any other feature, blocks the comparison.

Atom rows are joined by `(chain, sequence ID, component, atom name, occurrence)`
before comparison. The report includes all-atom and CA Kabsch RMSD, unaligned
coordinate drift, pair-distance drift, PAE/PDE/pLDDT drift, rankings, and
clashes. Missing or extra atoms fail instead of being silently dropped.

Two unaligned coordinate metrics must not be conflated. The historical
`coordinate_rmse` field is xyz component RMSE,
`sqrt(mean((jax_xyz - torch_xyz)^2))`. Conventional raw all-atom RMSD is
`sqrt(mean(sum((jax_xyz - torch_xyz)^2, axis=-1)))`, which is exactly
`sqrt(3)` times component RMSE for the same joined atoms. Release thresholds
use the latter `all_atom_raw_rmsd`; Kabsch-aligned RMSD is reported separately.

## Randomness boundary

The public Torch and JAX APIs use different PRNG algorithms. An equal integer
seed therefore matches the schedule and distributions, not the actual initial
noise, churn noise, rotations, or translations. End-to-end reports generated
that way are explicitly labeled `seed-and-distribution` and are **fold-level
diagnostics, not a scientific parity pass**.

`benchmark_sampler_parity.py` is the component/noise-matched route. It creates
one NumPy random tape and supplies the exact initial noise, per-step churn
noise, rotations, and translations to the official TorchScript diffusion
component and the mapped JAX component. Public end-to-end noise replay remains
blocked because neither public orchestration API currently accepts a random
tape.

## Fixtures and branches

`benchmarks/scientific_parity/manifest.json` defines protein, RNA+DNA, and
protein-ligand fixtures. It also defines baseline, recycle-2, restraint, MSA,
and template branch comparisons. The MSA case requires matching `.aligned.pqt`
assets, and the template case requires matching m8 and CIF assets; these are
not fabricated by the harness.

## Small-step evidence (2026-07-14)

These runs used one recycle, one sample, seed 0, and only two diffusion
timesteps. They validate the harness and **do not test 200-step production
quality**.

- Protein: 32/32 prepared features and all source weights matched. All-atom
  Kabsch RMSD was 542.751 A and CA RMSD was 407.959 A under distribution-only
  randomness. PAE/PDE/pLDDT MAE was 0.08186/0.01147/0.01940. This is RED for
  coordinate parity.
- Protein + ligand: 32/32 prepared features and all source weights matched.
  All-atom Kabsch RMSD was 524.338 A and CA RMSD was 391.456 A. PAE/PDE/pLDDT
  MAE was 0.92707/0.28502/0.02089. This is RED for coordinate parity.
- RNA + DNA: 31 features matched exactly. The distance feature differed in
  1,364 values with MAE `1.65e-11`, maximum `1.19e-7`, and zero values above
  `1e-6`; the tensor-specific raw-array check passed. All-atom Kabsch RMSD was
  586.112 A (there are no CA atoms). PAE/PDE/pLDDT MAE was
  0.06485/0.01849/0.01202. Coordinate parity remains RED.
- Noise-matched diffusion component/sampler: coordinate-component RMSE
  0.23143 A, component MAE
  0.12360 A, maximum absolute drift 3.55127 A, correlation 0.99999978. This
  isolates numerical component/sampler drift, but is still not a production
  end-to-end parity result.
- MSA, template, restraint, recycle-2, and 200-step branches were not run in
  this small-step validation. They remain open gates.

The machine-readable evidence is in
`benchmarks/scientific_parity/results/2026-07-14-small-step.json`.

## Commands

Run each backend in its own environment:

```bash
uv run python scripts/export_scientific_parity.py jax \
  --fixture-id protein \
  --fasta benchmarks/scientific_parity/fixtures/protein.fasta \
  --artifact /tmp/protein.jax.npz --run-dir /tmp/protein-jax \
  --bundle /tmp/chai_jax_native_bundle \
  --conformers /tmp/chai_jax_conformers_v1.npz \
  --recycles 1 --timesteps 2 --samples 1 --seed 0

../chai/.venv/bin/python scripts/export_scientific_parity.py torch \
  --fixture-id protein \
  --fasta benchmarks/scientific_parity/fixtures/protein.fasta \
  --artifact /tmp/protein.torch.npz --run-dir /tmp/protein-torch \
  --torch-model-directory ../chai/downloads/models_v2 \
  --recycles 1 --timesteps 2 --samples 1 --seed 0

uv run python scripts/scientific_parity.py \
  /tmp/protein.torch.npz /tmp/protein.jax.npz \
  --output /tmp/protein.report.json

XLA_PYTHON_CLIENT_PREALLOCATE=false uv run python \
  scripts/benchmark_sampler_parity.py \
  --timesteps 2 --samples 1 --seed 0 \
  --output /tmp/sampler-matched.report.json
```

Do not set a pass threshold from these two-step runs. A release threshold must
be pre-registered and evaluated with matched 200-step randomness or with a
clearly separated fold-level multi-seed protocol.

## Matched 200-step evidence (2026-07-14)

The production-length sampler uses one NumPy random tape for the initial atom
noise and all 199 churn noises, rotations, and translations. Both backends use
one sample and one recycle on the five-residue protein fixture. Latencies are
warm sampler-only measurements after one complete warm-up.

- With one shared JAX-prepared/trunk context, Torch/JAX coordinate-component
  RMSE was `4.30e-6 A`, correlation was effectively 1, and JAX was `2.96x` faster
  (`2.478 s` versus `7.346 s`). This isolates the sampler and mapped diffusion
  component.
- With framework-native preprocessing, embedding, and trunk and no ESM,
  coordinate-component RMSE was `0.02590 A`, maximum component error
  `0.11003 A`, correlation
  `0.9999567`, and JAX was `3.02x` faster (`2.462 s` versus `7.442 s`).
- With each framework's default ESM path enabled, coordinate-component RMSE was
  `0.02520 A`, maximum component error `0.08715 A`, correlation `0.9999596`, and
  JAX was `2.92x` faster (`2.515 s` versus `7.334 s`).
- The RNA+DNA fixture with ESM disabled had `0.06999 A` component RMSE,
  `0.21871 A` maximum component error, correlation `0.999873`, and `3.01x`
  JAX speedup.
- The protein-ligand fixture with default ESM had `0.31838 A` component RMSE
  and `0.55145 A` conventional raw atom RMSD,
  `1.95927 A` maximum per-coordinate error, correlation `0.993536`, and
  `2.97x` JAX speedup. Its raw atom RMSD is sub-angstrom, but it is materially
  less exact than the protein and nucleic-acid fixtures, so it remains a
  distinct ligand-sensitive slice.

An earlier native-context run had `2.386 A` component RMSE. A Torch-masks-only
hybrid reduced it to `0.0371 A`, identifying 802 mismatched atom block-pair mask
values as the dominant cause. Commit `81b2e64` preserved the reference-space
mask; the post-fix mismatch count is zero. Remaining valid-region trunk NRMSE
is approximately `0.0115`, but does not materially amplify under the matched
200-step tape on this fixture.

Prepared-feature comparisons also pass for the restraint and recycle-2
branches: 31/32 features match exactly, and the distance feature differs by at
most `8.94e-8`, within its tensor-specific `2e-7` tolerance. Recycle count is a
runtime branch and intentionally does not alter prepared features.
The actual two-step JAX restraint inference also writes its CIF and score
artifact after runtime fix `fb2bcba`.

Machine-readable results are in
`benchmarks/scientific_parity/results/2026-07-14-200-step.json`. These are
single-seed gates on small fixtures; multi-seed or larger-complex coverage
remains separate release work.

## Real MSA and template gates (2026-07-15)

The previously open MSA and template branches were evaluated with one recycle,
one sample, seed 0, no ESM, and an identical 200-step random tape. Both use the
released Torch weights and the native bundle made from those same source files.

- The MSA fixture is a 100-residue protein with 33,066 aligned-Parquet rows.
  Raw Torch/JAX parsing matches exactly, including source ordering, pairing,
  deletions, and masks. The joined valid depth is 9,308 and the profile depth is
  12,909 in both backends. Prepared features pass 32/32: 31 are exact and the
  blocked-distance maximum drift is `1.19e-7`.
- The template fixture is ubiquitin (76 residues) with four source-consistent
  m8/CIF templates. Prepared features pass 32/32: 30 are exact, blocked-distance
  maximum drift is `1.19e-7`, and template-unit-vector maximum drift is
  `1.79e-7`. Both are within their explicit `2e-7` tolerances.
- MSA valid-region trunk NRMSE is `0.00973` single and `0.01266` pair. Its
  active-versus-baseline relative L2 effect is `0.36009/0.74366` in Torch and
  `0.36142/0.74634` in JAX, showing that the scientific branch effect is
  preserved rather than merely producing similar final coordinates.
- Template valid-region trunk NRMSE is `0.00905` single and `0.01216` pair. Its
  active-versus-baseline effect is `0.33047/0.73092` in Torch and
  `0.33112/0.72982` in JAX.
- The MSA matched 200-step run has `0.12119 A` coordinate-component RMSE,
  correlation `0.999893`, and `2.94x` JAX warm speedup (`2.534 s` versus
  `7.453 s`). The template run has `0.15800 A` component RMSE, correlation
  `0.999722`, and `3.02x` speedup (`2.462 s` versus `7.431 s`).
  Shared-Torch-static controls are
  `2.29e-5 A` and `1.26e-5 A` component RMSE, respectively.

The official Torch trunk is bitwise deterministic across five repeats in one
process. Across three fresh processes, however, its CUDA token embedder changes
the initial single representation by NRMSE `0.00138-0.00143`; the 48-block
trunk amplifies this to single `0.00918-0.00939` and pair
`0.00886-0.01156`. This noise floor is recorded separately from JAX drift and
does not hide the reported coordinate errors.

The compact, hash-pinned evidence is in
`benchmarks/scientific_parity/results/2026-07-15-msa-template.json`. These MSA
and template gates are GREEN for the tested single-seed fixtures.

## Modified residue, glycan, and covalent-bond gate (2026-07-15)

The mixed fixture contains `AC(SEP)`, a `CS` ligand, a two-residue NAG glycan,
and a declared protein-ligand covalent bond. Non-standard conformers receive an
explicit three-occurrence rigid-transform tape in both preparation paths; the
public inference RNG defaults remain unchanged. Prepared parity passes 32/32
features: 30 are exact, `AtomRefPos` differs by at most `2.98e-8`, and the
inverse-distance feature differs by at most `1.19e-7`.

This fixture exposed two masked-atom bugs. Chai builds reference pair features
from `atom_ref_mask`, because a glycan leaving atom can be absent from predicted
coordinates but remain a valid reference atom. The JAX collator had used
`atom_exists_mask`. The deterministic token pooling path also assumed masked
atoms only occurred after every valid atom, which fails when the leaving-atom
token precedes later glycan atoms. The corrected path preserves the fast prefix
reduction for ordinary inputs and uses a deterministic masked-hole reduction
only when needed; diffusion `atom_single_mask` still uses `atom_exists_mask`.

With those fixes, masks and atom-token indices match exactly. Valid-region
single/pair trunk NRMSE is `0.01207/0.01756`. Under the identical 200-step
sampler tape, coordinate-component RMSE is `0.68226 A`, conventional raw atom
RMSD is `1.18168 A`, and maximum component error is
`2.06412 A`, and correlation is `0.97445`. The shared-Torch-static control is
`6.02e-5 A` component RMSE. JAX warm sampling is `2.94x` faster (`2.506 s`
versus `7.371 s`). This is not a sub-angstrom pass under the conventional raw
atom RMSD definition and remains a harder, separately reported chemistry slice.

The active-versus-no-covalent branch is nontrivial and directionally preserved.
Pair-initial effect RMS is `0.004875` in Torch and `0.004868` in JAX with
cosine `0.99991`. Single/pair trunk effect RMS is `8.864/8.384` in Torch and
`8.779/8.319` in JAX, with cosine `0.97145/0.98420`. The compact evidence is in
`benchmarks/scientific_parity/results/2026-07-15-modified-glycan-covalent.json`.

## Restraint and recycle-2 gates (2026-07-15)

The contact-restraint and recycle-2 cases use the same five-residue protein,
released weights, one sample, seed 0, and identical 200-step sampler tape.
Prepared restraint features pass 32/32 (31 exact; inverse-distance maximum drift
`8.94e-8`). Raw contact/docking/pocket dictionaries are consumed during feature
generation and excluded from JAX device arrays. A fresh two-step restraint smoke
also wrote its CIF, score archive, PAE, PDE, pLDDT, and ranking outputs.

The restraint case has `0.04735 A` coordinate-component RMSE, `0.15595 A`
maximum component error,
correlation `0.9998545`, and a `3.85e-6 A` shared-static control. Its pair-initial
active-versus-unrestrained effect has Torch/JAX RMS `0.082831/0.082836` and
cosine `0.999988`; single/pair trunk effects have cosine `0.98080/0.98868`.

Recycle-2 has `0.02518 A` coordinate-component RMSE, `0.07555 A` maximum
component error,
correlation `0.9999591`, and a `1.06e-5 A` shared-static control. Recycle-2
versus recycle-1 changes the single/pair trunk with Torch/JAX RMS
`30.930/30.719` and `13.622/13.405`; effect cosine is `0.99564/0.98938`.
Warm sampler speedup is `2.94x` in both cases.

Recycle MSA subsampling has a separate exact ordering gate. Two consecutive
official Torch random draws are replayed into the Torch-free implementation;
both selected feature rows and zero-padded masks match exactly, selected rows
retain source order, and the two recycle orders differ. The JAX runtime key
sequence is deterministic for a fixed seed and distinct between recycles.
Public Torch and JAX RNG algorithms remain intentionally framework-native; the
exact test controls the random draws to verify sampling and ordering semantics.
Compact evidence is in
`benchmarks/scientific_parity/results/2026-07-15-restraint-recycle.json`.

## Strict multi-seed release matrix (pending)

The pre-registered release matrix contains protein, protein-ligand,
modified-residue/glycan/covalent, 8CYO, and 1AC5 fixtures at seeds 0, 1, and 2.
Every run uses one recycle, 200 diffusion steps, one sample, and a complete
matched NumPy random tape. Each of the 15 runs must independently satisfy its
fixture's raw atom RMSD, maximum atom displacement, correlation, shared-static,
initial/trunk NRMSE, and identity-mismatch thresholds; fixture means cannot
rescue a failed run. The thresholds are committed in
`benchmarks/scientific_parity/release_thresholds.json`.

This gate remains open until the complete provenance-pinned aggregate is
committed and green. It establishes numerical port parity under matched tapes,
not biological accuracy or equivalence between framework-native PRNG streams.
