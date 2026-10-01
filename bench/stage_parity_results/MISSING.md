# What the stage table still does not measure

Every port now has all six stages measured. The six ESMFold2 and OpenDDE
stage cells (S3-S5) that the 2026-09-09 master captures could not support are
filled from a native *stage re-capture*, described first so its use can be
checked. What remains open is listed after it: two featurizer draws the
port cannot take, one inferred input document, the CPU-vs-GPU confound shared
by every S3-S6 cell, and the AlphaFold 3 row's lack of intermediate stages.

## The stage re-captures (ESMFold2, OpenDDE)

`foldjax-bench/stage-captures-20261001/<model>/protein_1ubq/native-A`, code
snapshot `code-24bbb78` beside them (commit `24bbb78`, the whole tree), job
scripts in `jobs/`, logs in `logs/`. Each is the stored capture's own
command, input, seed (101) and environment variables, plus one flag:

| model | Slurm job | wall | harness flag | hooks (run time only; upstream checkouts unchanged) |
| --- | --- | --- | --- | --- |
| ESMFold2 (Biohub `transformers` fork `ef32577f`, torch 2.13.0+cu130) | 4243 | 71 s | `bench/esmfold2_tape.py capture --capture-stages` | instance wrapper on `structure_head.sample` (its `z_trunk`, `s_inputs`, `relative_position_encoding` and returned `sample_atom_coords`); forward hook with kwargs on `confidence_head` (every tensor input and output); the returned `distogram_logits` |
| OpenDDE (`ddfa1df8`, torch 2.7.1+cu128) | 4244 | 30 s | `bench/opendde_closure_capture.py native --capture-stages` | class-level `functools.wraps` wrappers on `get_pairformer_output`, `expand_to_structural_tokens`, `run_sample_diffusion_stage`, `run_confidence_head_stage` |

Both wrote `stages.npz`, with each tensor's dtype at the boundary recorded in
`metadata.json:stages_schema` / `provenance.json:stages`. Upstream's
ESMFold2 `z_trunk`, `relative_position_encoding` and `token_bonds_encoding`
hold bf16-representable values (autocast), so the port receives them at its
own bf16 width losslessly; `s_inputs` is not bf16-exact and is injected as
float32, which is also the port's width there. In OpenDDE every boundary is
float32, and the same tensor seen at two boundaries is one value (the
structural trunk output is what the sampler received, the residue trunk is
what the confidence head received, the sampler's coordinates are the head's
`x_pred_coords`).

**Re-capture vs stored capture.** The draws are the same draws; what differs
is GPU rerun noise, of the size the same campaign's native-A vs native-B
rerun already shows:

| | ESMFold2 re-capture vs stored | ESMFold2 stored A vs B | OpenDDE re-capture vs stored | OpenDDE stored A vs B |
| --- | --- | --- | --- | --- |
| inputs, tape (MSA draws, noise, dropout masks) | bitwise equal | | bitwise equal (`native-input/derived/identity.npz`, `torch/tape.npz`, `torch/msa.npz`) | |
| ESM-C LM hidden states | bitwise equal | bitwise equal | n/a | n/a |
| all-atom RMSD per sample (A) | 0.0045, 0.0076, 0.0060, 0.0036, 0.0042 | 0.0059, 0.0052, 0.0044, 0.0031, 0.0039 | 0.0021, 0.0059, 0.0020, 0.0021, 0.0028 | 0.0021, 0.0063, 0.0021, 0.0020, 0.0019 |
| max unaligned displacement (A) | 0.083 | 0.066 | 0.105 | 0.127 |
| max \|d\| pLDDT / PAE (A) / pTM | 0.0016 / 0.54 / 4.0e-4 | 0.0042 / 0.30 / 2.1e-4 | atom pLDDT 2.6e-4 / token PAE 0.018 / 5.2e-6 | 1.5e-4 / 0.015 / 3.1e-6 |

So the re-capture is used as the native side of S3-S5 for both models: S4 and
S5 compare against the re-capture's *own* sampler and head outputs (which came
from the very tensors injected), and S1, S2 and S6 stay on the stored capture.
Every S3-S5 record carries this comparison as
`native_rerun_vs_stored_capture`. One provenance difference: the OpenDDE
checkout's `git status` hash is now that of a clean tree
(`e3b0c442...`, was `5a0e4e0e...` at the stored capture); the commit and the
tracked-diff hash (empty) are unchanged.

Port seams the injections use (all counted; a seam that never fires fails the
stage): ESMFold2 `models/model.py` module globals `inputs_embedding`,
`relative_position_encoding`, `_token_bonds_encoding`, `folding_trunk` (only
its `parcae_coda` call is replaced) and `diffusion.sample`, each stage traced
afresh with `jax.clear_caches()` on both sides; OpenDDE `models/model.py`
`pairformer_output_from_s_inputs`, `structural_token_expand` (the expanded
`s_inputs`; the expander still builds the structural pair features),
`structural_refiner_stack` and `sample_diffusion`.

## Still open

* **Boltz-2 S1 -- reference-conformer augmentation not replayed.** The
  featurizer rotates and translates each reference conformer at random.
  Upstream's draws are on disk (`native-A/preprocessing-tape.npz`, recorded by
  patching `boltz.data.feature.featurizerv2.center_random_augmentation` in
  `bench/boltz_closure_capture.py`), but the port featurizer has no injection
  point for them (`src/foldjax/models/boltz2/data/_external.py:center_random_augmentation`,
  called from `data/feature/featurizerv2.py:1561`). `ref_pos` differs by up to
  13 A raw and by 1.4e-6 A after one Kabsch fit per `ref_space_uid`, so the
  difference is that rigid draw and nothing else; every categorical feature
  is exact. Closing it is a port change (accept the tape), not a capture.
* **OpenFold3 S1 -- reference conformers not reproduced.** OpenFold3 builds
  each reference conformer with RDKit ETKDGv3 embedding
  (`openfold3/core/data/primitives/structure/conformer.py:_compute_conformer`,
  vendored at `src/foldjax/models/openfold3/_upstream/...`). The capture's
  tape holds only the MSA draws and the sampler draws, not these
  coordinates, and the port's seed-101 featurization does not reproduce them:
  all 118 conformers differ in internal geometry (intra-conformer distances
  up to 3.9 A apart; seeds 42 and 0 behave the same), so `ref_pos` stays
  3.9 A off even after a per-conformer fit. Every categorical feature is
  exact and the other float features agree to 1.2e-7. A re-capture should
  save the per-residue conformer coordinates returned by
  `_compute_conformer` / `multistrategy_compute_conformer` (they are what
  upstream writes into `ref_pos`), and the port featurizer needs a seam to
  take them. S3-S6 are unaffected: they run on the native `input.npz`.
* **ESMFold2 S1 -- input document inferred.** Both the stored capture and the
  re-capture ran the shared feature archive
  (`entity-parity-20260905/inputs/protein_1ubq/esmfold2/upstream-biohub-full.npz`;
  `metadata.json: core_only_shared_features = true`), which records only its
  own sha256, not the document or MSA it was built from. S1 re-featurizes the
  JCTC-matrix 1UBQ job with the bench's 1UBQ alignment (sha256 `4bf7b489...`,
  the alignment OpenDDE's capture recorded for the same sequence); the result
  agrees with the archive to 1.2e-7 on every shared array, which is what
  supports the inference. Closing it needs a capture that starts from the
  document (the backend's featurizer) rather than from the archive.
* **Every S3-S6 port stage runs CPU XLA against a GPU capture.** Triangle
  kernels are XLA where the native run used cuEquivariance, and bf16 is CPU
  bf16. The residuals are therefore a CPU-vs-GPU number on top of
  port-vs-native (see `docs/parity-cpu.md`); a GPU run of this script would
  separate the two.
* **AlphaFold 3 -- no intermediate stages.** The `alphafold3` row compares two
  complete CPU runs (DeepMind's `run_alphafold.py` v3.0.4 and FoldJAX's
  vendored AF3) on 8REH; S3/S4 are `not_captured`. Every compared output is
  bitwise identical (the featurised batch, all 405 parameters, every
  confidence leaf, the raw padded model outputs including the distogram
  contact probabilities, and both samples' coordinates, by both the harness and
  the CLI route), so there is no residual to localize. The check is one
  target, CPU, 2 samples and 1 recycle with FoldJAX given DeepMind's bucket
  list; it is not a GPU or default-bucket statement.
