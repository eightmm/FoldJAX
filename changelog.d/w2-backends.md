### Added

- **Boltz-2 affinity reaches the run's files.** The affinity head's results
  were computed and then dropped by `foldjax predict`. A run with an affinity
  binder now writes upstream's `affinity_<record>.json` beside the native
  structures (`affinity_pred_value`, `affinity_probability_binary`, and the
  two ensemble members `affinity_pred_value1`/`2` and
  `affinity_probability_binary1`/`2`, spelled as upstream spells them), and
  the same fields go into the `scores` of the one sample whose coordinates the
  affinity was computed on -- not every sample's, because affinity is scored
  on a single structure. `scores` already admits any numeric key, so
  `confidence.json` stays schema 1.x with no version change.

### Changed

- **OpenFold3's confidence Pairformer runs float32 by default, as upstream's
  does.** Upstream pins that stack to float32 in every regime
  (`heads/head_modules.py:106`, `heads/prediction_heads.py:224`); this port
  let it follow `dtype`, so the shipped bfloat16 trunk ran it bfloat16.
  `confidence_dtype` now defaults to `float32` whatever `dtype` is, and
  `--option confidence_dtype=bfloat16` restores the old head. The head reads
  predicted coordinates and emits scores, so structures are unchanged, but
  every confidence score -- and therefore the ranking and `best` sample -- of
  a default OpenFold3 run can differ, and the run gets a new cache namespace.

- **ESMFold2 takes its reference conformers from one source, whether or not
  an MSA is attached.** A bare single protein chain was featurized by the
  transformers fork's protein-only builder and its built-in conformer table,
  while the same chain with an alignment (or any multi-chain or non-protein
  job) went through Biohub's all-atom pipeline and the CCD's ideal
  conformers, so attaching an MSA also moved `ref_pos`. Every job now uses the
  all-atom pipeline (Biohub's released end-to-end path). A bare single chain
  therefore gets different `ref_pos` and the all-atom `entity_id` numbering
  (from 0 rather than 1), and can predict a different structure; it now needs
  the `ccd.pkl` staged beside the weights, as every other job already did. It
  is still folded by one `predict` call, as before.

- **`--msa auto` pairs a Boltz-2 heteromer, as Boltz-2's own server search
  does.** The search gave every chain an unpaired alignment only. Boltz-2
  submits its protein entities together as one `pairgreedy-env` job when
  there are two or more and writes each a CSV whose paired rows share a key
  (`main.py` `compute_msa`); a searched heteromer now gets the same: one
  complex pairing job, then per entity a `<out>/inputs/msa/entity_NNNN.csv`
  built exactly as upstream builds it (paired rows first, capped at 8,192,
  all-gap rows dropped; then the unpaired rows to 16,384 in all). Entities
  with one sequence are one query, as for OpenFold3 (upstream submits each
  entity, duplicates included); a homomer or monomer stays unpaired; a chain
  that arrived with its own alignment keeps the complex unpaired, with a
  warning. A common job still cannot name an `.csv` alignment. Searched
  heteromers can predict different structures.

- **Protenix featurizes each seed with that seed.** Native-JSON featurization
  -- the reference-conformer augmentation and the seeded chemistry draws --
  was keyed to the job's `modelSeeds[0]` (or 101) and computed once, so every
  seed of a run, and a `--seed` run, folded the first seed's input. Upstream
  reseeds before its dataloader builds each seed's input
  (`runner/inference.py:565-566`), and OpenDDE here already featurized per
  seed. A common-schema job is written with `modelSeeds` set to the run's
  seed, so it is unchanged; for native Protenix JSON, every seed but the first
  of a multi-seed run, and every request whose seed is not the job's first
  `modelSeeds` entry, now gets a different input and can predict a different
  structure. The ESM/ISM embedding depends on the
  sequence alone and is still computed once per job; a multi-seed native run
  holds one prepared feature set per seed.

- **ESMFold2 keys entities on sequence alone, as Biohub does.** Two polymer
  chains with one sequence and different modifications were two entities;
  Biohub's `_get_sequence_key` makes them one, so `entity_id` and `sym_id`
  change for such a job, and the prediction can.

### Fixed

- **Boltz-2 runs off a GPU without naming a GLU.** The released
  `glu_backend` is the fused Tokamax kernel, which refuses to run off a GPU,
  so every CPU Boltz-2 run failed unless it said `--option glu_backend=xla`
  -- while `doctor` reported that every model runs on CPU. An omitted
  `glu_backend` now resolves to `xla` off a GPU (and still to `pallas` on a
  serial GPU process); the cache namespace records the realised value. An
  explicit `tokamax` is still refused off a GPU. The native
  `foldjax.models.boltz2.api.predict` keeps its released default.

- **Boltz-2's affinity input is upstream's first-ranked sample.** The affinity
  stage re-featurized the sample with the highest ipTM; upstream scores the
  sample it ranks first by `confidence_score` (`data/write/writer.py:73-79,
  178`), which is also the sample `best` names.

- **A Boltz-2 `--max-msa-depth` above 8,192 reaches the features.**
  Preprocessing read at most 8,192 rows of each alignment file (upstream's
  released `--max_msa_seqs`) whatever depth was asked, so a deeper
  `max_msa_depth` changed nothing. The parse cap now follows `max_msa_depth`
  when that is larger; at or below 8,192 nothing changes.

- **OpenFold3 zeroes unused reference atoms before centring, as upstream.** An
  atom with no conformer position (NaN) and `annot_used_atom_mask` false
  poisoned the masked centre of its whole molecule, and an all-unused
  molecule kept its NaNs; either way validation then refused the job.
  Upstream (`featurization/conformer.py:141-156`) zeroes those rows first and
  refuses only a used atom without a position, which this port now does too.

- **Protenix refuses a constraint its weights cannot read while planning.** A
  common pocket constraint or a native `constraint.pocket`/`contact` on a
  checkpoint without a constraint embedder -- every released one but
  `protenix_base_constraint_v0.5.0` -- failed only at the embedder, after
  featurization and the weight load; `foldjax plan` and `predict` now refuse
  it up front, naming the channel. The embedder's own refusal now names every
  unweighted channel in a fixed order instead of one arbitrary one.

- **`representations.npz` and cached template mmCIFs are written with the
  umask mode,** not `tempfile`'s owner-only `0600`, as `confidence_full.npz`
  already was.

### Documented

- **OpenDDE's native writer has no `full_data_sample_*.json`.** Upstream
  writes it by default; its pair arrays are program outputs here only under
  `--option include_raw=true`. See `docs/ports/opendde/README.md`.
