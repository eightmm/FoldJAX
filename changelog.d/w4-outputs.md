### Added

- **OpenFold3 writes upstream's full confidence outputs by default:** expected
  PAE and PDE in angstroms per sample (`pae`, `pde` in each
  `confidence_full.npz`, as upstream's `write_full_confidence_scores=True`
  writes them), the contact-weighted `gpde` per sample in `confidence.json`
  scores, and per-chain pTM (`chain_ptm`) and upstream's ligand-aware bespoke
  chain-pair ipTM (`chain_pair_iptm_bespoke`) in the archive and in
  `<job>_confidences.json`. The expectation is taken inside the per-sample
  confidence map, so the 64-bin logits still never leave it; the PDE head now
  runs for every sample. `InferenceConfig.return_expected_errors` (and
  `released_config(return_expected_errors=False)`) turns the arrays off. A new
  config field changes every OpenFold3 compiled program, so the first run after
  upgrading recompiles. Values match upstream v0.5.0's own CPU results to 2e-5
  (`tests/models/openfold3/fixtures/full_confidence_upstream.npz`).

- **OpenFold3's `sample_ranking_score` for protein inputs.** The disorder term
  of upstream's `0.8*ipTM + 0.2*pTM + 0.5*disorder - 100*has_clash` is now
  computed on the host as upstream's RASA does (biotite SASA with ProtOr radii,
  Sander maximum areas, 25-residue reflect-padded smoothing, threshold 0.581),
  so protein runs report `disorder` and the exact `sample_ranking_score`, rank
  their samples and get a `best`. Without biotite (the `openfold3-preprocess`
  extra) or without decodable atom identities the run keeps reporting
  `sample_ranking_score_no_disorder` and says why in `disorder_unavailable`.

- **ESMFold2 writes `pae`, `pde` and `chain_pair_iptm` by default.**
  `ModelSettings.return_expected_errors` (default on) keeps the expected
  PAE/PDE matrices without the 64-bin logits `return_confidence_logits` still
  withholds; derived cost `2*S*L^2*4` bytes, 2.2 GiB at 3,012 tokens and the
  checkpoint's 32 samples against a fitted 67.7 GiB peak (3.2%, not yet
  measured on a GPU). The head's `pair_chains_iptm` is no longer projected out
  of the managed program.

- **Boltz-2 writes `chain_ptm` and `chain_pair_iptm`** (upstream's
  `chains_ptm`/`pair_chains_iptm`, `writer.py`) into `confidence_full.npz`, with
  `chain_id` naming the axis. The program computes them densely over a
  power-of-two chain bucket (32 by default) instead of one static dictionary per
  chain-label set, so inputs with up to 32 chains share one executable; `raw`
  carries the present chains' `pair_chains_iptm` and its diagonal `chains_ptm`.
  The affinity program is unchanged.

- **Viewer-ready exports for every model.** Each canonical sample directory
  now holds an AlphaFold-DB-schema `predicted_aligned_error.json`
  (`predicted_aligned_error`, `max_predicted_aligned_error` = 31.75 for all six
  models) whenever the model returned PAE, and every mmCIF is guaranteed to
  carry the model's pLDDT (0-100) in `B_iso_or_equiv`: the column is checked
  against `confidence_full.npz` and filled only where a writer left something
  else (all six native writers already write it). The outcome is recorded per
  sample under `metadata.confidence_arrays.exports`. At 3,012 tokens the JSON is
  about 50 MB per sample.
