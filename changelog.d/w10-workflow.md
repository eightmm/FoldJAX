## Unreleased

### Added

- **`--msa-pairing {model,greedy,complete,none}`** (`PredictionRequest.msa_pairing`,
  `ModelConfig.msa_pairing`), with `--msa auto`/`required`. `model` (default)
  is each model's existing choice, and keeps every existing MSA-cache entry.
  `greedy`/`complete` pair the complex in one ColabFold search
  (`pairgreedy-env`/`paircomplete-env`) for OpenFold3, and for Boltz-2 as
  upstream Boltz's keyed CSV (`boltz/main.py` `compute_msa`); AlphaFold 3,
  Protenix and OpenDDE, which re-pair by UniProt species that ColabFold
  pairing headers do not carry, refuse them. `none` delivers no paired
  alignment and skips the per-chain pairing ticket (a local wrapper sees
  `FOLDJAX_MSA_PAIRING=none`). The pairing mode is in the MSA cache key, in
  `foldjax_run.json` (`msa_pairing`), in `plan`, and in the resume identity.
- **Alignment depth and Neff per chain**: `foldjax_run.json` `msa_stats`
  (rows, and effective sequences at 80% identity with its definition), an
  `alignment` line in `foldjax show`, and an `msa_stats` column in
  `show --format csv|json` / `foldjax.results_table`.
- **`foldjax show --rank-by KEY`** (`plddt`, `ptm`, `iptm`, `ranking`, or a
  numeric column such as `score.<name>`; `KEY:asc` for smallest first): samples
  ordered within each model and configuration only, with
  `rank_within_model`, for table, CSV and JSON (`foldjax.results.rank_rows`).
- **`foldjax msa prefetch INPUTS [--model M ...] [--msa-pairing P]`**: search
  and cache every chain's alignment without loading weights or writing
  outputs; exits 3 when a chain's search failed. **`foldjax msa wrapper`**
  prints the path of `foldjax/search/colabfold_local.py`, a reference
  `FOLDJAX_MSA_COMMAND` running ColabFold's MMseqs2 search against local
  databases (optional `--gpu`) under ColabFold's own interpreter; no new
  FoldJAX dependency.
- **`--templates DIR`** (`PredictionRequest.template_dir`): a private folder of
  mmCIF files as the template source, searched on this machine with
  `mmseqs easy-search` or, without it, Kalign, then selected and delivered per
  model as `--templates auto` is; refused before anything runs when the needed
  aligner is missing. No release-date cutoff unless `--template-max-date`.
  The manifest records the folder's content digest (`template_dir`), and a
  changed folder is not resumed.
- **`--preset fast`** (`PredictionRequest.preset`): the publisher's reduced
  steps and recycles for the checkpoint being run, recorded as `preset` in the
  manifest. Published only for Protenix's Mini and Tiny profiles (5 steps,
  4 recycles, Protenix `docs/supported_models.md`); refused elsewhere with the
  reason.
- **ModelCIF confidence records in every written mmCIF**: `_ma_qa_metric`
  pLDDT `global` and `local` (`_ma_qa_metric_global`/`_local`) and `_software`
  rows for FoldJAX and the upstream model with versions; categories a writer
  already wrote (AlphaFold 3's, Boltz-2's local pLDDT) are kept and only the
  missing ones added.
