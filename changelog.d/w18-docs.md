## Unreleased

### Added

- **User documentation.** A documentation index (`docs/README.md`, linked from
  the README's new "Documentation" section); a quickstart from install to a
  first structure; six tutorials (`docs/tutorials/`: a heteromer with an MSA
  search and pairing, protein-ligand pocket and contact constraints with
  Boltz-2 affinity, glycans, templates, batches and Slurm arrays, resuming);
  `docs/outputs.md` (the output tree, `confidence.json`, `confidence_full.npz`,
  `predicted_aligned_error.json`, `foldjax_run.json` and `cost.breakdown`);
  `docs/configuration.md` (the store under `FOLDJAX_HOME`, every `FOLDJAX_*`
  variable with its default, compile-cache trust); `docs/faq.md`; and model
  pages for Boltz-2, Protenix and OpenDDE beside the existing three.
- `tests/test_docs_links.py`: every relative link in `docs/**/*.md` names a
  file that exists and, for a Markdown target, a heading that exists; the
  archive index lists every archived note.

### Changed

- The 49 dated engineering notes and one-off reports (`docs/*-2026-09-*.md`
  and `preprocessing-contract-audit.md`) moved to `docs/archive/`, indexed by
  `docs/archive/README.md`. Every reference to them -- docs, CHANGELOG, test
  and source comments, the parity manifests' notes and
  `bench/experiments/independent-input-entity-parity-2026-09-05.json` -- now
  names the new path. `docs/EXPERIMENTS.jsonl` stays where it was.
