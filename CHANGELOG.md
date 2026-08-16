# Changelog

What changed between releases, and why. Entries name the behaviour a user can
observe; the reasoning lives beside the code.

FoldJAX follows [semantic versioning](https://semver.org). Model weights,
upstream defaults and scientific results are covered by the project contract in
`PROJECT.md` rather than by this file: a release never changes what a recorded
command predicts unless it says so here, in its own paragraph.

## Unreleased

### Added

- **Alignment search.** `--msa auto` (`PredictionRequest(msa="auto")`) searches
  for a protein chain that arrived without one and caches the result under
  `$FOLDJAX_HOME/msa/`, keyed by sequence and search provenance rather than by
  model, so one target run through three backends searches once. `--msa
  required` refuses to fall back to single sequence. The default stays `none`,
  which is the previous behaviour — but it now warns once per run instead of
  silently folding a protein from its own sequence. **`auto` and `required`
  send the sequence to the configured server**; `FOLDJAX_MSA_COMMAND` and
  `FOLDJAX_RNA_MSA_COMMAND` point at a local workflow instead, and RNA is
  searchable only that way.
- **Input that is not a job file.** `--sequence`/`--dna`/`--rna`/`--ligand`/
  `--ligand-smiles`, FASTA files, `.pdb`/`.mmcif` depositions (re-folded from
  their chemistry, never their coordinates), `structure:PATH` for a `.cif`, and
  a directory as a batch. Each becomes an ordinary common-schema document under
  the runtime store, so `plan` names it and the manifest hashes it.
  `Job.from_sequences`, `Job.from_fasta` and `Job.from_structure` are the
  Python spellings.
- **Templates and binding affinity in the common schema.** A polymer takes
  `templates: [{mmcif, query_indices, template_indices}]` and a job takes
  `properties: [{affinity: {binder}}]`. The residue map is required by
  AlphaFold 3, Protenix and OpenDDE and refused by Boltz-2, which aligns the
  mmCIF itself; affinity reaches Boltz-2 alone. Whatever a backend cannot
  express is refused, never dropped.
- **`foldjax show`, `foldjax doctor`, `foldjax cache gc`, `foldjax models
  --for JOB`, `foldjax --version`.** `models --for` answers from the input
  translation table, so it needs no weights, GPU or network. `cache gc` reports
  by default and deletes only with `--apply`.
- **Progress on stderr and a summary table on stdout.** Stage lines during a
  run, and the model's own top-ranked structure at the end. stdout stays JSON
  whenever it is not a terminal, so pipes are unchanged; `--json` forces it and
  `--quiet` (or `FOLDJAX_PROGRESS=0`) silences the stage lines.
- **`--resume` and `--keep-going`**, and their `PredictionRequest(resume=...,
  on_error="continue")` spellings. Resume works at *seed* granularity: a
  five-seed job that died on the fourth repeats only what is missing. Failures
  are recorded in `foldjax_failures.json` beside the runs that succeeded, and
  `foldjax.predict_batch` returns results, skips and failures together.
- **`cost.phases` in the run manifest** — preparing input, predicting and
  writing, measured separately, because one number for the whole run cannot say
  which of them to act on.
- **`py.typed`.** The public surface is annotated; downstream type checkers
  now see it instead of `Any`.

### Changed

- Sequences are normalized where the document is validated: all whitespace
  removed (so a YAML block scalar works), upper-cased, and nucleic acids
  checked against IUPAC with the offending position named. A chain id may be
  omitted and is assigned in document order; an id that is present but blank is
  still an error.
- Unknown document fields name the field that was probably meant.
- `capabilities` gained `common_schema_features` and `native_only_features`.
  `supports_templates` describes the model; those two describe what a common
  job document can reach, which is not the same thing and used to be implied.
- `foldjax weights fetch` refuses a download the disk cannot hold and warns
  when conversion will be tight.
- The MSA client identifies itself to the search server as `foldjax/<version>`
  rather than `protenix-jax/0.1.0`.

### Fixed

- Ctrl-C ends a run with one line and exit code 130 instead of a traceback
  through FoldJAX's internals.
- A read-only output directory, a full disk and other `OSError`s are reported
  as the machine facts they are rather than as defects. A full disk names the
  store and the command that reclaims space; a missing optional package names
  the extra that provides it.
- The AlphaFold 3 runtime error names `foldjax runtime prepare --model
  alphafold3` instead of an internal function.
- `foldjax.progress` resolves its stream at write time. Holding the stream
  meant writing to whatever stderr *was*, which could fail a prediction that
  was otherwise fine.

## 0.3.0

FoldJAX became one standalone package: AlphaFold 3, Boltz-2, ESMFold2, OpenDDE,
OpenFold3 and Protenix inference with no sibling repository, plus opt-in
mask-aware padding for reusable JAX executables. See the git history and
`docs/ports/` for how each port got there.
