## Unreleased

### Added

- **`foldjax.api.preflight(request)`**: everything `predict` refuses before
  weights load, for one resolved request -- the backend's request checks and,
  for a FoldJAX job, the common-schema translation check, the alignment
  policy, the files the job names, CCD codes, SMILES and bond atoms -- with
  nothing searched or written. `plan` and `predict_batch` both call it.
- **Common-layer chemistry checks.** A ligand or modification CCD code absent
  from the shared `components.cif`, and a bond atom its residue's component
  does not have (named by the job's 1-based residue index), are refused before
  any model loads (`foldjax.ccd`). Skipped where no `components.cif` is
  installed. A SMILES string RDKit cannot parse or sanitize is refused the
  same way when RDKit is installed.
- **`foldjax models --for JOB --msa POLICY`** applies the alignment policy, so
  a protein chain with no alignment is "no" under the default `none` except on
  ESMFold2, and a missing alignment file is named. A `{jobs: [...]}` file is
  answered one job at a time instead of "unsupported top-level fields: 'jobs'".
- **`plan` shows the effective sampling**: `sampling` holds the value each
  neutral knob runs at and `sampling_source` says whether it came from the
  request, a native option or profile, the adapter's released default, or the
  checkpoint configuration (shown as null). `plan` also prints a generated job
  as `generated_input`, and with padding a `not_checked` note: the MSA rows a
  model stores are known only after featurization, so a `--pad-msa` below them
  is refused by `predict`, not by `plan`.
- **`foldjax capabilities`** reports `sampling_defaults`, and `--model` on
  `capabilities`, `runtime` and `weights path` lists every model and alias.
  `boltz-2` is an alias of `boltz2`; `home --path` accepts `msa` and
  `templates`.
- **`show`** names each run's input (a multi-job file's job by name) and ends
  with a footer listing recorded failures; `show` and `compare` warn when they
  find no structure.

### Changed

- **`plan` refuses what `predict` refuses**, with the same message: all 18
  malformed jobs of the audit now fail `plan`. It writes no generated job into
  the store; `predict` checks every run of a batch up front, before any runs
  (`--keep-going` records a refused run with `seed: null` and runs the rest).
- **Input auto-detection**: a JSON/YAML mapping without a native signature key
  (`sequences`, `modelSeeds`, `dialect`, `version`, `queries`) is validated as
  a FoldJAX job, so `entitys:` gets "did you mean 'entities'?" instead of a
  backend `KeyError`; an empty `{}` is therefore refused rather than passed to
  a backend. A `.json`/`.yaml` that does not parse is a one-line error
  (`... is not readable as YAML: <problem> (line L, column C)`) instead of a
  traceback; ESMFold2 no longer JSON-parses a `.yaml` job.
- **Generated jobs are content-keyed**: `Job.store()` writes
  `runtime/jobs/<digest>/<stem>.json` (it takes an optional `root` and
  `stem`), so the stem -- and the default output directory -- never depends on
  what was stored before. An unnamed `--sequence` job now runs into
  `foldjax-outputs/job-<8 hex of its SHA-256>`; `--name NAME` into
  `foldjax-outputs/NAME` (docs/input.md). A `--sequence` run finished before
  this change is recorded under its old input path, so `--resume` reruns it
  once.
- **A seed is drawn only after the run is validated**, so a refused job no
  longer first announces a seed it never used.
- **`predict` stdout is the result only**: everything printed while a request
  resolves and runs -- Python `print` and file descriptor 1 alike, such as
  Boltz-2's "Found explicit empty MSA" and the Protenix/OpenDDE runners'
  `compile cache:`/`job:`/`wrote:` lines -- goes to stderr, so
  `predict ... > out.json` is valid JSON (`cache warm` likewise).
- **CLI warnings** print once per command as `foldjax: warning: <message>`,
  without the source file and line.
- **`--seeds` help** describes the layout written: native files and a manifest
  per seed in `seed_<n>`, structures in `seed-<n>_sample-<NN>`, all listed in
  the top-level `foldjax_run.json`.

### Fixed

- **`--resume` with `--msa single`** reuses a finished run: it searches
  nothing, as `none` does, and was treated as unverifiable. With `--msa auto`
  or `required`, the MSA-cache entry each searched chain reads (alignments
  and the `provenance.json` holding their SHA-256) is recorded in
  `input_dependencies`, so a finished run is reused while those files are
  unchanged (RNA searches and OpenFold3's complex pairing stay unverifiable).
  A directory whose finished run is not reused now says why
  (`[foldjax] not resumable: ...`).
- **Multi-seed sample numbers**: the merged result records each sample's
  per-seed number (`metadata.sample`), so `show`, the CSV/JSON rows, `compare`
  and `best` no longer count samples across seeds, and `best_within_model`
  survives a best sample in a later seed. Merged manifests written before are
  read with per-seed numbering, `best` matched by structure path.
- **No traceback on bad user input**: an unknown CCD code, an invalid SMILES,
  a bond to an atom its residue lacks, a missing alignment file, and a YAML
  syntax error are one `foldjax:` line each.
