## Unreleased

### Fixed

- **`plan --json` and `plan --shard` write nothing into the store.** `--json`
  took a second code path that stored a `--sequence`, FASTA or structure
  job and the split jobs of a multi-job file under `runtime/jobs`, and
  `--shard` wrote its shard file to `runtime/jobs/shards`. Both now go to a
  scratch directory, as plain `plan` already did, and the paths shown are the
  ones `predict` would write. `--json` now also runs the same `preflight`
  checks as plain `plan`, so it refuses what `predict` refuses.
- **`run.schema.json` names every field the manifest writes.** `msa_search`,
  `weights.kind`, `weights.stat_signature` and AlphaFold 3's per-sample
  `metadata.native_sample` were written but undeclared. `schema_version` is
  now `1.1` for both `foldjax_run.json` and `confidence.json` (one shared
  constant); `1.0` files still validate and still resume, since resume gates on
  the integer `schema`. A contract test now fails on any written key the
  schemas do not name.
