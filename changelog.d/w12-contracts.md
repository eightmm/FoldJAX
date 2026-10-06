## Unreleased

### Fixed

- **`plan --json` and `plan --shard` write nothing into the store.** `--json`
  took a second code path that stored a `--sequence`, FASTA or structure
  job and the split jobs of a multi-job file under `runtime/jobs`, and
  `--shard` wrote its shard file to `runtime/jobs/shards`. Both now go to a
  scratch directory, as plain `plan` already did, and the paths shown are the
  ones `predict` would write. `--json` now also runs the same `preflight`
  checks as plain `plan`, so it refuses what `predict` refuses.
