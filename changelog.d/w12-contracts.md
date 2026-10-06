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
- **OpenFold3 `triangle_kernel=auto` is the omitted default.** It pinned
  `cueq` (attention only), which no omitted run selects; it now resolves as
  omitting the knob does -- `cueq-pallas` on a GPU, `cueq-full` elsewhere,
  `xla` under context parallelism -- and shares that cache namespace.
- **`triangle_kernel=cueq-pallas` is requestable** through the neutral knob on
  OpenFold3, the kernel its omitted GPU run selects; Boltz-2 and Protenix
  refuse it by name.
- **Width values have one vocabulary.** `bf16`/`bfloat16` and
  `fp32`/`float32`/`f32` are accepted in `dtype` and every `*_dtype` option on
  every port and rewritten to the port's own spelling, so `trunk_dtype=bf16`
  on OpenFold3 and `confidence_dtype=float32` on OpenDDE are no longer
  refused. Manifests still record options as typed, so `--resume` treats two
  spellings of one width as different requests.
- **AlphaFold 3 runs off a GPU with the default attention.** An omitted
  attention (or `attention_kernel=auto`) was upstream's `triton`, which
  tokamax refuses with `NotImplementedError` on a CPU (and TPU). It is now
  `triton` only on a GPU device and `xla` elsewhere, decided from the selected
  device at run time and from the `platform` option or JAX's default backend
  in the cache namespace, so an omitted CPU run shares the explicit `xla`
  namespace.
- **Checkpoint loading no longer needs `os.sched_getaffinity`**, which macOS
  lacks: the torch-archive prefetch and the ESMC cast-on-load size their
  thread pools with `os.process_cpu_count()` (the CPUs this process may use,
  Python 3.13+).
