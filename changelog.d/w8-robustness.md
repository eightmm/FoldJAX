### Fixed

- **`--keep-going` (`on_error="continue"`) now survives any error of one
  model/input pair.** It absorbed only a fixed list of exception types, so an
  unconverted checkpoint (`UnpicklingError`, a safetensors error), a kernel
  the device cannot run (`NotImplementedError`), a non-OOM XLA runtime error
  or a `KeyError` from a malformed input ended the whole batch. Every
  `Exception` is now recorded in `foldjax_failures.json` and the batch moves
  on; `KeyboardInterrupt` still stops it. A run directory that cannot be
  created, and an input that cannot be resolved or converted (an empty FASTA
  file in an input directory), are now that pair's recorded failure instead
  of a batch abort; without `--keep-going` both still refuse before anything
  runs.
- **A stale `foldjax_failures.json` is removed by a batch that has no
  failures,** so the file never describes an earlier invocation.
- **A run manifest the disk cannot take is a failure.** A full disk or a
  read-only directory turned the missing completion marker into a warning
  and exit status 0; it is now an `OSError` naming the manifest (exit 2, or a
  recorded failure and exit 3 under `--keep-going`), with the disk-full hint.
  A manifest whose provenance cannot be introspected still only warns.
- **Two runs into one output directory no longer share files.** Each
  model/input run holds an exclusive lock on `<run dir>/.foldjax.lock` for
  its whole duration; a second process is refused with the holder's process
  id and the lock path (a recorded failure under `--keep-going`). The lock is
  released by the kernel if its holder dies.
- **A structure whose header rewrite fails is left whole.** The canonical
  CIF's data-block title was rewritten in place, and any error was ignored,
  so a write cut short left a truncated structure that the manifest then
  digested as the result. It is now written to a sibling file and renamed
  over the original, and a failure is a `RuntimeWarning` naming the file.
- **Protenix runs on a CPU-only host.** The released denoiser attention is
  tokamax's fused kernel, which tokamax implements for GPUs only, so the
  default configuration failed off a GPU with `NotImplementedError` under a
  ~200-line traceback. An omitted `diffusion_attention_backend` now resolves
  to `xla_jit` there (as it already did under context parallelism); the
  resolved value is in the cache namespace and the native argv. A spelled
  `tokamax` attention option off a GPU is refused before featurization in one
  line that names the `--option` to use instead.
- **Failures name the file you wrote.** A FASTA or structure input is run
  through a generated job document under the store's `runtime/jobs/`; the
  failure line, `foldjax_failures.json` and the run manifest's
  `input.source` now name the original file (`source.kind` is `fasta` or
  `structure`; absent, as before, for a multi-job file).
- **OpenFold3 `attach_msas` without a `backend` raised `TypeError`.** The
  documented `attach_msas(spec, alignment_dir=...)` built its default
  `RemoteMMseqs2Client` with no endpoint. It now uses the same server and
  cache label as `--msa auto` (`FOLDJAX_MSA_SERVER_URL`,
  `FOLDJAX_MSA_SERVER_VERSION`, else the public ColabFold endpoint).
- **ESMFold2 `build_msa(msa_depth=...)` exceeded its cap and starved later
  chains.** The cap was checked against the shared row list inside the
  per-chain loop: the first chain filled it and every later chain still added
  one row past it. `msa_depth` is now one cap on the total row count (query
  included), shared one row at a time across the aligned chains; an uncapped
  or under-cap alignment is unchanged.
- **Managed-memory cleanup no longer swallows `KeyboardInterrupt` or
  `SystemExit`,** and a release that fails with an ordinary error is reported
  as a `RuntimeWarning` instead of passing silently (`models/_managed_memory`,
  the CCD session stack, ESMFold2's session).
- **An optional kernel that failed to import now says why when selected.**
  tokamax (Boltz-2/shared and Protenix wrappers, `cp_fused_attention`,
  `triangle_attention_ring_kernel='tokamax'`) and Boltz-2's Pallas triangle
  attention keep the import exception and chain it, with its type and message,
  into the error an explicit selection raises.
- **Input files with a UTF-8 byte-order mark are read.** A FASTA saved by a
  Windows editor failed with "FASTA must start with a '>' header line" and a
  JSON job with "Unexpected UTF-8 BOM"; the mark is now dropped. A file in
  another encoding (Latin-1, say) is refused naming the file and telling you to
  save it as UTF-8, instead of the bare codec message.
- **A trailing `*` stop codon in a FASTA protein is dropped.** UniProt and
  every translator write it; it used to be refused as an unsupported residue.
  An internal `*` is still refused.
- **`--name -dash` says how to pass it.** argparse reads the value as an
  option and only reported a missing argument; the error now adds
  `--name=-dash`.
- **Protenix's "cannot tell which model" error names the FoldJAX spelling.**
  It told `foldjax predict` users to pass `--model-name`, which that command
  does not have; it now names `--option model_name=NAME` (and `--model-name`
  for the native Protenix CLI).
- **A truncated converted checkpoint in the store is reported as truncated.**
  It was reported as "no converted weights" beside the file that was there.
  When the conversion record names a different size, the error now says
  `truncated: N bytes where their conversion wrote M` (or that an empty file
  is unreadable) and gives the `weights fetch` line that converts it again.

### Added

- **`foldjax cache gc --verify`** decompresses every JAX compile-cache entry
  with the codec this environment's JAX reads it with and selects the ones
  that do not decode, such as a write cut short by a full disk or a kill. JAX
  only warns about such an entry on each lookup and, because it never
  overwrites an existing entry, recompiles that program on every run until the
  file is removed. Like the other selectors it reports by default and deletes
  with `--apply`; it can be given alone or with `--older-than`/`--max-size`.
  JAX writes entries in place rather than through a temporary file, so entries
  modified in the last 10 minutes are left for a later pass. The JSON report
  gains a `verify` block (`checked_files`, `corrupt_files`, `corrupt_bytes`,
  `skipped_recent_files`, `unreadable_files`).

### Changed (resume staleness)

- **`--resume` no longer reuses a run whose featurizer or input writer has
  since changed.** Each port's recorded implementation files were a
  hand-picked list of model files, so an edit to Protenix's featurizer
  (`models/protenix/data/featurize_json.py`), its CCD tables
  (`ccd_nucleotides.npz` and the other two), or the common-job writer in
  `foldjax/input.py` (including its template map) changed predictions while an
  old run still resumed. Every port now binds its whole source tree, its
  packaged `.npz` tables, its adapter, the modules of another port its code
  imports (OpenDDE: all of Protenix; ESMFold2 and OpenFold3: Boltz-2's native
  norm), the shared `models/_*.py` helpers, `backends/base.py` and
  `input.py`; Boltz-2 and OpenFold3 also bind `template_search.py`, and
  OpenFold3 `_openfold3_compile.py`. Runs made before upgrading to this version
  are rerun rather than reused, once.
- **ESMFold2 binds the `ccd.pkl` beside its checkpoint.** An
  all-biomolecule job is featurized from it, and the run manifest recorded
  only the checkpoint and its config; it is now recorded (as absent when it
  is), so replacing or placing it reruns the job under `--resume`.
