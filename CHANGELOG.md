# Changelog

What changed between releases, and why. Entries name the behaviour a user can
observe; the reasoning lives beside the code.

FoldJAX follows [semantic versioning](https://semver.org). Model weights,
upstream defaults and scientific results are covered by the project contract in
`PROJECT.md` rather than by this file: a release never changes what a recorded
command predicts unless it says so here, in its own paragraph.

## Unreleased

### Added

- **Context parallelism (`--cp-devices P` / `cp_devices`).** Five backends --
  OpenDDE, Protenix, Boltz-2, OpenFold3, and ESMFold2 -- can shard their pair
  representations across `P` local JAX devices, the JAX form of OpenDDE
  upstream's Fold-CP `1 x P` mode. One process, no launcher: each consolidated
  graph carries `with_sharding_constraint` row shardings and the SPMD
  partitioner places the collectives; only triangle attention routes through
  `shard_map`, because its blocked row loop iterates the sharded axis
  (ESMFold2 has no triangle attention and needs no `shard_map` at all). Like
  upstream's mode it runs the XLA triangle kernels in CP mode -- the fused
  cueq/tokamax kernels are FFI calls the partitioner cannot split, and asking
  for one fails loudly; single-device runs keep their fused defaults
  unchanged. Verified numerically against the single-device programs on a
  forced four-device CPU mesh: module-level pair-stack parity for every
  covered backend at a token count indivisible by the mesh, plus real-weight
  end-to-end runs where local weights exist (OpenDDE 6.1e-4 Å, Protenix
  7.3e-4 Å max coordinate diff at fp32; Boltz-2 sits inside its own
  documented fp32 reduction-order floor, so its CP gate is the confidence
  summaries, which agree). Boltz-2 rejects steering and affinity jobs under
  CP for now. AlphaFold 3 is excluded: its vendored upstream implementation
  is authoritative and runs a Triton attention kernel the partitioner cannot
  split, and it is the least memory-bound of the six. Measured on real
  hardware (4x RTX A5000 24 GB, Slurm): a 1,003-residue OpenDDE job that
  OOMs one 24 GB card completes at CP=4 with balanced per-device peaks of
  11.7 GiB (bf16) / 18.0 GiB (fp32) -- 2.53x below the 45.5 GiB single-device
  fp32 peak, the gap to 4x being the triangle-mult all-gather that a future
  ring pass reclaims. Multi-GPU runs need `NCCL_P2P_DISABLE=1` on the tested
  nodes, and homogeneous cards (a mixed H100/RTX mesh is refused by XLA).
  **Multi-GPU runs can return silently non-finite output, and the cause is
  not yet established -- check every one.** Two things are known. A node
  that hands out four cards where only three initialise returns NaN from
  every run on it, so `jax.device_count()` must equal the requested shard
  count before a result is worth reading; that check is necessary and not
  sufficient. On one node where all four cards did initialise, CP=4 returned
  non-finite confidence and coordinates in 11 of 20 runs of the same input,
  at the same rate with and without XLA autotune flags -- which refutes the
  earlier entries in this file that blamed `--xla_gpu_autotune_level`, whose
  evidence was a single run either way against what is now known to be a
  coin-flip base rate. A second node of the same card model returned six
  finite runs from six. Whether the difference is that node's hardware or a
  timing-sensitive defect in the context-parallel path is under
  investigation; single-device runs are unaffected either way.

  What the search has settled so far: the corruption appears after the trunk.
  In a failing run every trunk array is finite and bit-identical to the ones
  the passing runs used -- so nothing the sharded pair code computes is wrong
  -- and only the diffusion coordinates are bad. It is also not always loud:
  Boltz-2 produced structures 34 Å from the single-device answer with every
  array in them finite, so **a finiteness check is the wrong gate** and the
  runs called clean by one should be re-scored by whether repeats of a seed
  agree with each other. Ten hypotheses have been measured and closed --
  autotune flags, a bad node, the step count, the diffusion-side sharding
  constraints, the sampler's `lax.scan`, NCCL's transport, the allocator,
  asynchronous collectives, and pinning the trunk/diffusion handover
  replicated.

  The sampler now runs inside a fully replicated `shard_map`, so the
  diffusion region provably contains no collective: a sharding constraint is
  a preference the partitioner may route around, which is why pinning those
  tensors replicated left the per-step collectives in the compiled program
  untouched, and `shard_map` is the guarantee instead. **The evidence for
  this is structural and single-device-equivalent, not yet a demonstrated
  fix**: the compiled HLO loses both sampler collectives, the CPU-mesh
  structure is unchanged to 0.0002 Å, and twelve GPU runs with and without it
  agree to 0.0002 Å -- but no session has yet been measured in which the
  unmodified arm fails and this one does not. Known v1 boundary: OpenDDE at ~3,000 residues still exceeds
  3x 96 GB because the fp32 diffusion side's atom-window gathers pull full
  pair tensors per device. Sharding those consumers was the planned next
  lever and is now on hold: the evidence above says that sharding *into*
  the diffusion is what the port cannot yet make reliable.
- **The square context-parallel grid (`--cp-layout 2d`).** Fold-CP's own
  layout, on OpenDDE, Protenix, Boltz-2 and OpenFold3: pair rows *and*
  columns split across a square device grid, with the triangle contraction
  run as Cannon's algorithm -- skew both operands once, then one local
  product per ring hop. Per-device pair cost falls from `O(N^2/P_rows)` with
  a full-width column axis to `O(N^2/P_total)`, and no all-gather appears in
  the contraction: the transient is two tiles. Triangle attention stays
  row-sharded under the grid, because its softmax spans whole columns.
  **`--cp-layout auto` deliberately stays on the 1-D layout**: every
  published number for context parallelism was measured there, and a default
  that silently changed the program would make those numbers describe a
  configuration nobody can reproduce, so the grid is opt-in until it has GPU
  evidence of its own. Verified on forced CPU meshes at 2x2 *and* 3x3 -- the
  3x3 case exists because modulo 2 a forward ring hop and a backward one are
  the same permutation, so a 2x2 grid cannot falsify a sign anywhere in the
  schedule. Every context-parallel parity gate is now mutation-tested and
  asserts from a trace-time witness that the sharded program really was
  traced.

- **Trunk representations (`representations=`, `stop_after="trunk"`).** Every
  model here builds a per-token *single* stream and a token-pair state before
  it predicts coordinates, and until now only two of them could hand those
  back, each under its own name and only through its native command line.
  `PredictionRequest(representations=("single", "pair"))` returns them from
  all five, and `stop_after="trunk"` stops once they exist rather than paying
  for a structure that will be discarded. The result carries a
  `Representations` handle: `result.representations["pair"]` reads one array
  from disk on access, `describe` gives its shape, dtype, axis names and --
  the part that keeps a cross-model comparison honest -- the *space* those
  axes count. Two models both call their pair state `pair`, but OpenDDE's
  `structural_pair` indexes a structural-token space its residue space does
  not cover, and nothing about the array says so. `foldjax capabilities
  --model M` lists what each model produces; ESMFold2 offers two names rather
  than three because it carries one single stream, not an input embedding and
  a trunk output.

  Underneath is a capture mechanism rather than a flag per tensor: a model
  annotates a value with `capture("pair", z)` and the traced program hands
  back the ones that were asked for. Off is free -- an un-requested tap leaves
  no trace in the jaxpr -- and the requested set is a static argument, so a
  different set is a different program rather than a stale cache hit. On costs
  liveness rather than bytes: a captured value becomes an entry output, and
  the temp arena is reused across a run where an entry output is not, which is
  the whole of the 15.2 GiB that returning Protenix's confidence logits used
  to cost at 3,012 tokens. **Nothing is returned by default**, because a pair
  representation is quadratic in token count -- half a gigabyte at a thousand
  tokens, nearly five at three thousand. A tap inside a scanned loop is
  refused unless the caller says what it is willing to spend: stacking
  OpenDDE's structural pair state over 48 refiner blocks is 63 GiB at 488
  residues, and learning that from an OOM twenty minutes in is much worse than
  learning it from the error.
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
