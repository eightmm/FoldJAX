# FAQ

## Which model should I use?

There is no single best model, and FoldJAX does not rank them: their
confidence scores are calibrated differently and are never comparable across
models ([outputs.md](outputs.md#confidencejson)). Running several on the same
input and comparing them is what FoldJAX makes cheap (`--model boltz2 protenix
openfold3`, then `foldjax compare`). What does differ is what each one can
read and under which terms its weights come:

| model | weights | reads that the others do not | does not read |
|---|---|---|---|
| [Boltz-2](boltz2.md) | MIT, fetched by `foldjax setup` | binding affinity (the only affinity head), pocket and contact constraints, template files it aligns itself | template residue maps, paired MSAs |
| [Protenix](protenix.md) | Apache-2.0 (v1.x), fetched; v2 is proprietary and user-supplied | pocket and contact constraints with the `base-constraint-v0.5.0` profile; small fast profiles (`mini-*`, `tiny-*`) | templates and RNA MSAs unless switched on (upstream's released default is off) |
| [OpenFold3](openfold3.md) | Apache-2.0, fetched | pocket constraints (as pocket-guided sampling), RNA alignments, templates as files or residue maps | contact constraints, ligands of several CCD components (glycans) |
| [OpenDDE](opendde.md) | Apache-2.0, fetched; antibody-antigen `abag` profile | the same input dialect as Protenix | any constraint: dropped, as upstream drops them, with a warning |
| [AlphaFold 3](alphafold3.md) | Google DeepMind's terms; you request them yourself | user-defined CCD entries (native input only) | pocket and contact constraints |
| [ESMFold2](esmfold2.md) | MIT plus Biohub's acceptable-use policy; 26.8 GB, opt-in | folds from its language model; an alignment is optional, not required | templates, paired MSAs, constraints |

`foldjax models --for job.yaml` answers for one job: which models can run it,
and the reason for each that cannot. It needs no weights, no GPU and no
network. `foldjax capabilities --model M` lists everything one model accepts.

## The run was refused because a chain has no alignment

Every model but ESMFold2 is meant to run with an MSA, so a protein chain
without one is refused unless you say what you want:

- `--msa auto` searches the ColabFold MMseqs2 server and caches the result.
  **This sends the sequence off the machine.** Point `FOLDJAX_MSA_SERVER_URL`
  at your own server, or `FOLDJAX_MSA_COMMAND` at a local search, for
  sequences that must not leave ([configuration.md](configuration.md#environment-variables)).
- `unpaired_msa: path/to/chain.a3m` on the entity uses your own alignment.
- `--msa single` folds from the sequence alone, on purpose. Expect a much
  worse structure for most proteins.

## Out of memory

Five models (Boltz-2, Protenix, OpenFold3, OpenDDE, ESMFold2) estimate their
peak before the first compile and refuse a run that will not fit, naming what
they estimated and the levers. AlphaFold 3 has no such estimate. In order of
what to try:

1. **Read the refusal.** It names the levers that apply to that model and
   size. `--memory-check warn` runs anyway and lets the allocator answer;
   `--memory-budget-gib N` asks whether a job would fit a smaller card.
2. **Give the run the card.** `--mem-fraction` (default 0.9) is how much of
   the device JAX preallocates; another process on the same GPU, or a run
   started right after a large one, leaves less than the estimate assumed.
   Run one prediction per GPU at a time.
3. **Change how it runs, not what it predicts.** On OpenDDE a float32 trunk
   costs about twice the bfloat16 default's peak. Context parallelism
   (`--option cp_devices=4` on Boltz-2, Protenix, OpenDDE, OpenFold3 and
   ESMFold2) splits one prediction over several GPUs
   ([context_parallel.md](context_parallel.md)).
4. **Change the input, knowingly.** `--max-msa-depth` reads fewer alignment
   rows: less memory, but a different prediction. Fewer `--num-samples` can
   lower a peak too, but not ESMFold2's, whose peak has no sample term.

`--padding` raises the peak: shapes are rounded up to a bucket. Nothing is
narrowed automatically to make a job fit. The measurements behind each law
and lever are in [cli.md](cli.md#memory) and
[engineering-notes.md](engineering-notes.md#memory).

## Can I run on a CPU only?

Yes, every model runs on a CPU, slowly: the first compile and the trunk are
much longer than on a GPU, so keep jobs small. Install without the CUDA group
and keep it out for the whole shell:

```bash
export UV_NO_DEFAULT_GROUPS=1
uv sync
uv run foldjax doctor          # the jax line says cpu
```

`JAX_PLATFORMS=cpu` keeps a GPU machine on the CPU. AlphaFold 3's default
attention kernel needs a GPU; off one it runs XLA's attention instead.
Memory estimates are for GPU allocators and do not apply on a CPU.

## FoldJAX runs on the CPU although the machine has a GPU

`foldjax doctor` prints the device JAX found. When it says `cpu` on a GPU
machine:

- The CUDA wheels are missing: a plain `uv sync` installs them, unless
  `UV_NO_DEFAULT_GROUPS=1` or `UV_NO_GROUP=gpu` is set in the shell. A CUDA 12
  driver needs the `cuda12` extra instead ([install.md](install.md)).
- Both `XLA_CLIENT_MEM_FRACTION` and `XLA_PYTHON_CLIENT_MEM_FRACTION` are set.
  The CUDA plugin fails to initialise when both spellings are present, and JAX
  then reports that no CUDA-enabled jaxlib is installed. Unset one.
- `JAX_PLATFORMS=cpu` is set.

## Does it run on macOS?

On the CPU. FoldJAX has no Apple GPU path: the GPU stack is CUDA (JAX's CUDA
plugin, cuEquivariance, Triton), so a Mac installs with the default `gpu`
group off (`export UV_NO_DEFAULT_GROUPS=1`) and runs every model on the CPU.
Checkpoint loading uses only interfaces macOS's Python provides. The
`templates` extra's Kalign has macOS wheels; AlphaFold 3's first-use runtime
build needs a C++ toolchain ([alphafold3.md](alphafold3.md#installing-the-alphafold-3-runtime)).
The CPU suite is tested on Linux; macOS is not part of CI.

## How do I get the AlphaFold 3 weights?

FoldJAX carries AlphaFold 3's Apache-2.0 source but never its parameters.
Google DeepMind releases them only to applicants who accept the
[model parameters terms](https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md),
which forbid redistribution. Request them from DeepMind, then put the file in
the store:

```bash
mkdir -p "$(foldjax home --path weights)/alphafold3"
cp af3.bin "$(foldjax home --path weights)/alphafold3/"   # or af3.bin.zst
foldjax doctor                                            # alphafold3 now ready
```

Nothing is converted or uploaded, and no FoldJAX command fetches them. The
terms bind what you may do with the parameters and with predictions made from
them; read them, not this page. Protenix v2's weights are the other set you
supply yourself, for a different reason: upstream declares them proprietary
([protenix.md](protenix.md#weights-and-profiles)). Per-model parameter terms:
[licences.md](licences.md).

## Why do two runs of the same job give different structures?

Usually because they were not the same run:

- **A different seed.** Boltz-2, ESMFold2, and AlphaFold 3 or OpenDDE on a
  job without `modelSeeds` seed nothing by default upstream, so FoldJAX draws
  a seed per run, prints it and records it in `foldjax_run.json`
  (`seed_source: random`). Pass `--seed N` (or `--seeds ...`) to repeat one.
  Protenix (101) and OpenFold3 (42) have fixed upstream defaults.
- **ESMFold2 is random by design**: random initial pair state, language-model
  dropout per loop and sampler noise, so two seeds give genuinely different
  structures ([esmfold2.md](esmfold2.md)).
- **A different alignment.** `--msa auto` reuses its cache, but a different
  server, database label or pairing mode is a different cache entry, and the
  server's own databases change over time. The manifest records what each
  chain read (`msa_search`, `msa_stats`).
- **A fresh compile.** At the same seed, a run that reuses its compiled
  program repeats bit for bit. Two fresh compilations can choose different
  GPU kernels and round differently, and a sensitive target amplifies that;
  keep one compile cache, or use `--option deterministic=on` when two
  independent compilations must agree (it costs wall time;
  [cli.md](cli.md#--option-deterministicon)).

Different models given the same integer seed never produce the same
coordinates: each has its own random-number scheme.

## Why is the first run so slow?

It compiles. These programs take minutes to compile and seconds to run once
compiled, and the compiled program is cached per model, weights, options and
**input shape**. `cost.breakdown` in `foldjax_run.json` shows how much went to
`compile` ([outputs.md](outputs.md#cost)). For many inputs of different
lengths, `--padding` rounds shapes to buckets so they share programs, and
`foldjax cache warm` compiles a representative shape ahead of time
([batches](tutorials/batch-slurm.md)).

## The compile cache warns that it is untrusted

FoldJAX uses a compile cache only when no other account can write into it,
because a cache entry is an executable JAX runs as found. A group-writable
store is refused with a warning, and the run compiles without a persistent
cache. If the sharing is deliberate, set `FOLDJAX_TRUST_SHARED_COMPILE_CACHE=1`
([configuration.md](configuration.md#compile-cache-trust)). `foldjax doctor`
says whether the cache is trusted.

## Can a lab share one store?

Yes: one group-owned, setgid `FOLDJAX_HOME`, a `0002` umask and
`FOLDJAX_TRUST_SHARED_COMPILE_CACHE=1` for every member from their first run.
[Sharing a store with a group](configuration.md#sharing-a-store-with-a-group)
has the setup, what each part of the store needs, and the `chmod` repair for
a store some member used before.

## Where did my files go?

`foldjax home` prints every location FoldJAX uses. Outputs go to
`foldjax-outputs/<input stem>` unless `--output-dir` says otherwise; the full
tree is in [outputs.md](outputs.md).
