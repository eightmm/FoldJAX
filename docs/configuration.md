# Configuration

FoldJAX has no configuration file. Where it keeps its files is decided by one
directory, the store, and a handful of environment variables change how
searches and caches behave. Everything else is a flag on the command
([cli.md](cli.md)) or a field of `PredictionRequest`
([python-api.md](python-api.md)).

## The store

```bash
foldjax home                 # every location, as JSON
foldjax home --path weights  # one location, for scripts
```

The store's root is the first of:

1. `$FOLDJAX_HOME`, when set;
2. a `.foldjax/` directory at the root of the source checkout FoldJAX was
   imported from, when that directory exists (`mkdir .foldjax` to opt in; an
   installed wheel never has one);
3. `$XDG_CACHE_HOME/foldjax`, else `~/.cache/foldjax`.

Nothing creates the root as a side effect of asking where it is.

```
$FOLDJAX_HOME/
├── downloads/<model>/       released files exactly as published, kept so a re-convert needs no network
├── weights/<root>/          prediction-ready checkpoints; one root per model, plus one per alternative
│                            profile (protenix-v2/, protenix-base-20250630/, opendde-abag/, ...)
├── assets/                  shared chemistry: CCD dictionaries, RDKit caches, template metadata
├── compile/                 the persistent XLA compilation cache, namespaced per model and program
├── msa/                     searched alignments, keyed by sequence and search provenance
├── templates/
│   ├── hits/                searched template hits, keyed like msa/
│   └── mmcif/<source>/      downloaded template structures, one directory per download source
└── runtime/
    ├── alphafold3/          AlphaFold 3's generated extension and CCD tables, keyed by source and ABI
    └── jobs/                job documents generated from --sequence, FASTA, structures and jobs files
```

`foldjax home --path` takes `home`, `downloads`, `weights`, `assets`,
`compile_cache`, `msa`, `templates` or `runtime`.

What each part costs and how to trim it:

| directory | grows with | reclaim with |
|---|---|---|
| `downloads/`, `weights/` | each model and profile you fetch (Boltz-2 about 6.2 GB, OpenDDE 3.3 GB, Protenix 2.1 GB, OpenFold3 2.3 GB, ESMFold2 about 26.8 GB) | delete a model's directory; `foldjax weights fetch` restores it |
| `compile/` | every distinct input shape, model, weight set and compile-relevant option | `foldjax cache gc` (reports; `--apply` deletes) |
| `msa/`, `templates/` | every distinct sequence searched | delete entries; the next `--msa auto` searches again |
| `runtime/alphafold3/` | each AlphaFold 3 source edit or Python move, about 1 GB each | `foldjax runtime gc --model alphafold3` (reports; `--apply` deletes) |

The MSA and template caches are deliberately not per model: running one
target through three models searches once. Entries are written with the
process umask, so a group can share `msa/`; the recorded hashes detect a
damaged entry, not a hostile writer, so a shared cache trusts everyone who can
write into it.

## Environment variables

Every `FOLDJAX_*` variable the package reads, with its default:

| variable | default | effect |
|---|---|---|
| `FOLDJAX_HOME` | see [the store](#the-store) | the store's root |
| `FOLDJAX_MSA_SERVER_URL` | `https://api.colabfold.com` | the MMseqs2 server `--msa auto`/`required` and `--templates auto` send sequences to. Part of the cache key, so two servers never share an alignment. An empty value is an error |
| `FOLDJAX_MSA_SERVER_VERSION` | `colabfold-mmseqs2` | a label for that server's databases, recorded in the cache key; change it when the server's databases change |
| `FOLDJAX_MSA_COMMAND` | unset | a local protein search instead of the server: called as `<command> --input query.fasta --output DIR`, it writes `DIR/non_pairing.a3m` and `DIR/pairing.a3m`. Nothing leaves the machine. `foldjax msa wrapper` prints a reference wrapper for ColabFold's local databases |
| `FOLDJAX_RNA_MSA_COMMAND` | unset | the same for RNA chains (writes `DIR/rna_msa.a3m`); no public server searches RNA |
| `FOLDJAX_MSA_LOCAL_VERSION` | `local` | a label for the local search's databases, part of the cache key: change it when the databases change, so old alignments are not reused |
| `FOLDJAX_MSA_MAX_WAIT_SECONDS` | `3600` | how long one remote search, with its retries, may take before it is abandoned |
| `FOLDJAX_ALLOW_INSECURE_HTTP` | unset | `1` allows plain `http://` for a server beyond this machine. Server URLs must otherwise be `https://` (loopback hosts excepted), because the sequence and any credential would cross the network in clear text |
| `FOLDJAX_TEMPLATE_COMMAND` | unset | a local template search for `--templates auto`: called as `<command> --input query.fasta --output DIR`, it writes `DIR/pdb70.m8` with hits named `<pdb id>_<author chain>` |
| `FOLDJAX_TEMPLATE_LOCAL_VERSION` | `local` | a label for that search's databases, part of the template cache key |
| `FOLDJAX_TEMPLATE_MMCIF_DIR` | unset | a local mmCIF mirror (flat, or wwPDB-divided `ab/1abc.cif.gz`) read before anything is downloaded. Its files are trusted as they are, so keep it writable only by you |
| `FOLDJAX_TEMPLATE_STRUCTURE_URL` | `https://files.rcsb.org/download` | where template structures are downloaded from; an empty value turns downloading off |
| `FOLDJAX_TRUST_SHARED_COMPILE_CACHE` | unset | `1`, `true`, `yes` or `on` uses a compile cache other accounts can write into ([below](#compile-cache-trust)) |
| `FOLDJAX_PROGRESS` | unset | `0`, `false`, `no` or `off` silences the stage lines on stderr, as `--quiet` does |
| `FOLDJAX_PROGRESS_MODE` | unset | `lines` prints download progress as tab-separated `[foldjax-progress]` lines when stderr is not a terminal (a batch log, say) |
| `FOLDJAX_DOCKQ` | `DockQ` on `PATH` | the DockQ executable `foldjax compare --metrics dockq` runs |
| `FOLDJAX_CP_RENDEZVOUS_TIMEOUT` | `600` | seconds a context-parallel run may wait at a collective before XLA ends the process, so one device's out-of-memory does not leave the others waiting forever. A negative value waits forever; `0` is refused. Only read before JAX initialises its backend |
| `FOLDJAX_SKIP_MESH_CHECK` | unset | `1` or `true` skips the check that a context-parallel mesh really moves data between its devices. The check exists because one node returned finite, wrong results from a broken interconnect; skip it only to diagnose |

`FOLDJAX_MSA_PAIRING` is not read by FoldJAX: under `--msa-pairing none` it
is set *for* a local `FOLDJAX_MSA_COMMAND` wrapper, to tell it no pairing
alignment is wanted. `FOLDJAX_REQUIRE_AF3_RUNTIME=1` is read only by the test
suite, where it turns "AlphaFold 3's runtime cannot be built here" from a skip
into a failure.

`foldjax doctor` prints which MSA and template search is configured, and every
`FOLDJAX_TEMPLATE_*` variable it sees.

### Variables FoldJAX sets or respects

| variable | what FoldJAX does with it |
|---|---|
| `XLA_CLIENT_MEM_FRACTION`, `XLA_PYTHON_CLIENT_MEM_FRACTION` | `foldjax predict` sets `XLA_CLIENT_MEM_FRACTION=0.9` when neither is set; an explicit `--mem-fraction` replaces whichever spelling is present. Setting both is an error inside the CUDA plugin, which then falls back to the CPU, so FoldJAX never adds its spelling next to yours |
| `XLA_FLAGS` | under context parallelism FoldJAX adds the collective timeout (`xla_gpu_nccl_termination_timeout_seconds`) unless it is already there |
| `JAX_PLATFORMS` | `cpu` keeps every model on the CPU (slow, but complete) |
| `SLURM_ARRAY_TASK_ID`, `SLURM_ARRAY_TASK_MIN`, `SLURM_ARRAY_TASK_COUNT` | read by `--shard auto` ([batches and Slurm](tutorials/batch-slurm.md)) |
| `UV_NO_DEFAULT_GROUPS` | not FoldJAX's, but `1` is how a CPU-only machine keeps `uv run` from re-installing the CUDA group ([install.md](install.md)) |

A few ports read their own variables to name a kernel explicitly
(`BOLTZ_JAX_TRIANGLE_MULTIPLICATION_BACKEND`,
`PROTENIX_TRIANGLE_MULTIPLICATION_BACKEND`, `OPENFOLD3_TRIANGLE_BACKEND`, ...);
those are described where the kernel is, in [cli.md](cli.md), and are not
needed for ordinary use.

## The compile cache

These models take minutes to compile and seconds to run once compiled, so the
persistent XLA cache is on by default under `compile/`. It is namespaced per
model, weight identity, runtime profile, and the options that change the
compiled program, so different models and weight sets never share a
namespace, and an option that only changes output formatting never splits
one. `--cache-dir DIR` moves it for one run, `--no-cache` turns it off for one
run (`use_compile_cache=False` in Python), and the two cannot be combined.

An entry is reused only for the same accelerator, JAX runtime, weights,
profile, static options **and input shapes**. A new shape compiles again, which
is why a screen over many inputs runs with `--padding` (shapes rounded to a
bucket; see [batches](tutorials/batch-slurm.md)) and why `foldjax cache warm`
exists:

```bash
foldjax cache warm --model boltz2 --input job.yaml --msa single
foldjax cache gc --older-than 30 --max-size 20G      # reports what it would remove
foldjax cache gc --older-than 30 --apply             # removes it
foldjax cache gc --verify --apply                    # removes entries that no longer decompress
```

`cache warm` runs the first seed of the request through the normal prediction
path, including GPU kernel autotuning, then discards the prediction unless
`--output-dir` is given. Warm the shapes and options you will run, on the
machine that will run them.

The cache setting is applied for one FoldJAX request and restored afterwards,
including on failure, even though JAX keeps it as process-wide configuration.
An application that embeds FoldJAX should therefore not compile unrelated JAX
programs in another thread at the same time.

### Compile-cache trust

JAX runs whatever executable it finds under a cache key. A directory another
account can write into is therefore a way for that account to run code as
you, so FoldJAX uses a cache directory only when the directory and every
directory above it

- belongs to you or to root, and
- is writable neither by the world nor by a group with another member (a
  root-owned sticky directory such as `/tmp` is fine as an ancestor).

New cache directories are created without the group-write bit whatever the
umask. When the check fails, the run warns once, names the directory and the
reason, and compiles without a persistent cache: it is slower, never wrong,
and an untrusted directory is never read.

A cache shared on purpose, such as a lab's group-writable setgid directory, is
opted in with

```bash
export FOLDJAX_TRUST_SHARED_COMPILE_CACHE=1
```

which trusts every account that can write into it. Use it only when that is
true of everyone in the group.
