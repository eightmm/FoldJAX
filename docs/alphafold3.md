# Running AlphaFold 3 through FoldJAX

FoldJAX carries AlphaFold 3's Apache-2.0 source and runner, but never its model
parameters. The parameters are released directly by Google under separate
terms that forbid redistribution. Everything else uses the same interface as
the other models — one job file, one model name.

```bash
foldjax predict --model alphafold3 --input job.yaml
```

## Installing the AlphaFold 3 runtime

Install FoldJAX's AlphaFold 3 extra for its non-JAX runtime dependencies. Keep
the CUDA extra that matches the machine, then inspect or
prepare the generated runtime explicitly:

```bash
uv sync --extra alphafold3
foldjax runtime status --model alphafold3
foldjax runtime prepare --model alphafold3
```

On the first AlphaFold 3 prediction, FoldJAX builds upstream's small
ABI-specific extension and the generated CCD lookup tables. These files go to
`$FOLDJAX_HOME/runtime/alphafold3`, never into the installed wheel or
site-packages, so a read-only container installation works. `foldjax home`
prints the exact runtime location. `runtime prepare` makes this potentially
long step explicit; prediction still prepares it automatically on first use if
you skip that command.

The first build needs a C++ compiler, Git, network access, and either `uv` or
Python's `pip`. The `alphafold3` extra supplies the Python *runtime*
dependencies. CMake and Ninja are isolated build requirements declared by the
vendored AlphaFold 3 project, so `uv`/`pip` obtains them for the temporary build
environment; the extra does not install them directly. CMake also fetches
pinned C++ sources from their upstream Git repositories. Later predictions
reuse the source-, Python-ABI-, platform-, and NumPy-version-keyed runtime.

FoldJAX deliberately ignores unmarked `cpp*.so` files and generated CCD
pickles left in a source checkout. They do not prove which source or NumPy ABI
created them, so reusing them after an update could fail much later inside the
model. Generated artifacts are published only through the keyed runtime store.

FoldJAX does not automatically adopt an independently installed
`alphafold3` distribution. The default is always the vendored,
source-and-ABI-keyed managed runtime, even when another distribution appears
earlier on `sys.path`. This makes the code provenance and JAX-only dependency
boundary reproducible.

If an external `alphafold3` package was already imported in the current Python
process, FoldJAX refuses to replace its native extension in place. Restart the
process before using the managed runtime.

For development against an upstream checkout, opt in explicitly with the
backend's `source` option. Place that checkout's `src` first on `PYTHONPATH` so
the selected runner and package have the same provenance; FoldJAX rejects a
mismatched pair:

```bash
PYTHONPATH=/path/to/alphafold3/src:$PYTHONPATH \
  foldjax predict --model alphafold3 --input job.yaml \
  --option source=/path/to/alphafold3
```

Without that explicit option FoldJAX uses the runner matching its vendored
source. The former `ALPHAFOLD3_SOURCE` ambient environment switch is not
honored. Do not install upstream's JAX dependency over FoldJAX's CUDA-specific
JAX environment.

## Weights

Request the parameters from DeepMind
([terms](https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md)),
then drop the file into the FoldJAX weight store:

```bash
weights_store=$(foldjax home --path weights)
mkdir -p "$weights_store/alphafold3"
cp af3.bin "$weights_store/alphafold3/"
```

`foldjax weights list` will then show `alphafold3` as ready. FoldJAX never
downloads, converts, or redistributes these parameters — it only knows where to
look. `--weights` still works if you keep them somewhere else.

## MSAs

AlphaFold 3 refuses to featurize a protein or RNA chain whose MSA is *absent*:
it assumes its own genetic-search pipeline will produce one, which needs
hundreds of gigabytes of databases. A protein chain with no alignment is
therefore refused by default. With `--msa single`, FoldJAX writes an **empty**
MSA, which AlphaFold 3 accepts as single-sequence mode. Predictions will be
much worse than with a real MSA; supply one for anything you care about:

```yaml
entities:
  - type: protein
    id: A
    sequence: LSDEDFKAVFGMTRSAFANLPLWKQQNLKKEKGLF
    unpaired_msa: msas/A.a3m
```

To use AlphaFold 3's own search pipeline instead, run its `run_alphafold.py`
directly with `--db_dir` — that path is outside FoldJAX.

## Cache warming and serving padding

AlphaFold 3's automatic `--padding` and synthetic `cache warm --token-buckets`
use the carried runner's grid: 256, 512, 768, 1024, 1280, 1536, 2048, 2560,
3072, 3584, 4096, 4608, 5120. This replaces the generic grid previously used
by neutral AF3 padding, whose first bucket was 128. An explicit
`--pad-tokens 128` still selects 128, and explicit targets above 5120 are also
supported. Inputs above the automatic grid require an explicit target or the
existing exact-overflow policy. Padding remains opt-in.

Single-sequence warming retains the native 16,384-row featurization pool;
real alignments use that same padded axis. The model retains its native
per-recycle selection, normally 1024 rows. No alignment is replaced with a
synthetic alignment for prediction. Use matching weights, device/runtime,
token target, MSA selection cap, and model options when warming and predicting:

```bash
foldjax cache warm --model alphafold3 --token-buckets 256 --msa single
foldjax predict --model alphafold3 --input job.yaml --padding --msa auto
```

Tokamax keys the actual numeric model input structure, shape, and dtype,
alongside source, device/runtime, model configuration, and padding mode. The
host's list of candidate buckets is excluded: a `(256,)` warm and an automatic
prediction that selects 256 run the same padded model. Older store identities
included that entire list, separating otherwise identical calls and causing
fresh tuning. This correction changes their keys, so existing installations
need one new warm; previous cache files are left in place.

For managed GPU runs with a persistent Tokamax store, `foldjax_run.json` and
warm JSON reports include bounded `kernel_tuning.calls` diagnostics. Each
unique call records a repeat count and kernel sources, up to eight calls;
`calls_truncated` counts additional unrecorded calls. `input_abi` lists at most
128 array leaves with feature-tree paths, shapes, dtypes, and weak-type flags,
including RNG metadata. Array contents, sequences, RNG values, and filesystem
paths are excluded; nonstandard tree metadata is redacted. Without an installed
persistent store, these call diagnostics are absent.

| Field | What a difference identifies |
| --- | --- |
| `model_input_abi_sha256`, `input_abi` | The numeric batch structure, shape, or dtype differs. |
| `call_sha256` | The batch ABI or RNG shape/dtype differs; RNG values do not affect it. |
| `compile_config`, `padding` | A whitelisted model option or padding mode differs. |
| `source_sha256`, `runtime_sha256`, `device_sha256` | Source assets, software versions, or device properties differ. |
| `store_identity_sha256`, `signature_sha256` | The full store identity or identity-plus-call differs. |
| `namespace_sha256` | A different cache-directory spelling was selected; its path is not printed. |
| `sources` | `store` reused saved tuning, `measured` tuned in this prediction, `partial` retained successful operations while missing operations fell back, or `tokamax` used Tokamax's own resolution. |

CPU regression tests run native featurization and the actual runner's numeric
input conversion for different MSA depths, lengths in the first bucket,
sequence compositions, a heterodimer, and a populated template. They verify
identical call descriptors and unchanged real MSA values/masks, and lower the
native MSA shuffle/selection with one shared abstract signature. These checks
do not establish GPU latency or guarantee hits for every small JIT; a saved
Tokamax result and a persistent XLA executable are separate caches.

## GPU kernels on an unfamiliar device

AlphaFold 3's transition blocks call `tokamax.gated_linear_unit`, which prefers
a Triton kernel and, on a cache miss, sizes it from heuristics. Those heuristics
are not right for every device: on an RTX PRO 6000 Blackwell they ask for
108 KiB of shared memory against the 99 KiB the device has, and the launch fails
with `RESOURCE_EXHAUSTED` from inside the diffusion transformer — after the
kernel has compiled, so Tokamax's own per-implementation fallback never sees it.

FoldJAX therefore runs AlphaFold 3 with Tokamax's autotuner enabled, which
benchmarks the candidate configurations on whatever device is in front of it and
keeps one that fits. For the managed runtime on a GPU, FoldJAX writes a complete,
hardware- and source-qualified Tokamax result beside the persistent JAX cache.
It includes Tokamax's pre-vmap lookups as well as the operations retained in
StableHLO, and is published only after the accumulated overlay can lower the
exact call under the strict `error` policy. A fresh process can therefore select
the same kernels before tracing. JAX executable-cache reuse is a separate,
best-effort layer; the persisted Tokamax result does not claim that every XLA
executable entry is readable.

A trusted shared namespace is read first, followed by the request's private
compilation-cache namespace. A shared hit stays in place. A complete new result
is published under a per-call lock in the writable shared tier, or in the
private tier when shared creation, locking, or publication fails. The private
namespace uses owner-only directories and files, including when shared-cache
trust is enabled. Missing shared namespaces can therefore use the personal
tier without modifying the shared installation.

**What the autotuning costs, measured.** With the compile cache warm on both
sides and only the persisted Tokamax store deleted between them, at seed 101 on
the released schedule:

| tokens | model | autotuning | XLA compile |
|---:|---:|---:|---:|
| 132 | 34.4 s | 187.9 s | 135.0 s |
| 1,003 | 87.3 s | 288.8 s | 137.3 s |
| 2,096 | 242.6 s | 325.9 s | 136.8 s |

Two things follow. Compilation is flat -- 137 s within two seconds across a
sixteen-fold range of sizes -- while **autotuning grows with the shape**,
because it benchmarks candidate configurations at the real dimensions. Treating
it as a constant, which is the obvious way to correct a wall time that carries
it, understates it at the top and overstates it at the bottom.

And it dominates. At 1,003 tokens the model is 87 s and the tuning 289 s, so a
process that tunes spends three quarters of its life not predicting. That is
what the persisted store removes, and why AlphaFold 3 rows measured before
2026-08-31 are several times their current cost.

These options control what happens when a persisted result is absent:

| option | effect |
|---|---|
| `--option kernel_autotuning=heuristics` | use a persisted result first; otherwise use upstream's faster guess, which may not fit your device |
| `--option kernel_autotuning=error` | use a persisted result first; fail loudly if any kernel remains uncovered |
| `--option attention_kernel=xla` | skip the Triton attention kernel entirely (the native spelling `attention_backend=xla` also works) |

All three values are cache-miss policies and share one cache namespace and one
store, so what a run executes depends on what that store already holds as well
as on the option. Measured on an RTX PRO 6000 Blackwell: from empty caches,
`autotune` and `heuristics` compile two different programs; `heuristics` on the
cache an `autotune` run filled compiles nothing new and runs the autotuned
program. Neither reuses the other's executable -- the compilation cache keys
each by its HLO -- but the option alone does not say which one ran. The run
manifest's `kernel_tuning` does: per model call, whether the configurations
came from the store (`store`), were measured by this run (`measured`), or were
partly retained after an incomplete measurement (`partial`), or left to Tokamax
(`tokamax`: its bundled data, else the cache-miss policy). It
is `null` without a store (CPU runs, `--no-cache`, an external source), where
every call is Tokamax's own. For upstream's heuristic programs whatever the
store holds, run `heuristics` with its own `--cache-dir`.

When two tiers are active, each recorded call also reports
`cache_read_tiers` for disk reads and `cache_write_tiers` for successful
publications, counted by `shared` or `private`. `cache_reuse_tiers` counts
in-memory reuse with a known storage origin; it does not count another disk
read. These describe storage separately from the existing
`store`/`measured`/`partial` kernel-configuration sources.

An omitted attention (or `attention_kernel=auto`) is upstream's `triton` only
when the selected device is a GPU; on a CPU or TPU it is `xla`, because
tokamax's Triton flash attention raises `NotImplementedError` there.
`attention_backend=triton` still names it explicitly.

Managed persistent tuning can measure a selected GPU synchronously, including
when JAX's process-default backend is CPU. During prediction, Tokamax's implicit
hardware queries use the selected device; explicit backend queries retain their
usual meaning. Worker-based tuning remains limited to the process-default GPU.
Explicit external sources, unverifiable assets, CPU runs, and `--no-cache` keep
Tokamax's process-local policy; an autotune miss outside the default GPU still
fails when the managed synchronous path is unavailable.

Timeouts and compilation errors exclude individual candidates. Successful
candidates still compete for the fastest configuration, and uncovered operations
get one heuristic retry. A complete result is persisted only after strict
coverage validation. If coverage remains incomplete on the default GPU,
successful configurations are retained for this process's inline fallback and
reported as `partial`; no incomplete manifest is published. Missing kernel
identities and failure reasons appear in the warning. On a selected GPU that
cannot safely use inline tuning, incomplete coverage raises that diagnostic.

A fresh-process `t0128` gate persisted 22 unique configurations, then completed
under `kernel_autotuning=error` with zero Tokamax misses: the writer used
1,500.2 MiB / 338.13 s and the strict reader used 1,323.4 MiB / 23.67 s. Their
timestamp-normalized CIF and confidence artifacts were byte-identical. A second
isolated-cache pair reproduced zero-miss strict reuse and exact paired output.
Both readers did expose one upstream JAX 0.11.1 cache issue: an auxiliary
`jit_apply_fn` executable contained an invalid XLA metadata-payload reference,
so JAX warned, treated only that entry as a cache miss, and recompiled it safely.
The existing key is not self-healed, and `JAX_RAISE_PERSISTENT_CACHE_ERRORS=1`
can turn that upstream cache defect into a hard failure. It does not weaken the
Tokamax coverage result. The complete manifest hashes, timings, parity digests,
and failed schema-v1 attempt are recorded in
[`bench/experiments/tokamax-persistence-gate-2026-08-31.json`](../bench/experiments/tokamax-persistence-gate-2026-08-31.json).

The released five-sample denoiser graph remains unchanged. When
`num_samples > 5`, FoldJAX maps the independent sample axis in groups of the
released `eval_batch_dim_shard_size=5`, preserving the explicit per-sample RNG
tape while bounding simultaneous denoiser activations. On the 132-token n=10
A/B, the final exact-source warm measurements reduced the XLA live-byte peak
from 1,531.6 to 1,351.0 MiB while wall time changed from 39.15 to 43.64
seconds. Matched structures had minimum TM-score 0.98740 and median CA RMSD
0.186 Å, against a 12.2 Å median spread among the model's own samples;
ranking-score maximum absolute drift was 0.01554.

In the earlier provenance-limited sweep, the absolute saving grew with sample
count and sequence length. At 490 tokens,
sharded runs used `2940.6 MiB / 316.22 s` for 10 samples and
`3058.2 MiB / 325.84 s` for 20, versus historical batched peaks of 3558.8 and
4484.7 MiB. At 970 tokens, a same-worktree 20-sample A/B measured
`9515.8 MiB / 422.25 s` batched and `6748.9 MiB / 401.48 s` sharded, a 29.1%
peak reduction. These are cold-Tokamax fresh-process times; use them to read
memory admission, not as warm-throughput claims. The 970-token, 20-sample A/B's
ranking-score maximum absolute drift was 0.000284.

## Options

`--num-samples` and `--num-recycles` set AlphaFold 3's diffusion sample count
and recycle count (upstream `run_alphafold.py`'s `--num_diffusion_samples` and
`--num_recycles`). Omitted, FoldJAX runs what `run_alphafold.py` runs: 5
samples and 10 recycles (eleven trunk passes). Before 2026-10-09 the omitted
count was 3, the SI Algorithm 1 `N_cycle=4`; pass `--num-recycles 3` for that
schedule (see [recycling defaults](recycling-defaults.md)). AlphaFold 3's own option names
(`diffusion_samples`, `recycles`) are not accepted through `--option`; the
native options it does take include `buckets`, `kernel_autotuning`,
`return_embeddings` and `return_distogram`:

```bash
foldjax predict --model alphafold3 --input job.yaml \
  --num-samples 5 --num-recycles 10 \
  --option buckets='[256,512,1024,2048,5120]'
```

AlphaFold 3 has no pocket or restraint input, so a common job's pocket
constraint is refused. `--option pocket_sampling=select` is the one way it
counts: FoldJAX's own route, which scores every finished sample against the
pocket with Boltz-2's rule and ranks a satisfying sample first; the model
itself still sees no pocket, and `max_distance` is required
([cli.md](cli.md#--option-pocket_samplingselect)).
