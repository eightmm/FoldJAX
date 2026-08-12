<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/banner-dark.png">
  <source media="(prefers-color-scheme: light)" srcset="docs/banner-light.png">
  <img alt="FoldJAX — biomolecular structure prediction, compiled" src="docs/banner-dark.png">
</picture>

# FoldJAX

Biomolecular structure prediction in JAX. Five models behind one interface —
four carried complete inside the package:

| model | vendored at | licence | upstream |
|---|---|---|---|
| `boltz2` | `foldjax.models.boltz2` | MIT | [jwohlwend/boltz](https://github.com/jwohlwend/boltz) |
| `opendde` | `foldjax.models.opendde` | Apache-2.0 | [aurekaresearch/OpenDDE](https://huggingface.co/aurekaresearch/OpenDDE) |
| `openfold3` | `foldjax.models.openfold3` | Apache-2.0 | [aqlaboratory/openfold-3](https://github.com/aqlaboratory/openfold-3) |
| `protenix` | `foldjax.models.protenix` | Apache-2.0 | [bytedance/Protenix](https://github.com/bytedance/Protenix) |

— plus `alphafold3`, which FoldJAX drives rather than carries, because its
parameters may not be redistributed. It takes the same job file as the rest
once installed; see [docs/alphafold3.md](docs/alphafold3.md).

OpenFold3 is carried twice over: the JAX port is a reimplementation, and
upstream's own *data pipeline* is vendored beside it, so featurization is exact
rather than approximately right and still needs nothing outside this repository.
Building features needs `--extra openfold3-preprocess`, because that pipeline is
upstream's torch code; predicting needs neither it nor torch. See
[docs/openfold3.md](docs/openfold3.md).

Give it one job file and a model name. Weights, the input dialect, the output
directory, and the XLA compile cache are all resolved for you.

```bash
uv sync --extra cuda13 --extra torch-bridge --group dev
uv run foldjax weights fetch --model boltz2      # download, verify, convert once
uv run foldjax predict --model boltz2 --input job.yaml
```

`torch-bridge` is only for that one conversion step: the upstream checkpoints
are torch files. **Prediction never imports torch for any model**, so once the
weights exist, `--extra cuda13` is the whole runtime.

```json
{
  "model": "boltz2",
  "output_dir": "foldjax-outputs/job",
  "samples": [
    {
      "seed": 0,
      "structure_path": "foldjax-outputs/job/boltz2_input.cif",
      "scores": {"mean_plddt": 0.9589, "iptm": 0.0}
    }
  ]
}
```

Every port keeps its own featurizer, sampler, checkpoint format, defaults, and
output writer, so results match running that project directly. FoldJAX supplies
the common request, the translation, and the plumbing.

## Input

One file, JSON or YAML, in the common schema:

```yaml
name: example
entities:
  - type: protein
    id: [A]
    sequence: ACDEFG
    unpaired_msa: msa.a3m
    modifications:
      - {ccd: SEP, position: 3}
  - {type: ligand, id: [L], ccd: ATP}
bonds:
  - [[A, 3, OG], [L, 1, PA]]
```

`--input-format` defaults to `auto` and decides on content, not extension: only
the common schema is a mapping with `entities`, and every native dialect is
passed through untouched. So a Boltz YAML or a Protenix job list works as-is.

Modifications and covalent bonds have one model-neutral spelling here and are
translated into each backend's own names — Protenix and OpenDDE want
`CCD_`-prefixed types and entity-indexed `covalent_bonds`, Boltz wants
`{ccd, position}` and a `bond` constraint, AlphaFold 3 wants bare `ptmType`.

**A field a backend cannot express is rejected, not dropped.** Boltz has no place
for a separate `paired_msa`, because it derives pairing from one per-chain a3m.
Silently discarding an alignment would change the science without changing the
exit code.

`unpaired_msa` reaches every model. That matters more than it sounds: a job that
pins an alignment must pin it everywhere, or any comparison across models becomes
a comparison of MSAs.

The same document can be built in Python instead of written by hand:

```python
from foldjax import Job, Ligand, Modification, Protein

job = Job(
    "example",
    [
        Protein("A", "ACDEFG", unpaired_msa="msa.a3m",
                modifications=[Modification("SEP", 3)]),
        Ligand("L", ccd="ATP"),
    ],
)
request = PredictionRequest(model="protenix", input=job.write("job.json"))
```

`Job` is a constructor, not a second schema: `to_document()` returns exactly the
mapping above, and every rule about what a document may say stays where the
model is known, so the class cannot drift from what the CLI accepts. `Job.read`
takes either file format back the other way.

## Choosing what to run

```bash
uv run foldjax predict \
  --model protenix \
  --input job.yaml \
  --num-samples 5 \
  --num-steps 200 \
  --num-recycles 10 \
  --seed 42
```

`--num-samples`, `--num-steps`, and `--num-recycles` are model-neutral. Each
backend calls them something different, and FoldJAX translates:

| neutral | alphafold3 | boltz2 | openfold3 | opendde / protenix |
|---|---|---|---|---|
| `--num-samples` | `heads.diffusion.eval.num_samples` | `diffusion_samples` | `num_samples` | `n_sample` |
| `--num-steps` | `heads.diffusion.eval.steps` | `steps` | `no_rollout_steps` | `n_step` |
| `--num-recycles` | `num_recycles` | `recycling` | `num_cycles` | `n_cycle` |
| `--max-msa-depth` | `evoformer.num_msa` | `max_msa_depth` | *feature cut* | `max_msa_rows` |

All four reach all five models. AlphaFold 3's step count and MSA depth are
not arguments of `make_model_config`, but the config it returns is an ordinary
mutable object and upstream already sets the sample count on it by assignment;
FoldJAX sets these the same way. That matters beyond convenience — a model that
cannot be held to the same schedule as the others cannot be compared with them.

Setting both a neutral knob and its native name is an error rather than a silent
preference. OpenFold3 has no MSA-depth argument at all, so there the knob is
applied to the features: the first *n* alignment rows are kept, in order, and
the per-token `profile` and `deletion_mean` are left as computed over the whole
alignment — which is what the other backends' native caps also do.
`--option KEY=VALUE` passes anything native straight through.

Seeds work the same way. `--seed` runs one, `--seeds 0 1 2` runs those three,
and `--num-seeds 3` is the same as the latter — counting up from `--seed`, so
`--seed 7 --num-seeds 3` is 7, 8 and 9. Each seed runs the whole job and every
structure comes back in one result.

`--seeds 1 2 3` runs the job once per seed, into a `seed_<n>` directory each,
and returns every structure together. The samples from one seed are correlated,
so this is the usual way to get independent predictions. Only some backends take
a seed list natively, so the loop lives in FoldJAX and every model gets it
identically.

Every run writes `foldjax_run.json` beside its structures: the model, the input
and its SHA-256, the resolved weights, the seeds and knobs actually used, the
JAX runtime, and each structure's confidence. A directory of `.cif` files
otherwise cannot say what produced it.

### Memory

The peaks in the [benchmark](#foldjax-against-upstream) are the defaults at
work; nothing below needs to be set to reach them.

- **Row-blocked pair stack.** The pair transition and triangle attention are
  evaluated a block of rows at a time -- mathematically exact, so there is no
  flag and no accuracy trade. Block sizes resolve from the token count.
- **Native-dtype triangle projections**, accumulating in float32 through
  `preferred_element_type`: float32 results at half the operand bytes.
- **Summaries, not logits.** Confidence is reduced in-graph; the full-bin
  PAE/PDE logits (`[samples, N, N, 64]` -- tens of GiB at long sequences) are
  returned only on request.

Design notes: [docs/engineering-notes.md](docs/engineering-notes.md).

### A bfloat16 trunk (`--option trunk_dtype=bf16`)

Protenix defaults to `bf16` (its upstream ships bf16-mixed); OpenDDE defaults
to `fp32` (so does its upstream) and takes the flag as an opt-in -- at 1,531
tokens it is the difference between completing and OOM on both sides. Boltz-2's
equivalent is `compute_dtype`, default `bfloat16`; OpenFold3 has no trunk
dtype: upstream runs `32-true` and a bf16 trunk destroys its prediction.
Details and measurements: [docs/engineering-notes.md](docs/engineering-notes.md).

### `--max-msa-depth`

Caps the `[depth, tokens, channels]` MSA representation, the dominant memory
term after the pair stack. What the cap costs in accuracy differs by model --
free on Protenix (its upstream subsamples anyway), pointless on OpenDDE,
harmful on some Chai-style stacks -- so FoldJAX translates the knob but never
imposes it. Per-model semantics and the measurements:
[docs/engineering-notes.md](docs/engineering-notes.md).

## Weights

Every upstream project publishes torch checkpoints; FoldJAX predicts from
torch-free JAX weights. `foldjax weights` owns that gap — it downloads from each
publisher, verifies against the published hash where one exists, converts once,
and stores the result under the FoldJAX home.

```bash
uv run foldjax setup                            # everything public, in one go
uv run foldjax weights list                     # what is downloaded and converted
uv run foldjax weights fetch --model opendde    # download, verify, convert
uv run foldjax weights path --model opendde     # where it landed
uv run foldjax home                             # every location FoldJAX uses
```

`setup` is the one to run on a new machine. It fetches and converts every model
whose weights are published — Boltz-2, OpenDDE, Protenix — along with the CCD
assets and the template metadata they share, and then reports what it could not
do for you: AlphaFold 3 and OpenFold3, whose publishers release parameters on
request, and whatever the template modality is still missing. MSA needs nothing
installed; it searches against the ColabFold MMseqs2 server.

```
~/.cache/foldjax/          # or $FOLDJAX_HOME, or a .foldjax/ in the checkout,
├── downloads/<model>/     #   or $XDG_CACHE_HOME/foldjax — `foldjax home` says
├── assets/                # CCD dictionaries shared by protenix and opendde
├── weights/<model>/       # converted JAX weights — what prediction loads
└── compile/               # XLA persistent compilation cache
```

`mkdir .foldjax` in a checkout and everything above lands there instead, which
keeps weights, assets and caches beside the source they belong to. `.gitignore`
already covers the directory, and `$FOLDJAX_HOME` still outranks it.

Conversion needs torch, which is why it is a separate step behind
`--extra torch-bridge`. Prediction never imports it.

No weights are redistributed. Each file is downloaded from its own publisher
under that project's terms, and nothing is fetched implicitly during prediction —
a missing model names the command that would get it.

Weight files exported before these ports were vendored recorded
`protenix_jax.*` / `opendde_jax.*` as their module names. The loader maps those
onto the vendored classes, so **existing weight files keep working**, and the
restricted unpickler still rejects arbitrary globals.

## GPU and the compile cache

These graphs take minutes to compile and seconds to replay, so the persistent
XLA cache is on by default and namespaced per model, weight identity, runtime
profile, and the options that actually change the compiled program. Different
models and weight sets never share a cache entry, and options that only affect
output formatting never fragment one.

The old comparison that lived here -- a 35-residue job at one sample and
20 steps, with three of its four upstream memory figures read off
`nvidia-smi` -- has been replaced by
[FoldJAX against upstream](#foldjax-against-upstream), which runs a real
schedule at three sizes and measures both sides the same way.

`--no-cache` turns it off for benchmark or ephemeral runs; `--cache-dir` moves
the root. Every model also runs on CPU, slowly.

## FoldJAX against upstream

Every model here is a JAX reimplementation of a published torch model, so the
only comparison worth making is against the repository it came from, on the
same job, under the same schedule, measured the same way.

That is what [`bench/`](bench/) does, and it is reproducible: one command per
row, one process per measurement, results written as they land.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/benchmark-dark.png">
  <source media="(prefers-color-scheme: light)" srcset="docs/benchmark-light.png">
  <img alt="FoldJAX vs upstream: wall time and peak GPU memory at 499, 1,003 and 3,012 tokens" src="docs/benchmark-dark.png">
</picture>

**The schedule is AlphaFold 3's released default — 5 diffusion samples, 200
diffusion steps, 10 recycles, seed 101 — not a number chosen here.** Protenix's
base model ships the same three. Boltz-2 defaults to 3 recycles rather than 10,
so this asks more of it than its own default does; it asks the same of its
upstream, which is what makes the comparison mean anything.

| tokens | model | FoldJAX s | upstream s | FoldJAX MiB | upstream MiB | speed | memory | confidence | FoldJAX | upstream |
|---|---|---|---|---|---|---|---|---|---|---|
| 499 | alphafold3 | 285 | - | 3,036 | - | - | - | - | - | - |
| 499 | boltz2 | 35 | 55 | 4,781 | 8,448 | 1.57x | 1.77x | complex_plddt | 0.9705 | 0.9700 |
| 499 | opendde | 81 | 75 | 12,649 | 18,555 | 0.92x | 1.47x | ranking_score | 0.1946 | 0.1947 |
| 499 | openfold3 | 39 | 97 | 10,442 | 19,193 | 2.46x | 1.84x | ptm | 0.9325 | 0.9274 |
| 499 | protenix | 28 | 63 | 5,045 | 5,130 | 2.26x | 1.02x | ranking_score | 0.1937 | 0.1935 |
| 1003 | alphafold3 | 368 | - | 6,901 | - | - | - | - | - | - |
| 1003 | boltz2 | 94 | 141 | 7,402 | 16,174 | 1.51x | 2.19x | complex_plddt | 0.9489 | 0.9482 |
| 1003 | opendde | 287 | 287 | 45,381 | 60,905 | 1.00x | 1.34x | ranking_score | 0.1926 | 0.1930 |
| 1003 | openfold3 | 126 | 323 | 12,043 | 25,620 | 2.56x | 2.13x | ptm | 0.9315 | 0.9300 |
| 1003 | protenix | 72 | 99 | 8,468 | 13,025 | 1.38x | 1.54x | ranking_score | 0.1921 | 0.1922 |
| 3012 | alphafold3 | 1,013 | - | 42,277 | - | - | - | - | - | - |
| 3012 | boltz2 | 981 | failed | 46,934 | failed | - | - | - | - | - |
| 3012 | opendde | failed | failed | failed | failed | - | - | - | - | - |
| 3012 | openfold3 | 1,374 | 6,722 | 66,264 | 83,210 | 4.89x | 1.26x | ptm | 0.8762 | 0.8748 |
| 3012 | protenix | 795 | 797 | 53,197 | 57,310 | 1.00x | 1.08x | ranking_score | 0.9554 | 0.9502 |

Read it with the method in mind:

- **The same job.** The upstream side's input is generated by FoldJAX's own
  translator, so neither side is running a hand-written approximation of the
  other's job file, and both name the same alignment.
- **The same measurement.** Peak memory is the live-bytes high-water mark on
  both sides — `peak_bytes_in_use` under JAX, `max_memory_allocated` under
  torch — never `nvidia-smi`, which reports the caching allocator's reserved
  pool and so tracks its doubling schedule rather than the model.
- **FoldJAX is timed warm.** Each case runs once to fill the XLA compile cache
  and that run is discarded. A cold JAX run is mostly compilation, which the
  torch side has no equivalent of: at 132 tokens it was 174 s against 29 s,
  almost all of it compiling. Compilation is paid once per shape and replayed
  from disk, which is what a second prediction costs.
- **The confidence columns are the same field**, chosen as the best one *both*
  implementations report. They are still different samples — the torch and JAX
  PRNG streams differ, so a shared seed is not a shared random tape. Same-seed
  numerical parity was established per port against a matched tape and is a
  separate exercise from this table.

**Every upstream runs on the fastest path this hardware lets it reach.**
[`bench/upstream-environments.md`](bench/upstream-environments.md) records each
virtualenv's provisioning, the kernels that cannot run on this card, and how to
verify both.

### What the table says

- **FoldJAX never uses more memory than the repository it reimplements.** The
  closest rows are Protenix at 499 tokens (1.02x) and 3,012 (1.08x); everywhere
  else the margin is 1.22x to 2.19x.
- **At 3,012 tokens, four FoldJAX implementations run; one upstream does.**
  Boltz-2's upstream reaches 96.2 GiB of the 97.9 GiB card even with
  `expandable_segments` and still cannot allocate -- a capacity limit, not
  fragmentation -- at a size FoldJAX completes in 46.9 GiB. OpenDDE fails on
  both sides: its structural branch wants a single 16.11 GiB buffer at ~2x
  the token count, which is model shape, not implementation.
- **On speed, read OpenFold3's column with its caveat.** Its upstream could
  reach no kernel on this card -- DS4Sci refuses sm_120 and 0.3.1's
  experimental cueq flag crashes at more than one diffusion sample -- and its
  predict preset chunks attention 4 rows at a time at every size, which is
  where the 6,722 s at 3,012 tokens lives. The other speed columns carry no
  such asterisk: Boltz-2 and Protenix upstreams run their released fast
  paths and FoldJAX is 1.0x to 2.3x against them.
- **AlphaFold 3 has no upstream column by design** -- FoldJAX drives the
  official implementation rather than reimplementing it, so both columns
  would run the same code. Its rows are the reference the ports are converging
  toward: flattest memory curve in the figure, every practice the ports
  adopted (bf16 throughout, fused attention, per-sample confidence,
  summaries-only outputs) is one it already had.
- **Confidence agrees on every complete row** -- ranking within 0.001-0.005,
  pTM within 0.005, FoldJAX marginally higher as often as lower. Two of those
  agreements took real fixes, both found the same way and both described
  below.

## Python API

One request type, every model. The knobs a request carries are model-neutral:
the same spelling means the same thing whichever backend runs it, and a knob a
backend cannot honour raises instead of silently doing something else.

```python
from foldjax import PredictionRequest, predict

result = predict(
    PredictionRequest(
        model="protenix",          # boltz2 / opendde / openfold3 / alphafold3
        input="job.yaml",          # FoldJAX job, or the model's native format
        output_dir="out/",
        seed=42,
        # Sampling, in one vocabulary. None keeps that backend's default.
        num_samples=5,
        num_steps=200,
        num_recycles=10,
        max_msa_depth=8192,
        # Execution, likewise: dtype, matmul_precision, triangle_kernel,
        # attention_kernel. "auto" is the fastest path this build can run --
        # never a silent fallback to a slower one.
        options={"dtype": "bfloat16", "triangle_kernel": "auto"},
    )
)
for sample in result.samples:
    print(sample.structure_path, sample.scores)
```

**One declaration, several runs.** Every axis that can name one thing can name
several -- the plural spelling fans out, nothing else changes. `models` x
`inputs` runs the cross product, each prediction into its own
`output_dir/<model>/<input stem>` subtree; `seeds` reruns each under every
seed. One result comes back per run, in declaration order:

```python
results = predict(
    PredictionRequest(
        models=("boltz2", "protenix", "openfold3"),
        inputs=("kinase.yaml", "complex.yaml"),
        seeds=(101, 102, 103),
        num_samples=5,
    )
)  # 3 models x 2 inputs -> 6 results, each holding 3 seeds x 5 samples
```

The CLI spells it the same way: `foldjax predict --model boltz2 protenix
--input kinase.yaml complex.yaml`.

`sample.scores` carries that model's released confidence summaries (pLDDT,
pTM/ipTM, ranking) per structure. The native option spellings each port grew
up with (`compute_dtype`, `trunk_dtype=bf16`, ...) still work and warn once;
`foldjax.execution.KNOBS` lists the neutral vocabulary.

Programs return summaries, not raw tensors, by default: the full-bin PAE/PDE
logits are `[n_samples, N, N, 64]` -- tens of GiB at long sequences -- so they
stay out of the outputs unless asked for (`return_confidence_logits=True` on
the Boltz-2 library API, `--include-raw` on the Protenix/OpenDDE CLIs, the raw
`.npz` formats elsewhere).

Leave a sampling knob unset and each backend runs its own upstream's released
default -- which differ, deliberately: matching each upstream is the whole
point of the ports. What `None` means per model:

| model | samples | steps | recycles | MSA depth |
|---|---|---|---|---|
| `alphafold3` | 5 | 200 | 10 | 1,024 rows (`evoformer.num_msa`) |
| `boltz2` | **1** | 200 | **3** | uncapped -- the whole alignment |
| `opendde` | 5 | 200 | 10 | 16,384 rows |
| `openfold3` | 5 | 200 | **3** | 1,024 rows, subsampled |
| `protenix` | 5 | 200 | 10 | 16,384 rows |

The bench table below pins all five to one schedule (5 / 200 / 10) precisely
because these defaults differ; set `num_recycles=10` on Boltz-2 and you are
asking more of it than `boltz predict` does.

`foldjax.resolve_request` applies every default without running anything, and
`foldjax.capabilities(model)` reports the input formats, entity types, and
sampling knobs a backend honours. Backend-specific output stays on
`PredictionResult.raw`.

## Environment

One Python 3.12 environment, no sibling checkouts. Mixed clusters pick a CUDA
generation per node from the same checkout and lockfile — the extras conflict
by construction, so each node type syncs its own venv:

```bash
uv sync --extra cuda13 --group dev                                  # CUDA 13 nodes
UV_PROJECT_ENVIRONMENT=.venv-cu12 uv sync --extra cuda12 --group dev  # CUDA 12 nodes
```

Weights, the compile cache, and every vendored source are shared between them;
nothing under `~/.foldjax` is CUDA-generation-specific.

- JAX / CUDA 13 plugin 0.10.1, cuequivariance-jax 0.10.0 with CUDA 13 ops 0.9.0,
  Tokamax 0.0.12

```bash
uv sync --extra cuda13 --group dev                       # predict with any model
uv sync --extra cuda13 --extra torch-bridge --group dev  # + convert checkpoints
```

**The prediction environment is torch-free for every model.** Boltz-2's
featurizer is vendored from upstream Boltz and written against torch; it runs
here on a NumPy array layer (`foldjax.models.boltz2.data._numpy_torch`) that
reproduces the torch featurizer exactly. `--extra boltz-preprocess` installs
real torch so the two can be compared, which is what
`tests/models/boltz2/test_featurize_torch_parity.py` does.

**AlphaFold 3** is carried at `foldjax.models.alphafold3` — upstream's source,
vendored verbatim under its Apache-2.0 licence, imported under its own name —
and runs from the same job file as the rest:
`foldjax predict --model alphafold3 --input job.yaml` with
`--extra alphafold3` installed. Its compiled half (one pybind module and the
chemical-component tables) is built in place on first use, a one-time step of
the same kind as fetching weights. The parameters are the one thing that
cannot ship: request them from Google under their own terms and place them in
the weight store — [docs/alphafold3.md](docs/alphafold3.md). An installed
`alphafold3` distribution, if you have one, is used instead of the vendored
copy.

**OpenFold3** is carried at `foldjax.models.openfold3` and predicts with no
extra: its inference path wants only `jax`, `numpy`, `safetensors` and `gemmi`,
all already in the base set. Its featurization is vendored beside it at
`models/openfold3/_upstream/` — upstream's own data pipeline, verbatim, under
its own licence — and needs `--extra openfold3-preprocess` for the torch and
lightning it imports; prediction needs neither.
[docs/openfold3.md](docs/openfold3.md) has the detail and the current state of
the port.

## Tests

```bash
JAX_PLATFORMS=cpu uv run --extra cuda13 --group dev pytest -q
uv run --extra cuda13 --group dev ruff check .

# One file, or one test, with no coverage gate in the way:
JAX_PLATFORMS=cpu uv run --extra cuda13 --group dev pytest -q tests/test_cache.py

# The 80% orchestration-layer gate, which only means anything on a full run.
# This is what CI enforces.
JAX_PLATFORMS=cpu uv run --extra cuda13 --group dev pytest -q \
    --cov=foldjax --cov-report=term-missing --cov-fail-under=80
```

1,669 tests pass with the optional extras installed, 88 skipped, 87% coverage on
the orchestration layer. Each vendored port brought its own suite to
`tests/models/<name>/`, including the torch-parity gates it was built against.

Those gates need an extra the torch-free environment deliberately omits, and the
suites handle it two different ways. Boltz-2's import torch at module scope, so
they have to be excluded at *collection* time;
`tests/models/test_optional_suite_gate.py` keeps that exclusion list honest
rather than letting a suite go quietly uncollected. OpenFold3's import torch
inside the test bodies instead, so they collect anywhere and skip at run time —
the better arrangement of the two, since a skipped test is reported and an
uncollected one is not.

Nothing in the suite needs a package that is on no index. The shared ColabFold
MMseqs2 client is `foldjax.search` — it has no third-party dependency, so
carrying it costs nothing and `--msa-server` works out of the box. The real-PDB
target loader is `tests/_foldbench`, kept under `tests/` rather than `src/`
because it fetches entries from RCSB over the network, which the installed
package has no business being able to do.

The tests built on real depositions still skip without `biotite`, which parses
them:

```bash
uv sync --extra cuda13 --extra openfold3-preprocess   # adds biotite
```

## Provenance

FoldJAX vendors these ports rather than depending on them, so one install covers
data intake through model output. Each keeps its upstream module layout,
`LICENSE`, and `NOTICE`, so it stays diffable against the repository it came
from; the top-level `NOTICE` lists every upstream, its licence, and where its
port lives.

Every port began as its own repository, and each carried notes the code alone
does not: an experiment log, a porting plan, the gates it had to pass, and the
benchmarks those gates were argued from. Those are in
[docs/ports/](docs/ports/), so nothing is left behind in a checkout that no
longer exists. They are history rather than instructions — the module names and
commands in them predate vendoring — but they are the only record of why a
given default was chosen and what was tried and rejected first.

Model weights are covered by their own licences and are never redistributed
here. AlphaFold 3 goes further — its parameters may not be redistributed at
all, so it is the one backend FoldJAX drives rather than carries. See
[docs/alphafold3.md](docs/alphafold3.md).
