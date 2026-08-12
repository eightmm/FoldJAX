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
uv sync --extra cuda13 --group dev
uv run foldjax weights fetch --model boltz2      # download, verify, convert once
uv run foldjax predict --model boltz2 --input job.yaml
```

**Neither prediction nor weight conversion imports torch.** Upstream
checkpoints are torch archives, and FoldJAX reads them with its own
deserializer (`foldjax.torch_archive`, verified bit-identical against
`torch.load` tensor for tensor), so `--extra cuda13` is the whole runtime from
download to structure. The `torch-bridge` extra remains only for the parity
suites that compare ports against real torch modules.

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

`--input-format auto` decides on content, so native dialects (a Boltz YAML, a
Protenix job list) pass through untouched. Modifications, covalent bonds and
`unpaired_msa` have one neutral spelling and are translated into each backend's
own; **a field a backend cannot express is rejected, not dropped** — silently
discarding an alignment would change the science without changing the exit
code.

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

`Job.write` emits exactly the document above; `Job.read` takes either format
back.

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

All four reach all five models; setting both a neutral knob and its native
name is an error, and `--option KEY=VALUE` passes anything native straight
through. Seeds fan out the same way — `--seeds 0 1 2` (or `--num-seeds 3`)
runs the job once per seed into `seed_<n>` directories and returns every
structure together. Every run writes `foldjax_run.json` beside its structures:
model, input SHA-256, resolved weights, the knobs actually used, and each
structure's confidence.

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

`setup` fetches and converts every publicly published model — Boltz-2,
OpenDDE, Protenix — plus shared CCD assets, then names what it cannot do for
you: AlphaFold 3 and OpenFold3 release parameters on request. MSA search needs
nothing installed (ColabFold MMseqs2 server).

```
~/.cache/foldjax/          # or $FOLDJAX_HOME, or a .foldjax/ in the checkout,
├── downloads/<model>/     #   or $XDG_CACHE_HOME/foldjax — `foldjax home` says
├── assets/                # CCD dictionaries shared by protenix and opendde
├── weights/<model>/       # converted JAX weights — what prediction loads
└── compile/               # XLA persistent compilation cache
```

`mkdir .foldjax` in a checkout keeps all of it beside the source. Conversion
reads the torch archives with FoldJAX's own deserializer, so the whole flow
runs in the base environment. No weights are redistributed — each file comes
from its own publisher under that project's terms, and nothing is fetched
implicitly during prediction.

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

| tokens | model | FoldJAX s | upstream s | FoldJAX GiB | upstream GiB | speed | memory | confidence | FoldJAX | upstream |
|---|---|---|---|---|---|---|---|---|---|---|
| 499 | alphafold3 | 285 | - | 3.0 | - | - | - | - | - | - |
| 499 | boltz2 | 35 | 55 | 4.7 | 8.2 | 1.57x | 1.77x | complex_plddt | 0.9705 | 0.9700 |
| 499 | opendde | 81 | 75 | 12.4 | 18.1 | 0.92x | 1.47x | ranking_score | 0.1946 | 0.1947 |
| 499 | openfold3 | 39 | 97 | 10.2 | 18.7 | 2.46x | 1.84x | ptm | 0.9325 | 0.9274 |
| 499 | protenix | 28 | 63 | 4.9 | 5.0 | 2.26x | 1.02x | ranking_score | 0.1937 | 0.1935 |
| 1003 | alphafold3 | 368 | - | 6.7 | - | - | - | - | - | - |
| 1003 | boltz2 | 94 | 141 | 7.2 | 15.8 | 1.51x | 2.19x | complex_plddt | 0.9489 | 0.9482 |
| 1003 | opendde | 287 | 287 | 44.3 | 59.5 | 1.00x | 1.34x | ranking_score | 0.1926 | 0.1930 |
| 1003 | openfold3 | 126 | 323 | 11.8 | 25.0 | 2.56x | 2.13x | ptm | 0.9315 | 0.9300 |
| 1003 | protenix | 72 | 99 | 8.3 | 12.7 | 1.38x | 1.54x | ranking_score | 0.1921 | 0.1922 |
| 3012 | alphafold3 | 1,013 | - | 41.3 | - | - | - | - | - | - |
| 3012 | boltz2 | 981 | failed | 45.8 | failed | - | - | - | - | - |
| 3012 | opendde | failed | failed | failed | failed | - | - | - | - | - |
| 3012 | openfold3 | 1,374 | 6,722 | 64.7 | 81.3 | 4.89x | 1.26x | ptm | 0.8762 | 0.8748 |
| 3012 | protenix | 795 | 797 | 52.0 | 56.0 | 1.00x | 1.08x | ranking_score | 0.9554 | 0.9502 |

Method: **same job** (the upstream input is generated by FoldJAX's own
translator, both sides name the same alignment), **same measurement**
(live-bytes high-water mark on both sides, never `nvidia-smi`'s reserved
pool), **FoldJAX timed warm** (compilation is paid once per shape and replayed
from disk; the discarded first run is what fills the cache), and **the same
confidence field on both sides** — different samples, since the PRNG streams
differ; same-seed parity is a separate, per-port exercise.

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

`sample.scores` carries that model's released confidence summaries. Native
option spellings (`compute_dtype`, `trunk_dtype=bf16`, ...) still work and
warn once; `foldjax.execution.KNOBS` lists the neutral vocabulary. Predictions
return summaries, not raw tensors -- the full-bin PAE/PDE logits are tens of
GiB at long sequences and come back only on request
(`return_confidence_logits=True`, `--include-raw`, or the raw `.npz` formats).

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

Weights, the compile cache, and every vendored source are shared between
them; nothing under `~/.foldjax` is CUDA-generation-specific.

**Prediction is torch-free for every model**; conversion too. The extras that
carry torch exist for two things only: featurization parity suites
(`boltz-preprocess`, `torch-bridge`) and OpenFold3's vendored upstream data
pipeline (`openfold3-preprocess`) — prediction never needs them.

**AlphaFold 3** is vendored at `foldjax.models.alphafold3` (Apache-2.0,
imported under its own name; an installed distribution wins if present). Its
compiled half is built in place on first use — a one-time step like fetching
weights — and its parameters must be requested from Google under their own
terms: [docs/alphafold3.md](docs/alphafold3.md). **OpenFold3** predicts with
no extra; its featurization pipeline is vendored beside it and needs
`--extra openfold3-preprocess` for the torch it imports:
[docs/openfold3.md](docs/openfold3.md).

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

1,669 tests pass with the optional extras installed, 88 skipped, 87% coverage
on the orchestration layer. Each vendored port brought its own suite to
`tests/models/<name>/`, including the torch-parity gates it was built against;
those need the torch extras and skip (or are excluded at collection, with
`test_optional_suite_gate.py` keeping the list honest) without them. Tests on
real depositions skip without `biotite` (`--extra openfold3-preprocess`).

## Provenance

FoldJAX vendors these ports rather than depending on them, so one install covers
data intake through model output. Each keeps its upstream module layout,
`LICENSE`, and `NOTICE`, so it stays diffable against the repository it came
from; the top-level `NOTICE` lists every upstream, its licence, and where its
port lives.

Each port's original experiment log, porting plan and gates live in
[docs/ports/](docs/ports/) — history rather than instructions, but the only
record of why each default was chosen. Model weights are covered by their own
licences and never redistributed here; AlphaFold 3's must be requested from
Google ([docs/alphafold3.md](docs/alphafold3.md)).
