<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/banner-dark.png">
  <source media="(prefers-color-scheme: light)" srcset="docs/banner-light.png">
  <img alt="FoldJAX — biomolecular structure prediction, compiled" src="docs/banner-dark.png">
</picture>

# FoldJAX

Biomolecular structure prediction in JAX. Six models behind one interface,
all carried inside the package:

| model | vendored at | licence | upstream |
|---|---|---|---|
| `alphafold3` | `foldjax.models.alphafold3` | Apache-2.0 | [google-deepmind/alphafold3](https://github.com/google-deepmind/alphafold3) |
| `boltz2` | `foldjax.models.boltz2` | MIT | [jwohlwend/boltz](https://github.com/jwohlwend/boltz) |
| `esmfold2` | `foldjax.models.esmfold2` | MIT | [biohub/ESMFold2](https://huggingface.co/biohub/ESMFold2) |
| `opendde` | `foldjax.models.opendde` | Apache-2.0 | [aurekaresearch/OpenDDE](https://huggingface.co/aurekaresearch/OpenDDE) |
| `openfold3` | `foldjax.models.openfold3` | Apache-2.0 | [aqlaboratory/openfold-3](https://github.com/aqlaboratory/openfold-3) |
| `protenix` | `foldjax.models.protenix` | Apache-2.0 | [bytedance/Protenix](https://github.com/bytedance/Protenix) |

Boltz-2, OpenDDE, Protenix, OpenFold3 and ESMFold2 are JAX reimplementations
that match their upstream's results; AlphaFold 3 is upstream's own source,
vendored. Weights are never redistributed — `foldjax weights fetch` gets each
from its publisher (AlphaFold 3's on request from Google).

ESMFold2 is the odd one out and worth knowing about before you use it. It has
no evolutionary trunk: it is a diffusion structure head on a linear-recurrence
pair trunk, folding the representations of **ESMC-6B**, a 25 GB protein
language model that upstream distributes separately (`scripts/fetch_esmc6b.sh`
puts it where the loader looks). It is also *random at inference by design* —
random initial pair state, per-loop language-model dropout, sampler noise — so
two seeds give genuinely different structures rather than the same structure
twice. Protein only for now; ligands and nucleic acids need reference
conformers this adapter does not build yet.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/benchmark-dark.png">
  <source media="(prefers-color-scheme: light)" srcset="docs/benchmark-light.png">
  <img alt="FoldJAX vs upstream: wall time and peak GPU memory at 499, 1,003 and 3,012 tokens" src="docs/benchmark-dark.png">
</picture>

*Each port against the repository it came from — same job, same schedule, same
measurement. Numbers, method and caveats: [docs/benchmark.md](docs/benchmark.md).*

## Installation

```bash
uv sync --extra cuda13 --group dev
uv run foldjax weights fetch --model boltz2      # download, verify, convert once
uv run foldjax predict --model boltz2 --input job.yaml
```

Mixed clusters pick a CUDA generation per node from the same checkout and
lockfile — the extras conflict by construction, so each node type syncs its
own venv; weights, caches and sources are shared:

```bash
uv sync --extra cuda13 --group dev                                    # CUDA 13 nodes
UV_PROJECT_ENVIRONMENT=.venv-cu12 uv sync --extra cuda12 --group dev  # CUDA 12 nodes
```

**Neither prediction nor weight conversion imports torch** — checkpoints are
read by FoldJAX's own deserializer (`foldjax.torch_archive`, bit-identical
against `torch.load`). The extras that carry torch exist for featurization
parity suites (`boltz-preprocess`, `torch-bridge`) and OpenFold3's vendored
upstream data pipeline (`openfold3-preprocess`); prediction never needs them.

**AlphaFold 3** needs `--extra alphafold3`; its compiled half is built in
place on first use — a one-time step like fetching weights — and an installed
`alphafold3` distribution wins if present: [docs/alphafold3.md](docs/alphafold3.md).
**OpenFold3** predicts with no extra; building features needs
`--extra openfold3-preprocess`: [docs/openfold3.md](docs/openfold3.md).

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

## CLI

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

| neutral | alphafold3 | boltz2 | esmfold2 | openfold3 | opendde / protenix |
|---|---|---|---|---|---|
| `--num-samples` | `heads.diffusion.eval.num_samples` | `diffusion_samples` | `num_samples` | `num_samples` | `n_sample` |
| `--num-steps` | `heads.diffusion.eval.steps` | `steps` | `num_steps` | `no_rollout_steps` | `n_step` |
| `--num-recycles` | `num_recycles` | `recycling` | `num_loops` | `num_cycles` | `n_cycle` |
| `--max-msa-depth` | `evoformer.num_msa` | `max_msa_depth` | `msa_max_depth` | *feature cut* | `max_msa_rows` |

All four reach all six models; setting both a neutral knob and its native
name is an error, and `--option KEY=VALUE` passes anything native straight
through. Seeds fan out the same way — `--seeds 0 1 2` (or `--num-seeds 3`)
runs the job once per seed into `seed_<n>` directories and returns every
structure together. Every run writes `foldjax_run.json` beside its structures:
model, input SHA-256, resolved weights, the knobs actually used, and each
structure's confidence.

### Memory

The peaks in the [benchmark](docs/benchmark.md) are the defaults at
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

### Weights and setup

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

### GPU and the compile cache

These graphs take minutes to compile and seconds to replay, so the persistent
XLA cache is on by default and namespaced per model, weight identity, runtime
profile, and the options that actually change the compiled program. Different
models and weight sets never share a cache entry, and options that only affect
output formatting never fragment one.

`--no-cache` turns it off for benchmark or ephemeral runs; `--cache-dir` moves
the root. Every model also runs on CPU, slowly.

## Python API

Everything the CLI does, as one request type — and the same neutral
vocabulary, so nothing below has a CLI-only or API-only spelling.

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
| `esmfold2` | **32** | **14** | **3** | 1,024 rows, resubsampled per loop |
| `opendde` | 5 | 200 | 10 | 16,384 rows |
| `openfold3` | 5 | 200 | **3** | 1,024 rows, subsampled |
| `protenix` | 5 | 200 | 10 | 16,384 rows |

ESMFold2's row is not a typo: its released `config.json` asks for 32 samples
off a 14-step schedule, and the schedule is then clipped at `sigma = 256`, so
it runs fewer denoiser calls than the number suggests. Its own source's
dataclass defaults say 68 steps and 20 loops; the file wins.

The benchmark pins the five AlphaFold-3-shaped models to one schedule
(5 / 200 / 10) precisely because these defaults differ; set `num_recycles=10`
on Boltz-2 and you are asking more of it than `boltz predict` does. ESMFold2 is
not on that chart: it has no evolutionary trunk and no comparable schedule, so
putting it on the same axes would compare two different questions.

`foldjax.resolve_request` applies every default without running anything, and
`foldjax.capabilities(model)` reports the input formats, entity types, and
sampling knobs a backend honours. Backend-specific output stays on
`PredictionResult.raw`.

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
