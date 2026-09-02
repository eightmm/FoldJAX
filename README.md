# FoldJAX

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eightmm/FoldJAX/blob/main/notebooks/FoldJAX_Colab.ipynb)

Biomolecular structure prediction in JAX. Six models behind one interface,
all carried inside the package:

| model | licence | upstream |
|---|---|---|
| `alphafold3` | Apache-2.0 code; parameters under DeepMind's own terms | [google-deepmind/alphafold3](https://github.com/google-deepmind/alphafold3) |
| `boltz2` | MIT | [jwohlwend/boltz](https://github.com/jwohlwend/boltz) |
| `esmfold2` | MIT + Biohub acceptable-use policy | [biohub/ESMFold2](https://huggingface.co/biohub/ESMFold2) |
| `opendde` | Apache-2.0 | [aurekaresearch/OpenDDE](https://huggingface.co/aurekaresearch/OpenDDE#license) |
| `openfold3` | Apache-2.0 | [OpenFold/OpenFold3](https://huggingface.co/OpenFold/OpenFold3) |
| `protenix` | Apache-2.0 | [bytedance/Protenix](https://github.com/bytedance/Protenix#license) |

Each is vendored at `foldjax.models.<name>`. The full parameter terms per
model, which are not always the code licence, are in
[docs/licences.md](docs/licences.md) — publisher summaries, not legal advice.

Boltz-2, OpenDDE, Protenix, OpenFold3 and ESMFold2 are JAX reimplementations
that match their upstream's results; AlphaFold 3 is upstream's own source,
vendored. Weights are never redistributed — each file comes from its own
publisher, under that project's terms. `foldjax setup` fetches every one that
is published, OpenFold3's exact p1 checkpoint included, and converts or stages
it. **AlphaFold 3 is the only model whose weights you have to supply
yourself**, because DeepMind releases its parameters only to applicants who
accept their terms. ESMFold2 is public but stays opt-in for its size: the full
bundle is about 26.8 GB.

Every advertised prediction path is Torch-free, including raw OpenFold3 jobs
and Protenix ESM/ISM conditioning. Publisher `.pt` checkpoints are a file
format here, not a runtime dependency: FoldJAX reads them with a restricted
NumPy archive reader. Neither the base install nor any public extra installs
PyTorch, Lightning, or TorchMetrics.

ESMFold2 is the odd one out: no evolutionary trunk, a 25 GB **ESMC-6B**
language model underneath, and *random at inference by design*, so two seeds
give genuinely different structures. Read [docs/esmfold2.md](docs/esmfold2.md)
before using it.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/benchmark-dark.png">
  <source media="(prefers-color-scheme: light)" srcset="docs/benchmark-light.png">
  <img alt="FoldJAX vs upstream: wall time and peak GPU memory at 499, 1,003, 1,354, 2,096, 3,012, 4,100 and 4,926 tokens" src="docs/benchmark-dark.png">
</picture>

*Each port against the repository it came from — same job, nominal schedule,
measurement, and precision on both sides. The daggered historical OpenFold3
rows used a different effective seed; numbers, method and caveats are in
[docs/benchmark.md](docs/benchmark.md).*

Across the 20 implementation/checkpoint pairs where both sides complete,
FoldJAX holds a **1.08x to 4.53x** lower peak and runs at **0.94x to 4.85x** the
speed. At 3,012 tokens, five FoldJAX rows finish (four model families plus the
Protenix-v2 checkpoint), versus two upstream rows. At 4,100 -- a size measured
on the FoldJAX side only, and where this card's ceiling turns out to sit --
three still finish and three do not. The caveats that belong with
those numbers — including the one row FoldJAX loses on time — are in
[docs/benchmark.md](docs/benchmark.md).

### Current memory-bound route changes

The default inference routes now bound sample-expanded or quadratic
intermediates where measurements justify it:

- AlphaFold 3 evaluates more than five diffusion samples in released-width
  groups of five. On the 970-token, 20-sample A/B this reduced peak live bytes
  from 9,515.8 to 6,748.9 MiB; matched structures had minimum TM-score 0.99982.
- OpenFold3 runs confidence one sample at a time whenever there is more than
  one. At 490 tokens its 5-, 10-, and 20-sample peaks are all about 4,264 MiB;
  the previous batched 20-sample route reached 31,930 MiB.
- OpenDDE projects structural relative positions directly from their four
  categorical indices. At 1,886 structural tokens this cuts that projection's
  XLA temporary allocation from 2,011,244,544 to 10,672,144 bytes while keeping
  the same output allocation. The five-sample whole-process peak stayed flat
  because the pair trunk, not this isolated projection, dominates that run.
- Managed AlphaFold 3 GPU runs persist a source-, hardware-, and call-qualified
  Tokamax configuration only after strict coverage succeeds, so a fresh process
  can select the verified Triton kernels before tracing.

These are targeted graph changes, not a global attention-backend switch;
one-sample controls and OpenFold3's publisher-native path remain covered by
their existing parity gates. The exact schedules, timings, numerical checks,
source-bound final GPU gate, and cold-autotuning caveats are in
[docs/benchmark.md](docs/benchmark.md),
[the final-tree evidence](bench/experiments/final-tree-gpu-gate-2026-08-31.json),
[docs/alphafold3.md](docs/alphafold3.md), and
[docs/openfold3.md](docs/openfold3.md).

### Two defaults are FoldJAX's own

Four ports run the precision their upstream runs — AlphaFold 3, Boltz-2 and
Protenix on bfloat16, OpenFold3 on float32. Two do not, and both are measured.

**OpenDDE defaults to a bfloat16 trunk where upstream ships float32**, measured
at 1,003 tokens over two runs per arm: peak 42.96 → 19.18 GiB, wall 280.4 →
174.4 s, identical best `ranking_score`, and a median CA RMSD of 0.0301 Å
against the 0.0142 Å you get by running either arm twice. It moves the wall
too — at 1,531 tokens float32 asks for 86.7 GiB and dies where bfloat16
finishes at 44.2 GiB. `--option dtype=float32` puts it back, and the benchmark
above pins exactly that, because a row that compared two precisions would not
be a comparison.

**ESMFold2 runs a bfloat16 trunk where upstream's released checkpoint is
float32.** Upstream's single autocast wraps its language model alone; the
folding trunk is not covered by it. At 1,003 tokens on the released schedule
the width costs 0.0992 Å all-atom against a 0.0230 Å rerun floor — resolvable,
unlike on Boltz-2 — and buys 27.38 → 16.78 s and 32.84 → 23.47 GiB with pLDDT
and pTM identical to four decimals. That is nine times inside the 0.9135 Å this
model's own diffusion samples already disagree by. There is no flag: the width
is not reachable from any knob, which is the honest description of a choice
rather than an option.

Both are recorded with their measurements in
[docs/engineering-notes.md](docs/engineering-notes.md).

## Installation

FoldJAX requires Python 3.13. Its lockfile pins JAX 0.11.1 and the matching
cuEquivariance JAX/CUDA packages for reproducible CPU and GPU environments.

```bash
uv sync
uv run foldjax weights fetch --model boltz2      # download, verify, convert once
uv run foldjax predict --model boltz2 --input job.yaml
```

`uv sync` installs the CUDA 13 runtime and the test tools without being asked;
a CPU-only machine wants `uv sync --no-default-groups --group dev` instead, and
a CUDA 12 one is in [docs/install.md](docs/install.md).

Two models want an extra: `--extra alphafold3` (its compiled half and CCD
tables build themselves on first use) and `--extra openfold3-preprocess` (only
to featurize raw OpenFold3 jobs — predicting from a feature `.npz` needs
nothing). Mixed CUDA generations, and what each extra actually installs:
[docs/install.md](docs/install.md).

## Try it

The [Colab notebook](notebooks/FoldJAX_Colab.ipynb) runs one input through
several models from a form, detecting the accelerator and installing the
matching JAX stack itself. What it caches, how it handles checkpoints, and why
its tutorial schedule is not any model's released one:
[docs/colab.md](docs/colab.md).

The same multi-model path in Python:

```python
from pathlib import Path

from foldjax import Job, PredictionRequest, predict_batch

job_path = Job.from_sequences(
    protein=("MKTAYIAKQRQISFVKSHFSRQDILDLWIYHTQGYFPDWQNYTPGPGIRYPLTFGWCFKLVPVDPEEVVEELEKAGVE",),
    rna=("GGGAAACCC",),
    ligand_ccd=("ATP",),
    name="protein-rna-atp-demo",
).store()
report = predict_batch(
    PredictionRequest(
        models=("protenix", "opendde", "openfold3"),
        input=job_path,
        output_dir=Path("foldjax-outputs"),
        msa="none",
        on_error="continue",
    )
)
```

## Reference

The three long ones live beside the code rather than in this file, because a
README that documents every flag stops being a README.

| | |
|---|---|
| [Input](docs/input.md) | every format FoldJAX reads, what each backend accepts, and how alignments, templates and binding affinity are supplied |
| [Command line](docs/cli.md) | the complete `foldjax` surface: prediction, batches, padding, memory knobs, weights, compile cache |
| [Python API](docs/python-api.md) | requests, results, sessions, and the structured events the CLI renders |
| [Benchmark](docs/benchmark.md) | the numbers above, their method, and what each one does not say |

Per-model notes: [AlphaFold 3](docs/alphafold3.md) ·
[OpenFold3](docs/openfold3.md) · [alignments](docs/alignment.md) ·
[context parallelism](docs/context_parallel.md)

## Tests

```bash
JAX_PLATFORMS=cpu uv run pytest -q
uv run ruff check .

# One file, or one test, with no coverage gate in the way:
JAX_PLATFORMS=cpu uv run pytest -q tests/test_cache.py

# The 80% orchestration-layer gate, which only means anything on a full run.
# This is what CI enforces.
JAX_PLATFORMS=cpu uv run pytest -q \
    --cov=foldjax --cov-report=term-missing --cov-fail-under=80
```

The CPU suite and the 80% orchestration coverage gate run in CI. Each vendored
port brought its own suite to `tests/models/<name>/`. Historical publisher
parity gates may use a separately provisioned reference environment, but those
dependencies are deliberately absent from FoldJAX's package metadata. Clean
runtime tests hard-block external Torch, Lightning, TorchMetrics, and fair-esm
imports. Tests on real depositions skip without `biotite`
(`--extra openfold3-preprocess`).

## Provenance

FoldJAX vendors these ports rather than depending on them, so one install covers
data intake through model output. Each keeps its upstream module layout,
`LICENSE`, and `NOTICE`, so it stays diffable against the repository it came
from; the top-level `NOTICE` lists every upstream, its licence, and where its
port lives.

Each port's original experiment log and gates live in
[docs/ports/](docs/ports/) — history rather than instructions, but the only
record of why each default was chosen. Model weights are covered by their own
licences and never redistributed here; AlphaFold 3's must be requested from
