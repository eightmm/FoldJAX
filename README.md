# FoldJAX

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eightmm/FoldJAX/blob/main/notebooks/FoldJAX_Colab.ipynb)

Biomolecular structure prediction in JAX. Six models behind one interface,
all carried inside the package:

| model | ported upstream target | license | upstream |
|---|---|---|---|
| `alphafold3` | [3.0.4 (`85c4d20`)](docs/model-versions.md#alphafold-3) | Apache-2.0 code; parameters under DeepMind's own terms | [google-deepmind/alphafold3](https://github.com/google-deepmind/alphafold3) |
| `boltz2` | [2.2.1 metadata (`b1ebfc4`)](docs/model-versions.md#boltz-2) | MIT | [jwohlwend/boltz](https://github.com/jwohlwend/boltz) |
| `esmfold2` | [Biohub snapshot (`ef32577`)](docs/model-versions.md#esmfold2) | MIT + Biohub acceptable-use policy | [biohub/ESMFold2](https://huggingface.co/biohub/ESMFold2) |
| `opendde` | [1.1.1 (`ddfa1df`); base + ABAG](docs/model-versions.md#opendde) | Apache-2.0 | [aurekaresearch/OpenDDE](https://huggingface.co/aurekaresearch/OpenDDE#license) |
| `openfold3` | [0.5.0/OpenBind (`c477165`)](docs/model-versions.md#openfold3) | Apache-2.0 | [aqlaboratory/openfold-3](https://github.com/aqlaboratory/openfold-3) |
| `protenix` | [2.0.0 metadata (`4c355be`); five weight profiles](docs/model-versions.md#protenix) | Apache-2.0 | [bytedance/Protenix](https://github.com/bytedance/Protenix#license) |

These are fixed implementation targets, not aliases for upstream `latest`.
[The version and validation ledger](docs/model-versions.md) records the full
source commits, checkpoint revisions and hashes, the support boundary of every
profile, and the stage-level parity of each port against its upstream.

Boltz-2, OpenDDE, Protenix, OpenFold3 and ESMFold2 are JAX reimplementations;
AlphaFold 3 is upstream's own source, vendored with a small set of FoldJAX
edits. "Matches upstream" is always a scoped claim: feature parity,
matched-random-tape model parity, and independently sampled agreement are
recorded separately in the ledger. It never means that different models, or
different RNG implementations given the same integer seed, emit the same
coordinates.

Every prediction path is Torch-free. Publisher `.pt` checkpoints are a file
format here, not a runtime dependency: FoldJAX reads them with a restricted
NumPy archive reader, and neither the base install nor any public extra
installs PyTorch, Lightning or TorchMetrics.

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
measurement, and precision on both sides.* Across the 20
implementation/checkpoint pairs where both sides complete in that sweep,
FoldJAX holds a **1.08x to 4.53x** lower peak and runs at **0.94x to 4.85x** the
speed. Numbers, method, and the caveats that belong with them (including the
one row FoldJAX loses on time) are in [docs/benchmark.md](docs/benchmark.md).

## Installation

FoldJAX requires Python 3.13. Its lockfile pins JAX 0.11.1 and the matching
cuEquivariance JAX/CUDA packages for reproducible CPU and GPU environments.

```bash
uv sync                                       # CUDA 13 runtime + test tools
uv sync --no-default-groups --group dev       # CPU-only machine instead
uv run foldjax setup                          # fetch and convert the public weights
```

A CUDA 12 machine and mixed CUDA generations are covered in
[docs/install.md](docs/install.md). Two models want an extra:
`--extra alphafold3` (its compiled half and CCD tables build themselves on
first use) and `--extra openfold3-preprocess` (only to featurize raw OpenFold3
jobs; predicting from a feature `.npz` needs nothing).

Weights are never redistributed: each file comes from its own publisher, under
that project's terms. `foldjax setup` fetches, verifies and converts the default
checkpoint of every public model, plus Protenix's v2. ESMFold2 (about 26.8 GB)
and the other alternative profiles are opt-in: `foldjax setup --all` takes them
too, and `foldjax weights fetch --model M [--profile P]` takes one.
**AlphaFold 3 is the only model whose weights you supply yourself**, because
DeepMind releases its parameters only to applicants who accept their terms.
`foldjax doctor` reports what is installed and what is missing.

## Quick start

```bash
# A sequence is enough; --msa single folds without an alignment on purpose.
uv run foldjax predict --model boltz2 --sequence MKTAYIAKQRQISFVK --ligand ATP --msa single

# A job file (JSON/YAML, FASTA, a .pdb/.mmcif deposition to re-fold, or a
# model's native input) -- see docs/input.md.
uv run foldjax predict --model protenix --input job.yaml

# What would run, resolved, without running it.
uv run foldjax plan --model openfold3 --input job.yaml
```

Without `--msa`, a protein chain with no alignment is refused (ESMFold2, which
folds without one upstream, is exempt). `--msa auto` searches and caches an
alignment, and **sends the sequence to the public ColabFold MMseqs2 server**
unless `FOLDJAX_MSA_SERVER_URL` points at your own.

The [Colab notebook](notebooks/FoldJAX_Colab.ipynb) runs one input through
several models from a form, detecting the accelerator and installing the
matching JAX stack itself. What it caches, how it handles checkpoints, and why
its tutorial schedule is not any model's released one:
[docs/colab.md](docs/colab.md).

## One job, many models

`--model` and `--input` both take several values; every model runs on every
input, each into `<output-dir>/<model>/<input stem>/`. A directory of job files
is a batch too. `--keep-going` runs the rest when one pair fails and records
the failure in `foldjax_failures.json`; `--resume` skips every pair whose
directory already holds a finished run.

```bash
uv run foldjax predict --model boltz2 protenix openfold3 \
    --input jobs/ --output-dir out --keep-going --resume
```

The same in Python:

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
        msa="single",  # fold without alignments; "auto" searches a server
        on_error="continue",
    )
)

# Read the directory back: one row per model/input/seed/sample, failures too.
from foldjax import load_results, results_table

rows = results_table(load_results("foldjax-outputs"))
```

**Several jobs in one file.** A mapping with a single `jobs` key holds a list
of ordinary common-schema jobs, each with a unique `name`, in JSON or YAML:

```yaml
jobs:
  - name: kinase
    entities:
      - {type: protein, id: A, sequence: MKTAYIAKQR, unpaired_msa: kinase.a3m}
  - name: kinase_atp
    entities:
      - {type: protein, id: A, sequence: MKTAYIAKQR, unpaired_msa: kinase.a3m}
      - {type: ligand, id: L, ccd: ATP}
```

It runs exactly as a directory holding one file per job would, and each run's
`foldjax_run.json` records which file and job it came from under
`input.source`. In Python it is a batch input: `inputs=("jobs.yaml",)`.
Details: [docs/input.md](docs/input.md#several-jobs-in-one-file).

## Outputs

Every model writes the same layout:

```
<run>/foldjax_run.json                                     the run manifest
<run>/seed-<seed>_sample-<nn>/<job>_seed-<seed>_sample-<nn>.cif
<run>/seed-<seed>_sample-<nn>/confidence.json              one per sample
<run>/seed-<seed>_sample-<nn>/confidence_full.npz          arrays, where the model has them
<batch>/<model>/<input stem>/...                           one run per pair
<batch>/foldjax_failures.json                              runs that failed
```

`<nn>` is the diffusion sample index for every model, never a rank; a model's
own rank is kept as `native_rank`.

**`confidence.json`** keeps the model's own scalar scores under its own names
in `scores`, and adds a common `summary` with four fields -- `plddt` (0-100),
`ptm`, `iptm` and `ranking` (the model's own ranking score) -- each recording
the native score it came from, the transform applied, and what it was averaged
over, or `{"value": null, "reason": ...}` when the model does not report it or
it does not apply. A value is never 0 by default: a single-chain structure's
`iptm` is null even where the model wrote 0.0.

> Common fields standardize names and numerical scales. They retain
> model-specific definitions and calibration and do not establish comparable
> accuracy probabilities or authorize pooled cross-model ranking.

**`confidence_full.npz`** holds the per-sample confidence arrays the model's
compiled program already returns, with their index maps, units and axes in a
`_meta` record; FoldJAX computes no confidence quantity itself. Read it with
`foldjax.load_confidence_arrays(sample_dir)`. What a default run writes
differs by model (`foldjax capabilities --model M` lists it):

| model | PAE in `confidence_full.npz` |
|---|---|
| AlphaFold 3 | yes, by default (with PDE and contact probabilities) |
| Boltz-2 | yes, by default (with PDE) |
| Protenix | with `--option output_format=both` (with PDE and contact probabilities) |
| OpenDDE | with `--option include_raw=true` (with PDE and contact probabilities) |
| OpenFold3 | no: only binned logits, with `--option all_arrays=true`, in its native `<job>_raw.npz`; never decoded to angstroms |
| ESMFold2 | no: not among its compiled program's outputs |

Arrays a model does not return are listed in the archive's metadata and the
manifest with the reason.

**`foldjax_run.json`** is the run manifest: the input and its hash, the
weights' identity, the seeds and where they came from, the sampling and native
options, padding, runtime, what the run cost and what the memory policy
predicted, every sample with its scores and structure hash, the confidence
arrays written, and any input the model never read (`ignored_msas`,
`ignored_templates`, `ignored_constraints`).

Both files are described by JSON Schemas shipped in the package,
`foldjax/schemas/confidence.schema.json` and `foldjax/schemas/run.schema.json`
(`foldjax.summary.load_schema`). A minor schema version only adds optional
fields; removing, renaming or reinterpreting a field is a new major version.
The full field tables are in [docs/cli.md](docs/cli.md#outputs).

## Reading results back

```bash
uv run foldjax show out/                          # per-run table, for reading
uv run foldjax show out/ --format csv > runs.csv  # one row per model/input/seed/sample
uv run foldjax show out/ --format json --aggregate
uv run foldjax compare out/                       # pairwise RMSD within each input
```

`foldjax.load_results(root)` and `foldjax.results_table(...)` return the same
rows as `show --format csv|json`: identity, the common summary, native scores
as `score.<name>`, the structure path and SHA-256, and one row per failure.
They read only the canonical files above, never a backend's native side files.
`--aggregate` and `foldjax.aggregate_table` summarize within one input, model
and configuration, never across models; `best_within_model` marks the top of
one model's own confidence ordering within one run.

`foldjax compare` (`foldjax.compare_directory`) aligns every structure of each
input to every other one, across models, seeds and samples -- proteins on CA,
nucleic acids on C4' -- and writes RMSD, coverage and the residue
correspondence to `compare.json` and `compare.csv`.

## Screening many inputs

A batch compiles one program per input shape, so a screen over many inputs
spends most of its time compiling unless the shapes repeat.

- `--padding` rounds each input up to a token bucket (and the MSA axis to a row
  bucket), so jobs in one band share one compiled executable. It is off by
  default, because exact shapes are what the published results were run at;
  padding never changes which alignment rows a model reads.
  [Profiles and pinning](docs/token-padding-profiles.md).
- The persistent compile cache is on by default (`foldjax home --path
  compile_cache`); `--cache-dir` moves it, `--no-cache` skips it.
  `foldjax cache warm` runs a representative job once to populate it, and
  `foldjax cache gc` reports what could be reclaimed (`--apply` deletes).
- `--keep-going`, `--resume` and `foldjax show --format csv` make a long batch
  restartable and readable as one table.

## Memory admission

Boltz-2, Protenix, OpenFold3, OpenDDE and ESMFold2 carry a peak-memory law
fitted to this repository's own measurements, so they can answer before the
first compile whether a run fits the device. `--memory-check refuse` (the
default) stops a run that does not fit and names the levers;
`--memory-check warn` prints the same message and runs anyway;
`--memory-budget-gib` plans against a smaller card than the one you are on.
A run that needs more than its law was fitted at (`--padding`, or a float32
precision option) is still refused when over budget, but is never called a
fit: its estimate is a lower bound, so it proceeds as `unknown`.
AlphaFold 3 has no law and answers `unknown` when asked. Nothing is narrowed
automatically to make a job fit. `--mem-fraction` (default 0.9) sets how much
of the device JAX preallocates. Details: [docs/cli.md](docs/cli.md#memory).

## Precision

Four ports depart from the precision their publisher ships; ESMFold2 and
AlphaFold 3 do not.

| model | FoldJAX default | upstream | float32 matmuls in FoldJAX | to run upstream's precision |
|---|---|---|---|---|
| AlphaFold 3 | upstream's own code: bfloat16 (`bfloat16: 'all'`) | the same | not pinned (JAX `DEFAULT`) | nothing to change |
| Boltz-2 | bfloat16 trunk with a **bfloat16 pair residual**; matmul scope **`high`** | bf16-mixed with a float32 pair residual; `highest` | `high` | `--option pair_residual_dtype=float32 --option matmul_precision=highest` |
| ESMFold2 | bfloat16 trunk under the released fork's autocast region | the same (Biohub fork `ef32577`) | not pinned (JAX `DEFAULT`) | nothing to change |
| OpenDDE | **bfloat16 trunk** and bfloat16 confidence Pairformer | float32 | not pinned (JAX `DEFAULT`) | `--option dtype=float32 --option confidence_dtype=fp32` |
| OpenFold3 | **bfloat16 partial token/pair track** (the confidence head follows it) | `32-true` | `high` | `--option dtype=float32` |
| Protenix | bfloat16 trunk; **confidence head bfloat16 at every size** | confidence head float32 up to 2,560 tokens (v1 profiles; `protenix-v2` runs it bfloat16 at every size) | `high` | `--option amp_policy=upstream` |

Bold marks a departure. Each default was kept because the structures held
under measurement, not because the arithmetic matches upstream's. On the GPU
this repository measures on, `DEFAULT` and `high` both execute float32
matmuls as TF32; `--option matmul_precision=highest` asks any model for full
float32. The measurements behind each row are in
[docs/engineering-notes.md](docs/engineering-notes.md#which-precision-each-model-runs).

## Reference

| | |
|---|---|
| [Input](docs/input.md) | every format FoldJAX reads, what each backend accepts, and how alignments, templates and binding affinity are supplied |
| [Command line](docs/cli.md) | the complete `foldjax` surface: prediction, batches, the output contract, padding, memory knobs, native options, weights, compile cache |
| [Python API](docs/python-api.md) | requests, results, sessions, and the structured events the CLI renders |
| [Common model interface](docs/model-interface.md) | `foldjax.get_model(name)` with `embed`, `encode` and `predict` stages, `ModelConfig` and `ExecutionConfig` |
| [Benchmark](docs/benchmark.md) | the numbers above, their method, and what each one does not say |
| [Version and validation ledger](docs/model-versions.md) | exact upstream targets and the parity evidence for each port |
| [Engineering notes](docs/engineering-notes.md) | how the ports reached their current memory, speed and precision defaults |

Per-model notes: [AlphaFold 3](docs/alphafold3.md) ·
[OpenFold3](docs/openfold3.md) · [ESMFold2](docs/esmfold2.md) ·
[alignments](docs/alignment.md) ·
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
port brought its own suite to `tests/models/<name>/`. Clean runtime tests
hard-block external Torch, Lightning, TorchMetrics and fair-esm imports. Tests
on real depositions skip without `biotite` (`--extra openfold3-preprocess`).

`tests/parity/` is a CPU replay of stored native captures, deselected by
default and selected with `--run-cpu-parity`. It needs released weights and
fixtures that are fetched by digest rather than committed, so it does not run
in the CI job; [docs/parity-cpu.md](docs/parity-cpu.md) has the tiers, the
manifest schema, and what a passing run does and does not certify.

## Licenses

FoldJAX vendors these ports rather than depending on them, so one install covers
data intake through model output. Each keeps its upstream module layout,
`LICENSE` and `NOTICE`, so it stays diffable against the repository it came
from; the top-level `NOTICE` lists every upstream, its license, and where its
port lives. The parameter terms per model, which are not always the code
license, are in [docs/licences.md](docs/licences.md) -- publisher summaries,
not legal advice. Model weights are never redistributed here; AlphaFold 3's
must be requested from Google DeepMind under its
[model parameters terms](https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md).

Each port's original experiment log and gates live in
[docs/ports/](docs/ports/): history rather than instructions, but the record of
why each default was chosen.

## Citing

A FoldJAX manuscript is in preparation; until it is published, cite this
repository. Predictions made with a model are that model's: cite its own paper
as well, linked from its upstream repository in the table at the top.
