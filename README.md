<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/banner-dark.png">
  <source media="(prefers-color-scheme: light)" srcset="docs/banner-light.png">
  <img alt="FoldJAX — biomolecular structure prediction, compiled" src="docs/banner-dark.png">
</picture>

# FoldJAX

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eightmm/FoldJAX/blob/main/notebooks/FoldJAX_Colab.ipynb)

Biomolecular structure prediction in JAX. Six models behind one interface,
all carried inside the package:

| model | vendored at | licence | upstream |
|---|---|---|---|
| `alphafold3` | `foldjax.models.alphafold3` | Apache-2.0 | [google-deepmind/alphafold3](https://github.com/google-deepmind/alphafold3) |
| `boltz2` | `foldjax.models.boltz2` | MIT | [jwohlwend/boltz](https://github.com/jwohlwend/boltz) |
| `esmfold2` | `foldjax.models.esmfold2` | MIT + upstream notices | [biohub/ESMFold2](https://huggingface.co/biohub/ESMFold2) |
| `opendde` | `foldjax.models.opendde` | Apache-2.0 | [aurekaresearch/OpenDDE](https://huggingface.co/aurekaresearch/OpenDDE) |
| `openfold3` | `foldjax.models.openfold3` | Apache-2.0 | [aqlaboratory/openfold-3](https://github.com/aqlaboratory/openfold-3) |
| `protenix` | `foldjax.models.protenix` | Apache-2.0 | [bytedance/Protenix](https://github.com/bytedance/Protenix) |

Boltz-2, OpenDDE, Protenix, OpenFold3 and ESMFold2 are JAX reimplementations
that match their upstream's results; AlphaFold 3 is upstream's own source,
vendored. Weights are never redistributed. `foldjax weights fetch` downloads
and converts or stages public assets, including the exact public OpenFold3 p1
checkpoint; AlphaFold 3 weights are supplied manually after accepting their
publisher's terms.

Every advertised prediction path is Torch-free, including raw OpenFold3 jobs
and Protenix ESM/ISM conditioning. Publisher `.pt` checkpoints are a file
format here, not a runtime dependency: FoldJAX reads them with a restricted
NumPy archive reader. Neither the base install nor any public extra installs
PyTorch, Lightning, or TorchMetrics.

ESMFold2 is the odd one out and worth knowing about before you use it. It has
no evolutionary trunk: it is a diffusion structure head on a linear-recurrence
pair trunk, folding the representations of **ESMC-6B**, a 25 GB protein
language model that upstream distributes separately (`foldjax weights fetch
--model esmfold2` puts the complete bundle where the loader looks). It is also
*random at inference by design* —
random initial pair state, per-loop language-model dropout, sampler noise — so
two seeds give genuinely different structures rather than the same structure
twice. Protein only for now: supported protein jobs use FoldJAX's in-package
canonical chemistry. Ligands and nucleic acids are rejected because this
adapter does not build their features; the current runtime does not consume
upstream's 417 MB `ccd.pkl`.

The complete released ESMFold2 profile is the compatible default. To run the
explicit structure-network-only variant without downloading ESMC-6B, fetch its
940 MB profile and select the same profile for prediction:

```bash
foldjax weights fetch --model esmfold2 --profile structure-only
foldjax predict --model esmfold2 --input protein.json \
  --profile structure-only
```

`--profile structure-only` sets `no_language_model=true` itself; conflicting
native options are rejected instead of silently overriding the profile.
The older explicit option still selects that managed profile automatically
when weights are omitted. Without either spelling, FoldJAX still requires the complete
structure+ESMC release and never silently falls back to a different model.
An explicit `esmc_weights=/path/to/esmc` also selects the 940 MB managed
profile: the released language-model branch still runs, but its weights come
from that path rather than a redundant managed ESMC copy.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/benchmark-dark.png">
  <source media="(prefers-color-scheme: light)" srcset="docs/benchmark-light.png">
  <img alt="FoldJAX vs upstream: wall time and peak GPU memory at 499, 1,003 and 3,012 tokens" src="docs/benchmark-dark.png">
</picture>

*Each port against the repository it came from — same job, same schedule, same
measurement. Numbers, method and caveats: [docs/benchmark.md](docs/benchmark.md).*

## Examples

### Google Colab: compare models from one input

The [Google Colab workflow](notebooks/FoldJAX_Colab.ipynb) is a practical GPU
example built around FoldJAX's common input API. A compact form accepts protein,
DNA, RNA, CCD ligands, and SMILES ligands. Independent checkboxes expose all six
carried models; the default sends the exact same two-chain job to Protenix and
OpenDDE. ESMFold2 is marked protein-only with its 26.35 GB complete bundle,
while OpenFold3 uses its managed public p1 checkpoint (with an optional path
override) and AlphaFold 3 accepts a user-supplied parameter directory.
Compatibility is checked before any download. The notebook keeps large model
sessions sequential, records per-model failures without losing successful runs,
and presents a comparison table plus an interactive structure selector. The
final ZIP contains the common input, batch report, score table, structures,
confidence artifacts, and reproducibility manifests for every run.

The notebook requires and checks Python 3.13, installs the pinned JAX 0.11.1
and cuEquivariance 0.11.1 CUDA 12 runtime, shows checkpoint source/licence/size
before downloading, and keeps its short 1-sample/20-step/1-recycle tutorial
schedule clearly separated from every model's released defaults. In Python,
the same multi-model path is simply:

```python
from pathlib import Path

from foldjax import Job, PredictionRequest, predict_batch

job_path = Job.from_sequences(
    protein=("GIVEQCCTSICSLYQLENYCN", "FVNQHLCGSHLVEALYLVCGERGFFYTPKT"),
    name="insulin-complex",
).write("insulin-complex.json")
report = predict_batch(
    PredictionRequest(
        models=("protenix", "opendde"),
        input=job_path,
        output_dir=Path("foldjax-outputs"),
        msa="none",
        on_error="continue",
    )
)
```

## Installation

FoldJAX requires Python 3.13. Its lockfile pins JAX 0.11.1 and the matching
cuEquivariance JAX/CUDA packages for reproducible CPU and GPU environments.

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

Managed weight conversion and all six inference paths use NumPy/JAX only.
OpenFold3 can predict from a self-contained feature `.npz`; turning raw
JSON/YAML into that archive uses the Torch-free in-package featurizer and the
non-Torch chemistry packages in `--extra openfold3-preprocess`. Protenix
ESM/ISM conditioning runs the exact ESM2 architecture in JAX.

**AlphaFold 3** needs `--extra alphafold3`; its compiled half and generated CCD
tables are built under `$FOLDJAX_HOME/runtime/alphafold3` on first use — a
one-time step that also works from a read-only wheel install. FoldJAX uses its
versioned managed runtime by default; external source is explicit opt-in:
[docs/alphafold3.md](docs/alphafold3.md).
**OpenFold3** predicts from an embedded-chemistry `.npz` with no extra; raw job
featurization needs `--extra openfold3-preprocess`:
[docs/openfold3.md](docs/openfold3.md).

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
`unpaired_msa` have one neutral spelling and are translated where the backend
supports them; **a field a backend cannot express is rejected, not dropped** — silently
discarding an alignment would change the science without changing the exit
code.

The same document can be built in Python instead of written by hand:

```python
from foldjax import Job, Ligand, Modification, PredictionRequest, Protein

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

Sequences are normalized where the document is validated: whitespace is
removed, not trimmed, so a YAML block scalar (`sequence: |`) works; letters are
upper-cased; a nucleic-acid sequence is checked against IUPAC and points at the
offending position. Chain ids may be omitted entirely and are assigned `A`,
`B`, ... in document order — an id that is *present but blank* is still an
error, because that is a typo rather than an omission.

**Or skip the file.** A sequence is the smallest thing anyone has, and it is
enough:

```bash
uv run foldjax predict --model boltz2 --sequence MKTAYIAKQRQISFVK --ligand ATP
uv run foldjax predict --model protenix --input target.fasta   # one chain per record
uv run foldjax predict --model boltz2   --input 1abc.pdb       # re-fold a deposition
uv run foldjax predict --model boltz2   --input jobs/          # a directory is a batch
```

A `.pdb` or `.mmcif` file is read for its **chemistry, never its coordinates**:
polymer chains keep their sequences and names, non-water heteroatoms become CCD
ligands, and the point is to predict the positions again. `.cif` needs the
explicit `structure:1abc.cif` spelling, because that suffix is also a perfectly
good name for a job document. A residue with no one-letter code is refused by
name rather than dropped — express it as a `modifications` entry instead.

`--sequence`/`--dna`/`--rna`/`--ligand`/`--ligand-smiles` and FASTA files are
turned into an ordinary common-schema job under `$FOLDJAX_HOME/runtime/jobs/`
and run through the same path as any other input, so `plan` prints the
generated file and the manifest hashes it. FASTA records become protein chains
unless `Job.from_fasta(path, kind="rna")` says otherwise: `ACGT` is a valid
protein as well as valid DNA, and guessing would fold the wrong polymer
silently. `--ligand` takes CCD codes and `--ligand-smiles` takes SMILES,
separately, because `CCO` is both a plausible CCD code and ethanol.

### Alignments

A protein chain with no `unpaired_msa` is folded from its single sequence. That
is unchanged and still the default — but it now says so, once per run, instead
of being the invisible difference between a good prediction and a poor one.

```bash
uv run foldjax predict --model openfold3 --input job.yaml --msa auto
```

`--msa auto` searches the ColabFold MMseqs2 server for every protein chain that
arrived without an alignment and caches the result under
`$FOLDJAX_HOME/msa/`, keyed by sequence and search provenance. **It sends the
sequence to a third-party server** — that is why it is opt-in and why nothing
is searched by default; point `FOLDJAX_MSA_SERVER_URL` at your own instance for
sequences that must not leave. The cache is not
per model, so running one target through three backends searches once.
`--msa required` fails instead of falling back, which is what a batch script
wants: the silent fallback it guards against is a *successful* single-sequence
run. `FOLDJAX_MSA_SERVER_URL` points at a different server, and the URL is part
of the cache identity, so two servers never read each other's alignments.
For sequences that must not leave the machine — and for RNA, which no public
endpoint answers — point FoldJAX at a locally installed search instead:

```bash
export FOLDJAX_MSA_COMMAND="/opt/msa/run.sh"          # protein
export FOLDJAX_RNA_MSA_COMMAND="/opt/msa/rna.sh"      # RNA
export FOLDJAX_MSA_LOCAL_VERSION="uniref-2026-06"     # part of the cache identity
```

Each wrapper is called as `<command> --input query.fasta --output DIR` and
writes `pairing.a3m` and `non_pairing.a3m` (`rna_msa.a3m` for the RNA one).
Nothing is sent anywhere on this path. The version string is part of the cache
key, so upgrading a database invalidates the alignments it produced instead of
mixing generations. `foldjax doctor` prints which search is configured.

OpenDDE's released inference configuration sets `use_rna_msa=False`. A
FoldJAX common-schema job therefore rejects an RNA `unpaired_msa` or
`paired_msa` instead of materializing a file the model will discard; protein
MSAs remain supported. Native OpenDDE input keeps the upstream-compatibility
policy and emits a warning when it drops an RNA `unpairedMsaPath`.

### Templates and binding affinity

```yaml
entities:
  - type: protein
    id: [A]
    sequence: ACDEFG
    templates:
      - mmcif: templates/5xyz.cif
        query_indices: [1, 2, 3]
        template_indices: [7, 8, 9]
  - {type: ligand, id: [L], ccd: ATP}
properties:
  - affinity: {binder: L}
```

The two forms of a template are different inputs, not one input in two
spellings. AlphaFold 3 and Protenix require the query→template residue map and
refuse a bare file; **Boltz-2 aligns the mmCIF itself** and refuses a map it
would have to ignore. OpenDDE has native template machinery, but its released
inference configuration sets `use_template=False`: common-schema templates are
rejected instead of being converted to a `templatesPath` that will be dropped.
Its runtime capability therefore reports `supports_templates=false`; native
OpenDDE input retains the compatibility warning/drop policy. Affinity
reaches Boltz-2 alone — it is the only carried model with that head. OpenFold3
builds template features from its own pipeline and has no per-job field, so a
template addressed to it is refused. `foldjax capabilities --model MODEL`
reports both `common_schema_features` and `native_only_features` for exactly
this reason.

## CLI

```bash
uv run foldjax predict \
  --model protenix \
  --input job.yaml \
  --profile mini-esm-v0.5.0 \
  --num-samples 5 \
  --num-steps 200 \
  --num-recycles 10 \
  --seed 42
```

With one model and one input, the compatible default remains
`foldjax-outputs/<input-stem>`. An explicit `--output-dir` replaces that scalar
default. Batch requests add canonical model namespaces below their output root
so different models cannot overwrite one another.

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
model, input SHA-256, resolved weights and their stat/tree identity, the knobs
actually used, and each structure's confidence.

A run reports its stages on stderr and its results as a table on stdout:

```
[foldjax] protenix · job.yaml · seed 101
  prepare input        1.2s
  predict             6m18s
  write                0.3s

model     protenix        weights  protenix-v0.5.0
samples   15              time     6m21s     peak  18.4 GiB
seeds     101, 102, 103   msa      auto
best      seed 102 / sample 01     ranking_score 0.873
          seed-102_sample-01/job_seed-102_sample-01.cif
```

stdout stays JSON whenever it is not a terminal, so pipes and scripts are
unchanged; `--json` forces it and `--quiet` silences the stage lines
(`FOLDJAX_PROGRESS=0` does the same). The same stage timings are kept in
`foldjax_run.json` under `cost.phases`, because "it took eleven minutes" cannot
be acted on until it is split into a search, a compile and a schedule.
`foldjax show <output-dir>` renders that table again later, for one run or a
whole batch, from the manifests alone.

Batches get two flags that only make sense per model/input pair:

```bash
uv run foldjax predict --model boltz2 --input jobs/ --resume --keep-going
```

`--resume` skips a run only when its finished `foldjax_run.json` matches the
exact request, input, checkpoint and referenced-local-asset stat/tree
identities, and the persisted structure/representation artifacts. Any relevant
metadata or directory-tree change triggers a conservative rerun; FoldJAX does
not reread large checkpoint or asset payloads merely to write or check a
manifest. Checkpoint trees containing symlinks or special entries are likewise
treated as non-resumable. Legacy or incomplete manifests rerun conservatively;
a manifest's presence alone is not evidence that today's request produced it.
Matching works at **seed** granularity: every seed writes
its own manifest, so a five-seed job that died on the fourth repeats only what
is missing rather than all five. Trunk archives are restored lazily, including
their logical BF16 dtype, without loading a quadratic pair array just to decide
whether it is reusable.

Inputs, checkpoints, and referenced local assets must remain immutable while a
prediction is running. Resume detects changes observed between runs; it does
not lock files against concurrent replacement during backend execution.
`--keep-going` runs the rest of the batch when one run fails and exits 3 if any
did, instead of losing seventeen good predictions to the third one's OOM; what
failed is recorded in `foldjax_failures.json` beside the runs that did not,
because a successful run leaves a manifest behind and a failed one used to
leave nothing at all.

Both are request fields, not CLI-only flags:
`PredictionRequest(resume=True, on_error="continue")`. `foldjax.predict_batch`
returns a `BatchReport` with the results, the reused runs and the failures
together; `foldjax.predict` keeps its original return type.

Two commands answer the questions that used to need a failed run:

```bash
uv run foldjax models --for job.yaml   # which models can run this, and why not
uv run foldjax doctor                  # install, accelerator, weights, templates
uv run foldjax cache gc --older-than 30 --max-size 20G   # reports; --apply deletes
```

`models --for` is answered from the input translation table, so it needs no
weights, no GPU and no network. `cache gc` reports by default and deletes only
with `--apply`: cache entries are pure derived data, but they are still someone's
disk.

### Optional shape padding

JAX reuses a compiled executable only when all dynamic feature axes have the
same shape. FoldJAX therefore offers one opt-in padding switch across all six
models:

```bash
uv run foldjax predict --model boltz2 --input job.yaml --padding
```

With no padding flag, the exact historical model path and shapes are unchanged.
With `--padding`, each backend selects the smallest conservative *complete*
shape profile: tokens plus the atom, MSA, template, structural-token, and/or
language-model axes that actually enter that model. Masks keep padded entries
out of the structure and confidence output, and the run summary and
`foldjax_run.json` report the concrete profile that ran.

For a deployment profile or an exact cache warm, pin only the axes you need;
the remaining model axes still use safe automatic buckets:

```bash
uv run foldjax cache warm \
  --model openfold3 \
  --input job.yaml \
  --pad-tokens 512 \
  --pad-atoms 4096 \
  --pad-msa 128
```

The advanced axes are `--pad-templates`, `--pad-structural-tokens` (OpenDDE),
and `--pad-language-model-tokens` (ESMFold2/ESMC and Protenix ESM/ISM). A
target smaller than the materialized input is an error—padding never truncates
tokens or atoms.
Automatic profiles fail before compilation above their standard grid; use
`--padding-overflow exact` only when an unplanned exact-shape compile is
intentional. Padding can substantially increase quadratic token work, so a
4096-token profile is explicit capacity planning, not a good universal default.

`--profile` is the prediction-side spelling of `foldjax weights ... --profile`.
For Protenix, `mini-esm-v0.5.0` and `mini-ism-v0.5.0` select the matching mini
structure model and the ESM/ISM checkpoint staged beside it, so callers do not
need to know `model_name` or `esm_checkpoint_dir`. The Python API uses the same
field: `PredictionRequest(..., profile="mini-esm-v0.5.0")`.

To choose a model before downloading anything, use `foldjax models --json`.
It reports each canonical name, accepted inputs and execution knobs, whether
its weights are ready, their location and download size when known, and the
next setup step. Each accepted format also has an `input_requirements` entry:
the prediction and preprocessing runtimes, required install extras, whether
preprocessing imports PyTorch, and a short explanation. `foldjax capabilities
--model MODEL` remains the compact capability-only view with the same format
requirements.

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
free on Protenix (its upstream subsamples anyway), pointless on OpenDDE, and
potentially harmful when a model embeds the complete alignment before recycling
-- so FoldJAX translates the knob but never imposes it. Per-model semantics and
the measurements:
[docs/engineering-notes.md](docs/engineering-notes.md).

### Weights and setup

Upstreams publish several formats. `foldjax weights` downloads public files,
checks their pinned size and SHA-256, converts Boltz-2/OpenDDE/Protenix once,
stages ESMFold2 and OpenFold3's publisher-native checkpoints, and locates
manually supplied AlphaFold 3 parameters. Managed checkpoint loading remains
PyTorch-free.

```bash
uv run foldjax setup                            # default public models, in one go
uv run foldjax weights list                     # what is downloaded and converted
uv run foldjax weights fetch --model opendde    # download, verify, convert
uv run foldjax weights path --model opendde     # where it landed
uv run foldjax home                             # every location FoldJAX uses
uv run foldjax home --path runtime              # one script-friendly location
```

Weight preparation keeps its lifecycle and progress on stderr. `weights fetch`
retains a human-readable summary on stdout; use `weights path` when a script
needs path-only stdout. A fresh conversion shows `resolve`, per-file `download`,
`convert` (or `stage`/`direct`), `validate`, and `ready`, with elapsed time and
sizes; a repeat run explicitly reports that the verified native bundle was
skipped. Library callers can receive the same structured `AssetEvent` objects
through `foldjax.assets.fetch(..., on_event=callback)` instead of parsing text.

`setup` fetches and converts the default publicly published models — Boltz-2,
OpenDDE, Protenix — plus shared CCD assets. ESMFold2 is public but deliberately
opt-in because its full structure+ESMC bundle is about 26 GB; fetch it with
`foldjax weights fetch --model esmfold2`, or fetch only its 940 MB structure
network with `--profile structure-only`, then pass that same profile to
`foldjax predict`. OpenFold3 is also opt-in: `foldjax weights fetch --model
openfold3` downloads the public p1 checkpoint supported by this port. Upstream
p2 and OpenBind v0.5 checkpoints target newer incompatible architectures.
AlphaFold 3 releases parameters on request. MSA search needs nothing installed
(ColabFold MMseqs2 server).

```
~/.cache/foldjax/          # or $FOLDJAX_HOME, or a .foldjax/ in the checkout,
├── downloads/<model>/     #   or $XDG_CACHE_HOME/foldjax — `foldjax home` says
├── assets/                # CCD dictionaries shared by protenix and opendde
├── weights/<model>/       # prediction-ready weights/assets
├── compile/               # XLA persistent compilation cache
└── runtime/<model>/       # generated native binaries/data (AlphaFold 3)
```

`mkdir .foldjax` in a checkout keeps all of it beside the source. Where a `.pt`
archive needs conversion or direct loading, FoldJAX's restricted reader returns
NumPy arrays without importing PyTorch. No weights are redistributed — each
file comes from its own publisher under that project's terms, and nothing is
fetched implicitly during prediction.

### GPU and the compile cache

These graphs take minutes to compile and seconds to replay, so the persistent
XLA cache is on by default and namespaced per model, weight identity, runtime
profile, and the options that actually change the compiled program. Different
models and weight sets never share a cache entry, and options that only affect
output formatting never fragment one.

Warm the exact request before a production run on its deployment GPU:

```bash
uv run foldjax cache warm \
  --model opendde \
  --input job.yaml \
  --num-samples 5 \
  --num-steps 200 \
  --num-recycles 10
```

This deliberately uses an `execute_once` strategy: FoldJAX runs the first
requested seed through the normal prediction path, including GPU kernel
autotuning, then discards its prediction files. Pass `--output-dir` if that
warm-up prediction should be kept. The command reports the exact cache
namespace, elapsed time, device allocator peak, and files/bytes before and
after. Persistent entries are accelerator-, JAX-runtime-, weight-, input-shape-,
profile-, and static-option-specific, so warm representative production shapes
with the same options on the machine that will run them. One small input cannot
prewarm arbitrary larger shapes. A padded warm reports the one concrete full
shape profile it selected; it does not claim that every bucket or every
token/atom/MSA combination was compiled.

A zero cache delta is reported neutrally as `no_new_entry_observed`: directory
contents alone cannot prove an XLA cache hit, and FoldJAX does not pretend that
they can.

The cache setting is scoped to one FoldJAX request even though JAX exposes it
as process-wide configuration: FoldJAX serializes its own predictions, applies
the resolved directory while the backend runs, and restores the embedding
application's previous setting afterwards, including on failure. An embedding
application must not compile unrelated JAX programs concurrently in another
thread; use a separate process for mixed workloads because JAX has no
thread-local cache configuration.
`--no-cache` therefore disables reads and writes for that request rather than
merely omitting a directory argument; `use_compile_cache=False` is the Python
spelling. It cannot be combined with `--cache-dir`/`cache_dir`, because one asks
to disable the cache while the other asks where to use it. Every model also
runs on CPU, slowly.

## Python API

Everything the CLI does, as one request type — and the same neutral
vocabulary, so nothing below has a CLI-only or API-only spelling.

```python
from foldjax import PaddingConfig, PredictionRequest, predict

result = predict(
    PredictionRequest(
        model="protenix",          # boltz2 / esmfold2 / opendde / openfold3 / alphafold3
        input="job.yaml",          # FoldJAX job, or the model's native format
        output_dir="out/",
        seed=42,
        # Sampling, in one vocabulary. None keeps that backend's default.
        num_samples=5,
        num_steps=200,
        num_recycles=10,
        max_msa_depth=8192,
        # Opt-in shape normalization. `padding=True` uses standard buckets;
        # pinning axes defines one exact serving/cache profile.
        padding=PaddingConfig(tokens=512, atoms=4096, msa=128),
        # Execution, likewise: dtype, matmul_precision, triangle_kernel,
        # attention_kernel. "auto" is the fastest path this build can run --
        # never a silent fallback to a slower one.
        options={"dtype": "bfloat16", "triangle_kernel": "auto"},
    )
)
for sample in result.samples:
    print(sample.structure_path, sample.scores)
```

The same prewarming operation is available without the CLI:

```python
from foldjax import PredictionRequest, warm_cache

report = warm_cache(
    PredictionRequest(model="opendde", input="job.yaml", num_steps=200)
)
print(report.summary())
```

**One declaration, several runs.** Every axis that can name one thing can name
several -- the plural spelling fans out, nothing else changes. `models` x
`inputs` runs the cross product, each prediction into its own
`output_dir/<model>/<input stem>` subtree; `seeds` reruns each under every
seed. Model aliases are canonicalized before the path is chosen, and two input
files with the same stem are refused if they would collide. One result comes
back per model/input run, in declaration order:

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

Representation capture is single-seed: `PredictionResult` carries one
`Representations` archive handle, so combining `representations=...` with more
than one resolved seed is rejected during request resolution instead of
silently retaining only one archive. Run one seed per request when capturing
trunk arrays. Empty selectors such as `representations=(",",)` are also
rejected; name at least one array or use `"all"`.

The CLI spells it the same way: `foldjax predict --model boltz2 protenix
--input kinase.yaml complex.yaml`.

For multiple different models, omit `weights` so each resolves its managed
checkpoint. One explicit `weights` path is accepted only when every run uses
the same canonical model; this prevents handing one model another model's
checkpoint by accident.

`resolve_request(request)` resolves one scalar model/input pair.
`resolve_requests(request)` handles either spelling and returns the complete
tuple of scalar requests, including the plural cross product and final output
paths, without executing a model. `foldjax plan` exposes the latter behavior in
the CLI and prints one object for a scalar request or a list for a plural one.

`sample.scores` carries that model's released confidence summaries. Native
option spellings (`compute_dtype`, `trunk_dtype=bf16`, ...) still work and
warn once; `foldjax.execution.KNOBS` lists the neutral vocabulary. Predictions
return summaries, not raw tensors -- the full-bin PAE/PDE logits are tens of
GiB at long sequences and come back only on request
(`--option return_confidence_logits=true` on Boltz-2, `--option
include_raw=true` on OpenDDE, or the raw `.npz` output formats). Both are
native option names passed through `--option`, not FoldJAX flags of their own.

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

`foldjax.capabilities(model)` reports the runnable backend's scientific input
and sampling surface. A dormant code path disabled by the shipped inference
configuration is not a capability; for example, OpenDDE reports
`supports_templates=false` because its fixed `use_template=False` path drops
template input. Two fields further describe the *common schema* route:
`common_schema_features` is what this backend's dialect can carry from a
FoldJAX job document, and `native_only_features` names abilities the model has
but the common adapter cannot safely reach. Templates are the case that
matters: backend support does not imply that every input route honours a
per-job template.
Its `input_requirements` mapping distinguishes
dependencies by input format — for example, OpenFold3's `openfold3-features` archive is JAX-only,
while its raw `native`, `openfold3`, and `foldjax` formats require the
non-Torch `openfold3-preprocess` chemistry extra for JAX featurization.
`foldjax.model_info(model)` adds execution choices and managed-weight
readiness, path, provenance, download size, and setup guidance; it is the
Python counterpart of `foldjax models --json`. Backend-specific output stays
on `PredictionResult.raw`.

A backend returning normally is not enough to count as success. FoldJAX checks
that it returned a `PredictionResult` for the selected model, at least one
sample, and for every sample either a non-empty structure file or non-empty
coordinates. Violations raise `PredictionOutputError`. A vendored native CLI
that calls `SystemExit` is converted to `PredictionError` instead of terminating
the host Python process; diagnosed allocator failures remain `MemoryError`.
Copied native structures retain their real `.pdb`, `.cif`, or `.mmcif` suffix,
and multi-job AlphaFold 3 inputs keep separate job names instead of overwriting
one another. Secret-looking option values and URL credentials are redacted from
plans and run manifests.

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

Each port's original experiment log, porting plan and gates live in
[docs/ports/](docs/ports/) — history rather than instructions, but the only
record of why each default was chosen. Model weights are covered by their own
licences and never redistributed here; AlphaFold 3's must be requested from
