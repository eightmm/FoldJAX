# Tutorial: batches, screens and Slurm

Run many jobs, through several models, as one restartable batch, and split it
over a Slurm array.

**Files you supply.** The jobs below name `target.a3m`, an alignment for the
protein, and a ligand library `library.smi` (one SMILES per line, a name after
it); none of them ships with FoldJAX. Put your own beside the job file, or drop
the `unpaired_msa` line and run with `--msa auto` (searches, sending the
sequence to a server) or `--msa single` (folds from the sequence alone) to try
the commands first.

## A batch is a list of inputs

Every model runs on every input; each pair gets its own run directory.

```bash
foldjax predict --model boltz2 protenix --input jobs/ --output-dir out \
    --msa auto --keep-going --resume
```

`--input` takes several files, a directory of job files, or a **multi-job
file**: a mapping with one `jobs` key whose list holds ordinary jobs, each with
a unique `name`.

```yaml
jobs:
  - name: target_apo
    entities:
      - {type: protein, id: A, sequence: MDTAYPREDTRAPTPSKAGAHTALTLGAPHPPPRDHLIWSVFSTLYLNLCCLGFLALAYSIKARDQKV, unpaired_msa: target.a3m}
  - name: target_atp
    entities:
      - {type: protein, id: A, sequence: MDTAYPREDTRAPTPSKAGAHTALTLGAPHPPPRDHLIWSVFSTLYLNLCCLGFLALAYSIKARDQKV, unpaired_msa: target.a3m}
      - {type: ligand, id: L, ccd: ATP}
```

Relative paths inside are resolved against the jobs file's directory. Each job
runs into `out/<model>/<job name>/`, and each run's `foldjax_run.json` names
the file and the job under `input.source`. `--keep-going` runs the rest when
one pair fails, records the failure in `out/foldjax_failures.json` and exits 3;
`--resume` skips every pair already finished
([resuming](resume.md)).

## A ligand screen

`foldjax jobs expand` writes one job per ligand of a library against one
target:

```bash
foldjax jobs expand --target target.json --ligands library.smi --affinity --out screen.json
```

The library is `.smi` (lines of `SMILES [name]`) or `.sdf` (the record title
as the name; coordinates are not carried). Jobs are named
`<target>__<ligand>`, independent of library order, so a grown or reordered
library keeps every finished directory. `--affinity` adds Boltz-2's affinity
request for the new ligand chain; unreadable records are refused unless
`--skip-invalid`. `foldjax jobs pulldown --baits A.fasta --candidates B.fasta`
does the same for protein-protein pairs.

Every job in a screen differs only in its ligand, so the shapes barely move.
Round them to buckets so that the screen compiles once per bucket rather than
once per ligand:

```bash
foldjax predict --model boltz2 --input screen.json --padding --output-dir screen-out --resume
foldjax show screen-out/ --screen              # top sample per job, affinity, ranked within each model
```

`--padding` is off by default because exact shapes are what the published
results were run at; it pads the MSA axis up to a bucket but never changes
which alignment rows a model reads. Profiles and pins:
[token-padding-profiles.md](../token-padding-profiles.md). `foldjax cache warm`
with the same flags compiles a bucket ahead of time
([configuration.md](../configuration.md#the-compile-cache)).

## Slurm arrays

`--shard I/N` runs every N-th unit of the batch starting at I (0-based), where
a unit is a plain input or one job of a multi-job file. Each unit lands in the
directory the unsharded batch would give it, with the same resume identity, so
shards run independently and any one of them can be resubmitted.
`--shard auto` reads `SLURM_ARRAY_TASK_ID` relative to `SLURM_ARRAY_TASK_MIN`
and the count from `SLURM_ARRAY_TASK_COUNT`; `auto/N` names the count
yourself.

Size the request first. `plan --json` adds a `slurm` block per run with the
`gres` to ask for and `min_device_memory_gib`, the model's fitted peak-law
estimate for that run (or `unknown` with the reason, outside a law's range,
for AlphaFold 3, and for models whose estimate needs the processed input).
A `--padding` run is `unknown` as well, with `serving padding` under
`exceeds_profile`: the laws were fitted unpadded and padded runs measured up
to 1.69x over the upper estimate, so the block keeps that estimate
(`upper_gib`) as a lower bound and sizes no card; leave headroom of your own.
`mem` is left null: no host-memory law is calibrated. `plan` checks the whole
batch without writing anything into the store.

```bash
foldjax plan --model boltz2 --input screen.json --padding --json > plan.json
```

Search the alignments once, where there is network, before the GPU jobs start:

```bash
foldjax msa prefetch screen.json --model boltz2
```

Then the array, one GPU per task:

```bash
#!/bin/bash
#SBATCH --job-name=screen
#SBATCH --array=0-7
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%A_%a.log

export FOLDJAX_PROGRESS_MODE=lines      # download progress as log lines
uv run foldjax predict --model boltz2 --input screen.json \
    --msa auto --padding --resume --keep-going \
    --output-dir screen-out --shard auto
```

With the alignments prefetched, `--msa auto` reads them from the cache and
the GPU node never contacts the server. All shards share one store
(`FOLDJAX_HOME`) and so one compile cache; if that store is group-writable on
purpose, see [compile-cache trust](../configuration.md#compile-cache-trust).

When the array finishes, read the whole batch as one table:

```bash
foldjax show screen-out/ --format csv > screen.csv     # one row per model/job/seed/sample, failures included
foldjax show screen-out/ --screen
```

Resubmitting the same array after a failure reruns only what is missing:
`--resume` skips every finished pair, shard by shard.

## From Python

```python
from pathlib import Path

from foldjax import PredictionRequest, predict_batch, results_table, load_results

report = predict_batch(
    PredictionRequest(
        models=("boltz2", "protenix"),
        inputs=("jobs.yaml",),
        output_dir=Path("out"),
        msa="auto",
        resume=True,
        on_error="continue",
    )
)
print(len(report.results), len(report.failures))
rows = results_table(load_results("out"))
```
