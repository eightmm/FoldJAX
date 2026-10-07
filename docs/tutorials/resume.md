# Tutorial: resuming a batch and handling failures

A long batch will be interrupted: a node is preempted, one input runs out of
memory, the queue's time limit arrives. `--resume` and `--keep-going` make the
same command safe to run again.

```bash
foldjax predict --model boltz2 protenix openfold3 --input jobs/ \
    --output-dir out --msa auto --keep-going --resume
```

## `--keep-going`: one failure does not stop the batch

Without it, the first failing model/input pair stops the batch. With it, the
rest run; each failure is recorded in `out/foldjax_failures.json` (model,
input, seed, output directory, error type and message, and the multi-job file
and job it came from), and the command exits 3. In Python:
`PredictionRequest(on_error="continue")`, and `predict_batch` returns the
results, the reused runs and the failures together.

```bash
foldjax show out/                    # ends with a footer listing recorded failures
foldjax show out/ --format csv       # a failed pair is a row with status: failed and its error
```

Every pair of a batch is checked before any of them runs, so a malformed job is
reported up front rather than an hour in.

## `--resume`: run only what is missing

A run's `foldjax_run.json` is written last, so its presence marks a finished
run. With `--resume`, a pair is skipped only when that manifest matches the
request exactly:

- the same model, input content, weights and profile, sampling knobs, native
  options (compared as the backend reads them, so `bf16` and `bfloat16` are one
  request), seeds, MSA policy and pairing, template policy and folder;
- unchanged files the job refers to (alignments, templates, and the MSA-cache
  entries a search read), and an unchanged checkpoint, which is compared by
  its file-system identity rather than re-read;
- the structures and arrays the manifest lists, still in place.

Anything else reruns, conservatively. A directory that is not reused says why,
as a line starting `[foldjax] not resumable:`. Matching works per **seed**:
every seed writes its own manifest, so a five-seed job that died on the fourth
seed reruns only what is missing.

A seed FoldJAX drew for a model with no upstream default seed is taken back
from the finished run's manifest, so `--resume` does not draw a new one.
Manifests from an older FoldJAX (schema `1.0`) still resume.

Inputs, checkpoints and the files a job names must not change while a run is
in progress: resume detects changes between runs, but nothing locks the files
during one.

## Out-of-memory failures in a batch

A model with a fitted memory law refuses a run that will not fit before it
compiles, which under `--keep-going` is a cheap, recorded failure rather than
a crash. Options for those pairs: run them on a larger card, under context
parallelism, or with `--memory-check warn` to let the allocator decide
([FAQ](../faq.md#out-of-memory)). Rerunning the same command with `--resume`
after changing flags reruns only the pairs whose request changed.

## Sharded batches

Under a Slurm array (`--shard auto`), each shard resumes independently, in the
directories the unsharded batch would use, so resubmitting the array, or one
task of it, reruns only what is missing ([batches and Slurm](batch-slurm.md)).
