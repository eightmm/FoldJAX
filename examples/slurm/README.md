# Slurm examples

These Bash examples submit one CPU MSA-prefetch job, then one GPU prediction job
per model. Each GPU job uses one GPU; `--shards N` makes each an array. Install
FoldJAX and its weights first, and make the same `FOLDJAX_HOME` visible on every
node. The scripts require Bash and Slurm, and never install software or weights.

Fill the settings at the top of the scripts or export them in your submission
environment. No site partition, module, GPU type, CPU count, account, or timing
measurement is supplied here:

```bash
export PARTITIONS_GPU=''  # your GPU partition(s), comma-separated if appropriate
export GRES='gpu:1'
export CPUS_PER_GPU=''    # your site's CPU allocation per GPU
export PARTITION_CPU=''   # CPU partition for MSA prefetch
export MODULE=''          # optional installed module name
export FOLDJAX_HOME=''    # shared installation's data store
export FIXED_MSA_DEPTH='' # optional measured MSA cap; no default is chosen
```

`foldjax-submit` requires the GPU partition and CPU count to be filled; auto MSA
also requires the CPU partition. `MODULE` must be usable by the site's `module`
shell function in both the submission shell and batch shell. Initialize that
function according to your site if necessary. Slurm cannot expand shell
variables in `#SBATCH` directives, so allocation flags are passed to `sbatch`.
The job scripts start in `SLURM_SUBMIT_DIR`; submit from a directory visible on
the compute nodes and keep relative input paths relative to that directory.

```bash
examples/slurm/foldjax-submit -m boltz2 -m protenix -o results \
  --msa auto --time 04:00:00 --shards 4 --dry-run -- \
  --input jobs/ --num-samples 5
```

Review the printed commands, then remove `--dry-run` to submit. Dry-run performs
no FoldJAX execution, input validation, directory creation, or scheduler calls.
A real submission runs `foldjax plan` for every model before submitting any job.
The CPU job uses `foldjax msa prefetch --model ... -- INPUTS...`, with two CPUs
and 8 GiB requested. Those resources suit a remote search client; size local
MMseqs2 searches separately. The GPU jobs depend on `afterok:<msa-job-id>` with
`--kill-on-invalid-dep=yes`; a failed CPU search prevents wasted GPU work.

`--msa auto` can send sequences to the public ColabFold service. Inform the user
before using it; configure `FOLDJAX_MSA_COMMAND` for local search or
`FOLDJAX_MSA_SERVER_URL` for an approved server as needed. Prefetch uses the same
`--msa-pairing` value as prediction. The wrapper requires `--input` for this
mode because prefetch accepts jobs, FASTA/structures, and directories rather
than raw `--sequence` arguments. `--msa single` submits no CPU search job and
accepts the usual prediction input forms.

`--gpu TYPE` requests `gpu:TYPE:1`. Choose a type supported by your installation:
FoldJAX GPU inference requires NVIDIA sm_80 or newer. GPU types and compilation
caches are specific to the selected hardware/runtime; warm every type you use.
The first prediction or warm job may spend substantial time compiling and
autotuning. Choose wall time and memory limits from your own measurements.

Prediction always receives `--resume --keep-going`. Resubmitting the same
command revisits failed or missing runs and reuses finished matching manifests.
With multiple models, this wrapper places each model below `OUTDIR/<model>`;
single-model submissions use `OUTDIR`. Logs live in `OUTDIR/.slurm` and include
the Slurm job/array identifiers. `--shards N` sets `--array=0-(N-1)` and passes
`--shard auto`; give it file/directory inputs, because raw sequences are one run.
Everything after `--` is passed as literal argv, including spaces and shell
metacharacters. Set model, output, MSA mode, fixed-MSA depth, and sharding with
the wrapper options before `--`; conflicting copies and duplicate model aliases
are rejected so independent jobs cannot write one output directory concurrently.

For Boltz-2, Protenix, OpenFold3, and OpenDDE, either `--fixed-msa N` or an
explicitly configured `FIXED_MSA_DEPTH=N` enables the same fixed-MSA preset in
planning and prediction. No unmeasured N is supplied. AlphaFold 3 and ESMFold2
receive no fixed-MSA flag, including in mixed-model submissions. `--exact`
disables the wrapper's automatic preset; it does not remove explicit padding
or depth options supplied after `--`. `--exact` and explicit `--fixed-msa` are
mutually exclusive. To reproduce the publisher's exact-shape path, omit any
other padding/depth overrides as well.

## Warm and verify a shared cache

Submit the following once per GPU type as the account that owns the shared
cache, after filling the site variables. For the four fixed-MSA models, add
the same **explicit** `--fixed-msa N` used for prediction; `warm.sbatch` passes
its arguments directly and does not invent or infer N from the environment.

```bash
mkdir -p warm-logs
sbatch --partition="$PARTITIONS_GPU" --gres="$GRES" \
  --cpus-per-task="$CPUS_PER_GPU" --output='warm-logs/warm-%j.out' \
  examples/slurm/warm.sbatch --model boltz2 --fixed-msa "$FIXED_MSA_DEPTH"
```

Omit `--fixed-msa` for AlphaFold 3/ESMFold2 or when no cap has been selected.
Warming uses `--token-buckets auto --msa single` and stops at the first OOM by
default; an explicit `--token-buckets SPEC` or warm policy after the script name
can override those defaults. It may be expensive: the GPU job must be submitted
deliberately with the site's resource/time settings. It is never run by
`foldjax-submit` implicitly.

After that job completes, have an ordinary **Unix user** with read-only access
to the warmed compile namespace submit one fresh verification prediction:

```bash
mkdir -p verification-logs
sbatch --partition="$PARTITIONS_GPU" --gres="$GRES" \
  --cpus-per-task="$CPUS_PER_GPU" --output='verification-logs/verify-%j.out' \
  examples/slurm/warm.sbatch --verify-only -- \
  --model boltz2 --input representative.json --padding \
  --fixed-msa "$FIXED_MSA_DEPTH" --output-dir fresh-verification-output
```

After the verification job finishes:

```bash
foldjax show fresh-verification-output --format csv
```

Use a representative input and matching bucket, MSA cap, weights, static
options, and GPU type from warming. Choose a fresh output directory and do not
add `--resume`, so verification actually executes. Examine the job log and
`foldjax doctor --json` for cache access; successful verification should use
existing entries without attempting to save new shared entries. A Slurm
`--account` selects accounting, **not the Unix user**, and is insufficient to
prove this boundary. These examples never use `su`, `sudo`, or change cache
permissions. `warm.sbatch` reports explicitly that its owner-side warm job has
not performed this separate-user check.

The default compile-cache trust rule accepts this user's or root's protected
namespaces. If the installation deliberately trusts another owner or shared
group writers, configure `FOLDJAX_TRUST_SHARED_COMPILE_CACHE=1` in both submission
environments according to that site's policy; permissions alone do not opt in.
