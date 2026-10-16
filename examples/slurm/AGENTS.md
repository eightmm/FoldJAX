# Running FoldJAX for a user on Slurm

Apply the repository `PROJECT.md` contract and the site's filled configuration.
No partition, account, queue limit, MSA depth, or timing measurement is assumed
by this template. Read the site's own instructions before submitting jobs.

- Never run prediction, cache warming, or a substantial local MSA search on a
  login node. Use the batch scripts and a deliberate resource/time request.
- Tell the user before `--msa auto` sends their sequences to an external search
  server. An approved local command/server may be configured instead.
- Run `foldjax plan` with the prediction's model/input/options before submitting.
  `foldjax-submit` does this for real submissions; `--dry-run` only prints.
- Use `--resume --keep-going`, consistent output paths, and job-specific logs.
  Array shards split inputs; they do not change prediction parameters.
- Select fixed MSA depth only from the user's explicit choice or a documented
  site measurement. No depth is a default until that evidence exists.
- Read completed results with `foldjax show OUTPUT_DIR --format csv`. Wait for
  scheduler completion before treating a missing manifest as a failed model.
- Verify a warmed shared cache from an ordinary Unix user. Do not change Unix
  identity or repair another owner's permissions without authorization.

| Symptom | Next action |
| --- | --- |
| MSA job failed; GPU dependency invalid | Read the CPU job log and repair search/cache access; resubmit the wrapper. |
| Permission denied in weights/runtime/MSA/cache | Run `foldjax doctor --json`; have the store owner inspect `foldjax setup --shared` before an authorized `--apply`. |
| CUDA module loading/autotuning OOM | Lower `XLA_CLIENT_MEM_FRACTION` (for example 0.85), then restart in a new batch process. |
| Pool OOM | Read the reported pool/usage; use `foldjax plan` and measured model-specific levers. Do not infer the peak from the pool size. |
| Unsupported GPU architecture | Select an NVIDIA sm_80+ GPU type. |
| Long first run | Check compilation/autotuning progress and scheduler limits; compare a matching warmed run before judging steady-state time. |
| Some batch inputs failed | Read failures with `foldjax show ... --format csv`; rerun the same wrapper command with matching output/options. |
