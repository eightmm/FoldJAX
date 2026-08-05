# bench

One comparison, run under one schedule, on both implementations of each model.

## What it measures

For every (model, size) it runs the prediction twice — once through FoldJAX and
once through that model's own upstream repository — and records wall time, peak
device memory, and the confidence the run reported.

Two things make it a controlled comparison rather than two runs that happen to
resemble each other:

- **The same job.** The upstream side's input is produced by FoldJAX's own
  translator (`materialize_native_input`), which emits each upstream's native
  dialect. Both sides read the same job naming the same alignment file, rather
  than two hand-written inputs believed to match.
- **The same measurement.** Peak memory is the live-bytes high-water mark on
  both sides: `peak_bytes_in_use` under JAX, `max_memory_allocated` under
  torch. `nvidia-smi` is not used — it reports the caching allocator's reserved
  pool, which grows by doubling and so tracks the allocator's schedule rather
  than the model's need. The torch side is read by `peakhook/sitecustomize.py`,
  injected on `PYTHONPATH`, so no upstream file is modified.

- **The same starting state.** FoldJAX is measured warm: the case is run once
  to fill the XLA compile cache and that run is discarded. A cold JAX run is
  dominated by compilation, which the torch side has no equivalent of, so
  timing one against the other measures the compiler rather than the model —
  at 132 tokens that was 174 s against 29 s, and almost all of the difference
  was compilation. Compilation is paid once per shape and replayed from disk
  afterwards, which is what a second prediction actually costs. Upstream needs
  no equivalent step: Protenix's fused layer-norm extension is the only thing
  it builds on first use, and it is already built in that checkout.

Each measurement is its own process, because a peak is a process-lifetime
high-water mark: two runs in one process report the larger, and the second
one's number is unknowable. The card is allowed to go idle between runs, since
a process exiting does not return its device memory synchronously and the next
job can otherwise fail an allocation that would have fit.

## The schedule

AlphaFold 3's released inference default — 5 diffusion samples, 200 diffusion
steps, 10 recycles, seed 101 — which Protenix's base model also ships. Chai and
Boltz-2 default to 3 recycles rather than 10, so this asks more of them than
their own default does; it asks the same of their upstream, which is what makes
the comparison mean anything. See `spec.py`.

MSA depth is deliberately not pinned: upstream Chai has no depth argument, so
fixing one would mean FoldJAX doing something its own reference cannot. Both
sides read the same alignment and apply the same default.

## Every upstream runs on its own fast path

A comparison against an upstream that is missing its own accelerated paths is
not a comparison, it is a handicap match. Two of the four arrived incomplete,
and both differences were large enough to change conclusions rather than shade
them:

- OpenDDE had no `cuequivariance`, so its `auto` triangle kernels fell back to
  plain torch. Installing the cu13 build took its 490-token peak from 91,191
  MiB to 18,580 and its time from 117 s to 71 s, and made 970 tokens run at all
  where it had been an out-of-memory failure that the runner reported as
  success.
- Protenix's fused layer norm is a CUDA extension built on first use, and this
  host has the CUDA runtime but no compiler. Completing the toolkit inside its
  own virtualenv — and adding sm_120 to an architecture list that stopped at
  compute_100 — took its 970-token time from 235 s to 94 s.

[`upstream-environments.md`](upstream-environments.md) has the exact commands
and, more importantly, how to verify each one actually took effect. A kernel
built for the wrong architecture fails *asynchronously*, so the call returns
normally and the error surfaces somewhere else entirely.

Boltz-2 and Chai needed nothing: Boltz-2 already had cuEquivariance, and Chai
does not use it.

## Running it

Data (alignments, jobs, sequences) lives outside the repository; point
`FOLDJAX_BENCH_DATA` at it.

```bash
python -m bench.drive --models boltz2 chai opendde protenix \
    --impls foldjax upstream --cases t0128 t0488 t0976 \
    --results results/ --work /tmp/bench --skip-existing
python -m bench.report --results results/
```

Rows are written as they finish, so an interrupted matrix keeps everything that
completed, and `--skip-existing` resumes it.

`alphafold3` has a FoldJAX column but no upstream one: FoldJAX drives the
installation you provide rather than reimplementing the model, so both columns
would be the same code.

## A failed row is a result, not a gap

`opendde` at 1,531 tokens does not run in fp32 on this card -- on either side.
FoldJAX asks for a 66.84 GiB program arena (92.21 with the fused kernels) and
upstream reaches 92,960 MiB before asking for 12.69 GiB more. Both rows carry
`"failed": true`, and upstream's carries `"reason": "exited 0 but produced no
structures"` -- its runner swallows the allocator error and exits zero, so the
row exists only because the harness checks for structures rather than for an
exit code.

`opendde-foldjax-t1531-bf16.json` is the configuration that does complete:
479.9 s at 58,747 MiB. It is stored under its own name rather than as the
`t1531` row, because it is not the same measurement -- upstream has no bf16
counterpart here, so that number belongs beside a stated dtype and not in a
column that implies one.
