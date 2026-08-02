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
