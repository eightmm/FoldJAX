# bench

A controlled comparison harness, plus explicitly labelled historical point
measurements.

## What it measures

For every model admitted by `bench.drive` that has an upstream runner, it
executes the prediction twice — once through FoldJAX and once through that model's own
upstream repository — and records wall time, peak device memory, and the
confidence the run reported. AlphaFold 3 is an intentional current-harness
FoldJAX-only row, for the reason described below. The checked-in benchmark table
also retains two historical FoldJAX-only ESMFold2 point measurements; ESMFold2
is not admitted by `bench.drive`, and
[`esmfold2_compare.py`](esmfold2_compare.py) is its separate
FoldJAX-vs-upstream harness.

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
  to populate eligible XLA compile-cache entries and that run is discarded. A
  cold JAX run is
  dominated by compilation, which the torch side has no equivalent of, so
  timing one against the other measures the compiler rather than the model —
  at 132 tokens that was 174 s against 29 s, and almost all of the difference
  was compilation. The measured process reuses every readable entry and includes
  any cache rejection or recompilation, which is what a second prediction
  actually costs. Upstream needs
  no equivalent step: Protenix's fused layer-norm extension is the only thing
  it builds on first use, and it is already built in that checkout.

Each measurement is its own process, because a peak is a process-lifetime
high-water mark: two runs in one process report the larger, and the second
one's number is unknowable. The card is allowed to go idle between runs, since
a process exiting does not return its device memory synchronously and the next
job can otherwise fail an allocation that would have fit.

**One quiet reading is not enough**, and the cost of believing it was an
OpenFold3 row at 3,012 tokens that looked like a memory regression. Its
measured run started 13 s after a 26-minute warm-up released 88 GiB, never
reserved its own pool -- three consecutive 15 s samples sat at 615 MiB, the
size of a bare CUDA context -- and died inside `BFCAllocator::Extend`, which is
the grow-on-demand path a process only takes when preallocation did not happen.
`gpu_idle` now wants several consecutive readings below the threshold and says
so when it gives up waiting. Re-run under that gate, the same warm-up-then-
measure sequence completed, and three independent runs of the row agreed to
0.1 MiB.

A failed row keeps its whole stderr in `<row>.stderr.txt` beside the JSON. The
3,000-character tail in the row itself is a summary, not the record: an XLA
allocation failure ends in a stack dump long enough to fill any tail on its
own, so the line naming the requested bytes -- the only part that identifies
the failure -- is exactly what a truncated tail drops. A warm-up that exits
non-zero is reported too, with its output in `<row>.warmup.txt`; it used to be
discarded, which left the measured run silently paying for compilation.

## Memory over time, not just its maximum

A peak is one number off a curve that has a shape: weights loading, a trunk that
climbs and holds, a sampler that spikes once per diffusion step. Two models with
the same peak can spend very different fractions of the run near it, and that is
what decides whether a job fits beside anything else on a card.

`trace.py` records that curve, and it records **the same quantity the table
does** — live bytes, sampled inside the measured process by
`peakhook/benchtrace.py`. Sampling `nvidia-smi` instead would have been wrong
twice over: it is the reserved pool this directory already refuses for peaks,
and it is *monotone over a run*, because neither allocator returns memory to the
driver mid-prediction. The shape a trace exists to show is not in that signal.
Measured on a torch probe that frees two thirds of what it allocated, live bytes
fall 1,536 → 768 MiB where the driver's counter stays at 1,536.

Sampling live bytes also means nothing has to be switched off: the trace is
taken under the same XLA preallocation the shipped command runs.

```bash
python -m bench.drive --models boltz2 protenix --impls foldjax upstream \
    --cases L1000_3og2 --results results/ --work /tmp/bench --traces traces/
python -m bench.trace plot --out ../docs/memory-trace.png \
    --case "1,003 tokens" traces/*.jsonl
```

`--traces` wraps each measurement and changes nothing about it, so a curve and
its table row are always of the same run. The warm-up is deliberately not
traced: it is discarded work, and its curve is compile-bound.

The 4 Hz sampler is an observer, so its largest sample is a *lower bound*; the
allocator's own high-water mark is written as a footer and drawn as the peak
rule in the figure, so the plot cannot understate what a run held.

## The schedule

AlphaFold 3's released inference default — 5 diffusion samples, 200 diffusion
steps, 10 recycles, seed 101 — which Protenix's base model also ships. Boltz-2
defaults to 3 recycles rather than 10, so this asks more of it than its own
default does; it asks the same of its upstream, which is what makes the
comparison mean anything. See `spec.py`.

The historical upstream OpenFold3 cells in `docs/benchmark.md` predate a seed
control fix: 0.3.1's `--num_model_seeds 1` replaced the YAML seed with one
generated from 42. Those rows remain valid time/memory measurements but their
confidence is explicitly marked not seed-matched. Current runs omit that flag
and use the YAML's sole seed, 101. They also pass `--use_templates false`:
these benchmark jobs have no template input, and letting 0.3.1's default remain
true only initialized template/CCD preprocessing that could not add a template.

MSA depth is deliberately not pinned: not every upstream exposes a depth
argument, so fixing one would mean FoldJAX doing something its own reference
cannot. Both sides read the same alignment and apply the same default.

## Every upstream path is disclosed

A comparison against an upstream that is missing its own accelerated paths is
a handicap match unless the limitation is explicit. The benchmark uses the
fastest path each reference can reach on this hardware and records exceptions
rather than presenting them as like-for-like speed claims. Two environments
arrived incomplete, and both differences were large enough to change
conclusions rather than shade them:

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

Boltz-2 needed nothing: it already had cuEquivariance. OpenFold3 is a different
case rather than an incomplete environment: its p1-compatible 0.3.1 kernels do
not support this sm_120 route, so its plain-Torch timing is explicitly qualified
in the OpenFold3 section below.

## Running it

Data (alignments, jobs, sequences) lives outside the repository; point
`FOLDJAX_BENCH_DATA` at it. Upstream repositories default to siblings under the
FoldJAX checkout's parent; set `FOLDJAX_BENCH_ROOT` when they live elsewhere.

```bash
python -m bench.drive --models boltz2 opendde openfold3 protenix \
    --impls foldjax upstream --cases t0128 t0488 t0976 \
    --results results/ --work /tmp/bench --skip-existing
python -m bench.report --results results/
```

Rows are written as they finish, so an interrupted matrix keeps everything that
completed. `--skip-existing` reuses a row only when it has the current result
schema, is successful, has the requested number of samples with finite scores
and valid measurements, and its recorded benchmark identity still matches.
Legacy, failed, malformed, or mismatched rows are run again; mere filename
existence is not a resume proof. The identity and row are checked again before
the driver skips, so an input change during that decision forces a rerun.

An upstream checkout with tracked changes is rejected by default. If a required
compatibility patch has been reviewed, hash the exact tracked diff and pin it on
the command line:

```bash
PATCH_SHA=$(git -C "$FOLDJAX_BENCH_ROOT/protenix" \
    diff --binary --no-ext-diff HEAD -- | sha256sum | cut -d' ' -f1)
python -m bench.drive ... --upstream-patch "protenix=$PATCH_SHA"
```

Every current-schema result hashes the common job and every MSA/template file it
references, the generated native document and its referenced companions when
applicable, and the exact checkpoint and implicit assets selected by the
runner. For AlphaFold 3 this includes the managed runtime's copied package,
native extension, generated CCD tables, and libcifpp dictionaries; transient
Python bytecode caches are excluded. It also binds the execution-affecting
FoldJAX/harness source, Python and
installed runtime versions, device model/compute capability/memory/driver, the
measurement protocol, trace state, and relevant effective environment. Path
values and device selectors are digested rather than written verbatim. These
identities are computed before and after inference; a change invalidates the
run. Upstream rows additionally bind the checkout commit, reviewed tracked-diff
digest and status. Ordinary untracked files are rejected, while ignored
in-tree source, config, executables, and native extensions are byte-hashed and
checked again afterwards. A source edit, native rebuild, or artifact
replacement therefore cannot silently become a reusable current result.

The checked-in historical result JSON files predate this schema and byte-level
artifact provenance. They remain dated measurement evidence used by the report,
not exact source-and-artifact replay capsules, and `--skip-existing` deliberately
does not treat them as resumable current rows.

## Optimization A/B evidence

[`experiments/final-tree-gpu-gate-2026-08-31.json`](experiments/final-tree-gpu-gate-2026-08-31.json)
is the final exact-source confirmation for the three memory-bound route
changes. Every arm starts with its own empty compilation/autotuning cache,
warms that cache in one successful process, then records a fresh measured
process. The file binds all six raw result hashes to one schema-2 source
identity, records the output-structure digests and paired structure/score
checks, and distinguishes whole-process peaks from OpenDDE's isolated
projection temporary.

[`experiments/gap-ablation-2026-08-31.json`](experiments/gap-ablation-2026-08-31.json)
contains the broader and longer sweeps, including AlphaFold 3 at 970 tokens and
20 samples. Those older rows retain the provenance limitation documented in
that file; the final-tree gate is the current exact-source release
confirmation.

`alphafold3` has a FoldJAX column but no upstream one: FoldJAX drives the
installation you provide rather than reimplementing the model, so both columns
would be the same code.

## The ablation tools

Three scripts here answer questions the main table cannot, and none of them is
reachable from a row in it -- which is why they read as dead code until you go
looking. They are not; each was written for a question that recurs.

- [`schedule_ablation.py`](schedule_ablation.py) splits a run's wall time across
  trunk, diffusion and confidence. Protenix and OpenDDE trace a whole prediction
  as one XLA program, so wrapping Python functions measures nothing; varying
  recycles, steps and samples one at a time and reading the slopes gives the
  split without a profiler. This is the only stage-level time attribution in the
  repository -- the manifest's `cost.phases` stops at prepare/predict/write.
- [`triangle_block_sweep.py`](triangle_block_sweep.py) times and sizes one
  triangle multiplication against its row-block size. Block budgets are not
  monotonic in the peak, so the answer has to be swept rather than reasoned to.
- [`run_gap_ablation.py`](run_gap_ablation.py) drives one named arm of a route
  A/B in its own process, which is what the `experiments/` records above are
  built from.

## A failed row is a result, not a gap

`opendde` at 1,531 tokens does not run in fp32 on this card -- on either side.
FoldJAX asks for a 66.84 GiB program arena (92.21 with the fused kernels) and
upstream reaches 92,960 MiB before asking for 12.69 GiB more. Both rows carry
`"failed": true`, and upstream's carries `"reason": "exited 0 but produced no
structures"` -- its runner swallows the allocator error and exits zero, so the
row exists only because the harness checks for structures rather than for an
exit code.

The checked-in historical `opendde-foldjax-t1531-bf16.json` row records a bf16
configuration that completed: 479.9 s at 58,747 MiB. It predates the current
default-bf16 measurement, 443 s at 44.2 GiB, reported in
[`docs/benchmark.md`](../docs/benchmark.md). The historical row is stored under
its own name rather than as the `t1531` row, because it is not the same
measurement -- upstream has no bf16 counterpart here, so that number belongs
beside a stated dtype and not in a column that implies one.

## OpenFold3's upstream route

OpenFold3 is in both `MODELS` and `REIMPLEMENTED`. `run_upstream.py`
drives the provisioned `openfold3-v031` environment rather than the current
upstream checkout: the `of3_ft3_v1.pt` p1 checkpoint implemented by FoldJAX has
`version_compatibility="<0.4"`, and current OpenFold3 rejects that architecture.

The command-line flag sets five diffusion samples. The other controlled
settings live in `openfold3_runner.yml`: it selects the
prediction and PAE presets, overrides the upstream default from 3 to 10
recycles, supplies the sole model seed, 101, and disables confidence-head host
offload; the latter changed the 3,012-token historical wall time by about one
minute and raised peak by about 2.2 GiB while remaining within capacity. The
harness deliberately does
not pass 0.3.1's `--num_model_seeds`: that flag discards the YAML seed list and
generates a new one from 42. The release already supplies 200 rollout steps.
This is the current harness configuration. Historical upstream cells in
`docs/benchmark.md` used the generated seed described above and retain their
explicit dagger rather than being relabeled as current runs.

Its timing needs one hardware caveat. OpenFold3 0.3.1's DS4Sci attention does
not build for sm_120, and its experimental cuEquivariance path fails in the
multi-sample confidence stack, so the recorded upstream runs use plain Torch
attention. That makes the comparison real but not a claim about upstream's
fastest path on supported older hardware. `upstream-environments.md` records
the environment, failures, and verification commands.
