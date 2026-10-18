# Performance diagnostics

Use a fresh process for every model and size. Keep input features, weights,
sampling settings, precision, padding and kernel choices in each measurement.
Compare ordinary warmed prediction latency separately from synchronized
diagnostics, which add overhead.

## OpenDDE memory and cache loading

```bash
python -m bench.opendde_profile --profile-out /tmp/opendde-768.json -- \
  predict --model opendde --input job.yaml --msa auto --fixed-msa 2048
```

This development tool runs the actual CLI prediction and records feature
preparation, parameter loading, inference, scoring and writing separately. The
report includes boundary device usage, process-lifetime device/RSS high-water
marks, and each executable's compile-or-load time and XLA memory accounting.
It writes the partial report if prediction fails. These are not per-phase
peak samples, and XLA executable accounting excludes CUDA context and module
memory. For operation-level attribution, also retain XLA buffer assignments
with `XLA_FLAGS="--xla_dump_to=/tmp/opendde-hlo --xla_dump_hlo_as_text"`.

The current graph already uses BF16 trunk/confidence, structural attention
blocking and scanned stacks/sampling. Its structural axis reserves twice the
token bucket, and its pair width is 384 rather than Protenix's 128. Therefore
one BF16 structural pair tensor has 12 times the storage of Protenix's pair
tensor at the same token bucket. At a 768-token bucket, it is 1.6875 GiB;
the default FP32 diffusion-boundary counterpart is 3.375 GiB. These are tensor
size calculations, not measured peak memory or evidence for changing dtype.

Shape-complementarity scoring runs after cropping to real cardinalities. Its
small executable can therefore vary with the real input length even when the
main model reuses its padded bucket. Moving that reduction before cropping
would change its reduction shape and needs separate numerical validation.

## Output crops and host staging

Recognized Protenix, OpenFold3, Boltz-2 and OpenDDE export arrays are transferred
to NumPy before unpadding. Final real lengths therefore do not require a
separate device slice executable for these arrays. Internal JAX slices and
unknown output fields keep their existing behavior. A narrowed export owns its
compact host storage, so a small result does not keep the full padded buffer.

JAX 0.11.1's `jit_stage` can come from a NumPy-to-`jnp.asarray` conversion; it
does not necessarily name a model preprocessing function. Direct `device_put`
avoids that identity program for pure host transfers. Existing cache file
counts can include artifacts from older code, so compare new entries from a
fresh diagnostic namespace when attributing an active path.

## ESMFold2 solver scope

```bash
python -m bench.esmfold2_performance_probe \
  --input small.json large.yaml --experiment alignment --output /scratch/esm-align
python -m bench.esmfold2_performance_probe \
  --input small.json large.yaml --experiment esmc --output /scratch/esm-esmc
```

Run this development probe inside a one-GPU allocation. Each experiment runs
baseline, candidate and baseline-repeat in separate processes, using compact LM
staging, the same features and an explicit native diffusion tape. It retains
five sample coordinates and raw confidence values, per-entity residuals after
a single global fit, the repeat floor, timings and allocator memory. The
alignment experiment changes only SVD versus Jacobi; the ESM-C experiment
changes only eager versus compiled blocks. Reports never automatically approve
a new default. `--smoke` reduces the sampling schedule and is excluded from
accuracy admission. Capacity measurements apply to the actual card that ran
the probe.

The experimental [Jacobi rigid-alignment option](esmfold2.md#solver-free-rigid-alignment-experimental)
removes the 3-by-3 SVD/determinant solver calls from diffusion alignment.
Component accuracy and solver-free lowering do not establish a full-model
accuracy result, a speedup, or a new 24 GiB token limit.

ESM-C retains its released eager-block default when deterministic reductions
are disabled. Compiled blocks are a benchmark candidate until actual model
coordinates, scores and warmed timings support changing that default.

Protenix's `tfg/potentials.py` also uses `jnp.linalg.solve`, for a square system
whose size is the number of geometric constraints. Ordinary prediction has
TFG guidance disabled and does not execute that solve. An explicit TFG-guided
run requires separate workspace profiling; this is not a fixed 3-by-3 problem
to which the quaternion alignment solver applies.
