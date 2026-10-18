# Performance diagnostics

Use a fresh process for every model and size. Keep input features, weights,
sampling settings, precision, padding and kernel choices in each measurement.
Compare ordinary warmed prediction latency separately from synchronized
diagnostics, which add overhead.

## Tokamax candidate cache scope

Candidate benchmarking bypasses persistent compilation caching while model
executables and selected tuning results still persist. A two-process GPU probe
with a real BF16 Triton GLU measured two candidates using the parallel autotuner:
two `jit_forward` compilations produced zero persistent candidate entries, while
one model executable and one tuning result were saved. A fresh process then
reported `sources={"store": 1}`, performed zero backend compilations and returned
the same output; the model call took 0.045 seconds. This bounded wallclock probe
checks the cache contract, not full AF3 prediction latency or tuning quality.

## OpenDDE memory and cache loading

```bash
python -m bench.opendde_profile --profile-out /tmp/opendde-768.json -- \
  predict --model opendde --input job.yaml --msa auto --fixed-msa 2048
```

This development tool runs the actual CLI prediction and records feature
preparation, parameter loading, inference, scoring and writing separately. The
report includes boundary device usage, process-lifetime device/RSS high-water
marks, and each executable's compile-or-load time and XLA memory accounting.
The native CLI applies allocator/runtime settings before profiling imports the
model, so `--mem-fraction` and caller allocator environment values still apply.
It writes the partial report if prediction fails. These are not per-phase
peak samples, and XLA executable accounting excludes CUDA context and module
memory. For operation-level attribution, also retain XLA buffer assignments
with `XLA_FLAGS="--xla_dump_to=/tmp/opendde-hlo --xla_dump_hlo_as_text"`.

The current graph already uses BF16 trunk/confidence and scanned stacks/sampling.
The XLA attention path blocks outer rows; the fused cuEquivariance path still
projects full structural Q/K/V arrays before its attention call. Its structural
axis reserves twice the token bucket, and its pair width is 384 rather than
Protenix's 128. Therefore one BF16 structural pair tensor has 12 times the
storage of Protenix's pair
tensor at the same token bucket. At a 768-token bucket, it is 1.6875 GiB;
the default FP32 diffusion-boundary counterpart is 3.375 GiB. These are tensor
size calculations, not measured peak memory or evidence for changing dtype.

Shape-complementarity scoring runs after cropping to real cardinalities. Its
small executable can therefore vary with the real input length even when the
main model reuses its padded bucket. Moving that reduction before cropping
would change its reduction shape and needs separate numerical validation.

An October 2026 diagnostic on a 96 GB Blackwell GPU, using the native 0.9 pool,
single-sequence MSA, fixed MSA 2048 and five samples, measured:

| Token bucket (real tokens) | Structural capacity | Allocator peak | Main executable load | Auxiliary score load |
| --- | --- | --- | --- | --- |
| 256 (254) | 512 | 3.44 GiB | 1.67 s | 0.06 s |
| 512 (499) | 1024 | 6.57 GiB | 1.74 s | 0.09 s |
| 1024 (1003) | 2048 | 21.17 GiB | 1.93 s | 0.14 s |

All three fresh-process cache runs reused shared entries without compiling.
At the largest bucket, XLA buffer assignments place six 3 GiB BF16 Q/K/V,
layout and pair buffers together at the attention peak. FP32 promotion was
not the cause of that measured peak. Main executable loading, rather than
shape-complementarity loading, accounts for most of the observed cache-load
delay; these timings combine cache I/O and backend executable loading.
Projection-level outer-row chunking for the fused path needs its own numerical
and performance comparison before adoption. These measurements do not establish
a 24 GB card limit or ordinary prediction latency.

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

The October 2026 paired probe on a 96 GB Blackwell GPU used 254- and 499-token
inputs, five samples, native recycling/diffusion settings and the same actual
diffusion tape. It retained `num_recycles=3` and configured `num_steps=14`
(ten denoiser calls after native schedule clipping), reducing the checkpoint's
32 samples to five. Other stochastic branches used the same keys
and consumers; their individual draws were not independently captured.
Jacobi's warmed structure times were 0.922/3.933 seconds versus
SVD's 0.921/3.904 seconds. Maximum entity RMSDs against SVD were 0.0887/0.0627 Å;
SVD repeats differed by 0.0754/0.0571 Å. Both candidate and repeat comparisons
exceeded the fixed 0.05 Å diagnostic. This does not support changing the default:
SVD remains the default and Jacobi remains experimental. The lifetime allocator
peak was unchanged at these sizes; external CUDA workspace and 24 GB capacity
remain unmeasured.

ESM-C retains its released eager-block default when deterministic reductions
are disabled. In the same two-size probe, compiled LM blocks reduced warmed LM
time from 1.516/1.413 seconds to 0.0457/0.0618 seconds. This is a 23–33 times
improvement in the LM phase, not the full prediction. Candidate maximum entity
RMSDs were 0.0807/0.0683 Å versus eager repeats' 0.0728/0.0423 Å. At 499 tokens
the repeat passed the 0.05 Å diagnostic while the candidate failed it. The LM
embedding also changed, so the speed gain does not establish numerical
equivalence. Compiled blocks remain a benchmark candidate.

Protenix's `tfg/potentials.py` also uses `jnp.linalg.solve`, for a square system
whose size is the number of geometric constraints. Ordinary prediction has
TFG guidance disabled and does not execute that solve. An explicit TFG-guided
run requires separate workspace profiling; this is not a fixed 3-by-3 problem
to which the quaternion alignment solver applies.
