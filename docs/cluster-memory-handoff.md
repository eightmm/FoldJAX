# Cluster memory investigation: 4090, OpenDDE and ESMFold2

This is a measurement handoff, not a claim that a new capacity has passed.
Run these commands inside your existing one-GPU allocation. Nothing here
submits Slurm work. Use a new process and a new output directory for each row;
keep the shared installation and warmed caches intact.

## Common controls and returned artifacts

Use the actual jobs that reproduced the failures, retaining their alignments,
templates, ligands, weights, seed, sample count, steps and recycles. A synthetic
protein or fewer samples can have a different peak. Record both actual and
padded token/atom/MSA/structural shapes. Freeze MSA results before comparing:
`--msa auto` can search on a miss; a job with explicit local alignments and
`--msa none` avoids that network and preprocessing change.

Choose a writable node-local `FJ_DIAG_ROOT` for reports and logs. Save:

- `git rev-parse HEAD`, the complete command/environment, device kind, driver,
  compute capability, ECC state and usable total/free memory before launch;
- full stdout/stderr and return status, including failed rows;
- the diagnostic JSON and successful `foldjax_run.json`, especially shapes,
  `resolved_chunking`, compilation cache tiers and `kernel_tuning`;
- XLA dumps when available, and a separate total-GPU-memory trace with its
  sampling interval. NVIDIA's total/reserved usage is not allocator live bytes;
- cold compile/load and a fresh-process warm run separately. Do not time a
  `--resume` run as prediction.

`peak_bytes_in_use` is a process-lifetime allocator peak. It excludes CUDA
context/modules and external workspaces. Neither a post-failure free-memory
reading nor total usage minus allocator live bytes proves a workspace size:
that difference also includes the allocator's reserved, unused pool. External
sampling can miss short peaks. Keep all three sources rather than substituting
one for another.

## A. AF3 2048 and Boltz-2 1536 on 3090 versus 4090

Use these matched rows before changing any kernel options:

| Model / original input | Card | Target | Allocator | Execution |
| --- | --- | --- | --- | --- |
| AF3 failing 2048 job | 3090, 4090 | 2048 | existing default | ordinary, then diagnostic |
| Boltz-2 failing 1536 job | 3090, 4090 | 1536 | existing default | ordinary, then diagnostic |
| Same Boltz-2 job | 4090 | 1536 | explicit 0.85 | ordinary, then launch-blocking |
| Same Boltz-2 job | 4090 | 1536 | existing default | launch-blocking |

First reproduce with the unchanged `foldjax predict` command and save its full
log. Then replace only the Python entry point with the development profiler:

```bash
: "${FJ_DIAG_ROOT:?set a writable results directory}"
mkdir -p "$FJ_DIAG_ROOT/af3-2048"
python -m bench.prediction_profile \
  --profile-out "$FJ_DIAG_ROOT/af3-2048/profile.json" -- \
  predict --model alphafold3 --input af3-failing-job.yaml \
  --msa auto --padding --pad-tokens 2048 \
  --output-dir "$FJ_DIAG_ROOT/af3-2048/output" \
  >"$FJ_DIAG_ROOT/af3-2048/run.log" 2>&1
```

Append the original seed/sample/step/recycle and other options to every arm;
the example deliberately does not replace those with a smaller schedule.
The profiler applies native allocator settings before importing JAX/models,
records selected-device identity, effective fraction and each executable's
argument/output/temporary bytes, and observes actual cache hits/backend
compilation. It does not change kernel selection or enable launch blocking.
It keeps partial reports on ordinary exceptions; a killed process cannot
flush one. A report-write failure must not replace the original failure.

```bash
mkdir -p "$FJ_DIAG_ROOT/boltz-1536-blocking"
CUDA_LAUNCH_BLOCKING=1 python -m bench.prediction_profile \
  --profile-out "$FJ_DIAG_ROOT/boltz-1536-blocking/profile.json" -- \
  predict --model boltz2 --input boltz-failing-job.yaml \
  --msa auto --fixed-msa 2048 --pad-tokens 1536 --mem-fraction 0.85 \
  --output-dir "$FJ_DIAG_ROOT/boltz-1536-blocking/output" \
  >"$FJ_DIAG_ROOT/boltz-1536-blocking/run.log" 2>&1
```

Also run the blocking arm without `--mem-fraction 0.85`. Blocking changes
execution timing and overlap; it is not a performance measurement. An earlier
illegal-address/assert error appearing under blocking supports an asynchronous
fault. An OOM at the same launch with no earlier fault is consistent with a
launch/module resource failure, but does not alone prove an external allocation
or distinguish all driver/kernel causes. Retain the first error, not only the
last XLA exception. If necessary, use Compute Sanitizer as a separate, slower
follow-up on the smallest failing case.

For AF3, obtain an XLA buffer assignment for the failing module as well:

```bash
mkdir -p "$FJ_DIAG_ROOT/af3-2048-dump/dump"
XLA_FLAGS="${XLA_FLAGS:+$XLA_FLAGS }--xla_dump_to=$FJ_DIAG_ROOT/af3-2048-dump/dump --xla_dump_hlo_as_text" \
  python -m bench.prediction_profile \
  --profile-out "$FJ_DIAG_ROOT/af3-2048-dump/profile.json" -- \
  predict --model alphafold3 --input af3-failing-job.yaml \
  --msa auto --padding --pad-tokens 2048 \
  --output-dir "$FJ_DIAG_ROOT/af3-2048-dump/output" \
  >"$FJ_DIAG_ROOT/af3-2048-dump/run.log" 2>&1
python -m bench.kernel_shootout.attrib \
  --dump "$FJ_DIAG_ROOT/af3-2048-dump/dump" \
  --json-out "$FJ_DIAG_ROOT/af3-2048-dump/buffers.json"
```

Retain existing XLA flags, avoid duplicate dump flags, and use result paths
without spaces in the flag value. A persistent cache hit need not emit a buffer
assignment. First return its executable memory accounting; an empty dump is
unknown, not zero memory. If a cold dump is needed, work on a private copy of
the resolved namespace and remove model executable entries only from that
copy. Keep the matching Tokamax tuning store/configuration unchanged. Do not
delete or alter the shared cache. A wholly empty `--cache-dir` also retunes
Tokamax and is a different kernel-selection experiment; label it separately.

The analyzer selects the module with the largest temporary arena. Compare its
name with the failing model module in the log/profile; also return the raw
dumps, including other modules. It reports allocation sizes, constituent values
and offsets, live peak, shape/dtype and producing HLO operations. A 14.35 GiB
allocation can be an entire temporary arena with reused storage. It must not
be called one tensor, or the sum of all its assigned values called live memory.
Missing live-range information is reported as unavailable.

Compare the 3090/4090 optimized HLO producers, custom-call targets, temporary
arena and largest simultaneously live buffers. Keep Tokamax source/call/device
identities and cache source with the results; the same input alone does not
prove the same selected kernels. These observations decide whether a 4090-only
kernel/workspace candidate is warranted. No 3090 profile or kernel default is
changed by this diagnostic patch.

## B. OpenDDE projection-row chunking candidate

The measured default peak contains whole structural Q/K/V and layout buffers.
The new experimental option is:

```bash
foldjax predict --model opendde --input job.yaml --msa auto --fixed-msa 2048 \
  --option structural_triangle_att_row_chunk_size=32 --output-dir candidate
```

This splits outer rows in structural-refiner cuEq/Tokamax attention, including
projection, gates and output projection. The column softmax, full triangle bias,
masks and dtypes are preserved. It does not change the residue trunk,
confidence stack or structural capacity. Context parallelism and an XLA
attention backend with this candidate are rejected. Actual fused GPU support,
rounding, memory improvement and speed still need measurement.

Start with default versus widths **64, 32, 16** at **256 and 768** tokens on
4090. Use the same two real jobs across all arms, including the existing
1TSR case at 768. If their memory/accuracy results support continuing, test:

| Cards | Padded targets | Purpose |
| --- | --- | --- |
| 3090 / A5000 / 4090 | 768, 896, 1024, 1280, 1536 | old boundary, target and next rung |
| A6000 | 1024, 1280, 1536, 1792, 2048 | old boundary through target |

Use a real job near each target, with representative structural-token/atom
content, rather than padding one tiny protein and claiming a general limit.
For each baseline/candidate row, keep the same job, sampling and all other
options. Run the stage observer separately from ordinary timing:

```bash
python -m bench.opendde_profile --profile-out "$FJ_DIAG_ROOT/opendde-1280.json" -- \
  predict --model opendde --input jobs/1280.yaml --msa auto \
  --fixed-msa 2048 --pad-tokens 1280 \
  --option structural_triangle_att_row_chunk_size=32 \
  --output-dir "$FJ_DIAG_ROOT/opendde-1280-output"
```

Remove the option for the baseline. Keep native steps/recycles, five paired
sample indices for numerical comparisons, and separately check the site's
normal sample count for capacity. Preserve baseline repeats and raw confidence,
audit atom/entity matching and compare each entity after one global proper fit.
Reference lDDT is a separate result. Ordinary same-seed CLI arms are useful
screening but do not prove equality of actual stochastic draws: promotion to a
default still requires the project's full consumed-tape/input audit and frozen
accuracy gates at two sizes. No promotion is part of this patch.

If admission refuses an off-profile capacity trial, use `--memory-check warn`
in **both** arms and record the override. Do not change dtype or the existing
cost law to make a candidate look admitted. The targets of 1280 on 24 GB and
2048 on A6000 remain unverified until these rows pass.

Cache-load records now split inclusive loading into layered filesystem/trust
reading, decompression and a residual including executable deserialization,
CUDA module loading and adapter overhead. The residual is not a measured
CUDA-only interval. Compare the main executable against auxiliary score loads;
do not infer the cause from the count of small score files. For an NFS/local
comparison, reuse an identical namespace copied to a trusted node-local cache
root, keep the tuning store intact, and check both arms actually hit their
entries. OS cache state and copy time are separate from deserialization.

**OpenDDE rewarm is required only when opting into a row-chunk width.** Warm
and predict with the same option in its separate candidate namespace. Existing
default model namespaces are retained; no default rewarm is requested.

## C. ESMFold2 SVD versus Jacobi

The [alignment diagnostic](esmfold2-alignment-diagnostics.md) gives the actual
failed full-model criterion, sweep/precision/conditioning results and runnable
1024/1280/1536/1792/2048 capacity plus five-sample paired commands. Ten sweeps
already reached the FP32 residual floor on the synthetic panel; increasing the
count is not an established fix. Actual covariances from differing full-model
samples are still missing. SVD remains the default.

Run padded serving capacity separately from the native unpadded fixed-tape
core comparison; they are different graphs. On each real card compare both
solvers under the same allocator fraction, native schedule and sample count.
Report LM and structure peaks separately where possible. If the solver-free
arm still fails at the same LM/trunk peak, obtain that module's buffer
assignment before proposing chunking. A solver-free lowering alone does not
establish that cuSolver caused the capacity limit.

## D. Host staging

Pure host-to-device transfers in AF3's managed padded/representation feature
boundary, Boltz's explicitly placed feature boundary, and OpenDDE's summary
score boundary now use `device_put`. CPU tests preserve values, dtype, weak
type, selected device and the downstream lowering. This removes the redundant
identity `stage` program at those boundaries, not every stage/cast or internal
array operation. Old cache files remain on disk: count **new** `jit_stage`
entries during fresh-process runs and keep real model executables/cache keys
separate from these transfer artifacts.
