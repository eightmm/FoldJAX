# GPU memory follow-up after e29548cf

The [2026-10-20 follow-up](cluster-memory-2026-10-20.md) supersedes the pending
Boltz-2 fraction control and adds the OpenDDE padding results and corrected
interpretation of ESMFold2's cross-process LM differences. Earlier conditions
below are retained as the history of those measurements.

The next useful artifact is one cold dump of the original failing 1024-token
profile. Do this before another row-chunk sweep or a precision change. No GPU
jobs were submitted for this follow-up, and no new kernel or dtype default is
proposed.

## What the current failure establishes

The two 6.02 GiB BFC messages explicitly say that the caller does not regard
them as failures. They are compilation-time warnings, not the final OOM.
The actual failed main executable requests a **20,847,646,976-byte temporary
arena**, approximately **19.42 GiB** or **20.85 decimal GB**. Keep the raw byte
count alongside either unit; GB and GiB are not interchangeable.

That arena is not one tensor. XLA assigns many values to it, often reusing the
same offsets at different times. Its constituent values cannot be added up as
simultaneously live memory. The reported rematerialization reduction from
25.41 to 22.42 GiB also remained above its 18.34 GiB target; those are compiler
estimates, not separate observed process-memory peaks.

The failing job's reported dimensions are:

| Axis | Actual | Padded |
| --- | ---: | ---: |
| Residue tokens | 952 | 1024 |
| Atoms | 7584 | 24576 |
| MSA rows | 722 | 2048 |
| Structural tokens | 1816 | 2048 |

On the reported 768-token 1TSR run, row width 32 reduced the main temporary
accounting from 12.61 to 12.29 in the report's displayed units, approximately
2.5%; widths 64, 32 and 16 did not improve latency. This does not support
promoting the row-chunk candidate. Confirm the original raw byte counts before
combining those displayed values with GiB measurements.

Source-level hypotheses remain: structural triangle multiplication and its
layout/accumulator liveness; the default FP32 structural-to-diffusion pair
boundary and conditioning intermediates; atom/diffusion caches; or scheduling
and buffer-reuse constraints. A 2048-by-2048-by-384 FP32 tensor would occupy
6 GiB, but matching that size to an earlier warning does not identify an
allocation or establish that it sets the peak. The buffer assignment and live
ranges must distinguish these hypotheses on the actual 4090 executable.

## One original-profile cold dump

Use the same actual job, existing MSA, weights, device, allocator fraction,
seed, sample count, steps, recycles and kernel options as the failed run. In
particular, keep its existing `--memory-check warn`; do not replace the actual
sample count with a diagnostic default or reduce the MSA/sampling schedule.
Freeze the existing alignment inputs so a search miss cannot change the arm.

In a **fresh process**, `JAX_ENABLE_COMPILATION_CACHE=false` bypasses persistent
JAX executable reads and writes even when FoldJAX selects its normal cache
namespace. It forces a backend compilation that can emit a dump, without
deleting, copying or redirecting the shared cache. It also recompiles auxiliary
JITs, so its elapsed time is not ordinary warmed prediction latency. Set this
environment value before Python starts; switching it after JAX initialization
is not this procedure.

Choose a new result directory whose path has no whitespace. Preserve existing
XLA flags, but resolve any existing `--xla_dump_to` or
`--xla_dump_hlo_as_text` setting before adding these flags. The `original`
array below stands for the **complete original predict arguments**, beginning
with `predict --model opendde --input ...`; copy the actual invocation and
change only the output location appended here. Retain its original
`--cache-dir` root if one was supplied: do not pass an already resolved
`model/weights/digest` namespace as a new cache root.

```bash
: "${FJ_DIAG_ROOT:?set a writable results directory}"
# Define original=(predict --model opendde --input ... <all original options>).
# Keep the original seed, sample count, MSA, sampling schedule and allocator.
case_dir="$FJ_DIAG_ROOT/opendde-1024-default-cold"
mkdir -p "$FJ_DIAG_ROOT"
mkdir "$case_dir"
mkdir "$case_dir/dump"
rc=0
JAX_ENABLE_COMPILATION_CACHE=false \
XLA_FLAGS="${XLA_FLAGS:+$XLA_FLAGS }--xla_dump_to=$case_dir/dump --xla_dump_hlo_as_text" \
  python -m bench.opendde_profile --profile-out "$case_dir/profile.json" -- \
  "${original[@]}" --output-dir "$case_dir/output" \
  >"$case_dir/run.log" 2>&1 || rc=$?
printf '%s\n' "$rc" >"$case_dir/status.txt"
python -m bench.kernel_shootout.attrib --dump "$case_dir/dump" \
  --json-out "$case_dir/buffers.json" \
  >"$case_dir/buffers.txt" 2>&1
```

Check `profile.json` for the effective allocator fraction, selected device,
full prediction arguments, cache namespace locations and backend compilation
of `jit__opendde_infer_graph` (use the module's actual spelling in the report).
Persistent cache reads should be absent for this cold process. A cache hit
without a buffer assignment is not a successful cold dump. The diagnostic now
retains cache/tuning/device/environment metadata on ordinary prediction
exceptions too; an externally killed process may not flush its report.

Retain the full log and every dump file, especially the main module's optimized
HLO, `*-buffer-assignment.txt`, `*-live-range.txt` and
`*-memory-usage-report.txt`. If compilation fails before producing some files,
retain the partial directory and say which artifacts are missing. Keep any
emitted PTX/cubin files too when investigating a separate kernel-launch failure;
they are optional for this arena diagnosis. The analyzer chooses the largest
temporary arena, so verify its module name against the main executable rather
than assuming every dump belongs to that graph.

### Separate Tokamax store control

JAX's executable-cache switch does not disable FoldJAX's separate Tokamax JSON
tuning store. For an AF3 diagnostic using this procedure, retain the same
namespace, input and tuning configuration and require
`kernel_tuning.sources == {"store": 1}` for the one-call case. A measured or
heuristic fallback is a different selected-kernel experiment. OpenDDE's default
triangle path uses cuEquivariance; it need not report a Tokamax store hit.
The CPU cache-switch regression verifies backend compilation, zero persistent
cache I/O and unchanged tuning-store bytes; it does not establish a real AF3
store hit on a GPU.

## Then isolate atom and structural padding, if needed

After inspecting the original dump, these existing pins distinguish padding
axes without changing dtype or removing real input data:

| Arm | `--pad-atoms` | `--pad-structural-tokens` |
| --- | ---: | ---: |
| Original capacities, explicit | 24576 | 2048 |
| Atom capacity only | 8192 | 2048 |
| Structural capacity only | 24576 | 1856 |
| Both capacities | 8192 | 1856 |

For this reported job, 8192 exceeds 7584 real atoms and is divisible by the
32-atom window; 1856 exceeds 1816 structural tokens and is divisible by 8 and
64. Serial explicit targets are accepted when they hold the actual/storage
extent. Recheck those extents on the exact featurized input: a pin below them
must fail rather than truncate. This is not a universal capacity recommendation
and does not cover a context-parallel mesh.

Keep tokens at 1024 and MSA at 2048 in every arm, retain the original sample
count and all other controls, and use a new process/output directory per arm:

```bash
for atoms in 24576 8192; do
  for structural in 2048 1856; do
    case_dir="$FJ_DIAG_ROOT/atoms-$atoms-structural-$structural"
    mkdir "$case_dir"
    rc=0
    python -m bench.opendde_profile --profile-out "$case_dir/profile.json" -- \
      "${original[@]}" --pad-atoms "$atoms" \
      --pad-structural-tokens "$structural" \
      --output-dir "$case_dir/output" \
      >"$case_dir/run.log" 2>&1 || rc=$?
    printf '%s\n' "$rc" >"$case_dir/status.txt"
  done
done
```

These are separate compile profiles. Retain the real-prefix RNG controls and
audit the actual draws; different concrete dimensions can change fusion and
rounding despite unchanged real input and dtype. Check structures/confidence
and a baseline repeat rather than asserting bit-identical outputs. Dump a
particular arm using the cold procedure only
when its result helps distinguish the leading hypothesis. Do not combine a
row-chunk option or BF16 diffusion change into this two-axis comparison.

## Distinguish memory attribution from runtime hotspots

The analyzer reports arena sizes, reused offsets, largest simultaneous live
values, shapes/dtypes and producing HLO operations. Its opcode census counts
static instructions; it is not kernel time or dynamic loop invocation count.
Scanned recycling/diffusion bodies may execute many times despite appearing
once. Use the phase report to separate compile/load from inference/scoring,
then obtain a separate warmed CUDA activity trace with the site's existing
profiler if inference still dominates. Compare kernel self-time and call count
for triangle/GEMM/attention kernels, copies and launch gaps. A synchronized
phase timer or a count of auxiliary executable files cannot establish the
cause of the reported 143 seconds. Keep that timing investigation separate
from the cold dump and from ordinary latency measurements.

## ESMFold2: buffered state capacity and accuracy

The reported 4090 GFP×5 input (1190 real tokens, padded to 1280) failed in
`jit_stack` with both SVD and Jacobi before reaching alignment. The allocator
was at its limit and the full LM state input/output coexisted. SVD remains
the default; the solver is not the capacity lever for that observed failure.

The new `--option lm_state_storage=buffer` gathers and masks each layer into
one donated `[batch, tokens, layers, width]` destination. It keeps all layers,
the original stack dtype promotion and the final layer-combination program;
it does not stream a differently ordered floating-point reduction, lower the
storage dtype or enable compiled ESMC blocks. CPU tests preserve every layer
and compact-embedding bits and confirm destination aliasing. GPU memory and
full-model accuracy are still to be measured; the default remains `stack`.

Start with **the same GFP×5/1280 and GFP×6/1536 jobs that failed**, then 1792
and 2048 on A6000. On 24 GB cards include the known-successful 1024 control.
Keep `--padding`, every other original shape/sampling setting and the same
allocator in both arms. Run each independently, even if the baseline fails:

```bash
for bucket in 1024 1280 1536 1792 2048; do
  for storage in stack buffer; do
    case_dir="$FJ_DIAG_ROOT/esm-$bucket-$storage"
    mkdir "$case_dir"
    rc=0
    python -m bench.prediction_profile --profile-out "$case_dir/profile.json" -- \
      predict --model esmfold2 --input "jobs/$bucket.yaml" \
      --padding --pad-tokens "$bucket" --msa auto --seed 101 \
      --option "lm_state_storage=$storage" \
      --output-dir "$case_dir/output" \
      >"$case_dir/run.log" 2>&1 || rc=$?
    printf '%s\n' "$rc" >"$case_dir/status.txt"
  done
done
```

Fill in the original sampling/weight/template/other options before running;
the sample count is deliberately not replaced in this capacity ladder. Jobs
must contain the intended actual input and fit the target. Preserve full logs
and per-executable argument/output/temp bytes: if the next peak becomes a pair
projection/einsum, removing `stack` alone does not prove that it too will fit.
Compare cold and fresh-process warm runs separately, and retain total device
usage as well as the allocator's lifetime high-water mark.

Use two sizes where **both** arms complete for the separate numerical panel:

```bash
python -m bench.esmfold2_performance_probe \
  --input jobs/256.yaml jobs/1024.yaml --experiment lm-storage \
  --output "$FJ_DIAG_ROOT/esm-storage-paired"
```

This runs stack, buffer and stack-repeat in separate processes with five paired
samples and the same explicit diffusion tape; it retains LM embedding errors,
per-entity residuals, confidence and the repeat floor. SVD and eager blocks are
fixed in all arms. The existing `both` experiment still means alignment plus
compiled-block experiments, not this new panel. Do not use `--smoke` for
accuracy admission. The core probe uses native unpadded features, unlike the
serving capacity ladder; do not equate their padded shapes or capacity limits.
An A6000 comparison at larger sizes is useful only where its stack arm also
completes. Keep reference lDDT separate from same-index agreement. New capacity
targets are not declared passed by the CPU component tests.

## Boltz-2: wait for the smaller-pool control before changing policy

The supplied 4090 GFP×6/1536 runs failed at the same kernel with ordinary and
launch-blocking execution, at both 0.9 and 0.85, with allocator live usage far
below its pool ceiling. This weakens the earlier asynchronous-fault hypothesis;
it does not measure CUDA module/workspace bytes or prove a particular external
allocator is at fault. Reserved unused pool space still consumes card capacity.

Return the ongoing 0.8/0.75/0.7 results with the first CUDA error, full log,
effective fraction, total GPU-memory trace and executable generated-code/temp
bytes. Include the same controls on 3090 if a 4090-only policy is proposed.
Success only after lowering the pool would support a reservation/resource
interaction, but would not identify cuBLAS versus Triton/module loading.
Failure even at 0.7 also does not by itself prove that memory is uninvolved.
Use optimized HLO/kernel metadata and retained PTX/cubin sizes to decide the
next loader/resource probe. File/code sizes are not resident GPU memory.

No sm_89-specific allocator default is introduced while these measurements are
pending. The measured A6000 48 GB policy stays at 0.85 and caller overrides
continue to win. Do not combine a module-loading-mode experiment, precision
change or new kernel policy with the fraction ladder.

## AF3: test preparation followed by fresh cached execution

The user reported that GFP×8 (1904 tokens, padded 2048) succeeds on 4090 when
the saved model executable and tuning store are loaded in a fresh process,
with compilation about 0.05 seconds and allocator peak about 17.05 in the
reported units. The original cold warm failed after compilation/tuning while
requesting its 14.35 temporary arena. Keep raw byte counts when comparing
these quantities; the original report mixed GB and GiB displays.

Run the same cold warm on an isolated writable diagnostic cache, then a
separate normal prediction against it. GPU process warming now prepares the
model/store in one child and exits before model dispatch, then verifies exact
main cache-key hits and actual execution in another. Default preparation can
tune; verification uses the error-on-tuning-miss policy. A miss or execution
failure stays a failed warm. Both process identities and cache evidence are
returned, rather than treating cache publication alone as successful execution.

Use a new results directory so neither cache tier can supply an older main
executable. Keep the original warm/prediction options in the commands below;
in particular, use the same weights and sampling schedule. The private root is
scoped to each command and reused by this pair, without changing the shared
installation or the user's persistent cache setting:

```bash
case_dir="$FJ_DIAG_ROOT/af3-2048-fresh-lifecycle"
mkdir "$case_dir"
rc=0
FOLDJAX_PRIVATE_COMPILE_CACHE="$case_dir/private-compile" \
  foldjax cache warm --model alphafold3 --token-buckets 2048 \
  --isolation process --cache-dir "$case_dir/compile" --json \
  >"$case_dir/warm.json" 2>"$case_dir/warm.log" || rc=$?
printf '%s\n' "$rc" >"$case_dir/warm-status.txt"
if [ "$rc" -eq 0 ]; then
  FOLDJAX_PRIVATE_COMPILE_CACHE="$case_dir/private-compile" \
    python -m bench.prediction_profile --profile-out "$case_dir/predict.json" -- \
    predict --model alphafold3 --input jobs/gfp8.yaml --msa auto \
    --padding --pad-tokens 2048 --cache-dir "$case_dir/compile" \
    --output-dir "$case_dir/output" \
    >"$case_dir/predict.log" 2>&1 || rc=$?
  printf '%s\n' "$rc" >"$case_dir/predict-status.txt"
fi
```

Retain the two synthetic lengths, actual padded dimensions, preparation and
verification records, `sources={"store": 1}` for each one-call verification,
main-key hit evidence, allocator/RSS peaks and full logs. Also retain the real
GFP×8 prediction as a separate result: a synthetic warm and a real input can
have different valid content despite their shared graph. Preparation still
has to compile/load the module in its own context; this only separates the
subsequent model dispatch, so a preparation-stage OOM is not resolved by
definition. The lifecycle has CPU process/key/cleanup coverage but has not yet
been run on the target GPU. In-process Python warming keeps its old behavior.
