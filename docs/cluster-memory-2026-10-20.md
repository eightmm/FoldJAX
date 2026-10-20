# Cluster follow-up: 2026-10-20

These are user-reported GPU results and next discriminating measurements, not
local GPU reproductions. No Slurm jobs are submitted by these tools. Use the
existing one-GPU allocation, fresh processes, unchanged weights/features/native
sampling options, and a new writable output directory for each arm. Keep normal
unobserved capacity/latency runs separate from diagnostic runs.

## ESMFold2: process variation, not an established buffer defect

The corrected `a452a0a4` panel used the original paired probe on one RTX 4090,
with seed 101, five structure samples, eager ESMC blocks, SVD, fraction 0.9,
no deterministic option, and empty `XLA_FLAGS`. Every process had exactly zero
LM cold/warm error. The compact BF16 embedding was saved as FP32:

| Comparison | GFP: 238 tokens, 1 chain | GFP×4: 952 tokens, 4 chains |
| --- | ---: | ---: |
| stack baseline ↔ buffer | 1.3125 | 1.7930 |
| stack baseline ↔ stack repeat | 1.3125 | 0 |
| buffer ↔ stack repeat | 0.125 | 1.7930 |

The large differences affected about 90% of values. An outlying stack process
in the first case and an outlying buffer process in the second do not isolate
storage as the cause. Small CPU storage tests also cannot establish the GPU
cause. Default `stack` is retained; no new numerical default is promoted.

At the unmasked LM boundary, ESMC has no random draws and no seed input. The
downstream model does have keyed stochastic operations, but those do not
explain a difference already present in the compact LM embedding. First
distinguish different loaded values/packing from execution/compilation choices.
Matching HLO does not exclude nondeterminism inside a GPU kernel or effects of
workspace/address selection. Kernel selection remains a hypothesis.

### Repeat the LM boundary before collecting instrumented layers

Reuse the original panel's `case-1` and `case-2` **features and bindings**. The
tool does not search MSA, download weights, run diffusion, or repeat structure
inference. Define absolute `FJ_PANEL`, `FJ_WEIGHTS` and `FJ_DIAG_ROOT` paths:

```bash
: "${FJ_PANEL:?original paired panel directory}"
: "${FJ_WEIGHTS:?installed ESMFold2 weight directory}"
: "${FJ_DIAG_ROOT:?new writable results root}"
mkdir -p "$FJ_DIAG_ROOT"
for case_id in 1 2; do
  rc=0
  python -m bench.esmc_state_diagnostics \
    --features "$FJ_PANEL/case-$case_id/features.npz" \
    --binding "$FJ_PANEL/case-$case_id/binding.json" \
    --weights "$FJ_WEIGHTS" --mode native --repeats 5 \
    --output "$FJ_DIAG_ROOT/esmc-native-case-$case_id" || rc=$?
  printf '%s\n' "$rc" >"$FJ_DIAG_ROOT/esmc-native-case-$case_id.status.txt"
done
```

Leave `--packed-length` unset for the original unpadded paired-panel LM shape.
For a separate serving-profile comparison, use that run's actual packed target
and record it; do not mix the two shapes in one frequency comparison. Keep
the same allocator/environment as the original panel. The default ESMC storage
dtype is BF16 and compute dtype follows the checkpoint, as in the old probe.
The weights are synchronized before LM evaluation, also as in that probe;
this is the staged LM boundary, not an observation of every asynchronous
serving-startup operation.

Each case interleaves five stack and five buffer workers. Each worker computes
the LM twice, then hashes the actual device parameters. These post-run hashes
do not prove values remained unchanged at every instant of evaluation.
Save `plan.json`,
`comparison.json`, every worker's `report.json`, embedding and log. Inspect
cold/warm differences, packed-input/settings/device-weight identity agreement,
and all 45 process pairs: 10 within-stack, 10 within-buffer, 25 between-storage.
All workers are attempted; after reports are collected, any failed worker
makes the command exit nonzero. Missing repeats are not successful runs.
Five repeats are a descriptive frequency panel, not statistical admission of a
new default or an equivalence claim.

Bindings retain strict source/content checks. If moving to another NFS client
only changes local stat anchors, explicitly add `--reanchor-weights`: every
checkpoint must have a matching full SHA256 and stable before/after stat, and
source/features/probe hashes must still match. A local binding copy is written
inside the new result directory; the original is not edited. Do not bypass a
content/source mismatch to reuse an old panel. Source checks cover the files
listed by that binding, not a hash of every dependency in the environment.

Only after that panel, collect separately labelled observations with the same
arguments and a new output directory:

- `--mode capture`: two layer-capture workers and replay of their identical
  stored layers through stack/scatter, donated updates and non-donated updates,
  followed by the original compact projection. Layer hashes, dtypes/layouts
  and the first divergence distinguish generation from transport/projection.
- Add `--replay-barrier` in another capture run to test the replay timing
  control. Capture already adds host reads, file writes and barriers, so it can
  remove the original failure; absence there does not prove native correctness.
- Add `--compiler-fingerprints` to a **separate native repeat** to record
  precompile MLIR/options and optimized HLO fingerprints. Hashing adds overhead
  and can perturb compilation/startup scheduling; retain the first unobserved
  panel. Fingerprint mismatch needs a real HLO diff, not a causal conclusion.
- `--deterministic` is a separate combined control: it also enables compiled
  ESMC blocks. Do not present that comparison as changing only determinism.

## OpenDDE: use the finer actual-storage profile

Automatic atoms now use the smallest existing atom bucket holding stored
atoms; structural tokens round stored capacity to 64, composed with the mesh
row multiple. The reported 952/7584/1816 input resolves to 1024 tokens,
8192 atoms and 1856 structural tokens. Explicit pins, existing stored suffixes,
validity masks and prefix RNG contracts remain intact.

The user reports that structural 2048→1856 allowed 4090/1024 to complete;
reducing atoms did not move that observed peak but reduced runtime about 30%.
The new automatic profile still needs a real GPU accuracy/capacity check. Warm
the actual GFP×4 job at fixed MSA 2048, then predict in a fresh process with the
same options. Compare 896 and 1024 on 24 GB; on A6000 first compare the old
1280 control, then try 1536/1792/2048 one fresh process at a time. Retain the
full native sampling schedule and real alignments; a failed size is a failed
measurement, not a reason to silently reduce them.

**OpenDDE automatic padding profiles whose axes change need warming again.**
The namespace and graph revision stay stable: exact and explicit profiles with
the same concrete shapes remain valid. Synthetic all-alanine token warming
does not cover every structural/atom combination. Its report now says so;
use real-input warming or common sufficient pins for the desired workload.

## Boltz-2: confirm automatic 0.8 on the RTX 4090

The user reports 4090/1536 failing at 0.9 and 0.85, including launch-blocking
runs, and succeeding at 0.8 or below. A single-model Boltz-2 CLI command now
selects 0.8 on a confirmed RTX 4090. Clear both fraction variables only for
the automatic-policy control; keep the original values in other controls.
Verify the effective fraction in the run record, then compare 1280 and 1536
in fresh processes. The 3090/1536 control retains 0.9, and A6000 retains 0.85.
Unknown mixed cards or multi-model commands retain the general policy;
explicit fraction choices still win. No cache rebuild is required just for
the fraction change. This success supports a resource-reservation interaction,
but does not identify which CUDA allocation or library caused the earlier OOM.

## Large ESMFold2 allocations and the AF3 cold/warm arena difference

The reported ESMFold2 1280 failure asks for 15.46 GiB; multichain A6000 runs
at 1792/2048 ask for 29–37 GiB. Those sizes alone do not identify an operation
or even establish that the request is one tensor rather than the executable's
temporary arena. Keep the exact exception, executable name and raw byte count.
Do not reduce an estimated tensor merely because its calculated size resembles
the allocator message. The A6000 1792 single-chain success and seven-chain
failure remain separate observations; the requested single-chain recheck is
pending, and no single-chain limit is lowered from the multichain result.

For each original failing job, retain the complete predict arguments as a Bash
array `original`, starting with `predict --model ...`. Preserve every sampling,
MSA, precision, padding and allocator setting. Use the existing
[cold-dump procedure](cluster-memory-followup.md#one-original-profile-cold-dump),
substituting `bench.prediction_profile` and adding `--hlo-fingerprints` before
`--` for ESMFold2/AF3. `JAX_ENABLE_COMPILATION_CACHE=false` forces cold JAX
compilation without deleting shared cache entries or disabling Tokamax JSON
records. For example, in a fresh result directory without whitespace:

```bash
case_dir="$FJ_DIAG_ROOT/original-cold"
mkdir "$case_dir"
mkdir "$case_dir/dump"
rc=0
JAX_ENABLE_COMPILATION_CACHE=false \
XLA_FLAGS="${XLA_FLAGS:+$XLA_FLAGS }--xla_dump_to=$case_dir/dump --xla_dump_hlo_as_text" \
  python -m bench.prediction_profile --hlo-fingerprints \
  --profile-out "$case_dir/profile.json" -- \
  "${original[@]}" --output-dir "$case_dir/output" \
  >"$case_dir/run.log" 2>&1 || rc=$?
printf '%s\n' "$rc" >"$case_dir/status.txt"
python -m bench.kernel_shootout.attrib --dump "$case_dir/dump" \
  --json-out "$case_dir/buffers.json" >"$case_dir/buffers.txt" 2>&1
```

Resolve any old dump flags first. Return the profile, complete log and whole
dump directory, especially optimized HLO, buffer assignment, live ranges and
memory-usage report. Compare the failed allocation's bytes with
`temp_size_in_bytes`, largest buffers and simultaneous live values. Match the
analyzer's chosen module to the exception; reused arena offsets are not
additive tensors. Extra fingerprinting/serialization is diagnostic overhead,
not normal compile/load latency or a verified production memory peak.

For AF3, compare the **same real input and dimensions** in two new processes:
a normal load of the successful existing cache and a cold JAX compilation
using the same existing Tokamax store. Require store-sourced tuning in both;
do not compare a measured/heuristic fallback with a stored configuration as if
only compilation changed. Check the main JAX cache key when present, input ABI,
tuning identities, compile-options and precompile/optimized HLO fingerprints,
plus raw argument/output/alias/temp bytes. Namespace equality alone is weaker
than executable equality. The reported 15.41 versus 14.35 displayed GB/GiB
needs raw bytes and this provenance before blaming a kernel or memory-aware
compiler schedule. The existing prepare/exit/fresh-verify warm lifecycle stays
in place; no AF3 graph or tuning key is changed by this investigation.

The compile-options fingerprint hashes the raw serialized options, including
dump paths/debug settings; different diagnostic destinations can change it
without changing an operation. Match those settings or account for that known
difference before interpreting a hash mismatch. Optimized HLO fingerprints
omit debug metadata but retain layouts and backend configurations.
