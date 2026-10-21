# ESMC process variation: 2026-10-21 handoff

These are user-reported RTX 4090 results from `032ba1f5`, followed by a
measurement protocol. No target-GPU reproduction or numerical default change
is claimed here. ESMFold2 continues to use `stack` storage by default.

## What the repeat panel established

Bindings, packed inputs and actual parameter values matched across the five
fresh processes for each storage mode. Within every process, the first and
second compact-embedding evaluations were bitwise equal. Across processes:

| Input | Within stack | Within buffer | Between storage modes |
| --- | --- | --- | --- |
| GFP, 238 tokens | median 0.125; maximum 1.047 | median 1.047; maximum 1.789 | maximum 1.789 |
| GFP×4, 952 tokens | maximum 1.977 | maximum 1.320 | maximum 1.977 |

These are maximum absolute differences of the compact BF16 embedding
`[1, tokens, 256]`, saved losslessly as FP32. The reported differences cluster
around a few values, including 0, 0.125, 1.047, 1.32, 1.38 and 1.79. This is
consistent with a choice becoming stable within a process, but it does not
identify a different kernel or prove an algorithm-selection defect. Both
storage modes vary. Buffer storage alone is not an established cause.

At the unmasked LM boundary, ESMC takes no random key and makes no random
draws. Downstream diffusion randomness therefore cannot explain a difference
already present in the LM embedding. Conversely, an LM difference does not
by itself measure its eventual structural effect.

## Repeat controls on the same two archived inputs

Run the matching diagnostic checkout with the fixed installation's interpreter
inside an existing single-GPU allocation. These commands do not submit jobs.
Keep the same GPU, weights, allocator settings and original non-deterministic
environment as the reported panel. Record site-supplied `XLA_FLAGS`; do not
silently erase them. A native control with deterministic flags already set
would not reproduce the original policy.

Define absolute paths for `FJ_PYTHON`, `FJ_PANEL`, `FJ_WEIGHTS` and a new writable
`FJ_DIAG_ROOT`. `FJ_PANEL` contains the original `case-1` and `case-2` archives.
Leave `--packed-length` unset to preserve their original unpadded LM shapes.
Serving padding is a separate shape control and must not be pooled with this
panel. Keep the default checkpoint compute policy and BF16 ESMC parameter
storage; do not add dtype overrides.

The probe revision changed, so archived creator hashes require the explicit
`--rebind-probe` migration. It verifies feature and model-source hashes, full
checkpoint content hashes and stable local anchors before writing a new
`local-binding.json`. The original binding is unchanged; its creator hash is
retained as provenance. The four bound model modules must still match:
`inference.py`, `models/model.py`, `models/diffusion.py` and `models/esmc.py`
under `foldjax/models/esmfold2`. A mismatch is a failed control, not permission
to edit those hashes. These bindings cover the listed modules, not every
dependency in the environment.

If only local checkpoint stat anchors changed on another NFS client, explicitly
set `FJ_REANCHOR_WEIGHTS=1`. That adds `--reanchor-weights`; matching full
checkpoint contents are still required. After the first verified migration,
the later controls reuse the local binding, avoiding another full checkpoint
read from NFS for every panel. Workers still check anchors and model sources
before/after execution and hash actual loaded parameter values.

```bash
: "${FJ_PYTHON:?fixed installation Python interpreter}"
: "${FJ_PANEL:?original paired panel directory}"
: "${FJ_WEIGHTS:?installed ESMFold2 weight directory}"
: "${FJ_DIAG_ROOT:?new writable result root}"
mkdir -p "$FJ_DIAG_ROOT"
migration=(--rebind-probe)
if [[ ${FJ_REANCHOR_WEIGHTS:-0} == 1 ]]; then
  migration+=(--reanchor-weights)
fi

run_lm_panel() {
  local case_id=$1 panel=$2 binding=$3 rc=0
  shift 3
  "$FJ_PYTHON" -m bench.esmc_state_diagnostics \
    --features "$FJ_PANEL/case-$case_id/features.npz" \
    --binding "$binding" --weights "$FJ_WEIGHTS" \
    --mode native --repeats 5 \
    --output "$FJ_DIAG_ROOT/case-$case_id-$panel" "$@" || rc=$?
  printf '%s\n' "$rc" >"$FJ_DIAG_ROOT/case-$case_id-$panel.status.txt"
}

for case_id in 1 2; do
  run_lm_panel "$case_id" native \
    "$FJ_PANEL/case-$case_id/binding.json" "${migration[@]}"
  verified="$FJ_DIAG_ROOT/case-$case_id-native/local-binding.json"
  if [[ ! -f "$verified" ]]; then
    printf 'case-%s: binding verification failed; controls skipped\n' "$case_id"
    continue
  fi
  run_lm_panel "$case_id" fingerprints "$verified" --compiler-fingerprints
  run_lm_panel "$case_id" deterministic "$verified" --deterministic
  if [[ ${FJ_COMPILED_CONTROLS:-0} == 1 ]]; then
    run_lm_panel "$case_id" compiled "$verified" --compile-esmc-blocks
    run_lm_panel "$case_id" combined "$verified" \
      --deterministic --compile-esmc-blocks
  fi
done
```

Each panel starts five stack and five buffer processes, interleaved. The three
required panels across two inputs are 60 fresh processes; the optional compiled
controls add 40. Each worker evaluates the LM twice. Failed workers retain logs,
all scheduled workers are attempted, and a nonzero status means the panel is
incomplete. A local binding's existence does not mean every native repeat
succeeded; return the status files too.

| Panel | Changed setting | Interpretation |
| --- | --- | --- |
| native | None | Original eager LM policy without a compiler observer |
| fingerprints | `--compiler-fingerprints` | Records precompile MLIR/options and optimized HLO hashes; observer overhead can change startup scheduling |
| deterministic | `--deterministic` | Process-wide deterministic XLA policy; eager block dispatch retained |
| compiled, optional | `--compile-esmc-blocks` | Compiled LM blocks under the original process policy |
| combined, optional | Both preceding flags | Compiled LM blocks under deterministic XLA policy |

Here `--deterministic` applies **two** shared XLA options before backend
initialization: `--xla_gpu_deterministic_ops=true` and
`--xla_gpu_enable_triton_gemm=false`. It is a combined compiler policy, not an
isolated test of one flag. Explicit conflicting XLA options are rejected.
Block compilation is independent. The product option
`foldjax predict --option deterministic=true` has its existing per-executable
semantics, including compiled ESMC blocks; it is a different interface from
this probe's process-wide control.

For each case/panel, return `plan.json`, `comparison.json`, all worker
`report.json`, `embedding.npy`, logs, status files and the migrated binding.
Reports contain actual `xla_flags`, `control_policy`, loaded parameter and
packed-input identities, device/runtime versions and optional
`compilation_observation`. Complete ten-worker panels contain 45 comparisons:
10 within stack, 10 within buffer and 25 between modes. Missing or mismatched
controls are excluded from distributions, while raw comparisons remain.
Post-evaluation parameter hashes do not prove values were unchanged at every
instant of evaluation; the staged weight-loading boundary is also not every
asynchronous serving-startup path.

Compare the frequency distributions first. If deterministic policy removes
variation, it implicates execution policy but does not isolate which of its
two options mattered. Different HLO hashes need an actual HLO/configuration
diff. Identical HLO hashes do not rule out kernel-internal nondeterminism,
workspace/address effects or runtime library choices. Observer-only changes
cannot establish the unobserved cause.

## Measure the final structural effect separately

The existing paired performance probe supplies a useful shared-core control:
two input sizes, five original-index samples, the same saved actual diffusion
draws for all three arms, and otherwise checkpoint-native settings. It uses
stack baseline, buffer candidate and an independent stack baseline repeat.
Do not add `--smoke`: that changes recycles and diffusion steps. The explicit
five-sample setting is a recorded deviation from any different checkpoint
sample count; it is not silently described as the released sample count.

```bash
: "${FJ_SMALL_JOB:?original smaller common-schema job}"
: "${FJ_LARGE_JOB:?original larger common-schema job}"
"$FJ_PYTHON" -m bench.esmfold2_performance_probe \
  --input "$FJ_SMALL_JOB" "$FJ_LARGE_JOB" \
  --weights "$FJ_WEIGHTS" --experiment lm-storage --seed 101 \
  --output "$FJ_DIAG_ROOT/structure-native"
```

After the LM controls, repeat this command in separate new output directories
with `--deterministic`, `--compile-esmc-blocks`, or both if those interventions
are relevant. Add `--compiler-fingerprints` only to a separately labelled
observer run. Within each case all three arms must report the same diffusion
tape SHA. When comparing separate panels, verify the actual tape arrays and
feature bindings as well; equal seeds alone do not prove equal random draws.

Retain the case `features.npz`, `binding.json`, `diffusion.npz`,
`diffusion.json`, tape-prepare report/log, and every arm's
`report.json`, `lm_embedding.npz`, `jax_coords.npz`, `jax_confidence.npz`,
output schemas and log. Retain the per-experiment `comparison.json` and root
`report.json`. The helper
[`compare_coordinates`](../bench/esmfold2_tape_report.py) checks five samples
against shared feature masks/atom identities and delegates to the common
entity comparison. Those shared features do not establish independent native
preprocessing or writer-output parity.

Declare the assessment before inspecting candidate results:

- Match features, atom keys/order, entity-instance labels and validity masks.
  Mismatches, nonfinite values or missing samples fail the control. Pair sample
  indices 0–4 directly; never reorder by pLDDT or select the best structure.
- Fit one proper whole-complex Kabsch transform on all valid system atoms per
  sample. Under that transform report all five RMSDs for every entity instance
  and each entity's maximum. Do not refit individual chains or hide an outlying
  copy in pooled complex RMSD.
- Report raw confidence leaves and per-sample/per-entity changes in their native
  units, including pLDDT and available pTM/ipTM/chain-pair outputs. Do not combine
  unlike scores into one acceptance number. The existing array helper records
  its strict `atol=1e-4, rtol=1e-4` diagnostics and preserves undefined-position
  checks; retain those results rather than retroactively relaxing them. Its
  pooled leaf summaries do not replace the per-sample/entity breakdown, so keep
  raw confidence archives for that analysis.
- Compare candidate residuals with the independent stack-repeat spread at the
  same boundary. That spread is a noise floor, not a statistical-equivalence
  test or an automatic allowance. A difference inside the observed floor is
  inconclusive about equivalence; a repeated excursion beyond it merits
  investigation. Keep the existing 0.05 Å coordinate diagnostic. The historical
  `>0.05` and `<=0.1` Å gray zone and `>0.1` Å priority rule govern triage, not
  a new universal acceptance threshold. Do not relax a gate after seeing results.

Keep ordinary native-RNG predictions as another arm, with the same two jobs,
five samples and recorded seed/settings, repeated in fresh processes. They
measure serving repeat spread without a substituted diffusion tape; they do
not replace the tape-fixed comparison. The paired core probe bypasses writers,
so assess final written structures separately if those are the user-visible
quantity of interest.

The existing paired probe varies LM values and may independently compile the
downstream core in each process. Even with equal diffusion draws, a structural
difference therefore does not isolate LM variation from downstream numerical
variation. A causal follow-up must interchange the saved compact embeddings
through the same downstream executable with identical inputs and actual RNG
draws, and separately hold the embedding fixed while repeating downstream
execution. Those interchange controls are not implemented by the command
above. This finite panel can establish observed differences and their repeat
floor; it does not by itself promote buffer storage, prove full-model accuracy
equivalence or complete a release gate.
