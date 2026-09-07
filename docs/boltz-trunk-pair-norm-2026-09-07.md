# Boltz trunk: matched-input operator localization

Status: **pair normalization defect repaired; whole-model parity still open**.
The user prioritizes Boltz trunk before moving on to OpenBind or another model.
This follows the [conditioning repair](boltz-conditioning-fma-2026-09-07.md).
[Portable results](../bench/experiments/boltz-trunk-pair-norm-2026-09-07.json)
bind the immutable raw reports by SHA256.

## Controls that did not resolve the trunk discrepancy

All rows below use the original native first-recycle MSA operands, all 4,436
MSA rows, 437 tokens, all four MSA blocks and exact mapped checkpoint leaves.
Native standalone MSA first reproduced the original capture bitwise. Native
policy is Boltz 2.2.1, Torch 2.12 CUDA BF16 AMP with its FP32 islands, cuEq
triangle kernels, and `highest` matmul precision. FoldJAX preserves explicit
BF16 rounding with `xla_allow_excess_precision=False`.

| Diagnostic arm | Final MSA pair-representation RMSE |
| --- | ---: |
| Existing production (v26) | 0.0319687435 |
| MSA/transition native norm control | 0.0320470260 |
| Dense native-width MSA embedding GEMM | 0.0320953161 |
| Dense embedding plus norm control | 0.0320470260 |
| Native full-row PWA/transition shapes (v27) | 0.0319650454 |
| Full-row shapes plus norm control | 0.0320470260 |

These are internal representation errors, **not coordinate RMSDs in Angstrom**.
The initial MSA embedding arrays are identical across all four v26 arms; the
small downstream dense-arm change therefore cannot be attributed to improved
embedding values. Norms alone and removing row chunks do not fix global MSA
error. Those experimental embedding/row-shape changes were **not adopted**.

The norm control covers MSA norms and transitions, not triangle-attention
normalization or fused cuEq multiplication norms. A 32-row PWA diagnostic
reproduced native width-64 and width-128 norm outputs exactly, but its whole
PWA result is not a full-MSA reference: native row slicing itself changes GEMM
tiling. Full-MSA contraction comparisons above avoid that sliced-target error.

## Confirmed pair-block defect and production repair

The first native MSA pair block was replayed with its own FP32 residual.
All five operator input/output hooks and the final pair output reproduced
the original capture exactly. Sequential FP32 residual reconstruction also
matches the final captured pair output bitwise. The captures are copied at
each hook, not aliases of subsequently mutated tensors.

FoldJAX then executes **each operator on that operator's native input**; it
does not feed one candidate result to the next operator. Holding cuEq cores,
weights, dtype and input values fixed gives:

| Operator | Previous local RMSE | Corrected production local RMSE |
| --- | ---: | ---: |
| Triangle attention, starting | 0.0000419058 | **0, bitwise exact** |
| Triangle attention, ending | 0.0000617346 | **0, bitwise exact** |
| Pair transition | 0.0000814216 | **0, bitwise exact** |
| cuEq multiplication, outgoing | 0.0000016013 | unchanged |
| cuEq multiplication, incoming | 0.0000016631 | unchanged |

All three corrected LayerNorm outputs are also bitwise exact. Native CUDA
vector-4 Welford and FP32 FMA arithmetic matter before BF16 projection casts;
ordinary JAX normalization differed at those rounding boundaries. This is
a reproduced local cause, not a claim that it is the only trunk discrepancy.

The repair is in installable FoldJAX code, not only in the benchmark wrapper:
mixed FP32-residual/BF16-projection triangle attention selects the existing
CUDA norm, and both MSA and main Pairformer pair transitions opt into it.
Pure FP32 and custom BF16-input attention behavior remain unchanged. CPU,
TPU, ROCm, context-parallel and unsupported-width fallbacks remain generic JAX.
No Torch or FP64 dependency, checkpoint change, or new precision default was
introduced. Width-64 MSA normalization remains diagnostic, not production.
The shared call sites also reach template and confidence pair stacks; the
three bitwise operator witnesses above cover the first MSA pair block only.
The Pallas norms are non-fusible and materialize FP32 pair buffers. Their
uninstrumented latency and peak-memory cost is unmeasured and may regress
throughput or memory, particularly at large token counts. This numerical
repair does not admit a faster/memory-optimized production profile.

## Verification and limits

GPU jobs 388 and 390 establish native reproduction and the causal control;
391 independently verifies the actual production paths. Job 389 failed in
the diagnostic wrapper because epsilon was a traced Pallas closure constant;
the retry made this diagnostic argument static. Its incomplete output is
not admitted evidence. The production kernel already accepts dynamic epsilon
as an explicit input and did not need that repair.

Focused CPU checks passed: 66 tests covering fallback/selection, native AMP
operators, teacher-forced residual validation and MSA observations. The
initial full-suite attempt used a stale JAX 0.10.1 test environment while the
project lock requires 0.11.1; its `top_k(is_stable=...)` failure also exists
in HEAD. That attempt was stopped, the test environment was synchronized to
the frozen CI dependencies, and it is not counted as a release gate.

The frozen-dependency full CPU gate subsequently passed: **4,297 passed,
400 skipped, 8 network tests deselected; orchestration coverage 87.44%**
(required minimum 80%). Command:

```sh
JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 .venv-ci/bin/python -m pytest -q \
  -m 'not network' --tb=short --cov=foldjax \
  --cov-report=term-missing --cov-fail-under=80
```

This includes the installable-source/import and clean-wheel distribution
tests. GPU/Torch-only and optional-runtime checks that skip in CPU CI are
not counted as validated there. Ruff, frozen lock and staged diff checks
pass. All **934 src/bench/test Python files** match the immutable v30
snapshot; the newer documents and portable result files are recording its
completed runs, not changing the executed program.

No independent preprocessing, device-consumer tape identity, strict confidence
closure or uninstrumented speed/memory improvement follows from these operator
controls. Full n=5 structure/confidence replay is a separate required result.

## Full n=5 5SAK replay after the production repair

GPU job 392 completed with the v30 production snapshot, native captured
features and sampler tape, 200 steps, 3 recycles (4 passes), full MSA and
native BF16/FP32 policy. There are **no substituted native trunk or other
intermediate tensors** in this full run. One proper whole-system Kabsch per
original sample pair precedes entity measurements; no entity refit or crystal.
The [bound full report](../bench/experiments/boltz-amp-5sak-v30-2026-09-07.json)
retains raw confidence, dtype and artifact checks.

| Full FoldJAX snapshot | Protein max RMSD (Å) | Ligand max RMSD (Å) |
| --- | ---: | ---: |
| Conditioning-only repair v24 | 2.864511 | 0.380902 |
| Pair-normalization repair v30 | 1.912050 | 0.149726 |

| Original sample index | Protein RMSD (Å) | Ligand RMSD (Å) |
| --- | ---: | ---: |
| 0 | 1.912050 | 0.149726 |
| 1 | 0.081661 | 0.014427 |
| 2 | 0.047404 | 0.009336 |
| 3 | 1.488854 | 0.144400 |
| 4 | 0.642291 | 0.102242 |

The full-model discrepancy is smaller in this observed comparison, but **the
0.05 Å structure diagnostic and raw/public strict confidence still fail**.
The public `confidence_score` field is still missing. Do not infer general
improvement or closure from this single captured panel or calibrate tolerance
using this candidate result.

Final trunk RMSE changes from v24 `s=0.055363, z=0.080111` to v30
`s=0.050958, z=0.068882`. First-recycle MSA input-pair error remains
`0.000611619`, and MSA output-pair error remains `0.0324655`. Correcting pair
norms therefore leaves a substantial MSA discrepancy. Next localization should
teacher-force native MSA transition/OPM operands and separate FP32 single-track
operations. No additional width-64 or width-384 CUDA route is admitted here.

### Other modalities on the same production snapshot

GPU jobs 393–394 run the full own-trunk model, five original samples each,
with each case's native captured features/tape and the same settings/metric.
The historical v14 comparator predates **both** conditioning and pair-norm
repairs; it does not isolate the effect of the latest pair repair alone.

| Case / entity | Historical v14 max RMSD (Å) | Current v30 max RMSD (Å) |
| --- | ---: | ---: |
| 1URN protein | 0.042364 | 0.049661 |
| 1URN RNA | 0.311269 | 0.311173 |
| 3GCA RNA | 0.009712 | 0.004484 |
| 3GCA ligand | 0.011555 | 0.005300 |

3GCA retains its structural diagnostic pass. The 1URN RNA outlier remains
at original sample index 1; it is not resolved, and its protein maximum is
slightly worse than the historical value. **Both raw/public strict confidence
reports still fail on both cases**. No all-modality improvement is claimed.
Full reports: [1URN](../bench/experiments/boltz-amp-1urn-v30-2026-09-07.json),
[3GCA](../bench/experiments/boltz-amp-3gca-v30-2026-09-07.json).

## Review boundaries and deferred issues

An independent source review requested the explicit historical/current scope
and unmeasured Pallas-cost warnings above. Separate follow-ups remain: generic
`linear` can narrow a BF16-input/FP32-kernel result when bias is present (no
active native FP32-island caller with that input combination was found);
padding across 384 tokens changes native hidden/head chunk selection; and
featurizer/bridge sources are not universally bound into resume identities.
These are not covered or admitted by this finite, unpadded native-feature panel.
Private cuEq APIs are tied to the existing 0.11.1 runtime pin, not future
dependency versions. Context-parallel normalization remains a separate policy.

The local `docs/EXPERIMENTS.jsonl` is an existing tracked audit resource, not a
new portable benchmark report. Its new machine-specific rows are deliberately
left unstaged. Publication must verify its staged diff is empty; removing the
tracked project audit resource or rewriting its history is outside this repair.
