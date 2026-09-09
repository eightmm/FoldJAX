# OpenBind ordinary-RNG warm inference

Candidate-only measurement, not an upstream speedup claim. Queue job 1095
exited zero. Bundle: `openbind-ordinary-warm-20260909-Ev8LIw/candidate`.

3GCA: 46 tokens, 718 atoms, FP32 native-private backend, five samples, 200
diffusion steps, four trunk passes, released MSA depth 1024 and native
per-sample cutoff 750. GPU-resident features and parameters are prepared before
timing. Model RNG is ordinary JAX key 101 reset per call, without any draw
replay or internal observer. Input features still come from the native archive;
this measures model inference, not independent raw-input preprocessing.

The standard output policy is retained: no returned trunk representations and
no optional pLDDT logits, while this small case still returns the normal pair
logits. Only coordinates and public confidence are written after measurement.

| Phase | Seconds |
| --- | ---: |
| First call, including compilation | 70.227535 |
| Warm 1 | 1.208808 |
| Warm 2 | 1.210323 |
| Warm 3 | 1.209794 |
| Warm median | 1.209794 |

All four calls produced bitwise-equal coordinates, pLDDT, pTM and ipTM. Output
archive SHA256s were independently rechecked. Timing includes device completion
but excludes observation, host transfer, compression and file writing. Each
GPU result is released before the next call; host snapshots remain for checks.

Allocator process-lifetime peak was 1,780,143,616 bytes (1.657888 GiB) after
both first and warm calls. **This includes setup and compilation and is not a
reset warm-only peak.** Preallocation was disabled, an explicit benchmark
deviation from the shipped CLI allocator default. Reserved process VRAM was
not substituted for allocator live-byte peak.

Verification: measurement-boundary and result-lifetime tests: 6 passed;
Ruff and diff checks passed. GPU job 1095, source/input/checkpoint/harness
postflight checks and four output archive hashes passed.

## Native comparison

Native job 1096 exited zero. Bundle:
`openbind-native-warm-20260909-olFOBK/native`. It uses the pinned native
prediction runner, FP32 (`32-true`), Triton triangles, n=5, 200 steps and four
trunk passes. Its recorded cutoff is 750 and triangle chunk size is 1024.
PyTorch is `2.12.1+cu130`, CUDA runtime 13.0. The runner substitutes matching
captured tensor inputs at the prediction boundary (38 tensor keys, with exact
shape/dtype checks), not intermediate model outputs. Native seed 101 is
reapplied by its ordinary `predict_step`; no RNG draw interception is used.
Container copies outside timing prevent mapping mutations leaking between
calls without duplicating tensor storage.

| Measurement | FoldJAX | Native |
| --- | ---: | ---: |
| First call (s) | 70.227535 | 4.550226 |
| Warm 1 (s) | 1.208808 | 3.897979 |
| Warm 2 (s) | 1.210323 | 3.877507 |
| Warm 3 (s) | 1.209794 | 3.893276 |
| Warm median (s) | 1.209794 | 3.893276 |
| Allocator lifetime peak (GiB) | 1.657888 | 1.531043 |

The observed warm-time ratio is 3.218, while FoldJAX's allocator peak is 8.3%
higher. This is a single small-case measurement, not a general speedup or
memory-reduction claim. Native peak is `torch.cuda.max_memory_allocated`,
never reset during this process; candidate peak is JAX `peak_bytes_in_use`.
Both include setup/first-call work and exclude reserved-pool sizes. These
framework allocator metrics are not a measurement of all driver allocations.

**Timing work differs:** native `predict_step` includes native reseeding and
its full confidence/ranking postprocessing; candidate times `compile_predict`
and its standard prediction outputs. Both exclude file writing, but they are
not identical metric workloads. The ratio must not be reported as a
controlled same-operations kernel speedup. A common-output workload control
and an end-to-end comparison remain separate tasks.

Native ordinary repeats are not bitwise equal despite the same integer seed.
Against warm 1, warm 2/3 global-fit entity RMSD maxima are respectively
RNA 0.010567130/0.009857117 Å and ligand 0.004896023/0.004763152 Å. These are
ordinary-repeat variations, not a proven deterministic-kernel noise floor:
RNG streams were not taped. FoldJAX's four public fields were bitwise stable.
No cross-framework ordinary-RNG coordinate parity is claimed.

Verification: native finished receipt binds measurements by SHA256; all four
native output archive hashes were checked. Timing/container-copy tests now
pass 8 cases; Ruff and diff checks pass. Native memory settings and n=5/200/
four-pass configuration were inspected from its preflight record.

Remaining: the rest of the performance panel, precise common-workload and
warm-only memory controls, stronger candidate runtime provenance, review and
commit. These timings do not change fixed-tape parity failures.

## Native input-mutation control

Job 1097 completed with per-call checks that all 38 shared tensor features
retained their exact shapes, dtypes and values after each prediction. Warm
times were 3.844669, 3.838060 and 3.849909 s (median 3.844669 s). Public
outputs still differed from the first call. Thus persistent in-place changes
to those shared inputs are excluded for this run; RNG-stream equality and
the numerical cause of native repeat variation remain unproven.

Bundle: `openbind-native-warm-guard-20260909-g9JPtM/native`. Finished-to-
measurement and four output archive hashes passed. Measurement/input-guard
tests: 9 passed; Ruff/diff checks pass. The original timing table is retained
as its own measured run rather than replaced by the faster follow-up.
