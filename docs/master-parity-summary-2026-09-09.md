# FoldJAX parity on master, all models, 2026-09-09/10

One night on the lab Slurm server (four RTX PRO 6000 Blackwell), one recipe
for every model: upstream configured with the kernel family the port uses,
the same RNG tape, n=5, seed 101, two native processes for upstream's own
floor, two port processes for the port's own floor, and the residual read
against both native draws. Per-model detail lives in the linked documents;
this page is the cross-model reading.

| model | cases | pass | deferred | at-floor | open | detail |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| OpenFold3 (OpenBind, `cueq`) | 7 + ion | 2 + ion | 1 | 2 | 0 (2 excluded: native attention fell back below 100 tokens) | `openbind-master-cueq-panel-2026-09-09.md` |
| Boltz-2 | 5SAK + ion | 0 | 0 | 5SAK (kernel-toggle scale), ion (sample 3 bistable under same-size random trunk noise; sampler exact given the trunk) | 0 | `boltz2-master-kernel-toggle-2026-09-09.md`, `ion-case-1aay-master-2026-09-09.md` |
| Protenix | 7 + ion | 2 (3 with deterministic XLA ops: 1UBQ 0.037) | ion | 5 (4) | 0 | `protenix-master-panel-2026-09-09.md` |
| OpenDDE | 7 + ion | 6 | ion | 1 | 0 | `opendde-master-panel-2026-09-09.md` |
| ESMFold2 | 7 | 1 | 2 | 3 | 7ST3 chain B | `esmfold2-master-panel-2026-09-09.md` |

Bands: pass < 0.05 Å, deferred < 0.1 Å (the user's tolerance), at-floor =
above 0.1 Å but within upstream's own process movement, or basin sharing, or
(Boltz-2 1AAY) reproduced by random perturbations of native's own trunk at
the port's measured band.

## What the night established

1. **Upstream is the scale, not a fixed threshold.** Every residual that
   looked like a port defect on the workstation turned out to be the size of
   upstream's own movement under an equivalent implementation: OpenFold3's
   Triton→cuEq swap (0.8-6.9 Å on the "bad" cases), Boltz-2's `use_kernels`
   toggle (2.9 Å on 5SAK), Protenix's process-to-process basin selection
   (1-25 Å on 7ST3/7R6R/3V7E), ESMFold2's ordinary CUDA nondeterminism
   (1.1 Å on 5SAK sample 1). OpenDDE alone is tight on both sides
   (0.002-0.016 Å) except 1URN, where upstream itself moves 0.15 Å.
2. **Chaotic samples land in shared basins.** With three implementations
   (native A, native B, port) each chaotic sample shows the port agreeing with
   one native process at the same distance the two natives agree with each
   other when they share a basin (Protenix 7ST3 sample 1: 0.38 Å vs 10.8 Å).
   The port's distribution of outcomes is upstream's.
3. **The port's own floor is mostly XLA kernel selection.** Freezing autotune
   makes OpenFold3 and ESMFold2 bitwise repeatable across processes and
   removed the one 1UBQ "residual" on ESMFold2 (0.122 → 0.02 Å), and
   collapses Boltz-2's 5SAK port floor from 0.83 Å to 0.0008 Å (jobs
   631/632) without moving the native residual. Protenix needed one more
   flag: its frozen pair still differed by 0.17 Å on 1UBQ sample 2, already
   present in `s_trunk`/`z_trunk` with bitwise-equal inputs; with
   `--xla_gpu_deterministic_ops=true` (jobs 654/656) two port processes are
   bitwise and the port sits 0.037 Å from native-A on every sample, moving
   1UBQ from at-floor into the pass band. On the other six cases (jobs
   657-662) the flag leaves every verdict where it was (3 pass, 1 deferred,
   3 at-floor in total) and no wall-time penalty is visible at ≤ 300 tokens.
   Recommended for every Protenix parity replay; default promotion waits on a
   1-3k-token timing and a port-level setting.
4. **Precision policy decisions**: OpenDDE keeps `high` (matches native's
   torch TF32; `highest` is worse everywhere). OpenFold3 keeps `cueq` as the
   default triangle kernel; `cueq-full` is 4-10% faster (9.9% at 1003
   tokens, jobs 638-641) for 3-8% more peak and one stable 0.22 Å case, so it
   stays opt-in; the 3k-token point is unmeasured.

## Open items

- **ESMFold2 7ST3 chain B**: samples 1/3/5 sit 2-4× above native's own
  scatter with autotune frozen (1.07/0.53/3.58 Å vs 0.24/0.08/1.09), above
  upstream's scatter on every pairing. Narrowed 2026-09-10: the ESMC-6B
  language model is excluded (its bf16-order drift is uniform across cases
  and injecting native's exact LM moves the port further, 3.2/1.3/4.7 Å);
  the residual lives in the pair trunk or structure head on three
  near-chaotic samples. Next step is a deterministic-array bisect there.
- **Boltz-2 ion case (1AAY)**: protein 0.175 Å, all of it on sample 3 (the
  other four samples 0.006-0.028 Å), against a bitwise-repeatable native and
  a 0.018 Å port floor. Upstream's own kernels-off toggle moves this case by
  only 0.020 Å (job 634), so at the coordinates the port residual is about
  9× upstream's own implementation scatter. Placed (jobs 635/636): with
  native's trunk injected the port's conditioning and sampler reproduce
  native to 1e-5 Å on every sample, and at every trunk boundary the port is
  closer to native than upstream's own kernels-off arm (delta_z relative
  1.31e-3 vs 1.45e-3, output_z 2.94e-3 vs 3.31e-3). The near-tie is
  measured, not asserted (jobs 647-650): four random Gaussian perturbations
  of native's own trunk at the port's band put sample 3 at 0.1745 Å twice
  and at 0.008 Å twice, the other samples staying at 0.005-0.02 Å. Sample 3
  is bistable at exactly the port's distance; at-floor, same class as 5SAK.
- **cuEq release independence** was shown for Boltz-2 only.

## Interface unification (in parallel, CPU)

Inventory in `interface-inventory-2026-09-09.md`. Landed on main:
shared CCD session mixins, one released-default strip helper, Backend hooks
for managed asset profiles, one summary-score reader (`70946af..4902a7a`),
then one `WeightAnchors` implementation behind the shared weight session and
the three backends that hand-rolled it (Boltz-2, ESMFold2, AlphaFold3), and
both port compile-cache paths routed through `cache.compilation_cache_scope`
(`cb3e632..9a4c70d`, 14 files, +388/-174). The CI gate on main
(`pytest -m 'not network' --cov`, CPU): after the first wave 6048 passed /
28 failed / 87.72% coverage; after the second wave 6053 passed / 27 failed /
87.73% (gate 80%). 26 of the failures are the pre-existing AlphaFold3
data-cache/small-input set and one is the mtime-resolution flake in
`test_stable_compile.py`; the 28th in the first run was real and is fixed:
`cueq-full` had never been registered in the shared execution vocabulary
(`7a757a2`). Left out on purpose: `enable_compilation_cache` stays exported
(documented direct-library entry), and the Protenix backend's
`--no-compile-cache` forcing stays (that CLI's cwd-relative default).

## Ion case

`ion-case-1aay-master-2026-09-09.md`: 1AAY (Zif268 + DNA duplex + 3 Zn, 115
tokens) accepted by all four ion-capable input translations; OpenBind pass
(0.034 Å worst, on a Zn), Protenix 0.063 and OpenDDE 0.070 Å deferred at
their native floors (0.054/0.070), Boltz-2 as above.
