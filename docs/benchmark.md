# FoldJAX against upstream

Every FoldJAX-side runtime here is JAX. Five models are reimplementations of
published torch models; AlphaFold 3 is the official JAX distribution driven
through the same FoldJAX interface. The useful comparison is against each
model's own reference repository, on the same job, under the same nominal
schedule, measured the same way.

The [model version ledger](model-versions.md) gives the exact source and
checkpoint target implemented by each port. A benchmark's upstream checkout
is a separate identity and is taken from that row's manifest, never inferred
from the short model name. [`bench/`](../bench/) is the harness that runs
these jobs; it reruns one row per process and records byte identities for its
external inputs, checkpoints, and (where applicable) upstream source/runtime
identity.

**The schedule is AlphaFold 3's released default — 5 diffusion samples, 200
diffusion steps, 10 recycles, seed 101 — not a number chosen here.** Protenix
ships the same three. Boltz-2 defaults to 3 recycles rather than 10, so this
asks more of it than its own default does; it asks the same of its upstream,
which is what makes the comparison mean anything.

Rows were measured on one **NVIDIA RTX PRO 6000 Blackwell Server Edition**
(`sm_120`; `nvidia-smi` reports 97,887 MiB of total memory).

FoldJAX runs at its own default precision in the comparison below, which is
reduced from upstream's on four of the six models (Boltz-2, OpenDDE,
OpenFold3, Protenix — see the precision table and
[docs/engineering-notes.md#which-precision-each-model-runs](engineering-notes.md#which-precision-each-model-runs)
for which arithmetic each side ran). **Part of the gain below is that choice,
not the implementation alone.** The matched-precision reruns further down
show what is left once both sides use upstream's arithmetic.

## Default-precision comparison

Over the 24 model/length cells where both FoldJAX and upstream complete under
this schedule, FoldJAX's warm wall time is **0.82x to 4.16x** upstream's
(median **2.41x**), and its device peak memory is **1.03x to 4.45x** lower
than upstream's (median **2.24x**). The one cell where FoldJAX is slower is
ESMFold2 at 254 tokens.

| model | tokens | FoldJAX s | upstream s | speed-up (up/FJ) | device peak (up/FJ) |
|---|---|---|---|---|---|
| Boltz-2 | 254 | 15.4 | 37.6 | 2.44x | 2.28x |
| Boltz-2 | 499 | 23.4 | 55.8 | 2.38x | 1.47x |
| Boltz-2 | 1,003 | 54.5 | 136.7 | 2.51x | 1.91x |
| Boltz-2 | 1,354 | 77.0 | 192.0 | 2.49x | 2.41x |
| Boltz-2 | 2,096 | 186.9 | 480.2 | 2.57x | 2.72x |
| ESMFold2 | 254 | 24.8 | 20.3 | 0.82x | 1.28x |
| ESMFold2 | 499 | 34.4 | 47.2 | 1.37x | 1.97x |
| ESMFold2 | 1,003 | 76.9 | 192.4 | 2.50x | 4.45x |
| OpenDDE | 254 | 22.8 | 30.2 | 1.33x | 2.19x |
| OpenDDE | 499 | 45.5 | 71.9 | 1.58x | 2.73x |
| OpenDDE | 1,003 | 136.4 | 247.6 | 1.81x | 3.08x |
| OpenFold3 | 254 | 15.8 | 29.7 | 1.88x | 2.30x |
| OpenFold3 | 499 | 25.1 | 51.2 | 2.04x | 4.30x |
| OpenFold3 | 1,003 | 56.3 | 140.7 | 2.50x | 2.60x |
| OpenFold3 | 1,354 | 84.4 | 239.5 | 2.84x | 2.26x |
| OpenFold3 | 2,096 | 192.7 | 626.3 | 3.25x | 2.26x |
| OpenFold3 | 3,012 | 418.6 | 1,492.9 | 3.57x | 1.51x |
| Protenix | 254 | 16.6 | 69.1 | 4.16x | 1.03x |
| Protenix | 499 | 23.3 | 73.4 | 3.15x | 1.14x |
| Protenix | 1,003 | 50.0 | 98.1 | 1.96x | 2.09x |
| Protenix | 1,354 | 74.7 | 132.3 | 1.77x | 3.29x |
| Protenix | 2,096 | 164.4 | 241.3 | 1.47x | 2.22x |
| Protenix | 3,012 | 373.8 | 656.3 | 1.76x | 1.59x |
| Protenix | 4,100 | 720.9 | 2,190.8 | 3.04x | 1.59x |

A number of additional cells run on FoldJAX where upstream fails (OOM) at the
same size; those are not ratios and are reported, with confidence intervals
and host-peak figures, in the paper.

## Matched-precision reruns

Forcing FoldJAX to upstream's own precision isolates what the port itself
contributes. Nine model/length cells were rerun this way; `wall` and `device
peak` below are matched-precision over FoldJAX's default (so `1.0x` means no
cost from matching precision), and `upstream/matched wall` is upstream's warm
time over the matched-precision run's (so values near `1.0x` mean upstream
parity).

| model | tokens | upstream/matched wall | matched/default wall | matched/default device peak | ΔTM (matched − upstream) |
|---|---|---|---|---|---|
| Boltz-2 | 1,003 | 1.01x | 2.49x | 1.09x | +0.001 |
| Boltz-2 | 2,096 | 0.96x | 2.69x | 1.24x | 0.000 |
| OpenDDE | 1,003 | 1.16x | 1.57x | 1.94x | -0.001 |
| OpenFold3 | 1,003 | 1.83x | 1.36x | 1.25x | -0.001 |
| OpenFold3 | 2,096 | 2.01x | 1.61x | 1.20x | 0.000 |
| OpenFold3 | 3,012 | 1.97x | 1.81x | 1.10x | +0.118 |
| Protenix | 1,003 | 1.93x | 1.02x | 1.00x | +0.001 |
| Protenix | 2,096 | 1.44x | 1.02x | 1.03x | 0.000 |
| Protenix | 3,012 | 1.76x | 1.00x | 1.00x | 0.000 |

At matched precision, Boltz-2 runs at essentially upstream's own speed
(0.96x-1.01x) — its whole default-precision speed-up is the precision
choice. OpenFold3 and Protenix keep most of their advantage even once forced
to upstream's precision (1.4x-2.0x). Structure quality (TM) moves by at most
0.001 except for OpenFold3 at 3,012 tokens, where the matched-precision arm
scores 0.118 higher — see the paper for that case.

## Where this comes from

These numbers are the FoldJAX release benchmark; the full method, confidence
intervals, per-seed variance, admission-law accuracy, and figures are in the
accompanying benchmark paper. A detailed, dated 2026-09 measurement report
(an earlier, 300 W Max-Q RTX PRO 6000 Blackwell card, and upstream OpenFold3
at 0.3.1 rather than the current 0.5.0/OpenBind target) is kept for
provenance at
[docs/archive/benchmark-full-report-2026-09-12.md](archive/benchmark-full-report-2026-09-12.md);
it predates this page and should not be read as current.
