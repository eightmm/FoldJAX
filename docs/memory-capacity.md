# Observed GPU capacity

Reported capacity and fitted peak estimates answer different questions. The
frozen peak laws retain their calibration data, coefficients, and refusal
rules. The following user reports from 2026-10-17 add warnings; they do not
provide peak measurements from which to fit new coefficients.

| Model | Reported maximum successful token bucket, 24 GB class | RTX A6000, 48 GB |
| --- | --- | --- |
| ESMFold2 | 640–768 | 1024 |
| Boltz-2 | 1280–1536 | 1280 |
| Protenix | 2048 | 2048 |
| OpenFold3 | 2048 | 2048 |
| OpenDDE | 896 | 1280 |
| AlphaFold 3 | 1536 | 2048 |

The reports do not specify MSA depth, sample count, padding, exact shapes, or
whether runs started in fresh processes. They are finite tested limits, not
guarantees that every smaller input fits or every larger input fails. Counts
refer to public/model token buckets; OpenDDE's expanded structural-token count
used by its peak law is a separate axis and is never compared directly with
this table.

Separate failure reports place ESMFold2 failures around 640–768 tokens and
OpenDDE failures at 1024 tokens on 24 GB cards. A 1435-token Boltz-2 run on a
4090 failed with a CUDA kernel-launch OOM while only 2.0 GiB of a 21.2 GiB
allocator pool was in use. Boltz-2 also failed at 1536 on A6000 despite some
24 GB successes in that range. Cumulative allocations and memory outside the
JAX pool are possible explanations for the discrepancy; the reports do not
establish its cause.

An ECC-enabled 4090 exposing about 22 GiB has less headroom than its nominal
24 GB class. A6000 is an option to investigate, not a promised solution.
Warnings recommend a fresh process and, if necessary, a larger card. No
precision default, RNG, MSA, sampling schedule, or actual input is reduced to
make a run fit.

Planning and execution use the same `observed_capacity` assessment. It records
the model, token axis/count, actual card and allocator/planning budget when
known, device kind, MSA rows when actually known, sample count, and shape
profile. Unknown observation conditions remain explicit. Planning stays
JAX-free: a supplied memory budget selects conditional observations and never
establishes an actual GPU's identity or capacity. Runtime uses the actual card
capacity first, so a small allocator pool on a larger card does not relabel
that device as a 24 GB card.

The run manifest's `memory.observed_capacity` supplements the fitted decision.
A fitted `fits` verdict still refers to that law and budget; the separate
capacity warning can identify risks it did not measure. Reports outside the
tested limits warn rather than inventing a hard refusal. Native fitted
over-budget refusals remain in force.

CUDA module loading or kernel-launch OOMs receive outside-pool guidance:
leave more memory outside the allocator, for example by lowering a 0.9
fraction to 0.8–0.85, and restart the process. Ordinary allocator-buffer OOMs
retain their existing pool diagnosis. Pool capacity plus post-failure usage
does not establish the peak memory requirement.
