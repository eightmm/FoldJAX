# Observed GPU capacity

Reported capacity and fitted peak estimates answer different questions. The
frozen peak laws retain their calibration data, coefficients, and refusal
rules. User-reported successful buckets are not peak measurements from which
to fit new coefficients.

## Reports from 2026-10-17 (historical)

The following earlier reports are retained with their original conditions and
limits. Later reports below update particular tested configurations.

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

## Reports from 2026-10-19

The later cluster report gives these maximum successful token buckets:

| Model | 24 GB class | RTX A6000, 48 GB |
| --- | --- | --- |
| Boltz-2 | 1536 on RTX 3090/RTX A5000; 1280 on RTX 4090 | 2048 with pool fraction 0.85 |
| ESMFold2 | 1024 | 1792 |
| OpenDDE | 896 | 1280 |

The deployment used `--padding` for AlphaFold 3 and ESMFold2, and
`--fixed-msa 2048` for Boltz-2, Protenix, OpenFold3 and OpenDDE. The reported
4090 runs had ECC disabled. Other conditions, including sample counts and the
exact input compositions, are not fully specified; these reports do not
establish that every input below a successful bucket fits.

On A6000, Boltz-2 reportedly completed 1536, 1792 and 2048 tokens with fraction
0.85, while a run with fraction 0.9 failed at 1536. This motivates reserving
more headroom outside the allocator on the 48 GB class, but does not identify
the failed allocation's owner or prove the same outcome for every A6000 run.
These are user-reported A6000 results, not an independent local A6000
reproduction. Separate measurements on 96 GB hardware do not verify 48 GB or
24 GB capacity. These successful-bucket reports do not change fitted peak
coefficients, precision, RNG or sampling settings.

## Using reports in planning

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

## Allocator policy and failures

An explicit `--mem-fraction` takes precedence over an existing allocator
environment variable, which takes precedence over the automatic default.
Without either fraction environment variable, FoldJAX checks total visible CUDA
device capacities before JAX backend initialization. All visible GPUs must have
at least 44 GiB and less than 52 GiB for the automatic fraction to be 0.85.
Other capacity classes, mixed classes and unknown capacities use 0.9. This
policy does not change compile-cache keys or model static execution profiles.
See [configuration](configuration.md#allocator-policy).

An explicit CUDA module-loading OOM receives outside-pool guidance: leave more
memory outside the allocator, for example by lowering a 0.9 fraction to
0.8–0.85, and restart the process. A generic CUDA kernel-launch OOM is
ambiguous; it does not by itself prove outside-pool exhaustion or justify an
instruction to increase the fraction to 0.95. Ordinary allocator-buffer OOMs
retain their pool diagnosis. Pool capacity plus post-failure usage does not
establish the failed allocation's peak requirement.
