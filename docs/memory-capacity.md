# Observed GPU capacity

Reported capacity and fitted peak estimates answer different questions. The
frozen peak laws retain their calibration data and coefficients. User-reported
successful buckets are not peak measurements from which to fit new
coefficients. The dated admission changes below distinguish a fitted peak
from an operating limit or an uncertainty allowance.

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

The [2026-10-20 follow-up](cluster-memory-2026-10-20.md) reports RTX 4090
Boltz-2 at 1536 tokens succeeding at fraction 0.8 or below, and OpenDDE at
1024 tokens succeeding with structural capacity 1856. These are conditional
observations, not replacements for the historical fitted laws. ESMFold2's
A6000 1792-token success was single-chain warming; the later actual-input
checks below supersede it for the deployment's default admission policy.

## Reports and admission changes from 2026-10-21

The fixed installation at `f787bdda` passed the cluster's 21-case validation
on RTX 3090, RTX 4090, RTX A5000 and RTX A6000. That finite panel does not
establish every input's capacity or local real-weight GPU reproduction.

| Model/configuration | Reported completion | Reported failure |
| --- | --- | --- |
| ESMFold2, 24 GB RTX 3090/4090/A5000 | 1024-token warming | No new larger-input failure bound supplied |
| ESMFold2, A6000 48 GB | 1536-token warming and multichain gfp6 | 1792 tokens, both single-chain gfp7sc (1726 real residues) and multichain gfp7 |
| OpenDDE, A6000 48 GB, gfp6, pool fraction 0.85 | 1536 public tokens / 2752 structural tokens / 12288 atoms / 2048 MSA rows, using `--memory-check warn` | gfp7, fitted nominal peak 52.2 GiB |

ESMFold2's failure requested a 29.16 GiB block from `jit__run`. That is a
failed allocation, not a measurement of the whole run's peak. The historical
continuous peak fit is unchanged. On a physically identified 48 GB class
device, the default single-device released path now refuses **executable
token counts of 1792 or more** before the LM forward. The count includes
serving padding, so 1726 residues padded to 1792 reaches that refusal. This
is an explicitly conservative operating policy from finite outcomes, not a
proof of a physical boundary or a promise that every smaller input fits.
`--memory-check warn` runs it anyway. Existing unmeasured alternatives such
as context parallelism, buffered LM states or solver-free alignment retain
their advisory treatment; no new capacity is claimed for them.

OpenDDE's gfp6 peak was **42,529,933,568 bytes (39.61 GiB)**, with
37,961,062,880 temporary bytes and a reported 980-second execution. The
frozen law predicts 39.65 GiB at 2752 structural tokens; its historical
1171 MiB uncertainty allowance raises that to 40.79 GiB. The approximately
40.3 GiB allocator pool already excludes the device memory reserved outside
the pool. Applying another factor of 0.9 reduced admission to 36.3 GiB and
falsely refused this completed run.

For OpenDDE on identified 24/48 GB physical classes, admission now compares
against the **full effective pool budget**, including any smaller explicit
`--memory-budget-gib` ceiling. The fitted coefficients and uncertainty
allowance stay in the record. An upper estimate within that ceiling fits;
a nominal estimate that fits while its allowance crosses the ceiling is
`unknown` and proceeds with an uncertainty warning. A nominal estimate above
the ceiling still refuses inside the fitted domain: the 52.2 GiB gfp7
prediction remains over a 40.3 GiB pool. Below the fitted domain, the existing
rule against refusing an extrapolation remains in force.

These physical classes use the existing observed-capacity bins, 20–24.5 GiB
and 40–48.5 GiB. Unidentified cards, other classes and the historical 96 GB
measurements retain the original factor of 0.9; a 40 GiB planning override
does not establish that a card is an A6000. No new repeat spread is invented
from the single gfp6 peak, and no precision, model graph or cache key changes
are involved.

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
tested limits normally warn. ESMFold2's dated 48 GB operating refusal is the
explicit exception: `refusal` records it for an identified card, while
`conditional_refusal` records it for planning without a physical device.
Planning remains `unknown` about an actual card and explains that the same
policy would refuse on that class; it does not initialize JAX to infer one.

## Allocator policy and failures

An explicit `--mem-fraction` takes precedence over an existing allocator
environment variable, which takes precedence over the automatic default.
Without either fraction environment variable, FoldJAX checks total visible CUDA
device capacities before JAX backend initialization. All visible GPUs must have
at least 44 GiB and less than 52 GiB for the automatic fraction to be 0.85.
Other capacity classes, mixed classes and unknown capacities use 0.9. This
policy does not change compile-cache keys or model static execution profiles.
A single-model Boltz-2 command on a confirmed RTX 4090 now uses 0.8 instead;
unknown mixed GPU types and multi-model commands keep the general policy.
See [configuration](configuration.md#allocator-policy).

An explicit CUDA module-loading OOM receives outside-pool guidance: leave more
memory outside the allocator, for example by lowering a 0.9 fraction to
0.8–0.85, and restart the process. A generic CUDA kernel-launch OOM is
ambiguous; it does not by itself prove outside-pool exhaustion or justify an
instruction to increase the fraction to 0.95. Ordinary allocator-buffer OOMs
retain their pool diagnosis. Pool capacity plus post-failure usage does not
establish the failed allocation's peak requirement.

A `gpusolverDnCreate(&handle) failed` / `cusolverDnCreate(...) failed` message
also receives CUDA-library/context guidance. Handle creation alone does not
establish OOM: the API retains a non-OOM error unless an allocation-failure
status is present, and bucket warming stops only for confirmed OOM. Inspect
the original status before attributing it to outside-pool exhaustion.
