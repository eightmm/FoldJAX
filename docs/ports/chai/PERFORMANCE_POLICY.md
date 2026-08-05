# Performance default policy

A runtime optimization may become a balanced production default only after
scientific parity passes and one of these Pareto gates passes on the same GPU,
inputs, precision, weights, seeds, and cached static shapes:

- latency gate: median warm latency improves by at least 3% over three or more
  synchronized runs and peak device memory increases by no more than 5%;
- memory gate: peak device memory improves by at least 10% and median warm
  latency regresses by no more than 5%;
- the selected gate holds at representative small, medium, and large sizes.

Cold compilation time is reported separately and does not replace the cache-hit
gate. An optimization that passes only some sizes must be size-gated. Only a
change that passes the latency gate may be described as a faster default.

The production trunk remains scan-based. No Chai execution variant is promoted
from a microbenchmark alone; a default change requires the full native inference
path and the existing Torch-compatibility gates.
