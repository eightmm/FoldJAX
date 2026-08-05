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

The production execution default keeps sampler scan enabled. The automatic
chunk policy uses chunk 256 for 769–1024 tokens because the measured 952-token
case reduced peak memory by 48–54% with warm latency within 3.2% of the unchunked
controlled run and slightly faster latency on the official five-sample run.
