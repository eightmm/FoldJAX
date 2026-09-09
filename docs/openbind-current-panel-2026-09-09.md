# OpenBind native-default panel preflight

The existing `bench/openfold3_runner.yml` is an equal-workload benchmark
configuration: it sets ten recycles and disables confidence-head offload.
It must not be described as an unchanged native-default execution.

Pinned OpenFold3 v0.5.0 model configuration sets three recycles, five samples
and 200 rollout steps. The forward-tape adapter explicitly requires this
four-trunk-pass n5/200 profile. New `bench/openfold3_native_runner.yml` selects
the publisher `predict` preset and seed 101 only, without overriding model
settings. The existing `--openfold3-runner-yaml` option can select it; the
historical runner and its results remain unchanged.

This is invocation preparation, not a GPU result. Effective precision,
dynamic chunk choices, actual forward tape, independent preprocessing,
coordinates/confidence and warm memory/time still require execution and
verification. The existing forward adapter explicitly lacks preprocessing RNG
proof; its successful adaptation is not full model admission.

The caller must also record `--num-recycles 3` when using this runner;
`bench.run_upstream` otherwise inherits the equal-workload schedule of ten
and correctly rejects the mismatch. Its command builder checks the preset,
seed, recycle count and rollout steps, then passes the selected runner and
five diffusion samples to upstream without injecting a model recycle override.

Verification: eight selected OpenFold3 harness tests pass (47 unrelated tests
deselected), including the new real-file runner test, correct command wiring
and rejection of a falsely recorded ten-recycle schedule. Scoped Ruff passes.
This proves command construction, not execution-time effective settings.
