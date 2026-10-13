# GPU kernel compatibility

FoldJAX's Python/JAX installation, a working JAX GPU backend, and support for
an individual fused kernel are separate requirements. `foldjax doctor` reports
the selected runtime and adds architecture-specific kernel guidance. The
supported dependency versions and CUDA installation choices are in
[install.md](install.md).

## CUDA device identity

Pallas-Triton in the pinned JAX version recognizes a finite list of GPU names.
FoldJAX registers unlisted local CUDA devices using their reported compute
capability, without pretending a 3090 or A6000 is an A100. Existing JAX names,
custom registry entries, and deliberate abstract/AOT targets remain intact.
CPU, ROCm and devices without a readable CUDA capability are not registered.
Registration runs before prediction, including cache warm, and at GLU, pair,
and attention kernel entry points for direct model callers. Explicit device
contexts (including JAX's platform-name strings) and AlphaFold 3's selected
platform/device are registered even when the process-default backend is CPU.

## Shared memory and global memory

A fused GLU needs per-block shared memory as well as global device memory.
Increasing `XLA_CLIENT_MEM_FRACTION` cannot increase a block's shared-memory
limit. The common Tokamax GLU applies the same heuristic correction as the
AlphaFold 3 backend on these architectures:

| Compute capability | Per-block opt-in shared-memory limit |
| --- | ---: |
| 8.6 (consumer/workstation Ampere) | 99 KiB |
| 8.9 (Ada) | 99 KiB |
| 12.0 / 12.1 (consumer/workstation Blackwell) | 99 KiB |

The capability includes its minor version: 8.0/8.7 have a larger budget and
must not inherit the 8.6 limit. The correction lowers the heuristic's pipeline
stage count first, then its tile dimensions if necessary; it retains the K
reduction step and leaves configurations already within budget alone. It
changes a launch schedule, not precision or the requested kernel backend.
Nested and concurrent device contexts keep separate budgets. Autotuned
configurations remain governed by Tokamax's tuning policy.

The limits above are per block, not per multiprocessor. See the
[NVIDIA CUDA compute-capability table](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html).
Shared-memory resource errors retain their original cause and are excluded
from FoldJAX's allocator-OOM advice. `RESOURCE_EXHAUSTED` alone does not
establish a memory failure: thread, file and disk limits do not receive GPU
memory advice. Nested all-OOM exception groups are recognized; mixed groups
retain their original cause, and multiple failed attempts do not establish
one allocation total. Memory admission/diagnosis reads the explicitly selected
device when one is configured, rather than always reading device zero.

## Turing and older accelerators

All six FoldJAX models require NVIDIA compute capability `sm_80+` for GPU
inference. Turing (`sm_75`, for example Quadro RTX 8000) is rejected before
backend dispatch, with the same reason shown by `foldjax doctor`. Selecting
XLA kernels or float32 does not bypass this support policy. CPU execution
remains supported. A kernel being able to compile does not establish full-model
accuracy, performance, or bitwise agreement on a GPU.

CUDA installation has its own hardware and driver limits. The JAX
[installation guide](https://docs.jax.dev/en/latest/installation.html) documents
CUDA 12/13 requirements; a software compatibility shim cannot add an
architecture omitted from the installed compiler or driver.

## Selected devices, memory and cache warming

An explicit `jax.default_device` context or `jax_default_device` setting governs
execution, runtime metadata, memory statistics and the compile-cache namespace.
AlphaFold 3 inherits that device unless its `platform`/`device` options explicitly
select another. Runtime metadata includes the selected device identity. This
correction changes cache namespaces; warm the corrected profiles before a
latency-sensitive run.

Private new cache directories under root-owned sticky `/tmp` are supported.
An existing ancestor is checked as a parent before creation; the completed
cache directory is checked as a leaf. `/tmp` itself, another user's directory
and non-sticky shared writable ancestors remain untrusted without explicit
shared-cache opt-in.

Allocator limits and peaks come from the selected device's JAX statistics.
`memory.card_bytes` comes from the CUDA driver using that device's stream,
rather than dividing the pool by a live environment variable. If the driver
cannot report capacity (including older drivers without `cuStreamGetDevice`),
the capacity is unknown. The allocator budget remains usable. `memory.selected`
is an admitted memory-policy profile name, not a GPU identifier; it is `null`
when the admission result is unknown, for example below the fitted size range.

AlphaFold 3's first trace can include many Tokamax candidate compilations and
measurements. Prepare frequently used size buckets and sampling settings with
the exact prediction profile:

```bash
foldjax cache warm --model alphafold3 --input job.json --cache-dir /tmp/foldjax-cache
```

The persisted tuning results are reused on matching profiles. On a device other
than the process-default backend's first GPU, persisted tuning compiles and
measures candidates synchronously in the selected device context. Uncontrolled
inline worker tuning remains refused there. Tokamax 0.0.13 already excludes
failed candidates, including OOMs, from its fastest successful configuration;
an operation whose candidates all fail cannot produce a valid tuning result.

## Verification scope

Regression tests cover capability formats, unlisted CUDA names, multiple GPU
kinds, preservation of known/custom entries, prediction/cache-warm wiring,
shared-GLU launch correction, nested/concurrent budgets, and resource-error
classification. CPU interpret tests exercise Pallas indexing and arithmetic.
Actual GPU kernel and full-model checks must be reported separately, with the
physical device and precision named. No claim covers every GPU or environment.
