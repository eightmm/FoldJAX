"""Deterministic reductions asked for per executable, not per process.

Running this port under the process-wide environment
``XLA_FLAGS=--xla_gpu_deterministic_ops=true`` makes it bitwise repeatable
across processes and moves one master-panel case into the pass band, at 13% of
its wall time at 1,003 tokens (``docs/protenix-master-panel-2026-09-09.md``).

An environment variable is the wrong carrier for that. It is read once when
the process starts, so it cannot say "this prediction and not the next one",
and it reaches every other model in a benchmark process -- including the ones
whose recorded numbers were measured without it. The setting belongs on the
executable that wants it, which is what this constant is for.

One dict, in one place, so nothing outside this module spells a flag name.
At 3,012 tokens XLA's deterministic autotuner has no candidate for two batched
``gemm_fusion_dot`` instructions of shape ``f32[5,16,4,96,48]`` and the
compile fails ("Failed to get configs for 2 of 171 instructions") under both
``xla_gpu_deterministic_ops`` and the narrower
``xla_gpu_exclude_nondeterministic_ops``. Disabling Triton gemms alongside
routes those two through cuBLAS and compiles: 635 s warm against 579 s
default at 3,012 tokens, peak unchanged (jobs 665/747, 2026-09-10). That is
why the option carries the second key.
"""

from __future__ import annotations

from typing import Any

#: XLA compile options for a run that asked for `deterministic=on`.
DETERMINISTIC_COMPILER_OPTIONS: dict[str, Any] = {
    "xla_gpu_deterministic_ops": True,
    "xla_gpu_enable_triton_gemm": False,
}
