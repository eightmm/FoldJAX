"""Deterministic reductions asked for per executable, not per process.

Running a port under the process-wide environment
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
why the option carries the second key. Every port pays for the second key in
its own trunk, because it reroutes every Triton gemm to cuBLAS; the per-port
wall-time row is a promotion gate, not a reason to grow a second spelling.

The promise covers the ops XLA emits -- scatters, reductions, autotuned gemm
selection. Custom calls are outside it: the cuEquivariance triangle kernels,
the tokamax attention and GLU kernels, and the Pallas elementwise divide are
compiled by their own toolchains, and their repeatability is observed rather
than documented.

This module is deliberately free of JAX imports so that a backend can read it
while planning a run, before any accelerator client exists.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from foldjax.models._jit_pool import BoundedJitPool

#: XLA compile options for a run that asked for `deterministic=on`.
DETERMINISTIC_COMPILER_OPTIONS: dict[str, Any] = {
    "xla_gpu_deterministic_ops": True,
    "xla_gpu_enable_triton_gemm": False,
}


def compiler_options(
    *,
    deterministic: bool,
    base: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """The XLA options one executable is built under, or `None` for "none".

    `None` rather than `{}` is the whole point: an owner that asks for nothing
    calls `jax.jit` with exactly the arguments it always did, so the default
    program is the one every recorded measurement describes rather than a
    fresh compile under an empty option map.

    `base` is whatever the port already asked for -- Boltz-2's bfloat16 run
    passes `{"xla_allow_excess_precision": False}` -- and survives both
    answers; the deterministic keys are merged on top of it. The result is
    always a fresh dict, so neither the caller's `base` nor the constant can
    be edited through the return value.
    """
    if deterministic:
        return {**(base or {}), **DETERMINISTIC_COMPILER_OPTIONS}
    return dict(base) if base else None


def policy_pools(
    function: Callable[..., Any],
    **pool_kwargs: Any,
) -> tuple[BoundedJitPool, BoundedJitPool]:
    """The two executable owners one function needs, `(default, deterministic)`.

    A separate owner rather than an option on the call: the setting is part of
    how the executable is built, so the two cannot share one cache entry, and
    a run that asked for repeatable reductions must never be handed the
    program compiled without them. Neither pool compiles anything until it is
    called, and the default one keeps the arguments it always had.

    A `compiler_options` keyword is taken as the port's own base and merged,
    not replaced.
    """
    from foldjax.models._jit_pool import BoundedJitPool

    base = pool_kwargs.pop("compiler_options", None)
    default = BoundedJitPool(
        function,
        compiler_options=compiler_options(deterministic=False, base=base),
        **pool_kwargs,
    )
    deterministic = BoundedJitPool(
        function,
        compiler_options=compiler_options(deterministic=True, base=base),
        **pool_kwargs,
    )
    return default, deterministic


def select(
    pools: tuple[BoundedJitPool, BoundedJitPool],
    deterministic: bool,
) -> BoundedJitPool:
    """The owner in `pools` that carries this run's reduction policy."""
    default, repeatable = pools
    return repeatable if deterministic else default
