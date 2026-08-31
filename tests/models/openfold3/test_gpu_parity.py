"""GPU parity for both the algebraic reference and the shipped kernel path.

The strict gate compares upstream Torch and FoldJAX with full-precision matmuls
and FoldJAX's XLA triangle attention. That isolates parameter mapping and model
algebra from backend scheduling. A separate, multi-seed gate exercises the real
production pair: Torch ``high`` and FoldJAX ``high`` with cuEquivariance. TF32
and fused-kernel reduction order are not bitwise equivalent, so that gate uses a
measured, scale-normalized envelope instead of pretending the strict 1e-4 tensor
tolerance applies to two different accelerator programs.

The numerical comparisons skip without a GPU. A CPU-safe test still checks that
the production runner enters the real ``high`` scope and restores the caller's
setting afterwards.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.openfold3.bridge.torch_mapping import map_pairformer_stack
from foldjax.models.openfold3.inference import (
    _MATMUL_PRECISION,
    openfold3_precision,
)
from foldjax.models.openfold3.models.pairformer import pairformer_stack
from foldjax.models.openfold3.models.triangle_attention import _default_backend

pytestmark = pytest.mark.torch_parity

C_S, C_Z, N = 10, 8, 6
HEADS = 2
STRICT_NORMALIZED_ERROR = 1e-5
PRODUCTION_NORMALIZED_ERROR = 5e-3


def _gpu_devices():
    try:
        return jax.devices("gpu")
    except RuntimeError:
        return []


requires_gpu = pytest.mark.skipif(
    not _gpu_devices(), reason="no GPU device visible to JAX"
)


def _torch():
    import torch

    return torch


def _stack():
    from openfold3.core.model.latent.pairformer import PairFormerStack

    return PairFormerStack(
        c_s=C_S,
        c_z=C_Z,
        c_hidden_pair_bias=4,
        no_heads_pair_bias=HEADS,
        c_hidden_mul=6,
        c_hidden_pair_att=4,
        no_heads_pair=HEADS,
        no_blocks=2,
        transition_type="swiglu",
        transition_n=2,
        pair_dropout=0.0,
        fuse_projection_weights=False,
        blocks_per_ckpt=None,
        inf=1e9,
    )


def _inputs(torch, *, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    s = torch.randn((1, N, C_S), generator=generator)
    z = torch.randn((1, N, N, C_Z), generator=generator)
    single = torch.ones(1, N)
    pair = single[..., None] * single[..., None, :]
    return s, z, single, pair


def _asarray(value):
    detach = getattr(value, "detach", None)
    if detach is not None:
        value = detach().cpu()
    to_numpy = getattr(value, "numpy", None)
    return jnp.asarray(to_numpy() if to_numpy is not None else value)


def _call(params, s, z, single, pair, *, runner=pairformer_stack):
    return runner(
        _asarray(s),
        _asarray(z),
        params,
        single_mask=_asarray(single),
        pair_mask=_asarray(pair),
        no_heads_pair=HEADS,
        no_heads_pair_bias=HEADS,
    )


def _run_reference_on(device, params, s, z, single, pair):
    with jax.default_device(device), jax.default_matmul_precision("highest"):
        return _call(params, s, z, single, pair)


@openfold3_precision
def _run_production_on(device, params, s, z, single, pair, *, runner=pairformer_stack):
    with jax.default_device(device):
        return _call(params, s, z, single, pair, runner=runner)


def _run_torch(module, inputs, torch, *, precision: str):
    previous_precision = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision(precision)
        with torch.no_grad():
            return module(
                s=inputs[0],
                z=inputs[1],
                single_mask=inputs[2],
                pair_mask=inputs[3],
            )
    finally:
        torch.set_float32_matmul_precision(previous_precision)


def _assert_normalized_close(got, want, *, limit: float, name: str) -> None:
    actual = np.asarray(got, dtype=np.float64)
    expected = np.asarray(want, dtype=np.float64)
    absolute_error = float(np.max(np.abs(actual - expected)))
    scale = max(float(np.max(np.abs(expected))), np.finfo(np.float32).tiny)
    normalized_error = absolute_error / scale
    assert np.isfinite(normalized_error), f"{name} error is not finite"
    assert normalized_error <= limit, (
        f"{name} normalized max error {normalized_error:.6g} exceeds {limit:.6g} "
        f"(max_abs={absolute_error:.6g}, reference_scale={scale:.6g})"
    )


def test_gpu_runner_uses_and_restores_the_shipped_high_scope() -> None:
    """The GPU helper must exercise production's scope, even in CPU-only CI."""
    seen: list[str | None] = []

    def record(s, z, params, **kwargs):
        del params, kwargs
        seen.append(jax.config.jax_default_matmul_precision)
        return s, z

    s = np.zeros((1, N, C_S), dtype=np.float32)
    z = np.zeros((1, N, N, C_Z), dtype=np.float32)
    single = np.ones((1, N), dtype=np.float32)
    pair = np.ones((1, N, N), dtype=np.float32)
    outside = jax.config.jax_default_matmul_precision

    actual_s, actual_z = _run_production_on(
        jax.devices("cpu")[0],
        None,
        s,
        z,
        single,
        pair,
        runner=record,
    )

    assert _MATMUL_PRECISION == "high"
    assert seen == ["high"]
    assert jax.config.jax_default_matmul_precision == outside
    np.testing.assert_array_equal(actual_s, s)
    np.testing.assert_array_equal(actual_z, z)


@requires_gpu
def test_reference_pairformer_parity_holds_on_gpu(
    openfold3_source: Path, randomized, monkeypatch
) -> None:
    """Full-precision XLA isolates mapping and algebra at the strict budget."""
    monkeypatch.setenv("OPENFOLD3_TRIANGLE_BACKEND", "xla")
    torch = _torch()
    module = randomized(_stack()).cuda()
    inputs = tuple(value.cuda() for value in _inputs(torch))
    expected = _run_torch(module, inputs, torch, precision="highest")
    params = map_pairformer_stack(dict(module.state_dict()))
    actual = _run_reference_on(_gpu_devices()[0], params, *inputs)

    for got, want, name in zip(actual, expected, ("s", "z"), strict=True):
        _assert_normalized_close(
            got,
            want.detach().cpu().numpy(),
            limit=STRICT_NORMALIZED_ERROR,
            name=f"strict GPU {name}",
        )


@requires_gpu
@pytest.mark.parametrize("seed", (0, 1, 2))
def test_production_pairformer_stays_within_numeric_envelope(
    openfold3_source: Path, randomized, monkeypatch, seed: int
) -> None:
    """The shipped TF32/cuEq path stays inside a mutation-sensitive envelope."""
    monkeypatch.delenv("OPENFOLD3_TRIANGLE_BACKEND", raising=False)
    assert _default_backend() == "cueq"
    torch = _torch()
    module = randomized(_stack(), seed=seed).cuda()
    inputs = tuple(value.cuda() for value in _inputs(torch, seed=seed + 1000))
    expected = _run_torch(module, inputs, torch, precision="high")
    params = map_pairformer_stack(dict(module.state_dict()))
    actual = _run_production_on(_gpu_devices()[0], params, *inputs)

    for got, want, name in zip(actual, expected, ("s", "z"), strict=True):
        _assert_normalized_close(
            got,
            want.detach().cpu().numpy(),
            limit=PRODUCTION_NORMALIZED_ERROR,
            name=f"production seed={seed} {name}",
        )


@requires_gpu
def test_reference_gpu_and_cpu_agree(
    openfold3_source: Path, randomized, monkeypatch
) -> None:
    """Full-precision XLA agrees across devices at the strict budget."""
    monkeypatch.setenv("OPENFOLD3_TRIANGLE_BACKEND", "xla")
    torch = _torch()
    module = randomized(_stack())
    inputs = _inputs(torch)
    params = map_pairformer_stack(dict(module.state_dict()))

    cpu = _run_reference_on(jax.devices("cpu")[0], params, *inputs)
    gpu = _run_reference_on(_gpu_devices()[0], params, *inputs)
    for got, want, name in zip(gpu, cpu, ("s", "z"), strict=True):
        _assert_normalized_close(
            got,
            want,
            limit=STRICT_NORMALIZED_ERROR,
            name=f"CPU/GPU {name}",
        )
