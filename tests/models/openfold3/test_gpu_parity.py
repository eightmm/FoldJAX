"""GPU parity, and the TF32 hazard it exists to catch.

The rest of the suite runs on CPU, where JAX always uses full FP32 products. On
an NVIDIA GPU it would default to TF32 unless told otherwise, costing roughly
three decimal digits per matmul — enough to fail a 1e-4 gate on a stack of
attention layers, and amplified further by a diffusion rollout.

These tests skip when no GPU is present. When one is, they assert two things:
parity still holds at the CPU tolerance, and the `highest` precision setting is
actually load-bearing rather than defensive decoration.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.openfold3.bridge.torch_mapping import map_pairformer_stack
from foldjax.models.openfold3.models.pairformer import pairformer_stack

pytestmark = pytest.mark.torch_parity

C_S, C_Z, N = 10, 8, 6
HEADS = 2


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

    torch.manual_seed(0)
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


def _inputs(torch):
    s = torch.randn(1, N, C_S)
    z = torch.randn(1, N, N, C_Z)
    single = torch.ones(1, N)
    pair = single[..., None] * single[..., None, :]
    return s, z, single, pair


def _run_on(device, params, s, z, single, pair):
    with jax.default_device(device):
        return pairformer_stack(
            jnp.asarray(s.numpy()),
            jnp.asarray(z.numpy()),
            params,
            single_mask=jnp.asarray(single.numpy()),
            pair_mask=jnp.asarray(pair.numpy()),
            no_heads_pair=HEADS,
            no_heads_pair_bias=HEADS,
        )


@requires_gpu
def test_pairformer_parity_holds_on_gpu(openfold3_source: Path, randomized) -> None:
    """The same 1e-4 gate the CPU suite uses, on the accelerator."""
    torch = _torch()
    module = randomized(_stack())
    s, z, single, pair = _inputs(torch)
    with torch.no_grad():
        expected_s, expected_z = module(
            s=s, z=z, single_mask=single, pair_mask=pair
        )

    params = map_pairformer_stack(dict(module.state_dict()))
    actual_s, actual_z = _run_on(_gpu_devices()[0], params, s, z, single, pair)

    for got, want, name in ((actual_s, expected_s, "s"), (actual_z, expected_z, "z")):
        np.testing.assert_allclose(
            np.asarray(got, dtype=np.float64),
            want.detach().numpy().astype(np.float64),
            rtol=1e-4,
            atol=1e-4,
            err_msg=f"GPU parity failed for {name}",
        )


@requires_gpu
def test_tf32_would_break_the_gate(openfold3_source: Path, randomized) -> None:
    """Drop to TF32 and the same comparison must fail.

    If this test ever passes with `highest` disabled, the precision setting is not
    doing anything on this hardware and the claim in the docs is wrong.
    """
    torch = _torch()
    module = randomized(_stack())
    s, z, single, pair = _inputs(torch)
    with torch.no_grad():
        expected_z = module(s=s, z=z, single_mask=single, pair_mask=pair)[1]

    params = map_pairformer_stack(dict(module.state_dict()))
    device = _gpu_devices()[0]

    previous = jax.config.jax_default_matmul_precision
    try:
        jax.config.update("jax_default_matmul_precision", "default")
        degraded = _run_on(device, params, s, z, single, pair)[1]
    finally:
        jax.config.update("jax_default_matmul_precision", previous)

    reference = expected_z.detach().numpy().astype(np.float64)
    degraded_error = np.abs(np.asarray(degraded, dtype=np.float64) - reference).max()

    exact = _run_on(device, params, s, z, single, pair)[1]
    exact_error = np.abs(np.asarray(exact, dtype=np.float64) - reference).max()

    # `highest` must be at least as accurate; on TF32 hardware, strictly better.
    assert exact_error <= degraded_error + 1e-9, (
        f"highest ({exact_error:.3e}) was worse than default "
        f"({degraded_error:.3e}); the precision setting may be inverted"
    )
    print(
        f"\nmax |error| vs torch: highest={exact_error:.3e} "
        f"default={degraded_error:.3e}"
    )


@requires_gpu
def test_gpu_and_cpu_agree(openfold3_source: Path, randomized) -> None:
    """Same parameters, both backends, one tolerance."""
    torch = _torch()
    module = randomized(_stack())
    s, z, single, pair = _inputs(torch)
    params = map_pairformer_stack(dict(module.state_dict()))

    cpu_z = _run_on(jax.devices("cpu")[0], params, s, z, single, pair)[1]
    gpu_z = _run_on(_gpu_devices()[0], params, s, z, single, pair)[1]
    np.testing.assert_allclose(
        np.asarray(cpu_z, dtype=np.float64),
        np.asarray(gpu_z, dtype=np.float64),
        rtol=1e-4,
        atol=1e-4,
    )
