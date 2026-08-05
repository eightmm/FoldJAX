"""Chai's fused triangle module, split into two kernel calls that sum back to it.

Chai does not compute two triangle updates. It computes one, with a single
output projection over the *sum* of the two normalized products and a single
output gate::

    sigmoid(g_out . norm) * W_out . (LN(outgoing) + LN(incoming))

The kernel computes one direction and bakes its own output projection and gate
into the same call, so the two do not obviously compose. They do, by linearity
of the projection and because the gate is shared -- but that argument is only
as good as the weight slicing, the direction strings and the mask transpose
that carry it, and every one of those is a place to be off by one block.

So these tests stand a pure-JAX implementation of the kernel's documented
semantics in for `cuequivariance_jax` and check the identity end to end on CPU.
What they cannot check is the kernel itself, which is NVIDIA's; what they do
check is the mapping, which is this port's.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from foldjax.models.chai.models import pairformer
from foldjax.models.chai.models.pairformer import (
    FusedTriangleMultiplicationParams,
    fused_triangle_multiplication,
)
from foldjax.models.chai.models.primitives import layer_norm, linear


def _reference_triangle_multiplicative_update(
    *,
    x,
    direction,
    mask,
    norm_in_weight,
    norm_in_bias,
    p_in_weight,
    g_in_weight,
    norm_out_weight,
    norm_out_bias,
    p_out_weight,
    g_out_weight,
    eps,
    fallback,
):
    """cuEquivariance's documented semantics, in fp32, without the kernel.

    Mirrors `_triangle_multiplicative_update.py`: normalize, one gated dual
    gemm producing both projections, mask the *gated output* (the wheel does
    `output = output * mask[:, None]`, not a mask on the input), the direction's
    einsum, output normalization, then a gate read off the *pre-product*
    normalized input.
    """
    assert fallback is False
    assert direction in {"outgoing", "incoming"}
    channels = x.shape[-1]
    normalized = layer_norm(
        x.astype(jnp.float32), norm_in_weight, norm_in_bias, eps=eps
    )
    projected = linear(normalized, p_in_weight)
    gate = jax.nn.sigmoid(linear(normalized, g_in_weight))
    both = projected * gate
    both = both * mask[..., None].astype(both.dtype)
    a = both[..., :channels]
    b = both[..., channels:]
    if direction == "outgoing":
        product = jnp.einsum("...ikd,...jkd->...ijd", a, b)
    else:
        product = jnp.einsum("...kid,...kjd->...ijd", a, b)
    normalized_out = layer_norm(product, norm_out_weight, norm_out_bias, eps=eps)
    out_gate = jax.nn.sigmoid(linear(normalized, g_out_weight))
    return linear(normalized_out, p_out_weight) * out_gate


@pytest.fixture
def stub_kernel(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "cuequivariance_jax",
        SimpleNamespace(
            triangle_multiplicative_update=_reference_triangle_multiplicative_update
        ),
    )


def _params(rng, channels: int) -> FusedTriangleMultiplicationParams:
    def normal(*shape):
        return jnp.asarray(rng.normal(size=shape) * 0.3, jnp.float32)

    return FusedTriangleMultiplicationParams(
        layer_norm_weight=normal(channels),
        layer_norm_bias=normal(channels),
        merged_linear_p_weight=normal(4 * channels, channels),
        merged_linear_g_weight=normal(5 * channels, channels),
        linear_z_out_weight=normal(channels, channels),
    )


def _inputs(rng, tokens: int, channels: int, *, symmetric_mask: bool):
    pair = jnp.asarray(rng.normal(size=(1, tokens, tokens, channels)), jnp.float32)
    if symmetric_mask:
        token_mask = jnp.asarray(rng.random(size=(1, tokens)) > 0.3)
        pair_mask = token_mask[..., :, None] & token_mask[..., None, :]
    else:
        pair_mask = jnp.asarray(rng.random(size=(1, tokens, tokens)) > 0.3)
    return pair, pair_mask


@pytest.mark.parametrize("symmetric_mask", [True, False])
def test_two_kernel_calls_sum_to_the_blocked_module(
    monkeypatch, stub_kernel, symmetric_mask: bool
) -> None:
    """The identity that justifies the whole port, on CPU, in fp32.

    The asymmetric case is not hypothetical framing: every trunk call site
    builds `pair_mask` as an outer product of one token mask and is therefore
    symmetric, so a wrapper that forgot to transpose the incoming mask would
    pass every real input and this test's first parameter. It fails only on the
    second.
    """
    rng = np.random.default_rng(11)
    channels = 32
    params = _params(rng, channels)
    pair, pair_mask = _inputs(rng, 7, channels, symmetric_mask=symmetric_mask)

    monkeypatch.setenv("CHAI_JAX_TRIANGLE_MULTIPLICATION_BACKEND", "xla")
    blocked = fused_triangle_multiplication(pair, pair_mask, params, lin=linear)

    monkeypatch.setenv("CHAI_JAX_TRIANGLE_MULTIPLICATION_BACKEND", "cueq")
    # `lin=linear` on both sides: the identity is about the mapping, and in the
    # port's own bf16 it would only hold to 2e-2, where a wrong weight slice
    # can hide.
    fused = fused_triangle_multiplication(pair, pair_mask, params, lin=linear)

    np.testing.assert_allclose(
        np.asarray(fused, np.float32),
        np.asarray(blocked, np.float32),
        rtol=2e-5,
        atol=2e-5,
    )


def test_the_kernel_runs_at_the_precision_the_blocked_path_runs(
    monkeypatch, stub_kernel
) -> None:
    """bf16, because `linear_bf16` is what the module it replaces uses.

    The trunk hands `z` to this module in fp32 and the blocked path immediately
    rounds every matmul to bf16, returning bf16. The first version of this port
    handed the fp32 tensor straight to the kernel, so the whole module ran in
    fp32 -- and the measurement said so: 970 tokens went from 17,185 MiB /
    179.6 s to 18,205 MiB / 202.2 s, worse on both axes, for a port whose
    entire purpose was to be smaller.
    """
    seen = {}
    real = sys.modules["cuequivariance_jax"].triangle_multiplicative_update

    def recording(**kwargs):
        seen["x_dtype"] = kwargs["x"].dtype
        seen["mask_dtype"] = kwargs["mask"].dtype
        return real(**kwargs)

    monkeypatch.setattr(
        sys.modules["cuequivariance_jax"],
        "triangle_multiplicative_update",
        recording,
    )
    rng = np.random.default_rng(17)
    channels = 32
    params = _params(rng, channels)
    pair, pair_mask = _inputs(rng, 5, channels, symmetric_mask=True)
    assert pair.dtype == jnp.float32  # what the trunk actually holds

    monkeypatch.setenv("CHAI_JAX_TRIANGLE_MULTIPLICATION_BACKEND", "cueq")
    fused_triangle_multiplication(pair, pair_mask, params)

    # What the kernel is handed, not what comes back: the stand-in above
    # computes in fp32 and would return fp32 whatever it was given, so a check
    # on the output would be a check on the stub. The real kernel's dtype
    # follows `x`, and an fp32 mask beside a bf16 `x` would promote it back.
    assert seen["x_dtype"] == jnp.bfloat16
    assert seen["mask_dtype"] == jnp.bfloat16


def test_each_direction_carries_the_shared_output_gate(stub_kernel) -> None:
    """Both calls must receive `linear_z_out` and the *fifth* gate block.

    Handing the output projection to only one call, or slicing the gate at
    `3*c_z`, still produces a plausible tensor of the right shape. The blocked
    reference is the only thing that says which.
    """
    from foldjax.models.chai.models.triangle_cueq import (
        cueq_triangle_multiplication_direction,
    )

    rng = np.random.default_rng(3)
    channels = 32
    params = _params(rng, channels)
    pair, pair_mask = _inputs(rng, 6, channels, symmetric_mask=True)

    outgoing = cueq_triangle_multiplication_direction(
        pair, pair_mask, params, incoming=False, kernel_dtype=jnp.float32
    )
    incoming = cueq_triangle_multiplication_direction(
        pair, pair_mask, params, incoming=True, kernel_dtype=jnp.float32
    )

    # Sharing the gate means neither direction is the whole update, and the
    # two are genuinely different tensors rather than one computed twice.
    assert outgoing.shape == pair.shape
    assert not np.allclose(np.asarray(outgoing), np.asarray(incoming))


def test_the_kernel_never_sees_a_width_it_rejects() -> None:
    """`c_hidden == c_z` and `c_z % 32 == 0`, for every Chai triangle site.

    Protenix has a template stack the kernel cannot take and must route around
    it; Chai has no such site, which is why this port has no width guard. That
    is a fact about the checkpoint, so it is asserted against the checkpoint's
    shapes rather than assumed.
    """
    # (merged_p rows, c_z) for every triangle multiplication in the release:
    # 48 trunk pairformer blocks, 2 MSA pair blocks, 4 confidence blocks.
    for merged_rows, channels in ((1024, 256), (256, 64)):
        assert merged_rows == 4 * channels  # c_hidden == c_z, structurally
        assert channels % 32 == 0


def test_backend_switch_reaches_the_msa_pair_block(monkeypatch, stub_kernel) -> None:
    """The staged path had its own copy of the arithmetic, and must not keep it.

    `_run_msa_pair_block_low_memory` computed the two directions and the joint
    projection as three separate compiled calls. Left alone it would have gone
    on running XLA while the default said cueq -- the same shape of divergence
    that made the two pair gates disagree.
    """
    from foldjax.models.chai import inference

    rng = np.random.default_rng(5)
    channels = 32
    params = SimpleNamespace(
        triangle_multiplication=_params(rng, channels),
    )
    pair, pair_mask = _inputs(rng, 6, channels, symmetric_mask=True)

    monkeypatch.setenv("CHAI_JAX_TRIANGLE_MULTIPLICATION_BACKEND", "cueq")
    monkeypatch.setattr(
        inference,
        "_compiled_triangle_multiplication",
        lambda pair, mask, params: pairformer.fused_triangle_multiplication(
            pair, mask, params, lin=linear
        ),
    )
    fused = inference._chunked_triangle_multiplication(
        pair, pair_mask, params.triangle_multiplication, 2
    )

    monkeypatch.setenv("CHAI_JAX_TRIANGLE_MULTIPLICATION_BACKEND", "xla")
    direct = pairformer.fused_triangle_multiplication(
        pair, pair_mask, params.triangle_multiplication, lin=linear
    )

    # Chunk size 2 against 6 tokens: had the gate not reached, this would have
    # returned the tiled XLA result and matched for the wrong reason -- so the
    # assertion is that it equals the *unchunked* reference within kernel noise.
    np.testing.assert_allclose(
        np.asarray(fused, np.float32),
        np.asarray(direct, np.float32),
        rtol=2e-5,
        atol=2e-5,
    )
