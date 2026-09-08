"""Private OpenBind full-operator candidates; not a selectable model backend.

Contracts follow pinned c4771653's layers/triangular_multiplicative_update.py
(``_inference_forward``) and layers/triangular_attention.py (``forward``).
Only B=1, FP32, released widths 64/128, four attention heads, and no context
parallelism are supported. Ordinary BLAS operands are rounded to the TF32 RNE
grid before JAX ``high`` dots, not the Triton linear's separate RTZ arithmetic.
Teacher-forced GPU projection controls passed 15/15 leaves (N16/17/437). Extending
that policy to stock QK/PV and triangle contraction still requires the full-module
panel; those contractions were not admitted by the projection-only control.

The immutable multiplication input replaces native's overwrite-recovery cache;
this preserves chunk operands, not native's in-place memory bound. Interpretation
omits ordinary RNE rounding, just as the custom kernel interpreters omit RTZ/FMA;
it checks the continuous FP32 formula, not GPU instruction-level parity.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp

from foldjax.models._cp import cp_mesh
from foldjax.models.openfold3.models.attention import flatten_heads, split_heads
from foldjax.models.openfold3.models.native_triton_attention import (
    native_triangle_attention as native_attention_core,
)
from foldjax.models.openfold3.models.native_triton_linear import (
    native_linear,
    native_linear_fused,
)
from foldjax.models.openfold3.models.native_triton_norm import native_layer_norm
from foldjax.models.openfold3.models.primitives import jax_sigmoid, layer_norm
from foldjax.models.openfold3.models.triangle import permute_final_dims


def _tf32_rne(x):
    """Ordinary cuBLAS tie policy, not the custom Triton kernels' RTZ policy.

    RNE-grid values survive XLA's subsequent RNA conversion unchanged. Keep this
    at the actual dot inputs: rounding a norm, sigmoid, mask or residual earlier
    would change a different operation. CPU can check these bits but cannot
    certify the GPU accumulator. NaN payloads, infinities and signed zero survive.
    """
    if x.dtype != jnp.float32:
        raise TypeError("ordinary native TF32 RNE requires FP32")
    bits = jax.lax.bitcast_convert_type(x, jnp.uint32)
    finite = (bits & jnp.uint32(0x7F800000)) != jnp.uint32(0x7F800000)
    rounded = (bits + jnp.uint32(0xFFF) + ((bits >> 13) & jnp.uint32(1))) & jnp.uint32(
        0xFFFFE000
    )
    return jax.lax.bitcast_convert_type(jnp.where(finite, rounded, bits), jnp.float32)


def _ordinary_linear(x, params, *, interpret=False):
    lhs, rhs = (
        (x, params.weight) if interpret else (_tf32_rne(x), _tf32_rne(params.weight))
    )
    result = jnp.matmul(lhs, rhs.T, precision="high")
    return result if params.bias is None else result + params.bias


def _ordinary_norm(x, params, eps, *, interpret=False):
    if interpret:
        return layer_norm(x, params, eps=eps)
    from foldjax.models.boltz2.models.primitives.native_amp_norm import _cuda_layer_norm

    # Ordinary Torch LayerNorm, not OpenBind's custom second-moment Triton norm.
    # The complete N16/17/437 start-attention norm control matched bitwise;
    # this private full-module route still needs separate panel verification.
    return jax.lax.platform_dependent(
        x,
        params,
        cuda=lambda a, p: _cuda_layer_norm(a, p.weight, p.bias, eps)[0],
        default=lambda a, p: layer_norm(a, p, eps=eps),
    )


def _ordinary_einsum(equation, lhs, rhs, *, interpret=False):
    if not interpret:
        lhs, rhs = _tf32_rne(lhs), _tf32_rne(rhs)
    return jnp.einsum(equation, lhs, rhs, precision="high")


def _stock_attention(query, key, value, biases, *, interpret=False):
    # The caller has already performed native's Q / sqrt(D) scaling. Biases are
    # still sequential FP32 adds, and probabilities are rounded only for PV.
    scores = _ordinary_einsum("...qc,...kc->...qk", query, key, interpret=interpret)
    for bias in biases:
        scores = scores + bias
    probability = jax.nn.softmax(scores, axis=-1)
    return _ordinary_einsum(
        "...qk,...kc->...qc", probability, value, interpret=interpret
    )


def chunk_ranges(length: int, chunk_size: int, *, split_half: bool = False):
    """Native static slices, including the multiplication cache's halfway split."""
    if (
        not isinstance(length, int)
        or isinstance(length, bool)
        or length < 1
        or not isinstance(chunk_size, int)
        or isinstance(chunk_size, bool)
        or chunk_size < 1
    ):
        raise ValueError("length and chunk_size must be positive integers")
    boundaries = (0, (length + 1) // 2, length) if split_half else (0, length)
    return tuple(
        (start, min(start + chunk_size, end))
        for begin, end in zip(boundaries[:-1], boundaries[1:], strict=True)
        for start in range(begin, end, chunk_size)
    )


def _validate(z, params, mask, eps):
    if cp_mesh() is not None:
        raise ValueError("private native operators do not support context parallelism")
    if (
        z.ndim != 4
        or z.shape[0] != 1
        or z.shape[1] < 1
        or z.shape[1] != z.shape[2]
        or z.shape[-1] not in (64, 128)
    ):
        raise ValueError("expected B=1 square pair tensor [1,N,N,C], C=64/128")
    if any(x.dtype != jnp.float32 for x in (z, *jax.tree.leaves(params))):
        raise ValueError("private native operators require FP32 inputs and parameters")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be positive and finite")
    if mask is None:
        mask = jnp.ones(z.shape[:-1], dtype=z.dtype)
    if mask.shape != z.shape[:-1] or mask.dtype != z.dtype:
        raise ValueError("mask must match pair shape and FP32 dtype")
    return mask


def _multiplication(z, params, *, outgoing, mask, eps, chunk_size, interpret, add):
    mask = _validate(z, params, mask, eps)
    width = z.shape[-1]
    for name in ("layer_norm_in", "layer_norm_out"):
        norm = getattr(params, name)
        if any(x is None or x.shape != (width,) for x in norm):
            raise ValueError("released multiplication requires affine norms of width C")
    for name in (
        "linear_a_p",
        "linear_a_g",
        "linear_b_p",
        "linear_b_g",
        "linear_g",
        "linear_z",
    ):
        projection = getattr(params, name)
        if projection.weight.shape != (width, width) or projection.bias is not None:
            raise ValueError("released multiplication requires bias-free C x C weights")
    n = z.shape[1]
    a_ranges = chunk_ranges(n, chunk_size)
    output_ranges = chunk_ranges(n, chunk_size, split_half=True)

    def norm(x, p):
        return native_layer_norm(x, p.weight, p.bias, eps, interpret=interpret)

    def projection(pair, pair_mask, *, a):
        x = norm(pair, params.layer_norm_in)
        gate = params.linear_a_g if a else params.linear_b_g
        proj = params.linear_a_p if a else params.linear_b_p
        g = native_linear_fused(x, gate.weight, apply_sigmoid=True, interpret=interpret)
        p = native_linear_fused(
            x, proj.weight, other=g, mask=pair_mask[..., None], interpret=interpret
        )
        p = permute_final_dims(p, (2, 0, 1))
        return jnp.swapaxes(p, -1, -2) if outgoing ^ a else p

    a = jnp.concatenate(
        [projection(z[:, i:j], mask[:, i:j], a=True) for i, j in a_ranges],
        axis=-2 if outgoing else -1,
    )
    outputs = []
    for i, j in output_ranges:
        pair = z[:, i:j] if outgoing else z[:, :, i:j]
        pair_mask = mask[:, i:j] if outgoing else mask[:, :, i:j]
        b = projection(pair, pair_mask, a=False)
        # Native is torch.einsum/cuBLAS here, not triton_linear_fused.
        x = _ordinary_einsum("...ij,...jk->...ik", a, b, interpret=interpret)
        x = norm(permute_final_dims(x, (1, 2, 0)), params.layer_norm_out)
        x = native_linear(x, params.linear_z.weight, interpret=interpret)
        original = z[:, :, i:j]
        g = norm(original, params.layer_norm_in)
        outputs.append(
            native_linear_fused(
                g,
                params.linear_g.weight,
                other=x,
                add_tensor=original if add else None,
                apply_sigmoid=True,
                interpret=interpret,
            )
        )
    return jnp.concatenate(outputs, axis=2)


def native_triangle_multiplication_update(
    z, params, *, outgoing, mask=None, eps=1e-5, chunk_size=256, interpret=False
):
    """Update only: native inplace_safe=True, _add_with_inplace=False."""
    return _multiplication(
        z,
        params,
        outgoing=outgoing,
        mask=mask,
        eps=eps,
        chunk_size=chunk_size,
        interpret=interpret,
        add=False,
    )


def native_triangle_multiplication_residual(
    z, params, *, outgoing, mask=None, eps=1e-5, chunk_size=256, interpret=False
):
    """Already includes z: native PairBlock's _add_with_inplace=True route.

    This must never be passed to the existing update-only caller's ``z + update``.
    """
    return _multiplication(
        z,
        params,
        outgoing=outgoing,
        mask=mask,
        eps=eps,
        chunk_size=chunk_size,
        interpret=interpret,
        add=True,
    )


def native_triangle_attention_update(
    z,
    params,
    *,
    mask=None,
    starting=True,
    transpose_bias=False,
    eps=1e-5,
    inf=1e9,
    chunk_size=1024,
    interpret=False,
):
    """Native module update, with ordinary LN/projections and stock N<=16 core.

    PairBlock ending attention calls this with transposed z/mask, starting=True,
    transpose_bias=True, then transposes its result. ``starting=False`` instead
    represents the standalone native module's own transpose contract.
    """
    mask = _validate(z, params, mask, eps)
    c = z.shape[-1]
    if not math.isfinite(inf) or inf <= 0:
        raise ValueError("finite positive mask inf is required")
    if any(x is None or x.shape != (c,) for x in params.layer_norm):
        raise ValueError("released attention requires affine norm of width C")
    projections = (params.linear_z, *params.mha)
    expected = ((4, c), *((c, c),) * 5)
    if any(
        p is None or p.weight.shape != shape or p.bias is not None
        for p, shape in zip(projections, expected, strict=True)
    ):
        raise ValueError("released attention requires four heads and bias-free weights")
    ranges = chunk_ranges(z.shape[1], chunk_size)
    if not starting:
        z, mask = jnp.swapaxes(z, 1, 2), jnp.swapaxes(mask, 1, 2)
    # This context affects only ordinary matmuls; Pallas kernels own their policy.
    with jax.default_matmul_precision("high"):
        x = _ordinary_norm(z, params.layer_norm, eps, interpret=interpret)
        bias = permute_final_dims(
            _ordinary_linear(x, params.linear_z, interpret=interpret),
            (2, 1, 0) if transpose_bias else (2, 0, 1),
        )[:, None]
        additive_mask = (inf * (mask - 1))[:, :, None, None, :]
        outputs = []
        for i, j in ranges:
            row = x[:, i:j]
            q, k, v = (
                split_heads(_ordinary_linear(row, p, interpret=interpret), 4)
                for p in (params.mha.linear_q, params.mha.linear_k, params.mha.linear_v)
            )
            row_mask = additive_mask[:, i:j]
            if z.shape[1] <= 16:
                q, k, v = (jnp.swapaxes(t, -2, -3) for t in (q, k, v))
                core = _stock_attention(
                    q / math.sqrt(c // 4), k, v, (row_mask, bias), interpret=interpret
                )
                core = jnp.swapaxes(core, -2, -3)
            else:
                core = native_attention_core(
                    q, k, v, row_mask, bias, interpret=interpret
                )
            gate = split_heads(
                jax_sigmoid(
                    _ordinary_linear(row, params.mha.linear_g, interpret=interpret)
                ),
                4,
            )
            outputs.append(
                _ordinary_linear(
                    flatten_heads(core * gate), params.mha.linear_o, interpret=interpret
                )
            )
        output = jnp.concatenate(outputs, axis=1)
    return output if starting else jnp.swapaxes(output, 1, 2)
