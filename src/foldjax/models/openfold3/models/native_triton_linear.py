"""Isolated OpenBind Triton linear/fused-linear forward candidate.

The contract is ``triangular_multiplicative_update.py::triton_linear_fused``
and ``linear_fused_kernel`` at OpenFold3 commit
c4771653c5d0a3ebb0b3af71b05efd64bc44ee86 (wrapper lines495-628, kernel232-364).
The same program with every optional flag off represents ``triton_linear``.

This is not selected by production inference. Only input/output widths 64 and
128 are implemented, covering the released pair and template projections.
Native CUDA uses TF32 inputs for FP32 dot products and FP32 accumulators, not
an arbitrary enclosing JAX matmul policy. Its observed finite FP32 operands
truncate toward zero, unlike Pallas's implicit nearest TF32 conversion.
CPU interpretation checks formula, shapes and storage boundaries; it does not
emulate TF32, CUDA fast sigmoid or FMA. Operator panel evidence does not admit
production selection or establish parity on other hardware or model inputs.
"""

from __future__ import annotations

import functools
import math

import jax
import jax.numpy as jnp

try:
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import triton as pt
except ImportError:  # pragma: no cover - optional accelerator implementation
    pl = None
    pt = None


_STORAGE_DTYPES = (jnp.float16, jnp.bfloat16, jnp.float32)
_WIDTHS = (64, 128)


def _tf32_rtz(value):
    if value.dtype != jnp.float32:
        raise TypeError("TF32 operand truncation requires FP32")
    bits = jax.lax.bitcast_convert_type(value, jnp.uint32)
    truncated = bits & jnp.uint32(0xFFFFE000)
    # Preserve special-value payloads instead of turning a low-payload NaN into
    # infinity; native parity evidence covers finite operands only.
    special = (bits & jnp.uint32(0x7F800000)) == jnp.uint32(0x7F800000)
    return jax.lax.bitcast_convert_type(
        jnp.where(special, bits, truncated), jnp.float32
    )


def _fp32_instruction(instruction, *values):
    values = jnp.broadcast_arrays(*values)
    registers = ", ".join(f"${index}" for index in range(len(values) + 1))
    return pt.elementwise_inline_asm(
        f"{instruction} {registers};",
        args=values,
        constraints="=f" + ",f" * len(values),
        pack=1,
        result_shape_dtypes=[jax.ShapeDtypeStruct(values[0].shape, jnp.float32)],
    )[0]


def _sigmoid(acc, *, interpret):
    acc = jnp.where(acc > 20, jnp.float32(20), acc)
    acc = jnp.where(acc < -20, jnp.float32(-20), acc)
    if interpret:
        return jnp.float32(1) / (jnp.float32(1) + jnp.exp(-acc))
    # The pinned native CUDA kernel lowers tl.exp to this log2(e)/ex2 sequence,
    # followed by add and div.full. jnp.exp would call a different libdevice op.
    return pt.elementwise_inline_asm(
        "{ .reg .f32 value;"
        " sub.f32 value, 0f00000000, $1;"
        " mul.f32 value, value, 0f3FB8AA3B;"
        " ex2.approx.f32 value, value;"
        " add.f32 value, value, 0f3F800000;"
        " div.full.f32 $0, 0f3F800000, value; }",
        args=(acc,),
        constraints="=f,f",
        pack=1,
        result_shape_dtypes=[jax.ShapeDtypeStruct(acc.shape, jnp.float32)],
    )[0]


def _epilogue(acc, bias, other, mask, add_tensor, *, apply_sigmoid, interpret):
    def multiply(a, b):
        return a * b if interpret else _fp32_instruction("mul.rn.f32", a, b)

    def add(a, b):
        return a + b if interpret else _fp32_instruction("add.rn.f32", a, b)

    def multiply_add(a, b, c):
        return a * b + c if interpret else _fp32_instruction("fma.rn.f32", a, b, c)

    if bias is not None:
        acc = add(acc, bias)
    if apply_sigmoid:
        acc = _sigmoid(acc, interpret=interpret)
    if other is not None:
        # Native's enabled FP fusion contracts the last multiply with residual
        # addition. With a mask, the earlier multiply must remain a separate op.
        if mask is None and add_tensor is not None:
            acc = multiply_add(acc, other, add_tensor)
            add_tensor = None
        else:
            acc = multiply(acc, other)
    if mask is not None:
        if add_tensor is not None:
            acc = multiply_add(acc, mask, add_tensor)
            add_tensor = None
        else:
            acc = multiply(acc, mask)
    if add_tensor is not None:
        acc = add(acc, add_tensor)
    return acc


def _linear_kernel(
    x_ref,
    weight_ref,
    bias_ref,
    other_ref,
    mask_ref,
    add_ref,
    shape_ref,
    out_ref,
    *,
    block_m,
    block_n,
    block_k,
    has_bias,
    has_other,
    has_mask,
    has_add,
    apply_sigmoid,
    interpret,
):
    rows, inputs, outputs = shape_ref[0], shape_ref[1], shape_ref[2]
    rm = pl.program_id(0) * block_m + jnp.arange(block_m)
    rn = pl.program_id(1) * block_n + jnp.arange(block_n)
    output_mask = (rm[:, None] < rows) & (rn[None, :] < outputs)
    if interpret:
        precision = jax.lax.Precision.HIGHEST
    elif x_ref.dtype == jnp.float32:
        precision = jax.lax.DotAlgorithmPreset.TF32_TF32_F32
    elif x_ref.dtype == jnp.bfloat16:
        precision = jax.lax.DotAlgorithmPreset.BF16_BF16_F32
    else:
        precision = jax.lax.DotAlgorithmPreset.F16_F16_F32

    def accumulate(block, acc):
        rk = block * block_k + jnp.arange(block_k)
        x = pt.load(
            x_ref.at[rm[:, None], rk[None, :]],
            mask=(rm[:, None] < rows) & (rk[None, :] < inputs),
            other=0.0,
        )
        weight = pt.load(
            weight_ref.at[rn[None, :], rk[:, None]],
            mask=(rn[None, :] < outputs) & (rk[:, None] < inputs),
            other=0.0,
        )
        if not interpret and x_ref.dtype == jnp.float32:
            # RTZ-input replay matches all 52 pinned native FP32 controls;
            # truncate loaded dot tiles, never bias/epilogue or half storage.
            x, weight = _tf32_rtz(x), _tf32_rtz(weight)
        product = jax.lax.dot_general(
            x,
            weight,
            (((1,), (0,)), ((), ())),
            precision=precision,
            preferred_element_type=jnp.float32,
        )
        return acc + product

    acc = jax.lax.fori_loop(
        0,
        (inputs + block_k - 1) // block_k,
        accumulate,
        jnp.zeros((block_m, block_n), jnp.float32),
    )
    bias = (
        pt.load(bias_ref.at[rn], mask=rn < outputs, other=0.0).astype(jnp.float32)[
            None, :
        ]
        if has_bias
        else None
    )

    def load_operand(ref):
        return pt.load(
            ref.at[rm[:, None], rn[None, :]], mask=output_mask, other=0.0
        ).astype(jnp.float32)

    acc = _epilogue(
        acc,
        bias,
        load_operand(other_ref) if has_other else None,
        load_operand(mask_ref) if has_mask else None,
        load_operand(add_ref) if has_add else None,
        apply_sigmoid=apply_sigmoid,
        interpret=interpret,
    )
    pt.store(
        out_ref.at[rm[:, None], rn[None, :]],
        acc.astype(out_ref.dtype),
        mask=output_mask,
    )


def _check_array(name, value, *, mask=False):
    if not hasattr(value, "shape") or not hasattr(value, "dtype"):
        raise TypeError(f"{name} must be an array")
    allowed = (*_STORAGE_DTYPES, jnp.bool_) if mask else _STORAGE_DTYPES
    if value.dtype not in allowed:
        suffix = ", or boolean" if mask else ""
        raise TypeError(f"{name} must have FP16, BF16 or FP32{suffix} dtype")


def native_linear_fused(
    x,
    weight,
    bias=None,
    other=None,
    mask=None,
    add_tensor=None,
    apply_sigmoid=False,
    *,
    interpret=False,
):
    """Standalone candidate for the pinned native fused-linear wrapper.

    x is ``[..., K]`` and weight ``[N, K]``, with K,N each 64 or 128. Bias is
    ``[N]``. Only weight/bias are converted to x's storage dtype, as upstream
    does; optional operands retain their dtype until FP32 kernel arithmetic.
    The output is ``[..., N]`` in x's dtype, cast once after the epilogue.

    Other/add_tensor must contain exactly M*N values (native reshape semantics).
    A mask's final dimension must be 1 or N: feature-1 masks support one or M
    rows, while feature-N masks require M rows. In particular [N] is NOT a
    per-channel broadcast when M>1. These are native wrapper restrictions.

    ``interpret=True`` is formula-only CPU evidence, never native GPU parity.
    No ordinary-JAX fallback or production backend selection is provided.
    """
    if not isinstance(apply_sigmoid, bool) or not isinstance(interpret, bool):
        raise TypeError("apply_sigmoid and interpret must be static booleans")
    _check_array("x", x)
    _check_array("weight", weight)
    if x.ndim < 1 or weight.ndim != 2 or x.shape[-1] != weight.shape[1]:
        raise ValueError("x must be [..., K] and weight [N, K] with the same K")
    inputs, outputs = x.shape[-1], weight.shape[0]
    if inputs not in _WIDTHS or outputs not in _WIDTHS:
        raise ValueError("native linear candidate supports only K,N in {64,128}")
    rows = math.prod(x.shape[:-1])
    shape = (*x.shape[:-1], outputs)
    if bias is not None:
        _check_array("bias", bias)
        if bias.shape != (outputs,):
            raise ValueError(f"bias must have shape ({outputs},)")
        bias = bias.astype(x.dtype)

    def reshape_operand(name, value):
        if value is None:
            return None
        _check_array(name, value)
        if math.prod(value.shape) != rows * outputs:
            raise ValueError(f"{name} must contain exactly M*N output values")
        return value.reshape(rows, outputs)

    other = reshape_operand("other", other)
    add_tensor = reshape_operand("add_tensor", add_tensor)
    if mask is not None:
        _check_array("mask", mask, mask=True)
        if mask.ndim < 1 or mask.shape[-1] not in (1, outputs):
            raise ValueError("mask must have final dimension 1 or N")
        mask_rows = math.prod(mask.shape[:-1])
        features = mask.shape[-1]
        if (features == 1 and mask_rows not in (1, rows)) or (
            features == outputs and mask_rows != rows
        ):
            raise ValueError("mask row count does not match native broadcasting")
        mask = jnp.broadcast_to(mask.reshape(mask_rows, features), (rows, outputs))
    if pl is None or pt is None:
        raise RuntimeError("Pallas/Triton is required for native OpenBind linear")
    if rows == 0:
        return jnp.empty(shape, x.dtype)

    block_m = min(128, 1 << (rows - 1).bit_length())
    block_n = min(64, 1 << (outputs - 1).bit_length())
    block_k = min(64, 1 << (inputs - 1).bit_length())
    kernel = functools.partial(
        _linear_kernel,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        has_bias=bias is not None,
        has_other=other is not None,
        has_mask=mask is not None,
        has_add=add_tensor is not None,
        apply_sigmoid=apply_sigmoid,
        interpret=interpret,
    )
    unused = jnp.zeros((1,), x.dtype)
    result = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((rows, outputs), x.dtype),
        grid=(pl.cdiv(rows, block_m), pl.cdiv(outputs, block_n)),
        compiler_params=pt.CompilerParams(num_warps=4, num_stages=3),
        interpret=interpret,
        name="openbind_native_triangle_linear_fused",
    )(
        x.reshape(rows, inputs),
        weight.astype(x.dtype),
        bias if bias is not None else unused,
        other if other is not None else unused,
        mask if mask is not None else unused,
        add_tensor if add_tensor is not None else unused,
        jnp.asarray((rows, inputs, outputs), jnp.int32),
    )
    return result.reshape(shape)


def native_linear(x, weight, bias=None, *, interpret=False):
    """Plain native linear contract, with every fused epilogue option disabled."""
    return native_linear_fused(x, weight, bias, interpret=interpret)
