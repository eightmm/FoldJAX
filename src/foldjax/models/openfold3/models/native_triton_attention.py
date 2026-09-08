"""Isolated candidate for OpenBind's native FP32 triangle attention core.

Source: OpenFold3 c4771653c5d0a3ebb0b3af71b05efd64bc44ee86,
``core/kernels/triton/evoformer.py`` lines 58-367 and 1060-1238. This follows
the default specialized forward, Q64/KV16, natural exponential softmax and
four heads with head width 16 or 32. LayerNorm, projections, gating and the
public Attention dispatch are outside this module; production never selects it.

The FP32 RTZ dot-input policy is a candidate inferred from native linear
controls, not established native attention parity. CPU interpretation checks
the online formula only, not CUDA TF32, fast exponential, FMA or reduction
order. The private carried-dot primitive has only a Pallas Triton lowering:
its third tt.dot operand is the live accumulator, never a separate add.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp

from foldjax.models.openfold3.models.native_triton_linear import (
    _fp32_instruction,
    _tf32_rtz,
)

try:
    from jax._src import core as jax_core
    from jax._src.lib.triton import dialect as tt_dialect
    from jax._src.pallas.triton import lowering as triton_lowering
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import triton as pt
except ImportError:  # pragma: no cover - optional accelerator implementation
    pl = pt = None
    _carried_dot_p = None
else:
    # This registration is for our own primitive only. It neither replaces
    # lax.dot_general nor changes any model's ordinary JAX lowering policy.
    _carried_dot_p = jax_core.Primitive("openbind_triangle_carried_dot")

    @_carried_dot_p.def_abstract_eval
    def _carried_dot_abstract(a, b, accumulator):
        if any(x.dtype != jnp.float32 for x in (a, b, accumulator)):
            raise TypeError("native triangle carried dot requires FP32")
        if (
            any(x.ndim != 2 for x in (a, b, accumulator))
            or a.shape[1] != b.shape[0]
            or accumulator.shape != (a.shape[0], b.shape[1])
        ):
            raise ValueError("carried dot requires [M,K], [K,N], [M,N]")
        if any(size < 16 or size & (size - 1) for size in (*a.shape, b.shape[1])):
            raise ValueError("carried dot tile dimensions must be powers of two >=16")
        return accumulator

    @_carried_dot_p.def_impl
    def _carried_dot_eager(*args):
        raise RuntimeError("native carried dot is Pallas Triton-only")

    @triton_lowering.register_lowering(_carried_dot_p)
    def _carried_dot_lowering(ctx, a, b, accumulator):
        del ctx
        return tt_dialect.dot(
            a, b, accumulator, input_precision=tt_dialect.InputPrecision.TF32
        )


def _dot(a, b, accumulator, *, interpret):
    if interpret:
        # Explicitly formula-only: a CPU matmul plus add does not certify the
        # native Tensor Core accumulator's arithmetic or instruction sequence.
        return (
            jax.lax.dot_general(
                a,
                b,
                (((1,), (0,)), ((), ())),
                precision=jax.lax.Precision.HIGHEST,
                preferred_element_type=jnp.float32,
            )
            + accumulator
        )
    return _carried_dot_p.bind(_tf32_rtz(a), _tf32_rtz(b), accumulator)


def _exp(value, *, interpret):
    if interpret:
        return jnp.exp(value)
    # Native USE_EXP2=False still lowers tl.exp to log2(e) times its argument
    # followed by CUDA ex2. Do not scale the logits before subtracting max.
    return pt.elementwise_inline_asm(
        "{ .reg .f32 scaled; mul.f32 scaled, $1, 0f3FB8AA3B;"
        " ex2.approx.f32 $0, scaled; }",
        args=(value,),
        constraints="=f,f",
        pack=1,
        result_shape_dtypes=[jax.ShapeDtypeStruct(value.shape, jnp.float32)],
    )[0]


def _arithmetic(instruction, *values, interpret):
    if not interpret:
        return _fp32_instruction(instruction, *values)
    if instruction == "mul.rn.f32":
        return values[0] * values[1]
    if instruction == "add.rn.f32":
        return values[0] + values[1]
    if instruction == "sub.rn.f32":
        return values[0] - values[1]
    if instruction == "fma.rn.f32":
        return values[0] * values[1] + values[2]
    if instruction == "div.full.f32":
        return values[0] / values[1]
    raise ValueError("unsupported native attention arithmetic")


def _attention_kernel(
    q_ref, k_ref, v_ref, mask_ref, bias_ref, out_ref, *, rows, length, dim, interpret
):
    q_index = pl.program_id(0) * 64 + jnp.arange(64)
    batch_row_head = pl.program_id(1)
    head = batch_row_head % 4
    row = (batch_row_head // 4) % rows
    batch = batch_row_head // (rows * 4)
    d_index = jnp.arange(32)
    q = pt.load(
        q_ref.at[batch, row, q_index[:, None], head, d_index[None, :]],
        mask=(q_index[:, None] < length) & (d_index[None, :] < dim),
        other=0.0,
    )
    arith = functools.partial(_arithmetic, interpret=interpret)
    q = arith("mul.rn.f32", q, jnp.asarray(dim**-0.5, q.dtype))

    def step(block, state):
        maximum, normalizer, output = state
        kv_index = block * 16 + jnp.arange(16)
        k = pt.load(
            k_ref.at[batch, row, kv_index[None, :], head, d_index[:, None]],
            mask=(kv_index[None, :] < length) & (d_index[:, None] < dim),
            other=0.0,
        )
        v = pt.load(
            v_ref.at[batch, row, kv_index[:, None], head, d_index[None, :]],
            mask=(kv_index[:, None] < length) & (d_index[None, :] < dim),
            other=0.0,
        )
        mask = pt.load(
            mask_ref.at[batch, row, 0, 0, kv_index],
            mask=kv_index < length,
            other=-jnp.inf,
        )[None, :]
        bias = pt.load(
            bias_ref.at[batch, 0, head, q_index[:, None], kv_index[None, :]],
            mask=(q_index[:, None] < length) & (kv_index[None, :] < length),
            other=-jnp.inf,
        )
        logits = _dot(q, k, jnp.zeros((64, 16), jnp.float32), interpret=interpret)
        logits = arith("add.rn.f32", logits, mask)
        logits = arith("add.rn.f32", logits, bias)
        if length % 16:
            logits = arith(
                "add.rn.f32",
                logits,
                jnp.where(kv_index[None, :] < length, 0.0, -jnp.inf),
            )
        new_maximum = jnp.maximum(maximum, jnp.max(logits, axis=1))
        probability = _exp(
            arith("sub.rn.f32", logits, new_maximum[:, None]), interpret=interpret
        )
        alpha = _exp(arith("sub.rn.f32", maximum, new_maximum), interpret=interpret)
        normalizer = arith(
            "fma.rn.f32", normalizer, alpha, jnp.sum(probability, axis=1)
        )
        probability = probability.astype(v.dtype)
        output = arith("mul.rn.f32", output, alpha[:, None])
        output = _dot(probability, v, output, interpret=interpret)
        return new_maximum, normalizer, output

    _, normalizer, output = jax.lax.fori_loop(
        0,
        (length + 15) // 16,
        step,
        (
            jnp.full((64,), -jnp.inf, jnp.float32),
            jnp.ones((64,), jnp.float32),
            jnp.zeros((64, 32), jnp.float32),
        ),
    )
    output = arith("div.full.f32", output, normalizer[:, None])
    pt.store(
        out_ref.at[batch, row, q_index[:, None], head, d_index[None, :]],
        output.astype(out_ref.dtype),
        mask=(q_index[:, None] < length) & (d_index[None, :] < dim),
    )


def native_triangle_attention(
    query, key, value, additive_mask, pair_bias, *, interpret=False
):
    """Return the native-layout triangle core, without projections or gating.

    Q/K/V are equal FP32 shapes [B,R,N,4,D], D in {16,32}, N>16. Additive mask
    [B,R,1,1,N] and pair bias [B,1,4,N,N] must be FP32, not binary validity
    flags. R may be a chunk of the triangle row axis. Inputs are unscaled QKV;
    the native D**-0.5 scaling happens inside the kernel in input dtype.

    Finite all-masked rows (e.g. -1e9) are not zeroed; true -inf masking may
    produce NaN, including a fully masked initial KV tile. This deliberately
    retains native online-softmax behavior, with no safe-softmax replacement.
    N<=16 is rejected: the upstream public wrapper chooses stock attention
    there. Only inference output is returned; backward logsumexp is omitted.

    interpret=True is CPU formula evidence only. FP32 RTZ, exponential and
    reduction fidelity require an actual native attention GPU comparison.
    """
    if not isinstance(interpret, bool):
        raise TypeError("interpret must be a static boolean")
    arrays = (query, key, value, additive_mask, pair_bias)
    if any(not hasattr(x, "shape") or not hasattr(x, "dtype") for x in arrays):
        raise TypeError("attention operands must be arrays")
    if any(x.dtype != jnp.float32 for x in arrays):
        raise TypeError("native attention candidate requires FP32 operands")
    if query.ndim != 5 or key.shape != query.shape or value.shape != query.shape:
        raise ValueError("Q/K/V must have equal [B,R,N,H,D] shapes")
    batch, rows, length, heads, dim = query.shape
    if batch < 1 or rows < 1 or heads != 4 or dim not in (16, 32):
        raise ValueError("requires positive B,R, four heads and D in {16,32}")
    if length <= 16:
        raise ValueError("N<=16 uses native stock attention, not this Triton core")
    if additive_mask.shape != (batch, rows, 1, 1, length):
        raise ValueError("additive mask must have shape [B,R,1,1,N]")
    if pair_bias.shape != (batch, 1, 4, length, length):
        raise ValueError("pair bias must have shape [B,1,4,N,N]")
    if pl is None or pt is None or _carried_dot_p is None:
        raise RuntimeError("Pallas Triton carried-dot lowering is required")
    return pl.pallas_call(
        functools.partial(
            _attention_kernel, rows=rows, length=length, dim=dim, interpret=interpret
        ),
        out_shape=jax.ShapeDtypeStruct(query.shape, query.dtype),
        grid=(pl.cdiv(length, 64), batch * rows * 4, 1),
        compiler_params=pt.CompilerParams(num_warps=4, num_stages=1),
        interpret=interpret,
        name="openbind_native_triangle_attention",
    )(*arrays)
