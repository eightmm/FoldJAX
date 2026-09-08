"""Bounded native BF16 PairWeightedAveraging contraction.

The residue-first k8 route matched all 62,033,024 native raw FP32 and BF16
outputs for the captured head2, S4436/N437/D32 operands (v46). This is not
whole-PWA, other-shape or cross-hardware parity admission. Guarded runtime
full rows and its default 1199/1199/1199/839 row slices also matched (v49/v48).
Chunk dispatch requires explicit original S4436 context; no layout is padded
or enlarged to select it, and custom row profiles keep the existing einsum.
Only CUDA supports this forward kernel; native evidence is from SM120.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from foldjax.models._cp import cp_mesh

try:
    from jax._src import core as jax_core
    from jax._src.lib.triton import dialect as tt_dialect
    from jax._src.pallas.triton import lowering as triton_lowering
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import triton as pt
except ImportError:  # pragma: no cover - optional accelerator implementation
    pl = pt = None
    _mma1688_p = None
else:
    _mma1688_p = jax_core.Primitive("boltz_pwa_mma1688")
    _mma1688_p.multiple_results = True

    @_mma1688_p.def_abstract_eval
    def _mma_abstract(*registers):
        if len(registers) != 7 or any(r.shape != (32,) for r in registers):
            raise ValueError("MMA requires seven one-warp register vectors")
        if [r.dtype for r in registers] != [jnp.uint32] * 3 + [jnp.float32] * 4:
            raise TypeError("MMA requires three uint32 words and four FP32 C registers")
        return [jax_core.ShapedArray((32,), jnp.float32)] * 4

    @_mma1688_p.def_impl
    def _mma_eager(*registers):
        raise RuntimeError("Boltz carried MMA is Pallas Triton-only")

    @triton_lowering.register_lowering(_mma1688_p)
    def _mma_lowering(ctx, *registers):
        if ctx.context.platform != "cuda":
            raise NotImplementedError("Boltz carried MMA requires CUDA")
        capability = ctx.context.compute_capability
        if capability is None or capability < 80:
            raise NotImplementedError("BF16 MMA requires known compute capability >=80")
        # Register only our primitive. JAX 0.11.1's generic multi-result helper
        # wraps OpResultList in an extra list; return the four IR values directly.
        results = list(
            tt_dialect.elementwise_inline_asm(
                [registers[3].type] * 4,
                _ASM,
                constraints=_CONSTRAINTS,
                pure=True,
                packed_element=1,
                args=registers,
            )
        )
        if len(results) != 4:
            raise ValueError("MMA lowering did not return four IR results")
        return results


# PTX ISA 9.0, mma.m16n8k8 floating-point fragment layout:
# https://docs.nvidia.com/cuda/archive/13.0.0/parallel-thread-execution/index.html#warp-level-matrix-fragment-mma-1688
_ASM = (
    "mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 "
    "{$0,$1,$2,$3}, {$4,$5}, {$6}, {$7,$8,$9,$10};"
)
_CONSTRAINTS = "=f,=f,=f,=f,r,r,r,f,f,f,f"


def _residue_k(position):
    # CUTLASS v3.8.0 predicated_tile_access_iterator.h:187-220, 514-523.
    # https://github.com/NVIDIA/cutlass/blob/v3.8.0/include/cutlass/transform/threadblock/predicated_tile_access_iterator.h
    # Preserve K0:21, insert eleven zeros, then K21:437; no K-order search.
    return jnp.where(
        position < 21, position, jnp.where(position < 32, -1, position - 11)
    )


def _guard_lane_identity(carry, physical_lane):
    # Fragment packing requires logical element i to execute on physical lane i.
    # Poison every output of an invalid tile on-device: this is a nonfinite
    # failure signal, not a Python exception, and needs no host callback.
    mismatch = physical_lane != jnp.arange(32, dtype=jnp.int32)
    valid = jnp.sum(mismatch.astype(jnp.int32)) == 0
    return tuple(jnp.where(valid, value, jnp.float32(jnp.nan)) for value in carry)


def _pwa_raw_fp32(weights, values, initial, *, debug=False):
    """Private positive-S kernel; caller dispatch admits only verified profiles."""
    if pl is None:
        raise ImportError("native PWA MMA requires Pallas Triton")
    if weights.dtype != jnp.bfloat16 or values.dtype != jnp.bfloat16:
        raise TypeError("native PWA MMA accepts already-BF16 operands only")
    if (
        weights.shape != (437, 437)
        or values.ndim != 3
        or values.shape[0] <= 0
        or values.shape[1:] != (437, 32)
        or initial.shape != (1,)
        or initial.dtype != jnp.float32
    ):
        raise ValueError("private PWA requires positive S, fixed N437/D32")
    weights = jax.lax.bitcast_convert_type(weights, jnp.uint16)
    values = jax.lax.bitcast_convert_type(values, jnp.uint16)

    def kernel(w_ref, v_ref, c_ref, out_ref, lane_ref):
        tile_m, tile_n = pl.program_id(0), pl.program_id(1)
        lane = jnp.arange(32, dtype=jnp.int32)
        group, thread = lane // 4, lane % 4

        def body(step, carry):
            kk = tuple(_residue_k(step * 8 + 2 * thread + half) for half in (0, 1))
            registers = []
            for register in (0, 1):
                m = tile_m * 16 + group + register * 8
                halves = tuple(
                    pt.load(
                        v_ref.at[m // 32, jnp.maximum(k, 0), m % 32],
                        mask=k >= 0,
                        other=0,
                    ).astype(jnp.uint32)
                    for k in kk
                )
                registers.append(halves[0] | (halves[1] << jnp.uint32(16)))
            n = tile_n * 8 + group
            halves = tuple(
                pt.load(
                    w_ref.at[n, jnp.maximum(k, 0)],
                    mask=(k >= 0) & (n < 437),
                    other=0,
                ).astype(jnp.uint32)
                for k in kk
            )
            registers.append(halves[0] | (halves[1] << jnp.uint32(16)))
            return tuple(_mma1688_p.bind(*registers, *carry))

        carry = (jnp.broadcast_to(c_ref[0], (32,)),) * 4
        carry = jax.lax.fori_loop(0, 56, body, carry)
        physical_lane = pt.elementwise_inline_asm(
            "mov.u32 $0, %laneid;",
            args=(lane,),
            constraints="=r,r",
            pack=1,
            result_shape_dtypes=[jax.ShapeDtypeStruct((32,), jnp.int32)],
        )[0]
        carry = _guard_lane_identity(carry, physical_lane)
        for register, value in enumerate(carry):
            m = tile_m * 16 + group + (register // 2) * 8
            n = tile_n * 8 + 2 * thread + register % 2
            pt.store(out_ref.at[m // 32, n, m % 32], value, mask=n < 437)
        pt.store(lane_ref.at[lane], physical_lane, mask=(tile_m == 0) & (tile_n == 0))

    return pl.pallas_call(
        kernel,
        out_shape=(
            jax.ShapeDtypeStruct(values.shape, jnp.float32),
            jax.ShapeDtypeStruct((32,), jnp.int32),
        ),
        grid=(values.shape[0] * 2, 55),
        compiler_params=pt.CompilerParams(num_warps=1, num_stages=1),
        debug=debug,
        name="boltz_native_pwa_mma1688",
    )(weights, values, initial)


def _pwa_outputs(weights, values, initial, *, debug=False):
    raw, lanes = _pwa_raw_fp32(weights, values, initial, debug=debug)
    return raw, raw.astype(jnp.bfloat16), lanes


def native_pwa_contraction(weight, value, *, original_msa_rows: int | None = None):
    """Return BF16 values; invalid physical lane packing poisons output with NaN."""
    if (
        not _observed_shape(weight, value, original_msa_rows=original_msa_rows)
        or cp_mesh() is not None
    ):
        raise ValueError(
            "native PWA requires unsharded singleton B/H, original S4436/N437/D32 "
            "BF16 and full rows or verified default row chunks"
        )
    _, output, _ = _pwa_outputs(weight[0, 0], value[0, 0], jnp.zeros(1, jnp.float32))
    return output[None, None]


def _observed_shape(weight, value, *, original_msa_rows: int | None = None):
    return (
        weight.shape == (1, 1, 437, 437)
        and value.ndim == 5
        and value.shape[:2] == (1, 1)
        and value.shape[3:] == (437, 32)
        and weight.dtype == value.dtype == jnp.bfloat16
        and (
            (value.shape[2] == 4436 and original_msa_rows in (None, 4436))
            or (original_msa_rows == 4436 and value.shape[2] in (1199, 839))
        )
    )


def pair_weighted_contraction(weight, value, *, original_msa_rows: int | None = None):
    """Keep every unobserved shape, non-AMP, CP and non-CUDA route unchanged."""

    def existing(w, v):
        return jnp.einsum("bhij,bhsjd->bhsid", w, v)

    if (
        not _observed_shape(weight, value, original_msa_rows=original_msa_rows)
        or cp_mesh() is not None
        or _mma1688_p is None
    ):
        return existing(weight, value)
    return jax.lax.platform_dependent(
        weight,
        value,
        cuda=partial(native_pwa_contraction, original_msa_rows=original_msa_rows),
        default=existing,
    )
