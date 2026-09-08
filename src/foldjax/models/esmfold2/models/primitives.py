"""The small pieces every ESMFold2 block is built from.

Each function takes the parameters as the mapping upstream's `state_dict`
uses, so a block's parameters are that block's sub-tree and nothing has to be
renamed twice. Every one of these is checked against the torch module it
mirrors in `tests/models/esmfold2/test_primitives_parity.py`.

Two conventions, both upstream's rather than this project's:

* `nn.Linear` stores its weight transposed for a right-multiply, so a linear
  is `x @ w.T`, and this module does not pre-transpose anything on load --
  the checkpoint keys stay recognisable against upstream's own source.
* SwiGLU packs its two projections into one `w12` and splits the result, so
  the split point is `hidden_features` and the *first* half is the one that
  passes through silu.
"""

from __future__ import annotations

from collections.abc import Mapping

import jax
import jax.numpy as jnp

Params = Mapping[str, jnp.ndarray]


def linear(x: jnp.ndarray, params: Params, prefix: str) -> jnp.ndarray:
    """`nn.Linear`, bias included when the checkpoint has one."""
    out = jnp.matmul(x, params[f"{prefix}.weight"].T)
    bias = params.get(f"{prefix}.bias")
    return out if bias is None else out + bias


def _cuda_profile_divide(numerator, denominator):
    """Native masked-MSA count division, without reciprocal approximation."""
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import triton as pt

    if numerator.dtype != jnp.float32 or denominator.dtype != jnp.float32:
        raise ValueError("native profile division requires FP32 operands")
    if numerator.size == 0:
        return numerator / denominator
    denominator = jnp.broadcast_to(denominator, numerator.shape)
    size = numerator.size
    padding = (-size) % 256
    left = jnp.pad(numerator.reshape(-1), (0, padding))
    right = jnp.pad(denominator.reshape(-1), (0, padding), constant_values=1)

    def kernel(a, b, out):
        out[:] = pt.elementwise_inline_asm(
            "div.rn.f32 $0, $1, $2;",
            args=(a[:], b[:]),
            constraints="=f,f,f",
            pack=1,
            result_shape_dtypes=[jax.ShapeDtypeStruct((256,), jnp.float32)],
        )[0]

    result = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(left.shape, jnp.float32),
        grid=(left.size // 256,),
        in_specs=(pl.BlockSpec((256,), lambda i: (i,)),) * 2,
        out_specs=pl.BlockSpec((256,), lambda i: (i,)),
        compiler_params=pt.CompilerParams(num_warps=4),
    )(left, right)
    return result[:size].reshape(numerator.shape)


def layer_norm(
    x: jnp.ndarray,
    weight: jnp.ndarray | None = None,
    bias: jnp.ndarray | None = None,
    eps: float = 1e-5,
) -> jnp.ndarray:
    """`F.layer_norm` over the last axis, with optional affine terms.

    Upstream calls this three ways -- with both terms, with a weight alone
    (`AdaptiveLayerNorm`'s conditioning path), and with neither (its activation
    path) -- so both are optional rather than there being three functions.

    Statistics and affine arithmetic use float32; this helper returns the
    input dtype. Native CUDA autocast instead returns float32 from LayerNorm,
    so callers implementing that policy must pass a float32 input and retain
    the original affine parameters. FP32 statistics alone do not establish
    native mixed-precision equivalence.
    """
    dtype = x.dtype
    x = x.astype(jnp.float32)
    mean = jnp.mean(x, axis=-1, keepdims=True)
    variance = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
    out = (x - mean) * jax.lax.rsqrt(variance + eps)
    if weight is not None:
        out = out * weight.astype(jnp.float32)
    if bias is not None:
        out = out + bias.astype(jnp.float32)
    return out.astype(dtype)


#: Bytes of widened SwiGLU allowed live at once before the rows are blocked.
#: The same budget `protenix` uses for the transition it shares this shape
#: with; see `models/protenix/models/primitives/primitives.py`.
_SWIGLU_WIDE_BUDGET_BYTES = 512 * 1024**2


def _swiglu_block(x: jnp.ndarray, params: Params, dot: str) -> jnp.ndarray:
    """One SwiGLU over whatever rows it is handed."""
    packed = linear(x, params, f"{dot}w12")
    hidden = packed.shape[-1] // 2
    gate, value = packed[..., :hidden], packed[..., hidden:]
    return linear(jax.nn.silu(gate) * value, params, f"{dot}w3")


def _swiglu_row_axis(x: jnp.ndarray) -> int | None:
    """The first leading axis with something to divide, or None."""
    for axis in range(x.ndim - 1):
        if x.shape[axis] >= 2:
            return axis
    return None


def _swiglu_rows(x: jnp.ndarray, axis: int, wide: int) -> int | None:
    """Rows along `axis` whose widened form fits the budget, or None for whole."""
    per_row = wide * x.dtype.itemsize
    for index, size in enumerate(x.shape[:-1]):
        if index != axis:
            per_row *= size
    if per_row <= 0 or per_row * x.shape[axis] <= _SWIGLU_WIDE_BUDGET_BYTES:
        return None
    return max(1, _SWIGLU_WIDE_BUDGET_BYTES // per_row)


def swiglu(x: jnp.ndarray, params: Params, prefix: str = "") -> jnp.ndarray:
    """`SwiGLU`: packed `w12`, split, silu the first half, gate, project.

    The split point comes from the checkpoint rather than a configuration
    value: `w12` is `[2 * hidden, in]`, so hidden is half its first axis.

    **Blocked along a leading axis when the widened form is large.** `w12`
    widens `[..., C]` to `[..., 2 * hidden]` and the split halves plus their
    product are all live at once, so on this model's pair representation the
    packed buffer alone is `[N, N, 2048]` -- 3,930 MiB at 1,003 tokens in
    bfloat16, and XLA's own arena accounting named seven of them. Every
    operation here contracts over the channel axis only, so dividing any
    leading axis is exact arithmetic and needs no flag; it is not bit-identical,
    because the blocked shape gets a different GEMM tiling, which is the same
    caveat every other blocked path in this repository carries.

    Nothing is blocked below the budget, which leaves the MSA-shaped callers
    and every small input on the original single-call route.
    """
    dot = f"{prefix}." if prefix else ""
    axis = _swiglu_row_axis(x)
    if axis is None:
        return _swiglu_block(x, params, dot)
    wide = params[f"{dot}w12.weight"].shape[0]
    rows = _swiglu_rows(x, axis, wide)
    if rows is None or rows >= x.shape[axis]:
        return _swiglu_block(x, params, dot)
    return jnp.concatenate(
        [
            _swiglu_block(
                # `min` because a trailing block is shorter whenever the axis
                # does not divide, and `slice_in_dim` rejects an overrun rather
                # than clamping it the way Python slicing would.
                jax.lax.slice_in_dim(
                    x, start, min(start + rows, x.shape[axis]), axis=axis
                ),
                params,
                dot,
            )
            for start in range(0, x.shape[axis], rows)
        ],
        axis=axis,
    )


def transition_layer(
    x: jnp.ndarray,
    params: Params,
    prefix: str = "",
    eps: float = 1e-5,
    *,
    linear_dtype: object | None = None,
) -> jnp.ndarray:
    """`TransitionLayer`: norm, two projections, silu gate, project back.

    The same arithmetic as `swiglu` with the projections kept apart and a
    normalisation in front; upstream keeps both forms, so this port does too
    rather than unifying them and having to un-unify them at load time.
    """
    dot = f"{prefix}." if prefix else ""
    x = layer_norm(x, params[f"{dot}norm.weight"], params[f"{dot}norm.bias"], eps=eps)
    if linear_dtype is not None:
        # Autocast narrows Linear operands, not the preceding FP32 LayerNorm.
        x = x.astype(linear_dtype)
        params = {
            key: value.astype(linear_dtype)
            for key, value in params.items()
            if key.startswith(
                tuple(f"{dot}{name}." for name in ("a_proj", "b_proj", "out_proj"))
            )
        }
    a = linear(x, params, f"{dot}a_proj")
    b = linear(x, params, f"{dot}b_proj")
    return linear(jax.nn.silu(a) * b, params, f"{dot}out_proj")


def adaptive_layer_norm(
    a: jnp.ndarray,
    s: jnp.ndarray,
    params: Params,
    prefix: str = "",
    eps: float = 1e-5,
) -> jnp.ndarray:
    """`AdaptiveLayerNorm` (adaLN-Zero).

    The activation is normalised without affine terms and the conditioning
    with a weight but no bias; the gate is a sigmoid and multiplies, the shift
    adds. Getting the two normalisations the same way round is the whole of
    this function -- they use different parameter sets on purpose.
    """
    dot = f"{prefix}." if prefix else ""
    a_norm = layer_norm(a, eps=eps)
    s_norm = layer_norm(s, params[f"{dot}s_scale"], eps=eps)
    gate = jax.nn.sigmoid(linear(s_norm, params, f"{dot}s_gate"))
    return gate * a_norm + linear(s_norm, params, f"{dot}s_shift")


def fourier_embedding(
    t_hat: jnp.ndarray, params: Params, prefix: str = ""
) -> jnp.ndarray:
    """`FourierEmbedding`: cos(2 pi (t w + b)), with w and b as buffers.

    They are `register_buffer`s rather than parameters -- drawn once at
    construction and frozen -- so they travel in the checkpoint and are read
    from it here rather than being redrawn, which would silently change the
    embedding.
    """
    dot = f"{prefix}." if prefix else ""
    w = params[f"{dot}w"]
    b = params[f"{dot}b"]
    t = jnp.reshape(jnp.asarray(t_hat, dtype=w.dtype), (-1,))
    return jnp.cos(2.0 * jnp.pi * (t[:, None] * w[None, :] + b[None, :]))
