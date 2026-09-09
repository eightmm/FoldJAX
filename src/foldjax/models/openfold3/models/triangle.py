"""Triangular multiplicative update (AF3 Algorithms 12 and 13).

Upstream ships four variants of this layer: a plain one, a chunked in-place
inference one, a cuEquivariance-fused one, and a "fused" one that concatenates
the ``a``/``b`` projections into single weights. They compute the same function
from differently-laid-out weights. This module implements the plain reference
path, which is what the released non-fused checkpoints store.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from foldjax._openfold3_compile import resolve_triangle_kernel
from foldjax.models._cp import (
    CP_COL_AXIS,
    CP_ROW_AXIS,
    col_skew_perm,
    cp_grid,
    cp_layout,
    cp_mesh,
    pair_spec,
    permute,
    ring_perm,
    row_skew_perm,
    shard_pair_rows,
    transpose_perm,
)
from foldjax.models.openfold3.models.primitives import (
    LayerNormParams,
    LinearParams,
    jax_sigmoid,
    layer_norm,
    linear,
)


class TriangleMultiplicationParams(NamedTuple):
    """Parameters for ``TriangleMultiplicativeUpdate``.

    Every projection is bias-free upstream (``tri_mul_init``). The two layer
    norms carry both scale and offset.
    """

    layer_norm_in: LayerNormParams
    layer_norm_out: LayerNormParams
    linear_a_p: LinearParams
    linear_a_g: LinearParams
    linear_b_p: LinearParams
    linear_b_g: LinearParams
    linear_g: LinearParams
    linear_z: LinearParams


def permute_final_dims(x: jnp.ndarray, inds: tuple[int, ...]) -> jnp.ndarray:
    """Permute the final ``len(inds)`` axes, matching OpenFold's helper."""
    zero_index = -len(inds)
    leading = list(range(len(x.shape[:zero_index])))
    return jnp.transpose(x, leading + [len(x.shape) + zero_index + i for i in inds])


def combine_projections(
    a: jnp.ndarray, b: jnp.ndarray, *, outgoing: bool
) -> jnp.ndarray:
    """Contract the two pair projections along the triangle's shared axis.

    Outgoing and incoming differ only in which of the two spatial axes is
    contracted, expressed upstream as a different pre-permutation.
    """
    if outgoing:
        a = permute_final_dims(a, (2, 0, 1))
        b = permute_final_dims(b, (2, 1, 0))
    else:
        a = permute_final_dims(a, (2, 1, 0))
        b = permute_final_dims(b, (2, 0, 1))
    p = jnp.einsum("...ij,...jk->...ik", a, b)
    return permute_final_dims(p, (1, 2, 0))


def triangle_multiplication(
    z: jnp.ndarray,
    params: TriangleMultiplicationParams,
    *,
    outgoing: bool,
    mask: jnp.ndarray | None = None,
    eps: float = 1e-5,
) -> jnp.ndarray:
    """Apply one triangular multiplicative update.

    Args:
        z: ``[..., N, N, C_z]`` pair representation.
        params: mapped layer parameters.
        outgoing: ``True`` for AF3 Algorithm 12, ``False`` for Algorithm 13.
        mask: ``[..., N, N]`` pair mask; ``None`` means all ones.
        eps: layer norm epsilon.

    Returns:
        ``[..., N, N, C_z]`` update. Upstream returns the update only; the
        caller adds it to ``z``.

    The ambient ``cueq-full`` kernel (see :func:`resolve_triangle_kernel`)
    routes this through cuEquivariance's fused update, which is the same
    kernel upstream's ``use_cueq_triangle_kernels`` selects. ``cueq`` alone
    keeps attention fused and this multiplication in XLA, so the two remain
    separately measurable arms.
    """
    if resolve_triangle_kernel(None, cp_shards=1) == "cueq-full":
        if cp_mesh() is not None:
            raise ValueError(
                "cueq-full triangle multiplication does not support context "
                "parallelism; select cueq or xla"
            )
        return _cueq_triangle_multiplication(
            z, params, outgoing=outgoing, mask=mask, eps=eps
        )
    z = layer_norm(z, params.layer_norm_in, eps=eps)
    gate_mask = 1.0 if mask is None else mask[..., None]

    a = gate_mask * jax_sigmoid(linear(z, params.linear_a_g)) * linear(
        z, params.linear_a_p
    )
    b = gate_mask * jax_sigmoid(linear(z, params.linear_b_g)) * linear(
        z, params.linear_b_p
    )

    if cp_layout() == "2d":
        # Fold-CP's own layout shards both pair axes, so neither operand is
        # full-width on any device and the contraction runs as Cannon's
        # algorithm instead: skew once, then one local product per ring hop.
        x = _cannon_combine(a, b, outgoing=outgoing)
    else:
        contract_outgoing = outgoing
        if cp_mesh() is not None and not outgoing:
            # The incoming contraction sums over the sharded row axis, which the
            # partitioner realises as a full-size partial sum plus an all-reduce
            # per device. Swapping the pair axes of both projections and using
            # the outgoing form is the same arithmetic (out[i,j] = sum_k
            # a[k,i] b[k,j] either way) with the partials sharded. The gate below
            # still reads the untransposed `z`.
            a = shard_pair_rows(jnp.swapaxes(a, -3, -2))
            b = shard_pair_rows(jnp.swapaxes(b, -3, -2))
            contract_outgoing = True
        x = combine_projections(a, b, outgoing=contract_outgoing)
    x = shard_pair_rows(x)
    x = layer_norm(x, params.layer_norm_out, eps=eps)
    x = linear(x, params.linear_z)
    return x * jax_sigmoid(linear(z, params.linear_g))


def _cueq_triangle_multiplication(
    z: jnp.ndarray,
    params: TriangleMultiplicationParams,
    *,
    outgoing: bool,
    mask: jnp.ndarray | None,
    eps: float,
) -> jnp.ndarray:
    """Run the fused cuEquivariance update with upstream's parameter packing.

    Upstream's ``_cueq_triangle_mult`` concatenates the ``a``/``b`` gate and
    projection weights along the output axis; the kernel then computes the
    gated projections, the contraction, the output norm and the output gate
    in one pass and returns the update without the residual. Leading axes
    beyond ``[N, N, C]`` fold into the kernel's batch axis.
    """
    from foldjax.models._cueq import load_cueq, triangle_multiplication_precision

    required = {
        "layer_norm_in.weight": params.layer_norm_in.weight,
        "layer_norm_in.bias": params.layer_norm_in.bias,
        "layer_norm_out.weight": params.layer_norm_out.weight,
        "layer_norm_out.bias": params.layer_norm_out.bias,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError(f"cueq-full requires affine layer norms; missing {missing}")
    if z.ndim < 3 or z.shape[-3] != z.shape[-2]:
        raise ValueError("triangle multiplication requires square pair axes")
    cuex = load_cueq()
    lead, (n, channels) = z.shape[:-3], z.shape[-2:]
    x = z.reshape((-1, n, n, channels))
    pair_mask = (
        jnp.ones(z.shape[:-1], dtype=z.dtype)
        if mask is None
        else jnp.broadcast_to(mask, z.shape[:-1]).astype(z.dtype)
    ).reshape((-1, n, n))

    def packed(first, second):
        weight = jnp.concatenate((first.weight, second.weight), axis=0)
        if (first.bias is None) != (second.bias is None):
            raise ValueError("paired projections must both carry or omit a bias")
        bias = (
            None
            if first.bias is None
            else jnp.concatenate((first.bias, second.bias), axis=0)
        )
        return weight, bias

    p_in_weight, p_in_bias = packed(params.linear_a_p, params.linear_b_p)
    g_in_weight, g_in_bias = packed(params.linear_a_g, params.linear_b_g)
    out = cuex.triangle_multiplicative_update(
        x=x,
        direction="outgoing" if outgoing else "incoming",
        mask=pair_mask,
        norm_in_weight=params.layer_norm_in.weight,
        norm_in_bias=params.layer_norm_in.bias,
        p_in_weight=p_in_weight,
        p_in_bias=p_in_bias,
        g_in_weight=g_in_weight,
        g_in_bias=g_in_bias,
        norm_out_weight=params.layer_norm_out.weight,
        norm_out_bias=params.layer_norm_out.bias,
        p_out_weight=params.linear_z.weight,
        p_out_bias=params.linear_z.bias,
        g_out_weight=params.linear_g.weight,
        g_out_bias=params.linear_g.bias,
        eps=eps,
        precision=triangle_multiplication_precision(cuex, dtype=x.dtype),
        fallback=False,
    )
    return out.reshape((*lead, n, n, out.shape[-1]))


def _cannon_combine(
    a: jnp.ndarray, b: jnp.ndarray, *, outgoing: bool
) -> jnp.ndarray:
    """Contract two doubly-sharded pair projections by Cannon's algorithm.

    Under the ``2d`` layout device ``(p, q)`` holds the ``(p, q)`` tile of every
    pair tensor, so the contracted axis lives on a different device from the
    operand that needs it and no plain sharding constraint can express the
    schedule. Cannon's algorithm is the answer Fold-CP uses: align the operands
    once so every device starts on a matching block of ``k``, then alternate a
    local product with a one-hop shift, ``side`` times. The per-device transient
    is two tiles -- ``O((N/side)^2 C)`` -- and nothing full-width is ever built.

    The alignment. ``transpose_perm`` puts the ``(q, p)`` tile on ``(p, q)``;
    ``row_skew_perm`` leaves ``(p, q)`` holding row-block ``p``, column-block
    ``p + q``; ``col_skew_perm`` leaves it holding row-block ``p + q``, column-
    block ``q``. Composing those for the outgoing direction leaves device
    ``(p, q)`` with ``a[block p, block p+q]`` and ``b[block q, block p+q]``, so
    the two share block ``p + q`` of the contracted axis and the product belongs
    at output block ``(p, q)`` -- which is where ``out_specs`` puts it. Each ring
    step advances the shared block by one, so ``side`` steps cover every ``k``.

    The local einsum is deliberately the *same* one the dense path uses, once per
    direction. Cannon permutes whole tiles; it does not change what the axes
    inside a tile mean, so a tile of ``a`` is still indexed ``[i, k]`` for
    outgoing and ``[k, i]`` for incoming. Rewriting the step as a canonical
    ``[i,k] @ [k,j]`` matmul is the tempting error and it is wrong in both
    directions: measured against the dense result at N=12 on a 2x2 grid it is off
    by 15.9 (outgoing) and 12.3 (incoming) on values of order one.
    """
    mesh = cp_mesh()
    side = cp_grid()[0]
    n = a.shape[-3]
    pad = (-n) % side
    if pad:
        # `shard_map` needs both pair axes to divide the grid. The projections
        # are padded rather than `z`, so the padding is exactly zero by
        # construction instead of by way of the mask: a padded block of `k`
        # contributes nothing to the sum, and the padded output region is sliced
        # off below.
        widths = [(0, 0)] * a.ndim
        widths[-3] = widths[-2] = (0, pad)
        a = jnp.pad(a, widths)
        b = jnp.pad(b, widths)
    spec = pair_spec(a.ndim)
    # Leading batch axes -- the template stack's templates, the confidence
    # re-embedding's samples -- are unsharded and ride along through both the
    # permutations and the einsum, exactly as they do in the CP attention.
    subscript = "...ikd,...jkd->...ijd" if outgoing else "...kid,...kjd->...ijd"

    def body(lhs: jnp.ndarray, rhs: jnp.ndarray) -> jnp.ndarray:
        if outgoing:
            rhs = permute(rhs, transpose_perm(side))
        else:
            lhs = permute(lhs, transpose_perm(side))
        lhs = permute(lhs, row_skew_perm(side))
        rhs = permute(rhs, col_skew_perm(side))
        total = None
        correction = None
        for step in range(side):
            # float32 accumulation without widening the operands, which is what
            # AF3's BF16_BF16_F32 algorithm does; with a float32 trunk the two
            # forms are identical.
            term = jnp.einsum(
                subscript, lhs, rhs, preferred_element_type=jnp.float32
            )
            if total is None:
                total = term
                correction = jnp.zeros_like(term)
            else:
                updated = total + term
                residual = jnp.where(
                    jnp.abs(total) >= jnp.abs(term),
                    (total - updated) + term,
                    (term - updated) + total,
                )
                total = updated
                correction = correction + residual
            if step + 1 < side:
                lhs = permute(lhs, ring_perm(side, axis=CP_COL_AXIS, delta=-1))
                rhs = permute(rhs, ring_perm(side, axis=CP_ROW_AXIS, delta=-1))
        return total + correction

    out = jax.shard_map(body, mesh=mesh, in_specs=(spec, spec), out_specs=spec)(a, b)
    if pad:
        out = jax.lax.slice_in_dim(out, 0, n, axis=-3)
        out = jax.lax.slice_in_dim(out, 0, n, axis=-2)
    # Back to the projections' own dtype, which is what `combine_projections`
    # returns on the other branch; the accumulator above is the only part that
    # runs wider.
    return out.astype(a.dtype)
