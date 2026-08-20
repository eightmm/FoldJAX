"""Pure JAX triangle multiplication blocks for the Boltz-2 port."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Literal

import jax
import jax.numpy as jnp

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
from foldjax.models.boltz2.models.primitives._common import layer_norm as _layer_norm
from foldjax.models.boltz2.models.primitives.glu_backend import gated_linear_unit

TriangleDirection = Literal["outgoing", "incoming"]
TriangleMultiplicationParams = Mapping[str, Mapping[str, jnp.ndarray]]


def triangle_multiplication_forward(
    params: TriangleMultiplicationParams,
    x: jnp.ndarray,
    mask: jnp.ndarray,
    direction: TriangleDirection,
    eps: float = 1e-5,
    chunk_size: int = 128,
    glu_backend: str = "xla",
    contraction_precision: str = "float32",
) -> jnp.ndarray:
    """Run Boltz triangle multiplication with mapped PyTorch parameters.

    ``glu_backend="tokamax"`` runs the projection+gate sigmoid GLU through the
    fused Triton kernel (GPU, low precision); the triangle contraction stays in
    XLA either way (cuBLAS-bound, no custom kernel — matches AF3). ``"xla"``
    (default) keeps the bit-exact elementwise gate.
    """

    # Context parallelism (an active `foldjax.models._cp` mesh) reroutes two
    # decisions, mirroring the OpenDDE/Protenix ports: the cueq kernel is an
    # FFI call the SPMD partitioner cannot split, so the environment default
    # resolves to the XLA einsum; and the chunked contraction slices the
    # sharded row axis, so it runs whole and the partitioner splits the one
    # einsum. A fused GLU cannot be partitioned either.
    cp_active = cp_mesh() is not None
    if cp_active:
        chunk_size = 0
        if glu_backend != "xla":
            msg = (
                "context-parallel triangle multiplication requires "
                f"glu_backend='xla'; got {glu_backend!r}"
            )
            raise ValueError(msg)

    backend = os.getenv("BOLTZ_JAX_TRIANGLE_MULTIPLICATION_BACKEND", "cueq")
    if backend == "cueq" and not cp_active:
        from foldjax.models.boltz2.models.triangle.triangle_cueq import (
            cueq_triangle_multiplication_forward,
        )

        return cueq_triangle_multiplication_forward(params, x, mask, direction, eps=eps)
    if backend not in ("xla", "cueq"):
        msg = f"Unsupported triangle multiplication backend: {backend!r}"
        raise ValueError(msg)

    x = _layer_norm(x, params["norm_in"]["scale"], params["norm_in"]["bias"], eps)
    out_dtype = x.dtype
    x_in = x
    mask = mask.astype(x.dtype)
    # sigmoid GLU: sigmoid(g_in(x)) * p_in(x)
    projected = gated_linear_unit(
        x,
        params["g_in"]["kernel"],
        params["p_in"]["kernel"],
        jax.nn.sigmoid,
        backend=glu_backend,
    )
    projected = projected * mask[..., None]
    # Coordinate sampling amplifies the small reduction-order differences of a
    # BF16-output contraction. Keep the historical FP32 contraction as the
    # accuracy default; the explicit BF16 mode is available for throughput
    # experiments where that drift has been validated for the target workload.
    if contraction_precision == "float32":
        contraction_input = projected.astype(jnp.float32)
    elif contraction_precision == "bf16":
        contraction_input = projected
    else:
        msg = f"Unsupported contraction_precision: {contraction_precision!r}"
        raise ValueError(msg)
    a, b = jnp.split(contraction_input, 2, axis=-1)

    if cp_layout() == "2d":
        # Fold-CP's own schedule: both pair axes are sharded, so the
        # contraction runs as Cannon's algorithm -- skew, then one local matmul
        # per ring hop. Nothing full-width is ever built, so the row-transpose
        # rewrite below (which only moves the all-reduce off the sharded axis)
        # has nothing left to fix.
        out = _cannon_contract(a, b, direction)
    else:
        if cp_active and direction == "incoming":
            # The incoming contraction sums over the sharded axis; the
            # partitioner realises that as a full-size partial sum plus an
            # all-reduce on every device. Swapping the pair axes of both
            # operands and contracting in the outgoing form is the same
            # arithmetic (out[i,j] = sum_k a[k,i] b[k,j] either way) with the
            # partials sharded.
            a = shard_pair_rows(jnp.swapaxes(a, 1, 2))
            b = shard_pair_rows(jnp.swapaxes(b, 1, 2))
            direction = "outgoing"
        out = _chunked_triangle_einsum(a, b, direction, chunk_size)
    out = shard_pair_rows(out)
    out = out.astype(out_dtype)

    out = _layer_norm(out, params["norm_out"]["scale"], params["norm_out"]["bias"], eps)
    out = _linear(out, params["p_out"]["kernel"])
    gate = jax.nn.sigmoid(_linear(x_in, params["g_out"]["kernel"]))
    return out * gate


def _cannon_contract(
    a: jnp.ndarray,
    b: jnp.ndarray,
    direction: TriangleDirection,
) -> jnp.ndarray:
    """Contract two 2-D-sharded pair projections by Cannon's algorithm.

    The triangle contraction is a per-channel matrix product: ``outgoing`` is
    ``out[i,j] = sum_k a[i,k] b[j,k]`` and ``incoming`` is
    ``out[i,j] = sum_k a[k,i] b[k,j]``. Both become a plain ``A @ B`` once the
    operand whose contracted axis sits on the wrong grid axis is transposed --
    ``b`` for outgoing, ``a`` for incoming -- which is why the two directions
    differ by one flag rather than by a rewrite.

    Cannon then aligns the operands so every device starts on matching
    indices: the left operand is skewed along grid columns by its row
    coordinate, the right along grid rows by its column coordinate. Each of
    the ``side`` steps multiplies the two resident tiles and shifts both one
    hop, so the per-device transient is two tiles -- ``O((N/side)^2 C)`` -- and
    no all-gather appears anywhere.
    """
    mesh = cp_mesh()
    side = cp_grid()[0]
    if a.ndim != 4:
        msg = (
            "the 2-D context-parallel triangle multiplication takes the native "
            f"[B, N, N, C] layout, got rank {a.ndim}"
        )
        raise ValueError(msg)
    if direction not in ("outgoing", "incoming"):
        msg = f"Unsupported triangle multiplication direction: {direction!r}"
        raise ValueError(msg)
    n = a.shape[1]
    pad = (-n) % side
    if pad:
        # Both pair axes have to divide the grid. The projections are already
        # masked, and the padding is zeros, so a padded k block contributes
        # nothing to the sum and the padded output region is sliced away.
        widths = ((0, 0), (0, pad), (0, pad), (0, 0))
        a = jnp.pad(a, widths)
        b = jnp.pad(b, widths)
    spec = pair_spec(4)

    def body(lhs, rhs):
        # Transposing a tensor that is blocked over both grid axes is two
        # moves, not one: `transpose_perm` sends the tile on (i, j) to (j, i),
        # and `swapaxes` transposes each tile's own pair axes. Doing only the
        # grid half leaves every device holding the right tile indexed the
        # wrong way round, which is silently wrong rather than a shape error --
        # every tile here is square.
        if direction == "outgoing":
            rhs = jnp.swapaxes(permute(rhs, transpose_perm(side)), 1, 2)
        else:
            lhs = jnp.swapaxes(permute(lhs, transpose_perm(side)), 1, 2)
        lhs = permute(lhs, row_skew_perm(side))
        rhs = permute(rhs, col_skew_perm(side))
        total = jnp.zeros(
            lhs.shape[:2] + rhs.shape[2:3] + lhs.shape[3:], dtype=jnp.float32
        )
        for step in range(side):
            total = total + jnp.einsum(
                "bikd,bkjd->bijd", lhs, rhs, preferred_element_type=jnp.float32
            )
            if step + 1 < side:
                lhs = permute(lhs, ring_perm(side, axis=CP_COL_AXIS, delta=-1))
                rhs = permute(rhs, ring_perm(side, axis=CP_ROW_AXIS, delta=-1))
        return total

    out = jax.shard_map(body, mesh=mesh, in_specs=(spec, spec), out_specs=spec)(a, b)
    if pad:
        out = out[:, :n, :n]
    return out


def _chunked_triangle_einsum(
    a: jnp.ndarray,
    b: jnp.ndarray,
    direction: TriangleDirection,
    chunk_size: int,
) -> jnp.ndarray:
    """Compute the triangle contraction in chunks over the output i axis.

    The contraction is over k (not split), so chunking the output i axis is
    exact: out[:, i_block] depends only on the i-block slice of ``a``.
    """

    if direction == "outgoing":
        n = a.shape[1]  # i is axis 1 of a in "bikd,bjkd->bijd"

        def block(start: int, size: int) -> jnp.ndarray:
            return jnp.einsum(
                "bikd,bjkd->bijd", jax.lax.dynamic_slice_in_dim(a, start, size, 1), b
            )

    elif direction == "incoming":
        n = a.shape[2]  # i is axis 2 of a in "bkid,bkjd->bijd"

        def block(start: int, size: int) -> jnp.ndarray:
            return jnp.einsum(
                "bkid,bkjd->bijd", jax.lax.dynamic_slice_in_dim(a, start, size, 2), b
            )

    else:
        msg = f"Unsupported triangle multiplication direction: {direction!r}"
        raise ValueError(msg)

    if chunk_size <= 0 or chunk_size >= n:
        return block(0, n)

    out = jnp.zeros((a.shape[0], n, b.shape[1], a.shape[-1]), dtype=a.dtype)
    for start in range(0, n, chunk_size):
        size = min(chunk_size, n - start)
        out = out.at[:, start : start + size].set(block(start, size))
    return out


def _linear(x: jnp.ndarray, kernel: jnp.ndarray) -> jnp.ndarray:
    return x @ kernel
