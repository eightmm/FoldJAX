"""Triangular multiplicative update (AF3 Algorithms 12 and 13).

Upstream ships four variants of this layer: a plain one, a chunked in-place
inference one, a cuEquivariance-fused one, and a "fused" one that concatenates
the ``a``/``b`` projections into single weights. They compute the same function
from differently-laid-out weights. This module implements the plain reference
path, which is what the released non-fused checkpoints store.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

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
    """
    z = layer_norm(z, params.layer_norm_in, eps=eps)
    gate_mask = 1.0 if mask is None else mask[..., None]

    a = gate_mask * jax_sigmoid(linear(z, params.linear_a_g)) * linear(
        z, params.linear_a_p
    )
    b = gate_mask * jax_sigmoid(linear(z, params.linear_b_g)) * linear(
        z, params.linear_b_p
    )

    x = combine_projections(a, b, outgoing=outgoing)
    x = layer_norm(x, params.layer_norm_out, eps=eps)
    x = linear(x, params.linear_z)
    return x * jax_sigmoid(linear(z, params.linear_g))
