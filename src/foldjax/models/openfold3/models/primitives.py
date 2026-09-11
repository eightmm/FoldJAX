"""JAX primitives matching OpenFold3 inference modules.

Each parameter container mirrors one ``torch.nn.Module``'s ``state_dict`` layout
so the mapping from a checkpoint stays explicit and greppable. Forward passes are
pure functions over arrays.

References are to ``openfold3/core/model/primitives`` and
``openfold3/core/model/layers/transition.py`` in the upstream checkout.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from foldjax.models._cp import cp_mesh
from foldjax.models._glu import gated_linear_unit


class LinearParams(NamedTuple):
    """Parameters for ``openfold3.core.model.primitives.Linear``.

    OpenFold3's ``Linear`` subclasses ``torch.nn.Linear`` and only customizes
    initialization, so the inference math is a plain affine map. PyTorch stores
    the weight as ``[out_features, in_features]``; JAX contracts over the final
    input axis, so the forward pass multiplies by ``weight.T``.
    """

    weight: jnp.ndarray
    bias: jnp.ndarray | None = None


class LayerNormParams(NamedTuple):
    """Parameters for ``openfold3.core.model.primitives.LayerNorm``.

    Both the scale and the offset are optional upstream (``create_scale`` /
    ``create_offset``), and AdaLN relies on that: it normalizes ``a`` with
    neither and ``s`` with a scale only.
    """

    weight: jnp.ndarray | None = None
    bias: jnp.ndarray | None = None


class SwiGLUParams(NamedTuple):
    """Parameters for ``openfold3.core.model.primitives.SwiGLU``.

    Both projections are bias-free upstream (``swiglu_init``).
    """

    linear_a: LinearParams
    linear_b: LinearParams


class AdaLNParams(NamedTuple):
    """Parameters for ``openfold3.core.model.primitives.AdaLN`` (AF3 Alg. 26).

    ``linear_g`` carries a bias and ``linear_s`` does not (``ada_ln_init``).
    """

    layer_norm_a: LayerNormParams
    layer_norm_s: LayerNormParams
    linear_g: LinearParams
    linear_s: LinearParams


class SwiGLUTransitionParams(NamedTuple):
    """Parameters for ``SwiGLUTransition`` (AF3 Algorithm 11)."""

    layer_norm: LayerNormParams
    swiglu: SwiGLUParams
    linear_out: LinearParams


def linear(x: jnp.ndarray, params: LinearParams) -> jnp.ndarray:
    """Apply a PyTorch-layout linear projection."""
    y = jnp.matmul(x, jnp.swapaxes(params.weight, -1, -2))
    if params.bias is not None:
        y = y + params.bias
    return y


def layer_norm(
    x: jnp.ndarray, params: LayerNormParams, *, eps: float = 1e-5
) -> jnp.ndarray:
    """Apply layer norm over the final axis with optional scale and offset.

    Upstream disables autocast here for the *reverse* of the usual reason --
    to force float32 rather than to hold bfloat16. ``LayerNorm.forward``
    (``core/model/primitives/normalization.py:54-70``) takes ``x.float()``
    with ``weight.float()`` and ``bias.float()``, normalises in float32 and
    rounds once on the way out, under the comment "LayerNorm should be
    upcasted to fp32 anyway in torch / This enforces it if not running with
    autocast context". Accumulating narrow instead rounds the mean and the
    variance back before ``x - mean`` and applies the affine narrow as well:
    four extra roundings per norm against upstream's one, at every layer norm
    in a narrowed region. Protenix's ``layer_norm`` already does this and its
    bfloat16 arm holds at 3,012 tokens where OpenFold3's drifts.

    The arrangement is Protenix's; the guard reads the *promoted* dtype rather
    than ``x.dtype`` alone, and the difference is forced by this port's
    parameter split. Protenix narrows parameters and activations together per
    region, so there ``x.dtype`` settles both questions. Here they come apart:
    the diffusion conditioning's pair branch is narrowed while the whole
    denoiser is pinned float32, so a bfloat16 ``zij_trunk`` reaches float32
    norm parameters at ``atom_features.py:170``. Reading the promotion
    normalises that site wide too -- it is narrow today, and it is the one
    place where rounding the *output* to bfloat16, as upstream's
    ``out.to(dtype=d)`` would, narrows a region this port pins wide. So the
    accumulation widens everywhere and no realised dtype moves anywhere.
    """
    operands = (x, *(value for value in params if value is not None))
    output_dtype = jnp.result_type(*operands)
    if any(value.dtype == jnp.bfloat16 for value in operands):
        x = x.astype(jnp.float32)
        params = LayerNormParams(
            *(None if value is None else value.astype(jnp.float32) for value in params)
        )
    mean = jnp.mean(x, axis=-1, keepdims=True)
    variance = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
    y = (x - mean) * jax_rsqrt(variance + eps)
    if params.weight is not None:
        y = y * params.weight
    if params.bias is not None:
        y = y + params.bias
    return y if y.dtype == output_dtype else y.astype(output_dtype)


def jax_rsqrt(x: jnp.ndarray) -> jnp.ndarray:
    """Reciprocal square root, spelled out to keep the layer norm readable."""
    return jnp.reciprocal(jnp.sqrt(x))


def silu(x: jnp.ndarray) -> jnp.ndarray:
    """SiLU activation matching ``torch.nn.SiLU``."""
    return x * jax_sigmoid(x)


def jax_sigmoid(x: jnp.ndarray) -> jnp.ndarray:
    """Numerically stable logistic sigmoid."""
    return jnp.where(
        x >= 0,
        jnp.reciprocal(1.0 + jnp.exp(-jnp.abs(x))),
        jnp.exp(-jnp.abs(x)) * jnp.reciprocal(1.0 + jnp.exp(-jnp.abs(x))),
    )


def swiglu(
    x: jnp.ndarray, params: SwiGLUParams, *, glu_backend: str = "xla"
) -> jnp.ndarray:
    """Apply ``swish(linear_a(x)) * linear_b(x)``.

    Upstream uses its ordinary SiLU path unless explicitly given
    ``use_kernel=True``; SwiGLUTransition does not pass that flag, so ``"xla"``
    is what the released architecture computes and stays the default here.

    ``glu_backend="tokamax"`` runs the same product through the fused Triton
    kernel, which never materializes the two widened projections. The kernel
    applies ``jax.nn.silu`` at its own width, where this path applies the
    port's :func:`silu`; treat the switch as a numerics change and read it
    against the port's rerun floor. See :mod:`foldjax.models._glu`.
    """
    if glu_backend != "xla":
        if cp_mesh() is not None:
            raise ValueError(
                "context parallelism requires glu_backend='xla'; a fused GLU "
                "cannot be partitioned"
            )
        if params.linear_a.bias is not None or params.linear_b.bias is not None:
            raise ValueError(
                "the fused GLU takes bias-free projections; upstream's "
                "swiglu_init builds both of them without a bias"
            )
        # ``LinearParams.weight`` is torch's ``[out, in]``; the kernel wants
        # ``[in, out]`` per branch, with the activated branch first.
        return gated_linear_unit(
            x,
            jnp.swapaxes(params.linear_a.weight, -1, -2),
            jnp.swapaxes(params.linear_b.weight, -1, -2),
            jax.nn.silu,
            backend=glu_backend,
        )
    return silu(linear(x, params.linear_a)) * linear(x, params.linear_b)


def adaln(
    a: jnp.ndarray, s: jnp.ndarray, params: AdaLNParams, *, eps: float = 1e-5
) -> jnp.ndarray:
    """Apply adaptive layer norm (AF3 Algorithm 26)."""
    a = layer_norm(a, params.layer_norm_a, eps=eps)
    s = layer_norm(s, params.layer_norm_s, eps=eps)
    gate = jax_sigmoid(linear(s, params.linear_g))
    return gate * a + linear(s, params.linear_s)


def swiglu_transition(
    x: jnp.ndarray,
    params: SwiGLUTransitionParams,
    *,
    mask: jnp.ndarray | None = None,
    eps: float = 1e-5,
    glu_backend: str = "xla",
) -> jnp.ndarray:
    """Apply the SwiGLU transition (AF3 Algorithm 11).

    ``mask`` is ``[..., N]``; upstream expands it to ``[..., N, 1]`` and
    multiplies the output. A missing mask means all-ones, matching upstream.
    """
    y = layer_norm(x, params.layer_norm, eps=eps)
    y = swiglu(y, params.swiglu, glu_backend=glu_backend)
    y = linear(y, params.linear_out)
    if mask is not None:
        y = y * mask[..., None]
    return y


class ConditionedTransitionBlockParams(NamedTuple):
    """Parameters for ``ConditionedTransitionBlock`` (AF3 Algorithm 25).

    ``linear_g`` carries a bias initialized to -2 (``gating_ada_zero``), which is
    what makes the block start near-closed; ``linear_out`` is bias-free.
    """

    layer_norm: AdaLNParams
    swiglu: SwiGLUParams
    linear_g: LinearParams
    linear_out: LinearParams


def conditioned_transition_block(
    a: jnp.ndarray,
    s: jnp.ndarray,
    params: ConditionedTransitionBlockParams,
    *,
    mask: jnp.ndarray | None = None,
    eps: float = 1e-5,
    glu_backend: str = "xla",
) -> jnp.ndarray:
    """Apply an AdaLN-conditioned SwiGLU transition with an AdaLN-zero gate.

    Args:
        a: ``[..., N, C_a]`` activation to update.
        s: ``[..., N, C_s]`` conditioning single representation.
        params: mapped parameters.
        mask: ``[..., N]`` mask; ``None`` means all ones.
        eps: layer norm epsilon.
        glu_backend: which gated-linear-unit path the SwiGLU takes.

    Returns:
        ``[..., N, C_a]`` update. Unlike ``swiglu_transition`` there is no
        residual here: the gate replaces it.
    """
    normed = adaln(a, s, params.layer_norm, eps=eps)
    hidden = swiglu(normed, params.swiglu, glu_backend=glu_backend)
    out = jax_sigmoid(linear(s, params.linear_g)) * linear(hidden, params.linear_out)
    if mask is not None:
        out = out * mask[..., None]
    return out
