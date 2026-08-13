"""ESMFold2's pair trunk: triangle updates, transitions, and the block stack.

Written against `docs/ports/esmfold2/trunk-spec.md`, which records what the
torch source does and where it is easy to read it wrongly. The three places
that matter here, all checked by the parity tests beside this file:

* the triangular contraction runs in float32 whatever the surrounding dtype,
  because upstream calls `.float()` on its operands -- a bfloat16 trunk that
  skipped this would drift in a way no shape check catches;
* `proj_bundle` packs `[left | right | gate_left | gate_right]`, the input
  gate multiplies before the mask, and the output gate is computed from the
  same normalised input rather than from the contraction;
* `Transition` carries its residual internally and `PairTransition` does not,
  so the caller adds one and not the other.
"""

from __future__ import annotations

from collections.abc import Mapping

import jax
import jax.numpy as jnp

from foldjax.models.esmfold2.models.primitives import layer_norm, linear, swiglu

Params = Mapping[str, jnp.ndarray]


def triangle_multiplicative(
    pair: jnp.ndarray,
    params: Params,
    prefix: str,
    *,
    outgoing: bool,
    mask: jnp.ndarray | None = None,
    eps: float = 1e-5,
) -> jnp.ndarray:
    """`TriangleMultiplicativeBlock`, reference path.

    `prefix` names the `TriangleMultiplicativeUpdate` that owns it; the
    `_engine` level upstream inserts is added here so callers spell the module
    the way the checkpoint does.
    """
    engine = f"{prefix}._engine" if prefix else "_engine"
    normalised = layer_norm(
        pair,
        params[f"{engine}.norm_start.weight"],
        params[f"{engine}.norm_start.bias"],
        eps=eps,
    )
    bundled = linear(normalised, params, f"{engine}.proj_bundle")
    width = bundled.shape[-1] // 2
    signal, gate_logits = bundled[..., :width], bundled[..., width:]
    routed = signal * jax.nn.sigmoid(gate_logits)
    if mask is not None:
        routed = routed * mask[..., None].astype(routed.dtype)

    # Upstream forces float32 here and keeps `norm_mix` in it too.
    routed = routed.astype(jnp.float32)
    half = routed.shape[-1] // 2
    left, right = routed[..., :half], routed[..., half:]
    equation = "bikd,bjkd->bijd" if outgoing else "bkid,bkjd->bijd"
    contracted = jnp.einsum(equation, left, right, preferred_element_type=jnp.float32)

    mixed = layer_norm(
        contracted,
        params[f"{engine}.norm_mix.weight"],
        params[f"{engine}.norm_mix.bias"],
        eps=eps,
    )
    mixed = linear(mixed.astype(pair.dtype), params, f"{engine}.proj_emit")
    out_gate = jax.nn.sigmoid(linear(normalised, params, f"{engine}.proj_gate"))
    return mixed * out_gate


def transition(
    x: jnp.ndarray, params: Params, prefix: str, *, residual: bool, eps: float = 1e-5
) -> jnp.ndarray:
    """Norm, SwiGLU, and upstream's two different opinions about the residual.

    `common.Transition` adds `x` back; `modeling.PairTransition` returns the
    update alone and lets its caller add it. Same parameters, same shapes --
    only the caller can tell them apart, so it is a flag rather than a guess.
    """
    dot = f"{prefix}." if prefix else ""
    normalised = layer_norm(
        x, params[f"{dot}norm.weight"], params[f"{dot}norm.bias"], eps=eps
    )
    update = swiglu(normalised, params, f"{dot}ffn")
    return x + update if residual else update


def pair_update_block(
    pair: jnp.ndarray,
    params: Params,
    prefix: str,
    *,
    mask: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """One `PairUpdateBlock`: two triangle updates then a transition.

    The residuals are sequential rather than parallel -- the incoming update
    reads the pair the outgoing one just wrote -- and the transition's own
    residual is internal. Dropout is `r=0.0` throughout the trunk and disabled
    at inference besides, so it is simply absent here.
    """
    dot = f"{prefix}." if prefix else ""
    pair = pair + triangle_multiplicative(
        pair, params, f"{dot}tri_mul_out", outgoing=True, mask=mask
    )
    pair = pair + triangle_multiplicative(
        pair, params, f"{dot}tri_mul_in", outgoing=False, mask=mask
    )
    return transition(pair, params, f"{dot}pair_transition", residual=True)


def folding_trunk(
    pair: jnp.ndarray,
    params: Params,
    prefix: str = "",
    *,
    n_layers: int,
    mask: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """`FoldingTrunk`: `n_layers` blocks in sequence, no output norm.

    Whether the result replaces the pair or is added to it is the caller's
    business and upstream disagrees with itself about it: the main loop
    overwrites, the confidence head adds. Neither is done here.
    """
    dot = f"{prefix}." if prefix else ""
    for index in range(n_layers):
        pair = pair_update_block(pair, params, f"{dot}blocks.{index}", mask=mask)
    return pair


def outer_product_mean(
    msa: jnp.ndarray,
    params: Params,
    prefix: str,
    *,
    msa_mask: jnp.ndarray,
    eps: float = 1e-5,
) -> jnp.ndarray:
    """`OuterProductMean`, in the released arrangement.

    The division happens *after* the projection, so `Wout`'s bias is scaled by
    it too. The experimental module divides before, under the same key names;
    reproducing the release means this order.
    """
    dot = f"{prefix}." if prefix else ""
    normalised = layer_norm(
        msa, params[f"{dot}norm.weight"], params[f"{dot}norm.bias"], eps=eps
    )
    projected = linear(normalised, params, f"{dot}W")
    projected = projected * msa_mask[..., None].astype(projected.dtype)
    half = projected.shape[-1] // 2
    a, b = projected[..., :half], projected[..., half:]

    mask = msa_mask.astype(a.dtype)
    valid = jnp.maximum(jnp.einsum("bim,bjm->bij", mask, mask)[..., None], 1.0)
    outer = jnp.einsum("bimc,bjmd->bijcd", a, b)
    outer = outer.reshape(outer.shape[:-2] + (half * half,))
    return linear(outer, params, f"{dot}Wout") / valid


def msa_pair_weighted_averaging(
    msa: jnp.ndarray,
    pair: jnp.ndarray,
    params: Params,
    prefix: str,
    *,
    pair_mask: jnp.ndarray,
    eps: float = 1e-5,
) -> jnp.ndarray:
    """`MSAPairWeightedAveraging`.

    The softmax is over the *second* token axis and the gate multiplies after
    the sum over it, not before. The masked-out bias is -1e5 rather than
    -inf: upstream fills a finite value, and a fully masked row therefore
    still produces a finite softmax.
    """
    dot = f"{prefix}." if prefix else ""
    normalised = layer_norm(
        msa,
        params[f"{dot}norm_single.weight"],
        params[f"{dot}norm_single.bias"],
        eps=eps,
    )
    bias = layer_norm(
        pair,
        params[f"{dot}compute_bias.0.weight"],
        params[f"{dot}compute_bias.0.bias"],
        eps=eps,
    )
    bias = linear(bias, params, f"{dot}compute_bias.1")
    bias = jnp.where(pair_mask[..., None].astype(bool), bias, -1e5)
    attention = jax.nn.softmax(bias, axis=-2)

    heads = bias.shape[-1]
    value = linear(normalised, params, f"{dot}Wv")
    value = value.reshape(value.shape[:-1] + (heads, value.shape[-1] // heads))
    gate = jax.nn.sigmoid(linear(normalised, params, f"{dot}Wgate"))
    gate = gate.reshape(value.shape)

    out = jnp.einsum("bijh,bjmhd,bimhd->bimhd", attention, value, gate)
    out = out.reshape(out.shape[:-2] + (heads * value.shape[-1],))
    return linear(out, params, f"{dot}Wout")
