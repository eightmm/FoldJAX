"""Chai-1 component state_dict -> JAX param pytree mappers.

Pure-mapping functions are torch-free; tensors are duck-typed and converted with
``jnp.asarray``. Keeps torch weight layout (Linear stores ``weight`` as
``(out, in)``) so a JAX linear computes ``x @ weight.T (+ bias)``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, NamedTuple

import jax.numpy as jnp


def require_key(state: Mapping[str, Any], key: str) -> Any:
    if key not in state:
        raise KeyError(f"missing weight key: {key!r}")
    return state[key]


class LinearParams(NamedTuple):
    weight: jnp.ndarray  # (out, in) torch layout
    bias: jnp.ndarray | None


def map_linear(
    state: Mapping[str, Any], prefix: str, *, bias: bool = True
) -> LinearParams:
    wkey = f"{prefix}.weight" if prefix else "weight"
    w = jnp.asarray(require_key(state, wkey))
    b = None
    if bias:
        bkey = f"{prefix}.bias" if prefix else "bias"
        b = jnp.asarray(require_key(state, bkey))
    return LinearParams(weight=w, bias=b)


def apply_linear(params: LinearParams, x: jnp.ndarray) -> jnp.ndarray:
    y = x @ params.weight.T
    if params.bias is not None:
        y = y + params.bias
    return y


class LayerNormParams(NamedTuple):
    weight: jnp.ndarray | None
    bias: jnp.ndarray | None


def map_layer_norm(
    state: Mapping[str, Any],
    prefix: str,
    *,
    scale: bool = True,
    offset: bool = True,
) -> LayerNormParams:
    w = jnp.asarray(require_key(state, f"{prefix}.weight")) if scale else None
    b = jnp.asarray(require_key(state, f"{prefix}.bias")) if offset else None
    return LayerNormParams(weight=w, bias=b)


def apply_layer_norm(
    params: LayerNormParams, x: jnp.ndarray, *, eps: float = 1e-5
) -> jnp.ndarray:
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    y = (x - mean) / jnp.sqrt(var + eps)
    if params.weight is not None:
        y = y * params.weight
    if params.bias is not None:
        y = y + params.bias
    return y


# --- First proven leaf: bond_loss_input_proj is a no-bias Linear(1, 512) ---
def map_bond_loss_input_proj(state: Mapping[str, Any]) -> LinearParams:
    """Map ``bond_loss_input_proj.pt`` (single no-bias linear, key 'weight')."""
    return map_linear(state, "", bias=False)
