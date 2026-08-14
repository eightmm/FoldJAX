"""Publisher-runtime counterparts for the JAX-only micro-module probes."""

from __future__ import annotations

from collections.abc import Mapping
from math import sqrt
from typing import Any

import numpy as np

ArrayTree = dict[str, Any]


def torch_pairformer_forward(
    params: list[ArrayTree],
    s: Any,
    z: Any,
    mask: Any,
    pair_mask: Any,
) -> tuple[Any, Any]:
    """Run the Pairformer-like stack in the publisher runtime."""

    for layer in params:
        s, z = _torch_pairformer_layer(layer, s, z, mask, pair_mask)
    return s, z


def torch_structure_forward(
    params: ArrayTree,
    s: Any,
    z: Any,
    coords: Any,
    atom_token: Any,
    steps: int,
) -> Any:
    """Run a structure-like iterative coordinate update for parity."""

    import torch

    pair_context = torch.mean(z, dim=2)
    sigma = _torch_sigma_schedule(steps, coords.device)
    for sigma_t in sigma:
        centered = coords - torch.mean(coords, dim=-2, keepdim=True)
        token_atom = torch.index_select(s, 1, atom_token)
        pair_atom = torch.index_select(pair_context, 1, atom_token)
        hidden = torch.nn.functional.gelu(
            _torch_linear(centered, params["coord_in"])
            + _torch_linear(token_atom, params["token_in"])
            + _torch_linear(pair_atom, params["pair_in"])
            + params["hidden_bias"]
        )
        update = _torch_linear(hidden, params["coord_out"])
        scale = sigma_t / (sigma_t + 1.0)
        coords = centered + scale * update
    return coords


def to_torch_tree(tree: Mapping[str, Any], device: str) -> ArrayTree:
    """Convert a NumPy pytree to publisher-runtime tensors."""

    import torch

    def convert(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            dtype = torch.int64 if np.issubdtype(value.dtype, np.integer) else None
            return torch.as_tensor(value, device=device, dtype=dtype)
        return value

    return _tree_map(convert, tree)


def _torch_pairformer_layer(
    p: ArrayTree,
    s: Any,
    z: Any,
    mask: Any,
    pair_mask: Any,
) -> tuple[Any, Any]:
    zn = _torch_layer_norm(z, p["z_norm_scale"], p["z_norm_bias"])
    z = z + _torch_triangle_out(zn, p) * pair_mask[..., None]
    z = z + _torch_triangle_in(zn, p) * pair_mask[..., None]
    z = z + _torch_transition(z, p["z_fc1"], p["z_fc2"]) * pair_mask[..., None]

    sn = _torch_layer_norm(s, p["s_norm_scale"], p["s_norm_bias"])
    s = s + _torch_attention(sn, z, mask, p)
    s = s + _torch_transition(s, p["s_fc1"], p["s_fc2"]) * mask[..., None]
    return s, z


def _torch_attention(s: Any, z: Any, mask: Any, p: ArrayTree) -> Any:
    import torch

    heads = p["pair_bias"].shape[-1]
    head_dim = s.shape[-1] // heads
    q = _torch_linear(s, p["q"]).reshape(*s.shape[:-1], heads, head_dim)
    k = _torch_linear(s, p["k"]).reshape(*s.shape[:-1], heads, head_dim)
    v = _torch_linear(s, p["v"]).reshape(*s.shape[:-1], heads, head_dim)
    q = torch.swapaxes(q, 1, 2)
    k = torch.swapaxes(k, 1, 2)
    v = torch.swapaxes(v, 1, 2)
    logits = torch.einsum("bhid,bhjd->bhij", q, k) / sqrt(head_dim)
    logits = logits + torch.einsum("bijc,ch->bhij", z, p["pair_bias"])
    logits = torch.where(mask[:, None, None, :] > 0, logits, -1e9)
    attn = torch.softmax(logits, dim=-1)
    out = torch.einsum("bhij,bhjd->bhid", attn, v)
    out = torch.swapaxes(out, 1, 2).reshape(s.shape)
    return _torch_linear(out, p["o"]) * mask[..., None]


def _torch_triangle_out(z: Any, p: ArrayTree) -> Any:
    import torch

    a = _torch_linear(z, p["tri_out_a"])
    b = _torch_linear(z, p["tri_out_b"])
    out = torch.einsum("bikc,bjkc->bijc", a, b) / sqrt(z.shape[1])
    return _torch_linear(out, p["tri_out_o"])


def _torch_triangle_in(z: Any, p: ArrayTree) -> Any:
    import torch

    a = _torch_linear(z, p["tri_in_a"])
    b = _torch_linear(z, p["tri_in_b"])
    out = torch.einsum("bkic,bkjc->bijc", a, b) / sqrt(z.shape[1])
    return _torch_linear(out, p["tri_in_o"])


def _torch_transition(x: Any, fc1: Any, fc2: Any) -> Any:
    import torch

    return _torch_linear(torch.nn.functional.gelu(_torch_linear(x, fc1)), fc2)


def _torch_layer_norm(x: Any, scale: Any, bias: Any, eps: float = 1e-5) -> Any:
    import torch

    mean = torch.mean(x, dim=-1, keepdim=True)
    var = torch.mean((x - mean) ** 2, dim=-1, keepdim=True)
    return (x - mean) * torch.rsqrt(var + eps) * scale + bias


def _torch_linear(x: Any, kernel: Any) -> Any:
    return x @ kernel


def _torch_sigma_schedule(steps: int, device: Any) -> Any:
    import torch

    return torch.linspace(1.0, 0.01, steps, dtype=torch.float32, device=device)


def _tree_map(fn: Any, tree: Any) -> Any:
    if isinstance(tree, dict):
        return {key: _tree_map(fn, value) for key, value in tree.items()}
    if isinstance(tree, list):
        return [_tree_map(fn, value) for value in tree]
    return fn(tree)
