"""Small PyTorch/JAX-equivalent modules for early Boltz JAX probes.

These modules are not a full Boltz-2 port. They preserve the broad compute
structure needed for first-pass latency and memory experiments:

- Pairformer-like token attention with pair bias.
- Triangle-like O(N^3) pair updates.
- Transition MLPs.
- Structure-like iterative coordinate updates conditioned on token and pair
  activations.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import sqrt
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

ArrayTree = dict[str, Any]


@dataclass(frozen=True)
class MicroConfig:
    """Static shapes for a micro Pairformer/Structure probe."""

    batch: int = 1
    residues: int = 128
    token_s: int = 256
    token_z: int = 128
    heads: int = 8
    pairformer_blocks: int = 2
    structure_steps: int = 20
    atoms_per_residue: int = 4

    @property
    def atoms(self) -> int:
        return self.residues * self.atoms_per_residue


def init_micro_params(config: MicroConfig, seed: int = 0) -> ArrayTree:
    """Initialize shared NumPy parameters for PyTorch and JAX backends."""

    rng = np.random.default_rng(seed)
    return {
        "pairformer": [
            _init_pairformer_layer(config, rng) for _ in range(config.pairformer_blocks)
        ],
        "structure": _init_structure_params(config, rng),
    }


def init_micro_inputs(config: MicroConfig, seed: int = 1) -> ArrayTree:
    """Create deterministic random inputs for both backends."""

    rng = np.random.default_rng(seed)
    atom_token = np.repeat(np.arange(config.residues), config.atoms_per_residue)
    return {
        "s": rng.standard_normal(
            (config.batch, config.residues, config.token_s), dtype=np.float32
        ),
        "z": rng.standard_normal(
            (config.batch, config.residues, config.residues, config.token_z),
            dtype=np.float32,
        ),
        "coords": rng.standard_normal(
            (config.batch, config.atoms, 3), dtype=np.float32
        ),
        "mask": np.ones((config.batch, config.residues), dtype=np.float32),
        "pair_mask": np.ones(
            (config.batch, config.residues, config.residues), dtype=np.float32
        ),
        "atom_token": atom_token.astype(np.int32),
    }


def to_jax_tree(tree: Mapping[str, Any]) -> ArrayTree:
    """Convert a NumPy pytree to JAX arrays."""

    return jax.tree.map(
        lambda x: jnp.asarray(x) if isinstance(x, np.ndarray) else x,
        tree,
    )


def jax_pairformer_forward(
    params: list[ArrayTree],
    s: jnp.ndarray,
    z: jnp.ndarray,
    mask: jnp.ndarray,
    pair_mask: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Run the Pairformer-like stack in JAX."""

    for layer in params:
        s, z = _jax_pairformer_layer(layer, s, z, mask, pair_mask)
    return s, z


def jax_structure_forward(
    params: ArrayTree,
    s: jnp.ndarray,
    z: jnp.ndarray,
    coords: jnp.ndarray,
    atom_token: jnp.ndarray,
    steps: int,
) -> jnp.ndarray:
    """Run a structure-like iterative coordinate update in JAX."""

    pair_context = jnp.mean(z, axis=2)
    sigma = _jax_sigma_schedule(steps)

    def body(carry: jnp.ndarray, sigma_t: jnp.ndarray) -> jnp.ndarray:
        centered = carry - jnp.mean(carry, axis=-2, keepdims=True)
        token_atom = jnp.take(s, atom_token, axis=1)
        pair_atom = jnp.take(pair_context, atom_token, axis=1)
        hidden = jax.nn.gelu(
            _jax_linear(centered, params["coord_in"])
            + _jax_linear(token_atom, params["token_in"])
            + _jax_linear(pair_atom, params["pair_in"])
            + params["hidden_bias"]
        )
        update = _jax_linear(hidden, params["coord_out"])
        scale = sigma_t / (sigma_t + 1.0)
        return centered + scale * update

    return jax.lax.fori_loop(0, steps, lambda i, x: body(x, sigma[i]), coords)


def torch_pairformer_forward(
    params: list[ArrayTree],
    s: Any,
    z: Any,
    mask: Any,
    pair_mask: Any,
) -> tuple[Any, Any]:
    """Run the Pairformer-like stack in PyTorch."""

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
    """Run a structure-like iterative coordinate update in PyTorch."""

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
    """Convert a NumPy pytree to PyTorch tensors."""

    import torch

    def convert(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            dtype = torch.int64 if np.issubdtype(value.dtype, np.integer) else None
            return torch.as_tensor(value, device=device, dtype=dtype)
        return value

    return _tree_map(convert, tree)


def _init_pairformer_layer(config: MicroConfig, rng: np.random.Generator) -> ArrayTree:
    head_dim = config.token_s // config.heads
    if head_dim * config.heads != config.token_s:
        msg = "token_s must be divisible by heads"
        raise ValueError(msg)

    return {
        "s_norm_scale": np.ones((config.token_s,), dtype=np.float32),
        "s_norm_bias": np.zeros((config.token_s,), dtype=np.float32),
        "z_norm_scale": np.ones((config.token_z,), dtype=np.float32),
        "z_norm_bias": np.zeros((config.token_z,), dtype=np.float32),
        "q": _normal(rng, config.token_s, config.token_s),
        "k": _normal(rng, config.token_s, config.token_s),
        "v": _normal(rng, config.token_s, config.token_s),
        "o": _normal(rng, config.token_s, config.token_s),
        "pair_bias": _normal(rng, config.token_z, config.heads),
        "tri_out_a": _normal(rng, config.token_z, config.token_z),
        "tri_out_b": _normal(rng, config.token_z, config.token_z),
        "tri_out_o": _normal(rng, config.token_z, config.token_z),
        "tri_in_a": _normal(rng, config.token_z, config.token_z),
        "tri_in_b": _normal(rng, config.token_z, config.token_z),
        "tri_in_o": _normal(rng, config.token_z, config.token_z),
        "s_fc1": _normal(rng, config.token_s, config.token_s * 4),
        "s_fc2": _normal(rng, config.token_s * 4, config.token_s),
        "z_fc1": _normal(rng, config.token_z, config.token_z * 4),
        "z_fc2": _normal(rng, config.token_z * 4, config.token_z),
    }


def _init_structure_params(config: MicroConfig, rng: np.random.Generator) -> ArrayTree:
    hidden = config.token_s
    return {
        "coord_in": _normal(rng, 3, hidden),
        "token_in": _normal(rng, config.token_s, hidden),
        "pair_in": _normal(rng, config.token_z, hidden),
        "coord_out": _normal(rng, hidden, 3),
        "hidden_bias": np.zeros((hidden,), dtype=np.float32),
    }


def _normal(rng: np.random.Generator, fan_in: int, fan_out: int) -> np.ndarray:
    scale = 1.0 / sqrt(fan_in)
    return (rng.standard_normal((fan_in, fan_out)) * scale).astype(np.float32)


def _jax_pairformer_layer(
    p: ArrayTree,
    s: jnp.ndarray,
    z: jnp.ndarray,
    mask: jnp.ndarray,
    pair_mask: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    zn = _jax_layer_norm(z, p["z_norm_scale"], p["z_norm_bias"])
    z = z + _jax_triangle_out(zn, p) * pair_mask[..., None]
    z = z + _jax_triangle_in(zn, p) * pair_mask[..., None]
    z = z + _jax_transition(z, p["z_fc1"], p["z_fc2"]) * pair_mask[..., None]

    sn = _jax_layer_norm(s, p["s_norm_scale"], p["s_norm_bias"])
    s = s + _jax_attention(sn, z, mask, p)
    s = s + _jax_transition(s, p["s_fc1"], p["s_fc2"]) * mask[..., None]
    return s, z


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


def _jax_attention(
    s: jnp.ndarray, z: jnp.ndarray, mask: jnp.ndarray, p: ArrayTree
) -> jnp.ndarray:
    heads = p["pair_bias"].shape[-1]
    head_dim = s.shape[-1] // heads
    q = _jax_linear(s, p["q"]).reshape(*s.shape[:-1], heads, head_dim)
    k = _jax_linear(s, p["k"]).reshape(*s.shape[:-1], heads, head_dim)
    v = _jax_linear(s, p["v"]).reshape(*s.shape[:-1], heads, head_dim)
    q = jnp.swapaxes(q, 1, 2)
    k = jnp.swapaxes(k, 1, 2)
    v = jnp.swapaxes(v, 1, 2)
    logits = jnp.einsum("bhid,bhjd->bhij", q, k) / sqrt(head_dim)
    logits = logits + jnp.einsum("bijc,ch->bhij", z, p["pair_bias"])
    logits = jnp.where(mask[:, None, None, :] > 0, logits, -1e9)
    attn = jax.nn.softmax(logits, axis=-1)
    out = jnp.einsum("bhij,bhjd->bhid", attn, v)
    out = jnp.swapaxes(out, 1, 2).reshape(s.shape)
    return _jax_linear(out, p["o"]) * mask[..., None]


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


def _jax_triangle_out(z: jnp.ndarray, p: ArrayTree) -> jnp.ndarray:
    a = _jax_linear(z, p["tri_out_a"])
    b = _jax_linear(z, p["tri_out_b"])
    out = jnp.einsum("bikc,bjkc->bijc", a, b) / sqrt(z.shape[1])
    return _jax_linear(out, p["tri_out_o"])


def _jax_triangle_in(z: jnp.ndarray, p: ArrayTree) -> jnp.ndarray:
    a = _jax_linear(z, p["tri_in_a"])
    b = _jax_linear(z, p["tri_in_b"])
    out = jnp.einsum("bkic,bkjc->bijc", a, b) / sqrt(z.shape[1])
    return _jax_linear(out, p["tri_in_o"])


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


def _jax_transition(x: jnp.ndarray, fc1: jnp.ndarray, fc2: jnp.ndarray) -> jnp.ndarray:
    return _jax_linear(jax.nn.gelu(_jax_linear(x, fc1)), fc2)


def _torch_transition(x: Any, fc1: Any, fc2: Any) -> Any:
    import torch

    return _torch_linear(torch.nn.functional.gelu(_torch_linear(x, fc1)), fc2)


def _jax_layer_norm(
    x: jnp.ndarray, scale: jnp.ndarray, bias: jnp.ndarray, eps: float = 1e-5
) -> jnp.ndarray:
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)
    return (x - mean) * jax.lax.rsqrt(var + eps) * scale + bias


def _torch_layer_norm(x: Any, scale: Any, bias: Any, eps: float = 1e-5) -> Any:
    import torch

    mean = torch.mean(x, dim=-1, keepdim=True)
    var = torch.mean((x - mean) ** 2, dim=-1, keepdim=True)
    return (x - mean) * torch.rsqrt(var + eps) * scale + bias


def _jax_linear(x: jnp.ndarray, kernel: jnp.ndarray) -> jnp.ndarray:
    return jnp.matmul(x, kernel)


def _torch_linear(x: Any, kernel: Any) -> Any:
    return x @ kernel


def _jax_sigma_schedule(steps: int) -> jnp.ndarray:
    return jnp.linspace(1.0, 0.01, steps, dtype=jnp.float32)


def _torch_sigma_schedule(steps: int, device: Any) -> Any:
    import torch

    return torch.linspace(1.0, 0.01, steps, dtype=torch.float32, device=device)


def _tree_map(fn: Any, tree: Any) -> Any:
    if isinstance(tree, dict):
        return {key: _tree_map(fn, value) for key, value in tree.items()}
    if isinstance(tree, list):
        return [_tree_map(fn, value) for value in tree]
    return fn(tree)
