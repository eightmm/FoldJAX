"""PyTorch state_dict to JAX parameter mapping for the Boltz-2 AffinityModule."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax.numpy as jnp

from foldjax.models.boltz2.bridge.torch_mapping import (
    _linear_kernel,
    _module_list_indices,
    _to_jax_array,
    map_pairformer_no_seq_layer_state_dict,
    map_pairwise_conditioning_state_dict,
)

AffinityModuleParams = dict[str, Any]


def map_affinity_ensemble_state_dict(
    state_dict: Mapping[str, Any],
    *,
    mw_correction: bool = False,
) -> AffinityModuleParams:
    """Map every affinity member present in a Boltz-2 checkpoint."""

    prefixes = [
        prefix
        for prefix in ("affinity_module1", "affinity_module2")
        if any(key.startswith(f"{prefix}.") for key in state_dict)
    ]
    if not prefixes and any(key.startswith("affinity_module.") for key in state_dict):
        prefixes = ["affinity_module"]
    if not prefixes:
        raise KeyError("checkpoint contains no affinity module weights")
    return {
        "ensemble": len(prefixes) > 1,
        "mw_correction": mw_correction,
        "modules": [
            map_affinity_module_state_dict(state_dict, prefix) for prefix in prefixes
        ],
    }


def _map_linear_with_bias(
    state_dict: Mapping[str, Any], prefix: str
) -> dict[str, jnp.ndarray]:
    return {
        "kernel": _linear_kernel(state_dict[f"{prefix}.weight"]),
        "bias": _to_jax_array(state_dict[f"{prefix}.bias"]),
    }


def _map_mlp(
    state_dict: Mapping[str, Any], prefix: str, indices: tuple[int, ...]
) -> list[dict[str, jnp.ndarray]]:
    return [_map_linear_with_bias(state_dict, f"{prefix}.{i}") for i in indices]


def _map_pairformer_no_seq_module(
    state_dict: Mapping[str, Any], prefix: str
) -> dict[str, list[dict[str, Any]]]:
    indices = _module_list_indices(state_dict, f"{prefix}.layers", None)
    return {
        "layers": [
            map_pairformer_no_seq_layer_state_dict(state_dict, f"{prefix}.layers.{i}")
            for i in indices
        ]
    }


def map_affinity_module_state_dict(
    state_dict: Mapping[str, Any],
    prefix: str = "affinity_module1",
) -> AffinityModuleParams:
    """Map a Boltz-2 AffinityModule to a JAX pytree."""

    required_keys = (
        f"{prefix}.boundaries",
        f"{prefix}.dist_bin_pairwise_embed.weight",
        f"{prefix}.s_to_z_prod_in1.weight",
        f"{prefix}.s_to_z_prod_in2.weight",
        f"{prefix}.z_norm.weight",
        f"{prefix}.z_norm.bias",
        f"{prefix}.z_linear.weight",
    )
    missing = [k for k in required_keys if k not in state_dict]
    if missing:
        msg = (
            "Missing required AffinityModule state_dict keys "
            f"for prefix {prefix!r}: {', '.join(missing)}"
        )
        raise KeyError(msg)

    heads = f"{prefix}.affinity_heads"
    return {
        "boundaries": _to_jax_array(state_dict[f"{prefix}.boundaries"]),
        "dist_bin_pairwise_embed": _to_jax_array(
            state_dict[f"{prefix}.dist_bin_pairwise_embed.weight"]
        ),
        "s_to_z_prod_in1": {
            "kernel": _linear_kernel(state_dict[f"{prefix}.s_to_z_prod_in1.weight"])
        },
        "s_to_z_prod_in2": {
            "kernel": _linear_kernel(state_dict[f"{prefix}.s_to_z_prod_in2.weight"])
        },
        "z_norm": {
            "scale": _to_jax_array(state_dict[f"{prefix}.z_norm.weight"]),
            "bias": _to_jax_array(state_dict[f"{prefix}.z_norm.bias"]),
        },
        "z_linear": {"kernel": _linear_kernel(state_dict[f"{prefix}.z_linear.weight"])},
        "pairwise_conditioner": map_pairwise_conditioning_state_dict(
            state_dict, f"{prefix}.pairwise_conditioner"
        ),
        "pairformer_stack": _map_pairformer_no_seq_module(
            state_dict, f"{prefix}.pairformer_stack"
        ),
        "affinity_heads": {
            "affinity_out_mlp": _map_mlp(
                state_dict, f"{heads}.affinity_out_mlp", (0, 2)
            ),
            "to_affinity_pred_value": _map_mlp(
                state_dict, f"{heads}.to_affinity_pred_value", (0, 2, 4)
            ),
            "to_affinity_pred_score": _map_mlp(
                state_dict, f"{heads}.to_affinity_pred_score", (0, 2, 4)
            ),
            "to_affinity_logits_binary": _map_linear_with_bias(
                state_dict, f"{heads}.to_affinity_logits_binary"
            ),
        },
    }
