"""Checkpoint mapping for the Boltz-2 ConfidenceModule (JAX port).

Mirrors the real ``boltz2_conf.ckpt`` configuration:
``add_s_to_z_prod=True``, ``add_s_input_to_s=True``, ``add_z_input_to_z=True``,
``bond_type_feature=True``, ``no_update_s=False``, ``token_level_confidence=True``,
``use_separate_heads=True``, 8 pairformer layers.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from foldjax.models.boltz2.bridge.torch_mapping import (
    _linear_kernel,
    _to_jax_array,
    map_contact_conditioning_state_dict,
    map_pairformer_module_state_dict,
    map_relative_position_state_dict,
)

Params = dict[str, Any]


def map_confidence_module_state_dict(
    state_dict: Mapping[str, Any],
    prefix: str = "confidence_module",
    num_pairformer_layers: int | None = None,
) -> Params:
    """Map ConfidenceModule weights from a Boltz checkpoint to a JAX pytree."""

    def k(name: str) -> str:
        return f"{prefix}.{name}"

    def has(name: str) -> bool:
        return k(name) in state_dict

    def lin(name: str) -> dict[str, Any]:
        return {"kernel": _linear_kernel(state_dict[k(name)])}

    def lnorm(name: str) -> dict[str, Any]:
        return {
            "scale": _to_jax_array(state_dict[k(f"{name}.weight")]),
            "bias": _to_jax_array(state_dict[k(f"{name}.bias")]),
        }

    params: Params = {}
    params["boundaries"] = _to_jax_array(state_dict[k("boundaries")])
    params["dist_bin_pairwise_embed"] = _to_jax_array(
        state_dict[k("dist_bin_pairwise_embed.weight")]
    )

    params["s_to_z"] = {"kernel": _linear_kernel(state_dict[k("s_to_z.weight")])}
    params["s_to_z_transpose"] = {
        "kernel": _linear_kernel(state_dict[k("s_to_z_transpose.weight")])
    }

    # add_s_to_z_prod
    params["s_to_z_prod_in1"] = {
        "kernel": _linear_kernel(state_dict[k("s_to_z_prod_in1.weight")])
    }
    params["s_to_z_prod_in2"] = {
        "kernel": _linear_kernel(state_dict[k("s_to_z_prod_in2.weight")])
    }
    params["s_to_z_prod_out"] = {
        "kernel": _linear_kernel(state_dict[k("s_to_z_prod_out.weight")])
    }

    params["s_inputs_norm"] = lnorm("s_inputs_norm")
    if has("s_norm.weight"):
        params["s_norm"] = lnorm("s_norm")
    params["z_norm"] = lnorm("z_norm")

    # add_s_input_to_s
    params["s_input_to_s"] = {
        "kernel": _linear_kernel(state_dict[k("s_input_to_s.weight")])
    }

    # add_z_input_to_z
    params["rel_pos"] = map_relative_position_state_dict(
        state_dict, prefix=k("rel_pos")
    )
    params["token_bonds"] = {
        "kernel": _linear_kernel(state_dict[k("token_bonds.weight")])
    }
    if has("token_bonds_type.weight"):
        params["token_bonds_type"] = _to_jax_array(
            state_dict[k("token_bonds_type.weight")]
        )
    params["contact_conditioning"] = map_contact_conditioning_state_dict(
        state_dict, prefix=k("contact_conditioning")
    )

    # map_pairformer_module_state_dict assumes a dot-free prefix (it parses the
    # layer index via key.split(".")[2]). Strip the "confidence_module." prefix.
    strip = f"{prefix}."
    pf_sub = {
        key[len(strip) :]: value
        for key, value in state_dict.items()
        if key.startswith(f"{strip}pairformer_stack.")
    }
    params["pairformer_stack"] = map_pairformer_module_state_dict(
        pf_sub,
        prefix="pairformer_stack",
        num_layers=num_pairformer_layers,
    )

    # Confidence heads (use_separate_heads=True, token_level_confidence=True)
    hp = "confidence_heads"
    heads: Params = {}
    heads["to_pae_intra_logits"] = lin(f"{hp}.to_pae_intra_logits.weight")
    heads["to_pae_inter_logits"] = lin(f"{hp}.to_pae_inter_logits.weight")
    heads["to_pde_intra_logits"] = lin(f"{hp}.to_pde_intra_logits.weight")
    heads["to_pde_inter_logits"] = lin(f"{hp}.to_pde_inter_logits.weight")
    heads["to_plddt_logits"] = lin(f"{hp}.to_plddt_logits.weight")
    heads["to_resolved_logits"] = lin(f"{hp}.to_resolved_logits.weight")
    params["confidence_heads"] = heads

    return params
