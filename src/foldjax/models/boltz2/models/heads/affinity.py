"""Pure JAX Boltz-2 AffinityModule port."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp

from foldjax.models.boltz2.models.diffusion.atom import (
    _repeat_index,
    gather_rep_atoms_to_tokens,
    token_to_rep_atom_index_from_feats,
)
from foldjax.models.boltz2.models.primitives._common import layer_norm as _layer_norm
from foldjax.models.boltz2.models.primitives._common import linear as _linear
from foldjax.models.boltz2.models.trunk_blocks.conditioning import (
    pairwise_conditioning_forward,
)
from foldjax.models.boltz2.models.trunk_blocks.pairformer_noseq import (
    pairformer_no_seq_module_forward,
)

AffinityParams = Mapping[str, Any]


def affinity_module_forward(
    params: AffinityParams,
    s_inputs: jnp.ndarray,
    z: jnp.ndarray,
    x_pred: jnp.ndarray,
    feats: Mapping[str, jnp.ndarray],
    multiplicity: int = 1,
    eps: float = 1e-5,
) -> dict[str, jnp.ndarray]:
    """Run Boltz-2 AffinityModule (eval mode, no kernels)."""

    # z = z_linear(z_norm(z)); repeat_interleave over batch
    z = _linear(
        _layer_norm(z, params["z_norm"]["scale"], params["z_norm"]["bias"], eps),
        params["z_linear"]["kernel"],
    )
    z = jnp.repeat(z, multiplicity, axis=0)

    z = (
        z
        + _linear(s_inputs, params["s_to_z_prod_in1"]["kernel"])[:, :, None, :]
        + _linear(s_inputs, params["s_to_z_prod_in2"]["kernel"])[:, None, :, :]
    )

    rep_atom_index = _repeat_index(
        token_to_rep_atom_index_from_feats(feats), multiplicity
    )
    if x_pred.ndim == 4:
        b, mult, n, _ = x_pred.shape
        x_pred = x_pred.reshape(b * mult, n, -1)
    x_pred = x_pred.astype(jnp.float32)

    x_pred_repr = gather_rep_atoms_to_tokens(None, x_pred, index=rep_atom_index)
    x2 = jnp.sum(x_pred_repr * x_pred_repr, axis=-1, keepdims=True)
    d2 = (
        x2
        + jnp.swapaxes(x2, -1, -2)
        - 2.0 * (x_pred_repr @ jnp.swapaxes(x_pred_repr, -1, -2))
    )
    d = jnp.sqrt(jnp.maximum(d2, 0.0))

    boundaries = params["boundaries"]
    distogram_idx = jnp.sum(d[..., None] > boundaries, axis=-1)
    distogram = params["dist_bin_pairwise_embed"][distogram_idx]

    z = z + pairwise_conditioning_forward(
        params["pairwise_conditioner"],
        z_trunk=z,
        token_rel_pos_feats=distogram,
        eps=eps,
    )

    pad_token_mask = jnp.repeat(feats["token_pad_mask"], multiplicity, axis=0)
    rec_mask = jnp.repeat(
        (feats["mol_type"] == 0).astype(z.dtype), multiplicity, axis=0
    )
    rec_mask = rec_mask * pad_token_mask
    lig_mask = jnp.repeat(
        (feats["affinity_token_mask"] != 0).astype(z.dtype), multiplicity, axis=0
    )
    lig_mask = lig_mask * pad_token_mask
    cross_pair_mask = (
        lig_mask[:, :, None] * rec_mask[:, None, :]
        + rec_mask[:, :, None] * lig_mask[:, None, :]
        + lig_mask[:, :, None] * lig_mask[:, None, :]
    )

    z = pairformer_no_seq_module_forward(
        params["pairformer_stack"], z, cross_pair_mask, eps=eps
    )

    return _affinity_heads_forward(params["affinity_heads"], z, feats, multiplicity)


def _affinity_heads_forward(
    params: AffinityParams,
    z: jnp.ndarray,
    feats: Mapping[str, jnp.ndarray],
    multiplicity: int,
) -> dict[str, jnp.ndarray]:
    pad_token_mask = jnp.repeat(feats["token_pad_mask"], multiplicity, axis=0)[
        ..., None
    ]
    rec_mask = jnp.repeat(
        (feats["mol_type"] == 0).astype(z.dtype), multiplicity, axis=0
    )[..., None]
    rec_mask = rec_mask * pad_token_mask
    lig_mask = (
        jnp.repeat(
            (feats["affinity_token_mask"] != 0).astype(z.dtype), multiplicity, axis=0
        )[..., None]
        * pad_token_mask
    )
    n = lig_mask.shape[1]
    eye = jnp.eye(n, dtype=z.dtype)[None, :, :, None]
    cross_pair_mask = (
        lig_mask[:, :, None] * rec_mask[:, None, :]
        + rec_mask[:, :, None] * lig_mask[:, None, :]
        + lig_mask[:, :, None] * lig_mask[:, None, :]
    ) * (1 - eye)

    g = jnp.sum(z * cross_pair_mask, axis=(1, 2)) / (
        jnp.sum(cross_pair_mask, axis=(1, 2)) + 1e-7
    )

    g = _mlp(params["affinity_out_mlp"], g, final_relu=True)
    pred_value = _mlp(params["to_affinity_pred_value"], g, final_relu=False).reshape(
        -1, 1
    )
    pred_score = _mlp(params["to_affinity_pred_score"], g, final_relu=False).reshape(
        -1, 1
    )
    logits_binary = _linear(
        pred_score,
        params["to_affinity_logits_binary"]["kernel"],
        params["to_affinity_logits_binary"]["bias"],
    ).reshape(-1, 1)

    return {
        "affinity_pred_value": pred_value,
        "affinity_logits_binary": logits_binary,
    }


def _mlp(
    layers: list[Mapping[str, jnp.ndarray]],
    x: jnp.ndarray,
    final_relu: bool,
) -> jnp.ndarray:
    """Sequential of Linear+ReLU; ReLU after every Linear except optionally last."""

    n = len(layers)
    for i, layer in enumerate(layers):
        x = _linear(x, layer["kernel"], layer["bias"])
        if i < n - 1 or final_relu:
            x = jax.nn.relu(x)
    return x
