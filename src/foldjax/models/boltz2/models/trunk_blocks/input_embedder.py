"""Pure JAX InputEmbedder for the Boltz-2 port."""

from __future__ import annotations

from collections.abc import Mapping

import jax.numpy as jnp

from foldjax.models.boltz2.models.diffusion.atom import (
    atom_attention_encoder_forward,
    get_indexing_matrix,
    single_to_keys,
)
from foldjax.models.boltz2.models.diffusion.diffusion_conditioning import (
    atom_encoder_forward,
)
from foldjax.models.boltz2.models.primitives._common import layer_norm as _layer_norm
from foldjax.models.boltz2.models.primitives._common import linear as _linear

Params = Mapping[str, object]


def input_embedder_forward(
    params: Params,
    feats: Mapping[str, jnp.ndarray],
    eps: float = 1e-5,
    attention_backend: str = "xla",
    *,
    affinity: bool = False,
) -> jnp.ndarray:
    """Run Boltz InputEmbedder."""

    q, c, p = atom_encoder_forward(
        params["atom_encoder"],
        feats,
        structure_prediction=False,
        eps=eps,
    )
    atom_enc_proj = params["atom_enc_proj_z"]
    atom_enc_bias = _linear(
        _layer_norm(
            p,
            atom_enc_proj["norm"]["scale"],
            atom_enc_proj["norm"]["bias"],
            eps,
        ),
        atom_enc_proj["linear"]["kernel"],
    )
    a, _, _ = atom_attention_encoder_forward(
        params["atom_attention_encoder"],
        feats=feats,
        q=q,
        c=c,
        atom_enc_bias=atom_enc_bias,
        to_keys=_to_keys_from_feats(feats),
        structure_prediction=False,
        eps=eps,
        attention_backend=attention_backend,
    )

    s = a
    res_type_kernel = params["res_type_encoding"]["kernel"]
    s = s + _linear(feats["res_type"].astype(res_type_kernel.dtype), res_type_kernel)
    profile_kernel = params["msa_profile_encoding"]["kernel"]
    profile = feats["profile_affinity"] if affinity else feats["profile"]
    deletion_mean = (
        feats["deletion_mean_affinity"] if affinity else feats["deletion_mean"]
    )
    msa_profile = jnp.concatenate(
        (
            profile.astype(profile_kernel.dtype),
            deletion_mean[..., None].astype(profile_kernel.dtype),
        ),
        axis=-1,
    )
    s = s + _linear(msa_profile, profile_kernel)
    s = (
        s
        + params["method_conditioning_init"][feats["method_feature"].astype(jnp.int32)]
    )
    s = s + params["modified_conditioning_init"][feats["modified"].astype(jnp.int32)]
    cyclic_kernel = params["cyclic_conditioning_init"]["kernel"]
    cyclic = jnp.minimum(feats["cyclic_period"], 1.0)[..., None].astype(
        cyclic_kernel.dtype
    )
    s = s + _linear(cyclic, cyclic_kernel)
    s = s + params["mol_type_conditioning_init"][feats["mol_type"].astype(jnp.int32)]
    return s


def _to_keys_from_feats(feats: Mapping[str, jnp.ndarray]):
    atoms = feats["ref_pos"].shape[1]
    w = 32
    h_keys = 128
    indexing = get_indexing_matrix(k=atoms // w, w=w, h_keys=h_keys)
    return lambda x: single_to_keys(x, indexing, w=w, h_keys=h_keys)
