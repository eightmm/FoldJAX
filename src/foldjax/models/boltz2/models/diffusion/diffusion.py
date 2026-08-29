"""Pure JAX diffusion score model assembly for the Boltz-2 port."""

from __future__ import annotations

from collections.abc import Mapping

import jax.numpy as jnp

from foldjax.models._cp import cp_mesh
from foldjax.models._cp_atom import single_to_keys_cp
from foldjax.models.boltz2.models.diffusion.atom import (
    atom_attention_decoder_forward,
    atom_attention_encoder_forward,
    diffusion_transformer_forward,
    get_indexing_matrix,
    single_to_keys,
)
from foldjax.models.boltz2.models.diffusion.diffusion_conditioning import (
    diffusion_conditioning_forward,
)
from foldjax.models.boltz2.models.primitives._common import layer_norm as _layer_norm
from foldjax.models.boltz2.models.primitives._common import linear as _linear
from foldjax.models.boltz2.models.trunk_blocks.conditioning import (
    single_conditioning_forward,
)

Params = Mapping[str, object]


def diffusion_score_model_forward(
    params: Params,
    s_inputs: jnp.ndarray,
    s_trunk: jnp.ndarray,
    r_noisy: jnp.ndarray,
    times: jnp.ndarray,
    feats: Mapping[str, jnp.ndarray],
    diffusion_conditioning: Mapping[str, object],
    multiplicity: int = 1,
    eps: float = 1e-5,
    use_scan: bool = True,
    attention_backend: str = "xla",
    token_attention_chunk: int | None = None,
    token_layers: int | None = None,
    atom_context_parallel: bool = False,
    atom_encoder_s_terms: tuple[jnp.ndarray, ...] | None = None,
    atom_decoder_s_terms: tuple[jnp.ndarray, ...] | None = None,
) -> jnp.ndarray:
    """Run Boltz DiffusionModule.forward using precomputed conditioning."""

    compute_dtype = r_noisy.dtype
    s, _ = single_conditioning_forward(
        params["single_conditioner"],
        times,
        jnp.repeat(s_trunk, multiplicity, axis=0),
        jnp.repeat(s_inputs, multiplicity, axis=0),
        eps=eps,
    )

    atom_to_token_idx = diffusion_conditioning.get("atom_to_token_idx")
    a, q_skip, c_skip = atom_attention_encoder_forward(
        params["atom_attention_encoder"],
        feats=feats,
        q=diffusion_conditioning["q"].astype(compute_dtype),
        c=diffusion_conditioning["c"].astype(compute_dtype),
        atom_enc_bias=diffusion_conditioning["atom_enc_bias"].astype(compute_dtype),
        to_keys=diffusion_conditioning["to_keys"],
        r=r_noisy,
        multiplicity=multiplicity,
        eps=eps,
        attention_backend=attention_backend,
        atom_to_token_idx=atom_to_token_idx,
        atom_context_parallel=atom_context_parallel,
        transformer_s_terms=atom_encoder_s_terms,
    )

    s_to_a = params["s_to_a_linear"]
    a = a + _linear(
        _layer_norm(
            s,
            s_to_a["norm"]["scale"],
            s_to_a["norm"]["bias"],
            eps,
        ),
        s_to_a["linear"]["kernel"],
    )

    mask = jnp.repeat(feats["token_pad_mask"], multiplicity, axis=0)
    token_bias = diffusion_conditioning.get("token_trans_bias")
    a = diffusion_transformer_forward(
        params["token_transformer"],
        a=a,
        s=s,
        bias=None if token_bias is None else token_bias.astype(compute_dtype),
        mask=mask.astype(jnp.float32),
        multiplicity=multiplicity,
        eps=eps,
        use_scan=use_scan,
        attention_backend=attention_backend,
        chunk_size=token_attention_chunk,
        layer_limit=token_layers,
        bias_params=diffusion_conditioning.get("token_trans_bias_params"),
        bias_input=diffusion_conditioning.get("token_trans_bias_input"),
        bias_normed_input=diffusion_conditioning.get("token_trans_bias_normed_input"),
    )
    a_norm = params["a_norm"]
    a = _layer_norm(a, a_norm["scale"], a_norm["bias"], eps)

    return atom_attention_decoder_forward(
        params["atom_attention_decoder"],
        a=a,
        q=q_skip,
        c=c_skip,
        atom_dec_bias=diffusion_conditioning["atom_dec_bias"].astype(compute_dtype),
        feats=feats,
        to_keys=diffusion_conditioning["to_keys"],
        multiplicity=multiplicity,
        eps=eps,
        attention_backend=attention_backend,
        atom_to_token_idx=atom_to_token_idx,
        atom_context_parallel=atom_context_parallel,
        transformer_s_terms=atom_decoder_s_terms,
    )


def conditioned_diffusion_score_forward(
    params: Params,
    s_inputs: jnp.ndarray,
    s_trunk: jnp.ndarray,
    z_trunk: jnp.ndarray,
    relative_position_encoding: jnp.ndarray,
    r_noisy: jnp.ndarray,
    times: jnp.ndarray,
    feats: Mapping[str, jnp.ndarray],
    token_layers: int | None = None,
    multiplicity: int = 1,
    atoms_per_window_queries: int = 32,
    atoms_per_window_keys: int = 128,
    eps: float = 1e-5,
    use_scan: bool = True,
    attention_backend: str = "xla",
    token_attention_chunk: int | None = None,
    lazy_token_trans_bias: bool = False,
    atom_context_parallel: bool = False,
) -> jnp.ndarray:
    """Run diffusion conditioning and score model as one JAX graph."""

    active = atom_context_parallel and cp_mesh() is not None
    conditioning = diffusion_conditioning_forward(
        params["diffusion_conditioning"],
        s_trunk=s_trunk,
        z_trunk=z_trunk,
        relative_position_encoding=relative_position_encoding,
        feats=feats,
        token_layers=token_layers,
        atoms_per_window_queries=atoms_per_window_queries,
        atoms_per_window_keys=atoms_per_window_keys,
        eps=eps,
        lazy_token_trans_bias=lazy_token_trans_bias,
        atom_context_parallel=active,
    )
    atoms = feats["ref_pos"].shape[1]
    num_windows = atoms // atoms_per_window_queries
    indexing = None
    if not active:
        indexing = get_indexing_matrix(
            k=num_windows,
            w=atoms_per_window_queries,
            h_keys=atoms_per_window_keys,
        )

    def to_keys(x: jnp.ndarray) -> jnp.ndarray:
        if active:
            return single_to_keys_cp(
                x,
                query_window=atoms_per_window_queries,
                key_window=atoms_per_window_keys,
            )
        return single_to_keys(
            x,
            indexing,
            w=atoms_per_window_queries,
            h_keys=atoms_per_window_keys,
        )

    conditioning["to_keys"] = to_keys
    return diffusion_score_model_forward(
        params["score_model"],
        s_inputs=s_inputs,
        s_trunk=s_trunk,
        r_noisy=r_noisy,
        times=times,
        feats=feats,
        diffusion_conditioning=conditioning,
        multiplicity=multiplicity,
        eps=eps,
        use_scan=use_scan,
        attention_backend=attention_backend,
        token_attention_chunk=token_attention_chunk,
        token_layers=token_layers,
        atom_context_parallel=active,
    )
