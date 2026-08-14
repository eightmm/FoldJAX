"""OpenDDE denoiser composition over shared Protenix-JAX leaf modules."""

from __future__ import annotations

import jax.numpy as jnp

from foldjax.models.opendde.models.diffusion_conditioning import diffusion_conditioning
from foldjax.models.protenix.models.diffusion.diffusion import DiffusionModuleParams
from foldjax.models.protenix.models.diffusion.diffusion import (
    diffusion_module_f_forward as _protenix_diffusion_module_f_forward,
)


def diffusion_module_f_forward(
    atom_to_token_idx: jnp.ndarray,
    ref_pos: jnp.ndarray,
    ref_charge: jnp.ndarray,
    ref_mask: jnp.ndarray,
    ref_atom_name_chars: jnp.ndarray,
    ref_element: jnp.ndarray,
    d_lm: jnp.ndarray,
    v_lm: jnp.ndarray,
    pad_info: dict[str, jnp.ndarray],
    r_noisy: jnp.ndarray,
    t_hat_noise_level: jnp.ndarray,
    relp_feature: jnp.ndarray,
    s_inputs: jnp.ndarray,
    s_trunk: jnp.ndarray,
    z_trunk: jnp.ndarray,
    params: DiffusionModuleParams,
    *,
    pair_z: jnp.ndarray | None = None,
    p_lm: jnp.ndarray | None = None,
    c_l: jnp.ndarray | None = None,
    transformer_z: jnp.ndarray | None = None,
    extra_attn_bias: jnp.ndarray | None = None,
    n_token: int,
    atom_encoder_heads: int,
    token_heads: int,
    atom_decoder_heads: int,
    n_queries: int,
    n_keys: int,
    sigma_data: float = 16.0,
    use_conditioning: bool = True,
    use_scan: bool = False,
    use_efficient_fusion: bool = False,
    token_q_chunk_size: int | None = None,
    attention_backend: str = "xla",
    token_mask: jnp.ndarray | None = None,
    atom_mask: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Run OpenDDE's raw denoising network for one noise level."""

    single_s, pair_z = diffusion_conditioning(
        t_hat_noise_level,
        relp_feature,
        s_inputs,
        s_trunk,
        z_trunk,
        params.conditioning,
        pair_z=pair_z,
        sigma_data=sigma_data,
        use_conditioning=use_conditioning,
    )
    return _protenix_diffusion_module_f_forward(
        atom_to_token_idx,
        ref_pos,
        ref_charge,
        ref_mask,
        ref_atom_name_chars,
        ref_element,
        d_lm,
        v_lm,
        pad_info,
        r_noisy,
        t_hat_noise_level,
        relp_feature,
        s_inputs,
        s_trunk,
        z_trunk,
        params,
        pair_z=pair_z,
        p_lm=p_lm,
        c_l=c_l,
        transformer_z=transformer_z,
        extra_attn_bias=extra_attn_bias,
        conditioned_single_s=single_s,
        n_token=n_token,
        atom_encoder_heads=atom_encoder_heads,
        token_heads=token_heads,
        atom_decoder_heads=atom_decoder_heads,
        n_queries=n_queries,
        n_keys=n_keys,
        sigma_data=sigma_data,
        use_conditioning=use_conditioning,
        use_scan=use_scan,
        use_efficient_fusion=use_efficient_fusion,
        token_q_chunk_size=token_q_chunk_size,
        attention_backend=attention_backend,
        token_mask=token_mask,
        atom_mask=atom_mask,
    )


def diffusion_module_forward(
    atom_to_token_idx: jnp.ndarray,
    ref_pos: jnp.ndarray,
    ref_charge: jnp.ndarray,
    ref_mask: jnp.ndarray,
    ref_atom_name_chars: jnp.ndarray,
    ref_element: jnp.ndarray,
    d_lm: jnp.ndarray,
    v_lm: jnp.ndarray,
    pad_info: dict[str, jnp.ndarray],
    x_noisy: jnp.ndarray,
    t_hat_noise_level: jnp.ndarray,
    relp_feature: jnp.ndarray,
    s_inputs: jnp.ndarray,
    s_trunk: jnp.ndarray,
    z_trunk: jnp.ndarray,
    params: DiffusionModuleParams,
    *,
    pair_z: jnp.ndarray | None = None,
    p_lm: jnp.ndarray | None = None,
    c_l: jnp.ndarray | None = None,
    transformer_z: jnp.ndarray | None = None,
    extra_attn_bias: jnp.ndarray | None = None,
    n_token: int,
    atom_encoder_heads: int,
    token_heads: int,
    atom_decoder_heads: int,
    n_queries: int,
    n_keys: int,
    sigma_data: float = 16.0,
    use_conditioning: bool = True,
    use_scan: bool = False,
    use_efficient_fusion: bool = False,
    token_q_chunk_size: int | None = None,
    attention_backend: str = "xla",
    token_mask: jnp.ndarray | None = None,
    atom_mask: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Run one OpenDDE EDM denoising step."""

    scale = jnp.sqrt(sigma_data**2 + t_hat_noise_level**2)[..., None, None]
    r_noisy = x_noisy / scale
    r_update = diffusion_module_f_forward(
        atom_to_token_idx,
        ref_pos,
        ref_charge,
        ref_mask,
        ref_atom_name_chars,
        ref_element,
        d_lm,
        v_lm,
        pad_info,
        r_noisy,
        t_hat_noise_level,
        relp_feature,
        s_inputs,
        s_trunk,
        z_trunk,
        params,
        pair_z=pair_z,
        p_lm=p_lm,
        c_l=c_l,
        transformer_z=transformer_z,
        extra_attn_bias=extra_attn_bias,
        n_token=n_token,
        atom_encoder_heads=atom_encoder_heads,
        token_heads=token_heads,
        atom_decoder_heads=atom_decoder_heads,
        n_queries=n_queries,
        n_keys=n_keys,
        sigma_data=sigma_data,
        use_conditioning=use_conditioning,
        use_scan=use_scan,
        use_efficient_fusion=use_efficient_fusion,
        token_q_chunk_size=token_q_chunk_size,
        attention_backend=attention_backend,
        token_mask=token_mask,
        atom_mask=atom_mask,
    )
    s_ratio = (t_hat_noise_level / sigma_data)[..., None, None].astype(r_update.dtype)
    output = (
        x_noisy / (1.0 + s_ratio**2)
        + t_hat_noise_level[..., None, None] / jnp.sqrt(1.0 + s_ratio**2) * r_update
    ).astype(r_update.dtype)
    if atom_mask is not None:
        output = output * jnp.asarray(atom_mask, dtype=output.dtype)[..., None]
    return output
