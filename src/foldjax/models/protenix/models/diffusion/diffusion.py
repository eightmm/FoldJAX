"""Diffusion conditioning blocks for the Protenix JAX port."""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import jax
import jax.numpy as jnp

from foldjax.models._random import masked_prefix_draw, supports_masked_prefix_draw
from foldjax.models.protenix.models.diffusion.atom import (
    AtomAttentionDecoderParams,
    AtomAttentionEncoderParams,
    atom_attention_decoder,
    atom_attention_encoder,
)
from foldjax.models.protenix.models.diffusion.transformer import (
    DiffusionTransformerStackParams,
    diffusion_transformer_stack,
)
from foldjax.models.protenix.models.primitives.primitives import (
    LayerNormParams,
    LinearParams,
    TransitionParams,
    layer_norm,
    linear,
    transition,
)
from foldjax.models.protenix.models.trunk_blocks.embedders import (
    FourierParams,
    RelativePositionParams,
    fourier_embedding,
    relative_position_encoding,
)


class DiffusionConditioningParams(NamedTuple):
    """Parameters for Protenix ``DiffusionConditioning``."""

    relpe: RelativePositionParams
    layernorm_z: LayerNormParams
    linear_z: LinearParams
    transition_z1: TransitionParams
    transition_z2: TransitionParams
    layernorm_s: LayerNormParams
    linear_s: LinearParams
    fourier: FourierParams
    layernorm_n: LayerNormParams
    linear_n: LinearParams
    transition_s1: TransitionParams
    transition_s2: TransitionParams


class DiffusionModuleParams(NamedTuple):
    """Parameters for infer-only Protenix ``DiffusionModule``."""

    conditioning: DiffusionConditioningParams
    atom_encoder: AtomAttentionEncoderParams
    layernorm_s: LayerNormParams
    linear_s: LinearParams
    diffusion_transformer: DiffusionTransformerStackParams
    layernorm_a: LayerNormParams
    atom_decoder: AtomAttentionDecoderParams


def inference_noise_schedule(
    *,
    num_steps: int = 200,
    s_max: float = 160.0,
    s_min: float = 4.0e-4,
    rho: float = 7.0,
    sigma_data: float = 16.0,
    dtype: jnp.dtype = jnp.float32,
) -> jnp.ndarray:
    """Return the Protenix inference noise schedule."""

    step_indices = jnp.arange(num_steps + 1, dtype=dtype)
    schedule = (
        sigma_data
        * (
            s_max ** (1.0 / rho)
            + step_indices
            / jnp.asarray(num_steps, dtype=dtype)
            * (s_min ** (1.0 / rho) - s_max ** (1.0 / rho))
        )
        ** rho
    )
    return schedule.at[-1].set(0.0)


def centre_random_augmentation(
    x: jnp.ndarray,
    atom_mask: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Center coordinates over the atom axis, matching inference centering."""

    if atom_mask is None:
        return x - jnp.mean(x, axis=-2, keepdims=True)
    mask = jnp.asarray(atom_mask, dtype=x.dtype)
    numerator = jnp.sum(x * mask[..., None], axis=-2, keepdims=True)
    denominator = jnp.maximum(jnp.sum(mask), 1.0)
    return (x - numerator / denominator) * mask[..., None]


def _prefix_atom_normal(
    key: jax.Array,
    atom_mask: jnp.ndarray,
    *,
    num_samples: int,
    dtype: jnp.dtype,
) -> jnp.ndarray:
    """Draw one padded coordinate field from the compact sample/atom prefix."""

    prefix_mask = jnp.broadcast_to(
        jnp.asarray(atom_mask, dtype=bool), (num_samples, atom_mask.shape[0])
    )
    return masked_prefix_draw(
        lambda normal_key, shape: jax.random.normal(
            normal_key, shape, dtype=dtype
        ),
        key,
        prefix_mask,
        trailing_shape=(3,),
    )


def sample_diffusion(
    denoise_fn,
    noise_schedule: jnp.ndarray,
    *,
    num_samples: int,
    n_atom: int,
    key: jax.Array | None,
    init_noise: jnp.ndarray | None = None,
    step_noises: jnp.ndarray | Sequence[jnp.ndarray] | None = None,
    gamma0: float = 0.8,
    gamma_min: float = 1.0,
    noise_scale_lambda: float = 1.003,
    step_scale_eta: float = 1.5,
    diffusion_chunk_size: int | None = None,
    dtype: jnp.dtype = jnp.float32,
    centre_each_step: bool = True,
    use_scan: bool = False,
    guidance_config=None,
    guidance_features=None,
    atom_mask: jnp.ndarray | None = None,
    preserve_prefix_rng: bool = False,
) -> jnp.ndarray:
    """Run Protenix Algorithm 18 diffusion sampling with a JAX denoiser."""

    if diffusion_chunk_size is None or diffusion_chunk_size <= 0:
        return _sample_diffusion_chunk(
            denoise_fn,
            noise_schedule,
            num_samples=num_samples,
            n_atom=n_atom,
            key=key,
            init_noise=init_noise,
            step_noises=step_noises,
            gamma0=gamma0,
            gamma_min=gamma_min,
            noise_scale_lambda=noise_scale_lambda,
            step_scale_eta=step_scale_eta,
            dtype=dtype,
            centre_each_step=centre_each_step,
            use_scan=use_scan,
            guidance_config=guidance_config,
            guidance_features=guidance_features,
            atom_mask=atom_mask,
            preserve_prefix_rng=preserve_prefix_rng,
        )

    outputs = []
    keys = None
    if key is not None:
        n_chunks = (num_samples + diffusion_chunk_size - 1) // diffusion_chunk_size
        keys = jax.random.split(key, n_chunks)
    for chunk_index, start in enumerate(range(0, num_samples, diffusion_chunk_size)):
        chunk_n = min(diffusion_chunk_size, num_samples - start)
        init_chunk = None
        if init_noise is not None:
            init_chunk = _slice_sample_axis(init_noise, start, chunk_n)
        step_chunks = None
        if step_noises is not None:
            if hasattr(step_noises, "shape"):
                step_chunks = _slice_sample_axis(step_noises, start, chunk_n)
            else:
                step_chunks = tuple(
                    _slice_sample_axis(noise, start, chunk_n)
                    for noise in step_noises
                )
        outputs.append(
            _sample_diffusion_chunk(
                denoise_fn,
                noise_schedule,
                num_samples=chunk_n,
                n_atom=n_atom,
                key=None if keys is None else keys[chunk_index],
                init_noise=init_chunk,
                step_noises=step_chunks,
                gamma0=gamma0,
                gamma_min=gamma_min,
                noise_scale_lambda=noise_scale_lambda,
                step_scale_eta=step_scale_eta,
                dtype=dtype,
                centre_each_step=centre_each_step,
                use_scan=use_scan,
                guidance_config=guidance_config,
                guidance_features=guidance_features,
                atom_mask=atom_mask,
                preserve_prefix_rng=preserve_prefix_rng,
            )
        )
    return jnp.concatenate(outputs, axis=-3)


def sample_diffusion_with_module(
    input_feature_dict: dict[str, jnp.ndarray | dict[str, jnp.ndarray]],
    s_inputs: jnp.ndarray,
    s_trunk: jnp.ndarray,
    z_trunk: jnp.ndarray,
    params: DiffusionModuleParams,
    noise_schedule: jnp.ndarray,
    *,
    num_samples: int,
    key: jax.Array | None,
    init_noise: jnp.ndarray | None = None,
    step_noises: jnp.ndarray | Sequence[jnp.ndarray] | None = None,
    pair_z: jnp.ndarray | None = None,
    p_lm: jnp.ndarray | None = None,
    c_l: jnp.ndarray | None = None,
    extra_attn_bias: jnp.ndarray | None = None,
    atom_encoder_heads: int = 4,
    token_heads: int = 16,
    atom_decoder_heads: int = 4,
    n_queries: int = 32,
    n_keys: int = 128,
    sigma_data: float = 16.0,
    use_conditioning: bool = True,
    use_scan: bool = False,
    use_sampler_scan: bool = False,
    use_denoiser_jit: bool = False,
    use_efficient_fusion: bool = False,
    attention_backend: str = "xla",
    token_q_chunk_size: int | None = None,
    diffusion_chunk_size: int | None = None,
    gamma0: float = 0.8,
    gamma_min: float = 1.0,
    noise_scale_lambda: float = 1.003,
    step_scale_eta: float = 1.5,
    dtype: jnp.dtype = jnp.float32,
    centre_each_step: bool = True,
    guidance_config=None,
    guidance_features=None,
    preserve_prefix_rng: bool = False,
) -> jnp.ndarray:
    """Sample coordinates using ``DiffusionModuleParams`` and static features."""

    atom_to_token_idx = input_feature_dict["atom_to_token_idx"]
    atom_padding_mask = input_feature_dict.get("atom_padding_mask")
    token_padding_mask = input_feature_dict.get("token_padding_mask")
    n_atom = int(atom_to_token_idx.shape[-1])
    n_token = int(s_inputs.shape[-2])
    transformer_z = None
    if use_efficient_fusion and pair_z is not None:
        transformer_z = layer_norm(
            jnp.expand_dims(pair_z, axis=-4).astype(jnp.float32),
            LayerNormParams(),
        )

    def denoise_fn(x_noisy: jnp.ndarray, t_hat: jnp.ndarray) -> jnp.ndarray:
        return diffusion_module_forward(
            atom_to_token_idx,
            input_feature_dict["ref_pos"],
            input_feature_dict["ref_charge"],
            input_feature_dict["ref_mask"],
            input_feature_dict["ref_atom_name_chars"],
            input_feature_dict["ref_element"],
            input_feature_dict["d_lm"],
            input_feature_dict["v_lm"],
            input_feature_dict["pad_info"],
            x_noisy,
            t_hat,
            input_feature_dict["relp"],
            s_inputs,
            s_trunk,
            z_trunk,
            params,
            pair_z=pair_z,
            p_lm=p_lm,
            c_l=c_l,
            extra_attn_bias=extra_attn_bias,
            transformer_z=transformer_z,
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
            token_mask=token_padding_mask,
            atom_mask=atom_padding_mask,
        )

    denoiser = jax.jit(denoise_fn) if use_denoiser_jit else denoise_fn
    return sample_diffusion(
        denoiser,
        noise_schedule,
        num_samples=num_samples,
        n_atom=n_atom,
        key=key,
        init_noise=init_noise,
        step_noises=step_noises,
        gamma0=gamma0,
        gamma_min=gamma_min,
        noise_scale_lambda=noise_scale_lambda,
        step_scale_eta=step_scale_eta,
        diffusion_chunk_size=diffusion_chunk_size,
        dtype=dtype,
        centre_each_step=centre_each_step,
        use_scan=use_sampler_scan,
        guidance_config=guidance_config,
        guidance_features=guidance_features,
        atom_mask=atom_padding_mask,
        preserve_prefix_rng=preserve_prefix_rng,
    )


def diffusion_conditioning_prepare_cache(
    relp_feature: jnp.ndarray,
    z_trunk: jnp.ndarray,
    params: DiffusionConditioningParams,
) -> jnp.ndarray:
    """Build diffusion pair conditioning cache."""

    pair_z = jnp.concatenate(
        [z_trunk, relative_position_encoding(relp_feature, params.relpe)],
        axis=-1,
    )
    pair_z = linear(layer_norm(pair_z, params.layernorm_z), params.linear_z)
    pair_z = pair_z + transition(pair_z, params.transition_z1)
    return pair_z + transition(pair_z, params.transition_z2)


def diffusion_conditioning(
    t_hat_noise_level: jnp.ndarray,
    relp_feature: jnp.ndarray,
    s_inputs: jnp.ndarray,
    s_trunk: jnp.ndarray,
    z_trunk: jnp.ndarray,
    params: DiffusionConditioningParams,
    *,
    pair_z: jnp.ndarray | None = None,
    sigma_data: float = 16.0,
    use_conditioning: bool = True,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Apply Protenix ``DiffusionConditioning`` in inference mode."""

    if pair_z is None:
        if not use_conditioning:
            s_trunk = jnp.zeros_like(s_trunk)
            z_trunk = jnp.zeros_like(z_trunk)
        pair_z = diffusion_conditioning_prepare_cache(relp_feature, z_trunk, params)

    single_s = jnp.concatenate([s_trunk, s_inputs], axis=-1)
    single_s = linear(layer_norm(single_s, params.layernorm_s), params.linear_s)
    noise = jnp.log(t_hat_noise_level / sigma_data) / 4.0
    noise = fourier_embedding(noise, params.fourier).astype(single_s.dtype)
    noise = linear(layer_norm(noise, params.layernorm_n), params.linear_n)
    single_s = single_s[..., None, :, :] + noise[..., :, None, :]
    single_s = single_s + transition(single_s, params.transition_s1)
    single_s = single_s + transition(single_s, params.transition_s2)
    return single_s, pair_z


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
    conditioned_single_s: jnp.ndarray | None = None,
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
    """Run the raw Protenix denoising network ``F`` for one noise level."""

    if conditioned_single_s is None:
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
    else:
        if pair_z is None:
            raise ValueError("preconditioned single features require a pair cache")
        single_s = conditioned_single_s
    s_trunk_sample = jnp.expand_dims(s_trunk, axis=-3)
    z_pair_sample = jnp.expand_dims(pair_z, axis=-4)
    if transformer_z is None:
        transformer_z = z_pair_sample.astype(jnp.float32)
        if use_efficient_fusion:
            transformer_z = layer_norm(transformer_z, LayerNormParams())
    a_token, q_skip, c_skip, p_skip = atom_attention_encoder(
        atom_to_token_idx,
        ref_pos,
        ref_charge,
        ref_mask,
        ref_atom_name_chars,
        ref_element,
        d_lm,
        v_lm,
        pad_info,
        params.atom_encoder,
        r_l=r_noisy,
        s=s_trunk_sample,
        z=z_pair_sample,
        p_lm=p_lm,
        c_l=c_l,
        n_token=n_token,
        n_heads=atom_encoder_heads,
        n_queries=n_queries,
        n_keys=n_keys,
        use_scan=use_scan,
        attention_backend=attention_backend,
        atom_mask=atom_mask,
    )
    a_token = a_token.astype(jnp.float32)
    a_token = a_token + linear(
        layer_norm(single_s, params.layernorm_s),
        params.linear_s,
    )
    a_token = diffusion_transformer_stack(
        a_token.astype(jnp.float32),
        single_s.astype(jnp.float32),
        transformer_z,
        params.diffusion_transformer,
        num_heads=token_heads,
        use_scan=use_scan,
        global_q_chunk_size=token_q_chunk_size,
        attention_backend=attention_backend,
        z_is_normalized=use_efficient_fusion,
        extra_attn_bias=extra_attn_bias,
        sequence_mask=token_mask,
    )
    a_token = layer_norm(a_token, params.layernorm_a)
    return atom_attention_decoder(
        atom_to_token_idx,
        a_token,
        q_skip,
        c_skip,
        p_skip,
        params.atom_decoder,
        n_heads=atom_decoder_heads,
        n_queries=n_queries,
        n_keys=n_keys,
        use_scan=use_scan,
        attention_backend=attention_backend,
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
    """Run one Protenix EDM denoising step."""

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


# Algorithm 18 raises the noise level from `c_tau_last` to `t_hat = c_tau_last *
# (gamma + 1)` and adds the difference `sqrt(t_hat^2 - c_tau_last^2)`.
#
# Written that way the term is a difference of two nearly equal squares, and it
# is *exactly* zero whenever gamma is zero — which is every step once the noise
# level drops below `gamma_min`. Under `lax.scan` XLA contracts the expression,
# so the cancellation leaves rounding noise instead of zero: measured at
# num_steps=20 on CPU it produced -4.3e-10 at one step, and `sqrt` of that is NaN,
# which then poisons every coordinate and every confidence score downstream.
# Neighbouring steps landed on the positive side and injected ~1e-4 of noise
# that Algorithm 18 does not call for. GPU fuses differently and happened not to
# cross zero, which is why this looked like a CPU-only fault.
#
# `t_hat^2 - c^2 == c^2 * ((gamma + 1)^2 - 1) == c^2 * gamma * (gamma + 2)`, so
# taking the square root of that form is algebraically identical, exactly zero
# at gamma == 0, never negative, and free of the cancellation.
def _sample_diffusion_chunk(
    denoise_fn,
    noise_schedule: jnp.ndarray,
    *,
    num_samples: int,
    n_atom: int,
    key: jax.Array | None,
    init_noise: jnp.ndarray | None,
    step_noises: jnp.ndarray | Sequence[jnp.ndarray] | None,
    gamma0: float,
    gamma_min: float,
    noise_scale_lambda: float,
    step_scale_eta: float,
    dtype: jnp.dtype,
    centre_each_step: bool,
    use_scan: bool,
    guidance_config,
    guidance_features,
    atom_mask: jnp.ndarray | None = None,
    preserve_prefix_rng: bool = False,
) -> jnp.ndarray:
    guidance_engine = None
    if guidance_config is not None:
        from foldjax.models.protenix.tfg import TFGConfig, TFGEngine, parse_tfg_config

        config = (
            guidance_config
            if isinstance(guidance_config, TFGConfig)
            else parse_tfg_config(guidance_config)
        )
        if config.enable:
            if guidance_features is None:
                raise ValueError("guidance_features are required when TFG is enabled")
            if use_scan:
                raise ValueError("TFG guidance currently requires use_scan=False")
            guidance_engine = TFGEngine(config)
    n_steps = int(noise_schedule.shape[0]) - 1
    if atom_mask is not None:
        atom_mask = jnp.asarray(atom_mask, dtype=dtype)
        if atom_mask.shape != (n_atom,):
            raise ValueError("atom_mask must have shape [N_atom]")
    if preserve_prefix_rng:
        if not supports_masked_prefix_draw():
            raise ValueError(
                "prefix-preserving padding noise requires "
                "jax_default_prng_impl='threefry2x32' and "
                "jax_threefry_partitionable=True"
            )
        if atom_mask is None:
            raise ValueError("prefix-preserving padding noise requires atom_mask")
        if init_noise is not None or step_noises is not None:
            raise ValueError(
                "prefix-preserving padding noise and explicit noise tapes are "
                "mutually exclusive"
            )

        def draw_normal(draw_key):
            return _prefix_atom_normal(
                draw_key,
                atom_mask,
                num_samples=num_samples,
                dtype=dtype,
            )

    else:

        def draw_normal(draw_key):
            return jax.random.normal(
                draw_key, (num_samples, n_atom, 3), dtype=dtype
            )

    if init_noise is None:
        if key is None:
            raise ValueError("key is required when init_noise is not provided")
        key, init_key = jax.random.split(key)
        init_noise = draw_normal(init_key)
    if atom_mask is not None:
        init_noise = init_noise * atom_mask[None, :, None]
    x_l = noise_schedule[0].astype(dtype) * init_noise.astype(dtype)

    step_noise_keys = preserve_prefix_rng and step_noises is None
    packed_step_noises = step_noises is not None and hasattr(step_noises, "shape")
    if step_noises is None:
        if key is None:
            raise ValueError("key is required when step_noises is not provided")
        step_keys = jax.random.split(key, n_steps)
        step_noises = (
            step_keys
            if step_noise_keys
            else tuple(
                jax.random.normal(step_key, x_l.shape, dtype=dtype)
                for step_key in step_keys
            )
        )
    if packed_step_noises:
        expected_step_shape = (n_steps, *x_l.shape)
        if tuple(step_noises.shape) != expected_step_shape:
            raise ValueError(
                f"packed step_noises expected shape {expected_step_shape}, "
                f"got {step_noises.shape}"
            )
    elif len(step_noises) != n_steps:
        raise ValueError("step_noises length must equal len(noise_schedule) - 1")
    guidance_keys = None
    if guidance_engine is not None and config.eps_std != 0.0:
        if key is None:
            raise ValueError("key is required when TFG mc.std > 0")
        guidance_key = jax.random.fold_in(key, 0x544647)
        guidance_keys = jax.random.split(guidance_key, n_steps)

    if use_scan:
        stacked_noises = (
            step_noises
            if packed_step_noises or step_noise_keys
            else jnp.stack(step_noises, axis=0)
        )

        def body(x_carry, xs):
            c_tau_last, c_tau, step_noise = xs
            if step_noise_keys:
                step_noise = draw_normal(step_noise)
            if centre_each_step:
                x_carry = centre_random_augmentation(x_carry, atom_mask)
            gamma = jnp.where(c_tau > gamma_min, gamma0, 0.0).astype(dtype)
            t_hat_scalar = c_tau_last * (gamma + 1.0)
            delta_noise_level = c_tau_last * jnp.sqrt(gamma * (gamma + 2.0))
            x_noisy = x_carry + noise_scale_lambda * delta_noise_level * step_noise
            if atom_mask is not None:
                x_noisy = x_noisy * atom_mask[None, :, None]
            t_hat = jnp.full(x_noisy.shape[:-2], t_hat_scalar, dtype=dtype)
            x_denoised = denoise_fn(x_noisy, t_hat)
            delta = (x_noisy - x_denoised) / t_hat[..., None, None]
            dt = c_tau - t_hat
            x_next = x_noisy + step_scale_eta * dt[..., None, None] * delta
            if atom_mask is not None:
                x_next = x_next * atom_mask[None, :, None]
            return x_next, None

        xs = (noise_schedule[:-1], noise_schedule[1:], stacked_noises)
        x_l, _ = jax.lax.scan(body, x_l, xs)
        return x_l

    for step_i in range(n_steps):
        c_tau_last = noise_schedule[step_i].astype(dtype)
        c_tau = noise_schedule[step_i + 1].astype(dtype)
        if centre_each_step:
            x_l = centre_random_augmentation(x_l, atom_mask)
        gamma = jnp.where(c_tau > gamma_min, gamma0, 0.0).astype(dtype)
        t_hat_scalar = c_tau_last * (gamma + 1.0)
        delta_noise_level = c_tau_last * jnp.sqrt(gamma * (gamma + 2.0))
        step_noise = step_noises[step_i]
        if step_noise_keys:
            step_noise = draw_normal(step_noise)
        x_noisy = x_l + noise_scale_lambda * delta_noise_level * step_noise
        if atom_mask is not None:
            x_noisy = x_noisy * atom_mask[None, :, None]
        t_hat = jnp.full(x_noisy.shape[:-2], t_hat_scalar, dtype=dtype)
        if guidance_engine is None:
            x_denoised = denoise_fn(x_noisy, t_hat)
            delta = (x_noisy - x_denoised) / t_hat[..., None, None]
            dt = c_tau - t_hat
            x_l = x_noisy + step_scale_eta * dt[..., None, None] * delta
        else:
            next_level = jnp.full(x_noisy.shape[:-2], c_tau, dtype=dtype)
            x_l = guidance_engine.step(
                denoise_fn,
                x=x_noisy,
                t_hat=t_hat,
                c_tau=next_level,
                step_scale_eta=step_scale_eta,
                step_i=step_i,
                num_diffusion_steps=n_steps,
                input_feature_dict=guidance_features,
                key=None if guidance_keys is None else guidance_keys[step_i],
            )
        if atom_mask is not None:
            x_l = x_l * atom_mask[None, :, None]
    return x_l


def _slice_sample_axis(x: jnp.ndarray, start: int, size: int) -> jnp.ndarray:
    if x.ndim < 3:
        raise ValueError("sample noise must have at least sample/atom/coord axes")
    return x[..., start : start + size, :, :] if x.ndim > 3 else x[start : start + size]
