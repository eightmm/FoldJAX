"""Pure JAX Boltz-2 trunk graph."""

from __future__ import annotations

from collections.abc import Mapping
from math import pi

import jax
import jax.numpy as jnp

from foldjax.models.boltz2.models.diffusion.diffusion import (
    conditioned_diffusion_score_forward,
    diffusion_score_model_forward,
)
from foldjax.models.boltz2.models.diffusion.diffusion_conditioning import (
    diffusion_conditioning_forward,
)
from foldjax.models.boltz2.models.primitives._common import layer_norm as _layer_norm
from foldjax.models.boltz2.models.primitives._common import linear as _linear
from foldjax.models.boltz2.models.triangle.triangle_attention import (
    resolve_triangle_attention_chunk,
    resolve_triangle_attention_q_chunk,
)
from foldjax.models.boltz2.models.trunk_blocks.input_embedder import (
    input_embedder_forward,
)
from foldjax.models.boltz2.models.trunk_blocks.msa import msa_module_forward
from foldjax.models.boltz2.models.trunk_blocks.pairformer import (
    pairformer_module_forward,
)
from foldjax.models.boltz2.models.trunk_blocks.template import (
    has_template_feats,
    template_module_forward,
)

Params = Mapping[str, object]


def resolve_long_sequence_chunks(
    num_tokens: int,
    *,
    chunk_size: int,
    triangle_attention_chunk: int | None,
    triangle_attention_q_chunk: int | None,
    token_attention_chunk: int | None,
) -> dict[str, int | None]:
    """Return weight-compatible chunk policy for long inference shapes.

    Caller-provided values win unless the general ``chunk_size`` is larger than
    the long-shape cap. The policy is based on the Stage13/14 synthetic probes:
    2048 tokens fits with 128-class chunks, while 3072 needs smaller chunks.
    """

    effective_chunk_size = chunk_size
    effective_triangle_chunk = triangle_attention_chunk
    effective_triangle_q_chunk = triangle_attention_q_chunk
    effective_token_chunk = token_attention_chunk

    if num_tokens > 2048:
        if chunk_size > 64:
            effective_chunk_size = 64
        if effective_triangle_chunk is None:
            effective_triangle_chunk = 16
        if effective_triangle_q_chunk is None:
            effective_triangle_q_chunk = 256
        if effective_token_chunk is None:
            effective_token_chunk = 64
    else:
        effective_triangle_chunk = resolve_triangle_attention_chunk(
            num_tokens, effective_chunk_size, effective_triangle_chunk
        )
        effective_triangle_q_chunk = resolve_triangle_attention_q_chunk(
            num_tokens, effective_triangle_q_chunk
        )

    return {
        "chunk_size": effective_chunk_size,
        "triangle_attention_chunk": effective_triangle_chunk,
        "triangle_attention_q_chunk": effective_triangle_q_chunk,
        "token_attention_chunk": effective_token_chunk,
    }


def _shard_pair(
    z: jnp.ndarray, mesh: object, token_axis: str, shard_tokens: bool
) -> jnp.ndarray:
    """Constrain pair tensor ``z`` [B, N, N, C] over ``mesh``.

    ``mesh is None`` -> no-op (default path stays bit-identical). When a mesh is
    given and ``shard_tokens`` is True the first token (N) axis is partitioned
    over ``token_axis``; when False a replicated constraint is emitted (the
    buffer still lives on the mesh, distribution is layout-only, numerics stay
    bit-exact). See ``boltz2_trunk_forward`` for the bit-exactness trade-off.
    """

    if mesh is None:
        return z
    from jax.sharding import NamedSharding, PartitionSpec

    spec = (
        PartitionSpec(None, token_axis, None, None) if shard_tokens else PartitionSpec()
    )
    return jax.lax.with_sharding_constraint(z, NamedSharding(mesh, spec))


def _shard_single(
    s: jnp.ndarray, mesh: object, token_axis: str, shard_tokens: bool
) -> jnp.ndarray:
    """Constrain single tensor ``s`` [B, N, C] over ``mesh`` (see ``_shard_pair``)."""

    if mesh is None:
        return s
    from jax.sharding import NamedSharding, PartitionSpec

    spec = PartitionSpec(None, token_axis, None) if shard_tokens else PartitionSpec()
    return jax.lax.with_sharding_constraint(s, NamedSharding(mesh, spec))


def _cast_params(params: object, dtype: jnp.dtype) -> object:
    """Cast only floating-point leaves of a param pytree to ``dtype``.

    Integer / boolean tables (e.g. embedding index tables stored as floats are
    still floats and get cast; genuine int tables are left untouched). Weights
    on disk are unchanged; this is a runtime cast only.
    """

    def _cast(x: object) -> object:
        if hasattr(x, "dtype") and jnp.issubdtype(x.dtype, jnp.floating):
            return x.astype(dtype)
        return x

    return jax.tree.map(_cast, params)


def _cast_float_feats(
    feats: Mapping[str, jnp.ndarray], dtype: jnp.dtype
) -> dict[str, jnp.ndarray]:
    """Cast floating-point feature arrays to ``dtype``; keep ints/bools as-is."""

    out: dict[str, jnp.ndarray] = {}
    for k, v in feats.items():
        if hasattr(v, "dtype") and jnp.issubdtype(v.dtype, jnp.floating):
            out[k] = v.astype(dtype)
        else:
            out[k] = v
    return out


def boltz2_graph_score_forward(
    params: Params,
    feats: Mapping[str, jnp.ndarray],
    r_noisy: jnp.ndarray,
    times: jnp.ndarray,
    *,
    recycling_steps: int = 0,
    token_layers: int | None = None,
    multiplicity: int = 1,
    eps: float = 1e-5,
    use_scan: bool = True,
    trunk_use_scan: bool | None = None,
    score_use_scan: bool | None = None,
    chunk_size: int = 128,
    triangle_attention_chunk: int | None = None,
    triangle_attention_q_chunk: int | None = None,
    transition_hidden_chunk: int | None = None,
    token_attention_chunk: int | None = None,
    matmul_precision: str = "highest",
    attention_backend: str = "xla",
    triangle_backend: str = "cueq",
    glu_backend: str = "xla",
    lazy_token_trans_bias: bool = True,
) -> jnp.ndarray:
    """Run non-template Boltz-2 trunk, conditioning, and score in one JAX graph."""

    trunk_scan = use_scan if trunk_use_scan is None else trunk_use_scan
    score_scan = use_scan if score_use_scan is None else score_use_scan
    chunks = resolve_long_sequence_chunks(
        feats["token_pad_mask"].shape[1],
        chunk_size=chunk_size,
        triangle_attention_chunk=triangle_attention_chunk,
        triangle_attention_q_chunk=triangle_attention_q_chunk,
        token_attention_chunk=token_attention_chunk,
    )
    trunk = boltz2_trunk_forward(
        params["trunk"],
        feats,
        recycling_steps=recycling_steps,
        eps=eps,
        use_scan=trunk_scan,
        chunk_size=chunks["chunk_size"],
        triangle_attention_chunk=chunks["triangle_attention_chunk"],
        triangle_attention_q_chunk=chunks["triangle_attention_q_chunk"],
        transition_hidden_chunk=transition_hidden_chunk,
        matmul_precision=matmul_precision,
        attention_backend=attention_backend,
        triangle_backend=triangle_backend,
        glu_backend=glu_backend,
    )
    return conditioned_diffusion_score_forward(
        params["conditioned_diffusion"],
        s_inputs=trunk["s_inputs"],
        s_trunk=trunk["s"],
        z_trunk=trunk["z"],
        relative_position_encoding=trunk["relative_position_encoding"],
        r_noisy=r_noisy,
        times=times,
        feats=feats,
        multiplicity=multiplicity,
        eps=eps,
        use_scan=score_scan,
        attention_backend=attention_backend,
        token_attention_chunk=chunks["token_attention_chunk"],
        token_layers=token_layers,
        lazy_token_trans_bias=lazy_token_trans_bias,
    )


def boltz2_sample_forward(
    params: Params,
    feats: Mapping[str, jnp.ndarray],
    key: jnp.ndarray,
    *,
    recycling_steps: int = 0,
    num_sampling_steps: int = 5,
    token_layers: int | None = None,
    multiplicity: int = 1,
    sigma_min: float = 0.0001,  # match torch Boltz2DiffusionParams (main.py:137)
    sigma_max: float = 160.0,
    sigma_data: float = 16.0,
    rho: float = 7.0,
    gamma_0: float = 0.8,
    gamma_min: float = 1.0,
    noise_scale: float = 1.003,
    step_scale: float = 1.5,
    augmentation: bool = True,
    alignment_reverse_diff: bool = True,
    s_trans: float = 1.0,
    eps: float = 1e-5,
    steering_args: Mapping[str, object] | None = None,
    init_noise: jnp.ndarray | None = None,
    step_noises: jnp.ndarray | None = None,
    aug_transforms: tuple[jnp.ndarray, jnp.ndarray] | None = None,
    use_scan: bool = True,
    trunk_use_scan: bool | None = None,
    score_use_scan: bool | None = None,
    trunk: Mapping[str, jnp.ndarray] | None = None,
    chunk_size: int = 128,
    triangle_attention_chunk: int | None = None,
    triangle_attention_q_chunk: int | None = None,
    transition_hidden_chunk: int | None = None,
    token_attention_chunk: int | None = None,
    matmul_precision: str = "highest",
    attention_backend: str = "xla",
    triangle_backend: str = "cueq",
    glu_backend: str = "xla",
    compute_dtype: jnp.dtype = jnp.float32,
    mesh: object | None = None,
    token_axis: str = "tok",
    shard_tokens: bool = True,
    lazy_token_trans_bias: bool = True,
) -> dict[str, jnp.ndarray]:
    """Run a JAX inference sampler.

    Mirrors ``AtomDiffusion.sample`` in eval mode: per-step centered random
    augmentation and ``alignment_reverse_diff`` (weighted rigid align of the
    noisy coords onto the denoised coords) are on by default to match the
    PyTorch inference path.

    When ``steering_args`` is None (default) the no-steering path is run and is
    byte-identical to the previous behaviour. When provided with FK steering
    and/or physical/contact guidance enabled, the FK resampling + guidance
    update loop from ``diffusionv2.AtomDiffusion.sample`` is mirrored. The
    stochastic multinomial resampling only runs when steering is enabled.

    ``matmul_precision`` ("highest" default) controls the triangle-attention
    projection precision pin (the one matmul the global flag cannot reach). The
    default "highest" path is bit-identical to before. To run the full graph in
    TF32, the caller should ALSO set
    ``jax.config.update("jax_default_matmul_precision", "default")`` so the
    remaining unpinned matmuls (diffusion score, triangle mult, einsums) match.

    ``aug_transforms`` (default None) is a test-only injection hook for parity
    studies: a ``(R, tr)`` tuple with ``R`` shape ``(num_sampling_steps,
    multiplicity, 3, 3)`` and ``tr`` shape ``(num_sampling_steps, multiplicity,
    1, 3)``. When provided AND ``augmentation`` is True, the per-step random
    augmentation samples are replaced by these fixed transforms instead of being
    drawn from ``key``. Default None is fully inert (original behaviour).

    ``chunk_size`` (default 128) is threaded to the triangle / outer-product-mean
    ops; it is bit-exact (the reduction axis is never split) and only reorders
    independent blocks. A larger value (e.g. 256) issues fewer, larger matmuls.
    """

    steering_on = steering_args is not None and (
        steering_args["fk_steering"]
        or steering_args["physical_guidance_update"]
        or steering_args["contact_guidance_update"]
    )
    trunk_scan = use_scan if trunk_use_scan is None else trunk_use_scan
    score_scan = use_scan if score_use_scan is None else score_use_scan
    chunks = resolve_long_sequence_chunks(
        feats["token_pad_mask"].shape[1],
        chunk_size=chunk_size,
        triangle_attention_chunk=triangle_attention_chunk,
        triangle_attention_q_chunk=triangle_attention_q_chunk,
        token_attention_chunk=token_attention_chunk,
    )

    compute_dtype = jnp.dtype(compute_dtype)
    low_precision = compute_dtype != jnp.float32
    # Mirror Boltz's precision profile: the trunk/pairformer/MSA run in low
    # precision (bf16-mixed), but the diffusion structure module is an fp32
    # island -- Boltz wraps `structure_module.sample` in autocast(enabled=False)
    # with s/z/s_inputs cast to float. So we cast ONLY the trunk inputs to low
    # precision, keep the diffusion params + feats in fp32, and cast the trunk
    # outputs back to fp32 before conditioning/sampling. The default fp32 path
    # is untouched and bit-identical. (Uniform low precision through the
    # sampling loop diverges: bf16 coordinate updates accumulate over steps.)
    trunk_params = params["trunk"]
    trunk_feats = feats
    if low_precision:
        trunk_params = _cast_params(params["trunk"], compute_dtype)
        trunk_feats = _cast_float_feats(feats, compute_dtype)

    if trunk is None:
        trunk = boltz2_trunk_forward(
            trunk_params,
            trunk_feats,
            recycling_steps=recycling_steps,
            eps=eps,
            use_scan=trunk_scan,
            chunk_size=chunks["chunk_size"],
            triangle_attention_chunk=chunks["triangle_attention_chunk"],
            triangle_attention_q_chunk=chunks["triangle_attention_q_chunk"],
            transition_hidden_chunk=transition_hidden_chunk,
            matmul_precision=matmul_precision,
            attention_backend=attention_backend,
            triangle_backend=triangle_backend,
            glu_backend=glu_backend,
            mesh=mesh,
            token_axis=token_axis,
            shard_tokens=shard_tokens,
        )
    if low_precision:
        # Enter the fp32 diffusion island: cast trunk outputs back to fp32 so
        # conditioning + the sampling loop (which derive dtype from these and
        # from the fp32 diffusion params/feats) run fully in fp32, as Boltz does.
        trunk = {
            k: (v.astype(jnp.float32) if hasattr(v, "dtype") else v)
            for k, v in trunk.items()
        }
    diffusion_params = params["conditioned_diffusion"]
    diffusion_conditioning = diffusion_conditioning_forward(
        diffusion_params["diffusion_conditioning"],
        s_trunk=trunk["s"],
        z_trunk=trunk["z"],
        relative_position_encoding=trunk["relative_position_encoding"],
        feats=feats,
        token_layers=token_layers,
        eps=eps,
        lazy_token_trans_bias=lazy_token_trans_bias,
    )
    sigmas = _sample_schedule(
        num_sampling_steps,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        sigma_data=sigma_data,
        rho=rho,
    )
    # Steering state -------------------------------------------------------
    potentials = []
    num_particles = 1
    energy_traj = None
    scaled_guidance_update = None
    if steering_on:
        from foldjax.models.boltz2.models.heads.potentials import get_potentials

        potentials = get_potentials(steering_args, boltz2=True)
        if steering_args["fk_steering"]:
            num_particles = int(steering_args["num_particles"])
            multiplicity = multiplicity * num_particles
            energy_traj = jnp.empty((multiplicity, 0), dtype=jnp.float32)
        if (
            steering_args["physical_guidance_update"]
            or steering_args["contact_guidance_update"]
        ):
            scaled_guidance_update = jnp.zeros(
                (multiplicity, *feats["atom_pad_mask"].shape[1:], 3),
                dtype=jnp.float32,
            )

    atom_mask = jnp.repeat(feats["atom_pad_mask"], multiplicity, axis=0)
    shape = (*atom_mask.shape, 3)
    key, init_key = jax.random.split(key)
    if init_noise is not None:
        atom_coords = sigmas[0] * jnp.asarray(init_noise, dtype=jnp.float32)
    else:
        atom_coords = sigmas[0] * jax.random.normal(init_key, shape, dtype=jnp.float32)
    gammas = jnp.where(sigmas > gamma_min, gamma_0, 0.0)
    atom_coords_denoised = None

    # Scan the sampling loop unless steering is active (steering needs the
    # Python loop for its per-step guidance); fall back to eager rather than
    # erroring so use_scan can default on.
    if use_scan and steering_on:
        use_scan = False

    if use_scan:

        def _scan_body(carry, xs):
            atom_coords_c, atom_coords_denoised_c, has_denoised, key_c = carry
            sigma_tm, sigma_t, gamma, injected, aug_r, aug_tr = xs

            if augmentation:
                if aug_transforms is not None:
                    random_r = aug_r.astype(atom_coords_c.dtype)
                    random_tr = aug_tr.astype(atom_coords_c.dtype)
                else:
                    key_c, aug_key = jax.random.split(key_c)
                    random_r, random_tr = _compute_random_augmentation(
                        aug_key, atom_coords_c.shape[0], s_trans, atom_coords_c.dtype
                    )
                atom_coords_c = atom_coords_c - atom_coords_c.mean(
                    axis=-2, keepdims=True
                )
                atom_coords_c = atom_coords_c @ random_r + random_tr
                # Step 0 has no prior denoised coords (eager keeps it None and
                # skips the transform). Gate on ``has_denoised`` to replicate.
                denoised_aug = (
                    atom_coords_denoised_c
                    - atom_coords_denoised_c.mean(axis=-2, keepdims=True)
                ) @ random_r + random_tr
                atom_coords_denoised_c = jnp.where(
                    has_denoised, denoised_aug, atom_coords_denoised_c
                )

            t_hat = sigma_tm * (1.0 + gamma)
            noise_var = jnp.maximum(noise_scale**2 * (t_hat**2 - sigma_tm**2), 0.0)
            key_c, noise_key = jax.random.split(key_c)
            if step_noises is not None:
                eps_noise = injected.astype(atom_coords_c.dtype)
            else:
                eps_noise = jax.random.normal(
                    noise_key, shape, dtype=atom_coords_c.dtype
                )
            noise = jnp.sqrt(noise_var) * eps_noise
            atom_coords_noisy = atom_coords_c + noise
            atom_coords_denoised_c = _preconditioned_score_forward(
                diffusion_params["score_model"],
                s_inputs=trunk["s_inputs"],
                s_trunk=trunk["s"],
                r_noisy=atom_coords_noisy,
                sigma=t_hat,
                feats=feats,
                diffusion_conditioning=diffusion_conditioning,
                multiplicity=multiplicity,
                sigma_data=sigma_data,
                eps=eps,
                use_scan=score_scan,
                attention_backend=attention_backend,
                token_attention_chunk=chunks["token_attention_chunk"],
                token_layers=token_layers,
            )
            if alignment_reverse_diff:
                atom_coords_noisy = _weighted_rigid_align(
                    atom_coords_noisy.astype(jnp.float32),
                    atom_coords_denoised_c.astype(jnp.float32),
                    atom_mask.astype(jnp.float32),
                    atom_mask.astype(jnp.float32),
                ).astype(atom_coords_denoised_c.dtype)
            denoised_over_sigma = (atom_coords_noisy - atom_coords_denoised_c) / t_hat
            atom_coords_c = atom_coords_noisy + (
                step_scale * (sigma_t - t_hat) * denoised_over_sigma
            )
            new_carry = (
                atom_coords_c,
                atom_coords_denoised_c,
                jnp.bool_(True),
                key_c,
            )
            return new_carry, None

        if step_noises is not None:
            injected_xs = jnp.asarray(step_noises, dtype=jnp.float32)
        else:
            injected_xs = jnp.zeros((num_sampling_steps,), dtype=jnp.float32)
        if aug_transforms is not None:
            aug_r_xs = jnp.asarray(aug_transforms[0], dtype=jnp.float32)
            aug_tr_xs = jnp.asarray(aug_transforms[1], dtype=jnp.float32)
        else:
            aug_r_xs = jnp.zeros((num_sampling_steps, multiplicity, 3, 3))
            aug_tr_xs = jnp.zeros((num_sampling_steps, multiplicity, 1, 3))
        xs = (sigmas[:-1], sigmas[1:], gammas[1:], injected_xs, aug_r_xs, aug_tr_xs)
        denoised_init = jnp.zeros(shape, dtype=jnp.float32)
        init_carry = (atom_coords, denoised_init, jnp.bool_(False), key)
        (atom_coords, _, _, key), _ = jax.lax.scan(_scan_body, init_carry, xs)
        return {"sample_atom_coords": atom_coords}

    for step_idx in range(num_sampling_steps):
        sigma_tm = sigmas[step_idx]
        sigma_t = sigmas[step_idx + 1]
        gamma = gammas[step_idx + 1]

        if augmentation:
            if aug_transforms is not None:
                random_r = jnp.asarray(
                    aug_transforms[0][step_idx], dtype=atom_coords.dtype
                )
                random_tr = jnp.asarray(
                    aug_transforms[1][step_idx], dtype=atom_coords.dtype
                )
            else:
                key, aug_key = jax.random.split(key)
                random_r, random_tr = _compute_random_augmentation(
                    aug_key, atom_coords.shape[0], s_trans, atom_coords.dtype
                )
            atom_coords = atom_coords - atom_coords.mean(axis=-2, keepdims=True)
            atom_coords = atom_coords @ random_r + random_tr
            if atom_coords_denoised is not None:
                atom_coords_denoised = atom_coords_denoised - atom_coords_denoised.mean(
                    axis=-2, keepdims=True
                )
                atom_coords_denoised = atom_coords_denoised @ random_r + random_tr
            if scaled_guidance_update is not None:
                scaled_guidance_update = scaled_guidance_update @ random_r

        t_hat = sigma_tm * (1.0 + gamma)
        noise_var = jnp.maximum(noise_scale**2 * (t_hat**2 - sigma_tm**2), 0.0)
        key, noise_key = jax.random.split(key)
        if step_noises is not None:
            eps_noise = jnp.asarray(step_noises[step_idx], dtype=atom_coords.dtype)
        else:
            eps_noise = jax.random.normal(noise_key, shape, dtype=atom_coords.dtype)
        noise = jnp.sqrt(noise_var) * eps_noise
        atom_coords_noisy = atom_coords + noise
        atom_coords_denoised = _preconditioned_score_forward(
            diffusion_params["score_model"],
            s_inputs=trunk["s_inputs"],
            s_trunk=trunk["s"],
            r_noisy=atom_coords_noisy,
            sigma=t_hat,
            feats=feats,
            diffusion_conditioning=diffusion_conditioning,
            multiplicity=multiplicity,
            sigma_data=sigma_data,
            eps=eps,
            use_scan=score_scan,
            attention_backend=attention_backend,
            token_attention_chunk=chunks["token_attention_chunk"],
            token_layers=token_layers,
        )

        if steering_on:
            steering_t = 1.0 - (step_idx / num_sampling_steps)
            noise_positive = bool(noise_var > 0)
            do_resample = steering_args["fk_steering"] and (
                (
                    step_idx % steering_args["fk_resampling_interval"] == 0
                    and noise_positive
                )
                or step_idx == num_sampling_steps - 1
            )

            if do_resample:
                energy = jnp.zeros(multiplicity, dtype=jnp.float32)
                for potential in potentials:
                    parameters = potential.compute_parameters(steering_t)
                    if parameters["resampling_weight"] > 0:
                        component = potential.compute(
                            atom_coords_denoised, feats, parameters
                        )
                        energy = energy + parameters["resampling_weight"] * component
                energy_traj = jnp.concatenate((energy_traj, energy[:, None]), axis=1)
                if step_idx == 0:
                    log_g = -energy
                else:
                    log_g = energy_traj[:, -2] - energy_traj[:, -1]

                if (
                    steering_args["physical_guidance_update"]
                    or steering_args["contact_guidance_update"]
                ) and noise_positive:
                    ll_difference = (
                        noise**2 - (noise + scaled_guidance_update) ** 2
                    ).sum(axis=(-1, -2)) / (2 * noise_var)
                else:
                    ll_difference = jnp.zeros_like(energy)

                resample_weights = jax.nn.softmax(
                    (ll_difference + steering_args["fk_lambda"] * log_g).reshape(
                        -1, num_particles
                    ),
                    axis=1,
                )

            if (
                steering_args["physical_guidance_update"]
                or steering_args["contact_guidance_update"]
            ) and step_idx < num_sampling_steps - 1:
                guidance_update = jnp.zeros_like(atom_coords_denoised)
                for guidance_step in range(steering_args["num_gd_steps"]):
                    energy_gradient = jnp.zeros_like(atom_coords_denoised)
                    for potential in potentials:
                        parameters = potential.compute_parameters(steering_t)
                        if (
                            parameters["guidance_weight"] > 0
                            and guidance_step % parameters["guidance_interval"] == 0
                        ):
                            energy_gradient = energy_gradient + parameters[
                                "guidance_weight"
                            ] * potential.compute_gradient(
                                atom_coords_denoised + guidance_update,
                                feats,
                                parameters,
                            )
                    guidance_update = guidance_update - energy_gradient
                atom_coords_denoised = atom_coords_denoised + guidance_update
                scaled_guidance_update = (
                    guidance_update * -1 * step_scale * (sigma_t - t_hat) / t_hat
                )

            if do_resample:
                n_groups = resample_weights.shape[0]
                n_draw = num_particles if step_idx < num_sampling_steps - 1 else 1
                key, rkey = jax.random.split(key)
                draws = jax.random.categorical(
                    rkey,
                    jnp.log(resample_weights),
                    axis=1,
                    shape=(n_groups, n_draw),
                )
                resample_indices = (
                    draws + num_particles * jnp.arange(n_groups)[:, None]
                ).reshape(-1)
                atom_coords = atom_coords[resample_indices]
                atom_coords_noisy = atom_coords_noisy[resample_indices]
                atom_mask = atom_mask[resample_indices]
                atom_coords_denoised = atom_coords_denoised[resample_indices]
                energy_traj = energy_traj[resample_indices]
                if scaled_guidance_update is not None:
                    scaled_guidance_update = scaled_guidance_update[resample_indices]

        if alignment_reverse_diff:
            atom_coords_noisy = _weighted_rigid_align(
                atom_coords_noisy.astype(jnp.float32),
                atom_coords_denoised.astype(jnp.float32),
                atom_mask.astype(jnp.float32),
                atom_mask.astype(jnp.float32),
            ).astype(atom_coords_denoised.dtype)

        denoised_over_sigma = (atom_coords_noisy - atom_coords_denoised) / t_hat
        atom_coords = atom_coords_noisy + (
            step_scale * (sigma_t - t_hat) * denoised_over_sigma
        )

    return {"sample_atom_coords": atom_coords}


def _compute_random_augmentation(
    key: jnp.ndarray,
    multiplicity: int,
    s_trans: float,
    dtype: jnp.dtype,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Uniform random rotation matrices and translations (batched).

    Returns ``R`` of shape ``(multiplicity, 3, 3)`` to be applied as
    ``coords @ R`` and ``tr`` of shape ``(multiplicity, 1, 3)``. The rotation
    is sampled uniformly via QR of a Gaussian matrix with sign/det correction.
    """

    rot_key, trans_key = jax.random.split(key)
    a = jax.random.normal(rot_key, (multiplicity, 3, 3), dtype=jnp.float32)
    q, r = jnp.linalg.qr(a)
    # Make the decomposition unique: fix signs by the diagonal of R.
    sign = jnp.sign(jnp.diagonal(r, axis1=-2, axis2=-1))
    sign = jnp.where(sign == 0, 1.0, sign)
    q = q * sign[:, None, :]
    # Ensure a proper rotation (det = +1) by flipping the last column.
    det = jnp.linalg.det(q)
    q = q.at[:, :, -1].multiply(det[:, None])
    random_tr = (
        jax.random.normal(trans_key, (multiplicity, 1, 3), dtype=jnp.float32) * s_trans
    )
    return q.astype(dtype), random_tr.astype(dtype)


def _weighted_rigid_align(
    true_coords: jnp.ndarray,
    pred_coords: jnp.ndarray,
    weights: jnp.ndarray,
    mask: jnp.ndarray,
) -> jnp.ndarray:
    """JAX port of ``boltz.model.loss.diffusionv2.weighted_rigid_align``.

    Aligns ``true_coords`` onto ``pred_coords`` with a weighted rigid (proper
    rotation + translation) transform. Algorithm 28.
    """

    dim = true_coords.shape[-1]
    w = (mask * weights)[..., None]

    true_centroid = (true_coords * w).sum(axis=-2, keepdims=True) / w.sum(
        axis=-2, keepdims=True
    )
    pred_centroid = (pred_coords * w).sum(axis=-2, keepdims=True) / w.sum(
        axis=-2, keepdims=True
    )
    true_centered = true_coords - true_centroid
    pred_centered = pred_coords - pred_centroid

    # cov[..., i, j] = sum_n (w * pred_centered)[..., n, i] * true_centered[..., n, j]
    cov = jnp.einsum("...ni,...nj->...ij", w * pred_centered, true_centered)
    u, _, vh = jnp.linalg.svd(cov.astype(jnp.float32), full_matrices=True)
    v = jnp.swapaxes(vh, -1, -2)

    rot = jnp.einsum("...ij,...kj->...ik", u, v)
    det = jnp.linalg.det(rot)
    f = jnp.broadcast_to(jnp.eye(dim, dtype=jnp.float32), cov.shape[:-2] + (dim, dim))
    f = f.at[..., -1, -1].set(det)
    rot = jnp.einsum("...ij,...jk,...lk->...il", u, f, v)

    aligned = (
        jnp.einsum("...ni,...ji->...nj", true_centered, rot.astype(true_coords.dtype))
        + pred_centroid
    )
    return aligned


def boltz2_trunk_forward(
    params: Params,
    feats: Mapping[str, jnp.ndarray],
    *,
    recycling_steps: int = 0,
    use_bond_type_feature: bool = True,
    # Boltz-2 conf checkpoint hyper_parameters set fix_sym_check=True and
    # cyclic_pos_enc=True (the nn.Module constructor defaults are False, but
    # Lightning restores the trained hparams). The relative-position chain
    # encoding differs under fix_sym_check even for single-chain inputs, so the
    # port must match the checkpoint to reproduce torch s/z.
    cyclic_pos_enc: bool = True,
    fix_sym_check: bool = True,
    eps: float = 1e-5,
    use_scan: bool = True,
    chunk_size: int = 128,
    triangle_attention_chunk: int | None = None,
    triangle_attention_q_chunk: int | None = None,
    transition_hidden_chunk: int | None = None,
    matmul_precision: str = "highest",
    attention_backend: str = "xla",
    triangle_backend: str = "cueq",
    glu_backend: str = "xla",
    subsample_msa: bool = False,
    num_subsampled_msa: int = 1024,
    msa_key: jnp.ndarray | None = None,
    msa_rows: jnp.ndarray | None = None,
    mesh: object | None = None,
    token_axis: str = "tok",
    shard_tokens: bool = True,
    use_template: bool | None = None,
) -> dict[str, jnp.ndarray]:
    """Run the non-template Boltz-2 trunk in eval mode.

    ``msa_rows`` is the per-pass MSA subsample injection point, shape
    ``[recycling_steps + 1, num_subsampled_msa]``; it overrides ``msa_key`` so
    a parity harness can replay the row draws upstream made. ``None``
    (default) is inert.

    OPT-IN multi-device sharding. With ``mesh=None`` (DEFAULT) no sharding
    constraints are emitted and the path is bit-identical to before. When
    ``mesh`` is a ``jax.sharding.Mesh`` the trunk single ``s`` [B, N, C] and
    pair ``z`` [B, N, N, C] activations are constrained over the mesh via
    ``jax.lax.with_sharding_constraint``; params stay replicated and XLA/GSPMD
    auto-inserts the cross-token communication for triangle multiplication,
    triangle attention and outer-product-mean.

    Bit-exactness trade-off (fp32):
      - ``mesh`` size 1 (e.g. a single physical device): degenerates to
        replicated -> BIT-EXACT vs the no-mesh path regardless of
        ``shard_tokens``.
      - ``mesh`` size > 1 with ``shard_tokens=True``: the token (N) axis is
        partitioned, so the i/j/k token contractions become distributed
        partial-sum matmuls whose fp32 reduction order differs from the
        single-device order. This is NOT bit-exact (~1e-3 per trunk pass on
        CPU GSPMD, amplified by diffusion) -- it trades a tiny fp reorder for
        real activation distribution across devices.
      - ``mesh`` size > 1 with ``shard_tokens=False``: replicated constraints
        are emitted (buffers placed on the mesh, layout-only distribution, no
        partitioned reductions) -> BIT-EXACT vs the no-mesh path. Used by the
        ``scripts/check_sharding.py`` parity gate to validate the sharding path
        on 8 simulated CPU devices.
    """

    chunks = resolve_long_sequence_chunks(
        feats["token_pad_mask"].shape[1],
        chunk_size=chunk_size,
        triangle_attention_chunk=triangle_attention_chunk,
        triangle_attention_q_chunk=triangle_attention_q_chunk,
        token_attention_chunk=None,
    )
    chunk_size = chunks["chunk_size"]
    triangle_attention_chunk = chunks["triangle_attention_chunk"]
    triangle_attention_q_chunk = chunks["triangle_attention_q_chunk"]

    s_inputs = input_embedder_forward(
        params["input_embedder"],
        feats,
        eps=eps,
        attention_backend=attention_backend,
    )
    s_init = _linear(s_inputs, params["s_init"]["kernel"])
    z_init = (
        _linear(s_inputs, params["z_init_1"]["kernel"])[:, :, None, :]
        + _linear(s_inputs, params["z_init_2"]["kernel"])[:, None, :, :]
    )
    relative_position_encoding = relative_position_forward(
        params["rel_pos"],
        feats,
        cyclic_pos_enc=cyclic_pos_enc,
        fix_sym_check=fix_sym_check,
    )
    z_init = z_init + relative_position_encoding
    bond_kernel = params["token_bonds"]["kernel"]
    z_init = z_init + _linear(
        feats["token_bonds"].astype(bond_kernel.dtype),
        bond_kernel,
    )
    if use_bond_type_feature and "token_bonds_type" in params:
        z_init = (
            z_init + params["token_bonds_type"][feats["type_bonds"].astype(jnp.int32)]
        )
    z_init = z_init + contact_conditioning_forward(
        params["contact_conditioning"],
        feats,
    )

    s_init = _shard_single(s_init, mesh, token_axis, shard_tokens)
    z_init = _shard_pair(z_init, mesh, token_axis, shard_tokens)
    s = jnp.zeros_like(s_init)
    z = jnp.zeros_like(z_init)
    mask = feats["token_pad_mask"].astype(jnp.float32)
    pair_mask = mask[:, :, None] * mask[:, None, :]
    # Gate the template path statically: only when template weights are present
    # AND the input carries a real (non-dummy) template. No-template inputs take
    # the original path and stay bit-identical to the non-template trunk.
    #
    # The "is there a real template" question is answered from array *values*,
    # which is a Python-level decision that cannot be made on a tracer. Callers
    # under jit therefore decide it from the concrete features and pass it in;
    # `None` keeps the eager behaviour for direct callers and tests.
    if use_template is None:
        use_template = has_template_feats(feats)
    use_template = bool(use_template) and "template_module" in params

    def _recycle_step(
        s: jnp.ndarray,
        z: jnp.ndarray,
        step_key: jnp.ndarray | None,
        step_rows: jnp.ndarray | None = None,
    ) -> tuple[jnp.ndarray, ...]:
        s = s_init + _linear(
            _layer_norm(
                s,
                params["s_norm"]["scale"],
                params["s_norm"]["bias"],
                eps,
            ),
            params["s_recycle"]["kernel"],
        )
        z = z_init + _linear(
            _layer_norm(
                z,
                params["z_norm"]["scale"],
                params["z_norm"]["bias"],
                eps,
            ),
            params["z_recycle"]["kernel"],
        )
        s = _shard_single(s, mesh, token_axis, shard_tokens)
        z = _shard_pair(z, mesh, token_axis, shard_tokens)
        if use_template:
            z = z + template_module_forward(
                params["template_module"],
                z,
                feats,
                pair_mask,
                eps=eps,
                chunk_size=chunk_size,
                triangle_attention_chunk=triangle_attention_chunk,
                triangle_attention_q_chunk=triangle_attention_q_chunk,
                transition_hidden_chunk=transition_hidden_chunk,
                triangle_backend=triangle_backend,
            )
        z = z + msa_module_forward(
            params["msa_module"],
            z,
            s_inputs,
            feats,
            eps=eps,
            use_scan=use_scan,
            chunk_size=chunk_size,
            triangle_attention_chunk=triangle_attention_chunk,
            triangle_attention_q_chunk=triangle_attention_q_chunk,
            transition_hidden_chunk=transition_hidden_chunk,
            matmul_precision=matmul_precision,
            triangle_backend=triangle_backend,
            glu_backend=glu_backend,
            subsample_msa=subsample_msa,
            num_subsampled_msa=num_subsampled_msa,
            msa_key=step_key,
            msa_rows=step_rows,
        )
        s, z = pairformer_module_forward(
            params["pairformer_module"],
            s,
            z,
            mask,
            pair_mask,
            eps=eps,
            use_scan=use_scan,
            chunk_size=chunk_size,
            triangle_attention_chunk=triangle_attention_chunk,
            triangle_attention_q_chunk=triangle_attention_q_chunk,
            transition_hidden_chunk=transition_hidden_chunk,
            matmul_precision=matmul_precision,
            attention_backend=attention_backend,
            triangle_backend=triangle_backend,
            glu_backend=glu_backend,
        )
        s = _shard_single(s, mesh, token_axis, shard_tokens)
        z = _shard_pair(z, mesh, token_axis, shard_tokens)
        return s, z

    # Recycling is a uniform fixed-point iteration over the (s, z) carry; the
    # body is identical every step, so `lax.scan` traces the trunk once instead
    # of unrolling `recycling_steps + 1` copies into the HLO (faster compile,
    # smaller executable). Runtime/peak is unchanged. Eager path kept for parity
    # debugging and when scan is disabled.
    # Upstream re-draws its MSA subsample inside every forward pass, so each
    # recycle needs its own key. Splitting up front keeps the scan body's carry
    # free of the key and makes the eager and scanned paths draw the same rows.
    step_keys = (
        jax.random.split(msa_key, recycling_steps + 1) if msa_key is not None else None
    )
    step_rows = None
    if msa_rows is not None:
        step_rows = jnp.asarray(msa_rows, dtype=jnp.int32)
        if step_rows.ndim != 2 or step_rows.shape[0] != recycling_steps + 1:
            raise ValueError(
                "msa_rows must have shape [recycling_steps + 1, "
                f"num_subsampled_msa], got {step_rows.shape} for "
                f"{recycling_steps + 1} passes"
            )
    if use_scan:

        def _scan_body(carry, xs):
            step_key, step_row = xs
            return _recycle_step(*carry, step_key, step_row), None

        (s, z), _ = jax.lax.scan(
            _scan_body,
            (s, z),
            xs=(step_keys, step_rows),
            length=recycling_steps + 1,
        )
    else:
        for cycle in range(recycling_steps + 1):
            s, z = _recycle_step(
                s,
                z,
                None if step_keys is None else step_keys[cycle],
                None if step_rows is None else step_rows[cycle],
            )

    return {
        "s_inputs": s_inputs,
        "s": s,
        "z": z,
        "relative_position_encoding": relative_position_encoding,
    }


def relative_position_forward(
    params: Params,
    feats: Mapping[str, jnp.ndarray],
    *,
    r_max: int = 32,
    s_max: int = 2,
    # Match the Boltz-2 conf checkpoint config (see boltz2_trunk_forward).
    cyclic_pos_enc: bool = True,
    fix_sym_check: bool = True,
) -> jnp.ndarray:
    """Run Boltz RelativePositionEncoder."""

    kernel = params["linear_layer"]["kernel"]
    b_same_chain = feats["asym_id"][:, :, None] == feats["asym_id"][:, None, :]
    b_same_residue = (
        feats["residue_index"][:, :, None] == feats["residue_index"][:, None, :]
    )
    b_same_entity = feats["entity_id"][:, :, None] == feats["entity_id"][:, None, :]

    d_residue = feats["residue_index"][:, :, None] - feats["residue_index"][:, None, :]
    if cyclic_pos_enc:
        period = jnp.where(
            feats["cyclic_period"] > 0,
            feats["cyclic_period"],
            jnp.zeros_like(feats["cyclic_period"]) + 10000,
        )
        d_residue = d_residue - period * jnp.round(d_residue / period)
    d_residue = jnp.clip(d_residue + r_max, 0, 2 * r_max).astype(jnp.int32)
    d_residue = jnp.where(b_same_chain, d_residue, 2 * r_max + 1)

    d_token = jnp.clip(
        feats["token_index"][:, :, None] - feats["token_index"][:, None, :] + r_max,
        0,
        2 * r_max,
    ).astype(jnp.int32)
    d_token = jnp.where(b_same_chain & b_same_residue, d_token, 2 * r_max + 1)

    d_chain = jnp.clip(
        feats["sym_id"][:, :, None] - feats["sym_id"][:, None, :] + s_max,
        0,
        2 * s_max,
    ).astype(jnp.int32)
    same_chain_condition = ~b_same_entity if fix_sym_check else b_same_chain
    d_chain = jnp.where(same_chain_condition, 2 * s_max + 1, d_chain)

    n_rel = 2 * r_max + 2
    n_chain = 2 * s_max + 2
    rel_pos_kernel = kernel[:n_rel]
    rel_token_kernel = kernel[n_rel : 2 * n_rel]
    entity_kernel = kernel[2 * n_rel]
    rel_chain_kernel = kernel[2 * n_rel + 1 : 2 * n_rel + 1 + n_chain]
    return (
        rel_pos_kernel[d_residue]
        + rel_token_kernel[d_token]
        + b_same_entity[..., None].astype(kernel.dtype) * entity_kernel
        + rel_chain_kernel[d_chain]
    )


def contact_conditioning_forward(
    params: Params,
    feats: Mapping[str, jnp.ndarray],
    *,
    cutoff_min: float = 4.0,
    cutoff_max: float = 20.0,
) -> jnp.ndarray:
    """Run Boltz ContactConditioning."""

    compute_dtype = params["encoder"]["kernel"].dtype
    contact_conditioning = feats["contact_conditioning"][:, :, :, 2:]
    contact_threshold_normalized = (feats["contact_threshold"] - cutoff_min) / (
        cutoff_max - cutoff_min
    )
    fourier_proj = params["fourier_embedding"]["proj"]
    flat = contact_threshold_normalized.astype(compute_dtype).reshape((-1, 1))
    fourier = jnp.cos(
        2.0 * pi * _linear(flat, fourier_proj["kernel"], fourier_proj["bias"])
    ).reshape((*contact_threshold_normalized.shape, -1))
    contact_conditioning = jnp.concatenate(
        (
            contact_conditioning.astype(compute_dtype),
            contact_threshold_normalized[..., None].astype(compute_dtype),
            fourier.astype(compute_dtype),
        ),
        axis=-1,
    )
    encoded = _linear(
        contact_conditioning,
        params["encoder"]["kernel"],
        params["encoder"]["bias"],
    )
    flags = feats["contact_conditioning"].astype(encoded.dtype)
    one = jnp.asarray(1.0, dtype=encoded.dtype)
    return (
        encoded * (one - jnp.sum(flags[:, :, :, 0:2], axis=-1, keepdims=True))
        + params["encoding_unspecified"] * flags[:, :, :, 0:1]
        + params["encoding_unselected"] * flags[:, :, :, 1:2]
    )


def _preconditioned_score_forward(
    params: Params,
    *,
    s_inputs: jnp.ndarray,
    s_trunk: jnp.ndarray,
    r_noisy: jnp.ndarray,
    sigma: jnp.ndarray,
    feats: Mapping[str, jnp.ndarray],
    diffusion_conditioning: Mapping[str, object],
    multiplicity: int,
    sigma_data: float,
    eps: float,
    use_scan: bool = True,
    attention_backend: str = "xla",
    token_attention_chunk: int | None = None,
    token_layers: int | None = None,
) -> jnp.ndarray:
    padded_sigma = jnp.reshape(sigma, (1, 1, 1))
    scaled_input = r_noisy / jnp.sqrt(padded_sigma**2 + sigma_data**2)
    # Run the score network activations in the param compute dtype (bf16/fp16
    # when opted in), but keep the c_skip/c_out preconditioning (which feeds the
    # fp32 Euler step) in fp32. r_noisy itself stays fp32.
    compute_dtype = params["s_to_a_linear"]["linear"]["kernel"].dtype
    times = jnp.full(
        (r_noisy.shape[0],),
        jnp.log(sigma / sigma_data) * 0.25,
        dtype=compute_dtype,
    )
    r_update = diffusion_score_model_forward(
        params,
        s_inputs=s_inputs,
        s_trunk=s_trunk,
        r_noisy=scaled_input.astype(compute_dtype),
        times=times,
        feats=feats,
        diffusion_conditioning=diffusion_conditioning,
        multiplicity=multiplicity,
        eps=eps,
        use_scan=use_scan,
        attention_backend=attention_backend,
        token_attention_chunk=token_attention_chunk,
        token_layers=token_layers,
    )
    r_update = r_update.astype(jnp.float32)
    c_skip = sigma_data**2 / (padded_sigma**2 + sigma_data**2)
    c_out = padded_sigma * sigma_data / jnp.sqrt(sigma_data**2 + padded_sigma**2)
    return c_skip * r_noisy + c_out * r_update


def _sample_schedule(
    num_sampling_steps: int,
    *,
    sigma_min: float,
    sigma_max: float,
    sigma_data: float,
    rho: float,
) -> jnp.ndarray:
    inv_rho = 1.0 / rho
    steps = jnp.arange(num_sampling_steps, dtype=jnp.float32)
    sigmas = (
        sigma_max**inv_rho
        + steps / (num_sampling_steps - 1) * (sigma_min**inv_rho - sigma_max**inv_rho)
    ) ** rho
    sigmas = sigmas * sigma_data
    return jnp.pad(sigmas, (0, 1))
