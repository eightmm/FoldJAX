"""Single Protenix inference wrapper.

This module is the torch-free library API equivalent of the static inference
CLI: static features + JAX params + PRNG key -> one prediction dictionary.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp

from foldjax.models.protenix.models.diffusion.diffusion import inference_noise_schedule
from foldjax.models.protenix.models.model import (
    ProtenixInferenceParams,
    protenix_infer_compiled,
    protenix_infer_static,
)


def protenix_predict_static(
    params: ProtenixInferenceParams,
    features: Mapping[str, jnp.ndarray | Mapping[str, jnp.ndarray]],
    key: jax.Array | None,
    *,
    n_sample: int = 1,
    num_sampling_steps: int = 200,
    s_max: float = 160.0,
    s_min: float = 4.0e-4,
    rho: float = 7.0,
    sigma_data: float = 16.0,
    recycling_steps: int = 10,
    init_noise: jnp.ndarray | None = None,
    step_noises: tuple[jnp.ndarray, ...] | None = None,
    pair_mask: jnp.ndarray | None = None,
    input_atom_heads: int = 4,
    atom_encoder_heads: int = 4,
    token_heads: int = 16,
    atom_decoder_heads: int = 4,
    n_queries: int = 32,
    n_keys: int = 128,
    use_pairformer_scan: bool = False,
    use_confidence_scan: bool = False,
    use_diffusion_scan: bool = False,
    use_sampler_scan: bool = True,
    use_denoiser_jit: bool = False,
    use_diffusion_efficient_fusion: bool = False,
    diffusion_attention_backend: str = "xla_jit",
    trunk_single_attention_backend: str = "xla_jit",
    trunk_triangle_attention_backend: str | None = None,
    #: ``None`` follows the trunk backend; see model.py for why this exists.
    confidence_triangle_attention_backend: str | None = None,
    use_confidence_embedding: bool = True,
    run_confidence: bool = True,
    run_confidence_scores: bool = True,
    triangle_mul_chunk_size: int | None = None,
    triangle_att_q_chunk_size: int | None = None,
    single_att_q_chunk_size: int | None = None,
    token_q_chunk_size: int | None = None,
    opm_chunk_size: int | None = None,
    diffusion_chunk_size: int | None = None,
    gamma0: float = 0.8,
    gamma_min: float = 1.0,
    noise_scale_lambda: float = 1.003,
    step_scale_eta: float = 1.5,
    centre_each_step: bool = True,
    # TF32, which is what upstream's released inference config turns on:
    # `configs_inference.py:32` sets `enable_tf32: True`, and
    # `protenix/model/protenix.py:99` applies it as
    # `torch.backends.cuda.matmul.allow_tf32`. This was "highest" -- true
    # float32 -- which made the port strictly more precise than the model it
    # ports, and paid ~17% of its runtime for the privilege. Measured at 1,003
    # tokens over five samples: 49.9 -> 41.5 s warm, peak unchanged, and the
    # structures move less than running the same program twice does (CA RMSD
    # 0.12 A median against a 0.55 A rerun floor; this model's own sample spread
    # is 0.9-6.3 A). Pass "highest" for a float32 reference tape.
    matmul_precision: str = "high",
    trunk_dtype: jnp.dtype | None = None,
    cycle_msa_features: tuple[dict[str, jnp.ndarray], ...] | None = None,
    guidance_config: Mapping[str, Any] | None = None,
    guidance_features: Mapping[str, Any] | None = None,
    graph_jit: bool = True,
) -> dict[str, jnp.ndarray]:
    """Run the static-feature Protenix inference graph.

    ``graph_jit`` traces the whole model as one program instead of dispatching
    it primitive by primitive; see :func:`protenix_infer_compiled`. Guidance
    configuration is a plain mapping and cannot travel as a static argument,
    so a guided run falls back to the eager path on its own.
    """

    # Pin the f32 matmul/einsum precision rather than inheriting it: the setting
    # is process-global in JAX, so without a pin this port runs at whatever
    # another model left behind, and it is invisible on CPU where the parity gate
    # runs. Pinned to what upstream selects for itself -- see the argument's
    # default. Islands needing more ask for it explicitly, which is why
    # `heads/confidence.py` pins `Precision.HIGHEST` on the pTM contraction.
    #
    # A scope, not a latch. This was `jax.config.update`, which left the setting
    # behind for everything later in the process -- another port, a notebook
    # cell, a test collected after this one. It went unnoticed while every port
    # pinned the same value; they no longer do.
    with jax.default_matmul_precision(matmul_precision):
        noise_schedule = inference_noise_schedule(
            n_step=num_sampling_steps,
            s_max=s_max,
            s_min=s_min,
            rho=rho,
            sigma_data=sigma_data,
        )
        guided = guidance_config is not None and guidance_config.get("enable")
        if graph_jit and not guided:
            infer = protenix_infer_compiled
            # Rolling the repeated stacks into `lax.scan` only pays once the whole
            # graph is traced: unrolled, every block becomes its own copy in one
            # executable, which is what makes the consolidated program slow to
            # build and slow to load back from the compile cache.
            use_pairformer_scan = True
            use_confidence_scan = True
            use_diffusion_scan = True
        else:
            infer = protenix_infer_static
        return infer(
            dict(features),
            params,
            noise_schedule,
            key=key,
            n_sample=n_sample,
            init_noise=init_noise,
            step_noises=step_noises,
            n_cycle=recycling_steps,
            pair_mask=pair_mask,
            input_atom_heads=input_atom_heads,
            atom_encoder_heads=atom_encoder_heads,
            token_heads=token_heads,
            atom_decoder_heads=atom_decoder_heads,
            n_queries=n_queries,
            n_keys=n_keys,
            sigma_data=sigma_data,
            use_pairformer_scan=use_pairformer_scan,
            use_confidence_scan=use_confidence_scan,
            use_diffusion_scan=use_diffusion_scan,
            use_sampler_scan=use_sampler_scan,
            use_denoiser_jit=use_denoiser_jit,
            use_diffusion_efficient_fusion=use_diffusion_efficient_fusion,
            diffusion_attention_backend=diffusion_attention_backend,
            trunk_single_attention_backend=trunk_single_attention_backend,
            trunk_triangle_attention_backend=trunk_triangle_attention_backend,
            confidence_triangle_attention_backend=confidence_triangle_attention_backend,
            use_confidence_embedding=use_confidence_embedding,
            run_confidence=run_confidence,
            run_confidence_scores=run_confidence_scores,
            triangle_mul_chunk_size=triangle_mul_chunk_size,
            triangle_att_q_chunk_size=triangle_att_q_chunk_size,
            single_att_q_chunk_size=single_att_q_chunk_size,
            token_q_chunk_size=token_q_chunk_size,
            opm_chunk_size=opm_chunk_size,
            diffusion_chunk_size=diffusion_chunk_size,
            gamma0=gamma0,
            gamma_min=gamma_min,
            noise_scale_lambda=noise_scale_lambda,
            step_scale_eta=step_scale_eta,
            centre_each_step=centre_each_step,
            trunk_dtype=trunk_dtype,
            cycle_msa_features=cycle_msa_features,
            guidance_config=guidance_config,
            guidance_features=guidance_features,
        )
