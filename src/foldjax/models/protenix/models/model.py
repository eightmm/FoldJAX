"""Top-level infer-only Protenix JAX orchestration."""

from __future__ import annotations

from collections.abc import Mapping
from functools import partial
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from foldjax.models._graph import (
    merge_static_flags,
    split_static_flags,
    traceable_features,
)
from foldjax.models.protenix.models.diffusion.atom import (
    atom_attention_encoder_prepare_diffusion_cache,
)
from foldjax.models.protenix.models.diffusion.diffusion import (
    DiffusionModuleParams,
    diffusion_conditioning_prepare_cache,
    sample_diffusion_with_module,
)
from foldjax.models.protenix.models.heads.confidence import (
    ConfidenceHeadParams,
    confidence_head,
    confidence_scores_from_logits,
)
from foldjax.models.protenix.models.heads.head import DistogramParams, distogram_head
from foldjax.models.protenix.models.trunk_blocks.embedders import (
    InputFeatureEmbedderParams,
    input_feature_embedder,
)
from foldjax.models.protenix.models.trunk_blocks.trunk import (
    PairformerOutputParams,
    pairformer_output_from_s_inputs,
)


class ProtenixInferenceParams(NamedTuple):
    """Parameters for the current infer-only Protenix wrapper."""

    input_embedder: InputFeatureEmbedderParams
    pairformer_output: PairformerOutputParams
    diffusion: DiffusionModuleParams
    distogram: DistogramParams
    confidence: ConfidenceHeadParams


def cast_trunk_params(
    params: ProtenixInferenceParams,
    dtype: jnp.dtype,
) -> ProtenixInferenceParams:
    """Cast input/trunk parameters while preserving FP32 output-head islands."""

    def cast_leaf(value):
        if hasattr(value, "dtype") and jnp.issubdtype(value.dtype, jnp.floating):
            return value.astype(dtype)
        return value

    return params._replace(
        input_embedder=jax.tree.map(cast_leaf, params.input_embedder),
        pairformer_output=jax.tree.map(cast_leaf, params.pairformer_output),
    )


def protenix_infer_static(
    input_feature_dict: dict[str, jnp.ndarray | dict[str, jnp.ndarray]],
    params: ProtenixInferenceParams,
    noise_schedule: jnp.ndarray,
    *,
    key: jax.Array | None,
    n_sample: int,
    init_noise: jnp.ndarray | None = None,
    step_noises: tuple[jnp.ndarray, ...] | None = None,
    n_cycle: int = 1,
    pair_mask: jnp.ndarray | None = None,
    input_atom_heads: int = 4,
    atom_encoder_heads: int = 4,
    token_heads: int = 16,
    atom_decoder_heads: int = 4,
    n_queries: int = 32,
    n_keys: int = 128,
    sigma_data: float = 16.0,
    use_pairformer_scan: bool = False,
    use_confidence_scan: bool = False,
    use_diffusion_scan: bool = False,
    use_sampler_scan: bool = True,
    use_denoiser_jit: bool = False,
    use_diffusion_efficient_fusion: bool = False,
    diffusion_attention_backend: str = "xla_jit",
    trunk_single_attention_backend: str = "xla_jit",
    trunk_triangle_attention_backend: str | None = None,
    #: Backend for the confidence head's triangle attention. ``None`` follows the
    #: trunk. The head runs on the same [n_token, n_token, c_z] pair tensor with
    #: the same head layout, so it has no business defaulting to a different
    #: kernel than the trunk -- but it did: this call site passed no backend, so
    #: the head took the module default while the trunk ran whatever was asked
    #: for. The XLA path materialises the [rows, heads, q_chunk, keys] scores; at
    #: 2030 tokens that is a 15.72 GiB buffer inside a 39.07 GiB temp arena,
    #: which is the allocation that killed the target after the compute had
    #: already finished.
    confidence_triangle_attention_backend: str | None = None,
    use_confidence_embedding: bool = True,
    run_confidence: bool = True,
    run_confidence_scores: bool = True,
    triangle_mul_chunk_size: int | None = None,
    triangle_att_q_chunk_size: int | None = None,
    single_att_q_chunk_size: int | None = None,
    token_q_chunk_size: int | None = None,
    diffusion_chunk_size: int | None = None,
    gamma0: float = 0.8,
    gamma_min: float = 1.0,
    noise_scale_lambda: float = 1.003,
    step_scale_eta: float = 1.5,
    centre_each_step: bool = True,
    trunk_dtype: jnp.dtype | None = None,
    cycle_msa_features: tuple[dict[str, jnp.ndarray], ...] | None = None,
    guidance_config: Mapping[str, Any] | None = None,
    guidance_features: Mapping[str, Any] | None = None,
    n_chain: int | None = None,
) -> dict[str, jnp.ndarray]:
    """Run the currently ported static-feature Protenix inference path.

    ``n_chain`` is the number of chains in the job. The confidence scores need
    it as a Python integer -- it sizes per-chain loops -- and by default they
    read it back off ``asym_id``, which is a tracer inside ``jit``. The
    consolidated graph resolves it on the host and passes it here.
    """

    n_token = int(input_feature_dict["restype"].shape[-2])
    trunk_features = input_feature_dict
    trunk_pair_mask = pair_mask
    if trunk_dtype is not None:
        trunk_features = jax.tree.map(
            lambda value: (
                value.astype(trunk_dtype)
                if hasattr(value, "dtype") and jnp.issubdtype(value.dtype, jnp.floating)
                else value
            ),
            input_feature_dict,
        )
        if pair_mask is not None:
            trunk_pair_mask = pair_mask.astype(trunk_dtype)
    s_inputs = input_feature_embedder(
        trunk_features,
        params.input_embedder,
        n_token=n_token,
        n_heads=input_atom_heads,
        n_queries=n_queries,
        n_keys=n_keys,
        use_scan=use_diffusion_scan,
    )
    s_inputs, s_trunk, z_trunk = pairformer_output_from_s_inputs(
        trunk_features,
        s_inputs,
        params.pairformer_output,
        n_cycle=n_cycle,
        pair_mask=trunk_pair_mask,
        use_pairformer_scan=use_pairformer_scan,
        triangle_mul_chunk_size=triangle_mul_chunk_size,
        triangle_att_q_chunk_size=triangle_att_q_chunk_size,
        single_att_q_chunk_size=single_att_q_chunk_size,
        single_attention_backend=trunk_single_attention_backend,
        triangle_attention_backend=trunk_triangle_attention_backend,
        cycle_msa_features=cycle_msa_features,
    )
    diffusion_s_inputs = s_inputs.astype(jnp.float32)
    diffusion_s_trunk = s_trunk.astype(jnp.float32)
    diffusion_z_trunk = z_trunk.astype(jnp.float32)
    relp = input_feature_dict["relp"]
    pair_z = diffusion_conditioning_prepare_cache(
        relp,
        diffusion_z_trunk,
        params.diffusion.conditioning,
    )
    p_lm, c_l = atom_attention_encoder_prepare_diffusion_cache(
        input_feature_dict["atom_to_token_idx"],
        input_feature_dict["ref_pos"],
        input_feature_dict["ref_charge"],
        input_feature_dict["ref_mask"],
        input_feature_dict["ref_element"],
        input_feature_dict["ref_atom_name_chars"],
        input_feature_dict["d_lm"],
        input_feature_dict["v_lm"],
        input_feature_dict["pad_info"],
        jnp.expand_dims(pair_z, axis=-4),
        params.diffusion.atom_encoder,
        n_queries=n_queries,
        n_keys=n_keys,
    )
    coordinates = sample_diffusion_with_module(
        input_feature_dict,
        diffusion_s_inputs,
        diffusion_s_trunk,
        diffusion_z_trunk,
        params.diffusion,
        noise_schedule,
        n_sample=n_sample,
        key=key,
        pair_z=pair_z,
        p_lm=p_lm,
        c_l=c_l,
        atom_encoder_heads=atom_encoder_heads,
        token_heads=token_heads,
        atom_decoder_heads=atom_decoder_heads,
        n_queries=n_queries,
        n_keys=n_keys,
        sigma_data=sigma_data,
        use_scan=use_diffusion_scan,
        use_sampler_scan=use_sampler_scan,
        use_denoiser_jit=use_denoiser_jit,
        use_efficient_fusion=use_diffusion_efficient_fusion,
        attention_backend=diffusion_attention_backend,
        token_q_chunk_size=token_q_chunk_size,
        diffusion_chunk_size=diffusion_chunk_size,
        gamma0=gamma0,
        gamma_min=gamma_min,
        noise_scale_lambda=noise_scale_lambda,
        step_scale_eta=step_scale_eta,
        centre_each_step=centre_each_step,
        init_noise=init_noise,
        step_noises=step_noises,
        guidance_config=guidance_config,
        guidance_features=(
            input_feature_dict if guidance_features is None else guidance_features
        ),
    )
    output = {
        "s_inputs": s_inputs,
        "s_trunk": s_trunk,
        "z_trunk": z_trunk,
        "coordinate": coordinates,
        "distogram_logits": distogram_head(diffusion_z_trunk, params.distogram),
    }
    if run_confidence:
        confidence_logits = confidence_head(
            input_feature_dict,
            diffusion_s_inputs,
            diffusion_s_trunk,
            diffusion_z_trunk,
            pair_mask,
            coordinates,
            params.confidence,
            use_embedding=use_confidence_embedding,
            use_scan=use_confidence_scan,
            triangle_mul_chunk_size=triangle_mul_chunk_size,
            triangle_att_q_chunk_size=triangle_att_q_chunk_size,
            single_att_q_chunk_size=single_att_q_chunk_size,
            triangle_attention_backend=(
                trunk_triangle_attention_backend
                if confidence_triangle_attention_backend is None
                else confidence_triangle_attention_backend
            ),
        )
        output.update(confidence_logits)
        if run_confidence_scores:
            output.update(
                confidence_scores_from_logits(
                    plddt_logits=confidence_logits["plddt"],
                    pae_logits=confidence_logits["pae"],
                    pde_logits=confidence_logits["pde"],
                    distogram_logits=output["distogram_logits"],
                    token_has_frame=input_feature_dict.get("has_frame"),
                    token_asym_id=input_feature_dict.get("asym_id"),
                    atom_to_token_idx=input_feature_dict.get("atom_to_token_idx"),
                    atom_coordinate=coordinates,
                    elements_one_hot=input_feature_dict.get("ref_element"),
                    mol_id=input_feature_dict.get("mol_id"),
                    token_is_ligand=input_feature_dict.get(
                        "token_is_ligand",
                        jnp.argmax(input_feature_dict["restype"], axis=-1) == 20,
                    ),
                    num_recycles=n_cycle,
                    n_chain=n_chain,
                )
            )
    return output


# Static arguments of the consolidated graph: everything that changes the
# program rather than the data. Anything not listed here is traced.
GRAPH_STATIC_ARGNAMES = (
    "atom_decoder_heads",
    "atom_encoder_heads",
    "centre_each_step",
    "confidence_triangle_attention_backend",
    "diffusion_attention_backend",
    "diffusion_chunk_size",
    "gamma0",
    "gamma_min",
    "input_atom_heads",
    "n_cycle",
    "n_keys",
    "n_queries",
    "n_sample",
    "noise_scale_lambda",
    "run_confidence",
    "run_confidence_scores",
    "sigma_data",
    "single_att_q_chunk_size",
    "step_scale_eta",
    "token_heads",
    "token_q_chunk_size",
    "triangle_att_q_chunk_size",
    "triangle_mul_chunk_size",
    "trunk_dtype",
    "n_chain",
    "trunk_single_attention_backend",
    "trunk_triangle_attention_backend",
    "use_confidence_embedding",
    "use_confidence_scan",
    "use_denoiser_jit",
    "use_diffusion_efficient_fusion",
    "use_diffusion_scan",
    "use_pairformer_scan",
    "use_sampler_scan",
)


@partial(
    jax.jit,
    static_argnames=(*GRAPH_STATIC_ARGNAMES, "params_treedef", "params_flags"),
)
def _compiled_protenix_infer(
    input_feature_dict: dict[str, Any],
    param_arrays: tuple,
    noise_schedule: jnp.ndarray,
    *,
    params_treedef: Any,
    params_flags: tuple[tuple[int, bool], ...],
    **kwargs: Any,
) -> dict[str, jnp.ndarray]:
    return protenix_infer_static(
        input_feature_dict,
        merge_static_flags(param_arrays, params_treedef, params_flags),
        noise_schedule,
        **kwargs,
    )


def protenix_infer_compiled(
    input_feature_dict: Mapping[str, Any],
    params: ProtenixInferenceParams,
    noise_schedule: jnp.ndarray,
    **kwargs: Any,
) -> dict[str, jnp.ndarray]:
    """Run Protenix as one compiled program instead of op by op.

    :func:`protenix_infer_static` is written eagerly -- it calls individually
    jitted primitives -- so one 35-residue prediction dispatched 666 separate
    executables and took 31 s on a warm cache. Tracing the whole function once
    gives XLA the entire graph, so it fuses across the module boundaries the
    eager form hides and the per-call overhead collapses to one dispatch.

    Same computation; only how much of it XLA sees at once changes. Guidance
    (TFG) is the exception: its configuration is a plain mapping that would
    have to be hashable to travel as a static argument, so a guided run stays
    on the eager path. :func:`protenix_infer_static` remains available and
    ``--no-graph-jit`` selects it.
    """
    # The confidence scores size their per-chain loops by the chain count,
    # which they read off `asym_id`. That is a value, so it is resolved here on
    # the concrete features and travels as a static argument.
    kwargs.setdefault("n_chain", int(jnp.max(input_feature_dict["asym_id"])) + 1)
    param_arrays, treedef, flags = split_static_flags(params)
    return _compiled_protenix_infer(
        traceable_features(input_feature_dict),
        param_arrays,
        noise_schedule,
        params_treedef=treedef,
        params_flags=flags,
        **kwargs,
    )
