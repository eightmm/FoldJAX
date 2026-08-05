"""OpenDDE dual-branch inference parameters and feature preparation."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from functools import partial, wraps
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from foldjax.models._graph import (
    merge_static_flags,
    split_static_flags,
    traceable_features,
)
from foldjax.models.opendde.models.diffusion_conditioning import (
    diffusion_conditioning_prepare_cache,
)
from foldjax.models.opendde.models.diffusion_module import diffusion_module_forward
from foldjax.models.opendde.models.sampling import sample_diffusion
from foldjax.models.opendde.models.structural_refiner import structural_refiner_stack
from foldjax.models.opendde.models.structural_tokens import (
    STRUCTURAL_TOKEN_ROLES,
    StructuralTokenExpanderParams,
    structural_token_expand,
)
from foldjax.models.protenix.models.diffusion.atom import (
    atom_attention_encoder_prepare_diffusion_cache,
)
from foldjax.models.protenix.models.diffusion.diffusion import DiffusionModuleParams
from foldjax.models.protenix.models.heads.confidence import (
    ConfidenceHeadParams,
    confidence_head,
)
from foldjax.models.protenix.models.heads.head import DistogramParams, distogram_head
from foldjax.models.protenix.models.trunk_blocks.embedders import (
    InputFeatureEmbedderParams,
    input_feature_embedder,
    relative_position_features,
)
from foldjax.models.protenix.models.trunk_blocks.pairformer import PairformerStackParams
from foldjax.models.protenix.models.trunk_blocks.trunk import (
    PairformerOutputParams,
    pairformer_output_from_s_inputs,
)


class OpenDDEInferenceParams(NamedTuple):
    """Parameters for the OpenDDE residue/structural dual-branch graph."""

    input_embedder: InputFeatureEmbedderParams
    pairformer_output: PairformerOutputParams
    structural_expander: StructuralTokenExpanderParams
    structural_refiner: PairformerStackParams
    diffusion: DiffusionModuleParams
    distogram: DistogramParams
    confidence: ConfidenceHeadParams


def cast_trunk_params(
    params: OpenDDEInferenceParams,
    dtype: jnp.dtype,
) -> OpenDDEInferenceParams:
    """Cast the embedder and both trunks, leaving the output heads in FP32.

    OpenDDE's memory is in the structural refiner: its pair representation is
    ``[n_structural, n_structural, 384]``, and the XLA triangle path holds two
    full projections of that shape at once -- 1,311 MiB each on a 488-residue
    job. Halving the element width is the only lever that touches all three at
    once, and it is the one AlphaFold 3 already pulls (its released trunk runs
    bfloat16 weights and activations, which is most of why its peak is 3,985
    MiB where the FP32 ports sit two to four times higher).

    The diffusion module, distogram, and confidence heads keep FP32 weights.
    The sampler already upcasts its inputs at the boundary, so the cast stops
    where the coordinates start.
    """

    def cast_leaf(value):
        if hasattr(value, "dtype") and jnp.issubdtype(value.dtype, jnp.floating):
            return value.astype(dtype)
        return value

    return params._replace(
        input_embedder=jax.tree.map(cast_leaf, params.input_embedder),
        pairformer_output=jax.tree.map(cast_leaf, params.pairformer_output),
        structural_expander=jax.tree.map(cast_leaf, params.structural_expander),
        structural_refiner=jax.tree.map(cast_leaf, params.structural_refiner),
    )


def _with_cueq_triangle_defaults(function):
    """Run OpenDDE's triangle kernels fused, like upstream does.

    This used to pin both backends to XLA. The pin was measured -- at 1,531
    tokens the fused path asks for a 92.21 GiB arena on a 97,887 MiB card and
    dies -- but it was measured at one size and applied to every size, which is
    the part that was wrong. OpenDDE peaks at 10,552 MiB on a 490-token job and
    34,408 MiB on a 970-token one; there is 60+ GiB spare in both.

    Upstream resolves `auto` to cuequivariance for both kernels
    (config/model_base.py:31-32) and OpenDDE's `c_hidden == c_z == 384` clears
    the kernel's guard, so fused is what upstream actually runs. Protenix, which
    shares this triangle module, went 254.5 -> 167.1 s on the same switch.

    `setdefault` with a restore: an explicit environment still wins, which is
    how a job too large for the fused arena asks for the blocked path, and
    nothing leaks into the process for the next model to inherit.
    """

    @wraps(function)
    def wrapped(*args, **kwargs):
        defaults = {
            "PROTENIX_TRIANGLE_BACKEND": "cueq_jit",
            "PROTENIX_TRIANGLE_MULTIPLICATION_BACKEND": "cueq",
        }
        inserted = []
        for name, value in defaults.items():
            if name not in os.environ:
                os.environ[name] = value
                inserted.append(name)
        try:
            return function(*args, **kwargs)
        finally:
            for name in inserted:
                os.environ.pop(name, None)

    return wrapped


def _validate_static_structural_features(
    features: Mapping[str, Any],
    *,
    require_residue_confidence_mask: bool,
    check_values: bool = True,
) -> tuple[int, int, int]:
    required = {
        "restype",
        "ref_pos",
        "parent_residue_idx",
        "subtoken_role_id",
        "structural_token_index",
        "atom_to_structural_token_idx",
        "atom_to_structural_tokatom_idx",
        "structural_distogram_rep_atom_mask",
        "structural_pae_rep_atom_mask",
        "structural_has_frame",
        "structural_frame_atom_index",
        "token_index",
        "asym_id",
        "residue_index",
        "entity_id",
        "sym_id",
        "atom_to_token_idx",
        "atom_to_tokatom_idx",
        "has_frame",
        "frame_atom_index",
        "pae_rep_atom_mask",
        "distogram_rep_atom_mask",
    }
    missing = sorted(required - features.keys())
    if missing:
        raise KeyError(
            "OpenDDE static inference is missing structural feature(s): "
            + ", ".join(missing)
        )

    restype = jnp.asarray(features["restype"])
    ref_pos = jnp.asarray(features["ref_pos"])
    if restype.ndim != 2 or restype.shape[-1] != 32:
        raise ValueError(
            "OpenDDE static inference requires unbatched restype [N_residue, 32]"
        )
    if ref_pos.ndim != 2 or ref_pos.shape[-1] != 3:
        raise ValueError(
            "OpenDDE static inference requires unbatched ref_pos [N_atom, 3]"
        )
    n_residue = int(restype.shape[0])
    n_atom = int(ref_pos.shape[0])

    parent = jnp.asarray(features["parent_residue_idx"])
    role = jnp.asarray(features["subtoken_role_id"])
    if parent.ndim != 1 or role.ndim != 1:
        raise ValueError(
            "OpenDDE static inference currently supports unbatched structural metadata"
        )
    n_structural = int(parent.shape[0])
    if n_residue < 1 or n_structural < 1 or n_atom < 1:
        raise ValueError("OpenDDE static inference requires non-empty inputs")

    expected_shapes = {
        "subtoken_role_id": (n_structural,),
        "structural_token_index": (n_structural,),
        "atom_to_structural_token_idx": (n_atom,),
        "atom_to_structural_tokatom_idx": (n_atom,),
        "structural_distogram_rep_atom_mask": (n_atom,),
        "structural_pae_rep_atom_mask": (n_atom,),
        "structural_has_frame": (n_structural,),
        "structural_frame_atom_index": (n_structural, 3),
        "token_index": (n_residue,),
        "asym_id": (n_residue,),
        "residue_index": (n_residue,),
        "entity_id": (n_residue,),
        "sym_id": (n_residue,),
        "atom_to_token_idx": (n_atom,),
        "atom_to_tokatom_idx": (n_atom,),
        "has_frame": (n_residue,),
        "frame_atom_index": (n_residue, 3),
        "pae_rep_atom_mask": (n_atom,),
        "distogram_rep_atom_mask": (n_atom,),
    }
    for name, expected in expected_shapes.items():
        actual = tuple(jnp.asarray(features[name]).shape)
        if actual != expected:
            raise ValueError(f"{name} expected shape {expected}, got {actual}")

    integer_indices = (
        "parent_residue_idx",
        "subtoken_role_id",
        "structural_token_index",
        "atom_to_structural_token_idx",
        "atom_to_structural_tokatom_idx",
        "token_index",
        "asym_id",
        "residue_index",
        "entity_id",
        "sym_id",
        "atom_to_token_idx",
        "atom_to_tokatom_idx",
    )
    for name in integer_indices:
        if not jnp.issubdtype(jnp.asarray(features[name]).dtype, jnp.integer):
            raise TypeError(f"{name} must have an integer dtype")

    def has_out_of_range(
        value: jnp.ndarray,
        upper_bound: int,
    ) -> bool:
        if not check_values:
            # Under `jax.jit` these arrays are tracers, so the comparison cannot
            # be resolved in Python. The caller runs these checks on the
            # concrete features before tracing instead; see
            # `opendde_infer_compiled`.
            return False
        invalid = jnp.any((value < 0) | (value >= upper_bound))
        return bool(jax.device_get(invalid))

    if has_out_of_range(parent, n_residue):
        raise ValueError("parent_residue_idx contains an out-of-range residue index")
    if has_out_of_range(
        role,
        max(STRUCTURAL_TOKEN_ROLES.values()) + 1,
    ):
        raise ValueError("subtoken_role_id contains an unsupported structural role")
    if has_out_of_range(
        jnp.asarray(features["atom_to_structural_token_idx"]),
        n_structural,
    ):
        raise ValueError(
            "atom_to_structural_token_idx contains an out-of-range token index"
        )
    residue_atom_map = jnp.asarray(features["atom_to_token_idx"])
    if has_out_of_range(residue_atom_map, n_residue):
        raise ValueError("atom_to_token_idx contains an out-of-range residue index")

    if require_residue_confidence_mask and check_values:
        representative_mask = jnp.asarray(features["distogram_rep_atom_mask"])
        representative_counts = (
            jnp.zeros(
                (n_residue,),
                dtype=jnp.int32,
            )
            .at[residue_atom_map]
            .add(representative_mask.astype(bool).astype(jnp.int32))
        )
        valid_counts = bool(jax.device_get(jnp.all(representative_counts == 1)))
        if not valid_counts:
            raise ValueError(
                "distogram_rep_atom_mask must select exactly one representative "
                "atom per residue token"
            )
    return n_residue, n_structural, n_atom


def prepare_structural_features(
    residue_features: Mapping[str, Any],
    structural_pair_features: Mapping[str, jnp.ndarray],
) -> dict[str, Any]:
    """Replace residue aliases with OpenDDE structural-token metadata."""

    structural = dict(residue_features)
    residue_aliases = (
        "token_index",
        "asym_id",
        "residue_index",
        "entity_id",
        "sym_id",
        "atom_to_token_idx",
        "atom_to_tokatom_idx",
        "has_frame",
        "frame_atom_index",
        "pae_rep_atom_mask",
        "distogram_rep_atom_mask",
    )
    for name in residue_aliases:
        structural[f"residue_level_{name}"] = residue_features[name]

    parent = jnp.asarray(residue_features["parent_residue_idx"], dtype=jnp.int32)
    structural["token_index"] = residue_features["structural_token_index"]
    structural["atom_to_token_idx"] = residue_features["atom_to_structural_token_idx"]
    structural["atom_to_tokatom_idx"] = residue_features[
        "atom_to_structural_tokatom_idx"
    ]
    for name in ("asym_id", "residue_index", "entity_id", "sym_id"):
        structural[name] = jnp.take(residue_features[name], parent, axis=-1)
    structural["has_frame"] = residue_features["structural_has_frame"]
    structural["frame_atom_index"] = residue_features["structural_frame_atom_index"]
    structural["pae_rep_atom_mask"] = residue_features["structural_pae_rep_atom_mask"]
    structural["distogram_rep_atom_mask"] = residue_features[
        "structural_distogram_rep_atom_mask"
    ]
    structural.update(structural_pair_features)
    structural["relp"] = relative_position_features(structural)

    residue_only = {
        "msa",
        "has_deletion",
        "deletion_value",
        "msa_mask",
        "profile",
        "deletion_mean",
        "token_bonds",
    }
    return {
        name: value
        for name, value in structural.items()
        if name not in residue_only and not name.startswith("template_")
    }


@_with_cueq_triangle_defaults
def opendde_infer_static(
    input_feature_dict: dict[str, Any],
    params: OpenDDEInferenceParams,
    noise_schedule: jnp.ndarray,
    *,
    key: jax.Array | None,
    n_sample: int,
    init_noise: jnp.ndarray | None = None,
    step_noises: Sequence[jnp.ndarray] | None = None,
    rotations: jnp.ndarray | None = None,
    translations: jnp.ndarray | None = None,
    n_cycle: int = 10,
    pair_mask: jnp.ndarray | None = None,
    input_atom_heads: int = 4,
    atom_encoder_heads: int = 4,
    token_heads: int = 16,
    atom_decoder_heads: int = 4,
    n_queries: int = 32,
    n_keys: int = 128,
    sigma_data: float = 16.0,
    use_pairformer_scan: bool = False,
    use_structural_refiner_scan: bool = False,
    use_confidence_scan: bool = False,
    use_diffusion_scan: bool = False,
    use_sampler_scan: bool = False,
    use_diffusion_efficient_fusion: bool = False,
    diffusion_attention_backend: str = "xla_jit",
    trunk_single_attention_backend: str = "xla_jit",
    trunk_triangle_attention_backend: str | None = None,
    structural_single_attention_backend: str = "xla_jit",
    structural_triangle_attention_backend: str | None = None,
    # `None`, like the trunk and structural branches above, so all three resolve
    # through the one shared rule. It read "xla_jit" and so was the one place an
    # env-var or default change could not reach -- on a stack that runs per
    # sample over N^2, which is not a small corner to leave behind.
    confidence_triangle_attention_backend: str | None = None,
    use_confidence_embedding: bool = True,
    run_confidence: bool = True,
    triangle_mul_chunk_size: int | None = None,
    triangle_att_q_chunk_size: int | None = None,
    single_att_q_chunk_size: int | None = None,
    token_q_chunk_size: int | None = None,
    gamma0: float = 0.8,
    gamma_min: float = 1.0,
    noise_scale_lambda: float = 1.003,
    step_scale_eta: float = 1.5,
    cycle_msa_features: tuple[dict[str, jnp.ndarray], ...] | None = None,
    validate_feature_values: bool = True,
    return_representations: bool = False,
    trunk_dtype: jnp.dtype | None = None,
) -> dict[str, jnp.ndarray]:
    """Run OpenDDE from already-featurized, unbatched static inputs.

    The residue trunk remains the source of distogram and confidence logits,
    while the expanded/refined structural-token branch exclusively drives
    diffusion. Exact OpenDDE postprocessing is intentionally outside this raw
    model function.
    """

    n_residue_token, n_structural_token, n_atom = _validate_static_structural_features(
        input_feature_dict,
        require_residue_confidence_mask=run_confidence,
        check_values=validate_feature_values,
    )
    if n_sample < 1:
        raise ValueError("n_sample must be at least one")
    if n_cycle < 1:
        raise ValueError("n_cycle must be at least one")
    if pair_mask is not None and tuple(pair_mask.shape) != (
        n_residue_token,
        n_residue_token,
    ):
        raise ValueError(
            "pair_mask expected shape "
            f"{(n_residue_token, n_residue_token)}, got {tuple(pair_mask.shape)}"
        )
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
    s_inputs_residue = input_feature_embedder(
        trunk_features,
        params.input_embedder,
        n_token=n_residue_token,
        n_heads=input_atom_heads,
        n_queries=n_queries,
        n_keys=n_keys,
        use_scan=use_diffusion_scan,
    )
    s_inputs_residue, s_residue, z_residue = pairformer_output_from_s_inputs(
        trunk_features,
        s_inputs_residue,
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
        # OpenDDE's MSA block runs the MSA stack before the communication, the
        # reverse of Protenix's, whose trunk function this shares. Its own
        # source says why: "Boltz updates MSA first, then writes the refreshed
        # MSA state back to z". Both orders preserve every shape, so taking
        # Protenix's here cost a pair representation that correlated 0.47 with
        # upstream's while every score still matched to within 1%.
        msa_stack_first=True,
    )

    (
        s_inputs_structural,
        s_structural,
        z_structural,
        structural_pair_features,
    ) = structural_token_expand(
        trunk_features,
        s_inputs_residue,
        s_residue,
        z_residue,
        params.structural_expander,
    )
    structural_features = prepare_structural_features(
        trunk_features,
        structural_pair_features,
    )
    structural_pair_bias = structural_features.get("structural_pair_attn_bias")
    expected_pair_shape = (n_structural_token, n_structural_token)
    if structural_pair_bias is None or tuple(structural_pair_bias.shape) != (
        expected_pair_shape
    ):
        actual = (
            None if structural_pair_bias is None else tuple(structural_pair_bias.shape)
        )
        raise ValueError(
            "structural_pair_attn_bias expected shape "
            f"{expected_pair_shape}, got {actual}"
        )
    expected_relp_shape = (*expected_pair_shape, 139)
    if tuple(structural_features["relp"].shape) != expected_relp_shape:
        raise ValueError(
            "structural relp expected shape "
            f"{expected_relp_shape}, got {tuple(structural_features['relp'].shape)}"
        )
    s_structural, z_structural = structural_refiner_stack(
        s_structural,
        z_structural,
        None,
        structural_pair_bias,
        params.structural_refiner,
        use_scan=use_structural_refiner_scan,
        triangle_mul_chunk_size=triangle_mul_chunk_size,
        triangle_att_q_chunk_size=triangle_att_q_chunk_size,
        single_att_q_chunk_size=single_att_q_chunk_size,
        single_attention_backend=structural_single_attention_backend,
        triangle_attention_backend=structural_triangle_attention_backend,
    )

    def as_float32(value: jnp.ndarray) -> jnp.ndarray:
        return value if value.dtype == jnp.float32 else value.astype(jnp.float32)

    diffusion_s_inputs = as_float32(s_inputs_structural)
    diffusion_s_trunk = as_float32(s_structural)
    diffusion_z_trunk = as_float32(z_structural)
    pair_z = diffusion_conditioning_prepare_cache(
        structural_features["relp"],
        diffusion_z_trunk,
        params.diffusion.conditioning,
    )
    p_lm, c_l = atom_attention_encoder_prepare_diffusion_cache(
        structural_features["atom_to_token_idx"],
        structural_features["ref_pos"],
        structural_features["ref_charge"],
        structural_features["ref_mask"],
        structural_features["ref_element"],
        structural_features["ref_atom_name_chars"],
        structural_features["d_lm"],
        structural_features["v_lm"],
        structural_features["pad_info"],
        z=jnp.expand_dims(pair_z, axis=-4),
        params=params.diffusion.atom_encoder,
        n_queries=n_queries,
        n_keys=n_keys,
    )

    def denoise_fn(
        x_noisy: jnp.ndarray,
        t_hat_noise_level: jnp.ndarray,
    ) -> jnp.ndarray:
        return diffusion_module_forward(
            structural_features["atom_to_token_idx"],
            structural_features["ref_pos"],
            structural_features["ref_charge"],
            structural_features["ref_mask"],
            structural_features["ref_atom_name_chars"],
            structural_features["ref_element"],
            structural_features["d_lm"],
            structural_features["v_lm"],
            structural_features["pad_info"],
            x_noisy=x_noisy,
            t_hat_noise_level=t_hat_noise_level,
            relp_feature=structural_features["relp"],
            s_inputs=diffusion_s_inputs,
            s_trunk=diffusion_s_trunk,
            z_trunk=diffusion_z_trunk,
            params=params.diffusion,
            pair_z=pair_z,
            p_lm=p_lm,
            c_l=c_l,
            extra_attn_bias=structural_pair_bias,
            n_token=n_structural_token,
            atom_encoder_heads=atom_encoder_heads,
            token_heads=token_heads,
            atom_decoder_heads=atom_decoder_heads,
            n_queries=n_queries,
            n_keys=n_keys,
            sigma_data=sigma_data,
            use_scan=use_diffusion_scan,
            use_efficient_fusion=use_diffusion_efficient_fusion,
            token_q_chunk_size=token_q_chunk_size,
            attention_backend=diffusion_attention_backend,
        )

    coordinates = sample_diffusion(
        denoise_fn,
        noise_schedule,
        n_sample=n_sample,
        n_atom=n_atom,
        key=key,
        init_noise=init_noise,
        step_noises=step_noises,
        rotations=rotations,
        translations=translations,
        batch_shape=(),
        gamma0=gamma0,
        gamma_min=gamma_min,
        noise_scale_lambda=noise_scale_lambda,
        step_scale_eta=step_scale_eta,
        use_scan=use_sampler_scan,
    )

    head_s_inputs = as_float32(s_inputs_residue)
    head_s_trunk = as_float32(s_residue)
    head_z_trunk = as_float32(z_residue)
    output = {
        "coordinate": coordinates,
        "distogram_logits": distogram_head(
            head_z_trunk,
            params.distogram,
        ),
    }
    if return_representations:
        # Off by default because these are the largest buffers the program can
        # hand back and nothing downstream reads them: the structural pair
        # representation alone is [946, 946, 384] float32 -- 1,311 MiB on a
        # 488-residue job, most of the 1,869 MiB output. Returning it also
        # keeps it live to the end of the program, so the refiner's working
        # buffer cannot be reused by anything after it. Scores and structures
        # need only `coordinate`, `distogram_logits`, and the confidence heads.
        output.update(
            s_inputs=s_inputs_residue,
            s_trunk=s_residue,
            z_trunk=z_residue,
            structural_s_inputs=s_inputs_structural,
            structural_s_trunk=s_structural,
            structural_z_trunk=z_structural,
        )
    if run_confidence:
        output.update(
            confidence_head(
                input_feature_dict,
                head_s_inputs,
                head_s_trunk,
                head_z_trunk,
                None,
                coordinates,
                params.confidence,
                use_embedding=use_confidence_embedding,
                use_scan=use_confidence_scan,
                triangle_mul_chunk_size=triangle_mul_chunk_size,
                triangle_att_q_chunk_size=triangle_att_q_chunk_size,
                single_att_q_chunk_size=single_att_q_chunk_size,
                triangle_attention_backend=confidence_triangle_attention_backend,
            )
        )
    return output


# Every keyword of :func:`opendde_infer_static` that shapes the program rather
# than feeding it: loop counts, head counts, chunk sizes, backend names, and the
# sampler's scalar constants. They are static so XLA sees one concrete graph.
GRAPH_STATIC_ARGNAMES = (
    "atom_decoder_heads",
    "atom_encoder_heads",
    "confidence_triangle_attention_backend",
    "diffusion_attention_backend",
    "gamma0",
    "gamma_min",
    "input_atom_heads",
    "n_cycle",
    "n_keys",
    "n_queries",
    "n_sample",
    "noise_scale_lambda",
    "return_representations",
    "run_confidence",
    "sigma_data",
    "single_att_q_chunk_size",
    "step_scale_eta",
    "structural_single_attention_backend",
    "structural_triangle_attention_backend",
    "token_heads",
    "token_q_chunk_size",
    "trunk_dtype",
    "triangle_att_q_chunk_size",
    "triangle_mul_chunk_size",
    "trunk_single_attention_backend",
    "trunk_triangle_attention_backend",
    "use_confidence_scan",
    "use_diffusion_efficient_fusion",
    "use_diffusion_scan",
    "use_pairformer_scan",
    "use_sampler_scan",
    "use_structural_refiner_scan",
    "validate_feature_values",
)

@partial(
    jax.jit,
    static_argnames=(*GRAPH_STATIC_ARGNAMES, "params_treedef", "params_flags"),
)
def _compiled_opendde_infer(
    input_feature_dict: dict[str, Any],
    param_arrays: tuple,
    noise_schedule: jnp.ndarray,
    *,
    params_treedef: Any,
    params_flags: tuple[tuple[int, bool], ...],
    **kwargs: Any,
) -> dict[str, jnp.ndarray]:
    return opendde_infer_static(
        input_feature_dict,
        merge_static_flags(param_arrays, params_treedef, params_flags),
        noise_schedule,
        validate_feature_values=False,
        **kwargs,
    )


def opendde_infer_compiled(
    input_feature_dict: dict[str, Any],
    params: OpenDDEInferenceParams,
    noise_schedule: jnp.ndarray,
    **kwargs: Any,
) -> dict[str, jnp.ndarray]:
    """Run OpenDDE as one compiled program instead of op by op.

    :func:`opendde_infer_static` is written eagerly: it calls individually
    jitted primitives, so a single prediction dispatches — and, without a
    persistent cache, compiles — well over a thousand small programs. Measured
    on a 35-residue job that was 1,824 compilations and about 21 s of dispatch
    on a warm cache, against 3.5 s for the torch original.

    Tracing the whole function once gives XLA the entire graph, so it fuses
    across the module boundaries the eager form hides and the per-call overhead
    collapses to one dispatch. This is the same computation: only how much of it
    XLA sees at once changes.
    """
    # Index-range checks read array *values*, which become tracers inside
    # `jit`. They run here on the concrete features, so the traced graph keeps
    # the same guarantees without a Python branch on a tracer.
    _validate_static_structural_features(
        input_feature_dict,
        require_residue_confidence_mask=kwargs.get("run_confidence", True),
    )
    param_arrays, treedef, flags = split_static_flags(params)
    return _compiled_opendde_infer(
        traceable_features(input_feature_dict),
        param_arrays,
        noise_schedule,
        params_treedef=treedef,
        params_flags=flags,
        **kwargs,
    )
