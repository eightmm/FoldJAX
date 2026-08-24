"""Top-level infer-only Protenix JAX orchestration."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import partial
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from foldjax.models import _capture
from foldjax.models._cp import (
    context_parallel,
    replicate_tree,
    shard_pair_rows,
)
from foldjax.models._cp import (
    cp_layout as _active_cp_layout,
)
from foldjax.models._cp import (
    cp_shards as _active_cp_shards,
)
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
    can_compact_confidence_distance_embedding,
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


_PADDED_MODEL_FEATURES = frozenset(
    {
        "atom_padding_mask",
        "atom_to_tokatom_idx",
        "atom_to_token_idx",
        "asym_id",
        "deletion_mean",
        "deletion_value",
        "distogram_rep_atom_mask",
        "d_lm",
        "entity_id",
        "esm_token_embedding",
        "has_deletion",
        "has_frame",
        "mol_id",
        "msa",
        "msa_mask",
        "pad_info",
        "profile",
        "ref_atom_name_chars",
        "ref_charge",
        "ref_element",
        "ref_mask",
        "ref_pos",
        "relp",
        "residue_index",
        "restype",
        "sym_id",
        "template_aatype",
        "template_backbone_frame_mask",
        "template_distogram",
        "template_multiplicity",
        "template_pseudo_beta_mask",
        "template_unit_vector",
        "token_bonds",
        "token_index",
        "token_padding_mask",
        "v_lm",
    }
)


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
    step_noises: jnp.ndarray | Sequence[jnp.ndarray] | None = None,
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
    compact_confidence_distance_bins: bool = False,
    run_confidence: bool = True,
    run_confidence_scores: bool = True,
    #: Whether the raw representations and full-bin logits are program outputs.
    #: XLA entry outputs stay resident for the whole execution, alongside the
    #: temp arena -- and at 3,012 tokens the PAE and PDE logits are
    #: f32[n_sample, N, N, 64] = 10.8 GiB *each*, returned right next to the
    #: summaries computed from them in-graph. A caller writing cif + confidence
    #: JSON reads none of it; only the raw-npz path does. Defaults stay True so
    #: the library API keeps its shape; the CLI passes what its output format
    #: actually consumes.
    return_trunk: bool = True,
    stop_after_trunk: bool = False,
    capture_names: tuple[str, ...] = (),
    return_confidence_logits: bool = True,
    #: Score each sample's logits inside the per-sample confidence loop instead
    #: of stacking [n_sample, N, N, 64] first. Bit-identical outputs; the False
    #: path exists as the reference the equivalence test compares against.
    confidence_sample_sequential: bool = True,
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
    trunk_dtype: jnp.dtype | None = None,
    cycle_msa_features: tuple[dict[str, jnp.ndarray], ...] | None = None,
    guidance_config: Mapping[str, Any] | None = None,
    guidance_features: Mapping[str, Any] | None = None,
    n_chain: int | None = None,
    #: Context-parallel shard count; same contract as OpenDDE's. More than one
    #: requires an active `foldjax.models._cp.context_parallel` mesh of the
    #: same size, and the value is a static argument so a mesh change is a
    #: retrace. `protenix_infer_compiled` manages both together.
    cp_shards: int = 1,
    #: Mesh layout, same contract as OpenDDE's: ``"2d"`` is Fold-CP's square
    #: grid (rows and columns split), ``"1d"`` splits rows only.
    cp_layout: str = "1d",
) -> dict[str, jnp.ndarray]:
    """Run the currently ported static-feature Protenix inference path.

    ``n_chain`` is the number of chains in the job. The confidence scores need
    it as a Python integer -- it sizes per-chain loops -- and by default they
    read it back off ``asym_id``, which is a tracer inside ``jit``. The
    consolidated graph resolves it on the host and passes it here.
    """

    if cp_layout != (_active_cp_layout() or "1d"):
        raise RuntimeError(
            f"cp_layout={cp_layout!r} but the active mesh is "
            f"{_active_cp_layout()!r}"
        )
    if cp_shards != _active_cp_shards():
        raise RuntimeError(
            f"cp_shards={cp_shards} but the active context-parallel mesh has "
            f"{_active_cp_shards()} shard(s); run through "
            "protenix_infer_compiled or activate context_parallel() yourself"
        )
    # Same rule as OpenDDE: no backend override for context parallelism.
    # Attention keeps its configured kernel (run per-shard inside
    # `shard_map`); triangle multiplication falls back to the partitionable
    # XLA einsum on its own when a mesh is active.
    n_token = int(input_feature_dict["restype"].shape[-2])
    token_padding_mask = input_feature_dict.get("token_padding_mask")
    atom_padding_mask = input_feature_dict.get("atom_padding_mask")
    if token_padding_mask is not None:
        token_valid = jnp.asarray(token_padding_mask).astype(bool)
        padding_pair_mask = token_valid[..., :, None] & token_valid[..., None, :]
        pair_mask = (
            padding_pair_mask
            if pair_mask is None
            else jnp.asarray(pair_mask).astype(bool) & padding_pair_mask
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
        opm_chunk_size=opm_chunk_size,
        single_attention_backend=trunk_single_attention_backend,
        triangle_attention_backend=trunk_triangle_attention_backend,
        cycle_msa_features=cycle_msa_features,
    )
    s_inputs = _capture.capture("single_inputs", s_inputs)
    s_trunk = _capture.capture("single", s_trunk)
    z_trunk = _capture.capture("pair", z_trunk)

    if stop_after_trunk:
        # The representations exist; the sampler and the confidence heads
        # are the rest of the cost, and a caller who wants embeddings
        # should not pay for a structure they will discard.
        return dict(_capture.collected())

    diffusion_s_inputs = s_inputs.astype(jnp.float32)
    diffusion_s_trunk = s_trunk.astype(jnp.float32)
    diffusion_z_trunk = z_trunk.astype(jnp.float32)
    # `relp` is a host-supplied [N, N, 139] pair feature here (the trunk
    # shards its own copy); pinned so the conditioning projection is
    # shard-local and `pair_z` is born sharded.
    relp = shard_pair_rows(jnp.asarray(input_feature_dict["relp"]))
    pair_z = shard_pair_rows(
        diffusion_conditioning_prepare_cache(
            relp,
            diffusion_z_trunk,
            params.diffusion.conditioning,
        )
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

    def sample(key, init_noise, step_noises):
        return sample_diffusion_with_module(
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

    coordinates = sample(key, init_noise, step_noises)
    distogram_logits = distogram_head(diffusion_z_trunk, params.distogram)
    output = {"coordinate": coordinates}
    if return_trunk:
        # The historical names, kept because the native writers and the
        # parity harnesses read them; the shared vocabulary arrives
        # through the taps below.
        output["s_inputs"] = s_inputs
        output["s_trunk"] = s_trunk
        output["z_trunk"] = z_trunk
    output.update(_capture.collected())
    if return_confidence_logits:
        output["distogram_logits"] = distogram_logits
    if run_confidence:
        if run_confidence_scores and n_chain is None:
            # The per-chain loops need a Python int. Resolve it once, outside
            # the per-sample loop: inside `lax.map` even a constant `asym_id`
            # reaches the fallback as a tracer. On the consolidated-graph path
            # the caller has already passed it (see the docstring).
            asym_id = input_feature_dict.get("asym_id")
            if asym_id is not None and not isinstance(asym_id, jax.core.Tracer):
                asym_id = jnp.asarray(asym_id)
                if token_padding_mask is not None:
                    asym_id = asym_id[jnp.asarray(token_padding_mask).astype(bool)]
                n_chain = int(jnp.max(asym_id)) + 1

        def _confidence(sample_coordinates):
            """Head and summaries for one sample's coordinates.

            The head already loops samples one at a time -- what kept three
            f32[n_sample, N, N, 64] stacks (32 GiB at 3,012 tokens) live at
            once was stacking every sample's logits and only then reducing
            them. Scoring inside the per-sample loop reduces each sample's
            logits while they are [1, N, N, 64], the same shape of fix as
            Boltz-2's `confidence_sequentially` and AF3's shard_size=1.
            """
            logits = confidence_head(
                input_feature_dict,
                diffusion_s_inputs,
                diffusion_s_trunk,
                diffusion_z_trunk,
                pair_mask,
                sample_coordinates,
                params.confidence,
                use_embedding=use_confidence_embedding,
                compact_distance_bins=compact_confidence_distance_bins,
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
            piece = dict(logits) if return_confidence_logits else {}
            if run_confidence_scores:
                piece.update(
                    confidence_scores_from_logits(
                        plddt_logits=logits["plddt"],
                        pae_logits=logits["pae"],
                        pde_logits=logits["pde"],
                        distogram_logits=distogram_logits,
                        token_has_frame=input_feature_dict.get("has_frame"),
                        token_asym_id=input_feature_dict.get("asym_id"),
                        atom_to_token_idx=input_feature_dict.get(
                            "atom_to_token_idx"
                        ),
                        atom_coordinate=sample_coordinates,
                        elements_one_hot=input_feature_dict.get("ref_element"),
                        mol_id=input_feature_dict.get("mol_id"),
                        token_is_ligand=input_feature_dict.get(
                            "token_is_ligand",
                            jnp.argmax(input_feature_dict["restype"], axis=-1)
                            == 20,
                        ),
                        num_recycles=n_cycle,
                        n_chain=n_chain,
                        token_mask=token_padding_mask,
                        atom_mask=atom_padding_mask,
                    )
                )
            return piece

        n_conf_sample = int(coordinates.shape[-3])
        if confidence_sample_sequential and n_conf_sample > 1:
            conf = jax.lax.map(
                _confidence,
                jnp.moveaxis(coordinates, -3, 0)[..., None, :, :],
            )
            # Every leaf gains a leading map axis over its single-sample shape.
            # A leaf whose next axis is the size-1 sample axis is collapsed
            # back onto it; a sample-independent leaf (contact_probs is
            # [N, N] whichever sample scored it) keeps one copy, as the
            # batched call produced.
            conf = {
                name: (
                    jnp.squeeze(value, axis=1)
                    if value.ndim > 1 and value.shape[1] == 1
                    else value[0]
                )
                for name, value in conf.items()
            }
        else:
            conf = _confidence(coordinates)
        output.update(conf)
    return output


# Static arguments of the consolidated graph: everything that changes the
# program rather than the data. Anything not listed here is traced.
GRAPH_STATIC_ARGNAMES = (
    "atom_decoder_heads",
    "atom_encoder_heads",
    "centre_each_step",
    "confidence_triangle_attention_backend",
    "compact_confidence_distance_bins",
    "cp_layout",
    "cp_shards",
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
    "confidence_sample_sequential",
    "return_trunk",
    "stop_after_trunk",
    "capture_names",
    "return_confidence_logits",
    "sigma_data",
    "opm_chunk_size",
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
    *,
    padded_generated_schema: bool = False,
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
    asym_id = jnp.asarray(input_feature_dict["asym_id"])
    token_padding_mask = input_feature_dict.get("token_padding_mask")
    if token_padding_mask is not None:
        asym_id = asym_id[jnp.asarray(token_padding_mask).astype(bool)]
    kwargs.setdefault("n_chain", int(jnp.max(asym_id)) + 1)
    compact_requested = bool(kwargs.get("compact_confidence_distance_bins", True))
    distance_params = getattr(
        getattr(params, "confidence", None), "distance_embedding", None
    )
    kwargs["compact_confidence_distance_bins"] = (
        compact_requested
        and can_compact_confidence_distance_embedding(distance_params)
    )
    param_arrays, treedef, flags = split_static_flags(params)
    model_features = traceable_features(input_feature_dict)
    if padded_generated_schema:
        # Generated features also carry writer metadata and variable-length
        # bond lists. They are not model inputs, but handing them to ``jit``
        # would still put their shapes in the Python/XLA cache key and defeat a
        # token/atom/MSA profile. The padding helper has already rejected
        # constraint/private schemas, so this allow-list is complete for the
        # supported generated-JSON contract. This is keyed by explicit
        # provenance rather than by ``token_padding_mask``: static archives
        # may legitimately contain that mask alongside constraint/private
        # model inputs, which must never be silently discarded.
        model_features = {
            name: value
            for name, value in model_features.items()
            if name in _PADDED_MODEL_FEATURES
        }
    cp = int(kwargs.pop("cp_shards", 1))
    layout = str(kwargs.pop("cp_layout", "auto"))
    if layout == "auto":
        # "auto" stays on the 1-D layout for now. The 2-D grid is the better
        # design and is verified on CPU meshes at 2x2 and 3x3, but every
        # measurement published for this feature -- the size ladders, the
        # per-device peaks -- was taken on the 1-D layout, and a default that
        # silently changes the program would make those numbers describe a
        # configuration nobody can reproduce. Ask for "2d" explicitly; this
        # flips once the square grid has its own GPU parity and memory
        # evidence.
        layout = "1d"
    # The capture set has to be live while the program is *traced*: a tap
    # records a tracer of the graph being built, not a value from a run.
    capture_names = tuple(kwargs.get("capture_names", ()))
    with _capture.capturing(capture_names), context_parallel(cp, layout=layout):
        if cp > 1:
            # A checkpoint committed to one device fails the multi-device
            # jit's device-assignment check; everything token-linear is
            # replicated onto the mesh, and the graph's own constraints
            # shard the pair-shaped state from its first materialization.
            param_arrays = replicate_tree(param_arrays)
            model_features = replicate_tree(model_features)
            noise_schedule = replicate_tree(noise_schedule)
            kwargs = replicate_tree(kwargs)
        return _compiled_protenix_infer(
            model_features,
            param_arrays,
            noise_schedule,
            params_treedef=treedef,
            params_flags=flags,
            cp_shards=cp,
            cp_layout=layout,
            **kwargs,
        )
