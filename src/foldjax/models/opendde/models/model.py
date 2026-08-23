"""OpenDDE dual-branch inference parameters and feature preparation."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from functools import partial, wraps
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

    **Those are float32 numbers**, which is the CLI default and faithful to
    upstream's own float32 storage (`inference_defaults.py:24`). Everything
    below is measured in bfloat16 unless it says otherwise; the two regimes
    have different answers and reading one for the other is how the paragraph
    after this one came to be wrong.

    Upstream resolves `auto` to cuequivariance for both kernels
    (config/model_base.py:31-32) and OpenDDE's `c_hidden == c_z == 384` clears
    the kernel's guard, so fused is what upstream actually runs. Protenix, which
    shares this triangle module, went 254.5 -> 167.1 s on the same switch --
    **Protenix's figure, not OpenDDE's**, and not transferable: OpenDDE's
    structural pair space is ~1.9x residues, a different shape with a different
    answer.

    `setdefault` with a restore: an explicit environment still wins, and
    nothing leaks into the process for the next model to inherit.

    This used to add "which is how a job too large for the fused arena asks for
    the blocked path". **That is false and was pointing users into a second
    OOM.** Measured at 1,531 tokens in float32, neither path runs: fused asks
    for 93.40 GiB and dies, blocked asks for 80.72 GiB and dies. The blocked
    path moves the wall; it does not rescue the job. In float32 the backend
    choice cannot save an oversized job and no size threshold makes it possible
    -- the lever there is bfloat16, where both paths run at 1,531 (fused 53,068
    MiB, blocked 46,596).

    So any adaptive policy here keys on **dtype first**, then size. Blocked is
    now the cheaper arm at every size measured -- about 24% by the identity in
    `protenix/models/triangle/triangle.py`, once its destination stopped being
    float32 -- but the default stays fused until a clocked A/B settles the time
    question, which single warm samples on this card cannot.
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
    optional_shapes = {
        "token_padding_mask": (n_residue,),
        "atom_padding_mask": (n_atom,),
        "structural_token_padding_mask": (n_structural,),
    }
    expected_shapes.update(
        {
            name: shape
            for name, shape in optional_shapes.items()
            if name in features
        }
    )
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
        token_mask = jnp.asarray(
            features.get(
                "token_padding_mask",
                jnp.ones((n_residue,), dtype=bool),
            )
        ).astype(bool)
        atom_mask = jnp.asarray(
            features.get(
                "atom_padding_mask",
                jnp.ones((n_atom,), dtype=bool),
            )
        ).astype(bool)
        representative_mask = (
            jnp.asarray(features["distogram_rep_atom_mask"]).astype(bool)
            & atom_mask
        )
        representative_counts = (
            jnp.zeros(
                (n_residue,),
                dtype=jnp.int32,
            )
            .at[residue_atom_map]
            .add(representative_mask.astype(jnp.int32))
        )
        valid_counts = bool(
            jax.device_get(
                jnp.all(representative_counts == token_mask.astype(jnp.int32))
            )
        )
        if not valid_counts:
            raise ValueError(
                "distogram_rep_atom_mask must select exactly one representative "
                "atom per residue token for every real token and none for "
                "padded tokens"
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
    if "token_padding_mask" in residue_features:
        structural["residue_level_token_padding_mask"] = residue_features[
            "token_padding_mask"
        ]

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
    if "structural_token_padding_mask" in residue_features:
        structural["token_padding_mask"] = residue_features[
            "structural_token_padding_mask"
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
    #: Compute the released confidence summaries inside the graph. Off, the
    #: caller must reduce the raw logits on the host -- which forces the
    #: program to return plddt/pae/pde at [n_sample, N, N, bins] full width:
    #: 21.6 GiB of entry outputs at 3,012 tokens, resident next to the temp
    #: arena. Same finding and same shape of fix as Protenix and Boltz-2.
    run_confidence_scores: bool = False,
    #: With the summaries computed in-graph nothing downstream reads the raw
    #: logits unless a raw dump was asked for; False drops them from the
    #: program outputs.
    return_confidence_logits: bool = True,
    #: Chains in the job, resolved on the host: the per-chain score loops need
    #: a Python int, and reading it off `asym_id` inside jit is a tracer.
    #: Same contract as Protenix's `n_chain`.
    n_chain: int | None = None,
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
    stop_after_trunk: bool = False,
    capture_names: tuple[str, ...] = (),
    trunk_dtype: jnp.dtype | None = None,
    #: Context-parallel shard count. More than one requires an active
    #: `foldjax.models._cp.context_parallel` mesh of the same size; the value
    #: is also a static argument so a mesh change is a retrace, never a stale
    #: cache hit. `opendde_infer_compiled` manages both together.
    cp_shards: int = 1,
    #: Sharding layout of the context-parallel mesh: ``"1d"`` splits pair rows
    #: only, ``"2d"`` splits rows and columns on Fold-CP's square grid.
    #: Carried as a static argument so a layout change is a retrace.
    cp_layout: str = "1d",
) -> dict[str, jnp.ndarray]:
    """Run OpenDDE from already-featurized, unbatched static inputs.

    The residue trunk remains the source of distogram and confidence logits,
    while the expanded/refined structural-token branch exclusively drives
    diffusion. Exact OpenDDE postprocessing is intentionally outside this raw
    model function.
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
            "opendde_infer_compiled or activate context_parallel() yourself"
        )
    # Triangle backends are NOT overridden for context parallelism. Attention
    # keeps whatever kernel is configured -- `_triangle_attention_cp` runs it
    # per-shard inside `shard_map`, where each device holds whole rows -- and
    # triangle multiplication falls back to the partitionable XLA einsum on
    # its own inside `triangle_multiplication` when a mesh is active. Forcing
    # a value here also made PROTENIX_TRIANGLE_BACKEND inert, so an A/B that
    # set it compared a backend against itself.
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
    residue_token_mask = input_feature_dict.get("token_padding_mask")
    atom_mask = input_feature_dict.get("atom_padding_mask")
    structural_token_mask = input_feature_dict.get(
        "structural_token_padding_mask"
    )
    residue_padding_pair_mask = None
    if residue_token_mask is not None:
        residue_token_mask = jnp.asarray(residue_token_mask).astype(bool)
        residue_padding_pair_mask = (
            residue_token_mask[:, None] & residue_token_mask[None, :]
        )
    structural_pair_mask = None
    if structural_token_mask is not None:
        structural_token_mask = jnp.asarray(structural_token_mask).astype(bool)
        structural_pair_mask = (
            structural_token_mask[:, None] & structural_token_mask[None, :]
        )
    trunk_features = input_feature_dict
    if pair_mask is None:
        trunk_pair_mask = residue_padding_pair_mask
    elif residue_padding_pair_mask is None:
        trunk_pair_mask = pair_mask
    else:
        trunk_pair_mask = (
            jnp.asarray(pair_mask).astype(bool) & residue_padding_pair_mask
        )
    if trunk_dtype is not None:
        trunk_features = jax.tree.map(
            lambda value: (
                value.astype(trunk_dtype)
                if hasattr(value, "dtype") and jnp.issubdtype(value.dtype, jnp.floating)
                else value
            ),
            input_feature_dict,
        )
        if trunk_pair_mask is not None:
            trunk_pair_mask = trunk_pair_mask.astype(trunk_dtype)
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
    if structural_token_mask is not None:
        s_inputs_structural = s_inputs_structural * structural_token_mask.astype(
            s_inputs_structural.dtype
        )[..., None]
    structural_features = prepare_structural_features(
        trunk_features,
        structural_pair_features,
    )
    # The structural relp is [N_s, N_s, 139] and feeds the diffusion
    # conditioning's pair projection; born sharded, that projection stays
    # shard-local.
    structural_features["relp"] = shard_pair_rows(structural_features["relp"])
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
        structural_pair_mask,
        structural_pair_bias,
        params.structural_refiner,
        use_scan=use_structural_refiner_scan,
        triangle_mul_chunk_size=triangle_mul_chunk_size,
        triangle_att_q_chunk_size=triangle_att_q_chunk_size,
        single_att_q_chunk_size=single_att_q_chunk_size,
        single_attention_backend=structural_single_attention_backend,
        triangle_attention_backend=structural_triangle_attention_backend,
    )

    s_inputs_residue = _capture.capture("single_inputs", s_inputs_residue)
    s_residue = _capture.capture("single", s_residue)
    z_residue = _capture.capture("pair", z_residue)
    s_inputs_structural = _capture.capture(
        "structural_single_inputs", s_inputs_structural
    )
    s_structural = _capture.capture("structural_single", s_structural)
    z_structural = _capture.capture("structural_pair", z_structural)

    if stop_after_trunk:
        # Everything a capture request can ask for on the trunk exists by
        # now, and the diffusion sampler plus the confidence heads are most
        # of the run's cost. Stopping here is the difference between an
        # embedding and a structure nobody asked for.
        return dict(_capture.collected())

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
    if structural_pair_mask is not None:
        pair_z = pair_z * structural_pair_mask.astype(pair_z.dtype)[..., None]
    pair_z = shard_pair_rows(pair_z)
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
            token_mask=structural_token_mask,
            atom_mask=atom_mask,
        )

    def sample(key, init_noise, step_noises, rotations, translations):
        return sample_diffusion(
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
            atom_mask=atom_mask,
        )

    coordinates = sample(key, init_noise, step_noises, rotations, translations)

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
        # The historical names, unchanged: parity harnesses and the native
        # writers read them. The shared vocabulary arrives separately, through
        # the taps, so neither spelling has to move for the other.
        #
        # Off by default because these are the largest buffers the program can
        # hand back and nothing downstream reads them: the structural pair
        # representation alone is [946, 946, 384] float32 -- 1,311 MiB on a
        # 488-residue job, most of the 1,869 MiB output. Returning it also
        # keeps it live to the end of the program, so the refiner's working
        # buffer cannot be reused by anything after it.
        output.update(
            s_inputs=s_inputs_residue,
            s_trunk=s_residue,
            z_trunk=z_residue,
            structural_s_inputs=s_inputs_structural,
            structural_s_trunk=s_structural,
            structural_z_trunk=z_structural,
        )
    output.update(_capture.collected())
    if run_confidence:
        output.update(
            confidence_head(
                input_feature_dict,
                head_s_inputs,
                head_s_trunk,
                head_z_trunk,
                # The explicit pair mask historically affects the residue
                # trunk only. Confidence needs the padding mask to exclude
                # dummy tokens, while retaining that default/custom-mask API.
                residue_padding_pair_mask,
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
        if run_confidence_scores:
            from foldjax.models.opendde.postprocess import (
                opendde_confidence_scores,
            )

            output.update(
                opendde_confidence_scores(
                    output,
                    input_feature_dict,
                    num_recycles=n_cycle,
                    n_chain=n_chain,
                    # numpy-based, untraceable; finished on the host by the
                    # postprocess passthrough.
                    include_shape_complementarity=False,
                )
            )
        if not return_confidence_logits:
            for name in ("plddt", "pae", "pde", "distogram_logits"):
                output.pop(name, None)
    return output


# Every keyword of :func:`opendde_infer_static` that shapes the program rather
# than feeding it: loop counts, head counts, chunk sizes, backend names, and the
# sampler's scalar constants. They are static so XLA sees one concrete graph.
GRAPH_STATIC_ARGNAMES = (
    "atom_decoder_heads",
    "atom_encoder_heads",
    "confidence_triangle_attention_backend",
    "cp_layout",
    "cp_shards",
    "run_confidence_scores",
    "n_chain",
    "return_confidence_logits",
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
    "stop_after_trunk",
    "capture_names",
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
    # The capture set has to be live while the program is *traced*, not
    # while it runs: a tap records a tracer of the graph being built. On a
    # cache hit nothing is traced and the compiled program already returns
    # what was asked for, so entering this is harmless either way.
    capture_names = tuple(kwargs.get("capture_names", ()))
    with _capture.capturing(capture_names), context_parallel(cp, layout=layout):
        if cp > 1:
            # A checkpoint committed to one device fails the multi-device
            # jit's device-assignment check; everything token-linear is
            # replicated onto the mesh, and the graph's own constraints
            # shard the pair-shaped state from its first materialization.
            param_arrays = replicate_tree(param_arrays)
            input_feature_dict = replicate_tree(input_feature_dict)
            noise_schedule = replicate_tree(noise_schedule)
            kwargs = replicate_tree(kwargs)
        return _compiled_opendde_infer(
            traceable_features(input_feature_dict),
            param_arrays,
            noise_schedule,
            params_treedef=treedef,
            params_flags=flags,
            cp_shards=cp,
            cp_layout=layout,
            **kwargs,
        )
