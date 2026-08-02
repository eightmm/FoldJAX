"""Public, Torch-free orchestration boundary for Chai-JAX inference.

This module deliberately separates preprocessing from model execution.  The
released preprocessing path is usable today for standard protein, RNA, and DNA
inputs.  A production model backend can consume :class:`PreparedInference` and
the fully mapped six-component bundle; until that backend is complete the
public entry point fails at the model-execution boundary instead of returning a
placeholder structure.
"""

from __future__ import annotations

import secrets
from collections.abc import Mapping
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import Any, Protocol

import jax
import jax.numpy as jnp
import numpy as np

from foldjax.models._stacking import prestack_layer_lists
from foldjax.models.chai.bridge.bundle_io import NativeBundle, load_native_bundle
from foldjax.models.chai.bridge.conformer_io import (
    ConformerData,
    load_native_conformers,
)
from foldjax.models.chai.data.chemistry import parse_covalent_bonds_csv
from foldjax.models.chai.data.collate import (
    block_atom_pair_mask,
    get_pad_sizes,
    qkv_indices_for_blocks,
)
from foldjax.models.chai.data.esm import (
    assemble_esm_context,
    generate_native_esm_embeddings_for_inputs,
    load_native_esm_embeddings,
)
from foldjax.models.chai.data.features import generate_features
from foldjax.models.chai.data.input import EntityType, Input, read_inputs
from foldjax.models.chai.data.msa import (
    FULL_DEPTH,
    MSAContext,
    load_msa_contexts,
    msa_subsample_indices_and_mask,
)
from foldjax.models.chai.data.restraints import parse_restraints_csv
from foldjax.models.chai.data.search import (
    MsaSearchPipeline,
    RemoteMMseqs2Backend,
)
from foldjax.models.chai.data.structure import StructureContext, tokenize_inputs
from foldjax.models.chai.data.templates import (
    TemplateContext,
    build_template_context_from_m8,
    load_native_template_context,
    materialize_server_template_hits,
)
from foldjax.models.chai.models.confidence import (
    TRIANGLE_QUERY_CHUNK_SIZE,
    ConfidenceBlockParams,
    ConfidenceHeadParams,
    TriangleAttentionParams,
    confidence_attention_pair_bias_projection,
    confidence_attention_pair_bias_with_bias,
    confidence_block_add_pair_updates,
    confidence_block_without_triangle_attention,
    confidence_head_forward,
    confidence_head_initialize,
    confidence_head_project,
    confidence_transition_output,
    confidence_transition_projection,
    confidence_triangle_multiplication_direction_tile,
    confidence_triangle_multiplication_output_tile,
    map_confidence_head,
    triangle_attention_finalize,
    triangle_attention_prepare,
    triangle_attention_query_slice,
)
from foldjax.models.chai.models.diffusion import (
    center_augmentation,
    chai_diffusion_gammas,
    chai_noise_schedule,
    edm_heun_step,
)
from foldjax.models.chai.models.diffusion_atom_decoder import atom_decoder_forward
from foldjax.models.chai.models.diffusion_atom_encoder import (
    diffusion_atom_encoder_with_aux,
)
from foldjax.models.chai.models.diffusion_conditioning import (
    diffusion_pair_conditioning_projection,
    diffusion_single_conditioning,
)
from foldjax.models.chai.models.diffusion_denoiser import (
    FullDiffusionDenoiserParams,
    full_diffusion_denoiser,
    map_full_diffusion_denoiser,
)
from foldjax.models.chai.models.diffusion_transformer import (
    _transition_update as diffusion_transformer_transition,
)
from foldjax.models.chai.models.diffusion_transformer import (
    diffusion_transformer_attention_with_bias,
    diffusion_transformer_pair_bias,
)
from foldjax.models.chai.models.feature_embedding import (
    embed_features,
    embed_msa,
    embed_non_msa_features,
)
from foldjax.models.chai.models.msa import (
    _msa_pair_block,
    msa_pair_weighted_averaging,
    msa_transition,
    outer_product_mean,
)
from foldjax.models.chai.models.pairformer import (
    _triangle_attention_backend,
    attention_pair_bias,
    fused_triangle_attention_bias,
    fused_triangle_attention_direction,
    fused_triangle_attention_direction_chunk,
    fused_triangle_attention_output,
    fused_triangle_multiplication_direction,
    fused_triangle_multiplication_direction_chunk,
    fused_triangle_multiplication_output,
    pairformer_transition,
    pairformer_transition_output,
    pairformer_transition_projection,
)
from foldjax.models.chai.models.primitives import (
    contiguous_segment_mean,
    layer_norm,
    linear,
    linear_bf16,
)
from foldjax.models.chai.models.template import template_embedding
from foldjax.models.chai.models.token_embedder import (
    TokenEmbedderParams,
    map_token_embedder,
    token_embedder_forward,
)
from foldjax.models.chai.models.trunk import (
    TrunkParams,
    _pairformer_stack,
    map_trunk,
    recycling_projection,
    trunk_forward,
)
from foldjax.models.chai.output import (
    StructureCandidates,
    TrunkPrediction,
    write_msa_coverage_pdf,
    write_prediction_outputs,
)
from foldjax.models.chai.ranking.frames import get_frames_and_mask
from foldjax.models.chai.ranking.rank import rank


class InferenceStageUnavailableError(NotImplementedError):
    """Raised when inference reaches a known, unfinished production stage."""


def _default_cache(*parts: str) -> Path:
    """Resolve a chai cache path, honouring the pre-FoldJAX location.

    These assets are exported once and are large (the ESM2 bundle is several
    GB), so a checkout that already populated ``~/.cache/chai_jax`` from the
    standalone package keeps using it instead of silently re-exporting.
    """
    current = Path.home().joinpath(".cache", "foldjax", "chai", *parts)
    if current.exists():
        return current
    legacy = Path.home().joinpath(".cache", "chai_jax", *parts)
    return legacy if legacy.exists() else current


DEFAULT_ESM_MODEL_PATH = _default_cache("models", "esm2_t36_3B")
_MSA_ROW_FEATURE_NAMES = (
    "IsPairedMSA",
    "MSADataSource",
    "MSADeletionValue",
    "MSAHasDeletion",
    "MSAOneHot",
)
_MODEL_PADDED_INPUT_KEYS = frozenset(
    {
        "atom_exists_mask",
        "atom_ref_mask",
        "atom_ref_space_uid",
        "atom_token_index",
        "atom_within_token_index",
        "block_atom_pair_kv_idces",
        "block_atom_pair_mask",
        "block_atom_pair_q_idces",
        "esm_embeddings",
        "msa_mask",
        "template_mask",
        "token_asym_id",
        "token_backbone_frame_index",
        "token_backbone_frame_mask",
        "token_centre_atom_index",
        "token_entity_type",
        "token_exists_mask",
        "token_ref_atom_index",
        "token_residue_index",
        "token_residue_type",
    }
)


def _random_seed() -> int:
    """Return a nonzero uint32 seed for one unseeded public inference call."""
    return secrets.randbelow(2**32 - 1) + 1


@dataclass(frozen=True)
class InferenceConfig:
    """Validated subset of the public Chai-1 inference configuration."""

    num_trunk_recycles: int = 3
    recycle_msa_subsample: int = 0
    max_msa_depth: int | None = None
    num_diffusion_timesteps: int = 200
    num_diffusion_samples: int = 5
    num_trunk_samples: int = 1
    seed: int | None = None
    use_esm_embeddings: bool = True
    esm_embeddings_path: Path | None = None
    esm_model_path: Path | None = DEFAULT_ESM_MODEL_PATH
    esm_cache_directory: Path = _default_cache("esm_embeddings")
    esm_attention_implementation: str = "xla"
    use_msa_server: bool = False
    msa_directory: Path | None = None
    msa_server_url: str = "https://api.colabfold.com"
    msa_server_version: str = "colabfold-public-api-v1"
    msa_cache_directory: Path | None = None
    use_templates_server: bool = False
    template_hits_path: Path | None = None
    template_cif_directory: Path | None = None
    template_cache_directory: Path = _default_cache("templates")
    kalign_executable: str | Path = "kalign"
    constraint_path: Path | None = None
    fasta_names_as_cif_chains: bool = False
    compilation_cache_dir: Path | None = None

    def validate(self) -> None:
        if self.num_trunk_recycles < 1:
            raise ValueError("num_trunk_recycles must be positive")
        if self.num_diffusion_timesteps < 2:
            raise ValueError("num_diffusion_timesteps must be at least 2")
        if self.num_diffusion_samples < 1:
            raise ValueError("num_diffusion_samples must be positive")
        if self.num_trunk_samples < 1:
            raise ValueError("num_trunk_samples must be positive")
        if self.max_msa_depth is not None and self.max_msa_depth < 1:
            raise ValueError("max_msa_depth must be positive")
        if self.use_msa_server and self.msa_directory is not None:
            raise ValueError("use_msa_server and msa_directory are mutually exclusive")
        if self.use_templates_server and self.template_hits_path is not None:
            raise ValueError(
                "use_templates_server and template_hits_path are mutually exclusive"
            )
        if self.use_templates_server and not self.use_msa_server:
            raise ValueError("use_templates_server requires use_msa_server=True")
        if self.template_cif_directory is not None and self.template_hits_path is None:
            raise ValueError("template_cif_directory requires template_hits_path")
        if (
            self.use_esm_embeddings
            and self.esm_embeddings_path is None
            and self.esm_model_path is None
        ):
            raise ValueError(
                "use_esm_embeddings requires esm_embeddings_path or esm_model_path"
            )
        if not self.use_esm_embeddings and self.esm_embeddings_path is not None:
            raise ValueError("esm_embeddings_path requires use_esm_embeddings=True")
        if self.esm_attention_implementation not in {"xla", "cudnn"}:
            raise ValueError("esm_attention_implementation must be 'xla' or 'cudnn'")
        if self.msa_cache_directory is not None and not self.use_msa_server:
            raise ValueError("msa_cache_directory requires use_msa_server=True")

    def configure_compilation_cache(self) -> None:
        """Enable JAX's persistent executable cache before model dispatch."""
        if self.compilation_cache_dir is None:
            return
        cache = self.compilation_cache_dir.expanduser().resolve()
        cache.mkdir(parents=True, exist_ok=True)
        jax.config.update("jax_compilation_cache_dir", str(cache))
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)


@dataclass(frozen=True)
class InferenceAssets:
    """Strict native model and chemistry assets."""

    bundle: NativeBundle
    conformers: Mapping[str, ConformerData]


@dataclass(frozen=True)
class ModelComponents:
    """All six native components, with callable model states mapped to JAX."""

    feature_embedding: Mapping[str, np.ndarray]
    bond_loss_input_proj: Mapping[str, np.ndarray]
    token_embedder: TokenEmbedderParams
    trunk: TrunkParams
    diffusion: FullDiffusionDenoiserParams
    confidence: ConfidenceHeadParams


@dataclass(frozen=True)
class PreparedInference:
    """Padded model input plus the seed that produced its stochastic features."""

    inputs: tuple[Input, ...]
    structure_context: StructureContext
    model_size: int
    realized_seed: int
    padded_inputs: Mapping[str, np.ndarray]
    features: Mapping[str, np.ndarray]


class InferenceBackend(Protocol):
    """Execution hook while the production sampler/output path is completed."""

    def __call__(
        self,
        prepared: PreparedInference,
        components: ModelComponents,
        config: InferenceConfig,
    ) -> Any: ...


def load_inference_assets(
    bundle_path: str | Path,
    conformer_path: str | Path,
    *,
    expected_conformer_version: str | None = None,
) -> InferenceAssets:
    """Load and checksum-validate every standalone inference asset."""
    return InferenceAssets(
        bundle=load_native_bundle(bundle_path),
        conformers=load_native_conformers(
            conformer_path, expected_asset_version=expected_conformer_version
        ),
    )


def map_model_components(bundle: NativeBundle) -> ModelComponents:
    """Map components while keeping their parameters host-resident.

    Each compiled stage transfers only its own parameters. This mirrors Chai's
    ``low_memory=True`` comparison mode instead of pinning all six components
    in accelerator memory for the lifetime of the process.
    """

    def on_host(value: Any) -> Any:
        # Stack the repeated blocks here, while they are still NumPy. Doing it
        # inside the traced stage instead makes XLA copy the whole component
        # into its temp arena -- see `foldjax.models._stacking`.
        return prestack_layer_lists(
            jax.tree.map(lambda leaf: np.asarray(jax.device_get(leaf)), value)
        )

    states = bundle.components
    return ModelComponents(
        feature_embedding=states["feature_embedding"],
        bond_loss_input_proj=states["bond_loss_input_proj"],
        token_embedder=on_host(map_token_embedder(states["token_embedder"])),
        trunk=on_host(map_trunk(states["trunk"])),
        diffusion=on_host(map_full_diffusion_denoiser(states["diffusion_module"])),
        confidence=on_host(map_confidence_head(states["confidence_head"])),
    )


def _batched_structure_inputs(context: StructureContext) -> dict[str, np.ndarray]:
    excluded = {"atom_ref_name", "atom_covalent_bond_indices"}
    result = {
        name: np.asarray(value)[None]
        for name, value in context.to_dict().items()
        if name not in excluded and isinstance(value, np.ndarray)
    }
    # These are structure-level upstream fields, not token/atom axes.
    result["resolution"] = np.asarray(context.resolution)[None]
    result["is_distillation"] = np.asarray(context.is_distillation)[None]
    return result


def _empty_msa(context: StructureContext) -> MSAContext:
    return MSAContext.create_empty(context.num_tokens, depth=FULL_DEPTH)


def _collate_inference_inputs(
    structure: StructureContext,
    parsed_inputs: list[Input],
    config: InferenceConfig,
    msa_search_pipeline: MsaSearchPipeline | None = None,
) -> tuple[int, dict[str, np.ndarray]]:
    pad_sizes = get_pad_sizes(structure.num_tokens, structure.num_atoms)
    padded = structure.pad(pad_sizes.n_tokens, pad_sizes.n_atoms)
    values = _batched_structure_inputs(padded)

    # Padding is the explicit residue class used by Chai's feature embedder.
    values["token_residue_type"] = values["token_residue_type"].copy()
    values["token_residue_type"][~values["token_exists_mask"]] = 32

    query, key, key_valid = qkv_indices_for_blocks(pad_sizes.n_atoms)
    # Chai builds reference-geometry pair features from ``atom_ref_mask``.
    # Glycan leaving atoms can be valid reference atoms while intentionally
    # absent from predicted coordinates (``atom_exists_mask=False``).
    atom_mask = jnp.asarray(values["atom_ref_mask"], dtype=bool)
    values["block_atom_pair_q_idces"] = np.asarray(query)
    values["block_atom_pair_kv_idces"] = np.asarray(key)
    pair_mask = block_atom_pair_mask(atom_mask, query, key, key_valid)
    # Chai's blocked-distance feature generator intersects this batch tensor
    # in-place with atom_ref_space equality before the same tensor is passed to
    # the token embedder and diffusion module.  Preserve that final batch
    # contract explicitly instead of depending on feature-generator mutation.
    reference_space = jnp.asarray(values["atom_ref_space_uid"])
    same_reference_space = (
        reference_space[:, query][..., :, None] == reference_space[:, key][..., None, :]
    )
    values["block_atom_pair_mask"] = np.asarray(pair_mask & same_reference_space)

    if config.msa_directory is not None or msa_search_pipeline is not None:
        search_results = None
        if msa_search_pipeline is not None:
            protein_sequences = [
                item.sequence
                for item in parsed_inputs
                if item.entity_type == EntityType.PROTEIN.value
            ]
            search_results = (
                msa_search_pipeline.search(protein_sequences)
                if protein_sequences
                else []
            )
        msa, profile_msa = load_msa_contexts(
            parsed_inputs,
            structure,
            msa_directory=config.msa_directory,
            search_results=search_results,
        )
        msa = msa.pad(
            max_num_tokens=pad_sizes.n_tokens,
            max_msa_depth=FULL_DEPTH,
        )
        profile_msa = profile_msa.pad(max_num_tokens=pad_sizes.n_tokens)
    else:
        msa = _empty_msa(structure).pad(max_num_tokens=pad_sizes.n_tokens)
        profile_msa = msa
    values.update(
        {
            "msa_tokens": msa.tokens[None],
            "msa_mask": msa.mask[None],
            "msa_deletion_matrix": msa.deletion_matrix[None],
            "msa_pairkey": msa.pairing_key_hash[None],
            "msa_sequence_source": msa.sequence_source[None],
            "main_msa_tokens": profile_msa.tokens[None],
            "main_msa_mask": profile_msa.mask[None],
            "main_msa_deletion_matrix": profile_msa.deletion_matrix[None],
        }
    )

    if not config.use_esm_embeddings:
        values["esm_embeddings"] = np.zeros((1, pad_sizes.n_tokens, 2560), np.float32)
    else:
        if config.esm_embeddings_path is not None:
            embeddings = load_native_esm_embeddings(
                config.esm_embeddings_path
            ).embeddings
        else:
            if config.esm_model_path is None:
                raise ValueError("native ESM2 model path is required")
            embeddings = generate_native_esm_embeddings_for_inputs(
                parsed_inputs,
                model_path=config.esm_model_path,
                cache_dir=config.esm_cache_directory,
                attention_implementation=config.esm_attention_implementation,
            )
        esm = assemble_esm_context(
            parsed_inputs,
            token_asym_id=structure.token_asym_id,
            token_residue_index=structure.token_residue_index,
            embeddings=embeddings,
        ).pad(pad_sizes.n_tokens)
        values["esm_embeddings"] = esm.esm_embeddings[None]

    template_hits_path = config.template_hits_path
    if config.use_templates_server:
        if search_results is None:
            raise ValueError("template server search did not produce results")
        template_hits_path = materialize_server_template_hits(
            parsed_inputs,
            search_results,
            cache_dir=config.template_cache_directory,
        )
    if template_hits_path is None:
        templates = TemplateContext.empty(4, structure.num_tokens)
    elif template_hits_path.suffix.lower() == ".npz":
        if config.template_cif_directory is not None:
            raise ValueError(
                "template_cif_directory cannot be used with a native template archive"
            )
        templates = load_native_template_context(template_hits_path).context
    else:
        templates = build_template_context_from_m8(
            parsed_inputs,
            token_asym_id=structure.token_asym_id,
            token_residue_index=structure.token_residue_index,
            m8_path=template_hits_path,
            template_cif_dir=config.template_cif_directory,
            cache_dir=config.template_cache_directory,
            kalign_executable=config.kalign_executable,
        ).context
    if templates.num_tokens != structure.num_tokens:
        raise ValueError("template token count does not match the input structure")
    padded_templates = templates.pad(max_templates=4, max_tokens=pad_sizes.n_tokens)
    values.update(padded_templates.to_model_inputs(batched=True))
    values["template_mask"] = padded_templates.template_mask[None]
    if config.constraint_path is not None:
        values.update(parse_restraints_csv(config.constraint_path))
    return pad_sizes.n_tokens, values


def prepare_inference_from_assets(
    fasta_file: str | Path,
    assets: InferenceAssets,
    *,
    config: InferenceConfig | None = None,
    msa_search_pipeline: MsaSearchPipeline | None = None,
) -> PreparedInference:
    """Parse, tokenize, pad, collate, and featurize one public Chai FASTA."""
    config = InferenceConfig() if config is None else config
    config.validate()
    realized_seed = config.seed if config.seed is not None else _random_seed()
    rng = np.random.default_rng(realized_seed)
    path = Path(fasta_file)
    if not path.is_file():
        raise FileNotFoundError(f"FASTA input is not a regular file: {path}")
    parsed = read_inputs(path)
    covalent_bonds = (
        parse_covalent_bonds_csv(config.constraint_path)
        if config.constraint_path is not None
        else None
    )
    structure = tokenize_inputs(
        parsed,
        dict(assets.conformers),
        identifier=path.stem,
        entity_name_as_subchain=config.fasta_names_as_cif_chains,
        covalent_bonds=covalent_bonds,
        rng=rng,
    )
    if config.msa_directory is not None and msa_search_pipeline is not None:
        raise ValueError("msa_directory and msa_search_pipeline are mutually exclusive")
    if config.use_msa_server and msa_search_pipeline is None:
        if config.msa_cache_directory is None:
            raise ValueError(
                "use_msa_server requires msa_cache_directory outside run_inference"
            )
        remote_options: dict[str, Any] = {}
        if config.use_templates_server:
            remote_options["use_templates"] = True
        msa_search_pipeline = MsaSearchPipeline(
            config.msa_cache_directory,
            RemoteMMseqs2Backend(
                config.msa_server_url,
                version=config.msa_server_version,
                **remote_options,
            ),
        )
    model_size, padded_inputs = _collate_inference_inputs(
        structure,
        parsed,
        config,
        msa_search_pipeline,
    )
    features = generate_features(padded_inputs, rng=rng)
    return PreparedInference(
        inputs=tuple(parsed),
        structure_context=structure,
        model_size=model_size,
        realized_seed=realized_seed,
        padded_inputs=padded_inputs,
        features=features,
    )


def prepare_inference(
    fasta_file: str | Path,
    *,
    bundle_path: str | Path,
    conformer_path: str | Path,
    config: InferenceConfig | None = None,
    expected_conformer_version: str | None = None,
    msa_search_pipeline: MsaSearchPipeline | None = None,
) -> tuple[PreparedInference, InferenceAssets]:
    """Load strict assets and build deterministic model-ready inputs."""
    assets = load_inference_assets(
        bundle_path,
        conformer_path,
        expected_conformer_version=expected_conformer_version,
    )
    return (
        prepare_inference_from_assets(
            fasta_file,
            assets,
            config=config,
            msa_search_pipeline=msa_search_pipeline,
        ),
        assets,
    )


def _bin_centers(minimum: float, maximum: float, count: int) -> jnp.ndarray:
    return jnp.linspace(minimum, maximum, 2 * count + 1, dtype=jnp.float32)[1::2]


def _random_rotations(key: jax.Array, count: int) -> jnp.ndarray:
    quaternion = jax.random.normal(key, (count, 4), dtype=jnp.float32)
    norm = jnp.linalg.norm(quaternion, axis=-1, keepdims=True)
    quaternion = quaternion / jnp.maximum(norm, 1e-8)
    quaternion = quaternion * jnp.where(quaternion[:, :1] < 0.0, -1.0, 1.0)
    real, i, j, k = jnp.moveaxis(quaternion, -1, 0)
    two = 2.0
    return jnp.stack(
        (
            1 - two * (j * j + k * k),
            two * (i * j - k * real),
            two * (i * k + j * real),
            two * (i * j + k * real),
            1 - two * (i * i + k * k),
            two * (j * k - i * real),
            two * (i * k - j * real),
            two * (j * k + i * real),
            1 - two * (i * i + j * j),
        ),
        axis=-1,
    ).reshape(count, 3, 3)


def _random_translations(key: jax.Array, count: int) -> jnp.ndarray:
    """Draw one broadcastable translation vector per diffusion sample."""
    return jax.random.normal(key, (count, 1, 3), dtype=jnp.float32)


def _token_bond_feature(prepared: PreparedInference) -> jnp.ndarray:
    token_count = prepared.model_size
    feature = np.zeros((1, token_count, token_count, 1), np.float32)
    left, right = prepared.structure_context.atom_covalent_bond_indices
    if left.size:
        atom_token = prepared.structure_context.atom_token_index
        feature[0, atom_token[left], atom_token[right], 0] = 1.0
    return jnp.asarray(feature)


_RAW_RESTRAINT_CONTROL_KEYS = frozenset(
    {"contact_constraints", "docking_constraints", "pocket_constraints"}
)


def _model_padded_inputs(
    prepared: PreparedInference,
) -> dict[str, jax.Array]:
    """Move numeric model inputs to JAX, excluding consumed restraint controls."""
    values: dict[str, jax.Array] = {}
    for name, value in prepared.padded_inputs.items():
        if name in _RAW_RESTRAINT_CONTROL_KEYS:
            continue
        if not isinstance(value, np.ndarray):
            raise TypeError(f"padded input {name!r} must be a NumPy array")
        if name in _MODEL_PADDED_INPUT_KEYS:
            values[name] = jnp.asarray(value)
    return values


def _embed_and_initialize(
    prepared: PreparedInference,
    components: ModelComponents,
    *,
    msa_mode: str = "cropped",
    max_msa_depth: int | None = None,
) -> tuple[
    dict[str, jnp.ndarray],
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray | None,
    jnp.ndarray,
]:
    if msa_mode not in {"cropped", "full", "split"}:
        raise ValueError(f"invalid MSA embedding mode: {msa_mode!r}")
    row_indices = None
    if msa_mode == "cropped":
        # This is the selection that sizes the embedded MSA -- the rows chosen
        # here are the ones that get embedded, and the result is the largest
        # tensor in the trunk. Capping only the per-recycle crop downstream
        # would leave that tensor at full depth and move nothing.
        row_indices, _ = _msa_row_indices_and_mask(
            prepared.padded_inputs["msa_mask"], key=None, max_depth=max_msa_depth
        )
    features = {}
    for name, value in prepared.features.items():
        if name in _MSA_ROW_FEATURE_NAMES:
            if msa_mode == "split":
                continue
            if row_indices is not None:
                value = np.take(value, row_indices, axis=1)
        features[name] = jnp.asarray(value)
    compiled_embed = (
        _compiled_embed_non_msa_features
        if msa_mode == "split"
        else _compiled_embed_features
    )
    embedded = compiled_embed(features, components.feature_embedding)
    token_pair_input, token_pair_structure = jnp.split(
        embedded["TOKEN_PAIR"], 2, axis=-1
    )
    atom_input, atom_structure = jnp.split(embedded["ATOM"], 2, axis=-1)
    atom_pair_input, atom_pair_structure = jnp.split(embedded["ATOM_PAIR"], 2, axis=-1)
    bond_weight = jnp.asarray(components.bond_loss_input_proj["weight"])
    bond = _token_bond_feature(prepared) @ jnp.swapaxes(bond_weight, -1, -2)
    trunk_bond, structure_bond = jnp.split(bond, 2, axis=-1)
    token_pair_input = token_pair_input + trunk_bond
    token_pair_structure = token_pair_structure + structure_bond

    values = _model_padded_inputs(prepared)
    token_inputs = {
        "token_single_input_feats": embedded["TOKEN"],
        "token_pair_input_feats": token_pair_input,
        "atom_single_input_feats": atom_input,
        "block_atom_pair_feat": atom_pair_input,
        "block_atom_pair_mask": values["block_atom_pair_mask"],
        "block_indices_h": values["block_atom_pair_q_idces"],
        "block_indices_w": values["block_atom_pair_kv_idces"],
        "atom_single_mask": values["atom_exists_mask"],
        "atom_token_indices": values["atom_token_index"],
    }
    single_initial, single_structure, pair_initial = _compiled_token_embedder(
        token_inputs,
        components.token_embedder,
        bf16=_TOKEN_EMBEDDER_COMPATIBILITY_BF16,
    )
    diffusion_inputs = {
        "token_single_initial_repr": single_structure.astype(jnp.float32),
        "token_pair_initial_repr": token_pair_structure.astype(jnp.float32),
        "atom_single_input_feats": atom_structure.astype(jnp.float32),
        "atom_block_pair_input_feats": atom_pair_structure.astype(jnp.float32),
        "atom_single_mask": values["atom_exists_mask"],
        "atom_block_pair_mask": values["block_atom_pair_mask"],
        "token_single_mask": values["token_exists_mask"],
        "block_indices_h": values["block_atom_pair_q_idces"],
        "block_indices_w": values["block_atom_pair_kv_idces"],
        "atom_token_indices": values["atom_token_index"],
    }
    result = (
        diffusion_inputs,
        single_initial,
        pair_initial,
        embedded.get("MSA"),
        embedded["TEMPLATES"],
    )
    # Split mode launches a separate MSA executable, so finish all consumers
    # before releasing its non-MSA raw feature buffers.  Combined modes retain
    # the original async schedule and have no cross-executable MSA overlap.
    if msa_mode == "split":
        jax.block_until_ready(result)
    del embedded, features, token_inputs, values
    return result


@jax.jit
def _compiled_embed_features(
    features: Mapping[str, jax.Array], params: Mapping[str, jax.Array]
) -> Mapping[str, jax.Array]:
    return embed_features(features, params)


@jax.jit
def _compiled_embed_non_msa_features(
    features: Mapping[str, jax.Array], params: Mapping[str, jax.Array]
) -> Mapping[str, jax.Array]:
    return embed_non_msa_features(features, params)


@jax.jit
def _compiled_embed_msa(features, weight, bias):
    return embed_msa(features, weight, bias)


@partial(jax.jit, static_argnames=("bf16",))
def _compiled_token_embedder(
    inputs: Mapping[str, jax.Array],
    params: TokenEmbedderParams,
    *,
    bf16: bool,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    return token_embedder_forward(inputs, params, bf16=bf16)


@partial(jax.jit, static_argnames=("skip_templates",))
def _compiled_trunk(
    single_initial: jax.Array,
    pair_initial: jax.Array,
    single: jax.Array,
    pair: jax.Array,
    msa_features: jax.Array,
    msa_mask: jax.Array,
    template_features: jax.Array,
    template_pair_mask: jax.Array,
    token_mask: jax.Array,
    pair_mask: jax.Array,
    params: TrunkParams,
    *,
    skip_templates: bool,
) -> tuple[jax.Array, jax.Array]:
    return trunk_forward(
        single_initial,
        pair_initial,
        single,
        pair,
        msa_features,
        msa_mask,
        template_features,
        template_pair_mask,
        token_mask,
        pair_mask,
        params,
        weighted_averaging_chunk_size=1024,
        skip_templates=skip_templates,
    )


@partial(jax.jit, static_argnames=("skip_templates",))
def _compiled_recycle_template(
    single_initial: jax.Array,
    pair_initial: jax.Array,
    single: jax.Array,
    pair: jax.Array,
    template_features: jax.Array,
    template_pair_mask: jax.Array,
    pair_mask: jax.Array,
    params: TrunkParams,
    *,
    skip_templates: bool,
) -> tuple[jax.Array, jax.Array]:
    pair = pair_initial + recycling_projection(pair, params.pair_recycling)
    single = single_initial + recycling_projection(single, params.single_recycling)
    if not skip_templates:
        pair = template_embedding(
            pair,
            template_features,
            template_pair_mask,
            pair_mask,
            params.template,
        )
    return single, pair


_compiled_msa_embedding = jax.jit(
    lambda features, single, weight: features + linear_bf16(single, weight)[:, None]
)

# The official component accepts BF16 inputs, but CUDA and XLA reduce the
# 833-atom local attention in different orders. XLA FP32 accumulation followed
# by the same downstream trunk is substantially closer to the released CUDA
# BF16 result on real proteins than XLA BF16 accumulation.
_TOKEN_EMBEDDER_COMPATIBILITY_BF16 = False
_compiled_outer_product_mean = jax.jit(
    lambda msa, mask, params: outer_product_mean(msa, mask, params, chunk_size=4096)
)
_compiled_msa_transition = jax.jit(
    lambda msa, params: msa_transition(msa, params, chunk_size=1024)
)
_compiled_msa_weighted = jax.jit(
    lambda msa, pair, msa_mask, pair_mask, params: msa_pair_weighted_averaging(
        msa,
        pair,
        msa_mask,
        pair_mask,
        params,
        chunk_size=1024,
    )
)
_compiled_msa_weighted_128 = jax.jit(
    lambda msa, pair, msa_mask, pair_mask, params: msa_pair_weighted_averaging(
        msa,
        pair,
        msa_mask,
        pair_mask,
        params,
        chunk_size=128,
    )
)
_compiled_msa_pair = jax.jit(_msa_pair_block)
_compiled_msa_pair_triangle_outgoing = jax.jit(
    lambda pair, pair_mask, params: fused_triangle_multiplication_direction(
        pair, pair_mask, params, incoming=False
    )
)
_compiled_msa_pair_triangle_incoming = jax.jit(
    lambda pair, pair_mask, params: fused_triangle_multiplication_direction(
        pair, pair_mask, params, incoming=True
    )
)
_compiled_msa_pair_triangle_output = jax.jit(fused_triangle_multiplication_output)
_compiled_msa_pair_transition_first = jax.jit(
    lambda pair, params: pairformer_transition_projection(pair, params, second=False)
)
_compiled_msa_pair_transition_second = jax.jit(
    lambda pair, params: pairformer_transition_projection(pair, params, second=True)
)
_compiled_msa_pair_transition_update = jax.jit(pairformer_transition_output)
_compiled_msa_pair_pre_attention = jax.jit(
    lambda pair, triangle_update, transition_update: (
        pair + triangle_update + transition_update
    ),
    donate_argnums=(1,),
)
_compiled_msa_pair_attention_first = jax.jit(
    lambda pair, pair_mask, params: fused_triangle_attention_direction(
        pair, pair_mask, params, direction=0
    )
)
_compiled_msa_pair_attention_second = jax.jit(
    lambda pair, pair_mask, params: fused_triangle_attention_direction(
        pair, pair_mask, params, direction=1
    )
)
_compiled_msa_pair_attention_output = jax.jit(fused_triangle_attention_output)
_compiled_msa_pair_add_attention = jax.jit(
    lambda pair, update: pair + update,
    donate_argnums=(1,),
)
_compiled_pair_triangle_tile = jax.jit(
    fused_triangle_multiplication_direction_chunk,
    static_argnames=(
        "incoming",
        "start",
        "size",
        "column_start",
        "column_size",
    ),
)
_compiled_pair_triangle_output_chunk = jax.jit(fused_triangle_multiplication_output)
_compiled_pair_transition_projection_chunk = jax.jit(
    lambda pair, params, second: pairformer_transition_projection(
        pair, params, second=second
    ),
    static_argnames=("second",),
)
_compiled_pair_transition_output_chunk = jax.jit(pairformer_transition_output)
_compiled_pair_attention_bias = jax.jit(fused_triangle_attention_bias)
_compiled_pair_attention_direction_chunk = jax.jit(
    fused_triangle_attention_direction_chunk,
    static_argnames=("direction",),
)
_compiled_pair_attention_output_chunk = jax.jit(fused_triangle_attention_output)
_compiled_pairformer_single_attention = jax.jit(attention_pair_bias)
_compiled_pairformer_single_transition = jax.jit(pairformer_transition)
_compiled_residual_add = jax.jit(
    lambda value, update: value + update,
    donate_argnums=(1,),
)
_compiled_pairformer_stack = jax.jit(
    lambda blocks, single, pair, token_mask, pair_mask: _pairformer_stack(
        single,
        pair,
        token_mask,
        pair_mask,
        blocks,
    )
)


def _chunk_starts(pair, chunk_size):
    return range(0, pair.shape[-2], chunk_size)


def _chunked_triangle_multiplication(pair, pair_mask, params, chunk_size):
    output_rows = []
    for start in _chunk_starts(pair, chunk_size):
        size = min(chunk_size, pair.shape[-2] - start)
        output_columns = []
        for column_start in _chunk_starts(pair, chunk_size):
            column_size = min(chunk_size, pair.shape[-2] - column_start)
            outgoing = _compiled_pair_triangle_tile(
                pair,
                pair_mask,
                params,
                incoming=False,
                start=start,
                size=size,
                column_start=column_start,
                column_size=column_size,
            )
            incoming = _compiled_pair_triangle_tile(
                pair,
                pair_mask,
                params,
                incoming=True,
                start=start,
                size=size,
                column_start=column_start,
                column_size=column_size,
            )
            pair_row = jax.lax.dynamic_slice_in_dim(pair, start, size, axis=-3)
            pair_tile = jax.lax.dynamic_slice_in_dim(
                pair_row, column_start, column_size, axis=-2
            )
            output = _compiled_pair_triangle_output_chunk(
                pair_tile, outgoing, incoming, params
            )
            jax.block_until_ready(output)
            output_columns.append(output)
        output_rows.append(jnp.concatenate(output_columns, axis=-2))
    output = jnp.concatenate(output_rows, axis=-3)
    jax.block_until_ready(output)
    return output


def _chunked_pair_transition(pair, params, chunk_size):
    outputs = []
    for start in _chunk_starts(pair, chunk_size):
        size = min(chunk_size, pair.shape[-2] - start)
        pair_part = jax.lax.dynamic_slice_in_dim(pair, start, size, axis=-3)
        first = _compiled_pair_transition_projection_chunk(
            pair_part, params, second=False
        )
        second = _compiled_pair_transition_projection_chunk(
            pair_part, params, second=True
        )
        output = _compiled_pair_transition_output_chunk(first, second, params)
        jax.block_until_ready(output)
        outputs.append(output)
    return jnp.concatenate(outputs, axis=-3)


def _triangle_attention_host_chunk_size(chunk_size: int) -> int:
    # XLA SDPA materializes work proportional to outer_rows * heads * N^2.
    # Keep cuEq's larger launch granularity, but bound XLA's host-side outer
    # rows so the advertised CUDA 12 / forced-XLA 2048 path remains viable.
    if _triangle_attention_backend() == "xla":
        return min(chunk_size, 64)
    return chunk_size


def _chunked_triangle_attention(pair, pair_mask, params, chunk_size):
    chunk_size = _triangle_attention_host_chunk_size(chunk_size)
    bias = _compiled_pair_attention_bias(pair, pair_mask, params)
    jax.block_until_ready(bias)
    outputs = []
    for start in _chunk_starts(pair, chunk_size):
        size = min(chunk_size, pair.shape[-2] - start)
        first_input = jax.lax.dynamic_slice_in_dim(pair, start, size, axis=-3)
        second_input = jax.lax.dynamic_slice_in_dim(pair, start, size, axis=-2)
        second_input = jnp.swapaxes(second_input, -3, -2)
        first = _compiled_pair_attention_direction_chunk(
            first_input, bias, params, direction=0
        )
        second = _compiled_pair_attention_direction_chunk(
            second_input, bias, params, direction=1
        )
        output = _compiled_pair_attention_output_chunk(first, second, params)
        jax.block_until_ready(output)
        outputs.append(output)
    return jnp.concatenate(outputs, axis=-3)


def _run_msa_pair_block_low_memory(pair, pair_mask, params, *, subchunk_size=None):
    """Preserve MSA pair-block order with bounded triangle-attention stages."""

    if subchunk_size is not None and subchunk_size < pair.shape[-2]:
        triangle_update = _chunked_triangle_multiplication(
            pair, pair_mask, params.triangle_multiplication, subchunk_size
        )
        transition_update = _chunked_pair_transition(
            pair, params.transition_pair, subchunk_size
        )
        intermediate = _compiled_msa_pair_pre_attention(
            pair, triangle_update, transition_update
        )
        jax.block_until_ready(intermediate)
        del pair, triangle_update, transition_update
        update = _chunked_triangle_attention(
            intermediate, pair_mask, params.triangle_attention, subchunk_size
        )
        output = _compiled_msa_pair_add_attention(intermediate, update)
        jax.block_until_ready(output)
        return output

    outgoing = _compiled_msa_pair_triangle_outgoing(
        pair, pair_mask, params.triangle_multiplication
    )
    jax.block_until_ready(outgoing)
    incoming = _compiled_msa_pair_triangle_incoming(
        pair, pair_mask, params.triangle_multiplication
    )
    jax.block_until_ready(incoming)
    triangle_update = _compiled_msa_pair_triangle_output(
        pair, outgoing, incoming, params.triangle_multiplication
    )
    jax.block_until_ready(triangle_update)
    transition_first = _compiled_msa_pair_transition_first(pair, params.transition_pair)
    jax.block_until_ready(transition_first)
    transition_second = _compiled_msa_pair_transition_second(
        pair, params.transition_pair
    )
    jax.block_until_ready(transition_second)
    transition_update = _compiled_msa_pair_transition_update(
        transition_first, transition_second, params.transition_pair
    )
    jax.block_until_ready(transition_update)
    intermediate = _compiled_msa_pair_pre_attention(
        pair, triangle_update, transition_update
    )
    jax.block_until_ready(intermediate)
    first = _compiled_msa_pair_attention_first(
        intermediate, pair_mask, params.triangle_attention
    )
    jax.block_until_ready(first)
    second = _compiled_msa_pair_attention_second(
        intermediate, pair_mask, params.triangle_attention
    )
    jax.block_until_ready(second)
    update = _compiled_msa_pair_attention_output(
        first, second, params.triangle_attention
    )
    jax.block_until_ready(update)
    output = _compiled_msa_pair_add_attention(intermediate, update)
    jax.block_until_ready(output)
    return output


def _run_pairformer_block_low_memory(
    single, pair, token_mask, pair_mask, params, *, subchunk_size
):
    """Run one official parallel Pairformer block with bounded pair branches."""

    triangle = _chunked_triangle_multiplication(
        pair, pair_mask, params.triangle_multiplication, subchunk_size
    )
    pair_output = _compiled_residual_add(pair, triangle)
    jax.block_until_ready(pair_output)
    del triangle
    triangle_attention = _chunked_triangle_attention(
        pair, pair_mask, params.triangle_attention, subchunk_size
    )
    pair_output = _compiled_residual_add(pair_output, triangle_attention)
    jax.block_until_ready(pair_output)
    del triangle_attention
    pair_transition = _chunked_pair_transition(
        pair, params.transition_pair, subchunk_size
    )
    pair_output = _compiled_residual_add(pair_output, pair_transition)
    jax.block_until_ready(pair_output)
    del pair_transition
    single_attention = _compiled_pairformer_single_attention(
        single, pair, pair_mask, token_mask, params.attention_pair_bias
    )
    jax.block_until_ready(single_attention)
    single_output = _compiled_residual_add(single, single_attention)
    jax.block_until_ready(single_output)
    del single_attention
    single_transition = _compiled_pairformer_single_transition(
        single, params.transition_single
    )
    single_output = _compiled_residual_add(single_output, single_transition)
    output = single_output, pair_output
    jax.block_until_ready(output)
    return output


def _msa_pair_subchunk_size(token_count: int) -> int | None:
    if token_count >= 2048 or (
        token_count >= 1536 and _triangle_attention_backend() == "xla"
    ):
        return 512
    return None


# Chai's pair representation is quadratic in the padded token count, so a
# threshold written as a token count crosses budget somewhere between two
# buckets rather than at one. It was 1536, which left the 1024 bucket on the
# full-memory path: a 976-token job peaked at 21,023 MiB where the low-memory
# path takes it to 19,229 with identical scores, for 12% more wall time. Keyed
# to the tensor's own size instead, the same rule holds at every bucket and on
# every card.
_LOW_MEMORY_PAIR_BUDGET_BYTES = 256 * 1024**2


def _pair_bytes(pair: jax.Array) -> int:
    total = pair.dtype.itemsize
    for dim in pair.shape:
        total *= dim
    return total


def _use_low_memory_pairformer(pair: jax.Array | int) -> bool:
    if isinstance(pair, int):  # token count, kept for the backend-selection tests
        token_count = pair
    else:
        if _pair_bytes(pair) >= _LOW_MEMORY_PAIR_BUDGET_BYTES:
            return True
        token_count = pair.shape[-2]
    return token_count >= 2048 or (
        token_count >= 1536 and _triangle_attention_backend() == "xla"
    )


def _run_staged_trunk(
    single_initial: jax.Array,
    pair_initial: jax.Array,
    single: jax.Array,
    pair: jax.Array,
    msa_features: jax.Array,
    msa_mask: jax.Array,
    template_features: jax.Array,
    template_pair_mask: jax.Array,
    token_mask: jax.Array,
    pair_mask: jax.Array,
    params: TrunkParams,
    *,
    skip_templates: bool,
) -> tuple[jax.Array, jax.Array]:
    """Execute bounded trunk stages so XLA can release MSA intermediates."""
    compiled_msa_weighted = (
        _compiled_msa_weighted_128 if pair.shape[-2] >= 1536 else _compiled_msa_weighted
    )
    single, pair = _compiled_recycle_template(
        single_initial,
        pair_initial,
        single,
        pair,
        template_features,
        template_pair_mask,
        pair_mask,
        params,
        skip_templates=skip_templates,
    )
    # The exported trunk keeps the recycle/template-conditioned pair as an
    # outer residual around the complete MSA module.  The residuals inside
    # each MSA block do not replace this graph-level addition.
    pair_before_msa = pair
    msa = _compiled_msa_embedding(msa_features, single, params.msa.linear_s2m_weight)
    msa_input = None
    for index, block in enumerate(params.msa.blocks):
        # `a + b` on device arrays dispatches an eager `add`, which is its own
        # XLA program: both operands and the result are live at once and none
        # of the buffers can be reused. On the MSA representation that is three
        # live copies of the largest tensor in the trunk. `_compiled_residual_add`
        # is the same arithmetic with the update donated, so the sum is written
        # back into it -- the pair path below already goes through it.
        pair = _compiled_residual_add(
            pair,
            _compiled_outer_product_mean(msa, msa_mask, block.outer_product_mean),
        )
        if index < 3:
            msa_input = msa
            msa = _compiled_residual_add(
                msa_input,
                _compiled_msa_transition(msa_input, block.msa_transition),
            )
            msa = _compiled_residual_add(
                msa,
                compiled_msa_weighted(
                    msa_input,
                    pair,
                    msa_mask,
                    pair_mask,
                    block.weighted_averaging,
                ),
            )
            # Last use: holding it through the pair block below would keep a
            # second copy of the MSA alive for the most expensive part of the
            # iteration. Rebound rather than deleted, because the low-memory
            # branch after the loop deletes it by name.
            msa_input = None
        if pair.shape[-2] >= 1536:
            pair = _run_msa_pair_block_low_memory(
                pair,
                pair_mask,
                block.pair,
                subchunk_size=_msa_pair_subchunk_size(pair.shape[-2]),
            )
        else:
            pair = _compiled_msa_pair(pair, pair_mask, block.pair)
    pair = _compiled_residual_add(pair_before_msa, pair)
    if _use_low_memory_pairformer(pair):
        jax.block_until_ready((single, pair))
        del msa, msa_input, pair_before_msa, block
        for block in params.pairformer_blocks:
            single, pair = _run_pairformer_block_low_memory(
                single,
                pair,
                token_mask,
                pair_mask,
                block,
                subchunk_size=512,
            )
        return single, pair
    return _compiled_pairformer_stack(
        params.pairformer_blocks, single, pair, token_mask, pair_mask
    )


def _subsample_msa_for_recycle(
    msa_features: jax.Array,
    msa_mask: np.ndarray,
    key: jax.Array,
    max_depth: int | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Subsample real MSA rows with an explicit JAX PRNG draw."""

    row_indices, sampled_mask = _msa_row_indices_and_mask(
        msa_mask, key=key, max_depth=max_depth
    )
    return jnp.take(msa_features, jnp.asarray(row_indices), axis=1), jnp.asarray(
        sampled_mask
    )


def _msa_row_indices_and_mask(
    msa_mask: np.ndarray,
    *,
    key: jax.Array | None,
    max_depth: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Choose which alignment rows the trunk sees, and pad to a bucket.

    ``max_depth`` caps the count before bucketing. Chai's rows arrive in
    priority order, so the first N are the alignment's best N -- the same
    tradeoff the other backends make under ``max_msa_depth``, and off by
    default so the released behaviour is unchanged unless it is asked for.
    """
    def bucket_plan(
        row_indices: np.ndarray, selected_mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        buckets = (1, 128, 256, 512, 1024, 2048, 4096, 8192, 16384)
        selected_depth = len(row_indices)
        try:
            bucket_depth = next(
                depth for depth in buckets if depth >= selected_depth
            )
        except StopIteration as error:
            raise ValueError(
                f"MSA depth {selected_depth} exceeds supported maximum 16384"
            ) from error
        if bucket_depth == selected_depth:
            return row_indices, selected_mask
        padded_indices = np.pad(
            row_indices, (0, bucket_depth - selected_depth), constant_values=0
        )
        padded_mask = np.pad(
            selected_mask,
            ((0, 0), (0, bucket_depth - selected_depth), (0, 0)),
            constant_values=False,
        )
        return padded_indices, padded_mask

    if max_depth is not None and max_depth < 1:
        raise ValueError("max_depth must be positive")

    mask = np.asarray(msa_mask, dtype=np.bool_)
    if key is not None:
        depth = mask.shape[1]
        select_n_rows = 4096 if max_depth is None else min(4096, max_depth)
        random_values = np.asarray(jax.random.uniform(key, (depth,), dtype=jnp.float16))
        plan = msa_subsample_indices_and_mask(
            mask,
            select_n_rows=select_n_rows,
            random_values=random_values,
        )
        if plan is not None:
            row_order, sampled_mask = plan
            selected_rows = min(select_n_rows, row_order.size)
            return bucket_plan(
                row_order[:selected_rows], sampled_mask[:, :selected_rows]
            )

    active_rows = np.any(mask, axis=(0, 2))
    active_indices = np.flatnonzero(active_rows)
    kept_rows = int(active_indices[-1] + 1) if active_indices.size else 1
    if max_depth is not None:
        kept_rows = min(kept_rows, max_depth)
    return bucket_plan(np.arange(kept_rows), mask[:, :kept_rows])


def _crop_trailing_masked_msa_rows(
    msa_features: jax.Array,
    msa_mask: np.ndarray | jax.Array,
    max_depth: int | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Remove trailing MSA rows that cannot affect any masked computation."""

    row_indices, cropped_mask = _msa_row_indices_and_mask(
        np.asarray(msa_mask), key=None, max_depth=max_depth
    )
    return jnp.take(msa_features, jnp.asarray(row_indices), axis=1), jnp.asarray(
        cropped_mask
    )


def _embed_msa_for_recycle(
    prepared: PreparedInference,
    feature_embedding_params: Mapping[str, np.ndarray],
    *,
    key: jax.Array | None,
) -> tuple[jax.Array, jax.Array]:
    """Crop raw host MSA rows before any row-dependent device transfer."""

    row_indices, sampled_mask = _msa_row_indices_and_mask(
        prepared.padded_inputs["msa_mask"], key=key
    )
    raw_features = {
        name: jnp.asarray(np.take(prepared.features[name], row_indices, axis=1))
        for name in _MSA_ROW_FEATURE_NAMES
    }
    weight = jnp.asarray(feature_embedding_params["input_projs.MSA.0.weight"])
    bias = jnp.asarray(feature_embedding_params["input_projs.MSA.0.bias"])
    embedded = _compiled_embed_msa(raw_features, weight, bias)
    jax.block_until_ready(embedded)
    return embedded, jnp.asarray(sampled_mask)


_compiled_diffusion_pair_projection = jax.jit(diffusion_pair_conditioning_projection)
_compiled_diffusion_pair_transition = jax.jit(
    lambda pair, params: pair + pairformer_transition(pair, params, lin=linear),
    donate_argnums=(0,),
)
_compiled_diffusion_pair_norm = jax.jit(
    lambda pair, weight, bias: layer_norm(pair.astype(jnp.float32), weight, bias),
    donate_argnums=(0,),
)
_compiled_diffusion_single_conditioning = jax.jit(diffusion_single_conditioning)
_compiled_diffusion_atom_encoder = jax.jit(diffusion_atom_encoder_with_aux)
_compiled_diffusion_transformer_initialize = jax.jit(
    lambda token_single, conditioned_single, weight: (
        token_single + linear(conditioned_single, weight)
    )
)
_compiled_diffusion_transformer_pair_bias = jax.jit(diffusion_transformer_pair_bias)
_compiled_diffusion_transformer_attention = jax.jit(
    lambda single, cond, pair_bias, params: diffusion_transformer_attention_with_bias(
        single,
        cond,
        pair_bias,
        params,
        query_chunk_size=64,
    )
)
_compiled_diffusion_transformer_transition = jax.jit(diffusion_transformer_transition)
_compiled_diffusion_transformer_add = jax.jit(
    lambda single, attention, transition: single + attention + transition,
    donate_argnums=(1,),
)
_compiled_diffusion_post_norms = jax.jit(
    lambda transformed, atom_condition, top: (
        layer_norm(
            transformed,
            top.post_attention_norm_weight,
            top.post_attention_norm_bias,
        ),
        layer_norm(
            atom_condition,
            top.post_atom_condition_norm_weight,
            top.post_atom_condition_norm_bias,
        ),
    )
)
_compiled_diffusion_atom_decoder = jax.jit(atom_decoder_forward)


def _run_diffusion_pair_conditioning(diffusion_inputs, params, *, row_chunk_size=128):
    initial = diffusion_inputs["token_pair_initial_repr"]
    trunk = diffusion_inputs["token_pair_trunk_repr"]

    def map_rows(function, *values):
        outputs = []
        for start in range(0, initial.shape[-3], row_chunk_size):
            size = min(row_chunk_size, initial.shape[-3] - start)
            chunks = [
                jax.lax.dynamic_slice_in_dim(value, start, size, axis=-3)
                for value in values
            ]
            output = function(*chunks)
            jax.block_until_ready(output)
            outputs.append(output)
        result = jnp.concatenate(outputs, axis=-3)
        jax.block_until_ready(result)
        return result

    pair = map_rows(
        lambda initial_chunk, trunk_chunk: _compiled_diffusion_pair_projection(
            initial_chunk, trunk_chunk, params
        ),
        initial,
        trunk,
    )
    for transition in (params.pair_transition_1, params.pair_transition_2):
        pair = map_rows(
            lambda pair_chunk, transition=transition: (
                _compiled_diffusion_pair_transition(pair_chunk, transition)
            ),
            pair,
        )
    return map_rows(
        lambda pair_chunk: _compiled_diffusion_pair_norm(
            pair_chunk, params.pair_norm_weight, params.pair_norm_bias
        ),
        pair,
    )


def _run_diffusion_transformer_staged(
    single, conditioned_single, conditioned_pair, token_mask, params
):
    pair_mask = token_mask[:, :, None] & token_mask[:, None, :]
    for block in params.blocks:
        bias_chunks = []
        for start in range(0, conditioned_pair.shape[-3], 128):
            size = min(128, conditioned_pair.shape[-3] - start)
            pair_chunk = jax.lax.dynamic_slice_in_dim(
                conditioned_pair, start, size, axis=-3
            )
            mask_chunk = jax.lax.dynamic_slice_in_dim(pair_mask, start, size, axis=-2)
            bias = _compiled_diffusion_transformer_pair_bias(
                pair_chunk, mask_chunk, block
            )
            jax.block_until_ready(bias)
            bias_chunks.append(bias)
        pair_bias = jnp.concatenate(bias_chunks, axis=-2)
        jax.block_until_ready(pair_bias)
        attention = _compiled_diffusion_transformer_attention(
            single, conditioned_single, pair_bias, block
        )
        jax.block_until_ready(attention)
        transition = _compiled_diffusion_transformer_transition(
            single, conditioned_single, block
        )
        single = _compiled_diffusion_transformer_add(single, attention, transition)
        jax.block_until_ready(single)
    return single


def _run_diffusion_denoiser_staged(
    position, sigma, diffusion_inputs, conditioned_pair, params
):
    sample_count = position.shape[0]
    noise_sigma = jnp.broadcast_to(sigma, (1, sample_count))
    conditioned_single = _compiled_diffusion_single_conditioning(
        diffusion_inputs["token_single_initial_repr"],
        diffusion_inputs["token_single_trunk_repr"],
        noise_sigma,
        params.conditioning,
    )
    sigma_expanded = noise_sigma[..., None, None]
    scaled_coords = position[None] / jnp.sqrt(sigma_expanded**2 + 256.0)
    encoder = _compiled_diffusion_atom_encoder(
        diffusion_inputs["atom_single_input_feats"],
        diffusion_inputs["token_single_trunk_repr"],
        conditioned_pair,
        scaled_coords,
        diffusion_inputs["atom_block_pair_input_feats"],
        diffusion_inputs["atom_single_mask"],
        diffusion_inputs["atom_block_pair_mask"],
        diffusion_inputs["block_indices_h"],
        diffusion_inputs["block_indices_w"],
        diffusion_inputs["atom_token_indices"],
        params.atom_encoder,
    )
    jax.block_until_ready(encoder)
    transformed = _compiled_diffusion_transformer_initialize(
        encoder.token_single,
        conditioned_single,
        params.top_level.structure_projection_weight,
    )
    transformed = _run_diffusion_transformer_staged(
        transformed,
        conditioned_single,
        conditioned_pair,
        diffusion_inputs["token_single_mask"],
        params.transformer,
    )
    transformed, atom_condition = _compiled_diffusion_post_norms(
        transformed, encoder.atom_condition, params.top_level
    )
    update = _compiled_diffusion_atom_decoder(
        transformed,
        encoder.atom_single,
        atom_condition,
        encoder.atom_pair,
        diffusion_inputs["atom_single_mask"],
        diffusion_inputs["atom_block_pair_mask"],
        diffusion_inputs["atom_token_indices"],
        params.atom_decoder,
    ).reshape(position.shape)
    output_scale = (
        sigma_expanded.reshape(sample_count, 1, 1)
        * 16.0
        / jnp.sqrt(sigma_expanded.reshape(sample_count, 1, 1) ** 2 + 256.0)
    )
    skip_scale = 256.0 / (sigma_expanded.reshape(sample_count, 1, 1) ** 2 + 256.0)
    output = position * skip_scale + update * output_scale
    jax.block_until_ready(output)
    return output


@jax.jit
def _compiled_edm_prepare(
    atom_pos, atom_mask, sigma_curr, gamma, noise, rotations, translations
):
    augmented = center_augmentation(atom_pos, atom_mask, rotations, translations)
    sigma_hat = sigma_curr * (1.0 + gamma)
    noise_scale = jnp.sqrt(jnp.maximum(sigma_hat**2 - sigma_curr**2, 1e-6))
    return augmented + 1.003 * noise * noise_scale, sigma_hat


@jax.jit
def _compiled_edm_euler(atom_pos_hat, denoised, sigma_hat, sigma_next):
    derivative = (atom_pos_hat - denoised) / sigma_hat
    return atom_pos_hat + (sigma_next - sigma_hat) * derivative, derivative


@jax.jit
def _compiled_edm_finish(euler, denoised_next, derivative, sigma_hat, sigma_next):
    derivative_next = (euler - denoised_next) / sigma_next
    return euler + (sigma_next - sigma_hat) * ((derivative_next + derivative) / 2.0)


def _run_diffusion_step_staged(
    atom_pos,
    atom_mask,
    sigma_curr,
    sigma_next,
    gamma,
    noise,
    rotations,
    translations,
    diffusion_inputs,
    conditioned_pair,
    params,
):
    atom_pos_hat, sigma_hat = _compiled_edm_prepare(
        atom_pos, atom_mask, sigma_curr, gamma, noise, rotations, translations
    )
    denoised = _run_diffusion_denoiser_staged(
        atom_pos_hat, sigma_hat, diffusion_inputs, conditioned_pair, params
    )
    euler, derivative = _compiled_edm_euler(
        atom_pos_hat, denoised, sigma_hat, sigma_next
    )
    denoised_next = _run_diffusion_denoiser_staged(
        euler, sigma_next, diffusion_inputs, conditioned_pair, params
    )
    output = _compiled_edm_finish(
        euler, denoised_next, derivative, sigma_hat, sigma_next
    )
    jax.block_until_ready(output)
    return output


@jax.jit
def _compiled_diffusion_step(
    atom_pos: jax.Array,
    atom_mask: jax.Array,
    sigma_curr: jax.Array,
    sigma_next: jax.Array,
    gamma: jax.Array,
    noise: jax.Array,
    rotations: jax.Array,
    translations: jax.Array,
    diffusion_inputs: Mapping[str, jax.Array],
    params: FullDiffusionDenoiserParams,
) -> jax.Array:
    sample_count = atom_pos.shape[0]

    def denoise(position: jax.Array, sigma: jax.Array) -> jax.Array:
        return full_diffusion_denoiser(
            **diffusion_inputs,
            atom_noised_coords=position[None],
            noise_sigma=jnp.broadcast_to(sigma, (1, sample_count)),
            params=params,
        )

    return edm_heun_step(
        atom_pos,
        atom_mask,
        sigma_curr=sigma_curr,
        sigma_next=sigma_next,
        gamma=gamma,
        noise=noise,
        rotations=rotations,
        translations=translations,
        denoise=denoise,
        second_order=True,
    )


@jax.jit
def _compiled_confidence(
    single_initial: jax.Array,
    single: jax.Array,
    pair: jax.Array,
    token_mask: jax.Array,
    atom_mask: jax.Array,
    atom_pos: jax.Array,
    token_ref_atom_index: jax.Array,
    atom_token_index: jax.Array,
    atom_within_token_index: jax.Array,
    params: ConfidenceHeadParams,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    return confidence_head_forward(
        single_initial,
        single,
        pair,
        token_mask,
        atom_mask,
        atom_pos,
        token_ref_atom_index,
        atom_token_index,
        atom_within_token_index,
        params,
    )


@jax.jit
def _compiled_confidence_initialize(
    single_initial: jax.Array,
    single: jax.Array,
    pair: jax.Array,
    atom_mask: jax.Array,
    atom_pos: jax.Array,
    token_ref_atom_index: jax.Array,
    params: ConfidenceHeadParams,
) -> tuple[jax.Array, jax.Array]:
    return confidence_head_initialize(
        single_initial,
        single,
        pair,
        atom_mask,
        atom_pos,
        token_ref_atom_index,
        params,
    )


@jax.jit
def _compiled_confidence_block_updates(
    single: jax.Array,
    pair: jax.Array,
    token_mask: jax.Array,
    params: ConfidenceBlockParams,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    return confidence_block_without_triangle_attention(single, pair, token_mask, params)


_compiled_confidence_triangle_tile = jax.jit(
    confidence_triangle_multiplication_direction_tile,
    static_argnames=(
        "incoming",
        "start",
        "size",
        "column_start",
        "column_size",
    ),
)
_compiled_confidence_triangle_output_tile = jax.jit(
    confidence_triangle_multiplication_output_tile
)
_compiled_confidence_transition_projection = jax.jit(
    confidence_transition_projection,
    static_argnames=("second",),
)
_compiled_confidence_transition_output = jax.jit(confidence_transition_output)
_compiled_confidence_attention_bias_projection = jax.jit(
    confidence_attention_pair_bias_projection
)
_compiled_confidence_attention_with_bias = jax.jit(
    confidence_attention_pair_bias_with_bias
)
_compiled_confidence_single_add = jax.jit(
    lambda single, attention, transition: single + attention + transition,
    donate_argnums=(1,),
)


@jax.jit
def _compiled_triangle_attention_prepare(
    pair: jax.Array,
    pair_mask: jax.Array,
    params: TriangleAttentionParams,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    return triangle_attention_prepare(pair, pair_mask, params)


@jax.jit
def _compiled_triangle_attention_query(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    bias: jax.Array,
) -> jax.Array:
    return triangle_attention_query_slice(q, k, v, bias)


@jax.jit
def _compiled_triangle_attention_finalize(
    attended: jax.Array,
    gate: jax.Array,
    params: TriangleAttentionParams,
) -> tuple[jax.Array, jax.Array]:
    return triangle_attention_finalize(attended, gate, params)


@jax.jit
def _compiled_confidence_block_add_pair_updates(
    pair: jax.Array,
    multiplication: jax.Array,
    attention_start: jax.Array,
    attention_end: jax.Array,
    pair_transition: jax.Array,
) -> jax.Array:
    return confidence_block_add_pair_updates(
        pair,
        multiplication,
        attention_start,
        attention_end,
        pair_transition,
    )


def _run_confidence_triangle_multiplication_staged(
    pair, pair_mask, params, *, tile_size=512
):
    rows = []
    for start in range(0, pair.shape[-2], tile_size):
        size = min(tile_size, pair.shape[-2] - start)
        columns = []
        for column_start in range(0, pair.shape[-2], tile_size):
            column_size = min(tile_size, pair.shape[-2] - column_start)
            outgoing = _compiled_confidence_triangle_tile(
                pair,
                pair_mask,
                params,
                incoming=False,
                start=start,
                size=size,
                column_start=column_start,
                column_size=column_size,
            )
            incoming = _compiled_confidence_triangle_tile(
                pair,
                pair_mask,
                params,
                incoming=True,
                start=start,
                size=size,
                column_start=column_start,
                column_size=column_size,
            )
            pair_row = jax.lax.dynamic_slice_in_dim(pair, start, size, axis=-3)
            pair_tile = jax.lax.dynamic_slice_in_dim(
                pair_row, column_start, column_size, axis=-2
            )
            output = _compiled_confidence_triangle_output_tile(
                pair_tile, outgoing, incoming, params
            )
            jax.block_until_ready(output)
            columns.append(output)
        rows.append(jnp.concatenate(columns, axis=-2))
    output = jnp.concatenate(rows, axis=-3)
    jax.block_until_ready(output)
    return output


def _run_confidence_pair_transition_staged(pair, params, *, row_chunk_size=128):
    outputs = []
    for start in range(0, pair.shape[-2], row_chunk_size):
        size = min(row_chunk_size, pair.shape[-2] - start)
        pair_chunk = jax.lax.dynamic_slice_in_dim(pair, start, size, axis=-3)
        first = _compiled_confidence_transition_projection(
            pair_chunk, params, second=False
        )
        second = _compiled_confidence_transition_projection(
            pair_chunk, params, second=True
        )
        output = _compiled_confidence_transition_output(first, second, params)
        jax.block_until_ready(output)
        outputs.append(output)
    return jnp.concatenate(outputs, axis=-3)


def _run_confidence_single_attention_staged(
    single, pair, pair_mask, single_mask, params, *, row_chunk_size=128
):
    bias_chunks = []
    for start in range(0, pair.shape[-2], row_chunk_size):
        size = min(row_chunk_size, pair.shape[-2] - start)
        pair_chunk = jax.lax.dynamic_slice_in_dim(pair, start, size, axis=-3)
        mask_chunk = jax.lax.dynamic_slice_in_dim(pair_mask, start, size, axis=-2)
        bias = _compiled_confidence_attention_bias_projection(
            pair_chunk, mask_chunk, params
        )
        jax.block_until_ready(bias)
        bias_chunks.append(bias)
    bias = jnp.concatenate(bias_chunks, axis=-2)
    output = _compiled_confidence_attention_with_bias(single, bias, single_mask, params)
    jax.block_until_ready(output)
    return output


def _run_staged_confidence_block(
    single: jax.Array,
    pair: jax.Array,
    token_mask: jax.Array,
    params: ConfidenceBlockParams,
    *,
    force_low_memory: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """Run confidence attention through host-enforced bounded query kernels."""
    pair_mask = token_mask[..., :, None] & token_mask[..., None, :]
    q, k, v, gate, bias = _compiled_triangle_attention_prepare(
        pair, pair_mask, params.triangle_attention
    )
    query_count = q.shape[-2]
    attended_chunks = [
        _compiled_triangle_attention_query(
            q[:, :, start : start + TRIANGLE_QUERY_CHUNK_SIZE],
            k,
            v,
            bias[:, :, start : start + TRIANGLE_QUERY_CHUNK_SIZE],
        )
        for start in range(0, query_count, TRIANGLE_QUERY_CHUNK_SIZE)
    ]
    attended = jnp.concatenate(attended_chunks, axis=2)
    attention_start, attention_end = _compiled_triangle_attention_finalize(
        attended, gate, params.triangle_attention
    )
    jax.block_until_ready((attention_start, attention_end))
    del attended, attended_chunks, bias, gate, k, q, v

    if force_low_memory or pair.shape[-2] >= 2048:
        pair_out = _compiled_residual_add(pair, attention_start)
        pair_out = _compiled_residual_add(pair_out, attention_end)
        jax.block_until_ready(pair_out)
        del attention_start, attention_end

        multiplication = _run_confidence_triangle_multiplication_staged(
            pair, pair_mask, params.triangle_multiplication
        )
        pair_out = _compiled_residual_add(pair_out, multiplication)
        jax.block_until_ready(pair_out)
        del multiplication

        pair_transition = _run_confidence_pair_transition_staged(
            pair, params.transition_pair
        )
        pair_out = _compiled_residual_add(pair_out, pair_transition)
        jax.block_until_ready(pair_out)
        del pair_transition

        single_attention = _run_confidence_single_attention_staged(
            single,
            pair,
            pair_mask,
            token_mask,
            params.attention_pair_bias,
        )
        transition_first = _compiled_confidence_transition_projection(
            single, params.transition_single, second=False
        )
        transition_second = _compiled_confidence_transition_projection(
            single, params.transition_single, second=True
        )
        single_transition = _compiled_confidence_transition_output(
            transition_first, transition_second, params.transition_single
        )
        single_out = _compiled_confidence_single_add(
            single, single_attention, single_transition
        )
        jax.block_until_ready((single_out, pair_out))
        return single_out, pair_out

    single_out, multiplication, pair_transition = _compiled_confidence_block_updates(
        single, pair, token_mask, params
    )
    pair_out = _compiled_confidence_block_add_pair_updates(
        pair,
        multiplication,
        attention_start,
        attention_end,
        pair_transition,
    )
    return single_out, pair_out


@jax.jit
def _compiled_confidence_project(
    single: jax.Array,
    pair: jax.Array,
    atom_token_index: jax.Array,
    atom_within_token_index: jax.Array,
    params: ConfidenceHeadParams,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    return confidence_head_project(
        single,
        pair,
        atom_token_index,
        atom_within_token_index,
        params,
    )


def execute_prepared_inference(
    prepared: PreparedInference,
    components: ModelComponents,
    config: InferenceConfig,
) -> TrunkPrediction:
    """Execute the complete native JAX folding, confidence, and ranking path."""
    config.validate()
    config.configure_compilation_cache()
    effective_seed = prepared.realized_seed if config.seed is None else config.seed
    values = _model_padded_inputs(prepared)
    split_msa = config.recycle_msa_subsample > 0 and prepared.model_size < 1536
    msa_mode = (
        "split"
        if split_msa
        else "full"
        if config.recycle_msa_subsample > 0
        else "cropped"
    )
    (
        diffusion_inputs,
        single_initial,
        pair_initial,
        msa_features,
        template_features,
    ) = _embed_and_initialize(
        prepared,
        components,
        msa_mode=msa_mode,
        max_msa_depth=config.max_msa_depth,
    )
    if split_msa:
        jax.block_until_ready(
            (
                diffusion_inputs,
                single_initial,
                pair_initial,
                msa_features,
                template_features,
            )
        )
    token_mask = values["token_exists_mask"]
    pair_mask = token_mask[..., :, None] & token_mask[..., None, :]
    template_mask = values["template_mask"]
    template_pair_mask = template_mask[..., :, None] & template_mask[..., None, :]
    skip_templates = not bool(np.any(prepared.padded_inputs["template_mask"]))
    single, pair = single_initial, pair_initial
    msa_key = jax.random.PRNGKey(effective_seed)
    for _ in range(config.num_trunk_recycles):
        if split_msa:
            msa_key, recycle_key = jax.random.split(msa_key)
            recycle_msa_features, recycle_msa_mask = _embed_msa_for_recycle(
                prepared, components.feature_embedding, key=recycle_key
            )
        elif config.recycle_msa_subsample > 0:
            msa_key, recycle_key = jax.random.split(msa_key)
            recycle_msa_features, recycle_msa_mask = _subsample_msa_for_recycle(
                msa_features,
                prepared.padded_inputs["msa_mask"],
                recycle_key,
                max_depth=config.max_msa_depth,
            )
        else:
            recycle_msa_features, recycle_msa_mask = _crop_trailing_masked_msa_rows(
                msa_features,
                prepared.padded_inputs["msa_mask"],
                max_depth=config.max_msa_depth,
            )
        single, pair = _run_staged_trunk(
            single_initial,
            pair_initial,
            single,
            pair,
            recycle_msa_features,
            recycle_msa_mask,
            template_features,
            template_pair_mask,
            token_mask,
            pair_mask,
            components.trunk,
            skip_templates=skip_templates,
        )

    diffusion_inputs["token_single_trunk_repr"] = single.astype(jnp.float32)
    diffusion_inputs["token_pair_trunk_repr"] = pair.astype(jnp.float32)
    conditioned_pair = (
        _run_diffusion_pair_conditioning(
            diffusion_inputs, components.diffusion.conditioning
        )
        if prepared.model_size >= 2048
        else None
    )
    sample_count = config.num_diffusion_samples
    atom_count = values["atom_exists_mask"].shape[1]
    key = jax.random.PRNGKey(effective_seed)
    key, initial_key = jax.random.split(key)
    sigmas = chai_noise_schedule(config.num_diffusion_timesteps)
    gammas = chai_diffusion_gammas(sigmas, num_timesteps=config.num_diffusion_timesteps)
    atom_pos = sigmas[0] * jax.random.normal(
        initial_key, (sample_count, atom_count, 3), dtype=jnp.float32
    )
    repeated_atom_mask = jnp.repeat(values["atom_exists_mask"], sample_count, axis=0)

    for sigma_curr, sigma_next, gamma in zip(
        sigmas[:-1], sigmas[1:], gammas[:-1], strict=True
    ):
        key, rotation_key, translation_key, noise_key = jax.random.split(key, 4)
        step_args = (
            atom_pos,
            repeated_atom_mask,
            sigma_curr,
            sigma_next,
            gamma,
            jax.random.normal(noise_key, atom_pos.shape, dtype=jnp.float32),
            _random_rotations(rotation_key, sample_count),
            _random_translations(translation_key, sample_count),
            diffusion_inputs,
        )
        atom_pos = (
            _run_diffusion_step_staged(
                *step_args,
                conditioned_pair,
                components.diffusion,
            )
            if conditioned_pair is not None
            else _compiled_diffusion_step(*step_args, components.diffusion)
        )

    pae_logits = []
    pde_logits = []
    plddt_logits = []
    rankings = []
    plddt_bins = _bin_centers(0.0, 1.0, 50)
    pae_bins = _bin_centers(0.0, 32.0, 64)
    for sample_index in range(sample_count):
        confidence_single, confidence_pair = _compiled_confidence_initialize(
            single_initial,
            single,
            pair,
            values["atom_exists_mask"],
            atom_pos[sample_index : sample_index + 1],
            values["token_ref_atom_index"],
            components.confidence,
        )
        for confidence_block in components.confidence.blocks:
            confidence_single, confidence_pair = _run_staged_confidence_block(
                confidence_single,
                confidence_pair,
                token_mask,
                confidence_block,
            )
        pae, pde, plddt = _compiled_confidence_project(
            confidence_single,
            confidence_pair,
            values["atom_token_index"],
            values["atom_within_token_index"],
            components.confidence,
        )
        _, valid_frames = get_frames_and_mask(
            atom_pos[sample_index : sample_index + 1],
            values["token_asym_id"],
            values["token_residue_index"],
            values["token_backbone_frame_mask"],
            values["token_centre_atom_index"],
            token_mask,
            values["atom_exists_mask"],
            values["token_backbone_frame_index"],
            values["atom_token_index"],
        )
        rankings.append(
            rank(
                atom_pos[sample_index : sample_index + 1],
                values["atom_exists_mask"],
                values["atom_token_index"],
                token_mask,
                values["token_asym_id"],
                values["token_entity_type"],
                valid_frames,
                plddt,
                plddt_bins,
                pae,
                pae_bins,
            )
        )
        pae_logits.append(pae[0])
        pde_logits.append(pde[0])
        plddt_logits.append(plddt[0])

    pae_array = jnp.stack(pae_logits)
    pde_array = jnp.stack(pde_logits)
    plddt_array = jnp.stack(plddt_logits)
    pae_scores = jnp.sum(jax.nn.softmax(pae_array, -1) * pae_bins, axis=-1)
    pde_scores = jnp.sum(jax.nn.softmax(pde_array, -1) * pae_bins, axis=-1)
    atom_plddt = jnp.sum(jax.nn.softmax(plddt_array, -1) * plddt_bins, axis=-1)
    token_plddt = contiguous_segment_mean(
        atom_plddt[..., None],
        jnp.broadcast_to(values["atom_exists_mask"], atom_plddt.shape),
        jnp.broadcast_to(values["atom_token_index"], atom_plddt.shape).astype(
            jnp.int32
        ),
        num_segments=prepared.model_size,
    )[..., 0]

    padded_context = prepared.structure_context.pad(prepared.model_size, atom_count)
    return TrunkPrediction(
        atom_coords=atom_pos,
        atom_plddt=atom_plddt,
        atom_mask=values["atom_exists_mask"][0],
        token_mask=token_mask[0],
        ranking_data=rankings,
        pae=pae_scores,
        pde=pde_scores,
        plddt=token_plddt,
        structure_context=padded_context,
    )


def run_inference(
    fasta_file: str | Path,
    *,
    output_dir: str | Path,
    bundle_path: str | Path,
    conformer_path: str | Path,
    config: InferenceConfig | None = None,
    expected_conformer_version: str | None = None,
    backend: InferenceBackend | None = None,
    msa_search_pipeline: MsaSearchPipeline | None = None,
) -> Any:
    """Run standalone Chai-JAX inference and write ranked public outputs."""
    effective_config = InferenceConfig() if config is None else config
    if effective_config.use_msa_server and msa_search_pipeline is None:
        cache_dir = effective_config.msa_cache_directory
        if cache_dir is None:
            cache_dir = Path(output_dir) / "msas"
        remote_options: dict[str, Any] = {}
        if effective_config.use_templates_server:
            remote_options["use_templates"] = True
        msa_search_pipeline = MsaSearchPipeline(
            cache_dir,
            RemoteMMseqs2Backend(
                effective_config.msa_server_url,
                version=effective_config.msa_server_version,
                **remote_options,
            ),
        )
    prepared, assets = prepare_inference(
        fasta_file,
        bundle_path=bundle_path,
        conformer_path=conformer_path,
        config=effective_config,
        expected_conformer_version=expected_conformer_version,
        msa_search_pipeline=msa_search_pipeline,
    )
    components = map_model_components(assets.bundle)
    realized_config = replace(effective_config, seed=prepared.realized_seed)
    if backend is not None:
        return backend(prepared, components, realized_config)
    predictions = []
    for trunk_index in range(effective_config.num_trunk_samples):
        trunk_config = replace(
            effective_config, seed=prepared.realized_seed + trunk_index
        )
        predictions.append(
            execute_prepared_inference(prepared, components, trunk_config)
        )
    candidates = write_prediction_outputs(output_dir, predictions)
    if isinstance(candidates, StructureCandidates):
        msa_mask = np.asarray(prepared.padded_inputs["msa_mask"])[0]
        token_count = prepared.structure_context.num_tokens
        msa_mask = msa_mask[:, :token_count]
        if msa_mask.any():
            plot_directories = (
                [Path(output_dir)]
                if len(predictions) == 1
                else [
                    Path(output_dir) / f"trunk_{index}"
                    for index in range(len(predictions))
                ]
            )
            msa_plot_paths = [
                write_msa_coverage_pdf(
                    prepared.structure_context.token_residue_type,
                    np.asarray(prepared.padded_inputs["msa_tokens"])[
                        0, :, :token_count
                    ],
                    msa_mask,
                    directory / "msa_depth.pdf",
                )
                for directory in plot_directories
            ]
            candidates = replace(candidates, msa_coverage_plot_path=msa_plot_paths[0])
    return candidates


__all__ = [
    "InferenceAssets",
    "InferenceBackend",
    "InferenceConfig",
    "InferenceStageUnavailableError",
    "ModelComponents",
    "PreparedInference",
    "load_inference_assets",
    "map_model_components",
    "execute_prepared_inference",
    "prepare_inference",
    "prepare_inference_from_assets",
    "run_inference",
]
