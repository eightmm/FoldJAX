"""Running ESMFold2 end to end: weights in, structures out.

The one thing worth knowing before calling this is that ESMFold2 is two models.
The 235M-parameter structure network is what this repository ports and what
`weights_dir("esmfold2")` holds; the representations it folds come from
**ESMC-6B**, a separate 25.4 GB checkpoint that upstream distributes apart from
it. Without ESMC the structure network still runs -- its language-model branch
is simply absent, which is what upstream does when no PLM is loaded -- but it
is not the model anyone benchmarked, so asking for that has to be explicit.

Nothing here imports torch.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from foldjax.models import _capture
from foldjax.models._cp import (
    context_parallel,
    replicate_tree,
)
from foldjax.models._cp import (
    cp_shards as _active_cp_shards,
)
from foldjax.models._feature_storage import compact_msa_storage
from foldjax.models._jit_pool import BoundedJitPool
from foldjax.models.esmfold2.bridge import checkpoint as structure_checkpoint
from foldjax.models.esmfold2.bridge import esmc as esmc_checkpoint
from foldjax.models.esmfold2.data import all_atom as all_atom_featurisation
from foldjax.models.esmfold2.data import features as featurisation
from foldjax.models.esmfold2.models import esmc as esmc_model
from foldjax.models.esmfold2.models import model as structure_model
from foldjax.models.esmfold2.models.segments import MAX_ATOMS_PER_TOKEN

#: Where `assets.py` stages the language model beside the structure weights.
ESMC_SUBDIRECTORY = "esmc"

# Single authority for the semantic arrays consumed by ESMC.  The backend
# hashes this exact vocabulary when it reuses seed-independent hidden states.
LANGUAGE_MODEL_FEATURES = (
    "input_ids",
    "asym_id",
    "residue_index",
    "mol_type",
    "token_attention_mask",
)

# Explicit capability marker for backend wrappers. Merely having a helper with
# a similar name is not enough to authorize the new predict keyword.
COMPACT_LANGUAGE_MODEL_API = True

# Explicit capability marker for the common backend's writer-only graph
# result. Wrappers without this marker keep receiving the historical keyword
# set and the native output mapping.
MANAGED_AUXILIARY_OUTPUT_API = True


@dataclass(frozen=True)
class LoadedModel:
    """Everything a run needs, with the language model optional."""

    parameters: Mapping[str, jnp.ndarray]
    settings: structure_model.ModelSettings
    esmc_parameters: Mapping[str, jnp.ndarray] | None = None
    esmc_settings: esmc_model.ESMCSettings | None = None

    @property
    def has_language_model(self) -> bool:
        return self.esmc_parameters is not None


def esmc_directory(weights: str | Path) -> Path:
    """Where the language model lives relative to the structure weights."""
    return Path(weights) / ESMC_SUBDIRECTORY


def load(
    weights: str | Path,
    *,
    esmc: str | Path | None = None,
    language_model: bool = True,
    dtype: str | None = None,
    esmc_dtype: str | None = "bfloat16",
) -> LoadedModel:
    """Read the checkpoint, and the language model beside it.

    `weights` is the *directory* holding `model.safetensors`, `config.json` and
    `esmc/` -- the configuration is not optional here, since the released one
    departs from upstream's dataclass defaults in most fields. A path to the
    weights file itself is accepted too, because that is what the store's
    `native=` entry resolves to and a caller has no reason to know the
    difference.

    `language_model` is on by default and, when the checkpoint is missing,
    refuses rather than quietly folding a different model: a run without ESMC
    is a legitimate thing to ask for and not a thing to get by accident.
    Turning it off skips the load entirely -- it does not merely tolerate an
    absence, which would make the flag mean different things on two machines.
    """
    weights = Path(weights)
    if weights.is_file():
        weights = weights.parent
    parameters = structure_checkpoint.load_parameters(weights, dtype=dtype)
    settings = structure_checkpoint.load_settings(weights)

    if not language_model:
        return LoadedModel(parameters=parameters, settings=settings)

    directory = Path(esmc) if esmc is not None else esmc_directory(weights)
    if not directory.exists():
        raise FileNotFoundError(
            f"ESMC-6B is not at {directory}. ESMFold2 folds the "
            "representations of a 6B protein language model that upstream "
            "distributes separately (25.4 GB); fetch it with "
            "foldjax weights fetch --model esmfold2, or pass "
            "language_model=False to run the "
            "structure network without it -- which is not the released model."
        )

    return LoadedModel(
        parameters=parameters,
        settings=settings,
        esmc_parameters=esmc_checkpoint.load_parameters(directory, dtype=esmc_dtype),
        esmc_settings=esmc_checkpoint.load_settings(directory),
    )


def language_model_states(
    features: Mapping[str, np.ndarray],
    model: LoadedModel,
    *,
    packed_length: int | None = None,
) -> jnp.ndarray | None:
    """ESMC's stacked hidden states for these tokens, or `None` without it."""
    if model.esmc_parameters is None or model.esmc_settings is None:
        return None
    values = [np.asarray(features[name]) for name in LANGUAGE_MODEL_FEATURES]
    return esmc_model.lm_hidden_states(
        *values,
        model.esmc_parameters,
        settings=model.esmc_settings,
        packed_length=packed_length,
    )


_LANGUAGE_MODEL_EMBEDDING_PARAMETERS = (
    "language_model.base_z_linear.0.weight",
    "language_model.base_z_linear.0.bias",
    "language_model.base_z_linear.1.weight",
    "language_model.base_z_linear.1.bias",
    "language_model.base_z_combine",
)


@lru_cache(maxsize=8)
def _compiled_language_model_embedding(
    compute_dtype: str,
    input_signature: tuple[object, ...],
) -> Callable[[jnp.ndarray, Mapping[str, jnp.ndarray]], jnp.ndarray]:
    """One bounded JIT owner for an exact hidden/parameter signature."""

    del input_signature

    compute = jnp.dtype(compute_dtype)

    def run(
        hidden_states: jnp.ndarray, parameters: Mapping[str, jnp.ndarray]
    ) -> jnp.ndarray:
        trunk_params = structure_model._cast(  # noqa: SLF001
            parameters, structure_model.TRUNK_PREFIXES, compute
        )
        return structure_model.language_model_embedding(
            hidden_states.astype(compute), trunk_params
        )

    return jax.jit(run)


def _language_model_embedding_from_states(
    hidden_states: jnp.ndarray,
    model: LoadedModel,
) -> jnp.ndarray:
    """Project one stack through a bounded, signature-specific JIT owner."""
    parameters = {
        name: model.parameters[name]
        for name in _LANGUAGE_MODEL_EMBEDDING_PARAMETERS
    }
    signature = (
        tuple(hidden_states.shape),
        str(hidden_states.dtype),
        tuple(
            (name, tuple(value.shape), str(value.dtype))
            for name, value in parameters.items()
        ),
    )
    return _compiled_language_model_embedding(
        str(model.settings.trunk_dtype), signature
    )(hidden_states, parameters)


def language_model_embedding(
    features: Mapping[str, np.ndarray],
    model: LoadedModel,
    *,
    packed_length: int | None = None,
) -> jnp.ndarray | None:
    """Return the compact seed-independent ESMC embedding for one input."""

    hidden_states = language_model_states(
        features, model, packed_length=packed_length
    )
    if hidden_states is None:
        return None
    return _language_model_embedding_from_states(hidden_states, model)


def language_model_length(features: Mapping[str, np.ndarray]) -> int:
    """Natural packed ESMC axis for a built structure feature dictionary."""

    return esmc_model.packed_lm_length(
        np.asarray(features["input_ids"]),
        np.asarray(features["asym_id"]),
        np.asarray(features["residue_index"]),
        np.asarray(features["mol_type"]),
        np.asarray(features["token_attention_mask"]),
    )


def _has_contiguous_atom_groups(features: Mapping[str, np.ndarray]) -> bool:
    """Whether the confidence head may use its bounded grouped reducer."""

    try:
        owners = np.asarray(features["atom_to_token"])
        raw_mask = np.asarray(features["atom_attention_mask"])
        mask = raw_mask.astype(bool)
        token_mask = np.asarray(features["token_attention_mask"])
    except (AttributeError, KeyError, TypeError, ValueError):
        return False
    if (
        owners.ndim != 2
        or mask.shape != owners.shape
        or token_mask.ndim != 2
        or token_mask.shape[0] != owners.shape[0]
        or not np.issubdtype(owners.dtype, np.integer)
        or not np.array_equal(raw_mask, mask)
    ):
        return False

    n_tokens = token_mask.shape[-1]
    # `sum_by_token` reserves the value `n_tokens` as its inactive sentinel,
    # so that value itself (not only the last token ID) must fit this dtype.
    if n_tokens < 1 or n_tokens > np.iinfo(owners.dtype).max:
        return False
    prefix = np.arange(owners.shape[-1])[None, :] < np.sum(mask, axis=-1)[:, None]
    if not np.array_equal(mask, prefix):
        return False
    for row, row_mask in zip(owners, mask, strict=True):
        active = row[row_mask]
        if active.size == 0:
            continue
        if (
            np.any(active < 0)
            or np.any(active >= n_tokens)
            or np.any(active[1:] < active[:-1])
        ):
            return False
        index = active.astype(np.intp, copy=False)
        if np.bincount(index, minlength=n_tokens).max() > MAX_ATOMS_PER_TOKEN:
            return False
    return True


def _has_compact_token_bond_encoding(
    features: Mapping[str, np.ndarray],
    parameters: Mapping[str, jnp.ndarray],
    *,
    pair_width: int,
    compute_dtype: object,
) -> bool:
    """Whether the pair-wide projection has an exact compact zero form.

    The package's protein featurizer emits only zero token bonds, and the
    released projection has no bias.  External feature mappings and custom
    checkpoints are less constrained, so they keep the generic graph unless
    the complete input shape, values, supported source/compute dtype route,
    weight shape, post-cast finiteness, and no-bias parameter contract can be
    checked on the host.
    """

    try:
        bonds = np.asarray(features["token_bonds"])
        token_mask = np.asarray(features["token_attention_mask"])
        source_weight = parameters["token_bonds.weight"]
        source_dtype = jnp.dtype(source_weight.dtype)
        compute = jnp.dtype(compute_dtype)
    except (AttributeError, KeyError, TypeError, ValueError):
        return False
    if not (
        jnp.issubdtype(bonds.dtype, jnp.integer)
        or jnp.issubdtype(bonds.dtype, jnp.floating)
    ):
        return False
    # Mirror `_cast`'s supported checkpoint paths exactly. Released FP32
    # weights are narrowed when the trunk computes in BF16; already-BF16
    # weights remain BF16. Other source/compute combinations retain the
    # generic graph, including BF16 source under an FP32 trunk, because
    # `_cast` deliberately does not widen it.
    if compute not in {jnp.dtype(jnp.float32), jnp.dtype(jnp.bfloat16)}:
        return False
    if source_dtype == jnp.dtype(jnp.float32):
        compute_weight = source_weight.astype(compute)
    elif source_dtype == jnp.dtype(jnp.bfloat16) and compute == source_dtype:
        compute_weight = source_weight
    else:
        return False
    if token_mask.ndim != 2:
        return False
    expected = (*token_mask.shape, token_mask.shape[-1])
    if bonds.shape not in {expected, (*expected, 1)}:
        return False
    if (
        compute_weight.shape != (pair_width, 1)
        or "token_bonds.bias" in parameters
    ):
        return False
    try:
        return bool(
            np.all(bonds == 0)
            and not np.any(np.signbit(bonds))
            and np.all(np.isfinite(np.asarray(compute_weight)))
        )
    except (TypeError, ValueError):
        return False


def _model_bound_features(
    features: Mapping[str, np.ndarray],
    *,
    compact_token_bond_encoding: bool,
) -> Mapping[str, Any]:
    """Return the private feature tree transferred into the structure graph.

    The public feature mapping keeps its historical ``token_bonds`` array for
    direct callers and output handling.  Once the host contract above has
    proved that the projection has the exact compact signed-zero form, the
    quadratic all-zero leaf is dead input data: the static graph choice carries
    all information needed to reproduce its projection.  Drop it only from a
    shallow model-bound copy, after the existing conservative MSA narrowing.
    """

    model_features = compact_msa_storage(features)
    if not compact_token_bond_encoding or "token_bonds" not in model_features:
        return model_features
    compact = dict(model_features)
    del compact["token_bonds"]
    return compact


def predict(
    key: jnp.ndarray,
    features: Mapping[str, np.ndarray],
    model: LoadedModel,
    *,
    num_recycles: int | None = None,
    num_samples: int | None = None,
    num_steps: int | None = None,
    max_msa_depth: int | None = None,
    language_model_tokens: int | None = None,
    precomputed_lm_states: jnp.ndarray | None = None,
    precomputed_lm_embedding: jnp.ndarray | None = None,
    compile_it: bool = True,
    preserve_prefix_rng: bool = False,
    #: Context-parallel shard count; same contract as the other ports. More
    #: than one shards the pair state across that many visible JAX devices,
    #: requires the compiled path, and replicates everything token-linear --
    #: the ESMC hidden states or compact embedding and the checkpoint included.
    cp_shards: int = 1,
    return_representations: tuple[str, ...] = (),
    stop_after_trunk: bool = False,
    return_distogram_logits: bool = True,
    return_auxiliary_outputs: bool = True,
) -> dict[str, jnp.ndarray]:
    """One forward over already-built features.

    Compiled by default. Eager JAX dispatches this model an operation at a
    time -- forty-eight trunk layers, four loops, twelve diffusion blocks per
    sampling step -- so the difference is not a tuning detail; `compile_it` is
    there for debugging, where a traced error message is worth the wait.
    """
    settings = structure_model.with_overrides(
        model.settings,
        num_recycles=num_recycles,
        num_samples=num_samples,
        num_steps=num_steps,
        max_msa_depth=max_msa_depth,
    )
    compact_token_bond_encoding = _has_compact_token_bond_encoding(
        features,
        model.parameters,
        pair_width=settings.d_pair,
        compute_dtype=settings.trunk_dtype,
    )
    model_features = _model_bound_features(
        features,
        compact_token_bond_encoding=compact_token_bond_encoding,
    )
    arrays = {
        name: jnp.asarray(value)
        for name, value in model_features.items()
        if name not in all_atom_featurisation.OUTPUT_METADATA_FEATURES
    }
    if precomputed_lm_states is not None and precomputed_lm_embedding is not None:
        raise ValueError(
            "pass precomputed_lm_states or precomputed_lm_embedding, not both"
        )
    compact_lm_input = precomputed_lm_embedding is not None
    hidden = (
        precomputed_lm_embedding
        if compact_lm_input
        else precomputed_lm_states
    )
    if hidden is None:
        hidden = language_model_states(
            features, model, packed_length=language_model_tokens
        )
    # Read on the host: it sizes the confidence head's per-chain matrix, and a
    # traced maximum cannot size anything.
    n_chains = int(np.asarray(features["asym_id"]).max()) + 1
    contiguous_atom_groups = (
        False if stop_after_trunk else _has_contiguous_atom_groups(features)
    )
    if cp_shards > 1 and not compile_it:
        raise ValueError(
            "context parallelism requires the compiled graph; drop "
            "compile_it=False or cp_shards"
        )
    auxiliary_output_kwargs = (
        {} if return_auxiliary_outputs else {"return_auxiliary_outputs": False}
    )
    runner = (
        compiled_predict(
            settings, n_chains, preserve_prefix_rng, cp_shards,
            return_representations, stop_after_trunk, contiguous_atom_groups,
            compact_token_bond_encoding, return_distogram_logits,
            compact_lm_input,
            **auxiliary_output_kwargs,
        )
        if compile_it
        else _run
    )
    parameters = model.parameters
    # A tap records a tracer of the graph being built, so the capture set
    # has to be live while the program is traced, not while it runs.
    with _capture.capturing(return_representations), context_parallel(cp_shards):
        if cp_shards > 1:
            # A checkpoint committed to one device fails the multi-device
            # jit's device-assignment check; everything token-linear is
            # replicated onto the mesh, and the graph's own constraints
            # shard the pair state from its first materialization.
            key = replicate_tree(key)
            arrays = replicate_tree(arrays)
            parameters = replicate_tree(parameters)
            hidden = replicate_tree(hidden)
        return runner(
            key,
            arrays,
            parameters,
            hidden,
            settings,
            n_chains,
            preserve_prefix_rng,
            cp_shards,
            return_representations,
            stop_after_trunk,
            contiguous_atom_groups,
            compact_token_bond_encoding,
            return_distogram_logits,
            compact_lm_input,
            **auxiliary_output_kwargs,
        )


def _run(
    key: jnp.ndarray,
    features: Mapping[str, jnp.ndarray],
    parameters: Mapping[str, jnp.ndarray],
    lm_hidden_states: jnp.ndarray | None,
    settings: structure_model.ModelSettings,
    n_chains: int,
    preserve_prefix_rng: bool,
    cp_shards: int = 1,
    return_representations: tuple[str, ...] = (),
    stop_after_trunk: bool = False,
    contiguous_atom_groups: bool = False,
    compact_token_bond_encoding: bool = False,
    return_distogram_logits: bool = True,
    compact_lm_input: bool = False,
    *,
    return_auxiliary_outputs: bool = True,
) -> dict[str, jnp.ndarray]:
    if cp_shards != _active_cp_shards():
        raise RuntimeError(
            f"cp_shards={cp_shards} but the active context-parallel mesh has "
            f"{_active_cp_shards()} shard(s); run through `predict`, which "
            "activates context_parallel() around this call"
        )
    return structure_model.predict(
        key,
        features,
        parameters,
        settings=settings,
        lm_hidden_states=None if compact_lm_input else lm_hidden_states,
        lm_embedding=lm_hidden_states if compact_lm_input else None,
        n_chains=n_chains,
        preserve_prefix_rng=preserve_prefix_rng,
        return_representations=return_representations,
        stop_after_trunk=stop_after_trunk,
        contiguous_atom_groups=contiguous_atom_groups,
        compact_token_bond_encoding=compact_token_bond_encoding,
        return_distogram_logits=return_distogram_logits,
        return_auxiliary_outputs=return_auxiliary_outputs,
    )


_COMPILED_PREDICT_STATIC_ARGNAMES = (
    "settings",
    "n_chains",
    "preserve_prefix_rng",
    "cp_shards",
    "return_representations",
    "stop_after_trunk",
    "contiguous_atom_groups",
    "compact_token_bond_encoding",
    "return_distogram_logits",
    "compact_lm_input",
    "return_auxiliary_outputs",
)
_compiled_predict_pool = BoundedJitPool(
    _run,
    static_argnames=_COMPILED_PREDICT_STATIC_ARGNAMES,
    limit=8,
)


class _CompiledPredictFacade:
    """One public factory result backed by the shared bounded owner pool."""

    def __call__(self, *args: Any, **kwargs: Any) -> dict[str, jnp.ndarray]:
        return _compiled_predict_pool(*args, **kwargs)

    def lower(self, *args: Any, **kwargs: Any) -> Any:
        return _compiled_predict_pool.lower(*args, **kwargs)

    def clear_cache(self) -> None:
        _compiled_predict_pool.clear_cache()

    def _cache_size(self) -> int:
        return _compiled_predict_pool._cache_size()  # noqa: SLF001


@lru_cache(maxsize=8)
def _compiled_predict_factory(
    settings: structure_model.ModelSettings,
    n_chains: int,
    preserve_prefix_rng: bool = False,
    cp_shards: int = 1,
    return_representations: tuple[str, ...] = (),
    stop_after_trunk: bool = False,
    contiguous_atom_groups: bool = False,
    compact_token_bond_encoding: bool = False,
    return_distogram_logits: bool = True,
    compact_lm_input: bool = False,
    *,
    return_auxiliary_outputs: bool = True,
) -> _CompiledPredictFacade:
    del (
        settings,
        n_chains,
        preserve_prefix_rng,
        cp_shards,
        return_representations,
        stop_after_trunk,
        contiguous_atom_groups,
        compact_token_bond_encoding,
        return_distogram_logits,
        compact_lm_input,
        return_auxiliary_outputs,
    )
    return _CompiledPredictFacade()


def compiled_predict(
    settings: structure_model.ModelSettings,
    n_chains: int,
    preserve_prefix_rng: bool = False,
    cp_shards: int = 1,
    return_representations: tuple[str, ...] = (),
    stop_after_trunk: bool = False,
    contiguous_atom_groups: bool = False,
    compact_token_bond_encoding: bool = False,
    return_distogram_logits: bool = True,
    compact_lm_input: bool = False,
    *,
    return_auxiliary_outputs: bool = True,
) -> Callable[..., dict[str, jnp.ndarray]]:
    """`predict` as one jitted program, cached per settings, chains and RNG mode.

    Those values are static: the settings decide how many layers get traced,
    `n_chains` sizes an output, the RNG mode selects either the historical
    exact-shape calls or masked serving draws, and `cp_shards` decides whether
    the trace carries sharding constraints -- a mesh change must be a retrace,
    never a stale cache hit. The real token and atom counts remain data in the
    masks and never enter this cache key. The trailing booleans select the
    normalized contiguous-atom reducer, the proven-zero token-bond projection,
    native distogram retention, and whether the language-model input is the
    raw layer stack or its separately compiled compact embedding. Each choice
    contributes to a distinct executable identity in the shared eight-entry
    owner pool; the final writer-only result projection does too. The
    lightweight factory facades preserve the historical callable-identity API
    without retaining an unbounded executable set.
    """
    identity = (
        settings,
        n_chains,
        preserve_prefix_rng,
        cp_shards,
        return_representations,
        stop_after_trunk,
        contiguous_atom_groups,
        compact_token_bond_encoding,
        return_distogram_logits,
        compact_lm_input,
    )
    if return_auxiliary_outputs:
        return _compiled_predict_factory(*identity)
    return _compiled_predict_factory(
        *identity,
        return_auxiliary_outputs=False,
    )


def _clear_compiled_predict_cache() -> None:
    _compiled_predict_factory.cache_clear()
    _compiled_predict_pool.clear_cache()


compiled_predict.cache_clear = _clear_compiled_predict_cache  # type: ignore[attr-defined]
compiled_predict.cache_info = _compiled_predict_factory.cache_info  # type: ignore[attr-defined]


def predict_job(
    key: jnp.ndarray,
    chains: Sequence[tuple[str, str, int, int]],
    alignments: Mapping[int, Path] | None,
    model: LoadedModel,
    **overrides: int | None,
) -> tuple[dict[str, jnp.ndarray], dict[str, np.ndarray]]:
    """Featurise a job and fold it, returning `(output, features)`.

    The features come back because everything downstream -- the structure
    writer, the per-atom confidences -- is indexed by them.
    """
    built = featurisation.build_features(chains, dict(alignments or {}))
    return predict(key, built, model, **overrides), built


def build_job_features(
    chains: Sequence[tuple[str, str, int, int]],
    alignments: Mapping[int, Path] | None,
) -> dict[str, np.ndarray]:
    """Build one job without running either ESMC or the structure network."""

    return featurisation.build_features(chains, dict(alignments or {}))


def build_common_job_features(
    document: Mapping[str, object],
    *,
    base_dir: str | Path,
    ccd_path: str | Path,
    seed: int,
    msa_depth: int | None = None,
) -> dict[str, np.ndarray]:
    """Build Biohub's all-biomolecule feature contract from a common job."""

    return all_atom_featurisation.build_job_features(
        document,
        base_dir=base_dir,
        ccd_path=ccd_path,
        seed=seed,
        msa_depth=msa_depth,
    )


def pad_features(
    features: dict[str, np.ndarray],
    *,
    n_token: int,
    n_atom: int,
    n_msa: int,
) -> dict[str, np.ndarray]:
    """Schema-aware NumPy padding, exposed beside the inference entry points."""

    return featurisation.pad_features(
        features, n_token=n_token, n_atom=n_atom, n_msa=n_msa
    )


def normalize_msa_features(
    key: jnp.ndarray,
    features: dict[str, np.ndarray],
    *,
    n_msa: int,
    max_msa_depth: int | None,
    total_steps: int,
) -> dict[str, np.ndarray]:
    """Build the exact released per-loop row-selection tape on the host."""

    mask = np.asarray(features["msa_attention_mask"]).astype(bool)
    if mask.ndim != 3 or mask.shape[0] != 1:
        raise ValueError(
            "ESMFold2 MSA normalization requires one batched alignment with "
            f"shape [1, rows, tokens]; got {mask.shape}"
        )
    active_rows = np.flatnonzero(np.any(mask, axis=-1)[0])
    if active_rows.size == 0 or active_rows[0] != 0:
        raise ValueError(
            "ESMFold2 MSA normalization requires a valid query in row 0"
        )
    compact_indices = msa_loop_row_indices(
        key,
        depth=int(active_rows.size),
        max_msa_depth=max_msa_depth,
        total_steps=total_steps,
    )
    # The released selection is defined over real MSA rows. Storage padding is
    # an archive/layout concern, so map compact selections back only after the
    # exact key split and permutation have been reproduced.
    row_indices = active_rows[compact_indices]
    return featurisation.normalize_msa_features(
        features, n_msa=n_msa, row_indices=row_indices
    )


def msa_loop_row_indices(
    key: jnp.ndarray,
    *,
    depth: int,
    max_msa_depth: int | None,
    total_steps: int,
) -> np.ndarray:
    """Mirror the model's key splits and return each loop's exact MSA rows."""

    if depth < 1:
        raise ValueError(f"ESMFold2 MSA depth must be positive; got {depth}")
    if total_steps < 1:
        raise ValueError(
            f"ESMFold2 total loop steps must be positive; got {total_steps}"
        )

    # `structure_model.predict` splits the public key five ways and passes the
    # fourth key into `run_loops`. Its scan then carries the first child of a
    # three-way split and uses the third child for MSA sampling on each step.
    _, _, _, loop_key, _ = jax.random.split(key, 5)
    all_rows: list[np.ndarray] = []
    for _ in range(total_steps):
        loop_key, _, msa_key = jax.random.split(loop_key, 3)
        selected = structure_model._subsample_msa(
            msa_key, depth, max_msa_depth
        )
        rows = (
            np.arange(depth, dtype=np.int64)
            if selected is None
            else np.asarray(selected, dtype=np.int64)
        )
        all_rows.append(rows)
    return np.stack(all_rows, axis=0)


def seed_key(seed: int) -> jnp.ndarray:
    """One place that decides what a FoldJAX seed means for this model.

    It means rather more here than elsewhere: the trunk's initial pair state,
    the per-loop language-model dropout and the sampler's noise all come off
    this key, so two seeds give genuinely different structures rather than the
    same structure sampled twice.
    """
    return jax.random.key(seed)


__all__ = [
    "ESMC_SUBDIRECTORY",
    "LANGUAGE_MODEL_FEATURES",
    "LoadedModel",
    "build_job_features",
    "build_common_job_features",
    "esmc_directory",
    "language_model_length",
    "language_model_embedding",
    "language_model_states",
    "load",
    "msa_loop_row_indices",
    "normalize_msa_features",
    "pad_features",
    "predict",
    "predict_job",
    "seed_key",
]
