"""ESMFold2 adapter, driving FoldJAX's own JAX port.

ESMFold2 is not an AlphaFold-3 reimplementation with an evolutionary trunk: it
is a diffusion structure head on a linear-recurrence pair trunk, folding the
representations of `ESMC-6B`. Both halves are ported here --
`models/esmfold2/models/` for the 235M-parameter structure network and `esmc`
for the language model -- so this backend, like every other one, needs no torch
at prediction time.

Two things about it differ from the rest of the fleet and reach the interface:

* **The weights are two checkpoints.** The structure network is 940 MB; the
  language model it reads from is a separate 25.4 GB download that upstream
  distributes apart from it, staged at `<weights>/esmc`. Without it the trunk
  still runs with its language-model branch absent, which is what upstream does
  when no PLM is loaded, and which is not the released model -- so it is opt-in
  through `no_language_model` rather than a silent fallback.
* **The model is stochastic by construction.** The trunk starts from a random
  pair state, the language-model pair embedding is dropped out at `p = 0.25`
  *per loop* with training-mode semantics that upstream's release path enables,
  and the sampler adds noise. Two runs at the same seed agree; two seeds do not,
  and that is the model rather than the port.

Upstream ESMFold2 is an all-biomolecule model. FoldJAX implements that released
input contract in NumPy: proteins, DNA, RNA, CCD and SMILES ligands, modified
residues and explicit covalent bonds all reach the same JAX model. Biohub's
verified ``ccd.pkl`` supplies arbitrary component chemistry; an unchanged
protein-only job retains the smaller historical in-package chemistry path.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np

from foldjax.backends._representations import _representations_result
from foldjax.backends.base import MATMUL_PRECISION_OPTION, Backend
from foldjax.manifest import path_stat_identity
from foldjax.models import _representations
from foldjax.padding import PaddingPlan, resolve_axis
from foldjax.schema import (
    InputRequirement,
    ModelCapabilities,
    PaddingConfig,
    PredictionError,
    PredictionRequest,
    PredictionResult,
    PredictionSample,
    _strict_boolean,
)

#: Upstream's released `config.json`, which is what the port reads at load
#: time. These are named here because `capabilities()` reports them and the
#: README table quotes them -- and because they are *not* the dataclass
#: defaults in upstream's source, which say 20 loops and 68 steps.
DEFAULTS = {
    "num_loops": 3,
    "num_sampling_steps": 14,
    "num_diffusion_samples": 32,
    "msa_max_depth": 1024,
}


def _esmc_asset_paths(directory: Path) -> list[Path] | None:
    """Mirror the ESMC loader's index/single/glob checkpoint resolution."""

    config = directory / "config.json"
    index = directory / "model.safetensors.index.json"
    if index.is_file():
        try:
            document = json.loads(index.read_text(encoding="utf-8"))
            mapping = document["weight_map"]
        except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError):
            return None
        if not isinstance(mapping, Mapping) or any(
            not isinstance(value, str) or not value for value in mapping.values()
        ):
            return None
        shards = [
            directory / name for name in sorted(set(mapping.values()))
        ]
        return [config, index, *shards]
    single = directory / "model.safetensors"
    if single.is_file():
        return [config, single]
    shards = sorted(directory.glob("*.safetensors"))
    return [config, *shards] if shards else None


def _model_asset_snapshot(
    weights: Path,
    *,
    esmc: str | Path | None,
    language_model: bool,
) -> tuple[tuple[str, str], ...] | None:
    """Metadata identity of exactly the files ``inference.load`` will read.

    Reading 26 GB merely to decide whether an already-loaded model is reusable
    would erase most of the win.  The manifest layer's stat/tree identity binds
    mode, device, inode, size, mtime and ctime for every relevant entry and
    rejects symlinked/special trees it cannot prove.  Re-stat before every reuse
    catches ordinary replacement or in-place edits without touching payloads.
    """

    weights = Path(weights)
    root = weights.parent if weights.is_file() else weights
    paths = [root / "model.safetensors", root / "config.json"]
    # All-biomolecule jobs read the CCD beside the structure checkpoint. Keep
    # it in the session provenance whenever it is present, while preserving
    # compatibility with external protein-only bundles that predate the CCD
    # requirement and never enter the all-atom feature path.
    ccd = root / "ccd.pkl"
    if ccd.exists():
        paths.append(ccd)
    if language_model:
        lm_paths = _esmc_asset_paths(
            Path(esmc) if esmc is not None else root / "esmc"
        )
        if lm_paths is None:
            return None
        paths.extend(lm_paths)

    records: list[tuple[str, str]] = []
    for path in paths:
        identity = path_stat_identity(path)
        if identity is None:
            return None
        try:
            resolved = path.resolve(strict=True)
        except (OSError, RuntimeError):
            return None
        records.append(
            (
                str(resolved),
                json.dumps(identity, sort_keys=True, separators=(",", ":")),
            )
        )
    return tuple(records)


def _language_model_feature_key(
    features: Mapping[str, np.ndarray],
    packed_length: int | None,
    names: Sequence[str],
) -> str:
    """Content identity of every semantic input read by the ESMC adapter."""

    digest = hashlib.sha256()
    digest.update(f"packed_length={packed_length!r}\n".encode())
    for name in names:
        value = np.ascontiguousarray(np.asarray(features[name]))
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(value.dtype.str.encode())
        digest.update(b"\0")
        digest.update(json.dumps(value.shape, separators=(",", ":")).encode())
        digest.update(b"\0")
        digest.update(memoryview(value).cast("B"))
    return digest.hexdigest()


def _model_source_key(
    weights: Path,
    *,
    esmc: str | Path | None,
    language_model: bool,
) -> tuple[bool, str, str | None]:
    """Stable lexical identity; symlink retargeting must not create a new key."""

    weights = Path(weights)
    return (
        language_model,
        str(weights.absolute()),
        str(Path(esmc).absolute()) if language_model and esmc is not None else None,
    )


def _runtime_placement_key(inference: Any) -> tuple[str, ...]:
    """Identity of the JAX devices on which checkpoint arrays are created."""

    jax_module = getattr(inference, "jax", None)
    if jax_module is None:
        return ("unknown",)
    try:
        configured = str(getattr(jax_module.config, "jax_default_device", None))
        devices = tuple(
            f"{device.platform}:{device.id}:{device.device_kind}"
            for device in jax_module.devices()
        )
    except Exception:  # noqa: BLE001 - placement becomes conservatively unique
        return ("unavailable", str(id(jax_module)))
    return (configured, *devices)


def _padding_plan(
    features: Mapping[str, np.ndarray],
    config: PaddingConfig,
    *,
    msa_max_depth: int | None,
    language_model_tokens: int | None,
) -> PaddingPlan:
    """Resolve every ESMFold2 input axis without changing the feature values."""

    if config.atoms is not None and config.atoms % 32:
        raise ValueError(
            "padding.atoms for ESMFold2 must be a multiple of its 32-atom "
            f"attention block; got {config.atoms}"
        )
    token_mask = np.asarray(features["token_attention_mask"]).astype(bool)
    atom_mask = np.asarray(features["atom_attention_mask"]).astype(bool)
    msa_mask = np.asarray(features["msa_attention_mask"]).astype(bool)
    actual = {
        "tokens": int(token_mask.sum()),
        "atoms": int(atom_mask.sum()),
        "msa": int(np.any(msa_mask, axis=-1).sum()),
    }
    storage = {
        "tokens": int(token_mask.shape[-1]),
        "atoms": int(atom_mask.shape[-1]),
        "msa": int(msa_mask.shape[-2]),
    }
    target = {
        axis: resolve_axis(actual[axis], config, axis, minimum=storage[axis])
        for axis in ("tokens", "atoms")
    }

    if msa_max_depth is not None and msa_max_depth < 1:
        raise ValueError(
            f"ESMFold2 msa_max_depth must be positive; got {msa_max_depth}"
        )
    selected_msa = (
        actual["msa"]
        if msa_max_depth is None
        else min(actual["msa"], msa_max_depth)
    )
    if config.msa is not None:
        target_msa = config.msa
        if msa_max_depth is not None and target_msa > msa_max_depth:
            raise ValueError(
                f"padding.msa={target_msa} exceeds ESMFold2's active "
                f"msa_max_depth={msa_max_depth}; padded rows could be sampled"
            )
    else:
        target_msa = resolve_axis(selected_msa, config, "msa")
        # A non-standard user cap (for example 100) can sit between the shared
        # 64 and 128 buckets. Use that stable cap rather than crossing it.
        if msa_max_depth is not None:
            target_msa = min(target_msa, msa_max_depth)
    if target_msa < selected_msa:
        raise ValueError(
            f"padding.msa={target_msa} would discard rows from ESMFold2's "
            f"selected depth {selected_msa} (active {actual['msa']}, stored "
            f"{storage['msa']}); exact loop-wise normalization is only "
            "possible when padding.msa equals or exceeds the selected depth "
            f"min(active rows, msa_max_depth={msa_max_depth})"
        )
    target["msa"] = target_msa

    if language_model_tokens is not None:
        actual["language_model_tokens"] = language_model_tokens
        storage["language_model_tokens"] = language_model_tokens
        target["language_model_tokens"] = resolve_axis(
            language_model_tokens,
            config,
            "language_model_tokens",
            minimum=language_model_tokens,
        )
    return PaddingPlan(actual=actual, storage=storage, target=target)


def managed_asset_profile(options: Mapping[str, Any]) -> str:
    """Select the managed files implied by ESMFold2's model and LM source.

    This small pure helper is shared with request resolution so the weight store
    and the backend cannot disagree about whether ESMC-6B is required.
    """
    without_lm = _strict_boolean(
        options.get("no_language_model", False), name="no_language_model"
    )
    external_esmc = options.get("esmc_weights") is not None
    if without_lm and external_esmc:
        raise ValueError(
            "esmc_weights cannot be combined with no_language_model=true"
        )
    # An explicit ESMC checkpoint still runs the released structure+LM model,
    # but the managed store only has to provide the structure half. Requiring
    # its own additional 25.4 GB ESMC copy would confuse model variant with
    # where the language-model weights come from.
    return "structure-only" if without_lm or external_esmc else "released"


def apply_managed_profile(
    options: Mapping[str, Any], profile: str
) -> dict[str, Any]:
    """Turn a public prediction profile into an unambiguous ESMFold2 variant."""
    if profile not in {"released", "structure-only"}:
        raise ValueError(
            f"unsupported asset profile {profile!r} for esmfold2; "
            "choose one of released, structure-only"
        )

    merged = dict(options)
    if profile == "structure-only":
        # The profile names the managed *structure* bundle. With no external
        # ESMC it also names the structure-network-only model; with an explicit
        # ESMC it keeps the released LM branch while avoiding a redundant
        # managed 25 GB copy, matching the existing options-only behavior.
        if merged.get("esmc_weights") is not None:
            managed_asset_profile(merged)  # validate no_language_model
            return merged
        if "no_language_model" in merged and not _strict_boolean(
            merged["no_language_model"], name="no_language_model"
        ):
            raise ValueError(
                "profile 'structure-only' conflicts with no_language_model=false"
            )
        merged["no_language_model"] = True
        return merged

    if managed_asset_profile(merged) != "released":
        raise ValueError(
            "profile 'released' conflicts with options that select the "
            "structure-only managed bundle"
        )
    return merged


class ESMFold2Backend(Backend):
    name = "esmfold2"
    session_reuse = True
    padding_axes = ("tokens", "atoms", "msa", "language_model_tokens")
    native_options = frozenset({"cp_devices", "esmc_weights", "no_language_model"})
    # The neutral names, against the port's. `max_msa_depth` is the one that is
    # not simply a rename: the model resubsamples that many MSA rows *per trunk
    # loop* rather than cutting the alignment once, which is the same policy
    # Boltz-2 uses and the opposite of a head-of-file cut.
    sampling_options = {
        "num_samples": "num_samples",
        "num_steps": "num_steps",
        "num_recycles": "num_loops",
        "max_msa_depth": "msa_max_depth",
    }
    # No `attention_kernel`: the port's attention is XLA's, and the fused and
    # cuEquivariance paths the torch model selected between do not exist here.
    # `no_language_model` is not a performance knob -- it changes which model
    # runs -- but it is the only way to fold without the 25.4 GB download, so
    # it is exposed and named for what it does.
    execution_options: dict[str, tuple[str, dict[str, Any]]] = {
        **MATMUL_PRECISION_OPTION,
    }
    compile_options = (
        "num_samples",
        "num_steps",
        "num_loops",
        "msa_max_depth",
        "no_language_model",
        "cp_devices",
    )

    def __init__(self) -> None:
        self._session_open = False
        self._session_active = False
        self._session_poisoned: str | None = None
        self._asset_anchors: dict[
            tuple[bool, str, str | None], tuple[tuple[str, str], ...] | None
        ] = {}
        self._loaded_model: Any | None = None
        self._loaded_model_key: (
            tuple[
                tuple[bool, str, str | None],
                tuple[str, ...],
                tuple[tuple[str, str], ...],
            ]
            | None
        ) = None
        self._lm_embedding: Any | None = None
        self._lm_embedding_key: str | None = None

    @contextmanager
    def session(self, requests: Sequence[PredictionRequest]) -> Iterator[Backend]:
        """Reuse one model and one input's compact ESMC embedding."""

        if self._session_open:
            raise RuntimeError("nested ESMFold2 backend sessions are not supported")
        self._session_open = True
        attempts = sum(len(request.resolved_seeds) for request in requests)
        # A scalar run has nothing to reuse and keeps the historical direct
        # ``predict_job`` route, including compatibility with small wrappers.
        self._session_active = attempts > 1
        try:
            yield self
        finally:
            self.invalidate_session()
            self._asset_anchors.clear()
            self._session_poisoned = None
            self._session_active = False
            self._session_open = False

    def invalidate_session(self) -> None:
        self._lm_embedding = None
        self._lm_embedding_key = None
        self._loaded_model = None
        self._loaded_model_key = None

    def _poison(self, message: str) -> None:
        self.invalidate_session()
        self._session_poisoned = message
        raise PredictionError(message)

    def _request_model_source(
        self, request: PredictionRequest
    ) -> tuple[Path, str | Path | None, bool]:
        options = self.apply_sampling(request)
        managed_asset_profile(options)
        without_lm = _strict_boolean(
            options.get("no_language_model", False), name="no_language_model"
        )
        esmc = options.get("esmc_weights")
        assert request.weights is not None
        return request.weights, esmc, not without_lm

    def _anchor_assets(
        self,
        weights: Path,
        *,
        esmc: str | Path | None,
        language_model: bool,
        require_verifiable: bool = False,
    ) -> tuple[
        tuple[bool, str, str | None], tuple[tuple[str, str], ...] | None
    ]:
        if self._session_poisoned is not None:
            raise PredictionError(self._session_poisoned)
        source = _model_source_key(
            weights, esmc=esmc, language_model=language_model
        )
        snapshot = _model_asset_snapshot(
            weights, esmc=esmc, language_model=language_model
        )
        missing = object()
        expected = self._asset_anchors.get(source, missing)
        if expected is missing:
            self._asset_anchors[source] = snapshot
        elif expected is None:
            # Once this source is unverifiable, keep the entire session on the
            # uncached compatibility path even if its layout later changes.
            snapshot = None
        elif expected is not None and snapshot != expected:
            self._poison(
                "ESMFold2 weights changed while a prediction batch was active"
            )
        if require_verifiable and snapshot is None:
            self._poison(
                "ESMFold2 cannot verify weights used by a resumed prediction"
            )
        return source, snapshot

    def observe_resumed(self, request: PredictionRequest) -> None:
        if not self._session_active:
            return
        weights, esmc, language_model = self._request_model_source(request)
        self._anchor_assets(
            weights,
            esmc=esmc,
            language_model=language_model,
            require_verifiable=True,
        )

    def validate_session(self, request: PredictionRequest) -> None:
        if not self._session_active:
            return
        weights, esmc, language_model = self._request_model_source(request)
        self._anchor_assets(
            weights, esmc=esmc, language_model=language_model
        )

    def _load_model(
        self,
        inference: Any,
        weights: Path,
        *,
        esmc: str | Path | None,
        language_model: bool,
    ) -> Any:
        """Load lazily, and reuse only while the exact asset snapshot matches."""

        if not self._session_active:
            return inference.load(weights, esmc=esmc, language_model=language_model)

        source, snapshot = self._anchor_assets(
            weights,
            esmc=esmc,
            language_model=language_model,
        )
        placement = _runtime_placement_key(inference)
        key = (source, placement, snapshot) if snapshot is not None else None
        if self._loaded_model is not None:
            if key is not None and self._loaded_model_key == key:
                return self._loaded_model
            self.invalidate_session()

        model = inference.load(weights, esmc=esmc, language_model=language_model)
        after = _model_asset_snapshot(weights, esmc=esmc, language_model=language_model)
        if snapshot is None or after is None:
            # Unverifiable trees are still runnable, but not retained.
            return model
        if snapshot != after:
            self._poison("ESMFold2 weights changed while they were being loaded")
        self._loaded_model = model
        self._loaded_model_key = key
        return model

    def _language_model_states(
        self,
        inference: Any,
        features: Mapping[str, np.ndarray],
        model: Any,
        *,
        packed_length: int | None,
    ) -> Any | None:
        """Return raw ESMC states for a legacy split inference wrapper."""

        if not model.has_language_model:
            return None
        # Raw 81-layer stacks are intentionally never retained. Wrappers that
        # predate the compact embedding API remain compatible, but recompute
        # the stack for each seed instead of pinning hundreds of megabytes.
        return inference.language_model_states(
            features, model, packed_length=packed_length
        )

    def _language_model_embedding(
        self,
        inference: Any,
        features: Mapping[str, np.ndarray],
        model: Any,
        *,
        packed_length: int | None,
    ) -> Any | None:
        """Return one compact ESMC embedding, retaining at most one input."""

        if not model.has_language_model:
            return None
        # Derived state is reusable only when this session owns the exact model
        # object that produced it. Unverifiable checkpoints are deliberately
        # loaded afresh and must never participate in an identity cache.
        if not self._session_active or self._loaded_model is not model:
            return inference.language_model_embedding(
                features, model, packed_length=packed_length
            )
        key = _language_model_feature_key(
            features,
            packed_length,
            inference.LANGUAGE_MODEL_FEATURES,
        )
        if self._lm_embedding is not None and self._lm_embedding_key == key:
            return self._lm_embedding
        # Drop the prior compact result before ESMC and the projection allocate
        # the next input. The transient raw stack is owned only by the helper;
        # this session retains the roughly 810-times-smaller combined result.
        self._lm_embedding = None
        self._lm_embedding_key = None
        embedding = inference.language_model_embedding(
            features, model, packed_length=packed_length
        )
        self._lm_embedding = embedding
        self._lm_embedding_key = key
        return embedding

    def validate_native_options(self, options: dict[str, Any]) -> None:
        managed_asset_profile(options)

    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            representations=_representations.available("esmfold2"),
            model=self.name,
            sampling=dict(self.sampling_options),
            input_formats=("foldjax",),
            input_requirements={
                "foldjax": InputRequirement(
                    notes=(
                        "FoldJAX's NumPy featurizer and both JAX model components "
                        "are included in the base install; publisher-reference "
                        "environments are kept outside the runtime package."
                    )
                )
            },
            entity_types=("protein", "dna", "rna", "ligand"),
            supports_templates=False,
            padding_axes=self.padding_axes,
        )

    def predict(self, request: PredictionRequest) -> PredictionResult:
        options = self.apply_sampling(request)
        # Out before the leftover-option check: carried by the scope.
        matmul_precision = self.matmul_precision(options)
        # Reject malformed model-selection options before opening the input or
        # importing JAX. This keeps configuration errors deterministic even
        # when the job path is also unavailable.
        managed_asset_profile(options)
        without_lm = _strict_boolean(
            options.get("no_language_model", False), name="no_language_model"
        )
        # In-package imports, kept inside `predict` for the same reason every
        # other vendored backend does it: to keep `import foldjax` off JAX's
        # import cost.
        inference = import_module("foldjax.models.esmfold2.inference")
        output_module = import_module("foldjax.models.esmfold2.output")

        document, document_base = _job_document(request.input)
        chains, alignments = _chains_from_document(document, document_base)
        all_atom_input = _requires_all_atom_features(document)
        overrides = {
            name: int(options.pop(name))
            for name in ("num_loops", "num_steps", "num_samples", "msa_max_depth")
            if name in options
        }
        if "cp_devices" in options:
            cp_devices = int(options.pop("cp_devices"))
            if cp_devices < 1:
                raise ValueError("cp_devices must be positive")
            overrides["cp_shards"] = cp_devices
        wanted = _representations.resolve(
            request.representations, _representations.specs_for("esmfold2")
        )
        if wanted:
            overrides["return_representations"] = wanted
        if request.stop_after == "trunk":
            overrides["stop_after_trunk"] = True
        # The managed download profile answers *where* ESMC comes from, not
        # whether the model uses it.  An external ESMC checkpoint selects the
        # structure-only managed bundle while still running the released LM
        # branch.
        if (
            without_lm
            and request.padding is not None
            and request.padding.language_model_tokens is not None
        ):
            raise ValueError(
                "padding.language_model_tokens cannot be set when "
                "no_language_model=true"
            )
        options.pop("no_language_model", None)
        esmc = options.pop("esmc_weights", None)
        if options:
            raise ValueError(f"unsupported ESMFold2 options: {', '.join(options)}")

        model = self._load_model(
            inference,
            request.weights,
            esmc=esmc,
            language_model=not without_lm,
        )
        padding_plan = None
        lm_target = None
        split_lm_api = all(
            hasattr(inference, name)
            for name in (
                "LANGUAGE_MODEL_FEATURES",
                "build_job_features",
                "language_model_states",
                "predict",
            )
        )
        compact_lm_api = (
            split_lm_api
            and bool(getattr(inference, "COMPACT_LANGUAGE_MODEL_API", False))
            and hasattr(inference, "language_model_embedding")
        )
        prebuilt_features = None
        if all_atom_input:
            weights_root = Path(request.weights)
            if not weights_root.is_dir():
                weights_root = weights_root.parent
            prebuilt_features = inference.build_common_job_features(
                document,
                base_dir=document_base,
                ccd_path=weights_root / "ccd.pkl",
                seed=request.seed,
            )

        if not all_atom_input and request.padding is None and (
            not self._session_active or not split_lm_api
        ):
            # Preserve the original, public no-padding path exactly.  Besides
            # avoiding an unnecessary API split for ordinary callers, wrappers
            # that expose only ``predict_job`` keep working in a session; they
            # reuse weights but deliberately forgo the derived-state cache.
            with matmul_precision():
                prediction, features = inference.predict_job(
                    inference.seed_key(request.seed),
                    chains,
                    alignments,
                    model,
                    return_distogram_logits=False,
                    **overrides,
                )
        else:
            features = (
                prebuilt_features
                if prebuilt_features is not None
                else inference.build_job_features(chains, alignments)
            )
            prediction_key = inference.seed_key(request.seed)
            lm_tokens = (
                inference.language_model_length(features)
                if model.has_language_model and request.padding is not None
                else None
            )
            configured_msa_depth = overrides.get(
                "msa_max_depth", model.settings.msa_max_depth
            )
            active_msa_depth = (
                configured_msa_depth
                if model.settings.msa_n_layers is not None
                else None
            )
            if request.padding is not None:
                padding_plan = _padding_plan(
                    features,
                    request.padding,
                    msa_max_depth=active_msa_depth,
                    language_model_tokens=lm_tokens,
                )
                features = inference.normalize_msa_features(
                    prediction_key,
                    features,
                    n_msa=padding_plan.target["msa"],
                    msa_max_depth=active_msa_depth,
                    total_steps=max(
                        1,
                        overrides.get("num_loops", model.settings.num_loops) + 1,
                    ),
                )
                features = inference.pad_features(
                    features,
                    n_token=padding_plan.target["tokens"],
                    n_atom=padding_plan.target["atoms"],
                    n_msa=padding_plan.target["msa"],
                )
                lm_target = padding_plan.target.get("language_model_tokens")
            if self._session_active and compact_lm_api:
                lm_input = {
                    "precomputed_lm_embedding": self._language_model_embedding(
                        inference,
                        features,
                        model,
                        packed_length=lm_target,
                    )
                }
            else:
                lm_input = {
                    "precomputed_lm_states": self._language_model_states(
                        inference,
                        features,
                        model,
                        packed_length=lm_target,
                    )
                }
            with matmul_precision():
                prediction = inference.predict(
                    prediction_key,
                    features,
                    model,
                    language_model_tokens=lm_target,
                    preserve_prefix_rng=request.padding is not None,
                    return_distogram_logits=False,
                    **lm_input,
                    **overrides,
                )

        name = Path(request.input).stem
        shape_profile = None
        if padding_plan is not None:
            shape_profile = {
                **padding_plan.summary(),
                "static": {
                    "chains": int(np.asarray(features["asym_id"]).max()) + 1
                },
            }
        _representations.save(
            request.output_dir,
            {name: prediction[name] for name in wanted if name in prediction},
            _representations.specs_for("esmfold2"),
            model="esmfold2",
        )
        raw = {
            "overrides": overrides,
            "language_model": model.has_language_model,
        }
        if shape_profile is not None:
            raw["padding"] = shape_profile
        if request.stop_after == "trunk":
            # Nothing was folded, so there are no samples to describe -- and
            # the writer must stay below this line, not above it. It reads
            # `sample_atom_coords`, which the trunk graph never produces, so
            # reaching it at all raised KeyError before the branch was tested.
            return PredictionResult(
                model=self.name,
                samples=(),
                output_dir=request.output_dir,
                raw=raw,
                representations=_representations_result(
                    self.name, request.output_dir, wanted
                ),
            )
        written = output_module.write_prediction_outputs(
            prediction, features, request.output_dir, name=name
        )
        scores = {entry["sample"]: entry for entry in written["summary"]}
        return PredictionResult(
            model=self.name,
            samples=tuple(
                PredictionSample(
                    seed=request.seed,
                    structure_path=path,
                    scores={
                        key: float(value)
                        for key, value in scores.get(index, {}).items()
                        if key != "sample"
                    },
                )
                for index, path in enumerate(written["structures"])
            ),
            output_dir=request.output_dir,
            raw=raw,
            shape_profile=shape_profile,
            representations=_representations_result(
                self.name, request.output_dir, wanted
            ),
        )


def _job_document(path: Path) -> tuple[dict[str, Any], Path]:
    """Read the validated common document consumed by this native adapter."""

    document: Any = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(document, dict) or "entities" not in document:
        raise ValueError(
            "ESMFold2 takes a FoldJAX job document; it has no native dialect"
        )
    return document, Path(path).parent


def _chains_from_document(
    document: Mapping[str, Any], base: Path
) -> tuple[list[tuple[str, str, int, int]], dict[int, Path]]:
    """Legacy protein chain tuples used by unchanged protein-only jobs."""

    chains: list[tuple[str, str, int, int]] = []
    alignments: dict[int, Path] = {}
    for entity_index, entity in enumerate(document["entities"]):
        if entity.get("type") != "protein":
            continue
        ids = entity.get("id", ["A"])
        ids = ids if isinstance(ids, list) else [ids]
        for symmetry, chain_id in enumerate(ids):
            chains.append((entity["sequence"], str(chain_id), entity_index, symmetry))
        msa = entity.get("unpaired_msa")
        if msa:
            candidate = Path(msa)
            alignments[entity_index] = (
                candidate if candidate.is_absolute() else base / candidate
            )
    return chains, alignments


def _requires_all_atom_features(document: Mapping[str, Any]) -> bool:
    """Whether the official all-biomolecule tokenizer is semantically needed."""

    return bool(document.get("bonds")) or any(
        entity.get("type") != "protein" or entity.get("modifications")
        for entity in document["entities"]
    )


def _job_chains(
    path: Path,
) -> tuple[list[tuple[str, str, int, int]], dict[int, Path]]:
    """One entry per chain copy, and the alignment each entity pins.

    A FoldJAX entity is a sequence and the chains that carry it, which is
    exactly ESMFold2's entity/symmetry split: every copy becomes its own chain
    with its own `asym_id`, sharing an `entity_id` and counting up `sym_id`.
    Alignment paths are resolved against the job file, as everywhere else.
    """
    document, base = _job_document(path)
    chains, alignments = _chains_from_document(document, base)
    if not chains:
        raise ValueError("the job names no protein chains")
    return chains, alignments
