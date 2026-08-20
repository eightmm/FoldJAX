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

Ligands and nucleic acids are what remains: the model expresses them, but this
adapter does not build their features, so `capabilities()` reports protein
rather than accepting a job it would fold without its ligand. Supported protein
features use the canonical chemistry carried in this package; prediction never
reads upstream's 417 MB `ccd.pkl`.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np

from foldjax.backends._representations import _representations_result
from foldjax.backends.base import Backend
from foldjax.models import _representations
from foldjax.padding import PaddingPlan, resolve_axis
from foldjax.schema import (
    InputRequirement,
    ModelCapabilities,
    PaddingConfig,
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
    execution_options: dict[str, tuple[str, dict[str, Any]]] = {}
    compile_options = (
        "num_samples",
        "num_steps",
        "num_loops",
        "msa_max_depth",
        "no_language_model",
        "cp_devices",
    )

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
            # Deliberately narrow: the model expresses ligands and nucleic
            # acids, but this adapter does not build their features yet, and a
            # job that named them would otherwise be folded as protein alone.
            entity_types=("protein",),
            supports_templates=False,
            padding_axes=self.padding_axes,
        )

    def predict(self, request: PredictionRequest) -> PredictionResult:
        options = self.apply_sampling(request)
        # In-package imports, kept inside `predict` for the same reason every
        # other vendored backend does it: to keep `import foldjax` off JAX's
        # import cost.
        inference = import_module("foldjax.models.esmfold2.inference")
        output_module = import_module("foldjax.models.esmfold2.output")

        chains, alignments = _job_chains(request.input)
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
        managed_asset_profile(options)
        without_lm = _strict_boolean(
            options.get("no_language_model", False), name="no_language_model"
        )
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

        model = inference.load(
            request.weights, esmc=esmc, language_model=not without_lm
        )
        padding_plan = None
        lm_target = None
        if request.padding is None:
            # Preserve the original, public no-padding path exactly.  Besides
            # avoiding an unnecessary API split for ordinary callers, this
            # keeps third-party wrappers that implement ``predict_job`` but do
            # not expose the newer host-side feature helpers compatible.
            prediction, features = inference.predict_job(
                inference.seed_key(request.seed),
                chains,
                alignments,
                model,
                **overrides,
            )
        else:
            features = inference.build_job_features(chains, alignments)
            prediction_key = inference.seed_key(request.seed)
            lm_tokens = (
                inference.language_model_length(features)
                if model.has_language_model
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
            prediction = inference.predict(
                prediction_key,
                features,
                model,
                language_model_tokens=lm_target,
                preserve_prefix_rng=True,
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


def _job_chains(
    path: Path,
) -> tuple[list[tuple[str, str, int, int]], dict[int, Path]]:
    """One entry per chain copy, and the alignment each entity pins.

    A FoldJAX entity is a sequence and the chains that carry it, which is
    exactly ESMFold2's entity/symmetry split: every copy becomes its own chain
    with its own `asym_id`, sharing an `entity_id` and counting up `sym_id`.
    Alignment paths are resolved against the job file, as everywhere else.
    """
    document: Any = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(document, dict) or "entities" not in document:
        raise ValueError(
            "ESMFold2 takes a FoldJAX job document; it has no native dialect"
        )
    chains: list[tuple[str, str, int, int]] = []
    alignments: dict[int, Path] = {}
    base = Path(path).parent
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
    if not chains:
        raise ValueError("the job names no protein chains")
    return chains, alignments
