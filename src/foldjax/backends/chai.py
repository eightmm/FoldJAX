"""Chai-1 JAX adapter.

The Chai port needs two assets rather than one weight path: a native model bundle
directory and a native conformer archive. FoldJAX resolves both from the single
``weights`` path using the layout Chai-JAX's own asset exporters produce, and
accepts explicit ``bundle_path``/``conformer_path`` options otherwise.
"""

from __future__ import annotations

import dataclasses
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np

from foldjax.backends.base import Backend
from foldjax.schema import (
    ModelCapabilities,
    PredictionRequest,
    PredictionResult,
    PredictionSample,
)

# Scalar entries of Chai's score NPZ. The per-chain arrays stay in ``raw``.
_SCALAR_SCORES = ("aggregate_score", "ptm", "iptm", "has_inter_chain_clashes")


def _resolve_asset(
    weights: Path, override: Any, *, candidates: tuple[Path, ...], option: str
) -> Path:
    if override is not None:
        path = Path(override)
        if not path.exists():
            raise FileNotFoundError(f"chai {option} does not exist: {path}")
        return path
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"cannot resolve chai {option} under {weights}; "
        f"tried {', '.join(str(candidate) for candidate in candidates)}. "
        f"Pass options[{option!r}] explicitly."
    )


class ChaiBackend(Backend):
    name = "chai"
    sampling_options = {
        "num_samples": "num_diffusion_samples",
        "num_steps": "num_diffusion_timesteps",
        "num_recycles": "num_trunk_recycles",
        "max_msa_depth": "max_msa_depth",
    }
    compile_options = (
        "num_trunk_recycles",
        "recycle_msa_subsample",
        "max_msa_depth",
        "num_diffusion_timesteps",
        "num_diffusion_samples",
        "num_trunk_samples",
        "use_esm_embeddings",
        "esm_attention_implementation",
    )

    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            model=self.name,
            sampling=dict(self.sampling_options),
            input_formats=("native", "chai", "foldjax"),
        )

    def predict(self, request: PredictionRequest) -> PredictionResult:
        options = self.apply_sampling(request)
        native = import_module("foldjax.models.chai")
        bundle_path = _resolve_asset(
            request.weights,
            options.pop("bundle_path", None),
            candidates=(
                request.weights / "models" / "chai1",
                request.weights,
            ),
            option="bundle_path",
        )
        conformer_path = _resolve_asset(
            request.weights,
            options.pop("conformer_path", None),
            candidates=(
                request.weights / "conformers.npz",
                request.weights.parent / "conformers.npz",
            ),
            option="conformer_path",
        )
        expected_conformer_version = options.pop("expected_conformer_version", None)
        # A common-schema job that pins an alignment has it written next to the
        # materialized FASTA, because Chai reads user-supplied MSAs from a
        # directory rather than from its input file. An explicit msa_directory
        # option still wins.
        from foldjax.input import CHAI_MSA_DIRNAME

        materialized_msa = request.input.parent / CHAI_MSA_DIRNAME
        if "msa_directory" not in options and materialized_msa.is_dir():
            options["msa_directory"] = materialized_msa
        config = _config(native.InferenceConfig, options, request)
        config.validate()
        # Chai refuses to write into a non-empty directory, and FoldJAX has
        # already materialized a common-schema job under `<output_dir>/inputs`
        # by this point. Give Chai its own subtree, which also matches how the
        # other backends nest their predictions.
        candidates = native.run_inference(
            request.input,
            output_dir=request.output_dir / "predictions",
            bundle_path=bundle_path,
            conformer_path=conformer_path,
            config=config,
            expected_conformer_version=expected_conformer_version,
        )
        return PredictionResult(
            model=self.name,
            samples=_samples(candidates, request.seed),
            output_dir=request.output_dir,
            raw=candidates,
        )


def _config(config_type: type, options: dict[str, Any], request: PredictionRequest):
    """Build a Chai ``InferenceConfig`` from FoldJAX options.

    Option names are Chai's own field names. ``seed`` and
    ``compilation_cache_dir`` are owned by the FoldJAX request, so passing them
    as options is an error rather than a silent override.
    """
    fields = {field.name: field for field in dataclasses.fields(config_type)}
    for owned in ("seed", "compilation_cache_dir"):
        if owned in options:
            raise ValueError(
                f"chai option {owned!r} is set from the FoldJAX request instead"
            )
    unknown = set(options) - set(fields)
    if unknown:
        raise ValueError(f"unsupported chai options: {', '.join(sorted(unknown))}")
    values = {
        name: _coerce(str(fields[name].type), value) for name, value in options.items()
    }
    return config_type(
        seed=request.seed,
        compilation_cache_dir=request.cache_dir,
        **values,
    )


_TRUE_TEXT = frozenset({"true", "yes", "on", "1"})
_FALSE_TEXT = frozenset({"false", "no", "off", "0"})


def _coerce(annotation: str, value: Any) -> Any:
    """Coerce a CLI/JSON option value to the annotated Chai field type."""
    if value is None:
        return None
    if "Path" in annotation:
        return Path(value)
    if annotation.startswith("bool"):
        # `--option use_esm_embeddings=False` reaches here as the string
        # "False", which is truthy. Parse it, and reject anything ambiguous
        # rather than silently turning it on.
        if isinstance(value, str):
            text = value.strip().lower()
            if text in _TRUE_TEXT:
                return True
            if text in _FALSE_TEXT:
                return False
            raise ValueError(f"cannot read {value!r} as a boolean")
        return bool(value)
    if annotation.startswith("int"):
        return int(value)
    if annotation.startswith("str"):
        return str(value)
    return value


def _samples(candidates: Any, seed: int) -> tuple[PredictionSample, ...]:
    """Normalize Chai ``StructureCandidates`` into common samples.

    Chai already ranks its candidates, so sample order is rank order.
    """
    paths = list(getattr(candidates, "cif_paths", ()) or ())
    if not paths:
        return ()
    rankings = list(getattr(candidates, "ranking_data", ()) or ())
    plddt = np.asarray(getattr(candidates, "plddt", np.empty(0)))
    # ``get_scores`` lives on the ``rank`` module; the ``ranking`` package does
    # not re-export it, and three sibling modules define the same name.
    from foldjax.models.chai.ranking.rank import get_scores

    samples = []
    for index, path in enumerate(paths):
        scores: dict[str, float] = {}
        if index < len(rankings):
            native_scores = get_scores(rankings[index])
            for key in _SCALAR_SCORES:
                if key not in native_scores:
                    continue
                value = np.asarray(native_scores[key]).reshape(-1)
                if value.size:
                    scores[key] = float(value[0])
        if plddt.ndim == 2 and index < plddt.shape[0]:
            scores["mean_plddt"] = float(plddt[index].mean())
        samples.append(
            PredictionSample(
                seed=seed,
                structure_path=Path(path),
                scores=scores,
                metadata={"rank": index},
            )
        )
    return tuple(samples)
