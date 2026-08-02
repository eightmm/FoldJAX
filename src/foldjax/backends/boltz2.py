"""Boltz2-JAX adapter."""

from __future__ import annotations

from collections.abc import Mapping
from importlib import import_module
from pathlib import Path

import numpy as np

from foldjax.backends.base import Backend
from foldjax.schema import (
    ModelCapabilities,
    PredictionRequest,
    PredictionResult,
    PredictionSample,
)


def _native_module():
    """Import the Boltz-2 port and resolve its lazy prediction entry point.

    Boltz-2 runs inference in JAX, but its featurizer reuses upstream Boltz's
    torch/lightning data code. That dependency only surfaces when the lazy
    ``predict`` attribute is touched, so it is resolved here to turn a bare
    ``ModuleNotFoundError`` into an actionable message.
    """
    native = import_module("foldjax.models.boltz2")
    try:
        native.predict
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "the boltz2 backend needs Boltz's featurization dependencies; "
            "install them with `uv sync --extra cuda13 --extra boltz-preprocess` "
            f"(missing: {error.name})"
        ) from error
    return native


#: Scalar confidence the Boltz-2 confidence head reports, one value per
#: diffusion sample. `pae` and the `*_logits` arrays stay in `raw`.
#:
#: `complex_plddt` and `complex_iplddt` are the head's own aggregates,
#: `(plddt * token_pad_mask).sum() / token_pad_mask.sum()`. They are what
#: upstream writes into its confidence JSON, so they are the pLDDT that can be
#: compared against it -- unlike the plain mean below, which weights padding
#: and interface tokens the same as everything else.
_CONFIDENCE_FIELDS = (
    "ptm",
    "iptm",
    "ligand_iptm",
    "protein_iptm",
    "complex_plddt",
    "complex_iplddt",
)


def _sample_scores(
    output: Mapping[str, object],
    plddt: np.ndarray,
    index: int,
    sample_count: int,
) -> dict[str, float]:
    """Confidence for one diffusion sample.

    Every field here is produced per sample, so reading a whole-batch mean or
    always element 0 makes the samples indistinguishable. That is what used to
    happen: `mean_plddt` was the mean over *all* samples and `iptm` was always
    the first one's, and the identical dict was copied onto each sample. Asking
    Boltz-2 for five structures returned five identical score sets, so ranking
    them -- the reason to generate more than one -- silently could not work.
    """
    scores: dict[str, float] = {}
    if plddt.size:
        # [n_sample, n_token] when more than one was requested, else [n_token].
        per_sample = plddt[index] if plddt.ndim == 2 else plddt
        scores["mean_plddt"] = float(per_sample.mean())
    raw = output.get("raw")
    for field in _CONFIDENCE_FIELDS:
        value = output.get(field)
        if value is None and isinstance(raw, Mapping):
            value = raw.get(field)
        if value is None:
            continue
        flat = np.asarray(value).reshape(-1)
        if flat.size == 0:
            continue
        # A per-sample vector is indexed; a single value applies to them all.
        scores[field] = float(flat[index] if flat.size == sample_count else flat[0])
    return scores


def _default_mols(weights: Path) -> Path | None:
    """Find the CCD molecule directory that `foldjax weights fetch` unpacked.

    Boltz reads per-molecule pickles from a directory rather than from the
    checkpoint, so the weight file alone is never enough. The fetcher puts it
    beside the weights; a hand-managed layout can keep it one level up.
    """
    for candidate in (weights.parent / "mols", weights.parent.parent / "mols"):
        if candidate.is_dir():
            return candidate
    return None


class Boltz2Backend(Backend):
    name = "boltz2"
    sampling_options = {
        "num_samples": "diffusion_samples",
        "num_steps": "steps",
        "num_recycles": "recycling",
        "max_msa_depth": "max_msa_depth",
    }
    compile_options = (
        "steps",
        "recycling",
        "diffusion_samples",
        "affinity_steps",
        "affinity_diffusion_samples",
        "compute_dtype",
        "max_msa_depth",
        "attention_backend",
        "triangle_backend",
        "glu_backend",
        "bucket",
    )

    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            model=self.name,
            sampling=dict(self.sampling_options),
            input_formats=("native", "boltz", "foldjax"),
            supports_affinity=True,
        )

    def predict(self, request: PredictionRequest) -> PredictionResult:
        options = self.apply_sampling(request)
        mols = options.pop("mols", None) or _default_mols(request.weights)
        if mols is None:
            raise ValueError(
                "Boltz2 prediction needs its CCD molecule directory. Run "
                "`foldjax weights fetch --model boltz2`, which unpacks it beside "
                "the weights, or pass --option mols=/path/to/mols"
            )
        native = _native_module()
        output = native.predict(
            input=request.input,
            weights=request.weights,
            mols=Path(mols),
            out_dir=request.output_dir,
            seed=request.seed,
            compile_cache=request.cache_dir,
            write_fmt=options.pop("write_fmt", "cif"),
            **options,
        )
        coords = np.asarray(output["coords"])
        plddt = np.asarray(output.get("plddt", []))
        sample_count = coords.shape[0] if coords.ndim == 3 else 1
        paths = output.get("out_paths")
        if paths is None:
            paths = [output.get("out_path")] * sample_count
        samples = tuple(
            PredictionSample(
                seed=request.seed,
                structure_path=Path(paths[index]) if paths[index] else None,
                coordinates=coords[index] if coords.ndim == 3 else coords,
                scores=_sample_scores(output, plddt, index, sample_count),
            )
            for index in range(sample_count)
        )
        return PredictionResult(
            model=self.name,
            samples=samples,
            output_dir=request.output_dir,
            raw=output,
        )
