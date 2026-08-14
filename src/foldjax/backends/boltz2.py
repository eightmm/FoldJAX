"""Boltz2-JAX adapter."""

from __future__ import annotations

from collections.abc import Mapping
from importlib import import_module
from pathlib import Path

import numpy as np

from foldjax.backends.base import Backend
from foldjax.schema import (
    InputRequirement,
    ModelCapabilities,
    PredictionRequest,
    PredictionResult,
    PredictionSample,
    _strict_boolean,
    _strict_integer,
)


def _native_module():
    """Import the Boltz-2 port and resolve its lazy prediction entry point.

    Prediction and featurization are both part of the base, torch-free FoldJAX
    install. Resolve the lazy attribute here so an incomplete installation says
    which base dependency is missing instead of recommending a parity-only extra.
    """
    native = import_module("foldjax.models.boltz2")
    try:
        native.predict
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "the boltz2 backend is included in FoldJAX's base installation, but "
            "one of its required dependencies is missing; reinstall FoldJAX "
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


def _padding_shape_profile(metadata: object) -> dict[str, object] | None:
    """Report every independently compiled stage in a Boltz2 padded run."""

    if not isinstance(metadata, Mapping):
        return None
    primary = metadata.get("primary")
    if not isinstance(primary, Mapping):
        return None
    affinity = metadata.get("affinity")
    if isinstance(affinity, Mapping):
        return {"primary": dict(primary), "affinity": dict(affinity)}
    return dict(primary)


class Boltz2Backend(Backend):
    name = "boltz2"
    padding_axes = ("tokens", "atoms", "msa")
    native_options = frozenset(
        {
            "affinity_diffusion_samples",
            "affinity_mw_correction",
            "affinity_steps",
            "affinity_weights",
            "bucket",
            "feature_cache",
            "glu_backend",
            "mols",
            "msa_api_key_header",
            "msa_api_key_value",
            "msa_pairing_strategy",
            "msa_server_password",
            "msa_server_url",
            "msa_server_username",
            "return_confidence_logits",
            "steering_args",
            "use_msa_server",
            "write_fmt",
        }
    )
    sampling_options = {
        "num_samples": "diffusion_samples",
        "num_steps": "steps",
        "num_recycles": "recycling",
        "max_msa_depth": "max_msa_depth",
    }
    # Neutral knob -> (this port's name, {neutral value: its value}). Boltz-2
    # already spells the values the neutral way; the names are its own.
    execution_options = {
        "dtype": ("compute_dtype", {"float32": "float32", "bfloat16": "bfloat16"}),
        "triangle_kernel": (
            "triangle_backend",
            {"auto": "cueq", "cueq": "cueq", "xla": "xla"},
        ),
        "attention_kernel": (
            "attention_backend",
            {"auto": "xla", "tokamax": "tokamax", "xla": "xla"},
        ),
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

    def validate_native_options(self, options: dict[str, object]) -> None:
        if "steps" in options:
            # The published Karras schedule divides by ``steps - 1``.  One
            # step therefore produces a NaN schedule and only fails after an
            # expensive model compile; reject it while planning instead.
            _strict_integer(options["steps"], name="steps", minimum=2)
        for name in (
            "affinity_mw_correction",
            "bucket",
            "return_confidence_logits",
            "use_msa_server",
        ):
            if name in options:
                _strict_boolean(options[name], name=name)
        if "write_fmt" in options and options["write_fmt"] not in {
            None,
            "cif",
            "pdb",
        }:
            raise ValueError("write_fmt must be one of 'cif', 'pdb', or null")

    def capabilities(self) -> ModelCapabilities:
        requirement = InputRequirement(
            notes=(
                "NumPy featurization and JAX prediction are included in the base "
                "install and never select a second tensor runtime."
            )
        )
        return ModelCapabilities(
            model=self.name,
            sampling=dict(self.sampling_options),
            input_formats=("native", "boltz", "foldjax"),
            input_requirements={
                name: requirement for name in ("native", "boltz", "foldjax")
            },
            supports_affinity=True,
            padding_axes=self.padding_axes,
        )

    def validate_request(self, request: PredictionRequest) -> None:
        if request.padding is not None and "bucket" in request.options:
            raise ValueError(
                "padding and the native Boltz2 option 'bucket' were both set; "
                "pass one of them"
            )
        super().validate_request(request)

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
            padding=request.padding,
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
        shape_profile = _padding_shape_profile(output.get("padding"))
        return PredictionResult(
            model=self.name,
            samples=samples,
            output_dir=request.output_dir,
            raw=output,
            shape_profile=shape_profile,
        )
