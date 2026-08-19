"""OpenDDE-JAX adapter."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from importlib import import_module
from pathlib import Path

from foldjax.backends._representations import _representations_result
from foldjax.backends.base import Backend
from foldjax.models import _representations
from foldjax.schema import (
    InputRequirement,
    ModelCapabilities,
    PredictionRequest,
    PredictionResult,
    PredictionSample,
    _strict_boolean,
)
from foldjax.scores import scalar_scores

# Every environment variable the native CLI assigns. Asserted against its
# source in tests/test_native_contracts.py so a new export cannot escape.
_EXPORTED_ENVIRONMENT = (
    "JAX_PLATFORMS",
    "PROTENIX_CCD_COMPONENTS_FILE",
    "PROTENIX_CCD_RDKIT_MOL_FILE",
    "PROTENIX_KALIGN_BINARY",
    "PROTENIX_TEMPLATE_MMCIF_DIR",
    "PROTENIX_TEMPLATE_OBSOLETE_FILE",
    "PROTENIX_TEMPLATE_RELEASE_DATES_FILE",
)
_CLI_OPTIONS = {
    "ccd_rdkit_cache",
    "components_cif",
    "cp_devices",
    "diffusion_attention_backend",
    "kalign_binary",
    "max_msa_rows",
    "token_q_chunk_size",
    "single_att_q_chunk_size",
    "triangle_att_q_chunk_size",
    "triangle_mul_chunk_size",
    "chunk_policy",
    "n_cycle",
    "n_keys",
    "n_queries",
    "n_sample",
    "n_step",
    "structural_single_attention_backend",
    "template_mmcif_dir",
    "template_obsolete_map",
    "template_release_dates",
    "trunk_dtype",
    "trunk_single_attention_backend",
}
_CONFIDENCE_INFIX = "_summary_confidence_sample_"


class OpenDDEBackend(Backend):
    name = "opendde"
    # OpenDDE has two token spaces: the residue trunk and the expanded
    # structural diffusion branch.  Both, the atom axis, and sampled MSA rows
    # have end-to-end masks and are cropped before public output.
    padding_axes = ("tokens", "atoms", "msa", "structural_tokens")
    native_options = frozenset(_CLI_OPTIONS | {"include_raw"})
    sampling_options = {
        "num_samples": "n_sample",
        "num_steps": "n_step",
        "num_recycles": "n_cycle",
        "max_msa_depth": "max_msa_rows",
    }
    # OpenDDE has no triangle-kernel option of its own -- it drives Protenix's
    # trunk but exposes only the chunk sizes -- so `triangle_kernel` is absent
    # here and asking for it is an error rather than a silent no-op.
    execution_options = {
        "dtype": ("trunk_dtype", {"float32": "fp32", "bfloat16": "bf16"}),
        "attention_kernel": (
            "trunk_single_attention_backend",
            {"auto": "xla_jit", "xla": "xla_jit"},
        ),
    }
    compile_options = tuple(sorted(_CLI_OPTIONS))

    def validate_native_options(self, options: dict[str, object]) -> None:
        _strict_boolean(options.get("include_raw", False), name="include_raw")

    def capabilities(self) -> ModelCapabilities:
        requirement = InputRequirement(
            notes=(
                "NumPy/Gemmi/RDKit featurization and JAX prediction are included "
                "in the base install and do not import PyTorch."
            )
        )
        return ModelCapabilities(
            representations=_representations.available("opendde"),
            model=self.name,
            sampling=dict(self.sampling_options),
            input_formats=("native", "opendde", "foldjax"),
            input_requirements={
                name: requirement for name in ("native", "opendde", "foldjax")
            },
            supports_templates=True,
            padding_axes=self.padding_axes,
        )

    def predict(self, request: PredictionRequest) -> PredictionResult:
        if not request.weights.is_file():
            raise FileNotFoundError(
                f"OpenDDE-JAX weights must be a native weight file: {request.weights}"
            )
        if request.weights.suffix.lower() in {".pt", ".pth", ".ckpt"}:
            raise ValueError(
                "OpenDDE-JAX prediction requires converted native weights; "
                "run opendde-jax-export-weights first"
            )

        options = self.apply_sampling(request)
        include_raw = options.pop("include_raw", False)
        if not isinstance(include_raw, bool):
            raise ValueError("include_raw must be a boolean")

        argv = [
            "--input-json",
            str(request.input),
            "--weights",
            str(request.weights),
            "--out",
            str(request.output_dir),
            "--seed",
            str(request.seed),
        ]
        if request.cache_dir is not None:
            argv.extend(("--compile-cache", str(request.cache_dir)))
        for key in sorted(_CLI_OPTIONS):
            if key in options:
                argv.extend((f"--{key.replace('_', '-')}", str(options.pop(key))))
        if include_raw:
            argv.append("--include-raw")
        wanted = _representations.resolve(
            request.representations, _representations.specs_for("opendde")
        )
        if wanted:
            argv.extend(("--representations", ",".join(wanted)))
            # Pinned so that every backend puts the archive in the same place;
            # each model's own output tree is shaped differently.
            argv.extend(("--representations-dir", str(request.output_dir)))
        if request.stop_after == "trunk":
            argv.extend(("--stop-after", "trunk"))
        if options:
            raise ValueError(f"unsupported OpenDDE options: {', '.join(options)}")

        padding_profiles: list[dict[str, object]] = []
        native = import_module("foldjax.models.opendde.cli.predict")
        with _restored_environment():
            if request.padding is None:
                # Keep the historical one-argument native entry point exact for
                # embedding applications and default-off predictions.
                written = native.main(argv)
            else:
                unsupported = sorted(
                    set(request.padding.explicit_axes) - set(self.padding_axes)
                )
                if unsupported:
                    raise ValueError(
                        "opendde does not support explicit padding axes: "
                        + ", ".join(unsupported)
                    )
                written = native.main(
                    argv,
                    padding=request.padding,
                    padding_profiles=padding_profiles,
                )
        samples = tuple(
            PredictionSample(
                seed=request.seed,
                structure_path=path,
                scores=_scores(path),
            )
            for path in written
            if path.suffix == ".cif"
        )
        shape_profile = _shape_profile(
            padding_profiles,
            padded=request.padding is not None,
        )
        raw: dict[str, object] = {"argv": tuple(argv)}
        if shape_profile is not None:
            raw["padding"] = shape_profile
        return PredictionResult(
            model=self.name,
            samples=samples,
            output_dir=request.output_dir,
            raw=raw,
            shape_profile=shape_profile,
            representations=_representations_result(
                self.name, request.output_dir, wanted
            ),
        )


@contextmanager
def _restored_environment() -> Iterator[None]:
    """Undo the asset environment variables the native CLI exports.

    OpenDDE's CLI hands asset paths to the Protenix featurizer through
    ``os.environ``, which is process-scoped and therefore harmless for its own
    entry point. FoldJAX runs that CLI in-process, so without this a job that
    passes ``components_cif`` would silently leave it applied to every later
    prediction in the same session, including ones for other backends.
    """
    saved = {name: os.environ.get(name) for name in _EXPORTED_ENVIRONMENT}
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _scores(structure_path: Path) -> dict[str, float]:
    name, separator, rank = structure_path.stem.rpartition("_sample_")
    if not separator:
        return {}
    return scalar_scores(
        structure_path.with_name(f"{name}{_CONFIDENCE_INFIX}{rank}.json")
    )


def _shape_profile(
    profiles: list[dict[str, object]],
    *,
    padded: bool,
) -> dict[str, object] | None:
    """Collapse identical native-job profiles without hiding heterogeneous runs."""

    if not padded:
        return None
    if not profiles:
        raise RuntimeError(
            "OpenDDE padding completed without reporting a concrete shape profile"
        )
    first = profiles[0]
    if all(profile == first for profile in profiles[1:]):
        return dict(first)
    return {"per_run": [dict(profile) for profile in profiles]}
