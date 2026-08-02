"""OpenDDE-JAX adapter."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from importlib import import_module
from pathlib import Path

from foldjax.backends.base import Backend
from foldjax.schema import (
    ModelCapabilities,
    PredictionRequest,
    PredictionResult,
    PredictionSample,
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
    sampling_options = {
        "num_samples": "n_sample",
        "num_steps": "n_step",
        "num_recycles": "n_cycle",
        "max_msa_depth": "max_msa_rows",
    }
    compile_options = tuple(sorted(_CLI_OPTIONS))

    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            model=self.name,
            sampling=dict(self.sampling_options),
            input_formats=("native", "opendde", "foldjax"),
            supports_templates=True,
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
        if options:
            raise ValueError(f"unsupported OpenDDE options: {', '.join(options)}")

        with _restored_environment():
            written = import_module("foldjax.models.opendde.cli.predict").main(argv)
        samples = tuple(
            PredictionSample(
                seed=request.seed,
                structure_path=path,
                scores=_scores(path),
            )
            for path in written
            if path.suffix == ".cif"
        )
        return PredictionResult(
            model=self.name,
            samples=samples,
            output_dir=request.output_dir,
            raw={"argv": tuple(argv)},
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
