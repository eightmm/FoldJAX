"""Protenix-JAX adapter."""

from __future__ import annotations

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

_CLI_OPTIONS = {
    "n_sample",
    "n_step",
    "n_cycle",
    "model_name",
    "trunk_dtype",
    "max_msa_rows",
    "diffusion_attention_backend",
    "trunk_single_attention_backend",
    "trunk_triangle_attention_backend",
    "chunk_policy",
    "triangle_mul_chunk_size",
    "triangle_att_q_chunk_size",
    "single_att_q_chunk_size",
    "token_q_chunk_size",
    "diffusion_chunk_size",
}
# Protenix writes "<name>_sample_<rank>.cif" next to
# "<name>_summary_confidence_sample_<rank>.json" in one predictions directory.
_CONFIDENCE_INFIX = "_summary_confidence_sample_"


class ProtenixBackend(Backend):
    name = "protenix"
    sampling_options = {
        "num_samples": "n_sample",
        "num_steps": "n_step",
        "num_recycles": "n_cycle",
        "max_msa_depth": "max_msa_rows",
    }
    compile_options = (
        "n_sample",
        "n_step",
        "model_name",
        "trunk_dtype",
        "max_msa_rows",
        "diffusion_attention_backend",
        "trunk_single_attention_backend",
        "trunk_triangle_attention_backend",
        "chunk_policy",
        "triangle_mul_chunk_size",
        "triangle_att_q_chunk_size",
        "single_att_q_chunk_size",
        "token_q_chunk_size",
        "diffusion_chunk_size",
        "cli_args",
    )

    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            model=self.name,
            sampling=dict(self.sampling_options),
            input_formats=("native", "protenix", "foldjax"),
        )

    def predict(self, request: PredictionRequest) -> PredictionResult:
        options = self.apply_sampling(request)
        argv = [
            "--input-json",
            str(request.input),
            "--weights",
            str(request.weights),
            "--out",
            str(request.output_dir),
            "--output-format",
            str(options.pop("output_format", "protenix")),
            "--seed",
            str(request.seed),
        ]
        if request.cache_dir is not None:
            argv.extend(("--compile-cache", str(request.cache_dir)))
        else:
            # Protenix's native CLI defaults --compile-cache to
            # `outputs/compile_cache`, so leaving the flag off does not turn
            # the cache off -- it relocates it, to a path relative to whatever
            # the working directory happens to be. `--no-cache` has to be said
            # out loud, or a run asked to write nothing still seeds a cache
            # directory next to wherever it was launched.
            argv.append("--no-compile-cache")
        for key in sorted(_CLI_OPTIONS):
            if key in options:
                argv.extend((f"--{key.replace('_', '-')}", str(options.pop(key))))
        argv.extend(str(value) for value in options.pop("cli_args", ()))
        if options:
            raise ValueError(f"unsupported Protenix options: {', '.join(options)}")
        written = import_module("foldjax.models.protenix.cli.predict").main(argv)
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


def _scores(structure_path: Path) -> dict[str, float]:
    """Read the summary confidence JSON Protenix writes beside ``structure_path``."""
    name, separator, rank = structure_path.stem.rpartition("_sample_")
    if not separator:
        return {}
    return scalar_scores(
        structure_path.with_name(f"{name}{_CONFIDENCE_INFIX}{rank}.json")
    )
