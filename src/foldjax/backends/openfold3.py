"""OpenFold3-JAX adapter.

Unlike the other backends this drives the Python API rather than a CLI, because
OpenFold3 splits featurization from inference on purpose: featurization needs
upstream's data stack, inference needs only JAX and a checkpoint. Shelling out
would force both into one process for no benefit.
"""

from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path
from typing import Any

from foldjax.backends.base import Backend
from foldjax.schema import (
    ModelCapabilities,
    PredictionRequest,
    PredictionResult,
    PredictionSample,
)

# Options that change the compiled program, so they belong in the cache namespace.
# Everything else here only affects what is written.
_COMPILE_OPTIONS = ("num_samples", "no_rollout_steps", "num_cycles", "pair_chunk_size")


class OpenFold3Backend(Backend):
    name = "openfold3"
    # OpenFold3 spells these `no_rollout_steps` and `num_cycles`, which is what
    # `released_config` takes and what `predict` below pops. Mapping them onto
    # their own neutral names put `num_steps` and `num_recycles` into the option
    # dict, where nothing consumed them, so both knobs raised "unsupported
    # OpenFold3 options" -- capabilities advertised two knobs that could only
    # ever fail. `max_msa_depth` is deliberately absent: OpenFold3 exposes no
    # MSA-depth argument, so it is refused rather than quietly ignored.
    sampling_options = {
        "num_samples": "num_samples",
        "num_steps": "no_rollout_steps",
        "num_recycles": "num_cycles",
    }
    compile_options = _COMPILE_OPTIONS

    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            model=self.name,
            sampling=dict(self.sampling_options),
            input_formats=("native", "openfold3", "foldjax"),
        )

    def predict(self, request: PredictionRequest) -> PredictionResult:
        options = self.apply_sampling(request)
        try:
            data = import_module("openfold3_jax.data")
            inference = import_module("openfold3_jax.inference")
            output = import_module("openfold3_jax.output")
            chemistry = import_module("openfold3_jax.bridge.chemistry")
            checkpoint = import_module("openfold3_jax.bridge.checkpoint")
            mapping = import_module("openfold3_jax.bridge.torch_mapping")
        except ModuleNotFoundError as error:
            # It is published on no index and cannot be a dependency, so a bare
            # "No module named 'openfold3_jax'" leaves the reader with no way
            # to find out whether that is a bug, a missing extra, or by design.
            raise ModuleNotFoundError(
                "the openfold3 backend drives an OpenFold3-JAX installation you "
                "provide; it is on no package index, so `uv sync` cannot "
                "install it. See docs/openfold3.md -- the port is also still "
                f"incomplete and has never run end to end here (missing: "
                f"{error.name})"
            ) from error
        jax = import_module("jax")

        spec = json.loads(Path(request.input).read_text(encoding="utf-8"))
        query_id = options.pop("query_id", None)
        features = data.featurize_query(
            spec,
            query_id=query_id,
            seed=request.seed,
            ccd_file_path=options.pop("ccd_file_path", None),
        )
        n_token = features["token_mask"].shape[-1]
        n_atom = features["atom_mask"].shape[-1]

        overrides = {
            key: int(options.pop(key))
            for key in ("num_samples", "no_rollout_steps", "num_cycles")
            if key in options
        }
        chunk = options.pop("pair_chunk_size", None)
        if chunk is not None:
            overrides["pair_chunk_size"] = int(chunk)
        config = inference.released_config(n_token=n_token, n_atom=n_atom, **overrides)

        params = mapping.map_inference_params(
            checkpoint.load_checkpoint(request.weights),
            options.pop("prefix", None),
        )
        compile_it = not bool(options.pop("no_compile", False))
        if options:
            raise ValueError(f"unsupported OpenFold3 options: {', '.join(options)}")

        key = jax.random.key(request.seed)
        table = chemistry.representative_atom_table()
        if compile_it:
            prediction = inference.compile_predict(config, table)(key, features, params)
        else:
            prediction = inference.predict(key, features, params, config, table)

        name = query_id or Path(request.input).stem
        written = output.write_prediction_outputs(
            prediction, features, request.output_dir, name=name
        )
        scores = _scores(written["scores"])
        return PredictionResult(
            model=self.name,
            samples=tuple(
                PredictionSample(
                    seed=request.seed,
                    structure_path=path,
                    scores=scores.get(index, {}),
                )
                for index, path in enumerate(written["structures"])
            ),
            output_dir=request.output_dir,
            raw={"features": {"n_token": n_token, "n_atom": n_atom}},
        )


def _scores(path: Path) -> dict[int, dict[str, float]]:
    """Index the confidence JSON by sample, dropping non-numeric entries."""
    summary: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
    indexed: dict[int, dict[str, float]] = {}
    for entry in summary.get("samples", []):
        index = int(entry.get("sample", -1))
        indexed[index] = {
            key: float(value)
            for key, value in entry.items()
            if key != "sample" and isinstance(value, (int, float))
        }
    return indexed
