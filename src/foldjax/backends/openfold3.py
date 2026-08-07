"""OpenFold3-JAX adapter.

Unlike the other vendored backends this drives the port's Python API rather
than its CLI, because OpenFold3 splits featurization from inference on purpose:
featurization delegates to upstream's data stack, inference needs only JAX and
a checkpoint. Shelling out would force both into one process for no benefit.

That split is also the one thing about this backend that is not self-contained.
Prediction from a featurized batch needs nothing beyond FoldJAX's base
dependencies, but building that batch needs the ``openfold3-preprocess`` extra
and an upstream OpenFold3 checkout, which is a directory rather than a package.
`foldjax.models.openfold3.data` raises with both remedies named.
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
_COMPILE_OPTIONS = (
    "num_samples",
    "no_rollout_steps",
    "num_cycles",
    "pair_chunk_size",
    "max_msa_depth",
)

class OpenFold3Backend(Backend):
    name = "openfold3"
    # OpenFold3 spells these `no_rollout_steps` and `num_cycles`, which is what
    # `released_config` takes and what `predict` below pops. Mapping them onto
    # their own neutral names put `num_steps` and `num_recycles` into the option
    # dict, where nothing consumed them, so both knobs raised "unsupported
    # OpenFold3 options" -- capabilities advertised two knobs that could only
    # ever fail. `max_msa_depth` overrides `released_config`'s `msa_depth`, which
    # already carries upstream's own 1024; the knob narrows a setting the model
    # has rather than imposing one it lacks.
    sampling_options = {
        "num_samples": "num_samples",
        "num_steps": "no_rollout_steps",
        "num_recycles": "num_cycles",
        "max_msa_depth": "max_msa_depth",
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
        # The port is vendored, so these are ordinary in-package imports. They
        # stay inside `predict` only to keep `import foldjax` off JAX's import
        # cost, which is the same reason the other vendored backends do it.
        data = import_module("foldjax.models.openfold3.data")
        inference = import_module("foldjax.models.openfold3.inference")
        output = import_module("foldjax.models.openfold3.output")
        chemistry = import_module("foldjax.models.openfold3.bridge.chemistry")
        checkpoint = import_module("foldjax.models.openfold3.bridge.checkpoint")
        mapping = import_module("foldjax.models.openfold3.bridge.torch_mapping")
        compilation = import_module("foldjax.models.openfold3.compilation")
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
        depth = options.pop("max_msa_depth", None)
        if depth is not None:
            overrides["msa_depth"] = int(depth)
        config = inference.released_config(n_token=n_token, n_atom=n_atom, **overrides)
        # Upstream subsamples inside `MSAModuleEmbedder.forward`; this port does it
        # on the host, before the alignment reaches the device. Unconditional, so
        # the default path is the released one rather than a full-depth divergence.
        features = data.subsample_msa_rows(features, config.msa_depth)

        params = mapping.map_inference_params(
            checkpoint.load_checkpoint(request.weights),
            options.pop("prefix", None),
        )
        compile_it = not bool(options.pop("no_compile", False))
        if options:
            raise ValueError(f"unsupported OpenFold3 options: {', '.join(options)}")

        # This backend was the only one that ignored the request's cache
        # directory, which is the one it could least afford to: compiling the
        # released architecture takes minutes and grows with token count, so
        # without a persistent cache every process pays it again. `api.predict`
        # has already namespaced the directory per model, weight identity and
        # compile-relevant options.
        #
        # `enable_compilation_cache` rather than a bare `jax.config.update`,
        # because it also lifts the minimum entry size -- XLA otherwise skips
        # exactly the small-but-slow-to-compile graphs this port produces.
        if compile_it and request.cache_dir is not None:
            compilation.enable_compilation_cache(request.cache_dir)

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
