"""High-level FoldJAX API.

``predict`` fills in everything a request left unset — weights from the FoldJAX
weight store, an output directory derived from the input, the shared compile
cache, and the input dialect — so a caller can supply one job file and a model
name and get structures back.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

from foldjax.backends.base import Backend
from foldjax.cache import cache_namespace, runtime_profile, weight_identity
from foldjax.input import materialize_native_input, read_job_document
from foldjax.paths import compile_cache_dir
from foldjax.registry import get_backend
from foldjax.schema import PredictionRequest, PredictionResult

#: Suffixes that can hold the common FoldJAX schema.
_STRUCTURED_SUFFIXES = frozenset({".json", ".yaml", ".yml"})


def detect_input_format(path: Path) -> str:
    """Return ``"foldjax"`` for the common schema, ``"native"`` otherwise.

    Detection is on content, not extension: every backend's native dialect also
    uses JSON or YAML, and only the common schema is a mapping with
    ``entities``. Anything unreadable as JSON/YAML is native by definition.
    """
    if path.suffix.lower() not in _STRUCTURED_SUFFIXES:
        return "native"
    try:
        document = read_job_document(path)
    except (ValueError, json.JSONDecodeError, OSError):
        return "native"
    if isinstance(document, dict) and "entities" in document:
        return "foldjax"
    return "native"


def resolve_request(request: PredictionRequest) -> PredictionRequest:
    """Apply every default without running anything.

    Exposed separately so callers (and `foldjax plan`) can see exactly what a
    bare request turns into before any weights load.
    """
    backend = get_backend(request.model)
    updates: dict[str, Any] = {}

    if request.input_format == "auto":
        updates["input_format"] = detect_input_format(request.input)
    if request.weights is None:
        from foldjax.assets import resolve_weights

        updates["weights"] = resolve_weights(backend.name)
    if request.output_dir is None:
        updates["output_dir"] = Path("foldjax-outputs") / request.input.stem
    if request.cache_dir is None and request.use_compile_cache:
        updates["cache_dir"] = compile_cache_dir()
    return dataclasses.replace(request, **updates) if updates else request


def predict(request: PredictionRequest) -> PredictionResult:
    """Dispatch one request to its selected native backend.

    A request naming several seeds runs the job once per seed, into a
    ``seed_<n>`` subdirectory each, and returns every structure in one result.
    The loop lives here rather than in each adapter because only three of the
    six models take a seed list natively, and a knob that works on half the
    models is not a neutral knob. Each pass reloads the weights; the compiled
    program is read back from the cache, so the repeat cost is the load, not
    the compile.
    """
    request = resolve_request(request)
    seeds = request.resolved_seeds
    if len(seeds) == 1:
        return _predict_once(request, seeds[0], request.output_dir)

    results = [
        _predict_once(request, seed, request.output_dir / f"seed_{seed}")
        for seed in seeds
    ]
    return PredictionResult(
        model=results[0].model,
        samples=tuple(sample for result in results for sample in result.samples),
        output_dir=request.output_dir,
        raw=[result.raw for result in results],
    )


def _predict_once(
    request: PredictionRequest, seed: int, output_dir: Path
) -> PredictionResult:
    """Run the job under exactly one seed, writing into ``output_dir``."""
    request = dataclasses.replace(
        request, seed=seed, seeds=None, output_dir=output_dir
    )
    backend = get_backend(request.model)
    capabilities = backend.capabilities()

    if request.input_format == "foldjax":
        native_input = materialize_native_input(
            request.input,
            capabilities,
            request.output_dir / "inputs",
            seed=request.seed,
        )
        request = dataclasses.replace(
            request, input=native_input, input_format="native"
        )
    if request.input_format not in capabilities.input_formats:
        raise ValueError(
            f"{backend.name} does not support input format {request.input_format!r}"
        )
    if request.cache_dir is not None:
        request = dataclasses.replace(
            request, cache_dir=resolve_cache_dir(request, backend)
        )
    request.output_dir.mkdir(parents=True, exist_ok=True)
    return backend.predict(request)


def resolve_cache_dir(request: PredictionRequest, backend: Backend) -> Path:
    """Return the backend/weight/runtime-specific subtree of ``request.cache_dir``.

    Backends receive an already-namespaced directory, so no backend has to know
    that the root is shared with every other model.
    """
    if request.cache_dir is None:
        raise ValueError("cache_dir is required to resolve a cache namespace")
    label, identity = weight_identity(request.weights)
    return cache_namespace(
        request.cache_dir,
        model=backend.name,
        weight_id=label,
        profile={
            "weights": identity,
            "runtime": runtime_profile(),
            "options": backend.cache_profile(request),
        },
    )
