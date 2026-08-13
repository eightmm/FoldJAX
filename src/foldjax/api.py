"""High-level FoldJAX API.

``predict`` fills in everything a request left unset — weights from the FoldJAX
weight store, an output directory derived from the input, the shared compile
cache, and the input dialect — so a caller can supply one job file and a model
name and get structures back.
"""

from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path
from typing import Any

from foldjax.backends.base import Backend
from foldjax.cache import cache_namespace, runtime_profile, weight_identity
from foldjax.input import materialize_native_input, read_job_document
from foldjax.manifest import device_peak_bytes
from foldjax.manifest import write as write_manifest
from foldjax.oom import diagnose as diagnose_oom
from foldjax.output import normalize as normalize_output
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
    if request.models is not None or request.inputs is not None:
        raise ValueError(
            "resolve_request takes one model and one input; "
            "predict() is what fans the plural spellings out"
        )
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


def predict(
    request: PredictionRequest,
) -> PredictionResult | tuple[PredictionResult, ...]:
    """Dispatch one request to its selected native backend.

    A request naming several ``models`` or ``inputs`` runs every combination,
    each into ``output_dir/<model>/<input stem>``, and returns one result per
    run, in declaration order. A scalar request returns a single result, as it
    always has.

    A request naming several seeds runs the job once per seed, into a
    ``seed_<n>`` subdirectory each, and returns every structure in one result.
    The loop lives here rather than in each adapter because only three of the
    six models take a seed list natively, and a knob that works on half the
    models is not a neutral knob. Each pass reloads the weights; the compiled
    program is read back from the cache, so the repeat cost is the load, not
    the compile.
    """
    fan = [
        (model, path)
        for model in request.resolved_models
        for path in request.resolved_inputs
    ]
    if len(fan) > 1:
        # The cross product, one full prediction each, namespaced the way the
        # seed loop namespaces below: the fanned axes appear in the path, so
        # two models never write over each other's structures. Sequential on
        # purpose -- these share one GPU.
        root = request.output_dir or Path("foldjax-outputs")
        return tuple(
            predict(
                dataclasses.replace(
                    request,
                    model=model,
                    models=None,
                    input=path,
                    inputs=None,
                    output_dir=root / model / path.stem,
                )
            )
            for model, path in fan
        )
    request = resolve_request(request)
    seeds = request.resolved_seeds
    if len(seeds) == 1:
        return _predict_once(request, seeds[0], request.output_dir)

    started = time.perf_counter()
    results = [
        _predict_once(
            request,
            seed,
            request.output_dir / f"seed_{seed}",
            layout_root=request.output_dir,
        )
        for seed in seeds
    ]
    combined = PredictionResult(
        model=results[0].model,
        samples=tuple(sample for result in results for sample in result.samples),
        output_dir=request.output_dir,
        raw=[result.raw for result in results],
    )
    # Each seed already recorded its own; this one covers the whole request. The
    # peak is the process high-water mark, so it already spans every seed --
    # summing the per-seed peaks would report memory that was never held at once.
    write_manifest(
        request,
        combined,
        request.output_dir,
        cost={
            "seconds": round(time.perf_counter() - started, 2),
            "peak_bytes": device_peak_bytes(),
        },
    )
    return combined


def _job_name(request: PredictionRequest) -> str:
    """What to call this target in file names: the job's own name, or the file's.

    A common-schema document names itself, and that name is what the person
    running it recognizes. Native dialects are not all required to carry one, so
    the input's stem is the fallback -- never the model, which is already in the
    manifest and would make every run's files look alike.
    """
    try:
        document = read_job_document(request.input)
    except (ValueError, OSError):
        document = None
    if isinstance(document, dict):
        name = str(document.get("name") or "").strip()
        if name:
            return name
    return request.input.stem


def _predict_once(
    request: PredictionRequest,
    seed: int,
    output_dir: Path,
    *,
    layout_root: Path | None = None,
) -> PredictionResult:
    """Run the job under exactly one seed, writing into ``output_dir``.

    ``layout_root`` is where the canonical per-sample directories go when that
    is not ``output_dir`` itself -- a multi-seed run keeps each seed's native
    files apart but gathers every structure under one root.
    """
    request = dataclasses.replace(
        request, seed=seed, seeds=None, output_dir=output_dir
    )
    # What the caller asked for, before common-schema input is translated into
    # a backend dialect. The manifest records this rather than the generated
    # file: the generated one lives inside the output directory it describes,
    # so a manifest naming it says nothing about which job was run.
    asked = request
    backend = get_backend(request.model)
    capabilities = backend.capabilities()

    if request.input_format == "foldjax":
        native_input = materialize_native_input(
            request.input,
            capabilities,
            request.output_dir / "inputs",
            seed=request.seed,
        )
        # Most backends have a dialect of their own and the materialised file
        # is in it. ESMFold2 does not -- its adapter reads the common schema
        # directly -- so for it the written file is still FoldJAX's, and
        # relabelling it "native" made every `foldjax predict --model esmfold2`
        # fail the capability check below on a format it had just invented.
        materialized = "native" if "native" in capabilities.input_formats else "foldjax"
        request = dataclasses.replace(
            request, input=native_input, input_format=materialized
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
    started = time.perf_counter()
    try:
        result = backend.predict(request)
    except Exception as error:  # noqa: BLE001 - re-raised, only the message grows
        explanation = diagnose_oom(error)
        if explanation is None:
            raise
        raise MemoryError(f"{backend.name} ran out of memory: {explanation}") from error
    cost = {
        "seconds": round(time.perf_counter() - started, 2),
        "peak_bytes": device_peak_bytes(),
    }
    # Five backends wrote five layouts; this puts every structure in the same
    # place under the same name, and leaves everything else where it was.
    result = normalize_output(result, job=_job_name(asked), root=layout_root)
    # Written after the run, so its presence also says the run finished.
    write_manifest(
        asked,
        result,
        request.output_dir,
        native_input=request.input if request.input != asked.input else None,
        cost=cost,
    )
    return result


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
