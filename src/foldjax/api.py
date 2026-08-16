"""High-level FoldJAX API.

``predict`` fills in everything a request left unset — weights from the FoldJAX
weight store, an output directory derived from the input, the shared compile
cache, and the input dialect — so a caller can supply one job file and a model
name and get structures back.
"""

from __future__ import annotations

import dataclasses
import json
import math
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from foldjax import progress
from foldjax.backends.base import Backend
from foldjax.cache import (
    cache_namespace,
    compilation_cache_scope,
    runtime_profile,
    weight_identity,
)
from foldjax.input import materialize_native_input, read_job_document
from foldjax.manifest import MANIFEST_NAME, device_peak_bytes
from foldjax.manifest import write as write_manifest
from foldjax.oom import diagnose as diagnose_oom
from foldjax.output import normalize as normalize_output
from foldjax.paths import compile_cache_dir
from foldjax.registry import get_backend
from foldjax.schema import (
    BatchReport,
    PredictionError,
    PredictionFailure,
    PredictionOutputError,
    PredictionRequest,
    PredictionResult,
    PredictionSample,
)

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
            "use resolve_requests() to inspect plural runs or predict() to execute them"
        )
    backend = get_backend(request.model)
    updates: dict[str, Any] = {}

    if request.model != backend.name:
        updates["model"] = backend.name
    if request.input_format == "auto":
        detected = detect_input_format(request.input)
        # OpenFold3 feature archives are the one binary input dialect with a
        # materially different preprocessing path: they are already featurized,
        # while raw JSON/YAML runs FoldJAX's NumPy chemistry/MSA/template path.
        # Keep `auto` useful for the ordinary `.npz` path and make `plan` report
        # the input path it will actually use.
        if backend.name == "openfold3" and request.input.suffix.lower() == ".npz":
            detected = "openfold3-features"
        updates["input_format"] = detected

    options = dict(request.options)
    requested_profile = request.profile
    if requested_profile is not None:
        from foldjax.assets import available_profiles

        profiles = available_profiles(backend.name)
        if requested_profile not in profiles:
            choices = ", ".join(profiles)
            raise ValueError(
                f"unsupported asset profile {requested_profile!r} for "
                f"{backend.name}; choose one of {choices}"
            )
        if backend.name == "esmfold2":
            from foldjax.backends.esmfold2 import apply_managed_profile

            options = apply_managed_profile(options, requested_profile)
        elif backend.name == "protenix":
            from foldjax.backends.protenix import apply_managed_profile

            # The profile owns the architecture name. The matching ESM/ISM
            # directory is added once the structure-weight path is known.
            options = apply_managed_profile(options, requested_profile)

    asset_profile: str | None = requested_profile
    if backend.name == "esmfold2":
        from foldjax.backends.esmfold2 import managed_asset_profile

        # Validate the model variant even for explicitly supplied weights. This
        # catches ambiguous `no_language_model=true` + `esmc_weights` requests
        # during `plan`, before a large checkpoint is loaded.
        asset_profile = managed_asset_profile(options)
    elif backend.name == "protenix" and (
        request.weights is None or requested_profile is not None
    ):
        from foldjax.backends.protenix import managed_asset_profile

        # Managed mini ESM/ISM checkpoints live with their matching encoder.
        # A first-class profile also configures explicitly supplied weights,
        # while the legacy options-only external-weight path stays untouched.
        asset_profile = managed_asset_profile(options)
    resolved_weights = request.weights
    if request.weights is None:
        from foldjax.assets import resolve_weights

        resolved_weights = resolve_weights(
            backend.name,
            profile=asset_profile,
        )
        updates["weights"] = resolved_weights
        # ``plan`` and run manifests should expose the bundle that was actually
        # selected, including a model's ordinary released default.
        updates["profile"] = asset_profile or "released"
    if backend.name == "protenix" and asset_profile is not None:
        from foldjax.backends.protenix import apply_managed_profile

        assert resolved_weights is not None
        options = apply_managed_profile(
            options,
            asset_profile,
            weights=resolved_weights,
        )
    if options != request.options:
        updates["options"] = options
    if request.output_dir is None:
        # Preserve the scalar API's original output contract. Plural requests
        # add the model namespace in ``resolve_requests`` because they need it
        # to avoid collisions; a one-model call does not.
        updates["output_dir"] = Path("foldjax-outputs") / request.input.stem
    if request.cache_dir is None and request.use_compile_cache:
        updates["cache_dir"] = compile_cache_dir()
    resolved = dataclasses.replace(request, **updates) if updates else request
    backend.validate_request(resolved)
    return resolved


def resolve_requests(request: PredictionRequest) -> tuple[PredictionRequest, ...]:
    """Resolve every scalar run represented by ``request`` without executing it.

    Scalar requests return a one-item tuple. Plural ``models``/``inputs`` form
    their documented cross product and receive collision-free model namespaces
    under the requested output root. ``resolve_request`` remains the convenient
    scalar API and keeps rejecting plural requests.
    """
    plural = request.models is not None or request.inputs is not None
    if not plural:
        return (resolve_request(request),)

    root = request.output_dir or Path("foldjax-outputs")
    runs: list[PredictionRequest] = []
    destinations: dict[Path, tuple[str, Path]] = {}
    canonical_models = tuple(
        get_backend(model).name for model in request.resolved_models
    )
    if len(set(canonical_models)) != len(canonical_models):
        raise ValueError(
            "models resolve to the same backend more than once; remove aliases or "
            "duplicate model names"
        )
    if len(canonical_models) > 1 and request.weights is not None:
        raise ValueError(
            "one explicit weights path cannot be shared by several models; omit "
            "weights to use each model's managed checkpoint, or run them separately"
        )
    for model in canonical_models:
        backend = get_backend(model)
        for path in request.resolved_inputs:
            destination = root / backend.name / path.stem
            previous = destinations.get(destination)
            if previous is not None:
                previous_model, previous_path = previous
                raise ValueError(
                    f"runs ({previous_model}, {previous_path}) and "
                    f"({backend.name}, {path}) share output {destination}; remove "
                    "the duplicate or give same-named inputs separate output roots"
                )
            destinations[destination] = (backend.name, path)
            scalar = dataclasses.replace(
                request,
                model=backend.name,
                models=None,
                input=path,
                inputs=None,
                output_dir=destination,
            )
            runs.append(resolve_request(scalar))
    return tuple(runs)


#: Failures a `continue` policy absorbs. Everything else -- a defect in
#: FoldJAX, a `KeyboardInterrupt` -- still ends the request immediately: one
#: run's bug is not evidence that the next run is worth attempting, and a
#: cancelled batch must stop when it is cancelled.
_RESUMABLE_ERRORS = (
    PredictionError,
    MemoryError,
    ValueError,
    OSError,
    ModuleNotFoundError,
)

#: Where a batch records what did not work. A successful run has
#: `foldjax_run.json`; this is the other half of that story.
FAILURES_NAME = "foldjax_failures.json"


def _result_from_manifest(directory: Path) -> PredictionResult | None:
    """Rebuild a finished run's result from the manifest it left behind.

    A resumed run returns what is on disk rather than nothing, so a caller sees
    one result per requested run whether or not this invocation produced it.
    ``raw`` is absent by construction: model-specific arrays were never written
    to the manifest, and inventing them would be worse than their absence.
    """
    path = Path(directory) / MANIFEST_NAME
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(document, Mapping):
        return None
    samples = []
    for entry in document.get("samples") or []:
        if not isinstance(entry, Mapping):
            continue
        structure = entry.get("structure_path")
        samples.append(
            PredictionSample(
                seed=int(entry.get("seed", 0)),
                structure_path=Path(structure) if structure else None,
                scores={
                    str(key): float(value)
                    for key, value in (entry.get("scores") or {}).items()
                },
                metadata=dict(entry.get("metadata") or {}),
            )
        )
    if not samples:
        return None
    return PredictionResult(
        model=str(document.get("model") or ""),
        samples=tuple(samples),
        output_dir=Path(directory),
        shape_profile=document.get("shape_profile"),
    )


def _write_failures(directory: Path, failures: list[PredictionFailure]) -> None:
    """Record what failed, beside the runs that did not.

    Reported rather than raised, exactly like the manifest: a batch that
    produced seventeen good predictions must not be turned into a failure
    because the note about the other three could not be written.
    """
    if not failures:
        return
    try:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / FAILURES_NAME).write_text(
            json.dumps(
                [failure.summary() for failure in failures], indent=2, sort_keys=True
            )
            + "\n",
            encoding="utf-8",
        )
    except OSError:
        return


def predict_batch(request: PredictionRequest) -> BatchReport:
    """Run everything ``request`` names and report results, skips and failures.

    This is the complete answer; :func:`predict` is the same execution with the
    original return type. Only this one can say *which* runs were reused from a
    previous invocation and which failed, which is what a batch needs in order
    to be resumed or reported on.
    """
    plural = request.models is not None or request.inputs is not None
    if request.output_dir is None:
        output_root = Path.cwd()
    elif plural:
        output_root = Path(request.output_dir)
    else:
        output_root = None
    resolved = resolve_requests(request)
    results: list[PredictionResult] = []
    failures: list[PredictionFailure] = []
    skipped: list[Path] = []
    for item in resolved:
        outcome = _predict_resolved(
            item, output_root=output_root, failures=failures, skipped=skipped
        )
        if outcome is not None:
            results.append(outcome)
    _write_failures(
        Path(request.output_dir) if request.output_dir is not None else Path.cwd(),
        failures,
    )
    return BatchReport(
        results=tuple(results), failures=tuple(failures), skipped=tuple(skipped)
    )


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
    report = predict_batch(request)
    plural = request.models is not None or request.inputs is not None
    if plural:
        return report.results
    if not report.results:
        # A scalar request has nothing to continue *to*, so a policy that lets
        # a batch survive one failure must not turn one failed prediction into
        # a silent success here.
        failure = report.failures[0]
        raise PredictionError(
            f"{failure.model} failed on {failure.input}: {failure.error}"
        )
    return report.results[0]


def _prepare_output_directory(
    path: Path, *, boundary: Path | None = None
) -> None:
    """Create a run directory without following generated child symlinks.

    An explicitly supplied scalar output path is the caller's own boundary and
    may itself be a symlink. Batch model/input and multi-seed directories are
    generated by FoldJAX below that boundary, so accepting a pre-created child
    symlink would let native backends write the whole run somewhere else.
    """
    path = Path(path)
    if boundary is not None:
        boundary = Path(boundary)
        try:
            relative = path.absolute().relative_to(boundary.absolute())
        except ValueError as error:
            raise PredictionOutputError(
                f"generated output directory escapes its run root: {path}"
            ) from error
        cursor = boundary.absolute()
        for part in relative.parts:
            cursor /= part
            if cursor.is_symlink():
                raise PredictionOutputError(
                    f"generated output directory is a symlink: {cursor}"
                )
    if path.exists() and not path.is_dir():
        raise NotADirectoryError(f"output_dir is not a directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    if boundary is not None and not path.resolve().is_relative_to(
        boundary.resolve()
    ):
        raise PredictionOutputError(
            f"generated output directory escapes its run root: {path}"
        )


def _finished(directory: Path) -> bool:
    """Whether a completed run already left its manifest in ``directory``."""
    return (Path(directory) / MANIFEST_NAME).is_file()


def _attempt(
    request: PredictionRequest,
    seed: int,
    directory: Path,
    *,
    layout_root: Path | None,
    allowed_root: Path | None,
    failures: list[PredictionFailure],
    skipped: list[Path],
) -> PredictionResult | None:
    """One seed, honouring the request's resume and error policies."""
    if request.resume and _finished(directory):
        reused = _result_from_manifest(directory)
        if reused is not None:
            skipped.append(Path(directory))
            return reused
    try:
        return _predict_once(
            request,
            seed,
            directory,
            layout_root=layout_root,
            allowed_root=allowed_root,
        )
    except _RESUMABLE_ERRORS as error:
        if request.on_error != "continue":
            raise
        failures.append(
            PredictionFailure(
                model=request.model,
                input=Path(request.input),
                seed=seed,
                output_dir=Path(directory),
                error=str(error),
                error_type=type(error).__name__,
            )
        )
        return None


def _predict_resolved(
    request: PredictionRequest,
    *,
    output_root: Path | None = None,
    failures: list[PredictionFailure] | None = None,
    skipped: list[Path] | None = None,
) -> PredictionResult | None:
    """Run one already-resolved model/input pair, including all requested seeds.

    ``failures`` and ``skipped`` are appended to rather than returned so that
    the call shape stays what every existing caller -- and every test double --
    already uses. ``None`` comes back only when the request's error policy
    absorbed every seed's failure.
    """
    failures = [] if failures is None else failures
    skipped = [] if skipped is None else skipped
    _prepare_output_directory(request.output_dir, boundary=output_root)
    seeds = request.resolved_seeds
    if len(seeds) == 1:
        outcome = _attempt(
            request,
            seeds[0],
            request.output_dir,
            layout_root=None,
            allowed_root=None,
            failures=failures,
            skipped=skipped,
        )
        return outcome

    started = time.perf_counter()
    results = [
        outcome
        for seed in seeds
        if (
            outcome := _attempt(
                request,
                seed,
                request.output_dir / f"seed_{seed}",
                layout_root=request.output_dir,
                allowed_root=request.output_dir,
                failures=failures,
                skipped=skipped,
            )
        )
        is not None
    ]
    if not results:
        return None
    combined = PredictionResult(
        model=results[0].model,
        samples=tuple(sample for result in results for sample in result.samples),
        output_dir=request.output_dir,
        raw=[result.raw for result in results],
        shape_profile=(
            results[0].shape_profile
            if all(
                result.shape_profile == results[0].shape_profile
                for result in results[1:]
            )
            else {"per_seed": [result.shape_profile for result in results]}
        ),
    )
    # Each seed already recorded its own; this one covers the whole request. The
    # peak is the process high-water mark, so it already spans every seed --
    # summing the per-seed peaks would report memory that was never held at once.
    #
    # Written only when every seed succeeded. The manifest's presence means "this
    # run finished", and `resume` reads it as exactly that: writing one for a
    # partial run would make the next attempt skip the pair and never produce
    # the seeds that are still missing.
    if not failures:
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
    allowed_root: Path | None = None,
) -> PredictionResult:
    """Run the job under exactly one seed, writing into ``output_dir``.

    ``layout_root`` is where the canonical per-sample directories go when that
    is not ``output_dir`` itself -- a multi-seed run keeps each seed's native
    files apart but gathers every structure under one root.
    """
    request = dataclasses.replace(
        request,
        seed=seed,
        seeds=None,
        num_seeds=None,
        output_dir=output_dir,
    )
    # What the caller asked for, before common-schema input is translated into
    # a backend dialect. The manifest records this rather than the generated
    # file: the generated one lives inside the output directory it describes,
    # so a manifest naming it says nothing about which job was run.
    asked = request
    backend = get_backend(request.model)
    capabilities = backend.capabilities()

    # Keep direct execution and ``foldjax plan`` on the same cheap validation
    # boundary, and do it before creating outputs or materialising native input.
    backend.validate_request(request)

    # Validate and create the native output root before input materialisation:
    # that step writes generated dialect files (and, for OpenFold3, MSA links)
    # below the same directory.
    _prepare_output_directory(request.output_dir, boundary=allowed_root)

    # Started here rather than at the model call, so `seconds` and the sum of
    # `phases` describe the same span: preparing input is part of what the run
    # cost, and for a searched alignment it can be most of it.
    started = time.perf_counter()
    timeline = progress.Timeline()
    progress.header(backend.name, request.input.name, request.seed)
    if request.input_format == "foldjax":
        with timeline.stage("prepare input"):
            native_input = materialize_native_input(
                request.input,
                capabilities,
                request.output_dir / "inputs",
                seed=request.seed,
                msa=request.msa,
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
    try:
        with (
            timeline.stage("predict"),
            compilation_cache_scope(request.cache_dir),
        ):
            result = backend.predict(request)
    except SystemExit as error:
        explanation = diagnose_oom(error)
        if explanation is not None:
            raise MemoryError(
                f"{backend.name} ran out of memory: {explanation}"
            ) from error
        detail = str(error).strip() or f"exit status {error.code}"
        raise PredictionError(
            f"{backend.name} stopped its native runner: {detail}"
        ) from error
    except Exception as error:  # noqa: BLE001 - re-raised, only the message grows
        explanation = diagnose_oom(error)
        if explanation is None:
            raise
        raise MemoryError(f"{backend.name} ran out of memory: {explanation}") from error
    result = _validate_result(
        result,
        backend=backend,
        output_dir=request.output_dir,
        expected_seed=request.seed,
    )
    if request.padding is not None and result.shape_profile is None:
        raise PredictionOutputError(
            f"{backend.name} accepted padding but did not report the concrete "
            "shape profile it executed"
        )
    # Six backends wrote six layouts; this puts every structure in the same
    # place under the same name, and leaves everything else where it was.
    with timeline.stage("write"):
        result = normalize_output(
            result,
            job=_job_name(asked),
            root=layout_root or request.output_dir,
        )
    cost = {
        "seconds": round(time.perf_counter() - started, 2),
        "peak_bytes": device_peak_bytes(),
        # One number for the run cannot say whether the eleven minutes went to
        # an alignment search, a cold compile, or the sample schedule, and those
        # call for three different responses.
        "phases": timeline.summary(),
    }
    # Backends own their native layout, but the common result always reports
    # the directory this scalar request actually ran in.
    result = dataclasses.replace(result, output_dir=request.output_dir)
    # Written after the run, so its presence also says the run finished.
    write_manifest(
        asked,
        result,
        request.output_dir,
        native_input=request.input if request.input != asked.input else None,
        cost=cost,
    )
    return result


def _has_coordinates(value: Any) -> bool:
    """Whether a coordinates-only sample is a finite numeric ``[..., 3]`` array."""
    if value is None or isinstance(value, (str, bytes, bytearray, Mapping)):
        return False
    try:
        coordinates = np.asarray(value)
    except (TypeError, ValueError, OverflowError):
        return False
    if (
        coordinates.size < 3
        or coordinates.ndim == 0
        or coordinates.shape[-1] != 3
        or coordinates.dtype.kind not in "fiu"
    ):
        return False
    return bool(np.isfinite(coordinates).all())


def _sample_index(sample: Any, fallback: int) -> int:
    """Mirror the output layout's reported-index fallback for validation."""
    metadata = sample.metadata
    value = metadata.get("sample") if isinstance(metadata, Mapping) else None
    if value is None:
        return fallback
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError("sample index must be a non-negative integer")
    index = int(value)
    if index < 0:
        raise ValueError("sample index must be a non-negative integer")
    return index


def _validate_result(
    result: PredictionResult,
    *,
    backend: Backend,
    output_dir: Path,
    expected_seed: int,
) -> PredictionResult:
    """Refuse a nominally successful backend call that produced no prediction."""
    if not isinstance(result, PredictionResult):
        raise PredictionOutputError(
            f"{backend.name} returned {type(result).__name__}, not PredictionResult"
        )
    if result.shape_profile is not None and not isinstance(
        result.shape_profile, Mapping
    ):
        raise PredictionOutputError(
            f"{backend.name} shape_profile must be a mapping when present"
        )
    if result.model != backend.name:
        raise PredictionOutputError(
            f"{backend.name} returned a result labelled {result.model!r}"
        )
    if not isinstance(result.samples, tuple):
        raise PredictionOutputError(
            f"{backend.name} returned samples as {type(result.samples).__name__}; "
            "PredictionResult.samples must be a tuple"
        )
    if not result.samples:
        raise PredictionOutputError(
            f"{backend.name} returned no prediction samples in {output_dir}"
        )
    structures: dict[tuple[int, int], int] = {}
    slots: dict[tuple[int, int], int] = {}
    validated_samples: list[PredictionSample] = []
    for index, sample in enumerate(result.samples):
        if not isinstance(sample, PredictionSample):
            raise PredictionOutputError(
                f"{backend.name} sample {index} is "
                f"{type(sample).__name__}, not PredictionSample"
            )
        if not isinstance(sample.scores, Mapping) or not isinstance(
            sample.metadata, Mapping
        ):
            raise PredictionOutputError(
                f"{backend.name} sample {index} scores and metadata must be mappings"
            )
        if (
            isinstance(sample.seed, bool)
            or not isinstance(sample.seed, (int, np.integer))
            or int(sample.seed) < 0
        ):
            raise PredictionOutputError(
                f"{backend.name} sample {index} seed must be a non-negative integer"
            )
        seed = int(sample.seed)
        if seed != expected_seed:
            raise PredictionOutputError(
                f"{backend.name} sample {index} reports seed {seed}, "
                f"but the run used seed {expected_seed}"
            )
        invalid_scores = [
            key
            for key, value in sample.scores.items()
            if not isinstance(key, str)
            or isinstance(value, bool)
            or not isinstance(value, (int, float, np.integer, np.floating))
            or not bool(np.isfinite(value))
        ]
        if invalid_scores:
            raise PredictionOutputError(
                f"{backend.name} sample {index} has non-numeric or non-finite "
                f"scores: {invalid_scores}"
            )
        scores = {key: float(value) for key, value in sample.scores.items()}
        if not all(math.isfinite(value) for value in scores.values()):
            raise PredictionOutputError(
                f"{backend.name} sample {index} has scores outside the "
                "finite Python float range"
            )
        try:
            sample_number = _sample_index(sample, index)
        except ValueError as error:
            raise PredictionOutputError(
                f"{backend.name} sample {index} {error}"
            ) from error
        structure = sample.structure_path
        if structure is not None:
            try:
                path = Path(structure)
            except TypeError as error:
                raise PredictionOutputError(
                    f"{backend.name} sample {index} structure_path must be path-like"
                ) from error
            try:
                stat = path.stat()
                if path.is_file() and stat.st_size > 0:
                    identity = (stat.st_dev, stat.st_ino)
                    previous = structures.get(identity)
                    if previous is not None:
                        raise PredictionOutputError(
                            f"{backend.name} samples {previous} and {index} reuse "
                            f"the same structure: {path}"
                        )
                    slot = (seed, sample_number)
                    previous = slots.get(slot)
                    if previous is not None:
                        raise PredictionOutputError(
                            f"{backend.name} samples {previous} and {index} share "
                            f"canonical seed/sample slot {slot}; native output: "
                            f"{output_dir}"
                        )
                    structures[identity] = index
                    slots[slot] = index
                    validated_samples.append(
                        dataclasses.replace(sample, seed=seed, scores=scores)
                    )
                    continue
            except OSError:
                pass
        if _has_coordinates(sample.coordinates):
            slot = (seed, sample_number)
            previous = slots.get(slot)
            if previous is not None:
                raise PredictionOutputError(
                    f"{backend.name} samples {previous} and {index} share "
                    f"canonical seed/sample slot {slot}; native output: {output_dir}"
                )
            slots[slot] = index
            validated_samples.append(
                dataclasses.replace(
                    sample, seed=seed, structure_path=None, scores=scores
                )
            )
            continue
        detail = (
            f"missing or empty structure {structure}" if structure else "no structure"
        )
        raise PredictionOutputError(
            f"{backend.name} sample {index} has {detail} and no coordinates; "
            f"native output: {output_dir}"
        )
    return dataclasses.replace(result, samples=tuple(validated_samples))


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
