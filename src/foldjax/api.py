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
import os
import time
from collections.abc import Mapping
from itertools import groupby
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
from foldjax.manifest import (
    MANIFEST_NAME,
    device_peak_bytes,
    file_content_digest,
    matches_request,
    stat_identity_matches,
)
from foldjax.manifest import write as write_manifest
from foldjax.models._representations import archive_headers, archive_identity
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
    Representations,
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


class _ScalarBackendSource:
    """Fresh non-session backends, with the dispatcher probe consumed once."""

    def __init__(self, model: str, first: Backend) -> None:
        self._model = model
        self._first: Backend | None = first

    def take(self) -> Backend:
        backend = self._first
        self._first = None
        return backend if backend is not None else get_backend(self._model)

    def discard_probe(self) -> None:
        """Release an unused probe when resume avoids execution entirely."""

        self._first = None


def _nonempty_file(path: Path) -> bool:
    """Whether an artifact is a regular, non-empty file right now."""
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _resolve_artifact_path(
    directory: Path,
    recorded: str,
    *,
    allowed_root: Path,
) -> Path:
    """Resolve a manifest-relative artifact inside its generated run boundary."""
    relative = Path(recorded)
    if relative.is_absolute():
        raise PredictionOutputError("run manifest artifact path must be relative")
    root = Path(os.path.normpath(Path(allowed_root).absolute()))
    directory = Path(os.path.normpath(Path(directory).absolute()))
    candidate = Path(os.path.normpath(directory / relative))
    try:
        child = candidate.relative_to(root)
    except ValueError as error:
        raise PredictionOutputError(
            f"run manifest artifact escapes its output root: {recorded}"
        ) from error
    cursor = root
    for part in child.parts:
        cursor /= part
        if cursor.is_symlink():
            raise PredictionOutputError(
                f"run manifest artifact path is a symlink: {cursor}"
            )
    try:
        resolved = candidate.resolve(strict=True)
        resolved_root = root.resolve(strict=True)
    except (OSError, RuntimeError, ValueError) as error:
        raise PredictionOutputError(
            f"run manifest artifact is missing: {recorded}"
        ) from error
    if not resolved.is_relative_to(resolved_root):
        raise PredictionOutputError(
            f"run manifest artifact escapes its output root: {recorded}"
        )
    return candidate


def _requested_representation_names(
    requested: tuple[str, ...] | None,
    available: tuple[str, ...],
) -> tuple[str, ...]:
    """Resolve the request vocabulary exactly as the backend adapters do."""
    if not requested:
        return ()
    names: list[str] = []
    for entry in requested:
        for name in (part.strip() for part in entry.split(",")):
            if not name:
                continue
            if name == "all":
                return available
            if name not in names:
                names.append(name)
    return tuple(names)


def _representation_artifact_error(
    value: Any,
    *,
    expected_model: str,
    expected_names: tuple[str, ...] | None,
    expected_identity: str | None = None,
    expected_artifact: Mapping[str, Any] | None = None,
    allowed_root: Path | None = None,
) -> str | None:
    """Return why a representation result is not a restorable NPZ artifact."""
    if not isinstance(value, Representations):
        return f"has type {type(value).__name__}, not Representations"
    if value.model != expected_model:
        return f"is labelled {value.model!r}, not {expected_model!r}"
    if not isinstance(value.path, Path) or not _nonempty_file(value.path):
        return f"archive is missing or empty: {value.path!r}"
    if allowed_root is not None:
        try:
            recorded = os.path.relpath(
                value.path.absolute(), Path(allowed_root).absolute()
            )
            _resolve_artifact_path(
                Path(allowed_root),
                recorded,
                allowed_root=Path(allowed_root),
            )
        except (OSError, ValueError, PredictionOutputError) as error:
            return f"archive is outside its output root: {error}"
    if not isinstance(value.manifest, Mapping) or not value.manifest:
        return "manifest is empty or not a mapping"

    names = tuple(value.manifest)
    if not all(isinstance(name, str) and name for name in names):
        return "manifest names must be non-empty strings"
    if expected_names is not None and names != expected_names:
        return f"names {names!r} do not match requested {expected_names!r}"
    try:
        headers = archive_headers(value.path)
    except ValueError as error:
        return str(error)
    if tuple(headers) != names:
        return f"archive names {tuple(headers)!r} do not match manifest {names!r}"
    if (
        expected_identity is not None
        and archive_identity(headers) != expected_identity
    ):
        return "archive content identity does not match the run manifest"
    if expected_artifact is not None and not stat_identity_matches(
        value.path, expected_artifact
    ):
        return "archive stat identity does not match the run manifest"

    for name, header in headers.items():
        entry = value.manifest[name]
        if not isinstance(entry, Mapping):
            return f"manifest entry {name!r} is not a mapping"
        shape = entry.get("shape")
        if (
            not isinstance(shape, (list, tuple))
            or any(
                isinstance(dimension, bool) or not isinstance(dimension, int)
                for dimension in shape
            )
            or tuple(shape) != header.shape
        ):
            return (
                f"manifest shape for {name!r} is {shape!r}, "
                f"archive reports {header.shape!r}"
            )
        dtype = entry.get("dtype")
        if dtype == "bfloat16":
            # NumPy learns this logical dtype only after ml_dtypes/JAX is
            # imported. The persisted NPY descriptor is stable regardless.
            recorded_dtype = "bfloat16"
        else:
            try:
                recorded_dtype = str(np.dtype(dtype))
            except (TypeError, ValueError):
                return f"manifest dtype for {name!r} is invalid: {dtype!r}"
        dtype_matches = recorded_dtype == header.dtype or (
            recorded_dtype == "bfloat16" and header.dtype == "|V2"
        )
        if not dtype_matches:
            return (
                f"manifest dtype for {name!r} is {recorded_dtype!r}, "
                f"archive reports {header.dtype!r}"
            )
    return None


def _representations_from_manifest(
    document: Mapping[str, Any],
    request: PredictionRequest,
    *,
    directory: Path,
    allowed_root: Path,
) -> tuple[Representations | None, bool]:
    """Restore a lazy representation handle without loading its arrays."""
    record = document.get("representations")
    if record is None:
        return None, not request.representations
    if not isinstance(record, Mapping):
        return None, False

    path_value = record.get("path")
    names = record.get("names")
    entries = record.get("entries")
    identity = record.get("identity")
    artifact = record.get("artifact")
    if (
        not isinstance(path_value, str)
        or not path_value
        or not isinstance(names, list)
        or not names
        or not all(isinstance(name, str) and name for name in names)
        or len(set(names)) != len(names)
        or not isinstance(entries, Mapping)
        or set(entries) != set(names)
        or not isinstance(identity, str)
        or not identity
        or not isinstance(artifact, Mapping)
    ):
        return None, False
    try:
        path = _resolve_artifact_path(
            directory, path_value, allowed_root=allowed_root
        )
    except PredictionOutputError:
        return None, False
    restored = Representations(
        model=record.get("model"),
        path=path,
        manifest={name: entries[name] for name in names},
    )
    available = get_backend(request.model).capabilities().representations
    expected_names = _requested_representation_names(
        request.representations, available
    )
    error = _representation_artifact_error(
        restored,
        expected_model=request.model,
        expected_names=expected_names or None,
        expected_identity=identity,
        expected_artifact=artifact,
    )
    return (restored, True) if error is None else (None, False)


def _result_from_manifest(
    directory: Path,
    request: PredictionRequest,
    *,
    seed: int,
    allowed_root: Path,
) -> PredictionResult | None:
    """Rebuild a finished run's result from the manifest it left behind.

    A resumed run returns what is on disk rather than nothing, so a caller sees
    one result per requested run whether or not this invocation produced it.
    ``raw`` is absent by construction: model-specific arrays were never written
    to the manifest, and inventing them would be worse than their absence.
    """
    path = Path(directory) / MANIFEST_NAME
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(document, Mapping) or not matches_request(
        document, request, seed=seed
    ):
        return None
    representations, valid_representations = _representations_from_manifest(
        document,
        request,
        directory=Path(directory),
        allowed_root=Path(allowed_root),
    )
    if not valid_representations:
        return None

    sample_records = document["samples"]
    if not isinstance(sample_records, list):
        return None
    if request.stop_after == "trunk":
        if sample_records or representations is None:
            return None
    elif not sample_records:
        return None

    samples: list[PredictionSample] = []
    restored_structures: set[tuple[int, int]] = set()
    restored_slots: set[tuple[int, int]] = set()
    for entry in sample_records:
        if not isinstance(entry, Mapping):
            return None
        recorded_seed = entry.get("seed")
        structure = entry.get("structure_path")
        structure_digest = entry.get("structure_sha256")
        scores = entry.get("scores")
        metadata = entry.get("metadata")
        try:
            structure_path = (
                _resolve_artifact_path(
                    Path(directory), structure, allowed_root=Path(allowed_root)
                )
                if isinstance(structure, str) and structure
                else None
            )
        except PredictionOutputError:
            return None
        if (
            isinstance(recorded_seed, bool)
            or not isinstance(recorded_seed, int)
            or recorded_seed != seed
            or not isinstance(structure, str)
            or not structure
            or structure_path is None
            or not _nonempty_file(structure_path)
            or not isinstance(structure_digest, str)
            or not structure_digest
            or file_content_digest(structure_path) != structure_digest
            or not isinstance(scores, Mapping)
            or not isinstance(metadata, Mapping)
        ):
            return None
        if any(
            not isinstance(key, str)
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
            for key, value in scores.items()
        ):
            return None
        try:
            restored_scores = {key: float(value) for key, value in scores.items()}
        except (OverflowError, TypeError, ValueError):
            return None
        if any(not math.isfinite(value) for value in restored_scores.values()):
            return None
        sample = PredictionSample(
            seed=recorded_seed,
            structure_path=structure_path,
            scores=restored_scores,
            metadata=dict(metadata),
        )
        try:
            structure_info = structure_path.stat()
            sample_number = _sample_index(sample, len(samples))
        except (OSError, ValueError):
            return None
        structure_identity = (structure_info.st_dev, structure_info.st_ino)
        slot = (seed, sample_number)
        if structure_identity in restored_structures or slot in restored_slots:
            return None
        restored_structures.add(structure_identity)
        restored_slots.add(slot)
        samples.append(sample)
    shape_profile = document.get("shape_profile")
    if shape_profile is not None and not isinstance(shape_profile, Mapping):
        return None
    return PredictionResult(
        model=request.model,
        samples=tuple(samples),
        output_dir=Path(directory),
        representations=representations,
        shape_profile=dict(shape_profile) if shape_profile is not None else None,
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
    # Resolution orders the cross product model-first. Keep only one backend
    # session alive at a time: a session may own tens of gigabytes of weights,
    # so opening every model up front would trade repeated I/O for a larger and
    # less predictable peak. Adapters opt in explicitly; every other backend
    # keeps the historical fresh-instance-per-scalar contract.
    for _model, grouped in groupby(resolved, key=lambda item: item.model):
        items = tuple(grouped)
        backend = get_backend(items[0].model)
        if not backend.session_reuse:
            backend_source = _ScalarBackendSource(items[0].model, backend)
            del backend
            for item in items:
                outcome = _predict_resolved(
                    item,
                    backend_source=backend_source,
                    output_root=output_root,
                    failures=failures,
                    skipped=skipped,
                )
                if outcome is not None:
                    results.append(outcome)
            continue
        with backend.session(items) as active_backend:
            for item in items:
                outcome = _predict_resolved(
                    item,
                    backend=active_backend,
                    output_root=output_root,
                    failures=failures,
                    skipped=skipped,
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
    models is not a neutral knob. Backends may retain request-scoped model state
    across these passes; they never retain it globally or across model groups.
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


def _attempt(
    request: PredictionRequest,
    seed: int,
    directory: Path,
    *,
    backend: Backend | None,
    backend_source: _ScalarBackendSource | None,
    layout_root: Path | None,
    allowed_root: Path | None,
    failures: list[PredictionFailure],
    skipped: list[Path],
) -> PredictionResult | None:
    """One seed, honouring the request's resume and error policies."""
    try:
        if request.resume:
            if backend is not None:
                # Anchor the live resources *before* checking the persisted
                # request.  Otherwise a checkpoint replaced between manifest
                # validation and ``observe_resumed`` could bind the reused
                # seed to generation A and the active session to generation B.
                backend.validate_session(request)
            reused = _result_from_manifest(
                directory,
                request,
                seed=seed,
                allowed_root=layout_root or directory,
            )
            if reused is not None:
                if backend is not None:
                    backend.observe_resumed(request)
                elif backend_source is not None:
                    backend_source.discard_probe()
                skipped.append(Path(directory))
                return reused
        attempt_backend = backend
        session_managed = attempt_backend is not None
        if attempt_backend is None and backend_source is not None:
            attempt_backend = backend_source.take()
        return _predict_once(
            request,
            seed,
            directory,
            backend=attempt_backend,
            session_managed=session_managed,
            layout_root=layout_root,
            allowed_root=allowed_root,
        )
    except _RESUMABLE_ERRORS as error:
        if request.on_error != "continue":
            raise
        # A continued batch historically started the next scalar run with a
        # fresh backend. Preserve that isolation when an adapter now keeps
        # request-scoped weights or derived state alive.
        if backend is not None:
            backend.invalidate_session()
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
    backend: Backend | None = None,
    backend_source: _ScalarBackendSource | None = None,
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
    entry_failure_count = len(failures)
    _prepare_output_directory(request.output_dir, boundary=output_root)
    seeds = request.resolved_seeds
    if len(seeds) == 1:
        outcome = _attempt(
            request,
            seeds[0],
            request.output_dir,
            backend=backend,
            backend_source=backend_source,
            layout_root=None,
            allowed_root=None,
            failures=failures,
            skipped=skipped,
        )
        return outcome

    started = time.perf_counter()
    results: list[PredictionResult] = []
    for seed in seeds:
        outcome = _attempt(
            request,
            seed,
            request.output_dir / f"seed_{seed}",
            backend=backend,
            backend_source=backend_source,
            layout_root=request.output_dir,
            allowed_root=request.output_dir,
            failures=failures,
            skipped=skipped,
        )
        if outcome is not None:
            results.append(outcome)
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
    # Written only when every seed in this pair succeeded.  Resume also checks
    # the recorded request identity and artifacts, but a partial top-level
    # manifest would still falsely claim that the missing seeds had finished.
    if len(failures) == entry_failure_count:
        try:
            _write_session_manifest(
                backend,
                request,
                combined,
                request.output_dir,
                cost={
                    "seconds": round(time.perf_counter() - started, 2),
                    "peak_bytes": device_peak_bytes(),
                },
            )
        except _RESUMABLE_ERRORS as error:
            if request.on_error != "continue":
                raise
            if backend is not None:
                backend.invalidate_session()
            failures.append(
                PredictionFailure(
                    model=request.model,
                    input=Path(request.input),
                    seed=None,
                    output_dir=Path(request.output_dir),
                    error=str(error),
                    error_type=type(error).__name__,
                )
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
    backend: Backend | None = None,
    session_managed: bool = False,
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
    backend = backend if backend is not None else get_backend(request.model)
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
    if session_managed:
        try:
            backend.validate_session(request)
        except BaseException:
            _discard_manifest(request.output_dir)
            raise
    result = _validate_result(
        result,
        backend=backend,
        output_dir=request.output_dir,
        expected_seed=request.seed,
        stop_after=request.stop_after,
        requested_representations=request.representations,
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
    _write_session_manifest(
        backend if session_managed else None,
        asked,
        result,
        request.output_dir,
        native_input=request.input if request.input != asked.input else None,
        cost=cost,
    )
    return result


def _write_session_manifest(
    backend: Backend | None,
    request: PredictionRequest,
    result: PredictionResult,
    directory: Path,
    **kwargs: Any,
) -> None:
    """Write only while the session's actual runtime assets stay unchanged."""

    try:
        if backend is not None:
            backend.validate_session(request)
    except BaseException:
        _discard_manifest(directory)
        raise
    written = write_manifest(request, result, directory, **kwargs)
    if written is None:
        # ``write_manifest`` intentionally preserves a successful prediction
        # on provenance errors.  It must not, however, leave an older manifest
        # advertising output files that this run may just have overwritten.
        _discard_manifest(directory)
    if backend is None:
        return
    try:
        backend.validate_session(request)
    except BaseException:
        # The manifest may have observed a different checkpoint generation in
        # the narrow interval between validation and its own stat snapshot.
        # Keep the native outputs for diagnosis, but never advertise them as a
        # resumable completed run.
        _discard_manifest(directory)
        raise


def _discard_manifest(directory: Path) -> None:
    """Best-effort removal of a completion marker that is no longer sound."""

    try:
        (Path(directory) / MANIFEST_NAME).unlink(missing_ok=True)
    except OSError:
        pass


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
    stop_after: str = "full",
    requested_representations: tuple[str, ...] | None = None,
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
    if result.representations is None:
        if stop_after == "trunk":
            raise PredictionOutputError(
                f"{backend.name} stopped after the trunk but returned no "
                f"representations in {output_dir}"
            )
        if requested_representations:
            raise PredictionOutputError(
                f"{backend.name} returned no requested representations in "
                f"{output_dir}"
            )
    else:
        requested_names = _requested_representation_names(
            requested_representations,
            backend.capabilities().representations,
        )
        representation_error = _representation_artifact_error(
            result.representations,
            expected_model=backend.name,
            expected_names=requested_names or None,
            allowed_root=output_dir,
        )
        if representation_error is not None:
            raise PredictionOutputError(
                f"{backend.name} returned invalid representations: "
                f"{representation_error}"
            )
    if stop_after == "trunk":
        # A run that stopped at the trunk predicted no structure on purpose,
        # so "no samples" is the expected shape and the representations are
        # what has to be there instead.
        return result
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
