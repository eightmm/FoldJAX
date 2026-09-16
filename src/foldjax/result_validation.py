"""What a backend must have produced for a call to count as a prediction.

A backend that returns normally has not necessarily predicted anything: it can
hand back a result labelled with another model, samples that reuse one
structure file, a seed the run did not ask for, or scores that are not finite.
`foldjax.api` puts every backend result through here before it writes a
manifest, so a run that is recorded as finished is one whose samples were each
checked.

The checks are pure inspection -- no JAX, no backend, no port -- which is why
they live apart from `api.py`: they read as the contract a backend has to meet,
and they can be read without loading a runtime.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from foldjax.schema import PredictionOutputError, PredictionResult, PredictionSample

if TYPE_CHECKING:
    from foldjax.backends.base import Backend


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
    from foldjax.api import (
        _representation_artifact_error,
        _requested_representation_names,
    )

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
        if stop_after in {"trunk", "inputs"}:
            raise PredictionOutputError(
                f"{backend.name} stopped after {stop_after} but returned no "
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
    if stop_after in {"trunk", "inputs"}:
        # A run stopped before sampling predicts no structure on purpose,
        # so "no samples" is the expected shape and the representations are
        # what has to be there instead.
        if result.samples:
            raise PredictionOutputError(
                f"{backend.name} returned structure samples "
                f"after {stop_after} extraction"
            )
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
