"""Turn what a model wrote into the result object the common API hands back.

Each backend drives a different native writer, and those writers put their
output in different trees. The common layer pins one destination -- the
request's output directory -- so a caller comparing two models does not have
to know either one's layout: `result.representations.path` is in the same
place whichever model ran.
"""

from __future__ import annotations

import json
from pathlib import Path

from foldjax.models._representations import ARCHIVE_NAME, MANIFEST_NAME, resolve
from foldjax.schema import (
    ModelCapabilities,
    PredictionRequest,
    PredictionResult,
    Representations,
)


def _representations_result(
    model: str,
    output_dir: Path | None,
    wanted: tuple[str, ...],
) -> Representations | None:
    """Describe the archive a run wrote, or None when none was asked for.

    A request that asked for representations and got no archive is a bug in
    the backend wiring rather than a user error, so this is quiet about a
    missing file: the caller sees `representations is None` and the run's own
    output says what happened.
    """
    if not wanted or output_dir is None:
        return None
    archive = Path(output_dir) / ARCHIVE_NAME
    if not archive.is_file():
        return None
    manifest_path = Path(output_dir) / MANIFEST_NAME
    manifest: dict = {}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text()).get("representations", {})
    return Representations(model=model, path=archive, manifest=manifest)


def resolve_representations(
    request: PredictionRequest, capabilities: ModelCapabilities,
) -> tuple[str, ...]:
    """Apply one selector contract in planning, cache identity and execution."""
    if not request.representations:
        return ()
    inputs = request.stop_after == "inputs"
    available = (
        capabilities.input_representations if inputs else capabilities.representations
    )
    if not available:
        stage = "input" if inputs else "trunk"
        raise ValueError(
            f"{capabilities.model} does not expose {stage} representations"
        )
    try:
        names = resolve(request.representations, available, validate_all=inputs)
    except ValueError as error:
        raise ValueError(f"{error} (model: {capabilities.model})") from error
    if not names:
        raise ValueError(
            "representations must name at least one representation or 'all'"
        )
    if len(request.resolved_seeds) > 1:
        raise ValueError(
            "representations cannot be combined with multiple seeds: "
            "PredictionResult carries one representation archive; run one "
            "seed per request"
        )
    return names


def representation_result(
    request: PredictionRequest,
    wanted: tuple[str, ...],
    *,
    model: str,
    raw: object = None,
    shape_profile: dict | None = None,
) -> PredictionResult:
    """Describe a completed early stage after native extraction and cropping."""
    if request.stop_after not in {"inputs", "trunk"}:
        raise ValueError("representation_result requires an input or trunk stage")
    return PredictionResult(
        model=model,
        output_dir=request.output_dir,
        raw=raw,
        shape_profile=shape_profile,
        representations=_representations_result(
            model, request.output_dir, wanted
        ),
    )
