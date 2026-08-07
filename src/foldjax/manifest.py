"""What produced this output directory.

A structure file records coordinates and nothing about the run that made it.
Six months later a directory of `.cif` files cannot answer which model wrote
them, which checkpoint, how many recycles, or on which seed -- and neither can
FoldJAX, because the request that knew all of it is gone. Two directories from
two schedules look identical.

So every prediction writes `foldjax_run.json` beside its structures: the model
and the backend, the input and its digest, the resolved weights and their
identity, the seeds and sampling knobs actually used, the JAX runtime, and the
structures with the confidence each reported. It is written after the run, so
its presence also means the run finished.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from foldjax.schema import PredictionRequest, PredictionResult

MANIFEST_NAME = "foldjax_run.json"


def device_peak_bytes() -> int | None:
    """The device allocator's high-water mark, or None when there is no device.

    `peak_bytes_in_use` and not the pool: JAX preallocates most of the card by
    default, so anything read from `nvidia-smi` reports the reservation and is
    the same number for a 250-token job and a 3000-token one. This is the bytes
    actually in use at the worst moment.

    It is cumulative for the process and JAX exposes no reset, so a run that does
    several passes reports the largest of them -- which is what a memory figure
    for the whole run should say anyway. It is not comparable across processes
    that did unrelated work first.
    """
    try:
        import jax

        devices = jax.devices()
    except Exception:  # noqa: BLE001 - a CPU-only or broken runtime is not an error
        return None
    if not devices:
        return None
    stats = devices[0].memory_stats() or {}
    peak = stats.get("peak_bytes_in_use")
    return int(peak) if peak is not None else None


def _digest(path: Path) -> str | None:
    """SHA-256 of a file, or None if it is not one.

    The input is hashed rather than merely named because job files are edited
    in place far more often than they are renamed, and a path alone cannot tell
    two runs apart.
    """
    try:
        if not path.is_file():
            return None
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def describe_run(
    request: PredictionRequest,
    result: PredictionResult,
    *,
    native_input: Path | None = None,
    cost: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the manifest for one finished prediction.

    ``request`` is what the caller asked for. ``native_input`` is the backend
    dialect FoldJAX generated from it, when the job arrived in the common
    schema -- recorded alongside rather than instead, because that file lives
    inside the output directory being described and naming only it would say
    nothing about which job produced the directory.
    """
    from foldjax import __version__
    from foldjax.cache import runtime_profile, weight_identity
    from foldjax.output import best_sample

    weights = request.weights
    label = identity = None
    if weights is not None:
        label, identity = weight_identity(weights)

    return {
        "foldjax": __version__,
        "finished": datetime.now(UTC).isoformat(timespec="seconds"),
        "model": request.model,
        "input": {
            "path": str(request.input),
            "format": request.input_format,
            "sha256": _digest(request.input),
            "native": str(native_input) if native_input is not None else None,
        },
        "weights": {"path": str(weights) if weights else None, "label": label,
                    "identity": identity},
        "seeds": list(request.resolved_seeds),
        "sampling": request.sampling,
        "options": {key: str(value) for key, value in request.options.items()},
        "runtime": runtime_profile(),
        # What the run cost. Recorded here because the alternative is measuring
        # it from outside, and from outside the only visible memory number is
        # JAX's preallocated pool -- the same figure whatever the job.
        "cost": dict(cost) if cost else None,
        "samples": [sample.summary() for sample in result.samples],
        # Which sample this model ranks first, by the score it ranks with. A
        # pointer rather than a second copy of the coordinates: the top-ranked
        # structure used to be written twice, and two files with one content
        # invite the question of which is authoritative.
        "best": best_sample(result),
    }


def write(
    request: PredictionRequest,
    result: PredictionResult,
    directory: Path,
    *,
    native_input: Path | None = None,
    cost: dict[str, Any] | None = None,
) -> Path | None:
    """Write the manifest, or return None if the directory cannot take it.

    A prediction that succeeded must not be turned into a failure because its
    provenance could not be recorded, so this reports rather than raises.
    """
    try:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / MANIFEST_NAME
        path.write_text(
            json.dumps(
                describe_run(request, result, native_input=native_input, cost=cost),
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return path
    except OSError:
        return None
