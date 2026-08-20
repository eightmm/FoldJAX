"""What a model computed on the way to a structure, in a shared vocabulary.

Every model in this package builds the same two things before it predicts
coordinates: a per-token *single* representation and a token-pair
representation. They are what downstream work wants -- feeding another
model, training a head, comparing two backends on one target -- and until
now only two of the six could hand them back, each under its own name and
only through its native command line.

The vocabulary here is deliberately small, because a shared name is a claim
that two arrays mean the same thing, and that claim is false more often than
it looks. `single` and `pair` name the *role* an array plays. What the axes
count -- residues for one model, structural tokens for another -- is carried
beside each array rather than implied by its name, so a comparison that
would be meaningless is visibly meaningless instead of silently wrong.

Anything a model has that does not play one of these roles keeps its native
name and is described the same way.

These are the largest arrays a run produces: a pair representation is
quadratic in token count, which is 0.5 GB at a thousand tokens and 4.6 GB at
three thousand, and an array returned from a compiled program stays resident
for the whole run. So they are written to disk as they are produced, never
carried in the result object, and never requested by default.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

#: The roles every model in this package fills, in the order they are built.
CANONICAL_NAMES: tuple[str, ...] = ("single_inputs", "single", "pair")

#: Written beside the arrays; `foldjax show` and any reader can use it to
#: find out what an axis counts without guessing from the name.
MANIFEST_NAME = "representations.json"
ARCHIVE_NAME = "representations.npz"


@dataclass(frozen=True, slots=True)
class RepresentationSpec:
    """One array a model can hand back, and what its axes count.

    ``space`` is the thing an axis indexes -- ``"residue"`` and
    ``"structural_token"`` are different spaces even when both are called
    tokens, and two models agree only when their spaces agree.
    """

    name: str
    axes: tuple[str, ...]
    space: str
    description: str


def resolve(
    requested: Sequence[str] | str | None,
    available: Mapping[str, RepresentationSpec],
) -> tuple[str, ...]:
    """Turn a request into the names to produce, or fail naming the options.

    ``"all"`` means every array this model has, which is the useful default
    for exploration and the wrong default for a large target.
    """
    if requested is None:
        return ()
    if isinstance(requested, str):
        requested = (requested,)
    names: list[str] = []
    for entry in requested:
        for part in str(entry).split(","):
            part = part.strip()
            if not part:
                continue
            if part == "all":
                return tuple(available)
            if part not in available:
                raise ValueError(
                    f"unknown representation {part!r}; this model produces: "
                    + ", ".join(available)
                )
            if part not in names:
                names.append(part)
    return tuple(names)


def _conform(
    name: str, value: np.ndarray, spec: RepresentationSpec | None
) -> np.ndarray:
    """Return the array shaped the way its spec says it is shaped.

    Three of the five producers hand back a leading batch axis of one, which
    made the manifest contradict itself: four numbers in ``shape`` beside three
    names in ``axes``, so a consumer reading the axes to know what it had was
    reading a claim the file disproved one line earlier. A batch of one carries
    no information, so it is dropped here rather than in each producer, and
    anything else that does not match the spec is refused instead of written
    under a description that does not fit it.
    """
    if spec is None:
        return value
    if value.ndim == len(spec.axes) + 1 and value.shape[0] == 1:
        return value[0]
    if value.ndim != len(spec.axes):
        raise ValueError(
            f"{name} has {value.ndim} axes {value.shape}, but its spec names "
            f"{len(spec.axes)}: {spec.axes}"
        )
    return value


def save(
    directory: Path,
    arrays: Mapping[str, Any],
    specs: Mapping[str, RepresentationSpec],
    *,
    model: str,
) -> Path | None:
    """Write the requested arrays and their descriptions; return the archive.

    Returns ``None`` when nothing was requested, so callers can treat "off"
    and "wrote nothing" the same way.
    """
    if not arrays:
        return None
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    materialised = {
        name: _conform(name, np.asarray(value), specs.get(name))
        for name, value in arrays.items()
    }
    archive = directory / ARCHIVE_NAME
    np.savez(archive, **materialised)
    manifest = {
        "model": model,
        "representations": {
            name: {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "axes": list(specs[name].axes) if name in specs else None,
                "space": specs[name].space if name in specs else None,
                "description": specs[name].description if name in specs else None,
            }
            for name, value in materialised.items()
        },
    }
    (directory / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n")
    return archive


def load(directory: Path) -> dict[str, np.ndarray]:
    """Read back what ``save`` wrote, by name."""
    archive = Path(directory) / ARCHIVE_NAME
    if not archive.is_file():
        archive = Path(directory)
    with np.load(archive) as data:
        return {name: data[name] for name in data.files}


def describe(directory: Path) -> dict[str, Any]:
    """Read the manifest ``save`` wrote beside the arrays."""
    return json.loads((Path(directory) / MANIFEST_NAME).read_text())


def _spec(
    name: str,
    axes: tuple[str, ...],
    space: str,
    description: str,
) -> RepresentationSpec:
    return RepresentationSpec(
        name=name, axes=axes, space=space, description=description
    )


#: What each model can hand back, and what its axes count. The tables differ
#: because the models do: OpenDDE folds in a structural-token space that is
#: not its residue space, and ESMFold2 carries one single stream rather than
#: an input embedding and a trunk output.
SPECS: dict[str, dict[str, RepresentationSpec]] = {
    "opendde": {
        "single_inputs": _spec(
            "single_inputs", ("residue", "channel"), "residue",
            "per-residue input embedding, before the trunk",
        ),
        "single": _spec(
            "single", ("residue", "channel"), "residue",
            "per-residue single representation after the trunk",
        ),
        "pair": _spec(
            "pair", ("residue", "residue", "channel"), "residue",
            "residue-pair representation after the trunk",
        ),
        "structural_single_inputs": _spec(
            "structural_single_inputs", ("structural_token", "channel"),
            "structural_token",
            "input embedding in OpenDDE's structural-token space",
        ),
        "structural_single": _spec(
            "structural_single", ("structural_token", "channel"),
            "structural_token",
            "single representation the structural refiner produced",
        ),
        "structural_pair": _spec(
            "structural_pair",
            ("structural_token", "structural_token", "channel"),
            "structural_token",
            "pair representation the structural refiner produced; the input "
            "the diffusion module conditions on",
        ),
    },
    "protenix": {
        "single_inputs": _spec(
            "single_inputs", ("token", "channel"), "token",
            "per-token input embedding, before the trunk",
        ),
        "single": _spec(
            "single", ("token", "channel"), "token",
            "per-token single representation after the trunk",
        ),
        "pair": _spec(
            "pair", ("token", "token", "channel"), "token",
            "token-pair representation after the trunk",
        ),
    },
    "boltz2": {
        "single_inputs": _spec(
            "single_inputs", ("token", "channel"), "token",
            "per-token input embedding, before the trunk",
        ),
        "single": _spec(
            "single", ("token", "channel"), "token",
            "per-token single representation after the trunk",
        ),
        "pair": _spec(
            "pair", ("token", "token", "channel"), "token",
            "token-pair representation after the trunk",
        ),
    },
    "openfold3": {
        "single_inputs": _spec(
            "single_inputs", ("token", "channel"), "token",
            "per-token input embedding, before the trunk",
        ),
        "single": _spec(
            "single", ("token", "channel"), "token",
            "per-token single representation after the trunk",
        ),
        "pair": _spec(
            "pair", ("token", "token", "channel"), "token",
            "token-pair representation after the trunk",
        ),
    },
    "esmfold2": {
        # One single stream, not two: the inputs embedding is what the heads
        # read, and the trunk updates the pair state around it.
        "single": _spec(
            "single", ("token", "channel"), "token",
            "per-token single representation the heads read",
        ),
        "pair": _spec(
            "pair", ("token", "token", "channel"), "token",
            "token-pair representation after the folding trunk",
        ),
    },
}


def specs_for(model: str) -> dict[str, RepresentationSpec]:
    """The table for one model; empty for a model that has none."""
    return SPECS.get(model, {})


def available(model: str) -> tuple[str, ...]:
    """Representation names ``model`` can produce, in build order."""
    return tuple(specs_for(model))
