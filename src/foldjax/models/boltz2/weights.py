"""JAX-free resolution of one native Boltz-2 checkpoint bundle."""

from __future__ import annotations

from pathlib import Path

#: Suffixes the native loader accepts, in the order it tries them.
_SUFFIXES = (".safetensors", ".npz")


def native_weight_candidates(path: str | Path) -> tuple[Path, ...]:
    """Return candidates in the exact order used by the native loader."""

    path = Path(path)
    if path.suffix in _SUFFIXES:
        return (path,)
    return tuple(path.with_suffix(suffix) for suffix in _SUFFIXES)


def unresolved_bundle_reason(path: str | Path) -> str:
    """Say what was looked for, for a path that resolved to nothing.

    This exists because the failure is easy to cause and its bare form is
    unreadable. Every other model in this repository takes a weight
    *directory*; Boltz-2 alone takes a bundle *stem*, and
    :func:`native_weight_candidates` appends a suffix to whatever it is given.
    Hand it the directory and it silently looks for a sibling file named after
    it -- ``weights/boltz2.safetensors`` beside ``weights/boltz2/`` -- which
    nothing ever creates.

    A benchmark panel lost three of its arms to exactly that, each failing in
    under two seconds with only the directory echoed back. Naming the bundles
    that *are* in the directory turns that into a one-line correction.
    """

    path = Path(path)
    if path.is_dir():
        found = sorted(
            candidate.name
            for suffix in _SUFFIXES
            for candidate in path.glob(f"*{suffix}")
        )
        inside = ", ".join(found) if found else "no bundle files"
        return (
            f"{path} is a directory; this option takes one bundle path, with or "
            f"without its suffix. It contains: {inside}"
        )
    tried = ", ".join(str(candidate) for candidate in native_weight_candidates(path))
    return f"no native weight bundle at {path}; tried {tried}"


def resolve_native_weight_bundle(
    path: str | Path,
) -> tuple[Path, Path] | None:
    """Return the selected weight file and its scalar sidecar path.

    The sidecar is returned even when absent because absence is an observable
    loader state: creating it later changes the parameter tree.

    A directory is deliberately *not* resolved by looking inside it. The two
    bundles that live there are different models -- the structure checkpoint
    and the affinity one -- so picking one would be a guess, and this
    repository would rather refuse than have two machines quietly run two
    different models under one setting. :func:`unresolved_bundle_reason` says
    which ones are there so the caller can name the one they meant.
    """

    weights = next(
        (
            candidate
            for candidate in native_weight_candidates(path)
            if candidate.is_file()
        ),
        None,
    )
    if weights is None:
        return None
    return weights, weights.with_suffix(weights.suffix + ".json")
